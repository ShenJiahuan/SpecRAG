# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
import asyncio
import ctypes
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.c_ops import batched_read_files as _c_batched_read_files
from lmcache.c_ops import pread_file as _c_pread_file
from lmcache.c_ops import readv_chunked_file as _c_readv_chunked_file
from lmcache.c_ops import writev_chunked_file as _c_writev_chunked_file
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import (
    CacheEngineKey,
    ChunkedCacheEngineKey,
    DiskCacheMetadata,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


# TODO(Jiayi): handle cases where cache is repetitvely prefetched.
class LocalDiskWorker:
    def __init__(
        self, loop: asyncio.AbstractEventLoop, max_workers: int = 4
    ) -> None:
        self.put_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

        self.prefetch_lock = threading.Lock()
        self.prefetch_tasks: dict[CacheEngineKey, Future] = {}

        self.executor = AsyncPQThreadPoolExecutor(loop, max_workers=max_workers)
        self.loop = loop
        self._closed = False

    async def submit_task(
        self,
        task_type: str,
        task: Callable,
        *args,
        **kwargs,
    ) -> Any:
        if task_type == "prefetch":
            priority = 0
            # self.insert_prefetch_task(kwargs["key"], None)
        elif task_type == "delete":
            priority = 1
        elif task_type == "put":
            priority = 2
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return await self.executor.submit_job(
            task,
            *args,
            priority=priority,
            **kwargs,
        )

    def remove_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            if key in self.put_tasks:
                self.put_tasks.remove(key)
            else:
                logger.warning(f"Key {key} not found in put tasks.")

    def insert_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            self.put_tasks.append(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def close(self):
        # Gracefully shut down the executor
        if self._closed:
            return
        self._closed = True
        self.executor.shutdown(wait=True)


class LocalDiskBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
    ):
        if torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()

        self.dst_device = dst_device

        self.local_cpu_backend = local_cpu_backend

        self.disk_lock = threading.Lock()

        assert config.local_disk is not None
        self.path: str = config.local_disk
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            logger.info(f"Created local disk cache directory: {self.path}")

        self.loop = loop

        self.use_local_cpu = config.local_cpu

        # Block size (for file system I/O)
        stat = os.statvfs(self.path)
        self.os_disk_bs = stat.f_bsize
        self.use_odirect = False

        if config.extra_config is not None:
            self.use_odirect = config.extra_config.get("use_odirect", False)
        logger.info("Using O_DIRECT for disk I/O: %s", self.use_odirect)

        self.disk_worker = LocalDiskWorker(
            loop, max_workers=config.disk_io_workers
        )

        # TODO(Jiayi): We need a disk space allocator to avoid fragmentation
        # and hide the following details away from the backend.
        self.max_cache_size = int(config.max_local_disk_size * 1024**3)
        self.current_cache_size = 0.0

        # to help maintain suffix -> prefix order in the dict
        # assumption: only one request is looked up at a time
        # (only one worker per cache engine)
        self.keys_in_request: List[CacheEngineKey] = []

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

    def __str__(self):
        return "LocalDiskBackend"

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.disk_lock:
            if key not in self.dict:
                return False
            if pin:
                self.dict[key].pin()
                # vllm lookup sets pin to True
                self.keys_in_request.append(key)
            return True

    def touch_cache(self):
        # flip the order of the keys in the request
        with self.disk_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return self.disk_worker.exists_in_put_tasks(key)

    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
            else:
                return False

    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            else:
                return False

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        if force:
            self.disk_lock.acquire()

        if not (meta := self.dict.pop(key, None)):
            if force:
                self.disk_lock.release()
            return False

        path = meta.path
        size = meta.size
        self.usage -= size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # NOTE: The following code will cause deadlock
        # res = asyncio.run_coroutine_threadsafe(
        #     self.disk_worker.submit_task("delete", os.remove, path),
        #     self.loop,
        # )
        # res.result()

        os.remove(path)

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.disk_lock.release()

        # push kv evict msg
        if self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(
                KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
            )

        return True

    def insert_key(
        self,
        key: CacheEngineKey,
        size: int,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
        chunked: bool = False,
        num_layers: int = 1,
    ) -> None:
        path = self._key_to_path(key)

        has_stored = False
        with self.disk_lock:
            if key in self.dict:
                # Update cache recency
                self.cache_policy.update_on_hit(key, self.dict)
                has_stored = True
            else:
                self.dict[key] = DiskCacheMetadata(
                    path, size, shape, dtype, fmt, 0,
                    chunked=chunked, num_layers=num_layers,
                )

        # push kv admit msg
        if self.lmcache_worker is not None and not has_stored:
            self.lmcache_worker.put_msg(
                KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
            )

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ):
        assert memory_obj.tensor is not None

        # skip repeated save
        if self.exists_in_put_tasks(key):
            logger.debug(f"Put task for {key} is already in progress.")
            return None

        self.disk_worker.insert_put_task(key)

        # TODO(Jiayi): Fragmentation is not considered here.
        required_size = memory_obj.get_physical_size()
        all_evict_keys = []
        evict_success = True
        with self.disk_lock:
            while self.current_cache_size + required_size > self.max_cache_size:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.dict, num_candidates=1
                )
                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found. Disk space under pressure."
                    )
                    evict_success = False
                    break

                for evict_key in evict_keys:
                    self.current_cache_size -= self.dict[evict_key].size

                self.batched_remove(evict_keys, force=False)

                all_evict_keys.extend(evict_keys)
            if evict_success:
                self.current_cache_size += required_size

        if not evict_success:
            return None

        self.cache_policy.update_on_put(key)
        memory_obj.ref_count_up()

        asyncio.run_coroutine_threadsafe(
            self.disk_worker.submit_task(
                "put",
                self.async_save_bytes_to_disk,
                key=key,
                memory_obj=memory_obj,
            ),
            self.loop,
        )

    # TODO(Jiayi): enable real batching
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

    def submit_chunked_put_task(
        self,
        key: ChunkedCacheEngineKey,
        memory_objs: List[MemoryObj],
    ):
        """Submit a task to write all layers of a chunk into a single file."""
        for obj in memory_objs:
            assert obj.tensor is not None

        if self.exists_in_put_tasks(key):
            logger.debug(f"Put task for {key} is already in progress.")
            return None

        self.disk_worker.insert_put_task(key)

        required_size = sum(obj.get_physical_size() for obj in memory_objs)
        evict_success = True
        with self.disk_lock:
            while self.current_cache_size + required_size > self.max_cache_size:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.dict, num_candidates=1
                )
                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found. Disk space under pressure."
                    )
                    evict_success = False
                    break
                for evict_key in evict_keys:
                    self.current_cache_size -= self.dict[evict_key].size
                self.batched_remove(evict_keys, force=False)
            if evict_success:
                self.current_cache_size += required_size

        if not evict_success:
            return None

        self.cache_policy.update_on_put(key)
        for obj in memory_objs:
            obj.ref_count_up()

        asyncio.run_coroutine_threadsafe(
            self.disk_worker.submit_task(
                "put",
                self.async_save_chunked_to_disk,
                key=key,
                memory_objs=memory_objs,
            ),
            self.loop,
        )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def async_save_chunked_to_disk(
        self,
        key: ChunkedCacheEngineKey,
        memory_objs: List[MemoryObj],
    ) -> None:
        """Write all layers of a chunk into a single file via writev."""
        path = self._key_to_path(key)

        ptrs = []
        sizes = []
        total_size = 0
        for obj in memory_objs:
            buf = obj.byte_array
            ptrs.append(obj.data_ptr)
            sizes.append(len(buf))
            total_size += len(buf)

        self.usage += total_size
        self.stats_monitor.update_local_storage_usage(self.usage)

        start_time = time.perf_counter()
        written = _c_writev_chunked_file(path, ptrs, sizes,
                                         use_odirect=self.use_odirect)
        write_time = time.perf_counter() - start_time
        if written < 0:
            logger.error("async_save_chunked_to_disk: failed to write %s", path)

        write_bw = total_size / write_time / 1e6 if write_time > 0 else 0.0
        logger.info(
            "async_save_chunked_to_disk: %d layers, "
            "%.2f MB, %.1f ms, %.2f MB/s",
            len(memory_objs),
            total_size / 1e6,
            write_time * 1e3,
            write_bw,
        )

        shape = memory_objs[0].metadata.shape
        dtype = memory_objs[0].metadata.dtype
        fmt = memory_objs[0].metadata.fmt
        for obj in memory_objs:
            obj.ref_count_down()

        self.insert_key(
            key, total_size, shape, dtype, fmt,
            chunked=True, num_layers=len(memory_objs),
        )
        self.disk_worker.remove_put_task(key)

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return None

        # Update cache recency
        self.cache_policy.update_on_hit(key, self.dict)

        disk_meta = self.dict[key]
        path = disk_meta.path
        dtype = disk_meta.dtype
        shape = disk_meta.shape
        fmt = disk_meta.fmt
        assert dtype is not None
        assert shape is not None

        self.disk_lock.release()
        memory_obj = self.load_bytes_from_disk(
            key, path, dtype=dtype, shape=shape, fmt=fmt
        )

        return memory_obj

    def prepare_batched_read(
        self,
        keys: list[CacheEngineKey],
        memory_objs: Optional[list[MemoryObj]] = None,
    ) -> tuple[list[str], list[MemoryObj]]:
        """Synchronously collect file paths, pin entries, and update LRU.

        This separates the metadata work from the async disk I/O so that
        callers can perform it eagerly (e.g. before entering a per-layer
        wait loop), eliminating event-loop and thread-scheduling latency
        on the critical path.

        Returns ``(paths, memory_objs)`` ready for
        :meth:`submit_batched_disk_read`.
        """
        paths: list[str] = []

        with self.disk_lock:
            for key in keys:
                assert key in self.dict, (
                    f"Key {key} not found in disk cache after pinning"
                )
                paths.append(self.dict[key].path)
                self.dict[key].pin()
                self.cache_policy.update_on_hit(key, self.dict)

        if memory_objs is None:
            memory_objs = []
            with self.disk_lock:
                for key in keys:
                    dtype = self.dict[key].dtype
                    shape = self.dict[key].shape
                    fmt = self.dict[key].fmt
                    assert dtype is not None
                    assert shape is not None
                    memory_obj = self.local_cpu_backend.allocate(
                        shape, dtype, fmt,
                    )
                    assert memory_obj is not None, (
                        "Memory allocation failed during async disk load."
                    )
                    memory_objs.append(memory_obj)

        return paths, memory_objs

    async def submit_batched_disk_read(
        self,
        paths: list[str],
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
    ) -> list[MemoryObj]:
        """Submit only the disk I/O after metadata has been prepared.

        Use :meth:`prepare_batched_read` first to obtain *paths* and
        *memory_objs*, then call this method to schedule the actual
        file reads on the disk worker thread pool.
        """
        return await self.disk_worker.submit_task(
            "prefetch",
            self.batched_async_load_bytes_from_disk,
            paths=paths,
            keys=keys,
            memory_objs=memory_objs,
        )

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
        memory_objs: Optional[list[MemoryObj]] = None,
    ) -> list[MemoryObj]:
        logger.info(f"lookup_id: {lookup_id}; Prefetching {len(keys)} keys from disk.")
        paths, memory_objs = self.prepare_batched_read(keys, memory_objs)
        return await self.submit_batched_disk_read(paths, keys, memory_objs)

    def prepare_chunked_read(
        self,
        key: ChunkedCacheEngineKey,
        memory_objs: Optional[list[MemoryObj]] = None,
    ) -> tuple[str, int, list[MemoryObj]]:
        """Prepare metadata for reading a chunked (all-layers) file.

        Returns ``(path, num_layers, memory_objs)`` where memory_objs has
        one entry per layer.
        """
        with self.disk_lock:
            assert key in self.dict, (
                f"Chunked key {key} not found in disk cache"
            )
            meta = self.dict[key]
            path = meta.path
            num_layers = meta.num_layers
            dtype = meta.dtype
            shape = meta.shape
            fmt = meta.fmt
            meta.pin()
            self.cache_policy.update_on_hit(key, self.dict)

        if memory_objs is None:
            memory_objs = []
            for _ in range(num_layers):
                obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
                assert obj is not None, (
                    "Memory allocation failed during chunked read."
                )
                memory_objs.append(obj)

        return path, num_layers, memory_objs

    async def get_chunked_non_blocking(
        self,
        key: ChunkedCacheEngineKey,
        memory_objs: Optional[list[MemoryObj]] = None,
    ) -> list[MemoryObj]:
        """Read layers of a chunk from a single chunked file.

        When *memory_objs* is provided, only ``len(memory_objs)`` layers
        are read (from the start of the file).  This allows partial-layer
        prefetch to read fewer layers than the file contains.
        """
        path, file_num_layers, memory_objs = self.prepare_chunked_read(
            key, memory_objs
        )
        # Read only as many layers as we have MemoryObjs for.
        # When partial prefetch is active, len(memory_objs) < file_num_layers.
        num_to_read = len(memory_objs)
        return await self.disk_worker.submit_task(
            "prefetch",
            self.async_load_chunked_from_disk,
            path=path,
            key=key,
            num_layers=num_to_read,
            memory_objs=memory_objs,
        )

    def get_chunked_sync(
        self,
        key: ChunkedCacheEngineKey,
        memory_objs: Optional[list[MemoryObj]] = None,
    ) -> list[MemoryObj]:
        """Sync counterpart to `get_chunked_non_blocking`.

        Used by the retrieve-time miss fallback (speculative prefetch
        didn't cover this chunk). Bypasses the asyncio event loop so
        that reading a miss chunk doesn't re-introduce the GIL-contention
        tax that the sync prefetch path already avoided.
        """
        path, _file_num_layers, memory_objs = self.prepare_chunked_read(
            key, memory_objs
        )
        num_to_read = len(memory_objs)
        return self.async_load_chunked_from_disk(
            path=path,
            key=key,
            num_layers=num_to_read,
            memory_objs=memory_objs,
        )

    def async_load_chunked_from_disk(
        self,
        path: str,
        key: ChunkedCacheEngineKey,
        num_layers: int,
        memory_objs: list[MemoryObj],
    ) -> list[MemoryObj]:
        """Read *num_layers* layers from the start of a chunked file.

        Hot path (called both from the async retrieve fallback and from
        `get_chunked_sync`). Kept minimal for the same GIL-contention
        reason as `async_load_chunked_partial_from_disk`: no per-call
        logger, no timing, no decorator NVTX.
        """
        torch.cuda.nvtx.range_push(
            f"disk_read_chunked::chunk={key.chunk_hash}:L[0..{num_layers}]"
        )

        layer_size = len(memory_objs[0].byte_array)

        # Scatter-read: pread directly into each MemoryObj buffer,
        # no temporary buffer or memmove needed.
        ptrs = [obj.data_ptr for obj in memory_objs]
        sizes = [layer_size] * num_layers

        nbytes = _c_readv_chunked_file(path, ptrs, sizes, 0,
                                       use_odirect=self.use_odirect)
        if nbytes < 0:
            logger.error(
                "async_load_chunked_from_disk: failed to read %s", path
            )

        with self.disk_lock:
            self.dict[key].unpin()

        torch.cuda.nvtx.range_pop()
        return memory_objs

    async def get_chunked_partial_non_blocking(
        self,
        key: ChunkedCacheEngineKey,
        start_layer: int,
        end_layer: int,
        memory_objs: Optional[list[MemoryObj]] = None,
    ) -> list[MemoryObj]:
        """Read a range of layers [start_layer, end_layer) from a chunked file
        using pread with offset."""
        with self.disk_lock:
            assert key in self.dict, (
                f"Chunked key {key} not found in disk cache"
            )
            meta = self.dict[key]
            path = meta.path
            dtype = meta.dtype
            shape = meta.shape
            fmt = meta.fmt
            meta.pin()
            self.cache_policy.update_on_hit(key, self.dict)

        num_to_read = end_layer - start_layer
        if memory_objs is None:
            memory_objs = []
            for _ in range(num_to_read):
                obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
                assert obj is not None
                memory_objs.append(obj)

        return await self.disk_worker.submit_task(
            "prefetch",
            self.async_load_chunked_partial_from_disk,
            path=path,
            key=key,
            start_layer=start_layer,
            end_layer=end_layer,
            memory_objs=memory_objs,
        )

    def get_chunked_partial_sync(
        self,
        key: ChunkedCacheEngineKey,
        start_layer: int,
        end_layer: int,
        memory_objs: Optional[list[MemoryObj]] = None,
    ) -> list[MemoryObj]:
        """Sync version of get_chunked_partial_non_blocking used by the
        speculative prefetch hot path. Bypasses asyncio entirely — runs
        on a plain ThreadPoolExecutor worker thread, so no asyncio loop
        thread gets woken up, no coroutines are created, no priority
        queue is involved. The async path's asyncio orchestration was
        measured to hold the GIL ~1.2 s per 5 s prefetch window on the
        asyncio loop thread; this path eliminates that overhead.
        """
        with self.disk_lock:
            assert key in self.dict, (
                f"Chunked key {key} not found in disk cache"
            )
            meta = self.dict[key]
            path = meta.path
            dtype = meta.dtype
            shape = meta.shape
            fmt = meta.fmt
            meta.pin()
            self.cache_policy.update_on_hit(key, self.dict)

        if memory_objs is None:
            num_to_read = end_layer - start_layer
            memory_objs = []
            for _ in range(num_to_read):
                obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
                assert obj is not None
                memory_objs.append(obj)

        # async_load_chunked_partial_from_disk is already a sync function
        # that releases the GIL inside the preadv C call and handles the
        # unpin internally.
        return self.async_load_chunked_partial_from_disk(
            path=path,
            key=key,
            start_layer=start_layer,
            end_layer=end_layer,
            memory_objs=memory_objs,
        )

    def async_load_chunked_partial_from_disk(
        self,
        path: str,
        key: ChunkedCacheEngineKey,
        start_layer: int,
        end_layer: int,
        memory_objs: list[MemoryObj],
    ) -> list[MemoryObj]:
        """Read layers [start_layer, end_layer) from a chunked file via pread.

        Hot path — runs on every disk_worker thread for every prefetched
        layer. Every pybind call here is one GIL release/reacquire cycle,
        and under vLLM's 30-thread GIL contention those reacquires stall
        ~200 us each. The body is kept intentionally minimal: no timing
        instrumentation, no per-call logger, no decorator-based NVTX.
        The manual nvtx range below is the only observability.
        """
        num_to_read = end_layer - start_layer
        torch.cuda.nvtx.range_push(
            f"disk_read_chunked_partial::chunk={key.chunk_hash}"
            f":L[{start_layer}..{end_layer}]"
        )

        layer_size = len(memory_objs[0].byte_array)
        file_offset = start_layer * layer_size

        # Scatter-read directly into each MemoryObj buffer
        ptrs = [obj.data_ptr for obj in memory_objs]
        sizes = [layer_size] * num_to_read

        nbytes = _c_readv_chunked_file(path, ptrs, sizes, file_offset,
                                       use_odirect=self.use_odirect)
        if nbytes < 0:
            logger.error(
                "async_load_chunked_partial_from_disk: failed to read %s",
                path,
            )

        with self.disk_lock:
            self.dict[key].unpin()

        torch.cuda.nvtx.range_pop()
        return memory_objs

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        with self.disk_lock:
            for key in keys:
                if key not in self.dict:
                    return num_hit_counts
                if pin:
                    self.dict[key].pin()
                    self.keys_in_request.append(key)
                num_hit_counts += 1
        return num_hit_counts

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.
        """
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        buffer = memory_obj.byte_array
        path = self._key_to_path(key)

        size = len(buffer)
        self.usage += size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # TODO(Jiayi): need to add ref count in disk memory object
        self.write_file(buffer, path)

        # ref count down here because there's a ref_count_up in
        # `submit_put_task` above.
        # Ref count down better be before `insert_key` for testing
        # purposes (e.g., testing mem_leak).
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        memory_obj.ref_count_down()

        self.insert_key(key, size, shape, dtype, fmt)

        self.disk_worker.remove_put_task(key)

    @_lmcache_nvtx_annotate
    def batched_async_load_bytes_from_disk(
        self,
        paths: list[str],
        keys: list[CacheEngineKey],
        memory_objs: list[MemoryObj],
        write_back: bool = False,
    ) -> list[MemoryObj]:
        """
        Async load bytearray from disk.
        """

        logger.debug("Executing `async_load_bytes` from disk.")

        # NVTX: record which chunks/layers are being read
        if keys:
            # Group by chunk_hash to produce a compact label
            chunk_layers: dict[int, list[int]] = {}
            for k in keys:
                layer_id = getattr(k, "layer_id", -1)
                chunk_layers.setdefault(k.chunk_hash, []).append(layer_id)
            parts = []
            for ch, layers in chunk_layers.items():
                layers.sort()
                parts.append(f"chunk={ch}:L{layers}")
            torch.cuda.nvtx.range_push(
                f"disk_read::{';'.join(parts)}"
            )
        else:
            torch.cuda.nvtx.range_push("disk_read::empty")

        # Collect buffer pointers and sizes for the C extension
        c_paths = []
        c_ptrs = []
        c_sizes = []
        for path, mem_obj in zip(paths, memory_objs, strict=False):
            buf = mem_obj.byte_array
            # byte_array is a memoryview backed by ctypes array;
            # get the raw pointer via the underlying tensor.
            c_paths.append(path)
            c_ptrs.append(mem_obj.data_ptr)
            c_sizes.append(len(buf))

        total_bytes = sum(c_sizes)

        # All file reads happen in C with the GIL released
        start_time = time.perf_counter()
        results = _c_batched_read_files(c_paths, c_ptrs, c_sizes,
                                        use_odirect=self.use_odirect)
        read_time = time.perf_counter() - start_time

        # Check for errors
        for i, nbytes in enumerate(results):
            if nbytes < 0:
                logger.error(
                    "batched_async_load_bytes_from_disk: failed to "
                    "read %s",
                    c_paths[i],
                )

        # Batch unpin after all reads to avoid per-file lock overhead
        with self.disk_lock:
            for key in keys:
                self.dict[key].unpin()

        read_bw = total_bytes / read_time / 1e6 if read_time > 0 else 0.0
        logger.info(
            "batched_async_load_bytes_from_disk: %d files, "
            "%.2f MB, %.1f ms, %.2f MB/s",
            len(paths),
            total_bytes / 1e6,
            read_time * 1e3,
            read_bw,
        )

        torch.cuda.nvtx.range_pop()  # disk_read
        return memory_objs

    def load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        """
        Load bytearray from disk.
        """

        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        assert memory_obj is not None, "Memory allocation failed during disk load."

        buffer = memory_obj.byte_array
        self.read_file(key, buffer, path)
        return memory_obj

    def write_file(self, buffer, path):
        start_time = time.time()
        size = len(buffer)
        use_direct = self.use_odirect and (size % self.os_disk_bs == 0)
        if use_direct:
            # Use C++ writev with O_DIRECT (single-buffer case).
            ptr = ctypes.addressof(
                ctypes.c_char.from_buffer(buffer)
            )
            _c_writev_chunked_file(path, [ptr], [size],
                                   use_odirect=True)
        else:
            with open(path, "wb") as f:
                f.write(buffer)
        disk_write_time = time.time() - start_time
        logger.debug(
            f"Disk write size: {size} bytes, "
            f"Bandwidth: {size / disk_write_time / 1e6:.2f} MB/s"
        )

    def read_file(self, key, buffer, path):
        start_time = time.perf_counter()
        size = len(buffer)
        fblock_aligned = size % self.os_disk_bs == 0
        use_direct = self.use_odirect and fblock_aligned
        if self.use_odirect and not fblock_aligned:
            logger.warning(
                "Cannot use O_DIRECT for this file, "
                "size is not aligned to disk block size."
            )

        try:
            if use_direct:
                # Use C++ pread with O_DIRECT.
                ptr = ctypes.addressof(
                    ctypes.c_char.from_buffer(buffer)
                )
                nbytes = _c_pread_file(path, ptr, size, 0,
                                       use_odirect=True)
                if nbytes < 0:
                    raise OSError(f"pread_file failed for {path}")
            else:
                f = open(path, "rb")
                f.readinto(buffer)
                f.close()
        except FileNotFoundError:
            if self.dict.get(key, None):
                self.dict.pop(key)
            return

        disk_read_time = time.perf_counter() - start_time
        logger.debug(
            "read_file: %d bytes, total %.3f ms",
            size,
            disk_read_time * 1e3,
        )

    def get_allocator_backend(self):
        return self.local_cpu_backend

    def close(self) -> None:
        self.disk_worker.close()
