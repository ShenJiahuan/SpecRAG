# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    Any,
    Coroutine,
    Generator,
    List,
    Optional,
    Sequence,
    Union,
)
import asyncio
import functools
import threading
import time

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    ChunkedCacheEngineKey,
    _lmcache_nvtx_annotate,
    start_loop_in_thread_with_exceptions,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )

logger = init_logger(__name__)


# Helper function to get the class name of the backend
def get_backend_cname(backend: StorageBackendInterface):
    return backend.__class__.__name__


# Helper function to allocate and copy memory objects between D and H
def allocate_and_copy_objects(
    allocator_backend: AllocatorBackendInterface,
    keys: Sequence[CacheEngineKey],
    src_memory_objs: list[MemoryObj],
    stream: torch.cuda.Stream,
) -> tuple[Sequence[CacheEngineKey], list[MemoryObj]]:
    """
    Allocate the memory objects and copy the data from src_memory_objs to
    the newly allocated memory objects

    Args:
        allocator_backend: the allocator backend to allocate the new memory
          objects
        keys: the cache engine keys corresponding to the memory objects
        src_memory_objs: the memory objects to copy from
        stream: the cuda stream to run the copy in

    Returns:
        - list of cache engine keys that corresponds to the memory objects
          that has been successfully allocated
        - list of the memory objects that has been successfully allocated
    """
    allocated_objects = []
    for key, src_memory_obj in zip(keys, src_memory_objs, strict=False):
        if allocator_backend.contains(key):
            continue
        memory_obj = allocator_backend.allocate(
            shape=src_memory_obj.get_shape(),
            dtype=src_memory_obj.get_dtype(),
            fmt=src_memory_obj.meta.fmt,
            eviction=True,
            busy_loop=False,
        )

        if memory_obj is None or memory_obj.tensor is None:
            break

        with torch.cuda.stream(stream):
            memory_obj.tensor.copy_(src_memory_obj.tensor, non_blocking=True)
        allocated_objects.append(memory_obj)

    stream.synchronize()
    return keys[: len(allocated_objects)], allocated_objects


class WeightedSemaphore:
    def __init__(self, chunk_budget: int):
        # it is physically impossible to have more fragmentation than 50%
        # when all of the chunks are of the same size (save_unfull_chunk=False)
        # so we can safely allocate half of the chunk budget for concurrent requests
        self._concurrent_budget_cap = chunk_budget // 2
        self._chunk_budget_cap = chunk_budget
        self._current_chunks = self._concurrent_budget_cap
        self._cond = asyncio.Condition()

    async def acquire(self, n: int = 1) -> None:
        if n > self._chunk_budget_cap:
            raise ValueError(
                f"Trying to acquire {n} chunks, "
                f"Cannot acquire more than {self._chunk_budget_cap} chunks"
                "Please set the max local cpu size to a larger value"
            )

        async with self._cond:
            logger.debug(f"WeightedSemaphore: Attempting to acquire {n} chunks")
            if n <= self._concurrent_budget_cap:
                await self._cond.wait_for(lambda: self._current_chunks >= n)
                self._current_chunks -= n
            else:
                # Oversized case: require exclusive access
                await self._cond.wait_for(
                    lambda: self._current_chunks == self._concurrent_budget_cap
                )
                # Reserve everything
                self._current_chunks = 0
            logger.debug(
                f"WeightedSemaphore: Acquired {n} chunks, "
                f"remaining chunks: {self._current_chunks}"
            )

    async def release(self, n: int = 1) -> None:
        async with self._cond:
            if n <= self._concurrent_budget_cap:
                self._current_chunks += n
            else:
                self._current_chunks = self._concurrent_budget_cap
            self._cond.notify_all()


class AsyncSerializer:
    """
    Prevent race conditions where multiple batched_get's cause the local CPU
    backend to allocate memory objects in parallel and get deadlocked.
    """

    def __init__(
        self,
        allocator_backend: AllocatorBackendInterface,
        loop: asyncio.AbstractEventLoop,
    ):
        self.chunk_budget = allocator_backend.calculate_chunk_budget()
        self._sem = WeightedSemaphore(self.chunk_budget)
        self.loop = loop

    async def run(
        self,
        coro_fn: Coroutine[Any, Any, Any],
        num_chunks: int,
    ) -> Any:
        await self._sem.acquire(num_chunks)
        try:
            return await coro_fn
        finally:
            await self._sem.release(num_chunks)


# TODO: extend this class to implement caching policies and eviction policies
class StorageManager:
    """
    The StorageManager is responsible for managing the storage backends.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        event_manager: EventManager,
        lmcache_worker: Optional["LMCacheWorker"] = None,
    ):
        self.config = config
        self.metadata = metadata
        self.loop = asyncio.new_event_loop()

        self.thread = threading.Thread(
            target=start_loop_in_thread_with_exceptions,
            args=(self.loop,),
            name="storage-manger-event-loop",
        )
        self.thread.start()

        # Shared thread pool for short-lived tasks (transpose, callbacks)
        # to avoid spawning a new thread per operation.
        self.worker_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="sm-worker",
        )

        # Dedicated sync disk pool for the speculative prefetch hot path.
        # We deliberately bypass `self.loop` / AsyncPQThreadPoolExecutor
        # because the asyncio orchestration (coroutine creation, gather,
        # PriorityQueue.put/get, done callbacks) costs ~1.2 s of GIL hold
        # per 5 s of prefetch wall time on the asyncio loop thread, which
        # in turn starves the worker thread of GIL. See
        # `layerwise_get_chunked_partial_sync` below.
        self._prefetch_disk_pool = ThreadPoolExecutor(
            max_workers=max(4, config.disk_io_workers),
            thread_name_prefix="lmcache-prefetch-disk",
        )

        if torch.cuda.is_available():
            dst_device = "cuda"
        else:
            dst_device = "cpu"
        self.storage_backends: OrderedDict[str, StorageBackendInterface] = (
            CreateStorageBackends(
                config,
                metadata,
                self.loop,
                dst_device,
                lmcache_worker,
            )
        )

        # the backend used for actual storage
        self.non_allocator_backends = self.get_non_allocator_backends()

        self.enable_pd = config.enable_pd

        self.allocator_backend = self._get_allocator_backend(config)
        if config.local_cpu:
            self.local_cpu_backend = self.storage_backends["LocalCPUBackend"]

        self.manager_lock = threading.Lock()

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.worker_id = metadata.worker_id

        self.event_manager = event_manager

        self.async_lookup_server: Optional["LMCacheAsyncLookupServer"] = None
        self.async_serializer: Optional[AsyncSerializer] = None

        # The cuda stream for internal copies during put
        if torch.cuda.is_available():
            self.internal_copy_stream = torch.cuda.Stream()
        else:
            self.internal_copy_stream = None

    def post_init(self, **kwargs) -> None:
        if "async_lookup_server" in kwargs:
            assert not self.config.save_unfull_chunk, (
                "save_unfull_chunk should be automatically set to False when using "
                "async loading."
            )
            self.async_lookup_server = kwargs.pop("async_lookup_server")
        # PDBackend has't supported calculate_chunk_budget
        if not self.enable_pd and (
            self.config.enable_async_loading or self.config.use_layerwise
        ):
            self.async_serializer = AsyncSerializer(self.allocator_backend, self.loop)

    def _get_allocator_backend(
        self, config: LMCacheEngineConfig
    ) -> AllocatorBackendInterface:
        if self.enable_pd:
            allocator_backend = self.storage_backends["PDBackend"]
        else:
            allocator_backend = self.storage_backends["LocalCPUBackend"]
        assert isinstance(allocator_backend, AllocatorBackendInterface)
        return allocator_backend

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
        busy_loop=True,
    ) -> Optional[MemoryObj]:
        """
        Allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        return self.allocator_backend.allocate(
            shape, dtype, fmt, eviction=eviction, busy_loop=busy_loop
        )

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
        busy_loop=True,
    ) -> Optional[list[MemoryObj]]:
        """
        Batched allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        return self.allocator_backend.batched_allocate(
            shape, dtype, batch_size, fmt, eviction=eviction, busy_loop=busy_loop
        )

    @_lmcache_nvtx_annotate
    def batched_allocate_chunks(
        self,
        shapes: List[torch.Size],
        dtype: torch.dtype,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
    ) -> Optional[List[List[MemoryObj]]]:
        """
        Bulk version of batched_allocate that allocates `batch_size`
        equal-sized tensors for each chunk in `shapes`, returning a
        chunk-major nested list. Used by the speculative-prefetch hot
        loop to collapse per-chunk batched_allocate calls into a
        single first-fit search; the underlying allocator only
        acquires its lock once. Returns None on allocation miss; the
        caller is expected to fall back to the eviction-aware
        per-chunk batched_allocate path.
        """
        chunks_method = getattr(
            self.allocator_backend, "batched_allocate_chunks", None
        )
        if chunks_method is None:
            return None
        return chunks_method(shapes, dtype, batch_size, fmt)

    def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        location: Optional[str] = None,
    ) -> None:
        """
        Non-blocking function to put the memory object into the storages.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).
        """
        raise RuntimeError(
            "StorageManager.put is deprecated and should not be called anymore"
        )
        for backend_name, backend in self.storage_backends.items():
            if location and backend_name != location:
                continue
            backend.submit_put_task(key, memory_obj)

        memory_obj.ref_count_down()

    def batched_put(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
        location: Optional[str] = None,
    ) -> None:
        """
        Non-blocking function to batched put the memory objects into the
        storage backends.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).
        """
        # The dictionary from backend cname to objects and keys
        obj_dict: dict[
            str,
            tuple[Sequence[CacheEngineKey], list[MemoryObj]],
        ] = {}
        obj_dict[get_backend_cname(self.allocator_backend)] = (
            keys,
            memory_objs,
        )

        for backend_name, backend in self.storage_backends.items():
            if location and backend_name != location:
                continue

            allocator_backend = backend.get_allocator_backend()
            cname = get_backend_cname(allocator_backend)
            if cname not in obj_dict:
                new_keys, new_objs = allocate_and_copy_objects(
                    allocator_backend, keys, memory_objs, self.internal_copy_stream
                )
                obj_dict[cname] = (new_keys, new_objs)

            # NOTE: the handling of exists_in_put_tasks
            # is done in the backend
            ks, objs = obj_dict[cname]
            backend.batched_submit_put_task(ks, objs, transfer_spec=transfer_spec)

        for cname, (ks, objs) in obj_dict.items():
            for memory_obj in objs:
                memory_obj.ref_count_down()

    def batched_put_chunked(
        self,
        chunked_keys: List[ChunkedCacheEngineKey],
        chunked_memory_objs: List[List[MemoryObj]],
    ) -> None:
        """Write each chunk's layers as a single file to the disk backend.

        :param chunked_keys: One ChunkedCacheEngineKey per chunk.
        :param chunked_memory_objs: chunked_memory_objs[chunk_idx][layer_idx]
            = MemoryObj for that layer, already on CPU.
        """
        from lmcache.v1.storage_backend.local_disk_backend import (
            LocalDiskBackend,
        )

        for backend_name, backend in self.storage_backends.items():
            if not isinstance(backend, LocalDiskBackend):
                continue

            # Allocate + copy to the backend's allocator if needed
            allocator_backend = backend.get_allocator_backend()
            cname = get_backend_cname(allocator_backend)
            src_cname = get_backend_cname(self.allocator_backend)

            for key, layer_objs in zip(
                chunked_keys, chunked_memory_objs, strict=False
            ):
                if cname != src_cname:
                    _, copied_objs = allocate_and_copy_objects(
                        allocator_backend,
                        [key] * len(layer_objs),
                        layer_objs,
                        self.internal_copy_stream,
                    )
                else:
                    copied_objs = layer_objs

                backend.submit_chunked_put_task(key, copied_objs)

                if cname != src_cname:
                    for obj in copied_objs:
                        obj.ref_count_down()

        # Release the original memory objects
        for layer_objs in chunked_memory_objs:
            for obj in layer_objs:
                obj.ref_count_down()

    def get(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """
        Blocking function to get the memory object from the storages.
        """

        # Search all backends for blocking get
        for backend_name, backend in self.storage_backends.items():
            if location and backend_name != location:
                continue
            # TODO(Jiayi): need to make sure all memory_objs returned
            # are allocated by the allocator backend.
            memory_obj = backend.get_blocking(key)
            if memory_obj:
                if backend_name not in ["LocalCPUBackend", "PDBackend"]:
                    local_cpu_backend = self.storage_backends["LocalCPUBackend"]
                    assert isinstance(local_cpu_backend, LocalCPUBackend)
                    local_cpu_backend.submit_put_task(key, memory_obj)
                return memory_obj

        return None

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        """
        Non-blocking function to get the memory object from the storages.
        """
        # TODO (Jiayi): incorporate prefetching here

        # Search all backends for non-blocking get
        for backend_name, backend in self.storage_backends.items():
            if location and backend_name != location:
                continue
            # NOTE(Jiayi): bypass the allocator for now
            task = backend.get_non_blocking(key)
            if task:
                # TODO (Jiayi): add write-back logic here
                return task
        return None

    def batched_get(
        self,
        keys: List[CacheEngineKey],
        location: Optional[str] = None,
    ) -> Optional[List[Optional[MemoryObj]]]:
        """
        Blocking function to get the memory objects from the storages.
        """
        # TODO (ApostaC): remove the nested optional here
        for backend_name, storage_backend in self.storage_backends.items():
            if location and backend_name != location:
                continue
            memory_objs = storage_backend.batched_get_blocking(keys)
            if memory_objs:
                return memory_objs
        return None

    def layerwise_batched_get(
        self,
        keys: List[List[CacheEngineKey]],
        location: Optional[str] = None,
        pre_allocated_mem_objs: Optional[List[List[MemoryObj]]] = None,
        eager: bool = False,
    ) -> Union[List[Future], Generator[Future, None, None]]:
        """
        Non-blocking function to get the memory objects into the storages
        in a layerwise manner.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).

        :param List[List[CacheEngineKey]] keys: The keys to get. The first
            dimension corresponds to the number of layers, and the second
            dimension corresponds to the number of chunks.

        :param Optional[List[List[MemoryObj]]] pre_allocated_mem_objs:
            Pre-allocated memory objects in layer-major format
            [num_layers][num_chunks]. If provided, the backend will use
            these instead of allocating internally.

        :param bool eager: If True, submit all layers upfront and return
            a list of futures. If False, return a lazy generator that
            yields one future per layer on demand.

        :return: A list of futures (eager) or a generator (lazy).
        """
        if eager:
            return self._layerwise_batched_get_eager(
                keys, location, pre_allocated_mem_objs
            )
        return self._layerwise_batched_get_lazy(
            keys, location, pre_allocated_mem_objs
        )

    def _layerwise_batched_get_lazy(
        self,
        keys: List[List[CacheEngineKey]],
        location: Optional[str],
        pre_allocated_mem_objs: Optional[List[List[MemoryObj]]],
    ) -> Generator[Future, None, None]:
        if location is None:
            location = "LocalCPUBackend"

        backend = self.storage_backends[location]
        assert self.async_serializer is not None, (
            "Async serializer must be initialized via post_init before using "
            "layerwise_batched_get."
        )

        for layer_id, keys_multi_chunk in enumerate(keys):
            kwargs = {}
            if pre_allocated_mem_objs is not None:
                kwargs["memory_objs"] = pre_allocated_mem_objs[layer_id]
            coro = self.async_serializer.run(
                backend.batched_get_non_blocking(
                    "fake_lookup_id",
                    keys_multi_chunk,
                    **kwargs,
                ),
                len(keys_multi_chunk),
            )
            task = asyncio.run_coroutine_threadsafe(coro, self.loop)
            yield task

    def _layerwise_batched_get_eager(
        self,
        keys: List[List[CacheEngineKey]],
        location: Optional[str],
        pre_allocated_mem_objs: Optional[List[List[MemoryObj]]],
    ) -> List[Future]:
        if location is None:
            location = "LocalCPUBackend"

        backend = self.storage_backends[location]
        assert self.async_serializer is not None, (
            "Async serializer must be initialized via post_init before using "
            "layerwise_batched_get."
        )

        use_fast_path = hasattr(backend, "prepare_batched_read")

        tasks: List[Future] = []
        for layer_id, keys_multi_chunk in enumerate(keys):
            mem_objs = (
                pre_allocated_mem_objs[layer_id]
                if pre_allocated_mem_objs is not None
                else None
            )

            if use_fast_path:
                paths, mem_objs = backend.prepare_batched_read(
                    keys_multi_chunk, memory_objs=mem_objs
                )
                coro = self.async_serializer.run(
                    backend.submit_batched_disk_read(
                        paths, keys_multi_chunk, mem_objs
                    ),
                    len(keys_multi_chunk),
                )
            else:
                kwargs = {}
                if mem_objs is not None:
                    kwargs["memory_objs"] = mem_objs
                coro = self.async_serializer.run(
                    backend.batched_get_non_blocking(
                        "fake_lookup_id",
                        keys_multi_chunk,
                        **kwargs,
                    ),
                    len(keys_multi_chunk),
                )

            task = asyncio.run_coroutine_threadsafe(coro, self.loop)
            tasks.append(task)

        return tasks

    def layerwise_batched_get_chunked(
        self,
        chunked_keys: List[ChunkedCacheEngineKey],
        num_layers: int,
        location: Optional[str] = None,
        pre_allocated_mem_objs: Optional[List[List[MemoryObj]]] = None,
    ) -> List[Future]:
        """Read chunked files for spec-miss chunks and return per-layer Futures.

        Each chunked file is read as a single I/O. Results are reorganized into
        per-layer Futures to fit the layerwise retrieve loop.

        :param chunked_keys: One ChunkedCacheEngineKey per chunk.
        :param num_layers: Number of layers.
        :param pre_allocated_mem_objs: [num_layers][num_chunks] pre-allocated.
        :return: List of Futures, one per layer. Each resolves to
            List[MemoryObj] (one per chunk for that layer).
        """
        from lmcache.v1.storage_backend.local_disk_backend import (
            LocalDiskBackend,
        )

        if location is None:
            location = "LocalDiskBackend"
        backend = self.storage_backends[location]
        assert isinstance(backend, LocalDiskBackend)

        num_chunks = len(chunked_keys)

        # Submit one read per chunk (all layers)
        chunk_futures: List[Future] = []
        for cidx in range(num_chunks):
            chunk_mem = None
            if pre_allocated_mem_objs is not None:
                chunk_mem = [
                    pre_allocated_mem_objs[lid][cidx]
                    for lid in range(num_layers)
                ]
            coro = backend.get_chunked_non_blocking(
                chunked_keys[cidx], memory_objs=chunk_mem
            )
            task = asyncio.run_coroutine_threadsafe(coro, self.loop)
            chunk_futures.append(task)

        # Create per-layer futures that transpose chunk results
        layer_futures: List[Future] = []
        for layer_id in range(num_layers):
            f: Future = Future()
            layer_futures.append(f)

        def _transpose_results(futs=chunk_futures, lf=layer_futures,
                               nl=num_layers, nc=num_chunks):
            # Wait for all chunk futures, then distribute by layer
            all_results = []
            for cf in futs:
                all_results.append(cf.result())  # List[MemoryObj] per chunk
            for lid in range(nl):
                layer_objs = [all_results[cidx][lid] for cidx in range(nc)]
                lf[lid].set_result(layer_objs)

        self.worker_pool.submit(_transpose_results)

        return layer_futures

    def layerwise_batched_get_chunked_sync(
        self,
        chunked_keys: List[ChunkedCacheEngineKey],
        num_layers: int,
        location: Optional[str] = None,
        pre_allocated_mem_objs: Optional[List[List[MemoryObj]]] = None,
    ) -> List[Future]:
        """Sync counterpart to `layerwise_batched_get_chunked` for the
        retrieve-time miss fallback path (chunks that speculative prefetch
        did not cover).

        Submits one full-layers read per chunk directly to
        `_prefetch_disk_pool`, then joins chunk futures into per-layer
        futures via a completion callback — no asyncio loop involved.
        """
        from lmcache.v1.storage_backend.local_disk_backend import (
            LocalDiskBackend,
        )

        if location is None:
            location = "LocalDiskBackend"
        backend = self.storage_backends[location]
        assert isinstance(backend, LocalDiskBackend)

        num_chunks = len(chunked_keys)

        # Submit one full-chunk read per chunk in parallel.
        chunk_futs: List[Future] = [None] * num_chunks  # type: ignore
        for cidx in range(num_chunks):
            chunk_mem = None
            if pre_allocated_mem_objs is not None:
                chunk_mem = [
                    pre_allocated_mem_objs[lid][cidx]
                    for lid in range(num_layers)
                ]
            chunk_futs[cidx] = self._prefetch_disk_pool.submit(
                backend.get_chunked_sync,
                chunked_keys[cidx],
                chunk_mem,
            )

        layer_futures: List[Future] = [Future() for _ in range(num_layers)]

        remaining = [num_chunks]
        lock = threading.Lock()

        def _on_chunk_done(
            _fut,
            lf=layer_futures,
            cf=chunk_futs,
            nc=num_chunks,
            nl=num_layers,
        ):
            with lock:
                remaining[0] -= 1
                if remaining[0] > 0:
                    return
            try:
                all_results = [c.result() for c in cf]
            except BaseException as exc:
                for f in lf:
                    if not f.done():
                        f.set_exception(exc)
                return
            for lid in range(nl):
                layer_objs = [all_results[cidx][lid] for cidx in range(nc)]
                lf[lid].set_result(layer_objs)

        # Register callbacks only after chunk_futs is fully populated.
        for cf in chunk_futs:
            cf.add_done_callback(_on_chunk_done)

        return layer_futures

    def layerwise_get_chunked_partial(
        self,
        chunked_keys: List[ChunkedCacheEngineKey],
        start_layer: int,
        end_layer: int,
        location: Optional[str] = None,
        pre_allocated_mem_objs: Optional[List[List[MemoryObj]]] = None,
    ) -> List[Future]:
        """Read partial layers from chunked files via pread.

        Returns per-layer Futures for layers [start_layer, end_layer).
        pre_allocated_mem_objs should be [num_to_read_layers][num_chunks].
        """
        from lmcache.v1.storage_backend.local_disk_backend import (
            LocalDiskBackend,
        )

        if location is None:
            location = "LocalDiskBackend"
        backend = self.storage_backends[location]
        assert isinstance(backend, LocalDiskBackend)

        num_chunks = len(chunked_keys)
        num_to_read = end_layer - start_layer

        # Build per-chunk pre-allocated memory views once
        per_chunk_mem: List[Optional[List[MemoryObj]]] = []
        for cidx in range(num_chunks):
            if pre_allocated_mem_objs is not None:
                per_chunk_mem.append([
                    pre_allocated_mem_objs[lid][cidx]
                    for lid in range(num_to_read)
                ])
            else:
                per_chunk_mem.append(None)

        layer_futures: List[Future] = [Future() for _ in range(num_to_read)]
        submit_time = time.perf_counter()

        # Single cross-thread submission: one driver coroutine that
        # gathers all k per-chunk reads inside the asyncio loop. This
        # collapses k * run_coroutine_threadsafe + 1 worker_pool submit
        # down to a single run_coroutine_threadsafe — eliminating the
        # per-chunk fixed submission overhead that dominates the
        # naive layer-by-layer Phase 2 baseline.
        async def _driver(
            keys=chunked_keys, mems=per_chunk_mem,
            s=start_layer, e=end_layer,
            be=backend, nc=num_chunks,
        ):
            coros = [
                be.get_chunked_partial_non_blocking(
                    keys[c], s, e, memory_objs=mems[c],
                )
                for c in range(nc)
            ]
            return await asyncio.gather(*coros)

        driver_task = asyncio.run_coroutine_threadsafe(
            _driver(), self.loop,
        )

        def _on_done(task, lf=layer_futures,
                     nread=num_to_read, nc=num_chunks,
                     t0=submit_time):
            try:
                all_results = task.result()  # List[k] of List[nread]
            except BaseException as exc:
                for f in lf:
                    if not f.done():
                        f.set_exception(exc)
                return
            io_elapsed = time.perf_counter() - t0
            for lid in range(nread):
                layer_objs = [all_results[cidx][lid] for cidx in range(nc)]
                lf[lid].set_result(layer_objs)
            lf[0].io_elapsed = io_elapsed

        driver_task.add_done_callback(_on_done)

        return layer_futures

    def layerwise_get_chunked_partial_sync(
        self,
        chunked_keys: List[ChunkedCacheEngineKey],
        start_layer: int,
        end_layer: int,
        location: Optional[str] = None,
        pre_allocated_mem_objs: Optional[List[List[MemoryObj]]] = None,
    ) -> List[Future]:
        """Sync counterpart to `layerwise_get_chunked_partial` that skips
        the asyncio event loop entirely.

        Each per-chunk pread is submitted directly to `_prefetch_disk_pool`
        (a plain concurrent.futures ThreadPoolExecutor). When the last
        chunk future completes we resolve the per-layer futures in one
        batch inside the completion callback. No coroutines, no gather,
        no PriorityQueue — so the asyncio loop thread is never woken up
        during the prefetch hot path.
        """
        from lmcache.v1.storage_backend.local_disk_backend import (
            LocalDiskBackend,
        )

        if location is None:
            location = "LocalDiskBackend"
        backend = self.storage_backends[location]
        assert isinstance(backend, LocalDiskBackend)

        num_chunks = len(chunked_keys)
        num_to_read = end_layer - start_layer

        # Build per-chunk memory-object views once (matches the async
        # path's layout transposition).
        per_chunk_mem: List[Optional[List[MemoryObj]]] = []
        for cidx in range(num_chunks):
            if pre_allocated_mem_objs is not None:
                per_chunk_mem.append([
                    pre_allocated_mem_objs[lid][cidx]
                    for lid in range(num_to_read)
                ])
            else:
                per_chunk_mem.append(None)

        layer_futures: List[Future] = [Future() for _ in range(num_to_read)]

        # Submit all k chunk reads up front so the pool can run them in
        # parallel on disk_worker threads.
        chunk_futs: List[Future] = [None] * num_chunks  # type: ignore
        for cidx in range(num_chunks):
            chunk_futs[cidx] = self._prefetch_disk_pool.submit(
                backend.get_chunked_partial_sync,
                chunked_keys[cidx],
                start_layer,
                end_layer,
                per_chunk_mem[cidx],
            )

        # Combine the k chunk futures into num_to_read layer futures.
        # We only populate layer_futures once all chunks are done so that
        # a layer's result list has every chunk's MemoryObj ready.
        remaining = [num_chunks]
        lock = threading.Lock()

        def _on_chunk_done(
            _fut,
            lf=layer_futures,
            cf=chunk_futs,
            nc=num_chunks,
            nread=num_to_read,
        ):
            with lock:
                remaining[0] -= 1
                if remaining[0] > 0:
                    return
            # All chunks done — collect results (may raise) and fan out.
            try:
                all_results = [c.result() for c in cf]
            except BaseException as exc:
                for f in lf:
                    if not f.done():
                        f.set_exception(exc)
                return
            for lid in range(nread):
                layer_objs = [all_results[cidx][lid] for cidx in range(nc)]
                lf[lid].set_result(layer_objs)

        # Register callbacks AFTER all submissions so that chunk_futs is
        # fully populated before any callback can fire.
        for cf in chunk_futs:
            cf.add_done_callback(_on_chunk_done)

        return layer_futures

    def batched_get_all_layers(
        self,
        keys_layer_major: List[List[CacheEngineKey]],
        location: Optional[str] = None,
        pre_allocated_mem_objs: Optional[List[List[MemoryObj]]] = None,
    ) -> Future:
        """
        Non-blocking function to load all layers of multiple chunks from
        storage in a single batched operation. Unlike layerwise_batched_get()
        which yields one Future per layer, this method flattens all keys
        across layers into one batch and returns a single Future.

        This is used by the speculative prefetch path, which loads all
        layers at once (no interleaving with selective recompute).

        :param List[List[CacheEngineKey]] keys_layer_major: The keys to get.
            Shape: [num_layers][num_chunks]. The first dimension corresponds
            to layers, the second to chunks.

        :param Optional[str] location: The storage backend to use.
            Defaults to "LocalCPUBackend".

        :param Optional[List[List[MemoryObj]]] pre_allocated_mem_objs:
            Pre-allocated memory objects in layer-major format
            [num_layers][num_chunks]. If provided, the backend will use
            these instead of allocating internally.

        :return: A concurrent.futures.Future that resolves to
            List[MemoryObj] (flattened across all layers).
        """
        if location is None:
            location = "LocalCPUBackend"

        backend = self.storage_backends[location]
        assert self.async_serializer is not None, (
            "Async serializer must be initialized via post_init before using "
            "batched_get_all_layers."
        )

        # Flatten keys from [num_layers][num_chunks] into a single list
        all_keys = [
            key
            for layer_keys in keys_layer_major
            for key in layer_keys
        ]

        # Flatten pre-allocated memory objects similarly
        kwargs = {}
        if pre_allocated_mem_objs is not None:
            all_mem_objs = [
                obj
                for layer_objs in pre_allocated_mem_objs
                for obj in layer_objs
            ]
            kwargs["memory_objs"] = all_mem_objs

        coro = self.async_serializer.run(
            backend.batched_get_non_blocking(
                "speculative_prefetch",
                all_keys,
                **kwargs,
            ),
            len(all_keys),
        )
        task = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return task

    def batched_get_per_chunk(
        self,
        keys_layer_major: List[List[CacheEngineKey]],
        location: Optional[str] = None,
        pre_allocated_mem_objs: Optional[List[List[MemoryObj]]] = None,
        chunked_keys: Optional[List[ChunkedCacheEngineKey]] = None,
    ) -> List[Future]:
        """
        Submit one disk read task per chunk (all layers), enabling parallel
        reads across chunks via the thread pool. Returns one Future per chunk.

        Unlike batched_get_all_layers() which submits everything as a single
        task, this method submits each chunk independently so that up to
        max_workers chunks can be read from disk concurrently.

        Bypasses AsyncSerializer because CPU memory is pre-allocated by the
        caller (speculative prefetch path).

        :param keys_layer_major: Shape [num_layers][num_chunks].
        :param location: Storage backend name. Defaults to "LocalDiskBackend".
        :param pre_allocated_mem_objs: Pre-allocated memory objects in
            layer-major format [num_layers][num_chunks].
        :return: List of concurrent.futures.Future, one per chunk. Each
            resolves to List[MemoryObj] (one per layer for that chunk).
        """
        if location is None:
            location = "LocalDiskBackend"

        backend = self.storage_backends[location]
        num_layers = len(keys_layer_major)
        num_chunks = len(keys_layer_major[0])

        from lmcache.v1.storage_backend.local_disk_backend import (
            LocalDiskBackend,
        )

        futures: List[Future] = []
        for chunk_idx in range(num_chunks):
            # Extract pre-allocated mem objs for this chunk
            chunk_mem_objs = None
            if pre_allocated_mem_objs is not None:
                chunk_mem_objs = [
                    pre_allocated_mem_objs[layer_idx][chunk_idx]
                    for layer_idx in range(num_layers)
                ]

            # If chunked key available, read single file per chunk
            if chunked_keys is not None and isinstance(
                backend, LocalDiskBackend
            ):
                coro = backend.get_chunked_non_blocking(
                    chunked_keys[chunk_idx],
                    memory_objs=chunk_mem_objs,
                )
            else:
                chunk_keys = [
                    keys_layer_major[layer_idx][chunk_idx]
                    for layer_idx in range(num_layers)
                ]
                kwargs = {}
                if chunk_mem_objs is not None:
                    kwargs["memory_objs"] = chunk_mem_objs
                coro = backend.batched_get_non_blocking(
                    "speculative_prefetch",
                    chunk_keys,
                    **kwargs,
                )
            task = asyncio.run_coroutine_threadsafe(coro, self.loop)
            futures.append(task)

        return futures

    def prefetch_single_done_callback(
        self,
        future: asyncio.Future,
        keys: list[CacheEngineKey],
        backend_name: str,
    ) -> None:
        """
        Callback function when a single prefetch task
        (i.e., prefetching from a single backend) is done.
        """
        # TODO(Jiayi): support write-back policy here
        pass

    def prefetch_all_done_callback(
        self,
        task: asyncio.Future,
        lookup_id: str,
        cum_last_tier_chunk_lengths: list[int],
    ) -> None:
        """
        Callback function when all prefetch tasks
        (i.e., prefetching from all backends for the entire request) are done.
        """
        assert self.async_lookup_server is not None
        self.event_manager.update_event_status(
            EventType.LOADING, lookup_id, status=EventStatus.DONE
        )
        res = task.result()
        last_tier_retrieved_chunks = len(res[-1])
        retrieved_length = cum_last_tier_chunk_lengths[last_tier_retrieved_chunks]
        logger.info(
            f"Responding to scheduler for lookup id {lookup_id}"
            f" with retrieved length {retrieved_length}"
        )
        self.async_lookup_server.send_response_to_scheduler(lookup_id, retrieved_length)

    async def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        cum_chunk_lengths: list[int],
        search_range: Optional[list[str]] = None,
        pin: bool = False,
    ) -> None:
        """
        Perform asynchronous lookup and prefetching across all storage backends.

        :param str lookup_id: The unique id (e.g., request id) for the request.
        :param list[CacheEngineKey] keys: The keys to lookup and prefetch.
        :param list[int] cum_chunk_lengths: The cumulative lengths of the chunks.
        :param Optional[list[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now. If None, search in all backends.
        :param bool pin: Whether to pin the keys.
        """

        # NOTE(Jiayi): Currently, the retrieval pattern is always
        # prefix-based. That is, we retrieve 0-t1 tokens from backend 1
        # and retrieve t1-t2 tokens from backend 2, etc. The assumption
        # here is that the suffix chunks are more likely to be evicted
        # than the prefix chunks.
        # TODO(Jiayi): We need to change/optimize this for non-prefix
        # based retrieval patterns or cases where middle chunks are missing.

        # NOTE(Jiayi): We can tolerate the last tier to have fewer loaded
        # chunks than its lookup result indicated. This is especially helpful
        # for P2PBackend.

        num_total_chunks = len(keys)
        num_total_hit_chunks = 0
        num_last_tier_hit_chunks = 0
        cum_chunk_lengths_total = cum_chunk_lengths[:]
        loading_tasks = []
        for backend_name, backend in self.storage_backends.items():
            if search_range and backend_name not in search_range:
                continue
            num_hit_chunks = await backend.batched_async_contains(lookup_id, keys, pin)

            if num_hit_chunks == 0:
                continue

            num_last_tier_hit_chunks = num_hit_chunks

            num_total_hit_chunks += num_hit_chunks

            assert self.async_serializer is not None, (
                "Async serializer must be initialized via post_init before using "
                "async_lookup_and_prefetch."
            )
            get_coro = self.async_serializer.run(
                backend.batched_get_non_blocking(
                    lookup_id,
                    keys[:num_hit_chunks],
                    {"cum_chunk_lengths": cum_chunk_lengths[: num_hit_chunks + 1]},
                ),
                num_hit_chunks,
            )
            loading_task = asyncio.create_task(get_coro)
            loading_task.add_done_callback(
                functools.partial(
                    self.prefetch_single_done_callback,
                    keys=keys,
                    backend_name=backend_name,
                )
            )

            loading_tasks.append(loading_task)

            cum_chunk_lengths = cum_chunk_lengths[num_hit_chunks:]

            if num_total_hit_chunks == num_total_chunks:
                break
            keys = keys[num_hit_chunks:]

        # If no chunks were hit across all backends, respond immediately and return.
        if num_total_hit_chunks == 0:
            if self.async_lookup_server is not None:
                self.async_lookup_server.send_response_to_scheduler(lookup_id, 0)
            return

        all_done = asyncio.gather(*loading_tasks)
        # Register the event before adding the callback to avoid race conditions
        self.event_manager.add_event(
            EventType.LOADING,
            lookup_id,
            all_done,
        )

        all_done.add_done_callback(
            lambda future: self.prefetch_all_done_callback(
                future,
                lookup_id,
                cum_chunk_lengths_total[
                    num_total_hit_chunks - num_last_tier_hit_chunks :
                ],
            )
        )

    def contains(
        self,
        key: CacheEngineKey,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> Optional[str]:
        """
        Check whether the key exists in the storage backend.

        :param CacheEngineKey key: The key to check.

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: Whether to pin the key.

        return: True if the key exists in the specified storage backends.
        """

        for backend_name, backend in self.storage_backends.items():
            if search_range and backend_name not in search_range:
                continue

            # NOTE(Jiayi): We do not pin for PDBackend
            if backend_name == "PDBackend":
                pin = False

            if backend.contains(key, pin):
                return backend_name

        return None

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        search_range: Optional[List[str]] = None,
        pin: bool = False,
        stop_after_first_not_exits: bool = True,
    ) -> List[bool]:
        """
        Check whether the key exists in the storage backend.

        :param List[CacheEngineKey] keys: The keys to check.

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: Whether to pin the key.

        :param bool stop_after_first_not_exits: Stop when find the first not exists key,
        all subsequent results will return False directly.

        return: True if the key exists in the specified storage backends else False.
        """

        # TODO: Only single-layer batched_contains is supported currently.
        # Only allocate backend is LocalCPUBackend and do not enable hot cache,
        # check another backend is supported batched_contains
        if (
            len(self.storage_backends) == 2
            and not self.config.enable_pd
            and not self.config.local_cpu
            and (search_range is None or len(search_range) == 1)
        ):
            for backend_name, backend in self.storage_backends.items():
                if backend_name == "LocalCPUBackend":
                    continue
                if (
                    search_range is None or search_range[0] == backend_name
                ) and backend.support_batched_contains():
                    return backend.batched_contains(
                        keys, pin, stop_after_first_not_exits
                    )

        # default implementation
        contains_res = []
        for key in keys:
            res = self.contains(key, search_range, pin)
            if res is not None:
                contains_res.append(True)
            else:
                if stop_after_first_not_exits:
                    # fill the contains_res with None
                    current_len = len(contains_res)
                    contains_res.extend([False] * (len(keys) - current_len))
                    break
                else:
                    contains_res.append(False)

        return contains_res

    def touch_cache(self):
        for backend_name, backend in self.storage_backends.items():
            if backend_name == "LocalCPUBackend" or backend_name == "LocalDiskBackend":
                backend.touch_cache()

    def remove(
        self,
        key: CacheEngineKey,
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Remove the key and the corresponding cache in the specified
        locations.

        :param CacheEngineKey key: The key to remove.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `remove` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `remove` in all backends.

        return: Total number of removed caches in the specified
        storage backends.
        """

        num_removed = 0
        for backend_name, backend in self.storage_backends.items():
            # TODO(Jiayi): need to handle remove in non-cpu backends
            if locations is None or backend_name in locations:
                num_removed += backend.remove(key)

        return num_removed

    def batched_remove(
        self,
        keys: List[CacheEngineKey],
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Batched remove the keys and the corresponding cache in the specified
        locations.

        :param List[CacheEngineKey] keys: The keys to remove.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `remove` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `remove` in all backends.

        return: Total number of removed caches in the specified
        storage backends.
        """
        num_removed = 0
        for backend_name, backend in self.storage_backends.items():
            if locations is None or backend_name in locations:
                num_removed += backend.batched_remove(keys)

        return num_removed

    def batched_unpin(
        self,
        keys: List[CacheEngineKey],
        locations: Optional[List[str]] = None,
    ) -> None:
        """
        Unpin the keys in the specified locations.

        :param List[CacheEngineKey] keys: The keys to unpin.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `unpin` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `unpin` in all backends.
        """
        for backend_name, backend in self.storage_backends.items():
            if locations is None or backend_name in locations:
                for key in keys:
                    backend.unpin(key)

    def clear(
        self,
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Clear all caches in the specified locations.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `clear` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `clear` in all backends.

        return: Total number of cleared tokens in the specified
        storage backends.
        """

        num_cleared_tokens = 0
        for backend_name, backend in self.storage_backends.items():
            # TODO(Jiayi): need to handle remove in non-cpu backends
            if locations is None or backend_name in locations:
                if hasattr(backend, "clear"):
                    num_cleared_tokens += backend.clear()
                else:
                    logger.warning(
                        f"Storage backend {backend_name} does not support "
                        "clear operation. Skipping."
                    )

        return num_cleared_tokens

    def memcheck(self) -> bool:
        """
        Check the integrity of the underlying storage backend's
        memory allocators

        Returns:
            True if everything is good otherwise False
        """
        for backend in self.storage_backends.values():
            if not isinstance(backend, AllocatorBackendInterface):
                continue
            if not backend.get_memory_allocator().memcheck():
                return False
        return True

    def get_non_allocator_backends(self) -> List[str]:
        """
        Get the names of the actual storage backends. Some backends,
        such as LocalCPUBackend and PDBackend, in some cases, only
        serve as a backend for allocation.
        """
        storage_names = []
        for backend_name, backend in self.storage_backends.items():
            if "LocalCPUBackend" == backend_name and not self.config.local_cpu:
                # if local_cpu is False, means LocalCPUBackend is only a allocator
                continue
            if "PDBackend" == backend_name and backend.pd_config.role == "sender":  # type: ignore
                # if pd_config.role is sender, means PDBackend is only a allocator
                continue
            storage_names.append(backend_name)
        return storage_names

    def close(self):
        for backend in self.storage_backends.values():
            backend.close()

        self.worker_pool.shutdown(wait=False)

        # using threadsafe method here as stop modifies
        # the internal state of the loop (in another thread)
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join()

        logger.info("Storage manager closed.")
