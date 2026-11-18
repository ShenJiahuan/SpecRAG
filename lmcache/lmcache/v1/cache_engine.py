# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)
import asyncio
import concurrent.futures
import ctypes
import gc
import math
import multiprocessing
import os
import threading
import time

# libc memcpy for fast pageable→pinned CPU copies that bypass
# torch.copy_'s slow path on pageable→pinned tensors.
_libc = ctypes.CDLL("libc.so.6", use_errno=False)
_libc.memcpy.restype = ctypes.c_void_p
_libc.memcpy.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
]

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_context import InitializeUsageContext
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
import nvtx as _nvtx_mod
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventType
from lmcache.v1.gpu_connector import (
    GPUConnectorInterface,
    SGLangLayerwiseGPUConnector,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.memory_management import CuFileMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import (  # noqa: E501
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.speculative_prefetch import (
    SpeculativePrefetchResult,
    SpeculativeGPUBuffer,
    build_spec_chunk_hash_map,
    make_gpu_memory_obj,
)
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.system_detection import NUMADetector, NUMAMapping
from lmcache.v1.token_database import (
    ChunkedTokenDatabase,
    SegmentTokenDatabase,
    TokenDatabase,
)

logger = init_logger(__name__)


class CacheEngineEndSignal:
    pass


@dataclass
class LookupCachedResult:
    """Cached chunk matching results from lookup(), reusable by retrieve_layer()
    to avoid redundant process_tokens + contains calls."""

    starts: List[int]
    ends: List[int]
    keys: List[List[CacheEngineKey]]  # multi-layer keys per hit chunk
    ret_mask: torch.Tensor
    retrieve_backend: Optional[str]
    num_total_chunks: int
    num_hit_chunks: int


class LMCacheEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        token_database: TokenDatabase,
        gpu_connector: GPUConnectorInterface,
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.metadata = metadata
        self.token_database = token_database
        self.gpu_connector = gpu_connector
        self.broadcast_fn = broadcast_fn
        self.broadcast_object_fn = broadcast_object_fn
        # save_only_first_rank only works when use mla
        self.save_only_first_rank = (
            self.config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )

        if self.save_only_first_rank:
            self.broadcast_stream = (
                self.gpu_connector.load_stream
                if hasattr(self.gpu_connector, "load_stream")
                else torch.cuda.Stream()
            )

        self.enable_controller = config.enable_controller

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method("spawn", force=True)

        # avoid circular import
        # First Party
        from lmcache.v1.cache_controller import LMCacheWorker

        self.lmcache_worker: Optional[LMCacheWorker] = None
        if self.enable_controller:
            self.lmcache_worker = LMCacheWorker(config, metadata, self)

        self.async_loading = config.enable_async_loading
        self.event_manager = EventManager()

        self.storage_manager = StorageManager(
            config,
            metadata,
            # self.memory_allocator,
            event_manager=self.event_manager,
            lmcache_worker=self.lmcache_worker,
        )

        # HACK: remove this in the future
        # NOTE (Jiayi): This is currently used to support
        # dropping the kv cache from the buffer in PD backend
        # at decoder.
        self.remove_after_retrieve = config.enable_pd and config.pd_role == "receiver"

        self.use_layerwise = config.use_layerwise
        self.num_layers = metadata.kv_shape[0]
        self.fmt = None
        if self.use_layerwise:
            if config.enable_blending:
                self.fmt = MemoryFormat.KV_2TD
            else:
                self.fmt = MemoryFormat.KV_T2D
        if metadata.use_mla:
            self.fmt = MemoryFormat.KV_MLA_FMT

        # NOTE(ApostaC): we haven't support lookup-cache yet
        self.lookup_cache: dict[CacheEngineKey, Any] = {}

        # lookup_id -> [pinned keys]
        self.lookup_pins: dict[str, list] = defaultdict(list)

        InitializeUsageContext(config.to_original_config(), metadata)
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        self.post_inited = False

        # Whether to force store to wait if no CPU buffer is available
        self.force_store_wait = config.extra_config and config.extra_config.get(
            "force_store_wait", False
        )

        # Cached lookup results: lookup_id -> LookupCachedResult
        # Populated by lookup() when blending is enabled, consumed by
        # retrieve_layer() to skip redundant chunk matching.
        self._lookup_result_cache: Dict[str, LookupCachedResult] = {}

        # Ablation: preserve the speculative reader and completion scheduler,
        # but never perform any retrieval-time KV reads.
        self.completion_only = os.environ.get("SPECRAG_COMPLETION_ONLY", "0") == "1"
        if self.completion_only and not (
            config.speculative_prefetch_enabled
            and config.speculative_prefetch_naive_phase2
        ):
            raise ValueError("SPECRAG_COMPLETION_ONLY requires speculative prefetch + naive Phase 2")

        # Orthogonal ablation: keep retrieval-time prefetch, but consume
        # uncached layers through CacheBlend's original lazy per-layer reader.
        self.cacheblend_completion = (
            os.environ.get("SPECRAG_CACHEBLEND_COMPLETION", "0") == "1"
        )
        if self.cacheblend_completion and (
            self.completion_only
            or not config.speculative_prefetch_enabled
            or not config.speculative_prefetch_layerwise_baseline
        ):
            raise ValueError(
                "SPECRAG_CACHEBLEND_COMPLETION requires layerwise speculative "
                "and is incompatible with SPECRAG_COMPLETION_ONLY"
            )

        # Speculative prefetch state
        self.speculative_cache: Dict[str, SpeculativePrefetchResult] = {}
        self.speculative_cache_lock = threading.Lock()
        # Per-request speculative prefetch hit-rate stats, keyed by request_id.
        # Populated after retrieve_layer() finishes chunk classification.
        self.speculative_stats: Dict[str, dict] = {}

        # Pipelined Phase 2 timing stats for adaptive batch sizing
        _win = config.speculative_prefetch_pipeline_stats_window
        self._pipeline_timing_stats: Dict[str, deque] = {
            "t_read_per_layer": deque(maxlen=_win),
            "t_infer_per_layer": deque(maxlen=_win),
        }

        # Speculative prefetch GPU buffer manager and dedicated CUDA stream
        self.speculative_gpu_buffer: Optional[SpeculativeGPUBuffer] = None
        self.spec_load_stream: Optional[torch.cuda.Stream] = None
        self._ttl_cleanup_timer: Optional[threading.Timer] = None
        if config.speculative_prefetch_enabled and torch.cuda.is_available():
            budget_bytes = int(
                config.speculative_prefetch_gpu_memory_budget_gb
                * (1024 ** 3)
            )
            self.speculative_gpu_buffer = SpeculativeGPUBuffer(
                device=torch.device("cuda"),
                dtype=metadata.kv_dtype,
                memory_budget_bytes=budget_bytes,
            )
            self.spec_load_stream = torch.cuda.Stream()
            self._start_ttl_cleanup()

        gc.collect()
        if not config.py_enable_gc:
            gc.disable()

    def post_init(self, **kwargs) -> None:
        if "async_lookup_server" in kwargs:
            self.async_lookup_server = kwargs["async_lookup_server"]
        if not self.post_inited:
            self.storage_manager.post_init(**kwargs)
            logger.info("Post-initializing LMCacheEngine")
            self.gpu_connector.initialize_kvcaches_ptr(**kwargs)
            self.post_inited = True

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens/hashes and mask into the cache engine.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if self._is_passive():
            logger.debug(f"rank={self.metadata.worker_id} ignore store")
            return

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device="cuda"
            )

        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts = []
        ends = []
        keys = []
        memory_objs = []

        offload_time = 0.0
        put_time = 0.0
        tot_kv_size = 0
        tot_token_num = 0
        t = time.perf_counter()

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        for start, end, key in self.token_database.process_tokens(
            tokens,
            hashes,
            offsets,
            mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            # Allocate the memory object
            num_tokens = end - start
            kv_shape = self.gpu_connector.get_shape(num_tokens)
            kv_dtype = self.metadata.kv_dtype

            # TODO (Jiayi): should be batched in the future
            memory_obj = self.storage_manager.allocate(
                kv_shape,
                kv_dtype,
                busy_loop=self.force_store_wait,
                fmt=self.fmt,
            )
            if memory_obj is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

            logger.debug(
                "store chunk_hash=%d tokens[%d:%d]",
                key.chunk_hash, start, end,
            )
            starts.append(start)
            ends.append(end)
            keys.append(key)
            memory_objs.append(memory_obj)
            tot_kv_size += memory_obj.get_size()
            tot_token_num += num_tokens

        # memory_objs might be empty, directly return to avoid sending tokens
        if not memory_objs:
            return
        self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
        offload_time += time.perf_counter() - t

        t = time.perf_counter()

        transfer_spec = kwargs.get("transfer_spec", None)
        self.storage_manager.batched_put(keys, memory_objs, transfer_spec=transfer_spec)
        put_time += time.perf_counter() - t

        tot_time = offload_time + put_time

        logger.info(
            "Stored %d out of total %d tokens. size: %.4f gb, cost %.4f ms, "
            "throughput: %.4f GB/s; offload_time: %.4f ms, put_time: %.4f ms",
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3,
            offload_time * 1000,
            put_time * 1000,
        )

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts = []
        ends = []
        keys = []
        memory_objs = []
        tot_token_num = 0
        kv_dtype = self.metadata.kv_dtype
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        use_chunked_disk = self.config.speculative_prefetch_enabled

        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)
            # Check existence: chunked key if speculative, else first layer
            if use_chunked_disk:
                check_key = key.to_chunked_key(self.num_layers)
            else:
                check_key = keys_multi_layer[0]
            stored = self.storage_manager.contains(check_key)
            if self.cacheblend_completion:
                # Both representations are needed: unchanged chunked prefetch
                # and unchanged CacheBlend per-layer completion reads.
                stored = stored and all(
                    self.storage_manager.contains(k) for k in keys_multi_layer
                )
            if stored:
                continue

            # Allocate the memory object
            num_tokens = end - start
            kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)

            memory_objs_multi_layer = self.storage_manager.batched_allocate(
                kv_shape_single_layer,
                kv_dtype,
                batch_size=self.num_layers,
                fmt=self.fmt,
                busy_loop=self.force_store_wait,
            )

            if memory_objs_multi_layer is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            memory_objs.append(memory_objs_multi_layer)
            tot_token_num += num_tokens

        if keys:
            # Transpose the keys and memory objects into layer major format
            memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
            keys = [list(row) for row in zip(*keys, strict=False)]

            assert isinstance(
                self.gpu_connector,
                (
                    VLLMPagedMemLayerwiseGPUConnector,
                    VLLMBufferLayerwiseGPUConnector,
                    SGLangLayerwiseGPUConnector,
                ),
            )

            mem_obj_generator = self.gpu_connector.batched_from_gpu(
                memory_objs, starts, ends, **kwargs
            )

            next(mem_obj_generator)

            if use_chunked_disk:
                # Accumulate all layers, then write as chunked files.
                # No extra ref_count_up needed: batched_put_chunked will
                # do ref_count_up in submit and ref_count_down to release
                # the allocation ref, same as batched_put does.
                for layer_id in range(self.num_layers):
                    yield
                    next(mem_obj_generator)

                # Build per-chunk lists: chunked_objs[chunk_idx][layer_idx]
                num_chunks = len(memory_objs[0])
                base_keys = keys[0]  # LayerCacheEngineKeys for layer 0
                chunked_keys = []
                chunked_objs = []
                for cidx in range(num_chunks):
                    bk = base_keys[cidx]
                    chunked_keys.append(
                        bk.to_chunked_key(self.num_layers)
                    )
                    chunked_objs.append(
                        [memory_objs[lid][cidx] for lid in range(self.num_layers)]
                    )
                if self.cacheblend_completion:
                    # Each put consumes an allocation ref. Retain one for the
                    # chunked put before submitting the per-layer copy.
                    for layer_id in range(self.num_layers):
                        for obj in memory_objs[layer_id]:
                            obj.ref_count_up()
                        self.storage_manager.batched_put(
                            keys[layer_id], memory_objs[layer_id]
                        )
                self.storage_manager.batched_put_chunked(
                    chunked_keys, chunked_objs
                )
            else:
                for layer_id in range(self.num_layers):
                    yield
                    next(mem_obj_generator)
                    self.storage_manager.batched_put(
                        keys[layer_id], memory_objs[layer_id]
                    )
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
        logger.debug(f"Stored {tot_token_num} out of total {len(tokens)} tokens")
        yield

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        tot_kv_size = 0
        t = time.perf_counter()

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        reordered_chunks: List[Tuple[CacheEngineKey, MemoryObj, int, int]] = []
        if not self._is_passive():
            if self.async_loading:
                reordered_chunks, tot_kv_size = self._async_process_tokens_internal(  # noqa: E501
                    tokens,
                    mask,
                    ret_mask,
                    **kwargs,
                )
            else:
                reordered_chunks, tot_kv_size = self._process_tokens_internal(
                    tokens,
                    mask,
                    ret_mask,
                    **kwargs,
                )
        if self.save_only_first_rank:
            with torch.cuda.stream(self.broadcast_stream):
                self._broadcast_or_receive_memory_objs(
                    reordered_chunks,
                    ret_mask,
                )

            # if self.gpu_connector has load_stream, self.broadcast_stream is equals
            # to self.gpu_connector.load_stream, the broadcast and to_gpu operation
            # will execute sequentially within the stream.
            # if self.gpu_connector does not have load_stream, self.broadcast_stream
            # is created by torch.cuda.Stream(), we need to synchronize broadcast
            # operation, and then process to_cpu operation.
            if not hasattr(self.gpu_connector, "load_stream"):
                self.broadcast_stream.synchronize()

        # NOTE(Jiayi): memory_obj doesn't have to be a pinned
        # cpu tensor for the sake of performance.
        # For example, disk->gpu is faster than disk->cpu->gpu.
        # RDMA is another example.
        if len(reordered_chunks) > 0:
            _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
            self.gpu_connector.batched_to_gpu(
                list(memory_objs), list(starts), list(ends), **kwargs
            )

        # TODO(Jiayi): Remove the following for loop with batched operations
        # TODO(Jiayi): Need to refactor the `remove_after_retrieve` logic.
        for key, memory_obj, _, _ in reordered_chunks:
            if self.remove_after_retrieve and not self._is_passive():
                self.storage_manager.remove(key)
            memory_obj.ref_count_down()

        onload_time = time.perf_counter() - t

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        logger.info(
            "Retrieved %d out of %d required tokens (from %d total tokens)."
            " size: %.4f gb,"
            " cost %.4f ms, throughput: %.4f GB/s;",
            retrieved_tokens,
            num_required_tokens,
            len(tokens),
            tot_kv_size / 1024**3,
            onload_time * 1000,
            tot_kv_size / onload_time / 1024**3 if onload_time > 0 else 0,
        )
        return ret_mask

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. In the first iteration,
            the generator retrieve the memory objects of the first layer from
            the storage backends. In the next iterations, it moves the KV cache
            of layer i from the memory objects (on CPU) to GPU and retrieves
            the memory objects of layer i+1 from the storage backends. In the
            last iteration, it moves the memory objects of the last layer to
            the GPU.
        """

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        torch.cuda.nvtx.range_push("retrieve::chunk_matching")

        # Try to reuse cached lookup results to skip redundant
        # process_tokens + contains calls.
        request_id = kwargs.get("request_id")
        cached_lookup = (
            self._lookup_result_cache.pop(request_id, None)
            if request_id
            else None
        )

        if cached_lookup is not None:
            starts = cached_lookup.starts
            ends = cached_lookup.ends
            keys = cached_lookup.keys
            retrieve_backend = cached_lookup.retrieve_backend
            num_total_chunks = cached_lookup.num_total_chunks
            num_hit_chunks = cached_lookup.num_hit_chunks
            # Rebuild ret_mask with the correct length (tokens may be
            # truncated differently between lookup and retrieve).
            ret_mask = torch.zeros(
                len(tokens), dtype=torch.bool, device="cpu"
            )
            for s, e in zip(starts, ends):
                if e <= len(tokens):
                    ret_mask[s:e] = True
            logger.info(
                "Retrieve chunk matching: reused cached lookup result "
                "(request_id=%s, %d/%d chunks hit)",
                request_id,
                num_hit_chunks,
                num_total_chunks,
            )
        else:
            ret_mask = torch.zeros(
                len(tokens), dtype=torch.bool, device="cpu"
            )

            starts = []
            ends = []
            keys = []
            retrieve_backend = None

            request_configs = kwargs.get("request_configs")
            if request_configs is not None and len(request_configs) != 0:
                assert isinstance(request_configs, dict)
            num_total_chunks = 0
            num_hit_chunks = 0
            for start, end, key in self.token_database.process_tokens(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)
                num_total_chunks += 1

                keys_multi_layer = key.split_layers(self.num_layers)

                # Check chunked key first, then fall back to per-layer
                if self.config.speculative_prefetch_enabled:
                    backend_name = self.storage_manager.contains(
                        key.to_chunked_key(self.num_layers)
                    )
                    if not backend_name:
                        backend_name = self.storage_manager.contains(
                            keys_multi_layer[0]
                        )
                else:
                    backend_name = self.storage_manager.contains(
                        keys_multi_layer[0]
                    )
                if not backend_name:
                    # When blending is enabled, skip missing chunks
                    # instead of stopping, so that later chunks can
                    # still be matched and used by CacheBlend.
                    if self.config.enable_blending:
                        continue
                    break

                if retrieve_backend is None:
                    retrieve_backend = backend_name

                num_hit_chunks += 1
                starts.append(start)
                ends.append(end)
                keys.append(keys_multi_layer)

                ret_mask[start:end] = True

            if self.config.enable_blending:
                logger.info(
                    "Retrieve chunk stats: %d/%d chunks hit",
                    num_hit_chunks,
                    num_total_chunks,
                )

        # Store ret_mask so the blender can access it to distinguish
        # hit vs miss positions for importance-based recomputation.
        self._blend_ret_mask = ret_mask

        torch.cuda.nvtx.range_pop()  # retrieve::chunk_matching

        if keys:
            # --- Phase 5: Check for speculative prefetch result ---
            torch.cuda.nvtx.range_push("retrieve::speculative_lookup")
            spec_result = None
            spec_hash_map = None
            spec_was_loading = False
            if (
                request_id
                and self.config.speculative_prefetch_enabled
                and hasattr(self, "speculative_cache")
            ):
                # Look up speculative prefetch result, with fuzzy
                # matching for vLLM's "chatcmpl-" prefix.
                def _lookup_spec_result():
                    with self.speculative_cache_lock:
                        result = self.speculative_cache.get(request_id)
                        cache_key = request_id
                        if result is None:
                            for cached_id in self.speculative_cache:
                                if (request_id.endswith(cached_id)
                                        or cached_id.endswith(request_id)):
                                    result = self.speculative_cache[cached_id]
                                    cache_key = cached_id
                                    break
                    return result, cache_key

                spec_result, spec_cache_key = _lookup_spec_result()
                if self.completion_only:
                    if spec_result is not None:
                        raise RuntimeError("Completion-only control received speculative work")
                    # This is the real layerwise L=0 completion state, created
                    # only after the final inference request reaches retrieve.
                    # Empty buffers signify zero resident layers; no allocation,
                    # disk read, or H2D copy has been performed for prefetch.
                    completed = concurrent.futures.Future()
                    completed.set_result(None)
                    spec_result = SpeculativePrefetchResult(
                        request_id=request_id, chunk_keys=keys,
                        gpu_buffers=[], future=completed,
                        num_prefetch_layers=0, status="ready", e2e_time=0.0,
                    )
                    spec_cache_key = request_id
                    with self.speculative_cache_lock:
                        self.speculative_cache[spec_cache_key] = spec_result
                    logger.info("Completion-only retrieve %s: prefetch_reads=0, ready_layers=0", request_id)

                # Build chunk hash map eagerly: chunk_keys are populated
                # at SpeculativePrefetchResult creation time, even while
                # status is still "loading".  This lets us classify
                # hits/misses and start disk I/O for misses immediately,
                # overlapping with the remaining prefetch GPU copy.
                # Accept spec_result if it has GPU buffers, CPU
                # MemoryObjs, or is still loading (will have one of
                # the two when done).
                if spec_result and (
                    spec_result.gpu_buffers is not None
                    or spec_result.cpu_mem_objs is not None
                    or spec_result.status == "loading"
                ):
                    # Layerwise baseline: freeze the snapshot NOW,
                    # before we even build the hash map. This is the
                    # "decision-deferred" half of the baseline — L is
                    # whatever the worker happens to have ready at the
                    # moment retrieve arrives.
                    #
                    # Non-blocking snapshot:
                    # `ready_num_prefetch_layers` is only advanced
                    # after the corresponding H2D event is drained
                    # (via `_drain_completed`'s `event.query()` or
                    # the worker's `final_sync`), so layers
                    # `0..L_baseline-1` are guaranteed GPU-visible
                    # the instant we read the counter. We skip
                    # waiting for the worker's finalize here because
                    # that wait (~8 ms observed) would sit in the
                    # TTFT critical path between retrieve entry and
                    # Phase-1/2 disk submission, and would only buy
                    # the 0-2 layers the worker has mid-H2D when stop
                    # fires. The worker is asked to stop via
                    # `stop_event` and finalizes in the background;
                    # the corresponding wait is deferred to the
                    # cleanup site just before `free_chunks`, where
                    # retrieve has already done tens of ms of disk
                    # work and the wait is typically a no-op.
                    _snap_wait_ms: Optional[float] = None
                    _snap_was_loading: Optional[bool] = None
                    _snap_layers_at_entry: Optional[int] = None
                    _snap_layers_at_exit: Optional[int] = None
                    if self.config.speculative_prefetch_layerwise_baseline:
                        _snap_t0 = time.monotonic()
                        _snap_was_loading = (
                            spec_result.status == "loading"
                        )
                        _snap_layers_at_entry = (
                            spec_result.ready_num_prefetch_layers
                        )
                        spec_result.stop_event.set()
                        L_baseline = (
                            spec_result.ready_num_prefetch_layers
                        )
                        _snap_layers_at_exit = L_baseline
                        _snap_wait_ms = (
                            time.monotonic() - _snap_t0
                        ) * 1000
                        # Override num_prefetch_layers so all downstream
                        # code (Phase 1 / Phase 2 / per-layer loop)
                        # uses L as the partial-layer count.
                        spec_result.num_prefetch_layers = L_baseline
                        logger.info(
                            "Speculative retrieve %s [layerwise "
                            "baseline]: snapshot L=%d "
                            "(worker_status=%s, wait=%.2fms, "
                            "L_entry=%d->exit=%d)",
                            request_id, L_baseline,
                            spec_result.status,
                            _snap_wait_ms,
                            _snap_layers_at_entry,
                            _snap_layers_at_exit,
                        )

                    spec_hash_map = build_spec_chunk_hash_map(spec_result)
                    spec_was_loading = (spec_result.status == "loading")
                    spec_is_cpu_only = (
                        spec_result.gpu_buffers is None
                    )

                    # Layerwise baseline + L == 0: nothing usable in the
                    # speculative buffer. Discard the hash map so the
                    # else-branch below treats every chunk as a plain
                    # disk miss. spec_result is kept alive only for the
                    # cleanup block at the end of retrieve_layer to
                    # release its (still allocated) GPU buffer.
                    if (
                        (self.completion_only or self.config.speculative_prefetch_layerwise_baseline)
                        and spec_result.num_prefetch_layers == 0
                    ):
                        spec_hash_map = None
                        spec_was_loading = False
                        logger.info(
                            "Speculative retrieve %s: L=0, routing all layers through "
                            "Phase 2 (asyncio partial reads, "
                            "overlapped with forward)",
                            request_id,
                        )

                    logger.info(
                        "Speculative retrieve %s: using prefetch result "
                        "(%d chunks, %s, status=%s)",
                        request_id,
                        len(spec_result.chunk_keys),
                        "CPU-only" if spec_is_cpu_only else "GPU buffer",
                        spec_result.status,
                    )
                else:
                    if spec_result:
                        logger.info(
                            "Speculative retrieve %s: prefetch result "
                            "not usable (status=%s, gpu_buffer=%s), "
                            "freeing resources",
                            request_id,
                            spec_result.status,
                            spec_result.gpu_buffers is not None,
                        )
                        # Free leaked CPU MemoryObjs to prevent
                        # memory exhaustion.
                        if spec_result.cpu_mem_objs is not None:
                            for obj in spec_result.cpu_mem_objs:
                                if obj is not None:
                                    obj.ref_count_down()
                            spec_result.cpu_mem_objs = None
                        spec_result.status = "expired"
                        with self.speculative_cache_lock:
                            self.speculative_cache.pop(
                                spec_cache_key, None
                            )
                    spec_result = None
                    spec_was_loading = False
                    spec_is_cpu_only = False

            # Classify chunks into speculative hits and disk misses
            spec_chunk_map: Dict[int, int] = {}  # actual_idx -> spec_idx
            disk_chunk_indices: List[int] = []

            if spec_hash_map is not None:
                for i, keys_multi_layer in enumerate(keys):
                    chunk_hash = (keys_multi_layer[0].chunk_hash,)
                    spec_idx = spec_hash_map.get(chunk_hash)
                    if spec_idx is not None:
                        spec_chunk_map[i] = spec_idx
                        logger.info(
                            "Retrieve chunk %d: speculative %s "
                            "(chunk_hash=%s)",
                            i,
                            "CPU" if spec_is_cpu_only else "GPU buffer",
                            keys_multi_layer[0].chunk_hash,
                        )
                    else:
                        disk_chunk_indices.append(i)
                        logger.info(
                            "Retrieve chunk %d: disk fallback "
                            "(chunk_hash=%s)",
                            i, keys_multi_layer[0].chunk_hash,
                        )

                hit_rate = (
                    len(spec_chunk_map) / len(keys) * 100.0
                    if keys
                    else 0.0
                )
                logger.info(
                    "Speculative retrieve %s: %d hits, %d misses "
                    "out of %d chunks (hit rate: %.1f%%)",
                    request_id,
                    len(spec_chunk_map),
                    len(disk_chunk_indices),
                    len(keys),
                    hit_rate,
                )
                # Expose stats so the API endpoint can return them.
                _baseline_on = (
                    self.config.speculative_prefetch_layerwise_baseline
                )
                self.speculative_stats[request_id] = {
                    "hits": len(spec_chunk_map),
                    "misses": len(disk_chunk_indices),
                    "total_chunks": len(keys),
                    "hit_rate": round(hit_rate, 2),
                    "e2e_time_ms": round(spec_result.e2e_time * 1000, 1) if spec_result.e2e_time is not None else None,
                    "num_prefetch_layers": spec_result.num_prefetch_layers,
                    "baseline": _baseline_on,
                    "layerwise_L": (
                        spec_result.ready_num_prefetch_layers
                        if _baseline_on else None
                    ),
                    "layerwise_layers_issued": (
                        spec_result.worker_layers_issued
                        if _baseline_on else None
                    ),
                    "snapshot_wait_ms": (
                        round(_snap_wait_ms, 3)
                        if _snap_wait_ms is not None else None
                    ),
                    "snapshot_was_loading": _snap_was_loading,
                    "snapshot_L_entry": _snap_layers_at_entry,
                    "snapshot_L_exit": _snap_layers_at_exit,
                }
            else:
                disk_chunk_indices = list(range(len(keys)))

            if self.completion_only:
                self.speculative_stats[request_id] = {
                    "hits": 0, "misses": len(keys), "total_chunks": len(keys),
                    "hit_rate": 0.0, "e2e_time_ms": 0.0,
                    "num_prefetch_layers": 0, "layerwise_layers_issued": 0,
                    "baseline": self.config.speculative_prefetch_layerwise_baseline,
                    "layerwise_L": 0 if self.config.speculative_prefetch_layerwise_baseline else None,
                    "retrieval_time_prefetch": False,
                }
            torch.cuda.nvtx.range_pop()  # retrieve::speculative_lookup

            # --- Determine how many layers to read eagerly ---
            num_prefetch_layers = (
                spec_result.num_prefetch_layers
                if spec_result is not None
                else self.num_layers
            )
            need_remaining_layers = (
                spec_result is not None
                and num_prefetch_layers < self.num_layers
                and not self.cacheblend_completion
            )
            # In the CacheBlend-completion ablation, no Phase 1/2 reads are
            # submitted at entry. The generator below advances one retriever
            # layer at a time, preserving CacheBlend's existing one-layer CPU
            # read-ahead: read i+1 finishes before the blender computes i.
            cacheblend_disk_indices = None
            if self.cacheblend_completion:
                if spec_result is None:
                    num_prefetch_layers = 0
                cacheblend_disk_indices = [
                    [i for i in range(len(keys))
                     if layer_id >= num_prefetch_layers or i not in spec_chunk_map]
                    for layer_id in range(self.num_layers)
                ]
                stats = self.speculative_stats.setdefault(request_id, {})
                stats.update({
                    "completion_reader": "cacheblend_per_layer",
                    "completion_schedule": "lazy_on_demand",
                    "completion_layers_submitted": 0,
                    "completion_chunk_reads": 0,
                    "completion_ahead_submissions": 0,
                })
                logger.info(
                    "CacheBlend completion %s: snapshot_layers=%d, "
                    "prefetch_hits=%d, layers=%d, reader=per_layer, "
                    "schedule=lazy_on_demand, phase2=disabled",
                    request_id, num_prefetch_layers, len(spec_chunk_map),
                    self.num_layers,
                )

            # --- Set up disk retrieval for missing chunks ---
            torch.cuda.nvtx.range_push("retrieve::mem_prealloc_and_setup")
            get_generator = None
            disk_tasks = None
            if self.cacheblend_completion:
                # Allocate once per chunk, like the original CacheBlend path,
                # but omit data already resident in the prefetch GPU buffer.
                cb_memory = [{} for _ in range(self.num_layers)]
                for idx in range(len(keys)):
                    first_layer = num_prefetch_layers if idx in spec_chunk_map else 0
                    if first_layer == self.num_layers:
                        continue
                    if retrieve_backend != "LocalDiskBackend":
                        raise RuntimeError("CacheBlend completion requires LocalDiskBackend")
                    if not all(self.storage_manager.contains(keys[idx][lid])
                               for lid in range(first_layer, self.num_layers)):
                        # During warmup, the chunked put can finish before
                        # the companion per-layer puts. Release the GIL while
                        # their existing disk workers finish; never substitute
                        # the chunked reader or submit completion reads early.
                        wait_start = time.monotonic()
                        deadline = wait_start + 10.0
                        while not all(
                            self.storage_manager.contains(keys[idx][lid])
                            for lid in range(first_layer, self.num_layers)
                        ):
                            if time.monotonic() >= deadline:
                                raise RuntimeError(
                                    "CacheBlend completion needs per-layer files; "
                                    "pending warmup writes did not finish in 10s"
                                )
                            time.sleep(0.01)
                        logger.info(
                            "CacheBlend completion %s: waited_for_per_layer_files_ms=%.3f "
                            "chunk=%d (warmup write barrier)",
                            request_id, (time.monotonic() - wait_start) * 1000, idx,
                        )
                    objs = self.storage_manager.batched_allocate(
                        self.gpu_connector.get_shape(ends[idx] - starts[idx]),
                        self.metadata.kv_dtype,
                        batch_size=self.num_layers - first_layer,
                        fmt=self.fmt,
                    )
                    assert objs is not None, "CacheBlend completion allocation failed"
                    for lid, obj in enumerate(objs, start=first_layer):
                        cb_memory[lid][idx] = obj
                cb_keys = [[keys[i][lid] for i in indices]
                           for lid, indices in enumerate(cacheblend_disk_indices)]
                cb_objs = [[cb_memory[lid][i] for i in indices]
                           for lid, indices in enumerate(cacheblend_disk_indices)]
                # Empty layers are filtered out: no empty I/O jobs and no
                # additional eager submission. next() advances the current
                # retriever layer, which is one ahead of the blender's compute.
                get_generator = self.storage_manager.layerwise_batched_get(
                    [ks for ks in cb_keys if ks],
                    location=retrieve_backend,
                    pre_allocated_mem_objs=[objs for ks, objs in zip(cb_keys, cb_objs)
                                            if ks],
                    eager=False,
                )
            elif disk_chunk_indices:
                disk_keys = [keys[i] for i in disk_chunk_indices]
                disk_keys_layer_major = [
                    list(row) for row in zip(*disk_keys, strict=False)
                ]

                # Pre-allocate memory per chunk across all layers when
                # retrieving from disk, so that allocation is
                # O(num_chunks) batched_allocate calls instead of
                # O(num_chunks * num_layers) individual allocate calls.
                pre_allocated = None
                if retrieve_backend == "LocalDiskBackend":
                    kv_dtype = self.metadata.kv_dtype
                    chunk_mem_objs: list[list[MemoryObj]] = []
                    for idx in disk_chunk_indices:
                        num_tokens = ends[idx] - starts[idx]
                        shape = self.gpu_connector.get_shape(num_tokens)
                        objs = self.storage_manager.batched_allocate(
                            shape,
                            kv_dtype,
                            batch_size=self.num_layers,
                            fmt=self.fmt,
                        )
                        assert objs is not None, (
                            "Memory allocation failed during disk retrieval."
                        )
                        chunk_mem_objs.append(objs)
                    # Transpose to layer-major: [num_layers][num_chunks]
                    pre_allocated = [
                        list(row)
                        for row in zip(*chunk_mem_objs, strict=False)
                    ]

                use_eager = spec_result is not None

                # Check if chunked files are available for disk chunks.
                # This must run regardless of use_eager: when
                # speculative_prefetch is enabled data is saved in chunked
                # format, so per-layer keys do not exist on disk.
                disk_chunked_keys = None
                if (
                    retrieve_backend == "LocalDiskBackend"
                    and disk_chunk_indices
                ):
                    base_key = keys[disk_chunk_indices[0]][0]
                    ck = CacheEngineKey(
                        base_key.fmt,
                        base_key.model_name,
                        base_key.world_size,
                        base_key.worker_id,
                        base_key.chunk_hash,
                        base_key.request_configs,
                    ).to_chunked_key(self.num_layers)
                    if self.storage_manager.contains(ck):
                        disk_chunked_keys = []
                        for idx in disk_chunk_indices:
                            bk = keys[idx][0]
                            disk_chunked_keys.append(
                                CacheEngineKey(
                                    bk.fmt, bk.model_name, bk.world_size,
                                    bk.worker_id, bk.chunk_hash,
                                    bk.request_configs,
                                ).to_chunked_key(self.num_layers)
                            )

                if need_remaining_layers and disk_chunked_keys is not None:
                    # Phase 1: only read first num_prefetch_layers for
                    # miss chunks so we don't hog disk bandwidth.
                    # Remaining layers are read together with hit chunks
                    # in Phase 2 (submitted lazily in the per-layer loop).
                    # When num_prefetch_layers == 0 (L=0 fallback), Phase 1
                    # is empty and all layers are handled by Phase 2.
                    if num_prefetch_layers > 0:
                        disk_tasks = (
                            self.storage_manager
                            .layerwise_get_chunked_partial(
                                disk_chunked_keys,
                                start_layer=0,
                                end_layer=num_prefetch_layers,
                                location=retrieve_backend,
                                pre_allocated_mem_objs=(
                                    pre_allocated[:num_prefetch_layers]
                                ),
                            )
                        )
                    else:
                        disk_tasks = None
                elif disk_chunked_keys is not None:
                    # Sync path: bypasses the LMCache asyncio loop. This
                    # is the retrieve-time miss fallback — slow queries
                    # land here when speculative prefetch did not cover
                    # one or more chunks. The original async path paid
                    # the same ~1.2 s/5 s GIL tax we eliminated from the
                    # prefetch hot path.
                    disk_tasks = (
                        self.storage_manager
                        .layerwise_batched_get_chunked_sync(
                            disk_chunked_keys,
                            num_layers=self.num_layers,
                            location=retrieve_backend,
                            pre_allocated_mem_objs=pre_allocated,
                        )
                    )
                else:
                    disk_tasks = self.storage_manager.layerwise_batched_get(
                        disk_keys_layer_major,
                        location=retrieve_backend,
                        pre_allocated_mem_objs=pre_allocated,
                        eager=use_eager,
                    )
                if not use_eager and disk_chunked_keys is None:
                    get_generator = disk_tasks
                    disk_tasks = None

            # --- Prepare Phase 2: remaining layers for ALL chunks ---
            # When partial-layer prefetch is active, layers
            # [num_prefetch_layers, num_layers) must be loaded from disk
            # for both hit chunks (not in GPU buffer) and miss chunks
            # (only first N layers read in Phase 1).
            # Phase 2 I/O is submitted lazily in the per-layer loop
            # after Phase 1 completes, so the two phases don't compete
            # for disk bandwidth.
            remaining_disk_tasks: list = []
            remaining_chunk_indices = None
            phase2_params = None
            if need_remaining_layers:
                # Remaining chunk indices = ALL matched chunks (miss + hit)
                remaining_chunk_indices = sorted(
                    set(disk_chunk_indices) | set(spec_chunk_map.keys())
                )

                remaining_num_layers = self.num_layers - num_prefetch_layers

                # Pre-allocate memory for remaining layers.
                # For miss chunks we reuse pre_allocated[num_prefetch_layers:]
                # For hit chunks we allocate new memory.
                remaining_pre_allocated = None
                if retrieve_backend == "LocalDiskBackend":
                    kv_dtype = self.metadata.kv_dtype
                    rem_chunk_mem: list[list[MemoryObj]] = []
                    for idx in remaining_chunk_indices:
                        if idx in disk_chunk_indices and pre_allocated:
                            # Reuse already-allocated memory for miss chunk
                            miss_j = disk_chunk_indices.index(idx)
                            objs = [
                                pre_allocated[lid][miss_j]
                                for lid in range(
                                    num_prefetch_layers, self.num_layers
                                )
                            ]
                        else:
                            # Allocate new memory for hit chunk
                            num_tokens = ends[idx] - starts[idx]
                            shape = self.gpu_connector.get_shape(num_tokens)
                            objs = self.storage_manager.batched_allocate(
                                shape,
                                kv_dtype,
                                batch_size=remaining_num_layers,
                                fmt=self.fmt,
                            )
                            assert objs is not None, (
                                "Memory allocation failed for remaining "
                                "layers."
                            )
                        rem_chunk_mem.append(objs)
                    # Transpose to layer-major:
                    # [remaining_num_layers][num_remaining_chunks]
                    remaining_pre_allocated = [
                        list(row)
                        for row in zip(*rem_chunk_mem, strict=False)
                    ]

                # Build chunked keys for remaining layer reads
                remaining_chunked_keys = None
                if retrieve_backend == "LocalDiskBackend":
                    bk0 = keys[remaining_chunk_indices[0]][0]
                    ck0 = CacheEngineKey(
                        bk0.fmt, bk0.model_name, bk0.world_size,
                        bk0.worker_id, bk0.chunk_hash, bk0.request_configs,
                    ).to_chunked_key(self.num_layers)
                    if self.storage_manager.contains(ck0):
                        remaining_chunked_keys = []
                        for idx in remaining_chunk_indices:
                            bk = keys[idx][0]
                            remaining_chunked_keys.append(
                                CacheEngineKey(
                                    bk.fmt, bk.model_name, bk.world_size,
                                    bk.worker_id, bk.chunk_hash,
                                    bk.request_configs,
                                ).to_chunked_key(self.num_layers)
                            )

                remaining_keys_layer_major = None
                if remaining_chunked_keys is None:
                    all_remaining_keys = [
                        keys[i] for i in remaining_chunk_indices
                    ]
                    remaining_keys_layer_major = [
                        list(row)
                        for row in zip(*all_remaining_keys, strict=False)
                    ][num_prefetch_layers:]

                phase2_params = (
                    remaining_chunked_keys,
                    remaining_keys_layer_major,
                    remaining_pre_allocated,
                    retrieve_backend,
                )

            assert isinstance(
                self.gpu_connector,
                (
                    VLLMPagedMemLayerwiseGPUConnector,
                    VLLMBufferLayerwiseGPUConnector,
                    SGLangLayerwiseGPUConnector,
                ),
            )
            # When blending with non-contiguous hits, ensure the buffer
            # starts at position 0 so get_kv() returns tensors aligned
            # with the model's token indices used in process_qkv.
            if self.config.enable_blending:
                kwargs["buf_start"] = 0
            mem_obj_consumer = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
            next(mem_obj_consumer)
            # Serialize retrieve's H2D stream behind the speculative
            # prefetch worker's H2D stream. The worker writes pinned→GPU
            # on self.spec_load_stream up until the snapshot's
            # stop_event is observed; retrieve then writes pinned→GPU on
            # gpu_connector.load_stream. On the shared DMA engine these
            # enqueues occasionally collide and inflate h2d_enq to
            # 8-13 ms/layer (vs ~40 µs typical), producing 300+ ms tail
            # TTFT in layerwise baseline mode. wait_stream inserts a
            # GPU-side dependency — no CPU block — so retrieve's H2Ds
            # queue strictly after the worker's drain. A no-op for
            # chunkwise (worker's RPC is synchronous, so spec_load_stream
            # is already idle when retrieve runs).
            if (
                self.spec_load_stream is not None
                and hasattr(self.gpu_connector, "load_stream")
            ):
                self.gpu_connector.load_stream.wait_stream(
                    self.spec_load_stream
                )
            torch.cuda.nvtx.range_pop()  # retrieve::mem_prealloc_and_setup

            # Deferred wait: if the speculative prefetch was still
            # loading when we started, wait for it now — after disk
            # I/O for misses has already been submitted in parallel.
            if spec_result is not None and spec_was_loading:
                torch.cuda.nvtx.range_push(
                    "speculative_prefetch::deferred_wait_for_loading"
                )
                wait_start = time.monotonic()
                max_wait = max(
                    0.0,
                    self.config.speculative_prefetch_ttl_seconds
                    - (time.monotonic() - spec_result.created_at),
                )
                try:
                    spec_result.future.result(timeout=max_wait)
                except Exception:
                    pass
                waited_ms = (time.monotonic() - wait_start) * 1000
                torch.cuda.nvtx.range_pop()
                logger.info(
                    "Speculative retrieve %s: deferred wait %.1f ms "
                    "for prefetch to finish (final status=%s)",
                    request_id,
                    waited_ms,
                    spec_result.status,
                )
                if spec_result.status != "ready":
                    logger.warning(
                        "Speculative retrieve %s: prefetch did not "
                        "become ready (status=%s), dropping spec hits",
                        request_id,
                        spec_result.status,
                    )
                    # Prefetch timed out: spec-hit chunks have no
                    # usable GPU data and were not submitted for disk
                    # I/O.  Mark them as unretrieved in ret_mask so
                    # blending will recompute them.
                    for actual_idx in spec_chunk_map:
                        s, e = starts[actual_idx], ends[actual_idx]
                        ret_mask[s:e] = False
                    spec_chunk_map.clear()
                    spec_result = None
                    need_remaining_layers = False
                else:
                    # Prefetch is now ready — backfill e2e_time that
                    # was unavailable when stats were first recorded.
                    if (
                        request_id in self.speculative_stats
                        and spec_result.e2e_time is not None
                    ):
                        self.speculative_stats[request_id]["e2e_time_ms"] = (
                            round(spec_result.e2e_time * 1000, 1)
                        )

            phase2_submitted = False
            # Pipelined Phase 2 state
            _use_naive = (
                need_remaining_layers
                and self.config.speculative_prefetch_naive_phase2
            )
            _use_pipeline = (
                need_remaining_layers
                and self.config.speculative_prefetch_pipelined_phase2
                and not _use_naive
            )
            _use_optimal = (
                _use_pipeline
                and self.config.speculative_prefetch_optimal_phase2
            )
            _total_remaining = self.num_layers - num_prefetch_layers
            _p2_next_layer = num_prefetch_layers  # abs layer for next batch
            _p2_next_trigger = 0   # rem_layer_idx to trigger next batch
            _p2_last_batch_sz = 0  # size of last submitted batch
            _p2_batch_num = 0  # batch counter for NVTX annotations
            _p2_batch_resolve_start: Optional[int] = None

            # Pre-computed optimal schedule: list of
            # (trigger_rem_idx, batch_size) pairs.
            # Built once at request arrival; consumed in the per-layer
            # loop instead of the reactive _compute_next_batch_size.
            _p2_optimal_plan: List[Tuple[int, int]] = []
            _p2_optimal_idx = 0  # next entry to consume

            def _submit_first_phase2_batch(runway: int) -> list:
                """Submit the first (or only) Phase 2 batch."""
                nonlocal _p2_next_layer, _p2_next_trigger, _p2_last_batch_sz
                nonlocal _p2_batch_num, _p2_batch_resolve_start
                remaining = self.num_layers - num_prefetch_layers
                if _use_naive:
                    # Naive baseline: submit one batch per layer up
                    # front, in ascending layer order. No in-loop
                    # triggers fire afterwards.
                    futures: list = []
                    for L in range(
                        num_prefetch_layers, self.num_layers
                    ):
                        futures.extend(
                            self._submit_phase2_batch(
                                self.storage_manager,
                                L, L + 1,
                                num_prefetch_layers,
                                phase2_params,
                            )
                        )
                    _p2_next_layer = self.num_layers
                    _p2_next_trigger = remaining  # never fires
                    _p2_last_batch_sz = 1
                    _p2_batch_resolve_start = None  # disable NVTX in loop
                    _p2_batch_num += 1
                    logger.info(
                        "Speculative retrieve %s: Phase 2 NAIVE — "
                        "submitted %d layer-by-layer batches "
                        "(layers [%d, %d))",
                        request_id,
                        remaining,
                        num_prefetch_layers, self.num_layers,
                    )
                    return futures
                if _use_optimal:
                    schedule = self._compute_optimal_schedule(
                        runway, remaining,
                    )
                    # Build trigger plan: batch 0 is submitted now.
                    # Batch k (k>=1) is submitted when inference starts
                    # consuming batch k-1, giving batch k the full
                    # processing time of batch k-1 for disk I/O.
                    #
                    # Trigger points (rem_layer_idx):
                    #   batch 1 → trigger at 0 (start consuming batch 0)
                    #   batch 2 → trigger at schedule[0]
                    #   batch k → trigger at sum(schedule[:k-1])
                    cum = 0
                    for idx, sz in enumerate(schedule):
                        if idx == 0:
                            batch_sz = sz
                        else:
                            _p2_optimal_plan.append((cum, sz))
                        cum += sz
                    logger.info(
                        "Speculative retrieve %s: optimal schedule %s "
                        "(runway=%d, remaining=%d)",
                        request_id, schedule, runway, remaining,
                    )
                elif _use_pipeline:
                    batch_sz = self._compute_next_batch_size(
                        runway, remaining,
                    )
                else:
                    batch_sz = remaining
                end = num_prefetch_layers + batch_sz
                futures = self._submit_phase2_batch(
                    self.storage_manager,
                    num_prefetch_layers,
                    end,
                    num_prefetch_layers,
                    phase2_params,
                )
                _p2_next_layer = end
                _p2_next_trigger = 0  # trigger next when consuming batch 1
                _p2_last_batch_sz = batch_sz
                _p2_batch_resolve_start = 0
                torch.cuda.nvtx.range_push(
                    f"phase2::batch{_p2_batch_num}_layers"
                    f"[{num_prefetch_layers},{end})_wait"
                )
                _p2_batch_num += 1
                logger.info(
                    "Speculative retrieve %s: Phase 2 batch [%d, %d) "
                    "submitted (runway=%d, total_remaining=%d%s)",
                    request_id,
                    num_prefetch_layers, end,
                    runway, remaining,
                    ", pipelined" if _use_pipeline else
                    (", optimal" if _use_optimal else ""),
                )
                return futures

            # If there's no Phase 1 to wait for, submit Phase 2 immediately.
            # Two cases trigger this: (a) no miss chunks at all, or
            # (b) num_prefetch_layers == 0 (L=0 fallback — every layer
            # is handled by Phase 2 layer-by-layer with forward overlap).
            if (
                need_remaining_layers
                and phase2_params is not None
                and (not disk_chunk_indices or num_prefetch_layers == 0)
            ):
                phase2_submitted = True
                remaining_disk_tasks = _submit_first_phase2_batch(
                    num_prefetch_layers
                )


            to_count_down = []
            _p2_first_read_recorded = False
            _infer_times: list = []
            _blend_times: list = []  # yield → resume only
            _merge_times: list = []  # merge + send only
            _vllm_blend_time: float = 0.0  # yield → resume
            _merge_send_start: float = 0.0
            _disk_wait_times: list = []  # per-layer wall time for disk I/O wait
            _per_layer_build_ms: list = []  # merged_objs build only
            _per_layer_send_ms: list = []  # mem_obj_consumer.send only
            for layer_id in range(self.num_layers):
                # Per-layer wrapper range so nsys (lmcache domain) shows
                # which layer each sample falls in.
                _nvtx_layer_id = _nvtx_mod.start_range(
                    message=f"retrieve::layer_{layer_id}",
                    domain="lmcache",
                )
                # Get disk MemoryObjs for this layer (if any)
                torch.cuda.nvtx.range_push(
                    f"retrieve::disk_io_wait_layer_{layer_id}"
                )
                _nvtx_diskwait_id = _nvtx_mod.start_range(
                    message=f"retrieve::disk_io_wait_layer_{layer_id}",
                    domain="lmcache",
                )
                disk_objs_map: Dict[int, MemoryObj] = {}
                if self.cacheblend_completion:
                    indices = cacheblend_disk_indices[layer_id]
                    if indices:
                        _t_wait_start = time.perf_counter()
                        task = next(get_generator)
                        disk_objs_layer = task.result()
                        _disk_wait_times.append(time.perf_counter() - _t_wait_start)
                        if len(disk_objs_layer) != len(indices) or any(
                            obj is None for obj in disk_objs_layer
                        ):
                            raise RuntimeError("Incomplete CacheBlend completion read")
                        for idx, obj in zip(indices, disk_objs_layer):
                            disk_objs_map[idx] = obj
                            to_count_down.append(obj)
                        stats = self.speculative_stats[request_id]
                        stats["completion_layers_submitted"] += 1
                        stats["completion_chunk_reads"] += len(indices)
                        logger.debug(
                            "CacheBlend completion %s: consume_layer=%d "
                            "submitted_layer=%d chunks=%d",
                            request_id, layer_id, layer_id, len(indices),
                        )
                elif layer_id < num_prefetch_layers and disk_tasks is not None:
                    # Phase 1: first num_prefetch_layers for miss chunks
                    task = disk_tasks[layer_id]
                    assert task is not None
                    _t_wait_start = time.perf_counter()
                    disk_objs_layer = task.result()
                    _disk_wait_times.append(
                        time.perf_counter() - _t_wait_start
                    )
                    for j, actual_idx in enumerate(disk_chunk_indices):
                        disk_objs_map[actual_idx] = disk_objs_layer[j]
                        to_count_down.append(disk_objs_layer[j])

                    # Phase 1 done for this layer.  When the first
                    # Phase 1 future resolves, all chunk reads are
                    # complete (the _transpose thread waits for all
                    # chunks before setting any layer future).  Submit
                    # Phase 2 now so it gets full disk bandwidth.
                    if (
                        not phase2_submitted
                        and need_remaining_layers
                        and phase2_params is not None
                    ):
                        phase2_submitted = True
                        remaining_disk_tasks = _submit_first_phase2_batch(
                            num_prefetch_layers,
                        )
                elif (
                    layer_id < num_prefetch_layers
                    and disk_tasks is None
                    and get_generator is not None
                ):
                    # Lazy path (pure CacheBlend, no speculative):
                    # submit per layer
                    _t_wait_start = time.perf_counter()
                    task = next(get_generator)
                    assert task is not None
                    disk_objs_layer = task.result()
                    _disk_wait_times.append(
                        time.perf_counter() - _t_wait_start
                    )
                    for j, actual_idx in enumerate(disk_chunk_indices):
                        disk_objs_map[actual_idx] = disk_objs_layer[j]
                        to_count_down.append(disk_objs_layer[j])
                elif not need_remaining_layers:
                    # Full prefetch or no partial layers: original paths
                    if disk_tasks is not None:
                        task = disk_tasks[layer_id]
                        assert task is not None
                        _t_wait_start = time.perf_counter()
                        disk_objs_layer = task.result()
                        _disk_wait_times.append(
                            time.perf_counter() - _t_wait_start
                        )
                        for j, actual_idx in enumerate(
                            disk_chunk_indices
                        ):
                            disk_objs_map[actual_idx] = (
                                disk_objs_layer[j]
                            )
                            to_count_down.append(disk_objs_layer[j])
                    elif get_generator is not None:
                        _t_wait_start = time.perf_counter()
                        task = next(get_generator)
                        assert task is not None
                        disk_objs_layer = task.result()
                        _disk_wait_times.append(
                            time.perf_counter() - _t_wait_start
                        )
                        for j, actual_idx in enumerate(
                            disk_chunk_indices
                        ):
                            disk_objs_map[actual_idx] = (
                                disk_objs_layer[j]
                            )
                            to_count_down.append(disk_objs_layer[j])

                # Get remaining-layer disk data for ALL chunks (Phase 2)
                remaining_objs_map: Dict[int, MemoryObj] = {}
                if (
                    need_remaining_layers
                    and layer_id >= num_prefetch_layers
                ):
                    rem_layer_idx = layer_id - num_prefetch_layers

                    # --- Pipelined next-batch trigger ---
                    # When we start consuming the current batch, submit
                    # the next batch so its I/O overlaps with inference.
                    if _use_optimal:
                        # Optimal: consume pre-computed plan entries
                        while (
                            _p2_optimal_idx < len(_p2_optimal_plan)
                            and rem_layer_idx
                            >= _p2_optimal_plan[_p2_optimal_idx][0]
                            and _p2_next_layer < self.num_layers
                        ):
                            _, new_sz = _p2_optimal_plan[
                                _p2_optimal_idx
                            ]
                            _p2_optimal_idx += 1
                            new_end = _p2_next_layer + new_sz
                            new_futures = self._submit_phase2_batch(
                                self.storage_manager,
                                _p2_next_layer,
                                new_end,
                                num_prefetch_layers,
                                phase2_params,
                            )
                            remaining_disk_tasks.extend(new_futures)
                            torch.cuda.nvtx.range_push(
                                f"phase2::batch{_p2_batch_num}_layers"
                                f"[{_p2_next_layer},{new_end})_wait"
                            )
                            _p2_batch_num += 1
                            logger.info(
                                "Speculative retrieve %s: Phase 2 "
                                "optimal batch [%d, %d) submitted",
                                request_id,
                                _p2_next_layer, new_end,
                            )
                            _p2_last_batch_sz = new_sz
                            _p2_next_layer = new_end
                    elif (
                        _use_pipeline
                        and rem_layer_idx == _p2_next_trigger
                        and _p2_next_layer < self.num_layers
                    ):
                        # Reactive: adaptive batch sizing
                        runway = _p2_last_batch_sz
                        new_remaining = self.num_layers - _p2_next_layer
                        # Cap at 3 batches total: beyond that, read all
                        # remaining to avoid 1-layer-at-a-time overhead.
                        if _p2_batch_num >= 3:
                            new_sz = new_remaining
                        else:
                            new_sz = self._compute_next_batch_size(
                                runway, new_remaining,
                            )
                        new_end = _p2_next_layer + new_sz
                        new_futures = self._submit_phase2_batch(
                            self.storage_manager,
                            _p2_next_layer,
                            new_end,
                            num_prefetch_layers,
                            phase2_params,
                        )
                        remaining_disk_tasks.extend(new_futures)
                        logger.info(
                            "Speculative retrieve %s: Phase 2 batch "
                            "[%d, %d) submitted (runway=%d, remaining=%d)",
                            request_id,
                            _p2_next_layer, new_end,
                            runway, new_remaining,
                        )
                        _p2_next_trigger += _p2_last_batch_sz
                        _p2_last_batch_sz = new_sz
                        _p2_next_layer = new_end

                    rem_task = remaining_disk_tasks[rem_layer_idx]
                    assert rem_task is not None
                    _t_wait_start = time.perf_counter()
                    rem_objs = rem_task.result()
                    _disk_wait_times.append(
                        time.perf_counter() - _t_wait_start
                    )

                    # NVTX: pop current batch wait on first resolve,
                    # push next batch wait if applicable.
                    if rem_layer_idx == _p2_batch_resolve_start:
                        torch.cuda.nvtx.range_pop()  # phase2::batchN wait
                        # If there are more layers beyond this batch,
                        # start the next batch's NVTX range.
                        next_batch_start = (
                            _p2_batch_resolve_start + _p2_last_batch_sz
                        )
                        if next_batch_start < _total_remaining:
                            abs_start = (
                                num_prefetch_layers + next_batch_start
                            )
                            torch.cuda.nvtx.range_push(
                                f"phase2::batch{_p2_batch_num}_layers"
                                f"[{abs_start},{_p2_next_layer})_wait"
                            )
                            _p2_batch_num += 1
                            _p2_batch_resolve_start = next_batch_start

                    # Record t_read from Phase 2 first-batch I/O time.
                    # _transpose records io_elapsed on the first layer
                    # future: the wall-clock time from submit to all
                    # chunk futures resolving (pure I/O, excludes any
                    # inference time between submit and .result()).
                    if (
                        not _p2_first_read_recorded
                        and hasattr(rem_task, "io_elapsed")
                    ):
                        _p2_first_read_recorded = True
                        _p2_io = rem_task.io_elapsed
                        _p2_per_layer = (
                            _p2_io / max(_p2_last_batch_sz, 1)
                        )
                        # Only record samples from batches with enough
                        # layers; small batches have high per-layer
                        # fixed overhead that skews the average.
                        if _p2_last_batch_sz >= 5:
                            self._pipeline_timing_stats[
                                "t_read_per_layer"
                            ].append(_p2_per_layer)
                        logger.info(
                            "Pipeline t_read sampled (Phase 2): "
                            "%.4fms/layer (%d chunks, %d layers, "
                            "%.1fms total%s)",
                            _p2_per_layer * 1000,
                            len(remaining_chunk_indices),
                            _p2_last_batch_sz,
                            _p2_io * 1000,
                            "" if _p2_last_batch_sz >= 5
                            else ", SKIPPED <5 layers",
                        )

                    for j, actual_idx in enumerate(
                        remaining_chunk_indices
                    ):
                        remaining_objs_map[actual_idx] = rem_objs[j]
                        to_count_down.append(rem_objs[j])

                torch.cuda.nvtx.range_pop()
                _nvtx_mod.end_range(_nvtx_diskwait_id)

                # Measure yield → resume (= caller's blend forward time)
                _pre_yield = time.perf_counter()
                if layer_id == 0:
                    # NOTE(Yuwei): For sglang integration we need to provide retrieved
                    # tokens number in the first layer loading since there is no lookup
                    yield torch.sum(ret_mask)
                else:
                    yield None
                _vllm_blend_time = time.perf_counter() - _pre_yield

                # Build merged MemoryObj list in original chunk order
                _merge_send_start = time.perf_counter()
                torch.cuda.nvtx.range_push(
                    f"retrieve::merge_and_send_layer_{layer_id}"
                )
                _nvtx_merge_id = _nvtx_mod.start_range(
                    message=f"retrieve::merge_and_send_layer_{layer_id}",
                    domain="lmcache",
                )
                _nvtx_build_id = _nvtx_mod.start_range(
                    message=f"retrieve::build_merged_layer_{layer_id}",
                    domain="lmcache",
                )
                merged_objs = []
                for i in range(len(keys)):
                    if (
                        need_remaining_layers
                        and layer_id >= num_prefetch_layers
                    ):
                        # Phase 2: ALL chunks use remaining_objs_map
                        mem_obj = remaining_objs_map[i]
                    elif i in spec_chunk_map and layer_id < num_prefetch_layers:
                        spec_idx = spec_chunk_map[i]
                        # Prefetched layer: use speculative data
                        if spec_result.gpu_buffers is not None:
                            mem_obj = make_gpu_memory_obj(
                                spec_result, spec_idx, layer_id,
                            )
                        else:
                            # CPU-only hit
                            mem_obj = spec_result.cpu_mem_objs[
                                spec_idx * num_prefetch_layers
                                + layer_id
                            ]
                    else:
                        mem_obj = disk_objs_map[i]
                    merged_objs.append(mem_obj)
                _nvtx_mod.end_range(_nvtx_build_id)

                _build_done = time.perf_counter()
                _nvtx_send_id = _nvtx_mod.start_range(
                    message=f"retrieve::consumer_send_layer_{layer_id}",
                    domain="lmcache",
                )
                mem_obj_consumer.send(merged_objs)
                _nvtx_mod.end_range(_nvtx_send_id)
                _send_done = time.perf_counter()
                torch.cuda.nvtx.range_pop()
                _nvtx_mod.end_range(_nvtx_merge_id)
                _nvtx_mod.end_range(_nvtx_layer_id)

                _per_layer_build_ms.append(
                    (_build_done - _merge_send_start) * 1000.0
                )
                _per_layer_send_ms.append(
                    (_send_done - _build_done) * 1000.0
                )

                # Record per-layer inference time: vLLM blend forward
                # + LMCache merge+send overhead.  This is the total
                # processing time per layer that provides "runway" for
                # overlapping disk I/O with compute.
                if need_remaining_layers:
                    _merge_send_time = _send_done - _merge_send_start
                    _blend_times.append(_vllm_blend_time)
                    _merge_times.append(_merge_send_time)
                    _infer_times.append(
                        _vllm_blend_time + _merge_send_time
                    )

            # Log per-layer disk I/O wait times (includes Python
            # scheduling overhead + actual disk I/O).  Compare with
            # backend-level disk_read logs to isolate scheduling cost.
            if _disk_wait_times:
                total_wait = sum(_disk_wait_times)
                avg_wait = total_wait / len(_disk_wait_times)
                max_wait = max(_disk_wait_times)
                min_wait = min(_disk_wait_times)
                logger.info(
                    "Disk I/O scheduling overhead: %d layers, "
                    "total=%.2fms, avg=%.4fms/layer, "
                    "min=%.4fms, max=%.4fms "
                    "(includes Python asyncio + serializer + "
                    "thread dispatch + disk I/O; subtract backend "
                    "disk_read time to get pure scheduling overhead)",
                    len(_disk_wait_times),
                    total_wait * 1000,
                    avg_wait * 1000,
                    min_wait * 1000,
                    max_wait * 1000,
                )

            # Record per-layer inference time for pipeline estimation
            if _infer_times:
                avg_infer = sum(_infer_times) / len(_infer_times)
                self._pipeline_timing_stats["t_infer_per_layer"].append(
                    avg_infer
                )
                avg_blend = (
                    sum(_blend_times) / len(_blend_times)
                    if _blend_times else 0
                )
                avg_merge = (
                    sum(_merge_times) / len(_merge_times)
                    if _merge_times else 0
                )
                logger.info(
                    "Pipeline t_infer sampled: %.4fms/layer "
                    "(blend=%.4fms + merge_send=%.4fms, "
                    "%d samples this request)",
                    avg_infer * 1000,
                    avg_blend * 1000,
                    avg_merge * 1000,
                    len(_infer_times),
                )

            if _per_layer_send_ms:
                _max_send = max(_per_layer_send_ms)
                _max_send_idx = _per_layer_send_ms.index(_max_send)
                _max_build = max(_per_layer_build_ms)
                _max_build_idx = _per_layer_build_ms.index(_max_build)
                _fill = getattr(
                    self.gpu_connector, "_last_pinned_fill_ms", None
                )
                _h2d = getattr(
                    self.gpu_connector, "_last_h2d_enqueue_ms", None
                )
                _d2d = getattr(
                    self.gpu_connector, "_last_d2d_enqueue_ms", None
                )
                _extra = ""
                if _fill is not None and _fill:
                    _max_fill = max(_fill)
                    _max_fill_idx = _fill.index(_max_fill)
                    _max_h2d = max(_h2d) if _h2d else 0.0
                    _max_h2d_idx = _h2d.index(_max_h2d) if _h2d else -1
                    _max_d2d = max(_d2d) if _d2d else 0.0
                    _max_d2d_idx = _d2d.index(_max_d2d) if _d2d else -1
                    _extra = (
                        f" | pinned_fill max={_max_fill:.3f}ms@L{_max_fill_idx}"
                        f" | h2d_enq max={_max_h2d:.3f}ms@L{_max_h2d_idx}"
                        f" | d2d_enq max={_max_d2d:.3f}ms@L{_max_d2d_idx}"
                    )
                logger.info(
                    "retrieve send breakdown L=%d | "
                    "send max=%.3fms@L%d | build max=%.3fms@L%d%s",
                    num_prefetch_layers,
                    _max_send, _max_send_idx,
                    _max_build, _max_build_idx,
                    _extra,
                )

            for mem_obj in to_count_down:
                mem_obj.ref_count_down()

            # Consume speculative result: free GPU buffer and remove
            # from cache.  By this point, torch.cuda.synchronize()
            # inside batched_to_gpu has ensured all copies from the
            # speculative buffer to the load buffer are complete.
            if spec_result is not None:
                spec_result.status = "consumed"
                # Layerwise baseline defers the stop-event wait from
                # snapshot time to here, so the ~5-10 ms worker
                # finalize overlaps with retrieve's own Phase-1/2 disk
                # reads instead of sitting in the TTFT critical path.
                # In the common case retrieve has spent tens of ms on
                # disk I/O by this point, so the wait is a no-op; it's
                # still needed for correctness because the worker may
                # otherwise have pending H2D into the GPU buffers
                # we're about to release back to the pool.
                if (
                    self.config.speculative_prefetch_layerwise_baseline
                ):
                    _finalize_t0 = time.monotonic()
                    try:
                        _max_wait = max(
                            0.0,
                            self.config
                            .speculative_prefetch_ttl_seconds
                            - (
                                time.monotonic()
                                - spec_result.created_at
                            ),
                        )
                        spec_result.future.result(timeout=_max_wait)
                    except Exception:
                        pass
                    _finalize_wait_ms = (
                        time.monotonic() - _finalize_t0
                    ) * 1000
                    # Backfill stats that were unavailable when the
                    # stats dict was first populated at snapshot time
                    # (worker hadn't finalized, so `e2e_time` and
                    # `worker_layers_issued` were still None).
                    _stats = self.speculative_stats.get(request_id)
                    if _stats is not None:
                        if spec_result.e2e_time is not None:
                            _stats["e2e_time_ms"] = round(
                                spec_result.e2e_time * 1000, 1,
                            )
                        if spec_result.worker_layers_issued is not None:
                            _stats["layerwise_layers_issued"] = (
                                spec_result.worker_layers_issued
                            )
                        _stats["layerwise_finalize_wait_ms"] = round(
                            _finalize_wait_ms, 3,
                        )
                    logger.info(
                        "Speculative retrieve %s [layerwise "
                        "baseline]: finalize wait %.2fms "
                        "(worker layers issued=%s)",
                        request_id,
                        _finalize_wait_ms,
                        spec_result.worker_layers_issued,
                    )
                if (
                    self.speculative_gpu_buffer is not None
                    and spec_result.gpu_buffers is not None
                ):
                    self.speculative_gpu_buffer.free_chunks(spec_result.gpu_buffers)
                    spec_result.gpu_buffers = None
                # Free CPU MemoryObjs from CPU-only fallback path.
                # By this point batched_to_gpu has finished all H2D
                # copies so the tensors are no longer needed.
                if spec_result.cpu_mem_objs is not None:
                    for obj in spec_result.cpu_mem_objs:
                        if obj is not None:
                            obj.ref_count_down()
                    spec_result.cpu_mem_objs = None
                with self.speculative_cache_lock:
                    self.speculative_cache.pop(spec_cache_key, None)
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        yield None

        # synchronize the last layer
        next(mem_obj_consumer)

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        if self.config.enable_blending:
            logger.info(
                f"Retrieved {retrieved_tokens} tokens "
                f"({num_hit_chunks}/{num_total_chunks} chunks hit) "
                f"out of {num_required_tokens} "
                f"out of total {len(tokens)} tokens"
            )
        else:
            logger.info(
                f"Retrieved {retrieved_tokens} "
                f"out of {num_required_tokens} "
                f"out of total {len(tokens)} tokens"
            )

        yield ret_mask

    @_lmcache_nvtx_annotate
    def speculative_prefetch(
        self,
        request_id: str,
        chunk_tokens_list: List[List[int]],
        num_prefetch_layers: Optional[int] = None,
    ) -> None:
        """Speculatively prefetch KV caches for candidate chunks from disk
        to GPU memory. This is called before the final prompt is ready,
        during ANN search, to hide disk I/O latency.

        Loads layers of all chunks from disk into CPU MemoryObjs,
        then copies them to a contiguous GPU buffer via a dedicated CUDA
        stream. CPU MemoryObjs are freed after the GPU copy completes.

        :param str request_id: Unique ID to correlate with future chat
            completion request.
        :param List[List[int]] chunk_tokens_list: Token IDs for each
            candidate chunk. Each chunk is processed independently to
            resolve CacheEngineKeys via token hashing.
        :param Optional[int] num_prefetch_layers: Number of layers to
            prefetch. None or 0 means all layers. A positive value
            limits prefetch to the first N layers; remaining layers are
            loaded during inference.
        """
        if self.completion_only:
            raise RuntimeError("Retrieval-time prefetch RPC is forbidden in completion-only control")
        if not self.config.speculative_prefetch_enabled:
            logger.warning(
                "speculative_prefetch called but feature is disabled"
            )
            return

        # Layerwise baseline path: dispatch to the alternative
        # implementation that reads layers strictly one-by-one and
        # exposes a `ready_num_prefetch_layers` counter for retrieve.
        if self.config.speculative_prefetch_layerwise_baseline:
            self._speculative_prefetch_layerwise(
                request_id, chunk_tokens_list, num_prefetch_layers,
            )
            return

        # Resolve effective number of layers to prefetch.
        # Priority: API param > config > all layers.
        effective_layers = (
            num_prefetch_layers
            or self.config.speculative_prefetch_num_layers
            or self.num_layers
        )
        effective_layers = min(effective_layers, self.num_layers)

        # Start a cross-thread NVTX range covering the entire prefetch
        # lifecycle (setup + async disk reads + GPU copy + cleanup).
        # nvtx.start_range / end_range are process-wide, not thread-local.
        _spec_nvtx_range_id = _nvtx_mod.start_range(
            message=f"spec_prefetch::e2e::{request_id}",
        )

        logger.info(
            "Speculative prefetch %s: %d chunks, %d/%d layers",
            request_id,
            len(chunk_tokens_list),
            effective_layers,
            self.num_layers,
        )

        # Step 1: Resolve CacheEngineKeys for each chunk via token hashing.
        # Each chunk is processed independently (same as pre-computation).
        # chunk_keys: list of [layer_idx] keys per valid chunk
        chunk_keys: List[List[CacheEngineKey]] = []
        chunk_chunked_keys: List = []  # ChunkedCacheEngineKey per chunk
        chunk_num_tokens: List[int] = []
        retrieve_backend = None

        for chunk_tokens in chunk_tokens_list:
            for _start, _end, key in self.token_database.process_tokens(
                tokens=chunk_tokens,
            ):
                assert isinstance(key, CacheEngineKey)
                keys_multi_layer = key.split_layers(effective_layers)

                # Check existence: chunked key for speculative mode
                chunked_key = key.to_chunked_key(self.num_layers)
                backend_name = self.storage_manager.contains(chunked_key)
                if not backend_name:
                    # Fallback: check per-layer key for backward compat
                    backend_name = self.storage_manager.contains(
                        keys_multi_layer[0]
                    )
                if not backend_name:
                    logger.info(
                        "Speculative prefetch %s: chunk miss "
                        "(chunk_hash=%s)",
                        request_id,
                        key.chunk_hash,
                    )
                    continue
                logger.info(
                    "Speculative prefetch %s: chunk hit "
                    "(chunk_hash=%s, backend=%s)",
                    request_id,
                    key.chunk_hash,
                    backend_name,
                )

                if retrieve_backend is None:
                    retrieve_backend = backend_name

                chunk_keys.append(keys_multi_layer)
                chunk_chunked_keys.append(chunked_key)
                chunk_num_tokens.append(_end - _start)

        if not chunk_keys:
            logger.info(
                "Speculative prefetch %s: no valid chunks found on disk",
                request_id,
            )
            _nvtx_mod.end_range(_spec_nvtx_range_id)
            return

        num_valid = len(chunk_keys)
        total_tokens = sum(chunk_num_tokens)
        logger.info(
            "Speculative prefetch %s: %d valid chunks, %d total tokens",
            request_id,
            num_valid,
            total_tokens,
        )

        # Pre-register the result in speculative_cache *now*, before the
        # slower CPU/GPU allocation and disk I/O setup.  This closes a
        # race window where retrieve_layer() can run before the cache
        # write at the bottom of this function, causing it to miss the
        # spec_result entirely (the entry's GPU buffer is then never
        # consumed and only freed by the 30s TTL cleanup, which in turn
        # starves the GPU buffer budget for subsequent prefetches).
        #
        # The result starts with gpu_buffers=None and status="loading".
        # retrieve_layer() accepts loading entries, builds the chunk
        # hash map from chunk_keys (already populated), and waits on
        # `future` until either the GPU buffer is filled in (set below
        # after allocation) or the prefetch errors out.
        aggregate_future = concurrent.futures.Future()
        result = SpeculativePrefetchResult(
            request_id=request_id,
            chunk_keys=chunk_keys,
            gpu_buffers=None,
            future=aggregate_future,
            cpu_mem_objs=None,
            num_prefetch_layers=effective_layers,
            status="loading",
        )
        with self.speculative_cache_lock:
            self.speculative_cache[request_id] = result

        # Step 2: Transpose to layer-major format [num_layers][num_chunks]
        keys_layer_major = [
            list(row) for row in zip(*chunk_keys, strict=False)
        ]

        # Step 4: Pre-allocate CPU MemoryObjs for disk read.
        # Compute per-layer KV shape from fmt (not from gpu_connector,
        # which may return the full multi-layer shape).
        kv_shape = self.metadata.kv_shape
        hidden_dim = kv_shape[3] * kv_shape[4]  # num_heads * head_size

        pre_allocated = None
        if retrieve_backend == "LocalDiskBackend":
            kv_dtype = self.metadata.kv_dtype
            chunk_mem_objs: list[list[MemoryObj]] = []
            for nt in chunk_num_tokens:
                if self.fmt == MemoryFormat.KV_2TD:
                    shape = torch.Size([2, nt, hidden_dim])
                elif self.fmt == MemoryFormat.KV_T2D:
                    shape = torch.Size([nt, 2, hidden_dim])
                else:
                    shape = self.gpu_connector.get_shape(nt)
                objs = self.storage_manager.batched_allocate(
                    shape,
                    kv_dtype,
                    batch_size=effective_layers,
                    fmt=self.fmt,
                )
                if objs is None:
                    logger.warning(
                        "Speculative prefetch %s: CPU memory allocation "
                        "failed, aborting",
                        request_id,
                    )
                    # Clean up the pre-registered cache entry so
                    # retrieve_layer doesn't wait on a future that
                    # nobody will set.
                    result.status = "expired"
                    aggregate_future.set_result(None)
                    with self.speculative_cache_lock:
                        self.speculative_cache.pop(request_id, None)
                    _nvtx_mod.end_range(_spec_nvtx_range_id)
                    return
                chunk_mem_objs.append(objs)
            # Transpose to layer-major: [num_layers][num_chunks]
            pre_allocated = [
                list(row) for row in zip(*chunk_mem_objs, strict=False)
            ]

        # Step 5: Pre-allocate per-chunk GPU buffers BEFORE disk reads
        # so that per-chunk callbacks can copy to GPU immediately.
        # Each buffer is [effective_layers, 2, chunk_tokens, hidden_dim],
        # contiguous per chunk for a single H2D copy.
        num_chunks = num_valid
        gpu_bufs = None
        if self.speculative_gpu_buffer is not None:
            gpu_bufs = self.speculative_gpu_buffer.allocate_chunks(
                num_layers=effective_layers,
                chunk_num_tokens=chunk_num_tokens,
                hidden_dim=hidden_dim,
            )
            if gpu_bufs is None:
                logger.warning(
                    "Speculative prefetch %s: GPU budget exceeded, "
                    "will keep CPU MemoryObjs only",
                    request_id,
                )
            else:
                # Publish the freshly allocated buffers on the
                # pre-registered result so that any concurrent
                # retrieve_layer that already grabbed `result` can read
                # from GPU instead of falling into the CPU-only path.
                result.gpu_buffers = gpu_bufs

        # Step 6: Submit per-chunk parallel disk reads.
        # Each chunk becomes an independent task in the thread pool,
        # enabling up to max_workers concurrent disk reads.
        # Use chunked keys if the backend has chunked files
        use_chunked = (
            retrieve_backend == "LocalDiskBackend"
            and chunk_chunked_keys
            and self.storage_manager.contains(chunk_chunked_keys[0])
        )
        chunk_futures = self.storage_manager.batched_get_per_chunk(
            keys_layer_major,
            location=retrieve_backend,
            pre_allocated_mem_objs=pre_allocated,
            chunked_keys=chunk_chunked_keys if use_chunked else None,
        )

        # Note: `result` and `aggregate_future` were created and
        # registered in `speculative_cache` near the top of this
        # function (right after chunk resolution) to avoid a race with
        # retrieve_layer().  `result.gpu_buffers` was populated above
        # if GPU allocation succeeded.

        # Step 8: Register per-chunk callbacks for pipelined GPU copy.
        # As each chunk's disk read completes, immediately copy that
        # chunk's layers to the GPU buffer without waiting for others.
        chunks_remaining = [num_chunks]
        chunks_lock = threading.Lock()
        # Collect all CPU MemoryObjs for cleanup after GPU copy
        all_cpu_mem_objs: List[Optional[list]] = [None] * num_chunks
        had_error = [False]

        def _on_all_chunks_done():
            """Called by the last chunk callback to finalize."""
            torch.cuda.nvtx.range_push(
                "spec_prefetch::all_chunks_done"
            )
            # Flatten all CPU MemoryObjs for cleanup
            flat_objs = [
                obj
                for chunk_objs in all_cpu_mem_objs
                if chunk_objs is not None
                for obj in chunk_objs
                if obj is not None
            ]

            if had_error[0]:
                # At least one chunk failed — free CPU objs, mark expired
                for obj in flat_objs:
                    obj.ref_count_down()
                if gpu_bufs is not None:
                    result.gpu_buffers = None
                    self.speculative_gpu_buffer.free_chunks(gpu_bufs)
                result.status = "expired"
                aggregate_future.set_result(None)
                _nvtx_mod.end_range(_spec_nvtx_range_id)
                torch.cuda.nvtx.range_pop()  # all_chunks_done
                return

            if gpu_bufs is not None:
                # All chunks copied to GPU — wait for stream, then
                # free CPU MemoryObjs.
                event = self.spec_load_stream.record_event()

                def _on_gpu_copy_done(ev=event, objs=flat_objs):
                    torch.cuda.nvtx.range_push(
                        "spec_prefetch::gpu_sync_and_cleanup"
                    )
                    ev.synchronize()
                    for obj in objs:
                        obj.ref_count_down()
                    result.cpu_mem_objs = None
                    result.e2e_time = time.monotonic() - result.created_at
                    result.status = "ready"
                    aggregate_future.set_result(True)
                    logger.info(
                        "Speculative prefetch %s: all %d chunks "
                        "pipelined to GPU, CPU MemoryObjs freed",
                        request_id,
                        num_chunks,
                    )
                    _nvtx_mod.end_range(_spec_nvtx_range_id)
                    torch.cuda.nvtx.range_pop()  # gpu_sync_and_cleanup

                self.storage_manager.worker_pool.submit(
                    _on_gpu_copy_done
                )
            else:
                # CPU-only fallback
                result.cpu_mem_objs = flat_objs
                result.e2e_time = time.monotonic() - result.created_at
                result.status = "ready"
                aggregate_future.set_result(True)
                _nvtx_mod.end_range(_spec_nvtx_range_id)

            torch.cuda.nvtx.range_pop()  # all_chunks_done

        for chunk_idx, chunk_future in enumerate(chunk_futures):

            def _on_chunk_read_done(fut, cidx=chunk_idx):
                try:
                    torch.cuda.nvtx.range_push(
                        f"spec_prefetch::chunk{cidx}_disk_read_done"
                    )
                    _nvtx_drd_id = _nvtx_mod.start_range(
                        message=(
                            f"spec_prefetch::chunk{cidx}_disk_read_done"
                        ),
                        domain="lmcache",
                    )
                    mem_objs = fut.result()  # List[MemoryObj], one per layer
                    all_cpu_mem_objs[cidx] = mem_objs
                    logger.info(
                        "Speculative prefetch %s: chunk %d/%d disk "
                        "read done (%d layers)",
                        request_id,
                        cidx,
                        num_chunks,
                        len(mem_objs),
                    )

                    # Pipeline: immediately copy this chunk to GPU.
                    # Pre-allocate a single destination buffer and copy
                    # each layer directly into it, avoiding per-layer
                    # contiguous() allocations and the torch.stack copy.
                    if gpu_bufs is not None:
                        torch.cuda.nvtx.range_push(
                            f"spec_prefetch::chunk{cidx}_cpu_to_gpu"
                        )
                        _nvtx_c2g_id = _nvtx_mod.start_range(
                            message=(
                                f"spec_prefetch::chunk{cidx}_cpu_to_gpu"
                            ),
                            domain="lmcache",
                        )
                        # Find valid layers and determine target shape
                        valid_srcs = []
                        for mem_obj in mem_objs:
                            if mem_obj is None:
                                continue
                            valid_srcs.append(mem_obj.tensor)

                        if valid_srcs:
                            torch.cuda.nvtx.range_push(
                                f"spec_prefetch::chunk{cidx}_alloc_and_copy"
                            )
                            _nvtx_ac_id = _nvtx_mod.start_range(
                                message=(
                                    f"spec_prefetch::chunk{cidx}_alloc_and_copy"
                                ),
                                domain="lmcache",
                            )
                            src0 = valid_srcs[0]
                            if self.fmt == MemoryFormat.KV_T2D:
                                # src shape [A, B, C] → target [B, A, C]
                                target_shape = (
                                    len(valid_srcs),
                                    src0.shape[1],
                                    src0.shape[0],
                                    src0.shape[2],
                                )
                            else:
                                target_shape = (
                                    len(valid_srcs),
                                    *src0.shape,
                                )
                            _nvtx_alloc_id = _nvtx_mod.start_range(
                                message=(
                                    f"spec_prefetch::chunk{cidx}_torch_empty"
                                ),
                                domain="lmcache",
                            )
                            cpu_stacked = torch.empty(
                                target_shape, dtype=src0.dtype,
                            )
                            _nvtx_mod.end_range(_nvtx_alloc_id)
                            _nvtx_copy_id = _nvtx_mod.start_range(
                                message=(
                                    f"spec_prefetch::chunk{cidx}_torch_copy_loop"
                                ),
                                domain="lmcache",
                            )
                            for i, src in enumerate(valid_srcs):
                                if self.fmt == MemoryFormat.KV_T2D:
                                    cpu_stacked[i].copy_(
                                        src.permute(1, 0, 2)
                                    )
                                else:
                                    cpu_stacked[i].copy_(src)
                            _nvtx_mod.end_range(_nvtx_copy_id)
                            torch.cuda.nvtx.range_pop()  # alloc_and_copy
                            _nvtx_mod.end_range(_nvtx_ac_id)

                            torch.cuda.nvtx.range_push(
                                f"spec_prefetch::chunk{cidx}_pin_memory"
                            )
                            _nvtx_pin_id = _nvtx_mod.start_range(
                                message=(
                                    f"spec_prefetch::chunk{cidx}_pin_memory"
                                ),
                                domain="lmcache",
                            )
                            cpu_stacked = cpu_stacked.pin_memory()
                            _nvtx_mod.end_range(_nvtx_pin_id)
                            torch.cuda.nvtx.range_pop()  # pin_memory
                            torch.cuda.nvtx.range_push(
                                f"spec_prefetch::chunk{cidx}_h2d_copy"
                            )
                            _nvtx_h2d_id = _nvtx_mod.start_range(
                                message=(
                                    f"spec_prefetch::chunk{cidx}_h2d_copy"
                                ),
                                domain="lmcache",
                            )
                            with torch.cuda.stream(
                                self.spec_load_stream
                            ):
                                gpu_bufs[cidx].copy_(
                                    cpu_stacked, non_blocking=True
                                )
                            _nvtx_mod.end_range(_nvtx_h2d_id)
                            torch.cuda.nvtx.range_pop()  # h2d_copy
                        torch.cuda.nvtx.range_pop()  # chunk cpu_to_gpu
                        _nvtx_mod.end_range(_nvtx_c2g_id)

                    torch.cuda.nvtx.range_pop()  # chunk disk_read_done
                    _nvtx_mod.end_range(_nvtx_drd_id)

                except Exception as e:
                    had_error[0] = True
                    logger.error(
                        "Speculative prefetch %s: chunk %d disk "
                        "read failed: %s",
                        request_id,
                        cidx,
                        e,
                    )

                # Decrement counter; last chunk triggers finalization
                with chunks_lock:
                    chunks_remaining[0] -= 1
                    is_last = chunks_remaining[0] == 0

                if is_last:
                    _on_all_chunks_done()

            chunk_future.add_done_callback(_on_chunk_read_done)

    def _speculative_prefetch_layerwise(
        self,
        request_id: str,
        chunk_tokens_list: List[List[int]],
        num_prefetch_layers: Optional[int] = None,
    ) -> None:
        """Layerwise baseline implementation of speculative prefetch.

        Reads layers strictly one at a time in ascending order. For
        each layer the worker submits a single-layer disk read, then
        issues per-chunk H2D ``copy_(non_blocking=True)`` on the
        dedicated speculative load stream and **records (but does not
        synchronize on) a per-layer event**.  Completed events are
        drained opportunistically via non-blocking ``event.query()``
        between iterations to release CPU MemoryObjs and advance
        ``ready_num_prefetch_layers``.  A single ``synchronize()`` on
        the last recorded event runs at finalize time, which on a
        single stream guarantees every prior H2D has drained before
        the result becomes ``ready``.

        ``retrieve_layer()`` snapshots ``ready_num_prefetch_layers`` on
        arrival, asks the worker to stop via ``stop_event``, and treats
        the first L ready layers as partial-layer speculative hits —
        falling back to the existing Phase 2 disk path for the
        remainder.  Because the worker performs exactly one CPU-side
        sync (at finalize), the consumer that calls
        ``spec_result.future.result()`` is guaranteed to observe all
        committed GPU writes by the time it resumes.

        This is a baseline meant to demonstrate why the chunk-major
        default is preferable; do not enable in production.
        """
        # Resolve effective number of layers (used as the upper bound
        # for the worker loop). Same priority chain as the chunk-major
        # path so behaviour is consistent across modes.
        effective_layers = (
            num_prefetch_layers
            or self.config.speculative_prefetch_num_layers
            or self.num_layers
        )
        effective_layers = min(effective_layers, self.num_layers)

        _spec_nvtx_range_id = _nvtx_mod.start_range(
            message=f"spec_prefetch_layerwise::e2e::{request_id}",
        )

        logger.info(
            "Speculative prefetch %s [layerwise baseline]: "
            "%d chunks, upper-bound %d/%d layers",
            request_id,
            len(chunk_tokens_list),
            effective_layers,
            self.num_layers,
        )

        # Step 1: Resolve CacheEngineKeys (identical to chunk-major).
        chunk_keys: List[List[CacheEngineKey]] = []
        chunk_chunked_keys: List = []
        chunk_num_tokens: List[int] = []
        retrieve_backend: Optional[str] = None

        for chunk_tokens in chunk_tokens_list:
            for _start, _end, key in self.token_database.process_tokens(
                tokens=chunk_tokens,
            ):
                assert isinstance(key, CacheEngineKey)
                keys_multi_layer = key.split_layers(effective_layers)

                chunked_key = key.to_chunked_key(self.num_layers)
                backend_name = self.storage_manager.contains(chunked_key)
                if not backend_name:
                    backend_name = self.storage_manager.contains(
                        keys_multi_layer[0]
                    )
                if not backend_name:
                    logger.info(
                        "Speculative prefetch %s [layerwise]: chunk miss "
                        "(chunk_hash=%s)",
                        request_id,
                        key.chunk_hash,
                    )
                    continue
                logger.info(
                    "Speculative prefetch %s [layerwise]: chunk hit "
                    "(chunk_hash=%s, backend=%s)",
                    request_id,
                    key.chunk_hash,
                    backend_name,
                )

                if retrieve_backend is None:
                    retrieve_backend = backend_name

                chunk_keys.append(keys_multi_layer)
                chunk_chunked_keys.append(chunked_key)
                chunk_num_tokens.append(_end - _start)

        if not chunk_keys:
            logger.info(
                "Speculative prefetch %s [layerwise]: no valid chunks "
                "found on disk",
                request_id,
            )
            _nvtx_mod.end_range(_spec_nvtx_range_id)
            return

        num_chunks = len(chunk_keys)
        logger.info(
            "Speculative prefetch %s [layerwise]: %d valid chunks, "
            "%d total tokens",
            request_id,
            num_chunks,
            sum(chunk_num_tokens),
        )

        # Decide whether to use the chunked-key fast path (one file per
        # chunk, layers via pread) or the per-layer key path (one file
        # per (chunk, layer)). The disk layout depends on how chunks
        # were originally written.
        use_chunked = (
            retrieve_backend == "LocalDiskBackend"
            and bool(chunk_chunked_keys)
            and self.storage_manager.contains(chunk_chunked_keys[0])
        )
        # Pre-build layer-major key matrix for the per-layer fallback.
        keys_layer_major: Optional[List[List[CacheEngineKey]]] = None
        if not use_chunked:
            keys_layer_major = [
                list(row) for row in zip(*chunk_keys, strict=False)
            ]

        # Pre-register the result so retrieve_layer() can find it
        # immediately, even before the worker has run any layer.
        aggregate_future: concurrent.futures.Future = (
            concurrent.futures.Future()
        )
        result = SpeculativePrefetchResult(
            request_id=request_id,
            chunk_keys=chunk_keys,
            gpu_buffers=None,
            future=aggregate_future,
            cpu_mem_objs=None,
            num_prefetch_layers=effective_layers,  # upper bound
            status="loading",
        )
        # ready_num_prefetch_layers starts at 0; stop_event default unset.
        with self.speculative_cache_lock:
            self.speculative_cache[request_id] = result

        # Step 2: Pre-allocate CPU MemoryObjs once for all layers.
        # Layout: [effective_layers][num_chunks], same as
        # layerwise_get_chunked_partial expects.
        kv_shape = self.metadata.kv_shape
        hidden_dim = kv_shape[3] * kv_shape[4]
        kv_dtype = self.metadata.kv_dtype

        pre_allocated_layer_major: Optional[List[List[MemoryObj]]] = None
        if retrieve_backend == "LocalDiskBackend":
            # Build all per-chunk shapes up front so we can attempt a
            # single bulk allocation. Per-chunk batched_allocate held
            # host_mem_lock once per chunk and incurred a 5-deep
            # pybind frame per call; under load the allocator pool's
            # first-fit search becomes a GIL-stealing bottleneck
            # (NVTX trace: TensorMemoryAllocator.batched_allocate
            # tail latency hit 21 ms post-cliff). The chunks variant
            # collapses the loop into a single first-fit search held
            # under one lock acquire.
            chunk_shapes: list[torch.Size] = []
            for nt in chunk_num_tokens:
                if self.fmt == MemoryFormat.KV_2TD:
                    chunk_shapes.append(torch.Size([2, nt, hidden_dim]))
                elif self.fmt == MemoryFormat.KV_T2D:
                    chunk_shapes.append(torch.Size([nt, 2, hidden_dim]))
                else:
                    chunk_shapes.append(self.gpu_connector.get_shape(nt))

            chunk_mem_objs: Optional[list[list[MemoryObj]]] = (
                self.storage_manager.batched_allocate_chunks(
                    chunk_shapes,
                    kv_dtype,
                    batch_size=effective_layers,
                    fmt=self.fmt,
                )
            )
            if chunk_mem_objs is None:
                # Fragmented pool or eviction needed — fall back to the
                # original per-chunk loop so the eviction retry path in
                # LocalCPUBackend.batched_allocate still runs.
                chunk_mem_objs = []
                for shape in chunk_shapes:
                    objs = self.storage_manager.batched_allocate(
                        shape,
                        kv_dtype,
                        batch_size=effective_layers,
                        fmt=self.fmt,
                    )
                    if objs is None:
                        logger.warning(
                            "Speculative prefetch %s [layerwise]: CPU "
                            "memory allocation failed, aborting",
                            request_id,
                        )
                        result.status = "expired"
                        aggregate_future.set_result(None)
                        with self.speculative_cache_lock:
                            self.speculative_cache.pop(request_id, None)
                        _nvtx_mod.end_range(_spec_nvtx_range_id)
                        return
                    chunk_mem_objs.append(objs)
            # Transpose to layer-major: [effective_layers][num_chunks]
            pre_allocated_layer_major = [
                list(row) for row in zip(*chunk_mem_objs, strict=False)
            ]

        # Step 3: Pre-allocate per-chunk GPU buffers (full upper bound).
        gpu_bufs: Optional[List[torch.Tensor]] = None
        if self.speculative_gpu_buffer is not None:
            gpu_bufs = self.speculative_gpu_buffer.allocate_chunks(
                num_layers=effective_layers,
                chunk_num_tokens=chunk_num_tokens,
                hidden_dim=hidden_dim,
            )
            if gpu_bufs is None:
                logger.warning(
                    "Speculative prefetch %s [layerwise]: GPU budget "
                    "exceeded, falling back to CPU-only mode",
                    request_id,
                )
            else:
                result.gpu_buffers = gpu_bufs

        # Used by the CPU-only fallback path. We populate it in
        # chunk-major order so retrieve_layer's existing index formula
        # `spec_idx * num_prefetch_layers + layer_id` works as-is
        # (after retrieve overrides num_prefetch_layers to L).
        cpu_only_layer_objs: List[List[Optional[MemoryObj]]] = (
            [[None] * effective_layers for _ in range(num_chunks)]
        )

        # Precompute per-layer (srcs, dsts) tensor-view lists for the
        # H2D foreach_copy_ call, and allocate per-chunk PINNED staging
        # buffers so the H2D goes through the pinned→GPU DMA path
        # instead of pageable→GPU bounce buffers.  The pageable→GPU
        # scatter pattern (61 × _foreach_copy_ from pageable sources)
        # leaves CUDA driver state (bounce buffer mappings, DMA
        # descriptors) that causes 12-19 ms h2d_enq spikes in the
        # subsequent retrieve path.  Pinning matches the chunkwise
        # worker strategy and eliminates the tail.
        precomputed_h2d: List[
            Tuple[List[torch.Tensor], List[torch.Tensor]]
        ] = []
        # Per-chunk pinned staging: allocated with the ORIGINAL tensor
        # shape (before any KV_T2D permute) so that flat libc memcpy
        # from the pageable MemoryObj is valid.  A permuted VIEW of
        # the pinned buffer is passed to _foreach_copy_.
        _pinned_staging_chunks: List[torch.Tensor] = []
        # Per-layer memcpy descriptors: (dst_ptr, src_ptr, nbytes) per
        # chunk, precomputed so the worker loop body is pure C calls.
        _memcpy_descs: List[
            List[Tuple[int, int, int]]
        ] = []  # [num_layers][num_chunks] of (dst, src, nbytes)
        if gpu_bufs is not None and pre_allocated_layer_major is not None:
            # Allocate pinned staging per chunk — ORIGINAL shape
            for _cidx in range(num_chunks):
                _sample = pre_allocated_layer_major[0][_cidx]
                if _sample is None:
                    _pinned_staging_chunks.append(
                        torch.empty(0, pin_memory=True)
                    )
                    continue
                _pinned_staging_chunks.append(
                    torch.empty(_sample.tensor.shape,
                                dtype=_sample.tensor.dtype,
                                pin_memory=True)
                )

            for _L in range(effective_layers):
                _pinned_srcs: List[torch.Tensor] = []
                _dsts: List[torch.Tensor] = []
                _descs: List[Tuple[int, int, int]] = []
                for _cidx in range(num_chunks):
                    _mo = pre_allocated_layer_major[_L][_cidx]
                    if _mo is None:
                        continue
                    _pinned = _pinned_staging_chunks[_cidx]
                    # src for _foreach_copy_: apply permute on
                    # the pinned buffer (view, no data copy).
                    _pinned_src = (
                        _pinned.permute(1, 0, 2)
                        if self.fmt == MemoryFormat.KV_T2D
                        else _pinned
                    )
                    _pinned_srcs.append(_pinned_src)
                    _dsts.append(gpu_bufs[_cidx][_L])
                    # memcpy: flat copy of raw bytes in original
                    # layout from pageable MemoryObj → pinned buf.
                    _descs.append((
                        _pinned.data_ptr(),
                        _mo.tensor.data_ptr(),
                        _mo.tensor.nelement()
                        * _mo.tensor.element_size(),
                    ))
                precomputed_h2d.append((_pinned_srcs, _dsts))
                _memcpy_descs.append(_descs)

        # Disk-pipeline lookahead. K layers' reads are kept in flight
        # at any time so the asyncio loop can interleave preads across
        # layers without the submit-wait-submit gap of the strict
        # sequential version.
        LOOKAHEAD = 4

        def _layerwise_worker():
            torch.cuda.nvtx.range_push(
                f"spec_prefetch_layerwise::worker::{request_id}"
            )
            had_error = False
            layers_issued = 0
            # Per-layer disk-read futures. Slot L holds the Future
            # returned from layerwise_get_chunked_partial / batched_get
            # for layer L (or None if not yet submitted / already
            # collected). Used to drive the lookahead window and to
            # drain in-flight reads on early stop.
            io_futures: List[Optional[concurrent.futures.Future]] = (
                [None] * effective_layers
            )

            def _submit_layer(L: int) -> None:
                """Issue a single-layer disk read for layer L and stash
                the future in io_futures[L]. No-op if L is out of range
                or already submitted."""
                if L >= effective_layers or io_futures[L] is not None:
                    return
                per_layer_pre = (
                    [pre_allocated_layer_major[L]]
                    if pre_allocated_layer_major is not None
                    else None
                )
                if use_chunked:
                    # Sync path: bypasses the LMCache asyncio loop
                    # entirely. Measured to remove ~1.2 s of GIL hold
                    # from the asyncio-loop thread per 5 s prefetch
                    # window (65% of the worker's GIL wait time).
                    futs = (
                        self.storage_manager
                        .layerwise_get_chunked_partial_sync(
                            chunk_chunked_keys,
                            start_layer=L,
                            end_layer=L + 1,
                            location=retrieve_backend,
                            pre_allocated_mem_objs=per_layer_pre,
                        )
                    )
                else:
                    single_layer_keys = [keys_layer_major[L]]
                    futs = (
                        self.storage_manager.layerwise_batched_get(
                            single_layer_keys,
                            location=retrieve_backend,
                            pre_allocated_mem_objs=per_layer_pre,
                            eager=True,
                        )
                    )
                io_futures[L] = futs[0]

            # Pending H2D events awaiting GPU completion. Each entry
            # is (event, layer_objs, L) in issue order. Single stream
            # guarantees FIFO completion, so we can drain from the
            # left without worrying about out-of-order events.
            pending_events: deque = deque()

            def _drain_completed():
                """Non-blocking drain of finished H2D events.

                Called between iterations to release pinned CPU memory
                and advance ``ready_num_prefetch_layers`` as soon as
                the GPU confirms layers are written.  Never blocks the
                CPU — uses ``event.query()``.
                """
                while pending_events and pending_events[0][0].query():
                    _ev, _objs, _L = pending_events.popleft()
                    for mo in _objs:
                        if mo is not None:
                            mo.ref_count_down()
                    # Single-stream FIFO ⇒ contiguous publish is safe.
                    result.ready_num_prefetch_layers = _L + 1

            # Prime the disk pipeline: launch the first LOOKAHEAD layers
            # so the asyncio loop can interleave their preads from the
            # very first iteration.
            for _L_prime in range(min(LOOKAHEAD, effective_layers)):
                _submit_layer(_L_prime)

            try:
                for L in range(effective_layers):
                    if result.stop_event.is_set():
                        logger.info(
                            "Speculative prefetch %s [layerwise]: "
                            "stop_event fired, exiting at layer %d/%d",
                            request_id, L, effective_layers,
                        )
                        break

                    torch.cuda.nvtx.range_push(
                        f"spec_prefetch_layerwise::layer{L}_read"
                    )
                    # Collect this layer (its read was submitted either
                    # during priming or by a previous iteration's
                    # lookahead refill).
                    fut = io_futures[L]
                    assert fut is not None, (
                        f"layer {L} future missing — lookahead refill bug"
                    )
                    layer_objs = fut.result()  # List[MemoryObj]
                    io_futures[L] = None
                    torch.cuda.nvtx.range_pop()  # layer read

                    # Refill the lookahead window before issuing H2D so
                    # the disk worker keeps making progress on the next
                    # layer while we copy this one. Skip if stop fired
                    # in the meantime — the in-flight ones will still
                    # need draining at finalize, but we avoid widening
                    # the waste.
                    next_to_submit = L + LOOKAHEAD
                    if (
                        next_to_submit < effective_layers
                        and not result.stop_event.is_set()
                    ):
                        _submit_layer(next_to_submit)

                    if gpu_bufs is not None:
                        torch.cuda.nvtx.range_push(
                            f"spec_prefetch_layerwise::layer{L}_h2d_issue"
                        )
                        # Pre-computed (srcs, dsts) views: the hot
                        # loop does a few pybind calls per layer
                        # (memcpy + stream ctx + foreach_copy_ +
                        # record_event) — empirically the worker is
                        # GIL-starved and each extra pybind call in
                        # the loop body adds a GIL-reacquire stall.
                        # Fallback (non-LocalDiskBackend path) has no
                        # precompute because MemoryObjs are only
                        # materialised inside the disk future.
                        if precomputed_h2d:
                            srcs, dsts = precomputed_h2d[L]
                            # Copy pageable MemoryObj data → pinned
                            # staging via libc memcpy (GIL-free
                            # during the actual copy). This makes the
                            # subsequent _foreach_copy_ go through
                            # the pinned→GPU DMA path instead of the
                            # pageable→GPU bounce-buffer path that
                            # pollutes CUDA driver state and causes
                            # 12-19ms h2d_enq spikes in retrieve.
                            for _dst, _src, _nb in _memcpy_descs[L]:
                                _libc.memcpy(_dst, _src, _nb)
                        else:
                            srcs = []
                            dsts = []
                            for cidx, mo in enumerate(layer_objs):
                                if mo is None:
                                    continue
                                src = mo.tensor
                                if self.fmt == MemoryFormat.KV_T2D:
                                    src = src.permute(1, 0, 2)
                                srcs.append(src)
                                dsts.append(gpu_bufs[cidx][L])
                        with torch.cuda.stream(self.spec_load_stream):
                            if dsts:
                                torch._foreach_copy_(
                                    dsts, srcs, non_blocking=True,
                                )
                            layer_event = (
                                self.spec_load_stream.record_event()
                            )
                        torch.cuda.nvtx.range_pop()  # h2d_issue

                        pending_events.append(
                            (layer_event, layer_objs, L)
                        )
                    else:
                        # CPU-only fallback: keep MemoryObjs alive,
                        # store in chunk-major order for retrieve.
                        # No GPU writes ⇒ publish immediately.
                        for cidx, mo in enumerate(layer_objs):
                            cpu_only_layer_objs[cidx][L] = mo
                        result.ready_num_prefetch_layers = L + 1

                    layers_issued = L + 1
                    logger.debug(
                        "Speculative prefetch %s [layerwise]: "
                        "layer %d issued",
                        request_id, L,
                    )

                    # Opportunistic non-blocking drain. Cheap if no
                    # events are ready (a few event.query() calls).
                    if pending_events:
                        _drain_completed()
            except Exception as e:
                had_error = True
                logger.error(
                    "Speculative prefetch %s [layerwise]: worker "
                    "failed at layer %d: %s",
                    request_id, layers_issued, e,
                )

            # ---- Finalisation ----
            torch.cuda.nvtx.range_push(
                "spec_prefetch_layerwise::finalize"
            )

            # Drain remaining pending events with ONE blocking sync on
            # the last event. Single-stream FIFO ⇒ syncing the tail
            # guarantees every prior H2D copy on spec_load_stream has
            # completed and is visible to subsequent CUDA work on any
            # stream.  This is the *only* CPU-side sync in the worker.
            if pending_events:
                torch.cuda.nvtx.range_push(
                    "spec_prefetch_layerwise::final_sync"
                )
                try:
                    pending_events[-1][0].synchronize()
                except Exception as exc:
                    logger.warning(
                        "Speculative prefetch %s [layerwise]: final "
                        "event sync failed: %s",
                        request_id, exc,
                    )
                    had_error = True
                # Drain everything (events are FIFO-complete on a
                # single stream, so all are now done).
                while pending_events:
                    _ev, _objs, _L = pending_events.popleft()
                    for mo in _objs:
                        if mo is not None:
                            try:
                                mo.ref_count_down()
                            except Exception:
                                pass
                    if not had_error:
                        result.ready_num_prefetch_layers = _L + 1
                torch.cuda.nvtx.range_pop()  # final_sync

            # Drain any in-flight disk reads from the lookahead window
            # that we never collected (early-stop or exception). The
            # async drivers are still writing into the pre-allocated
            # CPU buffers, so we MUST wait for them before freeing
            # those buffers below.  We can't actually cancel them at
            # the storage backend, so this just bounds the wasted I/O
            # to LOOKAHEAD-1 layers in the worst case.
            inflight_drained = 0
            for L_inflight in range(effective_layers):
                fut_inflight = io_futures[L_inflight]
                if fut_inflight is None:
                    continue
                try:
                    fut_inflight.result()
                    inflight_drained += 1
                except Exception as exc:
                    logger.warning(
                        "Speculative prefetch %s [layerwise]: in-flight "
                        "layer %d drain failed: %s",
                        request_id, L_inflight, exc,
                    )
                io_futures[L_inflight] = None
            if inflight_drained:
                logger.debug(
                    "Speculative prefetch %s [layerwise]: drained %d "
                    "in-flight layers after stop",
                    request_id, inflight_drained,
                )

            # Free pre-allocated CPU MemoryObjs for layers we never
            # got to issue (early-stop or failure).
            if pre_allocated_layer_major is not None:
                for L_unread in range(layers_issued, effective_layers):
                    for obj in pre_allocated_layer_major[L_unread]:
                        if obj is not None:
                            try:
                                obj.ref_count_down()
                            except Exception:
                                pass

            if had_error:
                if (
                    gpu_bufs is not None
                    and self.speculative_gpu_buffer is not None
                ):
                    self.speculative_gpu_buffer.free_chunks(gpu_bufs)
                    result.gpu_buffers = None
                result.status = "expired"
                aggregate_future.set_result(None)
            else:
                if gpu_bufs is None:
                    # CPU-only: flatten in chunk-major order so
                    # retrieve_layer's existing index formula
                    # `spec_idx * num_prefetch_layers + layer_id`
                    # works as-is (after retrieve overrides
                    # num_prefetch_layers to L).
                    flat: List[Optional[MemoryObj]] = []
                    for cidx in range(num_chunks):
                        for L_idx in range(layers_issued):
                            flat.append(
                                cpu_only_layer_objs[cidx][L_idx]
                            )
                    result.cpu_mem_objs = flat

                # Adjust the published num_prefetch_layers to reflect
                # how many layers we actually filled. retrieve_layer
                # may further reduce this when it snapshots.
                result.num_prefetch_layers = layers_issued
                # Preserve the worker's own progress independent of
                # retrieve_layer's later override, so stats can tell
                # "worker was slow" apart from "retrieve arrived early".
                result.worker_layers_issued = layers_issued
                result.e2e_time = time.monotonic() - result.created_at
                result.status = "ready"
                aggregate_future.set_result(True)
                logger.info(
                    "Speculative prefetch %s [layerwise]: worker done, "
                    "%d/%d layers issued",
                    request_id, layers_issued, effective_layers,
                )
            torch.cuda.nvtx.range_pop()  # finalize
            torch.cuda.nvtx.range_pop()  # worker
            _nvtx_mod.end_range(_spec_nvtx_range_id)

        # Submit the worker on the storage manager's thread pool. This
        # is a single long-running task. Intra-worker concurrency
        # comes from two sources: (1) a LOOKAHEAD-deep disk pipeline
        # so layer L+1's pread overlaps layer L's collect+H2D, and
        # (2) per-layer H2D events recorded without per-layer CPU sync
        # so successive layers' DMA streams overlap.
        self.storage_manager.worker_pool.submit(_layerwise_worker)

    def _start_ttl_cleanup(self):
        """Start periodic background cleanup of expired speculative
        prefetch results."""
        interval = min(self.config.speculative_prefetch_ttl_seconds, 5.0)
        self._ttl_cleanup_timer = threading.Timer(
            interval, self._run_ttl_cleanup,
        )
        self._ttl_cleanup_timer.daemon = True
        self._ttl_cleanup_timer.start()

    def _run_ttl_cleanup(self):
        """Check for and clean up expired speculative prefetch entries,
        then reschedule."""
        try:
            self._cleanup_expired_speculative()
        except Exception as e:
            logger.error("Speculative TTL cleanup error: %s", e)
        # Reschedule if engine is still alive
        if self.speculative_gpu_buffer is not None:
            self._start_ttl_cleanup()

    @staticmethod
    def _submit_phase2_batch(
        storage_manager,
        start_layer: int,
        end_layer: int,
        num_prefetch_layers: int,
        phase2_params: tuple,
    ) -> list:
        """Submit a batch of remaining-layer disk reads.

        Returns a ``List[Future]`` with one future per layer in
        ``[start_layer, end_layer)``.
        """
        (
            remaining_chunked_keys,
            remaining_keys_layer_major,
            remaining_pre_allocated,
            retrieve_backend,
        ) = phase2_params
        offset = start_layer - num_prefetch_layers
        count = end_layer - start_layer
        batch_pre = (
            remaining_pre_allocated[offset:offset + count]
            if remaining_pre_allocated is not None
            else None
        )
        if remaining_chunked_keys is not None:
            return storage_manager.layerwise_get_chunked_partial(
                remaining_chunked_keys,
                start_layer=start_layer,
                end_layer=end_layer,
                location=retrieve_backend,
                pre_allocated_mem_objs=batch_pre,
            )
        else:
            batch_keys = remaining_keys_layer_major[offset:offset + count]
            return storage_manager.layerwise_batched_get(
                batch_keys,
                location=retrieve_backend,
                pre_allocated_mem_objs=batch_pre,
                eager=True,
            )

    def _compute_next_batch_size(
        self, runway_layers: int, remaining_layers: int
    ) -> int:
        """Estimate how many remaining layers can be read from disk during
        *runway_layers* of inference.

        Returns *remaining_layers* (read-all) when there is insufficient
        historical timing data (cold start), or when the computed batch
        would be too small (to avoid per-batch overhead dominating).
        """
        stats = self._pipeline_timing_stats
        if not stats["t_read_per_layer"] or not stats["t_infer_per_layer"]:
            return remaining_layers  # cold start: fall back to read-all

        t_read = sum(stats["t_read_per_layer"]) / len(
            stats["t_read_per_layer"]
        )
        t_infer = sum(stats["t_infer_per_layer"]) / len(
            stats["t_infer_per_layer"]
        )

        if t_read <= 0:
            return remaining_layers

        ratio = t_infer / t_read  # layers readable per layer inferred
        batch = int(runway_layers * ratio)
        # Floor: never go below 1/3 of remaining to avoid degenerate
        # 1-layer-at-a-time batches whose per-batch overhead (transpose
        # thread, asyncio scheduling) outweighs any pipelining benefit.
        min_batch = max(1, remaining_layers // 3)
        batch = max(min_batch, min(batch, remaining_layers))
        # Absorb tiny tails: if the leftover after this batch would be
        # very small, just read everything to avoid an extra pread round.
        if remaining_layers - batch <= 2:
            batch = remaining_layers
        logger.info(
            "Pipeline batch estimate: t_read=%.4fms t_infer=%.4fms "
            "ratio=%.3f runway=%d remaining=%d -> batch=%d",
            t_read * 1000, t_infer * 1000,
            ratio, runway_layers, remaining_layers, batch,
        )
        return batch

    # ------------------------------------------------------------------
    # Optimal Phase 2 scheduling (equal-batch strategy)
    # ------------------------------------------------------------------

    # Per-batch overhead: thread dispatch + asyncio + future creation.
    # Conservative estimate; the exact value is not critical because
    # the minimum-K formula is dominated by the R/K term.
    _BATCH_OVERHEAD_S = 0.5e-3  # 0.5 ms

    def _compute_optimal_schedule(
        self, runway: int, remaining: int
    ) -> List[int]:
        """Pre-compute an optimal batch schedule for Phase 2.

        Since the transpose thread resolves futures per-*batch* (not
        per-layer), batch size directly controls when layers become
        consumable.  Equal-sized batches minimise the maximum stall
        across batches while keeping the number of batches (and thus
        per-batch overhead) low.

        Strategy:
        1. alpha <= 1  →  single batch (disk keeps up).
        2. alpha > 1, zero-stall achievable  →  K equal batches where
           K is the smallest integer satisfying
           ``R*(alpha - 1 + 1/K) + K*c/t_i <= N``.
        3. alpha > 1, stall unavoidable (N < R*(alpha-1))  →  big
           first batch to absorb the stall, then equal-batch tail.
        """
        if remaining <= 0:
            return []

        stats = self._pipeline_timing_stats
        if not stats["t_read_per_layer"] or not stats["t_infer_per_layer"]:
            return [remaining]  # cold start

        t_r = sum(stats["t_read_per_layer"]) / len(
            stats["t_read_per_layer"]
        )
        t_i = sum(stats["t_infer_per_layer"]) / len(
            stats["t_infer_per_layer"]
        )

        if t_r <= 0 or t_i <= 0:
            return [remaining]

        alpha = t_r / t_i
        c = self._BATCH_OVERHEAD_S

        # Disk faster than compute — one batch, zero stalls
        if alpha <= 1.0:
            logger.info(
                "Optimal schedule: alpha=%.3f <= 1, single batch %d",
                alpha, remaining,
            )
            return [remaining]

        # --- Find minimum K for zero-stall equal batches ---
        # Condition: R*(alpha - 1 + 1/K) + K*c/t_i <= N
        best_K = self._find_min_K_equal(
            runway, remaining, alpha, t_i, c,
        )

        if best_K is not None:
            schedule = self._equal_batches(remaining, best_K)
            logger.info(
                "Optimal schedule (equal, zero-stall): alpha=%.3f "
                "R=%d runway=%d K=%d -> %s",
                alpha, remaining, runway, best_K, schedule,
            )
            return schedule

        # --- I/O-bound: big first batch + equal tail ---
        b1 = math.ceil(remaining * (alpha - 1) / alpha)
        b1 = min(b1, remaining)
        tail_remaining = remaining - b1
        if tail_remaining <= 0:
            logger.info(
                "Optimal schedule (IO-bound, single): alpha=%.3f "
                "R=%d runway=%d -> [%d]",
                alpha, remaining, runway, remaining,
            )
            return [remaining]

        tail_K = self._find_min_K_equal(
            b1, tail_remaining, alpha, t_i, c,
        )
        if tail_K is not None:
            tail = self._equal_batches(tail_remaining, tail_K)
        else:
            tail = [tail_remaining]

        schedule = [b1] + tail
        logger.info(
            "Optimal schedule (IO-bound): alpha=%.3f "
            "R=%d runway=%d b1=%d tail_K=%s -> %s",
            alpha, remaining, runway, b1, tail_K, schedule,
        )
        return schedule

    @staticmethod
    def _find_min_K_equal(
        runway: int, remaining: int,
        alpha: float, t_i: float, c: float,
    ) -> Optional[int]:
        """Find the smallest K such that K equal batches of
        ``remaining`` layers achieve zero stall with the given runway.

        Returns ``None`` if stall is unavoidable (I/O-bound regime).
        """
        max_K = min(remaining, 20)  # cap search
        for K in range(1, max_K + 1):
            lhs = remaining * (alpha - 1 + 1.0 / K) + K * c / t_i
            if lhs <= runway:
                return K
        return None

    @staticmethod
    def _equal_batches(total: int, K: int) -> List[int]:
        """Split ``total`` layers into ``K`` near-equal batches."""
        base = total // K
        extra = total % K
        # First 'extra' batches get base+1, rest get base.
        return [base + 1] * extra + [base] * (K - extra)

    def _cleanup_expired_speculative(self):
        """Free GPU buffers and remove entries that have exceeded TTL."""
        now = time.monotonic()
        ttl = self.config.speculative_prefetch_ttl_seconds
        expired_ids = []

        with self.speculative_cache_lock:
            for req_id, result in self.speculative_cache.items():
                if now - result.created_at > ttl:
                    expired_ids.append(req_id)

            for req_id in expired_ids:
                result = self.speculative_cache.pop(req_id)
                result.status = "expired"
                if (
                    result.gpu_buffers is not None
                    and self.speculative_gpu_buffer is not None
                ):
                    self.speculative_gpu_buffer.free_chunks(result.gpu_buffers)
                    result.gpu_buffers = None
                # Free any remaining CPU MemoryObjs
                if result.cpu_mem_objs is not None:
                    for obj in result.cpu_mem_objs:
                        if obj is not None:
                            obj.ref_count_down()
                    result.cpu_mem_objs = None

        if expired_ids:
            logger.info(
                "Speculative TTL cleanup: removed %d expired entries: %s",
                len(expired_ids),
                expired_ids,
            )

    @_lmcache_nvtx_annotate
    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: the input tokens,
        with shape [seq_len]

        :param Optional[List[int]] hashes: the input hashes, with length [num_chunks]
        :param Optional[List[int]] offsets: the offsets of each chunk,
        with length [num_chunks]

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param Optional[str] lookup_id: The lookup ID to
            associate with the lookup. When pin is true, this argument is
            required to be not None.

        :param bool pin: If True, pin the KV cache in the storage.

        :param Optional[dict] request_configs: the configs of the request.

        :return: An int indicating how many prefix tokens are cached.
        """

        if tokens is not None:
            lookup_request_id = self.stats_monitor.on_lookup_request(len(tokens))
        else:
            assert offsets is not None
            assert hashes is not None
            lookup_request_id = self.stats_monitor.on_lookup_request(sum(offsets))

        res = 0
        try:
            chunk_info_iterator = self.token_database.process_tokens(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                request_configs=request_configs,
            )

            # TODO: support batched_contains when layerwise is enabled
            if self.use_layerwise:
                with _nvtx_mod.annotate("engine_lookup::materialize_chunks", domain="lmcache"):
                    chunk_info_list_lw = list(chunk_info_iterator)
                num_total_chunks = 0
                num_hit_chunks = 0
                # Collect hit chunk info for caching (avoid redundant
                # chunk matching in retrieve_layer).
                cached_starts: List[int] = []
                cached_ends: List[int] = []
                cached_keys: List[List[CacheEngineKey]] = []
                cached_backend: Optional[str] = None
                with _nvtx_mod.annotate("engine_lookup::layerwise_contains_loop", domain="lmcache"):
                  for start, end, key in chunk_info_list_lw:
                    assert isinstance(key, CacheEngineKey)
                    num_total_chunks += 1

                    with _nvtx_mod.annotate("engine_lookup::split_layers", domain="lmcache"):
                        key_all_layers = key.split_layers(self.num_layers)

                    # Check chunked key first (speculative mode), then
                    # fall back to per-layer key for backward compat.
                    with _nvtx_mod.annotate("engine_lookup::contains_layer0", domain="lmcache"):
                        if self.config.speculative_prefetch_enabled:
                            found = self.storage_manager.contains(
                                key.to_chunked_key(self.num_layers),
                                search_range, pin=False,
                            )
                        else:
                            found = None
                        if not found:
                            found = self.storage_manager.contains(
                                key_all_layers[0], search_range, pin
                            )
                    if found:
                        num_hit_chunks += 1
                        if cached_backend is None:
                            cached_backend = found
                        cached_starts.append(start)
                        cached_ends.append(end)
                        cached_keys.append(key_all_layers)
                        if pin:
                            assert lookup_id is not None, (
                                "lookup_id is required when pin is True"
                            )
                            # Pin remaining layers so lookup_unpin
                            # can unpin all of them without going negative.
                            with _nvtx_mod.annotate("engine_lookup::pin_remaining_layers", domain="lmcache"):
                                for key_single_layer in key_all_layers[1:]:
                                    self.storage_manager.contains(
                                        key_single_layer, search_range, pin
                                    )
                            self.lookup_pins[lookup_id].extend(  # type: ignore
                                key_all_layers
                            )
                        res = end
                        continue
                    # When blending is enabled, skip missing chunks
                    # instead of stopping, so that later chunks can
                    # still be matched and used by CacheBlend.
                    if self.config.enable_blending:
                        continue
                    return res
                if self.config.enable_blending:
                    logger.info(
                        "Lookup chunk stats: %d/%d chunks hit",
                        num_hit_chunks,
                        num_total_chunks,
                    )
                    # Cache the chunk matching results so
                    # retrieve_layer() can skip re-matching.
                    if lookup_id is not None and cached_starts:
                        ret_mask = torch.zeros(
                            res, dtype=torch.bool, device="cpu"
                        )
                        for s, e in zip(cached_starts, cached_ends):
                            ret_mask[s:e] = True
                        self._lookup_result_cache[lookup_id] = (
                            LookupCachedResult(
                                starts=cached_starts,
                                ends=cached_ends,
                                keys=cached_keys,
                                ret_mask=ret_mask,
                                retrieve_backend=cached_backend,
                                num_total_chunks=num_total_chunks,
                                num_hit_chunks=num_hit_chunks,
                            )
                        )
            else:
                chunk_info_list = []
                keys = []
                for chunk_info in chunk_info_iterator:
                    assert isinstance(chunk_info[2], CacheEngineKey)
                    chunk_info_list.append(chunk_info)
                    keys.append(chunk_info[2])

                with _nvtx_mod.annotate("engine_lookup::batched_contains", domain="lmcache"):
                    batched_contains_res = self.storage_manager.batched_contains(
                        keys, search_range, pin, True
                    )
                assert len(batched_contains_res) == len(chunk_info_list)
                for (start, end, key), exists in zip(
                    chunk_info_list, batched_contains_res, strict=False
                ):
                    if exists:
                        if pin:
                            assert lookup_id is not None, (
                                "lookup_id is required when pin is True"
                            )
                            self.lookup_pins[lookup_id].append(key)
                        res = end
                        continue
                    return res

            # all tokens where found, return the maximal end
            return res
        finally:
            self.stats_monitor.on_lookup_finished(lookup_request_id, res)
            # vllm lookup sets pin to True
            if pin:
                self.storage_manager.touch_cache()

    @_lmcache_nvtx_annotate
    def move(
        self,
        tokens: Union[torch.Tensor, List[int]],
        old_position: str,
        new_position: tuple[str, str],
        event_id: str,
        do_copy: bool = True,
    ) -> int:
        """
        Perform cross-node move of the KV cache.
        """

        num_tokens = self.lookup(
            tokens,
            search_range=old_position,
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        keys = self.lookup_pins[event_id]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=old_position,
        )
        assert memory_objs is not None, "Failed to get memory objects to move"
        logger.debug(
            f"Trying to send {len(memory_objs)} memory objects to {new_position}"
        )

        # TODO: reduce loops
        token_dim = memory_objs[0].meta.fmt.token_dim()  # type: ignore
        offsets = [m.meta.shape[token_dim] for m in memory_objs]  # type: ignore

        transfer_spec = {
            "peer_init_url": new_position[0],
            "offsets": offsets,
        }

        logger.info(self.storage_manager.storage_backends)
        p2p_backend = self.storage_manager.storage_backends["P2PBackend"]

        future = asyncio.run_coroutine_threadsafe(
            p2p_backend.async_batched_submit_put_task(
                keys,
                memory_objs,  # type: ignore
                transfer_spec=transfer_spec,
            ),
            self.storage_manager.loop,
        )

        future.result()

        if not do_copy:
            self.storage_manager.batched_remove(keys, locations=[old_position])

        logger.debug(f"Moving {num_tokens} token from {old_position} to {new_position}")
        return num_tokens

    # TODO(Jiayi): Add layerwise support.
    @_lmcache_nvtx_annotate
    def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> None:
        """
        An async version of lookup + prefetch.

        There are three categories of backends:
        (1) sync lookup + sync retrieval (e.g., cpu)
        (2) sync lookup + async retrieval (e.g., disk)
        (3) async lookup + async retrieval (e.g., p2p)
        """

        keys: list[CacheEngineKey] = []
        cum_chunk_lengths = [0]

        # TODO(Jiayi): make token database able to return list.
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            keys.append(key)
            cum_chunk_lengths.append(end)

        asyncio.run_coroutine_threadsafe(
            self.storage_manager.async_lookup_and_prefetch(
                lookup_id, keys, cum_chunk_lengths, search_range, pin
            ),
            self.storage_manager.loop,
        )

    # TODO(Jiayi): Need to handle the case where `tokens=None`.
    # In this case, we compress all tokens.
    # TODO(Jiayi): support other compression methods.
    @_lmcache_nvtx_annotate
    def compress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported compression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        serializer, _ = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        keys = self.lookup_pins[event_id]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )
        assert memory_objs is not None, (
            "LMCacheEngine.compress: Failed to get memory objects to compress"
        )

        compressed_memory_objs = []
        for memory_obj in memory_objs:
            assert memory_obj is not None
            compressed_memory_obj = serializer.serialize(memory_obj)
            memory_obj.unpin()
            compressed_memory_objs.append(compressed_memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=compressed_memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def decompress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported decompression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        _, deserializer = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("there are no tokens to decompress.")
            return 0

        keys = self.lookup_pins[event_id]

        compressed_memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )

        assert compressed_memory_objs is not None, (
            "LMCacheEngine.compress: Failed to get compressed "
            "memory objects to decompress"
        )

        memory_objs = []
        for compressed_memory_obj in compressed_memory_objs:
            assert compressed_memory_obj is not None
            memory_obj = deserializer.deserialize(compressed_memory_obj)
            compressed_memory_obj.unpin()
            memory_objs.append(memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def lookup_unpin(self, lookup_id: str) -> None:
        if lookup_id in self.lookup_pins:
            self.storage_manager.batched_unpin(self.lookup_pins[lookup_id])
            del self.lookup_pins[lookup_id]
        # Clean up any unconsumed cached lookup result to prevent leaks.
        self._lookup_result_cache.pop(lookup_id, None)

    @_lmcache_nvtx_annotate
    def clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        # TODO: need to clear by request_configs
        if self.save_only_first_rank:
            if self.metadata.is_first_rank():
                num_removed = self._clear(tokens, locations, request_configs)
                return num_removed
            else:
                return 0
        return self._clear(tokens, locations, request_configs)

    def _clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        assert isinstance(self.storage_manager, StorageManager)
        # Clear all caches if tokens is None
        if tokens is None or len(tokens) == 0:
            num_cleared = self.storage_manager.clear(locations)
            return num_cleared

        num_removed = 0
        # Only remove the caches for the given tokens
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            removed = self.storage_manager.remove(key, locations)
            num_removed += removed
        return num_removed

    @_lmcache_nvtx_annotate
    def health(
        self,
    ) -> int:
        """
        Check the health of the cache engine.
        return: 0 if healthy, otherwise the error code
        """
        return 0 if self.storage_manager.memcheck() else -1

    def close(self) -> None:
        """Close the cache engine and free all the resources"""

        # Stop TTL cleanup timer
        if self._ttl_cleanup_timer is not None:
            self._ttl_cleanup_timer.cancel()
            self._ttl_cleanup_timer = None

        # Free all speculative prefetch resources
        with self.speculative_cache_lock:
            for req_id, result in self.speculative_cache.items():
                if (
                    result.gpu_buffers is not None
                    and self.speculative_gpu_buffer is not None
                ):
                    self.speculative_gpu_buffer.free_chunks(result.gpu_buffers)
                    result.gpu_buffers = None
                if result.cpu_mem_objs is not None:
                    for obj in result.cpu_mem_objs:
                        if obj is not None:
                            obj.ref_count_down()
                    result.cpu_mem_objs = None
            self.speculative_cache.clear()
        # Release the GPU buffer manager reference
        self.speculative_gpu_buffer = None

        if self.lmcache_worker is not None:
            self.lmcache_worker.close()

        self.storage_manager.close()

        logger.info("LMCacheEngine closed.")

    def _async_process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> tuple[list[tuple[CacheEngineKey, MemoryObj, int, int]], int]:
        """
        This function is used to get the memory objects from the event manager.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert "req_id" in kwargs, "req_id is required for async loading"
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        tot_kv_size = 0
        chunks: list[tuple[CacheEngineKey, MemoryObj, int, int]] = []
        future = self.event_manager.pop_event(EventType.LOADING, kwargs["req_id"])

        memory_objs = future.result()
        memory_objs = [mm for m in memory_objs for mm in m]

        # NOTE(Jiayi): here we assume the retrieved memory_objs have
        # the same order as the lookup order.
        # TODO(Jiayi): hashing inside `process_tokens` can be skipped.
        for idx, (start, end, key) in enumerate(
            self.token_database.process_tokens(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
            )
        ):
            assert isinstance(key, CacheEngineKey)
            memory_obj = memory_objs[idx]
            chunks.append((key, memory_obj, start, end))
            tot_kv_size += memory_obj.get_size()
            ret_mask[start:end] = True

        # NOTE: free the memory objects that are not hit.
        for unused_mem_obj in memory_objs[len(chunks) :]:
            unused_mem_obj.ref_count_down()

        return chunks, tot_kv_size

    def _process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> tuple[list[tuple[CacheEngineKey, MemoryObj, int, int]], int]:
        """Process tokens and populate the reordered lists.

        This function is used to process tokens and populate the reordered lists.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """

        tot_kv_size = 0
        # location -> [(CacheEngineKey, start, end)]
        block_mapping: dict[str, list[tuple[CacheEngineKey, int, int]]] = defaultdict(
            list
        )

        reordered_chunks: list[tuple[CacheEngineKey, MemoryObj, int, int]] = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        # In some scenarios, lookup is called first, and then the original tokens
        # is sliced based on the lookup result. In these scenarios, the tokens
        # passed in must exist in LMCache, and we can set skip_contains_check to True.
        # When skip_contains_check is True and there is only one backend, the `contains`
        # call can be skipped.
        skip_contains_check = (
            kwargs["skip_contains_check"] if "skip_contains_check" in kwargs else False
        )
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)

            location = None
            if key in self.lookup_cache:
                # TODO(Jiayi): we can reduce the number of `contains` calls
                # by checking the lookup cache first (should be updated in `lookup`)
                pass
            else:
                # NOTE: key should always be in the lookup cache once we support it.
                # TODO: use lookup_cache to skip the contains
                if (
                    skip_contains_check
                    and len(self.storage_manager.non_allocator_backends) == 1
                ):
                    location = self.storage_manager.non_allocator_backends[0]
                else:
                    location = self.storage_manager.contains(key)
                if location is None:
                    logger.debug(
                        "retrieve MISS chunk_hash=%d tokens[%d:%d]",
                        key.chunk_hash, start, end,
                    )
                    break
                logger.debug(
                    "retrieve HIT  chunk_hash=%d tokens[%d:%d] "
                    "location=%s",
                    key.chunk_hash, start, end, location,
                )

                # NOTE: Here we make the assumption that the underlying
                # storage backend support pin operation, and the memory
                # object is already pinned in the storage backend.
                ret_mask[start:end] = True

            assert location is not None

            block_mapping[location].append((key, start, end))

        last_failed_block_start = None
        for location, blocks in block_mapping.items():
            keys = [key for key, _, _ in blocks]
            memory_objs = self.storage_manager.batched_get(
                keys=keys,
                location=location,
            )
            assert memory_objs is not None, (
                "Failed to get memory objects from storage backend"
            )

            for (key, start, end), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is None:
                    logger.warning(
                        "The cache block is in the storage, but it can't be retrieved"
                    )
                    if (
                        last_failed_block_start is None
                        or last_failed_block_start < start
                    ):
                        last_failed_block_start = start
                    break
                reordered_chunks.append((key, memory_obj, start, end))
                tot_kv_size += memory_obj.get_size()

        if last_failed_block_start is not None:
            ret_mask[last_failed_block_start:] = False

            reordered_chunks = [
                (key, memory_obj, start, end)
                for key, memory_obj, start, end in reordered_chunks
                if end < last_failed_block_start
            ]
        return reordered_chunks, tot_kv_size

    def _broadcast_or_receive_memory_objs(
        self,
        reordered_chunks,
        ret_mask,
    ):
        """
        Handles broadcasting or receiving memory objects in a distributed environment.

        This function implements the communication logic where:
        - The first rank (coordinator) broadcasts memory objects and metadata to others
        - Other ranks receive and reconstruct the memory objects

        Parameters:
        reordered_chunks: List of tuples containing [key, memory object, start, end]
        ret_mask: Boolean mask indicating which positions have been processed

        Side Effects:
        - On first rank:
          * Broadcasts chunk count and each chunk's combined metadata
          * Broadcasts tensor data
        - On other ranks:
          * Receives chunk data and populates reordered_chunks
          * Updates ret_mask to mark received positions as True
        """
        if self.metadata.is_first_rank():
            # Broadcast total chunk count
            chunk_count = len(reordered_chunks)
            self.broadcast_object_fn(chunk_count, self.metadata.first_rank)

            # Broadcast each chunk's data
            for key, memory_obj, start, end in reordered_chunks:
                # Combine (start, end) and metadata into single broadcast
                metadata_dict = memory_obj.metadata.to_dict()
                combined_metadata = (start, end, metadata_dict)
                self.broadcast_object_fn(combined_metadata, self.metadata.first_rank)

                # Broadcast tensor data
                tensor_to_broadcast = memory_obj.tensor.to(
                    f"cuda:{self.metadata.worker_id}"
                )
                self.broadcast_fn(tensor_to_broadcast, self.metadata.first_rank)
        else:
            # Receive total chunk count
            chunk_count = self.broadcast_object_fn(None, self.metadata.first_rank)
            if chunk_count is None:
                logger.warning(
                    f"rank={self.metadata.worker_id} received None chunk_count"
                )
                return

            # Fill reordered_chunks with received data
            for _ in range(chunk_count):
                # Receive combined metadata (start, end, metadata_dict)
                combined_metadata = self.broadcast_object_fn(
                    None, self.metadata.first_rank
                )
                if combined_metadata is None:
                    logger.warning(
                        f"rank={self.metadata.worker_id} "
                        "received None combined_metadata"
                    )
                    break
                start, end, metadata_dict = combined_metadata
                ret_mask[start:end] = True

                # Create tensor and receive data
                metadata = MemoryObjMetadata.from_dict(metadata_dict)
                local_rank = self.metadata.worker_id % torch.cuda.device_count()
                tensor = torch.empty(
                    metadata.shape,
                    dtype=metadata.dtype,
                    device=f"cuda:{local_rank}",
                )
                self.broadcast_fn(tensor, self.metadata.first_rank)

                # Create temporary memory object (key not needed for other ranks)
                memory_obj = TensorMemoryObj(
                    raw_data=tensor, metadata=metadata, parent_allocator=None
                )
                reordered_chunks.append((None, memory_obj, start, end))

    def _is_passive(self):
        """
        A 'passive' CacheEngine means that the node itself will not store/retrieve
        the data directly, but from the "active" worker (i.e., rank 0 in MLA)
        """
        return self.save_only_first_rank and not self.metadata.is_first_rank()


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheEngineMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}

    # TODO(Jiayi): Please remove this helper function in the future.
    # Currently, it's only used for testing.
    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        numa_mapping: Optional[NUMAMapping] = None,
    ) -> MemoryAllocatorInterface:
        # NOTE: should remove this function after fixing the unit tests:
        # raise RuntimeError("_Create_memory_allocator is deprecated!")
        extra_config = config.extra_config
        enable_nixl_storage = extra_config is not None and extra_config.get(
            "enable_nixl_storage"
        )

        if enable_nixl_storage:
            # TODO(Jiayi): weird to import from transfer utils.
            # First Party
            from lmcache.v1.transfer_channel.transfer_utils import (
                get_correct_device,
            )

            corrected_device = get_correct_device(
                config.nixl_buffer_device,
                metadata.worker_id,
            )

            buffer = torch.empty(
                config.nixl_buffer_size,
                dtype=torch.uint8,
                device=corrected_device,
            )

            if corrected_device == "cpu":
                torch.cuda.cudart().cudaHostRegister(
                    buffer.data_ptr(), config.nixl_buffer_size, 0
                )
            else:
                logger.info(f"Setting cuda device to {corrected_device} ")
                torch.cuda.set_device(corrected_device)

            return PagedTensorMemoryAllocator(
                buffer,
                torch.Size(metadata.kv_shape),
                metadata.kv_dtype,
                MemoryFormat.KV_2LTD,
            )

        if config.weka_path is not None or config.gds_path is not None:
            assert config.cufile_buffer_size is not None
            return CuFileMemoryAllocator(config.cufile_buffer_size * 1024**2)

        max_local_cpu_size = config.max_local_cpu_size
        # save_only_first_rank only works when use mla
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if save_only_first_rank and metadata.is_first_rank():
            # Only the first rank will save the cache,
            # so we need to set it lager than other ranks
            first_rank_max_local_cpu_size = (
                config.extra_config.get(
                    "first_rank_max_local_cpu_size", max_local_cpu_size
                )
                if config.extra_config
                else max_local_cpu_size
            )
            return MixedMemoryAllocator(
                int(first_rank_max_local_cpu_size * 1024**3),
                numa_mapping=numa_mapping,
            )
        return MixedMemoryAllocator(
            int(max_local_cpu_size * 1024**3),
            numa_mapping=numa_mapping,
        )

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> TokenDatabase:
        if config.enable_blending:
            return SegmentTokenDatabase(config, metadata)
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        gpu_connector: GPUConnectorInterface,
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            numa_mapping = NUMADetector.get_numa_mapping(config)
            logger.info(f"NUMA mapping for instance {instance_id}: {numa_mapping}")
            token_database = cls._Create_token_database(config, metadata)
            stat_logger = LMCacheStatsLogger(metadata, log_interval=10)

            engine = LMCacheEngine(
                config,
                metadata,
                token_database,
                gpu_connector,
                broadcast_fn,
                broadcast_object_fn,
            )

            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger
            return engine
        else:
            if (
                cls._cfgs[instance_id] != config
                or cls._metadatas[instance_id] != metadata
            ):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata."
                )
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID,
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            stat_logger.shutdown()
            engine = cls._instances[instance_id]
            engine.close()
            cls._instances.pop(instance_id, None)
            cls._cfgs.pop(instance_id, None)
            cls._metadatas.pop(instance_id, None)
            cls._stat_loggers.pop(instance_id, None)
            LMCStatsMonitor.DestroyInstance()
