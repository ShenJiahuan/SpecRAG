# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from concurrent.futures import Future
from typing import Dict, List, Optional, Tuple
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata

logger = init_logger(__name__)


@dataclass
class SpeculativePrefetchResult:
    """Holds the state of a speculative prefetch operation."""

    request_id: str

    # chunk_keys[chunk_idx][layer_idx] = CacheEngineKey
    # Chunk identity is determined by token hashing
    # (same as normal CacheBlend lookup).
    chunk_keys: List[List[CacheEngineKey]]

    # Per-chunk GPU buffers holding speculative KV data.
    # Each element has shape [num_layers, 2, chunk_tokens, hidden_dim],
    # contiguous per chunk so that prefetch uses one H2D copy per chunk.
    # None during Phase 3 (CPU-only); populated in Phase 4 after GPU copy.
    gpu_buffers: Optional[List[torch.Tensor]]

    # Future that completes when disk read (+ GPU copy) are done
    future: Future

    # CPU MemoryObjs from disk read, in flattened layer-major order.
    # Populated after disk read completes; freed after GPU copy (Phase 4).
    cpu_mem_objs: Optional[List] = None

    # Number of layers that were actually prefetched (may be less than
    # the model's total num_layers when partial-layer prefetch is used).
    # retrieve_layer() uses this to know which layers are in the
    # speculative buffer vs. which must be fetched from disk.
    num_prefetch_layers: int = 0  # 0 means "all layers" (set by caller)

    # "loading" | "ready" | "consumed" | "expired"
    status: str = "loading"

    # For timeout/cleanup
    created_at: float = field(default_factory=time.monotonic)

    # Server-side end-to-end time in seconds (set when status becomes "ready")
    e2e_time: Optional[float] = None

    # Layerwise baseline only:
    # Number of layers that have been fully copied to gpu_buffers and
    # are safe for retrieve_layer to read. Written by the worker after
    # each layer's H2D copy syncs; read (and frozen) by retrieve_layer
    # on arrival. Plain int writes are atomic in CPython, so no extra
    # lock is required for a single producer / single consumer counter.
    ready_num_prefetch_layers: int = 0

    # Layerwise baseline only: set by retrieve_layer to tell the worker
    # to stop submitting further layers. The worker checks this at the
    # top of each iteration and exits the loop early.
    stop_event: threading.Event = field(default_factory=threading.Event)

    # Layerwise baseline only: how many layers the worker actually got
    # through before finalize (full run, early stop, or error). Written
    # once at finalize time by the worker and never touched afterward.
    # Distinct from ``num_prefetch_layers`` because retrieve_layer
    # overrides that field to the L it snapshotted, which would
    # otherwise erase the worker's real progress from stats.
    worker_layers_issued: int = 0

    def get_chunk_key_tuples(self) -> List[Tuple]:
        """Return a list of hashable tuples representing each chunk's
        identity (using the first layer's key per chunk)."""
        return [
            (keys[0].chunk_hash,) if keys else ()
            for keys in self.chunk_keys
        ]


class SpeculativeGPUBuffer:
    """Manages GPU memory for speculative KV cache prefetch."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        memory_budget_bytes: int,
    ):
        self.device = device
        self.dtype = dtype
        self.memory_budget_bytes = memory_budget_bytes
        self.used_bytes = 0
        self.lock = threading.Lock()

    def allocate_chunks(
        self,
        num_layers: int,
        chunk_num_tokens: List[int],
        hidden_dim: int,
    ) -> Optional[List[torch.Tensor]]:
        """Allocate per-chunk GPU buffers for speculative prefetch.

        Each returned tensor has shape
        ``[num_layers, 2, chunk_tokens, hidden_dim]`` and is contiguous,
        enabling a single H2D copy per chunk instead of per-layer copies.

        Returns None if the total allocation would exceed the budget.
        """
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        required_bytes = sum(
            num_layers * 2 * nt * hidden_dim * element_size
            for nt in chunk_num_tokens
        )
        with self.lock:
            if self.used_bytes + required_bytes > self.memory_budget_bytes:
                logger.warning(
                    "Speculative GPU buffer budget exceeded: "
                    f"used={self.used_bytes}, "
                    f"required={required_bytes}, "
                    f"budget={self.memory_budget_bytes}"
                )
                return None
            self.used_bytes += required_bytes
        bufs = [
            torch.empty(
                (num_layers, 2, nt, hidden_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for nt in chunk_num_tokens
        ]
        return bufs

    def free_chunks(self, bufs: List[torch.Tensor]):
        """Free a list of per-chunk GPU buffers."""
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        freed_bytes = sum(buf.numel() * element_size for buf in bufs)
        with self.lock:
            self.used_bytes -= freed_bytes
        for buf in bufs:
            del buf

    @property
    def available_bytes(self) -> int:
        with self.lock:
            return self.memory_budget_bytes - self.used_bytes


class GPUSliceMemoryObj(MemoryObj):
    """Lightweight MemoryObj wrapping a slice of the speculative GPU buffer.

    This is used to feed speculative KV data into the GPU connector's
    ``batched_to_gpu()`` generator, which expects MemoryObj instances with
    ``metadata.fmt == KV_2TD`` and ``tensor[0]``/``tensor[1]`` for K/V.

    The underlying GPU tensor slice is **not** owned by this object; it is
    managed by the ``SpeculativeGPUBuffer``.  ``ref_count_down()`` is a
    no-op so that the caller's cleanup loop does not accidentally free
    shared speculative memory.
    """

    def __init__(self, gpu_tensor: torch.Tensor, dtype: torch.dtype):
        """
        :param gpu_tensor: Tensor of shape ``[2, num_tokens, hidden_dim]``
            on GPU, already in KV_2TD layout.
        :param dtype: Logical dtype of the KV data.
        """
        meta = MemoryObjMetadata(
            shape=gpu_tensor.shape,
            dtype=dtype,
            address=gpu_tensor.data_ptr(),
            phy_size=gpu_tensor.numel() * gpu_tensor.element_size(),
            ref_count=1,
            fmt=MemoryFormat.KV_2TD,
        )
        super().__init__(meta)
        self._tensor = gpu_tensor
        self.valid = True

    # -- Required abstract method implementations --

    def invalidate(self):
        self.valid = False

    def is_valid(self):
        return self.valid

    def get_size(self) -> int:
        return self.meta.phy_size

    def get_shape(self) -> torch.Size:
        return self.meta.shape

    def get_dtype(self) -> Optional[torch.dtype]:
        return self.meta.dtype

    def get_memory_format(self) -> MemoryFormat:
        return self.meta.fmt

    def get_physical_size(self) -> int:
        return self.meta.phy_size

    def ref_count_up(self):
        pass  # No-op: GPU buffer lifetime managed by SpeculativeGPUBuffer

    def ref_count_down(self):
        pass  # No-op: GPU buffer lifetime managed by SpeculativeGPUBuffer

    def get_ref_count(self) -> int:
        return 1

    def get_num_tokens(self) -> int:
        return self._tensor.shape[1]  # [2, T, D]

    def pin(self) -> bool:
        return True

    def unpin(self) -> bool:
        return True

    @property
    def metadata(self) -> MemoryObjMetadata:
        return self.meta

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        if not self.valid:
            return None
        return self._tensor

    @property
    def byte_array(self) -> bytes:
        raise NotImplementedError(
            "GPUSliceMemoryObj does not support byte_array access"
        )

    @property
    def data_ptr(self) -> int:
        return self._tensor.data_ptr()

    @property
    def is_pinned(self) -> bool:
        return False

    @property
    def can_evict(self) -> bool:
        return False


def make_gpu_memory_obj(
    spec_result: "SpeculativePrefetchResult",
    chunk_idx: int,
    layer_idx: int,
) -> GPUSliceMemoryObj:
    """Wrap a slice of the speculative GPU buffer as a MemoryObj.

    The returned object has ``metadata.fmt == KV_2TD`` and ``tensor``
    of shape ``[2, num_tokens, hidden_dim]``, suitable for direct use
    in ``batched_to_gpu()``.

    :param spec_result: The speculative prefetch result containing the
        GPU buffers (one per chunk).
    :param chunk_idx: Index of the chunk within the speculative result.
    :param layer_idx: Layer index to extract.
    :return: A ``GPUSliceMemoryObj`` wrapping the GPU buffer slice.
    """
    # gpu_buffers[chunk_idx] shape: [num_layers, 2, chunk_tokens, hidden_dim]
    kv_slice = spec_result.gpu_buffers[chunk_idx][layer_idx]
    return GPUSliceMemoryObj(
        gpu_tensor=kv_slice,
        dtype=spec_result.gpu_buffers[chunk_idx].dtype,
    )


def build_spec_chunk_hash_map(
    spec_result: "SpeculativePrefetchResult",
) -> Dict[Tuple, int]:
    """Build a mapping from chunk hash tuple to speculative chunk index.

    Used by ``retrieve_layer()`` to quickly look up which actual chunks
    have matching speculative data.
    """
    mapping: Dict[Tuple, int] = {}
    for idx, keys_per_layer in enumerate(spec_result.chunk_keys):
        if keys_per_layer:
            mapping[(keys_per_layer[0].chunk_hash,)] = idx
    return mapping
