# SPDX-License-Identifier: Apache-2.0
"""EPIC (Efficient Position-Independent Cache) blender for LMCache.

Key differences from CacheBlend (LMCBlender):
1. Importance scoring uses V-based L2 difference (not K-based)
2. Partial attention: selected Q queries attend to FULL K/V via attention bias mask
3. Status transitions: 0 (prefill) → 1 (check layer) → 2 (post-check layers)
4. After check layer, hidden_states are filtered to imp_indices for subsequent layers
"""

# Standard
from typing import Optional, Union
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.epic_attention import make_partial_attention_bias
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.epic_metadata import EPICBlendMetadata
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


class EPICBlender:
    """EPIC cache blending backend for LMCache."""

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        # Deferred import to avoid circular dependency
        from lmcache.v1.compute.models.utils import infer_model_from_vllm

        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)

        self.layerwise_model = infer_model_from_vllm(
            vllm_model, self, enable_sparse, algorithm="epic"
        )

        self.num_layers = len(vllm_model.model.layers)

        # Check layers config (reuse CacheBlend's config key)
        check_layers = config.blend_check_layers or [1]
        self.check_layers = check_layers

        # EPIC metadata
        self.metadata = EPICBlendMetadata(
            recomp_ratio=config.epic_recomp_ratio,
        )

        # Extract head info from vllm_model for attention bias construction
        first_attn = vllm_model.model.layers[0].self_attn.attn
        self.num_kv_heads = first_attn.impl.num_kv_heads
        self.num_heads = first_attn.impl.num_heads
        self.head_size = first_attn.impl.head_size

        # Masks from retrieve_layer
        self._ret_mask: Optional[torch.Tensor] = None
        self._token_mask: Optional[torch.Tensor] = None

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata: LMCAttnMetadata,
    ):
        """Process QKV for one layer with EPIC's importance-based selection.

        Unlike CacheBlend which scores on K, EPIC scores on V:
            diff = sum((v_new - v_old)², dim=[1, 2])
            top_indices = topk(diff, k=int((seq_len-1) * recomp_ratio))
        """
        logger.debug(f"EPIC Blender processing layer {layer_id}")
        torch.cuda.nvtx.range_push(f"epic_blend::process_qkv_layer_{layer_id}")

        device = q.device
        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(
                q.shape,
                dtype=q.dtype,
                device=q.device,
            )

        # Perform positional encoding
        torch.cuda.nvtx.range_push("epic_blend::rope")
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn
        q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)
        torch.cuda.nvtx.range_pop()  # epic_blend::rope

        if layer_id in self.check_layers:
            self.metadata.status = 1

            # Reshape to per-head format for importance scoring
            torch.cuda.nvtx.range_push("epic_blend::importance_scoring")
            old_v_heads = old_v.view(-1, self.num_kv_heads, self.head_size)
            v_heads = v.view(-1, self.num_kv_heads, self.head_size)

            seq_len = v_heads.shape[0]

            # EPIC importance: L2 diff on V values (exclude last token from diff)
            topk_num = int((seq_len - 1) * self.metadata.recomp_ratio)

            # Detect cache-miss positions
            miss_mask = None
            if self._ret_mask is not None and self._token_mask is not None:
                miss_mask = ~self._ret_mask & self._token_mask

            if miss_mask is not None and miss_mask.any():
                miss_indices = torch.where(miss_mask)[0]
                non_miss_mask = ~miss_mask
                non_miss_indices = torch.where(non_miss_mask)[0]

                # Score only non-miss positions on V
                if len(non_miss_indices) > 0:
                    # Exclude last token from scoring
                    nm_no_last = non_miss_indices[non_miss_indices < seq_len - 1]
                    if len(nm_no_last) > 0:
                        diff_v = torch.sum(
                            (v_heads[nm_no_last] - old_v_heads[nm_no_last]) ** 2,
                            dim=[1, 2],
                        )
                        nm_topk = min(topk_num, len(nm_no_last))
                        if nm_topk > 0:
                            top_in_nm = torch.topk(diff_v, k=nm_topk).indices
                            selected = nm_no_last[top_in_nm]
                        else:
                            selected = nm_no_last.new_empty(0, dtype=torch.long)
                    else:
                        selected = non_miss_indices.new_empty(0, dtype=torch.long)
                else:
                    selected = miss_indices.new_empty(0, dtype=torch.long)

                # Always include last token
                last_token = torch.tensor([seq_len - 1], device=device)
                top_indices = torch.cat([miss_indices, selected, last_token])
                top_indices = torch.unique(torch.sort(top_indices).values)
            else:
                # No cache-miss gaps: standard EPIC scoring
                temp_diff = torch.sum(
                    (v_heads[:-1, :, :] - old_v_heads[:-1, :, :]) ** 2,
                    dim=[1, 2],
                )
                if topk_num > 0:
                    top_indices = torch.topk(temp_diff, k=topk_num).indices
                else:
                    top_indices = torch.tensor([], device=device, dtype=torch.long)

                # Always include last token
                last_token = torch.tensor([seq_len - 1], device=device)
                top_indices = torch.cat([top_indices, last_token])
                top_indices = torch.sort(top_indices).values
            torch.cuda.nvtx.range_pop()  # epic_blend::importance_scoring

            self.metadata.imp_indices = top_indices
            self.metadata.org_seq_len = seq_len

            # Build partial attention bias
            torch.cuda.nvtx.range_push("epic_blend::build_attn_bias")
            attn_bias = make_partial_attention_bias(
                top_indices, seq_len, self.num_kv_heads,
                dtype=q.dtype, device=device,
            )
            self.metadata.attn_bias = attn_bias
            torch.cuda.nvtx.range_pop()  # epic_blend::build_attn_bias

            # Select only important Q positions (EPIC: Q is sliced, K/V stay full)
            torch.cuda.nvtx.range_push("epic_blend::token_slicing")
            q = q[top_indices]
            residual = residual[top_indices]
            self.metadata.positions = self.metadata.positions[top_indices]

            logger.debug(
                f"EPIC check layer {layer_id}: selected {len(top_indices)}/{seq_len} "
                f"tokens (ratio={self.metadata.recomp_ratio})"
            )

            torch.cuda.nvtx.range_pop()  # epic_blend::token_slicing

            # Update old KV at important indices with new values
            torch.cuda.nvtx.range_push("epic_blend::kv_cache_update")
            old_k_flat = old_k.view(-1, self.num_kv_heads * self.head_size)
            k_flat = k.view(-1, self.num_kv_heads * self.head_size)
            old_k_flat[top_indices] = k_flat[top_indices]

            old_v_heads[top_indices] = v_heads[top_indices]
            torch.cuda.nvtx.range_pop()  # epic_blend::kv_cache_update

            attn_output = attn_output[:len(top_indices)]
            attn_metadata.update_from_top_indices(top_indices)

            torch.cuda.nvtx.range_pop()

            # Return: selected Q, FULL old_k, FULL old_v, selected residual
            return (q, old_k, old_v, residual, attn_output, attn_metadata)

        elif self.metadata.status >= 1:
            # Post-check layers (status 2): use previously selected imp_indices
            self.metadata.status = 2
            imp_indices = self.metadata.imp_indices

            # Q/K/V are already computed from filtered hidden_states (only imp_indices tokens)
            # Update old KV at important indices with new values
            torch.cuda.nvtx.range_push("epic_blend::kv_cache_update")
            old_k_flat = old_k.view(-1, self.num_kv_heads * self.head_size)
            k_flat = k.view(-1, self.num_kv_heads * self.head_size)
            old_k_flat[imp_indices] = k_flat

            old_v_flat = old_v.view(-1, self.num_kv_heads * self.head_size)
            v_flat = v.view(-1, self.num_kv_heads * self.head_size)
            old_v_flat[imp_indices] = v_flat
            torch.cuda.nvtx.range_pop()  # epic_blend::kv_cache_update

            torch.cuda.nvtx.range_pop()

            # Return: Q as-is, FULL old_k, FULL old_v
            return (q, old_k, old_v, residual, attn_output, attn_metadata)

        else:
            # Status 0: before check layer, normal forward
            torch.cuda.nvtx.range_pop()
            return (q, k, v, residual, attn_output, attn_metadata)

    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Perform layerwise retrieve + blending with EPIC algorithm."""

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        torch.cuda.nvtx.range_push("epic_blend::retrieve_init")
        next(layerwise_retriever)
        torch.cuda.nvtx.range_pop()

        # Capture ret_mask and token_mask for cache miss detection.
        # Move to GPU once here to avoid per-layer pageable HtoD copies.
        torch.cuda.nvtx.range_push("epic_blend::capture_masks")
        ret_mask = getattr(self.cache_engine, "_blend_ret_mask", None)
        self._ret_mask = ret_mask.to(device=self.gpu_connector.device) if ret_mask is not None else None
        self._token_mask = mask.to(device=self.gpu_connector.device) if mask is not None else None
        torch.cuda.nvtx.range_pop()

        yield

        retrieve_times_ms = []
        compute_times_ms = []
        for i in range(self.num_layers):
            torch.cuda.nvtx.range_push(f"epic_blend::retrieve_layer_{i}")
            t0 = time.perf_counter()
            next(layerwise_retriever)
            retrieve_times_ms.append((time.perf_counter() - t0) * 1000.0)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push(f"epic_blend::compute_layer_{i}")
            t0 = time.perf_counter()
            next(layerwise_model_executor)
            compute_times_ms.append((time.perf_counter() - t0) * 1000.0)
            torch.cuda.nvtx.range_pop()
            yield

        r_max = max(retrieve_times_ms)
        c_max = max(compute_times_ms)
        logger.info(
            "epic_blend layerwise CPU timings (ms) | "
            "retrieve total=%.3f max=%.3f@L%d | "
            "compute  total=%.3f max=%.3f@L%d",
            sum(retrieve_times_ms),
            r_max, retrieve_times_ms.index(r_max),
            sum(compute_times_ms),
            c_max, compute_times_ms.index(c_max),
        )

        torch.cuda.nvtx.range_push("epic_blend::retrieve_finalize")
        next(layerwise_retriever)
        torch.cuda.nvtx.range_pop()

        self.metadata.clean()
        self._ret_mask = None
        self._token_mask = None
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Perform blending for the given tokens."""

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        for i in range(self.num_layers + 2):
            next(layerwise_blender)
