# SPDX-License-Identifier: Apache-2.0
"""EPIC attention with partial bias support.

EPIC's attention differs from CacheBlend's:
- CacheBlend: slices both Q and K/V at imp_indices
- EPIC: slices only Q at imp_indices, uses FULL K/V with a causal attention bias mask

This module provides:
1. make_partial_attention_bias(): constructs the causal mask for selected query positions
2. EPICFlashAttnBackend: flash attention that supports the partial bias
"""

# Standard
from typing import TYPE_CHECKING, Optional

# Third Party
import torch
from xformers.ops import memory_efficient_attention

# First Party
from lmcache.v1.compute.attention.abstract import AttentionInterface
from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata

if TYPE_CHECKING:
    from lmcache.v1.compute.attention.metadata import LMCAttnMetadata


def make_partial_attention_bias(
    imp_indices: torch.Tensor,
    seq_len: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create causal attention mask where only imp_indices queries attend to full KV.

    Ported from EPIC's _make_partial_bias_mha() in xformers.py (lines 907-925).

    The mask is a causal upper-triangular mask, selecting only rows
    corresponding to imp_indices. Positions above the diagonal are set
    to -inf (finfo.min) to prevent attending to future tokens.

    Args:
        imp_indices: [num_imp] — selected important token indices
        seq_len: Original sequence length
        num_kv_heads: Number of KV heads
        dtype: Data type for the mask
        device: Device for the mask

    Returns:
        [1, num_kv_heads, len(imp_indices), seq_len] attention bias tensor
    """
    padded_len = (seq_len + 7) // 8 * 8
    # Upper triangular = 1 above diagonal, 0 on and below
    attn_mask = torch.triu(
        torch.ones(padded_len, padded_len, dtype=dtype, device=device),
        diagonal=1,
    )
    # Convert to attention bias: 0 where we can attend, -inf where we cannot
    attn_mask = (attn_mask * torch.finfo(dtype).min).view(
        1, 1, padded_len, padded_len
    )
    # Select only rows for important indices
    attn_mask = attn_mask[:, :, imp_indices]
    # Expand to all KV heads
    attn_mask = attn_mask.expand(1, num_kv_heads, -1, -1)
    # Trim to actual sequence length
    return attn_mask[:, :, :, :seq_len]


class EPICFlashAttnBackend(AttentionInterface):
    """Flash attention backend with partial bias support for EPIC.

    When attn_bias is provided (EPIC partial attention mode):
    - Uses xformers memory_efficient_attention with the bias mask
    - Q has shape [num_imp, num_heads, head_size] (selected queries only)
    - K/V have shape [seq_len, num_kv_heads, head_size] (full)

    When attn_bias is None (normal mode):
    - Delegates to standard flash_attn_varlen_func
    """

    def __init__(self, vllm_attn):
        from vllm.attention import Attention
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata

        self.vllm_attn = vllm_attn
        self.vllm_attn_impl: FlashAttentionImpl = vllm_attn.impl
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.get_scheduler_metadata = get_scheduler_metadata
        self.aot_schedule = False

        idx = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{idx}")

        self.num_heads = self.vllm_attn_impl.num_heads
        self.num_kv_heads = self.vllm_attn_impl.num_kv_heads
        self.head_size = self.vllm_attn_impl.head_size
        self.scale = self.vllm_attn_impl.scale
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "LMCAttnMetadata",
        attn_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if attn_bias is not None:
            return self._forward_with_bias(query, key, value, output, attn_bias)
        else:
            return self._forward_standard(query, key, value, output, attn_metadata)

    def _forward_with_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        """EPIC partial attention: selected Q attends to full K/V with bias mask.

        Uses xformers memory_efficient_attention which supports arbitrary
        attention bias masks.
        """
        # query: [num_imp, num_heads, head_size]
        # key: [seq_len, num_kv_heads, head_size]
        # value: [seq_len, num_kv_heads, head_size]
        # attn_bias: [1, num_kv_heads, num_imp, seq_len]

        if self.num_queries_per_kv > 1:
            # GQA: reshape to [B, M, G, H, K] format for xformers
            # query: [1, num_imp, num_kv_heads, num_queries_per_kv, head_size]
            q = query.view(
                query.shape[0], self.num_kv_heads, self.num_queries_per_kv, self.head_size
            ).unsqueeze(0)
            k = key.unsqueeze(0).unsqueeze(2).expand(
                1, key.shape[0], self.num_queries_per_kv, self.num_kv_heads, self.head_size
            ).permute(0, 1, 3, 2, 4)  # [1, seq_len, num_kv_heads, num_queries_per_kv, head_size]
            v = value.unsqueeze(0).unsqueeze(2).expand(
                1, value.shape[0], self.num_queries_per_kv, self.num_kv_heads, self.head_size
            ).permute(0, 1, 3, 2, 4)

            # For GQA with xformers, expand attn_bias
            # [1, num_kv_heads, num_queries_per_kv, num_imp, seq_len]
            bias = attn_bias.unsqueeze(2).expand(
                -1, -1, self.num_queries_per_kv, -1, -1
            )

            out = memory_efficient_attention(
                q, k, v,
                attn_bias=bias,
                scale=self.scale,
            )
            # out: [1, num_imp, num_kv_heads, num_queries_per_kv, head_size]
            out = out.squeeze(0).view(query.shape[0], self.num_heads, self.head_size)
        else:
            # MHA: simpler path
            # Reshape to [1, seq, heads, head_size] for xformers
            q = query.unsqueeze(0)  # [1, num_imp, num_heads, head_size]
            k = key.unsqueeze(0)    # [1, seq_len, num_kv_heads, head_size]
            v = value.unsqueeze(0)  # [1, seq_len, num_kv_heads, head_size]

            # attn_bias: [1, num_kv_heads, num_imp, seq_len]
            out = memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                scale=self.scale,
            )
            out = out.squeeze(0)  # [num_imp, num_heads, head_size]

        output[:out.shape[0]] = out
        return output

    def _forward_standard(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "LMCAttnMetadata",
    ) -> torch.Tensor:
        """Standard flash attention (same as LMCFlashAttnBackend)."""
        assert isinstance(attn_metadata, LMCFlashAttnMetadata)

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        cu_seqlens_k = attn_metadata.cu_seqlens_k
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len

        descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

        scheduler_metadata = self._schedule(
            batch_size=1,
            cu_query_lens=cu_seqlens_q,
            max_query_len=max_seqlen_q,
            seqlens=seqused_k,
            max_seq_len=max_seqlen_k,
            causal=True,
        )

        self.flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.vllm_attn_impl.alibi_slopes,
            window_size=self.vllm_attn_impl.sliding_window,
            block_table=None,
            softcap=self.vllm_attn_impl.logits_soft_cap,
            scheduler_metadata=scheduler_metadata,
            fa_version=self.vllm_attn_impl.vllm_flash_attn_version,
            q_descale=self.vllm_attn._q_scale.expand(descale_shape),
            k_descale=self.vllm_attn._k_scale.expand(descale_shape),
            v_descale=self.vllm_attn._v_scale.expand(descale_shape),
        )

        return output

    def _schedule(
        self, batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
    ):
        if self.aot_schedule:
            return self.get_scheduler_metadata(
                batch_size=batch_size,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_seq_len,
                cache_seqlens=seqlens,
                num_heads_q=self.vllm_attn_impl.num_heads_q,
                num_heads_kv=self.vllm_attn_impl.num_heads_kv,
                headdim=self.vllm_attn_impl.headdim,
                page_size=self.vllm_attn_impl.block_size,
                cu_seqlens_q=cu_query_lens,
                causal=causal,
                window_size=self.vllm_attn_impl.aot_sliding_window,
            )
        return None

    def init_attn_metadata(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> LMCFlashAttnMetadata:
        seq_len = input_ids.shape[0]
        device = input_ids.device
        return LMCFlashAttnMetadata(
            query_start_loc=torch.tensor(
                [0, seq_len], dtype=torch.int32, device=device
            ),
            seq_lens=torch.tensor([seq_len], device=device),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            max_query_len=seq_len,
            max_seq_len=seq_len,
        )
