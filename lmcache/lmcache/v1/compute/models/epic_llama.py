# SPDX-License-Identifier: Apache-2.0
"""EPIC-specific model executor for Llama-family models.

Key differences from LMCLlamaModel (CacheBlend):
1. After check layer (status=1), hidden_states and residual are filtered
   to imp_indices. Subsequent layers operate on the filtered subset.
2. Uses EPIC attention with partial bias instead of CacheBlend's index-sliced attention.
3. Status 2 layers: Q/K/V are computed from filtered hidden_states only.
"""

# Third Party
import torch
from torch import nn

# First Party
from lmcache.v1.compute.attention.utils import infer_attn_backend_from_vllm
from lmcache.v1.compute.positional_encoding import get_fused_rope


class LMCEPICBaseModel(nn.Module):
    """Base model executor for EPIC blending.

    Compared to LMCBaseModel, this handles:
    - Passing attn_bias from blender to attention layer
    - Filtering hidden_states after check layer for status 2 processing
    """

    def __init__(self, vllm_model, blender, enable_sparse: bool = False):
        super().__init__()
        self.vllm_model = vllm_model
        self.blender = blender

        self.num_layers = len(vllm_model.model.layers)

        self.vllm_attn_layers = []
        self.lmc_attn_layers = []
        for i in range(self.num_layers):
            vllm_attn = vllm_model.model.layers[i].self_attn.attn
            self.vllm_attn_layers.append(vllm_attn)

            # Use EPIC attention backend
            from lmcache.v1.compute.attention.epic_attention import (
                EPICFlashAttnBackend,
            )
            self.lmc_attn_layers.append(EPICFlashAttnBackend(vllm_attn))

        # Rotary embedding setup
        rotary_emb = vllm_model.model.layers[0].self_attn.rotary_emb
        head_dim = rotary_emb.head_size
        max_position_embeddings = rotary_emb.max_position_embeddings
        base = rotary_emb.base
        is_neox_style = rotary_emb.is_neox_style
        dtype = rotary_emb.dtype
        self.fused_rotary_emb = get_fused_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=base,
            rope_scaling=None,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

    def _process_qkv(self, q, k, v, layer):
        """Process QKV tensors. Override in subclasses for model-specific logic."""
        return q, k, v

    def _post_self_attention(self, hidden_states, layer):
        """Hook applied to attention output, before residual fusion.

        Default is identity. Models like GLM-4 override to insert an
        extra layernorm (post_self_attn_layernorm).
        """
        return hidden_states

    def _post_mlp(self, hidden_states, layer):
        """Hook applied to MLP output, before the next layer's input layernorm.

        Default is identity. Models like GLM-4 override to insert an
        extra layernorm (post_mlp_layernorm).
        """
        return hidden_states

    @torch.compile
    def compute_layer(self, input_ids: torch.Tensor):
        torch.cuda.nvtx.range_push("epic_compute::embedding_and_init")
        input_ids = input_ids.cuda()
        if hasattr(self.vllm_model, "get_input_embeddings"):
            hidden_states = self.vllm_model.get_input_embeddings(input_ids)
        else:
            hidden_states = self.vllm_model.embed_input_ids(input_ids)
        residual = None
        attn_output = None

        attn_metadata = self.lmc_attn_layers[0].init_attn_metadata(
            input_ids=input_ids,
        )
        torch.cuda.nvtx.range_pop()

        for idx, layer in enumerate(
            self.vllm_model.model.layers[
                self.vllm_model.model.start_layer : self.vllm_model.model.end_layer
            ]
        ):
            # Self Attention
            torch.cuda.nvtx.range_push(f"epic_compute::layernorm_qkv_layer_{idx}")
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(
                    hidden_states, residual
                )

            qkv, _ = layer.self_attn.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size,
                ],
                dim=-1,
            )

            # Model-specific QKV processing (e.g., Qwen3's q_norm/k_norm)
            q, k, v = self._process_qkv(q, k, v, layer)
            torch.cuda.nvtx.range_pop()

            # EPIC blender processing (importance scoring + partial attention setup)
            q, k, v, residual, attn_output, attn_metadata = (
                self.blender.process_qkv(
                    q, k, v, residual, idx, attn_output, attn_metadata
                )
            )

            num_heads = self.vllm_attn_layers[idx].num_heads
            num_kv_heads = self.vllm_attn_layers[idx].num_kv_heads
            head_size = self.vllm_attn_layers[idx].head_size

            torch.cuda.nvtx.range_push(f"epic_compute::attn_layer_{idx}")
            q = q.view(-1, num_heads, head_size)
            k = k.view(-1, num_kv_heads, head_size)
            v = v.view(-1, num_kv_heads, head_size)
            attn_output = attn_output.view(-1, num_heads, head_size)

            # EPIC attention: use partial bias when available
            attn_output = self.lmc_attn_layers[idx].forward_contiguous(
                q, k, v, attn_output, attn_metadata,
                attn_bias=self.blender.metadata.attn_bias,
            )
            torch.cuda.nvtx.range_pop()

            attn_output = attn_output.view(-1, num_heads * head_size)
            k = k.view(-1, num_kv_heads * head_size)
            v = v.view(-1, num_kv_heads * head_size)

            torch.cuda.nvtx.range_push(f"epic_compute::o_proj_layer_{idx}")
            hidden_states, _ = layer.self_attn.o_proj(attn_output)
            # Model-specific hook (e.g., GLM-4 post_self_attn_layernorm)
            hidden_states = self._post_self_attention(hidden_states, layer)
            torch.cuda.nvtx.range_pop()

            # Fully Connected
            torch.cuda.nvtx.range_push(f"epic_compute::ffn_layer_{idx}")
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = layer.mlp(hidden_states)
            # Model-specific hook (e.g., GLM-4 post_mlp_layernorm)
            hidden_states = self._post_mlp(hidden_states, layer)
            torch.cuda.nvtx.range_pop()

            yield


class LMCEPICLlamaModel(LMCEPICBaseModel):
    """EPIC model executor for Llama (no special QKV processing)."""

    def _process_qkv(self, q, k, v, layer):
        return q, k, v


class LMCEPICQwen3Model(LMCEPICBaseModel):
    """EPIC model executor for Qwen3 (with q_norm and k_norm)."""

    def _process_qkv(self, q, k, v, layer):
        q_by_head = q.view(
            *q.shape[:-1],
            q.shape[-1] // layer.self_attn.head_dim,
            layer.self_attn.head_dim,
        )
        q_by_head = layer.self_attn.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(
            *k.shape[:-1],
            k.shape[-1] // layer.self_attn.head_dim,
            layer.self_attn.head_dim,
        )
        k_by_head = layer.self_attn.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k, v


class LMCEPICGlm4Model(LMCEPICBaseModel):
    """EPIC model executor for GLM-4-0414 (Glm4ForCausalLM).

    Differences vs Llama:
      - Adds `post_self_attn_layernorm` after the attention output.
      - Adds `post_mlp_layernorm` after the MLP output.
      - Uses partial RoPE; this is handled by vllm's own `rotary_emb` which
        EPIC's blender invokes directly, so no model-specific RoPE wiring
        is needed here.

    There is no q_norm / k_norm, so QKV is passed through unchanged.
    """

    def _process_qkv(self, q, k, v, layer):
        return q, k, v

    def _post_self_attention(self, hidden_states, layer):
        return layer.post_self_attn_layernorm(hidden_states)

    def _post_mlp(self, hidden_states, layer):
        return layer.post_mlp_layernorm(hidden_states)
