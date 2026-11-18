# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.compute.models.base import LMCBaseModel


class LMCGlm4Model(LMCBaseModel):
    """LMCache compute model for GLM-4-0414 (Glm4ForCausalLM).

    Differences vs Llama:
      - Adds `post_self_attn_layernorm` after the attention output (before
        the residual is fused with the next layernorm).
      - Adds `post_mlp_layernorm` after the MLP output.
      - Uses partial RoPE (rotary_dim = head_size / 2, non-NEOX style).
        The partial RoPE is handled by the rope wiring in `base.py` and
        `positional_encoding.py`; nothing model-specific needed here.

    There is no q_norm / k_norm, so QKV is passed through unchanged.
    """

    def _process_qkv(self, q, k, v, layer):
        return q, k, v

    def _post_self_attention(self, hidden_states, layer):
        return layer.post_self_attn_layernorm(hidden_states)

    def _post_mlp(self, hidden_states, layer):
        return layer.post_mlp_layernorm(hidden_states)
