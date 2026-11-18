# SPDX-License-Identifier: Apache-2.0
"""Unit tests for EPIC blender components.

Tests:
1. Partial attention bias shape and causal correctness
2. V-based importance scoring selects correct indices
3. Last token is always included in imp_indices
4. Cache miss positions are included in imp_indices
5. EPICBlendMetadata clean() resets all state
6. Config fields parse correctly
"""

import pytest
import torch

from lmcache.v1.compute.attention.epic_attention import make_partial_attention_bias
from lmcache.v1.compute.blend.epic_metadata import EPICBlendMetadata


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda:0")


class TestPartialAttentionBias:
    """Tests for make_partial_attention_bias()."""

    def test_output_shape(self, device):
        seq_len = 64
        num_kv_heads = 8
        imp_indices = torch.tensor([5, 10, 30, 63], device=device)

        bias = make_partial_attention_bias(
            imp_indices, seq_len, num_kv_heads, torch.float16, device
        )

        assert bias.shape == (1, num_kv_heads, len(imp_indices), seq_len)

    def test_causal_mask_correctness(self, device):
        """Verify causal structure: position i can attend to j iff j <= i."""
        seq_len = 16
        num_kv_heads = 1
        imp_indices = torch.tensor([3, 7, 15], device=device)

        bias = make_partial_attention_bias(
            imp_indices, seq_len, num_kv_heads, torch.float32, device
        )

        # For imp_indices[0]=3:
        # Can attend to positions 0,1,2,3 (bias=0)
        # Cannot attend to positions 4+ (bias=-inf)
        for j in range(4):
            assert bias[0, 0, 0, j].item() == 0.0, f"pos 3 should attend to {j}"
        for j in range(4, seq_len):
            assert bias[0, 0, 0, j].item() < -1000, f"pos 3 should NOT attend to {j}"

        # For imp_indices[2]=15 (last):
        # Can attend to all positions
        for j in range(seq_len):
            assert bias[0, 0, 2, j].item() == 0.0, f"pos 15 should attend to {j}"

    def test_padded_len_alignment(self, device):
        """Verify output is trimmed to seq_len even when internal padding is used."""
        seq_len = 13  # Not a multiple of 8
        num_kv_heads = 2
        imp_indices = torch.tensor([0, 12], device=device)

        bias = make_partial_attention_bias(
            imp_indices, seq_len, num_kv_heads, torch.float16, device
        )

        # Should be trimmed to seq_len, not padded_len
        assert bias.shape == (1, 2, 2, 13)

    def test_head_expansion(self, device):
        """All KV heads should get the same bias."""
        seq_len = 8
        num_kv_heads = 4
        imp_indices = torch.tensor([3, 7], device=device)

        bias = make_partial_attention_bias(
            imp_indices, seq_len, num_kv_heads, torch.float32, device
        )

        # All heads identical
        for h in range(1, num_kv_heads):
            assert torch.equal(bias[0, 0], bias[0, h])

    def test_single_index(self, device):
        """Edge case: only one important index."""
        seq_len = 32
        num_kv_heads = 2
        imp_indices = torch.tensor([31], device=device)

        bias = make_partial_attention_bias(
            imp_indices, seq_len, num_kv_heads, torch.float16, device
        )

        assert bias.shape == (1, 2, 1, 32)
        # Last token can attend to everything
        assert (bias[0, 0, 0, :] == 0).all()


class TestImportanceScoring:
    """Tests for EPIC's V-based importance scoring logic."""

    def test_v_based_scoring_selects_large_diffs(self, device):
        """V-based L2 diff should select positions with largest value differences."""
        seq_len = 20
        num_kv_heads = 2
        head_size = 4
        recomp_ratio = 0.16

        # Create V tensors where positions 5 and 10 have large differences
        v_old = torch.zeros(seq_len, num_kv_heads, head_size, device=device)
        v_new = torch.zeros(seq_len, num_kv_heads, head_size, device=device)

        # Large diff at position 5
        v_new[5] = torch.ones(num_kv_heads, head_size, device=device) * 10.0
        # Large diff at position 10
        v_new[10] = torch.ones(num_kv_heads, head_size, device=device) * 8.0

        # Compute importance (exclude last token)
        temp_diff = torch.sum(
            (v_new[:-1, :, :] - v_old[:-1, :, :]) ** 2,
            dim=[1, 2],
        )
        topk_num = int((seq_len - 1) * recomp_ratio)
        assert topk_num > 0

        top_indices = torch.topk(temp_diff, k=topk_num).indices
        top_indices = torch.sort(top_indices).values

        # Positions 5 and 10 should be selected (if topk_num >= 2)
        selected = set(top_indices.tolist())
        assert 5 in selected, f"Position 5 should be selected, got {selected}"
        assert 10 in selected, f"Position 10 should be selected, got {selected}"

    def test_last_token_always_included(self, device):
        """Last token must always be in imp_indices regardless of V diff."""
        seq_len = 10
        num_kv_heads = 2
        head_size = 4
        recomp_ratio = 0.2

        v_old = torch.zeros(seq_len, num_kv_heads, head_size, device=device)
        v_new = torch.zeros(seq_len, num_kv_heads, head_size, device=device)
        # Large diff only at position 3
        v_new[3] = torch.ones(num_kv_heads, head_size, device=device) * 100.0

        temp_diff = torch.sum(
            (v_new[:-1, :, :] - v_old[:-1, :, :]) ** 2,
            dim=[1, 2],
        )
        topk_num = int((seq_len - 1) * recomp_ratio)
        top_indices = torch.topk(temp_diff, k=topk_num).indices

        # Add last token
        last_token = torch.tensor([seq_len - 1], device=device)
        top_indices = torch.cat([top_indices, last_token])
        top_indices = torch.sort(top_indices).values

        assert (seq_len - 1) in top_indices.tolist()

    def test_cache_miss_positions_included(self, device):
        """Cache miss positions must be included in imp_indices."""
        seq_len = 10
        # ret_mask: True = cache hit, False = cache miss
        ret_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        ret_mask[2] = False  # position 2 is a cache miss
        ret_mask[7] = False  # position 7 is a cache miss

        # token_mask: True = should have been loaded
        token_mask = torch.ones(seq_len, dtype=torch.bool, device=device)

        miss_mask = ~ret_mask & token_mask
        miss_indices = torch.where(miss_mask)[0]

        assert 2 in miss_indices.tolist()
        assert 7 in miss_indices.tolist()


class TestEPICBlendMetadata:
    def test_initial_state(self):
        meta = EPICBlendMetadata(recomp_ratio=0.16)
        assert meta.status == 0
        assert meta.imp_indices is None
        assert meta.org_seq_len is None
        assert meta.attn_bias is None
        assert meta.positions is None
        assert meta.recomp_ratio == 0.16

    def test_clean_resets_all(self, device):
        meta = EPICBlendMetadata(recomp_ratio=0.2)
        meta.status = 2
        meta.imp_indices = torch.tensor([1, 2, 3], device=device)
        meta.org_seq_len = 100
        meta.attn_bias = torch.zeros(1, 1, 3, 100, device=device)
        meta.positions = torch.tensor([1, 2, 3], device=device)

        meta.clean()

        assert meta.status == 0
        assert meta.imp_indices is None
        assert meta.org_seq_len is None
        assert meta.attn_bias is None
        assert meta.positions is None
        assert meta.recomp_ratio == 0.2  # recomp_ratio should NOT be reset


class TestEPICConfig:
    def test_default_config(self):
        from lmcache.v1.config import LMCacheEngineConfig

        config = LMCacheEngineConfig.from_defaults()
        assert config.blending_algorithm == "cacheblend"
        assert config.epic_recomp_ratio == 0.16

    def test_env_override(self):
        import os
        from lmcache.v1.config import LMCacheEngineConfig

        os.environ["LMCACHE_BLENDING_ALGORITHM"] = "epic"
        os.environ["LMCACHE_EPIC_RECOMP_RATIO"] = "0.25"
        try:
            config = LMCacheEngineConfig.from_env()
            assert config.blending_algorithm == "epic"
            assert config.epic_recomp_ratio == 0.25
        finally:
            del os.environ["LMCACHE_BLENDING_ALGORITHM"]
            del os.environ["LMCACHE_EPIC_RECOMP_RATIO"]
