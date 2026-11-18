// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>
#include <vector>

/**
 * Perform multiple device-to-device memory copies in a single kernel launch.
 *
 * This eliminates the CPU-side overhead of launching many small cudaMemcpyAsync
 * calls (e.g., per-chunk speculative KV cache copies in CacheBlend).
 *
 * @param srcs  List of source tensors (must be contiguous, on the same GPU).
 * @param dsts  List of destination tensors (must be contiguous, on the same GPU,
 *              with matching numel to corresponding src).
 */
void batched_d2d_copy(const std::vector<torch::Tensor>& srcs,
                      const std::vector<torch::Tensor>& dsts);

/**
 * Variant that avoids Python-side tensor slicing.
 *
 * Instead of pre-built src/dst view lists, the caller passes:
 *   - src_chunks: list of [2, chunk_len, H] tensors (one per chunk)
 *   - dst_buf:    the destination buffer [2, T, H]
 *   - offset_pairs: [N, 2] int64 tensor with (start, end) token offsets
 *
 * The function computes dst pointers internally via pointer arithmetic.
 */
void batched_d2d_copy_v2(const std::vector<torch::Tensor>& src_chunks,
                          const torch::Tensor& dst_buf,
                          const torch::Tensor& offset_pairs);
