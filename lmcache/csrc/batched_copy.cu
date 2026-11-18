// SPDX-License-Identifier: Apache-2.0

#include "batched_copy.cuh"
#include "cuda_compat.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace lmc {

// ---------------------------------------------------------------------------
// Kernel: each block handles one copy, using vectorized 16-byte (float4) loads
// to maximise memory bandwidth.  A 2-D grid is used so that large copies can
// span multiple blocks (grid.x) while grid.y indexes the copy.
// ---------------------------------------------------------------------------
__global__ void batched_memcpy_kernel(const int64_t* __restrict__ src_ptrs,
                                      int64_t* __restrict__ dst_ptrs,
                                      const int64_t* __restrict__ num_float4s,
                                      int num_copies) {
  const int copy_idx = blockIdx.y;
  if (copy_idx >= num_copies) return;

  const float4* src =
      reinterpret_cast<const float4*>(src_ptrs[copy_idx]);
  float4* dst = reinterpret_cast<float4*>(dst_ptrs[copy_idx]);
  const int64_t n = num_float4s[copy_idx];

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    dst[i] = src[i];
  }
}

}  // namespace lmc
  
// ---------------------------------------------------------------------------
// Host wrapper exposed via pybind11
// ---------------------------------------------------------------------------
void batched_d2d_copy(const std::vector<torch::Tensor>& srcs,
                      const std::vector<torch::Tensor>& dsts) {
  const int n = static_cast<int>(srcs.size());
  if (n == 0) return;

  TORCH_CHECK(static_cast<int>(dsts.size()) == n,
              "srcs and dsts must have the same length");

  // Build pointer / size arrays on the host.
  std::vector<int64_t> h_src_ptrs(n);
  std::vector<int64_t> h_dst_ptrs(n);
  std::vector<int64_t> h_num_float4s(n);

  int64_t max_float4s = 0;

  for (int i = 0; i < n; ++i) {
    TORCH_CHECK(srcs[i].is_cuda(), "src tensor ", i, " is not on CUDA");
    TORCH_CHECK(dsts[i].is_cuda(), "dst tensor ", i, " is not on CUDA");
    TORCH_CHECK(srcs[i].is_contiguous(), "src tensor ", i, " is not contiguous");
    TORCH_CHECK(dsts[i].is_contiguous(), "dst tensor ", i, " is not contiguous");

    const int64_t bytes = srcs[i].nbytes();
    TORCH_CHECK(bytes == dsts[i].nbytes(),
                "src and dst tensor ", i, " size mismatch: ",
                bytes, " vs ", dsts[i].nbytes());
    TORCH_CHECK(bytes % 16 == 0,
                "tensor ", i, " byte size (", bytes,
                ") is not a multiple of 16");

    h_src_ptrs[i] = reinterpret_cast<int64_t>(srcs[i].data_ptr());
    h_dst_ptrs[i] = reinterpret_cast<int64_t>(dsts[i].data_ptr());
    h_num_float4s[i] = bytes / 16;
    if (h_num_float4s[i] > max_float4s) {
      max_float4s = h_num_float4s[i];
    }
  }

  // Record stream usage for source tensors so the caching allocator
  // does not reuse their memory before this kernel finishes reading.
  const auto device = srcs[0].device();
  const at::cuda::OptionalCUDAGuard device_guard(device);
  const auto cuda_stream = at::cuda::getCurrentCUDAStream();

  for (int i = 0; i < n; ++i) {
    c10::cuda::CUDACachingAllocator::recordStream(
        srcs[i].storage().data_ptr(), cuda_stream);
  }

  // Copy the small metadata arrays to device on the SAME stream used for
  // the kernel launch.  We use cudaMemcpyAsync so the H2D and kernel are
  // ordered on the same stream; non-pinned host memory causes the driver
  // to stage through an internal buffer, so the stack vectors are consumed
  // before this function returns.
  const cudaStream_t stream = cuda_stream.stream();

  auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto d_src_ptrs = torch::empty({n}, opts);
  auto d_dst_ptrs = torch::empty({n}, opts);
  auto d_num_float4s = torch::empty({n}, opts);

  cudaMemcpyAsync(d_src_ptrs.data_ptr(), h_src_ptrs.data(),
                  n * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_dst_ptrs.data_ptr(), h_dst_ptrs.data(),
                  n * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_num_float4s.data_ptr(), h_num_float4s.data(),
                  n * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

  // Launch kernel.
  const int threads = 256;
  // Use enough x-blocks so that every thread has work for the largest copy.
  const int blocks_x =
      std::min<int64_t>((max_float4s + threads - 1) / threads, 1024);
  const dim3 grid(blocks_x, n);

  lmc::batched_memcpy_kernel<<<grid, threads, 0, stream>>>(
      d_src_ptrs.data_ptr<int64_t>(), d_dst_ptrs.data_ptr<int64_t>(),
      d_num_float4s.data_ptr<int64_t>(), n);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Variant that accepts src chunk tensors [2, chunk_len, H] + a single dst
// buffer [2, T, H] + an offset table [N, 2] (start, end per chunk).
// Avoids Python-side tensor slicing and list building.
// ---------------------------------------------------------------------------
void batched_d2d_copy_v2(const std::vector<torch::Tensor>& src_chunks,
                          const torch::Tensor& dst_buf,
                          const torch::Tensor& offset_pairs) {
  const int n = static_cast<int>(src_chunks.size());
  if (n == 0) return;

  TORCH_CHECK(offset_pairs.dim() == 2 && offset_pairs.size(0) == n &&
                  offset_pairs.size(1) == 2,
              "offset_pairs must have shape [N, 2]");
  TORCH_CHECK(dst_buf.is_cuda() && dst_buf.is_contiguous(),
              "dst_buf must be a contiguous CUDA tensor");
  TORCH_CHECK(dst_buf.dim() == 3 && dst_buf.size(0) == 2,
              "dst_buf must have shape [2, T, H]");

  const int num_copies = 2 * n;  // K and V for each chunk
  const int64_t H = dst_buf.size(2);
  const int64_t T = dst_buf.size(1);
  const int64_t elem_bytes = dst_buf.element_size();
  const int64_t row_bytes = H * elem_bytes;
  const int64_t plane_bytes = T * row_bytes;

  const char* dst_base = static_cast<const char*>(dst_buf.data_ptr());

  // offset_pairs may be on CPU (pre-computed), use accessor.
  auto offset_pairs_cpu = offset_pairs.cpu();
  auto acc = offset_pairs_cpu.accessor<int64_t, 2>();

  std::vector<int64_t> h_src_ptrs(num_copies);
  std::vector<int64_t> h_dst_ptrs(num_copies);
  std::vector<int64_t> h_num_float4s(num_copies);
  int64_t max_float4s = 0;

  for (int i = 0; i < n; ++i) {
    const auto& src = src_chunks[i];
    TORCH_CHECK(src.is_cuda() && src.is_contiguous(),
                "src_chunks[", i, "] must be contiguous CUDA tensor");
    TORCH_CHECK(src.dim() == 3 && src.size(0) == 2,
                "src_chunks[", i, "] must have shape [2, chunk_len, H]");

    const int64_t s = acc[i][0];
    const int64_t e = acc[i][1];
    const int64_t chunk_len = e - s;
    const int64_t bytes = chunk_len * row_bytes;
    TORCH_CHECK(bytes % 16 == 0,
                "chunk ", i, " byte size (", bytes,
                ") is not a multiple of 16");
    const int64_t nf4 = bytes / 16;

    const char* src_base = static_cast<const char*>(src.data_ptr());
    const int64_t src_plane_bytes = chunk_len * row_bytes;

    // K copy
    h_src_ptrs[2 * i] = reinterpret_cast<int64_t>(src_base);
    h_dst_ptrs[2 * i] = reinterpret_cast<int64_t>(dst_base + s * row_bytes);
    h_num_float4s[2 * i] = nf4;

    // V copy
    h_src_ptrs[2 * i + 1] =
        reinterpret_cast<int64_t>(src_base + src_plane_bytes);
    h_dst_ptrs[2 * i + 1] =
        reinterpret_cast<int64_t>(dst_base + plane_bytes + s * row_bytes);
    h_num_float4s[2 * i + 1] = nf4;

    if (nf4 > max_float4s) max_float4s = nf4;
  }

  // Tell PyTorch's caching allocator that the source tensors are being
  // read on the current stream (load_stream).  Without this, the
  // allocator only knows these buffers were written on spec_load_stream
  // and may reuse the memory before this kernel finishes reading.
  // This is exactly what torch.copy_() does internally.
  const auto device = dst_buf.device();
  const at::cuda::OptionalCUDAGuard device_guard(device);
  const auto cuda_stream = at::cuda::getCurrentCUDAStream();

  for (int i = 0; i < n; ++i) {
    c10::cuda::CUDACachingAllocator::recordStream(
        src_chunks[i].storage().data_ptr(), cuda_stream);
  }

  // Copy metadata arrays to device on the SAME stream as the kernel,
  // using cudaMemcpyAsync so ordering is guaranteed.  Non-pinned host
  // memory causes the driver to stage through an internal buffer, so
  // the stack vectors are consumed before this function returns.
  const cudaStream_t stream = cuda_stream.stream();

  auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto d_src_ptrs = torch::empty({num_copies}, opts);
  auto d_dst_ptrs = torch::empty({num_copies}, opts);
  auto d_num_float4s = torch::empty({num_copies}, opts);

  cudaMemcpyAsync(d_src_ptrs.data_ptr(), h_src_ptrs.data(),
                  num_copies * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_dst_ptrs.data_ptr(), h_dst_ptrs.data(),
                  num_copies * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_num_float4s.data_ptr(), h_num_float4s.data(),
                  num_copies * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

  const int threads = 256;
  const int blocks_x =
      std::min<int64_t>((max_float4s + threads - 1) / threads, 1024);
  const dim3 grid(blocks_x, num_copies);

  lmc::batched_memcpy_kernel<<<grid, threads, 0, stream>>>(
      d_src_ptrs.data_ptr<int64_t>(), d_dst_ptrs.data_ptr<int64_t>(),
      d_num_float4s.data_ptr<int64_t>(), num_copies);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
