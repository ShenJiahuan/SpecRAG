// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include "mem_kernels.cuh"
#include "cachegen_kernels.cuh"
#include "pos_kernels.cuh"
#include "batched_copy.cuh"
#include "mem_alloc.h"
#include "disk_reader.h"
#include "utils.h"
#include <torch/torch.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(c_ops, m) {
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer);
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer);
  m.def("single_layer_kv_transfer_sgl", &single_layer_kv_transfer_sgl);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("encode_fast_new", &encode_cuda_new);
  m.def("decode_fast_new", &decode_cuda_new);
  m.def("decode_fast_prefsum", &decode_cuda_prefsum);
  m.def("calculate_cdf", &calculate_cdf);
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr);
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr);
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("get_gpu_pci_bus_id", &get_gpu_pci_bus_id);
  m.def("cpu_spin_probe", &cpu_spin_probe,
        "Fixed-count register-only arithmetic loop. Returns tuple "
        "(wall_us, thread_cpu_us, tsc_cycles, cpu_before, cpu_after, "
        "freq_khz, voluntary_ctxsw, involuntary_ctxsw, perf_cycles, "
        "perf_ref_cycles, perf_instructions).",
        py::arg("num_iters"));
  m.def("batched_read_files", &batched_read_files,
        "Read multiple files into pre-allocated buffers without holding the GIL",
        py::arg("paths"), py::arg("buffer_ptrs"), py::arg("buffer_sizes"),
        py::arg("use_odirect") = false);
  m.def("pread_file", &pread_file,
        "Read a file (or portion via offset) into a pre-allocated buffer",
        py::arg("path"), py::arg("buffer_ptr"), py::arg("size"),
        py::arg("offset"), py::arg("use_odirect") = false);
  m.def("readv_chunked_file", &readv_chunked_file,
        "Read a single file into multiple scattered buffers (scatter-gather read)",
        py::arg("path"), py::arg("buffer_ptrs"), py::arg("buffer_sizes"),
        py::arg("file_offset"), py::arg("use_odirect") = false);
  m.def("writev_chunked_file", &writev_chunked_file,
        "Write multiple buffers into a single file using scatter-gather I/O",
        py::arg("path"), py::arg("buffer_ptrs"), py::arg("buffer_sizes"),
        py::arg("use_odirect") = false);
  m.def("batched_d2d_copy", &batched_d2d_copy,
        "Perform multiple D2D copies in a single kernel launch",
        py::arg("srcs"), py::arg("dsts"));
  m.def("batched_d2d_copy_v2", &batched_d2d_copy_v2,
        "D2D copies with offset-based dst addressing (no Python slicing)",
        py::arg("src_chunks"), py::arg("dst_buf"), py::arg("offset_pairs"));
}
