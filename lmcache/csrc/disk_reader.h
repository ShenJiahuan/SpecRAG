#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

namespace py = pybind11;

// Read multiple files into pre-allocated buffers without holding the GIL.
// Each entry: (path, buffer_ptr, buffer_size).
// When use_odirect is true, files are opened with O_DIRECT to bypass
// the kernel page cache.  Buffers must be page-aligned and sizes must
// be multiples of the filesystem block size (typically 4096).
// Returns a vector of bytes_read per file (-1 on error).
std::vector<ssize_t> batched_read_files(
    const std::vector<std::string>& paths,
    const std::vector<uintptr_t>& buffer_ptrs,
    const std::vector<size_t>& buffer_sizes,
    bool use_odirect = false);

// Read a single file (or a portion via offset) into a pre-allocated buffer.
// Returns bytes read or -1 on error.
ssize_t pread_file(const std::string& path, uintptr_t buffer_ptr,
                   size_t size, size_t offset,
                   bool use_odirect = false);

// Read a single file into multiple scattered buffers (scatter-gather read).
// Opens the file once, then reads sequentially into each buffer.
// Returns total bytes read or -1 on error.
ssize_t readv_chunked_file(const std::string& path,
                           const std::vector<uintptr_t>& buffer_ptrs,
                           const std::vector<size_t>& buffer_sizes,
                           size_t file_offset,
                           bool use_odirect = false);

// Write multiple buffers into a single file using writev (scatter-gather I/O).
// Returns total bytes written or -1 on error.
ssize_t writev_chunked_file(const std::string& path,
                            const std::vector<uintptr_t>& buffer_ptrs,
                            const std::vector<size_t>& buffer_sizes,
                            bool use_odirect = false);
