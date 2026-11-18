#include "disk_reader.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/uio.h>
#include <cerrno>
#include <cstring>
#include <algorithm>
#include <stdexcept>

// Helper: build open flags depending on use_odirect.
static int read_flags(bool use_odirect) {
    return use_odirect ? (O_RDONLY | O_DIRECT) : O_RDONLY;
}

static int write_flags(bool use_odirect) {
    return use_odirect
        ? (O_CREAT | O_WRONLY | O_TRUNC | O_DIRECT)
        : (O_CREAT | O_WRONLY | O_TRUNC);
}

// Read a single file fully into buf. Returns bytes read or -1 on error.
static ssize_t read_one_file(const char* path, void* buf, size_t size,
                             bool use_odirect) {
    int fd = open(path, read_flags(use_odirect));
    if (fd < 0) return -1;

    size_t total = 0;
    while (total < size) {
        ssize_t n = read(fd, static_cast<char*>(buf) + total, size - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            close(fd);
            return -1;
        }
        if (n == 0) break;  // EOF
        total += n;
    }
    close(fd);
    return static_cast<ssize_t>(total);
}

std::vector<ssize_t> batched_read_files(
    const std::vector<std::string>& paths,
    const std::vector<uintptr_t>& buffer_ptrs,
    const std::vector<size_t>& buffer_sizes,
    bool use_odirect)
{
    size_t n = paths.size();
    if (buffer_ptrs.size() != n || buffer_sizes.size() != n) {
        throw std::runtime_error(
            "batched_read_files: paths, buffer_ptrs, buffer_sizes must have same length");
    }

    std::vector<ssize_t> results(n);

    // Release the GIL for the entire batch of reads
    py::gil_scoped_release release;

    for (size_t i = 0; i < n; i++) {
        results[i] = read_one_file(
            paths[i].c_str(),
            reinterpret_cast<void*>(buffer_ptrs[i]),
            buffer_sizes[i],
            use_odirect);
    }

    return results;
}

ssize_t pread_file(const std::string& path, uintptr_t buffer_ptr,
                   size_t size, size_t offset,
                   bool use_odirect)
{
    py::gil_scoped_release release;

    int fd = open(path.c_str(), read_flags(use_odirect));
    if (fd < 0) return -1;

    void* buf = reinterpret_cast<void*>(buffer_ptr);
    size_t total = 0;
    while (total < size) {
        ssize_t n = ::pread(fd, static_cast<char*>(buf) + total,
                            size - total, offset + total);
        if (n < 0) {
            if (errno == EINTR) continue;
            close(fd);
            return -1;
        }
        if (n == 0) break;  // EOF
        total += n;
    }
    close(fd);
    return static_cast<ssize_t>(total);
}

ssize_t readv_chunked_file(const std::string& path,
                           const std::vector<uintptr_t>& buffer_ptrs,
                           const std::vector<size_t>& buffer_sizes,
                           size_t file_offset,
                           bool use_odirect)
{
    size_t n = buffer_ptrs.size();
    if (buffer_sizes.size() != n) {
        throw std::runtime_error(
            "readv_chunked_file: buffer_ptrs and buffer_sizes must have same length");
    }

    py::gil_scoped_release release;

    int fd = open(path.c_str(), read_flags(use_odirect));
    if (fd < 0) return -1;

    // Use preadv for scatter-gather read — one syscall per batch.
    // IOV_MAX is typically 1024; batch if n exceeds it.
    ssize_t total_read = 0;
    size_t idx = 0;
    off_t cur_offset = static_cast<off_t>(file_offset);
    while (idx < n) {
        size_t batch = std::min(n - idx, static_cast<size_t>(1024));
        std::vector<struct iovec> iovs(batch);
        size_t batch_bytes = 0;
        for (size_t i = 0; i < batch; i++) {
            iovs[i].iov_base = reinterpret_cast<void*>(buffer_ptrs[idx + i]);
            iovs[i].iov_len = buffer_sizes[idx + i];
            batch_bytes += buffer_sizes[idx + i];
        }

        ssize_t nr;
        do {
            nr = ::preadv(fd, iovs.data(), static_cast<int>(batch),
                          cur_offset);
        } while (nr < 0 && errno == EINTR);

        if (nr < 0) {
            close(fd);
            return -1;
        }

        total_read += nr;
        cur_offset += static_cast<off_t>(batch_bytes);
        idx += batch;
    }

    close(fd);
    return total_read;
}

ssize_t writev_chunked_file(const std::string& path,
                            const std::vector<uintptr_t>& buffer_ptrs,
                            const std::vector<size_t>& buffer_sizes,
                            bool use_odirect)
{
    size_t n = buffer_ptrs.size();
    if (buffer_sizes.size() != n) {
        throw std::runtime_error(
            "writev_chunked_file: buffer_ptrs and buffer_sizes must have same length");
    }

    py::gil_scoped_release release;

    int fd = open(path.c_str(), write_flags(use_odirect), 0644);
    if (fd < 0) return -1;

    // writev has IOV_MAX limit (typically 1024), so loop in batches
    size_t total_written = 0;
    size_t idx = 0;
    while (idx < n) {
        size_t batch = std::min(n - idx, static_cast<size_t>(1024));
        std::vector<struct iovec> iovs(batch);
        for (size_t i = 0; i < batch; i++) {
            iovs[i].iov_base = reinterpret_cast<void*>(buffer_ptrs[idx + i]);
            iovs[i].iov_len = buffer_sizes[idx + i];
        }
        ssize_t w = ::writev(fd, iovs.data(), static_cast<int>(batch));
        if (w < 0) {
            close(fd);
            return -1;
        }
        total_written += w;
        idx += batch;
    }
    close(fd);
    return static_cast<ssize_t>(total_written);
}
