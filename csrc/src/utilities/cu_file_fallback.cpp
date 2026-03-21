// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// This file implements a fallback path, to be used in cases in which cuFile is not available.
//

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string_view>

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <fmt/core.h>

#include "cu_file.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace {

/// Double-buffered pinned staging area, lazily allocated once per thread.
/// Two buffers enable overlapping pread() on one while cudaMemcpyAsync()
/// drains the other.
constexpr size_t CHUNK = 4 << 20;  // 4 MiB per buffer

struct PinnedBuffers {
    void* buf[2] = {nullptr, nullptr};
    cudaStream_t stream = nullptr;

    PinnedBuffers() = default;
    ~PinnedBuffers() {
        for (auto& b : buf)
            if (b) { cudaFreeHost(b); b = nullptr; }
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }

    void ensure_allocated() {
        if (buf[0]) return;
        CUDA_CHECK(cudaMallocHost(&buf[0], CHUNK));
        CUDA_CHECK(cudaMallocHost(&buf[1], CHUNK));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    PinnedBuffers(const PinnedBuffers&) = delete;
    PinnedBuffers& operator=(const PinnedBuffers&) = delete;
};

thread_local PinnedBuffers g_pinned;

} // namespace


/**
 * @brief Open a file for reading via the POSIX fallback path (no cuFile).
 *
 * @param file_name Path to the file to open.
 * @return A cuFileRef-like handle containing a POSIX file descriptor and the stored name.
 *
 * @throws std::runtime_error If the file cannot be opened (POSIX open() error).
 */
cuFileRef open_cufile(std::string file_name) {
    int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error(fmt::format("posix open error ({}) for file {}: {}", errno, file_name, strerror(errno)));
    }

    return {nullptr, fd, std::move(file_name)};
}

/**
 * @brief Read a byte range from a file into device memory using a pinned host staging buffer.
 *
 * Range semantics: [begin, end) in bytes.
 *
 * @param fd POSIX file descriptor opened for reading.
 * @param d_target Destination pointer in device memory; must be valid for (end - begin) bytes.
 * @param begin Starting byte offset within the file (inclusive).
 * @param end Ending byte offset within the file (exclusive).
 * @param file_name Name/path used only for diagnostics.
 *
 * @throws std::logic_error If @p end < @p begin.
 * @throws std::runtime_error On POSIX pread() failure, short reads, or CUDA memcpy failures.
 */
void cufile_read_bytes(int fd, std::byte* d_target, std::ptrdiff_t begin, std::ptrdiff_t end, std::string_view file_name) {
    if(end < begin) {
        throw std::logic_error(fmt::format("Invalid range {} - {} in cufile_read_bytes for {}", begin, end, file_name));
    }

    const size_t nbytes = static_cast<size_t>(end - begin);
    if (nbytes == 0) return;

    g_pinned.ensure_allocated();

    // Double-buffer: pread into buf[cur] while the GPU async-copies from buf[prev].
    int cur = 0;
    size_t done = 0;
    size_t in_flight = 0;  // bytes of the previous async copy still on the stream

    while (done < nbytes) {
        const size_t want = std::min(CHUNK, nbytes - done);
        const off_t off = static_cast<off_t>(begin + done);

        ssize_t r = ::pread(fd, g_pinned.buf[cur], want, off);
        if (r < 0)
            throw std::runtime_error(fmt::format("posix pread error ({}) for {}, range {} - {}: {}",
                                                 errno, file_name, off, off + want, strerror(errno)));
        if (r == 0) break;

        // Wait for the previous async copy to finish before reusing its buffer next iteration.
        if (in_flight > 0)
            CUDA_CHECK(cudaStreamSynchronize(g_pinned.stream));

        CUDA_CHECK(cudaMemcpyAsync(d_target + done, g_pinned.buf[cur],
                                   static_cast<size_t>(r),
                                   cudaMemcpyHostToDevice, g_pinned.stream));
        in_flight = static_cast<size_t>(r);
        done += in_flight;
        cur ^= 1;  // swap buffer
    }

    // Wait for the last async copy.
    if (in_flight > 0)
        CUDA_CHECK(cudaStreamSynchronize(g_pinned.stream));

    if (done != nbytes) {
        throw std::runtime_error(fmt::format("posix read short: expected {} bytes, got {}",
                                             nbytes, done));
    }
}

/**
 * @brief Dispatch dtype conversion between supported tensor element types.
 *
 * @param target Destination buffer (device memory) holding converted elements of type @p t_type.
 * @param source Source buffer (device memory) holding elements of type @p s_type.
 * @param size Number of elements (not bytes) to convert.
 * @param t_type Target element dtype.
 * @param s_type Source element dtype.
 *
 * @throws std::runtime_error If the dtype conversion pair is not supported.
 */
void convert_tensor_dispatch(std::byte* target, const std::byte* source, std::size_t size, ETensorDType t_type, ETensorDType s_type) {
    if(t_type == ETensorDType::FP32 && s_type == ETensorDType::BF16) {
        convert_dtype(reinterpret_cast<float*>(target), reinterpret_cast<const nv_bfloat16*>(source), size);
    } else if(t_type == ETensorDType::BF16 && s_type == ETensorDType::FP32) {
        convert_dtype(reinterpret_cast<nv_bfloat16*>(target), reinterpret_cast<const float*>(source), size);
    } else if(t_type == ETensorDType::BF16 && s_type == ETensorDType::FP16) {
        convert_dtype(reinterpret_cast<nv_bfloat16*>(target), reinterpret_cast<const half*>(source), size);
    } else if(t_type == ETensorDType::BF16 && s_type == ETensorDType::FP8_E4M3) {
        convert_dtype(reinterpret_cast<nv_bfloat16*>(target), reinterpret_cast<const __nv_fp8_e4m3*>(source), size);
    } else if ((t_type == ETensorDType::BYTE || s_type == ETensorDType::BYTE) &&
               get_dtype_size(t_type) == get_dtype_size(s_type)) {
        // BYTE is a raw storage type — identity copy when partner has same byte width.
        // Handles FP4_E2M1 <-> BYTE, FP8_E4M3 <-> BYTE, INT8 <-> BYTE, etc.
        CUDA_CHECK(cudaMemcpyAsync(target, source, size * get_dtype_size(t_type), cudaMemcpyDefault));
    } else {
        throw std::runtime_error(fmt::format("Unsupported conversion: {} -> {}", dtype_to_str(s_type), dtype_to_str(t_type)));
    }
}

/**
 * @brief Read a file byte range into a temporary device buffer and convert into a target dtype.
 *
 * This is a streaming fallback path: it reads chunks (up to @p buffer_size bytes) from the file
 * into @p d_buffer, converts them, and writes the converted elements into @p target.
 *
 * Range semantics: [begin, end) in bytes of the *source dtype* representation in the file.
 *
 * @param fd POSIX file descriptor opened for reading.
 * @param target Destination pointer in device memory for converted output (dtype @p t_type).
 * @param begin Starting byte offset within the file (inclusive).
 * @param end Ending byte offset within the file (exclusive).
 * @param file_name Name/path used only for diagnostics.
 * @param t_type Target element dtype for @p target.
 * @param s_type Source element dtype as stored in the file.
 * @param d_buffer Temporary device buffer used to hold raw bytes read from file.
 * @param buffer_size Capacity of @p d_buffer in bytes; also the chunk size for reads.
 *
 * @throws std::logic_error / std::runtime_error Propagates errors from reads, CUDA, and conversion.
 */
void cufile_convert_tensor(int fd, std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end,
                           std::string_view file_name, ETensorDType t_type, ETensorDType s_type,
                           std::byte* d_buffer, std::size_t buffer_size) {
    for(std::ptrdiff_t p = 0; p < end - begin; p += buffer_size) {
        std::ptrdiff_t amount = std::min(end - begin - p, (std::ptrdiff_t)buffer_size);
        cufile_read_bytes(fd, d_buffer, begin + p, begin + p + amount, file_name);
        convert_tensor_dispatch(target + p * get_dtype_size(t_type) / get_dtype_size(s_type),
                                d_buffer,
                                amount / get_dtype_size(s_type),
                                t_type, s_type);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}



cuFileRef::cuFileRef(std::string file_name) : cuFileRef(open_cufile(std::move(file_name)))
{

}

/**
 * @brief Destructor; closes the underlying POSIX file descriptor if open.
 *
 * Never throws.
 */
cuFileRef::~cuFileRef() noexcept
{
    if (mFileDescriptor >= 0) {
        close(mFileDescriptor);
        mFileDescriptor = -1;
    }
}

/**
 * @brief Read a byte range from the file into device memory.
 *
 * Range semantics: [begin, end) in bytes.
 *
 * @param target Destination pointer in device memory; must be valid for (end - begin) bytes.
 * @param begin Starting byte offset within the file (inclusive).
 * @param end Ending byte offset within the file (exclusive).
 *
 * @throws std::logic_error If @p end < @p begin.
 * @throws std::runtime_error On POSIX/CUDA errors or short reads.
 */
void cuFileRef::read_bytes(std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end)
{
    cufile_read_bytes(mFileDescriptor, target, begin, end, mFileName);
}

/**
 * @brief Read a source byte range from the file, convert element dtype, and write to device output.
 *
 * Range semantics: [begin, end) in bytes of the *source dtype* representation.
 *
 * @param target Destination pointer in device memory for converted output (dtype @p t_type).
 * @param begin Starting byte offset within the file (inclusive).
 * @param end Ending byte offset within the file (exclusive).
 * @param file_name Name/path used only for diagnostics (does not need to match the opened name).
 * @param t_type Target element dtype for @p target.
 * @param s_type Source element dtype as stored in the file.
 * @param d_buffer Temporary device buffer used to hold raw bytes read from file.
 * @param buffer_size Capacity of @p d_buffer in bytes; also the chunk size for reads.
 *
 * @throws std::logic_error / std::runtime_error Propagates errors from reads, CUDA, and conversion.
 */
void cuFileRef::read_and_convert(std::byte* target, std::ptrdiff_t begin, std::ptrdiff_t end,
        std::string_view file_name, ETensorDType t_type, ETensorDType s_type,
        std::byte* d_buffer, std::size_t buffer_size)
{
    cufile_convert_tensor(mFileDescriptor, target, begin, end, file_name, t_type, s_type, d_buffer, buffer_size);
}
