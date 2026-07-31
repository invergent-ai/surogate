// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Implementation shared by cu_file.cpp (GPUDirect Storage) and cu_file_fallback.cpp.
//

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cerrno>
#include <string_view>

#include <cuda_runtime.h>
#include <unistd.h>
#include <fmt/core.h>

#include "cu_file.h"
#include "cu_file_common.h"

#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace {

//! Staging chunk for the POSIX path. Large enough that a 4 GiB shard costs ~500 copies
//! rather than ~4000, small enough to stay negligible against model weights.
constexpr std::size_t kStagingChunk = 8u << 20;

/**
 * @brief Pinned host staging buffer, allocated once per thread and never freed.
 *
 * Reused across reads: cudaMallocHost is a synchronizing, page-locking call, and the
 * previous per-read alloc/free made loading a sharded checkpoint pay it thousands of
 * times. Deliberately leaked — releasing pinned memory from a static destructor races
 * CUDA teardown and surfaces as CUDA_ERROR_DEINITIALIZED.
 */
std::byte* staging_buffer() {
    thread_local std::byte* buffer = [] {
        void* p = nullptr;
        CUDA_CHECK(cudaMallocHost(&p, kStagingChunk));
        return static_cast<std::byte*>(p);
    }();
    return buffer;
}

}  // namespace

bool cufile_disabled_by_env() {
    const char* value = std::getenv("SUROGATE_DISABLE_CUFILE");
    return value != nullptr && value[0] != '\0' && std::string_view(value) != "0";
}

void posix_read_bytes(int fd,
                      std::byte* d_target,
                      std::ptrdiff_t begin,
                      std::ptrdiff_t end,
                      std::string_view file_name) {
    if (end < begin) {
        throw std::logic_error(fmt::format("Invalid range {} - {} in posix_read_bytes for {}", begin, end, file_name));
    }

    const std::size_t nbytes = static_cast<std::size_t>(end - begin);
    std::byte* hbuf = staging_buffer();

    std::size_t done = 0;
    while (done < nbytes) {
        const std::size_t want = std::min(kStagingChunk, nbytes - done);
        const off_t off = static_cast<off_t>(begin + done);
        ssize_t r = ::pread(fd, hbuf, want, off);
        if (r < 0) {
            throw std::runtime_error(fmt::format("posix pread error ({}) for {}, range {} - {}: {}",
                                                 errno,
                                                 file_name,
                                                 off,
                                                 off + static_cast<off_t>(want),
                                                 strerror(errno)));
        }
        if (r == 0) break;

        CUDA_CHECK(cudaMemcpy(d_target + done, hbuf, static_cast<std::size_t>(r), cudaMemcpyHostToDevice));
        done += static_cast<std::size_t>(r);
    }

    if (done != nbytes) {
        throw std::runtime_error(
            fmt::format("posix read short for {}: expected {} bytes, got {}", file_name, nbytes, done));
    }
}

void convert_tensor_dispatch(std::byte* target,
                             const std::byte* source,
                             std::size_t size,
                             ETensorDType t_type,
                             ETensorDType s_type) {
    if (t_type == ETensorDType::FP32 && s_type == ETensorDType::BF16) {
        convert_dtype(reinterpret_cast<float*>(target), reinterpret_cast<const nv_bfloat16*>(source), size);
    } else if (t_type == ETensorDType::BF16 && s_type == ETensorDType::FP32) {
        convert_dtype(reinterpret_cast<nv_bfloat16*>(target), reinterpret_cast<const float*>(source), size);
    } else if (t_type == ETensorDType::BF16 && s_type == ETensorDType::FP16) {
        convert_dtype(reinterpret_cast<nv_bfloat16*>(target), reinterpret_cast<const half*>(source), size);
    } else if (t_type == ETensorDType::BF16 && s_type == ETensorDType::FP8_E4M3) {
        convert_dtype(reinterpret_cast<nv_bfloat16*>(target), reinterpret_cast<const __nv_fp8_e4m3*>(source), size);
    } else if ((t_type == ETensorDType::BYTE || s_type == ETensorDType::BYTE) &&
               get_dtype_size(t_type) == get_dtype_size(s_type)) {
        // BYTE is a raw storage type — identity copy when partner has same byte width.
        // Handles FP4_E2M1 <-> BYTE, FP8_E4M3 <-> BYTE, INT8 <-> BYTE, etc.
        CUDA_CHECK(cudaMemcpyAsync(target, source, size * get_dtype_size(t_type), cudaMemcpyDefault));
    } else {
        throw std::runtime_error(
            fmt::format("Unsupported conversion: {} -> {}", dtype_to_str(s_type), dtype_to_str(t_type)));
    }
}

cuFileRef::cuFileRef(std::string file_name)
    : cuFileRef(open_cufile(std::move(file_name))) {
}

/**
 * @brief Read a byte range from the file and convert tensor elements on-device.
 *
 * Staged through @p d_buffer in chunks of @p buffer_size; each chunk is fetched via
 * read_bytes(), so this works unchanged on both the GDS and POSIX paths.
 *
 * @param target Destination tensor buffer (device pointer) in dtype @p t_type.
 * @param begin Start offset in the file (inclusive), in bytes.
 * @param end End offset in the file (exclusive), in bytes.
 * @param file_name File name used for diagnostics only.
 * @param t_type Target tensor element dtype.
 * @param s_type Source tensor element dtype (as stored in the file).
 * @param d_buffer Temporary device buffer used for staged reads.
 * @param buffer_size Size of @p d_buffer in bytes; also the maximum chunk size per iteration.
 * @throws std::logic_error / std::runtime_error as propagated from reads and conversion.
 */
void cuFileRef::read_and_convert(std::byte* target,
                                 std::ptrdiff_t begin,
                                 std::ptrdiff_t end,
                                 std::string_view file_name,
                                 ETensorDType t_type,
                                 ETensorDType s_type,
                                 std::byte* d_buffer,
                                 std::size_t buffer_size) {
    (void)file_name;  // diagnostics come from mFileName via read_bytes()
    for (std::ptrdiff_t p = 0; p < end - begin; p += static_cast<std::ptrdiff_t>(buffer_size)) {
        std::ptrdiff_t amount = std::min(end - begin - p, static_cast<std::ptrdiff_t>(buffer_size));
        read_bytes(d_buffer, begin + p, begin + p + amount);
        convert_tensor_dispatch(target + p * get_dtype_size(t_type) / get_dtype_size(s_type),
                                d_buffer,
                                amount / get_dtype_size(s_type),
                                t_type,
                                s_type);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
