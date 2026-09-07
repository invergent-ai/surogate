#pragma once

#include "core/arena.h"
#include "ops/linear/fp8/fp8_cublaslt.h"
#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace sinfer::ops::detail {

struct Fp8A8Workspace {
    std::uint8_t* codes = nullptr;
    float* scales       = nullptr;
    float* staging      = nullptr; // fp32 [rows x tokens] for the cuBLASLt route, else null
};

inline std::size_t fp8_a8_checked_bytes(std::int32_t tokens, std::size_t bytes_per_token) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 A8 workspace: T must be positive"); }
    const auto count = static_cast<std::size_t>(tokens);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_token) {
        throw std::overflow_error("fp8 A8 workspace size overflow");
    }
    return count * bytes_per_token;
}

// staging_rows > 0 reserves the cuBLASLt route's fp32 staging for that many output rows.
template <class Arena>
Fp8A8Workspace allocate_fp8_a8_workspace(Arena& arena, std::int32_t tokens,
                                         std::int32_t input_rows, std::int32_t staging_rows = 0) {
    if (input_rows <= 0 || (input_rows % 32) != 0) {
        throw std::invalid_argument("fp8 A8 workspace: invalid K");
    }
    const DeviceSpan codes =
        arena.alloc_bytes(fp8_a8_checked_bytes(tokens, static_cast<std::size_t>(input_rows)), 256);
    const DeviceSpan scales = arena.alloc_bytes(fp8_a8_checked_bytes(tokens, sizeof(float)), 256);
    float* staging          = nullptr;
    if (staging_rows > 0) {
        const DeviceSpan span =
            arena.alloc_bytes(fp8_cublaslt_staging_bytes(staging_rows, tokens), 256);
        staging = static_cast<float*>(span.data);
    }
    return {static_cast<std::uint8_t*>(codes.data), static_cast<float*>(scales.data), staging};
}

inline std::size_t fp8_a8_workspace_capacity_bytes(std::int32_t tokens, std::int32_t input_rows,
                                                   std::int32_t staging_rows = 0) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fp8_a8_workspace(layout, tokens, input_rows, staging_rows);
    return layout.peak_bytes(1);
}

void launch_fp8_a8_quantize(const Tensor& x, const Weight& weight, Fp8A8Workspace workspace,
                            cudaStream_t stream);

void launch_fp8_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace workspace,
                   cudaStream_t stream);

} // namespace sinfer::ops::detail
