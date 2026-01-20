// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_UTILS_UTILS_H
#define SUROGATE_SRC_UTILS_UTILS_H

#include <concepts>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <driver_types.h>
#include <library_types.h>

#ifndef __CUDACC__
#define HOST_DEVICE
#else
#define HOST_DEVICE __host__ __device__
#endif

/// This exception will be thrown for reported cuda errors
class cuda_error : public std::runtime_error {
public:
    cuda_error(cudaError_t err, const std::string& arg) :
            std::runtime_error(arg), code(err){};

    cudaError_t code;
};

/// Check `status`; if it isn't `cudaSuccess`, throw the corresponding `cuda_error`
void cuda_throw_on_error(cudaError_t status, const char* statement, const char* file, int line);

#define CUDA_CHECK(status) cuda_throw_on_error(status, #status, __FILE__, __LINE__)

/// Check cuBLAS status; throws on error with stack trace
void cublas_throw_on_error(cublasStatus_t status, const char* statement, const char* file, int line);

#define CUBLAS_CHECK(status) cublas_throw_on_error(status, #status, __FILE__, __LINE__)

template<std::integral T>
constexpr T HOST_DEVICE div_ceil(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

[[noreturn]] void throw_not_divisible(long long dividend, long long divisor);

template<std::integral T>
constexpr T div_exact(T dividend, T divisor) {
    if(dividend % divisor != 0) {
        throw_not_divisible(dividend, divisor);
    }
    return dividend / divisor;
}

template<std::integral Dst, std::integral Src>
constexpr Dst narrow(Src input) {
    if constexpr (std::is_signed_v<Src>) {
        if (std::is_unsigned_v<Dst> && input < 0) {
            throw std::out_of_range("Cannot convert negative number to unsigned");
        }
        if (std::is_signed_v<Dst> && input < std::numeric_limits<Dst>::min())
        {
            throw std::out_of_range("Out of range in integer conversion: underflow");
        }
    }

    if (input > std::numeric_limits<Dst>::max())
    {
        throw std::out_of_range("Out of range in integer conversion: overflow");
    }

    return static_cast<Dst>(input);
}

// ----------------------------------------------------------------------------
template<typename Scalar>
inline cudaDataType to_cuda_lib_type_enum;

template<> inline constexpr cudaDataType to_cuda_lib_type_enum<float> = cudaDataType::CUDA_R_32F;
template<> inline constexpr cudaDataType to_cuda_lib_type_enum<nv_bfloat16> = cudaDataType::CUDA_R_16BF;
template<> inline constexpr cudaDataType to_cuda_lib_type_enum<std::int8_t> = cudaDataType::CUDA_R_8I;
template<> inline constexpr cudaDataType to_cuda_lib_type_enum<__nv_fp8_e4m3> = cudaDataType::CUDA_R_8F_E4M3;
template<> inline constexpr cudaDataType to_cuda_lib_type_enum<__nv_fp8_e5m2> = cudaDataType::CUDA_R_8F_E5M2;

template<typename FP8Type>
inline __nv_fp8_interpretation_t fp8_interpretation_v;
template<> inline constexpr __nv_fp8_interpretation_t fp8_interpretation_v<__nv_fp8_e4m3> = __nv_fp8_interpretation_t::__NV_E4M3;
template<> inline constexpr __nv_fp8_interpretation_t fp8_interpretation_v<__nv_fp8_e5m2> = __nv_fp8_interpretation_t::__NV_E5M2;

/// @brief Maximum representable value for FP8 types (for quantization scaling).
/// E4M3: 4 exponent bits, 3 mantissa bits, max ~448
/// E5M2: 5 exponent bits, 2 mantissa bits, max ~57344 (larger dynamic range, less precision)
template<typename FP8Type>
inline constexpr float fp8_max_v = 0.f;
template<> inline constexpr float fp8_max_v<__nv_fp8_e4m3> = 448.f;
template<> inline constexpr float fp8_max_v<__nv_fp8_e5m2> = 57344.f;

// ----------------------------------------------------------------------------
// FP4 (NVFP4/E2M1) constants and types
// E2M1: 2 exponent bits, 1 mantissa bit
// Representable values: +/-{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

/// @brief Maximum representable value for FP4 E2M1 format
inline constexpr float FP4_E2M1_MAX = 6.0f;

/// @brief Block size for NVFP4 quantization (scale per 16 consecutive values)
inline constexpr int NVFP4_BLOCK_SIZE = 16;

/// @brief Tile size for NVFP4 quantization kernels (128x128 tiles)
inline constexpr int NVFP4_TILE_SIZE = 128;

/// @brief Type alias for packed FP4 data (2 values per byte)
/// The low nibble contains the first value, high nibble contains the second
using fp4_packed_t = std::uint8_t;

/// @brief Calculate packed FP4 storage size in bytes for N elements
constexpr std::size_t fp4_packed_size(std::size_t num_elements) {
    return (num_elements + 1) / 2;  // 2 elements per byte, round up
}

/// @brief Calculate FP4 block scale tensor shape
/// @param M Number of rows
/// @param K Number of columns
/// @return {scale_rows, scale_cols} where scale_rows is aligned to 128, scale_cols to 4
constexpr std::pair<long, long> fp4_scale_shape(long M, long K) {
    // F8_128x4 reordering expects the scale tensor to be sized for the padded (rounded-up)
    // dimensions: rows rounded to 128, cols (K/16) rounded to 4.
    long scale_rows = div_ceil(M, static_cast<long>(NVFP4_TILE_SIZE)) * static_cast<long>(NVFP4_TILE_SIZE);
    long scale_cols = div_ceil(div_ceil(K, static_cast<long>(NVFP4_BLOCK_SIZE)), 4L) * 4;
    return {scale_rows, scale_cols};
}

// ----------------------------------------------------------------------------
// NVTX utils

class NvtxRange {
public:
    explicit NvtxRange(const char* s) noexcept;
    NvtxRange(const std::string& base_str, int number);
    ~NvtxRange() noexcept;
};
#define NVTX_RANGE_FN() NvtxRange nvtx_range_##__COUNTER__ (__FUNCTION__)

cudaStream_t create_named_stream(const char* name);
cudaEvent_t create_named_event(const char* name, bool timing=false);


// ----------------------------------------------------------------------------
bool iequals(std::string_view lhs, std::string_view rhs);

/**
 * @brief Displays a simple progress bar on stderr.
 *
 * @param current Current item index (0-based).
 * @param total Total number of items.
 * @param label Prefix label for the progress bar.
 */
void show_progress_bar(int current, int total, const std::string& label = "Quantizing");

#endif //SUROGATE_SRC_UTILS_UTILS_H
