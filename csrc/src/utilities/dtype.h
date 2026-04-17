// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_UTILITIES_DTYPE_H
#define SUROGATE_SRC_UTILITIES_DTYPE_H

#include <cstdint>
#include <stdexcept>
#include <string_view>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

enum class ETensorDType : int {
    FP32,
    BF16,
    FP16,
    INT32,
    INT8,
    FP8_E4M3,
    FP8_E5M2,
    FP4_E2M1,  // NVFP4: 4-bit E2M1 format, packed 2 values per byte
    BYTE,      // use for generic buffers
};

template <class T>
consteval ETensorDType dtype_from_pointer(const T*) {
    if constexpr (std::is_same_v<T, float>) {
        return ETensorDType::FP32;
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
        return ETensorDType::BF16;
    } else if constexpr (std::is_same_v<T, half>) {
        return ETensorDType::FP16;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return ETensorDType::INT32;
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
        return ETensorDType::INT8;
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        return ETensorDType::FP8_E4M3;
    } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
        return ETensorDType::FP8_E5M2;
    } else if constexpr (std::is_same_v<T, std::byte>) {
        return ETensorDType::BYTE;
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        return ETensorDType::BYTE;  // uint8_t used for packed FP4 data
    }
    throw std::runtime_error("Invalid dtype");
}

template <typename T>
constexpr ETensorDType dtype_from_type = dtype_from_pointer((T*)nullptr);

ETensorDType dtype_from_str(std::string_view dtype);
const char* dtype_to_str(ETensorDType dtype);
const char* dtype_to_torch_str(ETensorDType dtype);

constexpr int get_dtype_size(const ETensorDType type) {
    switch (type) {
        case ETensorDType::FP32:
        case ETensorDType::INT32: return 4;
        case ETensorDType::BF16:
        case ETensorDType::FP16: return 2;
        case ETensorDType::INT8:
        case ETensorDType::FP8_E4M3:
        case ETensorDType::FP8_E5M2:
        case ETensorDType::BYTE:
        case ETensorDType::FP4_E2M1:  // Packed: 2 values per byte, report as 1 for storage
            return 1;
    }
    throw std::logic_error("Invalid dtype");
}

/// @brief Get the bit width of a dtype (useful for sub-byte types like FP4)
constexpr int get_dtype_bits(const ETensorDType type) {
    switch (type) {
        case ETensorDType::FP32:
        case ETensorDType::INT32: return 32;
        case ETensorDType::BF16:
        case ETensorDType::FP16: return 16;
        case ETensorDType::INT8:
        case ETensorDType::FP8_E4M3:
        case ETensorDType::FP8_E5M2:
        case ETensorDType::BYTE: return 8;
        case ETensorDType::FP4_E2M1: return 4;
    }
    throw std::logic_error("Invalid dtype");
}

constexpr bool is_fp8_dtype(ETensorDType dt) {
    return dt == ETensorDType::FP8_E4M3 || dt == ETensorDType::FP8_E5M2;
}

constexpr bool is_fp4_dtype(ETensorDType dt) {
    return dt == ETensorDType::FP4_E2M1;
}

/// @brief Check if dtype is a sub-byte quantized type (FP4 or FP8)
constexpr bool is_quantized_dtype(ETensorDType dt) {
    return is_fp8_dtype(dt) || is_fp4_dtype(dt);
}

#endif  //SUROGATE_SRC_UTILITIES_DTYPE_H
