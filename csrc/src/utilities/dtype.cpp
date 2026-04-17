// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "dtype.h"

#include <fmt/core.h>

#include "utilities/utils.h"

template <typename... Others>
bool iequals_any(std::string_view lhs, Others&&... others) {
    return (iequals(lhs, others) || ...);
}

ETensorDType dtype_from_str(std::string_view dtype) {
    if (iequals_any(dtype, "F32", "FP32", "float", "float32", "single")) {
        return ETensorDType::FP32;
    } else if (iequals_any(dtype, "FP16", "F16", "float16", "half")) {
        return ETensorDType::FP16;
    } else if (iequals_any(dtype, "BF16", "bfloat16")) {
        return ETensorDType::BF16;
    } else if (iequals_any(dtype, "Int32", "i32")) {
        return ETensorDType::INT32;
    } else if (iequals_any(dtype, "Int8", "i8")) {
        return ETensorDType::INT8;
    } else if (iequals_any(dtype, "E4M3", "fp8_e4m3", "F8_E4M3")) {
        return ETensorDType::FP8_E4M3;
    } else if (iequals_any(dtype, "E5M2", "fp8_e5m2", "F8_E5M2")) {
        return ETensorDType::FP8_E5M2;
    } else if (iequals_any(dtype, "E2M1", "fp4_e2m1", "F4_E2M1", "FP4", "F4", "NVFP4")) {
        return ETensorDType::FP4_E2M1;
    } else if (iequals_any(dtype, "byte", "U8", "u8", "uint8")) {
        return ETensorDType::BYTE;
    } else if (iequals_any(dtype, "FP8", "F8")) {
        throw std::runtime_error("Invalid dtype FP8: Please specify E4M3 or E5M2");
    }
    throw std::runtime_error(fmt::format("Invalid dtype: '{}'", dtype));
}

const char* dtype_to_str(ETensorDType dtype) {
    switch (dtype) {
        case ETensorDType::FP32: return "F32";
        case ETensorDType::FP16: return "F16";
        case ETensorDType::BF16: return "BF16";
        case ETensorDType::INT32: return "I32";
        case ETensorDType::INT8: return "I8";
        case ETensorDType::FP8_E4M3: return "F8_E4M3";
        case ETensorDType::FP8_E5M2: return "F8_E5M2";
        case ETensorDType::FP4_E2M1: return "F4_E2M1";
        case ETensorDType::BYTE: return "U8";
        default: throw std::logic_error("Invalid dtype");
    }
}

const char* dtype_to_torch_str(ETensorDType dtype) {
    switch (dtype) {
        case ETensorDType::FP32: return "float32";
        case ETensorDType::FP16: return "float16";
        case ETensorDType::BF16: return "bfloat16";
        case ETensorDType::INT32: return "int32";
        case ETensorDType::INT8: return "int8";
        case ETensorDType::FP8_E4M3: return "float8_e4m3fn";
        case ETensorDType::FP8_E5M2: return "float8_e5m2";
        case ETensorDType::FP4_E2M1: return "float4_e2m1";  // Note: PyTorch doesn't have native FP4, use custom name
        case ETensorDType::BYTE: return "uint8";
        default: throw std::logic_error("Invalid dtype");
    }
}
