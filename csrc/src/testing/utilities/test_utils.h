// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>
#include <cstddef>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_bf16.h>
#include <iterator>

namespace testing_utils {

// BF16 helper: round-to-nearest-even conversion (emulated on CPU)
inline uint16_t float_to_bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // round to nearest even on the cut at 16 LSBs
    uint32_t lsb = (u >> 16) & 1u;                // last bit that will remain
    uint32_t rounding_bias = 0x7FFFu + lsb;       // RN-even
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

inline float bf16_bits_to_float(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline nv_bfloat16 make_nvbf16_from_float(float f) {
    uint16_t b = float_to_bf16_bits(f);
    nv_bfloat16 v{};
    std::memcpy(&v, &b, sizeof(b));
    return v;
}

// -----------------------------------------------------------------------------

template<typename T>
inline thrust::device_vector<T> to_device(const std::vector<T>& h_vec) {
    thrust::device_vector<T> d_vec(h_vec.size());
    thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());
    return d_vec;
}

template<typename T>
inline std::vector<T> from_device(const thrust::device_vector<T>& d_vec) {
    std::vector<T> h_vec(d_vec.size());
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    return h_vec;
}

inline std::vector<nv_bfloat16> to_bf16(const std::vector<float>& vec) {
    std::vector<nv_bfloat16> result(vec.size());
    for(size_t i = 0; i < vec.size(); ++i) {
        result[i] = make_nvbf16_from_float(vec[i]);
    }
    return result;
}

inline std::vector<float> round_bf16(const std::vector<float>& vec) {
    std::vector<float> result(vec.size());
    for(size_t i = 0; i < vec.size(); ++i) {
        result[i] = bf16_bits_to_float(float_to_bf16_bits(vec[i]));
    }
    return result;
}

// -----------------------------------------------------------------------------

inline std::vector<float> uniform_host(long n, float low, float high, uint64_t seed = 12345ULL) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> data;
    data.reserve(n);
    std::generate_n(std::back_inserter(data), n, [&]() { return dist(gen); });
    return data;
}

inline void fill_normal(std::vector<float>& v, float mean = 0.0f, float stddev = 1.0f, uint64_t seed = 12345ULL) {
    std::mt19937_64 gen(seed);
    std::normal_distribution<float> dist(mean, stddev);
    for (auto& x : v) x = dist(gen);
}

} // namespace testing_utils
