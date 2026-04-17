// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file squirrel_noise.cuh
 * @brief Squirrel Noise random number generation for stochastic rounding.
 *
 * Provides fast, deterministic pseudo-random number generation suitable for
 * GPU-based stochastic rounding in reduced-precision training. Uses Squirrel
 * Noise 5 algorithm for high-quality noise from position + seed.
 *
 * Stochastic rounding randomly rounds to nearest representable value based on
 * the fractional distance, providing unbiased rounding that improves training
 * convergence with low-precision formats (BF16, FP8).
 */

#ifndef SUROGATE_SQUIRREL_NOISE_CUH
#define SUROGATE_SQUIRREL_NOISE_CUH

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ----------------------------------------------------------------------------
// Random Number Generation used in Stochastic Rounding

/**
 * @brief SquirrelNoise5 hash function for deterministic pseudo-random numbers.
 *
 * High-quality noise function that produces uniformly distributed 32-bit
 * unsigned integers from a position and seed. Uses bit manipulation and
 * prime-based mixing for good avalanche properties.
 *
 * Based on: http://eiserloh.net/noise/SquirrelNoise5.hpp
 *
 * @param positionX Position input (e.g., thread index).
 * @param seed Global seed value.
 * @return Pseudo-random 32-bit unsigned integer.
 */
__device__ __host__ constexpr unsigned int squirrel_noise_5(unsigned int positionX, unsigned int seed) {
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;  // 11010010101010000000101000111111
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;  // 10101000100001001111000110010111
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B;  // 01101100011100110110111101001011
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;  // 10110111100111110011101010111011
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;  // 00011011010101101100010011110101
    unsigned int mangledBits = positionX;
    mangledBits *= SQ5_BIT_NOISE1;
    mangledBits += seed;
    mangledBits ^= (mangledBits >> 9);
    mangledBits += SQ5_BIT_NOISE2;
    mangledBits ^= (mangledBits >> 11);
    mangledBits *= SQ5_BIT_NOISE3;
    mangledBits ^= (mangledBits >> 13);
    mangledBits += SQ5_BIT_NOISE4;
    mangledBits ^= (mangledBits >> 15);
    mangledBits *= SQ5_BIT_NOISE5;
    mangledBits ^= (mangledBits >> 17);
    return mangledBits;
}

/**
 * @brief Generates 2D noise from x/y coordinates and seed.
 *
 * Combines 2D indices into a single position using a large prime multiplier
 * to avoid correlation between nearby (x,y) pairs, then calls squirrel_noise_5.
 * Useful for generating unique random values per thread in a 2D grid.
 *
 * @param indexX X coordinate (e.g., threadIdx.x).
 * @param indexY Y coordinate (e.g., blockIdx.x * blockDim.x + blockIdx.y).
 * @param seed Global seed value.
 * @return Pseudo-random 32-bit unsigned integer.
 */
__device__ __host__ constexpr unsigned int get_noise_2d(int indexX, int indexY, unsigned int seed) {
    constexpr unsigned int PRIME_NUMBER = 198491317u;  // Large prime number with non-boring bits
    unsigned int x = static_cast<unsigned int>(indexX);
    unsigned int y = static_cast<unsigned int>(indexY);

    return squirrel_noise_5(x + (PRIME_NUMBER * y), seed);
}

/**
 * @brief Stochastic rounding from FP32 to BF16.
 *
 * Rounds to nearest BF16 with probability proportional to the distance to
 * each representable value. This provides unbiased rounding on average,
 * which helps training convergence with low-precision formats.
 *
 * Uses lower 16 bits of input as rounding bits, compared against random threshold.
 *
 * @param in Input FP32 value.
 * @param[out] out Output BF16 value.
 * @param seed Random seed (updated per step via xorshift).
 * @param noise If true, generates per-thread noise; if false, uses seed directly.
 */
__device__ __forceinline__ void stochastic_rounding(float in, nv_bfloat16* out, unsigned int seed, bool noise = true) {
    // todo - is this stochastic rounding *too good*? can we cut any corners?
    // makes sure each thread gets a different random number
    unsigned int random = noise ? get_noise_2d(threadIdx.x, blockIdx.x * blockDim.x + blockIdx.y, seed) : seed;
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = __float_as_uint(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits & ~0xFFFF);
    *out = __float2bfloat16_rn(__uint_as_float(float_bits));
}

/**
 * @brief Stochastic rounding from FP32 to FP8 E4M3.
 *
 * Rounds to nearest FP8 E4M3 with probability proportional to the distance.
 * Handles special cases:
 * - Values > 448 (FP8 max): saturates to max representable
 * - Values < 0.015625 (near subnormal): uses standard rounding
 * - Otherwise: applies stochastic rounding using lower 20 mantissa bits
 *
 * @param in Input FP32 value.
 * @param[out] out Output FP8 E4M3 value.
 * @param seed Random seed (updated per step via xorshift).
 * @param noise If true, generates per-thread noise; if false, uses seed directly.
 */
__device__ __forceinline__ void
stochastic_rounding(float in, __nv_fp8_e4m3* out, unsigned int seed, bool noise = true) {
    if (fabs(in) > 448.f) {
        out->__x = __nv_cvt_float_to_fp8(in, __NV_SATFINITE, __NV_E4M3);
    } else if (fabs(in) < 0.015625) {
        // TODO align SR to mantissa
        out->__x = __nv_cvt_float_to_fp8(in, __NV_SATFINITE, __NV_E4M3);
    } else {
        // makes sure each thread gets a different random number
        unsigned int random = noise ? get_noise_2d(threadIdx.x, blockIdx.x * blockDim.x + blockIdx.y, seed) : seed;
        unsigned int threshold = random & 0xFFFFF;
        unsigned int float_bits = __float_as_uint(in);
        unsigned int rounded_bits = float_bits & 0x000FFFFF;
        float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFFF) : (float_bits & ~0xFFFFF);
        out->__x = __nv_cvt_float_to_fp8(__uint_as_float(float_bits), __NV_SATFINITE, __NV_E4M3);
    }
}

/**
 * @brief Identity "stochastic rounding" for FP32 (no-op).
 *
 * Dummy overload for FP32 mode where no rounding is needed.
 * Simply copies the input to output unchanged.
 *
 * @param in Input FP32 value.
 * @param[out] out Output FP32 value (same as input).
 * @param seed Unused (kept for API consistency).
 * @param noise Unused (kept for API consistency).
 */
__device__ __forceinline__ void stochastic_rounding(float in, float* out, unsigned int seed, bool noise = true) {
    *out = in;
}

#endif  //SUROGATE_SQUIRREL_NOISE_CUH
