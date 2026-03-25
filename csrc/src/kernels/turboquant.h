// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file turboquant.h
 * @brief TurboQuant 3.5-bit KV cache quantization (arXiv 2504.19874).
 *
 * Data-oblivious vector quantization achieving near-optimal inner product preservation
 * at 3.5 bits/element. Uses randomized Walsh-Hadamard rotation + mixed 2/3-bit
 * Lloyd-Max scalar quantization + 1-bit QJL residual correction.
 *
 * Packed format per vector (head_dim=128, 60 bytes):
 *   [0..23]   3-bit MSE indices for first 64 elements  (24 bytes)
 *   [24..39]  2-bit MSE indices for last 64 elements   (16 bytes)
 *   [40..55]  QJL sign bits for all 128 elements        (16 bytes)
 *   [56..57]  Residual norm (float16)
 *   [58..59]  Vector norm (float16)
 *
 * Bits/element: 3.5 + 32/head_dim (norm overhead, negligible for typical head_dim).
 * Compression vs FP16: ~4.3x for head_dim=128.
 */

#ifndef SUROGATE_SRC_KERNELS_TURBOQUANT_H
#define SUROGATE_SRC_KERNELS_TURBOQUANT_H

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace turboquant {

/// Packed size in bytes for a single vector of given head_dim.
/// Formula: 7*D/16 + 4 bytes (3.5 bits/elem + 32 bits norms).
constexpr int packed_size(int head_dim) {
    return 7 * head_dim / 16 + 4;
}

/// Quantize BF16 vectors to TurboQuant 3.5-bit packed format.
///
/// @param output   [num_vectors, packed_size(head_dim)] packed output (device)
/// @param input    [num_vectors, head_dim] BF16 input vectors (device)
/// @param signs_d1 [head_dim/32] random sign bitmask for main rotation (device)
/// @param signs_d2 [head_dim/32] random sign bitmask for QJL rotation (device)
/// @param num_vectors  Number of vectors to quantize
/// @param head_dim     Dimension per vector (must be 64, 128, or 256)
/// @param stream       CUDA stream
void quantize(
    uint8_t* output,
    const nv_bfloat16* input,
    const uint32_t* signs_d1,
    const uint32_t* signs_d2,
    int num_vectors,
    int head_dim,
    cudaStream_t stream);

/// Dequantize TurboQuant packed format back to BF16.
///
/// @param output   [num_vectors, head_dim] BF16 output vectors (device)
/// @param input    [num_vectors, packed_size(head_dim)] packed input (device)
/// @param signs_d1 [head_dim/32] random sign bitmask for main rotation (device)
/// @param signs_d2 [head_dim/32] random sign bitmask for QJL rotation (device)
/// @param num_vectors  Number of vectors to dequantize
/// @param head_dim     Dimension per vector (must be 64, 128, or 256)
/// @param stream       CUDA stream
void dequantize(
    nv_bfloat16* output,
    const uint8_t* input,
    const uint32_t* signs_d1,
    const uint32_t* signs_d2,
    int num_vectors,
    int head_dim,
    cudaStream_t stream);

/// Generate random sign bitmasks on the host, then caller copies to device.
/// Produces two bitmasks (d1, d2) each of head_dim/32 uint32s.
///
/// @param signs_d1 [head_dim/32] output (host)
/// @param signs_d2 [head_dim/32] output (host)
/// @param head_dim Dimension
/// @param seed     RNG seed (use layer_idx * num_heads + head_idx for uniqueness)
void generate_signs_host(
    uint32_t* signs_d1,
    uint32_t* signs_d2,
    int head_dim,
    uint64_t seed);

}  // namespace turboquant

#endif  // SUROGATE_SRC_KERNELS_TURBOQUANT_H
