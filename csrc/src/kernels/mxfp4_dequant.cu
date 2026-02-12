// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file mxfp4_dequant.cu
 * @brief CUDA kernel for MXFP4 (Microscaling FP4) dequantization.
 *
 * Dequantizes packed FP4 E2M1 data with E8M0 shared exponents (one per
 * 32-element block) to BF16. This is the microscaling (MX) format used
 * by OpenAI GPT-OSS and other pre-quantized HuggingFace models.
 *
 * Data layout:
 * - Packed FP4: Two 4-bit E2M1 values per byte (lower nibble = even index,
 *   upper nibble = odd index). Total bytes = M*K/2.
 * - E8M0 scales: One uint8 exponent per 32 elements. The scale value is
 *   2^(exponent - 127) (bias-127 unsigned exponent, no mantissa).
 *   Total bytes = M*K/32.
 *
 * Dequantization formula:
 *   value[i] = fp4_decode(nibble[i]) * 2^(e8m0[i/32] - 127)
 *
 * FP4 E2M1 decode table (indexed by 4-bit value 0..15):
 *   {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,    (positive: 0..7)
 *   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0}  (negative: 8..15)
 *
 * Reference: Qutlass project (study/qutlass/tests/mxfp4_test.py::_dq_fp4)
 */

#include "kernel_utils.cuh"
#include "utilities/tensor.h"
#include "utilities/utils.h"

#include <cuda_bf16.h>

// ============================================================================
// Device Constants
// ============================================================================

/// FP4 E2M1 decode lookup table.
/// Index 0..7 = positive values, 8..15 = negative values (sign bit = bit 3).
__constant__ float kFP4E2M1DecodeLUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,   // 0..7
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f    // 8..15
};

// ============================================================================
// Kernels
// ============================================================================

/**
 * @brief MXFP4 dequantization kernel (scalar, grid-stride loop).
 *
 * Each thread processes one packed byte (= 2 FP4 values). The E8M0 exponent
 * for each 32-element block is loaded once and applied to both nibbles.
 *
 * @param[out] out Output BF16 array, shape (M*K).
 * @param[in] in_fp4 Input packed FP4 data, shape (M*K/2 bytes).
 *                   Lower nibble = element 2*i, upper nibble = element 2*i+1.
 * @param[in] e8m0_scales Input E8M0 exponents, shape (M*K/32).
 *                        Scale for block b = 2^(e8m0_scales[b] - 127).
 * @param total_elements Total number of output BF16 elements (M*K).
 */
__global__ void dequantize_mxfp4_kernel(
    nv_bfloat16* __restrict__ out,
    const uint8_t* __restrict__ in_fp4,
    const uint8_t* __restrict__ e8m0_scales,
    long total_elements)
{
    // Each thread processes one packed byte = 2 FP4 values
    const long total_bytes = total_elements / 2;

    for (long byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
         byte_idx < total_bytes;
         byte_idx += gridDim.x * (long)blockDim.x) {

        // Unpack two 4-bit values from one byte
        const uint8_t packed = in_fp4[byte_idx];
        const int lo_nibble = packed & 0xF;        // Even element (2*byte_idx)
        const int hi_nibble = (packed >> 4) & 0xF;  // Odd element (2*byte_idx + 1)

        // Decode FP4 E2M1 to float via LUT
        const float fp4_lo = kFP4E2M1DecodeLUT[lo_nibble];
        const float fp4_hi = kFP4E2M1DecodeLUT[hi_nibble];

        // Compute E8M0 scale: 2^(exponent - 127)
        // Both elements in the same byte belong to consecutive positions,
        // so they share the same 32-element block if they're in the same block.
        // Element indices: lo = 2*byte_idx, hi = 2*byte_idx + 1
        // Block index for element i = i / 32
        const long elem_lo = 2 * byte_idx;
        const long elem_hi = elem_lo + 1;
        const long block_lo = elem_lo / 32;
        const long block_hi = elem_hi / 32;

        // Load E8M0 exponent and compute power-of-2 scale
        const float scale_lo = exp2f((float)e8m0_scales[block_lo] - 127.0f);

        // Write dequantized values
        out[elem_lo] = (nv_bfloat16)(fp4_lo * scale_lo);

        // hi element may be in the same or next block (only differs at block boundary)
        if (block_hi == block_lo) {
            out[elem_hi] = (nv_bfloat16)(fp4_hi * scale_lo);
        } else {
            const float scale_hi = exp2f((float)e8m0_scales[block_hi] - 127.0f);
            out[elem_hi] = (nv_bfloat16)(fp4_hi * scale_hi);
        }
    }
}

/**
 * @brief Vectorized MXFP4 dequantization kernel.
 *
 * Processes 4 packed bytes (= 8 FP4 values) per thread for better memory
 * bandwidth utilization. Uses uint32_t loads for the packed FP4 data.
 *
 * @param[out] out Output BF16 array, shape (M*K).
 * @param[in] in_fp4 Input packed FP4 data, shape (M*K/2 bytes).
 * @param[in] e8m0_scales Input E8M0 exponents, shape (M*K/32).
 * @param total_elements Total number of output BF16 elements (M*K).
 */
__global__ void dequantize_mxfp4_vec_kernel(
    nv_bfloat16* __restrict__ out,
    const uint8_t* __restrict__ in_fp4,
    const uint8_t* __restrict__ e8m0_scales,
    long total_elements)
{
    // Process 4 bytes (8 elements) per thread
    constexpr int BYTES_PER_THREAD = 4;
    const long total_bytes = total_elements / 2;
    const long vec_bytes = total_bytes / BYTES_PER_THREAD * BYTES_PER_THREAD;

    // Vectorized path: process 4 bytes at a time
    for (long base_byte = (blockIdx.x * blockDim.x + threadIdx.x) * BYTES_PER_THREAD;
         base_byte < vec_bytes;
         base_byte += gridDim.x * (long)blockDim.x * BYTES_PER_THREAD) {

        // Load 4 bytes of packed FP4 data as uint32
        const uint32_t packed4 = *reinterpret_cast<const uint32_t*>(in_fp4 + base_byte);

        // First element index for this chunk
        const long base_elem = 2 * base_byte;

        // Precompute block index for first element and load its scale
        const long first_block = base_elem / 32;
        float current_scale = exp2f((float)e8m0_scales[first_block] - 127.0f);
        long current_block = first_block;

        #pragma unroll
        for (int b = 0; b < BYTES_PER_THREAD; ++b) {
            const uint8_t packed = (packed4 >> (b * 8)) & 0xFF;
            const int lo_nibble = packed & 0xF;
            const int hi_nibble = (packed >> 4) & 0xF;

            const float fp4_lo = kFP4E2M1DecodeLUT[lo_nibble];
            const float fp4_hi = kFP4E2M1DecodeLUT[hi_nibble];

            const long elem_lo = base_elem + b * 2;
            const long elem_hi = elem_lo + 1;

            // Check if we crossed a 32-element block boundary
            const long block_lo = elem_lo / 32;
            if (block_lo != current_block) {
                current_scale = exp2f((float)e8m0_scales[block_lo] - 127.0f);
                current_block = block_lo;
            }

            out[elem_lo] = (nv_bfloat16)(fp4_lo * current_scale);

            const long block_hi = elem_hi / 32;
            if (block_hi != current_block) {
                current_scale = exp2f((float)e8m0_scales[block_hi] - 127.0f);
                current_block = block_hi;
            }

            out[elem_hi] = (nv_bfloat16)(fp4_hi * current_scale);
        }
    }

    // Scalar tail: handle remaining bytes not covered by vectorized loop
    for (long byte_idx = vec_bytes + blockIdx.x * blockDim.x + threadIdx.x;
         byte_idx < total_bytes;
         byte_idx += gridDim.x * (long)blockDim.x) {

        const uint8_t packed = in_fp4[byte_idx];
        const int lo_nibble = packed & 0xF;
        const int hi_nibble = (packed >> 4) & 0xF;

        const float fp4_lo = kFP4E2M1DecodeLUT[lo_nibble];
        const float fp4_hi = kFP4E2M1DecodeLUT[hi_nibble];

        const long elem_lo = 2 * byte_idx;
        const long elem_hi = elem_lo + 1;
        const long block_lo = elem_lo / 32;
        const long block_hi = elem_hi / 32;

        const float scale_lo = exp2f((float)e8m0_scales[block_lo] - 127.0f);
        out[elem_lo] = (nv_bfloat16)(fp4_lo * scale_lo);

        if (block_hi == block_lo) {
            out[elem_hi] = (nv_bfloat16)(fp4_hi * scale_lo);
        } else {
            const float scale_hi = exp2f((float)e8m0_scales[block_hi] - 127.0f);
            out[elem_hi] = (nv_bfloat16)(fp4_hi * scale_hi);
        }
    }
}

// ============================================================================
// Host Launcher Functions
// ============================================================================

/**
 * @brief Host launcher for MXFP4 dequantization.
 *
 * Dequantizes packed FP4 E2M1 data with E8M0 shared exponents to BF16.
 *
 * @param[out] out Output BF16 array (M, K).
 * @param[in] in_fp4 Input packed FP4 data (M*K/2 bytes).
 * @param[in] e8m0_scales Input E8M0 exponents (M*K/32).
 * @param M Number of rows.
 * @param K Number of columns (must be even).
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void dequantize_mxfp4(
    nv_bfloat16* out,
    const uint8_t* in_fp4,
    const uint8_t* e8m0_scales,
    int M, int K,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const long total_elements = (long)M * K;

    if (total_elements == 0) return;

    if (K % 2 != 0) {
        throw std::runtime_error("dequantize_mxfp4: K must be even (FP4 packs 2 values per byte)");
    }

    const int threads_per_block = 256;
    // Each thread in the vec kernel processes 4 bytes = 8 elements
    const long total_bytes = total_elements / 2;
    const int num_blocks = std::min(
        (long)std::max(2 * dp.multiProcessorCount, 1),
        (total_bytes + threads_per_block * 4 - 1) / (threads_per_block * 4));

    dequantize_mxfp4_vec_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        out, in_fp4, e8m0_scales, total_elements);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Tensor-based wrapper for MXFP4 dequantization.
 *
 * @param[out] out Output BF16 tensor (must be pre-allocated with shape [M, K]).
 * @param[in] in_fp4 Input packed FP4 tensor (uint8, shape [M*K/2]).
 * @param[in] e8m0_scales Input E8M0 scale tensor (uint8, shape [M*K/32]).
 * @param M Number of rows in the weight matrix.
 * @param K Number of columns in the weight matrix.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void dequantize_mxfp4(
    Tensor& out,
    const Tensor& in_fp4,
    const Tensor& e8m0_scales,
    int M, int K,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (out.DType != ETensorDType::BF16) {
        throw std::runtime_error("dequantize_mxfp4: output must be BF16");
    }
    if (in_fp4.DType != ETensorDType::BYTE) {
        throw std::runtime_error("dequantize_mxfp4: input FP4 data must be BYTE (packed uint8)");
    }
    if (e8m0_scales.DType != ETensorDType::BYTE) {
        throw std::runtime_error("dequantize_mxfp4: E8M0 scales must be BYTE (uint8 exponents)");
    }

    dequantize_mxfp4(
        out.get<nv_bfloat16>(),
        in_fp4.get<uint8_t>(),
        e8m0_scales.get<uint8_t>(),
        M, K, dp, stream);
}

// ============================================================================
// Batched 2D BF16 Transpose
// ============================================================================

/**
 * @brief Batched 2D matrix transpose kernel for BF16.
 *
 * Uses shared-memory tiling for coalesced reads and writes.
 * Each thread block handles one TILE_DIM x TILE_DIM tile.
 *
 * @param[out] dst Output BF16 array, shape [batches, cols, rows].
 * @param[in]  src Input BF16 array, shape [batches, rows, cols].
 * @param batches Number of independent matrices.
 * @param rows    Rows per matrix before transpose.
 * @param cols    Columns per matrix before transpose.
 */
__global__ void batched_transpose_2d_bf16_kernel(
    nv_bfloat16* __restrict__ dst,
    const nv_bfloat16* __restrict__ src,
    int batches, int rows, int cols)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    __shared__ nv_bfloat16 tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    const long per_matrix = (long)rows * cols;

    // Grid: blockIdx.x covers column tiles, blockIdx.y covers row tiles,
    // blockIdx.z covers batches.
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;
    const int batch = blockIdx.z;
    if (batch >= batches) return;

    const nv_bfloat16* src_mat = src + batch * per_matrix;
    nv_bfloat16* dst_mat = dst + batch * per_matrix;

    // Load tile from src[by..by+TILE_DIM, bx..bx+TILE_DIM] into shared memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int src_row = by + threadIdx.y + j;
        int src_col = bx + threadIdx.x;
        if (src_row < rows && src_col < cols) {
            tile[threadIdx.y + j][threadIdx.x] = src_mat[(long)src_row * cols + src_col];
        }
    }

    __syncthreads();

    // Write transposed tile to dst[bx..bx+TILE_DIM, by..by+TILE_DIM]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int dst_row = bx + threadIdx.y + j;
        int dst_col = by + threadIdx.x;
        if (dst_row < cols && dst_col < rows) {
            dst_mat[(long)dst_row * rows + dst_col] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void batched_transpose_2d_bf16(
    nv_bfloat16* dst, const nv_bfloat16* src,
    int batches, int rows, int cols,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (batches == 0 || rows == 0 || cols == 0) return;

    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 blocks(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM,
        batches);

    batched_transpose_2d_bf16_kernel<<<blocks, threads, 0, stream>>>(
        dst, src, batches, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}
