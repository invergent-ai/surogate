// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// BitsAndBytes-style NF4 quantization kernels for QLoRA.
// Based on bitsandbytes (https://github.com/bitsandbytes-foundation/bitsandbytes)

/**
 * @file bnb_quant.cu
 * @brief CUDA kernels for BitsAndBytes-style NF4 blockwise quantization/dequantization.
 *
 * Provides GPU-accelerated blockwise NF4 quantization for memory-efficient QLoRA training:
 * - NF4 (Normal Float 4-bit) quantization with per-block absmax scaling
 * - Double quantization support (quantize absmax values to INT8)
 * - Works on any CUDA GPU (no SM89+ or SM100+ requirement)
 *
 * NF4 uses 16 asymmetric bins derived from a standard normal distribution N(0,1),
 * which better represents the weight distribution of neural networks compared to
 * uniform or FP4 quantization.
 */

#include "kernel_utils.cuh"
#include "utilities/tensor.h"
#include "utilities/utils.h"

#include <cuda_bf16.h>
#include <cub/cub.cuh>

// ============================================================================
// NF4 Lookup Tables
// ============================================================================

/**
 * @brief NF4 dequantization lookup table.
 *
 * 16 values derived from the normal distribution N(0,1) where each bin
 * has equal probability mass. Values are normalized to [-1, 1] range.
 * This distribution better matches neural network weight distributions.
 */
__device__ __constant__ float d_nf4_dequant_lut[16] = {
    -1.0f,                 // 0b0000
    -0.6961928009986877f,  // 0b0001
    -0.5250730514526367f,  // 0b0010
    -0.39491748809814453f, // 0b0011
    -0.28444138169288635f, // 0b0100
    -0.18477343022823334f, // 0b0101
    -0.09105003625154495f, // 0b0110
    0.0f,                  // 0b0111
    0.07958029955625534f,  // 0b1000
    0.16093020141124725f,  // 0b1001
    0.24611230194568634f,  // 0b1010
    0.33791524171829224f,  // 0b1011
    0.44070982933044434f,  // 0b1100
    0.5626170039176941f,   // 0b1101
    0.7229568362236023f,   // 0b1110
    1.0f                   // 0b1111
};

// Host-side copy of NF4 lookup table for initialization
static const float h_nf4_dequant_lut[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

// ============================================================================
// NF4 Quantization/Dequantization Device Functions
// ============================================================================

/**
 * @brief Quantize a normalized value [-1, 1] to 4-bit NF4.
 *
 * Uses a binary decision tree for O(log n) quantization, matching the
 * bitsandbytes implementation exactly. Thresholds are midpoints between
 * adjacent NF4 codebook values.
 *
 * @param x Input value normalized to [-1, 1] range
 * @return 4-bit quantized value (0-15)
 */
__device__ __forceinline__ unsigned char dQuantizeNF4(float x) {
    // Binary decision tree for NF4 quantization
    // Thresholds are midpoints between adjacent codebook values
    if (x > 0.03979014977812767f)
        if (x > 0.3893125355243683f)         // 1
            if (x > 0.6427869200706482f)     // 11
                if (x > 0.8614784181118011f) // 111
                    return 0b1111;
                else
                    return 0b1110;
            else if (x > 0.5016634166240692f) // 110
                return 0b1101;
            else
                return 0b1100;
        else if (x > 0.2035212516784668f) // 10
            if (x > 0.2920137718319893f)  // 101
                return 0b1011;
            else
                return 0b1010;
        else if (x > 0.1202552504837513f) // 100
            return 0b1001;
        else
            return 0b1000;
    else if (x > -0.33967943489551544f)     // 0
        if (x > -0.13791173323988914f)      // 01
            if (x > -0.045525018125772476f) // 011
                return 0b0111;
            else
                return 0b0110;
        else if (x > -0.23460740596055984f) // 010
            return 0b0101;
        else
            return 0b0100;
    else if (x > -0.6106329262256622f) // 00
        if (x > -0.4599952697753906f)  // 001
            return 0b0011;
        else
            return 0b0010;
    else if (x > -0.8480964004993439f) // 000
        return 0b0001;
    else
        return 0b0000;
}

/**
 * @brief Dequantize a 4-bit NF4 value to float.
 *
 * Simple lookup table dequantization.
 *
 * @param val 4-bit NF4 value (0-15)
 * @return Dequantized float value in [-1, 1]
 */
__device__ __forceinline__ float dDequantizeNF4(unsigned char val) {
    return d_nf4_dequant_lut[val & 0x0F];
}

// ============================================================================
// NF4 Blockwise Quantization Kernel
// ============================================================================

/**
 * @brief Fused NF4 blockwise quantization kernel.
 *
 * Single-pass kernel that computes absmax and quantizes in one kernel launch.
 * Each CUDA thread block handles one quantization block:
 * 1. Threads cooperatively load elements and compute local max
 * 2. CUB BlockReduce finds the block's absmax
 * 3. Threads quantize their elements and write packed output
 *
 * This eliminates the second global memory read of the two-pass approach.
 *
 * @tparam BLOCK_SIZE Quantization block size (must be power of 2: 64, 128, 256, 512)
 * @tparam THREADS Number of threads per CUDA block
 * @param[out] out Output packed 4-bit array (n/2 bytes)
 * @param[out] absmax Output absmax scales (n/block_size floats)
 * @param[in] in Input BF16 array (n elements)
 * @param n Total number of elements
 */
template <int BLOCK_SIZE, int THREADS>
__global__ void kQuantizeBnBNF4Fused(
    unsigned char* __restrict__ out,
    float* __restrict__ absmax,
    const nv_bfloat16* __restrict__ in,
    const long n)
{
    // CUB block reduce for finding max
    typedef cub::BlockReduce<float, THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // Shared memory for the computed absmax (broadcast to all threads)
    __shared__ float s_absmax;

    // Each CUDA block handles one quantization block
    const long quant_block_idx = blockIdx.x;
    const long base_elem = quant_block_idx * BLOCK_SIZE;

    // Early exit if this quantization block is entirely out of bounds
    if (base_elem >= n) return;

    // Number of valid elements in this quantization block
    const int valid_elems = min((long)BLOCK_SIZE, n - base_elem);

    // Phase 1: Each thread loads multiple elements and computes local max
    // Each thread handles BLOCK_SIZE / THREADS elements
    constexpr int ELEMS_PER_THREAD = BLOCK_SIZE / THREADS;
    float local_vals[ELEMS_PER_THREAD];
    float thread_max = 0.0f;

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        const int local_idx = threadIdx.x * ELEMS_PER_THREAD + i;
        const long global_idx = base_elem + local_idx;

        if (local_idx < valid_elems) {
            float val = (float)in[global_idx];
            local_vals[i] = val;
            thread_max = fmaxf(thread_max, fabsf(val));
        } else {
            local_vals[i] = 0.0f;
        }
    }

    // Phase 2: Reduce to find block absmax using CUB
    // Use a lambda for max reduction (compatible with all CUB versions)
    auto max_op = [] __device__ (float a, float b) { return fmaxf(a, b); };
    float block_max = BlockReduce(temp_storage).Reduce(thread_max, max_op);

    // Thread 0 stores absmax and broadcasts via shared memory
    if (threadIdx.x == 0) {
        absmax[quant_block_idx] = block_max;
        s_absmax = block_max;
    }
    __syncthreads();

    // Phase 3: Quantize using the computed absmax
    const float local_absmax = s_absmax;
    const float inv_absmax = (local_absmax > 0.0f) ? (1.0f / local_absmax) : 0.0f;

    // Each thread quantizes its elements and packs pairs into bytes
    // Output is packed: 2 elements per byte, so ELEMS_PER_THREAD / 2 bytes per thread
    constexpr int BYTES_PER_THREAD = ELEMS_PER_THREAD / 2;
    const long base_out_byte = (base_elem / 2) + threadIdx.x * BYTES_PER_THREAD;

    #pragma unroll
    for (int i = 0; i < BYTES_PER_THREAD; i++) {
        const int val_idx = i * 2;
        const int local_elem_idx = threadIdx.x * ELEMS_PER_THREAD + val_idx;

        // Normalize and quantize two values
        float v0 = local_vals[val_idx] * inv_absmax;
        float v1 = local_vals[val_idx + 1] * inv_absmax;

        unsigned char q0 = dQuantizeNF4(v0);
        unsigned char q1 = dQuantizeNF4(v1);

        // Pack: high nibble = first value, low nibble = second value
        const long out_idx = base_out_byte + i;
        if (local_elem_idx < valid_elems) {
            out[out_idx] = (q0 << 4) | q1;
        }
    }
}

/**
 * @brief Simple NF4 blockwise quantization kernel (legacy, for non-power-of-2 sizes).
 *
 * Quantizes BF16 weights to 4-bit NF4 with per-block absmax scaling.
 * Each block of `block_size` consecutive elements shares one FP32 scale.
 * Uses a simple approach without CUB for predictable memory layout.
 *
 * @param[out] out Output packed 4-bit array (n/2 bytes)
 * @param[out] absmax Output absmax scales (n/block_size floats)
 * @param[in] in Input BF16 array (n elements)
 * @param blocksize Number of elements per quantization block
 * @param n Total number of elements
 */
__global__ void kQuantizeBnBNF4Simple(
    unsigned char* __restrict__ out,
    float* __restrict__ absmax,
    const nv_bfloat16* __restrict__ in,
    const int blocksize,
    const long n)
{
    // Each thread handles one packed byte (2 elements)
    const long packed_n = (n + 1) / 2;
    const long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= packed_n) return;

    // Element indices for this packed byte
    const long elem_idx = idx * 2;

    // Which absmax block do these elements belong to?
    const int blocksize_shift = 31 - __clz(blocksize);
    const int absmax_idx = elem_idx >> blocksize_shift;

    // Load the absmax for this block (computed in first pass)
    const float local_absmax = __ldg(&absmax[absmax_idx]);
    const float inv_absmax = (local_absmax > 0.0f) ? (1.0f / local_absmax) : 0.0f;

    // Load two elements
    float v0 = (elem_idx < n) ? ((float)in[elem_idx]) * inv_absmax : 0.0f;
    float v1 = (elem_idx + 1 < n) ? ((float)in[elem_idx + 1]) * inv_absmax : 0.0f;

    // Pack: high nibble = first value, low nibble = second value
    out[idx] = (dQuantizeNF4(v0) << 4) | dQuantizeNF4(v1);
}

/**
 * @brief Compute absmax for each block (legacy, used with kQuantizeBnBNF4Simple).
 *
 * First pass: compute the maximum absolute value for each block of elements.
 *
 * @param[out] absmax Output absmax scales (n/blocksize floats)
 * @param[in] in Input BF16 array (n elements)
 * @param blocksize Number of elements per quantization block
 * @param n Total number of elements
 */
__global__ void kComputeAbsmax(
    float* __restrict__ absmax,
    const nv_bfloat16* __restrict__ in,
    const int blocksize,
    const long n)
{
    const long num_blocks = (n + blocksize - 1) / blocksize;
    const long block_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_idx >= num_blocks) return;

    const long start = block_idx * blocksize;
    const long end = min(start + blocksize, n);

    float local_max = 0.0f;
    for (long i = start; i < end; i++) {
        local_max = fmaxf(local_max, fabsf((float)in[i]));
    }

    absmax[block_idx] = local_max;
}

// ============================================================================
// NF4 Blockwise Dequantization Kernel
// ============================================================================

/**
 * @brief Optimized NF4 blockwise dequantization kernel with vectorized memory access.
 *
 * Each thread processes 4 packed bytes (8 BF16 outputs) using vectorized loads/stores.
 * Uses uint32_t loads for 4 packed bytes and nv_bfloat162 stores for coalesced writes.
 *
 * @param[out] out Output BF16 array (n elements, must be 4-byte aligned)
 * @param[in] in Input packed 4-bit array (n/2 bytes, must be 4-byte aligned)
 * @param[in] absmax Per-block absmax scales
 * @param blocksize_shift log2(blocksize) for fast division
 * @param n Total number of output elements
 */
__global__ void kDequantizeBnBNF4Vectorized(
    nv_bfloat162* __restrict__ out,  // Output as BF16x2 for vectorized stores
    const uint32_t* __restrict__ in,  // Input as uint32 (4 packed bytes = 8 values)
    const float* __restrict__ absmax,
    const int blocksize_shift,
    const long n)
{
    // Each thread handles 4 packed bytes = 8 BF16 outputs
    const long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long elem_idx = idx * 8;

    if (elem_idx >= n) return;

    // Load 4 packed bytes at once (8 x 4-bit values)
    const uint32_t packed4 = in[idx];

    // Extract individual bytes
    const unsigned char b0 = (packed4 >> 0) & 0xFF;
    const unsigned char b1 = (packed4 >> 8) & 0xFF;
    const unsigned char b2 = (packed4 >> 16) & 0xFF;
    const unsigned char b3 = (packed4 >> 24) & 0xFF;

    // Compute absmax indices for each pair of elements
    // For block_size >= 8, all 8 elements likely share the same absmax
    const int absmax_idx0 = elem_idx >> blocksize_shift;
    const int absmax_idx4 = (elem_idx + 4) >> blocksize_shift;

    // Load absmax values (often the same for all 8 elements)
    const float scale0 = __ldg(&absmax[absmax_idx0]);
    const float scale4 = (absmax_idx4 != absmax_idx0) ? __ldg(&absmax[absmax_idx4]) : scale0;

    // Dequantize and pack into BF16x2 for vectorized stores
    // Each byte contains 2 x 4-bit values: high nibble first, low nibble second
    nv_bfloat162 out0, out1, out2, out3;

    // Byte 0: elements 0,1
    out0.x = __float2bfloat16(dDequantizeNF4(b0 >> 4) * scale0);
    out0.y = __float2bfloat16(dDequantizeNF4(b0 & 0x0F) * scale0);

    // Byte 1: elements 2,3
    out1.x = __float2bfloat16(dDequantizeNF4(b1 >> 4) * scale0);
    out1.y = __float2bfloat16(dDequantizeNF4(b1 & 0x0F) * scale0);

    // Byte 2: elements 4,5
    out2.x = __float2bfloat16(dDequantizeNF4(b2 >> 4) * scale4);
    out2.y = __float2bfloat16(dDequantizeNF4(b2 & 0x0F) * scale4);

    // Byte 3: elements 6,7
    out3.x = __float2bfloat16(dDequantizeNF4(b3 >> 4) * scale4);
    out3.y = __float2bfloat16(dDequantizeNF4(b3 & 0x0F) * scale4);

    // Vectorized stores (4 x BF16x2 = 8 BF16 values)
    const long out_idx = idx * 4;  // 4 BF16x2 pairs per thread
    if (elem_idx + 7 < n) {
        // Fast path: all 8 elements valid
        out[out_idx + 0] = out0;
        out[out_idx + 1] = out1;
        out[out_idx + 2] = out2;
        out[out_idx + 3] = out3;
    } else {
        // Slow path: handle edge case
        nv_bfloat16* out_scalar = reinterpret_cast<nv_bfloat16*>(out);
        if (elem_idx + 0 < n) out_scalar[elem_idx + 0] = out0.x;
        if (elem_idx + 1 < n) out_scalar[elem_idx + 1] = out0.y;
        if (elem_idx + 2 < n) out_scalar[elem_idx + 2] = out1.x;
        if (elem_idx + 3 < n) out_scalar[elem_idx + 3] = out1.y;
        if (elem_idx + 4 < n) out_scalar[elem_idx + 4] = out2.x;
        if (elem_idx + 5 < n) out_scalar[elem_idx + 5] = out2.y;
        if (elem_idx + 6 < n) out_scalar[elem_idx + 6] = out3.x;
        if (elem_idx + 7 < n) out_scalar[elem_idx + 7] = out3.y;
    }
}

/**
 * @brief Simple NF4 blockwise dequantization kernel (fallback for unaligned data).
 *
 * Dequantizes packed 4-bit NF4 data back to BF16 using per-block absmax scales.
 * Each thread processes one packed byte at a time.
 *
 * @param[out] out Output BF16 array (n elements)
 * @param[in] in Input packed 4-bit array (n/2 bytes)
 * @param[in] absmax Per-block absmax scales
 * @param blocksize Number of elements per quantization block
 * @param n Total number of output elements
 */
__global__ void kDequantizeBnBNF4Simple(
    nv_bfloat16* __restrict__ out,
    const unsigned char* __restrict__ in,
    const float* __restrict__ absmax,
    const int blocksize,
    const long n)
{
    const long packed_n = (n + 1) / 2;  // Number of packed bytes
    const long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= packed_n) return;

    // Load packed byte
    unsigned char packed = in[idx];

    // Element indices for this packed byte
    const long elem_idx = idx * 2;

    // Compute absmax index based on element index
    const int blocksize_shift = 31 - __clz(blocksize);
    const int absmax_idx = elem_idx >> blocksize_shift;
    const float local_abs_max = __ldg(&absmax[absmax_idx]);

    // High nibble (bits 4-7) is first value
    if (elem_idx < n) {
        out[elem_idx] = (nv_bfloat16)(dDequantizeNF4(packed >> 4) * local_abs_max);
    }
    // Low nibble (bits 0-3) is second value
    if (elem_idx + 1 < n) {
        out[elem_idx + 1] = (nv_bfloat16)(dDequantizeNF4(packed & 0x0F) * local_abs_max);
    }
}

// ============================================================================
// Double Quantization Kernels (for absmax compression)
// ============================================================================

/**
 * @brief Quantize absmax values to INT8 for double quantization.
 *
 * Double quantization reduces memory overhead by quantizing the absmax
 * scaling factors themselves. This kernel:
 * 1. Groups absmax values into blocks of 256
 * 2. Computes per-group offset (mean) and scale (max after offset subtraction)
 * 3. Quantizes (absmax - offset) to INT8 using the scale
 *
 * @param[out] out_absmax_quant Output INT8 quantized absmax values
 * @param[out] out_absmax_scale Per-group FP32 scale for INT8 dequantization
 * @param[out] out_absmax_offset Per-group FP32 offset (subtracted before quantization)
 * @param[in] absmax Input FP32 absmax values
 * @param n Number of absmax values
 * @param group_size Number of absmax values per quantization group (default 256)
 */
__global__ void kQuantizeAbsmaxDouble(
    unsigned char* __restrict__ out_absmax_quant,
    float* __restrict__ out_absmax_scale,
    float* __restrict__ out_absmax_offset,
    const float* __restrict__ absmax,
    const int n,
    const int group_size = 256)
{
    const int group_idx = blockIdx.x;
    const int base_idx = group_idx * group_size;
    const int end_idx = min(base_idx + group_size, n);
    const int valid_items = end_idx - base_idx;

    // Shared memory for reduction
    __shared__ float s_sum;
    __shared__ float s_absmax;

    if (threadIdx.x == 0) {
        s_sum = 0.0f;
        s_absmax = 0.0f;
    }
    __syncthreads();

    // Phase 1: Compute sum for offset (mean)
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < valid_items; i += blockDim.x) {
        thread_sum += absmax[base_idx + i];
    }

    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, offset);
    }
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&s_sum, thread_sum);
    }
    __syncthreads();

    // Compute offset (mean)
    float offset = s_sum / (float)valid_items;

    // Phase 2: Compute absmax of (value - offset)
    float thread_max = 0.0f;
    for (int i = threadIdx.x; i < valid_items; i += blockDim.x) {
        float val = absmax[base_idx + i] - offset;
        thread_max = fmaxf(thread_max, fabsf(val));
    }

    // Warp reduction for max
    for (int off = 16; off > 0; off >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, off));
    }
    if (threadIdx.x % 32 == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(&s_absmax), __float_as_uint(thread_max));
    }
    __syncthreads();

    float scale = s_absmax;
    float inv_scale = (scale > 0.0f) ? (127.0f / scale) : 0.0f;

    // Thread 0 stores the scale and offset
    if (threadIdx.x == 0) {
        out_absmax_scale[group_idx] = scale / 127.0f;  // Store dequant scale
        out_absmax_offset[group_idx] = offset;
    }

    // Phase 3: Quantize to INT8
    for (int i = threadIdx.x; i < valid_items; i += blockDim.x) {
        float val = absmax[base_idx + i] - offset;
        // Quantize to [-127, 127] range, stored as uint8 with offset 128
        int qval = __float2int_rn(val * inv_scale) + 128;
        qval = max(0, min(255, qval));
        out_absmax_quant[base_idx + i] = (unsigned char)qval;
    }
}

/**
 * @brief Dequantize INT8 absmax values back to FP32.
 *
 * @param[out] out_absmax Output FP32 absmax values
 * @param[in] in_absmax_quant Input INT8 quantized absmax values
 * @param[in] in_absmax_scale Per-group FP32 dequantization scale
 * @param[in] in_absmax_offset Per-group FP32 offset
 * @param n Number of absmax values
 * @param group_size Number of absmax values per group
 */
__global__ void kDequantizeAbsmaxDouble(
    float* __restrict__ out_absmax,
    const unsigned char* __restrict__ in_absmax_quant,
    const float* __restrict__ in_absmax_scale,
    const float* __restrict__ in_absmax_offset,
    const int n,
    const int group_size = 256)
{
    const long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int group_idx = idx / group_size;
    const float scale = in_absmax_scale[group_idx];
    const float offset = in_absmax_offset[group_idx];

    // Dequantize: value = (qval - 128) * scale + offset
    float qval = (float)in_absmax_quant[idx] - 128.0f;
    out_absmax[idx] = qval * scale + offset;
}

// ============================================================================
// NF4 Dequantization with Double Quantization Support
// ============================================================================

/**
 * @brief Vectorized NF4 dequantization with inline absmax dequantization.
 *
 * Each thread processes 4 packed bytes (8 BF16 outputs) using vectorized loads/stores.
 * Handles double quantization by inline dequantizing INT8 absmax values.
 *
 * @param[out] out Output BF16 array (n elements, as BF16x2 for vectorized stores)
 * @param[in] in Input packed 4-bit array (as uint32 for vectorized loads)
 * @param[in] absmax_quant Quantized INT8 absmax values
 * @param[in] absmax_scale Per-group FP32 scale for absmax
 * @param[in] absmax_offset Per-group FP32 offset for absmax
 * @param blocksize_shift log2(blocksize) for fast division
 * @param absmax_group_size Group size for double quantization
 * @param n Total number of output elements
 */
__global__ void kDequantizeBnBNF4DoubleVectorized(
    nv_bfloat162* __restrict__ out,
    const uint32_t* __restrict__ in,
    const unsigned char* __restrict__ absmax_quant,
    const float* __restrict__ absmax_scale,
    const float* __restrict__ absmax_offset,
    const int blocksize_shift,
    const int absmax_group_size,
    const long n)
{
    const long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long elem_idx = idx * 8;

    if (elem_idx >= n) return;

    // Load 4 packed bytes at once
    const uint32_t packed4 = in[idx];
    const unsigned char b0 = (packed4 >> 0) & 0xFF;
    const unsigned char b1 = (packed4 >> 8) & 0xFF;
    const unsigned char b2 = (packed4 >> 16) & 0xFF;
    const unsigned char b3 = (packed4 >> 24) & 0xFF;

    // Compute absmax indices
    const int absmax_idx0 = elem_idx >> blocksize_shift;
    const int absmax_idx4 = (elem_idx + 4) >> blocksize_shift;

    // Dequantize absmax for first 4 elements
    const int group0 = absmax_idx0 / absmax_group_size;
    const float scale0_dq = __ldg(&absmax_scale[group0]);
    const float offset0_dq = __ldg(&absmax_offset[group0]);
    const unsigned char qabs0 = __ldg(&absmax_quant[absmax_idx0]);
    const float scale0 = ((float)qabs0 - 128.0f) * scale0_dq + offset0_dq;

    // Dequantize absmax for last 4 elements (may be same as first)
    float scale4;
    if (absmax_idx4 != absmax_idx0) {
        const int group4 = absmax_idx4 / absmax_group_size;
        const float scale4_dq = __ldg(&absmax_scale[group4]);
        const float offset4_dq = __ldg(&absmax_offset[group4]);
        const unsigned char qabs4 = __ldg(&absmax_quant[absmax_idx4]);
        scale4 = ((float)qabs4 - 128.0f) * scale4_dq + offset4_dq;
    } else {
        scale4 = scale0;
    }

    // Dequantize and pack into BF16x2
    nv_bfloat162 out0, out1, out2, out3;

    out0.x = __float2bfloat16(dDequantizeNF4(b0 >> 4) * scale0);
    out0.y = __float2bfloat16(dDequantizeNF4(b0 & 0x0F) * scale0);
    out1.x = __float2bfloat16(dDequantizeNF4(b1 >> 4) * scale0);
    out1.y = __float2bfloat16(dDequantizeNF4(b1 & 0x0F) * scale0);
    out2.x = __float2bfloat16(dDequantizeNF4(b2 >> 4) * scale4);
    out2.y = __float2bfloat16(dDequantizeNF4(b2 & 0x0F) * scale4);
    out3.x = __float2bfloat16(dDequantizeNF4(b3 >> 4) * scale4);
    out3.y = __float2bfloat16(dDequantizeNF4(b3 & 0x0F) * scale4);

    // Vectorized stores
    const long out_idx = idx * 4;
    if (elem_idx + 7 < n) {
        out[out_idx + 0] = out0;
        out[out_idx + 1] = out1;
        out[out_idx + 2] = out2;
        out[out_idx + 3] = out3;
    } else {
        nv_bfloat16* out_scalar = reinterpret_cast<nv_bfloat16*>(out);
        if (elem_idx + 0 < n) out_scalar[elem_idx + 0] = out0.x;
        if (elem_idx + 1 < n) out_scalar[elem_idx + 1] = out0.y;
        if (elem_idx + 2 < n) out_scalar[elem_idx + 2] = out1.x;
        if (elem_idx + 3 < n) out_scalar[elem_idx + 3] = out1.y;
        if (elem_idx + 4 < n) out_scalar[elem_idx + 4] = out2.x;
        if (elem_idx + 5 < n) out_scalar[elem_idx + 5] = out2.y;
        if (elem_idx + 6 < n) out_scalar[elem_idx + 6] = out3.x;
        if (elem_idx + 7 < n) out_scalar[elem_idx + 7] = out3.y;
    }
}

/**
 * @brief Simple NF4 dequantization with inline absmax dequantization (fallback).
 *
 * When double quantization is used, this kernel handles both:
 * 1. Dequantizing INT8 absmax -> FP32 absmax
 * 2. Dequantizing NF4 data -> BF16 using the recovered absmax
 *
 * @param[out] out Output BF16 array (n elements)
 * @param[in] in Input packed 4-bit array (n/2 bytes)
 * @param[in] absmax_quant Quantized INT8 absmax values
 * @param[in] absmax_scale Per-group FP32 scale for absmax
 * @param[in] absmax_offset Per-group FP32 offset for absmax
 * @param blocksize Quantization block size in elements
 * @param absmax_group_size Group size for double quantization (typically 256)
 * @param n Total number of output elements
 */
__global__ void kDequantizeBnBNF4DoubleSimple(
    nv_bfloat16* __restrict__ out,
    const unsigned char* __restrict__ in,
    const unsigned char* __restrict__ absmax_quant,
    const float* __restrict__ absmax_scale,
    const float* __restrict__ absmax_offset,
    const int blocksize,
    const int absmax_group_size,
    const long n)
{
    const long packed_n = (n + 1) / 2;  // Number of packed bytes
    const long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= packed_n) return;

    // Load packed byte
    unsigned char packed = in[idx];

    // Element indices for this packed byte
    const long elem_idx = idx * 2;

    // Compute absmax index based on element index
    const int blocksize_shift = 31 - __clz(blocksize);
    const int absmax_idx = elem_idx >> blocksize_shift;

    // Dequantize absmax
    const int absmax_group = absmax_idx / absmax_group_size;
    const float scale = __ldg(&absmax_scale[absmax_group]);
    const float offset = __ldg(&absmax_offset[absmax_group]);
    const unsigned char qabsmax = __ldg(&absmax_quant[absmax_idx]);
    const float local_abs_max = ((float)qabsmax - 128.0f) * scale + offset;

    // High nibble (bits 4-7) is first value
    if (elem_idx < n) {
        out[elem_idx] = (nv_bfloat16)(dDequantizeNF4(packed >> 4) * local_abs_max);
    }
    // Low nibble (bits 0-3) is second value
    if (elem_idx + 1 < n) {
        out[elem_idx + 1] = (nv_bfloat16)(dDequantizeNF4(packed & 0x0F) * local_abs_max);
    }
}

// ============================================================================
// Host Launcher Functions
// ============================================================================

/**
 * @brief Host launcher for NF4 blockwise quantization.
 *
 * Quantizes BF16 weights to packed 4-bit NF4 with per-block absmax scales.
 * Uses a fused single-pass kernel for supported block sizes (64, 128, 256, 512)
 * which eliminates the second global memory read for ~2x speedup.
 *
 * @param[out] out Output packed 4-bit array (M*K/2 bytes)
 * @param[out] absmax Output per-block absmax scales (M*K/blocksize floats)
 * @param[in] in Input BF16 array (M*K elements)
 * @param M Number of rows
 * @param K Number of columns
 * @param block_size Quantization block size (64, 128, 256, or 512)
 * @param dp CUDA device properties
 * @param stream CUDA stream
 */
void quantize_bnb_nf4(
    unsigned char* out,
    float* absmax,
    const nv_bfloat16* in,
    int M, int K,
    int block_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const long n = (long)M * K;
    const long num_absmax_blocks = (n + block_size - 1) / block_size;

    // Use fused kernel for supported block sizes (single-pass, ~2x faster)
    // Each CUDA block handles one quantization block
    // Thread count chosen so each thread handles 2-8 elements for good occupancy
    switch (block_size) {
        case 64: {
            // 64 elements / 32 threads = 2 elements per thread
            constexpr int THREADS = 32;
            kQuantizeBnBNF4Fused<64, THREADS><<<num_absmax_blocks, THREADS, 0, stream>>>(
                out, absmax, in, n);
            break;
        }
        case 128: {
            // 128 elements / 32 threads = 4 elements per thread
            constexpr int THREADS = 32;
            kQuantizeBnBNF4Fused<128, THREADS><<<num_absmax_blocks, THREADS, 0, stream>>>(
                out, absmax, in, n);
            break;
        }
        case 256: {
            // 256 elements / 64 threads = 4 elements per thread
            constexpr int THREADS = 64;
            kQuantizeBnBNF4Fused<256, THREADS><<<num_absmax_blocks, THREADS, 0, stream>>>(
                out, absmax, in, n);
            break;
        }
        case 512: {
            // 512 elements / 128 threads = 4 elements per thread
            constexpr int THREADS = 128;
            kQuantizeBnBNF4Fused<512, THREADS><<<num_absmax_blocks, THREADS, 0, stream>>>(
                out, absmax, in, n);
            break;
        }
        default: {
            // Fallback to two-pass approach for non-standard block sizes
            const long packed_n = (n + 1) / 2;
            constexpr int THREADS = 256;

            // First pass: compute absmax for each block
            const int absmax_grid = (num_absmax_blocks + THREADS - 1) / THREADS;
            kComputeAbsmax<<<absmax_grid, THREADS, 0, stream>>>(absmax, in, block_size, n);
            CUDA_CHECK(cudaGetLastError());

            // Second pass: quantize using the computed absmax values
            const int quant_grid = (packed_n + THREADS - 1) / THREADS;
            kQuantizeBnBNF4Simple<<<quant_grid, THREADS, 0, stream>>>(out, absmax, in, block_size, n);
            break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for NF4 blockwise dequantization.
 *
 * Dequantizes packed 4-bit NF4 data back to BF16.
 * Uses vectorized kernel (4x throughput) when data is aligned and n >= 8.
 *
 * @param[out] out Output BF16 array (M*K elements)
 * @param[in] in Input packed 4-bit array (M*K/2 bytes)
 * @param[in] absmax Per-block absmax scales
 * @param M Number of rows
 * @param K Number of columns
 * @param block_size Quantization block size
 * @param dp CUDA device properties
 * @param stream CUDA stream
 */
void dequantize_bnb_nf4(
    nv_bfloat16* out,
    const unsigned char* in,
    const float* absmax,
    int M, int K,
    int block_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const long n = (long)M * K;

    // Check alignment for vectorized kernel (4-byte aligned)
    const bool aligned = ((reinterpret_cast<uintptr_t>(out) % 4) == 0) &&
                         ((reinterpret_cast<uintptr_t>(in) % 4) == 0);

    // Use vectorized kernel for aligned data with sufficient elements
    // Each thread processes 8 elements, so we need n >= 8
    if (aligned && n >= 8) {
        // Vectorized kernel: each thread handles 4 packed bytes = 8 BF16 outputs
        constexpr int THREADS = 256;
        const long num_vec_elements = (n + 7) / 8;  // Number of 8-element groups
        const int num_blocks = (num_vec_elements + THREADS - 1) / THREADS;
        // Host-side log2 for power-of-2 block sizes
        const int blocksize_shift = 31 - __builtin_clz(block_size);

        kDequantizeBnBNF4Vectorized<<<num_blocks, THREADS, 0, stream>>>(
            reinterpret_cast<nv_bfloat162*>(out),
            reinterpret_cast<const uint32_t*>(in),
            absmax, blocksize_shift, n);
    } else {
        // Fallback: simple kernel for unaligned or small data
        const long packed_n = (n + 1) / 2;
        constexpr int THREADS = 256;
        const int num_blocks = (packed_n + THREADS - 1) / THREADS;

        kDequantizeBnBNF4Simple<<<num_blocks, THREADS, 0, stream>>>(
            out, in, absmax, block_size, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for double quantization of absmax values.
 *
 * Quantizes FP32 absmax values to INT8 for memory savings.
 *
 * @param[out] out_quant Output INT8 quantized absmax
 * @param[out] out_scale Per-group dequantization scale
 * @param[out] out_offset Per-group offset
 * @param[in] absmax Input FP32 absmax values
 * @param num_absmax Number of absmax values
 * @param group_size Values per quantization group (default 256)
 * @param dp CUDA device properties
 * @param stream CUDA stream
 */
void quantize_absmax_double(
    unsigned char* out_quant,
    float* out_scale,
    float* out_offset,
    const float* absmax,
    int num_absmax,
    int group_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const int num_groups = (num_absmax + group_size - 1) / group_size;
    const int threads = 256;

    kQuantizeAbsmaxDouble<<<num_groups, threads, 0, stream>>>(
        out_quant, out_scale, out_offset, absmax, num_absmax, group_size);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for dequantizing INT8 absmax back to FP32.
 *
 * @param[out] out_absmax Output FP32 absmax values
 * @param[in] in_quant Input INT8 quantized absmax
 * @param[in] in_scale Per-group dequantization scale
 * @param[in] in_offset Per-group offset
 * @param num_absmax Number of absmax values
 * @param group_size Values per quantization group
 * @param dp CUDA device properties
 * @param stream CUDA stream
 */
void dequantize_absmax_double(
    float* out_absmax,
    const unsigned char* in_quant,
    const float* in_scale,
    const float* in_offset,
    int num_absmax,
    int group_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const int threads = 256;
    const int num_blocks = (num_absmax + threads - 1) / threads;

    kDequantizeAbsmaxDouble<<<num_blocks, threads, 0, stream>>>(
        out_absmax, in_quant, in_scale, in_offset, num_absmax, group_size);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for NF4 dequantization with double quantization.
 *
 * Dequantizes NF4 data when absmax values are also quantized (double quant).
 * Uses vectorized kernel (4x throughput) when data is aligned and n >= 8.
 *
 * @param[out] out Output BF16 array (M*K elements)
 * @param[in] in Input packed 4-bit array (M*K/2 bytes)
 * @param[in] absmax_quant Quantized INT8 absmax values
 * @param[in] absmax_scale Per-group FP32 scale for absmax
 * @param[in] absmax_offset Per-group FP32 offset for absmax
 * @param M Number of rows
 * @param K Number of columns
 * @param block_size Quantization block size
 * @param absmax_group_size Group size for double quantization
 * @param dp CUDA device properties
 * @param stream CUDA stream
 */
void dequantize_bnb_nf4_double(
    nv_bfloat16* out,
    const unsigned char* in,
    const unsigned char* absmax_quant,
    const float* absmax_scale,
    const float* absmax_offset,
    int M, int K,
    int block_size,
    int absmax_group_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const long n = (long)M * K;

    // Check alignment for vectorized kernel
    const bool aligned = ((reinterpret_cast<uintptr_t>(out) % 4) == 0) &&
                         ((reinterpret_cast<uintptr_t>(in) % 4) == 0);

    if (aligned && n >= 8) {
        // Vectorized kernel: each thread handles 8 BF16 outputs
        constexpr int THREADS = 256;
        const long num_vec_elements = (n + 7) / 8;
        const int num_blocks = (num_vec_elements + THREADS - 1) / THREADS;
        // Host-side log2 for power-of-2 block sizes
        const int blocksize_shift = 31 - __builtin_clz(block_size);

        kDequantizeBnBNF4DoubleVectorized<<<num_blocks, THREADS, 0, stream>>>(
            reinterpret_cast<nv_bfloat162*>(out),
            reinterpret_cast<const uint32_t*>(in),
            absmax_quant, absmax_scale, absmax_offset,
            blocksize_shift, absmax_group_size, n);
    } else {
        // Fallback: simple kernel
        const long packed_n = (n + 1) / 2;
        constexpr int THREADS = 256;
        const int num_blocks = (packed_n + THREADS - 1) / THREADS;

        kDequantizeBnBNF4DoubleSimple<<<num_blocks, THREADS, 0, stream>>>(
            out, in, absmax_quant, absmax_scale, absmax_offset, block_size, absmax_group_size, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Tensor-based Wrapper Functions
// ============================================================================

/**
 * @brief Tensor-based wrapper for NF4 quantization.
 */
void quantize_bnb_nf4(
    Tensor& out,
    Tensor& absmax,
    const Tensor& in,
    int block_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (in.DType != ETensorDType::BF16) {
        throw std::runtime_error("quantize_bnb_nf4: input must be BF16");
    }
    if (out.DType != ETensorDType::INT8) {
        throw std::runtime_error("quantize_bnb_nf4: output must be INT8 (packed NF4)");
    }
    if (absmax.DType != ETensorDType::FP32) {
        throw std::runtime_error("quantize_bnb_nf4: absmax must be FP32");
    }

    const int M = in.Sizes[0];
    const int K = (in.Rank == 2) ? in.Sizes[1] : 1;

    quantize_bnb_nf4(
        out.get<unsigned char>(),
        absmax.get<float>(),
        in.get<nv_bfloat16>(),
        M, K, block_size, dp, stream);
}

/**
 * @brief Tensor-based wrapper for NF4 dequantization.
 */
void dequantize_bnb_nf4(
    Tensor& out,
    const Tensor& in,
    const Tensor& absmax,
    int block_size,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (out.DType != ETensorDType::BF16) {
        throw std::runtime_error("dequantize_bnb_nf4: output must be BF16");
    }
    if (in.DType != ETensorDType::INT8) {
        throw std::runtime_error("dequantize_bnb_nf4: input must be INT8 (packed NF4)");
    }
    if (absmax.DType != ETensorDType::FP32) {
        throw std::runtime_error("dequantize_bnb_nf4: absmax must be FP32");
    }

    const int M = out.Sizes[0];
    const int K = (out.Rank == 2) ? out.Sizes[1] : 1;

    dequantize_bnb_nf4(
        out.get<nv_bfloat16>(),
        in.get<unsigned char>(),
        absmax.get<float>(),
        M, K, block_size, dp, stream);
}

/**
 * @brief Get the NF4 codebook values (host-side).
 *
 * Returns a pointer to the 16-element NF4 lookup table for host-side
 * operations or debugging.
 *
 * @return Pointer to static array of 16 floats
 */
const float* get_nf4_codebook() {
    return h_nf4_dequant_lut;
}
