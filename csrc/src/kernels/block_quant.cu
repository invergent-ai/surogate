// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file block_quant.cu
 * @brief CUDA kernels for per-block quantization/dequantization (QLoRA-style).
 *
 * Provides GPU-accelerated per-block quantization for memory-efficient QLoRA training:
 * - Per-block (e.g., 128x128) FP8 quantization with fused abs_max computation
 * - Per-block dequantization for on-the-fly weight reconstruction
 *
 * Unlike per-tensor quantization, per-block scaling provides better numerical accuracy
 * for large weight matrices while maintaining memory efficiency.
 */

#include "kernel_utils.cuh"
#include "utilities/tensor.h"
#include "utilities/vec.cuh"
#include "utilities/utils.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

/**
 * @brief Per-block FP8 quantization kernel with fused abs_max computation.
 *
 * Each CUDA block processes one (BLOCK_SIZE x BLOCK_SIZE) tile of the weight matrix.
 * Phase 1: Compute abs_max for the tile using shared memory reduction.
 * Phase 2: Quantize each element to FP8 E4M3 using the computed scale.
 * Stores inverse scale for dequantization.
 *
 * @tparam BLOCK_SIZE Tile size (default 128, must be power of 2).
 * @param[out] out Output FP8 array (M, K), same layout as input.
 * @param[out] block_scales Output scales array (ceil(M/BLOCK_SIZE), ceil(K/BLOCK_SIZE)).
 *             Each scale is the inverse quantization scale (abs_max / 448) for dequantization.
 * @param[in] in Input BF16 array (M, K).
 * @param M Number of rows in the weight matrix.
 * @param K Number of columns in the weight matrix.
 * @param scale_cols Number of columns in the block_scales tensor (ceil(K/BLOCK_SIZE)).
 */
template <int BLOCK_SIZE = 128>
__global__ void quantize_per_block_kernel(__nv_fp8_e4m3* __restrict__ out,
                                          float* __restrict__ block_scales,
                                          const nv_bfloat16* __restrict__ in,
                                          int M,
                                          int K,
                                          int scale_cols) {
    // Each CUDA block handles one (BLOCK_SIZE x BLOCK_SIZE) tile
    const int block_row = blockIdx.x;
    const int block_col = blockIdx.y;

    // Tile boundaries in the weight matrix
    const int row_start = block_row * BLOCK_SIZE;
    const int col_start = block_col * BLOCK_SIZE;
    const int row_end = min(row_start + BLOCK_SIZE, M);
    const int col_end = min(col_start + BLOCK_SIZE, K);

    // Phase 1: Compute abs_max for this tile
    // Use shared memory for block-level reduction
    __shared__ float s_block_max;
    if (threadIdx.x == 0) {
        s_block_max = 0.f;
    }
    __syncthreads();

    float thread_max = 0.f;
    const int tile_rows = row_end - row_start;
    const int tile_cols = col_end - col_start;
    const int tile_size = tile_rows * tile_cols;

    // Each thread processes multiple elements in the tile
    for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
        const int local_row = idx / tile_cols;
        const int local_col = idx % tile_cols;
        const int global_row = row_start + local_row;
        const int global_col = col_start + local_col;

        float val = (float)in[global_row * K + global_col];
        thread_max = fmaxf(thread_max, fabsf(val));
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, offset));
    }

    // First thread in each warp updates shared memory
    if (threadIdx.x % 32 == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(&s_block_max), __float_as_uint(thread_max));
    }
    __syncthreads();

    // Compute scale: we want fp8_val = bf16_val * (448 / abs_max)
    // Store inverse scale for dequantization: bf16_val = fp8_val * (abs_max / 448)
    const float abs_max = s_block_max;
    const float quant_scale = 448.f / fmaxf(abs_max, 1e-10f);
    const float dequant_scale = abs_max / 448.f;  // Inverse scale for dequantization

    // Thread 0 stores the scale
    if (threadIdx.x == 0) {
        block_scales[block_row * scale_cols + block_col] = dequant_scale;
    }

    // Phase 2: Quantize each element using the computed scale
    for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
        const int local_row = idx / tile_cols;
        const int local_col = idx % tile_cols;
        const int global_row = row_start + local_row;
        const int global_col = col_start + local_col;
        const int global_idx = global_row * K + global_col;

        float val = (float)in[global_idx];
        __nv_fp8_e4m3 result;
        result.__x = __nv_cvt_float_to_fp8(val * quant_scale,
                                           __nv_saturation_t::__NV_SATFINITE,
                                           __nv_fp8_interpretation_t::__NV_E4M3);
        out[global_idx] = result;
    }
}

/**
 * @brief Per-block FP8 dequantization kernel.
 *
 * Each thread dequantizes one element by looking up the appropriate block scale.
 * This kernel is highly memory-bound; the scale lookup is negligible overhead.
 *
 * @tparam BLOCK_SIZE Tile size used during quantization (default 128).
 * @param[out] out Output BF16 array (M, K).
 * @param[in] in Input FP8 array (M, K).
 * @param[in] block_scales Scales array (ceil(M/BLOCK_SIZE), ceil(K/BLOCK_SIZE)).
 * @param M Number of rows in the weight matrix.
 * @param K Number of columns in the weight matrix.
 * @param scale_cols Number of columns in the block_scales tensor.
 */
template <int BLOCK_SIZE = 128>
__global__ void dequantize_per_block_kernel(nv_bfloat16* __restrict__ out,
                                            const __nv_fp8_e4m3* __restrict__ in,
                                            const float* __restrict__ block_scales,
                                            int M,
                                            int K,
                                            int scale_cols) {
    // Grid-stride loop for processing all elements
    const long total_elements = (long)M * K;

    for (long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const int row = idx / K;
        const int col = idx % K;

        // Compute which block this element belongs to
        const int block_row = row / BLOCK_SIZE;
        const int block_col = col / BLOCK_SIZE;

        // Load the scale for this block
        const float scale = block_scales[block_row * scale_cols + block_col];

        // Dequantize: bf16_val = fp8_val * scale
        const __nv_fp8_e4m3 fp8_val = in[idx];
        float fp32_val = __half2float(__nv_cvt_fp8_to_halfraw(fp8_val.__x, __nv_fp8_interpretation_t::__NV_E4M3));
        out[idx] = (nv_bfloat16)(fp32_val * scale);
    }
}

/**
 * @brief Vectorized per-block FP8 dequantization kernel.
 *
 * Processes multiple elements per thread using vectorized loads/stores for better
 * memory bandwidth utilization. Falls back to scalar processing for edge cases.
 *
 * @tparam BLOCK_SIZE Tile size used during quantization (default 128).
 * @param[out] out Output BF16 array (M, K).
 * @param[in] in Input FP8 array (M, K).
 * @param[in] block_scales Scales array (ceil(M/BLOCK_SIZE), ceil(K/BLOCK_SIZE)).
 * @param M Number of rows in the weight matrix.
 * @param K Number of columns in the weight matrix.
 * @param scale_cols Number of columns in the block_scales tensor.
 */
template <int BLOCK_SIZE = 128>
__global__ void dequantize_per_block_vec_kernel(nv_bfloat16* __restrict__ out,
                                                const __nv_fp8_e4m3* __restrict__ in,
                                                const float* __restrict__ block_scales,
                                                int M,
                                                int K,
                                                int scale_cols) {
    // Process 8 elements per thread (8 bytes FP8 -> 16 bytes BF16)
    constexpr int VEC_SIZE = 8;
    using fp8_vec_t = GenericVector<__nv_fp8_e4m3, VEC_SIZE>;
    using bf16_vec_t = GenericVector<nv_bfloat16, VEC_SIZE>;

    const long total_elements = (long)M * K;
    const long vec_elements = total_elements / VEC_SIZE * VEC_SIZE;

    // Vectorized processing
    for (long idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE; idx < vec_elements;
         idx += gridDim.x * blockDim.x * VEC_SIZE) {
        // Load 8 FP8 values
        fp8_vec_t fp8_vals = fp8_vec_t::load(in + idx);
        bf16_vec_t bf16_vals;

#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            const long elem_idx = idx + i;
            const int row = elem_idx / K;
            const int col = elem_idx % K;
            const int block_row = row / BLOCK_SIZE;
            const int block_col = col / BLOCK_SIZE;

            const float scale = block_scales[block_row * scale_cols + block_col];
            float fp32_val =
                __half2float(__nv_cvt_fp8_to_halfraw(fp8_vals[i].__x, __nv_fp8_interpretation_t::__NV_E4M3));
            bf16_vals[i] = (nv_bfloat16)(fp32_val * scale);
        }

        bf16_vals.store(out + idx);
    }

    // Handle remaining elements (if total_elements is not divisible by VEC_SIZE)
    for (long idx = vec_elements + blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
         idx += gridDim.x * blockDim.x) {
        const int row = idx / K;
        const int col = idx % K;
        const int block_row = row / BLOCK_SIZE;
        const int block_col = col / BLOCK_SIZE;

        const float scale = block_scales[block_row * scale_cols + block_col];
        const __nv_fp8_e4m3 fp8_val = in[idx];
        float fp32_val = __half2float(__nv_cvt_fp8_to_halfraw(fp8_val.__x, __nv_fp8_interpretation_t::__NV_E4M3));
        out[idx] = (nv_bfloat16)(fp32_val * scale);
    }
}

/**
 * @brief Fused per-block dequantization + per-tensor requantization kernel.
 *
 * This kernel fuses two operations:
 * 1. Dequantize from per-block FP8 (with block_scales) to FP32
 * 2. Compute global abs_max and requantize to per-tensor FP8 (with single scale)
 *
 * This eliminates the intermediate BF16 storage and separate quantization kernel,
 * reducing memory traffic and kernel launch overhead when using QLoRA-FP8 with
 * FP8 forward/hybrid modes.
 *
 * Algorithm:
 * - Phase 1: Each thread dequantizes its elements and computes local abs_max
 * - Phase 2: Warp-level reduction to find block abs_max
 * - Phase 3: Atomic update to global abs_max
 * - Phase 4: Requantize all elements using the global abs_max
 *
 * Note: This kernel requires two passes (or careful synchronization) because
 * we need the global abs_max before quantizing. We use atomics for simplicity.
 *
 * @tparam BLOCK_SIZE Tile size used for per-block scales (default 128).
 * @param[out] out Output per-tensor FP8 array (M, K).
 * @param[out] out_scale Output per-tensor scale (single float).
 * @param[in] in Input per-block FP8 array (M, K).
 * @param[in] block_scales Input per-block scales (ceil(M/BLOCK_SIZE), ceil(K/BLOCK_SIZE)).
 * @param M Number of rows.
 * @param K Number of columns.
 * @param scale_cols Number of columns in block_scales tensor.
 */
template <int BLOCK_SIZE = 128>
__global__ void fused_dequant_requant_per_block_to_tensor_kernel(__nv_fp8_e4m3* __restrict__ out,
                                                                 float* __restrict__ out_scale,
                                                                 const __nv_fp8_e4m3* __restrict__ in,
                                                                 const float* __restrict__ block_scales,
                                                                 int M,
                                                                 int K,
                                                                 int scale_cols) {
    const long total_elements = (long)M * K;

    // Shared memory for block-level abs_max reduction
    __shared__ float s_block_absmax;
    if (threadIdx.x == 0) {
        s_block_absmax = 0.0f;
    }
    __syncthreads();

    // Phase 1: Dequantize and find local abs_max
    float thread_absmax = 0.0f;

    for (long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const int row = idx / K;
        const int col = idx % K;
        const int block_row = row / BLOCK_SIZE;
        const int block_col = col / BLOCK_SIZE;

        // Dequantize from per-block FP8
        const float block_scale = block_scales[block_row * scale_cols + block_col];
        const __nv_fp8_e4m3 fp8_val = in[idx];
        float fp32_val = __half2float(__nv_cvt_fp8_to_halfraw(fp8_val.__x, __nv_fp8_interpretation_t::__NV_E4M3));
        fp32_val *= block_scale;

        // Track abs_max
        thread_absmax = fmaxf(thread_absmax, fabsf(fp32_val));
    }

    // Phase 2: Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_absmax = fmaxf(thread_absmax, __shfl_xor_sync(0xFFFFFFFF, thread_absmax, offset));
    }

    // Phase 3: Block-level reduction and global update
    if (threadIdx.x % 32 == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(&s_block_absmax), __float_as_uint(thread_absmax));
    }
    __syncthreads();

    // First thread in block updates global abs_max
    if (threadIdx.x == 0 && s_block_absmax > 0.0f) {
        atomicMax(reinterpret_cast<unsigned int*>(out_scale), __float_as_uint(s_block_absmax));
    }
}

/**
 * @brief Second pass: Requantize using the computed global scale.
 *
 * This kernel must run after the abs_max kernel completes.
 *
 * @tparam BLOCK_SIZE Tile size used for per-block scales.
 * @param[out] out Output per-tensor FP8 array (M, K).
 * @param[in] tensor_scale Per-tensor scale (abs_max / 448).
 * @param[in] in Input per-block FP8 array (M, K).
 * @param[in] block_scales Input per-block scales.
 * @param M Number of rows.
 * @param K Number of columns.
 * @param scale_cols Number of columns in block_scales tensor.
 */
template <int BLOCK_SIZE = 128>
__global__ void requantize_to_per_tensor_kernel(__nv_fp8_e4m3* __restrict__ out,
                                                float tensor_scale,
                                                const __nv_fp8_e4m3* __restrict__ in,
                                                const float* __restrict__ block_scales,
                                                int M,
                                                int K,
                                                int scale_cols) {
    const long total_elements = (long)M * K;
    const float quant_scale = 448.0f / fmaxf(tensor_scale, 1e-10f);

    for (long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const int row = idx / K;
        const int col = idx % K;
        const int block_row = row / BLOCK_SIZE;
        const int block_col = col / BLOCK_SIZE;

        // Dequantize from per-block FP8
        const float block_scale = block_scales[block_row * scale_cols + block_col];
        const __nv_fp8_e4m3 fp8_val = in[idx];
        float fp32_val = __half2float(__nv_cvt_fp8_to_halfraw(fp8_val.__x, __nv_fp8_interpretation_t::__NV_E4M3));
        fp32_val *= block_scale;

        // Requantize to per-tensor FP8
        __nv_fp8_e4m3 result;
        result.__x = __nv_cvt_float_to_fp8(fp32_val * quant_scale,
                                           __nv_saturation_t::__NV_SATFINITE,
                                           __nv_fp8_interpretation_t::__NV_E4M3);
        out[idx] = result;
    }
}

// ============================================================================
// Host Launcher Functions
// ============================================================================

/**
 * @brief Host launcher for per-block FP8 quantization.
 *
 * Quantizes a BF16 weight matrix to FP8 E4M3 with per-block scales.
 * Each (block_size x block_size) tile gets its own scale factor.
 *
 * @param[out] out Output FP8 array (M, K).
 * @param[out] block_scales Output scales array (ceil(M/block_size), ceil(K/block_size)).
 * @param[in] in Input BF16 array (M, K).
 * @param M Number of rows.
 * @param K Number of columns.
 * @param block_size Tile size for per-block quantization (must be 64, 128, or 256).
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void quantize_per_block(__nv_fp8_e4m3* out,
                        float* block_scales,
                        const nv_bfloat16* in,
                        int M,
                        int K,
                        int block_size,
                        const cudaDeviceProp& dp,
                        cudaStream_t stream) {
    const int scale_rows = div_ceil(M, block_size);
    const int scale_cols = div_ceil(K, block_size);

    // Launch one CUDA block per weight tile
    dim3 grid(scale_rows, scale_cols);
    const int threads_per_block = 256;  // Good for shared memory reduction

    if (block_size == 128) {
        quantize_per_block_kernel<128><<<grid, threads_per_block, 0, stream>>>(out, block_scales, in, M, K, scale_cols);
    } else if (block_size == 64) {
        quantize_per_block_kernel<64><<<grid, threads_per_block, 0, stream>>>(out, block_scales, in, M, K, scale_cols);
    } else if (block_size == 256) {
        quantize_per_block_kernel<256><<<grid, threads_per_block, 0, stream>>>(out, block_scales, in, M, K, scale_cols);
    } else {
        throw std::runtime_error("quantize_per_block: unsupported block_size (must be 64, 128, or 256)");
    }
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for per-block FP8 dequantization.
 *
 * Dequantizes an FP8 E4M3 weight matrix to BF16 using per-block scales.
 *
 * @param[out] out Output BF16 array (M, K).
 * @param[in] in Input FP8 array (M, K).
 * @param[in] block_scales Scales array (ceil(M/block_size), ceil(K/block_size)).
 * @param M Number of rows.
 * @param K Number of columns.
 * @param block_size Tile size used during quantization.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void dequantize_per_block(nv_bfloat16* out,
                          const __nv_fp8_e4m3* in,
                          const float* block_scales,
                          int M,
                          int K,
                          int block_size,
                          const cudaDeviceProp& dp,
                          cudaStream_t stream) {
    const int scale_cols = div_ceil(K, block_size);

    // Use vectorized kernel for better throughput
    // Use 256 threads per block as a safe default that works on all architectures
    const int threads_per_block = 256;
    // Compute number of blocks to saturate the GPU
    // Use at least 2x SM count to ensure good occupancy
    const int num_blocks = std::max(2 * dp.multiProcessorCount,
                                    (int)((M * (long)K + threads_per_block * 8 - 1) / (threads_per_block * 8)));

    if (block_size == 128) {
        dequantize_per_block_vec_kernel<128>
            <<<num_blocks, threads_per_block, 0, stream>>>(out, in, block_scales, M, K, scale_cols);
    } else if (block_size == 64) {
        dequantize_per_block_vec_kernel<64>
            <<<num_blocks, threads_per_block, 0, stream>>>(out, in, block_scales, M, K, scale_cols);
    } else if (block_size == 256) {
        dequantize_per_block_vec_kernel<256>
            <<<num_blocks, threads_per_block, 0, stream>>>(out, in, block_scales, M, K, scale_cols);
    } else {
        throw std::runtime_error("dequantize_per_block: unsupported block_size (must be 64, 128, or 256)");
    }
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for fused per-block to per-tensor FP8 conversion.
 *
 * Converts per-block quantized FP8 weights to per-tensor quantized FP8 in a
 * single fused operation (two kernel launches: absmax computation + requantization).
 *
 * This is faster than separate dequant→BF16→quant because:
 * - No intermediate BF16 storage
 * - Reduced memory bandwidth (read FP8 once, write FP8 once)
 * - Only 2 kernel launches instead of 3 (dequant + quant + scale computation)
 *
 * @param[out] out Output per-tensor FP8 array (M, K) - can be in-place (same as in).
 * @param[out] out_scale Output per-tensor scale (single float, device pointer).
 * @param[in] in Input per-block FP8 array (M, K).
 * @param[in] block_scales Input per-block scales.
 * @param M Number of rows.
 * @param K Number of columns.
 * @param block_size Tile size used for per-block quantization (64, 128, or 256).
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void fused_dequant_requant_per_block_to_tensor(__nv_fp8_e4m3* out,
                                               float* out_scale,
                                               const __nv_fp8_e4m3* in,
                                               const float* block_scales,
                                               int M,
                                               int K,
                                               int block_size,
                                               const cudaDeviceProp& dp,
                                               cudaStream_t stream) {
    const int scale_cols = div_ceil(K, block_size);
    const int threads_per_block = 256;
    const int num_blocks =
        std::max(2 * dp.multiProcessorCount, (int)((M * (long)K + threads_per_block - 1) / threads_per_block));

    // Initialize output scale to zero (will be updated by atomicMax)
    CUDA_CHECK(cudaMemsetAsync(out_scale, 0, sizeof(float), stream));

    // Pass 1: Compute global abs_max via atomic updates
    if (block_size == 128) {
        fused_dequant_requant_per_block_to_tensor_kernel<128>
            <<<num_blocks, threads_per_block, 0, stream>>>(out, out_scale, in, block_scales, M, K, scale_cols);
    } else if (block_size == 64) {
        fused_dequant_requant_per_block_to_tensor_kernel<64>
            <<<num_blocks, threads_per_block, 0, stream>>>(out, out_scale, in, block_scales, M, K, scale_cols);
    } else if (block_size == 256) {
        fused_dequant_requant_per_block_to_tensor_kernel<256>
            <<<num_blocks, threads_per_block, 0, stream>>>(out, out_scale, in, block_scales, M, K, scale_cols);
    } else {
        throw std::runtime_error("fused_dequant_requant: unsupported block_size (must be 64, 128, or 256)");
    }
    CUDA_CHECK(cudaGetLastError());

    // Pass 2: Requantize using the computed scale
    // We need to read the scale back to pass to the kernel
    float h_scale = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&h_scale, out_scale, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));  // Must sync to get the scale

    if (block_size == 128) {
        requantize_to_per_tensor_kernel<128>
            <<<num_blocks, threads_per_block, 0, stream>>>(out, h_scale, in, block_scales, M, K, scale_cols);
    } else if (block_size == 64) {
        requantize_to_per_tensor_kernel<64>
            <<<num_blocks, threads_per_block, 0, stream>>>(out, h_scale, in, block_scales, M, K, scale_cols);
    } else if (block_size == 256) {
        requantize_to_per_tensor_kernel<256>
            <<<num_blocks, threads_per_block, 0, stream>>>(out, h_scale, in, block_scales, M, K, scale_cols);
    }
    CUDA_CHECK(cudaGetLastError());

    // Store both absmax and dequant scale in the output buffer
    // out_scale is expected to point to [absmax, scale] where:
    // - absmax = h_scale (the computed tensor absmax)
    // - scale = h_scale / 448.0f (the dequant scale for matmul)
    //
    // This layout matches Tensor::scale() which returns Stats + 1
    float stats[2];
    stats[0] = h_scale;           // absmax
    stats[1] = h_scale / 448.0f;  // dequant scale
    CUDA_CHECK(cudaMemcpyAsync(out_scale, stats, 2 * sizeof(float), cudaMemcpyHostToDevice, stream));
}

// ============================================================================
// Tensor-based Wrapper Functions
// ============================================================================

/**
 * @brief Tensor-based wrapper for per-block quantization.
 *
 * @param[out] out Output FP8 tensor (must be pre-allocated with correct shape).
 * @param[out] block_scales Output scales tensor (must be pre-allocated).
 * @param[in] in Input BF16 tensor.
 * @param block_size Tile size for quantization.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void quantize_per_block(Tensor& out,
                        Tensor& block_scales,
                        const Tensor& in,
                        int block_size,
                        const cudaDeviceProp& dp,
                        cudaStream_t stream) {
    if (in.DType != ETensorDType::BF16) {
        throw std::runtime_error("quantize_per_block: input must be BF16");
    }
    if (out.DType != ETensorDType::FP8_E4M3) {
        throw std::runtime_error("quantize_per_block: output must be FP8_E4M3");
    }
    if (block_scales.DType != ETensorDType::FP32) {
        throw std::runtime_error("quantize_per_block: block_scales must be FP32");
    }
    if (in.Rank != 2 || out.Rank != 2) {
        throw std::runtime_error("quantize_per_block: tensors must be 2D");
    }

    const int M = in.Sizes[0];
    const int K = in.Sizes[1];

    quantize_per_block(out.get<__nv_fp8_e4m3>(),
                       block_scales.get<float>(),
                       in.get<nv_bfloat16>(),
                       M,
                       K,
                       block_size,
                       dp,
                       stream);
}

/**
 * @brief Tensor-based wrapper for per-block dequantization.
 *
 * @param[out] out Output BF16 tensor (must be pre-allocated with correct shape).
 * @param[in] in Input FP8 tensor.
 * @param[in] block_scales Scales tensor.
 * @param block_size Tile size used during quantization.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void dequantize_per_block(Tensor& out,
                          const Tensor& in,
                          const Tensor& block_scales,
                          int block_size,
                          const cudaDeviceProp& dp,
                          cudaStream_t stream) {
    if (out.DType != ETensorDType::BF16) {
        throw std::runtime_error("dequantize_per_block: output must be BF16");
    }
    if (in.DType != ETensorDType::FP8_E4M3) {
        throw std::runtime_error("dequantize_per_block: input must be FP8_E4M3");
    }
    if (block_scales.DType != ETensorDType::FP32) {
        throw std::runtime_error("dequantize_per_block: block_scales must be FP32");
    }
    if (in.Rank != 2 || out.Rank != 2) {
        throw std::runtime_error("dequantize_per_block: tensors must be 2D");
    }

    const int M = in.Sizes[0];
    const int K = in.Sizes[1];

    dequantize_per_block(out.get<nv_bfloat16>(),
                         in.get<__nv_fp8_e4m3>(),
                         block_scales.get<float>(),
                         M,
                         K,
                         block_size,
                         dp,
                         stream);
}
