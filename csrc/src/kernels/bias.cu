// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

/**
 * @file bias.cu
 * @brief CUDA kernels for bias addition and gradient computation.
 *
 * Provides forward bias addition and backward bias gradient accumulation
 * with support for multiple data types (FP32, BF16, FP8).
 */

#include "utilities/vec.cuh"
#include "utilities/utils.h"
#include "utilities/dtype.h"

#include <type_traits>
#include <cassert>

/**
 * @brief CUDA kernel for adding bias to output tensor.
 *
 * Adds a 1D bias vector to each row of a 3D output tensor using grid-stride loop.
 * Each thread processes multiple elements for coalesced memory access.
 *
 * @tparam floatO Output tensor data type.
 * @tparam floatB Bias tensor data type.
 * @param[in,out] out Output tensor of shape (B, T, OC), modified in-place.
 * @param[in] bias Bias vector of shape (OC,).
 * @param B Batch size.
 * @param T Sequence length.
 * @param OC Output channels (bias dimension).
 */
template <class floatO, class floatB>
__global__ void add_bias_kernel(floatO* out, const floatB* bias, int B, int T, int OC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < B * T * OC; i += stride) {
        int col = i % OC;
        out[i] += bias[col];
    }
}

/**
 * @brief Template implementation for launching bias addition kernel.
 *
 * Launches the add_bias_kernel with appropriate grid dimensions.
 *
 * @tparam floatO Output tensor data type.
 * @tparam floatB Bias tensor data type.
 * @param[in,out] out Output tensor of shape (B, T, OC), modified in-place.
 * @param[in] bias Bias vector of shape (OC,).
 * @param B Batch size.
 * @param T Sequence length.
 * @param OC Output channels.
 * @param stream CUDA stream for asynchronous execution.
 */
template <class floatO, class floatB>
void add_bias_impl(floatO* out, const floatB* bias, int B, int T, int OC, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = div_ceil(OC * B * T, block_size);
    add_bias_kernel<<<grid_size, block_size, 0, stream>>>(out, bias, B, T, OC);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Adds bias to FP32 output tensor.
 *
 * @param[in,out] out Output tensor of shape (B, T, OC) in FP32, modified in-place.
 * @param[in] bias Bias vector of shape (OC,) in FP32.
 * @param B Batch size.
 * @param T Sequence length.
 * @param OC Output channels.
 * @param stream CUDA stream for asynchronous execution.
 */
void add_bias(float* out, const float* bias, int B, int T, int OC, cudaStream_t stream) {
    add_bias_impl(out, bias, B, T, OC, stream);
}

/**
 * @brief Adds bias to BF16 output tensor.
 *
 * @param[in,out] out Output tensor of shape (B, T, OC) in BF16, modified in-place.
 * @param[in] bias Bias vector of shape (OC,) in BF16.
 * @param B Batch size.
 * @param T Sequence length.
 * @param OC Output channels.
 * @param stream CUDA stream for asynchronous execution.
 */
void add_bias(nv_bfloat16* out, const nv_bfloat16* bias, int B, int T, int OC, cudaStream_t stream) {
    add_bias_impl(out, bias, B, T, OC, stream);
}

/**
 * @brief CUDA kernel for computing bias gradients from upstream gradients.
 *
 * Computes dbias = sum(dout, axis=(0,1)) with optional FP8 scale factors.
 * Uses vectorized loads (128-bit), warp shuffles for intra-warp reduction,
 * and shared memory for cross-warp reduction within blocks.
 *
 * When UseAuxBuffer is true, writes partial sums to an auxiliary buffer for
 * later reduction. When false, accumulates directly into dbias.
 *
 * Block organization: 4x8xN threads where N depends on GPU architecture.
 * Each warp processes 64 output channels at BF16 (8 warps * 8 elements each).
 *
 * @tparam floatX Upstream gradient data type.
 * @tparam OutFloat Output bias gradient data type.
 * @tparam UseAuxBuffer If true, write to auxiliary buffer; if false, accumulate to dbias.
 * @param[out] dbias Output bias gradient of shape (OC,) or (grid_y, OC) if UseAuxBuffer.
 * @param[in] dout Upstream gradient tensor of shape (B, T, OC).
 * @param[in] scale_a Optional FP8 scale factor A (nullptr for non-FP8).
 * @param[in] scale_b Optional FP8 scale factor B (nullptr for non-FP8).
 * @param B Batch size.
 * @param T Sequence length.
 * @param OC Output channels.
 */
template <typename floatX, typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel(OutFloat* dbias,
                                            const floatX* dout,
                                            const float* scale_a,
                                            const float* scale_b,
                                            int B,
                                            int T,
                                            int OC,
                                            std::bool_constant<UseAuxBuffer>) {
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    using f128 = GenericVector<float, 16 / sizeof(float)>;
    constexpr const int bdx = 4;
    constexpr const int bdy = 32 / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    int warp_d = (int)threadIdx.x;
    int warp_c = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    float scale = 1.f;
    if (scale_a != nullptr && scale_b != nullptr) {
        scale = *scale_a * *scale_b;
    }

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if (global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T; idx += gridDim.y * bt_per_block) {
            x128 packed_dout = x128::load(dout + global_oc + idx * OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    __shared__ float sub_results[x128::size][32][bdy];

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if (warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    __syncthreads();

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += blockDim.z) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim.z; r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += __shfl_down_sync(0xffffffff, v, 1, 4);
            v += __shfl_down_sync(0xffffffff, v, 2, 4);
            a += v;
        }
        if (warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a * scale + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx.y * OC] = a * scale;
            }
        }
    }
}

/**
 * @brief CUDA kernel for reducing partial sums from auxiliary buffer to final output.
 *
 * Sums m partial results (each of size n) from the source buffer and accumulates
 * into the destination. Used as the second pass when bias backward requires
 * cross-block reduction.
 *
 * @tparam floatX Output data type.
 * @param[in,out] dst Destination tensor of shape (n,), accumulated in-place.
 * @param[in] src Source tensor of shape (m, n) containing partial sums.
 * @param n Number of output channels.
 * @param m Number of partial results to sum.
 */
template <class floatX>
__global__ void reduce_add_sum_kernel(floatX* dst, const float* src, size_t n, size_t m) {
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    using f128 = GenericVector<float, 16 / sizeof(float)>;
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
    assert(n % x128::size == 0);
    if (idx < n) {
        f128 acc;
        for (int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        for (int l = 0; l < m; ++l) {
            f128 s = f128::load(src + idx + n * l);
            for (int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        for (int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX)((float)dst[idx + k] + acc[k]);
        }
    }
}

/**
 * @brief Calculates required scratch buffer size for bias backward pass.
 *
 * Returns the size (in bytes) of the auxiliary buffer needed for cross-block
 * reduction when computing bias gradients. The size depends on the data type,
 * output channels, and GPU architecture.
 *
 * @param dtype Data type of the gradient tensor.
 * @param OC Number of output channels.
 * @param dp CUDA device properties.
 * @return Required scratch buffer size in bytes.
 */
int get_bias_backward_scratch_size(ETensorDType dtype, int OC, const cudaDeviceProp& dp) {
    const int block_size = dp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;
    const int OC_per_warp = 8 * (16 / get_dtype_size(dtype));  // 64 at BF16
    const int grid_size_x = div_ceil(OC, OC_per_warp);         // e.g. 12 horizontal blocks for 768 OCs at BF16
    const int grid_size_y = std::max(1, block_size * dp.multiProcessorCount / (block_size * grid_size_x));  // full GPU!
    return grid_size_y * OC * sizeof(float);
}

/**
 * @brief Template implementation for bias backward pass.
 *
 * Computes bias gradients by summing upstream gradients across batch and sequence
 * dimensions. Uses a two-pass algorithm when full GPU utilization requires more
 * blocks than output channels warrant:
 * 1. Partial sums written to auxiliary buffer
 * 2. Final reduction from auxiliary buffer to output
 *
 * For smaller workloads, writes directly to output in a single pass.
 *
 * @tparam floatX Output bias gradient data type.
 * @tparam FloatY Upstream gradient data type.
 * @param[out] dbias Output bias gradient of shape (OC,).
 * @param[in] dout Upstream gradient tensor of shape (B, T, OC).
 * @param[in] scale_a Optional FP8 scale factor A (nullptr for non-FP8).
 * @param[in] scale_b Optional FP8 scale factor B (nullptr for non-FP8).
 * @param dbias_buffer Scratch buffer for partial sums (size from get_bias_backward_scratch_size).
 * @param B Batch size.
 * @param T Sequence length.
 * @param OC Output channels.
 * @param dp CUDA device properties for kernel configuration.
 * @param stream CUDA stream for asynchronous execution.
 * @throws std::logic_error If scale_a and scale_b are inconsistently null.
 */
template <class floatX, class FloatY>
void backward_bias_imp(floatX* dbias,
                       const FloatY* dout,
                       const float* scale_a,
                       const float* scale_b,
                       float* dbias_buffer,
                       int B,
                       int T,
                       int OC,
                       const cudaDeviceProp& dp,
                       cudaStream_t stream) {
    // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
    // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end
    using x128 = GenericVector<floatX, 16 / sizeof(floatX)>;
    using f128 = GenericVector<float, 16 / sizeof(float)>;

    const int block_size = dp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

    dim3 block_dim = {4, 8, (unsigned)block_size / 32};
    const int OC_per_warp = block_dim.y * x128::size;   // 64 at BF16
    const int grid_size_x = div_ceil(OC, OC_per_warp);  // e.g. 12 horizontal blocks for 768 OCs at BF16
    const int grid_size_y = std::max(1, block_size * dp.multiProcessorCount / (block_size * grid_size_x));  // full GPU!

    if ((scale_a == nullptr) != (scale_b == nullptr)) {
        throw std::logic_error("backward_bias: scale_a and scale_b must be both nullptr or both non-nullptr");
    }

    // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
    // and write results directly to the output.
    if (grid_size_y == 1) {
        matmul_backward_bias_kernel<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(
            dbias,
            dout,
            scale_a,
            scale_b,
            B,
            T,
            OC,
            std::bool_constant<false>());
        CUDA_CHECK(cudaGetLastError());
    } else {
        // kernel 9 overwrites temp buffer, so no need to memset
        matmul_backward_bias_kernel<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(
            dbias_buffer,
            dout,
            scale_a,
            scale_b,
            B,
            T,
            OC,
            std::bool_constant<true>());
        CUDA_CHECK(cudaGetLastError());
        reduce_add_sum_kernel<<<div_ceil((size_t)OC, 256 * f128::size), 256, 0, stream>>>(dbias,
                                                                                          dbias_buffer,
                                                                                          OC,
                                                                                          grid_size_y);
        CUDA_CHECK(cudaGetLastError());
    }
}

/**
 * @brief Computes bias gradients for FP32 tensors.
 *
 * @param[out] dbias Output bias gradient of shape (OC,) in FP32.
 * @param[in] dout Upstream gradient tensor of shape (B, T, OC) in FP32.
 * @param[in] scale_a Unused for FP32 (pass nullptr).
 * @param[in] scale_b Unused for FP32 (pass nullptr).
 * @param dbias_buffer Scratch buffer for partial sums.
 * @param B Batch size.
 * @param T Sequence length.
 * @param OC Output channels.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void backward_bias(float* dbias,
                   const float* dout,
                   const float* scale_a,
                   const float* scale_b,
                   float* dbias_buffer,
                   int B,
                   int T,
                   int OC,
                   const cudaDeviceProp& dp,
                   cudaStream_t stream) {
    backward_bias_imp(dbias, dout, scale_a, scale_b, dbias_buffer, B, T, OC, dp, stream);
}

/**
 * @brief Computes bias gradients for BF16 tensors.
 *
 * @param[out] dbias Output bias gradient of shape (OC,) in BF16.
 * @param[in] dout Upstream gradient tensor of shape (B, T, OC) in BF16.
 * @param[in] scale_a Unused for BF16 (pass nullptr).
 * @param[in] scale_b Unused for BF16 (pass nullptr).
 * @param dbias_buffer Scratch buffer for partial sums.
 * @param B Batch size.
 * @param T Sequence length.
 * @param OC Output channels.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void backward_bias(nv_bfloat16* dbias,
                   const nv_bfloat16* dout,
                   const float* scale_a,
                   const float* scale_b,
                   float* dbias_buffer,
                   int B,
                   int T,
                   int OC,
                   const cudaDeviceProp& dp,
                   cudaStream_t stream) {
    backward_bias_imp(dbias, dout, scale_a, scale_b, dbias_buffer, B, T, OC, dp, stream);
}

/**
 * @brief Computes bias gradients for FP8 E4M3 upstream gradients.
 *
 * @param[out] dbias Output bias gradient of shape (OC,) in BF16.
 * @param[in] dout Upstream gradient tensor of shape (B, T, OC) in FP8 E4M3.
 * @param[in] scale_a FP8 scale factor A (required for dequantization).
 * @param[in] scale_b FP8 scale factor B (required for dequantization).
 * @param dbias_buffer Scratch buffer for partial sums.
 * @param B Batch size.
 * @param T Sequence length.
 * @param OC Output channels.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void backward_bias(nv_bfloat16* dbias,
                   const __nv_fp8_e4m3* dout,
                   const float* scale_a,
                   const float* scale_b,
                   float* dbias_buffer,
                   int B,
                   int T,
                   int OC,
                   const cudaDeviceProp& dp,
                   cudaStream_t stream) {
    backward_bias_imp(dbias, dout, scale_a, scale_b, dbias_buffer, B, T, OC, dp, stream);
}

/**
 * @brief Computes bias gradients for FP8 E5M2 upstream gradients.
 *
 * @param[out] dbias Output bias gradient of shape (OC,) in BF16.
 * @param[in] dout Upstream gradient tensor of shape (B, T, OC) in FP8 E5M2.
 * @param[in] scale_a FP8 scale factor A (required for dequantization).
 * @param[in] scale_b FP8 scale factor B (required for dequantization).
 * @param dbias_buffer Scratch buffer for partial sums.
 * @param B Batch size.
 * @param T Sequence length.
 * @param OC Output channels.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void backward_bias(nv_bfloat16* dbias,
                   const __nv_fp8_e5m2* dout,
                   const float* scale_a,
                   const float* scale_b,
                   float* dbias_buffer,
                   int B,
                   int T,
                   int OC,
                   const cudaDeviceProp& dp,
                   cudaStream_t stream) {
    backward_bias_imp(dbias, dout, scale_a, scale_b, dbias_buffer, B, T, OC, dp, stream);
}
