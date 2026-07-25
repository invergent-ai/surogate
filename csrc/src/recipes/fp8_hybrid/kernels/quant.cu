// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file quant.cu
 * @brief CUDA kernels for quantization operations (FP8, INT8, BF16).
 *
 * Provides GPU-accelerated quantization with scale computation:
 * - Absolute maximum reduction for scale factor calculation
 * - Quantization to various reduced-precision formats (BF16, INT8, FP8 E4M3/E5M2)
 * - Fused quantize-and-transpose operations for efficient memory access patterns
 *
 * Scale factors are computed as: scale = max_representable_value / abs_max(input)
 */

#include "kernels/transpose_template.cuh"
#include "kernels/kernel_utils.cuh"
#include "utilities/tensor.h"
#include "utilities/vec.cuh"
#include "utilities/utils.h"  // for fp8_max_v, fp8_interpretation_v

/**
 * @brief CUDA kernel to compute the maximum absolute value of an array.
 *
 * Uses vectorized loads and hierarchical reduction (thread → warp → block → global)
 * to efficiently find the maximum absolute value. Each thread processes multiple
 * elements via grid-stride loop with 128-bit vectorized loads.
 *
 * @tparam floatX Input data type (float or nv_bfloat16).
 * @param[out] result Global maximum absolute value (atomically updated).
 * @param[in] in Input array.
 * @param N Number of elements.
 */
template <class floatX>
__global__ void reduce_abs_max_kernel(float* __restrict__ result, const floatX* __restrict__ in, long N) {
    using vec_t = GenericVector<floatX, 16 / sizeof(floatX)>;

    __shared__ float block_abs_max;
    if (threadIdx.x == 0) {
        block_abs_max = 0.f;
    }
    __syncthreads();
    float thread_abs_max = 0.f;
    for (int i = vec_t::size * (blockIdx.x * blockDim.x + threadIdx.x); i < N;
         i += blockDim.x * gridDim.x * vec_t::size) {
        vec_t values = vec_t::load(in + i);
        for (int j = 0; j < vec_t::size; ++j) {
            thread_abs_max = fmaxf(thread_abs_max, fabsf((float)values[j]));
        }
    }

    handle_absmax_reduction(result, &block_abs_max, thread_abs_max);
}

/**
 * @brief Template launcher for absolute maximum reduction kernel.
 *
 * Launches reduce_abs_max_kernel with device-optimal block and grid sizes.
 * Initializes result to zero before kernel launch.
 *
 * @tparam floatX Input data type (float or nv_bfloat16).
 * @param[out] result Output scalar for maximum absolute value.
 * @param[in] in Input array.
 * @param N Number of elements.
 * @param dp CUDA device properties for optimal launch configuration.
 * @param stream CUDA stream for asynchronous execution.
 */
template <class floatX>
void reduce_abs_max_launcher(float* result, const floatX* in, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    int block_size = dp.maxThreadsPerMultiProcessor == 2048 ? 1024 : 768;
    int n_blocks = dp.maxThreadsPerMultiProcessor / block_size * dp.multiProcessorCount;
    CUDA_CHECK(cudaMemsetAsync(result, 0, sizeof(float), stream));
    reduce_abs_max_kernel<<<n_blocks, block_size, 0, stream>>>(result, in, N);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief CUDA kernel to quantize FP32 to BF16 (simple cast, no scaling).
 *
 * Converts FP32 values to BF16 using vectorized loads/stores. Sets scale to 1.0
 * since BF16 doesn't require per-tensor scaling for this use case.
 *
 * @param[out] out Output BF16 array.
 * @param[out] scale_ptr Scale factor output (set to 1.0), may be NULL.
 * @param[in] in Input FP32 array.
 * @param[in] abs_max Unused (kept for API consistency).
 * @param N Number of elements.
 */
__global__ void quantize_with_abs_max_kernel(nv_bfloat16* __restrict__ out,
                                             float* __restrict__ scale_ptr,
                                             const float* __restrict__ in,
                                             const float* __restrict__ abs_max,
                                             long N) {
    using vec_t = GenericVector<float, 16 / sizeof(float)>;
    using bfv_t = GenericVector<nv_bfloat16, 16 / sizeof(nv_bfloat16)>;
    if (threadIdx.x == 0 && blockIdx.x == 0 && scale_ptr) {
        *scale_ptr = 1.f;
    }

    for (int i = vec_t::size * (blockIdx.x * blockDim.x + threadIdx.x); i < N;
         i += blockDim.x * gridDim.x * vec_t::size) {
        vec_t values = vec_t::load(in + i);
        bfv_t quants;
        for (int j = 0; j < vec_t::size; ++j) {
            quants[j] = (nv_bfloat16)values[j];
        }
        quants.store(out + i);
    }
}

/**
 * @brief CUDA kernel to quantize floating-point to INT8 with scaling.
 *
 * Scales input values to [-127, 127] range based on absolute maximum,
 * then converts to int8. Uses vectorized loads for efficiency.
 *
 * @tparam floatX Input data type (float or nv_bfloat16).
 * @param[out] out Output INT8 array.
 * @param[out] scale_ptr Scale factor output (unused in this variant).
 * @param[in] in Input floating-point array.
 * @param[in] abs_max Pre-computed absolute maximum of input.
 * @param N Number of elements.
 */
template <class floatX>
__global__ void
quantize_with_abs_max_kernel(std::int8_t* out, float* scale_ptr, const floatX* in, const float* abs_max, long N) {
    using vec_t = GenericVector<floatX, 16 / sizeof(floatX)>;
    using i8v_t = GenericVector<std::int8_t, 16 / sizeof(floatX)>;
    float scale = (float)std::numeric_limits<std::int8_t>::max() / *abs_max;
    for (int i = vec_t::size * (blockIdx.x * blockDim.x + threadIdx.x); i < N;
         i += blockDim.x * gridDim.x * vec_t::size) {
        vec_t values = vec_t::load(in + i);
        i8v_t quants;
        for (int j = 0; j < vec_t::size; ++j) {
            out[i] = (std::int8_t)std::max(
                (float)std::numeric_limits<std::int8_t>::min(),
                std::min((float)std::numeric_limits<std::int8_t>::max(), scale * (float)values[j]));
        }
        quants.store(out + i);
    }
}
/**
 * @brief CUDA kernel to quantize floating-point to FP8 (E4M3 or E5M2) with scaling.
 *
 * Scales input values to FP8 range (max ~448) based on absolute maximum,
 * using saturating conversion. Stores inverse scale for dequantization.
 *
 * @tparam FloatOut Output FP8 type (__nv_fp8_e4m3 or __nv_fp8_e5m2).
 * @tparam FloatIn Input data type (float or nv_bfloat16).
 * @param[out] out Output FP8 array.
 * @param[out] scale_ptr Inverse scale factor for dequantization (1/scale), may be NULL.
 * @param[in] in Input floating-point array.
 * @param[in] abs_max Pre-computed absolute maximum of input.
 * @param N Number of elements.
 */
template <
    typename FloatOut,
    typename FloatIn,
    typename _ =
        std::enable_if_t<std::is_same_v<FloatOut, __nv_fp8_e4m3> || std::is_same_v<FloatOut, __nv_fp8_e5m2>, void>>
__global__ void quantize_with_abs_max_kernel(FloatOut* __restrict__ out,
                                             float* __restrict__ scale_ptr,
                                             const FloatIn* __restrict__ in,
                                             const float* __restrict__ abs_max,
                                             long N) {
    using vec_t = GenericVector<FloatIn, 16 / sizeof(FloatIn)>;
    using f8v_t = GenericVector<FloatOut, 16 / sizeof(FloatIn)>;
    // Use type-specific max: E4M3=448, E5M2=57344
    float scale = fp8_max_v<FloatOut> / fmaxf(*abs_max, 1e-10f);
    if (threadIdx.x == 0 && blockIdx.x == 0 && scale_ptr) {
        *scale_ptr = 1.f / scale;
    }
    for (int i = vec_t::size * (blockIdx.x * blockDim.x + threadIdx.x); i < N;
         i += blockDim.x * gridDim.x * vec_t::size) {
        vec_t values = vec_t::load(in + i);
        f8v_t quants;
        for (int j = 0; j < vec_t::size; ++j) {
            FloatOut result;
            result.__x = __nv_cvt_float_to_fp8(scale * (float)values[j],
                                               __nv_saturation_t::__NV_SATFINITE,
                                               fp8_interpretation_v<FloatOut>);
            quants[j] = result;
        }
        quants.store(out + i);
    }
}

/**
 * @brief Template launcher for quantization kernel.
 *
 * Launches quantize_with_abs_max_kernel with device-optimal configuration.
 *
 * @tparam floatY Output data type (nv_bfloat16, int8_t, FP8).
 * @tparam floatX Input data type (float or nv_bfloat16).
 * @param[out] out Output quantized array.
 * @param[out] scale_ptr Scale factor output for dequantization.
 * @param[in] in Input floating-point array.
 * @param[in] abs_max Pre-computed absolute maximum of input.
 * @param N Number of elements.
 * @param dp CUDA device properties for optimal launch configuration.
 * @param stream CUDA stream for asynchronous execution.
 */
template <class floatY, class floatX>
void quantize_with_abs_max_launcher(floatY* out,
                                    float* scale_ptr,
                                    const floatX* in,
                                    const float* abs_max,
                                    long N,
                                    const cudaDeviceProp& dp,
                                    cudaStream_t stream) {
    int block_size = dp.maxThreadsPerMultiProcessor == 2048 ? 1024 : 768;
    int n_blocks = dp.maxThreadsPerMultiProcessor / block_size * dp.multiProcessorCount;
    quantize_with_abs_max_kernel<<<n_blocks, block_size, 0, stream>>>(out, scale_ptr, in, abs_max, N);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Computes maximum absolute value of an FP32 array.
 *
 * @param[out] scale Output scalar for maximum absolute value.
 * @param[in] in Input FP32 array.
 * @param N Number of elements.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void abs_max(float* scale, const float* in, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    reduce_abs_max_launcher(scale, in, N, dp, stream);
}

/**
 * @brief Computes maximum absolute value of a BF16 array.
 *
 * @param[out] scale Output scalar for maximum absolute value (in FP32).
 * @param[in] in Input BF16 array.
 * @param N Number of elements.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void abs_max(float* scale, const nv_bfloat16* in, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    reduce_abs_max_launcher(scale, in, N, dp, stream);
}

/// @brief Quantizes FP32 array to BF16.
void quantize_with_abs_max(nv_bfloat16* out,
                           float* scale_ptr,
                           const float* in,
                           const float* abs_max,
                           long N,
                           const cudaDeviceProp& dp,
                           cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}

/// @brief Quantizes FP32 array to INT8 with scaling.
void quantize_with_abs_max(std::int8_t* out,
                           float* scale_ptr,
                           const float* in,
                           const float* abs_max,
                           long N,
                           const cudaDeviceProp& dp,
                           cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}

/// @brief Quantizes FP32 array to FP8 E4M3 with scaling.
void quantize_with_abs_max(__nv_fp8_e4m3* out,
                           float* scale_ptr,
                           const float* in,
                           const float* abs_max,
                           long N,
                           const cudaDeviceProp& dp,
                           cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}

/// @brief Quantizes FP32 array to FP8 E5M2 with scaling.
void quantize_with_abs_max(__nv_fp8_e5m2* out,
                           float* scale_ptr,
                           const float* in,
                           const float* abs_max,
                           long N,
                           const cudaDeviceProp& dp,
                           cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}

/// @brief Quantizes BF16 array to INT8 with scaling.
void quantize_with_abs_max(std::int8_t* out,
                           float* scale_ptr,
                           const nv_bfloat16* in,
                           const float* abs_max,
                           long N,
                           const cudaDeviceProp& dp,
                           cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}

/// @brief Quantizes BF16 array to FP8 E4M3 with scaling.
void quantize_with_abs_max(__nv_fp8_e4m3* out,
                           float* scale_ptr,
                           const nv_bfloat16* in,
                           const float* abs_max,
                           long N,
                           const cudaDeviceProp& dp,
                           cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}

/// @brief Quantizes BF16 array to FP8 E5M2 with scaling.
void quantize_with_abs_max(__nv_fp8_e5m2* out,
                           float* scale_ptr,
                           const nv_bfloat16* in,
                           const float* abs_max,
                           long N,
                           const cudaDeviceProp& dp,
                           cudaStream_t stream) {
    quantize_with_abs_max_launcher(out, scale_ptr, in, abs_max, N, dp, stream);
}

/**
 * @brief CUDA kernel to quantize FP32 to BF16 and transpose in a single pass.
 *
 * Fuses quantization and transpose to reduce memory bandwidth. Uses tiled
 * approach with shared memory for efficient transpose.
 *
 * @tparam BLK Tile size for transpose (elements per thread).
 * @param[out] out Output transposed BF16 array (cols x rows).
 * @param[out] scale_ptr Unused (kept for API consistency).
 * @param[in] in Input FP32 array (rows x cols).
 * @param[in] abs_max Unused (kept for API consistency).
 * @param rows Number of rows in input.
 * @param cols Number of columns in input.
 */
template <int BLK>
__global__ void quantize_and_transpose_with_abs_max_kernel(nv_bfloat16* out,
                                                           float* scale_ptr,
                                                           const float* in,
                                                           const float* abs_max,
                                                           int rows,
                                                           int cols) {
    apply_and_transpose_helper<BLK>([](auto&& a) { return (nv_bfloat16)a; }, out, in, rows, cols);
}

/**
 * @brief CUDA kernel to quantize to INT8 and transpose in a single pass.
 *
 * Fuses quantization (with scaling) and transpose. Scales input to [-127, 127]
 * range based on absolute maximum before converting to int8.
 *
 * @tparam BLK Tile size for transpose (elements per thread).
 * @tparam floatX Input data type (float or nv_bfloat16).
 * @param[out] out Output transposed INT8 array (cols x rows).
 * @param[out] scale_ptr Unused (scale embedded in conversion).
 * @param[in] in Input floating-point array (rows x cols).
 * @param[in] abs_max Pre-computed absolute maximum for scaling.
 * @param rows Number of rows in input.
 * @param cols Number of columns in input.
 */
template <int BLK, class floatX>
__global__ void quantize_and_transpose_with_abs_max_kernel(std::int8_t* out,
                                                           float* scale_ptr,
                                                           const floatX* in,
                                                           const float* abs_max,
                                                           int rows,
                                                           int cols) {
    float scale = static_cast<float>(std::numeric_limits<std::int8_t>::max()) / *abs_max;
    auto cvt = [scale](auto&& in_val) -> std::int8_t {
        auto out_val = std::max((float)std::numeric_limits<std::int8_t>::min(),
                                std::min((float)std::numeric_limits<std::int8_t>::max(), scale * (float)in_val));
        return out_val;
    };

    apply_and_transpose_helper<BLK>(cvt, out, in, rows, cols);
}

/**
 * @brief CUDA kernel to quantize to FP8 E4M3 and transpose in a single pass.
 *
 * Fuses quantization (with scaling) and transpose. Scales input to FP8 range
 * (~448 max) based on absolute maximum. Stores inverse scale for dequantization.
 *
 * @tparam BLK Tile size for transpose (elements per thread).
 * @tparam floatX Input data type (float or nv_bfloat16).
 * @param[out] out Output transposed FP8 E4M3 array (cols x rows).
 * @param[out] scale_ptr Inverse scale factor for dequantization, may be NULL.
 * @param[in] in Input floating-point array (rows x cols).
 * @param[in] abs_max Pre-computed absolute maximum for scaling.
 * @param rows Number of rows in input.
 * @param cols Number of columns in input.
 */
template <int BLK, class floatX>
__global__ void quantize_and_transpose_with_abs_max_kernel(__nv_fp8_e4m3* out,
                                                           float* scale_ptr,
                                                           const floatX* in,
                                                           const float* abs_max,
                                                           int rows,
                                                           int cols) {
    float scale = fp8_max_v<__nv_fp8_e4m3> / fmaxf(*abs_max, 1e-10f);
    if (threadIdx.x == 0 && blockIdx.x == 0 && scale_ptr) {
        *scale_ptr = 1.f / scale;
    }
    auto cvt = [scale](auto&& in_val) -> __nv_fp8_e4m3 {
        __nv_fp8_e4m3 out_val;
        out_val.__x = __nv_cvt_float_to_fp8(scale * (float)in_val,
                                            __nv_saturation_t::__NV_SATFINITE,
                                            __nv_fp8_interpretation_t::__NV_E4M3);
        return out_val;
    };

    apply_and_transpose_helper<BLK>(cvt, out, in, rows, cols);
}

/**
 * @brief CUDA kernel to quantize to FP8 E5M2 and transpose in a single pass.
 *
 * Fuses quantization (with scaling) and transpose. Scales input to FP8 E5M2 range
 * (~57344 max) based on absolute maximum. E5M2 has larger dynamic range (useful
 * for gradients) but less precision than E4M3.
 *
 * @tparam BLK Tile size for transpose (elements per thread).
 * @tparam floatX Input data type (float or nv_bfloat16).
 * @param[out] out Output transposed FP8 E5M2 array (cols x rows).
 * @param[out] scale_ptr Inverse scale factor for dequantization, may be NULL.
 * @param[in] in Input floating-point array (rows x cols).
 * @param[in] abs_max Pre-computed absolute maximum for scaling.
 * @param rows Number of rows in input.
 * @param cols Number of columns in input.
 */
template <int BLK, class floatX>
__global__ void quantize_and_transpose_with_abs_max_kernel(__nv_fp8_e5m2* out,
                                                           float* scale_ptr,
                                                           const floatX* in,
                                                           const float* abs_max,
                                                           int rows,
                                                           int cols) {
    float scale = fp8_max_v<__nv_fp8_e5m2> / fmaxf(*abs_max, 1e-10f);
    if (threadIdx.x == 0 && blockIdx.x == 0 && scale_ptr) {
        *scale_ptr = 1.f / scale;
    }
    auto cvt = [scale](auto&& in_val) -> __nv_fp8_e5m2 {
        __nv_fp8_e5m2 out_val;
        out_val.__x = __nv_cvt_float_to_fp8(scale * (float)in_val,
                                            __nv_saturation_t::__NV_SATFINITE,
                                            __nv_fp8_interpretation_t::__NV_E5M2);
        return out_val;
    };

    apply_and_transpose_helper<BLK>(cvt, out, in, rows, cols);
}

/**
 * @brief Template launcher for fused quantize-and-transpose kernel.
 *
 * Launches the appropriate quantize_and_transpose_with_abs_max_kernel with
 * 2D grid configuration optimized for the transpose operation.
 *
 * @tparam floatIn Input data type (float or nv_bfloat16).
 * @tparam floatOut Output data type (nv_bfloat16, int8_t, or FP8).
 * @param[out] out Output transposed quantized array (cols x rows).
 * @param[out] scale_ptr Scale factor output for dequantization.
 * @param[in] in Input floating-point array (rows x cols).
 * @param[in] abs_max Pre-computed absolute maximum for scaling.
 * @param rows Number of rows in input.
 * @param cols Number of columns in input.
 * @param dp CUDA device properties.
 * @param stream CUDA stream for asynchronous execution.
 */
template <class floatIn, class floatOut>
void quantize_and_transpose_with_abs_max_imp(floatOut* out,
                                             float* scale_ptr,
                                             const floatIn* in,
                                             const float* abs_max,
                                             int rows,
                                             int cols,
                                             const cudaDeviceProp& dp,
                                             cudaStream_t stream) {
    dim3 block_size = {8, 8};
    const int BLK = std::is_same_v<floatIn, float> ? 4 : 8;
    dim3 grid_size = {(unsigned)div_ceil(rows, BLK * (int)block_size.x),
                      (unsigned)div_ceil(cols, BLK * (int)block_size.y)};
    quantize_and_transpose_with_abs_max_kernel<BLK>
        <<<grid_size, block_size, 0, stream>>>(out, scale_ptr, in, abs_max, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

/// @brief Quantizes FP32 to BF16 and transposes in a single pass.
void quantize_and_transpose_with_abs_max(nv_bfloat16* out,
                                         float* scale_ptr,
                                         const float* in,
                                         const float* abs_max,
                                         int rows,
                                         int cols,
                                         const cudaDeviceProp& dp,
                                         cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, scale_ptr, in, abs_max, rows, cols, dp, stream);
}

/// @brief Quantizes FP32 to INT8 and transposes in a single pass.
void quantize_and_transpose_with_abs_max(std::int8_t* out,
                                         float* scale_ptr,
                                         const float* in,
                                         const float* abs_max,
                                         int rows,
                                         int cols,
                                         const cudaDeviceProp& dp,
                                         cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, scale_ptr, in, abs_max, rows, cols, dp, stream);
}

/// @brief Quantizes FP32 to FP8 E4M3 and transposes in a single pass.
void quantize_and_transpose_with_abs_max(__nv_fp8_e4m3* out,
                                         float* scale_ptr,
                                         const float* in,
                                         const float* abs_max,
                                         int rows,
                                         int cols,
                                         const cudaDeviceProp& dp,
                                         cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, scale_ptr, in, abs_max, rows, cols, dp, stream);
}

/// @brief Quantizes BF16 to INT8 and transposes in a single pass.
void quantize_and_transpose_with_abs_max(std::int8_t* out,
                                         float* scale_ptr,
                                         const nv_bfloat16* in,
                                         const float* abs_max,
                                         int rows,
                                         int cols,
                                         const cudaDeviceProp& dp,
                                         cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, scale_ptr, in, abs_max, rows, cols, dp, stream);
}

/// @brief Quantizes BF16 to FP8 E4M3 and transposes in a single pass.
void quantize_and_transpose_with_abs_max(__nv_fp8_e4m3* out,
                                         float* scale_ptr,
                                         const nv_bfloat16* in,
                                         const float* abs_max,
                                         int rows,
                                         int cols,
                                         const cudaDeviceProp& dp,
                                         cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, scale_ptr, in, abs_max, rows, cols, dp, stream);
}

/// @brief Quantizes FP32 to FP8 E5M2 and transposes in a single pass.
void quantize_and_transpose_with_abs_max(__nv_fp8_e5m2* out,
                                         float* scale_ptr,
                                         const float* in,
                                         const float* abs_max,
                                         int rows,
                                         int cols,
                                         const cudaDeviceProp& dp,
                                         cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, scale_ptr, in, abs_max, rows, cols, dp, stream);
}

/// @brief Quantizes BF16 to FP8 E5M2 and transposes in a single pass.
void quantize_and_transpose_with_abs_max(__nv_fp8_e5m2* out,
                                         float* scale_ptr,
                                         const nv_bfloat16* in,
                                         const float* abs_max,
                                         int rows,
                                         int cols,
                                         const cudaDeviceProp& dp,
                                         cudaStream_t stream) {
    quantize_and_transpose_with_abs_max_imp(out, scale_ptr, in, abs_max, rows, cols, dp, stream);
}

// ============================================================================
// Delayed Scaling Quantization Kernels
// ============================================================================

/**
 * @brief CUDA kernel for FP8 quantization with delayed (pre-computed) scale.
 *
 * This kernel is used in delayed scaling mode where the scale factor was computed
 * from previous iteration(s) abs_max values. It:
 * 1. Quantizes input using the provided delayed scale
 * 2. Records the current abs_max for future scale computation
 *
 * @tparam FloatOut Output FP8 type (__nv_fp8_e4m3 or __nv_fp8_e5m2).
 * @tparam FloatIn Input data type (float or nv_bfloat16).
 * @param[out] out Output FP8 array.
 * @param[out] recorded_amax Output: abs_max of input for history update.
 * @param[out] inv_scale_ptr Inverse scale (1/scale) for dequantization, may be NULL.
 * @param[in] in Input floating-point array.
 * @param[in] delayed_scale Pre-computed scale from previous iteration.
 * @param N Number of elements.
 */
template <
    typename FloatOut,
    typename FloatIn,
    typename =
        std::enable_if_t<std::is_same_v<FloatOut, __nv_fp8_e4m3> || std::is_same_v<FloatOut, __nv_fp8_e5m2>, void>>
__global__ void quantize_with_delayed_scale_kernel(FloatOut* __restrict__ out,
                                                   float* __restrict__ recorded_amax,
                                                   float* __restrict__ inv_scale_ptr,
                                                   const FloatIn* __restrict__ in,
                                                   const float* __restrict__ delayed_scale,
                                                   long N) {
    using vec_t = GenericVector<FloatIn, 16 / sizeof(FloatIn)>;
    using f8v_t = GenericVector<FloatOut, 16 / sizeof(FloatIn)>;

    __shared__ float block_abs_max;
    if (threadIdx.x == 0) {
        block_abs_max = 0.f;
    }
    __syncthreads();

    // Read the delayed scale
    const float scale = *delayed_scale;

    // Store inverse scale for dequantization
    if (threadIdx.x == 0 && blockIdx.x == 0 && inv_scale_ptr) {
        *inv_scale_ptr = 1.f / scale;
    }

    float thread_abs_max = 0.f;

    for (long i = vec_t::size * (blockIdx.x * blockDim.x + threadIdx.x); i < N;
         i += blockDim.x * gridDim.x * vec_t::size) {
        vec_t values = vec_t::load(in + i);
        f8v_t quants;

        for (int j = 0; j < vec_t::size; ++j) {
            float val = (float)values[j];
            thread_abs_max = fmaxf(thread_abs_max, fabsf(val));

            // Quantize with the delayed scale
            FloatOut result;
            result.__x =
                __nv_cvt_float_to_fp8(scale * val, __nv_saturation_t::__NV_SATFINITE, fp8_interpretation_v<FloatOut>);
            quants[j] = result;
        }
        quants.store(out + i);
    }

    // Reduce abs_max for recording
    handle_absmax_reduction(recorded_amax, &block_abs_max, thread_abs_max);
}

/**
 * @brief Template launcher for delayed scale quantization kernel.
 */
template <class floatY, class floatX>
void quantize_with_delayed_scale_launcher(floatY* out,
                                          float* recorded_amax,
                                          float* inv_scale_ptr,
                                          const floatX* in,
                                          const float* delayed_scale,
                                          long N,
                                          const cudaDeviceProp& dp,
                                          cudaStream_t stream) {
    int block_size = dp.maxThreadsPerMultiProcessor == 2048 ? 1024 : 768;
    int n_blocks = dp.maxThreadsPerMultiProcessor / block_size * dp.multiProcessorCount;
    // NOTE: recorded_amax is zeroed once at the start of each training step in forward(),
    // not here. Multiple layers share the same quantizer slot and accumulate their max
    // via atomicMax in handle_absmax_reduction().
    quantize_with_delayed_scale_kernel<<<n_blocks, block_size, 0, stream>>>(out,
                                                                            recorded_amax,
                                                                            inv_scale_ptr,
                                                                            in,
                                                                            delayed_scale,
                                                                            N);
    CUDA_CHECK(cudaGetLastError());
}

/// @brief Quantizes BF16 array to FP8 E4M3 with delayed scale, records abs_max.
void quantize_with_delayed_scale(__nv_fp8_e4m3* out,
                                 float* recorded_amax,
                                 float* inv_scale_ptr,
                                 const nv_bfloat16* in,
                                 const float* delayed_scale,
                                 long N,
                                 const cudaDeviceProp& dp,
                                 cudaStream_t stream) {
    quantize_with_delayed_scale_launcher(out, recorded_amax, inv_scale_ptr, in, delayed_scale, N, dp, stream);
}

/// @brief Quantizes BF16 array to FP8 E5M2 with delayed scale, records abs_max.
void quantize_with_delayed_scale(__nv_fp8_e5m2* out,
                                 float* recorded_amax,
                                 float* inv_scale_ptr,
                                 const nv_bfloat16* in,
                                 const float* delayed_scale,
                                 long N,
                                 const cudaDeviceProp& dp,
                                 cudaStream_t stream) {
    quantize_with_delayed_scale_launcher(out, recorded_amax, inv_scale_ptr, in, delayed_scale, N, dp, stream);
}

/// @brief Quantizes FP32 array to FP8 E4M3 with delayed scale, records abs_max.
void quantize_with_delayed_scale(__nv_fp8_e4m3* out,
                                 float* recorded_amax,
                                 float* inv_scale_ptr,
                                 const float* in,
                                 const float* delayed_scale,
                                 long N,
                                 const cudaDeviceProp& dp,
                                 cudaStream_t stream) {
    quantize_with_delayed_scale_launcher(out, recorded_amax, inv_scale_ptr, in, delayed_scale, N, dp, stream);
}

/// @brief Quantizes FP32 array to FP8 E5M2 with delayed scale, records abs_max.
void quantize_with_delayed_scale(__nv_fp8_e5m2* out,
                                 float* recorded_amax,
                                 float* inv_scale_ptr,
                                 const float* in,
                                 const float* delayed_scale,
                                 long N,
                                 const cudaDeviceProp& dp,
                                 cudaStream_t stream) {
    quantize_with_delayed_scale_launcher(out, recorded_amax, inv_scale_ptr, in, delayed_scale, N, dp, stream);
}
