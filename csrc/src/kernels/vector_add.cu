// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <stdexcept>
#include "squirrel_noise.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"

/**
 * @brief CUDA kernel for element-wise vector addition with stochastic rounding.
 *
 * Computes dest = scale * (left + right) with stochastic rounding for low-precision types.
 * Note: This calculates `scale * (x + y)`, NOT `scale * x + y` as in SAXPY.
 *
 * @tparam T Data type (float or nv_bfloat16)
 * @param[out] dest Output array to store the result
 * @param[in] left First input array
 * @param[in] right Second input array
 * @param[in] scale Scalar multiplier applied to the sum
 * @param[in] nelem Total number of elements to process
 * @param[in] seed Random seed for stochastic rounding
 */
template<typename T>
__global__ void vector_add_sr_kernel(T* dest, const T* left, const T* right, float scale, long nelem, unsigned seed) {
    using vec_t = GenericVector<T, 16/sizeof(T)>;
    long idx = (blockIdx.x * blockDim.x + threadIdx.x) * vec_t::size;
    if(idx + vec_t::size <= nelem) {
        // Vectorized path: full vector fits within bounds
        vec_t a = vec_t::load_cs(left + idx);
        vec_t b = vec_t::load_cs(right + idx);
        vec_t c = vec_t::zeros();
        for(int j = 0; j < vec_t::size; ++j) {
            float sum = scale * ((float)a[j] + (float)b[j]);
            stochastic_rounding(sum, &c[j], seed + idx + j);
        }
        c.store(dest + idx);
    } else if(idx < nelem) {
        // Scalar tail path: handle remaining elements one by one
        for(long j = idx; j < nelem; ++j) {
            float sum = scale * ((float)left[j] + (float)right[j]);
            stochastic_rounding(sum, &dest[j], seed + j);
        }
    }
}

/**
 * @brief CUDA kernel for reducing multiple shards into a single destination with stochastic rounding.
 *
 * Sums elements across multiple shards (stored contiguously in src) and optionally accumulates
 * into the existing destination values. The result is scaled and stochastically rounded.
 *
 * @tparam T Data type (float or nv_bfloat16)
 * @param[in,out] dest Output array to store the reduced result (read if accumulate is true)
 * @param[in] src Input array containing n_shards contiguous arrays, each of size nelem
 * @param[in] scale Scalar multiplier applied to the final sum
 * @param[in] n_shards Number of shards to reduce
 * @param[in] skip Index of shard to skip during reduction (-1 or out of range to include all)
 * @param[in] nelem Number of elements per shard
 * @param[in] accumulate If true, add existing dest values to the sum before storing
 * @param[in] seed Random seed for stochastic rounding
 */
template<typename T>
__global__ void vector_reduce_sr_kernel(T* dest, const T* src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed) {
    using vecx_t = GenericVector<T, 16/sizeof(T)>;
    using vecf_t = GenericVector<float, 16/sizeof(T)>;
    long idx = (blockIdx.x * blockDim.x + threadIdx.x) * vecx_t::size;

    if(idx + vecx_t::size <= nelem) {
        // Vectorized path: full vector fits within bounds
        vecf_t accumulator = vecf_t::zeros();
        if(accumulate) {
            vecx_t v = vecx_t::load_cs(dest + idx);
            for(int j = 0; j < vecx_t::size; ++j) {
                accumulator[j] += (float) v[j];
            }
        }
        for (int k = 0; k < n_shards; ++k) {
            if(k == skip) continue;
            vecx_t v = vecx_t::load_cs(src + idx + k * nelem);
            for(int j = 0; j < vecx_t::size; ++j) {
                accumulator[j] += (float) v[j];
            }
        }

        vecx_t result;
        for(int j = 0; j < vecx_t::size; ++j) {
            float sum = scale * accumulator[j];
            stochastic_rounding(sum, &result[j], seed + idx + j);
        }
        result.store(dest + idx);
    } else if(idx < nelem) {
        // Scalar tail path: handle remaining elements one by one
        for(long j = idx; j < nelem; ++j) {
            float accumulator = 0.0f;
            if(accumulate) {
                accumulator += (float)dest[j];
            }
            for (int k = 0; k < n_shards; ++k) {
                if(k == skip) continue;
                accumulator += (float)src[j + k * nelem];
            }
            float sum = scale * accumulator;
            stochastic_rounding(sum, &dest[j], seed + j);
        }
    }
}

/**
 * @brief Implementation helper for launching vector_add_sr_kernel.
 *
 * Calculates grid/block dimensions and launches the kernel on the specified stream.
 *
 * @tparam T Data type (float or nv_bfloat16)
 * @param[out] dest Output array to store the result
 * @param[in] left First input array
 * @param[in] right Second input array
 * @param[in] scale Scalar multiplier applied to the sum
 * @param[in] nelem Total number of elements to process
 * @param[in] seed Random seed for stochastic rounding
 * @param[in] stream CUDA stream for asynchronous execution
 */
template<typename T>
void vector_add_sr_imp(T* dest, const T* left, const T* right, float scale, long nelem, unsigned seed, cudaStream_t stream) {
    if (nelem == 0) return;
    if (nelem < 0) {
        char buf[256];
        snprintf(buf, sizeof(buf), "vector_add_sr_imp: negative nelem=%ld", nelem);
        throw std::runtime_error(buf);
    }
    constexpr long vec_size = 16 / sizeof(T);
    long block_size = 512;
    long grid_size = div_ceil(nelem, block_size * vec_size);
    if (grid_size <= 0) {
        char buf[256];
        snprintf(buf, sizeof(buf), "vector_add_sr_imp: grid_size=%ld is zero or negative (nelem=%ld, vec_size=%ld, block_size=%ld)",
                 grid_size, nelem, vec_size, block_size);
        throw std::runtime_error(buf);
    }
    // Ensure grid_size fits in unsigned int (CUDA requirement)
    if (grid_size > 2147483647L) {
        throw std::runtime_error("vector_add_sr_imp: grid_size too large");
    }
    vector_add_sr_kernel<T><<<(unsigned int)grid_size, (unsigned int)block_size, 0, stream>>>(dest, left, right, scale, nelem, seed);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Implementation helper for launching vector_reduce_sr_kernel.
 *
 * Calculates grid/block dimensions and launches the kernel on the specified stream.
 *
 * @tparam T Data type (float or nv_bfloat16)
 * @param[in,out] dest Output array to store the reduced result (read if accumulate is true)
 * @param[in] src Input array containing n_shards contiguous arrays
 * @param[in] scale Scalar multiplier applied to the final sum
 * @param[in] n_shards Number of shards to reduce
 * @param[in] skip Index of shard to skip during reduction
 * @param[in] nelem Number of elements per shard
 * @param[in] accumulate If true, add existing dest values to the sum
 * @param[in] seed Random seed for stochastic rounding
 * @param[in] stream CUDA stream for asynchronous execution
 */
template<typename T>
void vector_reduce_sr_imp(T* dest, const T* src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream) {
    if (nelem == 0) return;
    if (nelem < 0) {
        throw std::runtime_error("vector_reduce_sr_imp: negative nelem");
    }
    constexpr long vec_size = 16 / sizeof(T);
    long block_size = 512;
    long grid_size = div_ceil(nelem, block_size * vec_size);
    if (grid_size <= 0) {
        throw std::runtime_error("vector_reduce_sr_imp: grid_size is zero or negative");
    }
    if (grid_size > 2147483647L) {
        throw std::runtime_error("vector_reduce_sr_imp: grid_size too large");
    }
    vector_reduce_sr_kernel<T><<<(unsigned int)grid_size, (unsigned int)block_size, 0, stream>>>(dest, src, scale, n_shards, skip, nelem, accumulate, seed);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Compute element-wise vector addition with stochastic rounding (float version).
 *
 * Computes dest = scale * (left + right) on the GPU.
 *
 * @param[out] dest Output array to store the result
 * @param[in] left First input array
 * @param[in] right Second input array
 * @param[in] scale Scalar multiplier applied to the sum
 * @param[in] nelem Total number of elements to process
 * @param[in] seed Random seed for stochastic rounding
 * @param[in] stream CUDA stream for asynchronous execution
 */
void vector_add_sr(float* dest, const float* left, const float* right, float scale, long nelem, unsigned seed, cudaStream_t stream) {
    vector_add_sr_imp(dest, left, right, scale, nelem, seed, stream);
}

/**
 * @brief Compute element-wise vector addition with stochastic rounding (bfloat16 version).
 *
 * Computes dest = scale * (left + right) on the GPU with stochastic rounding
 * to handle precision loss when converting back to bfloat16.
 *
 * @param[out] dest Output array to store the result
 * @param[in] left First input array
 * @param[in] right Second input array
 * @param[in] scale Scalar multiplier applied to the sum
 * @param[in] nelem Total number of elements to process
 * @param[in] seed Random seed for stochastic rounding
 * @param[in] stream CUDA stream for asynchronous execution
 */
void vector_add_sr(nv_bfloat16* dest, const nv_bfloat16* left, const nv_bfloat16* right, float scale, long nelem, unsigned seed, cudaStream_t stream) {
    vector_add_sr_imp(dest, left, right, scale, nelem, seed, stream);
}

/**
 * @brief Reduce multiple shards into a single destination with stochastic rounding (float version).
 *
 * Sums elements across n_shards arrays (each of size nelem) stored contiguously in src.
 * Optionally skips one shard and/or accumulates into existing dest values.
 *
 * @param[in,out] dest Output array to store the reduced result (read if accumulate is true)
 * @param[in] src Input array containing n_shards contiguous arrays
 * @param[in] scale Scalar multiplier applied to the final sum
 * @param[in] n_shards Number of shards to reduce
 * @param[in] skip Index of shard to skip (-1 or out of range to include all)
 * @param[in] nelem Number of elements per shard
 * @param[in] accumulate If true, add existing dest values to the sum
 * @param[in] seed Random seed for stochastic rounding
 * @param[in] stream CUDA stream for asynchronous execution
 */
void vector_reduce_sr(float* dest, const float* src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream) {
    vector_reduce_sr_imp(dest, src, scale, n_shards, skip, nelem, accumulate, seed, stream);
}

/**
 * @brief Reduce multiple shards into a single destination with stochastic rounding (bfloat16 version).
 *
 * Sums elements across n_shards arrays (each of size nelem) stored contiguously in src.
 * Uses stochastic rounding to handle precision loss when converting back to bfloat16.
 *
 * @param[in,out] dest Output array to store the reduced result (read if accumulate is true)
 * @param[in] src Input array containing n_shards contiguous arrays
 * @param[in] scale Scalar multiplier applied to the final sum
 * @param[in] n_shards Number of shards to reduce
 * @param[in] skip Index of shard to skip (-1 or out of range to include all)
 * @param[in] nelem Number of elements per shard
 * @param[in] accumulate If true, add existing dest values to the sum
 * @param[in] seed Random seed for stochastic rounding
 * @param[in] stream CUDA stream for asynchronous execution
 */
void vector_reduce_sr(nv_bfloat16* dest, const nv_bfloat16* src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream) {
    vector_reduce_sr_imp(dest, src, scale, n_shards, skip, nelem, accumulate, seed, stream);
}

// ============================================================================
// Fused BF16 -> FP32 accumulation kernels for router LoRA
// ============================================================================

/**
 * @brief CUDA kernel for fused BF16->FP32 accumulation.
 *
 * Accumulates BF16 values into FP32 output: out[i] += (float)src[i]
 */
__global__ void fused_bf16_accum_to_fp32_kernel(float* out, const nv_bfloat16* src, std::size_t count) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] += __bfloat162float(src[idx]);
    }
}

/**
 * @brief CUDA kernel for fused BF16->FP32 strided accumulation.
 *
 * Accumulates BF16 values into strided FP32 output:
 * out[row * out_stride + col] += (float)src[row * src_cols + col]
 */
__global__ void fused_bf16_accum_to_fp32_strided_kernel(
    float* out, const nv_bfloat16* src,
    int rows, int src_cols, int out_stride) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < src_cols) {
        out[row * out_stride + col] += __bfloat162float(src[row * src_cols + col]);
    }
}

/**
 * @brief Fused BF16->FP32 accumulation for router LoRA.
 *
 * Accumulates BF16 values into FP32 output: out[i] += (float)src[i]
 *
 * @param[in,out] out FP32 output array
 * @param[in] src BF16 source array
 * @param[in] count Number of elements
 * @param[in] stream CUDA stream
 */
void fused_bf16_accum_to_fp32(float* out, const nv_bfloat16* src, std::size_t count, cudaStream_t stream) {
    if (count == 0) return;
    constexpr int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    fused_bf16_accum_to_fp32_kernel<<<grid_size, block_size, 0, stream>>>(out, src, count);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Fused BF16->FP32 strided accumulation for router LoRA.
 *
 * Accumulates BF16 values into strided FP32 output:
 * out[row * out_stride + col] += (float)src[row * src_cols + col]
 *
 * @param[in,out] out FP32 output array (strided)
 * @param[in] src BF16 source array (packed)
 * @param[in] rows Number of rows
 * @param[in] src_cols Number of columns in source (output columns too)
 * @param[in] out_stride Stride of output rows
 * @param[in] stream CUDA stream
 */
void fused_bf16_accum_to_fp32_strided(float* out, const nv_bfloat16* src,
                                       int rows, int src_cols, int out_stride, cudaStream_t stream) {
    if (rows == 0 || src_cols == 0) return;
    constexpr int block_size = 256;
    int grid_x = (src_cols + block_size - 1) / block_size;
    dim3 grid(grid_x, rows);
    fused_bf16_accum_to_fp32_strided_kernel<<<grid, block_size, 0, stream>>>(out, src, rows, src_cols, out_stride);
    CUDA_CHECK(cudaGetLastError());
}
