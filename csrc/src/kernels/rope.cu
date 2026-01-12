// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

/**
 * @file rope.cu
 * @brief CUDA kernels for Rotary Position Embedding (RoPE).
 *
 * Implements RoPE for LLaMA-style transformers:
 * - CPU precomputation of frequency tables (cos/sin pairs)
 * - GPU forward pass: applies rotation to Q and K heads
 * - GPU backward pass: applies inverse rotation (negated sin)
 *
 * RoPE encodes position by rotating pairs of dimensions using
 * position-dependent angles: x' = x*cos(θ) - y*sin(θ), y' = x*sin(θ) + y*cos(θ)
 */

#include <cuda_bf16.h>

#include <cassert>

#include "kernel_utils.cuh"
#include "utilities/utils.h"
#include "utilities/vec.cuh"
#include "utilities/tensor.h"

/**
 * @brief CPU function to precompute RoPE frequency table (cos/sin pairs).
 *
 * Computes the rotation angles for each position and dimension pair.
 * For dimension pair i at position t: angle = t * theta^(-2i/dim)
 * Stores interleaved cos/sin values: [cos(θ_0), sin(θ_0), cos(θ_1), sin(θ_1), ...]
 *
 * @tparam floatX Output data type (float or nv_bfloat16).
 * @param[out] freqs_cis Output array of shape (end, dim) with interleaved cos/sin.
 * @param dim Head dimension (must be even).
 * @param end Maximum sequence length.
 * @param theta Base frequency (typically 10000.0 or 500000.0).
 */
template<typename floatX>
void precompute_freqs_cis_imp(floatX *freqs_cis, int dim, int end, float theta) {
    for (int i = 0; i < dim / 2; i++) {

        // calculate the frequency for the (i, i+1)th dimension
        float inv_freq = 1.0f / powf(theta, (float)(2 * i) / dim);

        // iterate over all time steps, calculate the angle, and store the cos/sin
        for (int t = 0; t < end; t++) {
            float angle = (float)t * inv_freq;
            freqs_cis[t * dim + 2 * i] = (floatX)cosf(angle);     // real part
            freqs_cis[t * dim + 2 * i + 1] = (floatX)sinf(angle); // imaginary part
        }
    }
}

/// @brief Precomputes RoPE frequency table in FP32.
void precompute_freqs_cis(float *freqs_cis, int dim, int end, float theta) {
    return precompute_freqs_cis_imp(freqs_cis, dim, end, theta);
}

/// @brief Precomputes RoPE frequency table in BF16.
void precompute_freqs_cis(nv_bfloat16 *freqs_cis, int dim, int end, float theta) {
    return precompute_freqs_cis_imp(freqs_cis, dim, end, theta);
}

/**
 * @brief CUDA kernel for RoPE forward and backward passes with partial RoPE support.
 *
 * Applies rotary position embedding to Q and K heads. V heads are passed
 * through unchanged (just copied if out != inp). Uses complex multiplication
 * interpretation: treats pairs of dimensions as (real, imag) and rotates.
 *
 * For partial RoPE (GLM4 style), only dimensions [0, rotary_dim) are rotated.
 * Dimensions [rotary_dim, head_dim) are passed through unchanged.
 *
 * Forward: (x, y) -> (x*cos - y*sin, x*sin + y*cos)
 * Backward: same but with negated sin (inverse rotation)
 *
 * @tparam Backward If true, applies inverse rotation (negated sin).
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] out Output tensor of shape (B, T, Nq+Nk+Nv, head_dim).
 * @param[in] inp Input tensor of shape (B, T, Nq+Nk+Nv, head_dim).
 * @param[in] freqs_cis Precomputed frequency table of shape (T, rotary_dim).
 * @param[in,out] abs_max_ptr Optional global absolute maximum tracker.
 * @param B Batch size.
 * @param T Sequence length.
 * @param Nq Number of query heads.
 * @param Nkv Number of key/value heads (for GQA).
 * @param head_dim Full head dimension.
 * @param rotary_dim Number of dimensions to apply RoPE to (must be <= head_dim, even).
 * @param bw Compile-time backward flag.
 */
template<bool Backward, typename floatX>
__global__ void rope_kernel(floatX *out, const floatX *inp, const floatX *freqs_cis, const int* position_ids, float* abs_max_ptr,
                            int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, std::bool_constant<Backward> bw = {}) {
    using x64 = GenericVector<floatX, 8/sizeof(floatX)>;
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    __shared__ float block_abs_max;
    if (abs_max_ptr) {
        if(threadIdx.x == 0)
            block_abs_max = 0.f;
        __syncthreads();
    }
    float thread_abs_max = 0.f;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x64::size;
    int head_dim_half = head_dim / 2;
    int N = Nq + 2*Nkv;
    if (idx >= B * T * N * head_dim_half) return;
    // decode the qkv index early so we can early exit if it's a value index
    int h = (idx / head_dim_half) % N;
    int qkv = 2;
    if(h < Nq) {
        qkv = 0;        // query head
    } else if (h < Nq + Nkv) {
        qkv = 1;        // key head
        h -= Nq;
    }

    if (qkv == 2) {
        if(abs_max_ptr) {
            x128 val = x128::load_cs(inp + 2 * idx);
            for(int k = 0; k < x128::size; k++) {
                thread_abs_max = fmaxf(thread_abs_max, fabsf(val[k]));
            }
            if (out != inp) {
                val.store(out + 2 * idx);
            }
            // can't return yet, need to participate in abs max computation
        } else {
            // if not in place, need to copy the value heads
            if (out != inp) {
                x128::load_cs(inp + 2 * idx).store(out + 2 * idx);
            }
            return;
        }
    } else {
        // decode the individual indices and get the input index
        int b = idx / (T * N * head_dim_half);
        int t = (idx / (N * head_dim_half)) % T;
        int d = idx % head_dim_half;

        int t_pos = position_ids ? position_ids[b*T + t] : t;

        int idx_bt = b * (T * N * head_dim) + t * (N * head_dim);
        int idx_bth = idx_bt + qkv * (Nq * head_dim) + h * head_dim;
        int idxi = idx_bth + d; // index in the input

        int rotary_dim_half = rotary_dim / 2;
        x64 v_real = x64::load(inp + idxi);
        x64 v_imag = x64::load(inp + idxi + head_dim_half);
        x64 o_real;
        x64 o_imag;

        // Check if this dimension block is within rotary_dim (partial RoPE support)
        if (d + x64::size <= rotary_dim_half) {
            // All dimensions in this vector are within rotary range - apply rotation
            x128 freqs_vec = x128::load_ldg(freqs_cis + t_pos * rotary_dim + 2 * d);
            for(int k = 0; k < x64::size; k++) {
                float cos = (float)freqs_vec[2*k];
                float sin = (float)freqs_vec[2*k+1];
                if constexpr (Backward) {
                    sin = -sin;
                }
                float real = (float)v_real[k];
                float imag = (float)v_imag[k];
                o_real[k] = real * cos - imag * sin;
                o_imag[k] = real * sin + imag * cos;
                if (abs_max_ptr) {
                    thread_abs_max = fmaxf(thread_abs_max, fabsf(o_real[k]));
                    thread_abs_max = fmaxf(thread_abs_max, fabsf(o_imag[k]));
                }
            }
        } else if (d >= rotary_dim_half) {
            // All dimensions in this vector are beyond rotary range - pass through
            for(int k = 0; k < x64::size; k++) {
                o_real[k] = v_real[k];
                o_imag[k] = v_imag[k];
                if (abs_max_ptr) {
                    thread_abs_max = fmaxf(thread_abs_max, fabsf((float)v_real[k]));
                    thread_abs_max = fmaxf(thread_abs_max, fabsf((float)v_imag[k]));
                }
            }
        } else {
            // Mixed: some dimensions rotated, some passed through
            // This happens at the boundary of rotary_dim
            for(int k = 0; k < x64::size; k++) {
                int dk = d + k;
                if (dk < rotary_dim_half) {
                    // Within rotary range - apply rotation
                    float cos = (float)freqs_cis[t_pos * rotary_dim + 2 * dk];
                    float sin = (float)freqs_cis[t_pos * rotary_dim + 2 * dk + 1];
                    if constexpr (Backward) {
                        sin = -sin;
                    }
                    float real = (float)v_real[k];
                    float imag = (float)v_imag[k];
                    o_real[k] = real * cos - imag * sin;
                    o_imag[k] = real * sin + imag * cos;
                } else {
                    // Beyond rotary range - pass through
                    o_real[k] = v_real[k];
                    o_imag[k] = v_imag[k];
                }
                if (abs_max_ptr) {
                    thread_abs_max = fmaxf(thread_abs_max, fabsf((float)o_real[k]));
                    thread_abs_max = fmaxf(thread_abs_max, fabsf((float)o_imag[k]));
                }
            }
        }
        o_real.store(out + idxi);
        o_imag.store(out + idxi + head_dim_half);
    }

    handle_absmax_reduction(abs_max_ptr, &block_abs_max, thread_abs_max);
}

/**
 * @brief Template launcher for RoPE kernel with partial RoPE support.
 *
 * Configures and launches rope_kernel with 128 threads per block.
 * Supports both forward and backward passes via template parameter.
 *
 * @tparam Backward If true, applies inverse rotation for backward pass.
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] out Output tensor (can be same as in for in-place).
 * @param[in] in Input tensor of shape (B, T, Nq+Nk+Nv, head_dim).
 * @param[in] freqs_cis Precomputed frequency table of shape (T, rotary_dim).
 * @param[in,out] abs_max_ptr Optional absolute maximum tracker.
 * @param B Batch size.
 * @param T Sequence length.
 * @param Nq Number of query heads.
 * @param Nkv Number of key/value heads.
 * @param head_dim Full head dimension.
 * @param rotary_dim Number of dimensions to rotate (for partial RoPE, <= head_dim).
 * @param stream CUDA stream.
 * @param bw Compile-time backward flag.
 */
template<bool Backward, class floatX>
void rope_imp(floatX* out, const floatX* in, const floatX *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream, std::bool_constant<Backward> bw = {}) {
    if (abs_max_ptr)
        CUDA_CHECK(cudaMemsetAsync(abs_max_ptr, 0, sizeof(float), stream));

    const int block_size = 128;
    using x64 = GenericVector<floatX, 8/sizeof(floatX)>;
    assert(head_dim % (2*x64::size) == 0);
    assert(rotary_dim % 2 == 0 && rotary_dim <= head_dim);
    int total_threads = (B * T * (Nq + 2*Nkv) * head_dim / 2) / x64::size;
    int num_blocks = div_ceil(total_threads, block_size);
    rope_kernel<<<num_blocks, block_size, 0, stream>>>(out, in, freqs_cis, position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, rotary_dim, bw);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Public API: Full RoPE (rotary_dim == head_dim, backwards compatible)
// ============================================================================

/// @brief RoPE forward pass for FP32 tensors (full RoPE).
void rope_forward(float* out, const float* in, const float *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    rope_imp(out, in, freqs_cis, position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, head_dim, stream, std::bool_constant<false>());
}

/// @brief RoPE forward pass for BF16 tensors (full RoPE).
void rope_forward(nv_bfloat16* out, const nv_bfloat16* in, const nv_bfloat16 *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream)  {
    rope_imp(out, in, freqs_cis, position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, head_dim, stream, std::bool_constant<false>());
}

/// @brief RoPE backward pass for FP32 tensors (full RoPE, applies inverse rotation).
void rope_backward(float* dinp, const float* dout, const float *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    rope_imp(dinp, dout, freqs_cis, position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, head_dim, stream, std::bool_constant<true>());
}

/// @brief RoPE backward pass for BF16 tensors (full RoPE, applies inverse rotation).
void rope_backward(nv_bfloat16* dinp, const nv_bfloat16* dout, const nv_bfloat16 *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream)  {
    rope_imp(dinp, dout, freqs_cis, position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, head_dim, stream, std::bool_constant<true>());
}

// ============================================================================
// Public API: Partial RoPE (rotary_dim < head_dim, for GLM4 etc.)
// ============================================================================

/// @brief RoPE forward pass for FP32 tensors with partial rotation (GLM4 style).
void rope_forward(float* out, const float* in, const float *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream) {
    rope_imp(out, in, freqs_cis, position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, rotary_dim, stream, std::bool_constant<false>());
}

/// @brief RoPE forward pass for BF16 tensors with partial rotation (GLM4 style).
void rope_forward(nv_bfloat16* out, const nv_bfloat16* in, const nv_bfloat16 *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream)  {
    rope_imp(out, in, freqs_cis, position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, rotary_dim, stream, std::bool_constant<false>());
}

/// @brief RoPE backward pass for FP32 tensors with partial rotation (GLM4 style).
void rope_backward(float* dinp, const float* dout, const float *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream) {
    rope_imp(dinp, dout, freqs_cis, position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, rotary_dim, stream, std::bool_constant<true>());
}

/// @brief RoPE backward pass for BF16 tensors with partial rotation (GLM4 style).
void rope_backward(nv_bfloat16* dinp, const nv_bfloat16* dout, const nv_bfloat16 *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream)  {
    rope_imp(dinp, dout, freqs_cis, position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, rotary_dim, stream, std::bool_constant<true>());
}

// ============================================================================
// Fused RoPE kernel with shared memory cos/sin caching
// ============================================================================
// Inspired by TransformerEngine's fused_rope implementation.
// Key optimization: compute cos/sin via sincosf() and cache in shared memory,
// so all threads in a block share the same frequencies for a given position.
// Uses 2D thread blocking: threadIdx.x iterates over head dimension,
// threadIdx.y iterates over heads.

/**
 * @brief Fused RoPE kernel with shared memory cos/sin caching.
 *
 * Each block processes one (batch, time) position. Threads are organized as:
 * - threadIdx.x: iterates over head dimension (half, since we process pairs)
 * - threadIdx.y: iterates over Q/K/V heads
 *
 * Shared memory layout: [cos_0, cos_1, ..., cos_{d/2-1}, sin_0, sin_1, ..., sin_{d/2-1}]
 *
 * @tparam Backward If true, negate sin for inverse rotation.
 * @tparam floatX Data type (float or nv_bfloat16).
 */
template<bool Backward, typename floatX>
__global__ void rope_fused_kernel(floatX* __restrict__ out, const floatX* __restrict__ inp,
                                  const int* __restrict__ position_ids, float* __restrict__ abs_max_ptr,
                                  float theta, int B, int T, int Nq, int Nkv, int head_dim) {
    extern __shared__ float shared_cos_sin[];
    float* shared_cos = shared_cos_sin;
    float* shared_sin = shared_cos_sin + head_dim / 2;

    const int b = blockIdx.y;
    const int t = blockIdx.x;
    const int N = Nq + 2 * Nkv;  // Total heads (Q + K + V)
    const int head_dim_half = head_dim / 2;

    // Get actual position for RoPE (allows for position offsets in inference)
    const int pos = position_ids ? position_ids[b * T + t] : t;

    // Cooperatively load cos/sin into shared memory
    // Each thread computes one or more (cos, sin) pairs
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int num_threads = blockDim.x * blockDim.y;
    for (int d = tid; d < head_dim_half; d += num_threads) {
        float inv_freq = 1.0f / powf(theta, (float)(2 * d) / head_dim);
        float angle = (float)pos * inv_freq;
        sincosf(angle, &shared_sin[d], &shared_cos[d]);
    }
    __syncthreads();

    // Process Q and K heads (apply rotation), V heads (copy through)
    float thread_abs_max = 0.0f;

    // Base offset for this (batch, time) position
    const size_t bt_offset = ((size_t)b * T + t) * N * head_dim;

    for (int h = threadIdx.y; h < N; h += blockDim.y) {
        // Determine if this is Q, K, or V
        int qkv = 2;  // V by default
        if (h < Nq) {
            qkv = 0;  // Q
        } else if (h < Nq + Nkv) {
            qkv = 1;  // K
        }

        const size_t head_offset = bt_offset + h * head_dim;

        for (int d = threadIdx.x; d < head_dim_half; d += blockDim.x) {
            float real_in = (float)inp[head_offset + d];
            float imag_in = (float)inp[head_offset + d + head_dim_half];

            float real_out, imag_out;
            if (qkv < 2) {
                // Q or K: apply rotation
                float c = shared_cos[d];
                float s = shared_sin[d];
                if constexpr (Backward) {
                    s = -s;  // Inverse rotation
                }
                real_out = real_in * c - imag_in * s;
                imag_out = real_in * s + imag_in * c;
            } else {
                // V: pass through
                real_out = real_in;
                imag_out = imag_in;
            }

            out[head_offset + d] = (floatX)real_out;
            out[head_offset + d + head_dim_half] = (floatX)imag_out;

            if (abs_max_ptr) {
                thread_abs_max = fmaxf(thread_abs_max, fabsf(real_out));
                thread_abs_max = fmaxf(thread_abs_max, fabsf(imag_out));
            }
        }
    }

    // Reduce abs_max across block and update global
    if (abs_max_ptr) {
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_abs_max = fmaxf(thread_abs_max, __shfl_xor_sync(0xFFFFFFFF, thread_abs_max, offset));
        }

        // Block-level reduction via shared memory (reuse cos_sin buffer)
        __shared__ float block_max;
        if (tid == 0) block_max = 0.0f;
        __syncthreads();

        if (threadIdx.x % 32 == 0) {
            atomicMax(reinterpret_cast<unsigned*>(&block_max), __float_as_uint(thread_abs_max));
        }
        __syncthreads();

        // Thread 0 updates global max
        if (tid == 0) {
            atomicMax(reinterpret_cast<unsigned*>(abs_max_ptr), __float_as_uint(block_max));
        }
    }
}

/**
 * @brief Launcher for fused RoPE kernel.
 *
 * Chooses thread block dimensions based on head configuration.
 * Uses 2D grid: (T, B) and 2D blocks: (dim_threads, head_threads).
 */
template<bool Backward, typename floatX>
void rope_fused_imp(floatX* out, const floatX* inp, const int* position_ids, float* abs_max_ptr,
                    float theta, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    if (abs_max_ptr) {
        CUDA_CHECK(cudaMemsetAsync(abs_max_ptr, 0, sizeof(float), stream));
    }

    const int N = Nq + 2 * Nkv;
    const int head_dim_half = head_dim / 2;

    // Thread block configuration:
    // - threads_x: process head dimension (up to 32 for warp efficiency)
    // - threads_y: process multiple heads in parallel
    int threads_x = std::min(32, head_dim_half);
    int threads_y = std::min(8, N);  // Up to 8 warps per block

    // Ensure we have enough threads to cover head_dim_half for shared mem loading
    while (threads_x * threads_y < head_dim_half && threads_y < 32) {
        threads_y++;
    }

    dim3 block(threads_x, threads_y);
    dim3 grid(T, B);

    // Shared memory: cos + sin arrays, each of size head_dim/2 floats
    size_t smem_size = head_dim * sizeof(float);

    rope_fused_kernel<Backward><<<grid, block, smem_size, stream>>>(
        out, inp, position_ids, abs_max_ptr, theta, B, T, Nq, Nkv, head_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

/// @brief Fused RoPE forward for FP32 (computes cos/sin on-the-fly with shared memory caching).
void rope_fused_forward(float* out, const float* inp, const int* position_ids, float* abs_max_ptr,
                        float theta, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    rope_fused_imp<false>(out, inp, position_ids, abs_max_ptr, theta, B, T, Nq, Nkv, head_dim, stream);
}

/// @brief Fused RoPE forward for BF16 (computes cos/sin on-the-fly with shared memory caching).
void rope_fused_forward(nv_bfloat16* out, const nv_bfloat16* inp, const int* position_ids, float* abs_max_ptr,
                        float theta, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    rope_fused_imp<false>(out, inp, position_ids, abs_max_ptr, theta, B, T, Nq, Nkv, head_dim, stream);
}

/// @brief Fused RoPE backward for FP32 (inverse rotation with shared memory caching).
void rope_fused_backward(float* dinp, const float* dout, const int* position_ids, float* abs_max_ptr,
                         float theta, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    rope_fused_imp<true>(dinp, dout, position_ids, abs_max_ptr, theta, B, T, Nq, Nkv, head_dim, stream);
}

/// @brief Fused RoPE backward for BF16 (inverse rotation with shared memory caching).
void rope_fused_backward(nv_bfloat16* dinp, const nv_bfloat16* dout, const int* position_ids, float* abs_max_ptr,
                         float theta, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    rope_fused_imp<true>(dinp, dout, position_ids, abs_max_ptr, theta, B, T, Nq, Nkv, head_dim, stream);
}

/**
 * @brief CUDA kernel for RoPE forward pass with fused FP8 quantization.
 *
 * Applies rotary position embedding and quantizes output to FP8 E4M3 in a single pass.
 * Uses pre-computed abs-max from forward pass for scaling, eliminating redundant
 * quantization during recomputation.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] out Quantized FP8 output tensor.
 * @param[out] scale_ptr Inverse scale factor for dequantization.
 * @param[in] inp Input tensor of shape (B, T, Nq+Nk+Nv, head_dim).
 * @param[in] freqs_cis Precomputed frequency table.
 * @param[in] position_ids Position IDs for each token.
 * @param[in] abs_max_ptr Pre-computed absolute maximum for quantization.
 * @param B Batch size.
 * @param T Sequence length.
 * @param Nq Number of query heads.
 * @param Nkv Number of key/value heads (for GQA).
 * @param head_dim Head dimension.
 */
template<typename floatX>
__global__ void rope_forward_quant_kernel(__nv_fp8_e4m3* out, float* scale_ptr,
                                          const floatX* inp, const floatX* freqs_cis,
                                          const int* position_ids, const float* abs_max_ptr,
                                          int B, int T, int Nq, int Nkv, int head_dim) {
    using x64 = GenericVector<floatX, 8/sizeof(floatX)>;
    using f8v_t = GenericVector<__nv_fp8_e4m3, 8 / sizeof(floatX)>;

    // Compute quantization scale
    float scale = 448.f / fmaxf(*abs_max_ptr, 1e-10f);
    if(threadIdx.x == 0 && blockIdx.x == 0 && scale_ptr) {
        *scale_ptr = 1.f / scale;
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_dim_half = head_dim / 2;

    // Number of heads that get rotated (Q and K, but not V)
    int Nh_rope = Nq + 2*Nkv;
    int total_threads = (B * T * Nh_rope * head_dim_half) / x64::size;
    if(idx >= total_threads) return;

    // Decode thread coordinates
    int h_idx = idx / (head_dim_half / x64::size);
    int d = (idx % (head_dim_half / x64::size)) * x64::size;
    int t = h_idx / Nh_rope;
    int h = h_idx % Nh_rope;
    int b = t / T;
    t = t % T;

    // Get actual position for this token
    int pos = position_ids ? position_ids[b * T + t] : t;

    // Input/output offsets
    int idxi = ((b * T + t) * (Nq + 2*Nkv) * head_dim) + (h * head_dim + d);

    // Load input pairs (real, imaginary)
    x64 v_real = x64::load_cs(inp + idxi);
    x64 v_imag = x64::load_cs(inp + idxi + head_dim_half);

    // Load frequency pairs
    const floatX* freqs_ptr = freqs_cis + pos * head_dim + 2 * d;
    x64 freqs_real, freqs_imag;
    for(int k = 0; k < x64::size; k++) {
        freqs_real[k] = freqs_ptr[2*k];
        freqs_imag[k] = freqs_ptr[2*k+1];
    }

    // Rotate and quantize
    f8v_t o_real, o_imag;
    for(int k = 0; k < x64::size; k++) {
        float cos_val = (float)freqs_real[k];
        float sin_val = (float)freqs_imag[k];
        float real = (float)v_real[k];
        float imag = (float)v_imag[k];
        
        float rotated_real = real * cos_val - imag * sin_val;
        float rotated_imag = real * sin_val + imag * cos_val;
        
        // Quantize
        __nv_fp8_e4m3 q_real, q_imag;
        q_real.__x = __nv_cvt_float_to_fp8(scale * rotated_real, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
        q_imag.__x = __nv_cvt_float_to_fp8(scale * rotated_imag, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
        o_real[k] = q_real;
        o_imag[k] = q_imag;
    }

    // Store quantized output
    o_real.store(out + idxi);
    o_imag.store(out + idxi + head_dim_half);
}

/**
 * @brief RoPE forward pass with fused FP8 quantization.
 *
 * Applies rotary position embedding and quantizes to FP8 in a single kernel,
 * using pre-computed abs-max to eliminate redundant quantization overhead.
 *
 * @param[out] out Quantized FP8 output tensor.
 * @param[out] scale_ptr Inverse scale factor for dequantization.
 * @param[in] inp Input BF16 tensor.
 * @param[in] freqs_cis Precomputed frequency table.
 * @param[in] position_ids Position IDs for each token.
 * @param[in] abs_max_ptr Pre-computed absolute maximum.
 * @param B Batch size.
 * @param T Sequence length.
 * @param Nq Number of query heads.
 * @param Nkv Number of key/value heads.
 * @param head_dim Head dimension.
 * @param stream CUDA stream.
 */
void rope_forward_quant(__nv_fp8_e4m3* out, float* scale_ptr,
                        const nv_bfloat16* inp, const nv_bfloat16* freqs_cis,
                        const int* position_ids, const float* abs_max_ptr,
                        int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    const int block_size = 128;
    using x64 = GenericVector<nv_bfloat16, 8/sizeof(nv_bfloat16)>;
    assert(head_dim % (2*x64::size) == 0);
    
    int Nh_rope = Nq + 2*Nkv;
    int total_threads = (B * T * Nh_rope * head_dim / 2) / x64::size;
    int num_blocks = div_ceil(total_threads, block_size);
    
    rope_forward_quant_kernel<<<num_blocks, block_size, 0, stream>>>(
        out, scale_ptr, inp, freqs_cis, position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Tensor wrapper for rope_forward_quant.
 *
 * Type-safe wrapper that validates tensor dtypes and dispatches to implementation.
 */
void rope_forward_quant(Tensor& out, float* scale_ptr,
                        const Tensor& inp, const Tensor& freqs_cis,
                        const int* position_ids, const float* abs_max_ptr,
                        int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    if (inp.DType != ETensorDType::BF16) {
        throw std::runtime_error("rope_forward_quant: only BF16 input supported");
    }
    if (out.DType != ETensorDType::FP8_E4M3) {
        throw std::runtime_error("rope_forward_quant: output must be FP8_E4M3");
    }
    rope_forward_quant(out.get<__nv_fp8_e4m3>(), scale_ptr,
                       inp.get<nv_bfloat16>(), freqs_cis.get<nv_bfloat16>(),
                       position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, stream);
}
