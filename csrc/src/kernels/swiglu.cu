// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

/**
 * @file swiglu.cu
 * @brief CUDA kernels for SwiGLU activation function.
 *
 * Implements the SwiGLU (Swish-Gated Linear Unit) activation used in LLaMA:
 * - Forward: out = up * gate * sigmoid(gate) = up * silu(gate)
 * - Backward: computes gradients for both up and gate inputs
 *
 * Input is concatenated [up, gate] of shape (B, T, 2*C), output is (B, T, C).
 * Provides both simple and persistent kernel variants, with optional FP8 quantization.
 */

#include <cassert>

#include <cuda_bf16.h>
#include <cuda_pipeline_primitives.h>

#include "kernels.h"
#include "utilities/utils.h"
#include "utilities/vec.cuh"
#include "utilities/strided_iter.cuh"
#include "kernel_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

/**
 * @brief Basic CUDA kernel for SwiGLU forward pass.
 *
 * Computes: out = up * gate * sigmoid(gate) where inp = [up, gate] concatenated.
 * Each thread processes one 128-bit vector. Simple kernel suitable for smaller tensors.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] out Output tensor of shape (B*T, C).
 * @param[in] inp Input tensor of shape (B*T, 2*C) with [up, gate] concatenated.
 * @param[in,out] abs_max_ptr Optional global absolute maximum tracker.
 * @param C Hidden dimension (output width).
 */
template<typename floatX>
__global__ void swiglu_forward_kernel(floatX* out, const floatX* inp, float* abs_max_ptr, int C, int total) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    // thread coordinates
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    // Shared memory init must happen before bounds-check to keep all threads in sync
    __shared__ float block_max;
    if (abs_max_ptr) {
        if(threadIdx.x == 0) {
            block_max = 0.f;
        }
        __syncthreads();
    }
    float thread_max = 0.f;

    // Bounds check: with MoE+EP, total may not be aligned to block_size * x128::size
    if (idx < total) {
        floatX* out_ptr = out + idx;
        int bt = (idx / C);
        int c = idx % C;

        const floatX* up_ptr = inp + (bt * C * 2 + c);
        const floatX* gate_ptr = up_ptr + C;

        x128 packed_out;
        x128 up_inp = x128::load_cs(up_ptr);
        x128 gate_inp = x128::load_cs(gate_ptr);
        for(int k = 0; k < up_inp.size; ++k) {
            float x1 = (float)up_inp[k];
            float x2 = (float)gate_inp[k];
            packed_out[k] = (floatX)((x1 * x2) / (1.0f + expf(-x2)));
            if (abs_max_ptr) {
                thread_max = fmaxf(thread_max, fabsf(packed_out[k]));
            }
        }
        packed_out.store(out_ptr);
    }

    handle_absmax_reduction(abs_max_ptr, &block_max, thread_max);
}

/**
 * @brief CUDA kernel for SwiGLU forward with FP8 quantized output.
 *
 * Computes SwiGLU and quantizes output to FP8 E4M3 format using pre-computed
 * absolute maximum for scaling. Simple kernel variant for smaller tensors.
 *
 * @tparam floatX Input data type (float or nv_bfloat16).
 * @param[out] out Output tensor in FP8 E4M3 format.
 * @param[out] scale_ptr Inverse scale factor for dequantization.
 * @param[in] inp Input tensor of shape (B*T, 2*C).
 * @param[in] abs_max_ptr Pre-computed absolute maximum for scaling.
 * @param C Hidden dimension (output width).
 */
template<typename floatX>
__global__ void swiglu_forward_quant_kernel(__nv_fp8_e4m3* out, float* scale_ptr, const floatX* inp, const float* abs_max_ptr, int C) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using f8v_t = GenericVector<__nv_fp8_e4m3, 16 / sizeof(floatX)>;

    float scale = 448.f / fmaxf(*abs_max_ptr, 1e-10f);
    if(threadIdx.x == 0 && blockIdx.x == 0 && scale_ptr) {
        *scale_ptr = 1.f / scale;
    }

    // thread coordinates
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    __nv_fp8_e4m3* out_ptr = out + idx;
    int bt = (idx / C);
    int c = idx % C;

    const floatX* up_ptr = inp + (bt * C * 2 + c);
    const floatX* gate_ptr = up_ptr + C;

    f8v_t packed_out;
    x128 up_inp = x128::load_cs(up_ptr);
    x128 gate_inp = x128::load_cs(gate_ptr);
    for(int k = 0; k < up_inp.size; ++k) {
        float x1 = (float)up_inp[k];
        float x2 = (float)gate_inp[k];
        float result = (x1 * x2) / (1.0f + expf(-x2));
        floatX qr = (floatX)result;
        __nv_fp8_e4m3 quant;
        quant.__x = __nv_cvt_float_to_fp8(scale * (float)qr, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
        packed_out[k] = quant;
    }
    packed_out.store(out_ptr);
}


/**
 * @brief Persistent CUDA kernel for SwiGLU forward pass.
 *
 * Optimized kernel using async memory copies and double-buffered shared memory
 * for better memory bandwidth utilization on large tensors. Gives 5-10% speedup
 * over simple kernel when tensor is large enough to fill the GPU.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] out Output tensor of shape (BT, C).
 * @param[in] inp Input tensor of shape (BT, 2*C).
 * @param[in,out] abs_max_ptr Optional global absolute maximum tracker.
 * @param BT Batch size * sequence length.
 * @param C Hidden dimension.
 */
template<bool HasAbsMax, typename floatX>
__global__ __launch_bounds__(128) void swiglu_forward_persistent_kernel(floatX* out, const floatX* inp, float* abs_max_ptr, int BT, int C) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    const int start = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    const int stride = gridDim.x * blockDim.x * x128::size;

    // ensure alignment is multiple of cache-line-sector size, not just multiple of
    // transfer size, to avoid overfetch!
    __shared__ alignas(32) floatX up_buffer[2 * 128 * (16/sizeof(floatX))];
    __shared__ alignas(32) floatX gate_buffer[2 * 128 * (16/sizeof(floatX))];
    __shared__ float block_max;
    
    // only handle abs-max if requested; these are guaranteed to be warp-convergent branches,
    // so they don't cost us in this memory-bound kernel.
    if (HasAbsMax) {
        if(threadIdx.x == 0) {
            block_max = 0.f;
        }
        __syncthreads();
    }
    float thread_max = 0.f;
    // Per-thread slice within shared buffers
    const int lane_base = threadIdx.x * x128::size;
    
    floatX* up_ptr_smem = up_buffer + lane_base;
    floatX* gate_ptr_smem = gate_buffer + lane_base;

    StridedIterator<int> iter(start, stride, C);

    if(start < BT*C) {
        auto [bt, c] = iter;
        __pipeline_memcpy_async(up_ptr_smem, inp + (bt * C * 2 + c), 16);
        __pipeline_memcpy_async(gate_ptr_smem,  inp + (bt * C * 2 + c + C), 16);
    }
    __pipeline_commit();

    iter.advance();

    if(start + stride < BT*C) {
        auto [bt, c] = iter;
        __pipeline_memcpy_async(up_ptr_smem + 128 * x128::size, inp + (bt * C * 2 + c), 16);
        __pipeline_memcpy_async(gate_ptr_smem + 128 * x128::size,  inp + (bt * C * 2 + c + C), 16);
    }
    __pipeline_commit();

    int phase = 0;
    for(int idx = start; idx < BT*C; idx += stride) {
        // note: each thread reads only what it writes itself, so there is no need for further synchronization here
        __pipeline_wait_prior(1);
        x128 up_inp = x128::load(up_ptr_smem + 128 * x128::size * phase);
        x128 gate_inp = x128::load(gate_ptr_smem + 128 * x128::size * phase);

        iter.advance();
        if(idx + 2*stride < BT*C) {
            auto [bt, c] = iter;
            __pipeline_memcpy_async(up_ptr_smem + 128 * x128::size * phase, inp + (bt * C * 2 + c), 16);
            __pipeline_memcpy_async(gate_ptr_smem + 128 * x128::size * phase,  inp + (bt * C * 2 + c + C), 16);
        }
        __pipeline_commit();

        x128 packed_out;
        for(int k = 0; k < up_inp.size; ++k) {
            float x1 = (float)up_inp[k];
            float x2 = (float)gate_inp[k];
            packed_out[k] = (floatX)((x1 * x2) / (1.0f + expf(-x2)));
            if (HasAbsMax) {
                thread_max = fmaxf(thread_max, fabsf((float)packed_out[k]));
            }
        }
        packed_out.store(out + idx);
        phase = (phase + 1) % 2;
    }

    if (HasAbsMax) {
        handle_absmax_reduction(abs_max_ptr, &block_max, thread_max);
    }
}

/**
 * @brief Persistent CUDA kernel for SwiGLU forward with FP8 quantized output.
 *
 * Optimized kernel combining SwiGLU computation with FP8 quantization.
 * Uses async memory copies and double-buffered shared memory for large tensors.
 *
 * @tparam floatX Input data type (float or nv_bfloat16).
 * @param[out] out Output tensor in FP8 E4M3 format.
 * @param[out] scale_ptr Inverse scale factor for dequantization.
 * @param[in] inp Input tensor of shape (BT, 2*C).
 * @param[in] abs_max_ptr Pre-computed absolute maximum for scaling.
 * @param BT Batch size * sequence length.
 * @param C Hidden dimension.
 */
template<typename floatX>
__global__ void swiglu_forward_quant_persistent_kernel(__nv_fp8_e4m3* out, float* scale_ptr, const floatX* inp, const float* abs_max_ptr, int BT, int C) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    using f8v_t = GenericVector<__nv_fp8_e4m3, 16 / sizeof(floatX)>;

    float scale = 448.f / fmaxf(*abs_max_ptr, 1e-10f);
    if(threadIdx.x == 0 && blockIdx.x == 0 && scale_ptr) {
        *scale_ptr = 1.f / scale;
    }

    int start = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int stride = gridDim.x * blockDim.x * x128::size;

    __shared__ alignas(32) floatX up_buffer[2 * 128 * (16/sizeof(floatX))];
    __shared__ alignas(32) floatX gate_buffer[2 * 128 * (16/sizeof(floatX))];

    // Per-thread slice within shared buffers
    const int lane_base = threadIdx.x * x128::size;
    StridedIterator<int> iter(start, stride, C);
    if(start < BT*C) {
        auto [bt, c] = iter;
        __pipeline_memcpy_async(up_buffer + lane_base, inp + (bt * C * 2 + c), 16);
        __pipeline_memcpy_async(gate_buffer + lane_base,  inp + (bt * C * 2 + c + C), 16);
    }
    __pipeline_commit();
    iter.advance();
    if(start + stride < BT*C) {
        auto [bt, c] = iter;
        __pipeline_memcpy_async(up_buffer + lane_base + 128 * x128::size, inp + (bt * C * 2 + c), 16);
        __pipeline_memcpy_async(gate_buffer + lane_base + 128 * x128::size,  inp + (bt * C * 2 + c + C), 16);
    }
    __pipeline_commit();

    int phase = 0;
    for(int idx = start; idx < BT*C; idx += stride) {
        // note: each thread reads only what it writes itself, so there is no need for further synchronization here
        __pipeline_wait_prior(1);
        x128 up_inp = x128::load(up_buffer + lane_base + 128 * x128::size * phase);
        x128 gate_inp = x128::load(gate_buffer + lane_base + 128 * x128::size * phase);
        iter.advance();
        if(idx + 2*stride < BT*C) {
            auto [bt, c] = iter;
            __pipeline_memcpy_async(up_buffer + lane_base + 128 * x128::size * phase, inp + (bt * C * 2 + c), 16);
            __pipeline_memcpy_async(gate_buffer + lane_base + 128 * x128::size * phase,  inp + (bt * C * 2 + c + C), 16);
        }
        __pipeline_commit();

        f8v_t packed_out;
        for(int k = 0; k < up_inp.size; ++k) {
            float x1 = (float)up_inp[k];
            float x2 = (float)gate_inp[k];
            float result = (x1 * x2) / (1.0f + expf(-x2));
            floatX qr = (floatX)result;
            __nv_fp8_e4m3 quant;
            quant.__x = __nv_cvt_float_to_fp8(scale * (float)qr, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);
            packed_out[k] = quant;
        }
        packed_out.store(out + idx);

        phase = (phase + 1) % 2;
    }
}

/**
 * @brief CUDA kernel for SwiGLU backward pass.
 *
 * Computes gradients for both up and gate inputs given upstream gradient.
 * For out = up * silu(gate):
 * - d_up = dout * gate * sigmoid(gate)
 * - d_gate = dout * up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] dinp Output gradients of shape (B, T, 2*C) with [d_up, d_gate].
 * @param[in] dout Upstream gradient of shape (B, T, C).
 * @param[in] inp Original input tensor of shape (B, T, 2*C).
 * @param[in,out] abs_max_ptr Optional global absolute maximum tracker.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Hidden dimension.
 */
template<typename floatX>
__global__ void swiglu_backward_kernel1(floatX* dinp, const floatX* dout, const floatX* inp, float* abs_max_ptr, int B, int T, int C) {
    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    // Shared memory init must happen before bounds-check to keep all threads in sync
    __shared__ float block_max;
    if (abs_max_ptr) {
        if(threadIdx.x == 0) {
            block_max = 0.f;
        }
        __syncthreads();
    }

    float thread_max = 0.f;

    // Bounds check: with MoE+EP, B*T*C may not be aligned to block_size * x128::size,
    // so the last block can have OOB threads. Guard loads/stores but keep all threads
    // participating in the absmax reduction below.
    if (idx < B * T * C) {
        const floatX* dout_ptr = dout + idx;
        // b,t,c in the output
        int b = idx / (T * C);
        int t = (idx / C) % T;
        int c = idx % C;
        // coords in input
        int C2 = C * 2;
        const floatX* inp1_ptr = inp + (b * T * C2 + t * C2 + c);
        const floatX* inp2_ptr = inp1_ptr + C;
        floatX* dinp1_ptr = dinp + (b * T * C2 + t * C2 + c);
        floatX* dinp2_ptr = dinp1_ptr + C;
        // backward
        x128 dinp1;
        x128 dinp2;
        x128 packed_dout = x128::load_cs(dout_ptr);
        x128 packed_inp1 = x128::load_cs(inp1_ptr); // fc1
        x128 packed_inp2 = x128::load_cs(inp2_ptr); // fc2

        for(int k = 0; k < packed_inp1.size; ++k) {
            float x1 = (float)packed_inp1[k];
            float x2 = (float)packed_inp2[k];
            float dout = (float)packed_dout[k];

            float sx2 = 1.0f / (1.0f + expf(-x2)); // sigmoid of x2
            float dx1 = dout * x2 * sx2;
            float dx2 = dout * x1 * sx2 * (1.0f + x2 * (1.0f - sx2));

            dinp1[k] = (floatX)dx1;
            dinp2[k] = (floatX)dx2;

            if (abs_max_ptr) {
                thread_max = fmaxf(thread_max, fabsf(dinp1[k]));
                thread_max = fmaxf(thread_max, fabsf(dinp2[k]));
            }
        }
        dinp1.store(dinp1_ptr);
        dinp2.store(dinp2_ptr);
    }

    handle_absmax_reduction(abs_max_ptr, &block_max, thread_max);
}

// ----------------------------------------------------------------------------
// kernel launchers

/**
 * @brief Template launcher for SwiGLU forward pass.
 *
 * Automatically selects between simple and persistent kernel based on tensor size.
 * Uses persistent kernel when tensor is large enough to benefit from it.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] out Output tensor of shape (B, T, C).
 * @param[in] inp Input tensor of shape (B, T, 2*C) with [up, gate] concatenated.
 * @param[in,out] abs_max_ptr Optional absolute maximum tracker.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Hidden dimension.
 * @param stream CUDA stream.
 */
template<typename floatX>
void swiglu_forward_impl(floatX* out, const floatX* inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {

    if (2ll*B*T*C >= std::numeric_limits<int>::max()) {
        throw std::runtime_error("swiglu_forward: input too large");
    }

    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    if (abs_max_ptr)
        CUDA_CHECK(cudaMemsetAsync(abs_max_ptr, 0, sizeof(float), stream));

    const int block_size = 128;
    assert(C % x128::size == 0);
    // Note: B*T*C alignment to block_size*x128::size is NOT guaranteed with MoE+EP
    // (dynamic token counts). Kernels have bounds checking to handle this.
    int bpsm;
    if (abs_max_ptr) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, swiglu_forward_persistent_kernel<true, floatX>, block_size, 0));
    } else {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, swiglu_forward_persistent_kernel<false, floatX>, block_size, 0));
    }
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

    // only use persistent kernel if we get enough blocks
    const int num_blocks = div_ceil(B*T*C, (int)(block_size * x128::size));
    if (num_blocks < bpsm * sms) {
        swiglu_forward_kernel<<<num_blocks, block_size, 0, stream>>>(out, inp, abs_max_ptr, C, B*T*C);
    } else {
        if (abs_max_ptr) {
            swiglu_forward_persistent_kernel<true><<<bpsm * sms, block_size, 0, stream>>>(out, inp, abs_max_ptr, B * T, C);
        } else {
            swiglu_forward_persistent_kernel<false><<<bpsm * sms, block_size, 0, stream>>>(out, inp, nullptr, B * T, C);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

/// @brief SwiGLU forward pass for BF16 tensors.
void swiglu_forward(nv_bfloat16* out, const nv_bfloat16* inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {
    swiglu_forward_impl(out, inp, abs_max_ptr, B, T, C, stream);
}

/// @brief SwiGLU forward pass for FP32 tensors.
void swiglu_forward(float* out, const float* inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {
    swiglu_forward_impl(out, inp, abs_max_ptr, B, T, C, stream);
}

/**
 * @brief SwiGLU forward pass with FP8 quantized output.
 *
 * Computes SwiGLU from BF16 input and quantizes output to FP8 E4M3.
 * Automatically selects between simple and persistent kernel.
 *
 * @param[out] out Output tensor in FP8 E4M3 format.
 * @param[out] scale_ptr Inverse scale factor for dequantization.
 * @param[in] inp Input BF16 tensor of shape (B, T, 2*C).
 * @param[in] abs_max_ptr Pre-computed absolute maximum for scaling.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Hidden dimension.
 * @param stream CUDA stream.
 */
void swiglu_forward_quant(__nv_fp8_e4m3* out, float* scale_ptr, const nv_bfloat16* inp, const float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {
    if (2ll*B*T*C >= std::numeric_limits<int>::max()) {
        throw std::runtime_error("swiglu_forward_quant: input too large");
    }
    using x128 = GenericVector<nv_bfloat16, 16/sizeof(nv_bfloat16)>;
    const int block_size = 128;
    assert(C % x128::size == 0);
    assert((B*T*C) % (block_size * x128::size) == 0);
    int bpsm;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, swiglu_forward_quant_persistent_kernel<nv_bfloat16>, block_size, 0));
    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

    // only use persistent kernel if we get enough blocks
    const int num_blocks = div_ceil(B*T*C, (int)(block_size * x128::size));
    if (num_blocks < bpsm * sms) {
        swiglu_forward_quant_kernel<<<num_blocks, block_size, 0, stream>>>(out, scale_ptr, inp, abs_max_ptr, C);
    } else {
        swiglu_forward_quant_persistent_kernel<<<bpsm * sms, block_size, 0, stream>>>(out, scale_ptr, inp, abs_max_ptr, B * T, C);
    }
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Template launcher for SwiGLU backward pass.
 *
 * Computes gradients for both up and gate inputs.
 *
 * @tparam floatX Data type (float or nv_bfloat16).
 * @param[out] dinp Output gradients of shape (B, T, 2*C).
 * @param[in] dout Upstream gradient of shape (B, T, C).
 * @param[in] inp Original input tensor of shape (B, T, 2*C).
 * @param[in,out] abs_max Optional absolute maximum tracker.
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Hidden dimension.
 * @param stream CUDA stream.
 */
template<typename floatX>
void swiglu_backward_impl(floatX* dinp, const floatX* dout, const floatX* inp, float* abs_max, int B, int T, int C, cudaStream_t stream) {
    if (2ll*B*T*C >= std::numeric_limits<int>::max()) {
        throw std::runtime_error("swiglu_backward: output too large");
    }

    using x128 = GenericVector<floatX, 16/sizeof(floatX)>;
    // input is (B, T, 2C), output is (B, T, C)
    // we have that inp[b, t, :] = [fc1, fc2] (i.e. they are concatenated in each C-fiber)

    if (abs_max)
        CUDA_CHECK(cudaMemsetAsync(abs_max, 0, sizeof(float), stream));

    const int block_size = 256;
    // Note: B*T*C alignment to block_size*x128::size is NOT guaranteed with MoE+EP
    // (dynamic token counts). The kernel has bounds checking to handle this.
    const int grid_size = div_ceil((size_t)B*T*C, block_size * x128::size);
    swiglu_backward_kernel1<<<grid_size, block_size, 0, stream>>>(dinp, dout, inp, abs_max, B, T, C);
    CUDA_CHECK(cudaGetLastError());
}

/// @brief SwiGLU backward pass for BF16 tensors.
void swiglu_backward(nv_bfloat16* dinp, const nv_bfloat16* dout, const nv_bfloat16* inp, float* abs_max, int B, int T, int C, cudaStream_t stream) {
    swiglu_backward_impl(dinp, dout, inp, abs_max, B, T, C, stream);
}

/// @brief SwiGLU backward pass for FP32 tensors.
void swiglu_backward(float* dinp, const float* dout, const float* inp, float* abs_max, int B, int T, int C, cudaStream_t stream) {
    swiglu_backward_impl(dinp, dout, inp, abs_max, B, T, C, stream);
}
