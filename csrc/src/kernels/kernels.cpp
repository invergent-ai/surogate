// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels.h"

#include "utilities/tensor.h"
#include <cstdio>

/**
 * @brief Performs the forward pass of Root Mean Square Normalization (RMSNorm).
 *
 * This function dispatches the RMSNorm operation to the appropriate kernel implementation
 * based on the data type of the output tensor (BF16 or FP32). It normalizes the input
 * tensor using the RMS statistic and applies a learnable weight.
 *
 * @param out [out] The output tensor where the normalized results will be stored.
 *                  Supported types: ETensorDType::BF16, ETensorDType::FP32.
 * @param rms [out] The tensor to store the calculated root mean square statistics (typically float).
 * @param inp [in] The input tensor to be normalized. Must match the data type of `out`.
 * @param weight [in] The learnable weight tensor (gamma) for scaling. Must match the data type of `out`.
 * @param abs_max_ptr [out] Pointer to a float to store the absolute maximum value found during
 *                          processing (optional, depending on implementation).
 * @param epsilon [in] A small constant added to the root mean square to avoid division by zero.
 * @param B [in] Batch size dimension.
 * @param T [in] Time/Sequence length dimension.
 * @param C [in] Channel/Hidden dimension (size of the vector being normalized).
 * @param stream [in] The CUDA stream on which to execute the kernel.
 *
 * @throws std::logic_error If the output tensor data type is not supported (neither BF16 nor FP32).
 */
void rmsnorm_forward(Tensor& out, Tensor& rms, const Tensor& inp, const Tensor& weight, float* abs_max_ptr, float epsilon, int B, int T, int C, cudaStream_t stream) {
    if(out.DType == ETensorDType::BF16) {
        rmsnorm_forward(out.get<nv_bfloat16>(), rms.get<float>(), inp.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), abs_max_ptr, epsilon, B, T, C, stream);
    } else if (out.DType == ETensorDType::FP32) {
        rmsnorm_forward(out.get<float>(), rms.get<float>(), inp.get<float>(), weight.get<float>(), abs_max_ptr, epsilon, B, T, C, stream);
    } else {
        throw std::logic_error("rmsnorm_forward: unsupported dtype");
    }
}

void rmsnorm_apply_saved(Tensor& out, const Tensor& inp, const Tensor& weight, const Tensor& rstd, int B, int T, int C, cudaStream_t stream) {
    if(out.DType == ETensorDType::BF16) {
        rmsnorm_apply_saved(out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), rstd.get<float>(), B, T, C, stream);
    } else if (out.DType == ETensorDType::FP32) {
        rmsnorm_apply_saved(out.get<float>(), inp.get<float>(), weight.get<float>(), rstd.get<float>(), B, T, C, stream);
    } else {
        throw std::logic_error("rmsnorm_apply_saved: unsupported dtype");
    }
}

/**
 * @brief Computes the backward pass for Root Mean Square Normalization (RMSNorm).
 *
 * This function calculates the gradients for the input tensor and the weight tensor based on the
 * gradients coming from the subsequent layer. It dispatches the computation to specific
 * implementations based on the data type of the input tensor (BF16 or FP32).
 *
 * @param[out] dinp        Gradient of the input tensor. Must match the dimensions of `inp`.
 *                         Supported types: BF16, FP32.
 * @param[out] dweight     Gradient of the weight (scale) tensor. Must match the dimensions of `weight`.
 *                         Supported types: BF16, FP32.
 * @param[in,out] scratch  Scratch memory buffer used for intermediate calculations during the kernel execution.
 * @param[in] dresidual    Gradient of the residual connection. Can be null or empty if not applicable.
 *                         Supported types: BF16, FP32.
 * @param[in] dout         Gradient of the output tensor (upstream gradient).
 *                         Supported types: BF16, FP32.
 * @param[in] inp          Original input tensor from the forward pass.
 *                         Supported types: BF16, FP32.
 * @param[in] weight       Original weight (scale) tensor used in the forward pass.
 *                         Supported types: BF16, FP32.
 * @param[in] rstd         Reciprocal standard deviation calculated during the forward pass.
 *                         Expected type: FP32.
 * @param[out] abs_max_ptr Pointer to store the absolute maximum value found during computation (optional, for quantization/stats).
 * @param[in] B            Batch size dimension.
 * @param[in] T            Time/Sequence length dimension.
 * @param[in] C            Channel/Hidden dimension.
 * @param[in] dp           CUDA device properties, used to optimize kernel launch parameters.
 * @param[in] stream       CUDA stream to execute the kernels on.
 *
 * @throws std::logic_error If the data type of `dinp` is not ETensorDType::BF16 or ETensorDType::FP32.
 */
void rmsnorm_backward(Tensor& dinp, Tensor& dweight, Tensor& scratch, const Tensor& dresidual, const Tensor& dout, const Tensor& inp, const Tensor& weight, const Tensor& rstd, float* abs_max_ptr,
                      int B, int T, int C, const cudaDeviceProp& dp, cudaStream_t stream, bool skip_weight_grad) {
    if(dinp.DType == ETensorDType::BF16) {
        rmsnorm_backward(dinp.get<nv_bfloat16>(), dweight.get<nv_bfloat16>(), scratch.Data, dresidual.get<nv_bfloat16>(),
            dout.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), rstd.get<float>(), abs_max_ptr, B, T, C, dp, stream, skip_weight_grad);
    } else if (dinp.DType == ETensorDType::FP32) {
        rmsnorm_backward(dinp.get<float>(), dweight.get<float>(), scratch.Data, dresidual.get<float>(),
            dout.get<float>(), inp.get<float>(), weight.get<float>(), rstd.get<float>(), abs_max_ptr, B, T, C, dp, stream, skip_weight_grad);
    } else {
        throw std::logic_error("rmsnorm_backward: unsupported dtype");
    }
}

/**
 * @brief Performs a fused residual addition and Root Mean Square Layer Normalization (RMSNorm) forward pass.
 *
 * This function computes the element-wise sum of two input tensors (`inp1` and `inp2`), stores the result
 * in the `residual` tensor, and then applies RMSNorm to this sum, storing the normalized output in `normed`.
 * It supports dynamic dispatch based on the data type of the `residual` tensor (BF16 or FP32).
 *
 * The operation can be described as:
 * 1. residual = inp1 + inp2
 * 2. normed = RMSNorm(residual, weight, epsilon)
 *
 * @param[out] residual    Output tensor to store the result of the addition (inp1 + inp2).
 *                         Must match the data type of inp1, inp2, and weight.
 * @param[out] normed      Output tensor to store the normalized result.
 * @param[out] rrms        Output tensor to store the reciprocal root mean square values (used for backward pass).
 *                         Typically of type float.
 * @param[in]  inp1        First input tensor for the residual addition.
 * @param[in]  inp2        Second input tensor for the residual addition.
 * @param[in]  weight      Weight tensor (gain) for the RMSNorm affine transformation.
 * @param[out] abs_max_ptr Pointer to a float to store the absolute maximum value of the output (optional/context-dependent).
 * @param[in]  epsilon     Small constant added to the root mean square for numerical stability.
 * @param[in]  N           Batch size or number of tokens (outer dimension).
 * @param[in]  C           Hidden dimension size (inner dimension).
 * @param[in]  stream      CUDA stream on which to execute the kernel.
 *
 * @throws std::logic_error If the tensor data type is not supported (currently supports BF16 and FP32).
 */
void fused_residual_rmsnorm_forward(Tensor& residual, Tensor& normed, Tensor& rrms, const Tensor& inp1, const Tensor& inp2, const Tensor& weight, float* abs_max_ptr,
                                    float epsilon, int N, int C, cudaStream_t stream)
{
    if(residual.DType == ETensorDType::BF16) {
        fused_residual_rmsnorm_forward(residual.get<nv_bfloat16>(), normed.get<nv_bfloat16>(), rrms.get<float>(),
            inp1.get<nv_bfloat16>(), inp2.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), abs_max_ptr, epsilon, N, C, stream);
    } else if (residual.DType == ETensorDType::FP32) {
        fused_residual_rmsnorm_forward(residual.get<float>(), normed.get<float>(), rrms.get<float>(),
            inp1.get<float>(), inp2.get<float>(), weight.get<float>(), abs_max_ptr, epsilon, N, C, stream);
    } else {
        throw std::logic_error("fused_residual_rmsnorm_forward: unsupported dtype");
    }
}

void fused_residual_rmsnorm_apply_saved(Tensor& residual, Tensor& normed,
                                        const Tensor& inp1, const Tensor& inp2,
                                        const Tensor& weight, const Tensor& rstd,
                                        int N, int C, cudaStream_t stream) {
    if(residual.DType == ETensorDType::BF16) {
        fused_residual_rmsnorm_apply_saved(residual.get<nv_bfloat16>(), normed.get<nv_bfloat16>(),
            inp1.get<nv_bfloat16>(), inp2.get<nv_bfloat16>(), weight.get<nv_bfloat16>(), rstd.get<float>(), N, C, stream);
    } else if (residual.DType == ETensorDType::FP32) {
        fused_residual_rmsnorm_apply_saved(residual.get<float>(), normed.get<float>(),
            inp1.get<float>(), inp2.get<float>(), weight.get<float>(), rstd.get<float>(), N, C, stream);
    } else {
        throw std::logic_error("fused_residual_rmsnorm_apply_saved: unsupported dtype");
    }
}

/**
 * @brief Performs the SwiGLU (Swish-Gated Linear Unit) activation function forward pass.
 *
 * This function acts as a dispatcher that invokes the appropriate type-specific implementation
 * (FP32 or BF16) based on the data type of the output tensor.
 *
 * The SwiGLU operation typically computes: out = (x * sigmoid(x)) * y, where the input
 * is split into two halves (x and y) along the last dimension.
 *
 * @param out [out] The output tensor where the result will be stored. Must be pre-allocated.
 *                  Supported types: ETensorDType::FP32, ETensorDType::BF16.
 * @param inp [in] The input tensor containing the data to be activated.
 *                 Must match the data type of the output tensor.
 * @param abs_max_ptr [out] Pointer to a float to store the absolute maximum value of the output
 *                          (often used for quantization calibration). Can be nullptr if not needed.
 * @param B [in] Batch size dimension.
 * @param T [in] Time/Sequence length dimension.
 * @param C [in] Channel/Hidden dimension (size of the output vector).
 *               Note: The input tensor's last dimension is usually 2 * C.
 * @param stream [in] The CUDA stream on which to execute the kernel.
 *
 * @throws std::logic_error If the output tensor data type is not supported (neither FP32 nor BF16).
 */
void swiglu_forward(Tensor& out, const Tensor& inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {
    if(out.DType == ETensorDType::FP32) {
        swiglu_forward(out.get<float>(), inp.get<float>(), abs_max_ptr, B, T, C, stream);
    } else if (out.DType == ETensorDType::BF16) {
        swiglu_forward(out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), abs_max_ptr, B, T, C, stream);
    } else {
        throw std::logic_error("swiglu_forward: unsupported dtype");
    }
}

/**
 * @brief Performs the SwiGLU activation function with quantization support.
 *
 * This function applies the SwiGLU (Swish-Gated Linear Unit) activation to the input tensor
 * and quantizes the result into the output tensor. It specifically handles the conversion
 * from BF16 input to FP8_E4M3 output.
 *
 * The operation is defined as: SwiGLU(x, y) = Swish(x) * y = (x * sigmoid(x)) * y.
 * In the context of this kernel, the input tensor is expected to contain both the gate
 * and the value components (usually concatenated along the last dimension), which are split internally.
 *
 * @param out [out] The output tensor where the quantized results will be stored.
 *                  Must have DType `ETensorDType::FP8_E4M3`.
 * @param scale_ptr [out] Pointer to a float where the calculated quantization scale factor will be stored.
 * @param inp [in] The input tensor containing the pre-activation values.
 *                 Must have DType `ETensorDType::BF16`.
 * @param abs_max_ptr [in] Pointer to the absolute maximum value of the input (or relevant block), used for quantization scaling.
 * @param B [in] Batch size dimension.
 * @param T [in] Time/Sequence length dimension.
 * @param C [in] Channel/Hidden dimension size.
 * @param stream [in] The CUDA stream on which to execute the kernel.
 *
 * @throws std::logic_error If the input/output tensor data types are not supported (currently requires out=FP8_E4M3 and inp=BF16).
 */
void swiglu_forward_quant(Tensor& out, float* scale_ptr, const Tensor& inp, const float* abs_max_ptr, int B, int T, int C, cudaStream_t stream) {
    if(out.DType == ETensorDType::FP8_E4M3 && inp.DType == ETensorDType::BF16) {
        swiglu_forward_quant(out.get<__nv_fp8_e4m3>(), scale_ptr, inp.get<nv_bfloat16>(), abs_max_ptr, B, T, C, stream);
    } else {
        throw std::logic_error("swiglu_forward_quant: unsupported dtype");
    }
}

/**
 * @brief Computes the backward pass for the SwiGLU activation function.
 *
 * This function acts as a dispatcher, invoking the appropriate CUDA kernel implementation
 * based on the data type of the input tensor (FP32 or BF16). It calculates the gradients
 * for the input tensor given the gradients of the output and the original input values.
 *
 * @param[out] dinp    The gradient tensor for the input. This tensor will be populated with the computed gradients.
 *                     Supported data types: ETensorDType::FP32, ETensorDType::BF16.
 * @param[in]  dout    The gradient tensor coming from the subsequent layer (upstream gradient).
 *                     Must match the data type of `dinp`.
 * @param[in]  inp     The original input tensor used in the forward pass.
 *                     Must match the data type of `dinp`.
 * @param[out] abs_max Pointer to a float to store the absolute maximum value encountered during the operation
 *                     (often used for quantization calibration or tracking).
 * @param[in]  B       Batch size dimension.
 * @param[in]  T       Time/Sequence length dimension.
 * @param[in]  C       Channel/Hidden dimension.
 * @param[in]  stream  The CUDA stream on which to execute the kernel.
 *
 * @throws std::logic_error If the tensor data type is not supported (i.e., not FP32 or BF16).
 */
void swiglu_backward(Tensor& dinp, const Tensor& dout, const Tensor& inp, float* abs_max, int B, int T, int C, cudaStream_t stream) {
    if(dinp.DType == ETensorDType::FP32) {
        swiglu_backward(dinp.get<float>(), dout.get<float>(), inp.get<float>(), abs_max, B, T, C, stream);
    } else if (dinp.DType == ETensorDType::BF16) {
        swiglu_backward(dinp.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), abs_max, B, T, C, stream);
    } else {
        throw std::logic_error("swiglu_backward: unsupported dtype");
    }
}

void silu_forward(Tensor& out, const Tensor& inp, long n, cudaStream_t stream) {
    if (out.DType == ETensorDType::FP32) {
        silu_forward(out.get<float>(), inp.get<float>(), n, stream);
    } else if (out.DType == ETensorDType::BF16) {
        silu_forward(out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), n, stream);
    } else {
        throw std::logic_error("silu_forward: unsupported dtype");
    }
}

void silu_backward(Tensor& dinp, const Tensor& inp, const Tensor& dout, long n, cudaStream_t stream) {
    if (dinp.DType == ETensorDType::FP32) {
        silu_backward(dinp.get<float>(), inp.get<float>(), dout.get<float>(), n, stream);
    } else if (dinp.DType == ETensorDType::BF16) {
        silu_backward(dinp.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), n, stream);
    } else {
        throw std::logic_error("silu_backward: unsupported dtype");
    }
}

void relu2_forward(Tensor& out, const Tensor& inp, long n, cudaStream_t stream) {
    if (out.DType == ETensorDType::FP32) {
        relu2_forward(out.get<float>(), inp.get<float>(), n, stream);
    } else if (out.DType == ETensorDType::BF16) {
        relu2_forward(out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), n, stream);
    } else {
        throw std::logic_error("relu2_forward: unsupported dtype");
    }
}

void relu2_backward(Tensor& dinp, const Tensor& inp, const Tensor& dout, long n, cudaStream_t stream) {
    if (dinp.DType == ETensorDType::FP32) {
        relu2_backward(dinp.get<float>(), inp.get<float>(), dout.get<float>(), n, stream);
    } else if (dinp.DType == ETensorDType::BF16) {
        relu2_backward(dinp.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), n, stream);
    } else {
        throw std::logic_error("relu2_backward: unsupported dtype");
    }
}

/**
 * @brief Dispatches the Rotary Positional Embedding (RoPE) forward pass to the appropriate kernel based on tensor data type.
 *
 * This function handles type checking for the output tensor and calls the corresponding
 * template specialization or overloaded function for either FP32 or BF16 data types.
 *
 * @param out The output tensor where the result will be stored. Must be pre-allocated.
 *            Supported types: ETensorDType::FP32, ETensorDType::BF16.
 * @param in The input tensor containing the query and key vectors to be rotated.
 * @param freqs_cis The precomputed complex frequency tensor (cis) used for rotation.
 * @param abs_max_ptr Pointer to store the absolute maximum value (used for quantization calibration, optional).
 * @param B Batch size.
 * @param T Sequence length (Time steps).
 * @param Nq Number of query heads.
 * @param Nkv Number of key/value heads.
 * @param head_dim Dimension of each attention head.
 * @param stream The CUDA stream to execute the kernel on.
 *
 * @throws std::logic_error If the output tensor data type is not supported (neither FP32 nor BF16).
 */
void rope_forward(Tensor& out, const Tensor& in, const Tensor& freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream)  {
    if(out.DType == ETensorDType::FP32) {
        rope_forward(out.get<float>(), in.get<float>(), freqs_cis.get<float>(), position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, stream);
    } else if(out.DType == ETensorDType::BF16) {
        rope_forward(out.get<nv_bfloat16>(), in.get<nv_bfloat16>(), freqs_cis.get<nv_bfloat16>(), position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, stream);
    } else {
        throw std::logic_error("rope_forward: unsupported dtype");
    }
}

/**
 * @brief Computes the backward pass for Rotary Positional Embeddings (RoPE).
 *
 * This function acts as a dispatcher that invokes the appropriate CUDA kernel implementation
 * based on the data type of the input tensor (FP32 or BF16). It computes the gradients
 * for the input embeddings by applying the inverse RoPE rotation to the output gradients.
 *
 * @param dinp [Out] The gradient of the input tensor to be computed. This tensor will be modified in-place.
 *                   Supported data types: FP32, BF16.
 * @param dout [In] The incoming gradient tensor from the subsequent layer.
 * @param freqs_cis [In] Precomputed complex exponential frequency values used for rotation.
 *                       Shape is typically related to [T, head_dim/2].
 * @param abs_max_ptr [Out] Pointer to device memory where the absolute maximum value of the
 *                          computed gradients will be stored (often used for quantization calibration).
 *                          Can be nullptr if not required.
 * @param B [In] Batch size.
 * @param T [In] Sequence length (time steps).
 * @param Nq [In] Number of query heads.
 * @param Nkv [In] Number of key/value heads.
 * @param head_dim [In] Dimension of each attention head.
 * @param stream [In] The CUDA stream on which to execute the kernel.
 *
 * @throws std::logic_error If the data type of `dinp` is not supported (i.e., not FP32 or BF16).
 */
void rope_backward(Tensor& dinp, const Tensor& dout, const Tensor& freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    if(dinp.DType == ETensorDType::FP32) {
        rope_backward(dinp.get<float>(), dout.get<float>(), freqs_cis.get<float>(), position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, stream);
    } else if(dinp.DType == ETensorDType::BF16) {
        rope_backward(dinp.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), freqs_cis.get<nv_bfloat16>(), position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, stream);
    } else {
        throw std::logic_error("rope_backward: unsupported dtype");
    }
}

/**
 * @brief Computes the forward pass for Rotary Positional Embeddings (RoPE) with partial rotation support.
 *
 * For partial RoPE (GLM4 style), only the first rotary_dim dimensions are rotated.
 * Dimensions beyond rotary_dim are passed through unchanged.
 *
 * @param out The output tensor for the result of the RoPE transformation.
 * @param in The input tensor containing the embedding data.
 * @param freqs_cis Precomputed RoPE frequencies (cos and sin values interleaved), shape (T, rotary_dim).
 * @param position_ids Optional position IDs for variable-length sequences (can be nullptr).
 * @param abs_max_ptr Optional pointer to track the absolute maximum value.
 * @param B Batch size.
 * @param T Sequence length.
 * @param Nq Number of query heads.
 * @param Nkv Number of key/value heads.
 * @param head_dim Full head dimension.
 * @param rotary_dim Number of dimensions to rotate (must be <= head_dim and even).
 * @param stream The CUDA stream to execute the kernel on.
 */
void rope_forward(Tensor& out, const Tensor& in, const Tensor& freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream) {
    if(out.DType == ETensorDType::FP32) {
        rope_forward(out.get<float>(), in.get<float>(), freqs_cis.get<float>(), position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, rotary_dim, stream);
    } else if(out.DType == ETensorDType::BF16) {
        rope_forward(out.get<nv_bfloat16>(), in.get<nv_bfloat16>(), freqs_cis.get<nv_bfloat16>(), position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, rotary_dim, stream);
    } else {
        throw std::logic_error("rope_forward: unsupported dtype");
    }
}

/**
 * @brief Computes the backward pass for Rotary Positional Embeddings (RoPE) with partial rotation support.
 *
 * For partial RoPE (GLM4 style), only the first rotary_dim dimensions are rotated.
 * Dimensions beyond rotary_dim are passed through unchanged.
 *
 * @param dinp The gradient of the input tensor to be computed.
 * @param dout The incoming gradient tensor from the subsequent layer.
 * @param freqs_cis Precomputed RoPE frequencies, shape (T, rotary_dim).
 * @param position_ids Optional position IDs for variable-length sequences.
 * @param abs_max_ptr Optional pointer for tracking absolute maximum value.
 * @param B Batch size.
 * @param T Sequence length.
 * @param Nq Number of query heads.
 * @param Nkv Number of key/value heads.
 * @param head_dim Full head dimension.
 * @param rotary_dim Number of dimensions to rotate (must be <= head_dim and even).
 * @param stream The CUDA stream on which to execute the kernel.
 */
void rope_backward(Tensor& dinp, const Tensor& dout, const Tensor& freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream) {
    if(dinp.DType == ETensorDType::FP32) {
        rope_backward(dinp.get<float>(), dout.get<float>(), freqs_cis.get<float>(), position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, rotary_dim, stream);
    } else if(dinp.DType == ETensorDType::BF16) {
        rope_backward(dinp.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), freqs_cis.get<nv_bfloat16>(), position_ids, abs_max_ptr, B, T, Nq, Nkv, head_dim, rotary_dim, stream);
    } else {
        throw std::logic_error("rope_backward: unsupported dtype");
    }
}

/**
 * @brief Dispatches the fused RoPE forward pass (with on-the-fly cos/sin computation) based on tensor data type.
 *
 * This is an optimized version that computes cos/sin via sincosf() and caches them in shared memory,
 * eliminating the need for precomputed freqs_cis tensor and reducing global memory bandwidth.
 *
 * @param out The output tensor (can be same as inp for in-place operation).
 * @param inp The input tensor containing QKV vectors.
 * @param position_ids Optional position IDs for each token (can be nullptr for sequential positions).
 * @param abs_max_ptr Optional pointer for tracking absolute maximum value.
 * @param theta RoPE base frequency (e.g., 10000.0 or 500000.0).
 * @param B Batch size.
 * @param T Sequence length.
 * @param Nq Number of query heads.
 * @param Nkv Number of key/value heads.
 * @param head_dim Dimension of each attention head.
 * @param stream CUDA stream.
 */
void rope_fused_forward(Tensor& out, const Tensor& inp, const int* position_ids, float* abs_max_ptr, float theta, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    if (out.DType == ETensorDType::FP32) {
        rope_fused_forward(out.get<float>(), inp.get<float>(), position_ids, abs_max_ptr, theta, B, T, Nq, Nkv, head_dim, stream);
    } else if (out.DType == ETensorDType::BF16) {
        rope_fused_forward(out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), position_ids, abs_max_ptr, theta, B, T, Nq, Nkv, head_dim, stream);
    } else {
        throw std::logic_error("rope_fused_forward: unsupported dtype");
    }
}

/**
 * @brief Dispatches the fused RoPE backward pass (with on-the-fly cos/sin computation) based on tensor data type.
 *
 * This is an optimized version that computes cos/sin via sincosf() and caches them in shared memory.
 * Applies inverse rotation (negated sin) to propagate gradients through RoPE.
 *
 * @param dinp The output gradient tensor (can be same as dout for in-place operation).
 * @param dout The incoming gradient tensor.
 * @param position_ids Optional position IDs for each token (can be nullptr for sequential positions).
 * @param abs_max_ptr Optional pointer for tracking absolute maximum value.
 * @param theta RoPE base frequency.
 * @param B Batch size.
 * @param T Sequence length.
 * @param Nq Number of query heads.
 * @param Nkv Number of key/value heads.
 * @param head_dim Dimension of each attention head.
 * @param stream CUDA stream.
 */
void rope_fused_backward(Tensor& dinp, const Tensor& dout, const int* position_ids, float* abs_max_ptr, float theta, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream) {
    if (dinp.DType == ETensorDType::FP32) {
        rope_fused_backward(dinp.get<float>(), dout.get<float>(), position_ids, abs_max_ptr, theta, B, T, Nq, Nkv, head_dim, stream);
    } else if (dinp.DType == ETensorDType::BF16) {
        rope_fused_backward(dinp.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), position_ids, abs_max_ptr, theta, B, T, Nq, Nkv, head_dim, stream);
    } else {
        throw std::logic_error("rope_fused_backward: unsupported dtype");
    }
}

/**
 * @brief Dispatches the fused classifier kernel based on the data type of the logits tensor.
 *
 * This function handles the computation of cross-entropy loss and optionally the gradients
 * (dlogits) in a fused manner for efficiency. It supports both FP32 and BF16 data types
 * for the logits, dispatching to the appropriate template specialization.
 *
 * @param logits [in, out] The input logits tensor of shape (BT, V). If `write_dlogits` is true,
 *                         this tensor will be updated in-place with the gradients.
 * @param losses [out] The output tensor to store individual losses. Expected to be of type float.
 * @param dloss [in] The scaling factor for the loss gradient (usually 1.0 / batch_size).
 * @param targets [in] The ground truth target indices tensor of shape (BT). Expected to be of type int.
 * @param BT [in] The batch size times time dimension (total number of tokens).
 * @param V [in] The vocabulary size (dimension of the logits).
 * @param P [in] The padding index (or similar parameter depending on specific kernel implementation).
 * @param write_dlogits [in] If true, gradients are computed and written back to `logits`.
 *                           If false, only the forward pass (loss calculation) is performed.
 * @param stream [in] The CUDA stream on which to execute the kernel.
 *
 * @throws std::runtime_error If the `logits` tensor data type is not FP32 or BF16.
 */
void fused_classifier(Tensor& logits, Tensor& losses,
                      float dloss, const Tensor& targets, Tensor* valid_token_count,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream) {
    int* count_ptr = valid_token_count ? valid_token_count->get<int>() : nullptr;
    if(logits.DType == ETensorDType::FP32) {
        fused_classifier(logits.get<float>(), losses.get<float>(), dloss, targets.get<int>(), count_ptr, BT, V, P, write_dlogits, stream);
    } else if(logits.DType == ETensorDType::BF16) {
        fused_classifier(logits.get<nv_bfloat16>(), losses.get<float>(), dloss, targets.get<int>(), count_ptr, BT, V, P, write_dlogits, stream);
    } else {
        throw std::runtime_error("fused_classifier: unsupported dtype");
    }
}

void fused_classifier(Tensor& logits, Tensor& losses,
                      float dloss, const Tensor& targets, Tensor* valid_token_count,
                      Tensor* correct_count,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream) {
    int* count_ptr = valid_token_count ? valid_token_count->get<int>() : nullptr;
    int* correct_ptr = correct_count ? correct_count->get<int>() : nullptr;
    if(logits.DType == ETensorDType::FP32) {
        fused_classifier(logits.get<float>(), losses.get<float>(), dloss, targets.get<int>(), count_ptr, correct_ptr, BT, V, P, write_dlogits, stream);
    } else if(logits.DType == ETensorDType::BF16) {
        fused_classifier(logits.get<nv_bfloat16>(), losses.get<float>(), dloss, targets.get<int>(), count_ptr, correct_ptr, BT, V, P, write_dlogits, stream);
    } else {
        throw std::runtime_error("fused_classifier: unsupported dtype");
    }
}

void fused_cross_entropy_forward(Tensor& logits, Tensor& losses, Tensor* logsumexp,
                                 const Tensor& targets, Tensor* valid_token_count,
                                 Tensor* correct_count,
                                 int BT, int V, int P, cudaStream_t stream) {
    float* lse_ptr = logsumexp ? logsumexp->get<float>() : nullptr;
    int* count_ptr = valid_token_count ? valid_token_count->get<int>() : nullptr;
    int* correct_ptr = correct_count ? correct_count->get<int>() : nullptr;
    if (logits.DType == ETensorDType::FP32) {
        fused_cross_entropy_forward(logits.get<float>(), losses.get<float>(), lse_ptr,
                                    targets.get<int>(), count_ptr, correct_ptr,
                                    BT, V, P, stream);
    } else if (logits.DType == ETensorDType::BF16) {
        fused_cross_entropy_forward(logits.get<nv_bfloat16>(), losses.get<float>(), lse_ptr,
                                    targets.get<int>(), count_ptr, correct_ptr,
                                    BT, V, P, stream);
    } else {
        throw std::runtime_error("fused_cross_entropy_forward: unsupported dtype");
    }
}

void fused_cross_entropy_backward(Tensor& dlogits, const Tensor& logits, const Tensor* logsumexp,
                                  const Tensor& dloss, const Tensor& targets,
                                  int BT, int V, int P, cudaStream_t stream) {
    const float* lse_ptr = logsumexp ? logsumexp->get<float>() : nullptr;
    const float* dloss_ptr = dloss.get<float>();
    if (dlogits.DType == ETensorDType::FP32) {
        fused_cross_entropy_backward(dlogits.get<float>(), logits.get<float>(), lse_ptr,
                                     dloss_ptr, targets.get<int>(),
                                     BT, V, P, stream);
    } else if (dlogits.DType == ETensorDType::BF16) {
        fused_cross_entropy_backward(dlogits.get<nv_bfloat16>(), logits.get<nv_bfloat16>(), lse_ptr,
                                     dloss_ptr, targets.get<int>(),
                                     BT, V, P, stream);
    } else {
        throw std::runtime_error("fused_cross_entropy_backward: unsupported dtype");
    }
}

void chunked_cross_entropy_forward(Tensor& logits, Tensor& losses, Tensor* logsumexp,
                                   Tensor& chunk_logsumexp, const Tensor& targets,
                                   Tensor* valid_token_count, Tensor* correct_count,
                                   int BT, int V, int P, int n_chunks, cudaStream_t stream) {
    float* lse_ptr = logsumexp ? logsumexp->get<float>() : nullptr;
    if (!lse_ptr) {
        throw std::runtime_error("chunked_cross_entropy_forward: logsumexp buffer is required");
    }
    int* count_ptr = valid_token_count ? valid_token_count->get<int>() : nullptr;
    int* correct_ptr = correct_count ? correct_count->get<int>() : nullptr;
    if (logits.DType == ETensorDType::FP32) {
        chunked_cross_entropy_forward(logits.get<float>(), losses.get<float>(), lse_ptr,
                                      chunk_logsumexp.get<float>(), targets.get<int>(),
                                      count_ptr, correct_ptr,
                                      BT, V, P, n_chunks, stream);
    } else if (logits.DType == ETensorDType::BF16) {
        chunked_cross_entropy_forward(logits.get<nv_bfloat16>(), losses.get<float>(), lse_ptr,
                                      chunk_logsumexp.get<float>(), targets.get<int>(),
                                      count_ptr, correct_ptr,
                                      BT, V, P, n_chunks, stream);
    } else {
        throw std::runtime_error("chunked_cross_entropy_forward: unsupported dtype");
    }
}

void chunked_cross_entropy_backward(Tensor& dlogits, const Tensor& logits, const Tensor* logsumexp,
                                    const Tensor& dloss, const Tensor& targets,
                                    int BT, int V, int P, cudaStream_t stream) {
    const float* lse_ptr = logsumexp ? logsumexp->get<float>() : nullptr;
    if (!lse_ptr) {
        throw std::runtime_error("chunked_cross_entropy_backward: logsumexp buffer is required");
    }
    const float* dloss_ptr = dloss.get<float>();
    if (dlogits.DType == ETensorDType::FP32) {
        chunked_cross_entropy_backward(dlogits.get<float>(), logits.get<float>(), lse_ptr,
                                       dloss_ptr, targets.get<int>(),
                                       BT, V, P, stream);
    } else if (dlogits.DType == ETensorDType::BF16) {
        chunked_cross_entropy_backward(dlogits.get<nv_bfloat16>(), logits.get<nv_bfloat16>(), lse_ptr,
                                       dloss_ptr, targets.get<int>(),
                                       BT, V, P, stream);
    } else {
        throw std::runtime_error("chunked_cross_entropy_backward: unsupported dtype");
    }
}

/**
 * @brief Performs the forward pass of the encoder layer (embedding lookup + positional encoding).
 *
 * This function acts as a dispatcher that invokes the appropriate CUDA kernel implementation
 * based on the data type of the output tensor (FP32 or BF16). It computes the initial
 * hidden states by combining token embeddings and optional positional embeddings.
 *
 * @param out    [Output] The output tensor of shape (B, T, C). Must be pre-allocated.
 *               Supported data types: FP32, BF16.
 * @param inp    [Input] The input tensor of shape (B, T) containing integer token indices.
 *               Data type must be int32.
 * @param wte    [Input] The token embedding weight tensor of shape (V, C).
 *               Data type must match `out`.
 * @param wpe    [Input] Optional positional embedding weight tensor of shape (max_seq_len, C).
 *               If provided, data type must match `out`. If std::nullopt, positional embeddings are skipped.
 * @param B      Batch size.
 * @param T      Time step (sequence length).
 * @param C      Channel size (embedding dimension).
 * @param V      Vocabulary size.
 * @param stream The CUDA stream on which to execute the kernel.
 *
 * @throws std::runtime_error If the output tensor data type is not supported (neither FP32 nor BF16).
 */
void encoder_forward(Tensor& out, const Tensor& inp, const Tensor& wte, std::optional<Tensor> wpe, int B, int T, int C, int V, cudaStream_t stream) {
    if(out.DType == ETensorDType::FP32) {
        encoder_forward(out.get<float>(), inp.get<std::int32_t>(), wte.get<float>(), wpe.has_value() ? wpe->get<float>() : nullptr, B, T, C, V, stream);
    } else if(out.DType == ETensorDType::BF16) {
        encoder_forward(out.get<nv_bfloat16>(), inp.get<std::int32_t>(), wte.get<nv_bfloat16>(),  wpe.has_value() ? wpe->get<nv_bfloat16>() : nullptr, B, T, C, V, stream);
    } else {
        throw std::runtime_error("encoder_forward: unsupported dtype");
    }
}

/**
 * @brief Performs the backward pass for the encoder layer, computing gradients for word token embeddings.
 *
 * This function acts as a dispatcher that calls the appropriate template specialization of the
 * backward encoder kernel based on the data type of the gradient tensor (`dwte`). It supports
 * both FP32 and BF16 data types.
 *
 * @param dwte [Out] The gradient of the word token embeddings. Must be on the device.
 *                   Supported types: FP32, BF16.
 * @param scratch [In/Out] Scratch memory tensor used for intermediate calculations (e.g., atomic operations).
 * @param workload_indices [In] CPU tensor containing indices for workload distribution.
 *                              Must be on the host (Device == -1).
 * @param bucket_info [In] CPU tensor containing bucket information (packed as int4).
 *                         Must be on the host (Device == -1).
 * @param dout [In] The gradient of the output tensor flowing back from the next layer.
 * @param inp [In] The original input tensor (token indices) on the device.
 * @param inputs_cpu [In] The original input tensor (token indices) mirrored on the CPU.
 * @param B [In] Batch size.
 * @param T [In] Sequence length (Time steps).
 * @param C [In] Embedding dimension (Channels).
 * @param seed [In] Random seed used for stochastic operations (if any).
 * @param stream [In] The CUDA stream on which to execute the kernel.
 * @param sync_event [In] A CUDA event used for synchronization between streams.
 * @param copy_stream [In] A separate CUDA stream used for asynchronous memory copies.
 *
 * @throws std::logic_error If `dwte` has an unsupported data type (not FP32 or BF16).
 */
void encoder_backward(Tensor& dwte, Tensor& scratch,
                      Tensor& workload_indices, Tensor& bucket_info,
                      const Tensor& dout, const Tensor& inp, const Tensor& inputs_cpu,
                      int B, int T, int C, unsigned int seed, cudaStream_t stream, cudaEvent_t sync_event, cudaStream_t copy_stream) {
    assert(workload_indices.Device == -1);
    assert(bucket_info.Device == -1);
    if(dwte.DType == ETensorDType::FP32) {
        encoder_backward(dwte.get<float>(), scratch.get<int>(), workload_indices.get<int>(),
            (int4*)bucket_info.get<int>(), dout.get<float>(), inp.get<std::int32_t>(), inputs_cpu.get<std::int32_t>(),
            B, T, C, seed, stream, sync_event, copy_stream);
    } else if(dwte.DType == ETensorDType::BF16) {
        encoder_backward(dwte.get<nv_bfloat16>(), scratch.get<int>(), workload_indices.get<int>(),
            (int4*)bucket_info.get<int>(), dout.get<nv_bfloat16>(), inp.get<std::int32_t>(), inputs_cpu.get<std::int32_t>(),
            B, T, C, seed, stream, sync_event, copy_stream);
    } else {
        throw std::logic_error("encoder_backward: unsupported dtype");
    }
}

/**
 * @brief Computes the squared global L2 norm of a tensor's values.
 *
 * This function acts as a dispatcher that invokes the appropriate CUDA kernel
 * based on the data type of the input tensor (FP32 or BF16). The result is
 * stored as a float in the output tensor.
 *
 * @param[out] out The output tensor where the computed squared norm will be stored.
 *                 Must be capable of storing a float value.
 * @param[in] values The input tensor containing the values to be normalized.
 *                   Supported data types are ETensorDType::FP32 and ETensorDType::BF16.
 * @param[in] count The number of elements in the input tensor to process.
 * @param[in] dp The CUDA device properties, used to optimize kernel launch parameters.
 * @param[in] stream The CUDA stream on which to execute the kernel.
 *
 * @throws std::logic_error If the input tensor's data type is not supported.
 */
void global_norm_squared(Tensor& out, const Tensor& values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream) {
    if(values.DType == ETensorDType::FP32) {
        global_norm_squared(out.get<float>(), values.get<float>(), count, dp, stream);
    } else if(values.DType == ETensorDType::BF16) {
        global_norm_squared(out.get<float>(), values.get<nv_bfloat16>(), count, dp, stream);
    } else {
        throw std::logic_error("global_norm_squared: unsupported dtype");
    }
}


/**
 * @brief Transposes a 2D tensor asynchronously on a CUDA stream.
 *
 * This function acts as a dispatcher that invokes the appropriate type-specific
 * transpose kernel based on the data type of the destination tensor. It supports
 * floating-point 32-bit, BFloat16, and FP8 (E4M3 and E5M2) formats.
 *
 * @param dst The destination tensor where the transposed data will be stored.
 *            Must have the same dimensions (swapped) and data type as src.
 * @param src The source tensor to be transposed.
 * @param rows The number of rows in the source tensor.
 * @param cols The number of columns in the source tensor.
 * @param stream The CUDA stream on which the operation will be executed.
 *
 * @throws std::logic_error If the destination tensor's data type is not supported
 *                          (i.e., not FP32, BF16, FP8_E4M3, or FP8_E5M2).
 */
void transpose(Tensor& dst, const Tensor& src, int rows, int cols, cudaStream_t stream) {
    if(dst.DType == ETensorDType::FP32) {
        transpose(dst.get<float>(), src.get<float>(), rows, cols, stream);
    } else if(dst.DType == ETensorDType::BF16) {
        transpose(dst.get<nv_bfloat16>(), src.get<nv_bfloat16>(), rows, cols, stream);
    } else if(dst.DType == ETensorDType::FP8_E4M3) {
        transpose(dst.get<__nv_fp8_e4m3>(), src.get<__nv_fp8_e4m3>(), rows, cols, stream);
    }  else if(dst.DType == ETensorDType::FP8_E5M2) {
        transpose(dst.get<__nv_fp8_e5m2>(), src.get<__nv_fp8_e5m2>(), rows, cols, stream);
    } else {
        throw std::logic_error("transpose: unsupported dtype");
    }
}

/**
 * @brief Performs a stochastic rounding vector addition of two tensors.
 *
 * This function adds the `left` and `right` tensors, scales the result, and stores it in the `dest` tensor
 * using stochastic rounding. It acts as a dispatcher, invoking the appropriate type-specific implementation
 * (FP32 or BF16) based on the data type of the destination tensor.
 *
 * @param dest   [out] The destination tensor where the result will be stored.
 * @param left   [in]  The first input tensor.
 * @param right  [in]  The second input tensor.
 * @param scale  [in]  The scaling factor applied to the addition result.
 * @param nelem  [in]  The number of elements to process.
 * @param seed   [in]  The random seed used for stochastic rounding.
 * @param stream [in]  The CUDA stream on which the operation will be executed.
 *
 * @throws std::logic_error If the destination tensor data type is not FP32 or BF16.
 */
void vector_add_sr(Tensor& dest, const Tensor& left, const Tensor& right, float scale, long nelem, unsigned seed, cudaStream_t stream) {
    // Validate that nelem makes sense
    if (nelem < 0) {
        fprintf(stderr, "ERROR: negative nelem=%ld\n", nelem);
        throw std::runtime_error("vector_add_sr: negative nelem parameter");
    }
    if ((size_t)nelem != dest.nelem()) {
        fprintf(stderr, "WARNING: nelem parameter (%ld) doesn't match dest.nelem() (%zu)\n", nelem, dest.nelem());
    }

    if(dest.DType == ETensorDType::FP32) {
        vector_add_sr(dest.get<float>(), left.get<float>(), right.get<float>(), scale, nelem, seed, stream);
    } else if(dest.DType == ETensorDType::BF16) {
        vector_add_sr(dest.get<nv_bfloat16>(), left.get<nv_bfloat16>(), right.get<nv_bfloat16>(), scale, nelem, seed, stream);
    } else {
        throw std::logic_error("vector_add_sr: unsupported dtype");
    }
}

/**
 * @brief Adds a 2D slice from a source tensor to a destination tensor.
 *
 * This function dispatches the operation to the appropriate type-specific kernel
 * (FP32 or BF16) based on the data type of the destination tensor. It ensures
 * that both tensors have matching data types before proceeding.
 *
 * @param dst The destination tensor where the slice will be added.
 * @param src The source tensor containing the data to add.
 * @param rows The number of rows to process.
 * @param dst_cols The total number of columns in the destination tensor.
 * @param src_cols The total number of columns in the source tensor.
 * @param dst_col_offset The column offset in the destination tensor where the addition begins.
 * @param stream The CUDA stream to execute the kernel on.
 *
 * @throws std::logic_error If the data types of dst and src do not match.
 * @throws std::logic_error If the data type is not supported (only FP32 and BF16 are supported).
 */
void add_2d_slice(Tensor& dst, const Tensor& src, long rows, long dst_cols, long src_cols, long dst_col_offset, cudaStream_t stream) {
    if (dst.DType != src.DType) {
        fprintf(stderr, "[DEBUG] add_2d_slice dtype mismatch: dst.DType=%d src.DType=%d\n", (int)dst.DType, (int)src.DType);
        fprintf(stderr, "[DEBUG] dst shape: [%ld, %ld] src shape: [%ld, %ld]\n", dst.Sizes[0], dst.Sizes[1], src.Sizes[0], src.Sizes[1]);
        throw std::logic_error("add_2d_slice: dtype mismatch");
    }
    if (dst.DType == ETensorDType::FP32) {
        add_2d_slice(dst.get<float>(), src.get<float>(), rows, dst_cols, src_cols, dst_col_offset, stream);
    } else if (dst.DType == ETensorDType::BF16) {
        add_2d_slice(dst.get<nv_bfloat16>(), src.get<nv_bfloat16>(), rows, dst_cols, src_cols, dst_col_offset, stream);
    } else {
        throw std::logic_error("add_2d_slice: unsupported dtype");
    }
}

/**
 * @brief Performs a stochastic rounding vector reduction operation.
 *
 * This function dispatches the reduction operation to the appropriate type-specific implementation
 * (FP32 or BF16) based on the destination tensor's data type. It reduces elements from the source
 * tensor into the destination tensor, applying a scaling factor and stochastic rounding.
 *
 * @param dest The destination tensor where the results will be stored. Must be of type FP32 or BF16.
 * @param src The source tensor containing the data to be reduced.
 * @param scale A scaling factor applied to the source elements during reduction.
 * @param n_shards The number of shards involved in the reduction (used for normalization or partitioning).
 * @param skip The number of elements to skip between reductions (stride).
 * @param nelem The total number of elements to process.
 * @param accumulate If true, adds the reduced values to the existing values in `dest`. If false, overwrites `dest`.
 * @param seed The random seed used for stochastic rounding.
 * @param stream The CUDA stream on which to execute the kernel.
 */
void vector_reduce_sr(Tensor& dest, const Tensor& src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream) {
    if(dest.DType == ETensorDType::FP32) {
        vector_reduce_sr(dest.get<float>(), src.get<float>(), scale, n_shards, skip, nelem, accumulate, seed, stream);
    } else if(dest.DType == ETensorDType::BF16) {
        vector_reduce_sr(dest.get<nv_bfloat16>(), src.get<nv_bfloat16>(), scale, n_shards, skip, nelem, accumulate, seed, stream);
    }
}

/**
 * @brief Computes the absolute maximum value of a tensor and stores it in a scale variable.
 *
 * This function dispatches the computation to the appropriate type-specific implementation
 * based on the data type of the input tensor (FP32 or BF16).
 *
 * @param scale Pointer to the float where the result (absolute maximum value) will be stored.
 * @param in The input tensor containing the data to be processed.
 * @param N The number of elements in the input tensor to process.
 * @param dp The CUDA device properties, used for kernel configuration.
 * @param stream The CUDA stream on which the operation will be executed.
 *
 * @throws std::logic_error If the input tensor's data type is not supported (neither FP32 nor BF16).
 */
void abs_max(float* scale, const Tensor& in, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    if (in.DType == ETensorDType::FP32) {
        abs_max(scale, in.get<float>(), N, dp, stream);
    } else if (in.DType == ETensorDType::BF16) {
        abs_max(scale, in.get<nv_bfloat16>(), N, dp, stream);
    } else if (in.DType == ETensorDType::FP8_E4M3) {
        // FP8 tensors should already have their absmax/scale computed during quantization
        // If we're being asked to compute absmax on FP8, something is wrong in the caller
        throw std::logic_error("absmax_scale: FP8 tensors should not need absmax recomputation. "
                                "This likely indicates an issue with weight quantization path.");
    } else {
        std::string msg = "absmax_scale: unsupported dtype: " + std::to_string(static_cast<int>(in.DType));
        throw std::logic_error(msg);
    }
}

/**
 * @brief Quantizes a tensor using absolute maximum scaling.
 *
 * This function acts as a dispatcher that selects the appropriate template specialization
 * based on the data types of the input and output tensors. It supports quantization from
 * FP32 or BF16 source formats to BF16, FP8 (E4M3/E5M2), or INT8 destination formats.
 *
 * @param out The output tensor where the quantized data will be stored.
 * @param scale_ptr Pointer to the memory location where the calculated scale factor will be stored.
 * @param in The input tensor containing the data to be quantized.
 * @param abs_max Pointer to the absolute maximum value of the input tensor (used for scaling).
 * @param N The number of elements in the tensors.
 * @param dp The CUDA device properties, used to determine hardware capabilities for specific quantization paths.
 * @param stream The CUDA stream on which the quantization kernel will be executed.
 *
 * @throws std::logic_error If the combination of input and output tensor data types is not supported.
 */
void quantize_with_abs_max(Tensor& out, float* scale_ptr, const Tensor& in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream) {
    if (in.DType == ETensorDType::FP32) {
        if (out.DType == ETensorDType::BF16) {
            quantize_with_abs_max(out.get<nv_bfloat16>(), scale_ptr, in.get<float>(), abs_max, N, dp, stream);
        } else if (out.DType == ETensorDType::FP8_E4M3) {
            quantize_with_abs_max(out.get<__nv_fp8_e4m3>(), scale_ptr, in.get<float>(), abs_max, N, dp, stream);
        } else if (out.DType == ETensorDType::FP8_E5M2) {
            quantize_with_abs_max(out.get<__nv_fp8_e5m2>(), scale_ptr, in.get<float>(), abs_max, N, dp, stream);
        } else if (out.DType == ETensorDType::INT8) {
            quantize_with_abs_max(out.get<int8_t>(), scale_ptr, in.get<float>(), abs_max, N, dp, stream);
        } else {
            throw std::logic_error("quantize_with_abs_max: unsupported dtype");
        }
    } else if (in.DType == ETensorDType::BF16) {
        if (out.DType == ETensorDType::FP8_E4M3) {
            quantize_with_abs_max(out.get<__nv_fp8_e4m3>(), scale_ptr, in.get<nv_bfloat16>(), abs_max, N, dp, stream);
        } else if (out.DType == ETensorDType::FP8_E5M2) {
            quantize_with_abs_max(out.get<__nv_fp8_e5m2>(), scale_ptr, in.get<nv_bfloat16>(), abs_max, N, dp, stream);
        } else if (out.DType == ETensorDType::INT8) {
            quantize_with_abs_max(out.get<int8_t>(), scale_ptr, in.get<nv_bfloat16>(), abs_max, N, dp, stream);
        } else {
            throw std::logic_error("quantize_with_abs_max: unsupported dtype");
        }
    } else {
        throw std::logic_error("quantize_with_abs_max: unsupported dtype");
    }
}

/**
 * @brief Quantizes and transposes a tensor using an absolute maximum value for scaling.
 *
 * This function acts as a dispatcher that selects the appropriate kernel implementation based on the
 * data types of the input and output tensors. It supports various combinations of source and destination
 * types, including FP32 to BF16/INT8/FP8 and BF16 to INT8/FP8.
 *
 * @param out The output tensor where the quantized and transposed data will be stored.
 * @param scale_ptr Pointer to the memory location where the calculated scale factor will be stored.
 * @param in The input tensor containing the original data.
 * @param abs_max Pointer to the absolute maximum value used for quantization scaling.
 * @param rows The number of rows in the input matrix.
 * @param cols The number of columns in the input matrix.
 * @param dp The CUDA device properties, used to optimize kernel launch parameters.
 * @param stream The CUDA stream on which the operation will be executed.
 *
 * @throws std::logic_error If the combination of input and output tensor data types is not supported.
 *
 * @note Supported Type Combinations:
 * - FP32 -> BF16
 * - FP32 -> INT8
 * - BF16 -> INT8
 * - FP32 -> FP8_E4M3
 * - BF16 -> FP8_E4M3
 */
void quantize_and_transpose_with_abs_max(Tensor& out, float* scale_ptr, const Tensor& in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream) {
    if(out.DType == ETensorDType::BF16 && in.DType == ETensorDType::FP32) {
        quantize_and_transpose_with_abs_max(out.get<nv_bfloat16>(), scale_ptr, in.get<float>(), abs_max, rows, cols, dp, stream);
    } else if(out.DType == ETensorDType::INT8 && in.DType == ETensorDType::FP32) {
        quantize_and_transpose_with_abs_max(out.get<std::int8_t>(), scale_ptr, in.get<float>(), abs_max, rows, cols, dp, stream);
    } else if(out.DType == ETensorDType::INT8 && in.DType == ETensorDType::BF16) {
        quantize_and_transpose_with_abs_max(out.get<std::int8_t>(), scale_ptr, in.get<nv_bfloat16>(), abs_max, rows, cols, dp, stream);
    } else if(out.DType == ETensorDType::FP8_E4M3 && in.DType == ETensorDType::FP32) {
        quantize_and_transpose_with_abs_max(out.get<__nv_fp8_e4m3>(), scale_ptr, in.get<float>(), abs_max, rows, cols, dp, stream);
    } else if(out.DType == ETensorDType::FP8_E4M3 && in.DType == ETensorDType::BF16) {
        quantize_and_transpose_with_abs_max(out.get<__nv_fp8_e4m3>(), scale_ptr, in.get<nv_bfloat16>(), abs_max, rows, cols, dp, stream);
    } else {
        throw std::logic_error("Invalid DType combination");
    }
}

/**
 * @brief Fills a tensor with random numbers drawn from a normal distribution.
 *
 * This function dispatches to the appropriate type-specific implementation based on the
 * data type of the destination tensor (FP32 or BF16).
 *
 * @param dest The destination tensor to be filled. Must have DType set to either FP32 or BF16.
 * @param count The number of elements to generate.
 * @param mean The mean value of the normal distribution.
 * @param std The standard deviation of the normal distribution.
 * @param seed The seed for the random number generator.
 * @param subsequence The subsequence index for the random number generator (useful for parallel generation).
 * @param stream The CUDA stream on which the kernel will be executed.
 *
 * @throws std::logic_error If the destination tensor's data type is not supported (i.e., not FP32 or BF16).
 */
void fill_normal(Tensor& dest, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence, cudaStream_t stream) {
    if (dest.DType == ETensorDType::FP32) {
        fill_normal(dest.get<float>(), count, mean, std, seed, subsequence, stream);
    } else if (dest.DType == ETensorDType::BF16) {
        fill_normal(dest.get<nv_bfloat16>(), count, mean, std, seed, subsequence, stream);
    } else {
        throw std::logic_error("fill_normal: unsupported dtype");
    }
}

/**
 * @brief Fills a tensor with a constant floating-point value.
 *
 * This function dispatches the operation to the appropriate type-specific kernel
 * based on the data type of the destination tensor. Currently supports FP32 and BF16.
 *
 * @param dest The destination tensor to be filled.
 * @param value The constant value to fill the tensor with.
 * @param count The number of elements to fill.
 * @param stream The CUDA stream to execute the kernel on.
 *
 * @throws std::logic_error If the destination tensor has an unsupported data type.
 */
void fill_constant(Tensor& dest, float value, std::size_t count, cudaStream_t stream) {
    if (dest.DType == ETensorDType::FP32) {
        fill_constant(dest.get<float>(), value, count, stream);
    } else if (dest.DType == ETensorDType::BF16) {
        fill_constant(dest.get<nv_bfloat16>(), static_cast<nv_bfloat16>(value), count, stream);
    } else {
        throw std::logic_error("fill_constant: unsupported dtype");
    }
}

/**
 * @brief Performs matrix multiplication (C = A * B + Bias) with support for various data types and mixed-precision operations.
 *
 * This function acts as a high-level dispatcher that selects the appropriate specialized `matmul` implementation
 * based on the data types of the input tensors (A, B) and the output tensor (C). It handles pointer casting,
 * workspace management, and optional bias integration.
 *
 * Supported Data Type Combinations (C, A, B):
 * - FP32, FP32, FP32
 * - FP32, BF16, BF16
 * - FP32, FP8_E4M3, FP8_E4M3 (Bias can be BF16 or FP32)
 * - BF16, FP8_E4M3, FP8_E4M3
 * - BF16, FP8_E4M3, FP8_E5M2
 * - BF16, BF16, BF16
 *
 * @param c          [Output] The result tensor. Its data type determines the accumulation precision.
 * @param a          [Input] The left operand tensor.
 * @param b          [Input] The right operand tensor.
 * @param bias       [Input, Optional] An optional bias tensor to add to the result. If present, its type must match the expected bias type for the specific kernel.
 * @param scale_a    [Input] Pointer to the scaling factor for tensor A (used for quantized types like FP8).
 * @param scale_b    [Input] Pointer to the scaling factor for tensor B (used for quantized types like FP8).
 * @param handle     [Input] The cuBLASLt handle used to launch the operation.
 * @param workspace  [Input/Output] A tensor serving as scratch memory for the cuBLASLt operation.
 * @param M          [Input] The number of rows in matrices A and C.
 * @param N          [Input] The number of columns in matrices B and C.
 * @param K          [Input] The number of columns in A and rows in B.
 * @param mode       [Input] The transposition mode for the operation (e.g., Transpose, NoTranspose).
 * @param accumulate [Input] If true, accumulates the result into C (C += A * B). If false, overwrites C.
 * @param stream     [Input] The CUDA stream on which to execute the kernel.
 *
 * @throws std::logic_error If the combination of tensor data types is not supported.
 */
void matmul(Tensor& c, const Tensor& a, const Tensor& b, std::optional<Tensor> bias,
            const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, Tensor& workspace,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream) {
    std::byte* ws = workspace.get<std::byte>();
    std::size_t ws_size = workspace.bytes();

    if(c.DType == ETensorDType::FP32 && a.DType == ETensorDType::FP32) {
        float* bias_ptr = bias.has_value() ? bias.value().get<float>() : nullptr;
        matmul(c.get<float>(), a.get<float>(), b.get<float>(), bias_ptr, scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream);
    } else if(c.DType == ETensorDType::FP32 && a.DType == ETensorDType::BF16) {
        // Note: bias is expected to be FP32 when output is FP32
        // If bias is BF16, we skip it (TODO: add support for BF16 bias with FP32 output)
        float* bias_ptr = nullptr;
        if(bias.has_value()) {
            if(bias.value().DType == ETensorDType::FP32) {
                bias_ptr = bias.value().get<float>();
            } else {
                // Skip BF16 bias for now - this shouldn't happen in practice
                fprintf(stderr, "[WARNING] matmul: FP32 output with BF16 bias - skipping bias\n");
            }
        }
        matmul(c.get<float>(), a.get<nv_bfloat16>(), b.get<nv_bfloat16>(), bias_ptr, scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream);
    } else if(c.DType == ETensorDType::FP32 && a.DType == ETensorDType::FP8_E4M3) {
        if(bias.has_value()) {
            if(bias.value().DType == ETensorDType::BF16) {
                matmul(c.get<float>(), a.get<__nv_fp8_e4m3>(), b.get<__nv_fp8_e4m3>(), bias->get<nv_bfloat16>(), scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream);
            } else {
                matmul(c.get<float>(), a.get<__nv_fp8_e4m3>(), b.get<__nv_fp8_e4m3>(), bias->get<float>(), scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream);
            }
        } else {
            matmul(c.get<float>(), a.get<__nv_fp8_e4m3>(), b.get<__nv_fp8_e4m3>(), (nv_bfloat16*)nullptr, scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream);
        }
    } else if(c.DType == ETensorDType::BF16 && a.DType == ETensorDType::FP8_E4M3 && b.DType == ETensorDType::FP8_E4M3) {
        nv_bfloat16* bias_ptr = bias.has_value() ? bias.value().get<nv_bfloat16>() : nullptr;
        matmul(c.get<nv_bfloat16>(), a.get<__nv_fp8_e4m3>(), b.get<__nv_fp8_e4m3>(), bias_ptr, scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream);
    } else if(c.DType == ETensorDType::BF16 && a.DType == ETensorDType::FP8_E4M3 && b.DType == ETensorDType::FP8_E5M2) {
        nv_bfloat16* bias_ptr = bias.has_value() ? bias.value().get<nv_bfloat16>() : nullptr;
        matmul(c.get<nv_bfloat16>(), a.get<__nv_fp8_e4m3>(), b.get<__nv_fp8_e5m2>(), bias_ptr, scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream);
    } else if(c.DType == ETensorDType::BF16) {
        nv_bfloat16* bias_ptr = bias.has_value() ? bias.value().get<nv_bfloat16>() : nullptr;
        matmul(c.get<nv_bfloat16>(), a.get<nv_bfloat16>(), b.get<nv_bfloat16>(), bias_ptr, scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, stream);
    } else {
        fprintf(stderr, "[DEBUG] matmul error: c.DType=%d a.DType=%d b.DType=%d\n",
                (int)c.DType, (int)a.DType, (int)b.DType);
        throw std::logic_error("matmul_forward: invalid DType combination");
    }
}

/**
 * @brief Performs Tensor-based matrix multiplication with explicit alpha/beta: C = alpha * (A @ B) + beta * C
 *
 * This overload allows fusing scaling into the matmul epilogue for better performance.
 * Useful for LoRA backward pass where scaling factors need to be applied.
 *
 * @param c [in,out] Output tensor C.
 * @param a [in] Input tensor A.
 * @param b [in] Input tensor B.
 * @param bias [in] Optional bias tensor.
 * @param scale_a [in] Scaling factor for A (FP8 only).
 * @param scale_b [in] Scaling factor for B (FP8 only).
 * @param handle [in] cuBLASLt handle.
 * @param workspace [in] Workspace tensor.
 * @param M [in] Number of rows in C.
 * @param N [in] Number of columns in C.
 * @param K [in] Inner dimension.
 * @param mode [in] Transpose mode.
 * @param alpha [in] Output scaling factor.
 * @param beta [in] Accumulation factor.
 * @param stream [in] CUDA stream.
 */
void matmul(Tensor& c, const Tensor& a, const Tensor& b, std::optional<Tensor> bias,
            const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, Tensor& workspace,
            int M, int N, int K, EMMTranspose mode, float alpha, float beta, cudaStream_t stream) {
    std::byte* ws = workspace.get<std::byte>();
    std::size_t ws_size = workspace.bytes();

    // Currently only BF16 is supported with explicit alpha/beta
    // (this is the primary use case for LoRA backward)
    if(c.DType == ETensorDType::BF16 && a.DType == ETensorDType::BF16 && b.DType == ETensorDType::BF16) {
        nv_bfloat16* bias_ptr = bias.has_value() ? bias.value().get<nv_bfloat16>() : nullptr;
        matmul(c.get<nv_bfloat16>(), a.get<nv_bfloat16>(), b.get<nv_bfloat16>(), bias_ptr, scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, alpha, beta, stream);
    } else {
        // For other dtypes, fall back to the accumulate-based API
        // Note: this loses the ability to use non-1.0 alpha values
        bool accumulate = (beta != 0.0f);
        matmul(c, a, b, bias, scale_a, scale_b, handle, workspace, M, N, K, mode, accumulate, stream);
        // If alpha != 1.0, we'd need a separate scale kernel, but this shouldn't happen in practice
        if (alpha != 1.0f && alpha != 0.0f) {
            fprintf(stderr, "[WARNING] matmul with alpha!=1.0 not supported for dtype %d, scaling skipped\n", (int)c.DType);
        }
    }
}

/**
 * @brief Performs a strided matrix multiplication operation (C = A * B + Bias) with optional accumulation and scaling.
 *
 * This function acts as a high-level dispatcher that selects the appropriate implementation based on the
 * data types of the input tensors (FP32 or BF16). It handles workspace management and validates the
 * leading dimension of C (ldc).
 *
 * @param c [in,out] The output tensor C. Must match the data type of A and B.
 * @param a [in] The input tensor A.
 * @param b [in] The input tensor B.
 * @param bias [in] An optional bias tensor to be added to the result. If present, its data type must match C.
 * @param scale_a [in] Pointer to a scaling factor for matrix A (can be nullptr depending on implementation).
 * @param scale_b [in] Pointer to a scaling factor for matrix B (can be nullptr depending on implementation).
 * @param handle [in] The cuBLAS Lt handle used for the operation.
 * @param workspace [in,out] A tensor used as scratch memory for the matrix multiplication.
 * @param M [in] The number of rows in matrix A and C.
 * @param N [in] The number of columns in matrix B and C.
 * @param K [in] The number of columns in A and rows in B.
 * @param mode [in] The transposition mode for the operation (e.g., transpose A, transpose B).
 * @param accumulate [in] If true, accumulates the result into C (C += A * B). If false, overwrites C.
 * @param ldc [in] The leading dimension of matrix C. Must be greater than 0.
 * @param stream [in] The CUDA stream on which to execute the kernel.
 *
 * @throws std::logic_error If `ldc` is less than or equal to 0.
 * @throws std::logic_error If the combination of tensor data types is not supported (currently supports all-FP32 or all-BF16).
 */
void matmul_strided_c(Tensor& c, const Tensor& a, const Tensor& b, std::optional<Tensor> bias,
                      const float* scale_a, const float* scale_b,
                      cublasLtHandle_t handle, Tensor& workspace,
                      int M, int N, int K, EMMTranspose mode, bool accumulate, int ldc, cudaStream_t stream) {
    std::byte* ws = workspace.get<std::byte>();
    std::size_t ws_size = workspace.bytes();
    if (ldc <= 0) {
        throw std::logic_error("matmul_strided_c: invalid ldc");
    }
    if (c.DType == ETensorDType::FP32 && a.DType == ETensorDType::FP32) {
        float* bias_ptr = bias.has_value() ? bias.value().get<float>() : nullptr;
        matmul_strided_c(c.get<float>(), a.get<float>(), b.get<float>(), bias_ptr, scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, ldc, stream);
    } else if (c.DType == ETensorDType::BF16 && a.DType == ETensorDType::BF16 && b.DType == ETensorDType::BF16) {
        nv_bfloat16* bias_ptr = bias.has_value() ? bias.value().get<nv_bfloat16>() : nullptr;
        matmul_strided_c(c.get<nv_bfloat16>(), a.get<nv_bfloat16>(), b.get<nv_bfloat16>(), bias_ptr, scale_a, scale_b, handle, ws, ws_size, M, N, K, mode, accumulate, ldc, stream);
    } else {
        throw std::logic_error("matmul_strided_c: unsupported dtype combination");
    }
}

/**
 * @brief Computes the backward pass for the bias term.
 *
 * This function dispatches the backward bias calculation to the appropriate kernel implementation
 * based on the data types of the input tensors (`dbias` and `dout`). It supports various combinations
 * of floating-point types, including FP32, BF16, and FP8 variants.
 *
 * @param dbias [in, out] The gradient of the bias tensor. This tensor will be updated with the computed gradients.
 *                        Supported types: FP32, BF16.
 * @param dout [in] The gradient of the output tensor (upstream gradient).
 *                  Supported types: FP32, BF16, FP8_E4M3, FP8_E5M2.
 * @param scale_a [in] Pointer to the scaling factor for input A (optional/context-dependent).
 * @param scale_b [in] Pointer to the scaling factor for input B (optional/context-dependent).
 * @param dbias_buffer [in, out] A temporary buffer tensor used for intermediate accumulation, typically in FP32.
 * @param B [in] Batch size dimension.
 * @param T [in] Time/Sequence length dimension.
 * @param OC [in] Output Channel (feature) dimension.
 * @param dp [in] CUDA device properties, used to optimize kernel launch parameters.
 * @param stream [in] The CUDA stream on which to execute the kernels.
 *
 * @throws std::logic_error If the combination of `dbias` and `dout` data types is not supported.
 *
 * @note The specific type combinations supported are:
 *       - dbias: FP32, dout: FP32
 *       - dbias: BF16, dout: BF16
 *       - dbias: BF16, dout: FP8_E4M3
 *       - dbias: BF16, dout: FP8_E5M2
 */
void backward_bias(Tensor& dbias, const Tensor& dout, const float* scale_a, const float* scale_b, Tensor& dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream) {
    if(dbias.DType == ETensorDType::FP32 && dout.DType == ETensorDType::FP32) {
        backward_bias(dbias.get<float>(), dout.get<float>(), scale_a, scale_b, dbias_buffer.get<float>(), B, T, OC, dp, stream);
    } else if(dbias.DType == ETensorDType::BF16 && dout.DType == ETensorDType::BF16) {
        backward_bias(dbias.get<nv_bfloat16>(), dout.get<nv_bfloat16>(), scale_a, scale_b, dbias_buffer.get<float>(), B, T, OC, dp, stream);
    } else if(dbias.DType == ETensorDType::BF16 && dout.DType == ETensorDType::FP8_E4M3) {
        backward_bias(dbias.get<nv_bfloat16>(), dout.get<__nv_fp8_e4m3>(), scale_a, scale_b, dbias_buffer.get<float>(), B, T, OC, dp, stream);
    }  else if(dbias.DType == ETensorDType::BF16 && dout.DType == ETensorDType::FP8_E5M2) {
        backward_bias(dbias.get<nv_bfloat16>(), dout.get<__nv_fp8_e5m2>(), scale_a, scale_b, dbias_buffer.get<float>(), B, T, OC, dp, stream);
    } else {
        throw std::logic_error("backward_bias: unsupported dtype");
    }
}
