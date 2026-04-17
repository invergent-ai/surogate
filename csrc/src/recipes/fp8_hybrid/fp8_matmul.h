// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_RECIPES_FP8_HYBRID_FP8_MATMUL_H
#define SUROGATE_SRC_RECIPES_FP8_HYBRID_FP8_MATMUL_H

#include <optional>
#include <stdexcept>
#include <cuda_runtime.h>

#include "kernels/kernels.h"
#include "utilities/tensor.h"

namespace recipes::fp8_hybrid {

/**
 * @brief FP8 forward matmul (TN layout) with E4M3 quantization
 *
 * Quantizes input to FP8 E4M3 and performs matmul using FP8 Tensor Cores.
 * Supports both JIT (just-in-time) scaling and delayed scaling (TransformerEngine-style).
 *
 * The FP8 quantized data is transient and NOT saved for backward - backward uses
 * BF16 cached activations for training stability.
 *
 * @tparam RunState Run state type providing cuBLAS handles and temp allocation
 * @param out Output tensor (BF16/FP32)
 * @param inp Input tensor (BF16) - the full-precision input
 * @param inp_fp8 FP8 buffer for quantized input (transient, shared across layers)
 * @param weight Weight tensor (can be BF16 or FP8)
 * @param bias Optional bias tensor
 * @param rs Run state containing cuBLAS handle and workspace
 * @param B Batch size
 * @param T Sequence length
 * @param C Input channels (K dimension)
 * @param OC Output channels (M dimension)
 * @param stream CUDA stream
 * @param cached_fp8_weight Optional pre-cached FP8 weight
 * @param delayed_quantizer_idx Quantizer index for delayed scaling (-1 = use JIT scaling)
 */
template <typename RunState>
inline void forward_matmul(Tensor& out,
                           Tensor& inp,
                           Tensor& inp_fp8,
                           Tensor& weight,
                           std::optional<Tensor> bias,
                           RunState& rs,
                           int B,
                           int T,
                           int C,
                           int OC,
                           cudaStream_t stream,
                           const Tensor* cached_fp8_weight = nullptr,
                           int delayed_quantizer_idx = -1) {
    if (!inp_fp8.Data) {
        throw std::runtime_error("fp8_hybrid::forward_matmul: FP8 forward buffer not allocated");
    }
    if (!inp_fp8.abs_max() || !inp_fp8.scale()) {
        throw std::runtime_error("fp8_hybrid::forward_matmul: FP8 buffer missing abs_max/scale Stats");
    }

    // Quantize input activation to FP8 E4M3
    // Use delayed scaling if enabled and quantizer index is valid
    if (delayed_quantizer_idx >= 0 && rs.has_fp8_delayed_scaling()) {
        auto& scaling_state = rs.fp8_scaling_state();
        auto qidx = static_cast<QuantizerIndex>(delayed_quantizer_idx);

        // Quantize using delayed (pre-computed) scale, record amax for next iteration
        quantize_with_delayed_scale(inp_fp8.template get<__nv_fp8_e4m3>(),
                                    scaling_state.get_recorded_amax_ptr(qidx),  // Output: record amax
                                    inp_fp8.scale(),                            // Output: inverse scale for matmul
                                    inp.template get<nv_bfloat16>(),
                                    scaling_state.get_scale(qidx),  // Input: delayed scale from history
                                    (long)B * T * C,
                                    rs.DeviceProp,
                                    stream);
    } else {
        // JIT scaling: compute abs_max and scale on-the-fly
        quantize_with_abs_max(inp_fp8, inp_fp8.scale(), inp, inp_fp8.abs_max(), (long)B * T * C, rs.DeviceProp, stream);
    }

    // FP8 matmul requires both operands to be FP8 for cuBLASLt Tensor Core speedup.
    if (weight.DType == ETensorDType::FP8_E4M3) {
        // Weight already FP8 (persistent quantization mode)
        matmul(out,
               weight,
               inp_fp8,
               bias,
               weight.scale(),
               inp_fp8.scale(),
               rs.CublasLtHandle,
               rs.CuBlasWorkspace,
               OC,
               B * T,
               C,
               EMMTranspose::TN,
               /*accumulate=*/false,
               stream);
    } else if (cached_fp8_weight && cached_fp8_weight->Data) {
        // Use pre-computed FP8 weight from cache
        float* weight_scale = const_cast<Tensor*>(cached_fp8_weight)->scale();
        matmul(out,
               *cached_fp8_weight,
               inp_fp8,
               bias,
               weight_scale,
               inp_fp8.scale(),
               rs.CublasLtHandle,
               rs.CuBlasWorkspace,
               OC,
               B * T,
               C,
               EMMTranspose::TN,
               /*accumulate=*/false,
               stream);
    } else if (weight.DType == ETensorDType::BF16) {
        // Weight is BF16 and no cache - quantize to FP8 on-the-fly for forward pass.
        Tensor weight_fp8 = rs.temp_alloc(ETensorDType::FP8_E4M3, {OC, C}, "weight_fp8");
        Tensor weight_stats = rs.temp_alloc(ETensorDType::FP32, {2}, "weight_stats");  // [abs_max, scale]
        weight_fp8.Stats = reinterpret_cast<float*>(weight_stats.Data);

        // Compute abs_max and quantize weight to FP8
        abs_max(weight_fp8.abs_max(), weight, (long)OC * C, rs.DeviceProp, stream);
        quantize_with_abs_max(weight_fp8,
                              weight_fp8.scale(),
                              weight,
                              weight_fp8.abs_max(),
                              (long)OC * C,
                              rs.DeviceProp,
                              stream);

        // Perform FP8 × FP8 matmul
        matmul(out,
               weight_fp8,
               inp_fp8,
               bias,
               weight_fp8.scale(),
               inp_fp8.scale(),
               rs.CublasLtHandle,
               rs.CuBlasWorkspace,
               OC,
               B * T,
               C,
               EMMTranspose::TN,
               /*accumulate=*/false,
               stream);

        rs.temp_free(weight_stats);
        rs.temp_free(weight_fp8);
    } else {
        // Unsupported weight dtype - fall back to standard matmul
        matmul(out,
               weight,
               inp,
               bias,
               nullptr,
               nullptr,
               rs.CublasLtHandle,
               rs.CuBlasWorkspace,
               OC,
               B * T,
               C,
               EMMTranspose::TN,
               /*accumulate=*/false,
               stream);
    }
}

/**
 * @brief FP8 HYBRID backward matmul (E4M3 weights × E5M2 gradients)
 *
 * Uses E4M3 weights (quantized on-the-fly from BF16) and E5M2 gradients for backward matmuls.
 * The HYBRID format uses:
 * - E4M3 (max=448, higher precision) for weights/activations
 * - E5M2 (max=57344, larger dynamic range) for gradients
 *
 * Computes:
 * - dinp = W^T @ dout  (E4M3 weight × E5M2 gradient)
 * - dW += inp^T @ dout (E4M3 activation × E5M2 gradient)
 *
 * @tparam RunState Run state type providing cuBLAS handles and temp allocation
 * @param dinp Output: gradient w.r.t. input activation
 * @param dweight Output: gradient w.r.t. weight (accumulated)
 * @param dbias Optional output: gradient w.r.t. bias
 * @param dout Input: upstream gradient (BF16)
 * @param dout_e5m2 Buffer for E5M2 quantized gradient
 * @param inp Input activation (BF16, from forward pass)
 * @param inp_fp8 Buffer for E4M3 quantized activation (may have forward abs_max)
 * @param weight BF16 weight (will be quantized to E4M3 on-the-fly)
 * @param bias_buffer Optional buffer for bias gradient computation
 * @param accumulate_gradient Whether to accumulate into dweight
 * @param rs Run state with cuBLAS handles
 * @param B Batch size
 * @param T Sequence length
 * @param C Input channels
 * @param OC Output channels
 * @param stream CUDA stream
 * @param skip_weight_grad Skip weight gradient computation (for LoRA-only mode)
 */
template <typename RunState>
inline void backward_matmul(Tensor& dinp,
                            Tensor& dweight,
                            std::optional<Tensor> dbias,
                            Tensor& dout,
                            Tensor& dout_e5m2,
                            Tensor& inp,
                            Tensor& inp_fp8,
                            Tensor& weight,
                            std::optional<Tensor> bias_buffer,
                            bool accumulate_gradient,
                            RunState& rs,
                            int B,
                            int T,
                            int C,
                            int OC,
                            cudaStream_t stream,
                            bool skip_weight_grad = false) {
    // Validate E5M2 gradient buffer
    if (!dout_e5m2.Data) {
        throw std::runtime_error("fp8_hybrid::backward_matmul: E5M2 gradient buffer not allocated");
    }
    if (!dout_e5m2.abs_max() || !dout_e5m2.scale()) {
        throw std::runtime_error("fp8_hybrid::backward_matmul: E5M2 buffer missing abs_max/scale Stats");
    }

    // Quantize upstream gradient to E5M2 (larger dynamic range for gradients)
    quantize_with_abs_max(dout_e5m2,
                          dout_e5m2.scale(),
                          dout,
                          dout_e5m2.abs_max(),
                          (long)B * T * OC,
                          rs.DeviceProp,
                          stream);

    // Get E4M3 weight for backward.
    Tensor weight_e4m3{};
    Tensor weight_stats{};
    bool weight_is_temp = false;
    if (weight.DType == ETensorDType::FP8_E4M3) {
        weight_e4m3 = weight;
        if (!weight_e4m3.scale()) {
            throw std::runtime_error("fp8_hybrid::backward_matmul: FP8 weight missing scale Stats");
        }
    } else {
        // Quantize BF16 weight to E4M3 on-the-fly
        weight_e4m3 = rs.temp_alloc(ETensorDType::FP8_E4M3, {OC, C}, "weight_e4m3");
        weight_stats = rs.temp_alloc(ETensorDType::FP32, {2}, "weight_stats");
        weight_e4m3.Stats = weight_stats.template get<float>();

        abs_max(weight_e4m3.abs_max(), weight, (long)OC * C, rs.DeviceProp, stream);
        quantize_with_abs_max(weight_e4m3,
                              weight_e4m3.scale(),
                              weight,
                              weight_e4m3.abs_max(),
                              (long)OC * C,
                              rs.DeviceProp,
                              stream);
        weight_is_temp = true;
    }

    // Compute dinp: dinp = W^T @ dout
    auto weight_tp = rs.temp_alloc(ETensorDType::FP8_E4M3, {C, OC}, "weight_tp");
    Tensor weight_tp_stats = rs.temp_alloc(ETensorDType::FP32, {2}, "weight_tp_stats");
    weight_tp.Stats = weight_tp_stats.template get<float>();

    transpose(weight_tp, weight_e4m3, OC, C, stream);
    cudaMemcpyAsync(weight_tp.scale(), weight_e4m3.scale(), sizeof(float), cudaMemcpyDeviceToDevice, stream);

    matmul(dinp,
           weight_tp,
           dout_e5m2,
           std::nullopt,
           weight_tp.scale(),
           dout_e5m2.scale(),
           rs.CublasLtHandle,
           rs.CuBlasWorkspace,
           C,
           B * T,
           OC,
           EMMTranspose::TN,
           /*accumulate=*/false,
           stream);

    rs.temp_free(weight_tp_stats);
    rs.temp_free(weight_tp);
    if (weight_is_temp) {
        rs.temp_free(weight_stats);
        rs.temp_free(weight_e4m3);
    }

    // Compute dweight: dW += inp^T @ dout
    if (!skip_weight_grad) {
        auto activation_tp = rs.temp_alloc(ETensorDType::FP8_E4M3, {C, B * T}, "activation_tp");
        Tensor act_stats = rs.temp_alloc(ETensorDType::FP32, {2}, "act_stats");
        activation_tp.Stats = act_stats.template get<float>();

        if (inp_fp8.abs_max()) {
            quantize_and_transpose_with_abs_max(activation_tp,
                                                activation_tp.scale(),
                                                inp,
                                                inp_fp8.abs_max(),
                                                B * T,
                                                C,
                                                rs.DeviceProp,
                                                stream);
        } else {
            abs_max(activation_tp.abs_max(), inp, (long)B * T * C, rs.DeviceProp, stream);
            quantize_and_transpose_with_abs_max(activation_tp,
                                                activation_tp.scale(),
                                                inp,
                                                activation_tp.abs_max(),
                                                B * T,
                                                C,
                                                rs.DeviceProp,
                                                stream);
        }

        auto grad_tp = rs.temp_alloc(ETensorDType::FP8_E5M2, {OC, B * T}, "grad_tp");
        Tensor grad_stats = rs.temp_alloc(ETensorDType::FP32, {2}, "grad_stats");
        grad_tp.Stats = grad_stats.template get<float>();
        transpose(grad_tp, dout_e5m2, B * T, OC, stream);
        cudaMemcpyAsync(grad_tp.scale(), dout_e5m2.scale(), sizeof(float), cudaMemcpyDeviceToDevice, stream);

        matmul(dweight,
               activation_tp,
               grad_tp,
               std::nullopt,
               activation_tp.scale(),
               grad_tp.scale(),
               rs.CublasLtHandle,
               rs.CuBlasWorkspace,
               C,
               OC,
               B * T,
               EMMTranspose::TN,
               /*accumulate=*/accumulate_gradient,
               stream);

        if (dbias.has_value()) {
            if (!bias_buffer.has_value()) {
                throw std::runtime_error("fp8_hybrid::backward_matmul: dbias requested but bias_buffer not provided");
            }
            backward_bias(dbias.value(),
                          dout_e5m2,
                          activation_tp.scale(),
                          dout_e5m2.scale(),
                          bias_buffer.value(),
                          B,
                          T,
                          OC,
                          rs.DeviceProp,
                          stream);
        }

        rs.temp_free(grad_stats);
        rs.temp_free(grad_tp);
        rs.temp_free(act_stats);
        rs.temp_free(activation_tp);
    }
}

}  // namespace recipes::fp8_hybrid

#endif  // SUROGATE_SRC_RECIPES_FP8_HYBRID_FP8_MATMUL_H
