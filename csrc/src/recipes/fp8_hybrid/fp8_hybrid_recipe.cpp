// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "recipes/fp8_hybrid/fp8_hybrid_recipe.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "runtime/core/fp8_scaling_config.h"  // QuantizerIndex
#include "runtime/core/fp8_scaling_state.h"   // FP8ScalingState (must be before runtime/training/model.h)
#include "kernels/kernels.h"
#include "runtime/training/model.h"

namespace recipes {

void FP8HybridRecipe::forward_matmul(modules::MatmulContext& ctx) const {
    // FP8 forward matmul with delayed scaling support
    //
    // This implements the logic previously in detail::forward_qmm_fp8:
    // 1. Quantize input to FP8 E4M3 (JIT or delayed scaling)
    // 2. Handle weight: already FP8, cached FP8, or quantize on-the-fly
    // 3. Perform FP8 x FP8 matmul via cuBLASLt

    if (!ctx.run_state) {
        throw std::runtime_error("FP8HybridRecipe::forward_matmul: run_state is null");
    }
    if (!ctx.out || !ctx.inp || !ctx.weight) {
        throw std::runtime_error("FP8HybridRecipe::forward_matmul: required tensors are null");
    }

    // Fall back to BF16 matmul if FP8 is not allowed for this layer (skip_quant_first/last_layers)
    if (!ctx.allow_fp8) {
        IRunState& rs = *ctx.run_state;
        const int M = ctx.B * ctx.T;
        const int N = ctx.C_out;
        const int K = ctx.C_in;
        std::optional<Tensor> bias_opt = ctx.has_bias() ? std::make_optional(*ctx.bias) : std::nullopt;
        // BF16 forward: out = inp @ weight.T, using same layout as base Recipe class
        // Weight is (N, K), inp is (M, K), out is (M, N)
        matmul(*ctx.out, *ctx.weight, *ctx.inp, bias_opt, nullptr, nullptr,
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               N, M, K, EMMTranspose::TN, /*accumulate=*/false, ctx.stream);
        return;
    }

    if (!ctx.inp_quant || !ctx.inp_quant->Data) {
        throw std::runtime_error("FP8HybridRecipe::forward_matmul: FP8 buffer not allocated");
    }
    if (!ctx.inp_quant->abs_max() || !ctx.inp_quant->scale()) {
        throw std::runtime_error("FP8HybridRecipe::forward_matmul: FP8 buffer missing abs_max/scale Stats");
    }

    IRunState& rs = *ctx.run_state;
    Tensor& inp_fp8 = *ctx.inp_quant;
    const int M = ctx.B * ctx.T;
    const int N = ctx.C_out;
    const int K = ctx.C_in;
    const long num_elements = static_cast<long>(M) * K;

    // Verify inp_quant is FP8 E4M3
    if (inp_fp8.DType != ETensorDType::FP8_E4M3) {
        throw std::runtime_error(std::string(
            "FP8HybridRecipe::forward_matmul: inp_quant should be E4M3, got ") +
            dtype_to_str(inp_fp8.DType));
    }

    // Step 1: Quantize input to FP8 E4M3
    if (ctx.delayed_quantizer_idx >= 0 && rs.has_fp8_delayed_scaling()) {
        // Delayed scaling: use pre-computed scale from history
        auto* scaling_state = rs.get_fp8_scaling_state();
        if (scaling_state) {
            auto qidx = static_cast<modules::QuantizerIndex>(ctx.delayed_quantizer_idx);
            // Verify input dtype before calling typed function
            if (ctx.inp->DType != ETensorDType::BF16) {
                throw std::runtime_error(std::string(
                    "FP8HybridRecipe::forward_matmul: delayed scaling requires BF16 input, got ") +
                    dtype_to_str(ctx.inp->DType));
            }
            quantize_with_delayed_scale(
                inp_fp8.get<__nv_fp8_e4m3>(),
                scaling_state->get_recorded_amax_ptr(qidx),  // Record amax for next iteration
                inp_fp8.scale(),
                ctx.inp->get<nv_bfloat16>(),
                scaling_state->get_scale(qidx),
                num_elements, rs.DeviceProp, ctx.stream);
        } else {
            // Fallback to JIT scaling
            quantize_with_abs_max(inp_fp8, inp_fp8.scale(), *ctx.inp, inp_fp8.abs_max(),
                                  num_elements, rs.DeviceProp, ctx.stream);
        }
    } else {
        // JIT scaling: compute abs_max and scale on-the-fly
        quantize_with_abs_max(inp_fp8, inp_fp8.scale(), *ctx.inp, inp_fp8.abs_max(),
                              num_elements, rs.DeviceProp, ctx.stream);
    }

    // Step 2: Prepare weight and perform matmul
    std::optional<Tensor> bias_opt = ctx.has_bias() ? std::make_optional(*ctx.bias) : std::nullopt;

    // QLoRA-FP8: Check for cached FP8 weight from weight provider first
    // (QLoRA stores weights in FP8 but provides dequantized BF16 via cache)
    if (ctx.cached_weight && ctx.cached_weight->Data && ctx.cached_weight->DType == ETensorDType::FP8_E4M3) {
        // Use pre-cached FP8 weight (from QLoRA weight manager)
        float* weight_scale = const_cast<Tensor*>(ctx.cached_weight)->scale();
        matmul(*ctx.out, *ctx.cached_weight, inp_fp8, bias_opt,
               weight_scale, inp_fp8.scale(),
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               N, M, K, EMMTranspose::TN, /*accumulate=*/false, ctx.stream);
    } else if (ctx.weight->DType == ETensorDType::FP8_E4M3) {
        // Weight already FP8 (persistent quantization mode)
        matmul(*ctx.out, *ctx.weight, inp_fp8, bias_opt,
               ctx.weight->scale(), inp_fp8.scale(),
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               N, M, K, EMMTranspose::TN, /*accumulate=*/false, ctx.stream);
    } else if (ctx.cached_weight && ctx.cached_weight->Data) {
        // Use pre-cached FP8 weight (from weight manager)
        float* weight_scale = const_cast<Tensor*>(ctx.cached_weight)->scale();
        matmul(*ctx.out, *ctx.cached_weight, inp_fp8, bias_opt,
               weight_scale, inp_fp8.scale(),
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               N, M, K, EMMTranspose::TN, /*accumulate=*/false, ctx.stream);
    } else if (ctx.weight->DType == ETensorDType::BF16) {
        // Weight is BF16 - quantize on-the-fly for this forward pass
        Tensor weight_fp8 = rs.temp_alloc(ETensorDType::FP8_E4M3, {N, K});
        Tensor weight_stats = rs.temp_alloc(ETensorDType::FP32, {2});  // [abs_max, scale]
        weight_fp8.Stats = reinterpret_cast<float*>(weight_stats.Data);

        // Compute abs_max and quantize weight to FP8
        abs_max(weight_fp8.abs_max(), *ctx.weight, static_cast<long>(N) * K, rs.DeviceProp, ctx.stream);
        quantize_with_abs_max(weight_fp8, weight_fp8.scale(), *ctx.weight, weight_fp8.abs_max(),
                              static_cast<long>(N) * K, rs.DeviceProp, ctx.stream);

        // FP8 x FP8 matmul
        matmul(*ctx.out, weight_fp8, inp_fp8, bias_opt,
               weight_fp8.scale(), inp_fp8.scale(),
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               N, M, K, EMMTranspose::TN, /*accumulate=*/false, ctx.stream);

        // Free temporary buffers
        rs.temp_free(weight_stats);
        rs.temp_free(weight_fp8);
    } else {
        // Fallback: unsupported weight dtype - use BF16 matmul
        matmul(*ctx.out, *ctx.weight, *ctx.inp, bias_opt,
               /*scale_a=*/nullptr, /*scale_b=*/nullptr,
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               N, M, K, EMMTranspose::TN, /*accumulate=*/false, ctx.stream);
    }
}

void FP8HybridRecipe::backward_matmul(modules::MatmulContext& ctx) const {
    // FP8 backward matmul with E5M2 gradient quantization
    //
    // This implements the logic from detail::backward_qmm_fp8:
    // 1. Quantize upstream gradient (dout) to E5M2
    // 2. Compute dinp = W^T @ dout (E4M3 weight × E5M2 gradient)
    // 3. Compute dweight = inp^T @ dout (E4M3 activation × E5M2 gradient)
    // 4. Optionally compute dbias

    if (!ctx.run_state) {
        throw std::runtime_error("FP8HybridRecipe::backward_matmul: run_state is null");
    }
    if (!ctx.dinp || !ctx.dout || !ctx.inp || !ctx.weight) {
        throw std::runtime_error("FP8HybridRecipe::backward_matmul: required tensors are null");
    }

    // Fall back to BF16 matmul if FP8 is not allowed for this layer (skip_quant_first/last_layers)
    if (!ctx.allow_fp8) {
        IRunState& rs = *ctx.run_state;
        const int B = ctx.B;
        const int T = ctx.T;
        const int C = ctx.C_in;   // Input channels
        const int OC = ctx.C_out; // Output channels

        // dinp = W^T @ dout (always needed for gradient flow)
        // Use Tensor-based matmul identical to backward_qmm
        matmul(*ctx.dinp, *ctx.weight, *ctx.dout, std::nullopt, nullptr, nullptr,
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               C, B * T, OC, EMMTranspose::NN, /*accumulate=*/false, ctx.stream);

        // dweight = inp^T @ dout (skip if weights are frozen in LoRA-only mode)
        if (!ctx.skip_weight_grad && ctx.dweight) {
            matmul(*ctx.dweight, *ctx.inp, *ctx.dout, std::nullopt, nullptr, nullptr,
                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                   C, OC, B * T, EMMTranspose::NT, /*accumulate=*/ctx.accumulate, ctx.stream);

            // Bias gradient if needed
            if (ctx.dbias && ctx.bias_buffer) {
                backward_bias(*ctx.dbias, *ctx.dout, nullptr, nullptr, *ctx.bias_buffer,
                              B, T, OC, rs.DeviceProp, ctx.stream);
            }
        }
        return;
    }

    if (!ctx.dout_quant || !ctx.dout_quant->Data) {
        throw std::runtime_error("FP8HybridRecipe::backward_matmul: E5M2 gradient buffer not allocated");
    }
    if (!ctx.dout_quant->abs_max() || !ctx.dout_quant->scale()) {
        throw std::runtime_error("FP8HybridRecipe::backward_matmul: E5M2 buffer missing abs_max/scale Stats");
    }

    IRunState& rs = *ctx.run_state;
    Tensor& dout_e5m2 = *ctx.dout_quant;
    const int M = ctx.B * ctx.T;
    const int N = ctx.C_out;  // OC
    const int K = ctx.C_in;   // C

    // Verify dtypes
    if (dout_e5m2.DType != ETensorDType::FP8_E5M2) {
        throw std::runtime_error(std::string(
            "FP8HybridRecipe::backward_matmul: dout_quant should be E5M2, got ") +
            dtype_to_str(dout_e5m2.DType));
    }
    if (ctx.dout->DType != ETensorDType::BF16 && ctx.dout->DType != ETensorDType::FP32) {
        throw std::runtime_error(std::string(
            "FP8HybridRecipe::backward_matmul: dout should be BF16/FP32, got ") +
            dtype_to_str(ctx.dout->DType));
    }

    // Step 1: Quantize upstream gradient to E5M2 (larger dynamic range for gradients)
    quantize_with_abs_max(dout_e5m2, dout_e5m2.scale(), *ctx.dout, dout_e5m2.abs_max(),
                          static_cast<long>(M) * N, rs.DeviceProp, ctx.stream);

    // Step 2: Get E4M3 weight for backward
    Tensor weight_e4m3{};
    Tensor weight_stats{};
    bool weight_is_temp = false;

    // Optional: use cached FP8 weights to avoid per-op quantize+transpose.
    // - If cached weight is (K, N), we treat it as W^T and skip transpose.
    // - If cached weight is (N, K), we treat it as W and still transpose each call.
    const Tensor* cached_fp8 = (ctx.cached_weight && ctx.cached_weight->Data) ? ctx.cached_weight : nullptr;
    const bool cached_is_fp8 = (cached_fp8 && cached_fp8->DType == ETensorDType::FP8_E4M3);
    const bool cached_is_transposed = (cached_is_fp8 &&
                                       cached_fp8->Rank == 2 &&
                                       cached_fp8->Sizes[0] == K &&
                                       cached_fp8->Sizes[1] == N);

    if (ctx.weight->DType == ETensorDType::FP8_E4M3) {
        // Weight already FP8 (e.g., QLoRA FP8 base)
        weight_e4m3 = *ctx.weight;
        if (!weight_e4m3.scale()) {
            throw std::runtime_error("FP8HybridRecipe::backward_matmul: FP8 weight missing scale Stats");
        }
    } else if (cached_is_fp8 && cached_fp8->Rank == 2 &&
               cached_fp8->Sizes[0] == N && cached_fp8->Sizes[1] == K) {
        // Use cached FP8 weight (non-transposed) and still transpose per call.
        weight_e4m3 = *cached_fp8;
        if (!weight_e4m3.scale()) {
            throw std::runtime_error("FP8HybridRecipe::backward_matmul: cached FP8 weight missing scale Stats");
        }
    } else {
        // Quantize BF16/FP32 weight to E4M3 on-the-fly
        weight_e4m3 = rs.temp_alloc(ETensorDType::FP8_E4M3, {N, K});
        weight_stats = rs.temp_alloc(ETensorDType::FP32, {2});
        weight_e4m3.Stats = weight_stats.get<float>();

        abs_max(weight_e4m3.abs_max(), *ctx.weight, static_cast<long>(N) * K, rs.DeviceProp, ctx.stream);
        quantize_with_abs_max(weight_e4m3, weight_e4m3.scale(), *ctx.weight, weight_e4m3.abs_max(),
                              static_cast<long>(N) * K, rs.DeviceProp, ctx.stream);
        weight_is_temp = true;
    }

    // Step 3: Compute dinp = W^T @ dout
    if (cached_is_transposed) {
        // Use cached W^T in FP8 directly.
        if (!cached_fp8->scale()) {
            throw std::runtime_error("FP8HybridRecipe::backward_matmul: cached FP8 W^T missing scale Stats");
        }
        matmul(*ctx.dinp, *cached_fp8, dout_e5m2, std::nullopt, cached_fp8->scale(), dout_e5m2.scale(),
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               K, M, N, EMMTranspose::TN, /*accumulate=*/false, ctx.stream);
    } else {
        // Need to transpose weight: (N, K) -> (K, N)
        auto weight_tp = rs.temp_alloc(ETensorDType::FP8_E4M3, {K, N});
        Tensor weight_tp_stats = rs.temp_alloc(ETensorDType::FP32, {2});
        weight_tp.Stats = weight_tp_stats.get<float>();

        transpose(weight_tp, weight_e4m3, N, K, ctx.stream);
        // Copy scale (transpose doesn't change it)
        cudaMemcpyAsync(weight_tp.scale(), weight_e4m3.scale(), sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream);

        // dinp = W^T @ dout: (K, N) × (N, M)^T = (K, M)^T -> (M, K)
        matmul(*ctx.dinp, weight_tp, dout_e5m2, std::nullopt, weight_tp.scale(), dout_e5m2.scale(),
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               K, M, N, EMMTranspose::TN, /*accumulate=*/false, ctx.stream);

        rs.temp_free(weight_tp_stats);
        rs.temp_free(weight_tp);
    }
    if (weight_is_temp) {
        rs.temp_free(weight_stats);
        rs.temp_free(weight_e4m3);
    }

    // Step 4: Compute dweight = inp^T @ dout (skip if LoRA-only mode)
    if (!ctx.skip_weight_grad && ctx.dweight) {
        // Quantize-and-transpose input activation to E4M3
        auto activation_tp = rs.temp_alloc(ETensorDType::FP8_E4M3, {K, M});
        Tensor act_stats = rs.temp_alloc(ETensorDType::FP32, {2});
        activation_tp.Stats = act_stats.get<float>();

        // Use abs_max from forward pass if available
        if (ctx.inp_quant && ctx.inp_quant->abs_max()) {
            quantize_and_transpose_with_abs_max(activation_tp, activation_tp.scale(), *ctx.inp, ctx.inp_quant->abs_max(),
                                                M, K, rs.DeviceProp, ctx.stream);
        } else {
            // Compute abs_max and quantize
            abs_max(activation_tp.abs_max(), *ctx.inp, static_cast<long>(M) * K, rs.DeviceProp, ctx.stream);
            quantize_and_transpose_with_abs_max(activation_tp, activation_tp.scale(), *ctx.inp, activation_tp.abs_max(),
                                                M, K, rs.DeviceProp, ctx.stream);
        }

        // Transpose gradient to E5M2: (M, N) -> (N, M)
        auto grad_tp = rs.temp_alloc(ETensorDType::FP8_E5M2, {N, M});
        Tensor grad_stats = rs.temp_alloc(ETensorDType::FP32, {2});
        grad_tp.Stats = grad_stats.get<float>();
        transpose(grad_tp, dout_e5m2, M, N, ctx.stream);
        cudaMemcpyAsync(grad_tp.scale(), dout_e5m2.scale(), sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream);

        // dweight = inp^T @ dout: (K, M) × (M, N)^T = (K, N)^T -> (N, K)
        matmul(*ctx.dweight, activation_tp, grad_tp, std::nullopt, activation_tp.scale(), grad_tp.scale(),
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               K, N, M, EMMTranspose::TN, /*accumulate=*/ctx.accumulate, ctx.stream);

        // Optional: compute dbias
        if (ctx.dbias && ctx.bias_buffer) {
            backward_bias(*ctx.dbias, dout_e5m2, activation_tp.scale(), dout_e5m2.scale(), *ctx.bias_buffer,
                          ctx.B, ctx.T, N, rs.DeviceProp, ctx.stream);
        }

        rs.temp_free(grad_stats);
        rs.temp_free(grad_tp);
        rs.temp_free(act_stats);
        rs.temp_free(activation_tp);
    }
}

}  // namespace recipes
