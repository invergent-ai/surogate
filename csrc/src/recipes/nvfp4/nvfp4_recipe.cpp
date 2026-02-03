// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "nvfp4_recipe.h"

#include <stdexcept>
#include <tuple>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "runtime/core/matmul_context.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "runtime/training/model.h"

namespace recipes {

void NVFP4Recipe::forward_matmul(modules::MatmulContext& ctx) const {
    // Validate input
    if (!ctx.run_state) {
        throw std::runtime_error("NVFP4Recipe::forward_matmul: run_state is null");
    }
    if (!ctx.out || !ctx.inp || !ctx.weight) {
        throw std::runtime_error("NVFP4Recipe::forward_matmul: required tensors are null");
    }

    // Fall back to BF16 matmul if FP4 is not allowed for this layer (skip_quant_first/last_layers)
    if (!ctx.allow_fp4) {
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

    // Dispatch based on backend
    if (mConfig.backend == EMatmulBackend::CUTLASS) {
        forward_matmul_cutlass(ctx);
        return;
    }

    // cuDNN path (CUBLASLT backend)
    // FP4 cuDNN forward matmul with Hadamard transform support
    //
    // This implements the logic from detail::forward_qmm_fp4 for the cuDNN path:
    // 1. Allocate temporary FP4 buffers for input (and weight)
    // 2. Quantize input to NVFP4 E2M1 with cuDNN scale layout
    // 3. Quantize weight to NVFP4 E2M1 with cuDNN scale layout
    // 4. Perform FP4 x FP4 GEMM via cuDNN with FP32 accumulation
    // 5. Apply alpha scaling correction
    // 6. Cast to BF16 and add bias if present

    IRunState& rs = *ctx.run_state;
    const int BT = ctx.B * ctx.T;
    const int C_in = ctx.C_in;
    const int C_out = ctx.C_out;
    constexpr int FP4_BLOCK_SIZE = 16;

    // NOTE: TransformerEngine's NVFP4 recipe currently applies RHT only for "column-wise usage"
    // (primarily to support wgrad GEMM). For forward fprop GEMMs, keep inputs/weights untransformed.
    const Tensor* quant_input = ctx.inp;
    const Tensor* quant_weight = ctx.weight;

    // Step 1: Allocate FP4 input buffers
    // Input data: (BT, C_in) -> packed (BT, C_in/2)
    // Input scales: (ceil(BT/128)*128, ceil(C_in/16/4)*4) for F8_128x4 swizzling
    auto compute_input_scale_shape = [](int M, int K) -> std::pair<long, long> {
        const long scale_rows = (M + 127) / 128 * 128;
        const long scale_cols = ((K + 15) / 16 + 3) / 4 * 4;
        return {scale_rows, scale_cols};
    };

    auto [inp_scale_rows, inp_scale_cols] = compute_input_scale_shape(BT, C_in);

    Tensor inp_fp4_data = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(BT), static_cast<long>(C_in / 2)});
    Tensor inp_fp4_scales = rs.temp_alloc(ETensorDType::FP8_E4M3, {inp_scale_rows, inp_scale_cols});
    Tensor inp_global_amax = rs.temp_alloc(ETensorDType::FP32, {1});

    // Step 2: Quantize input to FP4 with two-level block scaling (auto scale)
    quantize_fp4_block_auto_scale(
        inp_fp4_data.get<uint8_t>(),
        inp_fp4_scales.get<__nv_fp8_e4m3>(),
        inp_global_amax.get<float>(),
        quant_input->get<nv_bfloat16>(),
        BT, C_in,
        rs.DeviceProp, ctx.stream);

    // Step 3: Allocate and quantize weight to FP4
    // Weight layout: (C_out, C_in) = (N, K) row-major -> FP4 packed (C_out, C_in/2)
    // cuDNN B tensor: interpreted as (K, N) = (C_in, C_out) column-major
    // cuDNN expects B (weight) block scales in a K-major layout: (K/16, N) with F8_128x4 reordering.
    auto compute_weight_scale_shape = [](int N, int K) -> std::pair<long, long> {
        // Rows = K/16 (aligned to 4 for F8_128x4)
        const long scale_rows = ((K + 15) / 16 + 3) / 4 * 4;
        // Cols = N (aligned to 128 for F8_128x4)
        const long scale_cols = (N + 127) / 128 * 128;
        return {scale_rows, scale_cols};
    };

    auto [w_scale_rows, w_scale_cols] = compute_weight_scale_shape(C_out, C_in);

    Tensor weight_fp4_data = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(C_out), static_cast<long>(C_in / 2)});
    Tensor weight_fp4_scales = rs.temp_alloc(ETensorDType::FP8_E4M3, {w_scale_rows, w_scale_cols});
    Tensor weight_global_amax = rs.temp_alloc(ETensorDType::FP32, {1});

    // Quantize weight to FP4 with column-major scale layout for cuDNN
    quantize_fp4_weight_2d_auto_scale(
        weight_fp4_data.get<uint8_t>(),
        weight_fp4_scales.get<__nv_fp8_e4m3>(),
        weight_global_amax.get<float>(),
        quant_weight->get<nv_bfloat16>(),
        C_out, C_in,  // N=C_out, K=C_in
        rs.DeviceProp, ctx.stream);

    // Step 4: Execute FP4 matmul via cuDNN
    // Linear layer: y = x @ W^T
    // - A = input (M, K) = (BT, C_in) row-major
    // - B = weight (N, K) = (C_out, C_in) row-major (interpreted as column-major by cuDNN)
    // - D = output (M, N) = (BT, C_out) row-major
    //
    // IMPORTANT:
    // The two-level scaling scheme bakes `global_encode_scale` into the FP8 block scales. This means the
    // dequantized A/B values inside cuDNN are in a *scaled-up* domain (roughly multiplied by global_encode_scale),
    // and we must apply an alpha correction afterwards.
    //
    // If we let cuDNN write BF16 directly, the BF16 rounding happens on the scaled-up outputs, which destroys
    // too much mantissa and leads to severe numerical error (observed as loss ~18 / inf). Fix by:
    //   1) computing the matmul output in FP32
    //   2) applying alpha correction in FP32
    //   3) casting down to BF16
    //
    // OPTIMIZATION (B200/datacenter GPUs):
    // Use fused fp4_alpha_scale_convert() to combine steps 2+3 into a single kernel,
    // eliminating one kernel launch and intermediate memory traffic.
    //
    // Global scales are baked into block scales during quantization; alpha scaling is applied separately.
    Tensor out_f32{};
    float* out_f32_ptr = nullptr;
    const bool need_bf16_output = (ctx.out->DType == ETensorDType::BF16);

    if (need_bf16_output) {
        out_f32 = rs.temp_alloc(ETensorDType::FP32, {static_cast<long>(BT), static_cast<long>(C_out)});
        out_f32_ptr = out_f32.get<float>();
    } else if (ctx.out->DType == ETensorDType::FP32) {
        out_f32_ptr = ctx.out->get<float>();
    } else {
        throw std::runtime_error("NVFP4Recipe::forward_matmul: output must be BF16 or FP32");
    }

    fp4_matmul_f32(
        out_f32_ptr,
        inp_fp4_data.get<uint8_t>(),       // A = input (BT, C_in)
        weight_fp4_data.get<uint8_t>(),    // B = weight (C_out, C_in)
        inp_fp4_scales.get<__nv_fp8_e4m3>(),
        weight_fp4_scales.get<__nv_fp8_e4m3>(),
        1.0f, 1.0f,  // Global scales (unused, alpha scaling applied separately)
        rs.CuBlasWorkspace.get<std::byte>(),
        rs.CuBlasWorkspace.bytes(),
        BT, C_out, C_in,   // M=BT, N=C_out, K=C_in
        FP4_BLOCK_SIZE,
        rs.CudnnHandle, ctx.stream);

    // Step 5+6: Apply alpha scaling and convert to BF16 (fused for better performance)
    // alpha = (global_amax_a * global_amax_b) / (FP4_MAX^2 * FP8_MAX^2)
    if (need_bf16_output) {
        // Fused: alpha scale in FP32 + convert to BF16 in single kernel
        fp4_alpha_scale_convert(
            ctx.out->get<nv_bfloat16>(),
            out_f32_ptr,
            inp_global_amax.get<float>(),
            weight_global_amax.get<float>(),
            static_cast<long>(BT) * C_out,
            rs.DeviceProp, ctx.stream);
        rs.temp_free(out_f32);
    } else {
        // FP32 output: just alpha scale in-place
        fp4_alpha_scale(
            out_f32_ptr,
            inp_global_amax.get<float>(),
            weight_global_amax.get<float>(),
            static_cast<long>(BT) * C_out,
            rs.DeviceProp, ctx.stream);
    }

    // Step 7: Add bias if present
    if (ctx.has_bias()) {
        // Add bias: out += bias (broadcast across B*T dimension)
        if (ctx.out->DType != ETensorDType::BF16) {
            throw std::runtime_error("NVFP4Recipe::forward_matmul: bias add currently requires BF16 output");
        }
        add_bias(ctx.out->get<nv_bfloat16>(), ctx.bias->get<nv_bfloat16>(),
                 ctx.B, ctx.T, C_out, ctx.stream);
    }

    // Step 8: Free temporary buffers (in LIFO order)
    rs.temp_free(weight_global_amax);
    rs.temp_free(weight_fp4_scales);
    rs.temp_free(weight_fp4_data);
    rs.temp_free(inp_global_amax);
    rs.temp_free(inp_fp4_scales);
    rs.temp_free(inp_fp4_data);
}

void NVFP4Recipe::backward_matmul(modules::MatmulContext& ctx) const {
    // Validate input
    if (!ctx.run_state) {
        throw std::runtime_error("NVFP4Recipe::backward_matmul: run_state is null");
    }
    if (!ctx.dinp || !ctx.dout || !ctx.inp || !ctx.weight) {
        throw std::runtime_error("NVFP4Recipe::backward_matmul: required tensors are null");
    }
    if (ctx.inp->DType != ETensorDType::BF16 || ctx.weight->DType != ETensorDType::BF16 ||
        ctx.dout->DType != ETensorDType::BF16) {
        throw std::runtime_error("NVFP4Recipe::backward_matmul: inp/weight/dout must be BF16");
    }

    // Fall back to BF16 matmul if FP4 is not allowed for this layer (skip_quant_first/last_layers)
    if (!ctx.allow_fp4) {
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

    // Dispatch based on backend
    if (mConfig.backend == EMatmulBackend::CUTLASS) {
        backward_matmul_cutlass(ctx);
        return;
    }

    // cuDNN path (CUBLASLT backend)
    // FP4 cuDNN backward matmul with stochastic rounding
    //
    // Computes gradients using NVFP4 E2M1:
    // - dinp = dout @ W (FP4 quantized with stochastic rounding)
    // - dweight = inp^T @ dout (FP4 quantized with optional RHT for wgrad)

    IRunState& rs = *ctx.run_state;
    const int B = ctx.B;
    const int T = ctx.T;
    const int C = ctx.C_in;   // Input channels
    const int OC = ctx.C_out; // Output channels
    const int BT = B * T;
    constexpr int FP4_BLOCK_SIZE = 16;

    // Use different sub-seeds for SR vs RHT
    const unsigned int sr_seed = ctx.seed ^ 0x9E3779B9u;
    const unsigned int rht_seed = ctx.seed ^ 0xA5A5A5A5u;

    // Helper lambda for scale shape computation
    auto compute_weight_scale_shape = [](int N, int K) -> std::pair<long, long> {
        const long scale_rows = ((K + 15) / 16 + 3) / 4 * 4;
        const long scale_cols = (N + 127) / 128 * 128;
        return {scale_rows, scale_cols};
    };

    // =========================================================================
    // dinp = dout @ W
    // =========================================================================
    {
        // Transpose weight to match cuDNN B operand convention for computing dout @ W:
        // fp4_matmul expects B stored as row-major (N, K) but interpreted as column-major (K, N).
        // For dinp (M=BT, N=C, K=OC), B must represent W (OC, C) as column-major (OC, C),
        // which corresponds to a row-major (C, OC) buffer.
        Tensor weight_tp = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(C), static_cast<long>(OC)});
        transpose(weight_tp, *ctx.weight, OC, C, ctx.stream);

        // Quantize dout (A) with stochastic rounding
        Tensor dout_fp4 = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(BT), static_cast<long>(OC / 2)});
        auto [a_scale_rows, a_scale_cols] = fp4_scale_shape(BT, OC);
        Tensor dout_scales = rs.temp_alloc(ETensorDType::FP8_E4M3, {a_scale_rows, a_scale_cols});
        Tensor dout_amax = rs.temp_alloc(ETensorDType::FP32, {1});
        quantize_fp4_block_stochastic_auto_scale(
            dout_fp4.get<uint8_t>(),
            dout_scales.get<__nv_fp8_e4m3>(),
            dout_amax.get<float>(),
            ctx.dout->get<nv_bfloat16>(),
            BT, OC,
            sr_seed,
            rs.DeviceProp, ctx.stream);

        // Quantize W^T (B) with 16x16 weight scaling
        Tensor w_fp4 = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(C), static_cast<long>(OC / 2)});
        auto [w_scale_rows, w_scale_cols] = compute_weight_scale_shape(C, OC);
        Tensor w_scales = rs.temp_alloc(ETensorDType::FP8_E4M3, {w_scale_rows, w_scale_cols});
        Tensor w_amax = rs.temp_alloc(ETensorDType::FP32, {1});
        quantize_fp4_weight_2d_auto_scale(
            w_fp4.get<uint8_t>(),
            w_scales.get<__nv_fp8_e4m3>(),
            w_amax.get<float>(),
            weight_tp.get<nv_bfloat16>(),
            C, OC,
            rs.DeviceProp, ctx.stream);

        // Matmul output in FP32, alpha-correct + convert in single fused kernel
        Tensor dinp_f32{};
        float* dinp_f32_ptr = nullptr;
        const bool need_bf16_dinp = (ctx.dinp->DType == ETensorDType::BF16);

        if (need_bf16_dinp) {
            dinp_f32 = rs.temp_alloc(ETensorDType::FP32, {static_cast<long>(BT), static_cast<long>(C)});
            dinp_f32_ptr = dinp_f32.get<float>();
        } else if (ctx.dinp->DType == ETensorDType::FP32) {
            dinp_f32_ptr = ctx.dinp->get<float>();
        } else {
            throw std::runtime_error("NVFP4Recipe::backward_matmul: dinp must be BF16 or FP32");
        }

        fp4_matmul_f32(
            dinp_f32_ptr,
            dout_fp4.get<uint8_t>(),   // A = dout (BT, OC)
            w_fp4.get<uint8_t>(),      // B = W^T storage (C, OC) row-major
            dout_scales.get<__nv_fp8_e4m3>(),
            w_scales.get<__nv_fp8_e4m3>(),
            1.0f, 1.0f,
            rs.CuBlasWorkspace.get<std::byte>(),
            rs.CuBlasWorkspace.bytes(),
            BT, C, OC,
            FP4_BLOCK_SIZE,
            rs.CudnnHandle, ctx.stream);

        // Fused alpha scale + BF16 conversion for better datacenter GPU performance
        if (need_bf16_dinp) {
            fp4_alpha_scale_convert(
                ctx.dinp->get<nv_bfloat16>(),
                dinp_f32_ptr,
                dout_amax.get<float>(),
                w_amax.get<float>(),
                static_cast<long>(BT) * C,
                rs.DeviceProp, ctx.stream);
            rs.temp_free(dinp_f32);
        } else {
            fp4_alpha_scale(
                dinp_f32_ptr,
                dout_amax.get<float>(),
                w_amax.get<float>(),
                static_cast<long>(BT) * C,
                rs.DeviceProp, ctx.stream);
        }

        rs.temp_free(w_amax);
        rs.temp_free(w_scales);
        rs.temp_free(w_fp4);
        rs.temp_free(dout_amax);
        rs.temp_free(dout_scales);
        rs.temp_free(dout_fp4);
        rs.temp_free(weight_tp);
    }

    // =========================================================================
    // dW += dout^T @ inp (skip if LoRA-only mode)
    // =========================================================================
    if (!ctx.skip_weight_grad && ctx.dweight) {
        // Transpose: A = dout^T (OC, BT), B = inp (BT, C) provided as column-major via row-major (C, BT)
        Tensor dout_tp = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(OC), static_cast<long>(BT)});
        Tensor inp_tp = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(C), static_cast<long>(BT)});
        transpose(dout_tp, *ctx.dout, BT, OC, ctx.stream);
        transpose(inp_tp, *ctx.inp, BT, C, ctx.stream);

        // Apply RHT for wgrad GEMM (matching TransformerEngine's NVFP4 recipe)
        if ((BT % 16) != 0) {
            throw std::runtime_error("NVFP4Recipe::backward_matmul: BT must be multiple of 16 for RHT");
        }
        Tensor dout_tp_r = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(OC), static_cast<long>(BT)});
        Tensor inp_tp_r = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(C), static_cast<long>(BT)});
        hadamard_transform_forward(dout_tp_r.get<nv_bfloat16>(), dout_tp.get<nv_bfloat16>(), nullptr, OC, BT, rht_seed, ctx.stream);
        hadamard_transform_forward(inp_tp_r.get<nv_bfloat16>(), inp_tp.get<nv_bfloat16>(), nullptr, C, BT, rht_seed, ctx.stream);
        const Tensor* q_dout_tp = &dout_tp_r;
        const Tensor* q_inp_tp = &inp_tp_r;

        // Quantize A (gradient) with stochastic rounding
        Tensor a_fp4 = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(OC), static_cast<long>(BT / 2)});
        auto [a_scale_rows, a_scale_cols] = fp4_scale_shape(OC, BT);
        Tensor a_scales = rs.temp_alloc(ETensorDType::FP8_E4M3, {a_scale_rows, a_scale_cols});
        Tensor a_amax = rs.temp_alloc(ETensorDType::FP32, {1});
        quantize_fp4_block_stochastic_auto_scale(
            a_fp4.get<uint8_t>(),
            a_scales.get<__nv_fp8_e4m3>(),
            a_amax.get<float>(),
            q_dout_tp->get<nv_bfloat16>(),
            OC, BT,
            sr_seed + 1,
            rs.DeviceProp, ctx.stream);

        // Quantize B (activation) with deterministic rounding in B-scale layout
        Tensor b_fp4 = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(C), static_cast<long>(BT / 2)});
        auto [b_scale_rows, b_scale_cols] = compute_weight_scale_shape(C, BT);
        Tensor b_scales = rs.temp_alloc(ETensorDType::FP8_E4M3, {b_scale_rows, b_scale_cols});
        Tensor b_amax = rs.temp_alloc(ETensorDType::FP32, {1});
        quantize_fp4_weight_auto_scale(
            b_fp4.get<uint8_t>(),
            b_scales.get<__nv_fp8_e4m3>(),
            b_amax.get<float>(),
            q_inp_tp->get<nv_bfloat16>(),
            C, BT,
            rs.DeviceProp, ctx.stream);

        // Matmul: (OC, BT) @ (BT, C) -> (OC, C)
        Tensor dw_f32 = rs.temp_alloc(ETensorDType::FP32, {static_cast<long>(OC), static_cast<long>(C)});
        float* dw_f32_ptr = dw_f32.get<float>();

        fp4_matmul_f32(
            dw_f32_ptr,
            a_fp4.get<uint8_t>(),
            b_fp4.get<uint8_t>(),
            a_scales.get<__nv_fp8_e4m3>(),
            b_scales.get<__nv_fp8_e4m3>(),
            1.0f, 1.0f,
            rs.CuBlasWorkspace.get<std::byte>(),
            rs.CuBlasWorkspace.bytes(),
            OC, C, BT,
            FP4_BLOCK_SIZE,
            rs.CudnnHandle, ctx.stream);

        // Apply alpha scaling and type conversion (fused when possible for datacenter GPUs)
        if (ctx.dweight->DType == ETensorDType::BF16) {
            if (!ctx.accumulate) {
                // Fused: alpha scale + convert to BF16 in single kernel
                fp4_alpha_scale_convert(
                    ctx.dweight->get<nv_bfloat16>(),
                    dw_f32_ptr,
                    a_amax.get<float>(),
                    b_amax.get<float>(),
                    static_cast<long>(OC) * C,
                    rs.DeviceProp, ctx.stream);
            } else {
                // Accumulate mode: need separate operations
                fp4_alpha_scale(
                    dw_f32_ptr,
                    a_amax.get<float>(),
                    b_amax.get<float>(),
                    static_cast<long>(OC) * C,
                    rs.DeviceProp, ctx.stream);
                Tensor dw_bf16 = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(OC), static_cast<long>(C)});
                convert_dtype(dw_bf16.get<nv_bfloat16>(), dw_f32_ptr,
                              static_cast<std::size_t>(OC) * static_cast<std::size_t>(C), ctx.stream);
                vector_add_sr(ctx.dweight->get<nv_bfloat16>(), ctx.dweight->get<nv_bfloat16>(), dw_bf16.get<nv_bfloat16>(),
                              1.0f, static_cast<long>(OC) * C, ctx.seed, ctx.stream);
                rs.temp_free(dw_bf16);
            }
        } else if (ctx.dweight->DType == ETensorDType::FP32) {
            fp4_alpha_scale(
                dw_f32_ptr,
                a_amax.get<float>(),
                b_amax.get<float>(),
                static_cast<long>(OC) * C,
                rs.DeviceProp, ctx.stream);
            if (!ctx.accumulate) {
                CUDA_CHECK(cudaMemcpyAsync(ctx.dweight->get<float>(), dw_f32_ptr,
                                           sizeof(float) * static_cast<std::size_t>(OC) * static_cast<std::size_t>(C),
                                           cudaMemcpyDeviceToDevice, ctx.stream));
            } else {
                vector_add_sr(ctx.dweight->get<float>(), ctx.dweight->get<float>(), dw_f32_ptr,
                              1.0f, static_cast<long>(OC) * C, ctx.seed, ctx.stream);
            }
        } else {
            throw std::runtime_error("NVFP4Recipe::backward_matmul: dweight must be BF16 or FP32");
        }

        // Compute dbias if needed
        if (ctx.dbias && ctx.bias_buffer) {
            backward_bias(*ctx.dbias, *ctx.dout, /*scale_a=*/nullptr, /*scale_b=*/nullptr,
                          *ctx.bias_buffer, B, T, OC, rs.DeviceProp, ctx.stream);
        }

        rs.temp_free(dw_f32);
        rs.temp_free(b_amax);
        rs.temp_free(b_scales);
        rs.temp_free(b_fp4);
        rs.temp_free(a_amax);
        rs.temp_free(a_scales);
        rs.temp_free(a_fp4);
        rs.temp_free(inp_tp_r);
        rs.temp_free(dout_tp_r);
        rs.temp_free(inp_tp);
        rs.temp_free(dout_tp);
    }
}

// ============================================================================
// CUTLASS Backend Implementations
// ============================================================================

void NVFP4Recipe::forward_matmul_cutlass(modules::MatmulContext& ctx) const {
    // CUTLASS FP4 forward matmul with two-level scaling
    //
    // Optimized path using alpha-in-epilogue fusion:
    // 1. Compute global amax for input and weight (or use cached weight)
    // 2. Bake global encode scale into block scales
    // 3. Compute alpha on device from amaxes
    // 4. Execute FP4 x FP4 GEMM via CUTLASS with alpha fused in epilogue (direct BF16 output)
    // 5. Add bias if present

    IRunState& rs = *ctx.run_state;
    const int BT = ctx.B * ctx.T;
    const int C_in = ctx.C_in;
    const int C_out = ctx.C_out;

    // Check if we have cached FP4 weights (eliminates weight quantization overhead)
    // The weight manager's 4/6 setting is synchronized with the recipe's config at
    // allocate_run_state(), so cached weights use the same quantization method.
    const bool use_cached_weight = (ctx.cached_fp4_data != nullptr &&
                                    ctx.cached_fp4_scales != nullptr &&
                                    ctx.cached_fp4_amax != nullptr);

    // Step 1: Allocate FP4 input buffers with CUTLASS scale layout
    const size_t inp_scale_size = compute_nvfp4_cutlass_scale_size(BT, C_in);
    Tensor inp_fp4_data = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(BT), static_cast<long>(C_in / 2)});
    Tensor inp_fp4_scales = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(inp_scale_size)});

    // Step 2: Quantize input to FP4 with two-level scaling.
    //
    // Fast path: reuse global amax computed in preceding fused ops (e.g., RMSNorm/SwiGLU)
    // to avoid a separate abs_max reduction over the full activation tensor. This is
    // especially important on very fast GPUs (B200) where GEMM is no longer dominant.
    float* inp_global_amax_ptr = nullptr;
    Tensor inp_global_amax{};
    bool inp_amax_is_temp = false;
    {
        Tensor* unused_data = nullptr;
        Tensor* unused_scales = nullptr;
        float* precomputed_amax = nullptr;
        std::tie(unused_data, unused_scales, precomputed_amax) =
            rs.get_fp4_forward_buffers(static_cast<int>(ctx.op));

        if (precomputed_amax) {
            inp_global_amax_ptr = precomputed_amax;
            // Note: 4/6 quantization requires computing amax internally, so we can't use the from_amax path
            if (mConfig.enable_four_over_six) {
                // 4/6 doesn't support from_amax, fall back to auto_scale (will recompute amax)
                inp_global_amax = rs.temp_alloc(ETensorDType::FP32, {1});
                inp_global_amax_ptr = inp_global_amax.get<float>();
                inp_amax_is_temp = true;
                quantize_nvfp4_4o6_cutlass_auto_scale(
                    inp_fp4_data.get<uint8_t>(),
                    inp_fp4_scales.get<uint8_t>(),
                    inp_global_amax_ptr,
                    ctx.inp->get<nv_bfloat16>(),
                    BT, C_in,
                    mConfig.four_over_six_metric,
                    rs.DeviceProp, ctx.stream);
            } else {
                quantize_nvfp4_cutlass_from_amax(
                    inp_fp4_data.get<uint8_t>(),
                    inp_fp4_scales.get<uint8_t>(),
                    inp_global_amax_ptr,
                    ctx.inp->get<nv_bfloat16>(),
                    BT, C_in,
                    rs.DeviceProp, ctx.stream);
            }
        } else {
            inp_global_amax = rs.temp_alloc(ETensorDType::FP32, {1});
            inp_global_amax_ptr = inp_global_amax.get<float>();
            inp_amax_is_temp = true;
            if (mConfig.enable_four_over_six) {
                quantize_nvfp4_4o6_cutlass_auto_scale(
                    inp_fp4_data.get<uint8_t>(),
                    inp_fp4_scales.get<uint8_t>(),
                    inp_global_amax_ptr,
                    ctx.inp->get<nv_bfloat16>(),
                    BT, C_in,
                    mConfig.four_over_six_metric,
                    rs.DeviceProp, ctx.stream);
            } else {
                quantize_nvfp4_cutlass_auto_scale(
                    inp_fp4_data.get<uint8_t>(),
                    inp_fp4_scales.get<uint8_t>(),
                    inp_global_amax_ptr,
                    ctx.inp->get<nv_bfloat16>(),
                    BT, C_in,
                    rs.DeviceProp, ctx.stream);
            }
        }
    }

    // Step 3: Get FP4 weight (cached or quantize on-the-fly)
    const uint8_t* weight_fp4_data_ptr;
    const uint8_t* weight_fp4_scales_ptr;
    const float* weight_amax_ptr;

    // Temporaries for on-the-fly quantization (only allocated if not using cache)
    Tensor weight_fp4_data{};
    Tensor weight_fp4_scales{};
    Tensor weight_global_amax{};

    if (use_cached_weight) {
        // Use pre-quantized cached weights (eliminates 2 quantization kernels per matmul)
        weight_fp4_data_ptr = ctx.cached_fp4_data->get<uint8_t>();
        weight_fp4_scales_ptr = ctx.cached_fp4_scales->get<uint8_t>();
        weight_amax_ptr = ctx.cached_fp4_amax;
    } else {
        // Quantize weight on-the-fly (original path)
        const size_t weight_scale_size = compute_nvfp4_cutlass_scale_size(C_out, C_in);
        weight_fp4_data = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(C_out), static_cast<long>(C_in / 2)});
        weight_fp4_scales = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(weight_scale_size)});
        weight_global_amax = rs.temp_alloc(ETensorDType::FP32, {1});

        if (mConfig.enable_four_over_six) {
            quantize_nvfp4_4o6_cutlass_auto_scale(
                weight_fp4_data.get<uint8_t>(),
                weight_fp4_scales.get<uint8_t>(),
                weight_global_amax.get<float>(),
                ctx.weight->get<nv_bfloat16>(),
                C_out, C_in,
                mConfig.four_over_six_metric,
                rs.DeviceProp, ctx.stream);
        } else {
            quantize_nvfp4_weight_cutlass_auto_scale(
                weight_fp4_data.get<uint8_t>(),
                weight_fp4_scales.get<uint8_t>(),
                weight_global_amax.get<float>(),
                ctx.weight->get<nv_bfloat16>(),
                C_out, C_in,
                rs.DeviceProp, ctx.stream);
        }

        weight_fp4_data_ptr = weight_fp4_data.get<uint8_t>();
        weight_fp4_scales_ptr = weight_fp4_scales.get<uint8_t>();
        weight_amax_ptr = weight_global_amax.get<float>();
    }

    // Step 4: Compute alpha on device
    // The alpha formula depends on which tensor scales are used:
    // - Standard:  both use 2688 => alpha = (amax_a * amax_b) / (2688^2)
    // - 4/6:       both use 1536 => alpha = (amax_a * amax_b) / (1536^2)
    // Note: Cached weights use the same 4/6 setting as on-the-fly quantization
    // (synchronized via set_four_over_six in allocate_run_state).
    Tensor alpha = rs.temp_alloc(ETensorDType::FP32, {1});
    if (mConfig.enable_four_over_six) {
        // Both input and weight use 4/6 tensor scale (1536)
        compute_fp4_alpha_4o6(
            alpha.get<float>(),
            inp_global_amax_ptr,
            weight_amax_ptr,
            ctx.stream);
    } else {
        compute_fp4_alpha(
            alpha.get<float>(),
            inp_global_amax_ptr,
            weight_amax_ptr,
            ctx.stream);
    }

    // Step 5: Execute FP4 matmul with alpha fused in CUTLASS epilogue (direct BF16 output)
    if (ctx.out->DType != ETensorDType::BF16) {
        throw std::runtime_error("NVFP4Recipe::forward_matmul_cutlass: output must be BF16");
    }

    matmul_cutlass_fp4_alpha(
        ctx.out->get<nv_bfloat16>(),
        inp_fp4_data.get<uint8_t>(),      // A = input (BT, C_in)
        weight_fp4_data_ptr,               // B = weight (C_out, C_in) as column-major
        inp_fp4_scales.get<uint8_t>(),
        weight_fp4_scales_ptr,
        alpha.get<float>(),
        rs.CuBlasWorkspace.get<std::byte>(),
        rs.CuBlasWorkspace.bytes(),
        BT, C_out, C_in,   // M=BT, N=C_out, K=C_in
        ctx.stream);

    // Step 6: Add bias if present
    if (ctx.has_bias()) {
        add_bias(ctx.out->get<nv_bfloat16>(), ctx.bias->get<nv_bfloat16>(),
                 ctx.B, ctx.T, C_out, ctx.stream);
    }

    // Step 7: Free temporary buffers (LIFO order)
    rs.temp_free(alpha);
    if (!use_cached_weight) {
        rs.temp_free(weight_global_amax);
        rs.temp_free(weight_fp4_scales);
        rs.temp_free(weight_fp4_data);
    }
    if (inp_amax_is_temp) {
        rs.temp_free(inp_global_amax);
    }
    rs.temp_free(inp_fp4_scales);
    rs.temp_free(inp_fp4_data);
}

void NVFP4Recipe::backward_matmul_cutlass(modules::MatmulContext& ctx) const {
    // CUTLASS FP4 backward matmul with two-level scaling and stochastic rounding
    //
    // Optimized path using alpha-in-epilogue fusion:
    // - dinp = dout @ W (FP4 quantized with SR + two-level scaling + alpha in epilogue)
    // - dweight = inp^T @ dout (FP4 quantized, skip if LoRA-only)

    IRunState& rs = *ctx.run_state;
    const int B = ctx.B;
    const int T = ctx.T;
    const int C = ctx.C_in;   // Input channels
    const int OC = ctx.C_out; // Output channels
    const int BT = B * T;

    const unsigned int sr_seed = ctx.seed ^ 0x9E3779B9u;

    // =========================================================================
    // dinp = dout @ W
    // =========================================================================
    {
        // Quantize dout (A) with stochastic rounding + two-level scaling
        // Note: For dgrad, we always use standard quantization (2688) because W^T uses standard.
        // CUTLASS FP4 matmul requires both inputs to use the same tensor scale.
        // 4/6 is only used for forward (input, weight) and wgrad (dout^T, inp^T).
        const size_t dout_scale_size = compute_nvfp4_cutlass_scale_size(BT, OC);
        Tensor dout_fp4 = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(BT), static_cast<long>(OC / 2)});
        Tensor dout_scales = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(dout_scale_size)});
        Tensor dout_amax = rs.temp_alloc(ETensorDType::FP32, {1});

        quantize_nvfp4_stochastic_cutlass_auto_scale(
            dout_fp4.get<uint8_t>(),
            dout_scales.get<uint8_t>(),
            dout_amax.get<float>(),
            ctx.dout->get<nv_bfloat16>(),
            BT, OC, sr_seed,
            rs.DeviceProp, ctx.stream);

        // Prepare W (B) for dgrad with two-level scaling.
        // Preferred path on B200/B300: use a cached FP4 W^T (transposed layout) produced by the weight manager.
        // Fallback: quantize-and-transpose directly from the original (OC, C) BF16 weight.
        // Note: dgrad always uses standard quantization (2688) for both dout and W^T, so we can use cached W^T.
        const bool use_cached_wt = (ctx.cached_fp4_data != nullptr &&
                                    ctx.cached_fp4_scales != nullptr &&
                                    ctx.cached_fp4_amax != nullptr);

        const uint8_t* w_fp4_ptr = nullptr;
        const uint8_t* w_scales_ptr = nullptr;
        const float* w_amax_ptr = nullptr;

        Tensor w_fp4{};
        Tensor w_scales{};
        Tensor w_amax{};

        if (use_cached_wt) {
            w_fp4_ptr = ctx.cached_fp4_data->get<uint8_t>();
            w_scales_ptr = ctx.cached_fp4_scales->get<uint8_t>();
            w_amax_ptr = ctx.cached_fp4_amax;
        } else {
            const size_t w_scale_size = compute_nvfp4_cutlass_scale_size(C, OC);
            w_fp4 = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(C), static_cast<long>(OC / 2)});
            w_scales = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(w_scale_size)});
            w_amax = rs.temp_alloc(ETensorDType::FP32, {1});

            // Note: No 4/6 variant for weight transpose quantization yet, use standard path
            quantize_nvfp4_weight_cutlass_transpose_auto_scale(
                w_fp4.get<uint8_t>(),
                w_scales.get<uint8_t>(),
                w_amax.get<float>(),
                ctx.weight->get<nv_bfloat16>(),
                /*N=*/OC, /*K=*/C,
                rs.DeviceProp, ctx.stream);

            w_fp4_ptr = w_fp4.get<uint8_t>();
            w_scales_ptr = w_scales.get<uint8_t>();
            w_amax_ptr = w_amax.get<float>();
        }

        // Compute alpha on device
        // For dgrad, both dout and W^T use standard quantization (2688), so always use standard alpha.
        Tensor alpha = rs.temp_alloc(ETensorDType::FP32, {1});
        compute_fp4_alpha(
            alpha.get<float>(),
            dout_amax.get<float>(),
            w_amax_ptr,
            ctx.stream);

        // dinp = dout @ W^T with alpha fused in epilogue (direct BF16 output)
        if (ctx.dinp->DType != ETensorDType::BF16) {
            throw std::runtime_error("NVFP4Recipe::backward_matmul_cutlass: dinp must be BF16");
        }

        matmul_cutlass_fp4_alpha(
            ctx.dinp->get<nv_bfloat16>(),
            dout_fp4.get<uint8_t>(),   // A = dout (BT, OC)
            w_fp4_ptr,                 // B = W^T (C, OC) column-major
            dout_scales.get<uint8_t>(),
            w_scales_ptr,
            alpha.get<float>(),
            rs.CuBlasWorkspace.get<std::byte>(),
            rs.CuBlasWorkspace.bytes(),
            BT, C, OC,
            ctx.stream);

        rs.temp_free(alpha);
        if (!use_cached_wt) {
            rs.temp_free(w_amax);
            rs.temp_free(w_scales);
            rs.temp_free(w_fp4);
        }
        rs.temp_free(dout_amax);
        rs.temp_free(dout_scales);
        rs.temp_free(dout_fp4);
    }

    // =========================================================================
    // dW = dout^T @ inp (skip if LoRA-only mode)
    // =========================================================================
    if (!ctx.skip_weight_grad && ctx.dweight) {
        // For wgrad: A = dout^T (OC, BT), B = inp^T (C, BT) -> result (OC, C)
        Tensor dout_tp = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(OC), static_cast<long>(BT)});
        Tensor inp_tp = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(C), static_cast<long>(BT)});
        transpose(dout_tp, *ctx.dout, BT, OC, ctx.stream);
        transpose(inp_tp, *ctx.inp, BT, C, ctx.stream);

        // Quantize A (dout^T) with stochastic rounding + two-level scaling
        const size_t a_scale_size = compute_nvfp4_cutlass_scale_size(OC, BT);
        Tensor a_fp4 = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(OC), static_cast<long>(BT / 2)});
        Tensor a_scales = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(a_scale_size)});
        Tensor a_amax = rs.temp_alloc(ETensorDType::FP32, {1});

        if (mConfig.enable_four_over_six) {
            quantize_nvfp4_4o6_stochastic_cutlass_auto_scale(
                a_fp4.get<uint8_t>(),
                a_scales.get<uint8_t>(),
                a_amax.get<float>(),
                dout_tp.get<nv_bfloat16>(),
                OC, BT,
                mConfig.four_over_six_metric,
                sr_seed + 1,
                rs.DeviceProp, ctx.stream);
        } else {
            quantize_nvfp4_stochastic_cutlass_auto_scale(
                a_fp4.get<uint8_t>(),
                a_scales.get<uint8_t>(),
                a_amax.get<float>(),
                dout_tp.get<nv_bfloat16>(),
                OC, BT, sr_seed + 1,
                rs.DeviceProp, ctx.stream);
        }

        // Quantize B (inp^T) with two-level scaling
        const size_t b_scale_size = compute_nvfp4_cutlass_scale_size(C, BT);
        Tensor b_fp4 = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(C), static_cast<long>(BT / 2)});
        Tensor b_scales = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(b_scale_size)});
        Tensor b_amax = rs.temp_alloc(ETensorDType::FP32, {1});

        if (mConfig.enable_four_over_six) {
            quantize_nvfp4_4o6_cutlass_auto_scale(
                b_fp4.get<uint8_t>(),
                b_scales.get<uint8_t>(),
                b_amax.get<float>(),
                inp_tp.get<nv_bfloat16>(),
                C, BT,
                mConfig.four_over_six_metric,
                rs.DeviceProp, ctx.stream);
        } else {
            quantize_nvfp4_weight_cutlass_auto_scale(
                b_fp4.get<uint8_t>(),
                b_scales.get<uint8_t>(),
                b_amax.get<float>(),
                inp_tp.get<nv_bfloat16>(),
                C, BT,
                rs.DeviceProp, ctx.stream);
        }

        // Compute alpha on device
        // - If 4/6 enabled: both A and B use 1536 => alpha = (amax_a * amax_b) / (1536^2)
        // - Standard:       both A and B use 2688 => alpha = (amax_a * amax_b) / (2688^2)
        Tensor alpha = rs.temp_alloc(ETensorDType::FP32, {1});
        if (mConfig.enable_four_over_six) {
            compute_fp4_alpha_4o6(
                alpha.get<float>(),
                a_amax.get<float>(),
                b_amax.get<float>(),
                ctx.stream);
        } else {
            compute_fp4_alpha(
                alpha.get<float>(),
                a_amax.get<float>(),
                b_amax.get<float>(),
                ctx.stream);
        }

        // dW = dout^T @ inp^T => (OC, BT) @ (BT, C) = (OC, C) with alpha in epilogue
        if (ctx.dweight->DType != ETensorDType::BF16) {
            throw std::runtime_error("NVFP4Recipe::backward_matmul_cutlass: dweight must be BF16");
        }

        if (!ctx.accumulate) {
            matmul_cutlass_fp4_alpha(
                ctx.dweight->get<nv_bfloat16>(),
                a_fp4.get<uint8_t>(),
                b_fp4.get<uint8_t>(),
                a_scales.get<uint8_t>(),
                b_scales.get<uint8_t>(),
                alpha.get<float>(),
                rs.CuBlasWorkspace.get<std::byte>(),
                rs.CuBlasWorkspace.bytes(),
                OC, C, BT,
                ctx.stream);
        } else {
            // For accumulate mode, need to compute to temp buffer then add
            Tensor dw_temp = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(OC), static_cast<long>(C)});
            matmul_cutlass_fp4_alpha(
                dw_temp.get<nv_bfloat16>(),
                a_fp4.get<uint8_t>(),
                b_fp4.get<uint8_t>(),
                a_scales.get<uint8_t>(),
                b_scales.get<uint8_t>(),
                alpha.get<float>(),
                rs.CuBlasWorkspace.get<std::byte>(),
                rs.CuBlasWorkspace.bytes(),
                OC, C, BT,
                ctx.stream);
            vector_add_sr(ctx.dweight->get<nv_bfloat16>(), ctx.dweight->get<nv_bfloat16>(),
                          dw_temp.get<nv_bfloat16>(), 1.0f,
                          static_cast<long>(OC) * C, ctx.seed, ctx.stream);
            rs.temp_free(dw_temp);
        }

        // Compute dbias if needed
        if (ctx.dbias && ctx.bias_buffer) {
            backward_bias(*ctx.dbias, *ctx.dout, /*scale_a=*/nullptr, /*scale_b=*/nullptr,
                          *ctx.bias_buffer, B, T, OC, rs.DeviceProp, ctx.stream);
        }

        rs.temp_free(alpha);
        rs.temp_free(b_amax);
        rs.temp_free(b_scales);
        rs.temp_free(b_fp4);
        rs.temp_free(a_amax);
        rs.temp_free(a_scales);
        rs.temp_free(a_fp4);
        rs.temp_free(inp_tp);
        rs.temp_free(dout_tp);
    }
}

}  // namespace recipes
