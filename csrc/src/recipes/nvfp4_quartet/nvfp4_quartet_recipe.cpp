// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Based on Quartet-II by IST Austria (Erik Schultheis et al.)
// SPDX-License-Identifier: Apache-2.0

#include "nvfp4_quartet_recipe.h"

#include <stdexcept>
#include <tuple>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "runtime/core/matmul_context.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "runtime/training/model.h"

// Quartet-II EDEN kernels
#include "kernels/quartet_kernels.h"

namespace recipes {

// ============================================================================
// Forward Matmul - Standard RTN NVFP4 (no Hadamard/EDEN)
// ============================================================================

void NVFP4QuartetRecipe::forward_matmul(modules::MatmulContext& ctx) const {
    // Validate input
    if (!ctx.run_state) {
        throw std::runtime_error("NVFP4QuartetRecipe::forward_matmul: run_state is null");
    }
    if (!ctx.out || !ctx.inp || !ctx.weight) {
        throw std::runtime_error("NVFP4QuartetRecipe::forward_matmul: required tensors are null");
    }

    // Fall back to BF16 matmul if FP4 is not allowed for this layer (skip_quant_first/last_layers)
    if (!ctx.allow_fp4) {
        IRunState& rs = *ctx.run_state;
        const int M = ctx.B * ctx.T;
        const int N = ctx.C_out;
        const int K = ctx.C_in;
        std::optional<Tensor> bias_opt = ctx.has_bias() ? std::make_optional(*ctx.bias) : std::nullopt;
        matmul(*ctx.out, *ctx.weight, *ctx.inp, bias_opt, nullptr, nullptr,
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               N, M, K, EMMTranspose::TN, /*accumulate=*/false, ctx.stream);
        return;
    }

    // Quartet-II forward pass uses standard RTN quantization (no Hadamard, no EDEN).
    // EDEN with Hadamard re-randomization is only applied in backward for gradient unbiasedness.
    IRunState& rs = *ctx.run_state;
    const int BT = ctx.B * ctx.T;
    const int C_in = ctx.C_in;
    const int C_out = ctx.C_out;

    // Check if we have cached FP4 weights
    const bool use_cached_weight = (ctx.cached_fp4_data != nullptr &&
                                    ctx.cached_fp4_scales != nullptr &&
                                    ctx.cached_fp4_amax != nullptr);

    // =========================================================================
    // Step 1: Allocate FP4 input buffers (CUTLASS layout)
    // =========================================================================
    const size_t inp_scale_size = compute_nvfp4_cutlass_scale_size(BT, C_in);
    Tensor inp_fp4_data = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(BT), static_cast<long>(C_in / 2)});
    Tensor inp_fp4_scales = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(inp_scale_size)});
    Tensor inp_global_amax = rs.temp_alloc(ETensorDType::FP32, {1});

    // =========================================================================
    // Step 2: Quantize input with standard RTN (no Hadamard, no EDEN for forward)
    // =========================================================================
    // Forward pass uses RTN quantization. EDEN is only for backward gradient unbiasedness.
    quantize_nvfp4_cutlass_auto_scale(
        inp_fp4_data.get<uint8_t>(),
        inp_fp4_scales.get<uint8_t>(),
        inp_global_amax.get<float>(),
        ctx.inp->get<nv_bfloat16>(),
        BT, C_in,
        rs.DeviceProp, ctx.stream);

    // =========================================================================
    // Step 3: Get FP4 weight (cached or quantize on-the-fly)
    // =========================================================================
    const uint8_t* weight_fp4_data_ptr;
    const uint8_t* weight_fp4_scales_ptr;
    const float* weight_amax_ptr;

    Tensor weight_fp4_data{};
    Tensor weight_fp4_scales{};
    Tensor weight_global_amax{};

    if (use_cached_weight) {
        // Use pre-quantized cached weights
        weight_fp4_data_ptr = ctx.cached_fp4_data->get<uint8_t>();
        weight_fp4_scales_ptr = ctx.cached_fp4_scales->get<uint8_t>();
        weight_amax_ptr = ctx.cached_fp4_amax;
    } else {
        // Quantize weight on-the-fly (RTN, no Hadamard per TransformerEngine convention)
        const size_t weight_scale_size = compute_nvfp4_cutlass_scale_size(C_out, C_in);
        weight_fp4_data = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(C_out), static_cast<long>(C_in / 2)});
        weight_fp4_scales = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(weight_scale_size)});
        weight_global_amax = rs.temp_alloc(ETensorDType::FP32, {1});

        // Weights use standard RTN quantization (no Hadamard, no EDEN)
        quantize_nvfp4_weight_cutlass_auto_scale(
            weight_fp4_data.get<uint8_t>(),
            weight_fp4_scales.get<uint8_t>(),
            weight_global_amax.get<float>(),
            ctx.weight->get<nv_bfloat16>(),
            C_out, C_in,
            rs.DeviceProp, ctx.stream);

        weight_fp4_data_ptr = weight_fp4_data.get<uint8_t>();
        weight_fp4_scales_ptr = weight_fp4_scales.get<uint8_t>();
        weight_amax_ptr = weight_global_amax.get<float>();
    }

    // =========================================================================
    // Step 4: Compute alpha (standard tensor scale since forward uses RTN)
    // =========================================================================
    // Forward uses standard NVFP4 quantization (FP8_MAX = 448), so use standard alpha
    Tensor alpha = rs.temp_alloc(ETensorDType::FP32, {1});
    compute_fp4_alpha(
        alpha.get<float>(),
        inp_global_amax.get<float>(),
        weight_amax_ptr,
        ctx.stream);

    // =========================================================================
    // Step 5: Execute FP4 matmul with alpha fused in CUTLASS epilogue
    // =========================================================================
    if (ctx.out->DType != ETensorDType::BF16) {
        throw std::runtime_error("NVFP4QuartetRecipe::forward_matmul: output must be BF16");
    }

    matmul_cutlass_fp4_alpha(
        ctx.out->get<nv_bfloat16>(),
        inp_fp4_data.get<uint8_t>(),      // A = input (BT, C_in)
        weight_fp4_data_ptr,               // B = weight (C_out, C_in) column-major
        inp_fp4_scales.get<uint8_t>(),
        weight_fp4_scales_ptr,
        alpha.get<float>(),
        rs.CuBlasWorkspace.get<std::byte>(),
        rs.CuBlasWorkspace.bytes(),
        BT, C_out, C_in,
        ctx.stream);

    // =========================================================================
    // Step 6: Add bias if present
    // =========================================================================
    if (ctx.has_bias()) {
        add_bias(ctx.out->get<nv_bfloat16>(), ctx.bias->get<nv_bfloat16>(),
                 ctx.B, ctx.T, C_out, ctx.stream);
    }

    // =========================================================================
    // Step 7: Free temporary buffers (LIFO order)
    // =========================================================================
    rs.temp_free(alpha);
    if (!use_cached_weight) {
        rs.temp_free(weight_global_amax);
        rs.temp_free(weight_fp4_scales);
        rs.temp_free(weight_fp4_data);
    }
    rs.temp_free(inp_global_amax);
    rs.temp_free(inp_fp4_scales);
    rs.temp_free(inp_fp4_data);
}

// ============================================================================
// Backward Matmul - Uses EDEN quantization with SR on scales only
// ============================================================================

void NVFP4QuartetRecipe::backward_matmul(modules::MatmulContext& ctx) const {
    // Validate input
    if (!ctx.run_state) {
        throw std::runtime_error("NVFP4QuartetRecipe::backward_matmul: run_state is null");
    }
    if (!ctx.dinp || !ctx.dout || !ctx.inp || !ctx.weight) {
        throw std::runtime_error("NVFP4QuartetRecipe::backward_matmul: required tensors are null");
    }
    if (ctx.inp->DType != ETensorDType::BF16 || ctx.weight->DType != ETensorDType::BF16 ||
        ctx.dout->DType != ETensorDType::BF16) {
        throw std::runtime_error("NVFP4QuartetRecipe::backward_matmul: inp/weight/dout must be BF16");
    }

    // Fall back to BF16 matmul if FP4 is not allowed for this layer
    if (!ctx.allow_fp4) {
        IRunState& rs = *ctx.run_state;
        const int B = ctx.B;
        const int T = ctx.T;
        const int C = ctx.C_in;
        const int OC = ctx.C_out;

        // dinp = W^T @ dout
        matmul(*ctx.dinp, *ctx.weight, *ctx.dout, std::nullopt, nullptr, nullptr,
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               C, B * T, OC, EMMTranspose::NN, /*accumulate=*/false, ctx.stream);

        // dweight = inp^T @ dout
        if (!ctx.skip_weight_grad && ctx.dweight) {
            matmul(*ctx.dweight, *ctx.inp, *ctx.dout, std::nullopt, nullptr, nullptr,
                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                   C, OC, B * T, EMMTranspose::NT, /*accumulate=*/ctx.accumulate, ctx.stream);

            if (ctx.dbias && ctx.bias_buffer) {
                backward_bias(*ctx.dbias, *ctx.dout, nullptr, nullptr, *ctx.bias_buffer,
                              B, T, OC, rs.DeviceProp, ctx.stream);
            }
        }
        return;
    }

    IRunState& rs = *ctx.run_state;
    const int B = ctx.B;
    const int T = ctx.T;
    const int C = ctx.C_in;
    const int OC = ctx.C_out;
    const int BT = B * T;

    const float fp4_max_bwd = quartet::FP4_MAX / mConfig.backward_scale_override;
    const float fp8_max_eden = mConfig.scales_max;

    // Dimension validation for 128-element Hadamard groups
    if (C % quartet::HADAMARD_DIM != 0 || OC % quartet::HADAMARD_DIM != 0) {
        throw std::runtime_error("NVFP4QuartetRecipe: C and OC must be multiples of 128 for Hadamard transform");
    }
    if (BT % quartet::HADAMARD_DIM != 0) {
        throw std::runtime_error("NVFP4QuartetRecipe: B*T must be multiple of 128 for Hadamard transform");
    }

    // NOTE: Quartet-II uses the *same* rerotated Hadamard matrix for both operands
    // within each GEMM (EW and EtX). If different Hadamards are used, the rotation
    // no longer cancels (HᵀH != I) and gradients become incorrect.
    const unsigned int hadamard_seed = ctx.seed ^ 0xDEADBEEFu;
    const unsigned int sr_seed = ctx.seed ^ 0x9E3779B9u;

    Tensor hadamard = rs.temp_alloc(ETensorDType::BF16,
        {quartet::HADAMARD_DIM, quartet::HADAMARD_DIM});
    quartet::initialize_hadamard_128(hadamard.get<nv_bfloat16>(), hadamard_seed, ctx.stream);

    // =========================================================================
    // dinp = dout @ W (grad_x path)
    // =========================================================================
    {
        // Allocate FP4 buffers for dout with EDEN quantization
        const size_t dout_scale_size = compute_nvfp4_cutlass_scale_size(BT, OC);
        Tensor dout_fp4 = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(BT), static_cast<long>(OC / 2)});
        const size_t dout_scale_elems = (static_cast<size_t>(BT) * OC) / quartet::QUANT_GROUP_SIZE;
        Tensor dout_scales_linear = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(dout_scale_elems)});
        Tensor dout_scales_cutlass = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(dout_scale_size)});
        Tensor dout_amax = rs.temp_alloc(ETensorDType::FP32, {1});

        // EDEN scratch buffers
        Tensor dout_scratch_scales = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(dout_scale_elems)});
        Tensor dout_max_scale = rs.temp_alloc(ETensorDType::INT32, {1});

        // EDEN quantize dout with Hadamard transform
        // Key Quartet-II features:
        // - RTN for FP4 values (not stochastic rounding)
        // - EDEN correction factor per 128-element group
        // - Stochastic rounding on E4M3 scales only
        // - Backward scale override (~1.054)
        quartet::group_transform_128_eden(
            reinterpret_cast<__nv_fp4x2_storage_t*>(dout_fp4.get<uint8_t>()),
            reinterpret_cast<__nv_fp8_e4m3*>(dout_scales_linear.get<uint8_t>()),
            dout_amax.get<float>(),
            dout_scratch_scales.get<nv_bfloat16>(),
            reinterpret_cast<unsigned*>(dout_max_scale.Data),
            hadamard.get<nv_bfloat16>(),
            ctx.dout->get<nv_bfloat16>(),
            sr_seed,  // For stochastic rounding on scales
            fp4_max_bwd,
            fp8_max_eden,
            BT, OC,
            /*transposeX=*/false,
            ctx.stream);

        // Quartet EDEN kernels emit scales in a simple linear row-major layout.
        // CUTLASS FP4 GEMM expects scales in Sm1xxBlkScaledConfig swizzled layout.
        nvfp4_scales_linear_to_cutlass(
            dout_scales_cutlass.get<uint8_t>(),
            dout_scales_linear.get<uint8_t>(),
            BT, OC,
            ctx.stream);

        // Prepare W^T for dgrad computation
        // If cached, use cached W^T. Otherwise, dequant -> transpose -> requant with new Hadamard
        const bool use_cached_wt = (ctx.cached_fp4_data != nullptr &&
                                    ctx.cached_fp4_scales != nullptr &&
                                    ctx.cached_fp4_amax != nullptr);

        const uint8_t* w_fp4_ptr = nullptr;
        const uint8_t* w_scales_ptr = nullptr;
        const float* w_amax_ptr = nullptr;

        Tensor w_fp4{};
        Tensor w_scales{};
        Tensor w_scales_linear{};
        Tensor w_amax{};
        Tensor w_scratch_scales{};
        Tensor w_max_scale{};

        if (use_cached_wt) {
            const size_t w_scale_size = compute_nvfp4_cutlass_scale_size(C, OC);
            w_fp4 = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(C), static_cast<long>(OC / 2)});
            const size_t w_scale_elems = (static_cast<size_t>(C) * OC) / quartet::QUANT_GROUP_SIZE;
            w_scales_linear = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(w_scale_elems)});
            w_scales = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(w_scale_size)});
            w_amax = rs.temp_alloc(ETensorDType::FP32, {1});

            w_scratch_scales = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(w_scale_elems)});
            w_max_scale = rs.temp_alloc(ETensorDType::INT32, {1});
            const size_t cached_w_scale_elems = (static_cast<size_t>(OC) * C) / quartet::QUANT_GROUP_SIZE;
            Tensor cached_w_scales_linear = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(cached_w_scale_elems)});
            Tensor cached_w_tensor_scale = rs.temp_alloc(ETensorDType::FP32, {1});

            // FP4 cache stores global amax; dequant kernel expects tensor_scale = amax / (FP4_MAX * FP8_MAX).
            // Cached weights are produced by the standard NVFP4 path (FP8_MAX=448).
            compute_fp4_tensor_scale(
                cached_w_tensor_scale.get<float>(),
                ctx.cached_fp4_amax,
                quartet::FP4_MAX,
                /*fp8_max=*/448.0f,
                ctx.stream);

            // Cache scales are in CUTLASS layout; dequant kernel expects a linear row-major scale matrix.
            nvfp4_scales_cutlass_to_linear(
                cached_w_scales_linear.get<uint8_t>(),
                ctx.cached_fp4_scales->get<uint8_t>(),
                OC, C,
                ctx.stream);

            // Dequant cached W -> transpose -> Hadamard -> EDEN requant
            quartet::dequant_tp_had_quant(
                reinterpret_cast<__nv_fp4x2_storage_t*>(w_fp4.get<uint8_t>()),
                reinterpret_cast<__nv_fp8_e4m3*>(w_scales_linear.get<uint8_t>()),
                w_amax.get<float>(),
                w_scratch_scales.get<nv_bfloat16>(),
                reinterpret_cast<unsigned*>(w_max_scale.Data),
                hadamard.get<nv_bfloat16>(),
                reinterpret_cast<const __nv_fp4x2_storage_t*>(ctx.cached_fp4_data->get<uint8_t>()),
                reinterpret_cast<const __nv_fp8_e4m3*>(cached_w_scales_linear.get<uint8_t>()),
                cached_w_tensor_scale.get<float>(),
                sr_seed + 1,
                fp4_max_bwd,
                fp8_max_eden,
                OC, C,  // Original W is (OC, C), transposed is (C, OC)
                ctx.stream);

            // Convert requantized scales to CUTLASS layout for GEMM.
            nvfp4_scales_linear_to_cutlass(
                w_scales.get<uint8_t>(),
                w_scales_linear.get<uint8_t>(),
                C, OC,
                ctx.stream);

            w_fp4_ptr = w_fp4.get<uint8_t>();
            w_scales_ptr = w_scales.get<uint8_t>();
            w_amax_ptr = w_amax.get<float>();

            rs.temp_free(cached_w_tensor_scale);
            rs.temp_free(cached_w_scales_linear);
        } else {
            // No cached weights: quantize W^T from BF16 directly
            // Use Hadamard-aware quantize-transpose
            const size_t w_scale_size = compute_nvfp4_cutlass_scale_size(C, OC);
            w_fp4 = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(C), static_cast<long>(OC / 2)});
            const size_t w_scale_elems = (static_cast<size_t>(C) * OC) / quartet::QUANT_GROUP_SIZE;
            w_scales_linear = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(w_scale_elems)});
            w_scales = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(w_scale_size)});
            w_amax = rs.temp_alloc(ETensorDType::FP32, {1});

            w_scratch_scales = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(w_scale_elems)});
            w_max_scale = rs.temp_alloc(ETensorDType::INT32, {1});

            // Transpose W (OC, C) -> W^T (C, OC) then EDEN quantize
            Tensor weight_tp = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(C), static_cast<long>(OC)});
            transpose(weight_tp, *ctx.weight, OC, C, ctx.stream);

            quartet::group_transform_128_eden(
                reinterpret_cast<__nv_fp4x2_storage_t*>(w_fp4.get<uint8_t>()),
                reinterpret_cast<__nv_fp8_e4m3*>(w_scales_linear.get<uint8_t>()),
                w_amax.get<float>(),
                w_scratch_scales.get<nv_bfloat16>(),
                reinterpret_cast<unsigned*>(w_max_scale.Data),
                hadamard.get<nv_bfloat16>(),
                weight_tp.get<nv_bfloat16>(),
                sr_seed + 1,
                fp4_max_bwd,
                fp8_max_eden,
                C, OC,
                /*transposeX=*/false,
                ctx.stream);

            rs.temp_free(weight_tp);

            // Convert scales to CUTLASS layout for GEMM.
            nvfp4_scales_linear_to_cutlass(
                w_scales.get<uint8_t>(),
                w_scales_linear.get<uint8_t>(),
                C, OC,
                ctx.stream);
            w_fp4_ptr = w_fp4.get<uint8_t>();
            w_scales_ptr = w_scales.get<uint8_t>();
            w_amax_ptr = w_amax.get<float>();
        }

        // EDEN kernels produce tensor_scale (amax / (fp4_max * fp8_max)), so alpha is just the product.
        Tensor alpha = rs.temp_alloc(ETensorDType::FP32, {1});
        compute_fp4_alpha_from_tensor_scale(
            alpha.get<float>(),
            dout_amax.get<float>(),
            w_amax_ptr,
            ctx.stream);

        // dinp = dout @ W^T with alpha fused in epilogue
        if (ctx.dinp->DType != ETensorDType::BF16) {
            throw std::runtime_error("NVFP4QuartetRecipe::backward_matmul: dinp must be BF16");
        }

        matmul_cutlass_fp4_alpha(
            ctx.dinp->get<nv_bfloat16>(),
            dout_fp4.get<uint8_t>(),   // A = dout (BT, OC)
            w_fp4_ptr,                  // B = W^T (C, OC) column-major
            dout_scales_cutlass.get<uint8_t>(),
            w_scales_ptr,
            alpha.get<float>(),
            rs.CuBlasWorkspace.get<std::byte>(),
            rs.CuBlasWorkspace.bytes(),
            BT, C, OC,
            ctx.stream);

        // Free dgrad temporaries (strict LIFO order)
        // Allocation order: dout_fp4, dout_scales, dout_amax, dout_scratch_scales,
        //                   dout_max_scale, w_fp4, w_scales, w_amax, w_scratch_scales,
        //                   w_max_scale, alpha
        // So deallocation must be reverse order:
        rs.temp_free(alpha);
        if (w_max_scale.Data) rs.temp_free(w_max_scale);
        if (w_scratch_scales.Data) rs.temp_free(w_scratch_scales);
        if (w_amax.Data) rs.temp_free(w_amax);
        if (w_scales.Data) rs.temp_free(w_scales);
        if (w_scales_linear.Data) rs.temp_free(w_scales_linear);
        if (w_fp4.Data) rs.temp_free(w_fp4);
        rs.temp_free(dout_max_scale);
        rs.temp_free(dout_scratch_scales);
        rs.temp_free(dout_amax);
        rs.temp_free(dout_scales_cutlass);
        rs.temp_free(dout_scales_linear);
        rs.temp_free(dout_fp4);
    }

    // =========================================================================
    // dW = dout^T @ inp (grad_w path, skip if LoRA-only)
    // =========================================================================
    if (!ctx.skip_weight_grad && ctx.dweight) {
        // Transpose tensors for wgrad computation
        Tensor dout_tp = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(OC), static_cast<long>(BT)});
        Tensor inp_tp = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(C), static_cast<long>(BT)});
        transpose(dout_tp, *ctx.dout, BT, OC, ctx.stream);
        transpose(inp_tp, *ctx.inp, BT, C, ctx.stream);

        // Quantize A (dout^T) with EDEN
        const size_t a_scale_size = compute_nvfp4_cutlass_scale_size(OC, BT);
        Tensor a_fp4 = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(OC), static_cast<long>(BT / 2)});
        const size_t a_scale_elems = (static_cast<size_t>(OC) * BT) / quartet::QUANT_GROUP_SIZE;
        Tensor a_scales_linear = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(a_scale_elems)});
        Tensor a_scales_cutlass = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(a_scale_size)});
        Tensor a_amax = rs.temp_alloc(ETensorDType::FP32, {1});
        Tensor a_scratch = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(a_scale_elems)});
        Tensor a_max_scale = rs.temp_alloc(ETensorDType::INT32, {1});

        quartet::group_transform_128_eden(
            reinterpret_cast<__nv_fp4x2_storage_t*>(a_fp4.get<uint8_t>()),
            reinterpret_cast<__nv_fp8_e4m3*>(a_scales_linear.get<uint8_t>()),
            a_amax.get<float>(),
            a_scratch.get<nv_bfloat16>(),
            reinterpret_cast<unsigned*>(a_max_scale.Data),
            hadamard.get<nv_bfloat16>(),
            dout_tp.get<nv_bfloat16>(),
            sr_seed + 2,
            fp4_max_bwd,
            fp8_max_eden,
            OC, BT,
            /*transposeX=*/false,
            ctx.stream);

        nvfp4_scales_linear_to_cutlass(
            a_scales_cutlass.get<uint8_t>(),
            a_scales_linear.get<uint8_t>(),
            OC, BT,
            ctx.stream);

        // Quantize B (inp^T) with EDEN (same Hadamard as dout^T for wgrad)
        const size_t b_scale_size = compute_nvfp4_cutlass_scale_size(C, BT);
        Tensor b_fp4 = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(C), static_cast<long>(BT / 2)});
        const size_t b_scale_elems = (static_cast<size_t>(C) * BT) / quartet::QUANT_GROUP_SIZE;
        Tensor b_scales_linear = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(b_scale_elems)});
        Tensor b_scales_cutlass = rs.temp_alloc(ETensorDType::BYTE, {static_cast<long>(b_scale_size)});
        Tensor b_amax = rs.temp_alloc(ETensorDType::FP32, {1});
        Tensor b_scratch = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(b_scale_elems)});
        Tensor b_max_scale = rs.temp_alloc(ETensorDType::INT32, {1});

        quartet::group_transform_128_eden(
            reinterpret_cast<__nv_fp4x2_storage_t*>(b_fp4.get<uint8_t>()),
            reinterpret_cast<__nv_fp8_e4m3*>(b_scales_linear.get<uint8_t>()),
            b_amax.get<float>(),
            b_scratch.get<nv_bfloat16>(),
            reinterpret_cast<unsigned*>(b_max_scale.Data),
            hadamard.get<nv_bfloat16>(),
            inp_tp.get<nv_bfloat16>(),
            sr_seed + 3,
            fp4_max_bwd,
            fp8_max_eden,
            C, BT,
            /*transposeX=*/false,
            ctx.stream);

        nvfp4_scales_linear_to_cutlass(
            b_scales_cutlass.get<uint8_t>(),
            b_scales_linear.get<uint8_t>(),
            C, BT,
            ctx.stream);

        // EDEN kernels produce tensor_scale (amax / (fp4_max * fp8_max)), so alpha is just the product.
        Tensor alpha = rs.temp_alloc(ETensorDType::FP32, {1});
        compute_fp4_alpha_from_tensor_scale(
            alpha.get<float>(),
            a_amax.get<float>(),
            b_amax.get<float>(),
            ctx.stream);

        // dW = dout^T @ inp^T => (OC, BT) @ (BT, C) = (OC, C) with alpha in epilogue
        if (ctx.dweight->DType != ETensorDType::BF16) {
            throw std::runtime_error("NVFP4QuartetRecipe::backward_matmul: dweight must be BF16");
        }

        if (!ctx.accumulate) {
            matmul_cutlass_fp4_alpha(
                ctx.dweight->get<nv_bfloat16>(),
                a_fp4.get<uint8_t>(),
                b_fp4.get<uint8_t>(),
                a_scales_cutlass.get<uint8_t>(),
                b_scales_cutlass.get<uint8_t>(),
                alpha.get<float>(),
                rs.CuBlasWorkspace.get<std::byte>(),
                rs.CuBlasWorkspace.bytes(),
                OC, C, BT,
                ctx.stream);
        } else {
            // Accumulate mode: compute to temp buffer then add
            Tensor dw_temp = rs.temp_alloc(ETensorDType::BF16, {static_cast<long>(OC), static_cast<long>(C)});
            matmul_cutlass_fp4_alpha(
                dw_temp.get<nv_bfloat16>(),
                a_fp4.get<uint8_t>(),
                b_fp4.get<uint8_t>(),
                a_scales_cutlass.get<uint8_t>(),
                b_scales_cutlass.get<uint8_t>(),
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

        // Free wgrad temporaries (LIFO)
        rs.temp_free(alpha);
        rs.temp_free(b_max_scale);
        rs.temp_free(b_scratch);
        rs.temp_free(b_amax);
        rs.temp_free(b_scales_cutlass);
        rs.temp_free(b_scales_linear);
        rs.temp_free(b_fp4);
        rs.temp_free(a_max_scale);
        rs.temp_free(a_scratch);
        rs.temp_free(a_amax);
        rs.temp_free(a_scales_cutlass);
        rs.temp_free(a_scales_linear);
        rs.temp_free(a_fp4);
        rs.temp_free(inp_tp);
        rs.temp_free(dout_tp);
    }

    rs.temp_free(hadamard);
}

}  // namespace recipes
