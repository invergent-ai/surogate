// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "recipes/recipe.h"

#include <stdexcept>

#include <cuda_bf16.h>

#include "kernels/kernels.h"
#include "runtime/training/model.h"

namespace recipes {

// =============================================================================
// Default forward matmul implementation
// =============================================================================

void Recipe::forward_matmul(modules::MatmulContext& ctx) const {
    // Default implementation: BF16 matmul using cuBLASLt
    //
    // This base class implementation handles the simple BF16 case.
    // Derived classes (FP8HybridRecipe, NVFP4Recipe, etc.) override this
    // to implement their specific quantization and matmul strategies.

    if (!ctx.run_state) {
        throw std::runtime_error("Recipe::forward_matmul: run_state is null");
    }
    if (!ctx.out || !ctx.inp || !ctx.weight) {
        throw std::runtime_error("Recipe::forward_matmul: required tensors are null");
    }

    IRunState& rs = *ctx.run_state;
    const int M = ctx.B * ctx.T;
    const int N = ctx.C_out;
    const int K = ctx.C_in;

    // Simple BF16 matmul: out = inp @ weight.T
    // Weight is (N, K), inp is (M, K), out is (M, N)
    std::optional<Tensor> bias_opt = ctx.has_bias() ? std::make_optional(*ctx.bias) : std::nullopt;

    matmul(*ctx.out, *ctx.weight, *ctx.inp, bias_opt,
           /*scale_a=*/nullptr, /*scale_b=*/nullptr,
           rs.CublasLtHandle, rs.CuBlasWorkspace,
           N, M, K, EMMTranspose::TN, /*accumulate=*/false, ctx.stream);
}

// =============================================================================
// Default backward matmul implementation
// =============================================================================

void Recipe::backward_matmul(modules::MatmulContext& ctx) const {
    // Default implementation: BF16 backward matmul using cuBLASLt
    //
    // Computes:
    // - dinp = weight @ dout  (gradient flow to previous layer)
    // - dweight += inp.T @ dout  (weight gradient, unless skip_weight_grad)

    if (!ctx.run_state) {
        throw std::runtime_error("Recipe::backward_matmul: run_state is null");
    }
    if (!ctx.dinp || !ctx.dout || !ctx.inp || !ctx.weight) {
        throw std::runtime_error("Recipe::backward_matmul: required tensors are null");
    }

    IRunState& rs = *ctx.run_state;
    const int M = ctx.B * ctx.T;
    const int N = ctx.C_out;
    const int K = ctx.C_in;

    // dinp = W^T @ dout => (K, N) @ (M, N)^T = (K, M)^T = (M, K)
    // Using NN layout: dinp = weight @ dout where weight is (N, K) -> need (K, N)
    matmul(*ctx.dinp, *ctx.weight, *ctx.dout, std::nullopt,
           /*scale_a=*/nullptr, /*scale_b=*/nullptr,
           rs.CublasLtHandle, rs.CuBlasWorkspace,
           K, M, N, EMMTranspose::NN, /*accumulate=*/false, ctx.stream);

    // dweight = inp^T @ dout => (K, M) @ (M, N) = (K, N) => stored as (N, K)
    if (!ctx.skip_weight_grad && ctx.dweight) {
        matmul(*ctx.dweight, *ctx.inp, *ctx.dout, std::nullopt,
               /*scale_a=*/nullptr, /*scale_b=*/nullptr,
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               K, N, M, EMMTranspose::NT, /*accumulate=*/ctx.accumulate, ctx.stream);

        // Bias gradient if requested
        if (ctx.dbias && ctx.dbias->Data) {
            // TODO: backward_bias needs a scratch buffer from run_state
            // For now, bias gradient is handled separately in the model code
        }
    }
}

// =============================================================================
// Default SwiGLU implementation
// =============================================================================

void Recipe::swiglu_forward(modules::SwiGLUContext& ctx) const {
    // Default implementation: standard SwiGLU without scaling
    //
    // SwiGLU(x) = SiLU(gate) * up
    // where gate = x[:, :D], up = x[:, D:]

    if (!ctx.out || !ctx.inp) {
        throw std::runtime_error("Recipe::swiglu_forward: required tensors are null");
    }

    // Call the global swiglu_forward kernel function
    ::swiglu_forward(*ctx.out, *ctx.inp, ctx.abs_max_out, ctx.B, ctx.T, ctx.D, ctx.stream);
}

void Recipe::swiglu_backward(modules::SwiGLUContext& ctx) const {
    // Default implementation: standard SwiGLU backward

    if (!ctx.dinp || !ctx.dout || !ctx.inp) {
        throw std::runtime_error("Recipe::swiglu_backward: required tensors are null");
    }

    ::swiglu_backward(*ctx.dinp, *ctx.dout, *ctx.inp, ctx.abs_max_out,
                      ctx.B, ctx.T, ctx.D, ctx.stream);
}

}  // namespace recipes
