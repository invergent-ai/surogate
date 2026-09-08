#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace sinfer::ops {

/**
 * Prepares Gated DeltaNet decay and update gates:
 *
 *   g[h,t]    = -exp(A_log[h]) * softplus(a[h,t] + dt_bias[h])
 *   beta[h,t] = sigmoid(b[h,t]).
 *
 * `a` and `b` are contiguous BF16 [H,T], `A_log` and `dt_bias` are contiguous FP32 [H], and
 * `g` and `beta` are contiguous FP32 [H,T]. The oracle evaluates the formula naively in FP64;
 * transcendental implementation and intermediate precision are private kernel choices. Inputs and
 * the two outputs must be mutually non-overlapping. There is no workspace or persistent state side
 * effect.
 */
void gdn_gating(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                Tensor& g, Tensor& beta, cudaStream_t stream);

/**
 * Op: kda_gating
 *
 * Kimi Delta Attention's decay and update gates, which differ from the delta net's above in
 * both shape and formula:
 *
 *   g[h*D + c, t] = lower_bound * sigmoid(exp(A_log[h]) * (a[h*D + c, t] + dt_bias[h*D + c]))
 *   beta[h,t]     = sigmoid(b[h,t])
 *
 * The decay is one value per key *channel* rather than per head, and it is a bounded logistic
 * rather than a softplus: `lower_bound` is the checkpoint's `kda.gate_lower_bound` (-5 for
 * GLM-5.3), so g lies in (lower_bound, 0) and the state can neither grow nor be erased outright.
 * A softplus form would be unbounded below, which is the reason the trained model uses this one.
 *
 * `a` is contiguous BF16 [H*D, T], `b` contiguous BF16 [H, T], `A_log` contiguous FP32 [H],
 * `dt_bias` contiguous FP32 [H*D], `g` contiguous FP32 [H*D, T] and `beta` contiguous FP32
 * [H, T], with H and D positive. The oracle evaluates the formula naively in FP64;
 * transcendental implementation and intermediate precision are private kernel choices. Inputs
 * and the two outputs must be mutually non-overlapping. No workspace, no persistent state.
 */
void kda_gating(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                float lower_bound, Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace sinfer::ops
