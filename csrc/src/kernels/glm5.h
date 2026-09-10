// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "utilities/tensor.h"

enum class Glm5Kernel {
    MhcMix,
    MhcCombine,
    KdaDecay,
    KdaRule,
    Clamp,
    CausalConv1d
};
struct Glm5Options {
    int streams = 4;
    int sinkhorn_iters = 20;
    float hc_eps = 1e-6f;
    float norm_eps = 1e-5f;
    float lower_bound = -5.f;
    float clamp_min = -INFINITY;
    float clamp_max = INFINITY;
    bool fused_gate_up = false;
};

// Forward outputs use activation precision except for the FP32 HC gates/decay.
// Backward outputs are FP32; the dispatcher converts/accumulates into gradient slots.
void glm5_forward(Glm5Kernel kind,
                  const std::vector<Tensor>& in,
                  std::vector<Tensor>& out,
                  const Glm5Options& options,
                  cudaStream_t stream);
void glm5_backward(Glm5Kernel kind,
                   const std::vector<Tensor>& in,
                   std::vector<Tensor>& out,
                   const Glm5Options& options,
                   Tensor checkpoints,
                   cudaStream_t stream);
void glm5_copy_gradient(const Tensor& src, Tensor& dst, bool accumulate, cudaStream_t stream);
void glm5_tile_expert_offsets(const int* offsets, int* output, int experts, int start, int rows, cudaStream_t stream);
void glm5_convolution_state(const Tensor& x, const Tensor& weight, const Tensor& state,
                            const Tensor& output, bool initial, cudaStream_t stream);

// Checkpointed recurrent backward recomputes at most 15 tokens per reverse step.
// This bounds scratch by O(B * ceil(T/16) * H * K * V), not O(B*T*H*K*V).
constexpr int GLM5_KDA_CHECKPOINT = 16;
