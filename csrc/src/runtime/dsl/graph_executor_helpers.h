// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Helper functions for DSL Graph executor (FP8/FP4, quantization, etc).

#ifndef SUROGATE_SRC_DSL_GRAPH_EXECUTOR_HELPERS_H
#define SUROGATE_SRC_DSL_GRAPH_EXECUTOR_HELPERS_H

#include "runtime/dsl/dsl_runtime.h"
#include "runtime/lora/lora_run_state.h"
#include "runtime/core/matmul_context.h"
#include "runtime/core/model_config.h"
#include "runtime/training/runtime_options.h"
#include "utilities/tensor.h"

class NCCLCommunicator;

namespace dsl {

// Quantization layer policy
bool allow_quant_layer(const RuntimeOptions& options, const ::modules::ModelConfig& config, int layer_idx);

// FP8 buffer accessors
Tensor* fp8_forward_buffer(DslRunState& rs, ::modules::MatmulOp op);
Tensor* fp8_grad_buffer(DslRunState& rs, ::modules::MatmulOp op);
int fp8_quantizer_index(const DslRunState& rs, ::modules::MatmulOp op, int layer_idx);

// Bias addition
void add_bias_tensor(Tensor& out, const Tensor& bias, int B, int T, int OC, cudaStream_t stream);

// Loss reduction
void reduce_loss(DslRunState& rs, long B, long T, NCCLCommunicator& comm);

// LoRA RMSNorm recomputation
Tensor recompute_lora_rmsnorm(::modules::LoRARunState& lora_rs, const Tensor& residual, const Tensor& weight,
                              float eps, int B, int T, int C, cudaStream_t stream);

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_GRAPH_EXECUTOR_HELPERS_H
