// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Mamba output projection operation dispatch.
// This is essentially a standard matmul, but we keep it separate for clarity
// and potential Mamba-specific optimizations (e.g., LoRA hooks).

#include "runtime/executor/compiled_ops.h"

#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_mamba_out_proj(const CompiledOp& op, const modules::ForwardHook* hook) {
    // This is a standard matmul: out = gated_out @ out_proj_weight.T
    // Input: gated_out [B*T, intermediate_size]
    // Weight: out_proj_weight [d_model, intermediate_size]
    // Output: out [B*T, d_model]

    // Delegate to the standard matmul dispatcher
    // The Mamba out_proj is functionally identical to a standard matmul
    dispatch_matmul(op, hook);
}

void CompiledExecutor::dispatch_mamba_out_proj_backward(const CompiledOp& op, const modules::BackwardHook* hook) {
    // This is a standard matmul backward
    // Delegate to the standard matmul backward dispatcher
    dispatch_matmul_backward(op, hook);
}

namespace {

// -----------------------------------------------------------------------------
// Mamba out_proj backward rule
// Forward: out = mamba_out_proj(inp, weight)
// This is just a matmul, so we delegate to the matmul rule via the
// registry (its definition lives in runtime/ops/matmul.cpp).
// -----------------------------------------------------------------------------
std::vector<Operation> mamba_out_proj_backward(const BackwardRuleContext& ctx) {
    const OpDescriptor* desc = OpRegistry::instance().find_by_name("matmul");
    if (!desc || !desc->autodiff_fn) {
        throw std::runtime_error("mamba_out_proj_backward: matmul autodiff rule not registered");
    }
    return desc->autodiff_fn(ctx);
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("mamba_out_proj", ::dsl::mamba_out_proj_backward);
