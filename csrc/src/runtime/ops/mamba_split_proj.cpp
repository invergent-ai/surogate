// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Mamba split_proj operation dispatch.

#include "runtime/dsl/compiled_ops.h"

#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_mamba_split_proj(const CompiledOp& op) {
    // Input: projected [B, T, P] where P = intermediate_size + conv_dim + num_heads
    // Outputs: gate [B, T, intermediate_size], conv_input [B, conv_dim, T], dt [B, T, num_heads]
    Tensor& proj = resolve_tensor(op.inputs[0]);

    const int B = static_cast<int>(proj.Sizes[0]);
    const int T = static_cast<int>(proj.Sizes[1]);
    const int P = static_cast<int>(proj.Sizes[2]);

    // Get Mamba dimensions from attributes
    const int intermediate_size = op.attrs.intermediate_size;
    const int conv_dim = op.attrs.conv_dim;
    const int num_heads = op.attrs.mamba_num_heads;
    const int head_dim = op.attrs.mamba_head_dim;


    // Allocate all output tensors upfront (before any mTemps.push_back)
    // to avoid dangling pointers from vector reallocation.
    Tensor gate_t = mRunState.temp_alloc(proj.DType, {B, T, intermediate_size});
    Tensor conv_t = mRunState.temp_alloc(proj.DType, {B, conv_dim, T});
    Tensor delta_t = mRunState.temp_alloc(proj.DType, {B, intermediate_size, T});
    mTemps.push_back(gate_t);
    mTemps.push_back(conv_t);
    mTemps.push_back(delta_t);

    // Call kernel
    mamba_split_proj(gate_t, conv_t, delta_t, proj,
                     B, T, intermediate_size, conv_dim, num_heads, head_dim,
                     mRunState.MainStream);

    // Store outputs
    store_tensor(op.outputs[0], gate_t);
    store_tensor(op.outputs[1], conv_t);
    store_tensor(op.outputs[2], delta_t);
}

void CompiledExecutor::dispatch_mamba_split_proj_backward(const CompiledOp& op) {
    // Inputs: d_gate, d_conv_in, d_delta
    // Output: d_proj
    // d_delta is [B, D, T] (expanded) and needs reduction to [B, T, num_heads]
    Tensor& d_gate = resolve_tensor(op.inputs[0]);
    Tensor& d_conv_in = resolve_tensor(op.inputs[1]);
    Tensor& d_delta = resolve_tensor(op.inputs[2]);

    const int B = static_cast<int>(d_gate.Sizes[0]);
    const int T = static_cast<int>(d_gate.Sizes[1]);
    const int intermediate_size = static_cast<int>(d_gate.Sizes[2]);

    const int conv_dim = op.attrs.conv_dim;
    const int num_heads = op.attrs.mamba_num_heads;
    const int head_dim = op.attrs.mamba_head_dim;

    // Allocate output
    const int proj_size = intermediate_size + conv_dim + num_heads;
    Tensor d_proj = mRunState.temp_alloc(d_gate.DType, {B, T, proj_size});
    mTemps.push_back(d_proj);

    // Call kernel — reduces d_delta [B, D, T] to per-head d_dt inline
    mamba_pack_dproj(d_proj, d_gate, d_conv_in, d_delta,
                     B, T, intermediate_size, conv_dim, num_heads, head_dim,
                     mRunState.MainStream);

    store_tensor(op.outputs[0], d_proj);
}

}  // namespace dsl
