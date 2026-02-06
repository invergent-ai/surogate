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

    // Allocate outputs
    Tensor gate = mRunState.temp_alloc(proj.DType, {B, T, intermediate_size});
    mTemps.push_back(gate);

    Tensor conv_in = mRunState.temp_alloc(proj.DType, {B, conv_dim, T});
    mTemps.push_back(conv_in);

    Tensor delta = mRunState.temp_alloc(proj.DType, {B, intermediate_size, T});
    mTemps.push_back(delta);

    // Call kernel
    mamba_split_proj(gate, conv_in, delta, proj,
                     B, T, intermediate_size, conv_dim, num_heads, head_dim,
                     mRunState.MainStream);

    // Store outputs
    mTensorMap[op.outputs[0].name] = gate;
    mTensorMap[op.outputs[1].name] = conv_in;
    mTensorMap[op.outputs[2].name] = delta;
}

void CompiledExecutor::dispatch_mamba_split_proj_backward(const CompiledOp& op) {
    // Inputs: d_gate, d_conv_in, d_dt
    // Output: d_proj
    Tensor& d_gate = resolve_tensor(op.inputs[0]);
    Tensor& d_conv_in = resolve_tensor(op.inputs[1]);
    Tensor& d_dt = resolve_tensor(op.inputs[2]);

    const int B = static_cast<int>(d_gate.Sizes[0]);
    const int T = static_cast<int>(d_gate.Sizes[1]);
    const int intermediate_size = static_cast<int>(d_gate.Sizes[2]);

    const int conv_dim = op.attrs.conv_dim;
    const int num_heads = op.attrs.mamba_num_heads;

    // Allocate output
    const int proj_size = intermediate_size + conv_dim + num_heads;
    Tensor d_proj = mRunState.temp_alloc(d_gate.DType, {B, T, proj_size});
    mTemps.push_back(d_proj);

    // Call kernel
    mamba_pack_dproj(d_proj, d_gate, d_conv_in, d_dt,
                     B, T, intermediate_size, conv_dim, num_heads,
                     mRunState.MainStream);

    mTensorMap[op.outputs[0].name] = d_proj;
}

}  // namespace dsl
