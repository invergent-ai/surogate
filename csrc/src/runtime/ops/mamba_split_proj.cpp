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


    auto shape_matches = [](const TensorRef& ref, long expected) -> bool {
        if (ref.shape.empty()) return false;
        long prod = 1;
        for (auto d : ref.shape) {
            if (d <= 0) return false;
            prod *= d;
        }
        return prod == expected;
    };

    const long gate_expected = static_cast<long>(B) * T * intermediate_size;
    const long conv_expected = static_cast<long>(B) * conv_dim * T;
    const long delta_expected = static_cast<long>(B) * intermediate_size * T;

    Tensor* gate_ptr = nullptr;
    Tensor* conv_ptr = nullptr;
    Tensor* delta_ptr = nullptr;

    if (shape_matches(op.outputs[0], gate_expected)) {
        Tensor& gate_ref = ensure_output_tensor(op.outputs[0]);
        if (gate_ref.nelem() == gate_expected) {
            gate_ptr = &gate_ref;
        }
    }
    if (!gate_ptr) {
        Tensor gate = mRunState.temp_alloc(proj.DType, {B, T, intermediate_size});
        mTemps.push_back(gate);
        gate_ptr = &mTemps.back();
    }

    if (shape_matches(op.outputs[1], conv_expected)) {
        Tensor& conv_ref = ensure_output_tensor(op.outputs[1]);
        if (conv_ref.nelem() == conv_expected) {
            conv_ptr = &conv_ref;
        }
    }
    if (!conv_ptr) {
        Tensor conv_in = mRunState.temp_alloc(proj.DType, {B, conv_dim, T});
        mTemps.push_back(conv_in);
        conv_ptr = &mTemps.back();
    }

    if (shape_matches(op.outputs[2], delta_expected)) {
        Tensor& delta_ref = ensure_output_tensor(op.outputs[2]);
        if (delta_ref.nelem() == delta_expected) {
            delta_ptr = &delta_ref;
        }
    }
    if (!delta_ptr) {
        Tensor delta = mRunState.temp_alloc(proj.DType, {B, intermediate_size, T});
        mTemps.push_back(delta);
        delta_ptr = &mTemps.back();
    }

    // Call kernel
    mamba_split_proj(*gate_ptr, *conv_ptr, *delta_ptr, proj,
                     B, T, intermediate_size, conv_dim, num_heads, head_dim,
                     mRunState.MainStream);

    // Store outputs (ensure_output_tensor may already insert into map)
    mTensorMap[op.outputs[0].name] = *gate_ptr;
    mTensorMap[op.outputs[1].name] = *conv_ptr;
    mTensorMap[op.outputs[2].name] = *delta_ptr;
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

    mTensorMap[op.outputs[0].name] = d_proj;
}

}  // namespace dsl
