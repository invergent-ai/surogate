// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Repeat-interleave heads operation dispatch.

#include "runtime/dsl/compiled_ops.h"

#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"

namespace dsl {

void CompiledExecutor::dispatch_repeat_interleave_heads(const CompiledOp& op) {
    // Input: x [B,T,H,D]
    // Output: y [B,T,H*repeats,D]
    if (op.inputs.empty()) {
        throw std::runtime_error("repeat_interleave_heads: expected input x");
    }
    Tensor& x = resolve_tensor(op.inputs[0]);
    if (x.Rank != 4) {
        throw std::runtime_error("repeat_interleave_heads: input must be rank-4 [B,T,H,D]");
    }
    const int repeats = op.attrs.repeat_factor > 0 ? op.attrs.repeat_factor : 1;
    const long B = x.Sizes[0];
    const long T = x.Sizes[1];
    const long H = x.Sizes[2];
    const long D = x.Sizes[3];

    const std::vector<long> out_shape{B, T, H * repeats, D};
    Tensor out = ensure_output_tensor_or_persistent(
        ensure_output_tensor(op.outputs[0]),
        mRunState, mMoeSavedBuffers, mMoeSavedSizes,
        op.op_id + "." + op.outputs[0].name + ".out",
        x.DType, out_shape, "repeat_interleave_heads");
    repeat_interleave_heads_forward(out, x, repeats, mRunState.MainStream);
    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_repeat_interleave_heads_backward(const CompiledOp& op) {
    // Inputs: d_out [B,T,H*repeats,D], x [B,T,H,D]
    // Output: d_x [B,T,H,D]
    if (op.inputs.size() < 2) {
        throw std::runtime_error("repeat_interleave_heads_backward: expected inputs (d_out, x)");
    }
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& x = resolve_tensor(op.inputs[1]);
    if (x.Rank != 4 || d_out.Rank != 4) {
        throw std::runtime_error("repeat_interleave_heads_backward: rank mismatch");
    }
    const int repeats = op.attrs.repeat_factor > 0 ? op.attrs.repeat_factor : 1;
    if (d_out.Sizes[0] != x.Sizes[0] || d_out.Sizes[1] != x.Sizes[1] ||
        d_out.Sizes[2] != x.Sizes[2] * repeats || d_out.Sizes[3] != x.Sizes[3]) {
        throw std::runtime_error("repeat_interleave_heads_backward: d_out shape mismatch");
    }

    const std::vector<long> d_x_shape{x.Sizes[0], x.Sizes[1], x.Sizes[2], x.Sizes[3]};
    Tensor d_x = ensure_output_tensor_or_persistent(
        ensure_output_tensor(op.outputs[0]),
        mRunState, mMoeSavedBuffers, mMoeSavedSizes,
        op.op_id + "." + op.outputs[0].name + ".d_x",
        x.DType, d_x_shape, "repeat_interleave_heads_backward");

    repeat_interleave_heads_backward(d_x, d_out, repeats, mRunState.MainStream);
    store_tensor(op.outputs[0], d_x);
}

}  // namespace dsl
