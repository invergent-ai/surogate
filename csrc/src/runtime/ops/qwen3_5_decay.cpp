// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 decay operation dispatch.

#include "runtime/dsl/compiled_ops.h"

#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"

namespace dsl {

void CompiledExecutor::dispatch_qwen3_5_decay(const CompiledOp& op) {
    // Inputs: a [B,T,H], A_log [H], dt_bias [H]
    // Output: g [B,T,H] = -exp(A_log) * softplus(a + dt_bias)
    if (op.inputs.size() < 3) {
        throw std::runtime_error("qwen3_5_decay: expected inputs (a, A_log, dt_bias)");
    }
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& a_log = resolve_tensor(op.inputs[1]);
    Tensor& dt_bias = resolve_tensor(op.inputs[2]);

    if (a.Rank != 3 || a_log.Rank != 1 || dt_bias.Rank != 1) {
        throw std::runtime_error("qwen3_5_decay: expected a rank-3 and A_log/dt_bias rank-1");
    }
    const long B = a.Sizes[0];
    const long T = a.Sizes[1];
    const long H = a.Sizes[2];
    if (a_log.Sizes[0] != H || dt_bias.Sizes[0] != H) {
        throw std::runtime_error("qwen3_5_decay: head dimension mismatch");
    }

    Tensor out = ensure_output_tensor(op.outputs[0]);
    if (out.Rank != 3 || out.Sizes[0] != B || out.Sizes[1] != T || out.Sizes[2] != H || out.DType != a.DType) {
        out = mRunState.temp_alloc(a.DType, {B, T, H});
        mTemps.push_back(out);
    }
    qwen3_5_decay_forward(out, a, a_log, dt_bias, mRunState.MainStream);
    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_qwen3_5_decay_backward(const CompiledOp& op) {
    // Inputs: d_out [B,T,H], a [B,T,H], A_log [H], dt_bias [H]
    // Outputs: d_a [B,T,H], d_A_log [H], d_dt_bias [H]
    if (op.inputs.size() < 4) {
        throw std::runtime_error("qwen3_5_decay_backward: expected inputs (d_out, a, A_log, dt_bias)");
    }
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& a = resolve_tensor(op.inputs[1]);
    Tensor& a_log = resolve_tensor(op.inputs[2]);
    Tensor& dt_bias = resolve_tensor(op.inputs[3]);

    if (d_out.Rank != 3 || a.Rank != 3 || a_log.Rank != 1 || dt_bias.Rank != 1) {
        throw std::runtime_error("qwen3_5_decay_backward: invalid ranks");
    }
    if (d_out.Sizes[0] != a.Sizes[0] || d_out.Sizes[1] != a.Sizes[1] || d_out.Sizes[2] != a.Sizes[2]) {
        throw std::runtime_error("qwen3_5_decay_backward: d_out shape must match a");
    }
    const long B = a.Sizes[0];
    const long T = a.Sizes[1];
    const long H = a.Sizes[2];
    if (a_log.Sizes[0] != H || dt_bias.Sizes[0] != H) {
        throw std::runtime_error("qwen3_5_decay_backward: head dimension mismatch");
    }

    auto ensure_or_temp = [&](std::size_t out_idx,
                              ETensorDType dtype,
                              const std::vector<long>& shape) -> Tensor {
        if (op.outputs.size() > out_idx && !op.outputs[out_idx].name.empty()) {
            Tensor out = ensure_output_tensor(op.outputs[out_idx]);
            if (out.DType == dtype &&
                out.Rank == static_cast<int>(shape.size())) {
                bool match = true;
                for (int i = 0; i < out.Rank; ++i) {
                    if (out.Sizes[i] != shape[static_cast<std::size_t>(i)]) {
                        match = false;
                        break;
                    }
                }
                if (match) return out;
            }
        }
        Tensor out = mRunState.temp_alloc(dtype, shape);
        mTemps.push_back(out);
        return out;
    };

    Tensor d_a = ensure_or_temp(0, a.DType, {B, T, H});
    Tensor d_a_log = ensure_or_temp(1, ETensorDType::FP32, {H});
    Tensor d_dt_bias = ensure_or_temp(2, ETensorDType::FP32, {H});

    qwen3_5_decay_backward(d_a, d_a_log, d_dt_bias,
                           d_out, a, a_log, dt_bias, mRunState.MainStream);

    if (op.outputs.size() > 0 && !op.outputs[0].name.empty()) store_tensor(op.outputs[0], d_a);
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) store_tensor(op.outputs[1], d_a_log);
    if (op.outputs.size() > 2 && !op.outputs[2].name.empty()) store_tensor(op.outputs[2], d_dt_bias);
}

}  // namespace dsl

