// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 decay operation dispatch.

#include "runtime/executor/compiled_ops.h"

#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
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
    if (out.Rank != 3 || out.Sizes[0] != B || out.Sizes[1] != T || out.Sizes[2] != H ||
        out.DType != ETensorDType::FP32) {
        out = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H}, "qwen3_5_decay_out");
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

    auto shape_matches = [](const Tensor& t, const std::vector<long>& shape) -> bool {
        if (t.Rank != static_cast<int>(shape.size())) {
            return false;
        }
        for (int i = 0; i < t.Rank; ++i) {
            if (t.Sizes[i] != shape[static_cast<std::size_t>(i)]) {
                return false;
            }
        }
        return true;
    };

    auto maybe_output = [&](std::size_t out_idx, const std::vector<long>& shape) -> Tensor {
        if (op.outputs.size() > out_idx && !op.outputs[out_idx].name.empty()) {
            Tensor out = ensure_output_tensor(op.outputs[out_idx]);
            if (shape_matches(out, shape)) {
                return out;
            }
        }
        return Tensor{};
    };

    auto alloc_temp = [&](ETensorDType dtype, const std::vector<long>& shape) -> Tensor {
        Tensor out = mRunState.temp_alloc(dtype, shape, "qwen3_5_decay_backward_temp");
        mTemps.push_back(out);
        return out;
    };

    auto copy_or_accumulate = [&](const TensorRef& out_ref, const Tensor& src, bool allow_accumulate) {
        if (out_ref.name.empty()) {
            return;
        }
        Tensor dst = ensure_output_tensor(out_ref);
        const std::vector<long> src_shape(src.Sizes.begin(), src.Sizes.begin() + src.Rank);
        bool is_param_grad = false;
        if (auto base = base_param_from_grad(out_ref.name)) {
            bool acc = false;
            if (Tensor* g = mGrads.get_param_grad(*base, acc)) {
                is_param_grad = (g && g->Data);
            }
        }
        if (!shape_matches(dst, src_shape)) {
            if (is_param_grad) {
                throw std::runtime_error("qwen3_5_decay_backward: output shape mismatch for " + out_ref.name);
            }
            dst = alloc_temp(src.DType, src_shape);
        }

        bool accumulate = false;
        if (allow_accumulate) {
            accumulate = mAccumulateTensors.count(out_ref.name) > 0;
            if (!accumulate) {
                if (auto base = base_param_from_grad(out_ref.name)) {
                    accumulate = mAccumulateTensors.count("d_" + *base) > 0;
                }
            }
        }

        Tensor src_use = src;
        if (dst.DType != src.DType) {
            src_use = alloc_temp(dst.DType, {static_cast<long>(src.nelem())});
            if (dst.DType == ETensorDType::BF16 && src.DType == ETensorDType::FP32) {
                convert_dtype(src_use.get<nv_bfloat16>(), src.get<float>(), src.nelem(), mRunState.MainStream);
            } else if (dst.DType == ETensorDType::FP32 && src.DType == ETensorDType::BF16) {
                convert_dtype(src_use.get<float>(), src.get<nv_bfloat16>(), src.nelem(), mRunState.MainStream);
            } else {
                throw std::runtime_error("qwen3_5_decay_backward: unsupported dtype conversion for " + out_ref.name);
            }
            src_use = view_tensor(src_use, std::vector<long>(src.Sizes.begin(), src.Sizes.begin() + src.Rank));
        }

        if (dst.Data != src_use.Data) {
            if (accumulate) {
                vector_add_sr(dst, dst, src_use, 1.0f, static_cast<long>(dst.nelem()), 0, mRunState.MainStream);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(dst.Data,
                                           src_use.Data,
                                           dst.bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           mRunState.MainStream));
            }
        }
        store_tensor(out_ref, dst);
    };

    Tensor d_a_target = maybe_output(0, {B, T, H});
    Tensor d_a_log_target = maybe_output(1, {H});
    Tensor d_dt_bias_target = maybe_output(2, {H});

    Tensor d_a = d_a_target.Data && d_a_target.DType == a.DType ? d_a_target : alloc_temp(a.DType, {B, T, H});

    Tensor d_a_log = d_a_log_target.Data && d_a_log_target.DType == ETensorDType::FP32
                         ? d_a_log_target
                         : alloc_temp(ETensorDType::FP32, {H});

    Tensor d_dt_bias = d_dt_bias_target.Data && d_dt_bias_target.DType == ETensorDType::FP32
                           ? d_dt_bias_target
                           : alloc_temp(ETensorDType::FP32, {H});

    qwen3_5_decay_backward(d_a, d_a_log, d_dt_bias, d_out, a, a_log, dt_bias, mRunState.MainStream);

    if (op.outputs.size() > 0 && !op.outputs[0].name.empty()) {
        if (d_a_target.Data && d_a_target.Data == d_a.Data) {
            store_tensor(op.outputs[0], d_a_target);
        } else {
            copy_or_accumulate(op.outputs[0], d_a, /*allow_accumulate=*/false);
        }
    }
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        if (d_a_log_target.Data && d_a_log_target.Data == d_a_log.Data) {
            store_tensor(op.outputs[1], d_a_log_target);
        } else {
            copy_or_accumulate(op.outputs[1], d_a_log, /*allow_accumulate=*/true);
        }
    }
    if (op.outputs.size() > 2 && !op.outputs[2].name.empty()) {
        if (d_dt_bias_target.Data && d_dt_bias_target.Data == d_dt_bias.Data) {
            store_tensor(op.outputs[2], d_dt_bias_target);
        } else {
            copy_or_accumulate(op.outputs[2], d_dt_bias, /*allow_accumulate=*/true);
        }
    }
}

namespace {

// -----------------------------------------------------------------------------
// Qwen3.5 decay backward rule
// Forward: g = qwen3_5_decay(a, A_log, dt_bias)
// Backward: d_a, d_A_log, d_dt_bias = qwen3_5_decay_backward(d_g, a, A_log, dt_bias)
// -----------------------------------------------------------------------------
std::vector<Operation> qwen3_5_decay_backward_rule(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;
    const auto& fwd = ctx.fwd_op;
    if (fwd.inputs.size() < 3) {
        return ops;
    }
    if (!(ctx.needs_grad(0) || ctx.needs_grad(1) || ctx.needs_grad(2))) {
        return ops;
    }

    const std::string& a = fwd.inputs[0];
    const std::string& A_log = fwd.inputs[1];
    const std::string& dt_bias = fwd.inputs[2];
    const std::string a_ref = ctx.is_param(a) ? a : saved_ref(a);
    const std::string a_log_ref = ctx.is_param(A_log) ? A_log : saved_ref(A_log);
    const std::string dt_bias_ref = ctx.is_param(dt_bias) ? dt_bias : saved_ref(dt_bias);

    ops.push_back(make_operation("qwen3_5_decay_backward_" + std::to_string(ctx.op_counter++),
                                 "qwen3_5_decay_backward",
                                 "qwen3_5_decay_backward",
                                 {ctx.d_output, a_ref, a_log_ref, dt_bias_ref},
                                 {ctx.needs_grad(0) ? ctx.d_inputs[0] : "",
                                  ctx.needs_grad(1) ? ctx.d_inputs[1] : "",
                                  ctx.needs_grad(2) ? ctx.d_inputs[2] : ""}));
    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("qwen3_5_decay", ::dsl::qwen3_5_decay_backward_rule);
