#include "runtime/executor/glm_decode_state.h"
#include "kernels/decode.h"
// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#include "runtime/executor/compiled_ops.h"

#include <numeric>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include "kernels/glm5.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/buffer_plan.h"
#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/executor/op_registry.h"

namespace dsl {
namespace {
Glm5Kernel kernel(CompiledOpType type) {
    switch (type) {
        case CompiledOpType::MhcMix:
        case CompiledOpType::MhcMixBackward: return Glm5Kernel::MhcMix;
        case CompiledOpType::MhcCombine:
        case CompiledOpType::MhcCombineBackward: return Glm5Kernel::MhcCombine;
        case CompiledOpType::KdaDecay:
        case CompiledOpType::KdaDecayBackward: return Glm5Kernel::KdaDecay;
        case CompiledOpType::KimiDeltaRule:
        case CompiledOpType::KimiDeltaRuleBackward: return Glm5Kernel::KdaRule;
        case CompiledOpType::Clamp:
        case CompiledOpType::ClampBackward: return Glm5Kernel::Clamp;
        case CompiledOpType::GlmCausalConv1d:
        case CompiledOpType::GlmCausalConv1dBackward: return Glm5Kernel::CausalConv1d;
        default: throw std::runtime_error("invalid GLM training operation");
    }
}
Glm5Options options(const CompiledAttrs& a) {
    return {a.hc_mult,
            a.hc_sinkhorn_iters,
            a.hc_eps,
            a.eps,
            a.kda_lower_bound,
            a.clamp_min,
            a.clamp_max,
            a.clamp_fused_gate_up};
}
std::vector<long> shape(const Tensor& t) {
    return {t.Sizes.begin(), t.Sizes.begin() + t.Rank};
}

void validate(Glm5Kernel k, const std::vector<Tensor>& x, const Glm5Options& o) {
    auto require = [](bool ok, const char* message) {
        if (!ok) throw std::runtime_error(message);
    };
    if (k == Glm5Kernel::KdaRule) {
        require((x.size() == 5 || x.size() == 6) && x[0].Rank == 4, "KDA requires q/k/v/g [B,T,H,D] and beta [B,T,H]");
        for (int i = 1; i < 4; ++i)
            require(shape(x[i]) == shape(x[0]), "KDA q/k/v/g shapes must match");
        require(x[4].Rank == 3 && x[4].nelem() * x[0].Sizes[3] == x[0].nelem(), "invalid KDA beta shape");
        require(x[3].DType == ETensorDType::FP32, "KDA decay must be FP32");
    } else if (k == Glm5Kernel::CausalConv1d) {
        require(x.size() == 3 && x[0].Rank == 3 && x[1].Rank == 3 && x[1].Sizes[0] == x[0].Sizes[2] &&
                    x[1].Sizes[1] == 1 && x[1].Sizes[2] > 0,
                "GLM convolution requires x [B,T,C], weight [C,1,K], positions [B,T]");
    } else if (k == Glm5Kernel::KdaDecay) {
        require(x.size() == 3 && x[0].Rank == 4, "KDA decay requires x [B,T,H,D], A_log [H], bias [H*D]");
        require(x[1].Rank == 1 && x[1].Sizes[0] == x[0].Sizes[2] && x[2].Rank == 1 &&
                    x[2].Sizes[0] == x[0].Sizes[2] * x[0].Sizes[3],
                "invalid KDA decay parameter shapes");
        require(std::isfinite(o.lower_bound) && o.lower_bound < 0, "KDA lower bound must be finite and negative");
    } else if (k == Glm5Kernel::MhcMix || k == Glm5Kernel::MhcCombine) {
        require(x.size() == 4 && x[0].Rank == 3 && o.streams > 0 && x[0].Sizes[2] % o.streams == 0,
                "mHC requires residual [B,T,H*C]");
        int H = o.streams, M = H * (H + 2);
        if (k == Glm5Kernel::MhcMix) {
            if (!(x[1].Rank == 2 && x[1].Sizes[0] == M && x[1].Sizes[1] == x[0].Sizes[2] && x[2].nelem() == M &&
                  x[3].nelem() == 3)) {
                std::ostringstream message;
                message << "invalid mHC residual/fn/base/scale shapes:";
                for (const auto& t : x) {
                    message << " [";
                    for (int i = 0; i < t.Rank; ++i)
                        message << (i ? "," : "") << t.Sizes[i];
                    message << "]";
                }
                throw std::runtime_error(message.str());
            }
        } else {
            require(x[1].nelem() * H == x[0].nelem() && x[2].nelem() == x[0].Sizes[0] * x[0].Sizes[1] * H &&
                        x[3].nelem() == x[2].nelem() * H,
                    "invalid mHC combine shapes");
        }
    } else
        require(x.size() == 1 && o.clamp_min <= o.clamp_max, "invalid clamp inputs/bounds");
    if (k == Glm5Kernel::CausalConv1d || (k == Glm5Kernel::KdaRule && x.size() == 6)) {
        const auto& pos = x.back();
        require(pos.DType == ETensorDType::INT32 && pos.Rank == 2 && pos.Sizes[0] == x[0].Sizes[0] &&
                    pos.Sizes[1] == x[0].Sizes[1],
                "GLM sequence positions must be INT32 [B,T]");
    }
}

std::vector<Operation> backward_rule(const BackwardRuleContext& ctx) {
    const auto& f = ctx.fwd_op;
    std::vector<std::string> in, out;
    if (f.kernel_type == "mhc_mix" || f.name == "mhc_mix") {
        for (size_t i = 0; i < 3; ++i)
            in.push_back(i < ctx.d_outputs.size() ? ctx.d_outputs[i] : "");
    } else
        in.push_back(ctx.d_output);
    for (size_t i = 0; i < f.inputs.size(); ++i) {
        in.push_back(ctx.is_param(f.inputs[i]) ? f.inputs[i] : saved_ref(f.inputs[i]));
        const auto& type = f.kernel_type;
        bool positions = (type == "chunk_kimi_delta_rule" && i == 5) || (type == "glm_causal_conv1d" && i == 2);
        if (!positions) out.push_back(ctx.needs_grad(i) ? ctx.d_inputs[i] : "");
    }
    auto name = (f.kernel_type.empty() || f.kernel_type == "custom" ? f.name : f.kernel_type) + "_backward";
    return {make_operation(name + "_" + std::to_string(ctx.op_counter++), name, name, in, out, f.attrs)};
}

long stack_bound(const CompiledOp& op, const BufferPlan& plan) {
    long bytes = 0;
    const size_t offset = op.type == CompiledOpType::MhcMixBackward ? 3 : 1;
    for (size_t i = offset; i < op.inputs.size(); ++i) {
        const auto& s = op.inputs[i].shape;
        if (!s.empty()) bytes += align_stack_bytes(std::accumulate(s.begin(), s.end(), 6L, std::multiplies<long>()));
    }
    const bool kda_backward = op.type == CompiledOpType::KimiDeltaRuleBackward;
    if (op.type == CompiledOpType::KimiDeltaRule || kda_backward) {
        const auto& s = op.inputs[kda_backward ? 1 : 0].shape;
        if (s.size() == 4) {
            // Documents can be one token at planning time. Execution allocates
            // only the bound for the actual document count.
            bytes += align_stack_bytes(
                KimiDeltaRuleKernels::workspace_bytes(plan.B, plan.T, s[2], s[3], plan.B * plan.T, kda_backward));
        }
    }
    return bytes;
}
void forward(CompiledExecutor& e, const CompiledOp& op, const void*) {
    e.dispatch_glm5(op);
}
void backward(CompiledExecutor& e, const CompiledOp& op, const void*) {
    e.dispatch_glm5_backward(op);
}
}  // namespace

void CompiledExecutor::dispatch_glm5(const CompiledOp& op) {
    auto k = kernel(op.type);
    auto opts = options(op.attrs);
    std::vector<Tensor> inputs, outputs;
    for (const auto& ref : op.inputs)
        inputs.push_back(resolve_tensor(ref));
    validate(k, inputs, opts);
    // The recurrent kernel currently processes a complete sequence, so don't
    // silently drop carry state when used by sequence-chunked dispatch-PP.
    if (k == Glm5Kernel::KdaRule && sequence_chunk_active())
        throw std::runtime_error("GLM KDA training does not yet support sequence-chunked dispatch-PP");
    for (size_t i = 0; i < op.outputs.size(); ++i) {
        auto s = shape(inputs[0]);
        auto dtype = inputs[0].DType;
        if (k == Glm5Kernel::MhcMix) {
            if (i == 0)
                s[2] /= opts.streams;
            else {
                s = {s[0] * s[1], opts.streams};
                if (i == 2) s.push_back(opts.streams);
                dtype = ETensorDType::FP32;
            }
        } else if (k == Glm5Kernel::KdaDecay)
            dtype = ETensorDType::FP32;
        Tensor out = ensure_output_tensor(op.outputs[i]);
        if (shape(out) != s || out.DType != dtype) {
            out = mRunState.temp_alloc(dtype, s, "glm5_forward_output");
            mTemps.push_back(out);
        }
        outputs.push_back(out);
    }
    if (mExecutionRequest && mExecutionRequest->decoding() &&
        (k == Glm5Kernel::KdaRule || k == Glm5Kernel::CausalConv1d)) {
        for (int row = 0; row < inputs[0].Sizes[0]; ++row) {
            auto* decode = mExecutionRequest->decode_state(row);
            auto sliced = inputs;
            if (k == Glm5Kernel::KdaRule) {
                for (auto& tensor : sliced)
                    tensor = decode_batch_row(tensor, row);
                const auto& q = sliced[0];
                auto state = decode->get(op_layer_idx(op),
                                         "kda_state",
                                         ETensorDType::FP32,
                                         {1, q.Sizes[2], q.Sizes[3], q.Sizes[3]});
                mKdaKernels.recurrent(sliced,
                                      decode_batch_row(outputs[0], row),
                                      state,
                                      decode->length == 0,
                                      mRunState.MainStream);
            } else {
                auto x = decode_batch_row(inputs[0], row);
                auto history =
                    decode->get(op_layer_idx(op), "convolution", x.DType, {x.Sizes[2], inputs[1].Sizes[2] - 1});
                glm5_convolution_state(x,
                                       inputs[1],
                                       history,
                                       decode_batch_row(outputs[0], row),
                                       decode->length == 0,
                                       mRunState.MainStream);
            }
        }
    } else if (k == Glm5Kernel::KdaRule && inputs[0].DType == ETensorDType::BF16 && mOptions.DocMasking) {
        const auto& q = inputs[0];
        if (mCuSeqlensGpu && mTotalDocTokens != q.Sizes[0] * q.Sizes[1])
            throw std::runtime_error("KDA document metadata must cover all batch tokens");
        Tensor workspace = mRunState.temp_alloc(ETensorDType::BYTE,
                                                {static_cast<long>(KimiDeltaRuleKernels::workspace_bytes(q.Sizes[0],
                                                                                                         q.Sizes[1],
                                                                                                         q.Sizes[2],
                                                                                                         q.Sizes[3],
                                                                                                         mNumDocs,
                                                                                                         false))},
                                                "kda_workspace");
        mKdaKernels.run(false, inputs, outputs, mCuSeqlensGpu, mNumDocs, workspace, mRunState.MainStream,
                        mOptions.GlmRolloutParity);
        mRunState.Stack.free(workspace);
    } else {
        glm5_forward(k, inputs, outputs, opts, mRunState.MainStream);
    }
    for (size_t i = 0; i < outputs.size(); ++i)
        store_tensor(op.outputs[i], outputs[i]);
}

void CompiledExecutor::dispatch_glm5_backward(const CompiledOp& op) {
    auto k = kernel(op.type);
    size_t offset = k == Glm5Kernel::MhcMix ? 3 : 1;
    std::vector<Tensor> inputs, outputs;
    for (const auto& ref : op.inputs)
        inputs.push_back(ref.name.empty() ? Tensor{} : resolve_tensor(ref));
    for (size_t i = offset; i < inputs.size(); ++i) {
        if (!inputs[i].Data || inputs[i].Rank == 0)
            throw std::runtime_error("GLM backward is missing saved input " + op.inputs[i].name);
    }
    validate(k, {inputs.begin() + offset, inputs.end()}, options(op.attrs));
    for (size_t i = 0; i < op.outputs.size(); ++i) {
        Tensor t;
        if (!op.outputs[i].name.empty() || k == Glm5Kernel::KdaRule) {
            t = mRunState.temp_alloc(ETensorDType::FP32, shape(inputs.at(offset + i)), "glm5_gradient");
            mTemps.push_back(t);
        }
        outputs.push_back(t);
    }
    if (k == Glm5Kernel::KdaRule && inputs[1].DType == ETensorDType::BF16 && mOptions.DocMasking) {
        const auto& q = inputs[1];
        if (mCuSeqlensGpu && mTotalDocTokens != q.Sizes[0] * q.Sizes[1])
            throw std::runtime_error("KDA document metadata must cover all batch tokens");
        Tensor workspace = mRunState.temp_alloc(
            ETensorDType::BYTE,
            {static_cast<long>(
                KimiDeltaRuleKernels::workspace_bytes(q.Sizes[0], q.Sizes[1], q.Sizes[2], q.Sizes[3], mNumDocs, true))},
            "kda_workspace");
        mKdaKernels.run(true, inputs, outputs, mCuSeqlensGpu, mNumDocs, workspace, mRunState.MainStream);
        mRunState.Stack.free(workspace);
    } else {
        // The reference path supports FP32 activations and position resets when
        // attention document masking (and thus its cu_seqlens) is disabled.
        Tensor checkpoints;
        if (k == Glm5Kernel::KdaRule) {
            const auto& q = inputs[1];
            checkpoints = mRunState.temp_alloc(ETensorDType::FP32,
                                               {q.Sizes[0],
                                                (q.Sizes[1] + GLM5_KDA_CHECKPOINT - 1) / GLM5_KDA_CHECKPOINT,
                                                q.Sizes[2],
                                                q.Sizes[3],
                                                q.Sizes[3]},
                                               "kda_reference_checkpoints");
        }
        glm5_backward(k, inputs, outputs, options(op.attrs), checkpoints, mRunState.MainStream);
        if (checkpoints.Data) mRunState.Stack.free(checkpoints);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& ref = op.outputs[i];
        if (ref.name.empty()) continue;
        Tensor dst = ensure_output_tensor(ref);
        bool accumulate = mAccumulateTensors.count(ref.name) > 0;
        if (mCurrentGraph)
            if (auto base = base_param_from_grad_kind(ref.tensor_id, *mCurrentGraph))
                accumulate = accumulate || mAccumulateTensors.count("d_" + *base) > 0;
        const bool param_grad = mCurrentGraph && base_param_from_grad_kind(ref.tensor_id, *mCurrentGraph).has_value();
        // Gradient storage is configurable independently of parameter dtype.
        // Small FP32 gates may accumulate into BF16 gradients in full training.
        const auto dtype = param_grad ? dst.DType : inputs.at(offset + i).DType;
        if (shape(dst) != shape(outputs[i]) || dst.DType != dtype) {
            if (accumulate || param_grad)
                throw std::runtime_error("GLM backward gradient shape/dtype mismatch: " + ref.name);
            dst = mRunState.temp_alloc(dtype, shape(outputs[i]), "glm5_gradient_output");
            mTemps.push_back(dst);
        }
        glm5_copy_gradient(outputs[i], dst, accumulate, mRunState.MainStream);
        store_tensor(ref, dst);
    }
}
}  // namespace dsl

#define GLM_OP(NAME, TYPE, SEMANTIC)                                                                    \
    REGISTER_COMPILED_OP_NO_COMM(NAME, TYPE, ::dsl::forward, nullptr, SEMANTIC);                        \
    REGISTER_COMPILED_OP_NO_COMM(NAME "_backward", TYPE##Backward, nullptr, ::dsl::backward, SEMANTIC); \
    REGISTER_AUTODIFF(NAME, ::dsl::backward_rule);                                                      \
    REGISTER_STACK_BOUND(NAME "_backward", TYPE##Backward, ::dsl::stack_bound)
GLM_OP("mhc_mix", MhcMix, Sequence);
GLM_OP("mhc_combine", MhcCombine, Sequence);
GLM_OP("kda_decay", KdaDecay, Sequence);
GLM_OP("chunk_kimi_delta_rule", KimiDeltaRule, Sequence);
REGISTER_STACK_BOUND("chunk_kimi_delta_rule", KimiDeltaRule, ::dsl::stack_bound);
GLM_OP("clamp", Clamp, Sequence);
GLM_OP("glm_causal_conv1d", GlmCausalConv1d, Sequence);
#undef GLM_OP
