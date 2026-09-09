// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#include "runtime/executor/compiled_ops.h"

#include "kernels/glm5.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/buffer_plan.h"
#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/executor/op_registry.h"

namespace dsl {
namespace {
void dispatch(CompiledExecutor& e, const CompiledOp& op, const void*) {
    e.dispatch_glm_dsa(op);
}
std::vector<Operation> attention_backward(const BackwardRuleContext& ctx) {
    const auto& f = ctx.fwd_op;
    return {make_operation("glm_dsa_attention_backward_" + std::to_string(ctx.op_counter++),
                           "glm_dsa_attention_backward",
                           "glm_dsa_attention_backward",
                           {ctx.d_output, saved_ref(f.inputs[0]), saved_ref(f.inputs[1])},
                           {ctx.d_inputs[0]},
                           {})};
}
long stack_bound(const CompiledOp& op, const BufferPlan&) {
    if (op.type == CompiledOpType::GlmDsaIndexer) {
        const auto& s = op.inputs.at(0).shape;
        const long P = op.inputs.at(6).shape[0], NP = (s[1] + P - 1) / P;
        return align_stack_bytes(
            s[0] * (s[1] * (s[3] * 2 + NP * 4 + op.outputs[0].shape[2] * 4 / P) + NP * (s[3] * 2 + 4)) + 2048);
    }
    const auto& s = op.inputs.at(op.type == CompiledOpType::GlmDsaAttentionBackward ? 1 : 0).shape;
    if (s.size() != 4) throw std::runtime_error("DSA stack planning needs QKV shape for " + op.inputs.at(1).name);
    return align_stack_bytes(s[0] * s[1] * s[2] * s[3] * 10 + 4096);
}
}  // namespace

void CompiledExecutor::dispatch_glm_dsa(const CompiledOp& op) {
    std::vector<Tensor> inputs;
    for (const auto& ref : op.inputs)
        inputs.push_back(resolve_tensor(ref));
    auto stream = mRunState.MainStream;
    auto* cache = mExecutionRequest ? mExecutionRequest->glm_decode_state : nullptr;
    const int layer = op_layer_idx(op);
    if (op.type == CompiledOpType::GlmDsaIndexer) {
        auto output = ensure_output_tensor(op.outputs[0]);
        const auto& q = inputs[0];
        auto scratch = mRunState.temp_alloc(
            ETensorDType::BYTE,
            {static_cast<long>(mDsaKernels.workspace_bytes(q.Sizes[0], q.Sizes[1], cache ? cache->length : 0))},
            "dsa_workspace");
        mDsaKernels.indexer(inputs, output, scratch, stream, cache, layer);
        mRunState.Stack.free(scratch);
        store_tensor(op.outputs[0], output);
        return;
    }
    bool backward = op.type == CompiledOpType::GlmDsaAttentionBackward;
    if (backward && cache) throw std::runtime_error("Decode caches cannot be used for training backward");
    const auto& qkv = inputs[backward ? 1 : 0];
    const auto& indices = inputs[backward ? 2 : 1];
    const long B = qkv.Sizes[0], T = qkv.Sizes[1], H = qkv.Sizes[2] / 3, D = qkv.Sizes[3];
    Tensor out, result;
    if (backward) result = ensure_output_tensor(op.outputs[0]);
    if (backward)
        out = mRunState.temp_alloc(ETensorDType::BF16, {B, T, H * D}, "dsa_recompute");
    else
        out = ensure_output_tensor(op.outputs[0]);
    auto lse = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H}, "dsa_lse");
    mDsaKernels.attention(qkv, indices, out, lse, stream, cache, layer);
    if (backward) {
        auto dout = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H * D}, "dsa_dout");
        glm5_copy_gradient(inputs[0], dout, false, stream);
        auto dqkv = mRunState.temp_alloc(ETensorDType::FP32, {B, T, 3 * H, D}, "dsa_dqkv");
        mDsaKernels.backward(dout, qkv, indices, out, lse, dqkv, stream);
        glm5_copy_gradient(dqkv, result, mAccumulateTensors.count(op.outputs[0].name) > 0, stream);
        store_tensor(op.outputs[0], result);
        mRunState.Stack.free(dqkv);
        mRunState.Stack.free(dout);
    } else
        store_tensor(op.outputs[0], out);
    mRunState.Stack.free(lse);
    if (backward) mRunState.Stack.free(out);
}
}  // namespace dsl

REGISTER_COMPILED_OP_NO_COMM("glm_dsa_indexer", GlmDsaIndexer, ::dsl::dispatch, nullptr, Sequence);
REGISTER_AUTODIFF("glm_dsa_indexer", [](const dsl::BackwardRuleContext&) { return std::vector<dsl::Operation>{}; });
REGISTER_COMPILED_OP_NO_COMM("glm_dsa_attention", GlmDsaAttention, ::dsl::dispatch, nullptr, Sequence);
REGISTER_AUTODIFF("glm_dsa_attention", ::dsl::attention_backward);
REGISTER_COMPILED_OP_NO_COMM("glm_dsa_attention_backward", GlmDsaAttentionBackward, nullptr, ::dsl::dispatch, Sequence);
REGISTER_STACK_BOUND("glm_dsa_indexer", GlmDsaIndexer, ::dsl::stack_bound);
REGISTER_STACK_BOUND("glm_dsa_attention", GlmDsaAttention, ::dsl::stack_bound);
REGISTER_STACK_BOUND("glm_dsa_attention_backward", GlmDsaAttentionBackward, ::dsl::stack_bound);
