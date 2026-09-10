// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#include "runtime/executor/compiled_ops.h"

#include "kernels/glm5.h"
#include "kernels/decode.h"
#include "kernels/kernels.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/buffer_plan.h"
#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/executor/op_registry.h"
#include "runtime/lora/lora_slice_dispatch.h"

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
        const long P = op.inputs.at(6).shape[0];
        return align_stack_bytes(
            GlmDsaKernels::indexer_workspace_bytes(s[0], s[1], s[1], s[3], P, op.outputs[0].shape[2] / P));
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
        const bool decoding = mExecutionRequest && mExecutionRequest->decoding();
        const int rows = decoding ? inputs[0].Sizes[0] : 1;
        for (int row = 0; row < rows; ++row) {
            auto sliced = inputs;
            auto target = output;
            auto* state = cache;
            if (decoding) {
                state = mExecutionRequest->decode_state(row);
                for (int i : {0, 1, 2, 3, 7})
                    sliced[i] = decode_batch_row(inputs[i], row);
                target = decode_batch_row(output, row);
            }
            const auto& q = sliced[0];
            auto scratch = mRunState.temp_alloc(
                ETensorDType::BYTE,
                {static_cast<long>(mDsaKernels.workspace_bytes(q.Sizes[0], q.Sizes[1], state ? state->length : 0))},
                "dsa_workspace");
            mDsaKernels.indexer(sliced, target, scratch, stream, state, layer);
            mRunState.Stack.free(scratch);
        }
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
    auto run_attention = [&](const std::vector<Tensor>& inputs,
                             const Tensor& qkv,
                             const Tensor& indices,
                             const Tensor& out,
                             const Tensor& lse,
                             GlmDecodeState* cache) {
        if (cache && inputs.size() == 4) {
            if (qkv.Sizes[0] != 1) throw std::runtime_error("Latent MLA decode requires one request");
            const auto& latent = inputs[2];
            const auto& weight = inputs[3];
            const long rank = weight.Sizes[1];
            auto& history = cache->pages(layer, "mla_latent", ETensorDType::BF16, rank);
            history.append(latent.Data, cache->length, T, stream);
            if (cache->length == 0) {
                // Prefill already has projected K/V for every new token. Retain
                // only latents for subsequent decode calls.
                mDsaKernels.attention(qkv, indices, out, lse, stream);
            } else {
                const CompiledOp* projection = nullptr;
                for (const auto& candidate : mCurrentGraph->ops) {
                    if (candidate.type == CompiledOpType::Matmul && candidate.inputs.size() == 2 &&
                        candidate.inputs[1].tensor_id == op.inputs[3].tensor_id) {
                        projection = &candidate;
                        break;
                    }
                }
                if (!projection) throw std::runtime_error("MLA cache cannot find its latent projection");
                const int slots = mDsaKernels.selection_slots();
                const long allocated_slots = std::max(slots, 1);
                auto selected_latent =
                    mRunState.temp_alloc(ETensorDType::BF16, {allocated_slots, rank}, "mla_selected_latent");
                auto selected_kv =
                    mRunState.temp_alloc(ETensorDType::BF16, {allocated_slots, 2 * H * D}, "mla_selected_kv");
                const long chunk = mLoRARunState ? std::min(128, mLoRARunState->B * mLoRARunState->T) : 128;
                auto projected = mRunState.temp_alloc(ETensorDType::BF16, {chunk, 2 * H * D}, "mla_projection_tile");
                auto slot_indices = mRunState.temp_alloc(ETensorDType::INT32, {allocated_slots}, "mla_slot_indices");
                for (long token = 0; token < T; ++token) {
                    Tensor selected = indices;
                    selected.Data += token * indices.Sizes[2] * 4;
                    selected.Sizes[1] = 1;
                    if (slots)
                        decode_gather_pages(history.table(),
                                            selected,
                                            selected_latent,
                                            slot_indices,
                                            slots,
                                            rank,
                                            stream);
                    for (long start = 0; start < slots; start += chunk) {
                        const long rows = std::min(chunk, slots - start);
                        Tensor x = selected_latent, y = projected;
                        x.Data += start * rank * 2;
                        x.Sizes[0] = y.Sizes[0] = rows;
                        // Reconstruct the original BF16 projection, then its LoRA
                        // contribution. Absorbing weights would change rounding.
                        matmul(y,
                               weight,
                               x,
                               std::nullopt,
                               nullptr,
                               nullptr,
                               mRunState.CublasLtHandle,
                               mRunState.CuBlasWorkspace,
                               2 * H * D,
                               rows,
                               rank,
                               EMMTranspose::TN,
                               false,
                               stream);
                        modules::detail::apply_lora_slices_forward(projection->attrs.lora_slices,
                                                                   layer,
                                                                   x,
                                                                   y,
                                                                   rows,
                                                                   mLoRAWeights,
                                                                   mLoRAConfig,
                                                                   mLoRARunState,
                                                                   mRunState.CublasLtHandle,
                                                                   mRunState.CuBlasWorkspace,
                                                                   stream);
                        Tensor destination = selected_kv;
                        destination.Data += start * 2 * H * D * 2;
                        destination.Sizes[0] = rows;
                        mDsaKernels.repack_kv(y, destination, stream);
                    }
                    Tensor query = qkv, output = out, logsumexp = lse;
                    query.Data += token * 3 * H * D * 2;
                    output.Data += token * H * D * 2;
                    logsumexp.Data += token * H * 4;
                    mDsaKernels.attention_selected(query, slot_indices, selected_kv, output, logsumexp, slots, stream);
                }
                mRunState.Stack.free(slot_indices);
                mRunState.Stack.free(projected);
                mRunState.Stack.free(selected_kv);
                mRunState.Stack.free(selected_latent);
            }
        } else {
            mDsaKernels.attention(qkv, indices, out, lse, stream, cache, layer);
        }
    };
    if (mExecutionRequest && mExecutionRequest->decoding()) {
        if (backward) throw std::runtime_error("Decode cannot execute backward");
        for (int row = 0; row < B; ++row) {
            auto sliced = inputs;
            sliced[0] = decode_batch_row(qkv, row);
            sliced[1] = decode_batch_row(indices, row);
            if (inputs.size() == 4) {
                sliced[2].Data += row * T * inputs[2].Sizes[1] * get_dtype_size(inputs[2].DType);
                sliced[2].Sizes[0] = T;
            }
            run_attention(sliced,
                          sliced[0],
                          sliced[1],
                          decode_batch_row(out, row),
                          decode_batch_row(lse, row),
                          mExecutionRequest->decode_state(row));
        }
    } else
        run_attention(inputs, qkv, indices, out, lse, nullptr);
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
