// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Causal language-model runtime profile.

#include "runtime/executor/causal_lm_execution_profile.h"

#include "runtime/core/model_config.h"
#include "runtime/dsl/dsl_run_state.h"
#include "runtime/executor/graph_executor.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/training/runtime_options.h"
#include "utilities/dtype.h"

#include <algorithm>
#include <string>

namespace dsl {
namespace {

void add_binding_if_present(ExecutionRequest& request, const std::string& name, const Tensor& tensor) {
    for (const auto& binding : request.bindings) {
        if (binding.name == name) return;
    }
    for (const auto& copy : request.input_copies) {
        if (copy.name == name) return;
    }
    if (tensor.Data) {
        request.bindings.push_back(RuntimeBinding{name, tensor});
    }
}

void add_copy_if_present(ExecutionRequest& request,
                         const std::string& name,
                         const Tensor& source,
                         const Tensor& destination) {
    for (const auto& binding : request.bindings) {
        if (binding.name == name) return;
    }
    for (const auto& copy : request.input_copies) {
        if (copy.name == name) return;
    }
    if (!source.Data || !destination.Data) return;

    const auto kind = (source.Device == -1) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    const auto transform =
        (name == "position_ids" && destination.Rank == 3 && destination.Sizes[0] == 3 &&
         source.bytes() <= static_cast<std::size_t>(destination.Sizes[1]) *
                               static_cast<std::size_t>(destination.Sizes[2]) * get_dtype_size(destination.DType))
            ? RuntimeCopyTransform::ReplicateSinglePlaneToThree
            : RuntimeCopyTransform::None;
    request.input_copies.push_back(RuntimeCopy{name, source, destination, kind, transform});
}

void add_causal_lm_bindings(ExecutionRequest& request,
                            DslRunState& rs,
                            const modules::ModelConfig& config,
                            const RuntimeOptions& options,
                            bool backward) {
    add_binding_if_present(request, "token_ids", rs.Inputs);
    add_binding_if_present(request, "position_ids", rs.PositionIDs);
    add_binding_if_present(request, "visual_pos_masks", rs.VisualPosMasks);
    add_binding_if_present(request, "visual_embeds", rs.VisualEmbeds);
    for (std::size_t i = 0; i < rs.DeepstackVisualEmbeds.size(); ++i) {
        add_binding_if_present(request, "deepstack_visual_embeds_" + std::to_string(i), rs.DeepstackVisualEmbeds[i]);
    }
    add_binding_if_present(request, "x0", rs.non_block_activations().encoded);
    add_binding_if_present(request, "encoded", rs.non_block_activations().encoded);
    add_binding_if_present(request, "ln_final", rs.non_block_activations().ln_final);
    add_binding_if_present(request, "xF", rs.non_block_activations().ln_final);
    add_binding_if_present(request, "ln_final_rstd", rs.non_block_activations().ln_final_rstd);
    add_binding_if_present(request, "targets", rs.Targets);
    add_binding_if_present(request, "loss", rs.Losses);
    add_binding_if_present(request, "losses", rs.Losses);
    add_binding_if_present(request, "d_loss", rs.scratch().cross_entropy_dloss);

    if (backward && options.LMHeadChunks <= 1 && rs.non_block_activations().output.Data) {
        Tensor logits_view =
            view_tensor(rs.non_block_activations().output, {request.batch, request.sequence, config.VocabSize});
        Tensor logits_flat =
            view_tensor(rs.non_block_activations().output, {request.batch * request.sequence, config.VocabSize});
        request.bindings.push_back(RuntimeBinding{"d_logits", logits_view});
        request.bindings.push_back(RuntimeBinding{"d_logits_flat", logits_flat});
    }
}

void add_forward_copies(ExecutionRequest& request, DslRunState& rs, Tensor inputs, Tensor position_ids) {
    add_copy_if_present(request, "token_ids", inputs, rs.Inputs);
    add_copy_if_present(request, "position_ids", position_ids, rs.PositionIDs);
    add_copy_if_present(request, "visual_pos_masks", rs.VisualPosMasks_CPU, rs.VisualPosMasks);
    add_copy_if_present(request, "visual_embeds", rs.VisualEmbeds_CPU, rs.VisualEmbeds);
    for (std::size_t i = 0; i < rs.DeepstackVisualEmbeds.size() && i < rs.DeepstackVisualEmbeds_CPU.size(); ++i) {
        add_copy_if_present(request,
                            "deepstack_visual_embeds_" + std::to_string(i),
                            rs.DeepstackVisualEmbeds_CPU[i],
                            rs.DeepstackVisualEmbeds[i]);
    }
}

void add_forward_weight_groups(ExecutionRequest& request) {
    request.gather_before_forward_weight_groups.push_back("embeddings");
    request.gather_before_forward_weight_groups.push_back("final_norm");
    request.release_after_forward_weight_groups.push_back("embeddings");
    request.release_after_forward_weight_groups.push_back("final_norm");
    request.fp8_forward_cache_weight_groups.push_back("lm_head");
}

void add_backward_weight_groups(ExecutionRequest& request, const RuntimeOptions& options) {
    request.gather_before_backward_weight_groups.push_back("final_norm");
    request.release_after_backward_weight_groups.push_back("final_norm");
    if (options.LMHeadChunks <= 1) {
        request.gather_before_backward_weight_groups.push_back("lm_head");
        request.release_after_backward_weight_groups.push_back("lm_head");
        request.fp8_backward_cache_weight_groups.push_back("lm_head");
    }
}

}  // namespace

std::optional<CausalLMDocMaskingInfo>
CausalLMExecutionProfile::compute_doc_masking(const std::int32_t* position_ids, int B, int T, bool mrope) const {
    if (!position_ids) return std::nullopt;

    std::vector<std::int32_t> cu_seqlens;
    cu_seqlens.push_back(0);
    int max_seqlen = 0;
    bool has_boundaries = false;

    for (int b = 0; b < B; ++b) {
        int doc_start = b * T;
        for (int t = 1; t < T; ++t) {
            const int idx = b * T + t;
            const int prev = position_ids[idx - 1];
            const int curr = position_ids[idx];
            const bool is_boundary = mrope ? (curr < prev) : (curr - prev != 1);
            if (is_boundary) {
                const int doc_len = (b * T + t) - doc_start;
                if (doc_len > 0) {
                    cu_seqlens.push_back(cu_seqlens.back() + doc_len);
                    max_seqlen = std::max(max_seqlen, doc_len);
                }
                doc_start = b * T + t;
                has_boundaries = true;
            }
        }

        const int last_len = (b + 1) * T - doc_start;
        if (last_len > 0) {
            cu_seqlens.push_back(cu_seqlens.back() + last_len);
            max_seqlen = std::max(max_seqlen, last_len);
        }
    }

    if (!has_boundaries) return std::nullopt;

    const int num_docs = static_cast<int>(cu_seqlens.size()) - 1;
    const int total_q = cu_seqlens.back();
    return CausalLMDocMaskingInfo{std::move(cu_seqlens), num_docs, max_seqlen, total_q};
}

bool CausalLMExecutionProfile::apply_doc_masking(IGraphExecutor& executor,
                                                 const RuntimeOptions& options,
                                                 const modules::ModelConfig& config,
                                                 Tensor inputs,
                                                 Tensor position_ids,
                                                 int micro_step) const {
    if (!options.DocMasking) return false;
    if (!position_ids.Data || position_ids.Device != -1) {
        executor.clear_doc_masking();
        return false;
    }

    const auto* pos_ptr = reinterpret_cast<const std::int32_t*>(position_ids.Data);
    const int B = static_cast<int>(inputs.Sizes[0]);
    const int T = static_cast<int>(inputs.Sizes[1]);
    auto doc_info = compute_doc_masking(pos_ptr, B, T, config.Rope.is_multimodal());
    if (!doc_info) {
        executor.clear_doc_masking();
        return false;
    }

    executor.set_doc_masking(doc_info->cu_seqlens.data(),
                             doc_info->num_docs,
                             doc_info->max_seqlen,
                             doc_info->total_q,
                             micro_step);
    return true;
}

ExecutionRequest CausalLMExecutionProfile::make_forward_request(DslRunState& rs,
                                                                const modules::ModelConfig& config,
                                                                const RuntimeOptions& options,
                                                                Tensor inputs,
                                                                Tensor position_ids,
                                                                int micro_step) const {
    ExecutionRequest request;
    request.batch = inputs.Sizes[0];
    request.sequence = inputs.Sizes[1];
    request.mode = ExecutionMode::Forward;
    request.micro_step = micro_step;
    request.initialize_loss_buffers = micro_step == 0;
    add_forward_weight_groups(request);
    add_forward_copies(request, rs, inputs, position_ids);
    add_copy_if_present(request, "targets", rs.Targets_CPU, rs.Targets);
    add_causal_lm_bindings(request, rs, config, options, /*backward=*/false);
    return request;
}

ExecutionRequest CausalLMExecutionProfile::make_eval_request(DslRunState& rs,
                                                             const modules::ModelConfig& config,
                                                             const RuntimeOptions& options,
                                                             Tensor inputs,
                                                             Tensor position_ids,
                                                             Tensor targets,
                                                             int micro_step) const {
    ExecutionRequest request;
    request.batch = inputs.Sizes[0];
    request.sequence = inputs.Sizes[1];
    request.mode = ExecutionMode::Eval;
    request.micro_step = micro_step;
    request.initialize_loss_buffers = true;
    request.reduce_loss_on_completion = true;
    add_forward_weight_groups(request);
    add_forward_copies(request, rs, inputs, position_ids);
    add_copy_if_present(request, "targets", targets, rs.Targets);
    add_causal_lm_bindings(request, rs, config, options, /*backward=*/false);
    return request;
}

ExecutionRequest CausalLMExecutionProfile::make_backward_request(DslRunState& rs,
                                                                 const modules::ModelConfig& config,
                                                                 const RuntimeOptions& options,
                                                                 Tensor inputs,
                                                                 Tensor targets,
                                                                 int grad_accum_steps,
                                                                 int micro_step) const {
    ExecutionRequest request;
    request.batch = inputs.Sizes[0];
    request.sequence = inputs.Sizes[1];
    request.mode = ExecutionMode::Backward;
    request.micro_step = micro_step;
    request.grad_accum_steps = grad_accum_steps;
    request.reduce_loss_on_completion = micro_step == grad_accum_steps - 1;
    request.last_inputs_cpu = inputs;
    request.has_last_inputs_cpu = true;
    add_backward_weight_groups(request, options);
    add_copy_if_present(request, "targets", targets, rs.Targets);
    if (options.LMHeadChunks > 1) {
        request.skipped_backward_tensors.push_back("d_logits");
        request.skipped_backward_tensors.push_back("d_logits_flat");
    }
    add_causal_lm_bindings(request, rs, config, options, /*backward=*/true);
    return request;
}

std::vector<std::string> CausalLMExecutionProfile::backward_save_exclusions() const {
    return {"logits", "logits_flat"};
}

}  // namespace dsl
