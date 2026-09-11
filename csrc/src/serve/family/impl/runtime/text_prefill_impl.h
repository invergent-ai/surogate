#include "family/impl/runtime/instance.h"
#include "family/impl/runtime/schedule.h"

#include "api/ops/linear.h"
#include "api/ops/sampling.h"
#include "api/ops/scalar.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

#include <cstdio>
#include <cstdlib>

#include "api/ops/lora_store.h"
#include "api/ops/sampled_logprob.h"

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule {
namespace {

// Every prefill publishes its policy, including base requests and vision encoding.
struct LoraPrefillScope {
    bool held = false;
    ops::LoraRound previous = ops::lora_current_round();
    LoraPrefillScope(std::int32_t slot, std::int32_t columns, cudaStream_t stream) {
        if (!ops::lora_active()) { return; }
        ops::LoraRound round;
        round.uniform = true;
        round.scratch = ops::lora_store_for_current_device().scratch(columns);
        if (round.scratch.data == nullptr) { throw std::logic_error("adapter prefill exceeds configured scratch capacity"); }
        ops::lora_store_for_current_device().write_uniform_slot(slot, stream);
        round.uniform_cell = ops::lora_store_for_current_device().uniform_cell();
        ops::lora_set_round(round);
        held = true;
    }
    ~LoraPrefillScope() {
        if (held) { ops::lora_set_round(previous); }
    }
};


DFlashFeatureSink make_dflash_prefill_sink(PrefillContext& state) {
    if (!state.execution.io.dflash_decode || state.dflash_host_ingress == nullptr) {
        throw std::logic_error("DFlash prefill controls are unavailable");
    }
    return dflash_feature_sink(
        state, [&state](const Tensor& features, const Tensor& positions, bool rewrite_checkpoint) {
            const auto& stage = state.execution.stage;
            if (stage.features) {
                const auto offset = stage.residual_bytes + static_cast<std::size_t>(features.ne[0]) * sizeof(std::uint16_t);
                *reinterpret_cast<std::int32_t*>(static_cast<std::byte*>(stage.export_pinned) + offset) =
                    rewrite_checkpoint ? 1 : 0;
                if (stage.last < state.execution.model.geometry.layers) { return; }
            }
            auto& frame  = *state.execution.io.dflash_decode;
            Tensor count = frame.append_counts.slice(0, 0, 1);
            Tensor lane  = frame.lanes.slice(0, 0, 1);
            Tensor row   = frame.dflash_kv_table_rows.slice(0, 0, 1);
            ops::set_i32_scalar(count, features.ne[1], state.execution.device.stream);
            const auto exact = static_cast<std::uint32_t>(features.ne[1]);
            dflash_append_context(state, features, positions, count, lane, row, {exact, exact});
            if (rewrite_checkpoint) {
                state.dflash->save_rewrite_checkpoint(state.dflash_host_ingress->lanes[0],
                                                      state.execution.device.stream);
            }
        });
}

} // namespace

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t current_state_slot,
                         std::int32_t rewrite_checkpoint_state_slot,
                         std::uint32_t mtp_proposal_extent) {
    card.set_sampling(sampling);
    card.set_linear_state_slots(current_state_slot, rewrite_checkpoint_state_slot);
    card.set_ple_state(execution.ple);
    card.set_stage(execution.stage);
    card.set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
    card.set_mtp_proposal_extent(mtp_proposal_extent);
    if (execution.proposal_head == ProposalHead::Full) {
        card.set_proposal_head(nullptr, nullptr, 0);
        return;
    }
    if (card.proposal_head() == nullptr || card.proposal_head_ids() == nullptr ||
        card.proposal_head_n() <= 0) {
        throw std::runtime_error("optimized proposal head is unavailable");
    }
}

PrefillChunkResult prefill_text_chunk(
    PrefillContext& state, std::span<const TokenId> ids, std::uint32_t nominal_length,
    std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier, bool finalize_at_end) {
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);
    card.logprob_observer = state.logprob_observer;
    card.score_prompt = state.score_prompt;
    card.score_prompt_start = state.score_prompt_start;
    card.score_prompt_end = state.score_prompt_end;
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_rewrite_checkpoint_frontier(
        rewrite_checkpoint_capture_frontier
            ? static_cast<std::int64_t>(*rewrite_checkpoint_capture_frontier)
            : -1);
    card.set_prefill_graph_family(state.prefill_graphs);
    const std::span<const int> prompt(ids.data(), ids.size());
    LoraPrefillScope lora_scope(state.lora_slot, static_cast<std::int32_t>(nominal_length),
                                 state.execution.device.stream);
    if (state.dflash != nullptr) {
        DFlashFeatureSink sink = make_dflash_prefill_sink(state);
        return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end,
                                  sink);
    }
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end);
}

PrefillChunkResult
prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                         VisionPrefillSession& vision, std::uint32_t nominal_length,
                         std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier,
                         bool finalize_at_end) {
    if (state.dflash != nullptr) {
        throw std::logic_error("DFlash staged multimodal prefill is unavailable");
    }
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);
    card.logprob_observer = state.logprob_observer;
    card.score_prompt = state.score_prompt;
    card.score_prompt_start = state.score_prompt_start;
    card.score_prompt_end = state.score_prompt_end;
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_rewrite_checkpoint_frontier(
        rewrite_checkpoint_capture_frontier
            ? static_cast<std::int64_t>(*rewrite_checkpoint_capture_frontier)
            : -1);
    LoraPrefillScope lora_scope(state.lora_slot, static_cast<std::int32_t>(nominal_length),
                                 state.execution.device.stream);
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, vision, finalize_at_end);
}

void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge) {
    if (!state.mtp_kv.valid() || bridge.previous_hidden == nullptr || state.text_kv_base == 0 ||
        bridge.position < 0 ||
        static_cast<std::uint32_t>(bridge.position) + 1 != state.text_kv_base) {
        throw std::logic_error("multimodal MTP bridge does not match the reusable frontier");
    }

    LoraPrefillScope lora_scope(state.lora_slot, 1, state.execution.device.stream);
    Tensor bridge_token = state.execution.io.mtp->target_input_ids.slice(0, 0, 1);
    const TokenId token = prompt.token_ids[state.text_kv_base];
    CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token), cudaMemcpyHostToDevice,
                               state.execution.device.stream));

    Tensor visual_embedding;
    const Tensor* composed_embedding = nullptr;
    if (prompt.token_types[state.text_kv_base] != 0) {
        const VisionChunk chunk = vision.prepare_chunk(state.text_kv_base, 1);
        if (chunk.control == nullptr) {
            throw std::logic_error("visual MTP bridge has no encoded Vision item");
        }
        const auto& scatter = chunk.control->scatter_indices;
        const auto column   = std::lower_bound(scatter.begin(), scatter.end(),
                                               static_cast<std::int32_t>(state.text_kv_base));
        if (column == scatter.end() || *column != static_cast<std::int32_t>(state.text_kv_base) ||
            static_cast<std::uint8_t>(chunk.control->modality) !=
                prompt.token_types[state.text_kv_base]) {
            throw std::logic_error("visual MTP bridge does not match Vision scatter metadata");
        }
        visual_embedding =
            chunk.embeddings.slice(1, static_cast<std::int32_t>(column - scatter.begin()), 1);
        composed_embedding = &visual_embedding;
    }

    mtp_bridge_and_propose(state, bridge_token, *bridge.previous_hidden, bridge.position,
                           bridge.rope_position, false, composed_embedding);
}

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose) {
    state.execution.work.reset();
    Tensor logits = state.execution.io.logits.slice(1, 0, 1);
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    LoraPrefillScope lora_scope(state.lora_slot, 1, state.execution.device.stream);
    card.logits_from_hidden(hidden, logits);
    CUDA_CHECK(cudaMemcpyAsync(state.execution.io.pos.data, &absolute_position,
                               sizeof(absolute_position), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    ops::sample(logits, state.execution.io.token,
                state.execution.model.geometry.token_domain, state.sampling,
                state.execution.io.pos, purpose, state.execution.work,
                state.execution.device.stream);
    // The same quantity the decode rounds record, for the one token a prefill
    // licenses -- a client that asks for probabilities wants the first one too.
    ops::sampled_logprob(logits, state.execution.io.token, state.execution.io.logprob,
                         state.execution.model.geometry.token_domain, state.sampling,
                         state.execution.device.stream);
    if (state.logprob_observer) state.logprob_observer(logits, absolute_position, true);
    state.execution.work.reset();
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule
