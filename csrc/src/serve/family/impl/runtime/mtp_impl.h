#include "family/impl/runtime/instance.h"
#include "family/impl/runtime/schedule.h"
#include "api/ops/sampled_logprob.h"

#include "api/ops/mtp_round.h"
#include "api/ops/scatter.h"
#include "api/ops/lora_store.h"
#include "api/ops/scalar.h"
#include "api/ops/position.h"
#include "api/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule {
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding) {
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    // The MTP head's envelopes are deliberately unwindowed: no target with a
    // draft head declares a sliding window, and the head is not a stack layer,
    // so layer_sliding_window() has no layer index to key on for it. A windowed
    // model that grows an MTP head must decide its window here, on purpose.
    const ops::GqaExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    if (!build_proposal) { return; }

    if (state.mtp_proposal_extent == 0 ||
        state.mtp_proposal_extent >
            static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.execution.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::GqaExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.execution.io.mtp->ar_hidden, ar_position,
                                 envelope, next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.execution.io.mtp->ar_hidden.data, next_hidden.data,
                                   state.execution.io.mtp->ar_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, state.execution.device.stream));
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
}

auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                           MtpGqaEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumBatchColumns) ||
            k == 0 || k > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        family::MtpDecodeState& frame = state.frame;
        const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(family::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, &state.mtp_cache);
        card.set_ple_state(state.execution.ple);
        if (std::getenv("SUROGATE_SERVE_TRACE_STAGE") != nullptr) {
            std::fprintf(stderr, "stage-trace: an MTP round carries stage [%d, %d)\n",
                         state.execution.stage.first, state.execution.stage.last);
        }
        card.set_stage(state.execution.stage);
        Tensor anchors           = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers         = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor budgets           = frame.remaining_budgets.slice(0, 0, batch_size);
        Tensor current_extents   = frame.current_extents.slice(0, 0, batch_size);
        Tensor target_valid      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor current_drafts    = frame.current_drafts.slice(1, 0, batch_size);
        Tensor target_rope       = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor text_rows         = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor mtp_rows          = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes             = frame.lanes.slice(0, 0, batch_size);
        Tensor rope_deltas       = frame.rope_deltas.slice(0, 0, batch_size);
        Tensor verify_ids        = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions  = frame.target_positions.slice(1, 0, batch_size);
        Tensor target_tokens     = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits     = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden     = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden   = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens   = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts   = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted          = frame.accepted_drafts.slice(0, 0, batch_size);
        Tensor next_extents      = frame.next_extents.slice(0, 0, batch_size);
        Tensor alignment_ids     = frame.alignment_ids.slice(1, 0, batch_size);
        Tensor alignment_hidden  = frame.alignment_hidden.slice(2, 0, batch_size);
        Tensor ar_hidden         = frame.ar_hidden.slice(1, 0, batch_size);
        Tensor next_hidden       = frame.next_hidden.slice(1, 0, batch_size);
        Tensor ar_positions      = frame.ar_positions.slice(0, 0, batch_size);
        Tensor ar_rope_positions = frame.ar_rope_positions.slice(0, 0, batch_size);
        Tensor ar_valid_columns  = frame.ar_valid_columns.slice(0, 0, batch_size);
        Tensor next_drafts       = frame.next_drafts.slice(0, 0, batch_size);

        ops::speculative_prepare_verify_inputs(anchors, current_drafts, frontiers, current_extents,
                                               verify_ids, target_positions,
                                               state.execution.device.stream);
        Tensor adapter_columns = ops::speculative_lora_columns(frame.lora_slots, frame.lora_columns,
            verify_ids.ne[0], batch_size, state.execution.device.stream);
        if (!card.stage_finishes()) {
            // A pipeline stage without the head: the verify forward over the draft columns
            // for its own layers, the recurrent state recorded for the fold, the residual
            // exported for the next stage -- and nothing more. Accept, select, align and
            // propose are the head stage's; what it decides reaches this stage as host
            // integers (adopt_speculative_outcome, then resolve_pending_batch).
            if (state.execution.replay_records == nullptr) {
                throw std::logic_error("speculative target verify has no ReplaySSM record storage");
            }
            card.set_gdn_state_action(GdnStateAction::RecordForReplay,
                                      state.execution.replay_records);
            ops::ScopedLoraColumns adapter(adapter_columns);
            card.target_verify_batch(verify_ids, target_positions, target_rope, target_valid,
                                     text_rows, lanes, envelopes.target_verify, target_hidden,
                                     target_logits, target_tokens);
            return;
        }
        target_verify_accept(state.execution, state.continuation_hidden_store, card,
                             TargetVerifyFrameView{
                                 .ids             = verify_ids,
                                 .cache_positions = target_positions,
                                 .rope_positions  = target_rope,
                                 .valid_columns   = target_valid,
                                 .kv_table_rows   = text_rows,
                                 .lanes           = lanes,
                                 .target_hidden   = target_hidden,
                                 .target_logits   = target_logits,
                                 .target_tokens   = target_tokens,
                                 .drafts          = current_drafts,
                                 .current_extents = current_extents,
                                 .frontiers       = frontiers,
                                 .anchors         = anchors,
                                 .licensed_tokens = licensed_tokens,
                                 .licensed_counts = licensed_counts,
                                 .accepted_drafts = accepted,
                                 .selected_hidden = selected_hidden,
                                 .replay_records  = state.execution.replay_records,
                                 .sampling        = frame.sampling,
                                 .lora_columns = adapter_columns,
                             },
                             envelopes.target_verify);

        ops::mtp_prepare_next_round(verify_ids, anchors, accepted, frontiers, budgets,
                                    licensed_counts, rope_deltas, alignment_ids, next_extents,
                                    ar_positions, ar_rope_positions, ar_valid_columns,
                                    static_cast<std::int32_t>(state.text_cache.max_context()),
                                    state.execution.device.stream);
        card.mtp_forward_decode_batch(alignment_ids, target_hidden, target_positions, target_rope,
                                      licensed_counts, mtp_rows, envelopes.batch, alignment_hidden);
        ops::speculative_select_accepted_hidden(alignment_hidden, accepted, ar_hidden,
                                                state.execution.device.stream);

        Tensor proposal_logits = frame.proposal_logits.slice(1, 0, batch_size);
        Tensor draft0          = next_drafts.slice(1, 0, 1).view({batch_size});
        card.mtp_propose_batch(ar_hidden, proposal_logits, draft0);
        for (std::uint32_t step = 0; step + 1 < k; ++step) {
            Tensor previous =
                next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
            Tensor next =
                next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1).view({batch_size});
            Tensor position =
                ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view({1, batch_size});
            Tensor rope = ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                              .view({1, batch_size});
            Tensor valid =
                ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
            Tensor previous_batch    = previous.view({1, batch_size});
            // The head's hidden is as wide as the buffers were sized: the model width for a
            // fixed-tail head, the wide residual for a trunk-block one.
            const std::int32_t hidden_width = ar_hidden.ne[0];
            Tensor hidden_batch      = ar_hidden.view({hidden_width, 1, batch_size});
            Tensor next_hidden_batch = next_hidden.view({hidden_width, 1, batch_size});
            card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                          mtp_rows, envelopes.ar[step], next_hidden_batch);
            card.mtp_propose_batch(next_hidden, proposal_logits, next);
            CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, next_hidden.data, ar_hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, state.execution.device.stream));
        }

        auto* scores = reinterpret_cast<RawTokenScores*>(static_cast<std::byte*>(frame.egress.data) +
            offsetof(family::MtpDecodeEgress, scores));
        ops::score_logprobs_device(target_logits, licensed_tokens, state.execution.model.geometry.token_domain,
            frame.sampling, scores, state.execution.device.stream, k + 1, static_cast<const int*>(licensed_counts.data));
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   offsetof(family::MtpDecodeEgress, scores) + batch_size * (k + 1) * sizeof(RawTokenScores), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

// The narrow round: what the head runs when the batch is too wide to pay for a verify. The
// trunk sees one column per lane -- the anchor at the frontier, exactly what an ordinary round
// sees -- sampled under the lane's config; the head is aligned on that one column, so its
// cache stays current, and proposes nothing. A round that is wide again verifies one column
// and proposes, as the first round after a prefill does. The wide frame's buffers are read as
// width-one views over their first B entries (the frame is one contiguous buffer per tensor,
// so the prefix is contiguous); the host fills the ingress with stride one and reads the
// egress the same way.
auto mtp_narrow_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                           MtpGqaEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumBatchColumns) ||
            k == 0 || k > kMtpDecodeMaximumDrafts || state.one.data == nullptr) {
            throw std::logic_error("MTP narrow batch state is incomplete");
        }
        family::MtpDecodeState& frame = state.frame;
        cudaStream_t stream            = state.execution.device.stream;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(family::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   stream));

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, &state.mtp_cache);
        card.set_ple_state(state.execution.ple);
        if (std::getenv("SUROGATE_SERVE_TRACE_STAGE") != nullptr) {
            std::fprintf(stderr, "stage-trace: a narrow MTP round carries stage [%d, %d)\n",
                         state.execution.stage.first, state.execution.stage.last);
        }
        card.set_stage(state.execution.stage);

        const auto over = [](const Tensor& wide, DType dtype,
                             std::initializer_list<std::int32_t> shape) {
            return Tensor(wide.data, dtype, shape);
        };
        const std::int32_t hidden_width  = frame.target_hidden.ne[0];
        const std::int32_t vocab         = frame.target_logits.ne[0];
        Tensor anchors           = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers         = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor target_valid      = frame.target_valid_columns.slice(0, 0, batch_size); // one
        Tensor text_rows         = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor mtp_rows          = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes             = frame.lanes.slice(0, 0, batch_size);
        Tensor verify_ids        = over(frame.verify_ids, DType::I32, {1, batch_size});
        Tensor target_positions  = over(frame.target_positions, DType::I32, {1, batch_size});
        Tensor target_rope       = over(frame.target_rope_positions, DType::I32, {1, batch_size});
        Tensor target_tokens     = over(frame.target_argmax, DType::I32, {1, batch_size});
        Tensor target_logits     = over(frame.target_logits, DType::BF16, {vocab, 1, batch_size});
        Tensor target_hidden     = over(frame.target_hidden, DType::BF16, {hidden_width, 1, batch_size});
        Tensor alignment_hidden  = over(frame.alignment_hidden, DType::BF16, {frame.alignment_hidden.ne[0], 1, batch_size});
        Tensor licensed_tokens   = over(frame.licensed_tokens, DType::I32, {batch_size});
        const std::size_t column_ids = static_cast<std::size_t>(batch_size) * sizeof(std::int32_t);
        const char* narrow_step = "prologue";
        try {

        // The one verify column: the anchor, at the frontier.
        CUDA_CHECK(cudaMemcpyAsync(verify_ids.data, anchors.data, column_ids,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(target_positions.data, frontiers.data, column_ids,
                                   cudaMemcpyDeviceToDevice, stream));
        // One column, nothing to accept or reject: the recurrent state updates in place, as
        // an ordinary round's does, and the resolve folds nothing for this round (a pending
        // candidate marked in-place commits zero columns to the fold).
        narrow_step = "verify";
        card.set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
        {
            ops::ScopedLoraColumns adapter(frame.lora_slots.slice(0, 0, batch_size));
            card.target_verify_batch(verify_ids, target_positions, target_rope, target_valid, text_rows,
                                     lanes, envelopes.target_verify, target_hidden, target_logits,
                                     target_tokens);
        }
        if (!card.stage_finishes()) { return; } // a stage without the head: exported, done

        // The token, sampled the way the ordinary round samples; it is the round's whole licence.
        narrow_step = "sample";
        Tensor logits_flat    = over(frame.target_logits, DType::BF16, {vocab, batch_size});
        Tensor positions_flat = over(frame.target_positions, DType::I32, {batch_size});
        ops::sample(logits_flat, licensed_tokens, state.execution.model.geometry.token_domain,
                    frame.sampling, positions_flat, ops::kSamplePurposeDecode, state.execution.work,
                    stream);
        narrow_step = "scatter";
        Tensor hidden_flat = over(frame.target_hidden, DType::BF16, {hidden_width, batch_size});
        ops::scatter(hidden_flat, lanes, state.continuation_hidden_store, stream);
        // The head, aligned on the one column so its cache stays current for the round that
        // is narrow no longer: the token pairs with the hidden it was sampled from, at that
        // position, exactly as the wide round pairs accepted tokens with their columns. It
        // proposes nothing -- three proposal steps for a token the next round will not verify
        // were most of what a narrow round cost -- so the first wide round after a narrow
        // stretch verifies one column and proposes, as after a prefill.
        narrow_step = "align";
        CUDA_CHECK(cudaMemcpyAsync(anchors.data, licensed_tokens.data, column_ids,
                                   cudaMemcpyDeviceToDevice, stream));
        Tensor alignment_ids = over(frame.anchors, DType::I32, {1, batch_size});
        card.mtp_forward_decode_batch(alignment_ids, target_hidden, target_positions, target_rope,
                                      target_valid, mtp_rows, envelopes.batch, alignment_hidden);

        narrow_step = "egress";
        auto* scores = reinterpret_cast<RawTokenScores*>(static_cast<std::byte*>(frame.egress.data) +
            offsetof(family::MtpDecodeEgress, scores));
        ops::score_logprobs_device(logits_flat, licensed_tokens, state.execution.model.geometry.token_domain,
            frame.sampling, scores, stream);
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   offsetof(family::MtpDecodeEgress, scores) + batch_size * sizeof(RawTokenScores), cudaMemcpyDeviceToHost,
                                   stream));
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string("narrow MTP round, step ") + narrow_step + ": " +
                                     error.what());
        }
    };
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpGqaEnvelopes envelopes, DecodeGraphDefinition& definition,
                              bool narrow) {
    if (narrow) {
        auto body = mtp_narrow_batch_body(state, batch_size, k, envelopes);
        capture_graph(state, definition, body);
        return;
    }
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    capture_graph(state, definition, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpGqaEnvelopes envelopes, DecodeGraphExecutable* executable, bool narrow) {
    if (narrow) {
        auto body = mtp_narrow_batch_body(state, batch_size, k, envelopes);
        run_prepared(state, executable, body);
        return;
    }
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    run_prepared(state, executable, body);
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule
