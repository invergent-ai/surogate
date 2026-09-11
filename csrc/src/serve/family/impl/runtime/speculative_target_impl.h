#include "family/impl/runtime/instance.h"
#include "family/impl/runtime/schedule.h"

#include "api/ops/scatter.h"
#include "api/ops/lora_store.h"
#include "api/ops/speculative_round.h"

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope, const MixedTargetForward& mixed_target) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    if (frame.ids.ne[0] == 1) {
        card.set_gdn_state_action(GdnStateAction::UpdateInPlace, nullptr);
    } else {
        card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    }
    {
        ops::ScopedLoraColumns adapter(frame.lora_columns);
        if (mixed_target) {
            mixed_target(card, frame, envelope);
        } else if (frame.feature_sink != nullptr) {
            card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                     frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                     frame.target_hidden, frame.target_logits, frame.target_tokens,
                                     *frame.feature_sink);
        } else {
            card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                     frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                     frame.target_hidden, frame.target_logits, frame.target_tokens);
        }
    }
    if (execution.stage.last >= 0 && execution.stage.last < execution.model.geometry.layers) { return; }
    if (execution.constraints) { execution.constraints->enqueue(frame.drafts, execution.device.stream); }
    ops::sampling_update_greedy_targets(frame.target_logits, frame.target_tokens,
                                       execution.model.geometry.token_domain, frame.sampling,
                                       execution.device.stream);
    ops::speculative_accept_greedy_drafts(
        frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
        frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
        frame.accepted_drafts, execution.model.geometry.token_domain, frame.sampling,
        execution.work, execution.device.stream, frame.ids.ne[0] == 1);
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule
