#pragma once

#include "core/layout.h"
#include "core/tensor.h"
#include "api/ops/sampling.h"
#include "api/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace sinfer::family {

inline constexpr std::uint32_t kMtpDecodeMaximumDrafts    = 5;
inline constexpr std::uint32_t kMtpDecodeMaximumWidth     = kMtpDecodeMaximumDrafts + 1;
inline constexpr std::uint32_t kDFlashDecodeMaximumDrafts = 15;
inline constexpr std::uint32_t kDFlashDecodeMaximumWidth  = kDFlashDecodeMaximumDrafts + 1;

struct RoundStateSpec {
    std::int32_t hidden          = 0;
    std::int32_t output_rows     = 0;
    std::uint32_t batch_capacity = 1;
    std::uint32_t draft_window   = 0;
    bool enable_mtp              = false;
    bool enable_dflash           = false;
};

// Stable pinned/device transfer format for ordinary decode. The full fixed-size object is copied
// once per round; only its exact-B prefixes are consumed by the model schedule.
struct OrdinaryDecodeIngress {
    std::array<TokenId, kMaximumBatchColumns> tokens{};
    std::array<std::int32_t, kMaximumBatchColumns> cache_positions{};
    std::array<std::int32_t, kMaximumBatchColumns> rope_positions{};
    std::array<std::int32_t, kMaximumBatchColumns> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumBatchColumns> lanes{};
    /// The LoRA slot each lane's request selected, or -1 for the base model.
    /// It rides in the ingress rather than a buffer of its own because this
    /// struct is already the round's host-to-device channel, staged in a way a
    /// captured graph replays correctly -- the property an adapter index needs.
    std::array<std::int32_t, kMaximumBatchColumns> lora_slots{};
    std::array<ops::SamplingConfig, kMaximumBatchColumns> sampling{};
};

struct OrdinaryDecodeEgress {
    std::array<TokenId, kMaximumBatchColumns> sampled_tokens{};
    /// The log-probability of each sampled token under that lane's full-vocabulary
    /// temperature-scaled distribution. It rides in the egress rather than a
    /// buffer of its own because this struct is already the round's
    /// device-to-host channel, and a reinforcement-learning client needs the
    /// number for every token it was given -- there is no request that wants the
    /// token but not its probability.
    std::array<float, kMaximumBatchColumns> sampled_logprobs{};
    std::array<RawTokenScores, kMaximumBatchColumns * 1> scores{};
};

// Stable pinned/device transfer formats for concurrent MTP decode. The arrays use the maximum
// product domain; RoundState binds only the configured [K,C] and [K+1,C] prefixes.
struct MtpDecodeIngress {
    std::array<TokenId, kMaximumBatchColumns> anchors{};
    std::array<std::int32_t, kMaximumBatchColumns> base_frontiers{};
    std::array<std::int32_t, kMaximumBatchColumns> remaining_budgets{};
    std::array<std::int32_t, kMaximumBatchColumns> current_extents{};
    std::array<std::int32_t, kMaximumBatchColumns> target_valid_columns{};
    std::array<TokenId, kMaximumBatchColumns * kMtpDecodeMaximumDrafts> current_drafts{};
    std::array<std::int32_t, kMaximumBatchColumns * kMtpDecodeMaximumWidth> target_rope_positions{};
    std::array<std::int32_t, kMaximumBatchColumns> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumBatchColumns> mtp_kv_table_rows{};
    std::array<std::int32_t, kMaximumBatchColumns> lanes{};
    std::array<std::int32_t, kMaximumBatchColumns> rope_deltas{};
    std::array<std::int32_t, kMaximumBatchColumns> lora_slots{};
    std::array<ops::SamplingConfig, kMaximumBatchColumns> sampling{};
};

struct MtpDecodeEgress {
    std::array<TokenId, kMaximumBatchColumns * kMtpDecodeMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumBatchColumns> licensed_counts{};
    std::array<std::int32_t, kMaximumBatchColumns> accepted_drafts{};
    // Step-major: all B rows for proposal step 0, followed by all B rows for step 1, etc.
    std::array<TokenId, kMaximumBatchColumns * kMtpDecodeMaximumDrafts> next_drafts{};
    std::array<std::int32_t, kMaximumBatchColumns> next_extents{};
    std::array<RawTokenScores, kMaximumBatchColumns * kMtpDecodeMaximumWidth> scores{};
};

// Stable pinned/device transfer formats for one exact-B DFlash transaction. The proposal is
// produced and verified in the same round, so no draft state crosses the round boundary.
struct DFlashDecodeIngress {
    std::array<TokenId, kMaximumBatchColumns> anchors{};
    std::array<std::int32_t, kMaximumBatchColumns> execution_frontiers{};
    std::array<std::int32_t, kMaximumBatchColumns> context_frontiers{};
    std::array<std::int32_t, kMaximumBatchColumns> proposal_extents{};
    std::array<std::int32_t, kMaximumBatchColumns> target_valid_columns{};
    // Target positions include the visual prompt's RoPE offset; draft/cache positions
    // remain absolute token indices. Packed [draft_window + 1, batch] like MTP.
    std::array<std::int32_t, kMaximumBatchColumns * kDFlashDecodeMaximumWidth>
        target_rope_positions{};
    std::array<std::int32_t, kMaximumBatchColumns> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumBatchColumns> dflash_kv_table_rows{};
    std::array<std::int32_t, kMaximumBatchColumns> lanes{};
    std::array<std::int32_t, kMaximumBatchColumns> lora_slots{};
    std::array<ops::SamplingConfig, kMaximumBatchColumns> sampling{};
};

struct DFlashDecodeEgress {
    std::array<TokenId, kMaximumBatchColumns * kDFlashDecodeMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumBatchColumns> licensed_counts{};
    std::array<std::int32_t, kMaximumBatchColumns> accepted_drafts{};
    std::array<RawTokenScores, kMaximumBatchColumns * kDFlashDecodeMaximumWidth> scores{};
};

struct OrdinaryDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion logits;
    TensorRegion hidden;
};

struct MtpPrefillStateLayout {
    TensorRegion position;
    TensorRegion ar_hidden;
    TensorRegion draft_tokens;
    TensorRegion target_input_ids;
    TensorRegion target_positions;
};

struct DFlashPrefillStateLayout {
    TensorRegion produced_count;
};

struct MtpDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion verify_ids;
    TensorRegion lora_columns;
    TensorRegion target_positions;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
    TensorRegion proposal_logits;
    TensorRegion alignment_ids;
    TensorRegion alignment_hidden;
    TensorRegion ar_hidden;
    TensorRegion next_hidden;
    TensorRegion ar_positions;
    TensorRegion ar_rope_positions;
    TensorRegion ar_valid_columns;
};

struct DFlashDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion proposal_ids;
    TensorRegion proposal_positions;
    TensorRegion append_positions;
    TensorRegion append_counts;
    TensorRegion draft_tokens;
    TensorRegion verify_ids;
    TensorRegion lora_columns;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
};

// Multi-prompt prefill (#80): the prompts that finish inside one mixed round sample together.
// Their last hidden columns gather here, one lm_head fills the logits, and the batched sampler
// writes the tokens - so F finishing prompts cost one vocabulary projection, not F.
inline constexpr std::int32_t kMaximumPrefillSegments = 8;

struct MixedPrefillFinalizeLayout {
    TensorRegion hidden;
    TensorRegion logits;
    TensorRegion positions;
    TensorRegion rope_positions;
    TensorRegion tokens;
    LayoutRegion sampling; // raw ops::SamplingConfig[F]
};

struct MixedPrefillFinalizeState {
    Tensor hidden;
    Tensor logits;
    Tensor positions;
    Tensor rope_positions;
    Tensor tokens;
    ops::SamplingConfig* sampling = nullptr;
};

struct RoundStateLayout {
    RoundStateSpec spec;
    std::optional<OrdinaryDecodeStateLayout> ordinary;
    MixedPrefillFinalizeLayout prefill_finalize;
    TensorRegion token;
    TensorRegion logprob;
    TensorRegion pos;
    TensorRegion rope_pos;
    TensorRegion rope_delta;
    TensorRegion logits;
    TensorRegion text_kv_table_row;
    TensorRegion backend_kv_table_row;
    std::optional<MtpPrefillStateLayout> mtp;
    std::optional<DFlashPrefillStateLayout> dflash_prefill;
    std::optional<MtpDecodeStateLayout> mtp_decode;
    std::optional<DFlashDecodeStateLayout> dflash_decode;
    bool complete = false;
};

struct OrdinaryDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor tokens;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor text_kv_table_rows;
    Tensor lanes;
    Tensor lora_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor sampled_tokens;
    Tensor sampled_logprobs;
    Tensor logits;
    Tensor hidden;

    OrdinaryDecodeState() = default;
    OrdinaryDecodeState(DeviceSpan backing, const OrdinaryDecodeStateLayout& layout,
                        std::uint32_t batch_capacity);
};

// The two planning calls expose one deliberate exact-target extension seam after scalar logits.
// This lets a target retain its schedule-sized prefill activation at the established physical
// address without making that activation part of the family round contract.
[[nodiscard]] RoundStateLayout begin_round_state_layout(LayoutBuilder& builder,
                                                        const RoundStateSpec& spec);
void complete_round_state_layout(LayoutBuilder& builder, RoundStateLayout& layout);

struct MtpPrefillState {
    Tensor position;
    Tensor ar_hidden;
    Tensor draft_tokens;
    Tensor target_input_ids;
    Tensor target_positions;

    MtpPrefillState() = default;
    MtpPrefillState(DeviceSpan backing, const MtpPrefillStateLayout& layout);
};

struct DFlashPrefillState {
    Tensor produced_count;

    DFlashPrefillState() = default;
    DFlashPrefillState(DeviceSpan backing, const DFlashPrefillStateLayout& layout);
};

struct MtpDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor base_frontiers;
    Tensor remaining_budgets;
    Tensor current_extents;
    Tensor target_valid_columns;
    Tensor current_drafts;
    Tensor target_rope_positions;
    Tensor text_kv_table_rows;
    Tensor mtp_kv_table_rows;
    Tensor lanes;
    Tensor lora_slots;
    Tensor lora_columns;
    Tensor rope_deltas;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor next_drafts;
    Tensor next_extents;
    Tensor verify_ids;
    Tensor target_positions;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;
    Tensor proposal_logits;
    Tensor alignment_ids;
    Tensor alignment_hidden;
    Tensor ar_hidden;
    Tensor next_hidden;
    Tensor ar_positions;
    Tensor ar_rope_positions;
    Tensor ar_valid_columns;

    MtpDecodeState() = default;
    MtpDecodeState(DeviceSpan backing, const MtpDecodeStateLayout& layout,
                   std::uint32_t batch_capacity, std::uint32_t draft_window);
};

struct DFlashDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor execution_frontiers;
    Tensor context_frontiers;
    Tensor proposal_extents;
    Tensor target_valid_columns;
    Tensor target_rope_positions;
    Tensor text_kv_table_rows;
    Tensor dflash_kv_table_rows;
    Tensor lanes;
    Tensor lora_slots;
    Tensor lora_columns;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor proposal_ids;
    Tensor proposal_positions;
    Tensor append_positions;
    Tensor append_counts;
    Tensor draft_tokens;
    Tensor verify_ids;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;

    void set_width(std::uint32_t width);
    DFlashDecodeState() = default;
    DFlashDecodeState(DeviceSpan backing, const DFlashDecodeStateLayout& layout,
                      std::uint32_t batch_capacity, std::uint32_t draft_window);
};

struct RoundState {
    std::optional<OrdinaryDecodeState> ordinary;
    MixedPrefillFinalizeState prefill_finalize;
    Tensor token;
    /// The step token's log-probability, the prefill twin of
    /// `OrdinaryDecodeEgress::sampled_logprobs`.
    Tensor logprob;
    Tensor pos;
    Tensor rope_pos;
    Tensor rope_delta;
    Tensor logits;
    Tensor text_kv_table_row;
    Tensor backend_kv_table_row;
    std::optional<MtpPrefillState> mtp;
    std::optional<DFlashPrefillState> dflash_prefill;
    std::optional<MtpDecodeState> mtp_decode;
    std::optional<DFlashDecodeState> dflash_decode;

    RoundState() = default;
    RoundState(DeviceSpan backing, const RoundStateLayout& layout);
};

} // namespace sinfer::family
