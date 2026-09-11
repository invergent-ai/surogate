#pragma once
#include "core/ngram_ple_state.h"
#include "family/impl/runtime/speculative_constraint.h"
#include "family/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "api/ops/sampling.h"
#include "api/ops/bidirectional_gqa_attention.h"
#include "api/ops/kv_cache_append_prefix.h"
#include "api/ops/swa.h"
#include "core/decode_graph.h"
#include "runtime/contract/transient_region.h"
#include <api/family/prepared_prompt.h>
#include <api/family/decoder_state.h>
#include "family/impl/runtime/text_context.h"
#include "family/impl/runtime/dflash_context.h"
#include "family/impl/runtime/vision_context.h"
#include "family/impl/runtime/vision_prefill.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <optional>
#include <span>

namespace sinfer::family::detail {
class PrefillGraphFamily;
} // namespace sinfer::family::detail

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule {

using family::PreparedPromptData;
using family::PromptModality;

struct ExecutionCore {
    DeviceContext& device;
    const LoadedModelData& model;
    WorkspaceArena& work;
    LinearAttentionStatePool& linear_attention;
    const GdnReplayRecords* replay_records;
    family::RoundState& io;
    Tensor& prefill_hidden;
    std::uint32_t prefill_chunk;
    ProposalHead proposal_head;
    NgramPleStatePool* ple = nullptr; ///< the layer prologue's per-slot state, when the target has one
    StageSpan stage{};                 ///< pipeline stage (whole model by default)
    SpeculativeConstraintRound* constraints = nullptr;
};

struct PrefillContext {
    ExecutionCore execution;
    family::PagedKVCacheView text_kv;
    family::PagedKVCacheView mtp_kv;
    const family::PagedKVCache& text_cache;
    const family::PagedKVCache* mtp_cache;
    DFlashPersistentState* dflash;
    std::uint32_t text_kv_base;
    const ops::SamplingConfig* sampling;
    Tensor* rewrite_checkpoint_hidden;
    std::int32_t current_state_slot                         = 0;
    std::int32_t rewrite_checkpoint_state_slot              = 0;
    std::uint32_t mtp_proposal_extent                       = 0;
    const family::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    PrefillGraphFamily* prefill_graphs                      = nullptr;
    /// The LoRA slot of the request being prefilled, -1 for the base model. Every
    /// column of a prefill chunk belongs to this one request, so the adapter is a
    /// scalar here rather than the per-lane vector a decode round carries.
    std::int32_t lora_slot = -1;
    bool score_prompt = false;
    int score_prompt_start = 1;
    int score_prompt_end = 0;
    std::function<void(const Tensor&, int, bool)> logprob_observer;
};

struct OrdinaryBatchContext {
    ExecutionCore execution;
    const family::PagedKVCache& text_cache;
    family::OrdinaryDecodeState& frame;
    const family::OrdinaryDecodeIngress& host_ingress;
    family::OrdinaryDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    // Round chaining (PATCHES.md #32): device I32 scalar holding 1 for the
    // in-graph position increments of the chained round flavor.
    Tensor chain_one;
};

struct MtpBatchContext {
    ExecutionCore execution;
    const family::PagedKVCache& text_cache;
    const family::PagedKVCache& mtp_cache;
    family::MtpDecodeState& frame;
    const family::MtpDecodeIngress& host_ingress;
    family::MtpDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
    /// Device I32 scalar holding 1: the narrow round advances the frontiers by it in-graph.
    Tensor one;
};

struct DFlashBatchContext {
    ExecutionCore execution;
    const family::PagedKVCache& text_cache;
    DFlashPersistentState& dflash;
    family::DFlashDecodeState& frame;
    const family::DFlashDecodeIngress& host_ingress;
    family::DFlashDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
};

struct DFlashAppendContext {
    ExecutionCore execution;
    DFlashPersistentState& dflash;
};

struct MtpGqaEnvelopes {
    ops::GqaExecutionEnvelope target_verify;
    ops::GqaExecutionEnvelope batch;
    // One envelope per autoregressive draft step after the first, so K-1 of them
    // -- and none at all for a target with no MTP head, where K is 0. Written as
    // an unsigned `K - 1` that was an array of four billion envelopes, which GCC
    // reports as "passing too large argument on stack" from the lambda that
    // captures this struct, several headers away from the cause.
    std::array<ops::GqaExecutionEnvelope,
               kMaximumMtpDraftTokens == 0 ? 0 : kMaximumMtpDraftTokens - 1>
        ar;
};

struct DFlashEnvelopes {
    ops::SwaContextExecutionEnvelope local;
    ops::GqaContextExecutionEnvelope full;
    ops::KVCacheAppendPrefixExecutionEnvelope append;
};

struct TargetVerifyFrameView {
    Tensor ids;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor valid_columns;
    Tensor kv_table_rows;
    Tensor lanes;
    Tensor target_hidden;
    Tensor target_logits;
    Tensor target_tokens;
    Tensor drafts;
    Tensor current_extents;
    Tensor frontiers;
    Tensor anchors;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor selected_hidden;
    const GdnReplayRecords* replay_records = nullptr;
    const ops::SamplingConfig* sampling    = nullptr;
    DFlashFeatureSink* feature_sink        = nullptr;
};

void configure_text_card(TextContext& card, const ExecutionCore& execution,
                         const ops::SamplingConfig* sampling, std::int32_t current_state_slot,
                         std::int32_t rewrite_checkpoint_state_slot,
                         std::uint32_t mtp_proposal_extent);
void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope);

[[nodiscard]] PrefillChunkResult prefill_text_chunk(
    PrefillContext& state, std::span<const TokenId> ids, std::uint32_t nominal_length,
    std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier, bool finalize_at_end);

[[nodiscard]] PrefillChunkResult
prefill_multimodal_chunk(PrefillContext& state, const PreparedPromptData& prompt,
                         VisionPrefillSession& vision, std::uint32_t nominal_length,
                         std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier,
                         bool finalize_at_end);

struct MtpBridgeInput {
    const Tensor* previous_hidden = nullptr;
    std::int32_t position         = 0;
    std::array<std::int32_t, 3> rope_position{};
};

void sample_from_hidden(PrefillContext& state, const Tensor& hidden, std::int32_t absolute_position,
                        std::int32_t purpose);
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding = nullptr);
void mtp_bridge_multimodal(PrefillContext& state, const PreparedPromptData& prompt,
                           VisionPrefillSession& vision, const MtpBridgeInput& bridge);

// Executes one exact-B ordinary decode traversal. All request rows enter through the stable
// ordinary ingress, share one model schedule, publish continuation hidden by selector, and leave
// through one compact egress transfer.
void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::GqaExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition);
void capture_ordinary_decode_batch_chained(OrdinaryBatchContext& state, std::int32_t batch_size,
                                           ops::GqaExecutionEnvelope envelope,
                                           DecodeGraphDefinition& definition);
void ordinary_decode_batch_chained(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::GqaExecutionEnvelope envelope,
                                   DecodeGraphExecutable* executable);
void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::GqaExecutionEnvelope envelope, DecodeGraphExecutable* executable);

// Executes one exact-B MTP verification/alignment/proposal transaction. Each row may carry a
// different current and next proposal extent while the model traversal remains batched.
// `narrow`: the head's round for a batch too wide to pay for a verify -- one column per lane
// through the trunk, sampled as an ordinary round samples, the head aligned on it and
// proposing the next drafts as usual.
void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpGqaEnvelopes envelopes, DecodeGraphDefinition& definition,
                              bool narrow = false);
void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpGqaEnvelopes envelopes, DecodeGraphExecutable* executable,
                      bool narrow = false);

[[nodiscard]] DFlashFeatureSink
dflash_feature_sink(PrefillContext& state, DFlashFeatureSink::PrefillConsumer consume_prefill = {});
void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);
void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::GqaExecutionEnvelope target_envelope,
                                 DecodeGraphDefinition& definition);
void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes, ops::GqaExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable);

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule
