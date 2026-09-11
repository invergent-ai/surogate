#include "family/impl/runtime/instance.h"
#include "family/impl/runtime/prefill_graph.h"
#include "family/impl/runtime/program.h"
#include <cstdlib>

#include "family/impl/runtime/schedule.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS {
namespace {

void validate_sampling(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty) ||
        !std::isfinite(sampling.repetition_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    // Zero would divide a positive logit by nothing; a negative one would flip the
    // sign of every seen token, which is not a penalty.
    if (sampling.repetition_penalty <= 0.0F) {
        throw std::invalid_argument("repetition_penalty must be positive");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
}

ops::SamplingConfig translate_sampling(const ResolvedSamplingParameters& source) {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.repetition_penalty = source.repetition_penalty;
    out.seed              = source.seed;
    out.token_counts      = nullptr;
    return out;
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint64_t projected_service_work(const runtime::RequestPlanSummary& summary,
                                     std::uint32_t reuse_base, std::uint32_t prefill_chunk,
                                     std::size_t prefill_splits) noexcept {
    const std::uint32_t suffix = summary.prompt_tokens - reuse_base;
    const std::uint64_t prefill_units =
        suffix == 0
            ? 1ULL
            : 1ULL + (static_cast<std::uint64_t>(suffix) - 1ULL) / prefill_chunk + prefill_splits;
    const std::uint64_t decode_units =
        summary.effective_output_tokens == 0 ? 0ULL : summary.effective_output_tokens - 1ULL;
    return prefill_units + decode_units;
}

} // namespace

RequestBasePlan
ProgramImplCore::plan_request_base(const PreparedPromptData& prompt,
                                   const runtime::ResolvedExecutionOptions& options) {
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (prompt.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= cfg.token_domain) {
            throw std::invalid_argument("prompt contains token outside the 248077-token domain");
        }
    }
    if (prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3ULL * prompt.token_ids.size()) {
        throw std::invalid_argument("prepared prompt token metadata has an invalid shape");
    }
    if (prompt.has_media() != !prompt.media_payloads.empty() ||
        prompt.media_payloads.size() != prompt.vision_items.size()) {
        throw std::invalid_argument("prepared prompt media payload is incomplete");
    }
    for (std::size_t i = 0; i < prompt.media_payloads.size(); ++i) {
        if (!prompt.media_payloads[i] ||
            prompt.media_payloads[i]->patch_elements !=
                prompt.vision_items[i].patch_count * static_cast<std::size_t>(model.vision_geometry.patch_dim)) {
            throw std::invalid_argument("prepared prompt media item payload has an invalid shape");
        }
    }
    if (prompt.has_media() && !vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    validate_sampling(options.sampling);
    for (const auto& [token, bias] : options.sampling.logit_bias) {
        if (token < 0 || token >= cfg.token_domain || !std::isfinite(bias) || bias < -100.0F || bias > 100.0F) {
            throw std::invalid_argument("logit_bias token is outside the model vocabulary or value is outside [-100,100]");
        }
    }

    auto base                             = std::make_unique<RequestBasePlanImpl>();
    base->summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.token_ids.size());
    base->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        capacity - base->summary.prompt_tokens + static_cast<std::uint32_t>(1);
    base->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason = options.requested_output_tokens <= capacity_output
                                               ? FinishReason::OutputLimit
                                               : FinishReason::ContextCapacity;
    base->summary.transient_alignment    = 1;
    base->summary.transient_bytes        = 0;
    base->sampling                       = translate_sampling(options.sampling);
    base->sampling.top_logprobs = options.top_logprobs;
    base->logit_bias                     = options.sampling.logit_bias;
    base->constraint                     = options.constraint;
    base->prompt_logprobs = options.prompt_logprobs;
    base->top_logprobs = options.top_logprobs;
    base->allow_prefix_reuse = options.allow_prefix_reuse;
    base->lora_slot                      = options.lora_slot;
    base->min_tokens                     = options.min_tokens;
    base->stop_barrier_count =
        std::min<std::uint32_t>(options.stop_barrier_count, ops::SamplingConfig::kMaxSuppressed);
    for (std::uint32_t i = 0; i < base->stop_barrier_count; ++i) {
        base->sampling.suppressed[i] = options.stop_barrier[i];
    }
    // Nothing has been produced yet, so the barrier is up for the prefill's own
    // token; the per-round staging lowers it once the request is long enough.
    base->sampling.suppressed_count = static_cast<std::int32_t>(base->stop_barrier_count);
    // A prefill graph chunk writes its whole 128-rounded bucket, pad columns included,
    // so the request must own the pages that window can reach as well as its output
    // extent, or mapping the chunk lands outside the entitlement and the round dies.
    // The window is rounded from the chunk's cursor, not from zero: a prefix-reuse
    // follow-up or a rewrite-checkpoint restore starts at an arbitrary frontier, and
    // a chunk at cursor c reaches c + roundup128(prompt - c), which is at most
    // prompt + 127. Reserving to the rounded prompt alone assumed every chunk starts
    // 128-aligned, and a follow-up with a short output budget then killed the engine.
    const std::uint32_t graph_reach_tokens =
        PrefillGraphFamily::graph_prefill_reach(base->summary.prompt_tokens);
    const std::uint32_t reserved_context_tokens = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        capacity,
        std::max<std::uint64_t>(
            static_cast<std::uint64_t>(base->summary.prompt_tokens) +
                (base->summary.effective_output_tokens == 0
                     ? 0U
                     : base->summary.effective_output_tokens - 1U),
            graph_reach_tokens)));
    base->text_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    if (speculative_backend == SpeculativeBackend::Mtp) {
        const std::uint32_t mtp_tokens    = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(reserved_context_tokens) + draft_window - 1ULL));
        base->backend_kv_page_entitlement = pages_for_tokens(mtp_tokens);
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        base->backend_kv_page_entitlement = pages_for_tokens(reserved_context_tokens);
    }
    base->summary.admission = runtime::AdmissionResources{
        .active_lanes     = 1,
        .main_kv_pages    = base->text_kv_page_entitlement,
        .backend_kv_pages = base->backend_kv_page_entitlement,
    };
    if (prompt.has_media()) {
        // The tower these weights carry, which is what the encode will actually run.
        const family::VisionGeometry vision =
            schedule::bound_vision_geometry(model.vision_geometry, model.geometry);
        auto control = std::make_shared<family::VisionControl>(
            family::build_vision_control(prompt, vision.position_embeddings));
        std::size_t max_merged     = 0;
        std::uint32_t previous_end = 0;
        for (const family::VisionItemControl& item : control->items) {
            if (item.scatter_indices.empty()) {
                throw std::invalid_argument("vision item has no Text consumer columns");
            }
            const auto first = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last  = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin =
                speculative_backend == SpeculativeBackend::Mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end = last + 1;
            if (begin < previous_end) {
                throw std::invalid_argument("vision item consumer spans overlap");
            }
            if (end > base->summary.prompt_tokens) {
                throw std::invalid_argument("vision item consumer span exceeds prompt");
            }
            if (schedule::VisionContext::workspace_bytes(vision, item) > work.capacity()) {
                throw std::invalid_argument("vision item exceeds the Program workspace envelope");
            }
            previous_end = end;
            max_merged   = std::max(max_merged, item.merged_count);
        }
        base->vision_transient_bytes =
            schedule::VisionContext::output_transient_bytes(vision, max_merged);
        base->vision_control         = std::move(control);
    }

    if (prompt.identity.rewrite_checkpoint) {
        const RewriteCheckpointSpec candidate = *prompt.identity.rewrite_checkpoint;
        if (candidate.frontier == 0 || candidate.frontier > base->summary.prompt_tokens) {
            throw std::invalid_argument(
                "rewrite checkpoint frontier must lie at or inside the prompt frontier");
        }
        base->rewrite_checkpoint = candidate;
    }
    const std::size_t cold_prefill_splits =
        (base->vision_control != nullptr ? base->vision_control->items.size() : 0ULL) +
        (base->rewrite_checkpoint &&
                 base->rewrite_checkpoint->frontier < base->summary.prompt_tokens
             ? 1ULL
             : 0ULL);
    base->summary.service_work_quanta =
        projected_service_work(base->summary, 0, prefill_chunk, cold_prefill_splits);
    return RequestBasePlan(std::move(base));
}

RequestPlan ProgramImplCore::plan_request_for_lane(std::uint32_t lane,
                                                   const PreparedPromptData& prompt,
                                                   const RequestBasePlan& base_plan) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    const RequestControl& request = requests[lane];
    const SequenceState& sequence = sequences[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("cannot plan a request while Program is active or pending");
    }
    if (base_plan.impl_ == nullptr) { throw std::logic_error("request base plan is empty"); }
    const RequestBasePlanImpl& base = *base_plan.impl_;

    auto plan                         = std::make_unique<RequestPlanImpl>();
    plan->summary                     = base.summary;
    plan->prompt_logprobs = base.prompt_logprobs;
    plan->top_logprobs = base.top_logprobs;
    plan->sampling                    = base.sampling;
    plan->logit_bias                   = base.logit_bias;
    plan->constraint                   = base.constraint;
    plan->text_kv_page_entitlement    = base.text_kv_page_entitlement;
    plan->backend_kv_page_entitlement = base.backend_kv_page_entitlement;
    plan->lora_slot                   = base.lora_slot;
    plan->min_tokens                  = base.min_tokens;
    plan->stop_barrier_count          = base.stop_barrier_count;

    if (base.allow_prefix_reuse && prompt.identity.reusable && sequence.retained && !sequence.cached_scores.empty()) {
        const auto limit = std::min({prompt.token_ids.size(), sequence.ledger.size(), sequence.cached_scores.size()});
        const auto needed = std::min(cfg.token_domain, std::max(0, base.prompt_logprobs));
        std::uint32_t count = 1;
        while (count < limit && sequence.cached_scores[count].selected.token_id == prompt.token_ids[count] &&
               sequence.cached_scores[count].top.size() >= std::size_t(needed)) ++count;
        if (limit >= 1 && family::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity, count, base.lora_slot)) {
            plan->reusable_scores = count;
        }
    }

    if (base.allow_prefix_reuse && prompt.identity.reusable && sequence.retained) {
        const bool dflash_append_ready =
            speculative_backend != SpeculativeBackend::DFlash ||
            sequence.dflash_context_frontier == sequence.execution_frontier;
        if (sequence.execution_frontier != 0 && dflash_append_ready &&
            family::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                            sequence.execution_frontier, base.lora_slot)) {
            plan->reuse      = ReusePath::AppendAtFrontier;
            plan->reuse_base = sequence.execution_frontier;
        } else if (sequence.rewrite_checkpoint.valid && sequence.rewrite_checkpoint.frontier != 0 &&
                   sequence.rewrite_checkpoint.frontier <= prompt.token_ids.size() &&
                   family::detail::prefix_matches(prompt, sequence.ledger,
                                                   sequence.prefix_identity,
                                                   sequence.rewrite_checkpoint.frontier,
                                                   base.lora_slot)) {
            plan->reuse      = restore_path(sequence.rewrite_checkpoint.kind);
            plan->reuse_base = sequence.rewrite_checkpoint.frontier;
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const bool append_ready =
            plan->reuse == ReusePath::AppendAtFrontier && sequence.tail_hidden_valid &&
            decoder->mtp_cache() != nullptr &&
            (plan->reuse_base == 0 || sequence.mtp_kv_valid >= plan->reuse_base - 1);
        const bool checkpoint_ready = is_rewrite_checkpoint_restore(plan->reuse) &&
                                      decoder->mtp_cache() != nullptr && plan->reuse_base != 0 &&
                                      sequence.mtp_kv_valid >= plan->reuse_base - 1;
        if (plan->reuse != ReusePath::FullReset && !append_ready && !checkpoint_ready) {
            plan->reuse      = ReusePath::FullReset;
            plan->reuse_base = 0;
        }
    }

    if (is_rewrite_checkpoint_restore(plan->reuse) &&
        speculative_backend == SpeculativeBackend::DFlash &&
        (!dflash || !sequence.kv || !sequence.kv->backend ||
         sequence.dflash_context_frontier < plan->reuse_base)) {
        plan->reuse      = ReusePath::FullReset;
        plan->reuse_base = 0;
    }

    // Missing scores before the retained frontier require prefill replay. Its
    // boundary token can be scored from the retained final hidden state.
    if (base.prompt_logprobs >= 0 && plan->reuse_base > 0 &&
        plan->reusable_scores < plan->reuse_base) {
        plan->reuse = ReusePath::FullReset;
        plan->reuse_base = 0;
    }

    // With rewrite checkpoints disabled the state pool holds no checkpoint
    // slots, so nothing may be desired, captured or deferred: the plan drops.
    static const std::optional<RewriteCheckpointSpec> kNoCheckpoint;
    const std::optional<RewriteCheckpointSpec>& desired =
        rewrite_checkpoints ? base.rewrite_checkpoint : kNoCheckpoint;
    const bool existing_checkpoint_matches =
        desired && plan->reuse != ReusePath::FullReset && sequence.rewrite_checkpoint.valid &&
        sequence.rewrite_checkpoint.frontier == desired->frontier &&
        family::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                        desired->frontier, base.lora_slot);
    if (!desired) {
        plan->rewrite_checkpoint_action = RewriteCheckpointAction::Drop;
    } else if (existing_checkpoint_matches) {
        plan->rewrite_checkpoint_action = sequence.rewrite_checkpoint.kind == desired->kind
                                              ? RewriteCheckpointAction::KeepExisting
                                              : RewriteCheckpointAction::ReclassifyExisting;
    } else if (desired->frontier > plan->reuse_base) {
        // surogate vendor patch (PATCHES.md #24): capturing the rewrite
        // checkpoint splits the final prefill chunk at frontier (prompt - 4)
        // and pays a second full-model pass for the tail (~12 ms at 4B, 15%
        // of a 1.9k prefill). With deferral the checkpoint is captured only
        // when a rewrite actually replays that prefix — rewind-heavy flows
        // pay the same cost later; everything else keeps the 12 ms.
        static const bool defer_capture = [] {
            const char* env = std::getenv("SUROGATE_SERVE_DEFER_REWRITE_CHECKPOINT");
            return env != nullptr && env[0] == '1';
        }();
        if (defer_capture) {
            plan->rewrite_checkpoint_action = RewriteCheckpointAction::DeferCapture;
        } else {
            plan->rewrite_checkpoint_action  = RewriteCheckpointAction::CaptureNew;
            plan->rewrite_checkpoint_capture = desired;
        }
    } else {
        // The selected continuation state is already past the desired boundary. It remains a
        // valid hit; do not replay an otherwise reusable prefix merely to materialize an older
        // auxiliary snapshot. A later request can still use the checkpoint currently retained.
        plan->rewrite_checkpoint_action = RewriteCheckpointAction::DeferCapture;
    }

    plan->summary.reusable_prompt_tokens = plan->reuse_base;
    // SUROGATE_SERVE_REUSE_TRACE=1: every non-FullReset plan, with enough identity to correlate
    // a later wrong answer with the reuse decision that produced it.
    static const bool reuse_trace = std::getenv("SUROGATE_SERVE_REUSE_TRACE") != nullptr;
    if (reuse_trace && plan->reuse != ReusePath::FullReset) {
        const auto head = [](const auto& tokens, std::size_t n) {
            std::string out;
            for (std::size_t i = 0; i < n && i < tokens.size(); ++i) {
                out += std::to_string(tokens[i]);
                out += ',';
            }
            return out;
        };
        std::fprintf(stderr,
                     "reuse-trace: plan lane %u path %d base %u frontier %u prompt %zu "
                     "prompt_head %s ledger_head %s\n",
                     lane, static_cast<int>(plan->reuse), plan->reuse_base,
                     sequence.execution_frontier, prompt.token_ids.size(),
                     head(prompt.token_ids, 6).c_str(), head(sequence.ledger, 6).c_str());
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        if (plan->reuse == ReusePath::FullReset) {
            plan->prepare_mtp = true;
        } else if (plan->reuse == ReusePath::AppendAtFrontier) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        } else if (is_rewrite_checkpoint_restore(plan->reuse)) {
            plan->prepare_mtp = true;
            plan->mtp_bridge  = plan->reuse_base < plan->summary.prompt_tokens
                                    ? MtpBridgeMode::BeforeSuffix
                                    : MtpBridgeMode::AfterExactHit;
        }
    }

    if (base.vision_control != nullptr) {
        VisionPrefillPlan vision;
        vision.control = base.vision_control;
        vision.uses.reserve(base.vision_control->items.size());
        for (std::size_t index = 0; index < base.vision_control->items.size(); ++index) {
            const family::VisionItemControl& item = base.vision_control->items[index];
            const auto first          = static_cast<std::uint32_t>(item.scatter_indices.front());
            const auto last           = static_cast<std::uint32_t>(item.scatter_indices.back());
            const std::uint32_t begin = plan->prepare_mtp && first != 0 ? first - 1 : first;
            const std::uint32_t end   = last + 1;
            if (end <= plan->reuse_base) { continue; }
            vision.uses.push_back(VisionUseSpan{begin, end, static_cast<std::uint32_t>(index)});
        }
        if (!vision.uses.empty()) {
            plan->summary.transient_alignment = 256;
            plan->summary.transient_bytes     = base.vision_transient_bytes;
            plan->vision                      = std::move(vision);
        }
    }

    const std::size_t prefill_splits =
        (plan->vision ? plan->vision->uses.size() : 0ULL) +
        (plan->rewrite_checkpoint_capture &&
                 plan->rewrite_checkpoint_capture->frontier < plan->summary.prompt_tokens
             ? 1ULL
             : 0ULL);
    plan->summary.service_work_quanta =
        projected_service_work(plan->summary, plan->reuse_base, prefill_chunk, prefill_splits);
    return RequestPlan(std::move(plan));
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS
