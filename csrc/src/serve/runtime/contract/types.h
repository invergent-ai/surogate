#pragma once

#include <array>
#include "api/types.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace sinfer::runtime {

using ::sinfer::FinishReason;
using ::sinfer::KvCapacityMode;
using ::sinfer::KvCapacityPolicy;
using ::sinfer::OutputChannel;
using ::sinfer::ResolvedSamplingParameters;
using ::sinfer::StopPolicy;
using ::sinfer::StopString;
using ::sinfer::TokenId;

// Engine has already selected the registered model/mode preset, applied every explicit override,
// and validated these values before constructing the runtime request.
struct ResolvedExecutionOptions {
    std::shared_ptr<const CompiledTokenConstraint> constraint;
    ResolvedSamplingParameters sampling;
    std::uint32_t requested_output_tokens = 0;
    int prompt_logprobs = -1;
    int top_logprobs = -1;
    bool allow_prefix_reuse               = true;
    /// Bank slot of the LoRA adapter this request selected, -1 for the base model.
    std::int32_t lora_slot = -1;
    /// A minimum length, and the stop ids barred until it is reached.
    std::uint32_t min_tokens         = 0;
    std::array<TokenId, 4> stop_barrier{};
    std::uint32_t stop_barrier_count = 0;
};

struct ResolvedRequestOptions {
    ResolvedExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

struct OutputDecision {
    std::uint32_t accepted_tokens = 0;
    FinishReason finish_reason    = FinishReason::None;

    [[nodiscard]] bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

// Complete request-lifetime ownership in the three independently exhausted admission domains.
// Values are already rounded to the physical allocation granularity by the target.
struct AdmissionResources {
    std::uint32_t active_lanes     = 0;
    std::uint32_t main_kv_pages    = 0;
    std::uint32_t backend_kv_pages = 0;
};

struct RequestPlanSummary {
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t reusable_prompt_tokens  = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    FinishReason effective_limit_reason   = FinishReason::None;
    std::size_t transient_bytes           = 0;
    std::size_t transient_alignment       = 1;
    AdmissionResources admission;
    std::uint64_t service_work_quanta = 0;
};

struct BeginSummary {
    std::uint32_t prompt_tokens        = 0;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::FullReset;
};

struct GeneratedRound {
    std::span<const TokenId> tokens;
    /// The log-probability of each token under the full vocabulary, same order and
    /// length as `tokens`. Empty from a route that does not produce them -- the
    /// speculative ones, whose licensed tokens come from a verify step rather than
    /// from one sampled column -- so a consumer checks before reading.
    std::span<const float> logprobs;
};

struct BatchedGeneratedRound {
    std::span<const TokenId> tokens;
    /// As `GeneratedRound::logprobs`, laid out exactly like `tokens`: the same
    /// `row_stride` per row. Empty when the route does not produce them.
    std::span<const float> logprobs;
    std::span<const std::int32_t> row_counts;
    std::uint32_t row_stride = 1;
};

struct PrefillStepResult {
    BeginSummary summary;
    GeneratedRound round;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    std::uint32_t feature_offset = 0;
    std::int32_t feature_base = -1;
    // A suspended text-layer slice still forwards its residual to the next stage.
    bool has_stage_residual = false;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

// A mixed round advances prompt chunks alongside ordinary decode or speculative
// verification. Decode rows use BatchedGeneratedRound's counts and stride. `prefills[i]` belongs to the i-th lane the caller passed in.
inline constexpr std::size_t kMaximumMixedPrefills = 8;

struct MixedRoundResult {
    std::array<PrefillStepResult, kMaximumMixedPrefills> prefills{};
    std::size_t prefill_count = 0;
    BatchedGeneratedRound round;

    [[nodiscard]] const PrefillStepResult& prefill_at(std::size_t index) const {
        if (index >= prefill_count) { throw std::out_of_range("mixed round prefill index"); }
        return prefills[index];
    }
};

// Target-produced affine reservation curve for one Main KV physical-capacity axis. The byte
// values come from complete target physical layout plans, not from a model geometry formula in
// the common runtime.
struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;
    std::uint32_t minimum_main_page_groups           = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::size_t minimum_device_reservation_bytes     = 0;
    std::size_t bytes_per_additional_main_page_group = 0;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                              = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::uint32_t resolved_tokens                    = 0;
    std::size_t minimum_runtime_reservation_bytes    = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::size_t runtime_reservation_bytes            = 0;
    std::size_t available_after_weights_bytes        = 0;
    std::size_t available_after_startup_bytes        = 0;
    std::size_t automatic_headroom_bytes             = 0;
    std::size_t planned_slack_bytes                  = 0;
};

} // namespace sinfer::runtime
