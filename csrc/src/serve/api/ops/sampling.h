#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace sinfer::ops {

// Counter-based RNG subkey. Distinct purposes keep draws at the same logical position separate.
enum SamplePurpose : std::int32_t {
    kSamplePurposePrefill               = 0,
    kSamplePurposeDecode                = 1,
    kSamplePurposeSpeculativeAccept     = 2,
    kSamplePurposeSpeculativeCorrection = 3,
    kSamplePurposeSpeculativeBonus      = 4,
};

// Device-resident sampling parameters. token_counts is an optional device I32
// [token_domain] occurrence-count array used by both penalties.
struct SamplingConfig {
    float temperature          = 0.0f; // <= 0 => greedy argmax after bias/masking
    /// <=0 means no limit of the caller's own: with top_p>=1 that is the whole
    /// vocabulary, otherwise the 20-candidate cap. A top_k above 20 is clamped to it.
    std::int32_t top_k         = 0;
    float top_p                = 1.0f; // >= 1 => disabled
    float min_p                = 0.0f; // <= 0 => disabled
    float presence_penalty     = 0.0f;
    float frequency_penalty    = 0.0f;
    /// vLLM's multiplicative penalty on tokens seen already: a positive logit is
    /// divided by it and a negative one multiplied, so a value above 1 pushes both
    /// towards zero. 1 disables it. Distinct from the two additive penalties above,
    /// and applied before them.
    float repetition_penalty   = 1.0f;
    unsigned long long seed    = 0;
    const std::int32_t* token_bitmask = nullptr; // optional allowed-token bitmap [ceil(token_domain/32)]
    const float* logit_bias = nullptr; // optional device F32 [token_domain], added before temperature
    std::int32_t* token_counts = nullptr; // device [token_domain] i32, or null
    /// Token ids barred from being drawn at all, and how many of the four slots
    /// are in use. This is how a minimum length is honoured: the stop tokens are
    /// barred until the request has produced enough, which is the only way to get
    /// one -- refusing to *finish* on a stop token would still have emitted it, and
    /// the model would go on proposing it.
    ///
    /// Inline rather than a device pointer because this struct is already copied
    /// to the device every round, so a bar that changes with the token count costs
    /// nothing extra to publish. Four is more than any stop set these models carry.
    static constexpr int kMaxSuppressed = 4;
    std::int32_t suppressed[kMaxSuppressed] = {-1, -1, -1, -1};
    std::int32_t suppressed_count           = 0;
};

// Recompute greedy verification tokens when a request has logit overrides.
// logits: BF16 [physical_rows,width,batch], targets: I32 [width,batch].
// Rows without overrides and stochastic rows are left unchanged.
void sampling_update_greedy_targets(const Tensor& logits, Tensor& targets,
                                    std::int32_t token_domain, const SamplingConfig* configs,
                                    cudaStream_t stream);

// Caller-owned transient capacity for every parallel sampling-lane count in the inclusive
// interval. For sample(), one lane is one batch row; speculative acceptance uses the same
// workspace substrate for its verification columns. token_domain is the fixed route profile.
// Invalid profiles or intervals throw; a legal single-block route returns zero.
[[nodiscard]] std::size_t sampling_workspace_capacity_bytes(std::int32_t token_domain,
                                                            std::int32_t min_lanes,
                                                            std::int32_t max_lanes);

/**
 * Produces one token id per independent request row. `logits` is contiguous BF16
 * [physical_rows,B], `out` and `logical_positions` are contiguous I32 [B], and only vocabulary
 * rows v in [0,token_domain) participate. `configs` is a device-resident contiguous
 * SamplingConfig[B] array. Greedy and stochastic rows may coexist in one invocation.
 *
 * For row b with configs[b].temperature<=0:
 *
 *   out[b] = min argmax_allowed_v (float(logits[v,b]) + bias_v).
 *
 * Penalties, filters, RNG, and token_counts updates are skipped for that row. With positive
 * temperature, let c_v=configs[b].token_counts[v] (or zero when the pointer is null):
 *
 *   adjusted_v = float(logits[v,b]) + bias_v
 *                - configs[b].presence_penalty * (c_v > 0)
 *                - configs[b].frequency_penalty * c_v.
 *
 * A row with top_k<=0 and top_p>=1 asked for no truncation at all and is drawn from the whole
 * vocabulary: its distribution is exp(adjusted_v/temperature) normalized over every v in
 * [0,token_domain), with min_p removing every v whose weight is below min_p times the largest.
 * That row never sees a candidate cap, so the tail of its distribution is reachable and its
 * probabilities are the ones a full-vocabulary log-softmax reports. It is realized by adding an
 * independent Gumbel(0,1) to each scaled logit and taking the argmax, which is that draw exactly.
 *
 * Every other row is drawn from a bounded candidate set. Candidates are sorted by adjusted_v
 * descending with lower token id breaking ties. Per-row top_k in [1,19] keeps that many
 * candidates; top_k>=20, or top_k<=0 alongside a top_p below 1, keeps min(20,token_domain) --
 * nucleus needs an ordered prefix, which the untruncated route does not produce. Candidate
 * weights are exp(adjusted_v/temperature-max). min_p removes the suffix below min_p*max_weight;
 * top_p keeps the shortest remaining prefix whose cumulative weight reaches top_p times the
 * pre-truncation candidate weight. At least the best candidate remains, the support is
 * renormalized, and one id is drawn for that row.
 *
 * Row b uses counter-based RNG key
 * (configs[b].seed,logical_positions[b],purpose), without mutable RNG state or dependence on the
 * compact row index. In the positive-temperature branch the selected token atomically increments
 * configs[b].token_counts when it is non-null. Non-null token-count arrays belonging to distinct
 * active requests must not alias. `out` must not overlap logits, configs, logical_positions, or any
 * token-count array. The Op writes all of out, uses caller-owned transient storage reported by
 * sampling_workspace_capacity_bytes(), and has no other persistent-state side effect.
 */
void sample(const Tensor& logits, Tensor& out, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, std::int32_t purpose,
            WorkspaceArena& workspace, cudaStream_t stream);

} // namespace sinfer::ops
