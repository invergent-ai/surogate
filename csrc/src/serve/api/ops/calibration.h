#pragma once

// Calibration statistics for structural pruning (rl/flash-next-student-2026-10-31/RECIPE.md
// section 9): per-expert routing statistics of every MoE layer -- the REAP score's numerator among
// them -- and residual samples at layer-group boundaries for the groups' block influence.
//
// Both are opt-in, process-wide, and configured once before the first model loads (surogate serve
// --moe-stats-out DIR, --residual-samples-out DIR); off, nothing here allocates or launches
// anything, and the served rounds are exactly the ones an engine without this computes.

#include "api/ops/sparse_moe.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace sinfer::ops {

// ---------------------------------------------------------------------------------------------
// MoE expert statistics
// ---------------------------------------------------------------------------------------------

struct MoeStatsOptions {
    /// Where dumps go (`DIR/<tag>/`); empty turns the statistics off.
    std::string directory;
    /// The REAP numerator needs each selected expert's own output norm, which the served round never
    /// forms (its kernels sum the weighted paths as they go). With `norms`, every mixture call is
    /// followed by experts_per_token probe rounds of the same kernels, each with the router weights
    /// rewritten to one slot (SparseMoeRoundHook::reweight), so prefill costs about
    /// (1 + experts_per_token) times the mixture's time. Off, only counts and gate sums are kept,
    /// and they cost nothing measurable.
    bool norms = true;
    /// Run the draft head (MTP) over every prompt column in prefill, so its mixture is observed on
    /// prompts too: the served prefill runs the head's attention keys over the prompt and its
    /// mixture on the last column only. Only meaningful with --spec mtp.
    bool mtp_prompt = true;
    /// What is being observed, for the record each dump carries (the artifact path).
    std::string model;
};

/// Sets the statistics up; call once, before any model loads. An empty directory turns them off.
void configure_moe_stats(const MoeStatsOptions& options);
[[nodiscard]] bool moe_stats_enabled() noexcept;
[[nodiscard]] bool moe_stats_norms() noexcept;
[[nodiscard]] bool moe_stats_mtp_prompt() noexcept;

/// Creates the calling device's accumulators: `mixtures` rows (the trunk's layers, then the draft
/// head's: row `trunk_layers` is the NextN block) of `experts` each. Call before any graph capture
/// on that device; a repeated call with the same shape is a no-op, another shape throws.
void moe_stats_prepare_device(std::int32_t mixtures, std::int32_t trunk_layers, std::int32_t experts,
                              std::int32_t experts_per_token, std::int32_t hidden);

/// A round hook that records the routing of one mixture call into row `layer`: one count and the
/// renormalised gate weight per (token, selected expert), and the call's token count. It reads the
/// round and rewrites nothing, so the call computes exactly what it does without it. `storage` holds
/// the hook's context and must outlive the sparse_moe call it is passed to.
struct MoeStatsObserver {
    void* stats            = nullptr;
    std::int32_t layer     = -1;
    std::int32_t top_k     = 0;
};
[[nodiscard]] SparseMoeRoundHook moe_stats_observer(std::int32_t layer, MoeStatsObserver& storage);

/// The probe rounds of one mixture call (MoeStatsOptions::norms): for each selected slot k, the
/// same sparse_moe call over `x` with the router weights rewritten to a one-hot on k and the shared
/// expert's scale to zero, so the call's output is that slot's unweighted expert output y; its L2
/// norm per token is added to the slot's expert as Sum ||y|| and Sum gate * ||y|| (gate = the
/// served round's weight of that slot). `moe_workspace` is a workspace the layer's own sparse_moe
/// call fits in (it is reused, call after call); `workspace` provides the probe's own planes,
/// `moe_stats_probe_workspace_bytes` of them. A no-op without norms or for a layer out of range.
void moe_stats_probe(std::int32_t layer, const Tensor& x, const SparseMoeWeights& weights,
                     SparseMoeRouting routing, WorkspaceArena& moe_workspace,
                     WorkspaceArena& workspace, cudaStream_t stream);
/// The probe's planes for rounds of up to `max_tokens` columns: zero unless norms are on.
[[nodiscard]] std::size_t moe_stats_probe_workspace_bytes(std::int32_t hidden,
                                                          std::int32_t max_tokens) noexcept;

/// Notes that layer `layer`'s mixture is served from the host expert bank, whose rounds this does
/// not observe (they go through the expert cache); the dump records the layer as unobserved.
void moe_stats_note_unobserved(std::int32_t layer) noexcept;

/// Writes the summed statistics of every device to `DIR/<tag>/` as NumPy arrays (counts.npy,
/// gate_sum.npy, gate_norm_sum.npy, norm_sum.npy: [mixtures, experts]; tokens.npy: [mixtures])
/// and meta.json, then zeroes them when `reset`. An empty tag names the dump by its UTC time.
/// Synchronises every device it reads, so call it while no request runs for exact totals. Returns
/// the directory written. Throws when the statistics are off.
std::string moe_stats_dump(const std::string& tag, bool reset);
/// Zeroes every device's accumulators.
void moe_stats_reset();
/// The tokens each mixture observed so far, as a JSON array (for the admin endpoint's answer).
[[nodiscard]] std::string moe_stats_tokens_json();

// ---------------------------------------------------------------------------------------------
// Residual samples at layer-group boundaries
// ---------------------------------------------------------------------------------------------

struct ResidualSampleOptions {
    /// Where the samples go (`DIR/residual-samples.bin` and `.json`); empty turns them off.
    std::string directory;
    /// Every Nth column of the forwards, counted over all forwards in the order they start: a
    /// forward of T columns started after S columns contributes its columns c with
    /// (S + c) % every == 0.
    std::int32_t every = 64;
    /// Sampling stops once this many columns have been sampled (bounds the file: each sampled
    /// column stores every boundary's residual, FP16). A forward's columns are reserved when it
    /// starts, so forwards in flight together never overshoot it.
    std::int64_t max_tokens = 8192;
    std::string model;
};

void configure_residual_samples(const ResidualSampleOptions& options);
[[nodiscard]] bool residual_samples_enabled() noexcept;

/// One boundary of one forward: `boundary` of `boundaries` (0 is the input of layer 0, b the input
/// of group b's first layer, `boundaries - 1` the final residual the output mixer reads); `layer`
/// the layer whose input it is (the trunk's layer count for the final one). `residual` is the
/// target's residual, [rows, T] BF16 or FP32 (a hyper-connected target's streams stacked), `ids`
/// the forward's token ids ([T] I32 on the device, may be empty: the final boundary has none).
/// Enqueues a copy of the sampled columns (and their ids) to pinned memory on `stream` without
/// waiting for it; the record is appended once the copy has landed, at a later call or a flush.
/// Rounds being captured into a CUDA graph are not sampled: their host code does not run when the
/// graph replays, so only eagerly executed forwards are (prefill under --spec mtp, or everything
/// with --enforce-eager).
///
/// Several forwards may be in flight at once -- a pipeline's first stage starts the next one while
/// later stages still run earlier ones -- as long as each boundary sees the forwards in the order
/// they started, which a pipeline's stages keep. A call at boundary b belongs to the oldest open
/// forward that has passed b-1; one that falls behind a younger forward is dropped.
void residual_sample(std::int32_t boundary, std::int32_t boundaries, std::int32_t layer,
                     const Tensor& residual, const Tensor& ids, cudaStream_t stream);
/// While one is alive, residual_sample ignores this thread's calls: for code that runs a boundary's
/// mix again on residuals a forward already passed (a head reading a stored residual), which is no
/// forward's boundary and would otherwise reach whichever forward waits there.
class ResidualSampleQuiet {
public:
    ResidualSampleQuiet() noexcept;
    ~ResidualSampleQuiet();
    ResidualSampleQuiet(const ResidualSampleQuiet&)            = delete;
    ResidualSampleQuiet& operator=(const ResidualSampleQuiet&) = delete;
};
/// Flushes the sample file and rewrites its JSON summary; returns the summary.
std::string residual_samples_flush();
/// Starts the sample file over, empty, and zeroes its counts (forwards in flight are forgotten,
/// and their later boundaries ignored); returns the new summary. For a clean calibration pass on an
/// engine that already ran startup or test forwards.
std::string residual_samples_reset();

} // namespace sinfer::ops
