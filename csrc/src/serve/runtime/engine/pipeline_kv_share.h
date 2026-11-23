#pragma once

// Window-bounded KV: the KV pool a pipeline's stages share under `--kv-capacity auto`.
//
// Every stage resolves its own capacity curve to a page count. With an elastic window class a stage's
// page ledger also books each lane's window pages as main-page equivalents -- its window charge -- and
// that charge differs between stages: it follows the stage's own window planes over its own
// full-context planes (Inkling with DSpark: 175 pages on the prelude / core stage, another figure on the
// stage that carries the drafter's planes). Sharing the smallest stage's page count therefore shared
// pages, not tokens: a stage with a larger charge got fewer usable tokens than the others, or was
// clamped up to its own minimum and held more than the pool the others shared.
//
// What the stages share is real KV: the pages left when every lane holds its charge. With R_s the
// stage's resolved pages, C_s its per-lane charge and L the lanes,
//   shared = min over stages of (R_s - L * C_s),          stage pages P_s = shared + L * C_s,
// clamped to [the stage's curve minimum, R_s]. P_s never exceeds R_s (shared <= R_s - L * C_s), so each
// stage still fits what it resolved; the minimum clamp only engages when a tiny pool would leave a stage
// unable to hold one full-context request, and then it is still within that stage's own resolution.
// Without a window class every charge is 0 and this is the old rule: every stage gets min R_s.

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace sinfer::runtime {

struct PipelineStageKvFit {
    std::uint32_t resolved_pages = 0; ///< the stage's own resolution of its curve
    std::uint32_t minimum_pages  = 0; ///< the curve's minimum (one full-context request + its charge)
    std::uint32_t charge_pages   = 0; ///< per-lane window charge in this stage's main pages (0 without a class)
    std::uint32_t page_tokens    = 0; ///< tokens per main page of this stage's curve (review R5 minor 12)
};

struct PipelineKvShare {
    std::uint32_t shared_pages = 0;        ///< real KV pages every stage holds with every lane charged
    std::uint32_t page_tokens  = 0;        ///< tokens per page, the same on every stage
    std::vector<std::uint32_t> stage_pages; ///< each stage's page count: shared + lanes x its charge
};

/// The KV tokens a stage counts against what a run needs when `--host-moe-layers auto` decides whether to offload.
/// Under auto, real KV: a window class's per-lane charge holds no tokens of the context. Under an explicit capacity, the
/// capacity itself: it never grows with memory, so measuring it net of the charge would stay short forever and offload
/// every mixture layer of the stage (review R5 minor 11 c).
[[nodiscard]] inline std::uint64_t pipeline_stage_held_tokens(std::uint32_t kv_tokens, const PipelineStageKvFit& fit,
                                                              std::uint32_t lanes, bool explicit_capacity) noexcept {
    if (explicit_capacity) { return kv_tokens; }
    const std::uint64_t charged = static_cast<std::uint64_t>(lanes) * fit.charge_pages * fit.page_tokens;
    return kv_tokens > charged ? kv_tokens - charged : 0;
}

[[nodiscard]] inline PipelineKvShare share_pipeline_kv(std::span<const PipelineStageKvFit> stages,
                                                       std::uint32_t lanes) {
    if (stages.empty() || lanes == 0) { throw std::invalid_argument("pipeline KV share needs stages and lanes"); }
    PipelineKvShare out;
    out.page_tokens = stages.front().page_tokens;
    bool first = true;
    for (const PipelineStageKvFit& stage : stages) {
        if (stage.page_tokens == 0 || stage.page_tokens != out.page_tokens) {
            throw std::invalid_argument("pipeline stages disagree on the KV page size");
        }
        const std::uint64_t charged = static_cast<std::uint64_t>(lanes) * stage.charge_pages;
        const std::uint64_t real    = stage.resolved_pages > charged ? stage.resolved_pages - charged : 0;
        out.shared_pages = first ? static_cast<std::uint32_t>(real)
                                 : std::min(out.shared_pages, static_cast<std::uint32_t>(real));
        first = false;
    }
    out.stage_pages.reserve(stages.size());
    for (const PipelineStageKvFit& stage : stages) {
        const std::uint64_t wanted =
            static_cast<std::uint64_t>(out.shared_pages) + static_cast<std::uint64_t>(lanes) * stage.charge_pages;
        out.stage_pages.push_back(static_cast<std::uint32_t>(std::min<std::uint64_t>(
            stage.resolved_pages, std::max<std::uint64_t>(stage.minimum_pages, wanted))));
    }
    return out;
}

} // namespace sinfer::runtime
