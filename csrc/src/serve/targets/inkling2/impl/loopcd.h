#pragma once

// LoopCD-Logits for Inkling-2-Mini (arXiv 2610.02185, "Decoding Looped Transformers Better for (Almost) Free"):
// contrast the final prediction with the prediction of the FIRST loop pass, z' = z + w (z - z_ref), before the
// sampler. Off by default; SUROGATE_SERVE_INKLING_LOOPCD turns it on for the whole engine:
//   unset, "", "0", "off"   off (the engine is exactly as without this file)
//   "<w>" or "adaptive:<w>" adaptive strength w = w_max (1 - (p1 - p2)), p1 / p2 the two largest probabilities of
//                           softmax(z) over the token domain (the paper's margin rule), w_max = <w>
//   "fixed:<w>"             constant strength w
//
// How it runs (the target's own schedule, nothing in the family runtime changes):
// - The reference: pass 0's state, i.e. the AttnRes stack [0, 1, 2, 3] after visit 17 (slots 2-3 hold pass 0's
//   blocks until pass 2 overwrites them). Right after visit 17 the coda's six layers run on it as six REFERENCE
//   visits with their own KV planes (the text pool's LAST six planes, after the drafter's: `extra_kv_planes`) and their own conv
//   states (state layers 48..53), exactly the slot pattern of visits 42..47; the output mix (output query + final
//   norm) of that state is written to slot 6, which nothing reads before visit 42. After visit 41 it moves to slot 5
//   (pass 1's second block, dead once pass 2 has read its messages).
// - The hidden that crosses a round boundary is [h ; h_ref] per column (the served geometry's hidden = 2 x 2304):
//   `final_residual_mix` writes both, `lm_head` projects both with the invariant head (one GEMM over 2T columns: a
//   column's bits do not depend on its neighbours, so the main logits are bit-identical to the plain head's) and
//   combines them. Every route to the sampler (prefill tail, decode, DSpark verify, a cached hidden) goes through
//   `lm_head`, so the verify accepts against the contrasted distribution and the output is exactly LoopCD's.
// - With w = 0 the logits are bit-identical to LoopCD off (the strongest test of the plumbing).
// - One card only (a pipeline split refuses), text only (the vision path checks the hidden width).

#include "targets/inkling2/impl/config.h"
#include "targets/inkling2/impl/loopcd_combine.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::targets::inkling2::detail {

struct LoopcdSettings {
    bool enabled   = false;
    bool adaptive  = true;
    float strength = 0.0F; ///< w_max (adaptive) or w (fixed)
};

/// The engine-wide setting, parsed once from SUROGATE_SERVE_INKLING_LOOPCD (throws on a malformed value).
[[nodiscard]] const LoopcdSettings& loopcd_settings();

/// The reference visits: the coda's six layers (visits 42..47's VisitInfo) on pass 0's state.
inline constexpr int kLoopcdVisits     = TextConfig::physical_layers - TextConfig::core_end; // 6
inline constexpr int kLoopcdAfterVisit = 17;  ///< the reference runs right after this visit (pass 0's last)
inline constexpr int kLoopcdMoveVisit  = 41;  ///< after this visit the reference hidden moves slot 6 -> 5
inline constexpr int kLoopcdHiddenSlot = 5;   ///< where finish() finds the reference hidden
/// Conv-state layers with LoopCD on: the 48 visits + the 6 reference visits.
[[nodiscard]] int loopcd_state_layers();

/// Head tokens per pass of the LoopCD head (each token = 2 columns) and its FP32 scratch (one per loaded model: the
/// head runs on the card's one compute stream). 64 = the widest decode / verify round of 8 lanes x (7 drafts + 1) in one
/// pass of the head weights (scratch 201024 x 128 x 4 B = 103 MB); wider rounds (prefill tails of many segments) take more passes.
inline constexpr int kLoopcdHeadTokens = 64;
[[nodiscard]] constexpr std::size_t loopcd_head_scratch_bytes() {
    return static_cast<std::size_t>(TextConfig::output_rows) * 2 * kLoopcdHeadTokens * sizeof(float);
}

} // namespace sinfer::targets::inkling2::detail
