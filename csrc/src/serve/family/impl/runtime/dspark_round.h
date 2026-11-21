#pragma once

// DSpark (Inkling-2-Mini's MTP drafter, ENGINE-INKLING lane e2) through the family's DFlash runtime
// (rl/engine-inkling-2026-11-21/lanes/e2/DESIGN-dspark.md 12). A target that declares `dspark_drafter` keeps the DFlash
// round structure -- feature sink, pending features, context append, one parallel proposal, the shared verify / accept,
// graphs, adaptive k -- but runs its own drafter: the runtime hands the context append and the proposal to the
// Variant's `dspark_append` / `dspark_propose` leaves instead of the DFlash bodies. Its draft KV lives in extra planes
// of the TEXT pool (the Variant's `extra_kv_planes`), addressed by the sequence's own block-table row. Nothing here is
// read by a target that declares none of it: every other instantiation is unchanged.

#include "api/family/decoder_state.h"
#include "api/family/text_geometry.h"
#include "api/family/startup_features.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sinfer::family {

/// Whether `Variant` runs the DSpark drafter through the DFlash runtime. False for every target that declares nothing.
template <class Variant>
[[nodiscard]] constexpr bool dspark_drafter() {
    if constexpr (requires { Variant::dspark_drafter; }) {
        return Variant::dspark_drafter;
    } else {
        return false;
    }
}

/// The windows of the extra text-pool planes a target appends after its own (0 = whole context), or empty.
template <class Variant>
[[nodiscard]] std::vector<std::int32_t> variant_extra_kv_planes(const TextGeometry& g, const StartupFeatures& f) {
    if constexpr (requires { Variant::extra_kv_planes(g, f); }) {
        return Variant::extra_kv_planes(g, f);
    } else {
        (void)g;
        (void)f;
        return {};
    }
}

/// A lane's draft acceptance by content kind, for the opt-in DSpark k policy "lane" (SUROGATE_SERVE_DSPARK_K_POLICY=lane;
/// the default is "static"; not carried across a KV preemption yet -- KvResumeState) (agent-EN-e2c): decayed
/// counts of accepted drafts and of rounds stopped by a rejection, per kind, from this request's own rounds only (reset
/// at every request start), so a lane's k depends on nothing but its own tokens -- batch- and timing-independent, and a
/// sampled rollout stays reproducible under its per-(request, position) seeds. Read by the Variant's
/// dspark_draft_cap; updated by the runtime after each round (observe).
struct DSparkLaneStats {
    static constexpr int kKinds = 4; ///< the Variant's content kinds (inkling2: tool JSON, structure, thinking, text)
    float accepted[kKinds] = {};     ///< decayed accepted drafts per kind
    float stopped[kKinds]  = {};     ///< decayed rounds that ended on a rejected draft per kind
    std::uint8_t kind      = 0;      ///< the kind of the round in flight
    std::uint32_t extent   = 0;      ///< the drafts verified in the round in flight
    bool noted             = false;  ///< a DSpark round of this lane was noted and not yet observed
    /// The round's outcome: `committed` tokens (accepted drafts + the correction / bonus). The kind's counts decay by
    /// `decay` every round of that kind, also at extent 0, so an estimate stuck low relaxes to the prior and re-probes.
    /// Only a round noted at launch counts (nothing else ever changes the counts).
    void observe(std::uint32_t committed, float decay) {
        if (!noted) { return; }
        noted = false;
        const int k = kind < kKinds ? kind : 0;
        accepted[k] *= decay;
        stopped[k] *= decay;
        if (extent == 0 || committed == 0) { return; }
        const std::uint32_t a = committed - 1 < extent ? committed - 1 : extent;
        accepted[k] += static_cast<float>(a);
        if (a < extent) { stopped[k] += 1.0f; }
    }
};

/// What the runtime gives a DSpark leaf: the text pool (draft planes included), the per-lane f_rnn state and the
/// round's workspace / stream.
struct DSparkRuntime {
    const PagedKVCache* text_kv = nullptr;
    Tensor f_rnn;                 ///< BF16 [hidden, lane capacity]: each lane's Markov feature (last committed column)
    WorkspaceArena* work = nullptr;
    cudaStream_t stream  = nullptr;
};

/// Context append of the committed columns (prefill chunk: B = 1, every column committed; decode: <= k + 1 each).
struct DSparkAppendArgs {
    Tensor features;      ///< BF16 [feature_rows, W, B]
    Tensor positions;     ///< I32 [W, B]
    Tensor counts;        ///< I32 [B] committed columns per lane
    Tensor lanes;         ///< I32 [B]
    Tensor table_rows;    ///< I32 [B] the sequences' TEXT-pool block-table rows
    bool all_committed = false; ///< every lane commits all W columns (host knowledge: a prefill chunk)
};

/// One proposal round: lane b's anchor token at anchor_positions[b] -> k drafts.
struct DSparkProposeArgs {
    Tensor anchors;          ///< I32 [B]
    Tensor anchor_positions; ///< I32 [B]
    Tensor lanes;            ///< I32 [B]
    Tensor table_rows;       ///< I32 [B] TEXT-pool rows
    Tensor drafts;           ///< I32 [k, B]
    std::int32_t batch = 0;
    std::int32_t k     = 0;
};

// Dispatch helpers: the Variant's DSpark leaves through a dependent name, so a runtime instantiated over any other
// target never names them (an `if constexpr` on a concrete Variant would still look them up).

template <class V>
[[nodiscard]] std::size_t dspark_append_workspace_bytes_if(std::int32_t columns) {
    if constexpr (dspark_drafter<V>()) {
        return V::dspark_append_workspace_bytes(columns);
    } else {
        (void)columns;
        return 0;
    }
}

template <class V>
[[nodiscard]] std::size_t dspark_propose_workspace_bytes_if(std::int32_t batch, std::uint32_t max_visible_keys) {
    if constexpr (dspark_drafter<V>()) {
        return V::dspark_propose_workspace_bytes(batch, max_visible_keys);
    } else {
        (void)batch;
        (void)max_visible_keys;
        return 0;
    }
}

template <class V, class Model>
void dspark_append_if(const Model& model, const DSparkRuntime& runtime, const DSparkAppendArgs& args) {
    if constexpr (dspark_drafter<V>()) {
        V::dspark_append(model, runtime, args);
    } else {
        (void)model; (void)runtime; (void)args;
    }
}

template <class V, class Model>
void dspark_propose_if(const Model& model, const DSparkRuntime& runtime, const DSparkProposeArgs& args) {
    if constexpr (dspark_drafter<V>()) {
        V::dspark_propose(model, runtime, args);
    } else {
        (void)model; (void)runtime; (void)args;
    }
}

} // namespace sinfer::family
