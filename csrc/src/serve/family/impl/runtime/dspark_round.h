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
