#pragma once

// The round a visit-schedule target runs (ENGINE-INKLING lane e1, rl/engine-inkling-2026-11-21/lanes/e1/
// DESIGN-target.md 1.3). A target that declares `visit_schedule` replaces the family's layer loop: every
// TextContext entry (prompt chunk, decode batch, verify batch, mixed round) stages its columns, embeds into
// the residual as usual, and hands this description to `Variant::run_visits`. Nothing here is read by any
// other target. Not instantiated per Variant: the Variant's own header names these types.

#include "api/family/decoder_state.h"
#include "api/ops/gqa_attention.h"
#include "core/arena.h"
#include "core/tensor.h"
#include <api/family/runtime.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace sinfer::family {

/// Whether `Variant` replaces the layer loop with its own visit schedule. False for every target that
/// declares nothing.
template <class Variant>
[[nodiscard]] constexpr bool visit_schedule() {
    if constexpr (requires { Variant::visit_schedule; }) {
        return Variant::visit_schedule;
    } else {
        return false;
    }
}

/// A target's own `--kv-cache-dtype auto` choice: FP8 e4m3 when it declares `kv_storage_fp8_default`.
template <class Variant>
[[nodiscard]] constexpr bool kv_storage_fp8_default() {
    if constexpr (requires { Variant::kv_storage_fp8_default; }) {
        return Variant::kv_storage_fp8_default;
    } else {
        return false;
    }
}

/// Whether the target keeps its logits FP32 after the softcap (`logits_fp32`), as its reference does.
template <class Variant>
[[nodiscard]] constexpr bool logits_fp32() {
    if constexpr (requires { Variant::logits_fp32; }) {
        return Variant::logits_fp32;
    } else {
        return false;
    }
}

/// The dtype of every target-logits plane the runtime allocates and the sampler reads.
template <class Variant>
[[nodiscard]] constexpr DType logits_dtype() {
    return logits_fp32<Variant>() ? DType::FP32 : DType::BF16;
}

/// How a round's conv / recurrent state is kept: updated in place (prefill, decode, a narrow speculative
/// round), or, for a speculative verify, updated in place after a backup with every column's inputs
/// recorded, so the resolve can commit the accepted prefix (`fold`).
enum class VisitStateMode : std::uint8_t {
    InPlace,
    Record,
};

/// One sequence's consecutive columns (a prompt chunk, or one prompt's segment of a mixed round).
struct VisitSegment {
    std::int32_t column       = 0;  ///< first round column
    std::int32_t count        = 0;  ///< columns
    std::int32_t kv_base      = 0;  ///< position of its first column (host; device positions are authoritative)
    std::int32_t kv_table_row = 0;  ///< its row in the text pool's block-table matrix
    /// The row as a device I32 [1] the entry already staged (`io.text_kv_table_row`); used instead of
    /// `kv_table_row` when set.
    Tensor table_row{};
    std::int32_t state_slot   = 0;  ///< its state slot
    /// A rewrite checkpoint: copy the state after this segment to that slot before any later segment on the
    /// same slot advances it (mixed rounds, `mixed_split_segments_supported`); -1 none.
    std::int32_t capture_slot = -1;
    /// A rewrite checkpoint inside the segment: the state before this column goes to `checkpoint_slot`.
    std::int32_t checkpoint_column = -1;
    std::int32_t checkpoint_slot   = -1;
};

/// B lanes of W columns behind the segments: an ordinary decode (W = 1) or a verify (W > 1).
struct VisitBlock {
    std::int32_t column = 0; ///< first round column (column of lane b's column j = column + j + W * b)
    std::int32_t width  = 1;
    std::int32_t batch  = 0;
    Tensor positions;        ///< I32 [W, B]
    Tensor kv_table_rows;    ///< I32 [B]
    Tensor state_slots;      ///< I32 [B]
    Tensor valid_columns;    ///< I32 [B], or empty (all W valid)
    VisitStateMode mode = VisitStateMode::InPlace;
    ops::GqaExecutionEnvelope envelope{};
};

/// Where a Record-mode round keeps its backup and inputs (planned per program; see DESIGN-target.md 2).
struct VisitRecords {
    void* old                = nullptr; ///< BF16 [channels, 3, visits, batch_max]
    void* inputs             = nullptr; ///< BF16 [channels, width_max, visits, batch_max]
    std::int32_t visits      = 0;
    std::int32_t width_max   = 0;
    std::int32_t batch_max   = 0;
};

/// The routed-expert capture of the round (`--capture-routing`): visit v's ids for round column c at
/// `ids + (v * columns + c) * top_k` (routing_rows.h layout), or null.
struct VisitRoutingCapture {
    std::int32_t* ids    = nullptr;
    std::int32_t columns = 0;
    std::int32_t top_k   = 0;
};

/// A target-feature capture (DSpark): `capture(context, feature, value, stream)` with value BF16 [hidden, T].
struct VisitFeatureCapture {
    void* context = nullptr;
    void (*capture)(void* context, int feature, const Tensor& value, cudaStream_t stream) = nullptr;
    [[nodiscard]] bool active() const noexcept { return capture != nullptr; }
};

struct VisitRound {
    TextPhase phase = TextPhase::Prefill;
    Tensor x;         ///< the residual [residual, T]; the target's embed hook has filled what it needs
    Tensor ids;       ///< I32 [T]
    Tensor positions; ///< I32 [T]: absolute position of every round column
    std::span<const VisitSegment> segments;
    const VisitBlock* block = nullptr;
    std::int32_t first_visit = 0; ///< the pipeline stage's visits
    std::int32_t last_visit  = 0;
    const PagedKVCache* text_kv              = nullptr; ///< batch views of the text pool (plane = visit)
    /// A single-sequence prefill's own view (`run_layers` from a prompt chunk): its block table is one row;
    /// segments then address it with row 0. Invalid when the round uses `text_kv`.
    PagedKVCacheView sequence_kv{};
    LinearAttentionStatePool* state          = nullptr;
    VisitRecords records{};
    VisitRoutingCapture routing{};
    VisitFeatureCapture features{};
    std::uint32_t max_visible_keys = 0; ///< host promise: max over rows of the last position + 1
    /// The target's persistent state for this compute stream (V::visit_target_state_bytes() bytes, zeroed once at
    /// startup, never touched by the runtime afterwards); null when the target asks for none.
    void* target_state             = nullptr;
    WorkspaceArena* work           = nullptr;
    cudaStream_t stream            = nullptr;
};

// Dispatch helpers: the Variant's visit leaves through a dependent name, so a runtime instantiated over any
// other target never names them (an `if constexpr` on a concrete Variant would still look them up).

template <class V, class Model>
void run_visits_if(const Model& model, VisitRound& round) {
    if constexpr (visit_schedule<V>()) {
        V::run_visits(model, round);
    } else {
        (void)model;
        (void)round;
    }
}

/// An FP32-logits target's own LM head (projection + softcap into FP32 logits).
template <class V, class Model>
void lm_head_if(const Model& model, const Tensor& hidden, Tensor& logits, cudaStream_t stream) {
    if constexpr (logits_fp32<V>()) {
        V::lm_head(model, hidden, logits, stream);
    } else {
        (void)model; (void)hidden; (void)logits; (void)stream;
    }
}

template <class V>
[[nodiscard]] std::size_t visit_target_state_bytes_if() {
    if constexpr (visit_schedule<V>()) {
        return V::visit_target_state_bytes();
    } else {
        return 0;
    }
}

template <class V>
[[nodiscard]] std::size_t visit_workspace_if(const TextGeometry& geometry, TextPhase phase, std::int32_t batch_size,
                                             std::int32_t max_width, std::int32_t max_columns,
                                             std::uint32_t max_visible_keys, bool lora_enabled) {
    if constexpr (visit_schedule<V>()) {
        return V::visit_workspace_capacity_bytes(geometry, phase, batch_size, max_width, max_columns,
                                                 max_visible_keys, lora_enabled);
    } else {
        (void)geometry; (void)phase; (void)batch_size; (void)max_width; (void)max_columns; (void)max_visible_keys;
        (void)lora_enabled;
        return 0;
    }
}

template <class V>
[[nodiscard]] LinearAttentionStatePoolSpec visit_state_spec_if(const TextGeometry& geometry, std::int32_t slots) {
    if constexpr (visit_schedule<V>()) {
        return V::visit_state_spec(geometry, slots);
    } else {
        (void)geometry;
        (void)slots;
        return {};
    }
}

} // namespace sinfer::family
