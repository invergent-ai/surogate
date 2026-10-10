#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace sinfer {

/// Whether a speculative round's GDN fold may wait for the lane's next verify
/// (GdnReplayRecordSpec::pending_slots). SUROGATE_SERVE_DEFER_GDN_FOLD=0 folds every round.
[[nodiscard]] inline bool deferred_gdn_fold_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("SUROGATE_SERVE_DEFER_GDN_FOLD");
        return value == nullptr || std::string_view(value) != "0";
    }();
    return enabled;
}

struct GdnReplayRecordSpec {
    std::int32_t layers          = 0;
    std::int32_t record_capacity = 0;
    std::int32_t width           = 0;
    std::int32_t conv_channels   = 0;
    std::int32_t qk_heads        = 0;
    std::int32_t value_heads     = 0;
    std::int32_t key_dim         = 0;
    std::int32_t value_dim       = 0;
    /// Whether the forget gate is one value per key channel (Kimi Delta Attention) rather than
    /// one per value head (the gated delta net). The gate plane is then `[key_dim, value_heads,
    /// width, outer]` -- laid out like the value, so a replay reads the channels it owns -- and
    /// beta, which stays one per head, has a plane of its own beside it.
    bool diagonal_gate           = false;
    /// Lanes (current-state slots [0, pending_slots)) that can carry a deferred fold; zero plans
    /// none. A deferred fold keeps a round's accepted transitions per lane until that lane's next
    /// verify reads its state anyway, which then applies them before verifying -- so a round
    /// reads the recurrent state once instead of twice (the verify, then the fold). Scalar gate
    /// only.
    std::int32_t pending_slots   = 0;

    /// Rows of the gate plane: {g, beta} pairs for a scalar gate, the key channels of g for a
    /// diagonal one.
    [[nodiscard]] constexpr std::int32_t gate_rows() const noexcept {
        return diagonal_gate ? key_dim : 2;
    }
};

struct GdnReplayRecordLayout {
    GdnReplayRecordSpec spec;
    TensorRegion conv;
    TensorRegion key;
    TensorRegion value;
    TensorRegion gate;
    /// Present only for a diagonal gate; a scalar gate's beta rides in `gate`.
    TensorRegion beta;
    /// Present only with `spec.pending_slots`: the deferred folds' transitions, laid out like
    /// the records with the lane's slot in place of the round's row, and their column counts.
    TensorRegion pending_key;
    TensorRegion pending_value;
    TensorRegion pending_gate;
    TensorRegion pending_columns;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

[[nodiscard]] GdnReplayRecordLayout plan_gdn_replay_records(LayoutBuilder& builder,
                                                            const GdnReplayRecordSpec& spec);

/// One layer's deferred folds, read by that layer's verify. Empty when none are planned.
struct GdnPendingFoldLayer {
    Tensor key;     // BF16 [key_dim, qk_heads, width, pending_slots]
    Tensor value;   // BF16 [value_dim, value_heads, width, pending_slots]
    Tensor gate;    // FP32 [2, value_heads, width, pending_slots]: {g, beta}
    Tensor columns; // I32 [pending_slots]: transitions pending per lane, for every layer

    [[nodiscard]] bool present() const noexcept { return columns.data != nullptr; }
};

struct GdnReplayRecordLayer {
    Tensor conv;  // BF16 [conv_channels, width, rows]
    Tensor key;   // BF16 [key_dim, qk_heads, width, rows]
    Tensor value; // BF16 [value_dim, value_heads, width, rows]
    Tensor gate;  // FP32 [gate_rows, value_heads, width, rows]: {g, beta}, or g per channel
    Tensor beta;  // FP32 [value_heads, width, rows] for a diagonal gate; empty otherwise
    GdnPendingFoldLayer pending;
};

/**
 * Bound non-owning all-layer ReplaySSM transition records.
 *
 * Layer and physical record row share the outer index `layer * record_capacity + row` in every
 * plane. The object owns no allocation and stores no active-row or valid-prefix metadata.
 */
struct GdnReplayRecords {
    Tensor conv;
    Tensor key;
    Tensor value;
    Tensor gate;
    Tensor beta;
    /// Deferred folds (spec.pending_slots > 0): key/value/gate planes with outer index
    /// `layer * pending_slots + slot` at the planned width, and I32 [pending_slots] counts.
    Tensor pending_key;
    Tensor pending_value;
    Tensor pending_gate;
    Tensor pending_columns;
    GdnReplayRecordSpec spec;

    [[nodiscard]] bool defers_fold() const noexcept { return pending_columns.data != nullptr; }

    GdnReplayRecords() = default;
    GdnReplayRecords(DeviceSpan backing, const GdnReplayRecordLayout& layout);

    [[nodiscard]] GdnReplayRecords with_width(std::int32_t width) const;
    [[nodiscard]] GdnReplayRecordLayer layer(std::int32_t layer, std::int32_t rows) const;
};

} // namespace sinfer
