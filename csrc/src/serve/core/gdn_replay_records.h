#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace sinfer {

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

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

[[nodiscard]] GdnReplayRecordLayout plan_gdn_replay_records(LayoutBuilder& builder,
                                                            const GdnReplayRecordSpec& spec);

struct GdnReplayRecordLayer {
    Tensor conv;  // BF16 [conv_channels, width, rows]
    Tensor key;   // BF16 [key_dim, qk_heads, width, rows]
    Tensor value; // BF16 [value_dim, value_heads, width, rows]
    Tensor gate;  // FP32 [gate_rows, value_heads, width, rows]: {g, beta}, or g per channel
    Tensor beta;  // FP32 [value_heads, width, rows] for a diagonal gate; empty otherwise
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
    GdnReplayRecordSpec spec;

    GdnReplayRecords() = default;
    GdnReplayRecords(DeviceSpan backing, const GdnReplayRecordLayout& layout);

    [[nodiscard]] GdnReplayRecordLayer layer(std::int32_t layer, std::int32_t rows) const;
};

} // namespace sinfer
