#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace sinfer {

/**
 * Per-slot state of an n-gram PLE layer: the token history the hash needs for the first
 * columns of a segment and the convolution history. Slots follow the linear-attention pool's
 * slot roles; the family resets and copies both pools at the same lifecycle points.
 */
struct NgramPleStatePoolSpec {
    std::int32_t history_tokens = 0; ///< ngram - 1
    std::int32_t conv_history   = 0; ///< (kernel - 1) * dilation
    std::int32_t channels       = 0; ///< streams * hidden
    std::int32_t slot_count     = 1;
    std::int32_t eos_token      = 0; ///< the reset value of the token history
    /// Columns of a speculative round whose state each slot keeps (0: none). A verify round
    /// writes the state after every column of its segment here, so the columns the round
    /// commits can be restored when it rejects the rest.
    std::int32_t snapshot_width = 0;
};

struct NgramPleStatePoolLayout {
    NgramPleStatePoolSpec spec;
    LayoutRegion history;
    LayoutRegion conv;
    LayoutRegion history_snapshots;
    LayoutRegion conv_snapshots;
};

[[nodiscard]] NgramPleStatePoolLayout plan_ngram_ple_state_pool(LayoutBuilder& builder,
                                                                const NgramPleStatePoolSpec& spec);

struct NgramPleStatePool {
    Tensor history;    ///< I32 [history_tokens, slots]
    Tensor conv_state; ///< BF16 [conv_history, channels, slots]
    /// The state after each of a segment's first `snapshot_width` columns, laid out per
    /// (slot, column) as one slot of `history` and `conv_state`: I32
    /// [history_tokens, snapshot_width, slots] and BF16 [conv_history, channels,
    /// snapshot_width, slots]. Empty when the spec keeps none.
    Tensor history_snapshots;
    Tensor conv_snapshots;
    NgramPleStatePoolSpec spec;
    std::vector<const NgramPleStatePool*> checkpoint_slots;

    NgramPleStatePool() = default;
    NgramPleStatePool(DeviceSpan backing, const NgramPleStatePoolLayout& layout);

    [[nodiscard]] bool empty() const noexcept { return history.data == nullptr; }
    [[nodiscard]] std::int32_t slot_count() const noexcept {
        return spec.slot_count + static_cast<std::int32_t>(checkpoint_slots.size());
    }
    [[nodiscard]] Tensor history_slot(std::int32_t slot) const;
    [[nodiscard]] Tensor conv_slot(std::int32_t slot) const;

    void copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream = nullptr);
    /// Token history becomes EOS, the convolution history zero.
    void reset_slot(std::int32_t slot, cudaStream_t stream = nullptr);
    /// The slot's state becomes the one its last segment left after `column` (0-based): what a
    /// speculative round that committed `column + 1` of its columns should have written.
    void commit_snapshot(std::int32_t slot, std::int32_t column, cudaStream_t stream = nullptr);
};

} // namespace sinfer
