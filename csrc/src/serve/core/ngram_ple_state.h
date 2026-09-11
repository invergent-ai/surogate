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
};

struct NgramPleStatePoolLayout {
    NgramPleStatePoolSpec spec;
    LayoutRegion history;
    LayoutRegion conv;
};

[[nodiscard]] NgramPleStatePoolLayout plan_ngram_ple_state_pool(LayoutBuilder& builder,
                                                                const NgramPleStatePoolSpec& spec);

struct NgramPleStatePool {
    Tensor history;    ///< I32 [history_tokens, slots]
    Tensor conv_state; ///< BF16 [conv_history, channels, slots]
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
};

} // namespace sinfer
