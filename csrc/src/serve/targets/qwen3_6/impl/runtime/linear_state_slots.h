#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

/** Qwen3.6's target-local mapping from a stable request lane to its two state roles. */
inline constexpr std::int32_t kNoRewriteCheckpointSlot = -1;

struct LinearStateSlots {
    // Layout: [0, N) the lanes' current state, N the shared prefill-graph
    // scratch slot (PATCHES.md #27), then — only when rewrite checkpoints are
    // enabled — [N+1, 2N+1) one checkpoint per lane. Checkpoints come last so
    // that disabling them truncates the pool: at 72 MiB per slot per lane on
    // the 27B they are what keeps the lane count off the memory cliff, and a
    // workload without prefix reuse never touches them. Any path that reaches
    // for a checkpoint slot while they are disabled indexes past the pool and
    // fails loudly in validate_layer_slot rather than silently aliasing.
    [[nodiscard]] static std::int32_t state_slot_count(std::uint32_t max_concurrency,
                                                       bool rewrite_checkpoints) {
        if (max_concurrency == 0 ||
            max_concurrency >
                static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max() / 2)) {
            throw std::invalid_argument("Qwen3.6 Linear Attention concurrency is invalid");
        }
        return static_cast<std::int32_t>(rewrite_checkpoints ? 2U * max_concurrency + 1U
                                                             : max_concurrency + 1U);
    }

    [[nodiscard]] static std::int32_t prefill_scratch_state_slot(std::uint32_t max_concurrency) {
        return static_cast<std::int32_t>(max_concurrency);
    }

    [[nodiscard]] static std::int32_t current_state_slot(std::uint32_t lane,
                                                         std::uint32_t max_concurrency) {
        if (lane >= max_concurrency) {
            throw std::out_of_range("Qwen3.6 Linear Attention lane is out of range");
        }
        return static_cast<std::int32_t>(lane);
    }

    [[nodiscard]] static std::int32_t rewrite_checkpoint_state_slot(std::uint32_t lane,
                                                                    std::uint32_t max_concurrency) {
        return static_cast<std::int32_t>(max_concurrency) + 1 +
               current_state_slot(lane, max_concurrency);
    }
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
