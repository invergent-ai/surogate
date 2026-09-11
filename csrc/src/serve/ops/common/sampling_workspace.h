#pragma once

// Private workspace layout shared by token sampling and stochastic speculative acceptance.
// The same builder supplies the public sizing query and launcher-side binding.

#include "core/layout.h"
#include "core/limits.h"
#include "ops/common/math.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops {

inline constexpr int kSamplerSortItems = 4;
inline constexpr int kSamplerSortTile = 256 * kSamplerSortItems;
inline constexpr int kSamplerBlock               = 256;
inline constexpr int kSamplerTileItems           = 256;
inline constexpr int kSamplerItemsPerThread      = 2;
inline constexpr int kSamplerPartialTileItems    = kSamplerBlock * kSamplerItemsPerThread;
inline constexpr int kSamplerGroupBlock          = 256;
inline constexpr int kSamplerGroupItemsPerThread = 2;
inline constexpr int kSamplerGroupTileItems      = kSamplerGroupBlock * kSamplerGroupItemsPerThread;
inline constexpr int kSamplerPartialsPerGroup    = 25;
inline constexpr int kSamplerFastCandidates      = 20;
inline constexpr int kSamplerCandidateCap        = kSamplerFastCandidates;
// Tied to the ops batch bound: a stale cap here does not fail, it silently
// drops decode rounds wider than the cap onto sample_row at ~4 ms a call
// (measured at 15% of device time when the ceiling moved to 64 and this
// stayed at 32). Never hardcode it again.
inline constexpr int kSamplerMaxColumns          = kMaximumBatchColumns;

static_assert(kSamplerPartialsPerGroup * kSamplerCandidateCap <= kSamplerGroupTileItems,
              "group merge tile must hold one group's candidates");

__host__ __device__ inline int sampler_group_count(int partial_blocks) {
    return div_up(partial_blocks, kSamplerPartialsPerGroup);
}

// The multi-block route is deliberately finite. A single final merge tile must
// hold every group candidate; the registered sampling/speculative routes use at
// most kSamplerMaxColumns columns per GPU round.
__host__ __device__ inline bool sampler_multiblock_ok(int vocab, int cols, int partial_blocks,
                                                      int group_count) {
    return vocab > kSamplerTileItems && cols > 0 && cols <= kSamplerMaxColumns &&
           partial_blocks > 0 && group_count > 0 &&
           (group_count * kSamplerCandidateCap) <= kSamplerGroupTileItems;
}

struct SamplingWorkspace {
    unsigned long long* partial_keys         = nullptr;
    std::int32_t* dist_idx                   = nullptr;
    float* dist_prob                         = nullptr;
    std::int32_t* dist_support               = nullptr;
    std::int32_t* group_done                 = nullptr;
    std::int32_t* speculative_finalize_count = nullptr;
    std::int32_t partial_stride              = 0;
    unsigned long long* sort_keys_a = nullptr;
    unsigned long long* sort_keys_b = nullptr;
};

struct SamplingWorkspaceLayout {
    TensorRegion sort_keys_a;
    TensorRegion sort_keys_b;
    TensorRegion partial_keys;
    TensorRegion dist_idx;
    TensorRegion dist_prob;
    TensorRegion dist_support;
    TensorRegion group_done;
    TensorRegion speculative_finalize_count;
    std::size_t bytes           = 0;
    std::int32_t partial_stride = 0;
    bool multiblock             = false;

    [[nodiscard]] SamplingWorkspace bind(DeviceSpan backing) const {
        return {
            multiblock ? static_cast<unsigned long long*>(partial_keys.bind(backing).data) : nullptr,
            multiblock ? static_cast<std::int32_t*>(dist_idx.bind(backing).data) : nullptr,
            multiblock ? static_cast<float*>(dist_prob.bind(backing).data) : nullptr,
            multiblock ? static_cast<std::int32_t*>(dist_support.bind(backing).data) : nullptr,
            multiblock ? static_cast<std::int32_t*>(group_done.bind(backing).data) : nullptr,
            multiblock ? static_cast<std::int32_t*>(speculative_finalize_count.bind(backing).data) : nullptr,
            partial_stride,
            sort_keys_a.region.bytes ? static_cast<unsigned long long*>(sort_keys_a.bind(backing).data) : nullptr,
            sort_keys_b.region.bytes ? static_cast<unsigned long long*>(sort_keys_b.bind(backing).data) : nullptr,
        };
    }
};

inline SamplingWorkspaceLayout make_sampling_workspace_layout(std::int32_t token_domain,
                                                              std::int32_t columns) {
    SamplingWorkspaceLayout out;
    if (token_domain <= 0 || columns <= 0) { return out; }

    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    out.multiblock = sampler_multiblock_ok(token_domain, columns, partial_blocks, groups);
    if (!out.multiblock && token_domain <= kSamplerSortTile) { return out; }
    out.partial_stride = partial_blocks + groups;

    LayoutBuilder layout;
    if (token_domain > kSamplerSortTile) {
        out.sort_keys_a = layout.add_tensor(DType::I64, {token_domain, columns}, 256, "sampling sort keys A");
        out.sort_keys_b = layout.add_tensor(DType::I64, {token_domain, columns}, 256, "sampling sort keys B");
    }
    if (out.multiblock) {
        out.partial_keys =
            layout.add_tensor(DType::I64, {kSamplerCandidateCap, out.partial_stride, columns}, 256,
                              "sampling partial keys");
        out.dist_idx  = layout.add_tensor(DType::I32, {kSamplerCandidateCap, columns}, 256,
                                          "sampling speculative distribution indices");
        out.dist_prob = layout.add_tensor(DType::FP32, {kSamplerCandidateCap, columns}, 256,
                                          "sampling speculative distribution probabilities");
        out.dist_support =
            layout.add_tensor(DType::I32, {columns}, 256, "sampling speculative support sizes");
        out.group_done = layout.add_tensor(DType::I32, {columns}, 256, "sampling group counters");
        out.speculative_finalize_count =
            layout.add_tensor(DType::I32, {1}, 256, "sampling speculative finalize counter");
    }
    // Speculative acceptance replicates this complete layout once per batch row. Preserve the
    // strongest member alignment in the row stride so every replicated partial-key array remains
    // correctly aligned.
    out.bytes = layout.finish(256, "sampling workspace");
    return out;
}

} // namespace sinfer::ops
