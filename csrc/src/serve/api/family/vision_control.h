#pragma once

#include <api/family/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sinfer::family {

struct VisionItemControl {
    PromptModality modality = PromptModality::Image;
    VisionGrid grid;
    std::size_t patch_begin     = 0;
    std::size_t patch_count     = 0;
    std::size_t merged_count    = 0;
    std::int32_t segment_length = 0;
    std::int32_t segment_count  = 0;
    std::vector<std::int32_t> position_ids;
    std::vector<std::int32_t> cu_seqlens;
    std::vector<std::int32_t> scatter_indices;
    std::vector<std::int32_t> position_table_indices;
    std::vector<float> position_table_weights;
};

struct VisionControl {
    std::vector<VisionItemControl> items;
};

/// The tower's view of one item, from its grid and the encoder's position-table size. `scatter_indices` is left empty:
/// it says where the merged tokens land in a text sequence, which the tower never reads.
[[nodiscard]] VisionItemControl build_vision_item_control(const VisionGrid& grid,
                                                          PromptModality modality,
                                                          std::int32_t position_embeddings = 48 * 48,
                                                          std::int32_t merge = 2, bool factorized = false);

[[nodiscard]] VisionControl build_vision_control(const PreparedPromptData& prompt,
                                                  std::int32_t position_embeddings = 48 * 48,
                                                          std::int32_t merge = 2, bool factorized = false);

} // namespace sinfer::family
