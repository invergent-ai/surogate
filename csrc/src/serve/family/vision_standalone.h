#pragma once
// A vision tower loaded on its own, without the text model it was converted beside.
//
// The serving path reaches the tower through a fully loaded model, because that is what
// it has. A trainer has an image and a grid and wants the features -- nothing else about
// the checkpoint -- and used to get them by running a second implementation of the same
// encoder out of transformers. This is the same tower the server runs, bound from the
// same artifact objects, so the two cannot drift.

#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "family/vision_tower.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace sinfer::family {

class StandaloneVisionTower {
public:
    /// Binds and materializes only the tower's objects. The artifact must declare one;
    /// a text-only checkpoint is refused here rather than at the first encode.
    StandaloneVisionTower(const std::filesystem::path& artifact, int device);
    ~StandaloneVisionTower();

    StandaloneVisionTower(StandaloneVisionTower&&) noexcept;
    StandaloneVisionTower& operator=(StandaloneVisionTower&&) noexcept;
    StandaloneVisionTower(const StandaloneVisionTower&)            = delete;
    StandaloneVisionTower& operator=(const StandaloneVisionTower&) = delete;

    [[nodiscard]] const VisionGeometry& geometry() const noexcept;

    /// Merged visual tokens for one image or video item.
    ///
    /// `patches` is the processor's output for this item: `patch_count * patch_dim` BF16
    /// values, patch-major, exactly as the serving frontend hands them over. `output` is
    /// filled with `[output_hidden, merged_tokens, 1 + deepstack_layers]` BF16, the last
    /// axis holding the projection first and then each deepstack merger in tower order.
    void encode(std::span<const std::uint16_t> patches, const VisionGrid& grid,
                PromptModality modality, Tensor& output);

    /// Bytes `output` must have for a grid, so a caller can size its own buffer.
    [[nodiscard]] std::size_t output_bytes(const VisionGrid& grid) const;
    [[nodiscard]] std::size_t merged_tokens(const VisionGrid& grid) const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace sinfer::family
