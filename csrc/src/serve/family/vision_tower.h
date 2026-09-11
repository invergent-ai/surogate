#pragma once
// The vision tower, compiled once.
//
// It used to live inside the per-variant runtime namespace, so every target that
// instantiated the runtime got its own copy of ~180 lines of device orchestration --
// thirteen copies of one encoder, differing in a single line. Nothing in it is
// target-dependent: the weights type is not templated, the geometry arrives as a value,
// and the only variant-specific call was a debug probe, which now arrives as a callback.
//
// Compiling it once is the smaller half of the reason. The larger one is that a caller
// which is not a loaded serving model -- the trainer, encoding images for a sequence it
// is assembling itself -- can now run this tower rather than a second implementation of
// it in another framework.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include <api/family/text_geometry.h>
#include <api/family/vision_control.h>
#include <api/family/vision_geometry.h>
#include <api/family/vision.h>
#include "core/arena.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace sinfer::family {

/// The alignment the tower's workspace and its output transient are carved at.
inline constexpr std::size_t kVisionWorkspaceAlignment = 256;
// Amortize scheduling without running a whole tower. A fixed number of steps
// also keeps pipeline stages at the same encoder boundary on different GPUs.
inline constexpr std::size_t kVisionEncodeStepsPerSlice = 4;

struct VisionItemView {
    std::span<const std::uint16_t> patches;
    const VisionItemControl* control = nullptr;
};

struct VisionEncodeState {
    enum class Phase { Embedding, Blocks, Projection, Complete };
    Phase phase = Phase::Embedding;
    std::size_t layer = 0;
    std::size_t deepstack = 0;
    // Serving keeps this outside the shared scratch arena across decode rounds.
    // A standalone encode can leave it empty and reuse the arena's residual.
    Tensor residual;
};

/// A tower's output must match the text model it was bound with.
[[nodiscard]] inline VisionGeometry
bound_vision_geometry(const VisionGeometry& tower, const TextGeometry& text) {
    if ((!tower.encoder_free && tower.layers <= 0) || tower.output_hidden != text.hidden) {
        throw std::invalid_argument("vision geometry is missing or disagrees with the text width");
    }
    return tower;
}

class VisionContext {
public:
    /// Called after the merger projects, when a variant asks for activation probes.
    /// Empty for a caller that has no variant -- the standalone tower.
    using Probe = std::function<void(const char*, const Tensor&, std::int32_t, cudaStream_t)>;

    VisionContext(DeviceContext& device, const VisionWeights& vision, const VisionGeometry& tower,
                  const TextGeometry& text, Probe probe = {});

    [[nodiscard]] static std::size_t output_transient_bytes(const VisionGeometry& geometry,
                                                            std::size_t merged_tokens);
    [[nodiscard]] static std::size_t encoding_transient_bytes(const VisionGeometry& geometry,
                                                              std::size_t merged_tokens);
    [[nodiscard]] static std::size_t workspace_bytes(const VisionGeometry& geometry,
                                                     const VisionItemControl& item);
    [[nodiscard]] static std::size_t workspace_capacity_bytes(const VisionGeometry& geometry,
                                                              std::uint32_t max_merged_tokens,
                                                              std::uint32_t max_segments);
    /// The tower this context encodes with.
    [[nodiscard]] const VisionGeometry& geometry() const noexcept { return cfg_; }
    void encode(const VisionItemView& item, Tensor& output, WorkspaceArena& workspace) const;
    [[nodiscard]] bool encode_step(const VisionItemView& item, Tensor& output,
                                    WorkspaceArena& workspace, VisionEncodeState& state) const;

private:
    struct BlockW {
        const Tensor* norm1_weight    = nullptr;
        const Tensor* norm1_bias      = nullptr;
        const Weight* qkv             = nullptr;
        const Tensor* qkv_bias        = nullptr;
        const Weight* projection      = nullptr;
        const Tensor* projection_bias = nullptr;
        const Tensor* norm2_weight    = nullptr;
        const Tensor* norm2_bias      = nullptr;
        const Weight* fc1             = nullptr;
        const Tensor* fc1_bias        = nullptr;
        const Weight* fc2             = nullptr;
        const Tensor* fc2_bias        = nullptr;
    };

    struct MergerW {
        const Tensor* norm_weight = nullptr;
        const Tensor* norm_bias   = nullptr;
        const Weight* fc1         = nullptr;
        const Tensor* fc1_bias    = nullptr;
        const Weight* fc2         = nullptr;
        const Tensor* fc2_bias    = nullptr;
    };

    const VisionWeights* weights_ = nullptr;
    DeviceContext& ctx_;
    VisionGeometry cfg_{};
    Probe probe_;
    const Weight* patch_embed_      = nullptr;
    const Tensor* patch_embed_bias_ = nullptr;
    const Tensor* position_embed_   = nullptr;
    const Tensor* post_norm_weight_ = nullptr;
    const Tensor* post_norm_bias_   = nullptr;
    /// Sized from the bound weights rather than by the type: two checkpoints of one family
    /// ship towers of different depths.
    std::vector<BlockW> blocks_;
    MergerW merger_{};
    std::vector<std::pair<std::int32_t, MergerW>> deepstack_;
};

} // namespace sinfer::family
