#pragma once
#include "family/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include <api/family/text_geometry.h>
#include <api/family/vision_control.h>
#include <api/family/vision_geometry.h>
#include "runtime/contract/transient_region.h"
#include "family/impl/runtime/vision_prefill.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule {

struct VisionItemView {
    std::span<const std::uint16_t> patches;
    const family::VisionItemControl* control = nullptr;
};

/// The tower this target compiles, as a value. The schedule reads every dimension from a
/// `VisionGeometry` it is handed rather than from `VisionConfig`, so a context sizes its buffers
/// from the tower the weights carry; this is the fallback for a caller that has no weights yet.
[[nodiscard]] inline family::VisionGeometry compiled_vision_geometry() {
    return family::VisionGeometry::compiled<VisionConfig>();
}

/// A tower as the schedule must run it. `output_hidden` is the width the merger projects into,
/// and that is the text model's hidden state -- merged visual tokens join the residual stream --
/// so it is taken from the text geometry rather than from the tower's own config. Only the text
/// side knows that width at the checkpoint's size, and the family's shared tower config cannot
/// name it at all; leaving it compiled while the text hidden moved is a mismatch the merger's
/// last GEMM finds, and only there.
[[nodiscard]] inline family::VisionGeometry
bound_vision_geometry(const family::VisionGeometry& tower, const family::TextGeometry& text) {
    family::VisionGeometry geometry = tower;
    geometry.output_hidden          = text.hidden;
    return geometry;
}

class VisionContext {
public:
    VisionContext(DeviceContext& device, const LoadedModelData& model);

    [[nodiscard]] static std::size_t output_transient_bytes(const family::VisionGeometry& geometry,
                                                            std::size_t merged_tokens);
    [[nodiscard]] static std::size_t workspace_bytes(const family::VisionGeometry& geometry,
                                                     const family::VisionItemControl& item);
    [[nodiscard]] static std::size_t workspace_capacity_bytes(const family::VisionGeometry& geometry,
                                                              std::uint32_t max_merged_tokens,
                                                              std::uint32_t max_segments);
    /// The tower this context encodes with.
    [[nodiscard]] const family::VisionGeometry& geometry() const noexcept { return cfg_; }
    void encode(const VisionItemView& item, Tensor& output, WorkspaceArena& workspace) const;

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

    DeviceContext& ctx_;
    family::VisionGeometry cfg_{};
    const Weight* patch_embed_      = nullptr;
    const Tensor* patch_embed_bias_ = nullptr;
    const Tensor* position_embed_   = nullptr;
    /// Sized from the bound weights rather than by the type: two checkpoints of one family
    /// ship towers of different depths.
    std::vector<BlockW> blocks_;
    MergerW merger_{};
};

struct VisionChunk {
    std::int32_t length                       = 0;
    const family::VisionItemControl* control = nullptr;
    Tensor embeddings;
};

class VisionPrefillSession {
public:
    VisionPrefillSession(DeviceContext& device, const LoadedModelData& model,
                         WorkspaceArena& workspace, family::PreparedPromptData& prompt,
                         const VisionPrefillPlan& plan, runtime::TransientRegion transient);

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    void release_encoded_media_payloads() noexcept;
    [[nodiscard]] double elapsed_seconds() const;

private:
    DeviceContext& device_;
    WorkspaceArena& workspace_;
    family::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    runtime::TransientRegion transient_;
    VisionContext context_;
    std::optional<std::uint32_t> active_item_;
    std::vector<std::uint32_t> encoded_payloads_pending_release_;
    std::vector<CudaEventTimer> timers_;
};

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule
