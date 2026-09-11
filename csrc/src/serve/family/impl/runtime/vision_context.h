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
#include "family/vision_tower.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule {

using family::VisionContext;
using family::VisionItemView;
using family::bound_vision_geometry;
inline constexpr std::size_t kWorkspaceAlignment = family::kVisionWorkspaceAlignment;

/// The tower itself is compiled once (family/vision_tower.h); this is the per-variant
/// convenience that reads it out of a loaded model and carries the variant's debug probe.
[[nodiscard]] family::VisionContext vision_context_for(DeviceContext& device,
                                                       const LoadedModelData& model);

struct VisionChunk {
    std::int32_t length                       = 0;
    const family::VisionItemControl* control = nullptr;
    Tensor embeddings;
    Tensor deepstack;
};

class VisionPrefillSession {
public:
    VisionPrefillSession(DeviceContext& device, const LoadedModelData& model,
                         WorkspaceArena& workspace, family::PreparedPromptData& prompt,
                         const VisionPrefillPlan& plan, runtime::TransientRegion transient);

    [[nodiscard]] VisionChunk prepare_chunk(std::uint32_t begin, std::uint32_t nominal_length);
    [[nodiscard]] std::uint32_t chunk_length(std::uint32_t begin, std::uint32_t nominal_length) const;
    void release_encoded_media_payloads() noexcept;
    [[nodiscard]] double elapsed_seconds() const;

private:
    DeviceContext& device_;
    WorkspaceArena& workspace_;
    family::PreparedPromptData& prompt_;
    const VisionPrefillPlan& plan_;
    runtime::TransientRegion transient_;
    family::VisionContext context_;
    std::optional<std::uint32_t> active_item_;
    std::vector<std::uint32_t> encoded_payloads_pending_release_;
    std::vector<CudaEventTimer> timers_;
};

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule
