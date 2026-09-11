#pragma once
#include "family/vision_tower.h"

namespace sinfer::family {
std::size_t gemma_vision_workspace_bytes(const VisionGeometry& g, std::size_t tokens);
void encode_gemma_vision(const VisionGeometry& g, const VisionWeights& weights,
                         const VisionItemView& item, Tensor& output, WorkspaceArena& workspace,
                         cudaStream_t stream, const VisionContext::Probe& probe);
} // namespace sinfer::family
