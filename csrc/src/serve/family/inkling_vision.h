#pragma once
#include "family/vision_tower.h"
#include "family/inkling_vision_ops.h"

namespace sinfer::family {
// Inkling-2's vision tower (vLLM vllm/models/inkling2/nvidia/vision.py, ops/vision.py): a PE-style ViT with one CLS
// token per image, LayerScale residuals and interleaved 2D RoPE, then a 2 x 2 merge projection (no bias, no norm) into
// the decoder's width. The patches arrive in merge order (frontend/inkling_vision.h); attention is bidirectional over
// one image, so the rows stay in that order throughout and the CLS row rides as the LAST column. Pooling the 2 x 2
// neighbourhoods is then a view of the first 4V columns.
std::size_t inkling_vision_workspace_bytes(const VisionGeometry& geometry, std::size_t tokens);
bool encode_inkling_vision_step(const VisionGeometry& geometry, const VisionWeights& weights,
                                const VisionItemView& item, Tensor& output, WorkspaceArena& workspace,
                                cudaStream_t stream, const VisionContext::Probe& probe,
                                VisionEncodeState& state);

} // namespace sinfer::family
