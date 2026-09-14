#pragma once
#include "family/vision_tower.h"

namespace sinfer::family {
std::size_t muse_vision_workspace_bytes(const VisionGeometry& geometry, std::size_t tokens);
bool encode_muse_vision_step(const VisionGeometry& geometry, const VisionWeights& weights,
                             const VisionItemView& item, Tensor& output, WorkspaceArena& workspace,
                             cudaStream_t stream, const VisionContext::Probe& probe,
                             VisionEncodeState& state);
// Muse stores rotary cos/sin tables in BF16 before applying them in FP32.
void muse_vision_rope(const Tensor& positions, float theta, Tensor& q, Tensor& k,
                      cudaStream_t stream);
// Input columns group each 2x2 neighborhood; output features keep channels outermost.
void muse_pixel_shuffle(const Tensor& input, Tensor& output, cudaStream_t stream);
} // namespace sinfer::family
