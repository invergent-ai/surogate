#include "ninfer/ops/vision_attention.h"

#include "core/layout.h"
#include "ops/launcher/vision_attention.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 72;
constexpr std::int32_t kHeads   = 16;

std::int32_t scratch_tiles(std::int32_t patches, std::int32_t segments) {
    if (segments == 1) { return 0; }
    constexpr std::int64_t tile_rows = 64;
    const std::int64_t tiles =
        (static_cast<std::int64_t>(patches) + tile_rows - 1) / tile_rows + segments - 1LL;
    if (tiles > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("vision_attention: descriptor tile count exceeds int32");
    }
    return static_cast<std::int32_t>(tiles);
}

template <class Allocator>
Tensor allocate_workspace(Allocator& allocator, std::int32_t patches, std::int32_t segments) {
    const std::int32_t tiles = scratch_tiles(patches, segments);
    return tiles == 0 ? Tensor{} : allocator.alloc(DType::I32, {4, tiles});
}

void require_qkv(const Tensor& tensor, std::int32_t patches, const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != kHeadDim || tensor.ne[1] != kHeads ||
        tensor.ne[2] != patches || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("vision_attention: invalid ") + name + " shape");
    }
    constexpr std::int64_t elem = 2;
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * kHeadDim ||
        tensor.nb[2] < elem * kHeadDim * kHeads || (tensor.nb[2] % elem) != 0) {
        throw std::invalid_argument(std::string("vision_attention: invalid ") + name + " strides");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("vision_attention: ") + name +
                                    " data must be non-null");
    }
}

} // namespace

std::size_t vision_attention_workspace_capacity_bytes(std::int32_t min_patches,
                                                      std::int32_t max_patches,
                                                      std::int32_t min_segments,
                                                      std::int32_t max_segments) {
    if (min_patches <= 0 || max_patches < min_patches || min_segments <= 0 ||
        max_segments < min_segments || min_segments > max_patches) {
        throw std::invalid_argument("vision_attention workspace: invalid execution envelope");
    }
    const std::int32_t segments = std::min(max_segments, max_patches);
    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, max_patches, segments);
    return layout.peak_bytes(1);
}

void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens,
                      WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    const std::int32_t patches  = q.ne[2];
    const std::int32_t segments = cu_seqlens.ne[0] - 1;
    if (patches <= 0) { throw std::invalid_argument("vision_attention: P must be positive"); }
    require_qkv(q, patches, "q");
    require_qkv(k, patches, "k");
    require_qkv(v, patches, "v");
    require_qkv(out, patches, "out");
    if (!out.is_contiguous()) {
        throw std::invalid_argument("vision_attention: out must be contiguous");
    }
    if (cu_seqlens.dtype != DType::I32 || segments <= 0 || segments > patches ||
        cu_seqlens.ne[1] != 1 || cu_seqlens.ne[2] != 1 || cu_seqlens.ne[3] != 1 ||
        !cu_seqlens.is_contiguous() || cu_seqlens.data == nullptr) {
        throw std::invalid_argument("vision_attention: cu_seqlens must be contiguous I32 [S+1]");
    }
    auto scratch_scope = workspace.scope();
    Tensor tiles       = allocate_workspace(workspace, patches, segments);
    Tensor* tiles_ptr  = tiles.data == nullptr ? nullptr : &tiles;
    detail::vision_attention_launch(q, k, v, cu_seqlens, tiles_ptr, out, stream);
}

void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                      std::int32_t segment_length, Tensor& out, cudaStream_t stream) {
    const std::int32_t patches = q.ne[2];
    if (patches <= 0) { throw std::invalid_argument("vision_attention: P must be positive"); }
    require_qkv(q, patches, "q");
    require_qkv(k, patches, "k");
    require_qkv(v, patches, "v");
    require_qkv(out, patches, "out");
    if (!out.is_contiguous()) {
        throw std::invalid_argument("vision_attention: out must be contiguous");
    }
    if (segment_length <= 0 || patches % segment_length != 0) {
        throw std::invalid_argument("vision_attention: invalid uniform segment length");
    }
    detail::vision_attention_uniform_launch(q, k, v, segment_length, out, stream);
}

} // namespace ninfer::ops
