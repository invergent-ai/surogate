#include "api/ops/vision_attention.h"

#include "core/layout.h"
#include "ops/launcher/vision_attention.h"
#include "ops/launcher/vision_attention_fp8.h"
#include <cmath>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

// Either tower: 72 (Qwen3.6 family, Flash-Next) or 64 (Qwen3.5).
constexpr std::int32_t kHeadDim   = 72;
constexpr std::int32_t kHeadDim64 = 64;

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
    const std::int32_t head_dim = static_cast<std::int32_t>(tensor.ne[0]);
    if (tensor.dtype != DType::BF16 || (head_dim != kHeadDim && head_dim != kHeadDim64 && head_dim != 96) ||
        tensor.ne[1] <= 0 || tensor.ne[2] != patches || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("vision_attention: invalid ") + name + " shape");
    }
    constexpr std::int64_t elem = 2;
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * head_dim ||
        tensor.nb[2] < elem * head_dim * tensor.ne[1] || (tensor.nb[2] % elem) != 0) {
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
                      WorkspaceArena& workspace, Tensor& out, cudaStream_t stream, float scale) {
    if (!std::isfinite(scale) || scale < 0) { throw std::invalid_argument("vision_attention: invalid scale"); }
    const std::int32_t patches  = q.ne[2];
    const std::int32_t segments = cu_seqlens.ne[0] - 1;
    if (patches <= 0) { throw std::invalid_argument("vision_attention: P must be positive"); }
    require_qkv(q, patches, "q");
    require_qkv(k, patches, "k");
    require_qkv(v, patches, "v");
    require_qkv(out, patches, "out");
    if (!std::equal(q.ne, q.ne + 4, k.ne) || !std::equal(q.ne, q.ne + 4, v.ne) ||
        !std::equal(q.ne, q.ne + 4, out.ne)) {
        throw std::invalid_argument("vision_attention: q/k/v/out shapes must match");
    }
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
    detail::vision_attention_launch(q, k, v, cu_seqlens, tiles_ptr, out, stream, scale);
}

std::size_t vision_attention_fp8_scratch_bytes(std::int32_t head_dim, std::int32_t heads,
                                               std::int32_t patches) {
    return detail::vision_attention_fp8_scratch_bytes(head_dim, heads, patches);
}

namespace {

bool run_fp8(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens,
             WorkspaceArena& workspace, DeviceSpan scratch, Tensor& out, cudaStream_t stream,
             float scale, const Tensor* rope_positions) {
    if (!std::isfinite(scale) || scale < 0) { throw std::invalid_argument("vision_attention: invalid scale"); }
    const std::int32_t patches  = q.ne[2];
    const std::int32_t segments = cu_seqlens.ne[0] - 1;
    if (patches <= 0) { throw std::invalid_argument("vision_attention: P must be positive"); }
    require_qkv(q, patches, "q");
    require_qkv(k, patches, "k");
    require_qkv(v, patches, "v");
    require_qkv(out, patches, "out");
    if (!std::equal(q.ne, q.ne + 4, k.ne) || !std::equal(q.ne, q.ne + 4, v.ne) ||
        !std::equal(q.ne, q.ne + 4, out.ne) || !out.is_contiguous()) {
        throw std::invalid_argument("vision_attention: q/k/v/out shapes must match, out contiguous");
    }
    if (cu_seqlens.dtype != DType::I32 || segments <= 0 || segments > patches ||
        cu_seqlens.ne[1] != 1 || cu_seqlens.ne[2] != 1 || cu_seqlens.ne[3] != 1 ||
        !cu_seqlens.is_contiguous() || cu_seqlens.data == nullptr) {
        throw std::invalid_argument("vision_attention: cu_seqlens must be contiguous I32 [S+1]");
    }
    const auto aligned16 = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0; };
    const std::int32_t head_dim = q.ne[0];
    if (!detail::vision_attention_fp8_supported(head_dim) || !aligned16(q.data) ||
        !aligned16(k.data) || !aligned16(v.data) || q.nb[1] != k.nb[1] || q.nb[1] != v.nb[1] ||
        q.nb[2] != k.nb[2] || q.nb[2] != v.nb[2] || (q.nb[2] % 16) != 0 || (k.nb[2] % 16) != 0 ||
        (v.nb[2] % 16) != 0 || scratch.data == nullptr || !aligned16(scratch.data) ||
        scratch.bytes < detail::vision_attention_fp8_scratch_bytes(head_dim, q.ne[1], patches)) {
        return false;
    }
    auto scratch_scope = workspace.scope();
    Tensor tiles       = allocate_workspace(workspace, patches, segments);
    Tensor* tiles_ptr  = tiles.data == nullptr ? nullptr : &tiles;
    detail::vision_attention_fp8_launch(q, k, v, cu_seqlens, tiles_ptr, scratch, out, stream, scale,
                                        rope_positions);
    return true;
}

} // namespace

bool vision_attention_fp8(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& cu_seqlens, WorkspaceArena& workspace,
                          DeviceSpan scratch, Tensor& out, cudaStream_t stream, float scale) {
    return run_fp8(q, k, v, cu_seqlens, workspace, scratch, out, stream, scale, nullptr);
}

bool vision_attention_fp8_rope(const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& positions, std::int32_t rotary_dim, float theta,
                               const Tensor& cu_seqlens, WorkspaceArena& workspace,
                               DeviceSpan scratch, Tensor& out, cudaStream_t stream, float scale) {
    if (positions.dtype != DType::I32 || positions.ne[0] != q.ne[2] || positions.ne[2] != 1 ||
        positions.ne[3] != 1 || !positions.is_contiguous() || positions.data == nullptr ||
        !detail::vision_attention_fp8_rope_supported(q.ne[0], q.ne[1], rotary_dim, theta, positions.ne[1])) {
        return false;
    }
    return run_fp8(q, k, v, cu_seqlens, workspace, scratch, out, stream, scale, &positions);
}

void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                      std::int32_t segment_length, Tensor& out, cudaStream_t stream, float scale) {
    if (!std::isfinite(scale) || scale < 0) { throw std::invalid_argument("vision_attention: invalid scale"); }
    const std::int32_t patches = q.ne[2];
    if (patches <= 0) { throw std::invalid_argument("vision_attention: P must be positive"); }
    require_qkv(q, patches, "q");
    require_qkv(k, patches, "k");
    require_qkv(v, patches, "v");
    require_qkv(out, patches, "out");
    if (!std::equal(q.ne, q.ne + 4, k.ne) || !std::equal(q.ne, q.ne + 4, v.ne) ||
        !std::equal(q.ne, q.ne + 4, out.ne)) {
        throw std::invalid_argument("vision_attention: q/k/v/out shapes must match");
    }
    if (!out.is_contiguous()) {
        throw std::invalid_argument("vision_attention: out must be contiguous");
    }
    if (segment_length <= 0 || patches % segment_length != 0) {
        throw std::invalid_argument("vision_attention: invalid uniform segment length");
    }
    detail::vision_attention_uniform_launch(q, k, v, segment_length, out, stream, scale);
}

} // namespace sinfer::ops
