#include "ops/launcher/vision_attention.h"

#include "ops/kernel/vision_attention.cuh"
#include "core/device.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

std::int64_t stride_elements(const Tensor& tensor, int dim) {
    return tensor.nb[dim] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

template <int Br, int Bc>
void launch_flash(const Tensor& q, const Tensor& k, const Tensor& v,
                  const VisionAttentionTile* tiles, std::int32_t uniform_segment_length,
                  std::int32_t query_tiles, Tensor& out, cudaStream_t stream) {
    constexpr int kThreads = Br * 2;
    constexpr int kSmemBytes =
        (Br + 2 * Bc) * kVisionAttentionPaddedD * static_cast<int>(sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(query_tiles),
                    static_cast<unsigned>(kVisionAttentionHeads), 1u);
    vision_attention_flash_kernel<Br, Bc><<<grid, kThreads, kSmemBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), tiles, q.ne[2], uniform_segment_length,
        static_cast<__nv_bfloat16*>(out.data), stride_elements(q, 0), stride_elements(q, 1),
        stride_elements(q, 2), stride_elements(k, 0), stride_elements(k, 1), stride_elements(k, 2),
        stride_elements(v, 0), stride_elements(v, 1), stride_elements(v, 2));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void vision_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor* tiles, Tensor& out,
                             cudaStream_t stream) {
    const bool packed_segments = tiles != nullptr;
    const int max_tiles =
        packed_segments ? tiles->ne[1] : (q.ne[2] + kVisionAttentionBr - 1) / kVisionAttentionBr;
    if (packed_segments) {
        vision_attention_prepare_tiles_kernel<<<1, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(cu_seqlens.data), cu_seqlens.ne[0] - 1,
            static_cast<VisionAttentionTile*>(tiles->data), max_tiles, q.ne[2]);
        CUDA_CHECK(cudaGetLastError());
    }

    launch_flash<kVisionAttentionBr, kVisionAttentionBc>(
        q, k, v, packed_segments ? static_cast<const VisionAttentionTile*>(tiles->data) : nullptr,
        0, max_tiles, out, stream);
}

std::int32_t vision_attention_uniform_tile(std::int32_t segment_length) {
    // Measured issued-MMA rates are approximately 3:6:8 for the 16/32/64
    // tiles. Minimize padded tile area divided by that rate.
    constexpr std::int32_t tiles[] = {16, 32, 64};
    constexpr std::int32_t rates[] = {3, 6, 8};
    std::int32_t best_tile         = tiles[0];
    std::int64_t best_padded       = ((segment_length + best_tile - 1) / best_tile) * best_tile;
    std::int64_t best_square       = best_padded * best_padded;
    std::int32_t best_rate         = rates[0];
    for (int i = 1; i < 3; ++i) {
        const std::int64_t padded = ((segment_length + tiles[i] - 1) / tiles[i]) * tiles[i];
        const std::int64_t square = padded * padded;
        if (square * best_rate < best_square * rates[i]) {
            best_tile   = tiles[i];
            best_square = square;
            best_rate   = rates[i];
        }
    }
    return best_tile;
}

void vision_attention_uniform_launch_with_tile(const Tensor& q, const Tensor& k, const Tensor& v,
                                               std::int32_t segment_length, std::int32_t tile_size,
                                               Tensor& out, cudaStream_t stream) {
    const std::int32_t segments    = q.ne[2] / segment_length;
    const std::int32_t query_tiles = segments * ((segment_length + tile_size - 1) / tile_size);
    switch (tile_size) {
    case 16:
        launch_flash<16, 16>(q, k, v, nullptr, segment_length, query_tiles, out, stream);
        return;
    case 32:
        launch_flash<32, 32>(q, k, v, nullptr, segment_length, query_tiles, out, stream);
        return;
    case 64:
        launch_flash<64, 64>(q, k, v, nullptr, segment_length, query_tiles, out, stream);
        return;
    default:
        throw std::invalid_argument("vision_attention: invalid uniform tile size");
    }
}

void vision_attention_uniform_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     std::int32_t segment_length, Tensor& out,
                                     cudaStream_t stream) {
    vision_attention_uniform_launch_with_tile(
        q, k, v, segment_length, vision_attention_uniform_tile(segment_length), out, stream);
}

} // namespace ninfer::ops::detail
