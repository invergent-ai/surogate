#include "ops/launcher/vision_attention_fp8.h"

#include "core/device.h"
#include "ops/kernel/vision_attention_fp8.cuh"

#include <algorithm>
#include <atomic>
#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

static_assert(kVisionFp8Br == kVisionAttentionBr,
              "the packed-segment tile descriptors are cut at the BF16 kernel's query tile");

constexpr std::size_t kAlignment = 256;
std::size_t aligned(std::size_t bytes) { return (bytes + kAlignment - 1) / kAlignment * kAlignment; }

bool device_is_sm12x() noexcept {
    static std::atomic<int> known[64] = {}; // 0 unread, 1 no, 2 yes
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 64) {
        (void)cudaGetLastError();
        return false;
    }
    int state = known[device].load(std::memory_order_relaxed);
    if (state == 0) {
        int major       = 0;
        const bool read = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess;
        if (!read) { (void)cudaGetLastError(); }
        state = read && major == 12 ? 2 : 1;
        known[device].store(state, std::memory_order_relaxed);
    }
    return state == 2;
}

struct Scratch {
    std::uint8_t* q8;
    std::uint8_t* k8;
    std::uint8_t* v8;
    float* sq;
    float* sk;
    float* vamax;
};

template <int D>
std::size_t scratch_bytes_for(std::int32_t heads, std::int32_t patches) {
    const std::size_t rows = static_cast<std::size_t>(heads) * static_cast<std::size_t>(patches);
    return 3 * aligned(rows * kVisionFp8RowBytes<D>) + 2 * aligned(rows * sizeof(float)) +
           aligned(static_cast<std::size_t>(heads) * kVisionFp8VColumns<D> * sizeof(float));
}

template <int D>
Scratch carve(void* base, std::int32_t heads, std::int32_t patches) {
    const std::size_t rows = static_cast<std::size_t>(heads) * static_cast<std::size_t>(patches);
    auto* cursor           = static_cast<unsigned char*>(base);
    const auto take        = [&](std::size_t bytes) {
        void* at = cursor;
        cursor += aligned(bytes);
        return at;
    };
    Scratch s{};
    s.q8    = static_cast<std::uint8_t*>(take(rows * kVisionFp8RowBytes<D>));
    s.k8    = static_cast<std::uint8_t*>(take(rows * kVisionFp8RowBytes<D>));
    s.v8    = static_cast<std::uint8_t*>(take(rows * kVisionFp8RowBytes<D>));
    s.sq    = static_cast<float*>(take(rows * sizeof(float)));
    s.sk    = static_cast<float*>(take(rows * sizeof(float)));
    s.vamax = static_cast<float*>(take(static_cast<std::size_t>(heads) * kVisionFp8VColumns<D> * sizeof(float)));
    return s;
}

std::int64_t elements(const Tensor& t, int dim) { return t.nb[dim] / 2; }

template <int D>
void launch_for(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens,
                Tensor* tiles, DeviceSpan scratch, Tensor& out, cudaStream_t stream, float scale,
                const Tensor* rope_positions) {
    const std::int32_t heads   = q.ne[1];
    const std::int32_t patches = q.ne[2];
    if (scratch.data == nullptr || scratch.bytes < scratch_bytes_for<D>(heads, patches)) {
        throw std::invalid_argument("vision_attention_fp8: scratch too small");
    }
    const Scratch s = carve<D>(scratch.data, heads, patches);
    CUDA_CHECK(cudaMemsetAsync(s.vamax, 0, static_cast<std::size_t>(heads) * kVisionFp8VColumns<D> * sizeof(float), stream));
    if (q.nb[1] != k.nb[1] || q.nb[1] != v.nb[1] || q.nb[2] != k.nb[2] || q.nb[2] != v.nb[2]) {
        throw std::invalid_argument("vision_attention_fp8: q, k and v must share their strides");
    }
    const std::int64_t rows = static_cast<std::int64_t>(heads) * patches;
    if (rope_positions != nullptr) {
        const int tokens_per_block = std::max(1, 256 / (2 * heads));
        const int threads          = tokens_per_block * 2 * heads;
        if (threads > 1024) { throw std::invalid_argument("vision_attention_fp8: too many heads"); }
        const std::size_t shared = static_cast<std::size_t>(tokens_per_block) * D * sizeof(float);
        vision_fp8_rope_qk_kernel<D><<<static_cast<unsigned>((patches + tokens_per_block - 1) / tokens_per_block),
                                       static_cast<unsigned>(threads), shared, stream>>>(
            static_cast<const std::int32_t*>(rope_positions->data),
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
            elements(q, 1), elements(q, 2), heads, patches, tokens_per_block, s.q8, s.k8, s.sq, s.sk);
    } else {
        vision_fp8_prepare_qk_kernel<D><<<static_cast<unsigned>((2 * rows + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
            elements(q, 1), elements(q, 2), heads, patches, s.q8, s.k8, s.sq, s.sk);
    }
    constexpr int kTokensPerAmaxBlock = 64;
    const int amax_threads = (heads * (D / 8) + 31) / 32 * 32;
    if (amax_threads > 1024) { throw std::invalid_argument("vision_attention_fp8: too many heads"); }
    vision_fp8_v_amax_kernel<D><<<static_cast<unsigned>((patches + kTokensPerAmaxBlock - 1) / kTokensPerAmaxBlock),
                                  static_cast<unsigned>(amax_threads), 0, stream>>>(
        static_cast<const __nv_bfloat16*>(v.data), elements(v, 1), elements(v, 2), heads, patches,
        kTokensPerAmaxBlock, s.vamax);
    vision_fp8_prepare_v_kernel<D><<<static_cast<unsigned>((rows + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(v.data), elements(v, 1), elements(v, 2), heads, patches,
        s.vamax, s.v8);
    int query_tiles = (patches + kVisionFp8Br - 1) / kVisionFp8Br;
    if (tiles != nullptr) {
        query_tiles = tiles->ne[1];
        vision_attention_prepare_tiles_kernel<<<1, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(cu_seqlens.data), cu_seqlens.ne[0] - 1,
            static_cast<VisionAttentionTile*>(tiles->data), query_tiles, patches);
    }
    vision_attention_fp8_kernel<D><<<dim3(static_cast<unsigned>(query_tiles), static_cast<unsigned>(heads)),
                                     kVisionFp8Threads, 0, stream>>>(
        s.q8, s.k8, s.v8, s.sq, s.sk, s.vamax,
        tiles != nullptr ? static_cast<const VisionAttentionTile*>(tiles->data) : nullptr, patches,
        static_cast<__nv_bfloat16*>(out.data), scale);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool vision_attention_fp8_supported(std::int32_t head_dim) {
    return (head_dim == 64 || head_dim == 72) && device_is_sm12x();
}

std::size_t vision_attention_fp8_scratch_bytes(std::int32_t head_dim, std::int32_t heads,
                                               std::int32_t patches) {
    switch (head_dim) {
    case 72: return scratch_bytes_for<72>(heads, patches);
    case 64: return scratch_bytes_for<64>(heads, patches);
    default: return 0;
    }
}

bool vision_attention_fp8_rope_supported(std::int32_t head_dim, std::int32_t heads,
                                         std::int32_t rotary_dim, float theta, std::int32_t axes) {
    return (head_dim == 72 || head_dim == 64) && heads == 16 && rotary_dim == head_dim &&
           theta == 10'000.0F && axes == 2;
}

void vision_attention_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& cu_seqlens, Tensor* tiles, DeviceSpan scratch,
                                 Tensor& out, cudaStream_t stream, float scale,
                                 const Tensor* rope_positions) {
    switch (q.ne[0]) {
    case 72: launch_for<72>(q, k, v, cu_seqlens, tiles, scratch, out, stream, scale, rope_positions); return;
    case 64: launch_for<64>(q, k, v, cu_seqlens, tiles, scratch, out, stream, scale, rope_positions); return;
    default: throw std::invalid_argument("vision_attention_fp8: head dim must be 64 or 72");
    }
}

} // namespace sinfer::ops::detail
