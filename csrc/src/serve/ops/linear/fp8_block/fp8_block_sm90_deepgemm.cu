// The DeepGEMM route of Hopper's block-scaled FP8 GEMM (fp8_block_sm90_deepgemm.h): DeepGEMM's own
// sm90 cost model (csrc/jit_kernels/heuristics/sm90.hpp, MIT) picks one of the compiled tiles per
// call, and the call builds the four TMA descriptors DeepGEMM's runtime would and launches it.

#include "ops/linear/fp8_block/fp8_block_sm90_deepgemm_tiles.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>

namespace sinfer::ops::detail::fp8_block::sm90 {
namespace {

struct Tile {
    int bm, bn, cm, cn;
    dg::LaunchFn launch;
};

// DeepGEMM enumerates cluster m, then cluster n, then block m (64, 128, 256), then block n, and
// keeps the first of equally cheap candidates; the table is sorted the same way.
const std::array<Tile, dg::kNumTiles>& tiles() {
    static const std::array<Tile, dg::kNumTiles> sorted = [] {
        std::array<Tile, dg::kNumTiles> t{{
#define SINFER_DG_ENTRY(BM, BN, CM, CN) Tile{BM, BN, CM, CN, &dg::SINFER_DG_NAME(BM, BN, CM, CN)},
            SINFER_DG_TILES(SINFER_DG_ENTRY)
#undef SINFER_DG_ENTRY
        }};
        std::stable_sort(t.begin(), t.end(), [](const Tile& a, const Tile& b) {
            if (a.cm != b.cm) { return a.cm < b.cm; }
            if (a.cn != b.cn) { return a.cn < b.cn; }
            if (a.bm != b.bm) { return a.bm < b.bm; }
            return a.bn < b.bn;
        });
        return t;
    }();
    return sorted;
}

std::int64_t div_up(std::int64_t a, std::int64_t b) { return (a + b - 1) / b; }

// DeepGEMM's SM90ArchSpec::get_layout_info: L1/L2 traffic over the tile's blocks, divided by the
// last wave's fill; clusters only when there is more than one wave.
std::int64_t cycles(const Tile& t, int m, int n, int k, int sms, bool residual) {
    const std::int64_t blocks = div_up(m, t.bm) * div_up(n, t.bn);
    const std::int64_t waves  = div_up(blocks, sms);
    const int l2_bandwidth    = static_cast<int>(std::min(64.0 * sms, 8e6 / 1.3e3));
    const int l1_bandwidth    = 128 * sms;
    const std::int64_t l2_ab  = static_cast<std::int64_t>(k) * (t.bm / t.cn + t.bn / t.cm);
    const std::int64_t l1_ab  = static_cast<std::int64_t>(k) * (t.bm + t.bn);
    const std::int64_t l1_tc  = static_cast<std::int64_t>(k) * (std::max(64, t.bm) + t.bn) + t.bm * t.bn * 2;
    const std::int64_t cd     = static_cast<std::int64_t>(t.bm) * t.bn * 2 * (residual ? 2 : 1);
    const std::int64_t l2     = (l2_ab + cd) * blocks / l2_bandwidth;
    const std::int64_t l1     = (l1_ab + l1_tc + cd) * blocks / l1_bandwidth;
    const float efficiency    = static_cast<float>(blocks) / static_cast<float>(waves * sms);
    if (t.cm * t.cn > 1 && waves <= 1) { return std::numeric_limits<std::int64_t>::max(); }
    return static_cast<std::int64_t>(static_cast<float>(std::max(l1, l2)) / efficiency);
}

int multiprocessors() {
    static std::mutex mutex;
    static std::map<int, int> counts;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return 0; }
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = counts.find(device); found != counts.end()) { return found->second; }
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) { return 0; }
    return counts[device] = sms;
}

bool encode(CUtensorMap& map, CUtensorMapDataType type, const void* base, std::uint64_t inner,
            std::uint64_t outer, std::uint64_t outer_stride_bytes, std::uint32_t box_inner,
            std::uint32_t box_outer, CUtensorMapSwizzle swizzle) {
    const cuuint64_t dims[2]          = {inner, outer};
    const cuuint64_t strides[1]       = {outer_stride_bytes};
    const cuuint32_t box[2]           = {box_inner, box_outer};
    const cuuint32_t element_steps[2] = {1, 1};
    return cuTensorMapEncodeTiled(&map, type, 2, const_cast<void*>(base), dims, strides, box, element_steps,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

CUtensorMapSwizzle swizzle_of(int bytes) {
    switch (bytes) {
    case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
    case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
    case 32: return CU_TENSOR_MAP_SWIZZLE_32B;
    default: return CU_TENSOR_MAP_SWIZZLE_NONE;
    }
}

} // namespace

bool deepgemm(const DeepGemmOperands& o) {
    const auto aligned = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; };
    if (o.tokens <= 0 || o.n <= 0 || o.k <= 0 || o.k % dg::kBlockK != 0 || o.n % 8 != 0 ||
        o.k / dg::kBlockK > dg::kMaxKBlocks || !aligned(o.act_codes) || !aligned(o.act_scales) ||
        !aligned(o.w_codes) || !aligned(o.w_scales) || !aligned(o.out_bf16)) {
        return false;
    }
    const int sms = multiprocessors();
    if (sms <= 0) { return false; }
    const Tile* best = nullptr;
    std::int64_t best_cycles = 0;
    for (const Tile& t : tiles()) {
        if (sms % (t.cm * t.cn) != 0) { continue; }
        const std::int64_t c = cycles(t, o.tokens, o.n, o.k, sms, o.residual);
        if (best == nullptr || c < best_cycles) { best = &t, best_cycles = c; }
    }
    if (best == nullptr) { return false; }

    const auto m          = static_cast<std::uint64_t>(o.tokens);
    const auto n          = static_cast<std::uint64_t>(o.n);
    const auto k          = static_cast<std::uint64_t>(o.k);
    const auto scale_rows = static_cast<std::uint64_t>((o.tokens + 3) / 4 * 4); // sm90_scale_stride
    const int swizzle_d   = dg::swizzle_d(best->bn);
    dg::Launch l{};
    if (!encode(l.a, CU_TENSOR_MAP_DATA_TYPE_UINT8, o.act_codes, k, m, k, dg::kBlockK, best->bm,
                CU_TENSOR_MAP_SWIZZLE_128B) ||
        !encode(l.b, CU_TENSOR_MAP_DATA_TYPE_UINT8, o.w_codes, k, n, k, dg::kBlockK, best->bn,
                CU_TENSOR_MAP_SWIZZLE_128B) ||
        !encode(l.d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, o.out_bf16, n, m, n * 2, swizzle_d / 2, best->bm,
                swizzle_of(swizzle_d)) ||
        !encode(l.sfa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, o.act_scales, scale_rows, k / dg::kBlockK,
                scale_rows * 4, best->bm, 1, CU_TENSOR_MAP_SWIZZLE_NONE)) {
        return false;
    }
    l.sfb      = const_cast<float*>(o.w_scales);
    l.residual = o.residual ? static_cast<const __nv_bfloat16*>(o.out_bf16) : nullptr;
    l.m        = static_cast<std::uint32_t>(o.tokens);
    l.n        = static_cast<std::uint32_t>(o.n);
    l.k        = static_cast<std::uint32_t>(o.k);
    l.sms      = sms / (best->cm * best->cn) * (best->cm * best->cn);
    l.smem     = dg::smem_bytes(best->bm, best->bn, o.k / dg::kBlockK);
    l.stream   = o.stream;
    if (best->launch(l) != cudaSuccess) {
        (void)cudaGetLastError(); // a launch refused at configuration leaves no sticky error
        return false;
    }
    return true;
}

} // namespace sinfer::ops::detail::fp8_block::sm90
