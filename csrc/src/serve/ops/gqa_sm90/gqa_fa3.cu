// sinfer::ops - FlashAttention-3's sm90 forward over the paged KV cache (see gqa_fa3.h).
#include "ops/gqa_sm90/gqa_fa3.h"

#include "ops/gqa_sm90/gqa_fa3_launch.h"

#include "tile_size.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/kernel_hardware_info.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace sinfer::ops::detail::gqa_fa3 {
namespace {

constexpr std::int32_t kPageSize = 64;
constexpr std::size_t kAlign     = 256;
constexpr int kArch              = 90;

struct Hardware {
    int device   = -1;
    int cc       = 0;
    int sm_count = 0;
};

const Hardware& hardware() {
    static thread_local Hardware cached;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return cached; }
    if (cached.device != device) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        cached.device   = device;
        cached.cc       = major * 10 + minor;
        cached.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
    }
    return cached;
}

// Whether the split combine starts as the forward's programmatic dependent (combine in
// gqa_fa3_launch.h), as FA3's API launches it; SUROGATE_SERVE_GQA_FA3_PDL=0 launches it after
// the forward instead.
bool combine_pdl() {
    static const bool enabled = [] {
        const char* value = std::getenv("SUROGATE_SERVE_GQA_FA3_PDL");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// vLLM's FlashAttention fork runs a launch whose segments pack at most 64 query rows (decode, and
// verify rows of a narrow group) on one MMA warpgroup with a 64-row M tile, at head dim 64 or 128
// without a window (its heuristics.h, use_one_mma_wg; here over a BF16 cache, the only one built
// that way). A decode row packs its query group into 64 rows where the two-warpgroup tile pads it
// to 128, and the smaller CTA leaves more of them in flight on the keys.
// SUROGATE_SERVE_GQA_FA3_ONE_WG=0 keeps the 128-row tile.
bool one_mma_wg_by_env() {
    static const bool enabled = [] {
        const char* value = std::getenv("SUROGATE_SERVE_GQA_FA3_ONE_WG");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool use_one_mma_wg(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads,
                    std::int32_t max_q, bool fp8_cache, bool local) {
    return one_mma_wg_by_env() && !fp8_cache && !local && (head_dim == 64 || head_dim == 128) &&
           static_cast<std::int64_t>(max_q) * (q_heads / kv_heads) <= 64;
}

bool enabled_by_env() {
    static const bool enabled = [] {
        const char* value = std::getenv("SUROGATE_SERVE_GQA_FA3");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

std::size_t align_up(std::size_t value) { return (value + kAlign - 1) / kAlign * kAlign; }

// FA3's prepared varlen scheduler keeps four per-segment vectors (dynamic split count, M blocks,
// the sorted order, heads per L2 window), each rounded to four entries, and the tile counter
// after them (workspace_bytes in the header sizes them).
std::int32_t rounded_segments(std::int32_t segments) { return (segments + 3) / 4 * 4; }

__global__ void prompt_metadata_kernel(const std::int32_t* positions, std::int32_t tokens,
                                       const std::int32_t* kv_row, std::int32_t* metadata) {
    if (threadIdx.x == 0) {
        metadata[0] = 0;
        metadata[1] = tokens;
        metadata[2] = tokens;
        metadata[3] = positions[0] + tokens;
        metadata[4] = kv_row != nullptr ? kv_row[0] : 0;
    }
}

// Row r of a decode or verify batch as segment r (rows_metadata in the header), and the tile
// counters of the launches that will share it zeroed.
__global__ void rows_metadata_kernel(const std::int32_t* positions, std::int32_t width,
                                     std::int32_t batch, const std::int32_t* valid_columns,
                                     const std::int32_t* rows, std::int32_t* metadata,
                                     std::int32_t* zero, std::int32_t counters) {
    const std::int32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < counters) { zero[r] = 0; }
    if (r >= batch) { return; }
    std::int32_t valid = valid_columns != nullptr ? valid_columns[r] : width;
    valid              = valid < 0 ? 0 : (valid > width ? width : valid);
    const std::int32_t keys = valid > 0 ? positions[r * width] + valid : 0;
    metadata[r]                 = r * width;
    metadata[batch + 1 + r]     = valid;
    metadata[2 * batch + 1 + r] = keys > 0 ? keys : 0;
    metadata[3 * batch + 1 + r] = rows != nullptr ? rows[r] : r;
    if (r == batch - 1) { metadata[batch] = batch * width; }
}

__global__ void segment_kv_lengths_kernel(const std::int32_t* positions,
                                          const std::int32_t* q_offsets,
                                          const std::int32_t* q_lengths, std::int32_t segments,
                                          std::int32_t* kv_lengths) {
    const std::int32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < segments) { kv_lengths[s] = positions[q_offsets[s]] + q_lengths[s]; }
}

// The queries' e4m3 codes for the FP8 kernel: a plain saturating cast, scale 1, as vLLM's static
// per-tensor query quantization does with its default scale and as the cache stores its keys.
// Eight values a thread.
__global__ void quantize_query_kernel(const uint4* __restrict__ q, uint2* __restrict__ codes,
                                      std::size_t groups) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= groups) { return; }
    const uint4 packed               = q[i];
    const std::uint32_t words[4]     = {packed.x, packed.y, packed.z, packed.w};
    __nv_fp8x2_storage_t out[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words[w]));
        out[w]         = __nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
    }
    codes[i] = make_uint2(static_cast<std::uint32_t>(out[0]) | static_cast<std::uint32_t>(out[1]) << 16,
                          static_cast<std::uint32_t>(out[2]) | static_cast<std::uint32_t>(out[3]) << 16);
}

template <int HeadDim, bool Split>
void dispatch_dim(Flash_fwd_params& p, bool fp8, bool local, bool one_wg, cudaStream_t stream) {
    if constexpr (HeadDim != 256) {
        if (one_wg) { return launch<HeadDim, false, false, Split, true>(p, stream); }
    }
    if constexpr (!Split) {
        if (fp8) {
            return local ? launch<HeadDim, true, true, false>(p, stream)
                         : launch<HeadDim, true, false, false>(p, stream);
        }
    }
    return local ? launch<HeadDim, false, true, Split>(p, stream)
                 : launch<HeadDim, false, false, Split>(p, stream);
}

template <bool Split>
void dispatch(Flash_fwd_params& p, bool fp8, bool local, bool one_wg, cudaStream_t stream) {
    switch (p.d) {
    case 64:  return dispatch_dim<64, Split>(p, fp8, local, one_wg, stream);
    case 128: return dispatch_dim<128, Split>(p, fp8, local, one_wg, stream);
    default:  return dispatch_dim<256, Split>(p, fp8, local, false, stream);
    }
}

// flash_api.cpp's num_splits_heuristic: the fewest splits within 85 % of the best wave
// efficiency, none when the segments' M blocks nearly fill the device or a segment holds at most
// four key blocks (FA3's sm90 kernels split causal and windowed launches only this way).
int num_splits_heuristic(int total_mblocks, int num_sms, int num_n_blocks, int max_splits) {
    if (total_mblocks >= 0.8f * num_sms || num_n_blocks <= 4) { return 1; }
    max_splits = std::min({max_splits, num_sms, num_n_blocks});
    float best = 0.0f;
    for (int splits = 1; splits <= max_splits; ++splits) {
        const float waves = float(total_mblocks * splits) / num_sms;
        best              = std::max(best, waves / std::ceil(waves));
    }
    for (int splits = 1; splits <= max_splits; ++splits) {
        const float waves = float(total_mblocks * splits) / num_sms;
        if (waves / std::ceil(waves) >= 0.85f * best) { return splits; }
    }
    return 1;
}


void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gqa_fa3 ") + what + ": " + cudaGetErrorString(status));
    }
}

} // namespace

bool available() noexcept { return hardware().cc == 90 && enabled_by_env(); }

bool supports(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads) noexcept {
    return (head_dim == 64 || head_dim == 128 || head_dim == 256) && kv_heads > 0 && q_heads > 0 &&
           q_heads % kv_heads == 0;
}

std::int32_t min_columns() noexcept {
    static const std::int32_t value = [] {
        const char* text = std::getenv("SUROGATE_SERVE_GQA_FA3_MIN_COLUMNS");
        if (text == nullptr || *text == '\0') { return 32; }
        char* end         = nullptr;
        const long parsed = std::strtol(text, &end, 10);
        return (end != nullptr && *end == '\0' && parsed >= 1 && parsed <= 1 << 20)
                   ? static_cast<std::int32_t>(parsed) : 32;
    }();
    return value;
}

bool rows_enabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("SUROGATE_SERVE_GQA_FA3_ROWS");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

void prompt_metadata(const std::int32_t* positions, std::int32_t tokens,
                     const std::int32_t* kv_row, std::int32_t* metadata, cudaStream_t stream) {
    prompt_metadata_kernel<<<1, 32, 0, stream>>>(positions, tokens, kv_row, metadata);
    check_launch("prompt metadata");
}

RowSplits row_splits(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads,
                     std::int32_t total_q, std::int32_t max_q, std::int32_t max_keys, bool fp8_cache,
                     std::int32_t sliding_window) noexcept {
    if (!supports(head_dim, q_heads, kv_heads) || total_q <= 0 || max_q <= 0 || max_keys <= 0 ||
        fp8_cache) {
        return {};
    }
    const int num_sms = hardware().sm_count;
    if (num_sms <= 0) { return {}; }
    const bool local = sliding_window > 0;
    const auto tile  = tile_size_fwd_sm90(
        head_dim, head_dim, !local, local, /*element_size=*/2, /*v_colmajor=*/false,
        /*paged_kv_non_TMA=*/true, /*softcap=*/false,
        use_one_mma_wg(head_dim, q_heads, kv_heads, max_q, fp8_cache, local));
    const int block_m = std::get<0>(tile);
    const int block_n = std::get<1>(tile);
    // A window loads at most its own keys plus one M tile's worth (FA3's seqlen_k_loaded).
    const int keys = local ? std::min(max_keys, sliding_window + block_m) : max_keys;
    const int n_blocks = (keys + block_n - 1) / block_n;
    const int m_blocks = (max_q * (q_heads / kv_heads) + block_m - 1) / block_m;
    const int splits = num_splits_heuristic(kv_heads * m_blocks, num_sms, n_blocks, kMaxSplits);
    if (splits <= 1) { return {}; }
    const std::size_t cap = kPartialBudget / split_bytes(head_dim, q_heads, total_q);
    return {static_cast<std::int32_t>(std::min<std::size_t>(splits, std::max<std::size_t>(1, cap))),
            true};
}

void rows_metadata(const std::int32_t* positions, std::int32_t width, std::int32_t batch,
                   const std::int32_t* valid_columns, const std::int32_t* rows,
                   std::int32_t* metadata, cudaStream_t stream, std::int32_t* zero,
                   std::int32_t counters) {
    if (batch <= 0 || width <= 0) { throw std::invalid_argument("gqa_fa3: empty row batch"); }
    if (counters < 0 || (counters > 0 && zero == nullptr)) {
        throw std::invalid_argument("gqa_fa3: invalid tile counters");
    }
    const std::int32_t threads = std::max(batch, counters);
    rows_metadata_kernel<<<(threads + 127) / 128, 128, 0, stream>>>(
        positions, width, batch, valid_columns, rows, metadata, zero, counters);
    check_launch("rows metadata");
}

void segment_kv_lengths(const std::int32_t* positions, const std::int32_t* q_offsets,
                        const std::int32_t* q_lengths, std::int32_t segments,
                        std::int32_t* kv_lengths, cudaStream_t stream) {
    if (segments <= 0) { return; }
    segment_kv_lengths_kernel<<<(segments + 127) / 128, 128, 0, stream>>>(
        positions, q_offsets, q_lengths, segments, kv_lengths);
    check_launch("segment kv lengths");
}

void run(const PagedLaunch& a, void* workspace, std::size_t workspace_capacity,
         cudaStream_t stream) {
    if (!available()) {
        throw std::runtime_error("gqa_fa3: needs an sm_90 device and a build with 90a");
    }
    if (!supports(a.head_dim, a.q_heads, a.kv_heads)) {
        throw std::invalid_argument("gqa_fa3: unsupported head geometry");
    }
    if (a.segments <= 0 || a.total_q <= 0 || a.max_q <= 0 || a.max_q > a.total_q ||
        a.logical_pages <= 0 || a.table_rows <= 0 || a.physical_pages <= 0 || a.q == nullptr ||
        a.out == nullptr || a.k_pages == nullptr || a.v_pages == nullptr ||
        a.block_tables == nullptr || a.q_offsets == nullptr || a.q_lengths == nullptr ||
        a.kv_lengths == nullptr || a.kv_rows == nullptr || !(a.scale > 0.0f) ||
        a.sliding_window < 0 || a.max_splits < 1 || a.max_splits > kMaxSplits ||
        (a.fp8_cache && a.max_splits > 1) || (!a.prepare && a.scheduler == nullptr) ||
        (a.scheduler != nullptr && a.tile_counter == nullptr) ||
        static_cast<std::size_t>(a.max_splits) * split_bytes(a.head_dim, a.q_heads, a.total_q) >
            std::max(kPartialBudget, split_bytes(a.head_dim, a.q_heads, a.total_q))) {
        throw std::invalid_argument("gqa_fa3: invalid launch");
    }
    if (workspace_capacity < workspace_bytes(a.head_dim, a.q_heads, a.total_q, a.segments,
                                             a.fp8_cache, a.max_splits > 1)) {
        throw std::invalid_argument("gqa_fa3: workspace smaller than workspace_bytes()");
    }

    const std::int64_t dim = a.head_dim;
    const std::size_t rows = static_cast<std::size_t>(a.q_heads) * a.total_q;
    const std::int32_t b_rounded = rounded_segments(a.segments);
    auto* lse       = static_cast<float*>(workspace);
    auto* scheduler = a.scheduler != nullptr
                          ? a.scheduler
                          : reinterpret_cast<int*>(static_cast<char*>(workspace) +
                                                   align_up(rows * sizeof(float)));
    const void* q   = a.q;
    char* cursor    = static_cast<char*>(workspace) + align_up(rows * sizeof(float)) +
                      align_up((static_cast<std::size_t>(b_rounded) * 4 + 1) * sizeof(std::int32_t));
    if (a.fp8_cache) {
        void* codes = cursor;
        cursor += align_up(rows * static_cast<std::size_t>(dim));
        const std::size_t groups = rows * static_cast<std::size_t>(dim) / 8;
        quantize_query_kernel<<<static_cast<unsigned>((groups + 255) / 256), 256, 0, stream>>>(
            static_cast<const uint4*>(a.q), static_cast<uint2*>(codes), groups);
        check_launch("query quantize");
        q = codes;
    }

    Flash_fwd_params p;
    std::memset(&p, 0, sizeof(p));
    p.is_bf16 = !a.fp8_cache;
    p.is_e4m3 = a.fp8_cache;  // the scheduler sizes its L2 windows by the element
    p.q_ptr   = const_cast<void*>(q);
    p.k_ptr   = const_cast<void*>(a.k_pages);
    p.v_ptr   = const_cast<void*>(a.v_pages);
    p.o_ptr   = a.out;
    // Strides in elements. Varlen queries carry no batch stride; the cache's "batch" is the page.
    p.q_row_stride   = a.q_heads * dim;
    p.q_head_stride  = dim;
    p.o_row_stride   = p.q_row_stride;
    p.o_head_stride  = dim;
    p.k_row_stride   = dim;
    p.v_row_stride   = dim;
    p.k_head_stride  = kPageSize * dim;
    p.v_head_stride  = p.k_head_stride;
    p.k_batch_stride = a.kv_heads * kPageSize * dim;
    p.v_batch_stride = p.k_batch_stride;
    p.v_dim_stride   = 1;
    p.softmax_lse_ptr = lse;

    p.b          = a.segments;
    p.b_k        = a.table_rows;
    p.h          = a.q_heads;
    p.h_k        = a.kv_heads;
    p.d          = a.head_dim;
    p.d_rounded  = a.head_dim;
    p.dv         = a.head_dim;
    p.dv_rounded = a.head_dim;
    p.seqlen_q   = a.max_q;
    p.total_q    = a.total_q;
    p.seqlen_k   = a.logical_pages * kPageSize;
    p.total_k    = a.physical_pages * kPageSize;
    p.seqlen_q_rounded = (a.max_q + 127) / 128 * 128;
    p.seqlen_k_rounded = (p.seqlen_k + 127) / 128 * 128;
    p.scale_softmax    = a.scale;

    p.cu_seqlens_q = const_cast<int*>(a.q_offsets);
    p.seqused_q    = const_cast<int*>(a.q_lengths);
    p.seqused_k    = const_cast<int*>(a.kv_lengths);
    p.kv_batch_idx = const_cast<int*>(a.kv_rows);
    p.page_table   = const_cast<int*>(a.block_tables);
    p.page_table_batch_stride = a.logical_pages;
    p.page_size    = kPageSize;
    p.num_pages    = a.physical_pages;
    p.pagedkv_tma  = false;

    const bool local    = a.sliding_window > 0;
    p.is_causal         = !local;
    p.is_local          = local;
    p.window_size_left  = local ? a.sliding_window - 1 : -1;
    p.window_size_right = 0;
    p.num_splits        = a.max_splits;
    p.pack_gqa          = true;
    if (a.max_splits > 1) {
        // [splits][q_heads][total_q][dv] partial outputs and [splits][q_heads][total_q]
        // log-sum-exps, as flash_api.cpp lays them out for a varlen launch.
        const std::size_t partial = static_cast<std::size_t>(a.max_splits) * rows;
        p.oaccum_ptr               = cursor;
        p.softmax_lseaccum_ptr     = cursor + align_up(partial * dim * sizeof(float));
        p.oaccum_split_stride      = static_cast<std::int64_t>(rows) * dim;
        p.oaccum_head_stride       = static_cast<std::int64_t>(a.total_q) * dim;
        p.oaccum_row_stride        = dim;
        p.lseaccum_split_stride    = static_cast<std::int64_t>(rows);
        p.lseaccum_head_stride     = a.total_q;
    }

    p.varlen_sort_batches   = !local;  // as FA3's API sets the scheduler's Sort and LPT order
    p.head_swizzle          = true;
    p.num_splits_dynamic_ptr = scheduler;
    p.num_m_blocks_ptr       = scheduler + b_rounded;
    // Only a sorted scheduler (causal) writes the virtual-to-real batch map, and the combine reads
    // it whenever it is set, so a window leaves it null, as flash_api.cpp does.
    p.varlen_batch_idx_ptr   = local ? nullptr : scheduler + b_rounded * 2;
    p.num_nheads_in_l2_ptr   = scheduler + b_rounded * 3;
    p.tile_count_semaphore   = a.scheduler != nullptr ? a.tile_counter : scheduler + b_rounded * 4;
    p.tile_count_semaphore_offset = b_rounded * 4;
    // The prepare kernel runs on the device lengths (it also zeroes the tile counter), which is
    // what lets a captured graph replay with new ones; a launch sharing metadata an earlier one
    // prepared skips it, its counter zeroed by whoever filled the segment arrays.
    p.skip_scheduler_metadata_computation = !a.prepare;
    p.prepare_varlen_pdl = false;
    p.arch   = kArch;
    p.num_sm = hardware().sm_count;
    // q/k/v descale pointers stay null: 1.0, the cache's (absent) scale.

    const bool one_wg =
        use_one_mma_wg(a.head_dim, a.q_heads, a.kv_heads, a.max_q, a.fp8_cache, local);
    if (a.max_splits == 1) { return dispatch<false>(p, a.fp8_cache, local, one_wg, stream); }
    dispatch<true>(p, a.fp8_cache, local, one_wg, stream);
    // The combine writes the output of every segment the forward split and leaves the rest,
    // which the forward wrote whole.
    p.is_bf16 = true;
    combine(p, stream, combine_pdl());
}

} // namespace sinfer::ops::detail::gqa_fa3
