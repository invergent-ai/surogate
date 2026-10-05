// sinfer::ops - FlashAttention-3's sm90 forward over the paged BF16 KV cache (see gqa_fa3.h).
#include "ops/gqa_sm90/gqa_fa3.h"

#include "flash_fwd_launch_template.h"

#include <cutlass/kernel_hardware_info.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace sinfer::ops::detail::gqa_fa3 {
namespace {

using Element = cutlass::bfloat16_t;

constexpr std::int32_t kPageSize = 64;
constexpr std::size_t kAlign     = 256;

// The one instantiation: BF16, head dim 256, causal, varlen, paged without TMA (a 64-slot page is
// not a multiple of the 80-key tile, the condition under which FA3 itself takes this path), the
// query group packed into the M tile, no split.
constexpr int kArch = 90;

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

__global__ void segment_kv_lengths_kernel(const std::int32_t* positions,
                                          const std::int32_t* q_offsets,
                                          const std::int32_t* q_lengths, std::int32_t segments,
                                          std::int32_t* kv_lengths) {
    const std::int32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < segments) { kv_lengths[s] = positions[q_offsets[s]] + q_lengths[s]; }
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
    return head_dim == kHeadDim && kv_heads > 0 && q_heads > 0 && q_heads % kv_heads == 0;
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

void prompt_metadata(const std::int32_t* positions, std::int32_t tokens,
                     const std::int32_t* kv_row, std::int32_t* metadata, cudaStream_t stream) {
    prompt_metadata_kernel<<<1, 32, 0, stream>>>(positions, tokens, kv_row, metadata);
    check_launch("prompt metadata");
}

void segment_kv_lengths(const std::int32_t* positions, const std::int32_t* q_offsets,
                        const std::int32_t* q_lengths, std::int32_t segments,
                        std::int32_t* kv_lengths, cudaStream_t stream) {
    if (segments <= 0) { return; }
    segment_kv_lengths_kernel<<<(segments + 127) / 128, 128, 0, stream>>>(
        positions, q_offsets, q_lengths, segments, kv_lengths);
    check_launch("segment kv lengths");
}

void run(const PagedPrefill& a, void* workspace, std::size_t workspace_capacity,
         cudaStream_t stream) {
    if (!available()) {
        throw std::runtime_error("gqa_fa3: needs an sm_90 device and a build with 90a");
    }
    if (!supports(kHeadDim, a.q_heads, a.kv_heads)) {
        throw std::invalid_argument("gqa_fa3: unsupported head geometry");
    }
    if (a.segments <= 0 || a.total_q <= 0 || a.max_q <= 0 || a.max_q > a.total_q ||
        a.logical_pages <= 0 || a.table_rows <= 0 || a.physical_pages <= 0 || a.q == nullptr ||
        a.out == nullptr || a.k_pages == nullptr || a.v_pages == nullptr ||
        a.block_tables == nullptr || a.q_offsets == nullptr || a.q_lengths == nullptr ||
        a.kv_lengths == nullptr || a.kv_rows == nullptr || !(a.scale > 0.0f)) {
        throw std::invalid_argument("gqa_fa3: invalid launch");
    }
    if (workspace_capacity < workspace_bytes(a.q_heads, a.total_q, a.segments)) {
        throw std::invalid_argument("gqa_fa3: workspace smaller than workspace_bytes()");
    }

    auto* lse = static_cast<float*>(workspace);
    auto* scheduler = reinterpret_cast<int*>(static_cast<char*>(workspace) +
        align_up(static_cast<std::size_t>(a.q_heads) * a.total_q * sizeof(float)));
    const std::int32_t b_rounded = rounded_segments(a.segments);

    Flash_fwd_params p;
    std::memset(&p, 0, sizeof(p));
    p.is_bf16 = true;
    p.q_ptr   = const_cast<void*>(a.q);
    p.k_ptr   = const_cast<void*>(a.k_pages);
    p.v_ptr   = const_cast<void*>(a.v_pages);
    p.o_ptr   = a.out;
    // Strides in elements. Varlen queries carry no batch stride; the cache's "batch" is the page.
    p.q_row_stride   = static_cast<std::int64_t>(a.q_heads) * kHeadDim;
    p.q_head_stride  = kHeadDim;
    p.o_row_stride   = p.q_row_stride;
    p.o_head_stride  = kHeadDim;
    p.k_row_stride   = kHeadDim;
    p.v_row_stride   = kHeadDim;
    p.k_head_stride  = static_cast<std::int64_t>(kPageSize) * kHeadDim;
    p.v_head_stride  = p.k_head_stride;
    p.k_batch_stride = static_cast<std::int64_t>(a.kv_heads) * kPageSize * kHeadDim;
    p.v_batch_stride = p.k_batch_stride;
    p.v_dim_stride   = 1;
    p.softmax_lse_ptr = lse;

    p.b          = a.segments;
    p.b_k        = a.table_rows;
    p.h          = a.q_heads;
    p.h_k        = a.kv_heads;
    p.d          = kHeadDim;
    p.d_rounded  = kHeadDim;
    p.dv         = kHeadDim;
    p.dv_rounded = kHeadDim;
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

    p.is_causal         = true;
    p.is_local          = false;
    p.window_size_left  = -1;
    p.window_size_right = 0;
    p.num_splits        = 1;
    p.pack_gqa          = true;

    p.varlen_sort_batches   = true;  // !is_local: the scheduler's Sort
    p.head_swizzle          = true;  // causal: the scheduler's LPT order
    p.num_splits_dynamic_ptr = scheduler;
    p.num_m_blocks_ptr       = scheduler + b_rounded;
    p.varlen_batch_idx_ptr   = scheduler + b_rounded * 2;
    p.num_nheads_in_l2_ptr   = scheduler + b_rounded * 3;
    p.tile_count_semaphore   = scheduler + b_rounded * 4;
    p.tile_count_semaphore_offset = b_rounded * 4;
    // The prepare kernel runs every launch: it zeroes the tile counter and reads the device
    // lengths, which is what lets a captured graph replay with new ones.
    p.skip_scheduler_metadata_computation = false;
    p.prepare_varlen_pdl = false;
    p.arch   = kArch;
    p.num_sm = hardware().sm_count;

    run_flash_fwd<kArch, kHeadDim, kHeadDim, 1, Element, Element, /*Is_causal=*/true,
                  /*Is_local=*/false, /*Has_softcap=*/false, /*Varlen=*/true,
                  /*PagedKVNonTMA=*/true, /*AppendKV=*/false, /*HasQv=*/false,
                  /*PackGQA=*/true, /*Split=*/false, /*V_colmajor=*/false>(p, stream);
}

} // namespace sinfer::ops::detail::gqa_fa3
