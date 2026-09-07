// Sparse-MoE prefill kernels, instantiated per geometry by sparse_moe_prefill_kernels.cu; the
// including namespace defines kGeometry, kHidden, kExperts, kRouterRows, kTopK, kIntermediate.

namespace {

/// The stored width of a row, against the width the math reads.
///
/// `row-split-k128-v1` pads K up to 128 and stores `padded / group` groups per row, so a
/// 704-wide expert's scale row is 48 bytes and not 44. Every mixture registered before Gemma 4
/// had widths that were multiples of 128, so the two coincided and every stride here could be
/// written against the logical width. They no longer coincide, and the failure is not a wrong
/// number: at 22 groups the second row's scales land 4 bytes short of where they are, the
/// `groups_per_row % 8` predicate picks the 8-byte staging path, and the load faults on an
/// address that is only 4-byte aligned. At the stored 24 the 16-byte path is taken and the
/// addresses are right -- correct *and* the faster branch.
///
/// Strides use this. Loop bounds keep the logical width, so the zero padding is never read.
/// The GGML superblock layouts store K exactly and are left alone.
constexpr int kStoredIntermediate = ((kIntermediate + 127) / 128) * 128;
constexpr int kStoredHidden       = ((kHidden + 127) / 128) * 128;


constexpr int kSparseMoeRouterScoreRows = (kRouterRows + 3) / 4 * 4;
constexpr int kRouterBM      = 16;
constexpr int kRouterBN      = 64;
constexpr int kRouterBK      = 64;
constexpr int kRouterStages  = 2;
constexpr int kRouterWarps   = 8;
constexpr int kRouterThreads = 32 * kRouterWarps;

__global__ __launch_bounds__(kRouterThreads, 2) void sparse_moe_prefill_router_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ scores, int tokens) {
    __shared__ __align__(16) __nv_bfloat16 Ws[kRouterStages][kRouterBM * kRouterBK];
    __shared__ __align__(16) __nv_bfloat16 Xs[kRouterStages][kRouterBN * kRouterBK];

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int row0   = static_cast<int>(blockIdx.x) * kRouterBM;
    const int token0 = static_cast<int>(blockIdx.y) * kRouterBN;

    float acc[4] = {};

    auto stage_inputs = [&](int stage, int kt) {
        const int k0 = kt * kRouterBK;
        for (int item = tid; item < kRouterBM * (kRouterBK / 8); item += kRouterThreads) {
            const int row   = item / (kRouterBK / 8);
            const int k8    = item - row * (kRouterBK / 8);
            const int rr    = row0 + row;
            auto* dst       = &Ws[stage][row * kRouterBK + gemm_swz64(row, k8 * 8)];
            const auto* src = weight +
                              static_cast<std::int64_t>(rr < kRouterRows ? rr : 0) * kHidden + k0 +
                              k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, rr < kRouterRows ? 16 : 0);
        }
        for (int item = tid; item < kRouterBN * (kRouterBK / 8); item += kRouterThreads) {
            const int col = item / (kRouterBK / 8);
            const int k8  = item - col * (kRouterBK / 8);
            const int tt  = token0 + col;
            auto* dst     = &Xs[stage][col * kRouterBK + gemm_swz64(col, k8 * 8)];
            const auto* src =
                x + static_cast<std::int64_t>(tt < tokens ? tt : 0) * kHidden + k0 + k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, tt < tokens ? 16 : 0);
        }
    };

#pragma unroll
    for (int stage = 0; stage < kRouterStages; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int kt = 0; kt < kHidden / kRouterBK; ++kt) {
        const int stage = kt & 1;
        cp_wait<kRouterStages - 1>();
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < kRouterBK / 16; ++ki) {
            unsigned af[4];
            unsigned bf[2];
            const int arow = a_rowoff;
            const int acol = ki * 16 + a_coloff;
            const int brow = warp * 8 + b_rin;
            const int bcol = ki * 16 + b_koff;
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(&Ws[stage][arow * kRouterBK + gemm_swz64(arow, acol)]));
            ldmatrix_x2(bf[0], bf[1],
                        smem_addr(&Xs[stage][brow * kRouterBK + gemm_swz64(brow, bcol)]));
            mma_bf16(acc[0], acc[1], acc[2], acc[3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
        }

        __syncthreads();
        const int next = kt + kRouterStages;
        if (next < kHidden / kRouterBK) { stage_inputs(stage, next); }
        cp_commit();
    }
    cp_wait<0>();

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int r0  = row0 + gid;
    const int r1  = r0 + 8;
    const int c0  = token0 + warp * 8 + 2 * lid;
    const int c1  = c0 + 1;
    if (r0 < kRouterRows && c0 < tokens) {
        scores[static_cast<std::int64_t>(c0) * kSparseMoeRouterScoreRows + r0] = acc[0];
    }
    if (r0 < kRouterRows && c1 < tokens) {
        scores[static_cast<std::int64_t>(c1) * kSparseMoeRouterScoreRows + r0] = acc[1];
    }
    if (r1 < kRouterRows && c0 < tokens) {
        scores[static_cast<std::int64_t>(c0) * kSparseMoeRouterScoreRows + r1] = acc[2];
    }
    if (r1 < kRouterRows && c1 < tokens) {
        scores[static_cast<std::int64_t>(c1) * kSparseMoeRouterScoreRows + r1] = acc[3];
    }
}

__global__ void sparse_moe_prefill_select_count_kernel(const float* __restrict__ scores,
                                                       int* __restrict__ ids,
                                                       float* __restrict__ alpha,
                                                       float* __restrict__ shared_scale,
                                                       int* __restrict__ local_rank,
                                                       int* __restrict__ tile_counts, int tokens,
                                                       const float* __restrict__ router_bias,
                                                       const float* __restrict__ per_expert_scale) {
    __shared__ int counts[kExperts];
    __shared__ float selected_logits[kSparseMoeRouteTileTokens][kTopK];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    for (int expert = tid; expert < kExperts; expert += kRouterThreads) { counts[expert] = 0; }
    __syncthreads();

    const int token = static_cast<int>(blockIdx.x) * kSparseMoeRouteTileTokens + warp;
    if (token < tokens) {
        sparse_moe_select_top_k_warp<kExperts, kTopK, kHasShared, kGating, kSharedGated,
                                     kPerExpertScaled>(
            scores + static_cast<std::int64_t>(token) * kSparseMoeRouterScoreRows,
            ids + token * kTopK, alpha + token * kTopK, shared_scale + token,
            selected_logits[warp], router_bias, kRoutedScale, per_expert_scale);
        __syncwarp();
        if (lane == 0) {
#pragma unroll
            for (int rank = 0; rank < kTopK; ++rank) {
                const int assignment   = token * kTopK + rank;
                local_rank[assignment] = atomicAdd(&counts[ids[assignment]], 1);
            }
        }
    }
    __syncthreads();
    for (int expert = tid; expert < kExperts; expert += kRouterThreads) {
        tile_counts[static_cast<std::int64_t>(blockIdx.x) * kExperts + expert] = counts[expert];
    }
}

__global__ void sparse_moe_prefill_scan_kernel(const int* __restrict__ tile_counts,
                                               int* __restrict__ tile_bases,
                                               int* __restrict__ expert_offsets,
                                               int* __restrict__ route_job_experts,
                                               int* __restrict__ route_job_columns,
                                               int* __restrict__ route_job_count, int route_tiles,
                                               int job_bn, int tokens, bool adaptive) {
    __shared__ int scan[kExperts];
    const int expert = static_cast<int>(threadIdx.x);
    int count        = 0;
    for (int tile = 0; tile < route_tiles; ++tile) {
        count += tile_counts[static_cast<std::int64_t>(tile) * kExperts + expert];
    }
    scan[expert] = count;
    __syncthreads();

#pragma unroll
    for (int offset = 1; offset < kExperts; offset <<= 1) {
        const int add = expert >= offset ? scan[expert - offset] : 0;
        __syncthreads();
        scan[expert] += add;
        __syncthreads();
    }

    const int base         = expert == 0 ? 0 : scan[expert - 1];
    expert_offsets[expert] = base;
    if (expert == kExperts - 1) { expert_offsets[kExperts] = scan[expert]; }

    int cursor = base;
    for (int tile = 0; tile < route_tiles; ++tile) {
        const std::int64_t index = static_cast<std::int64_t>(tile) * kExperts + expert;
        tile_bases[index]        = cursor;
        cursor += tile_counts[index];
    }

    scan[expert] = (count + job_bn - 1) / job_bn;
    __syncthreads();
#pragma unroll
    for (int offset = 1; offset < kExperts; offset <<= 1) {
        const int add = expert >= offset ? scan[expert - offset] : 0;
        __syncthreads();
        scan[expert] += add;
        __syncthreads();
    }
    const int job_begin = expert == 0 ? 0 : scan[expert - 1];
    const int jobs      = (count + job_bn - 1) / job_bn;
    for (int job = 0; job < jobs; ++job) {
        route_job_experts[job_begin + job] = expert;
        route_job_columns[job_begin + job] = job * job_bn;
    }
    if (expert == kExperts - 1) {
        const int jobs = scan[expert];
        // Above 3.5 grouped jobs per token, fixed token work wins by avoiding sparse expert tiles.
        *route_job_count = adaptive && jobs * 2 > 7 * tokens ? -jobs : jobs;
    }
}

template <bool Adaptive>
__global__ void
sparse_moe_prefill_gather_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
                                 int* __restrict__ packed_index, const int* __restrict__ tile_bases,
                                 __nv_bfloat16* __restrict__ gathered,
                                 const int* __restrict__ route_job_count,
                                 int* __restrict__ column_token) {
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int assignment = static_cast<int>(blockIdx.x);
    const int token      = assignment / kTopK;
    const int expert     = ids[assignment];
    const int tile       = token / kSparseMoeRouteTileTokens;
    const int packed =
        tile_bases[static_cast<std::int64_t>(tile) * kExperts + expert] + packed_index[assignment];
    // The int8 route quantises the input once per token and has its kernel gather by this
    // map instead, so it passes no destination for the copy.
    if (gathered != nullptr) {
        const int k       = static_cast<int>(threadIdx.x) * 8;
        const uint4 value = load_vec<uint4>(x + static_cast<std::int64_t>(token) * kHidden + k);
        store_vec(gathered + static_cast<std::int64_t>(packed) * kHidden + k, value);
    }
    if (threadIdx.x == 0) {
        packed_index[assignment] = packed;
        if (column_token != nullptr) { column_token[packed] = token; }
    }
}

constexpr int kExpertBM                = 64;
constexpr int kExpertBN                = 64;
constexpr int kExpertBK                = 64;
constexpr int kExpertStages            = 2;
static_assert(kExpertBM == ggml::kI8BM && kExpertBK == ggml::kI8BK && kExpertStages == ggml::kI8Stages,
              "the int8 tile's geometry is the expert body's");
constexpr int kExpertWarps             = 8;
constexpr int kExpertThreads           = 32 * kExpertWarps;
constexpr int kPrefillBlocksPerSm      = 3;
// Marlin's MoE block: 16 rows, the smallest that is not the m_block_size_8 path.
constexpr int kMarlinMoeBlock          = 16;

// The routed GEMMs stream every touched expert once and the bench puts them at ~30 % of peak
// bandwidth, so how many requests are in flight matters more than anything else here. The
// persistent grid is the knob for that, tunable so it can be swept rather than guessed.
// MoE Marlin residency (#89). The routed gate/up is 53 % of a MoE layer and its kernel is
// issue-bound, so vLLM's tuned Marlin is worth the repack - and our Q4G64 codes are already
// kU4B8 with group 64, so this regroups bytes rather than requantising. Planes are built on
// first use and keyed by the weight pointer.
struct MarlinMoePlane {
    void* b_tiles                     = nullptr;
    void* scales                      = nullptr;
    void* product                     = nullptr;
    void* c_tmp                       = nullptr;
    std::int32_t* sorted_token_ids    = nullptr;
    std::int32_t* expert_ids          = nullptr;
    std::int32_t* past_padded         = nullptr;
    std::int32_t* locks               = nullptr;
    std::int32_t capacity_assignments = 0;
};

MarlinMoePlane& marlin_moe_plane_for(const Weight& weight, std::int32_t experts, std::int32_t n,
                                     std::int32_t k, std::int32_t assignments,
                                     cudaStream_t stream) {
    static std::mutex mutex;
    static std::unordered_map<const void*, MarlinMoePlane> planes;
    const std::lock_guard<std::mutex> lock(mutex);
    MarlinMoePlane& plane = planes[weight.qdata];
    if (plane.b_tiles == nullptr) {
        const std::size_t b_bytes = detail::marlin_moe_b_bytes(n, k) * experts;
        const std::size_t s_bytes = detail::marlin_moe_scale_bytes(n, k) * experts;
        void* gptq_tmp            = nullptr;
        CUDA_CHECK(cudaMalloc(&plane.b_tiles, b_bytes));
        CUDA_CHECK(cudaMalloc(&plane.scales, s_bytes));
        CUDA_CHECK(cudaMalloc(&gptq_tmp, static_cast<std::size_t>(k / 8) * n * sizeof(std::uint32_t)));
        detail::marlin_moe_repack_q4g64(weight.qdata, weight.scales, experts, n, k, gptq_tmp,
                                        plane.b_tiles, plane.scales, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(gptq_tmp));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&plane.locks),
                              detail::marlin_moe_lock_bytes()));
        CUDA_CHECK(cudaMemsetAsync(plane.locks, 0, detail::marlin_moe_lock_bytes(), stream));
    }
    if (assignments > plane.capacity_assignments) {
        if (plane.product != nullptr) {
            CUDA_CHECK(cudaFree(plane.product));
            CUDA_CHECK(cudaFree(plane.c_tmp));
            CUDA_CHECK(cudaFree(plane.sorted_token_ids));
            CUDA_CHECK(cudaFree(plane.expert_ids));
            CUDA_CHECK(cudaFree(plane.past_padded));
        }
        const std::int32_t padded =
            detail::marlin_moe_padded_rows(assignments, experts, kMarlinMoeBlock);
        CUDA_CHECK(cudaMalloc(&plane.product,
                              static_cast<std::size_t>(assignments) * n * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&plane.c_tmp, detail::marlin_moe_c_tmp_bytes(assignments, n)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&plane.sorted_token_ids),
                              static_cast<std::size_t>(padded) * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&plane.expert_ids),
                              static_cast<std::size_t>(padded / kMarlinMoeBlock + 1) *
                                  sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&plane.past_padded), sizeof(std::int32_t)));
        plane.capacity_assignments = assignments;
    }
    return plane;
}

inline int prefill_persistent_blocks() {
    static const int blocks = [] {
        int device = 0;
        int sms    = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        const char* raw = std::getenv("SUROGATE_SERVE_MOE_BLOCKS_PER_SM");
        if (raw == nullptr || *raw == '\0') { return kPrefillBlocksPerSm * sms; }
        const long parsed = std::strtol(raw, nullptr, 10);
        if (parsed <= 0 || parsed > 32) { return kPrefillBlocksPerSm * sms; }
        return static_cast<int>(parsed) * sms;
    }();
    return blocks;
}

template <int ExpertWarps, int ExpertBN>
__global__ __launch_bounds__(ExpertWarps * 32, 3) void sparse_moe_prefill_q4_gate_up_kernel(
    const __nv_bfloat16* __restrict__ gathered, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ slot_of_expert,
    const int* __restrict__ route_job_count, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ activation) {
    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int GroupsPerRow  = kStoredHidden / 64;
    constexpr int WarpCols      = ExpertBN / ExpertWarps;
    constexpr int WarpNT        = WarpCols / 8;
    static_assert(ExpertBN % ExpertWarps == 0 && WarpCols % 8 == 0);
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][ExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertStages][kExpertBM * 32];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * GroupsPerRow * 2];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int a_mat          = lane >> 3;
    const int a_rin          = lane & 7;
    const int a_rowoff       = a_rin + ((a_mat & 1) << 3);
    const int a_coloff       = (a_mat >> 1) << 3;
    const int b_rin          = lane & 7;
    const int b_koff         = ((lane >> 3) & 1) << 3;
    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = kIntermediate / (kExpertBM / 2);
    const int total_work     = *route_job_count * row_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job     = work / row_blocks;
        const int row_block     = work - route_job * row_blocks;
        const int expert        = route_job_experts[route_job];
        const int block = slot_of_expert != nullptr ? slot_of_expert[expert] : expert;
        if (block < 0) { continue; } // computed on the host (expert split); uniform per CTA
        const int logical0      = row_block * (kExpertBM / 2);
        const int begin         = expert_offsets[expert];
        const int count         = expert_offsets[expert + 1] - begin;
        const int column_base   = route_job_columns[route_job];
        const int cols          = count - column_base < ExpertBN ? count - column_base : ExpertBN;
        float acc[4][WarpNT][4] = {};

        auto global_row = [&](int local_row) {
            const int logical = logical0 + (local_row & (kExpertBM / 2 - 1));
            return block * (2 * kIntermediate) + logical +
                   (local_row >= kExpertBM / 2 ? kIntermediate : 0);
        };

        auto stage_scales = [&] {
            constexpr int ScaleBytesPerRow = GroupsPerRow * 2;
            constexpr int ChunksPerRow     = ScaleBytesPerRow / 16;
            for (int item = tid; item < kExpertBM * ChunksPerRow; item += ExpertThreads) {
                const int row   = item / ChunksPerRow;
                const int chunk = item - row * ChunksPerRow;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * GroupsPerRow + chunk * 8;
                cp_async<16, Cache::cg>(&Sr[row * ScaleBytesPerRow + chunk * 16], &scales[gi * 2]);
            }
        };

        auto stage_inputs = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < ExpertBN * (kExpertBK / 8); item += ExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int packed_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    gathered +
                    static_cast<std::int64_t>(col < cols ? packed_col : begin) * kHidden + k0 +
                    k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }

            const int group = kt;
            for (int item = tid; item < kExpertBM * 2; item += ExpertThreads) {
                const int row  = item >> 1;
                const int half = item & 1;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * GroupsPerRow + group;
                cp_async<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                        &codes[gi * 32 + half * 16]);
            }
        };

        auto decode_weight = [&](int stage) {
            for (int row = warp; row < kExpertBM; row += ExpertWarps) {
                const std::uint8_t packed = Cr[stage][row * 32 + lane];
                const int q0              = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
                const int q1              = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
                const __nv_bfloat162 value =
                    __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
                store_vec(&As[row * kExpertBK + gemm_swz64(row, 2 * lane)], value);
            }
        };

        stage_scales();
        cp_commit();
#pragma unroll
        for (int stage = 0; stage < kExpertStages; ++stage) {
            stage_inputs(stage, stage);
            cp_commit();
        }

#pragma unroll 1
        for (int kt = 0; kt < kHidden / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<kExpertStages - 1>();
            __syncthreads();
            decode_weight(stage);
            __syncthreads();

            if (warp * WarpCols < cols) {
                float partial[4][WarpNT][4] = {};
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[4][4];
                    unsigned bf[WarpNT][2];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        const int brow = warp * WarpCols + ni * 8 + b_rin;
                        const int bcol = ki * 16 + b_koff;
                        ldmatrix_x2(
                            bf[ni][0], bf[ni][1],
                            smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi) {
                            mma_bf16(partial[mi][ni][0], partial[mi][ni][1], partial[mi][ni][2],
                                     partial[mi][ni][3], af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                     bf[ni][0], bf[ni][1]);
                        }
                    }
                }
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int row0 = mi * 16 + gid;
                    const int row1 = row0 + 8;
                    float scale0 =
                        lid == 0
                            ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                                  &Sr[(row0 * GroupsPerRow + kt) * 2])))
                            : 0.0f;
                    float scale1 =
                        lid == 0
                            ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                                  &Sr[(row1 * GroupsPerRow + kt) * 2])))
                            : 0.0f;
                    scale0 = __shfl_sync(0xffffffffu, scale0, gid * 4);
                    scale1 = __shfl_sync(0xffffffffu, scale1, gid * 4);
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        acc[mi][ni][0] = fmaf(partial[mi][ni][0], scale0, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(partial[mi][ni][1], scale0, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(partial[mi][ni][2], scale1, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(partial[mi][ni][3], scale1, acc[mi][ni][3]);
                    }
                }
            }

            __syncthreads();
            const int next = kt + kExpertStages;
            if (next < kHidden / kExpertBK) { stage_inputs(stage, next); }
            cp_commit();
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * WarpCols < cols) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0 = logical0 + mi * 16 + gid;
                const int row1 = row0 + 8;
#pragma unroll
                for (int ni = 0; ni < WarpNT; ++ni) {
                    const int col0       = begin + column_base + warp * WarpCols + ni * 8 + 2 * lid;
                    const int col1       = col0 + 1;
                    const int local_col0 = warp * WarpCols + ni * 8 + 2 * lid;
                    const int local_col1 = local_col0 + 1;
                    if (local_col0 < cols) {
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                            __float2bfloat16_rn(gated_clamped(acc[mi][ni][0], acc[mi + 2][ni][0], kSwigluLimit, kActivation));
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row1] =
                            __float2bfloat16_rn(gated_clamped(acc[mi][ni][2], acc[mi + 2][ni][2], kSwigluLimit, kActivation));
                    }
                    if (local_col1 < cols) {
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row0] =
                            __float2bfloat16_rn(gated_clamped(acc[mi][ni][1], acc[mi + 2][ni][1], kSwigluLimit, kActivation));
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                            __float2bfloat16_rn(gated_clamped(acc[mi][ni][3], acc[mi + 2][ni][3], kSwigluLimit, kActivation));
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <bool Routed, bool Adaptive = false>
__global__ __launch_bounds__(kExpertThreads, 1) void sparse_moe_prefill_w8_gate_up_kernel(
    const __nv_bfloat16* __restrict__ input, const int* __restrict__ expert_offsets,
    const int* __restrict__ slot_of_expert, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ activation, int tokens, const int* __restrict__ route_job_count) {
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][kExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertBM * kExpertBK];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * 16];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int expert   = Routed ? static_cast<int>(blockIdx.y) : 0;
    const int block  = (Routed && slot_of_expert != nullptr) ? slot_of_expert[expert] : expert;
    if (block < 0) { return; } // computed on the host (expert split)
    const int logical0 = static_cast<int>(blockIdx.x) * (kExpertBM / 2);
    const int begin    = Routed ? expert_offsets[expert] : static_cast<int>(blockIdx.y) * kExpertBN;
    const int count    = Routed ? expert_offsets[expert + 1] - begin
                                : (tokens - begin < kExpertBN ? tokens - begin : kExpertBN);
    if (count <= 0) { return; }

    const int a_mat              = lane >> 3;
    const int a_rin              = lane & 7;
    const int a_rowoff           = a_rin + ((a_mat & 1) << 3);
    const int a_coloff           = (a_mat >> 1) << 3;
    const int b_rin              = lane & 7;
    const int b_koff             = ((lane >> 3) & 1) << 3;
    const int gid                = lane >> 2;
    const int lid                = lane & 3;
    constexpr int groups_per_row = kStoredHidden / 32;

    const int column_iterations = Routed ? (count + kExpertBN - 1) / kExpertBN : 1;
    for (int iteration = 0; iteration < column_iterations; ++iteration) {
        const int column_base = iteration * kExpertBN;
        const int cols        = count - column_base < kExpertBN ? count - column_base : kExpertBN;
        float acc[4][4]       = {};

        auto global_row = [&](int local_row) {
            const int logical = logical0 + (local_row & (kExpertBM / 2 - 1));
            const int row     = logical + (local_row >= kExpertBM / 2 ? kIntermediate : 0);
            return (Routed ? block * (2 * kIntermediate) : 0) + row;
        };

        auto stage_x = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < kExpertBN * (kExpertBK / 8); item += kExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int source_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    input + static_cast<std::int64_t>(col < cols ? source_col : begin) * kHidden +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }
        };

        auto stage_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int group0                = kt * groups_per_tile;
            for (int item = tid; item < kExpertBM * (kExpertBK / 16); item += kExpertThreads) {
                const int row   = item / (kExpertBK / 16);
                const int chunk = item - row * (kExpertBK / 16);
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * groups_per_row + group0;
                cp_async<16, Cache::cg>(&Cr[row * kExpertBK + chunk * 16],
                                        &codes[gi * 32 + chunk * 16]);
            }
            if ((kt % scale_cache_tiles) == 0) {
                for (int row = tid; row < kExpertBM; row += kExpertThreads) {
                    const std::int64_t gi =
                        static_cast<std::int64_t>(global_row(row)) * groups_per_row + group0;
                    if constexpr (groups_per_row % 8 == 0) {
                        cp_async<16, Cache::cg>(&Sr[row * 16], &scales[gi * 2]);
                    } else {
                        // See the down kernel: an 8-byte-aligned scale row is staged in pieces.
                        cp_async<8>(&Sr[row * 16], &scales[gi * 2]);
                        if (group0 + 8 <= groups_per_row) {
                            cp_async<8>(&Sr[row * 16 + 8], &scales[gi * 2 + 8]);
                        }
                    }
                }
            }
        };

        auto decode_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int scale_offset          = (kt % scale_cache_tiles) * groups_per_tile * 2;
            const int half                  = lane >> 4;
            const int half_lane             = lane & 15;
            for (int row_pair = warp * 2; row_pair < kExpertBM; row_pair += kExpertWarps * 2) {
                const int row = row_pair + half;
                unsigned scale_pair =
                    half_lane == 0
                        ? *reinterpret_cast<const std::uint32_t*>(&Sr[row * 16 + scale_offset])
                        : 0;
                scale_pair = __shfl_sync(0xffffffffu, scale_pair, half * 16);
#pragma unroll
                for (int group = 0; group < groups_per_tile; ++group) {
                    const float scale = __half2float(__ushort_as_half(
                        static_cast<std::uint16_t>((scale_pair >> (group * 16)) & 0xffffu)));
                    const int col     = group * 32 + half_lane * 2;
                    const std::uint16_t packed =
                        *reinterpret_cast<const std::uint16_t*>(&Cr[row * kExpertBK + col]);
                    const int q0 = static_cast<int>(static_cast<std::int8_t>(packed & 0xffu));
                    const int q1 = static_cast<int>(static_cast<std::int8_t>(packed >> 8));
                    const __nv_bfloat162 value = __floats2bfloat162_rn(
                        static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
                    store_vec(&As[row * kExpertBK + gemm_swz64(row, col)], value);
                }
            }
        };

        stage_x(0, 0);
        stage_weight(0);
        cp_commit();

#pragma unroll 4
        for (int kt = 0; kt < kHidden / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<0>();
            __syncthreads();
            decode_weight(kt);
            __syncthreads();

            const int next = kt + 1;
            if (next < kHidden / kExpertBK) {
                stage_x(next & 1, next);
                stage_weight(next);
                cp_commit();
            }

            if (warp * 8 < cols) {
                unsigned af[2][4][4];
                unsigned bf[2][2];
                auto load_fragments = [&](int slot, int ki) {
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[slot][mi][0], af[slot][mi][1], af[slot][mi][2],
                                    af[slot][mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[slot][0], bf[slot][1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
                };
                load_fragments(0, 0);
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    const int slot = ki & 1;
                    if (ki + 1 < kExpertBK / 16) { load_fragments(slot ^ 1, ki + 1); }
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[slot][mi][0],
                                 af[slot][mi][1], af[slot][mi][2], af[slot][mi][3], bf[slot][0],
                                 bf[slot][1]);
                    }
                }
            }
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0       = logical0 + mi * 16 + gid;
                const int row1       = row0 + 8;
                const int col0       = begin + column_base + warp * 8 + 2 * lid;
                const int col1       = col0 + 1;
                const int local_col0 = warp * 8 + 2 * lid;
                const int local_col1 = local_col0 + 1;
                if (local_col0 < cols) {
                    activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                        __float2bfloat16_rn(gated_clamped(acc[mi][0], acc[mi + 2][0], kSwigluLimit, kActivation));
                    activation[static_cast<std::int64_t>(col0) * kIntermediate + row1] =
                        __float2bfloat16_rn(gated_clamped(acc[mi][2], acc[mi + 2][2], kSwigluLimit, kActivation));
                }
                if (local_col1 < cols) {
                    activation[static_cast<std::int64_t>(col1) * kIntermediate + row0] =
                        __float2bfloat16_rn(gated_clamped(acc[mi][1], acc[mi + 2][1], kSwigluLimit, kActivation));
                    activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                        __float2bfloat16_rn(gated_clamped(acc[mi][3], acc[mi + 2][3], kSwigluLimit, kActivation));
                }
            }
        }
        __syncthreads();
    }
}


// ---------------------------------------------------------------------------------------------
// GGML K-quant routed experts. Same tiling, staging and MMA as the row-split kernels above; what
// differs is where a weight's scale lives. A row-split group is 64 values with one scale, so
// those kernels feed the MMA raw integer codes and scale the accumulator afterwards. A K-quant's
// 64-wide tile spans two 32-value sub-blocks (four 16-value ones for Q6_K) with different scales,
// and Q4_K/Q5_K are affine -- their min term depends on the activation sum over the sub-block and
// cannot be folded into a post-MMA multiply at all. So these dequantise inside `decode_weight`
// and let the MMA accumulate finished BF16, which also makes the epilogue shorter.
// ---------------------------------------------------------------------------------------------

/// Staging a row's tile: sixteen-byte `cp_async` where the block size allows it, scalar pairs for
/// Q6_K, whose 210-byte block leaves consecutive blocks only two-byte aligned.
/// Launch a GGML prefill codec on the route its format has. `Int8` is a template parameter so
/// the discarded branch is never instantiated: the int8 tensor-core kernels ask a codec for
/// scale groups and code unpacking that only the three hand codecs have.
template <class Codec, bool Int8, class I8Launch, class Bf16Launch>
inline void launch_ggml_prefill_codec(I8Launch& i8, Bf16Launch& bf16, bool int8_route) {
    if constexpr (Int8) {
        if (int8_route) { i8.template operator()<Codec>(); }
        else { bf16.template operator()<Codec>(); }
    } else {
        (void)i8;
        (void)int8_route;
        bf16.template operator()<Codec>();
    }
}

template <class Codec, int ExpertWarps, int ExpertBN>
__global__ __launch_bounds__(ExpertWarps * 32, Codec::kMinBlocks) void
sparse_moe_prefill_ggml_gate_up_kernel(
    const __nv_bfloat16* __restrict__ gathered, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ slot_of_expert, const int* __restrict__ route_job_count,
    const std::uint8_t* __restrict__ blocks, __nv_bfloat16* __restrict__ activation) {
    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int BlocksPerRow  = kHidden / ggml::QK_K;
    constexpr int TileUnits     = Codec::kCpAsync ? Codec::kTileBytes / 16 : Codec::kTileBytes / 2;
    constexpr int HeaderUnits =
        Codec::kCpAsync ? Codec::kHeaderBytes / 16 : Codec::kHeaderBytes / 2;
    constexpr int WarpCols      = ExpertBN / ExpertWarps;
    constexpr int WarpNT        = WarpCols / 8;
    static_assert(ExpertBN % ExpertWarps == 0 && WarpCols % 8 == 0);
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][ExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertStages][kExpertBM * Codec::kTileBytes];
    __shared__ __align__(16) std::uint8_t Hdr[kExpertStages][kExpertBM * Codec::kHeaderBytes];
    __shared__ std::uint8_t Off[kExpertStages][kExpertBM]; // a covering codec's per-row offset

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int a_mat          = lane >> 3;
    const int a_rin          = lane & 7;
    const int a_rowoff       = a_rin + ((a_mat & 1) << 3);
    const int a_coloff       = (a_mat >> 1) << 3;
    const int b_rin          = lane & 7;
    const int b_koff         = ((lane >> 3) & 1) << 3;
    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = kIntermediate / (kExpertBM / 2);
    const int total_work     = *route_job_count * row_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job     = work / row_blocks;
        const int row_block     = work - route_job * row_blocks;
        const int expert        = route_job_experts[route_job];
        const int block = slot_of_expert != nullptr ? slot_of_expert[expert] : expert;
        if (block < 0) { continue; }
        const int logical0      = row_block * (kExpertBM / 2);
        const int begin         = expert_offsets[expert];
        const int count         = expert_offsets[expert + 1] - begin;
        const int column_base   = route_job_columns[route_job];
        const int cols          = count - column_base < ExpertBN ? count - column_base : ExpertBN;
        float acc[4][WarpNT][4] = {};

        auto global_row = [&](int local_row) {
            const int logical = logical0 + (local_row & (kExpertBM / 2 - 1));
            return block * (2 * kIntermediate) + logical +
                   (local_row >= kExpertBM / 2 ? kIntermediate : 0);
        };
        auto row_base = [&](int local_row) {
            return blocks + static_cast<std::int64_t>(global_row(local_row)) * BlocksPerRow *
                                Codec::kBlockBytes;
        };

        auto stage_inputs = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < ExpertBN * (kExpertBK / 8); item += ExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int packed_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    gathered +
                    static_cast<std::int64_t>(col < cols ? packed_col : begin) * kHidden + k0 +
                    k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }
            for (int item = tid; item < kExpertBM * TileUnits; item += ExpertThreads) {
                const int row  = item / TileUnits;
                const int unit = item - row * TileUnits;
                ggml::stage_ggml_tile<Codec>(&Cr[stage][row * Codec::kTileBytes], row_base(row), kt, unit);
            }
            // Four tiles share a superblock, so its header is staged when the superblock
            // changes rather than once per tile -- a third of this kernel's weight traffic
            // otherwise. The buffer is indexed by the superblock's parity, which differs from
            // the one being decoded exactly when a new superblock is being staged.
            if (kt % ggml::kTilesPerBlock == 0) {
                const int slot = (kt / ggml::kTilesPerBlock) & 1;
                for (int item = tid; item < kExpertBM * HeaderUnits; item += ExpertThreads) {
                    const int row  = item / HeaderUnits;
                    const int unit = item - row * HeaderUnits;
                    ggml::stage_ggml_header<Codec>(&Hdr[slot][row * Codec::kHeaderBytes], row_base(row),
                                             kt, unit);
                    if constexpr (Codec::kCover) {
                        if (unit == 0) {
                            Off[slot][row] = static_cast<std::uint8_t>(
                                ggml::cover_offset<Codec>(row_base(row), kt));
                        }
                    }
                }
            }
        };

        auto decode_weight = [&](int stage, int kt) {
            for (int row = warp; row < kExpertBM; row += ExpertWarps) {
                const int slot = (kt / ggml::kTilesPerBlock) & 1;
                const int off  = Codec::kCover ? Off[slot][row] : 0;
                const __nv_bfloat162 value = Codec::decode(
                    &Cr[stage][row * Codec::kTileBytes] + off,
                    &Hdr[slot][row * Codec::kHeaderBytes] + off, lane, kt % ggml::kTilesPerBlock);
                store_vec(&As[row * kExpertBK + gemm_swz64(row, 2 * lane)], value);
            }
        };

#pragma unroll
        for (int stage = 0; stage < kExpertStages; ++stage) {
            stage_inputs(stage, stage);
            cp_commit();
        }

#pragma unroll 1
        for (int kt = 0; kt < kHidden / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<kExpertStages - 1>();
            __syncthreads();
            decode_weight(stage, kt);
            __syncthreads();

            if (warp * WarpCols < cols) {
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[4][4];
                    unsigned bf[WarpNT][2];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        const int brow = warp * WarpCols + ni * 8 + b_rin;
                        const int bcol = ki * 16 + b_koff;
                        ldmatrix_x2(bf[ni][0], bf[ni][1],
                                    smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi) {
                            mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                                     af[mi][0], af[mi][1], af[mi][2], af[mi][3], bf[ni][0],
                                     bf[ni][1]);
                        }
                    }
                }
            }

            __syncthreads();
            const int next = kt + kExpertStages;
            if (next < kHidden / kExpertBK) { stage_inputs(stage, next); }
            cp_commit();
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * WarpCols < cols) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0 = logical0 + mi * 16 + gid;
                const int row1 = row0 + 8;
#pragma unroll
                for (int ni = 0; ni < WarpNT; ++ni) {
                    const int col0       = begin + column_base + warp * WarpCols + ni * 8 + 2 * lid;
                    const int col1       = col0 + 1;
                    const int local_col0 = warp * WarpCols + ni * 8 + 2 * lid;
                    const int local_col1 = local_col0 + 1;
                    if (local_col0 < cols) {
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                            __float2bfloat16_rn(gated_clamped(acc[mi][ni][0], acc[mi + 2][ni][0], kSwigluLimit, kActivation));
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row1] =
                            __float2bfloat16_rn(gated_clamped(acc[mi][ni][2], acc[mi + 2][ni][2], kSwigluLimit, kActivation));
                    }
                    if (local_col1 < cols) {
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row0] =
                            __float2bfloat16_rn(gated_clamped(acc[mi][ni][1], acc[mi + 2][ni][1], kSwigluLimit, kActivation));
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                            __float2bfloat16_rn(gated_clamped(acc[mi][ni][3], acc[mi + 2][ni][3], kSwigluLimit, kActivation));
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <class Codec, int ExpertWarps, int ExpertBN>
__global__ __launch_bounds__(ExpertWarps * 32, Codec::kMinBlocks) void
sparse_moe_prefill_ggml_down_kernel(
    const __nv_bfloat16* __restrict__ activation, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ slot_of_expert, const int* __restrict__ route_job_count,
    const std::uint8_t* __restrict__ blocks, __nv_bfloat16* __restrict__ output) {
    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int BlocksPerRow  = kIntermediate / ggml::QK_K;
    constexpr int TileUnits     = Codec::kCpAsync ? Codec::kTileBytes / 16 : Codec::kTileBytes / 2;
    constexpr int HeaderUnits =
        Codec::kCpAsync ? Codec::kHeaderBytes / 16 : Codec::kHeaderBytes / 2;
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][ExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertStages][kExpertBM * Codec::kTileBytes];
    __shared__ __align__(16) std::uint8_t Hdr[kExpertStages][kExpertBM * Codec::kHeaderBytes];
    __shared__ std::uint8_t Off[kExpertStages][kExpertBM]; // a covering codec's per-row offset

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int a_mat          = lane >> 3;
    const int a_rin          = lane & 7;
    const int a_rowoff       = a_rin + ((a_mat & 1) << 3);
    const int a_coloff       = (a_mat >> 1) << 3;
    const int b_rin          = lane & 7;
    const int b_koff         = ((lane >> 3) & 1) << 3;
    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = kHidden / kExpertBM;
    const int total_work     = *route_job_count * row_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job   = work / row_blocks;
        const int row_block   = work - route_job * row_blocks;
        const int expert      = route_job_experts[route_job];
        const int block = slot_of_expert != nullptr ? slot_of_expert[expert] : expert;
        if (block < 0) { continue; }
        const int row0        = row_block * kExpertBM;
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols        = count - column_base < ExpertBN ? count - column_base : ExpertBN;
        float acc[4][4]       = {};

        auto row_base = [&](int local_row) {
            const int global_row = block * kHidden + row0 + local_row;
            return blocks +
                   static_cast<std::int64_t>(global_row) * BlocksPerRow * Codec::kBlockBytes;
        };

        auto stage_inputs = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < ExpertBN * (kExpertBK / 8); item += ExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int packed_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    activation +
                    static_cast<std::int64_t>(col < cols ? packed_col : begin) * kIntermediate +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }
            for (int item = tid; item < kExpertBM * TileUnits; item += ExpertThreads) {
                const int row  = item / TileUnits;
                const int unit = item - row * TileUnits;
                ggml::stage_ggml_tile<Codec>(&Cr[stage][row * Codec::kTileBytes], row_base(row), kt, unit);
            }
            // Four tiles share a superblock, so its header is staged when the superblock
            // changes rather than once per tile -- a third of this kernel's weight traffic
            // otherwise. The buffer is indexed by the superblock's parity, which differs from
            // the one being decoded exactly when a new superblock is being staged.
            if (kt % ggml::kTilesPerBlock == 0) {
                const int slot = (kt / ggml::kTilesPerBlock) & 1;
                for (int item = tid; item < kExpertBM * HeaderUnits; item += ExpertThreads) {
                    const int row  = item / HeaderUnits;
                    const int unit = item - row * HeaderUnits;
                    ggml::stage_ggml_header<Codec>(&Hdr[slot][row * Codec::kHeaderBytes], row_base(row),
                                             kt, unit);
                    if constexpr (Codec::kCover) {
                        if (unit == 0) {
                            Off[slot][row] = static_cast<std::uint8_t>(
                                ggml::cover_offset<Codec>(row_base(row), kt));
                        }
                    }
                }
            }
        };

        auto decode_weight = [&](int stage, int kt) {
            for (int row = warp; row < kExpertBM; row += ExpertWarps) {
                const int slot = (kt / ggml::kTilesPerBlock) & 1;
                const int off  = Codec::kCover ? Off[slot][row] : 0;
                const __nv_bfloat162 value = Codec::decode(
                    &Cr[stage][row * Codec::kTileBytes] + off,
                    &Hdr[slot][row * Codec::kHeaderBytes] + off, lane, kt % ggml::kTilesPerBlock);
                store_vec(&As[row * kExpertBK + gemm_swz64(row, 2 * lane)], value);
            }
        };

#pragma unroll
        for (int stage = 0; stage < kExpertStages; ++stage) {
            stage_inputs(stage, stage);
            cp_commit();
        }

#pragma unroll 1
        for (int kt = 0; kt < kIntermediate / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<kExpertStages - 1>();
            __syncthreads();
            decode_weight(stage, kt);
            __syncthreads();

            if (warp * 8 < cols) {
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[4][4];
                    unsigned bf[2];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[mi][0],
                                 af[mi][1], af[mi][2], af[mi][3], bf[0], bf[1]);
                    }
                }
            }

            __syncthreads();
            const int next = kt + kExpertStages;
            if (next < kIntermediate / kExpertBK) { stage_inputs(stage, next); }
            cp_commit();
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int output_row0 = row0 + mi * 16 + gid;
                const int output_row1 = output_row0 + 8;
                const int col0        = begin + column_base + warp * 8 + 2 * lid;
                const int col1        = col0 + 1;
                const int local_col0  = warp * 8 + 2 * lid;
                const int local_col1  = local_col0 + 1;
                if (local_col0 < cols) {
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][0]);
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][2]);
                }
                if (local_col1 < cols) {
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][1]);
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][3]);
                }
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------
// The int8 tensor-core route for K-quant experts.
//
// The BF16 kernels above decode every weight value before the MMA and were measured
// compute-bound, with the tensor pipe running BF16 at a quarter of its int8 rate. Here the
// weights' own codes are the A operand -- unpacked from nibbles to int8 once per K tile,
// cooperatively, into a row-major tile -- and the activations arrive already quantised to int8
// per 32 values with a (scale, sum) pair each, which is what llama.cpp's MMQ consumes. The s8
// MMA yields an exact int32 dot product per 32-value group, and the group's scales apply to it
// in FP32:
//
//   acc += (d*sc)[row,g] * d_x[col,g] * dot - (dmin*m)[row,g] * sum_x[col,g]
//
template <class Codec, int ExpertWarps, int ExpertBN>
__global__ __launch_bounds__(ExpertWarps * 32, Codec::kMinBlocks) void
sparse_moe_prefill_ggml_i8_gate_up_kernel(
    const std::int8_t* __restrict__ codes, const __half2* __restrict__ ds,
    const int* __restrict__ column_token, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ slot_of_expert, const int* __restrict__ route_job_count,
    const std::uint8_t* __restrict__ blocks, __nv_bfloat16* __restrict__ activation) {
    constexpr int BlocksPerRow = kHidden / ggml::QK_K;
    constexpr int WarpCols     = ExpertBN / ExpertWarps;
    __shared__ ggml::GgmlI8Smem<Codec, ExpertBN> sm;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    constexpr int row_blocks = kIntermediate / (kExpertBM / 2);
    const int total_work     = *route_job_count * row_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job   = work / row_blocks;
        const int row_block   = work - route_job * row_blocks;
        const int expert      = route_job_experts[route_job];
        const int block       = slot_of_expert != nullptr ? slot_of_expert[expert] : expert;
        if (block < 0) { continue; }
        const int logical0    = row_block * (kExpertBM / 2);
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols        = count - column_base < ExpertBN ? count - column_base : ExpertBN;

        auto row_base = [&](int local_row) {
            const int logical = logical0 + (local_row & (kExpertBM / 2 - 1));
            const int row     = block * (2 * kIntermediate) + logical +
                                (local_row >= kExpertBM / 2 ? kIntermediate : 0);
            return blocks + static_cast<std::int64_t>(row) * BlocksPerRow * Codec::kBlockBytes;
        };
        auto act_row = [&](int local_col) {
            return static_cast<std::int64_t>(column_token[begin + column_base + local_col]);
        };
        float acc[4][4];
        // Every geometry that shares this body instantiates it; only those whose rows are whole
        // superblocks may run it, and their dispatch guards the launch.
        if constexpr (kHidden % ggml::QK_K != 0) { return; }
        ggml::ggml_i8_tile<Codec, ExpertWarps, ExpertBN>(sm, row_base, act_row, codes, ds, cols,
                                                          kHidden, acc);

        if (warp * WarpCols < cols) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0       = logical0 + mi * 16 + gid;
                const int row1       = row0 + 8;
                const int local_col0 = warp * WarpCols + 2 * lid;
                const int local_col1 = local_col0 + 1;
                const int col0       = begin + column_base + local_col0;
                const int col1       = col0 + 1;
                if (local_col0 < cols) {
                    activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                        __float2bfloat16_rn(gated_clamped(acc[mi][0], acc[mi + 2][0], kSwigluLimit, kActivation));
                    activation[static_cast<std::int64_t>(col0) * kIntermediate + row1] =
                        __float2bfloat16_rn(gated_clamped(acc[mi][2], acc[mi + 2][2], kSwigluLimit, kActivation));
                }
                if (local_col1 < cols) {
                    activation[static_cast<std::int64_t>(col1) * kIntermediate + row0] =
                        __float2bfloat16_rn(gated_clamped(acc[mi][1], acc[mi + 2][1], kSwigluLimit, kActivation));
                    activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                        __float2bfloat16_rn(gated_clamped(acc[mi][3], acc[mi + 2][3], kSwigluLimit, kActivation));
                }
            }
        }
        __syncthreads();
    }
}

template <class Codec, int ExpertWarps, int ExpertBN>
__global__ __launch_bounds__(ExpertWarps * 32, Codec::kMinBlocks) void
sparse_moe_prefill_ggml_i8_down_kernel(
    const std::int8_t* __restrict__ codes, const __half2* __restrict__ ds,
    const int* __restrict__ expert_offsets, const int* __restrict__ route_job_experts,
    const int* __restrict__ route_job_columns, const int* __restrict__ slot_of_expert,
    const int* __restrict__ route_job_count, const std::uint8_t* __restrict__ blocks,
    __nv_bfloat16* __restrict__ output) {
    constexpr int BlocksPerRow = kIntermediate / ggml::QK_K;
    constexpr int WarpCols     = ExpertBN / ExpertWarps;
    __shared__ ggml::GgmlI8Smem<Codec, ExpertBN> sm;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    constexpr int row_blocks = kHidden / kExpertBM;
    const int total_work     = *route_job_count * row_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job   = work / row_blocks;
        const int row_block   = work - route_job * row_blocks;
        const int expert      = route_job_experts[route_job];
        const int block       = slot_of_expert != nullptr ? slot_of_expert[expert] : expert;
        if (block < 0) { continue; }
        const int row0        = row_block * kExpertBM;
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols        = count - column_base < ExpertBN ? count - column_base : ExpertBN;

        auto row_base = [&](int local_row) {
            const int global_row = block * kHidden + row0 + local_row;
            return blocks +
                   static_cast<std::int64_t>(global_row) * BlocksPerRow * Codec::kBlockBytes;
        };
        auto act_row = [&](int local_col) {
            return static_cast<std::int64_t>(begin + column_base + local_col);
        };
        float acc[4][4];
        if constexpr (kIntermediate % ggml::QK_K != 0) { return; }
        ggml::ggml_i8_tile<Codec, ExpertWarps, ExpertBN>(sm, row_base, act_row, codes, ds, cols,
                                                          kIntermediate, acc);

        if (warp * WarpCols < cols) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int output_row0 = row0 + mi * 16 + gid;
                const int output_row1 = output_row0 + 8;
                const int local_col0  = warp * WarpCols + 2 * lid;
                const int local_col1  = local_col0 + 1;
                const int col0        = begin + column_base + local_col0;
                const int col1        = col0 + 1;
                if (local_col0 < cols) {
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][0]);
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][2]);
                }
                if (local_col1 < cols) {
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][1]);
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][3]);
                }
            }
        }
        __syncthreads();
    }
}

struct Q5DownMma {
    static constexpr int kHighBytes = Q5RowSplitStorage::kHighBytesPerGroup;

    __device__ static __forceinline__ __nv_bfloat162 decode(const std::uint8_t* codes,
                                                            const std::uint8_t* high,
                                                            const std::uint8_t* scale, int row,
                                                            int lane) {
        return Q5MmaDecodeAtom::decode_pair(codes, high, scale, row, lane);
    }
};

struct Q6DownMma {
    static constexpr int kHighBytes = Q6RowSplitStorage::kHighBytesPerGroup;

    __device__ static __forceinline__ __nv_bfloat162 decode(const std::uint8_t* codes,
                                                            const std::uint8_t* high,
                                                            const std::uint8_t* scale, int row,
                                                            int lane) {
        return Q6MmaDecodeAtom::decode_pair(codes, high, scale, row, lane);
    }
};

template <class Codec, int ExpertWarps, int ExpertBN>
__global__ __launch_bounds__(ExpertWarps * 32, 3) void sparse_moe_prefill_qx_down_kernel(
    const __nv_bfloat16* __restrict__ activation, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ slot_of_expert,
    const int* __restrict__ route_job_count, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ output) {
    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int GroupsPerRow  = kStoredIntermediate / 64;
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][ExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertStages][kExpertBM * 32];
    __shared__ __align__(16) std::uint8_t Hr[kExpertStages][kExpertBM * Codec::kHighBytes];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * GroupsPerRow * 2];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int a_mat          = lane >> 3;
    const int a_rin          = lane & 7;
    const int a_rowoff       = a_rin + ((a_mat & 1) << 3);
    const int a_coloff       = (a_mat >> 1) << 3;
    const int b_rin          = lane & 7;
    const int b_koff         = ((lane >> 3) & 1) << 3;
    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = kHidden / kExpertBM;
    const int total_work     = *route_job_count * row_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job   = work / row_blocks;
        const int row_block   = work - route_job * row_blocks;
        const int expert      = route_job_experts[route_job];
        const int block = slot_of_expert != nullptr ? slot_of_expert[expert] : expert;
        if (block < 0) { continue; } // computed on the host (expert split); uniform per CTA
        const int row0        = row_block * kExpertBM;
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols        = count - column_base < ExpertBN ? count - column_base : ExpertBN;
        float acc[4][4]       = {};

        auto stage_scales = [&] {
            for (int row = tid; row < kExpertBM; row += ExpertThreads) {
                const int global_row  = block * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow;
                cp_async<16, Cache::cg>(&Sr[row * GroupsPerRow * 2], &scales[gi * 2]);
            }
        };

        auto stage_inputs = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < ExpertBN * (kExpertBK / 8); item += ExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int packed_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    activation +
                    static_cast<std::int64_t>(col < cols ? packed_col : begin) * kIntermediate +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }

            for (int item = tid; item < kExpertBM * 2; item += ExpertThreads) {
                const int row         = item >> 1;
                const int half        = item & 1;
                const int global_row  = block * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow + kt;
                cp_async<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                        &codes[gi * 32 + half * 16]);
            }
            for (int row = tid; row < kExpertBM; row += ExpertThreads) {
                const int global_row  = block * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow + kt;
                if constexpr (Codec::kHighBytes == 8) {
                    cp_async<8>(&Hr[stage][row * Codec::kHighBytes], &high[gi * Codec::kHighBytes]);
                } else {
                    cp_async<16, Cache::cg>(&Hr[stage][row * Codec::kHighBytes],
                                            &high[gi * Codec::kHighBytes]);
                }
            }
        };

        auto decode_weight = [&](int stage, int kt) {
            for (int row = warp; row < kExpertBM; row += ExpertWarps) {
                const __nv_bfloat162 value = Codec::decode(
                    Cr[stage], Hr[stage], &Sr[(row * GroupsPerRow + kt) * 2], row, lane);
                store_vec(&As[row * kExpertBK + gemm_swz64(row, 2 * lane)], value);
            }
        };

        stage_scales();
        cp_commit();
#pragma unroll
        for (int stage = 0; stage < kExpertStages; ++stage) {
            stage_inputs(stage, stage);
            cp_commit();
        }

#pragma unroll
        for (int kt = 0; kt < kIntermediate / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<kExpertStages - 1>();
            __syncthreads();
            decode_weight(stage, kt);
            __syncthreads();

            if (warp * 8 < cols) {
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[4][4];
                    unsigned bf[2];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[mi][0],
                                 af[mi][1], af[mi][2], af[mi][3], bf[0], bf[1]);
                    }
                }
            }

            __syncthreads();
            const int next = kt + kExpertStages;
            if (next < kIntermediate / kExpertBK) { stage_inputs(stage, next); }
            cp_commit();
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int output_row0 = row0 + mi * 16 + gid;
                const int output_row1 = output_row0 + 8;
                const int col0        = begin + column_base + warp * 8 + 2 * lid;
                const int col1        = col0 + 1;
                const int local_col0  = warp * 8 + 2 * lid;
                const int local_col1  = local_col0 + 1;
                if (local_col0 < cols) {
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][0]);
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][2]);
                }
                if (local_col1 < cols) {
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][1]);
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][3]);
                }
            }
        }
        __syncthreads();
    }
}

/// `destination += routed_sum`, for a mixture with no always-on expert.
///
/// Where there is one, its down projection is what writes the output: it merges the
/// destination, the routed sum and its own gated value in a single epilogue. A routed-only
/// mixture never launches that kernel, so the routed sum would otherwise be computed and left
/// in the workspace -- every token's answer, discarded one step before it was written.
__global__ void sparse_moe_prefill_combine_routed_kernel(const float* __restrict__ routed_sum,
                                                         __nv_bfloat16* __restrict__ destination,
                                                         int tokens) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= tokens) { return; }
    const std::int64_t base = static_cast<std::int64_t>(token) * kHidden;
    for (int row = static_cast<int>(threadIdx.x); row < kHidden;
         row += static_cast<int>(blockDim.x)) {
        destination[base + row] = __float2bfloat16_rn(
            __bfloat162float(destination[base + row]) + routed_sum[base + row]);
    }
}

template <bool Routed, bool Adaptive = false>
__global__ __launch_bounds__(kExpertThreads, 1) void sparse_moe_prefill_w8_down_kernel(
    const __nv_bfloat16* __restrict__ input, const int* __restrict__ expert_offsets,
    const int* __restrict__ slot_of_expert, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ grouped_output, const float* __restrict__ routed_sum,
    const float* __restrict__ shared_scale, __nv_bfloat16* __restrict__ destination, int tokens,
    const int* __restrict__ route_job_count) {
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][kExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertBM * kExpertBK];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * 16];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int expert = Routed ? static_cast<int>(blockIdx.y) : 0;
    const int block  = (Routed && slot_of_expert != nullptr) ? slot_of_expert[expert] : expert;
    if (block < 0) { return; } // computed on the host (expert split)
    const int row0   = static_cast<int>(blockIdx.x) * kExpertBM;
    const int begin  = Routed ? expert_offsets[expert] : static_cast<int>(blockIdx.y) * kExpertBN;
    const int count  = Routed ? expert_offsets[expert + 1] - begin
                              : (tokens - begin < kExpertBN ? tokens - begin : kExpertBN);
    if (count <= 0) { return; }

    const int a_mat              = lane >> 3;
    const int a_rin              = lane & 7;
    const int a_rowoff           = a_rin + ((a_mat & 1) << 3);
    const int a_coloff           = (a_mat >> 1) << 3;
    const int b_rin              = lane & 7;
    const int b_koff             = ((lane >> 3) & 1) << 3;
    const int gid                = lane >> 2;
    const int lid                = lane & 3;
    constexpr int groups_per_row = kStoredIntermediate / 32;

    const int column_iterations = Routed ? (count + kExpertBN - 1) / kExpertBN : 1;
    for (int iteration = 0; iteration < column_iterations; ++iteration) {
        const int column_base = iteration * kExpertBN;
        const int cols        = count - column_base < kExpertBN ? count - column_base : kExpertBN;
        float acc[4][4]       = {};

        auto stage_x = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < kExpertBN * (kExpertBK / 8); item += kExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int source_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    input +
                    static_cast<std::int64_t>(col < cols ? source_col : begin) * kIntermediate +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }
        };

        auto stage_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int group0                = kt * groups_per_tile;
            for (int item = tid; item < kExpertBM * (kExpertBK / 16); item += kExpertThreads) {
                const int row        = item / (kExpertBK / 16);
                const int chunk      = item - row * (kExpertBK / 16);
                const int global_row = (Routed ? block * kHidden : 0) + row0 + row;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row) * groups_per_row + group0;
                cp_async<16, Cache::cg>(&Cr[row * kExpertBK + chunk * 16],
                                        &codes[gi * 32 + chunk * 16]);
            }
            if ((kt % scale_cache_tiles) == 0) {
                for (int row = tid; row < kExpertBM; row += kExpertThreads) {
                    const int global_row = (Routed ? block * kHidden : 0) + row0 + row;
                    const std::int64_t gi =
                        static_cast<std::int64_t>(global_row) * groups_per_row + group0;
                    if constexpr (groups_per_row % 8 == 0) {
                        cp_async<16, Cache::cg>(&Sr[row * 16], &scales[gi * 2]);
                    } else {
                        // A scale row of groups_per_row * 2 bytes is only 8-byte aligned when
                        // the group count is not a multiple of eight; stage in 8-byte pieces.
                        cp_async<8>(&Sr[row * 16], &scales[gi * 2]);
                        if (group0 + 8 <= groups_per_row) {
                            cp_async<8>(&Sr[row * 16 + 8], &scales[gi * 2 + 8]);
                        }
                    }
                }
            }
        };

        auto decode_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int scale_offset          = (kt % scale_cache_tiles) * groups_per_tile * 2;
            const int half                  = lane >> 4;
            const int half_lane             = lane & 15;
            for (int row_pair = warp * 2; row_pair < kExpertBM; row_pair += kExpertWarps * 2) {
                const int row = row_pair + half;
                unsigned scale_pair =
                    half_lane == 0
                        ? *reinterpret_cast<const std::uint32_t*>(&Sr[row * 16 + scale_offset])
                        : 0;
                scale_pair = __shfl_sync(0xffffffffu, scale_pair, half * 16);
#pragma unroll
                for (int group = 0; group < groups_per_tile; ++group) {
                    const float scale = __half2float(__ushort_as_half(
                        static_cast<std::uint16_t>((scale_pair >> (group * 16)) & 0xffffu)));
                    const int col     = group * 32 + half_lane * 2;
                    const std::uint16_t packed =
                        *reinterpret_cast<const std::uint16_t*>(&Cr[row * kExpertBK + col]);
                    const int q0 = static_cast<int>(static_cast<std::int8_t>(packed & 0xffu));
                    const int q1 = static_cast<int>(static_cast<std::int8_t>(packed >> 8));
                    const __nv_bfloat162 value = __floats2bfloat162_rn(
                        static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
                    store_vec(&As[row * kExpertBK + gemm_swz64(row, col)], value);
                }
            }
        };

        stage_x(0, 0);
        stage_weight(0);
        cp_commit();

#pragma unroll
        for (int kt = 0; kt < kIntermediate / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<0>();
            __syncthreads();
            decode_weight(kt);
            __syncthreads();

            const int next = kt + 1;
            if (next < kIntermediate / kExpertBK) {
                stage_x(next & 1, next);
                stage_weight(next);
                cp_commit();
            }

            if (warp * 8 < cols) {
                unsigned af[2][4][4];
                unsigned bf[2][2];
                auto load_fragments = [&](int slot, int ki) {
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[slot][mi][0], af[slot][mi][1], af[slot][mi][2],
                                    af[slot][mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[slot][0], bf[slot][1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
                };
                load_fragments(0, 0);
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    const int slot = ki & 1;
                    if (ki + 1 < kExpertBK / 16) { load_fragments(slot ^ 1, ki + 1); }
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[slot][mi][0],
                                 af[slot][mi][1], af[slot][mi][2], af[slot][mi][3], bf[slot][0],
                                 bf[slot][1]);
                    }
                }
            }
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int output_row0 = row0 + mi * 16 + gid;
                const int output_row1 = output_row0 + 8;
                const int col0        = begin + column_base + warp * 8 + 2 * lid;
                const int col1        = col0 + 1;
                const int local_col0  = warp * 8 + 2 * lid;
                const int local_col1  = local_col0 + 1;
                auto store            = [&](int col, int row, float value) {
                    if constexpr (Routed) {
                        grouped_output[static_cast<std::int64_t>(col) * kHidden + row] =
                            __float2bfloat16_rn(value);
                    } else {
                        const float merged =
                            __bfloat162float(
                                destination[static_cast<std::int64_t>(col) * kHidden + row]) +
                            routed_sum[static_cast<std::int64_t>(col) * kHidden + row] +
                            shared_scale[col] * value;
                        destination[static_cast<std::int64_t>(col) * kHidden + row] =
                            __float2bfloat16_rn(merged);
                    }
                };
                if (local_col0 < cols) {
                    store(col0, output_row0, acc[mi][0]);
                    store(col0, output_row1, acc[mi][2]);
                }
                if (local_col1 < cols) {
                    store(col1, output_row0, acc[mi][1]);
                    store(col1, output_row1, acc[mi][3]);
                }
            }
        }
        __syncthreads();
    }
}

union alignas(16) SparseMoeBf16x8 {
    uint4 raw;
    __nv_bfloat162 pair[4];
};

// Sums a token's routed paths from the grouped down outputs. A path whose expert the slot
// table maps to -1 was computed elsewhere (the round hook's host split): the gate/up and down
// kernels skipped it, so its grouped column holds whatever an earlier round left there — it
// must carry no weight here, or the token gets that stale column on top of the host's
// contribution.
template <bool Adaptive>
__global__ void sparse_moe_prefill_reduce_kernel(const __nv_bfloat16* __restrict__ grouped_output,
                                                 const int* __restrict__ packed_index,
                                                 const int* __restrict__ ids,
                                                 const float* __restrict__ alpha,
                                                 const int* __restrict__ slot_of_expert,
                                                 float* __restrict__ routed_sum,
                                                 const int* __restrict__ route_job_count) {
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    __shared__ int columns[kTopK];
    __shared__ float weights[kTopK];
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    if (tid < kTopK) {
        columns[tid] = packed_index[token * kTopK + tid];
        const bool served_here =
            slot_of_expert == nullptr || slot_of_expert[ids[token * kTopK + tid]] >= 0;
        weights[tid] = served_here ? alpha[token * kTopK + tid] : 0.0f;
    }
    __syncthreads();

    float values[8] = {};
#pragma unroll
    for (int route = 0; route < kTopK; ++route) {
        SparseMoeBf16x8 packed;
        packed.raw = load_vec<uint4>(grouped_output +
                                     static_cast<std::int64_t>(columns[route]) * kHidden + tid * 8);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 decoded = __bfloat1622float2(packed.pair[pair]);
            values[2 * pair] += weights[route] * decoded.x;
            values[2 * pair + 1] += weights[route] * decoded.y;
        }
    }
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        routed_sum[static_cast<std::int64_t>(token) * kHidden + tid * 8 + item] = values[item];
    }
}

} // namespace

/// This geometry's shape as the vendored runner spells it.
constexpr trtllm_moe::Geometry trtllm_geometry{kHidden, kExperts, kTopK, kIntermediate};

/// The routed weights as the runner reads them. The artifact stores an expert's gate/up rows as
/// [up; gate] for this profile precisely because that is the order the runner assumes.
trtllm_moe::Nvfp4RoutedExperts trtllm_experts(const SparseMoeWeights& weights) {
    if (weights.slot_of_expert != nullptr) {
        throw std::invalid_argument(
            "sparse_moe prefill: the TRT-LLM routed route does not take an expert slot pool");
    }
    trtllm_moe::Nvfp4RoutedExperts experts;
    experts.gate_up_codes        = weights.routed_gate_up.qdata;
    experts.gate_up_block_scales = weights.routed_gate_up.scales;
    experts.gate_up_act_scale    = weights.routed_gate_up_act_scale;
    experts.gate_up_alpha        = weights.routed_gate_up_alpha;
    experts.down_codes           = weights.routed_down.qdata;
    experts.down_block_scales    = weights.routed_down.scales;
    experts.down_act_scale       = weights.routed_down_act_scale;
    experts.down_alpha           = weights.routed_down_alpha;
    return experts;
}

// Every mixture here but Gemma 4 projects the router and the experts from the same tokens, and
// passes the same tensor twice. Gemma 4 does not: its router reads the post-attention residual
// normalised with no weight and scaled by a learned per-channel vector, its experts the same
// residual through `pre_feedforward_layernorm_2`. Two projections of one token.
void prefill_launch(const Tensor& x, const Tensor& router_x, const SparseMoeWeights& weights,
                    Tensor& destination, const SparseMoePrefillPlan& plan,
                    const SparseMoePrefillWorkspace& workspace, cudaStream_t stream,
                    const SparseMoeRoundHook* hook) {
    if (x.ne[1] != plan.tokens || destination.ne[1] != plan.tokens || plan.slice_tokens < 1) {
        throw std::invalid_argument("sparse_moe prefill: launch plan does not match tensors");
    }

    const auto* router = static_cast<const __nv_bfloat16*>(weights.router_shared_gate.qdata);
    const auto* routed_gate_codes = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_gate_scales =
        static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* routed_down_codes  = static_cast<const std::uint8_t*>(weights.routed_down.qdata);
    const auto* routed_down_high   = static_cast<const std::uint8_t*>(weights.routed_down.qhigh);
    const auto* routed_down_scales = static_cast<const std::uint8_t*>(weights.routed_down.scales);
    const auto* shared_gate_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_gate_scales =
        static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    const auto* shared_down_codes  = static_cast<const std::uint8_t*>(weights.shared_down.qdata);
    const auto* shared_down_scales = static_cast<const std::uint8_t*>(weights.shared_down.scales);

    auto* ids               = static_cast<int*>(workspace.token_ids.data);
    auto* alpha             = static_cast<float*>(workspace.token_alpha.data);
    auto* packed_index      = static_cast<int*>(workspace.packed_index.data);
    auto* shared_scale      = static_cast<float*>(workspace.shared_scale.data);
    auto* tile_counts       = static_cast<int*>(workspace.tile_counts.data);
    auto* tile_bases        = static_cast<int*>(workspace.tile_bases.data);
    auto* offsets           = static_cast<int*>(workspace.expert_offsets.data);
    auto* route_job_experts = static_cast<int*>(workspace.route_job_experts.data);
    auto* route_job_columns = static_cast<int*>(workspace.route_job_columns.data);
    auto* route_job_count   = static_cast<int*>(workspace.route_job_count.data);
    auto* scores            = static_cast<float*>(workspace.score_storage.data);
    auto* shared_activation = static_cast<__nv_bfloat16*>(workspace.shared_activation.data);
    auto* grouped_io        = static_cast<__nv_bfloat16*>(workspace.grouped_io.data);
    auto* routed_activation = static_cast<__nv_bfloat16*>(workspace.routed_storage.data);
    auto* column_token      = static_cast<int*>(workspace.column_token.data);
    auto* act_codes         = static_cast<std::int8_t*>(workspace.act_codes.data);
    auto* act_ds            = static_cast<__half2*>(workspace.act_ds.data);
    auto* mid_codes         = static_cast<std::int8_t*>(workspace.mid_codes.data);
    auto* mid_ds            = static_cast<__half2*>(workspace.mid_ds.data);
    auto* routed_sum        = static_cast<float*>(workspace.routed_sum.data);

    for (std::int32_t token0 = 0; token0 < plan.tokens; token0 += plan.slice_tokens) {
        const std::int32_t tokens =
            std::min(plan.slice_tokens, static_cast<std::int32_t>(plan.tokens - token0));
        const Tensor input_slice  = x.slice(1, token0, tokens);
        const Tensor router_slice = router_x.slice(1, token0, tokens);
        Tensor output_slice      = destination.slice(1, token0, tokens);
        const auto* input        = static_cast<const __nv_bfloat16*>(input_slice.data);
        const auto* router_input = static_cast<const __nv_bfloat16*>(router_slice.data);
        auto* output             = static_cast<__nv_bfloat16*>(output_slice.data);
        const int route_tiles =
            (tokens + kSparseMoeRouteTileTokens - 1) / kSparseMoeRouteTileTokens;
        const int assignments   = tokens * kTopK;
        const int adaptive_last = weights.routed_down.qtype == QType::Q5G64_F16S   ? 51
                                  : weights.routed_down.qtype == QType::Q6G64_F16S ? 52
                                                                                   : 0;
        const bool adaptive     = tokens >= 47 && tokens <= adaptive_last;

        sparse_moe_prefill_router_mma_kernel<<<dim3((kRouterRows + kRouterBM - 1) / kRouterBM,
                                                    (tokens + kRouterBN - 1) / kRouterBN),
                                               kRouterThreads, 0, stream>>>(router_input, router,
                                                                            scores, tokens);
        CUDA_CHECK(cudaGetLastError());

        sparse_moe_prefill_select_count_kernel<<<route_tiles, kRouterThreads, 0, stream>>>(
            scores, ids, alpha, shared_scale, packed_index, tile_counts, tokens,
            weights.router_bias, weights.per_expert_scale);
        CUDA_CHECK(cudaGetLastError());
        if (hook != nullptr && hook->resolve != nullptr) {
            const Tensor slice_ids(ids, DType::I32, {assignments});
            const Tensor slice_alpha(alpha, DType::FP32, {assignments});
            hook->resolve(hook->context, slice_ids, slice_alpha, input_slice, output_slice, stream);
        }

        if (plan.routed_trtllm) {
            // The routed experts are computed by the vendored TRT-LLM runner, which does its own
            // permutation from `ids` and its own top-k reduction with `alpha`. Everything the
            // route scan, the gather and this family's four routed kernels would produce is
            // internal to it, so the slice goes straight from selection to the shared expert and
            // the combine.
            if constexpr (kHasShared) {
                const dim3 shared_gate_grid(kIntermediate / (kExpertBM / 2),
                                            (tokens + kExpertBN - 1) / kExpertBN);
                sparse_moe_prefill_w8_gate_up_kernel<false, false>
                    <<<shared_gate_grid, kExpertThreads, 0, stream>>>(
                        input, nullptr, nullptr, shared_gate_codes, shared_gate_scales,
                        shared_activation, tokens, nullptr);
                CUDA_CHECK(cudaGetLastError());
            }

            trtllm_moe::run(trtllm_geometry, input, tokens, ids, alpha,
                            trtllm_experts(weights), workspace.trtllm_workspace.data,
                            workspace.trtllm_workspace.bytes, routed_sum, stream);

            if constexpr (kHasShared) {
                const dim3 shared_down_grid(kHidden / kExpertBM,
                                            (tokens + kExpertBN - 1) / kExpertBN);
                sparse_moe_prefill_w8_down_kernel<false, false>
                    <<<shared_down_grid, kExpertThreads, 0, stream>>>(
                        shared_activation, nullptr, nullptr, shared_down_codes, shared_down_scales,
                        nullptr, routed_sum, shared_scale, output, tokens, nullptr);
            } else {
                sparse_moe_prefill_combine_routed_kernel<<<tokens, 256, 0, stream>>>(
                    routed_sum, output, tokens);
            }
            CUDA_CHECK(cudaGetLastError());
            continue;
        }

        // A serving mixed round carries one prompt's chunk plus the decode batch - about 735
        // columns at the board's shape - which lands just under the 768 the wide plan asks
        // for. The threshold is tunable so that edge can be measured rather than assumed.
        static const std::int32_t wide_min = [] {
            const char* raw = std::getenv("SUROGATE_SERVE_MOE_WIDE_MIN");
            if (raw == nullptr || *raw == '\0') { return kSparseMoePrefillWideMin; }
            const long parsed = std::strtol(raw, nullptr, 10);
            return parsed > 0 ? static_cast<std::int32_t>(parsed) : kSparseMoePrefillWideMin;
        }();
        const bool wide_plan   = tokens >= wide_min;
        const int route_job_bn = wide_plan ? 64 : 32;
        sparse_moe_prefill_scan_kernel<<<1, kExperts, 0, stream>>>(
            tile_counts, tile_bases, offsets, route_job_experts, route_job_columns, route_job_count,
            route_tiles, route_job_bn, tokens, adaptive);
        CUDA_CHECK(cudaGetLastError());

        if (adaptive) {
            auto* adaptive_activations = reinterpret_cast<float*>(grouped_io);
            sparse_moe_decode_launch_d3_small_t(kGeometry, input_slice, weights, ids,
                                                adaptive_activations, tokens,
                                                SparseMoeSmallTD3Schedule::Paths3, stream,
                                                route_job_count);
            sparse_moe_decode_launch_d4_small_t(
                kGeometry, weights, output_slice, ids, alpha, shared_scale, adaptive_activations,
                tokens, SparseMoeSmallTD4Schedule::Rows4, stream, route_job_count);
            sparse_moe_prefill_gather_kernel<true><<<assignments, kHidden / 8, 0, stream>>>(
                input, ids, packed_index, tile_bases, plan.routed_int8 ? nullptr : grouped_io,
                route_job_count, column_token);
        } else {
            sparse_moe_prefill_gather_kernel<false><<<assignments, kHidden / 8, 0, stream>>>(
                input, ids, packed_index, tile_bases, plan.routed_int8 ? nullptr : grouped_io,
                nullptr, column_token);
        }
        CUDA_CHECK(cudaGetLastError());
        if (plan.routed_int8) {
            ggml::quantize_q8_1_planes_launch(input, kHidden, tokens, act_codes, act_ds, stream);
        }

        const dim3 routed_gate_grid(kIntermediate / (kExpertBM / 2), kExperts);
        if (weights.routed_gate_up.qtype == QType::Q4G64_F16S &&
            detail::marlin_moe_route_enabled()) {
            if (weights.slot_of_expert != nullptr) {
                throw std::invalid_argument(
                    "sparse_moe prefill: the Marlin gate/up route does not take an expert slot pool");
            }
            // Marlin produces the raw [rows, 2*intermediate] product; the fold that our own
            // kernel does inline becomes a separate pass (#89).
            const std::int32_t assignments = tokens * kTopK;
            MarlinMoePlane& plane          = marlin_moe_plane_for(
                weights.routed_gate_up, kExperts, 2 * kIntermediate, kHidden, assignments, stream);
            detail::marlin_moe_build_routing(offsets, kExperts, assignments, kMarlinMoeBlock,
                                             plane.sorted_token_ids, plane.expert_ids,
                                             plane.past_padded, stream);
            detail::marlin_moe_gemm_q4g64_bf16(
                grouped_io, plane.b_tiles, plane.scales, plane.product, plane.c_tmp,
                plane.sorted_token_ids, plane.expert_ids, plane.past_padded, assignments,
                2 * kIntermediate, kHidden, kExperts, kMarlinMoeBlock, plane.locks, stream);
            detail::marlin_moe_silu_mul(plane.product, routed_activation, assignments,
                                        kIntermediate, stream);
        } else if (weights.routed_gate_up.qtype == QType::Q4G64_F16S) {
            if (wide_plan) {
                sparse_moe_prefill_q4_gate_up_kernel<8, 64>
                    <<<prefill_persistent_blocks(), 8 * 32, 0, stream>>>(
                        grouped_io, offsets, route_job_experts, route_job_columns,
                        weights.slot_of_expert, route_job_count, routed_gate_codes,
                        routed_gate_scales, routed_activation);
            } else {
                sparse_moe_prefill_q4_gate_up_kernel<4, 32>
                    <<<prefill_persistent_blocks(), 4 * 32, 0, stream>>>(
                        grouped_io, offsets, route_job_experts, route_job_columns,
                        weights.slot_of_expert, route_job_count, routed_gate_codes,
                        routed_gate_scales, routed_activation);
            }
        } else if (ggml::is_ggml_qtype(weights.routed_gate_up.qtype)) {
            // A K-quant row is a whole number of 256-value superblocks. Every geometry that
            // serves one satisfies this; the guard keeps the kernel from being instantiated for
            // the others, which share this body.
            if constexpr (kHidden % ggml::QK_K != 0 || kIntermediate % ggml::QK_K != 0) {
                throw std::invalid_argument(
                    "sparse_moe prefill: this geometry's rows are not whole superblocks");
            } else {
            const auto launch_ggml_gate_up = [&]<class Codec>() {
                if (wide_plan) {
                    sparse_moe_prefill_ggml_gate_up_kernel<Codec, 8, 64>
                        <<<prefill_persistent_blocks(), 8 * 32, 0, stream>>>(
                            grouped_io, offsets, route_job_experts, route_job_columns,
                            weights.slot_of_expert, route_job_count, routed_gate_codes,
                            routed_activation);
                } else {
                    sparse_moe_prefill_ggml_gate_up_kernel<Codec, 4, 32>
                        <<<prefill_persistent_blocks(), 4 * 32, 0, stream>>>(
                            grouped_io, offsets, route_job_experts, route_job_columns,
                            weights.slot_of_expert, route_job_count, routed_gate_codes,
                            routed_activation);
                }
            };
            const auto launch_i8_gate_up = [&]<class Codec>() {
                if (wide_plan) {
                    sparse_moe_prefill_ggml_i8_gate_up_kernel<Codec, 8, 64>
                        <<<prefill_persistent_blocks(), 8 * 32, 0, stream>>>(
                            act_codes, act_ds, column_token, offsets, route_job_experts,
                            route_job_columns, weights.slot_of_expert, route_job_count,
                            routed_gate_codes, routed_activation);
                } else {
                    sparse_moe_prefill_ggml_i8_gate_up_kernel<Codec, 4, 32>
                        <<<prefill_persistent_blocks(), 4 * 32, 0, stream>>>(
                            act_codes, act_ds, column_token, offsets, route_job_experts,
                            route_job_columns, weights.slot_of_expert, route_job_count,
                            routed_gate_codes, routed_activation);
                }
            };
            const bool int8_route = plan.routed_int8;
            // One case per stored format, from the one list: the codec trait names the hand codec
            // for the three K-quants (with the int8 route) and the generic BF16 codec for the rest.
            switch (weights.routed_gate_up.qtype) {
#define SINFER_GATE_UP_CASE(NAME)                                                                  \
            case QType::NAME:                                                                      \
                launch_ggml_prefill_codec<ggml::PrefillCodecFor<ggml::GgmlType::NAME>::Codec,     \
                                          ggml::PrefillCodecFor<ggml::GgmlType::NAME>::kInt8Route>( \
                    launch_i8_gate_up, launch_ggml_gate_up, int8_route);                                                       \
                break;
            SINFER_GGML_FOR_EACH_TYPE(SINFER_GATE_UP_CASE)
#undef SINFER_GATE_UP_CASE
            default:
                throw std::invalid_argument("sparse_moe prefill: unsupported gate/up codec");
            }
            }
        } else if (weights.routed_gate_up.qtype == QType::W8G32_F16S) {
            sparse_moe_prefill_w8_gate_up_kernel<true>
                <<<routed_gate_grid, kExpertThreads, 0, stream>>>(
                    grouped_io, offsets, weights.slot_of_expert, routed_gate_codes, routed_gate_scales,
                    routed_activation, tokens, nullptr);
        } else {
            throw std::invalid_argument("sparse_moe prefill: unsupported gate/up codec");
        }
        CUDA_CHECK(cudaGetLastError());
        if (plan.routed_int8) {
            ggml::quantize_q8_1_planes_launch(routed_activation, kIntermediate, assignments,
                                              mid_codes, mid_ds, stream);
        }

if constexpr (kHasShared) {
            const dim3 shared_gate_grid(kIntermediate / (kExpertBM / 2),
                                        (tokens + kExpertBN - 1) / kExpertBN);
            if (adaptive) {
                sparse_moe_prefill_w8_gate_up_kernel<false, true>
                    <<<shared_gate_grid, kExpertThreads, 0, stream>>>(
                        input, nullptr, nullptr, shared_gate_codes, shared_gate_scales, shared_activation,
                        tokens, route_job_count);
            } else {
                sparse_moe_prefill_w8_gate_up_kernel<false, false>
                    <<<shared_gate_grid, kExpertThreads, 0, stream>>>(
                        input, nullptr, nullptr, shared_gate_codes, shared_gate_scales, shared_activation,
                        tokens, nullptr);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        const dim3 routed_down_grid(kHidden / kExpertBM, kExperts);
        if constexpr (kIntermediate / 64 != 8) {
            if (weights.routed_down.qtype == QType::Q5G64_F16S ||
                weights.routed_down.qtype == QType::Q6G64_F16S) {
                throw std::invalid_argument(
                    "sparse_moe prefill: Q5/Q6 routed down needs intermediate 512 for this geometry");
            }
        }
        if (weights.slot_of_expert != nullptr) {
            // With an expert slot pool some routed experts may be unmapped (slot -1: computed
            // on the host by the expert split). The down kernels skip them, so their
            // assignment rows must read as zero in the reduce that follows.
            CUDA_CHECK(cudaMemsetAsync(grouped_io, 0,
                                       static_cast<std::size_t>(tokens) * kTopK * kHidden * sizeof(__nv_bfloat16),
                                       stream));
        }
        switch (weights.routed_down.qtype) {
        case QType::Q5G64_F16S:
            if (wide_plan) {
                sparse_moe_prefill_qx_down_kernel<Q5DownMma, 8, 64>
                    <<<prefill_persistent_blocks(), 8 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        weights.slot_of_expert, route_job_count, routed_down_codes,
                        routed_down_high, routed_down_scales, grouped_io);
            } else {
                sparse_moe_prefill_qx_down_kernel<Q5DownMma, 4, 32>
                    <<<prefill_persistent_blocks(), 4 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        weights.slot_of_expert, route_job_count, routed_down_codes,
                        routed_down_high, routed_down_scales, grouped_io);
            }
            break;
        case QType::Q6G64_F16S:
            if (wide_plan) {
                sparse_moe_prefill_qx_down_kernel<Q6DownMma, 8, 64>
                    <<<prefill_persistent_blocks(), 8 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        weights.slot_of_expert, route_job_count, routed_down_codes,
                        routed_down_high, routed_down_scales, grouped_io);
            } else {
                sparse_moe_prefill_qx_down_kernel<Q6DownMma, 4, 32>
                    <<<prefill_persistent_blocks(), 4 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        weights.slot_of_expert, route_job_count, routed_down_codes,
                        routed_down_high, routed_down_scales, grouped_io);
            }
            break;
        case QType::W8G32_F16S:
            sparse_moe_prefill_w8_down_kernel<true>
                <<<routed_down_grid, kExpertThreads, 0, stream>>>(
                    routed_activation, offsets, weights.slot_of_expert, routed_down_codes,
                    routed_down_scales, grouped_io, nullptr, nullptr, nullptr, tokens, nullptr);
            break;
        // every GGML block format: the three K-quants with hand codecs and the rest generic
#define SINFER_GGML_DOWN_CASE(NAME) case QType::NAME:
        SINFER_GGML_FOR_EACH_TYPE(SINFER_GGML_DOWN_CASE)
#undef SINFER_GGML_DOWN_CASE
        {
            if constexpr (kHidden % ggml::QK_K != 0 || kIntermediate % ggml::QK_K != 0) {
                throw std::invalid_argument(
                    "sparse_moe prefill: this geometry's rows are not whole superblocks");
            } else {
            const auto launch_ggml_down = [&]<class Codec>() {
                if (wide_plan) {
                    sparse_moe_prefill_ggml_down_kernel<Codec, 8, 64>
                        <<<prefill_persistent_blocks(), 8 * 32, 0, stream>>>(
                            routed_activation, offsets, route_job_experts, route_job_columns,
                            weights.slot_of_expert, route_job_count, routed_down_codes, grouped_io);
                } else {
                    sparse_moe_prefill_ggml_down_kernel<Codec, 4, 32>
                        <<<prefill_persistent_blocks(), 4 * 32, 0, stream>>>(
                            routed_activation, offsets, route_job_experts, route_job_columns,
                            weights.slot_of_expert, route_job_count, routed_down_codes, grouped_io);
                }
            };
            const auto launch_i8_down = [&]<class Codec>() {
                if (wide_plan) {
                    sparse_moe_prefill_ggml_i8_down_kernel<Codec, 8, 64>
                        <<<prefill_persistent_blocks(), 8 * 32, 0, stream>>>(
                            mid_codes, mid_ds, offsets, route_job_experts, route_job_columns,
                            weights.slot_of_expert, route_job_count, routed_down_codes, grouped_io);
                } else {
                    sparse_moe_prefill_ggml_i8_down_kernel<Codec, 4, 32>
                        <<<prefill_persistent_blocks(), 4 * 32, 0, stream>>>(
                            mid_codes, mid_ds, offsets, route_job_experts, route_job_columns,
                            weights.slot_of_expert, route_job_count, routed_down_codes, grouped_io);
                }
            };
            const bool int8_route = plan.routed_int8;
            switch (weights.routed_down.qtype) {
#define SINFER_DOWN_CASE(NAME)                                                                  \
            case QType::NAME:                                                                      \
                launch_ggml_prefill_codec<ggml::PrefillCodecFor<ggml::GgmlType::NAME>::Codec,     \
                                          ggml::PrefillCodecFor<ggml::GgmlType::NAME>::kInt8Route>( \
                    launch_i8_down, launch_ggml_down, int8_route);                                                       \
                break;
            SINFER_GGML_FOR_EACH_TYPE(SINFER_DOWN_CASE)
#undef SINFER_DOWN_CASE
            default:
                throw std::invalid_argument("sparse_moe prefill: unsupported down codec");
            }
            }
            break;
        }
        default:
            throw std::invalid_argument("sparse_moe prefill: unsupported down codec");
        }
        CUDA_CHECK(cudaGetLastError());

        if (adaptive) {
            sparse_moe_prefill_reduce_kernel<true><<<tokens, kHidden / 8, 0, stream>>>(
                grouped_io, packed_index, ids, alpha, weights.slot_of_expert, routed_sum,
                route_job_count);
        } else {
            sparse_moe_prefill_reduce_kernel<false><<<tokens, kHidden / 8, 0, stream>>>(
                grouped_io, packed_index, ids, alpha, weights.slot_of_expert, routed_sum, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());

        if constexpr (kHasShared) {
            const dim3 shared_down_grid(kHidden / kExpertBM,
                                        (tokens + kExpertBN - 1) / kExpertBN);
            if (adaptive) {
                sparse_moe_prefill_w8_down_kernel<false, true>
                    <<<shared_down_grid, kExpertThreads, 0, stream>>>(
                        shared_activation, nullptr, nullptr, shared_down_codes, shared_down_scales,
                        nullptr, routed_sum, shared_scale, output, tokens, route_job_count);
            } else {
                sparse_moe_prefill_w8_down_kernel<false, false>
                    <<<shared_down_grid, kExpertThreads, 0, stream>>>(
                        shared_activation, nullptr, nullptr, shared_down_codes, shared_down_scales,
                        nullptr, routed_sum, shared_scale, output, tokens, nullptr);
            }
        } else {
            sparse_moe_prefill_combine_routed_kernel<<<tokens, 256, 0, stream>>>(routed_sum,
                                                                                 output, tokens);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

