// Sparse-MoE decode kernels, instantiated per geometry by sparse_moe_decode_kernels.cu; the
// including namespace defines kHidden, kExperts, kRouterRows, kTopK, and kIntermediate.

namespace {

constexpr int kD1Warps          = 8;
constexpr int kAdaptiveD3Blocks = 5 * kIntermediate;
constexpr int kAdaptiveD4Blocks = 5 * (kHidden / 4);

__device__ __forceinline__ float dot_bf16_eight(const __nv_bfloat16* a, const __nv_bfloat16* b) {
    const uint4 av  = load_vec<uint4>(a);
    const uint4 bv  = load_vec<uint4>(b);
    const float2 a0 = bf16x2_bits_to_float2(av.x);
    const float2 a1 = bf16x2_bits_to_float2(av.y);
    const float2 a2 = bf16x2_bits_to_float2(av.z);
    const float2 a3 = bf16x2_bits_to_float2(av.w);
    const float2 b0 = bf16x2_bits_to_float2(bv.x);
    const float2 b1 = bf16x2_bits_to_float2(bv.y);
    const float2 b2 = bf16x2_bits_to_float2(bv.z);
    const float2 b3 = bf16x2_bits_to_float2(bv.w);
    float sum       = 0.0f;
    sum             = fmaf(a0.x, b0.x, sum);
    sum             = fmaf(a0.y, b0.y, sum);
    sum             = fmaf(a1.x, b1.x, sum);
    sum             = fmaf(a1.y, b1.y, sum);
    sum             = fmaf(a2.x, b2.x, sum);
    sum             = fmaf(a2.y, b2.y, sum);
    sum             = fmaf(a3.x, b3.x, sum);
    sum             = fmaf(a3.y, b3.y, sum);
    return sum;
}

__device__ __forceinline__ float router_row_dot(const __nv_bfloat16* x, const __nv_bfloat16* row) {
    constexpr int kSlice = kHidden / kD1Warps;
    static_assert(kHidden % kD1Warps == 0 && kSlice % 8 == 0);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float sum      = 0.0f;
    // Each lane walks its slice eight values at a time; a slice that is not a multiple of
    // 256 leaves its tail to the low lanes.
    for (int k = warp * kSlice + lane * 8; k < (warp + 1) * kSlice; k += 32 * 8) {
        sum += dot_bf16_eight(row + k, x + k);
    }
    return warp_reduce_sum(sum);
}

__global__ void sparse_moe_d1_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ router,
                                     float* __restrict__ scores) {
    __shared__ float partial[kD1Warps];
    const int row   = static_cast<int>(blockIdx.x);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const float dot = router_row_dot(x, router + static_cast<std::int64_t>(row) * kHidden);
    if (lane == 0) { partial[warp] = dot; }
    __syncthreads();
    if (warp == 0) {
        float value = lane < kD1Warps ? partial[lane] : 0.0f;
        value       = warp_reduce_sum<kD1Warps>(value);
        if (lane == 0) { scores[row] = value; }
    }
}

__global__ void sparse_moe_d2_warp_kernel(const float* __restrict__ scores, int* __restrict__ ids,
                                          float* __restrict__ alpha,
                                          float* __restrict__ shared_scale) {
    __shared__ float selected_logits[kTopK];
    if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    sparse_moe_select_top_k_warp<kExperts, kTopK>(scores, ids, alpha, shared_scale,
                                                  selected_logits);
}

/// An expert's gate/up block stores the gate rows first in every groupwise-int and W8 artifact.
/// The NVFP4 profile stores them second, because it is served by TensorRT-LLM's fused MoE runner
/// wherever the round is wide enough to prefer it, and that runner reads [up; gate]. The order is
/// a property of the codec's artifact, so it rides on the codec rather than on a runtime flag.
struct Q4Codec {
    static constexpr bool kGateRowsFirst = true;
    static constexpr int kGroupK         = 64;
    static constexpr bool kD3PackedWord8 = true;
};

struct Q5Codec {
    static constexpr bool kGateRowsFirst = true;
    static constexpr int kGroupK       = 64;
    static constexpr bool kPackedWord8 = true;

    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
               std::int64_t group_index, int lane_in_group, float (&weights)[8]) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            codes + group_index * Q5RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint8_t high_bits =
            high[group_index * Q5RowSplitStorage::kHighBytesPerGroup + lane_in_group];
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q5RowSplitStorage::kScaleBytesPerGroup);
        Q5SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);
    }
};

struct Q6Codec {
    static constexpr bool kGateRowsFirst = true;
    static constexpr int kGroupK       = 64;
    static constexpr bool kPackedWord8 = true;

    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
               std::int64_t group_index, int lane_in_group, float (&weights)[8]) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            codes + group_index * Q6RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint16_t high_bits = *reinterpret_cast<const std::uint16_t*>(
            high + group_index * Q6RowSplitStorage::kHighBytesPerGroup + lane_in_group * 2);
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q6RowSplitStorage::kScaleBytesPerGroup);
        Q6SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);
    }
};

// NVFP4: e2m1 codes with an e4m3 scale every 16 values, in the same
// `BlockScaleK16M128x4` layout the dense NVFP4 path and the artifact use — one layout for the
// format across the engine rather than a routed-only variant (design/NVFP4_ROUTED_EXPERTS.md).
//
// Codes are row-major, two values per byte, so a group's eight-value lane slice is at
// `group_index * 8 + lane * 4` exactly as the other codecs index. Scales are swizzled: a tile
// covers 128 rows by 4 groups in 512 bytes, addressed as
//     (row/128 · tiles_per_row + group/4) · 512 + (row%128 %32) · 16 + (row%128 /32) · 4 + group%4
// which is why the codec needs the row width as a template parameter — the shared codec
// interface passes only the flattened `group_index`.
//
// The second level of the format, the per-expert `weight_global_scale`, is deliberately not
// applied here: a factor constant over an expert comes out of the dot product, so the kernels
// multiply the finished dot by `divisor[expert]` once instead of touching every weight.
template <int kGroupsPerRow>
struct Nvfp4CodecFor {
    static constexpr bool kGateRowsFirst        = false;
    static constexpr int kGroupK                = 16;
    static constexpr bool kPackedWord8          = true;
    // D3 has two specialised transaction shapes (Q4's four-group word and W8's one value per
    // lane); NVFP4 uses neither and takes the generic packed branch below.
    static constexpr bool kD3PackedWord8        = false;
    static constexpr bool kD3SingleValuePerLane = false;

    __device__ static __forceinline__ float scale_of(const std::uint8_t* scales,
                                                     std::int64_t group_index) {
        constexpr int kTilesPerRow = kGroupsPerRow / 4;
        const int row              = static_cast<int>(group_index / kGroupsPerRow);
        const int group            = static_cast<int>(group_index - static_cast<std::int64_t>(row) * kGroupsPerRow);
        const int row_inner        = row & 127;
        const std::int64_t offset =
            (static_cast<std::int64_t>(row >> 7) * kTilesPerRow + (group >> 2)) * 512 +
            static_cast<std::int64_t>(row_inner & 31) * 16 +
            static_cast<std::int64_t>(row_inner >> 5) * 4 + (group & 3);
        return detail::decode_nvfp4_e4m3(scales[offset]);
    }

    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t* /*high*/,
               const std::uint8_t* scales, std::int64_t group_index, int lane_in_group,
               float (&weights)[8]) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            codes + group_index * (kGroupK / 2) + lane_in_group * 4);
        const float scale = scale_of(scales, group_index);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 decoded = detail::decode_nvfp4_e2m1x2(
                static_cast<std::uint8_t>((packed >> (8 * pair)) & 0xFFu));
            weights[2 * pair]     = decoded.x * scale;
            weights[2 * pair + 1] = decoded.y * scale;
        }
    }
};

struct W8Codec {
    static constexpr bool kGateRowsFirst        = true;
    static constexpr int kGroupK                = 32;
    static constexpr bool kD3SingleValuePerLane = true;
    static constexpr bool kD3PackedWord8        = false;
    static constexpr bool kPackedWord8          = false;

    __device__ static __forceinline__ float load_one(const std::uint8_t* codes,
                                                     const std::uint8_t* scales,
                                                     std::int64_t group_index, int lane) {
        const float scale = __half2float(
            __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + group_index * 2)));
        return static_cast<float>(static_cast<std::int8_t>(codes[group_index * kGroupK + lane])) *
               scale;
    }

    __device__ static __forceinline__ void
    load_pair(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
              std::int64_t group_index, int lane, float& w0, float& w1) {
        W8ScalarDecodeAtom::load_pair(codes, high, scales, group_index, lane, w0, w1);
    }
};

template <class Codec, int K>
__device__ __forceinline__ void dot_two_rows(const std::uint8_t* codes, const std::uint8_t* high,
                                             const std::uint8_t* scales, int row0, int row1,
                                             const __nv_bfloat16* x, int k_begin, int k_end,
                                             float& result0, float& result1) {
    constexpr int kGroups = K / Codec::kGroupK;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc0            = 0.0f;
    float acc1            = 0.0f;
    const int first_group = k_begin / Codec::kGroupK;
    const int last_group  = k_end / Codec::kGroupK;
    if constexpr (Codec::kD3PackedWord8) {
        // Four adjacent Q4 groups form one 128-byte warp transaction. Each lane owns eight
        // consecutive K values, so one mantissa decode feeds eight FP32 FMAs instead of issuing
        // four scalar code-pair/decode iterations.
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
        for (int group_base = first_group; group_base < last_group; group_base += 4) {
            const int group           = group_base + lane_group;
            const std::int64_t index0 = static_cast<std::int64_t>(row0) * kGroups + group;
            const std::int64_t index1 = static_cast<std::int64_t>(row1) * kGroups + group;
            const std::uint32_t packed0 =
                *reinterpret_cast<const std::uint32_t*>(codes + index0 * 32 + lane_in_group * 4);
            const std::uint32_t packed1 =
                *reinterpret_cast<const std::uint32_t*>(codes + index1 * 32 + lane_in_group * 4);
            const auto scale0 = *reinterpret_cast<const std::uint16_t*>(scales + index0 * 2);
            const auto scale1 = *reinterpret_cast<const std::uint16_t*>(scales + index1 * 2);
            float weights0[8];
            float weights1[8];
            Q4SimtDecodeAtom::decode_eight(packed0, scale0, weights0);
            Q4SimtDecodeAtom::decode_eight(packed1, scale1, weights1);
            const uint4 input     = load_vec<uint4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float2 x0       = bf16x2_bits_to_float2(input.x);
            const float2 x1       = bf16x2_bits_to_float2(input.y);
            const float2 x2       = bf16x2_bits_to_float2(input.z);
            const float2 x3       = bf16x2_bits_to_float2(input.w);
            const float values[8] = {x0.x, x0.y, x1.x, x1.y, x2.x, x2.y, x3.x, x3.y};
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                acc0 = fmaf(weights0[item], values[item], acc0);
                acc1 = fmaf(weights1[item], values[item], acc1);
            }
        }
    } else if constexpr (Codec::kPackedWord8) {
        // Generic packed path: every lane owns eight consecutive K values, so a group spans
        // kGroupK/8 lanes and a warp steps 32/(kGroupK/8) groups at a time. Same shape as the
        // D4 loop; it serves any codec that offers load_eight (NVFP4 today, Q5/Q6 if they are
        // ever wanted for gate/up).
        constexpr int kLanesPerGroup = Codec::kGroupK / 8;
        constexpr int kGroupsPerWarp = 32 / kLanesPerGroup;
        static_assert(Codec::kGroupK % 8 == 0, "packed codecs own eight values per lane");
        const int lane_group    = lane / kLanesPerGroup;
        const int lane_in_group = lane % kLanesPerGroup;
        for (int group_base = first_group; group_base < last_group; group_base += kGroupsPerWarp) {
            const int group = group_base + lane_group;
            if (group >= last_group) { break; }
            const uint4 input = load_vec<uint4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float2 x0   = bf16x2_bits_to_float2(input.x);
            const float2 x1   = bf16x2_bits_to_float2(input.y);
            const float2 x2   = bf16x2_bits_to_float2(input.z);
            const float2 x3   = bf16x2_bits_to_float2(input.w);
            const float values[8] = {x0.x, x0.y, x1.x, x1.y, x2.x, x2.y, x3.x, x3.y};
            const std::int64_t index0 = static_cast<std::int64_t>(row0) * kGroups + group;
            const std::int64_t index1 = static_cast<std::int64_t>(row1) * kGroups + group;
            float weights0[8];
            float weights1[8];
            Codec::load_eight(codes, high, scales, index0, lane_in_group, weights0);
            Codec::load_eight(codes, high, scales, index1, lane_in_group, weights1);
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                acc0 = fmaf(weights0[item], values[item], acc0);
                acc1 = fmaf(weights1[item], values[item], acc1);
            }
        }
    } else if constexpr (Codec::kD3SingleValuePerLane) {
        for (int group = first_group; group < last_group; ++group) {
            const std::int64_t index0 = static_cast<std::int64_t>(row0) * kGroups + group;
            const std::int64_t index1 = static_cast<std::int64_t>(row1) * kGroups + group;
            const float w0            = Codec::load_one(codes, scales, index0, lane);
            const float w1            = Codec::load_one(codes, scales, index1, lane);
            const float xv            = __bfloat162float(x[group * Codec::kGroupK + lane]);
            acc0                      = fmaf(w0, xv, acc0);
            acc1                      = fmaf(w1, xv, acc1);
        }
    } else if (lane < Codec::kGroupK / 2) {
        for (int group = first_group; group < last_group; ++group) {
            float w00, w01, w10, w11;
            Codec::load_pair(codes, high, scales, static_cast<std::int64_t>(row0) * kGroups + group,
                             lane, w00, w01);
            Codec::load_pair(codes, high, scales, static_cast<std::int64_t>(row1) * kGroups + group,
                             lane, w10, w11);
            const int k     = group * Codec::kGroupK + lane * 2;
            const float2 xv = __bfloat1622float2(load_vec<__nv_bfloat162>(x + k));
            acc0            = fmaf(w00, xv.x, acc0);
            acc0            = fmaf(w01, xv.y, acc0);
            acc1            = fmaf(w10, xv.x, acc1);
            acc1            = fmaf(w11, xv.y, acc1);
        }
    }
    result0 = warp_reduce_sum(acc0);
    result1 = warp_reduce_sum(acc1);
}

template <class RoutedCodec>
__global__ void sparse_moe_d3_nine_warp_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales,
    const int* __restrict__ slot_of_expert, const float* __restrict__ routed_gate_up_scale,
    const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, float* __restrict__ act) {
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) { pdl::trigger_dependents(); }
    for (int vector = tid; vector < kHidden / 8; vector += static_cast<int>(blockDim.x)) {
        store_vec(x_shared + vector * 8, load_vec<uint4>(x + vector * 8));
    }
    __syncthreads();

    const int j = static_cast<int>(blockIdx.x);
    float gate  = 0.0f;
    float up    = 0.0f;
    if (warp < kTopK) {
        pdl::wait_for_dependencies();
        const int expert   = ids[warp];
        const int block  = slot_of_expert != nullptr ? slot_of_expert[expert] : expert;
        const int row_base = block * (2 * kIntermediate);
        if (block >= 0) {
            float& first  = RoutedCodec::kGateRowsFirst ? gate : up;
            float& second = RoutedCodec::kGateRowsFirst ? up : gate;
            dot_two_rows<RoutedCodec, kHidden>(routed_codes, routed_high, routed_scales,
                                               row_base + j, row_base + kIntermediate + j, x_shared,
                                               0, kHidden, first, second);
            // NVFP4's second level, one multiply per path instead of one per weight. It lands
            // before SwiGLU, which is where a gate/up scale has to be. The array is indexed by
            // row half, so it follows the codec's order rather than the names.
            if (routed_gate_up_scale != nullptr) {
                first *= routed_gate_up_scale[block * 2];
                second *= routed_gate_up_scale[block * 2 + 1];
            }
        }
    } else {
        dot_two_rows<W8Codec, kHidden>(shared_codes, nullptr, shared_scales, j, kIntermediate + j,
                                       x_shared, 0, kHidden, gate, up);
    }
    if (lane == 0) { act[static_cast<std::int64_t>(warp) * kIntermediate + j] = silu(gate) * up; }
}

template <class RoutedCodec, int PathsPerBlock, bool Adaptive>
__global__ void sparse_moe_d3_path_tiled_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ token_ids,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales,
    const int* __restrict__ slot_of_expert, const float* __restrict__ routed_gate_up_scale,
    const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, float* __restrict__ token_activations,
    int tokens, const int* __restrict__ adaptive_route_jobs) {
    // Three path CTAs per token/output row expose enough blocks for the 170-SM target and keep the
    // heavier shared W8 path from holding eight completed routed warps resident.
    static_assert(PathsPerBlock > 0 && (kTopK + 1) % PathsPerBlock == 0);
    constexpr int kPathBlocks = (kTopK + 1) / PathsPerBlock;
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (Adaptive) {
        if (*adaptive_route_jobs >= 0) { return; }
    }
    const int total_work = tokens * kPathBlocks * kIntermediate;
    const int first_work =
        Adaptive ? static_cast<int>(blockIdx.x)
                 : static_cast<int>(blockIdx.y) * kIntermediate + static_cast<int>(blockIdx.x);
    const int work_stride = Adaptive ? static_cast<int>(gridDim.x) : total_work;
    for (int work = first_work; work < total_work; work += work_stride) {
        const int token_path = work / kIntermediate;
        const int j          = work - token_path * kIntermediate;
        const int token      = token_path / kPathBlocks;
        const int path_block = token_path - token * kPathBlocks;
        const int path       = path_block * PathsPerBlock + warp;
        const auto* input    = x + static_cast<std::int64_t>(token) * kHidden;
        for (int vector = tid; vector < kHidden / 8; vector += static_cast<int>(blockDim.x)) {
            store_vec(x_shared + vector * 8, load_vec<uint4>(input + vector * 8));
        }
        __syncthreads();

        constexpr int kRouterPartitions = 4;
        const int activation_begin      = (token * (kTopK + 1) + path) * kIntermediate;
        // Routed paths need S2's ids. A shared path is independent only when its output lies
        // beyond the partial-score prefix that S2 may still read from the lifetime-unioned scratch.
        const bool must_wait_for_s2 =
            path < kTopK || activation_begin < tokens * kRouterRows * kRouterPartitions;
        if constexpr (!Adaptive) {
            if (must_wait_for_s2) { pdl::wait_for_dependencies(); }
        }
        float gate = 0.0f;
        float up   = 0.0f;
        if (path < kTopK) {
            const int expert   = token_ids[token * kTopK + path];
            const int block  = slot_of_expert != nullptr ? slot_of_expert[expert] : expert;
            const int row_base = block * (2 * kIntermediate);
            if (block >= 0) {
                float& first  = RoutedCodec::kGateRowsFirst ? gate : up;
                float& second = RoutedCodec::kGateRowsFirst ? up : gate;
                dot_two_rows<RoutedCodec, kHidden>(routed_codes, routed_high, routed_scales,
                                                   row_base + j, row_base + kIntermediate + j,
                                                   x_shared, 0, kHidden, first, second);
                if (routed_gate_up_scale != nullptr) {
                    first *= routed_gate_up_scale[block * 2];
                    second *= routed_gate_up_scale[block * 2 + 1];
                }
            }
        } else {
            dot_two_rows<W8Codec, kHidden>(shared_codes, nullptr, shared_scales, j,
                                           kIntermediate + j, x_shared, 0, kHidden, gate, up);
        }
        if (lane == 0) {
            token_activations[(static_cast<std::int64_t>(token) * (kTopK + 1) + path) *
                                  kIntermediate +
                              j] = silu(gate) * up;
        }
        if constexpr (Adaptive) { __syncthreads(); }
    }
}

template <class Codec, int Rows>
__device__ __forceinline__ void dot_fp32_rows(const std::uint8_t* codes, const std::uint8_t* high,
                                              const std::uint8_t* scales, int row_base,
                                              const float* x, int first_group, int last_group,
                                              float (&result)[Rows]) {
    constexpr int kGroups = kIntermediate / Codec::kGroupK;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc[Rows];
#pragma unroll
    for (int row = 0; row < Rows; ++row) { acc[row] = 0.0f; }
    if constexpr (Codec::kPackedWord8) {
        // Every lane owns eight consecutive K values, so a group spans kGroupK/8 lanes and a
        // warp covers 32/(kGroupK/8) groups per step. That is 8 lanes and 4 groups for the
        // 64-wide codecs (Q5/Q6, unchanged) and 2 lanes and 16 groups for NVFP4's 16-wide one.
        constexpr int kLanesPerGroup   = Codec::kGroupK / 8;
        constexpr int kGroupsPerWarp   = 32 / kLanesPerGroup;
        static_assert(Codec::kGroupK % 8 == 0, "packed codecs own eight values per lane");
        const int lane_group    = lane / kLanesPerGroup;
        const int lane_in_group = lane % kLanesPerGroup;
        for (int group_base = first_group; group_base < last_group; group_base += kGroupsPerWarp) {
            const int group = group_base + lane_group;
            const float4 x0 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float4 x1 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8 + 4);
            const float values[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row_base + row) * kGroups + group;
                float weights[8];
                Codec::load_eight(codes, high, scales, group_index, lane_in_group, weights);
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    acc[row] = fmaf(weights[item], values[item], acc[row]);
                }
            }
        }
    } else if (lane < Codec::kGroupK / 2) {
        for (int group = first_group; group < last_group; ++group) {
            const int k     = group * Codec::kGroupK + lane * 2;
            const float2 xv = load_vec<float2>(x + k);
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                float w0, w1;
                Codec::load_pair(codes, high, scales,
                                 static_cast<std::int64_t>(row_base + row) * kGroups + group, lane,
                                 w0, w1);
                acc[row] = fmaf(w0, xv.x, acc[row]);
                acc[row] = fmaf(w1, xv.y, acc[row]);
            }
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) { result[row] = warp_reduce_sum(acc[row]); }
}

template <class RoutedCodec, int Rows>
__global__ void sparse_moe_d4_nine_warp_kernel(
    const int* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ act,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales,
    const int* __restrict__ slot_of_expert, const float* __restrict__ routed_down_scale,
    const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination) {
    __shared__ float paths[kTopK + 1][Rows];
    pdl::wait_for_dependencies();
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int row_base = static_cast<int>(blockIdx.x) * Rows;
    if (warp < kTopK) {
        const int expert = ids[warp];
        const int block  = slot_of_expert != nullptr ? slot_of_expert[expert] : expert;
        float dot[Rows];
        // NVFP4's per-expert second level rides along with the routing weight: both are constant
        // over the path, so the epilogue costs the same multiply it already did.
        float weight = alpha[warp];
        if (block >= 0) {
            dot_fp32_rows<RoutedCodec, Rows>(routed_codes, routed_high, routed_scales,
                                             block * kHidden + row_base,
                                             act + static_cast<std::int64_t>(warp) * kIntermediate,
                                             0, kIntermediate / RoutedCodec::kGroupK, dot);
            if (routed_down_scale != nullptr) { weight *= routed_down_scale[block]; }
        } else {
            for (int r = 0; r < Rows; ++r) { dot[r] = 0.0f; }
        }
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) { paths[warp][row] = weight * dot[row]; }
        }
    } else {
        float dot[Rows];
        dot_fp32_rows<W8Codec, Rows>(shared_codes, nullptr, shared_scales, row_base,
                                     act + static_cast<std::int64_t>(kTopK) * kIntermediate, 0,
                                     kIntermediate / W8Codec::kGroupK, dot);
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) { paths[kTopK][row] = *shared_scale * dot[row]; }
        }
    }
    __syncthreads();
    if (warp == 0 && lane < Rows) {
        float value = __bfloat162float(destination[row_base + lane]);
#pragma unroll
        for (int path = 0; path < kTopK + 1; ++path) { value += paths[path][lane]; }
        destination[row_base + lane] = __float2bfloat16_rn(value);
    }
}

template <class RoutedCodec, int Rows, bool Adaptive>
__global__ void sparse_moe_d4_token_kernel(
    const int* __restrict__ token_ids, const float* __restrict__ token_alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ token_activations,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales,
    const int* __restrict__ slot_of_expert, const float* __restrict__ routed_down_scale,
    const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination,
    int tokens, const int* __restrict__ adaptive_route_jobs) {
    // Token is a grid dimension rather than an in-CTA serial loop. Rows lets one routed-weight
    // stream serve adjacent outputs while retaining the deterministic rank-order FP32 epilogue.
    __shared__ float paths[kTopK + 1][Rows];
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if constexpr (Adaptive) {
        if (*adaptive_route_jobs >= 0) { return; }
    }
    constexpr int kRowBlocks = kHidden / Rows;
    const int total_work     = tokens * kRowBlocks;
    const int first_work =
        Adaptive ? static_cast<int>(blockIdx.x)
                 : static_cast<int>(blockIdx.y) * kRowBlocks + static_cast<int>(blockIdx.x);
    const int work_stride = Adaptive ? static_cast<int>(gridDim.x) : total_work;
    for (int work = first_work; work < total_work; work += work_stride) {
        const int token     = work / kRowBlocks;
        const int row_block = work - token * kRowBlocks;
        const int row_base  = row_block * Rows;
        const float* act =
            token_activations + static_cast<std::int64_t>(token) * (kTopK + 1) * kIntermediate;
        if (warp < kTopK) {
            const int expert = token_ids[token * kTopK + warp];
            const int block  = slot_of_expert != nullptr ? slot_of_expert[expert] : expert;
            float dot[Rows];
            float weight = token_alpha[token * kTopK + warp];
            if (block >= 0) {
                dot_fp32_rows<RoutedCodec, Rows>(routed_codes, routed_high, routed_scales,
                                                 block * kHidden + row_base,
                                                 act + static_cast<std::int64_t>(warp) * kIntermediate,
                                                 0, kIntermediate / RoutedCodec::kGroupK, dot);
                if (routed_down_scale != nullptr) { weight *= routed_down_scale[block]; }
            } else {
                for (int r = 0; r < Rows; ++r) { dot[r] = 0.0f; }
            }
            if (lane == 0) {
#pragma unroll
                for (int row = 0; row < Rows; ++row) { paths[warp][row] = weight * dot[row]; }
            }
        } else {
            float dot[Rows];
            dot_fp32_rows<W8Codec, Rows>(shared_codes, nullptr, shared_scales, row_base,
                                         act + static_cast<std::int64_t>(kTopK) * kIntermediate, 0,
                                         kIntermediate / W8Codec::kGroupK, dot);
            if (lane == 0) {
#pragma unroll
                for (int row = 0; row < Rows; ++row) {
                    paths[kTopK][row] = shared_scale[token] * dot[row];
                }
            }
        }
        __syncthreads();
        if (warp == 0 && lane < Rows) {
            __nv_bfloat16* output =
                destination + static_cast<std::int64_t>(token) * kHidden + row_base + lane;
            float value = __bfloat162float(*output);
#pragma unroll
            for (int path = 0; path < kTopK + 1; ++path) { value += paths[path][lane]; }
            *output = __float2bfloat16_rn(value);
        }
        if constexpr (Adaptive) { __syncthreads(); }
    }
}

void launch_d1(const Tensor& x, const Weight& router_shared_gate,
               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    sparse_moe_d1_kernel<<<kRouterRows, kD1Warps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(router_shared_gate.qdata),
        static_cast<float*>(workspace.scratch.data));
    CUDA_CHECK(cudaGetLastError());
}

template <class Codec>
void launch_d3_dependent_codec(const Tensor& x, const SparseMoeWeights& weights,
                               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    const auto* ids           = static_cast<const int*>(workspace.ids.data);
    auto* act                 = static_cast<float*>(workspace.scratch.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_gate_up.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kIntermediate), dim3((kTopK + 1) * 32), 0, stream},
        sparse_moe_d3_nine_warp_kernel<Codec>,
        input, ids, routed_codes, routed_high, routed_scales, weights.slot_of_expert,
        weights.routed_gate_up_scale, shared_codes, shared_scales, act));
}

// How long the routed expert kernels run (SUROGATE_SERVE_D3_PROBE=1). This is the whole of
// the work that could ever be overlapped with an expert miss-gather: everything earlier in the
// layer feeds the routing the gather depends on. Measuring it sizes that idea before anyone
// builds it.
struct D3Probe {
    static constexpr int kRing = 16;
    bool enabled = std::getenv("SUROGATE_SERVE_D3_PROBE") != nullptr;
    bool ready   = false;
    int slot     = 0;
    bool pending[kRing]{};
    cudaEvent_t before[kRing]{};
    cudaEvent_t after[kRing]{};
    double ms      = 0.0;
    long long runs = 0;
    long long reported = 0;

    void begin(cudaStream_t stream) {
        if (!enabled) { return; }
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) {
            enabled = false;
            return;
        }
        if (!ready) {
            ready = true;
            for (int i = 0; i < kRing; ++i) {
                CUDA_CHECK(cudaEventCreate(&before[i]));
                CUDA_CHECK(cudaEventCreate(&after[i]));
            }
        }
        if (pending[slot]) {
            CUDA_CHECK(cudaEventSynchronize(after[slot]));
            float dt = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&dt, before[slot], after[slot]));
            ms += static_cast<double>(dt);
            ++runs;
            pending[slot] = false;
        }
        CUDA_CHECK(cudaEventRecord(before[slot], stream));
    }
    void end(cudaStream_t stream) {
        if (!enabled || !ready) { return; }
        CUDA_CHECK(cudaEventRecord(after[slot], stream));
        pending[slot] = true;
        slot          = (slot + 1) % kRing;
        if (runs >= reported + 512 && runs > 0) {
            reported = runs;
            std::fprintf(stderr,
                         "sparse_moe: routed expert kernels %.3f ms over %lld layers "
                         "(mean %.4f ms/layer)\n",
                         ms, runs, ms / static_cast<double>(runs));
        }
    }
};

inline D3Probe& d3_probe() {
    static D3Probe probe;
    return probe;
}

void launch_d2_d3(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                  const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream,
                  const SparseMoeRoundHook* hook) {
    const auto* scores = static_cast<const float*>(workspace.scratch.data);
    auto* ids          = static_cast<int*>(workspace.ids.data);
    auto* alpha        = static_cast<float*>(workspace.alpha.data);
    auto* shared_scale = static_cast<float*>(workspace.shared_scale.data);
    sparse_moe_d2_warp_kernel<<<1, 32, 0, stream>>>(scores, ids, alpha, shared_scale);
    CUDA_CHECK(cudaGetLastError());
    // The round hook sees the final ids before the expert kernels; its launches sit between
    // D2 and D3 on the stream, so D3's programmatic wait resolves against them.
    if (hook != nullptr && hook->resolve != nullptr) {
        hook->resolve(hook->context, workspace.ids, workspace.alpha, x, destination, stream);
    }

    d3_probe().begin(stream);
    switch (weights.routed_gate_up.qtype) {
    case QType::Q4G64_F16S:
        launch_d3_dependent_codec<Q4Codec>(x, weights, workspace, stream);
        break;
    case QType::W8G32_F16S:
        launch_d3_dependent_codec<W8Codec>(x, weights, workspace, stream);
        break;
    case QType::NVFP4:
        launch_d3_dependent_codec<Nvfp4CodecFor<kHidden / 16>>(x, weights, workspace, stream);
        break;
    case QType::Q4_K:
        launch_d3_dependent_codec<ggml::GgmlMoeCodec<ggml::GgmlType::Q4_K>>(x, weights, workspace,
                                                                            stream);
        break;
    case QType::Q5_K:
        launch_d3_dependent_codec<ggml::GgmlMoeCodec<ggml::GgmlType::Q5_K>>(x, weights, workspace,
                                                                            stream);
        break;
    case QType::Q6_K:
        launch_d3_dependent_codec<ggml::GgmlMoeCodec<ggml::GgmlType::Q6_K>>(x, weights, workspace,
                                                                            stream);
        break;
    default:
        throw std::invalid_argument("sparse_moe: unsupported D3 codec");
    }
    d3_probe().end(stream);
}

template <class Codec>
void require_d4_codec_geometry() {
    if constexpr (Codec::kPackedWord8) {
        // The packed D4 loop walks 32/(kGroupK/8) groups per warp step and does not bound the
        // last one, so the group count must divide by that step -- four for the 64-wide codecs
        // (which is where the constant four came from) but one for a 256-wide superblock, where
        // a whole warp covers a single group and every step is exact.
        constexpr int kD4GroupsPerWarp = 32 / (Codec::kGroupK / 8);
        if constexpr ((kIntermediate / Codec::kGroupK) % kD4GroupsPerWarp != 0) {
            throw std::invalid_argument(
                "sparse_moe: routed down codec needs intermediate divisible by its warp step");
        }
    }
}

template <class Codec>
void launch_d4_dependent_codec(const SparseMoeWeights& weights, Tensor& destination,
                               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    require_d4_codec_geometry<Codec>();
    const auto* ids           = static_cast<const int*>(workspace.ids.data);
    const auto* alpha         = static_cast<const float*>(workspace.alpha.data);
    const auto* shared_scale  = static_cast<const float*>(workspace.shared_scale.data);
    const auto* act           = static_cast<const float*>(workspace.scratch.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_down.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_down.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_down.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_down.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_down.scales);
    auto* output              = static_cast<__nv_bfloat16*>(destination.data);
    CUDA_CHECK(pdl::launch_dependent({dim3(kHidden), dim3((kTopK + 1) * 32), 0, stream},
                                     sparse_moe_d4_nine_warp_kernel<Codec, 1>, ids, alpha,
                                     shared_scale, act, routed_codes, routed_high, routed_scales,
                                     weights.slot_of_expert, weights.routed_down_scale,
                                     shared_codes, shared_scales, output));
}

void launch_d4_dependent(const SparseMoeWeights& weights, Tensor& destination,
                         const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    switch (weights.routed_down.qtype) {
    case QType::Q5G64_F16S:
        launch_d4_dependent_codec<Q5Codec>(weights, destination, workspace, stream);
        return;
    case QType::Q6G64_F16S:
        launch_d4_dependent_codec<Q6Codec>(weights, destination, workspace, stream);
        return;
    case QType::Q4_K:
        launch_d4_dependent_codec<ggml::GgmlMoeCodec<ggml::GgmlType::Q4_K>>(weights, destination,
                                                                            workspace, stream);
        return;
    case QType::Q5_K:
        launch_d4_dependent_codec<ggml::GgmlMoeCodec<ggml::GgmlType::Q5_K>>(weights, destination,
                                                                            workspace, stream);
        return;
    case QType::Q6_K:
        launch_d4_dependent_codec<ggml::GgmlMoeCodec<ggml::GgmlType::Q6_K>>(weights, destination,
                                                                            workspace, stream);
        return;
    case QType::W8G32_F16S:
        launch_d4_dependent_codec<W8Codec>(weights, destination, workspace, stream);
        return;
    case QType::NVFP4:
        launch_d4_dependent_codec<Nvfp4CodecFor<kIntermediate / 16>>(weights, destination,
                                                                     workspace, stream);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported D4 codec");
    }
}

template <class Codec, int PathsPerBlock, bool Adaptive>
void launch_d3_small_t_paths(const Tensor& x, const SparseMoeWeights& weights, const int* token_ids,
                             float* token_activations, std::int32_t tokens, cudaStream_t stream,
                             const int* adaptive_route_jobs) {
    constexpr int kPathBlocks = (kTopK + 1) / PathsPerBlock;
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_gate_up.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    if constexpr (Adaptive) {
        sparse_moe_d3_path_tiled_kernel<Codec, PathsPerBlock, true>
            <<<kAdaptiveD3Blocks, PathsPerBlock * 32, 0, stream>>>(
                input, token_ids, routed_codes, routed_high, routed_scales,
                weights.slot_of_expert, weights.routed_gate_up_scale, shared_codes, shared_scales,
                token_activations, tokens, adaptive_route_jobs);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(kIntermediate, tokens * kPathBlocks), dim3(PathsPerBlock * 32), 0, stream},
            sparse_moe_d3_path_tiled_kernel<Codec, PathsPerBlock, false>, input, token_ids,
            routed_codes, routed_high, routed_scales, weights.slot_of_expert,
            weights.routed_gate_up_scale, shared_codes, shared_scales, token_activations, tokens,
            nullptr));
    }
}

template <class Codec, bool Adaptive>
void launch_d3_small_t_codec(const Tensor& x, const SparseMoeWeights& weights, const int* token_ids,
                             float* token_activations, std::int32_t tokens,
                             SparseMoeSmallTD3Schedule schedule, cudaStream_t stream,
                             const int* adaptive_route_jobs) {
    switch (schedule) {
    case SparseMoeSmallTD3Schedule::Paths1:
        launch_d3_small_t_paths<Codec, 1, Adaptive>(x, weights, token_ids, token_activations,
                                                    tokens, stream, adaptive_route_jobs);
        return;
    case SparseMoeSmallTD3Schedule::Paths3:
        // Three paths per CTA needs the path count divisible by three; otherwise every path
        // of the token shares one CTA.
        if constexpr ((kTopK + 1) % 3 == 0) {
            launch_d3_small_t_paths<Codec, 3, Adaptive>(x, weights, token_ids,
                                                        token_activations, tokens, stream,
                                                        adaptive_route_jobs);
        } else {
            launch_d3_small_t_paths<Codec, kTopK + 1, Adaptive>(x, weights, token_ids,
                                                                token_activations, tokens,
                                                                stream, adaptive_route_jobs);
        }
        return;
    case SparseMoeSmallTD3Schedule::PathsAll:
        launch_d3_small_t_paths<Codec, kTopK + 1, Adaptive>(x, weights, token_ids,
                                                            token_activations, tokens, stream,
                                                            adaptive_route_jobs);
        return;
    }
    throw std::logic_error("sparse_moe: unknown small-T D3 schedule");
}

template <class Codec, int Rows, bool Adaptive>
void launch_d4_small_t_rows(const SparseMoeWeights& weights, Tensor& destination,
                            const int* token_ids, const float* token_alpha,
                            const float* shared_scale, const float* token_activations,
                            std::int32_t tokens, cudaStream_t stream,
                            const int* adaptive_route_jobs) {
    require_d4_codec_geometry<Codec>();
    const dim3 grid = Adaptive ? dim3(kAdaptiveD4Blocks) : dim3(kHidden / Rows, tokens);
    sparse_moe_d4_token_kernel<Codec, Rows, Adaptive><<<grid, (kTopK + 1) * 32, 0, stream>>>(
        token_ids, token_alpha, shared_scale, token_activations,
        static_cast<const std::uint8_t*>(weights.routed_down.qdata),
        static_cast<const std::uint8_t*>(weights.routed_down.qhigh),
        static_cast<const std::uint8_t*>(weights.routed_down.scales), weights.slot_of_expert,
        weights.routed_down_scale, static_cast<const std::uint8_t*>(weights.shared_down.qdata),
        static_cast<const std::uint8_t*>(weights.shared_down.scales),
        static_cast<__nv_bfloat16*>(destination.data), tokens, adaptive_route_jobs);
    CUDA_CHECK(cudaGetLastError());
}

template <class Codec, bool Adaptive>
void launch_d4_small_t_codec(const SparseMoeWeights& weights, Tensor& destination,
                             const int* token_ids, const float* token_alpha,
                             const float* shared_scale, const float* token_activations,
                             std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
                             cudaStream_t stream, const int* adaptive_route_jobs) {
    switch (schedule) {
    case SparseMoeSmallTD4Schedule::Rows1:
        launch_d4_small_t_rows<Codec, 1, Adaptive>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens, stream,
                                                   adaptive_route_jobs);
        return;
    case SparseMoeSmallTD4Schedule::Rows2:
        launch_d4_small_t_rows<Codec, 2, Adaptive>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens, stream,
                                                   adaptive_route_jobs);
        return;
    case SparseMoeSmallTD4Schedule::Rows4:
        launch_d4_small_t_rows<Codec, 4, Adaptive>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens, stream,
                                                   adaptive_route_jobs);
        return;
    }
    throw std::logic_error("sparse_moe: unknown small-T D4 schedule");
}

} // namespace

void decode_launch_d3_small_t(const Tensor& x, const SparseMoeWeights& weights,
                              const int* token_ids, float* token_activations, std::int32_t tokens,
                              SparseMoeSmallTD3Schedule schedule, cudaStream_t stream,
                              const int* adaptive_route_jobs) {
    switch (weights.routed_gate_up.qtype) {
    case QType::Q4G64_F16S:
        if (adaptive_route_jobs == nullptr) {
            launch_d3_small_t_codec<Q4Codec, false>(x, weights, token_ids, token_activations,
                                                    tokens, schedule, stream, nullptr);
        } else {
            launch_d3_small_t_codec<Q4Codec, true>(x, weights, token_ids, token_activations, tokens,
                                                   schedule, stream, adaptive_route_jobs);
        }
        return;
    case QType::W8G32_F16S:
        launch_d3_small_t_codec<W8Codec, false>(x, weights, token_ids, token_activations, tokens,
                                                schedule, stream, nullptr);
        return;
    case QType::Q4_K:
        launch_d3_small_t_codec<ggml::GgmlMoeCodec<ggml::GgmlType::Q4_K>, false>(
            x, weights, token_ids, token_activations, tokens, schedule, stream, nullptr);
        return;
    case QType::Q5_K:
        launch_d3_small_t_codec<ggml::GgmlMoeCodec<ggml::GgmlType::Q5_K>, false>(
            x, weights, token_ids, token_activations, tokens, schedule, stream, nullptr);
        return;
    case QType::Q6_K:
        launch_d3_small_t_codec<ggml::GgmlMoeCodec<ggml::GgmlType::Q6_K>, false>(
            x, weights, token_ids, token_activations, tokens, schedule, stream, nullptr);
        return;
    case QType::NVFP4:
        if (adaptive_route_jobs == nullptr) {
            launch_d3_small_t_codec<Nvfp4CodecFor<kHidden / 16>, false>(
                x, weights, token_ids, token_activations, tokens, schedule, stream, nullptr);
        } else {
            launch_d3_small_t_codec<Nvfp4CodecFor<kHidden / 16>, true>(
                x, weights, token_ids, token_activations, tokens, schedule, stream,
                adaptive_route_jobs);
        }
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported small-T D3 codec");
    }
}

void decode_launch_d4_small_t(const SparseMoeWeights& weights, Tensor& destination,
                              const int* token_ids, const float* token_alpha,
                              const float* shared_scale, const float* token_activations,
                              std::int32_t tokens, SparseMoeSmallTD4Schedule schedule,
                              cudaStream_t stream, const int* adaptive_route_jobs) {
    switch (weights.routed_down.qtype) {
    case QType::Q5G64_F16S:
        if (adaptive_route_jobs == nullptr) {
            launch_d4_small_t_codec<Q5Codec, false>(weights, destination, token_ids, token_alpha,
                                                    shared_scale, token_activations, tokens,
                                                    schedule, stream, nullptr);
        } else {
            launch_d4_small_t_codec<Q5Codec, true>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens,
                                                   schedule, stream, adaptive_route_jobs);
        }
        return;
    case QType::Q6G64_F16S:
        if (adaptive_route_jobs == nullptr) {
            launch_d4_small_t_codec<Q6Codec, false>(weights, destination, token_ids, token_alpha,
                                                    shared_scale, token_activations, tokens,
                                                    schedule, stream, nullptr);
        } else {
            launch_d4_small_t_codec<Q6Codec, true>(weights, destination, token_ids, token_alpha,
                                                   shared_scale, token_activations, tokens,
                                                   schedule, stream, adaptive_route_jobs);
        }
        return;
    case QType::W8G32_F16S:
        launch_d4_small_t_codec<W8Codec, false>(weights, destination, token_ids, token_alpha,
                                                shared_scale, token_activations, tokens, schedule,
                                                stream, nullptr);
        return;
    case QType::Q4_K:
        launch_d4_small_t_codec<ggml::GgmlMoeCodec<ggml::GgmlType::Q4_K>, false>(
            weights, destination, token_ids, token_alpha, shared_scale, token_activations, tokens,
            schedule, stream, nullptr);
        return;
    case QType::Q5_K:
        launch_d4_small_t_codec<ggml::GgmlMoeCodec<ggml::GgmlType::Q5_K>, false>(
            weights, destination, token_ids, token_alpha, shared_scale, token_activations, tokens,
            schedule, stream, nullptr);
        return;
    case QType::Q6_K:
        launch_d4_small_t_codec<ggml::GgmlMoeCodec<ggml::GgmlType::Q6_K>, false>(
            weights, destination, token_ids, token_alpha, shared_scale, token_activations, tokens,
            schedule, stream, nullptr);
        return;
    case QType::NVFP4:
        if (adaptive_route_jobs == nullptr) {
            launch_d4_small_t_codec<Nvfp4CodecFor<kIntermediate / 16>, false>(
                weights, destination, token_ids, token_alpha, shared_scale, token_activations,
                tokens, schedule, stream, nullptr);
        } else {
            launch_d4_small_t_codec<Nvfp4CodecFor<kIntermediate / 16>, true>(
                weights, destination, token_ids, token_alpha, shared_scale, token_activations,
                tokens, schedule, stream, adaptive_route_jobs);
        }
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported small-T D4 codec");
    }
}

void decode_launch(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                   const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream,
                   const SparseMoeRoundHook* hook) {
    launch_d1(x, weights.router_shared_gate, workspace, stream);
    launch_d2_d3(x, weights, destination, workspace, stream, hook);
    launch_d4_dependent(weights, destination, workspace, stream);
}

