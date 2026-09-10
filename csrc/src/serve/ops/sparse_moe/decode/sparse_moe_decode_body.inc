constexpr int kDecodeScratch = std::max(kPaths, (kRouterRows + kIntermediate - 1) / kIntermediate) * kIntermediate;

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

// Expert indices are selected on the device. Low-rank intermediates therefore
// follow routed paths, while each bank remains indexed by its checkpoint expert.
__global__ void moe_lora_low_kernel(const __nv_bfloat16* x, const float* act, const int* ids,
                                     const LoraBank* banks, const int* slot, float* low, bool down, int slot_stride) {
    const int token = blockIdx.z;
    if (x) { x += token * kHidden; }
    if (act) { act += token * kDecodeScratch; }
    ids += token * kTopK;
    slot += token * slot_stride;
    low += token * 2 * kPaths * kMoeLoraMaxRank;
    const int r = blockIdx.x;
    const int path = down ? blockIdx.y : blockIdx.y / 2;
    const int part = down ? 2 : blockIdx.y % 2;
    const int expert = path < kTopK ? ids[path] : kExperts;
    const LoraBank bank = banks[2 + 3 * expert + part];
    if (r >= bank.rank) { return; }
    float sum = 0;
    if (*slot >= 0 && bank.a) {
        const auto* a = static_cast<const __nv_bfloat16*>(bank.a) + *slot * bank.a_stride + r * bank.k;
        for (int k = threadIdx.x; k < bank.k; k += 32) {
            const float value = down ? act[path * kIntermediate + k] : __bfloat162float(x[k]);
            sum = fmaf(__bfloat162float(a[k]), value, sum);
        }
    }
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) { low[blockIdx.y * kMoeLoraMaxRank + r] = sum; }
}

__device__ float moe_lora_expand(const LoraBank* banks, int index, const int* slot,
                                  int row, const float* low) {
    if (!banks || !slot || *slot < 0) { return 0; }
    const LoraBank bank = banks[index];
    if (!bank.b) { return 0; }
    const auto* b = static_cast<const __nv_bfloat16*>(bank.b) + *slot * bank.b_stride + row * bank.rank;
    float sum = 0;
    for (int r = 0; r < bank.rank; ++r) { sum = fmaf(__bfloat162float(b[r]), low[r], sum); }
    return sum;
}

__device__ float moe_lora_adjust(const LoraBank* banks, int index, const int* slot, int row, float value) {
    if (!banks || !slot || *slot < 0) { return value; }
    const auto bank = banks[index];
    const auto offset = static_cast<std::int64_t>(*slot) * bank.n + row;
    return value * (bank.gain ? 1.0F + bank.gain[offset] : 1.0F) + (bank.bias ? bank.bias[offset] : 0.0F);
}

__global__ void moe_lora_router_kernel(const __nv_bfloat16* x, const LoraBank* banks,
                                        const int* slot, float* scores, int slot_stride) {
    const int token = blockIdx.y;
    x += token * kHidden;
    slot += token * slot_stride;
    scores += token * kDecodeScratch;
    if (*slot < 0) { return; }
    const int row = blockIdx.x;
    const LoraBank bank = banks[row < kExperts ? 0 : 1];
    if (!bank.a) { return; }
    const auto* a = static_cast<const __nv_bfloat16*>(bank.a) + *slot * bank.a_stride;
    const auto* b = static_cast<const __nv_bfloat16*>(bank.b) + *slot * bank.b_stride +
                    (row < kExperts ? row : 0) * bank.rank;
    float delta = 0;
    for (int r = 0; r < bank.rank; ++r) {
        float dot = 0;
        for (int k = threadIdx.x; k < bank.k; k += 32) {
            dot = fmaf(__bfloat162float(a[r * bank.k + k]), __bfloat162float(x[k]), dot);
        }
        dot = warp_reduce_sum(dot);
        if (threadIdx.x == 0) { delta = fmaf(__bfloat162float(b[r]), dot, delta); }
    }
    if (threadIdx.x == 0) { scores[row] = moe_lora_adjust(banks, row < kExperts ? 0 : 1, slot, row < kExperts ? row : 0, scores[row] + delta); }
}

__global__ void sparse_moe_d1_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ router,
                                     float* __restrict__ scores) {
    x += blockIdx.y * kHidden;
    scores += blockIdx.y * kDecodeScratch;
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
                                          float* __restrict__ shared_scale,
                                          const float* __restrict__ router_bias,
                                          const float* __restrict__ per_expert_scale) {
    scores += blockIdx.y * kDecodeScratch;
    ids += blockIdx.y * kTopK;
    alpha += blockIdx.y * kTopK;
    shared_scale += blockIdx.y;
    __shared__ float selected_logits[kTopK];
    if (threadIdx.x == 0) { pdl::trigger_dependents(); }
    sparse_moe_select_top_k_warp<kExperts, kTopK, kHasShared, kGating, kSharedGated,
                                 kPerExpertScaled>(
        scores, ids, alpha, shared_scale, selected_logits, router_bias, kRoutedScale,
        per_expert_scale);
}

/// An expert's gate/up block stores the gate rows first in every groupwise-int and W8 artifact.
/// The NVFP4 profile stores them second, because it is served by TensorRT-LLM's fused MoE runner
/// wherever the round is wide enough to prefer it, and that runner reads [up; gate]. The order is
/// a property of the codec's artifact, so it rides on the codec rather than on a runtime flag.
struct Q4Codec {
    static constexpr bool kGateRowsFirst = true;
    static constexpr int kGroupK         = 64;
    static constexpr bool kD3PackedWord8 = true;
    /// Row-split storage pads K to 128; see `kStoredGroups`.
    static constexpr int kStoredKAlignment = 128;
};

struct Q5Codec {
    static constexpr bool kGateRowsFirst = true;
    static constexpr int kGroupK       = 64;
    static constexpr bool kPackedWord8 = true;
    static constexpr int kStoredKAlignment = 128;

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
    static constexpr int kStoredKAlignment = 128;

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
    /// `BlockScaleK16M128x4` stores K exactly; it does not pad to 128.
    static constexpr int kStoredKAlignment      = 1;
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

/// W8 is the always-on expert's format in every artifact, so it runs in one warp of every D3
/// and D4 block beside the routed warps. It used to read one byte per lane per 32-value group:
/// 64 dependent DRAM round trips across a 2048-wide row where a Q4_K warp beside it makes 8,
/// and in D4 a loop that left half the warp idle. That warp was the long pole of both kernels
/// -- D4's barrier stall was the routed warps waiting for it. Eight consecutive values per
/// lane, as every other codec: one 8-byte load per lane per group, four lanes to a group,
/// eight groups per warp step.
struct W8Codec {
    static constexpr bool kGateRowsFirst        = true;
    static constexpr int kGroupK                = 32;
    static constexpr int kStoredKAlignment      = 128;
    static constexpr bool kD3SingleValuePerLane = false;
    static constexpr bool kD3PackedWord8        = false;
    static constexpr bool kPackedWord8          = true;

    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t* /*high*/,
               const std::uint8_t* scales, std::int64_t group_index, int lane_in_group,
               float (&weights)[8]) {
        const uint2 packed = load_vec<uint2>(
            codes + group_index * W8RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 8);
        const float scale = __half2float(
            __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + group_index * 2)));
        const std::uint32_t words[2] = {packed.x, packed.y};
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            // Arithmetic shift out of the word sign-extends the byte.
            const int code = static_cast<int>(words[item >> 2] << (24 - 8 * (item & 3))) >> 24;
            weights[item]  = static_cast<float>(code) * scale;
        }
    }
};

/// How wide a stored row is, against how wide the math reads.
///
/// `row-split-k128-v1` pads K up to 128 and stores `padded / group` groups per row
/// (`row_split_geometry` in `artifact/storage_layouts.cpp`), so a row's stride is the *padded*
/// width while the dot product still reads only the real one. Every mixture registered before
/// Gemma 4 had feed-forward widths that were multiples of 128 -- 512, 640, 768, 2048 -- so the
/// two coincided and nothing here distinguished them. Gemma 4's routed experts are 704 wide,
/// which is 5.5 x 128: the artifact holds 24 groups per row where the width is 22, and a stride
/// of 22 reads every row after the first from the middle of its predecessor.
///
/// The GGML superblock and NVFP4 layouts store K exactly, so their alignment is one. The
/// constant rides on the codec because the layout does.
template <class Codec, int K>
inline constexpr int kStoredGroups =
    ((K + Codec::kStoredKAlignment - 1) / Codec::kStoredKAlignment) * Codec::kStoredKAlignment /
    Codec::kGroupK;

template <class Codec, int K>
__device__ __forceinline__ void dot_two_rows(const std::uint8_t* codes, const std::uint8_t* high,
                                             const std::uint8_t* scales, int row0, int row1,
                                             const __nv_bfloat16* x, int k_begin, int k_end,
                                             float& result0, float& result1) {
    constexpr int kGroups = kStoredGroups<Codec, K>;
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
#pragma unroll
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
    const std::uint8_t* __restrict__ shared_scales, float* __restrict__ act,
    const LoraBank* adapters, const int* adapter_slot, const float* lora_low, int slot_stride) {
    const int token = blockIdx.y;
    x += token * kHidden;
    ids += token * kTopK;
    act += token * kDecodeScratch;
    if (adapter_slot) { adapter_slot += token * slot_stride; }
    if (lora_low) { lora_low += token * 2 * kPaths * kMoeLoraMaxRank; }
    if (adapters) { pdl::wait_for_dependencies(); }
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
    } else if constexpr (kHasShared) {
        dot_two_rows<W8Codec, kHidden>(shared_codes, nullptr, shared_scales, j, kIntermediate + j,
                                       x_shared, 0, kHidden, gate, up);
    }
    if (lane == 0) {
        if (adapters && (warp >= kTopK || !slot_of_expert || slot_of_expert[ids[warp]] >= 0)) {
            const int expert = warp < kTopK ? ids[warp] : kExperts;
            gate += moe_lora_expand(adapters, 2 + 3 * expert, adapter_slot, j,
                                     lora_low + (warp * 2) * kMoeLoraMaxRank);
            up += moe_lora_expand(adapters, 3 + 3 * expert, adapter_slot, j,
                                   lora_low + (warp * 2 + 1) * kMoeLoraMaxRank);
            gate = moe_lora_adjust(adapters, 2 + 3 * expert, adapter_slot, j, gate);
            up = moe_lora_adjust(adapters, 3 + 3 * expert, adapter_slot, j, up);
        }
        act[static_cast<std::int64_t>(warp) * kIntermediate + j] = gated_clamped(gate, up, kSwigluLimit, kActivation);
    }
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
    static_assert(PathsPerBlock > 0 && kPaths % PathsPerBlock == 0);
    constexpr int kPathBlocks = kPaths / PathsPerBlock;
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
        const int activation_begin      = (token * kPaths + path) * kIntermediate;
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
        } else if constexpr (kHasShared) {
            dot_two_rows<W8Codec, kHidden>(shared_codes, nullptr, shared_scales, j,
                                           kIntermediate + j, x_shared, 0, kHidden, gate, up);
        }
        if (lane == 0) {
            token_activations[(static_cast<std::int64_t>(token) * kPaths + path) *
                                  kIntermediate +
                              j] = gated_clamped(gate, up, kSwigluLimit, kActivation);
        }
        if constexpr (Adaptive) { __syncthreads(); }
    }
}

template <class Codec, int Rows>
__device__ __forceinline__ void dot_fp32_rows(const std::uint8_t* codes, const std::uint8_t* high,
                                              const std::uint8_t* scales, int row_base,
                                              const float* x, int first_group, int last_group,
                                              float (&result)[Rows]) {
    constexpr int kGroups = kStoredGroups<Codec, kIntermediate>;
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
        // Not unrolled, unlike D3's loop above: unrolling this one cost GLM-5.3-Flash's
        // 2048-wide Q6_K down projection 12 % (5090, cold), and gained nothing at FFN 512.
        for (int group_base = first_group; group_base < last_group; group_base += kGroupsPerWarp) {
            const int group = group_base + lane_group;
            // The last step is short whenever the group count is not a whole number of warp
            // steps -- 20 groups of 32 against a step of 8, which is what a 640-wide expert
            // gives. Lanes past the end contribute nothing rather than the codec refusing.
            if (group >= last_group) { continue; }
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
    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination,
    const LoraBank* adapters, const int* adapter_slot, const float* lora_low, int slot_stride) {
    const int token = blockIdx.y;
    ids += token * kTopK;
    alpha += token * kTopK;
    shared_scale += token;
    act += token * kDecodeScratch;
    destination += token * kHidden;
    if (adapter_slot) { adapter_slot += token * slot_stride; }
    if (lora_low) { lora_low += token * 2 * kPaths * kMoeLoraMaxRank; }
    __shared__ float paths[kPaths][Rows];
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
            if (routed_down_scale != nullptr) {
                if (!adapters) { weight *= routed_down_scale[block]; }
                else if (lane == 0) { for (int r = 0; r < Rows; ++r) { dot[r] *= routed_down_scale[block]; } }
            }
        } else {
            for (int r = 0; r < Rows; ++r) { dot[r] = 0.0f; }
        }
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                if (adapters && block >= 0) { dot[row] += moe_lora_expand(adapters, 4 + 3 * expert, adapter_slot,
                    row_base + row, lora_low + warp * kMoeLoraMaxRank);
                    dot[row] = moe_lora_adjust(adapters, 4 + 3 * expert, adapter_slot, row_base + row, dot[row]); }
                paths[warp][row] = weight * dot[row];
            }
        }
    } else if constexpr (kHasShared) {
        float dot[Rows];
        dot_fp32_rows<W8Codec, Rows>(shared_codes, nullptr, shared_scales, row_base,
                                     act + static_cast<std::int64_t>(kTopK) * kIntermediate, 0,
                                     kIntermediate / W8Codec::kGroupK, dot);
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                if (adapters) { dot[row] += moe_lora_expand(adapters, 4 + 3 * kExperts, adapter_slot,
                    row_base + row, lora_low + kTopK * kMoeLoraMaxRank);
                    dot[row] = moe_lora_adjust(adapters, 4 + 3 * kExperts, adapter_slot, row_base + row, dot[row]); }
                paths[kTopK][row] = *shared_scale * dot[row];
            }
        }
    }
    __syncthreads();
    if (warp == 0 && lane < Rows) {
        float value = __bfloat162float(destination[row_base + lane]);
#pragma unroll
        for (int path = 0; path < kPaths; ++path) { value += paths[path][lane]; }
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
    __shared__ float paths[kPaths][Rows];
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
            token_activations + static_cast<std::int64_t>(token) * kPaths * kIntermediate;
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
        } else if constexpr (kHasShared) {
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
            for (int path = 0; path < kPaths; ++path) { value += paths[path][lane]; }
            *output = __float2bfloat16_rn(value);
        }
        if constexpr (Adaptive) { __syncthreads(); }
    }
}

void launch_d1(const Tensor& x, const Weight& router_shared_gate,
               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream) {
    sparse_moe_d1_kernel<<<dim3(kRouterRows, x.ne[1]), kD1Warps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(router_shared_gate.qdata),
        static_cast<float*>(workspace.scratch.data));
    CUDA_CHECK(cudaGetLastError());
}

template <class Codec>
void launch_d3_dependent_codec(const Tensor& x, const SparseMoeWeights& weights,
                               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream,
                               const LoraBank* adapters = nullptr, const int* adapter_slot = nullptr) {
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    const auto* ids           = static_cast<const int*>(workspace.ids.data);
    auto* act                 = static_cast<float*>(workspace.scratch.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_gate_up.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    CUDA_CHECK(pdl::launch_dependent(
        {dim3(kIntermediate, x.ne[1]), dim3(kPaths * 32), 0, stream},
        sparse_moe_d3_nine_warp_kernel<Codec>,
        input, ids, routed_codes, routed_high, routed_scales, weights.slot_of_expert,
        weights.routed_gate_up_scale, shared_codes, shared_scales, act,
        adapters, adapter_slot, static_cast<const float*>(workspace.lora_low.data), workspace.slot_stride));
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
                  const SparseMoeRoundHook* hook,
                  const LoraBank* adapters = nullptr, const int* adapter_slot = nullptr) {
    const auto* scores = static_cast<const float*>(workspace.scratch.data);
    auto* ids          = static_cast<int*>(workspace.ids.data);
    auto* alpha        = static_cast<float*>(workspace.alpha.data);
    auto* shared_scale = static_cast<float*>(workspace.shared_scale.data);
    sparse_moe_d2_warp_kernel<<<dim3(1, x.ne[1]), 32, 0, stream>>>(scores, ids, alpha, shared_scale,
                                                   weights.router_bias,
                                                   weights.per_expert_scale);
    CUDA_CHECK(cudaGetLastError());
    // The round hook sees the final ids before the expert kernels; its launches sit between
    // D2 and D3 on the stream, so D3's programmatic wait resolves against them.
    if (hook != nullptr && hook->resolve != nullptr) {
        hook->resolve(hook->context, workspace.ids, workspace.alpha, x, destination, stream);
    }

    if (adapters) {
        moe_lora_low_kernel<<<dim3(kMoeLoraMaxRank, 2 * kPaths, x.ne[1]), 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), nullptr, ids, adapters, adapter_slot,
            static_cast<float*>(workspace.lora_low.data), false, workspace.slot_stride);
        CUDA_CHECK(cudaGetLastError());
    }
    d3_probe().begin(stream);
    switch (weights.routed_gate_up.qtype) {
    case QType::Q4G64_F16S:
        launch_d3_dependent_codec<Q4Codec>(x, weights, workspace, stream, adapters, adapter_slot);
        break;
    case QType::W8G32_F16S:
        launch_d3_dependent_codec<W8Codec>(x, weights, workspace, stream, adapters, adapter_slot);
        break;
    case QType::NVFP4:
        launch_d3_dependent_codec<Nvfp4CodecFor<kHidden / 16>>(x, weights, workspace, stream, adapters, adapter_slot);
        break;
#define SINFER_D3_DEP(NAME)                                                                        \
    case QType::NAME:                                                                              \
        launch_d3_dependent_codec<ggml::GgmlMoeCodec<ggml::GgmlType::NAME>>(x, weights, workspace,  \
                                                                            stream, adapters, adapter_slot); \
        break;
        SINFER_GGML_FOR_EACH_TYPE(SINFER_D3_DEP)
#undef SINFER_D3_DEP
    default:
        throw std::invalid_argument("sparse_moe: unsupported D3 codec");
    }
    d3_probe().end(stream);
}

template <class Codec>
void require_d4_codec_geometry() {
    // The packed D4 loop bounds its last warp step, so any group count works.
}

template <class Codec>
void launch_d4_dependent_codec(const SparseMoeWeights& weights, Tensor& destination,
                               const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream,
                               const LoraBank* adapters = nullptr, const int* adapter_slot = nullptr) {
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
    // Output rows one warp carries for a single token. A warp re-reads its path's activation
    // slice once per row it owns, and a block pays its barrier and epilogue once per block, so
    // short rows want several per warp: four at FFN 512 was 12 % off D4, one at FFN 2048
    // was the best of the three there. Measured on a 5090, cold, over Q4_K/Q5_K/Q6_K.
    constexpr int kD4Rows = kIntermediate <= 512 ? 4 : kIntermediate <= 1024 ? 2 : 1;
    static_assert(kHidden % kD4Rows == 0);
    CUDA_CHECK(pdl::launch_dependent({dim3(kHidden / kD4Rows, destination.ne[1]), dim3(kPaths * 32), 0, stream},
                                     sparse_moe_d4_nine_warp_kernel<Codec, kD4Rows>, ids, alpha,
                                     shared_scale, act, routed_codes, routed_high, routed_scales,
                                     weights.slot_of_expert, weights.routed_down_scale,
                                     shared_codes, shared_scales, output, adapters, adapter_slot,
                                     static_cast<const float*>(workspace.lora_low.data), workspace.slot_stride));
}

void launch_d4_dependent(const SparseMoeWeights& weights, Tensor& destination,
                         const SparseMoeDecodeWorkspace& workspace, cudaStream_t stream,
                               const LoraBank* adapters = nullptr, const int* adapter_slot = nullptr) {
    switch (weights.routed_down.qtype) {
    case QType::Q5G64_F16S:
        launch_d4_dependent_codec<Q5Codec>(weights, destination, workspace, stream, adapters, adapter_slot);
        return;
    case QType::Q6G64_F16S:
        launch_d4_dependent_codec<Q6Codec>(weights, destination, workspace, stream, adapters, adapter_slot);
        return;
#define SINFER_D4_DEP(NAME)                                                                        \
    case QType::NAME:                                                                              \
        launch_d4_dependent_codec<ggml::GgmlMoeCodec<ggml::GgmlType::NAME>>(weights, destination,   \
                                                                            workspace, stream, adapters, adapter_slot);     \
        return;
        SINFER_GGML_FOR_EACH_TYPE(SINFER_D4_DEP)
#undef SINFER_D4_DEP
    case QType::W8G32_F16S:
        launch_d4_dependent_codec<W8Codec>(weights, destination, workspace, stream, adapters, adapter_slot);
        return;
    case QType::NVFP4:
        launch_d4_dependent_codec<Nvfp4CodecFor<kIntermediate / 16>>(weights, destination,
                                                                     workspace, stream, adapters, adapter_slot);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported D4 codec");
    }
}

template <class Codec, int PathsPerBlock, bool Adaptive>
void launch_d3_small_t_paths(const Tensor& x, const SparseMoeWeights& weights, const int* token_ids,
                             float* token_activations, std::int32_t tokens, cudaStream_t stream,
                             const int* adaptive_route_jobs) {
    constexpr int kPathBlocks = kPaths / PathsPerBlock;
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
        if constexpr (kPaths % 3 == 0) {
            launch_d3_small_t_paths<Codec, 3, Adaptive>(x, weights, token_ids,
                                                        token_activations, tokens, stream,
                                                        adaptive_route_jobs);
        } else {
            launch_d3_small_t_paths<Codec, kPaths, Adaptive>(x, weights, token_ids,
                                                                token_activations, tokens,
                                                                stream, adaptive_route_jobs);
        }
        return;
    case SparseMoeSmallTD3Schedule::PathsAll:
        launch_d3_small_t_paths<Codec, kPaths, Adaptive>(x, weights, token_ids,
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
    sparse_moe_d4_token_kernel<Codec, Rows, Adaptive><<<grid, kPaths * 32, 0, stream>>>(
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
#define SINFER_D3_SMALL(NAME)                                                                      \
    case QType::NAME:                                                                              \
        launch_d3_small_t_codec<ggml::GgmlMoeCodec<ggml::GgmlType::NAME>, false>(                  \
            x, weights, token_ids, token_activations, tokens, schedule, stream, nullptr);           \
        return;
        SINFER_GGML_FOR_EACH_TYPE(SINFER_D3_SMALL)
#undef SINFER_D3_SMALL
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
#define SINFER_D4_SMALL(NAME)                                                                      \
    case QType::NAME:                                                                              \
        launch_d4_small_t_codec<ggml::GgmlMoeCodec<ggml::GgmlType::NAME>, false>(                  \
            weights, destination, token_ids, token_alpha, shared_scale, token_activations, tokens,  \
            schedule, stream, nullptr);                                                             \
        return;
        SINFER_GGML_FOR_EACH_TYPE(SINFER_D4_SMALL)
#undef SINFER_D4_SMALL
    default:
        throw std::invalid_argument("sparse_moe: unsupported small-T D4 codec");
    }
}

// Every mixture here but Gemma 4 projects the router and the experts from the same tokens, and
// passes the same tensor twice. Gemma 4 does not: its router reads the post-attention residual
// normalised with no weight and scaled by a learned per-channel vector, its experts the same
// residual through `pre_feedforward_layernorm_2`. Two projections of one token.
void decode_launch(const Tensor& x, const Tensor& router_x, const SparseMoeWeights& weights,
                   Tensor& destination, const SparseMoeDecodeWorkspace& workspace,
                   cudaStream_t stream, const SparseMoeRoundHook* hook,
                   const LoraBank* adapters, const int* adapter_slot) {
    launch_d1(router_x, weights.router_shared_gate, workspace, stream);
    if (adapters) {
        moe_lora_router_kernel<<<dim3(kRouterRows, x.ne[1]), 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(router_x.data), adapters, adapter_slot,
            static_cast<float*>(workspace.scratch.data), workspace.slot_stride);
        CUDA_CHECK(cudaGetLastError());
    }
    launch_d2_d3(x, weights, destination, workspace, stream, hook, adapters, adapter_slot);
    if (adapters) {
        moe_lora_low_kernel<<<dim3(kMoeLoraMaxRank, kPaths, x.ne[1]), 32, 0, stream>>>(
            nullptr, static_cast<const float*>(workspace.scratch.data),
            static_cast<const int*>(workspace.ids.data), adapters, adapter_slot,
            static_cast<float*>(workspace.lora_low.data), true, workspace.slot_stride);
        CUDA_CHECK(cudaGetLastError());
    }
    launch_d4_dependent(weights, destination, workspace, stream, adapters, adapter_slot);
}

