#pragma once

// ninfer::ops - signed int8, per-token group-wise KV cache codec (shared device
// helpers). Quantization (append) and dequantization (stage) are FUSED into the
// GQA attention kernels themselves (decode partial kernel, prefill fill/attention);
// this header only provides the index math, the vectorized dequant, and the scalar
// quantize helper they share. There is deliberately no standalone quant/dequant
// kernel: that would defeat the halved-bandwidth goal.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaKvQuantHeadDim = 256;
inline constexpr int kGqaKvQuantGroup   = 64;
inline constexpr int kGqaKvQuantGroups  = kGqaKvQuantHeadDim / kGqaKvQuantGroup;

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_code_index(int physical_page, int kv_head,
                                                                int d, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantHeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_scale_index(int physical_page, int kv_head,
                                                                 int group, int page_offset) {
    return paged_kv_element_offset<kGqaKvQuantGroups, Geometry::KVHeads>(physical_page, kv_head,
                                                                         page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_quant_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaKvQuantHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

// Quantize one bf16 value with a precomputed 1/scale (scale is the FP16-rounded
// per-group absmax/127). Round-to-nearest-even + symmetric clamp to keep codes
// bit-identical to the CPU oracle and to bf16 parity.
__device__ __forceinline__ std::int8_t gqa_kv_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

// Dequantize 8 consecutive int8 codes (dims [d, d+8), aligned to a multiple of 8
// so they lie inside one 64-group) into 8 bf16 packed as an int4, given a pointer
// to the 8 codes and the group's dequant scale. The codes are read with ONE 64-bit
// (int2) load; the pointer may be in global or shared memory. This keeps the dequant
// ALU identical whether the codes were streamed via cp.async into smem (decode) or
// read directly from the cache (prefill).
__device__ __forceinline__ int4 gqa_kv_dequant_i8x8_from(const std::int8_t* codes8, float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// ---------------------------------------------------------------------------
// E4M3 KV codec. Unlike the int8 codec above there is no scale plane and no
// group: a code is the raw e4m3 byte, which is what vLLM stores for
// --kv-cache-dtype fp8 when no calibration is present (its k_scale/v_scale
// default to 1.0, so its scaled_convert reduces to a cast). Values beyond
// e4m3's 448 saturate rather than rescaling, matching that behaviour.
//
// Storage is 1 byte per element with no side planes, so an fp8 cache is
// slightly smaller than the int8 one, which carries an FP16 scale per 64
// values. Compute stays bf16: the codes are widened at stage time and fed to
// the same MMA as the bf16 cache, so this buys memory traffic rather than
// tensor-core throughput -- which is the side decode attention is short on.
__device__ __forceinline__ std::uint8_t gqa_kv_fp8_code(float x) {
    return static_cast<std::uint8_t>(
        __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

// Widen 8 e4m3 codes already held in a register pair. Splitting the load from
// the conversion lets a caller issue several cache reads before spending ALU on
// any of them, which matters because the e4m3 stage cannot use cp.async: it has
// to touch the values, so the only latency hiding available is issuing the
// loads early.
__device__ __forceinline__ int4 gqa_kv_dequant_fp8x8_raw(int2 raw) {
    const std::uint8_t* c = reinterpret_cast<const std::uint8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(
            static_cast<__nv_fp8x2_storage_t>(static_cast<std::uint16_t>(c[2 * i]) |
                                              (static_cast<std::uint16_t>(c[2 * i + 1]) << 8)),
            __NV_E4M3);
        const __half2 hv = *reinterpret_cast<const __half2*>(&h);
        packed[i]        = pack_bf16x2(__low2float(hv), __high2float(hv));
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Widen 8 consecutive e4m3 codes into 8 bf16 packed as an int4, reading the
// codes with one 64-bit load. Mirrors gqa_kv_dequant_i8x8_from so the staging
// paths differ only in the codec they name. e4m3 is exactly representable in
// fp16, so the intermediate conversion is lossless.
__device__ __forceinline__ int4 gqa_kv_dequant_fp8x8_from(const std::uint8_t* codes8) {
    return gqa_kv_dequant_fp8x8_raw(load_vec<int2>(codes8));
}

// Narrow 8 consecutive bf16 values to 8 e4m3 codes with one 16-byte load and
// one 8-byte store, the append-side mirror of gqa_kv_dequant_fp8x8_from.
__device__ __forceinline__ void gqa_kv_store_fp8x8(std::uint8_t* dst8,
                                                   const __nv_bfloat16* src8) {
    const int4 raw            = load_vec<int4>(src8);
    const __nv_bfloat16* vals = reinterpret_cast<const __nv_bfloat16*>(&raw);
    std::uint8_t codes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { codes[i] = gqa_kv_fp8_code(__bfloat162float(vals[i])); }
    store_vec(dst8, *reinterpret_cast<const int2*>(codes));
}

// Selects the cache codec at compile time without pulling in <type_traits>:
// the KV kernels are shared between the bf16 cache and the e4m3 one, and this
// is the only thing that distinguishes their storage.
template <typename T>
struct GqaKvIsFp8 {
    static constexpr bool value = false;
};
template <>
struct GqaKvIsFp8<std::uint8_t> {
    static constexpr bool value = true;
};

} // namespace ninfer::ops
