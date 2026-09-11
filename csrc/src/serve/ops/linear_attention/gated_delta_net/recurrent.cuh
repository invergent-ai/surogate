#pragma once

#include "ops/common/bf16_vector.cuh"
#include "ops/linear_attention/gated_delta_net/common.cuh"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

namespace sinfer::ops::detail::gated_delta_net {

inline constexpr int kDvPerWarp = 4;
inline constexpr int kNumWarps  = 4;
inline constexpr int kBlockDv   = kNumWarps * kDvPerWarp;
inline constexpr int kQkPerLane = kStateDim / kWarpSize;

static_assert(kStateDim % kWarpSize == 0);
static_assert(kQkPerLane == 4);
static_assert(kStateDim % kBlockDv == 0);

// Storage type of the recurrent state. Compute is fp32 regardless; this is the
// width it occupies in HBM, and it is the single largest cost in a decode
// round. Declared in one place so a target or a quality-sensitive deployment
// can move it back to float without touching kernels.
using GdnStateStorage = __nv_bfloat16;

// The recurrent state is held in registers as fp32 and computed in fp32; only
// its STORAGE width is a choice. At 32 value heads x 128 x 128 the state is 2
// MiB per lane per layer, so at 64 lanes across 24 GDN layers a decode round
// moves ~6 GiB of it in fp32 — 22.6% of device time, measured at 71% of peak
// bandwidth, which makes this the largest single cost in a decode round and the
// reason lane count never bought throughput. Storing bf16 halves that traffic.
// (vLLM's gated_delta_net_state_dtype defaults to "auto", i.e. the model dtype,
// so bf16 storage is what the comparison engine already does.)
__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const float* base,
                                             std::uint32_t dqk_base) {
    store_vec(reg, load_vec<float4>(base + dqk_base));
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane], float* base,
                                              std::uint32_t dqk_base) {
    store_vec(base + dqk_base, load_vec<float4>(reg));
}

__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const __nv_bfloat16* base,
                                             std::uint32_t dqk_base) {
    static_assert(kQkPerLane == 4, "bf16 state lane load assumes a 4-wide lane");
    const __nv_bfloat162* pair = reinterpret_cast<const __nv_bfloat162*>(base + dqk_base);
    const float2 lo            = __bfloat1622float2(pair[0]);
    const float2 hi            = __bfloat1622float2(pair[1]);
    reg[0]                     = lo.x;
    reg[1]                     = lo.y;
    reg[2]                     = hi.x;
    reg[3]                     = hi.y;
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane], __nv_bfloat16* base,
                                              std::uint32_t dqk_base) {
    static_assert(kQkPerLane == 4, "bf16 state lane store assumes a 4-wide lane");
    __nv_bfloat162* pair = reinterpret_cast<__nv_bfloat162*>(base + dqk_base);
    pair[0]              = __floats2bfloat162_rn(reg[0], reg[1]);
    pair[1]              = __floats2bfloat162_rn(reg[2], reg[3]);
}

// `static`: a non-template kernel defined in a header has external linkage, so a second
// translation unit including this file collides with the first at link time -- which is what
// happened the moment the Kimi delta rule reused these lane helpers. Every other kernel here is
// a template and already has the linkage a header needs.
static __global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_fp32_kernel(const float* __restrict__ q, const float* __restrict__ k,
                          const float* __restrict__ v, const float* __restrict__ g,
                          const float* __restrict__ beta, float* __restrict__ ssm_state,
                          float* __restrict__ out, std::int64_t T, head_map heads, float scale) {
    const int lane           = threadIdx.x;
    const int warp_id        = threadIdx.y;
    const std::uint32_t h_v  = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t h_qk = static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(h_v)));

    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(blockIdx.z * kBlockDv + warp_id * kDvPerWarp);
    const std::uint32_t dqk_base = static_cast<std::uint32_t>(lane * kQkPerLane);

    float* state_h = ssm_state + static_cast<std::int64_t>(h_v) * kStateDim * kStateDim;

    __align__(16) float s_tile[kDvPerWarp][kQkPerLane];
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(s_tile[r], state_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                     dqk_base);
    }

    __align__(16) float k_reg[kQkPerLane];
    load_qk_lane(k_reg, k + static_cast<std::int64_t>(h_qk) * kStateDim, dqk_base);

    for (std::int64_t t = 0; t < T; ++t) {
        const float* v_t          = v + (t * heads.H_v + h_v) * kStateDim;
        const std::int64_t gb_off = t * heads.H_v + h_v;
        const float beta_val      = beta[gb_off];
        const float alpha         = expf(g[gb_off]);

        float v_local = 0.0f;
        if (lane < kDvPerWarp) { v_local = v_t[dv_base + lane]; }

#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) { partial += s_tile[r][c] * k_reg[c]; }
            partial = warp_sum<kWarpSize>(partial);

            const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
            const float delta = beta_val * (v_r - alpha * partial);

#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) {
                s_tile[r][c] = alpha * s_tile[r][c] + delta * k_reg[c];
            }
        }

        if (t + 1 < T) {
            load_qk_lane(k_reg, k + ((t + 1) * heads.H_qk + h_qk) * kStateDim, dqk_base);
        }

        __align__(16) float q_reg[kQkPerLane];
        load_qk_lane(q_reg, q + (t * heads.H_qk + h_qk) * kStateDim, dqk_base);

        float attn_val = 0.0f;
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < kQkPerLane; ++c) { partial += s_tile[r][c] * q_reg[c]; }
            partial = warp_sum<kWarpSize>(partial);
            if (lane == r) { attn_val = partial; }
        }

        if (lane < kDvPerWarp) {
            out[(t * heads.H_v + h_v) * kStateDim + dv_base + lane] = attn_val * scale;
        }
    }

#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_qk_lane(s_tile[r], state_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                      dqk_base);
    }
}

inline constexpr float kQkL2NormEps = 1.0e-6f;

struct RawQkLane {
    Bf16x4Pack bits;
    float value[kQkPerLane];
};

struct RawValueLane {
    __nv_bfloat16 bits;
    float value;
};

struct RawGatePair {
    uint2 bits;
    float g;
    float beta;
};

/// Which forget gate a recurrence has. The delta net's is one value per head, so it multiplies
/// the whole state and can be pulled out of the dot product; Kimi Delta Attention's is one per
/// *key channel*, so the state decays by a diagonal and the gate belongs inside. Everything
/// else about the two recurrences -- the tiling, the lane loads, the readout, the snapshot --
/// is the same, which is why they share this loop instead of each having one.
enum class ForgetGate { Scalar, Diagonal };

/// A diagonal gate as a lane sees it: the key channels it owns, and the head's beta.
struct RawDiagonalGate {
    float channel[kQkPerLane];
    float beta;
};

/// The payload the loop carries for a gate of either shape.
template <ForgetGate Gate>
using GateOf = std::conditional_t<Gate == ForgetGate::Diagonal, RawDiagonalGate, RawGatePair>;

__device__ __forceinline__ RawQkLane load_raw_qk_lane(const __nv_bfloat16* base,
                                                      std::uint32_t dqk_base) {
    RawQkLane out;
    out.bits        = load_vec<Bf16x4Pack>(base + dqk_base);
    const float2 lo = bf16x2_to_float2(out.bits.pair[0]);
    const float2 hi = bf16x2_to_float2(out.bits.pair[1]);
    out.value[0]    = lo.x;
    out.value[1]    = lo.y;
    out.value[2]    = hi.x;
    out.value[3]    = hi.y;
    return out;
}

template <bool Normalize>
__device__ __forceinline__ void normalize_qk_lane(float (&value)[kQkPerLane], int lane) {
    if constexpr (Normalize) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { sum += value[i] * value[i]; }
        sum       = warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum + kQkL2NormEps) : 0.0f;
        inv       = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
        // Chunked prefill materializes l2norm into BF16 tensors. Keep the same
        // rounding boundary when normalization is fused into a short recurrence,
        // speculative verification, or replay of an accepted prefix.
        for (int i = 0; i < kQkPerLane; ++i) {
            value[i] = __bfloat162float(__float2bfloat16_rn(value[i] * inv));
        }
    }
}

__device__ __forceinline__ RawValueLane load_value_lane(const __nv_bfloat16* base, int lane,
                                                        std::uint32_t dv_base) {
    RawValueLane out{__float2bfloat16(0.0f), 0.0f};
    if (lane < kDvPerWarp) {
        out.bits  = base[dv_base + lane];
        out.value = __bfloat162float(out.bits);
    }
    return out;
}

__device__ __forceinline__ RawGatePair load_source_gate(const float* g, const float* beta,
                                                        std::int64_t offset) {
    const float g_value    = g[offset];
    const float beta_value = beta[offset];
    return {make_uint2(__float_as_uint(g_value), __float_as_uint(beta_value)), g_value, beta_value};
}

__device__ __forceinline__ RawGatePair load_record_gate(const uint2* gate, std::int64_t offset) {
    const uint2 bits = load_vec<uint2>(gate + offset);
    return {bits, __uint_as_float(bits.x), __uint_as_float(bits.y)};
}

__device__ __forceinline__ void apply_gdn_transition(float (&state)[kDvPerWarp][kQkPerLane],
                                                     const float (&key)[kQkPerLane], float v_local,
                                                     float g, float beta) {
    const float alpha = expf(g);

#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * key[c]; }
        partial = warp_sum<kWarpSize>(partial);

        const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
        const float delta = beta * (v_r - alpha * partial);

#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { state[r][c] = alpha * state[r][c] + delta * key[c]; }
    }
}

/// One token of the recurrence with a diagonal forget gate.
///
///   partial_r = sum_c S[r][c] * alpha[c] * k[c]      the decayed state's prediction
///   delta_r   = beta * (v_r - partial_r)
///   S[r][c]   = alpha[c] * S[r][c] + delta_r * k[c]
///
/// Set every alpha to the same value and this is `apply_gdn_transition` exactly, which is the
/// check to make when reading the two side by side.
__device__ __forceinline__ void apply_kda_transition(float (&state)[kDvPerWarp][kQkPerLane],
                                                     const float (&key)[kQkPerLane],
                                                     const float (&gate)[kQkPerLane],
                                                     float v_local, float beta) {
    float alpha[kQkPerLane];
#pragma unroll
    for (int c = 0; c < kQkPerLane; ++c) { alpha[c] = __expf(gate[c]); }

#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * alpha[c] * key[c]; }
        partial = warp_sum<kWarpSize>(partial);

        const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
        const float delta = beta * (v_r - partial);

#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) {
            state[r][c] = alpha[c] * state[r][c] + delta * key[c];
        }
    }
}

/// The transition either gate asks for, so the shared loop names one thing.
template <ForgetGate Gate>
__device__ __forceinline__ void apply_transition(float (&state)[kDvPerWarp][kQkPerLane],
                                                 const float (&key)[kQkPerLane], float v_local,
                                                 const GateOf<Gate>& gate) {
    if constexpr (Gate == ForgetGate::Diagonal) {
        apply_kda_transition(state, key, gate.channel, v_local, gate.beta);
    } else {
        apply_gdn_transition(state, key, v_local, gate.g, gate.beta);
    }
}

template <bool Normalize>
__device__ __forceinline__ void readout_and_store(float (&state)[kDvPerWarp][kQkPerLane],
                                                  const __nv_bfloat16* query, __nv_bfloat16* output,
                                                  std::uint32_t dqk_base, std::uint32_t dv_base,
                                                  int lane, float scale) {
    RawQkLane q = load_raw_qk_lane(query, dqk_base);
    normalize_qk_lane<Normalize>(q.value, lane);

    float attn_val = 0.0f;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * q.value[c]; }
        partial = warp_sum<kWarpSize>(partial);
        if (lane == r) { attn_val = partial; }
    }
    if (lane < kDvPerWarp) { output[dv_base + lane] = __float2bfloat16(attn_val * scale); }
}

template <bool NormalizeQK>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_bf16_direct_kernel(const __nv_bfloat16* __restrict__ q,
                                 const __nv_bfloat16* __restrict__ k,
                                 const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                 const float* __restrict__ beta,
                                 const GdnStateStorage* __restrict__ state_read,
                                 GdnStateStorage* __restrict__ state_write,
                                 __nv_bfloat16* __restrict__ out, std::int32_t width, head_map heads,
                                 float scale) {
    const int lane           = threadIdx.x;
    const int warp_id        = threadIdx.y;
    const std::uint32_t h_v  = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t h_qk = static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(h_v)));
    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(blockIdx.z * kBlockDv + warp_id * kDvPerWarp);
    const std::uint32_t dqk_base = static_cast<std::uint32_t>(lane * kQkPerLane);
    const GdnStateStorage* read_h =
        state_read + static_cast<std::int64_t>(h_v) * kStateDim * kStateDim;

    __align__(16) float state[kDvPerWarp][kQkPerLane];
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(state[r], read_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                     dqk_base);
    }

    RawQkLane key = load_raw_qk_lane(k + static_cast<std::int64_t>(h_qk) * kStateDim, dqk_base);
    normalize_qk_lane<NormalizeQK>(key.value, lane);
    for (std::int32_t token = 0; token < width; ++token) {
        const std::int64_t column = token;
        const RawGatePair gate    = load_source_gate(g, beta, column * heads.H_v + h_v);
        const RawValueLane value =
            load_value_lane(v + (column * heads.H_v + h_v) * kStateDim, lane, dv_base);
        apply_gdn_transition(state, key.value, value.value, gate.g, gate.beta);

        if (token + 1 < width) {
            key = load_raw_qk_lane(k + ((column + 1) * heads.H_qk + h_qk) * kStateDim, dqk_base);
            normalize_qk_lane<NormalizeQK>(key.value, lane);
        }

        readout_and_store<NormalizeQK>(state, q + (column * heads.H_qk + h_qk) * kStateDim,
                                       out + (column * heads.H_v + h_v) * kStateDim, dqk_base,
                                       dv_base, lane, scale);
    }

    GdnStateStorage* write_h = state_write + static_cast<std::int64_t>(h_v) * kStateDim * kStateDim;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_qk_lane(state[r], write_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                      dqk_base);
    }
}

enum class RecurrentMode {
    Snapshot,
    Record,
    Fold,
};

struct RecurrentCoordinates {
    int lane;
    int warp;
    std::int32_t batch;
    std::int32_t layer;
    std::int32_t state_tile;
    std::uint32_t value_head;
    std::uint32_t qk_head;
    std::uint32_t dv_base;
    std::uint32_t dqk_base;
};

__device__ __forceinline__ RecurrentCoordinates make_coordinates(std::int32_t batch,
                                                                 std::int32_t layer,
                                                                 std::int32_t state_tile,
                                                                 head_map heads) {
    const int lane                 = threadIdx.x;
    const int warp                 = threadIdx.y;
    const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t qk_head =
        static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(value_head)));
    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
    return {lane,    warp,       batch,
            layer,   state_tile, value_head,
            qk_head, dv_base,    static_cast<std::uint32_t>(lane * kQkPerLane)};
}

template <bool Batched, bool Masked, ForgetGate Gate = ForgetGate::Scalar>
struct SnapshotAccess {
    static constexpr ForgetGate kGate = Gate;

    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    GdnStateStorage* states;
    const std::int32_t* valid_columns;
    const std::int32_t* initial_slots;
    const std::int32_t* snapshot_bases;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        const std::int32_t batch = Batched ? static_cast<std::int32_t>(blockIdx.y) : 0;
        return make_coordinates(batch, 0, static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        if constexpr (Masked) { return valid_columns[coord.batch]; }
        return width;
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        return static_cast<std::int64_t>(coord.batch) * width + token;
    }

    __device__ __forceinline__ const GdnStateStorage*
    state_read_base(const RecurrentCoordinates& coord) const {
        return states + static_cast<std::int64_t>(initial_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ GateOf<Gate> load_gate(const RecurrentCoordinates& coord,
                                                      std::int32_t token) const {
        const std::int64_t offset = column(coord, token) * heads.H_v + coord.value_head;
        if constexpr (Gate == ForgetGate::Diagonal) {
            // A diagonal gate is laid out like v -- one value per channel per head per token --
            // so a lane reads the same key channels of it that it owns of the key.
            RawDiagonalGate out_gate;
            load_qk_lane(out_gate.channel, g + offset * kStateDim, coord.dqk_base);
            out_gate.beta = beta[offset];
            return out_gate;
        } else {
            return load_source_gate(g, beta, offset);
        }
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ void
    store_snapshot(const RecurrentCoordinates& coord, std::int32_t token,
                   const float (&state)[kDvPerWarp][kQkPerLane]) const {
        GdnStateStorage* snapshot =
            states +
            static_cast<std::int64_t>(snapshot_bases[coord.batch] + token) * state_slot_stride +
            static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            store_qk_lane(state[r],
                          snapshot + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                          coord.dqk_base);
        }
    }
};

/// The record plane a gate of either shape writes: {g, beta} pairs for a scalar gate, the key
/// channels of g for a diagonal one, whose beta then has a plane of its own.
template <ForgetGate Gate>
using GateRecordOf = std::conditional_t<Gate == ForgetGate::Diagonal, float, uint2>;

template <bool Masked, ForgetGate Gate = ForgetGate::Scalar>
struct RecordAccess {
    static constexpr ForgetGate kGate = Gate;

    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const GdnStateStorage* states;
    const std::int32_t* valid_columns;
    const std::int32_t* initial_slots;
    __nv_bfloat16* key_record;
    __nv_bfloat16* value_record;
    GateRecordOf<Gate>* gate_record;
    float* beta_record; // diagonal gate only; null for a scalar one
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(static_cast<std::int32_t>(blockIdx.y), 0,
                                static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        if constexpr (Masked) { return valid_columns[coord.batch]; }
        return width;
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        return static_cast<std::int64_t>(coord.batch) * width + token;
    }

    __device__ __forceinline__ const GdnStateStorage*
    state_read_base(const RecurrentCoordinates& coord) const {
        return states + static_cast<std::int64_t>(initial_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ GateOf<Gate> load_gate(const RecurrentCoordinates& coord,
                                                      std::int32_t token) const {
        const std::int64_t offset = column(coord, token) * heads.H_v + coord.value_head;
        if constexpr (Gate == ForgetGate::Diagonal) {
            RawDiagonalGate out_gate;
            load_qk_lane(out_gate.channel, g + offset * kStateDim, coord.dqk_base);
            out_gate.beta = beta[offset];
            return out_gate;
        } else {
            return load_source_gate(g, beta, offset);
        }
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ void store_key(const RecurrentCoordinates& coord, std::int32_t token,
                                              const RawQkLane& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 &&
            static_cast<int>(coord.value_head) % heads.group_size() == 0) {
            __nv_bfloat16* destination =
                key_record + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
            store_vec(destination + coord.dqk_base, raw.bits);
        }
    }

    __device__ __forceinline__ void store_value(const RecurrentCoordinates& coord,
                                                std::int32_t token, const RawValueLane& raw) const {
        if (coord.lane < kDvPerWarp) {
            __nv_bfloat16* destination =
                value_record + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
            destination[coord.dv_base + coord.lane] = raw.bits;
        }
    }

    __device__ __forceinline__ void store_gate(const RecurrentCoordinates& coord,
                                               std::int32_t token, const GateOf<Gate>& raw) const {
        if (coord.state_tile != 0 || coord.warp != 0) { return; }
        const std::int64_t offset = column(coord, token) * heads.H_v + coord.value_head;
        if constexpr (Gate == ForgetGate::Diagonal) {
            // One warp per (head, token) writes the gate: every lane its own four channels, the
            // same ones it read, and lane 0 the head's beta.
            store_qk_lane(raw.channel, gate_record + offset * kStateDim, coord.dqk_base);
            if (coord.lane == 0) { beta_record[offset] = raw.beta; }
        } else {
            if (coord.lane == 0) { gate_record[offset] = raw.bits; }
        }
    }
};

template <ForgetGate Gate = ForgetGate::Scalar>
struct FoldAccess {
    static constexpr ForgetGate kGate = Gate;

    const __nv_bfloat16* key_record;
    const __nv_bfloat16* value_record;
    const GateRecordOf<Gate>* gate_record;
    const float* beta_record; // diagonal gate only; null for a scalar one
    const __nv_bfloat16* conv_record;
    GdnStateStorage* recurrent_layer0;
    __nv_bfloat16* conv_layer0;
    std::int64_t recurrent_layer_stride;
    std::int64_t conv_layer_stride;
    std::int32_t record_capacity;
    std::int32_t width;
    std::int32_t qk_heads;
    std::int32_t value_heads;
    std::int32_t conv_channels;
    GdnReplayFoldKernelRows rows;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        const std::int32_t batch       = static_cast<std::int32_t>(blockIdx.y);
        const std::int32_t layer_tile  = static_cast<std::int32_t>(blockIdx.z);
        const int lane                 = threadIdx.x;
        const int warp                 = threadIdx.y;
        const std::int32_t state_tile  = layer_tile & 7;
        const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
        const std::uint32_t kGroup = value_heads / qk_heads;
        const std::uint32_t qk_head    = value_head / kGroup;
        const std::uint32_t dv_base =
            static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
        return {lane,
                warp,
                batch,
                layer_tile >> 3,
                state_tile,
                value_head,
                qk_head,
                dv_base,
                static_cast<std::uint32_t>(lane * kQkPerLane)};
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        return rows.row[coord.batch].commit_columns;
    }

    __device__ __forceinline__ std::int64_t record_outer(const RecurrentCoordinates& coord) const {
        return static_cast<std::int64_t>(coord.layer) * record_capacity + coord.batch;
    }

    __device__ __forceinline__ GdnStateStorage* state_read_base(const RecurrentCoordinates& coord) const {
        const std::int64_t slot_stride =
            static_cast<std::int64_t>(value_heads) * kStateDim * kStateDim;
        return recurrent_layer0 + static_cast<std::int64_t>(coord.layer) * recurrent_layer_stride +
               static_cast<std::int64_t>(rows.row[coord.batch].linear_state_slot) * slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return key_record + (column * qk_heads + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return value_record + (column * value_heads + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ GateOf<Gate> load_gate(const RecurrentCoordinates& coord,
                                                      std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        const std::int64_t offset = column * value_heads + coord.value_head;
        if constexpr (Gate == ForgetGate::Diagonal) {
            RawDiagonalGate out_gate;
            load_qk_lane(out_gate.channel, gate_record + offset * kStateDim, coord.dqk_base);
            out_gate.beta = beta_record[offset];
            return out_gate;
        } else {
            return load_record_gate(gate_record, offset);
        }
    }

    __device__ __forceinline__ void
    store_final_state(const RecurrentCoordinates& coord,
                      const float (&state)[kDvPerWarp][kQkPerLane]) const {
        GdnStateStorage* destination = state_read_base(coord);
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            store_qk_lane(state[r],
                          destination + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                          coord.dqk_base);
        }
    }

    __device__ __forceinline__ void publish_final_conv_history(const RecurrentCoordinates& coord,
                                                               std::int32_t commit) const {
        const std::int32_t tile_block =
            static_cast<std::int32_t>(coord.value_head) * 8 + coord.state_tile;
        if (tile_block >= conv_channels / 128) { return; }

        const std::int32_t tid     = coord.warp * kWarpSize + coord.lane;
        const std::int32_t channel = tile_block * 128 + tid;
        __nv_bfloat16* history =
            conv_layer0 + static_cast<std::int64_t>(coord.layer) * conv_layer_stride +
            static_cast<std::int64_t>(rows.row[coord.batch].linear_state_slot) *
                (3LL * conv_channels) +
            channel;
        const __nv_bfloat16* record =
            conv_record + record_outer(coord) * width * conv_channels + channel;

        __nv_bfloat16 h0;
        __nv_bfloat16 h1;
        __nv_bfloat16 h2;
        if (commit == 1) {
            h0 = history[conv_channels];
            h1 = history[2LL * conv_channels];
            h2 = record[0];
        } else if (commit == 2) {
            h0 = history[2LL * conv_channels];
            h1 = record[0];
            h2 = record[conv_channels];
        } else {
            h0 = record[static_cast<std::int64_t>(commit - 3) * conv_channels];
            h1 = record[static_cast<std::int64_t>(commit - 2) * conv_channels];
            h2 = record[static_cast<std::int64_t>(commit - 1) * conv_channels];
        }
        history[0]                             = h0;
        history[conv_channels]       = h1;
        history[2LL * conv_channels] = h2;
    }
};

template <RecurrentMode Mode, bool NormalizeInputs, class Access>
__device__ __forceinline__ void recurrent_bf16_body(const Access& access,
                                                    const RecurrentCoordinates& coord,
                                                    std::int32_t width, std::int32_t valid) {
    if constexpr (Mode == RecurrentMode::Fold) {
        if (valid == 0) { return; }
    }

    const GdnStateStorage* initial = access.state_read_base(coord);
    __align__(16) float state[kDvPerWarp][kQkPerLane];
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(state[r], initial + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                     coord.dqk_base);
    }

    RawQkLane key = load_raw_qk_lane(access.key_ptr(coord, 0), coord.dqk_base);
    if constexpr (Mode == RecurrentMode::Record) { access.store_key(coord, 0, key); }
    normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);

    for (std::int32_t token = 0; token < valid; ++token) {
        const GateOf<Access::kGate> gate = access.load_gate(coord, token);
        const RawValueLane value =
            load_value_lane(access.value_ptr(coord, token), coord.lane, coord.dv_base);
        if constexpr (Mode == RecurrentMode::Record) {
            access.store_value(coord, token, value);
            access.store_gate(coord, token, gate);
        }

        apply_transition<Access::kGate>(state, key.value, value.value, gate);

        if (token + 1 < valid) {
            key = load_raw_qk_lane(access.key_ptr(coord, token + 1), coord.dqk_base);
            if constexpr (Mode == RecurrentMode::Record) {
                access.store_key(coord, token + 1, key);
            }
            normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);
        }

        if constexpr (Mode != RecurrentMode::Fold) {
            readout_and_store<NormalizeInputs>(state, access.query_ptr(coord, token),
                                               access.output_ptr(coord, token), coord.dqk_base,
                                               coord.dv_base, coord.lane, access.scale);
        }
        if constexpr (Mode == RecurrentMode::Snapshot) {
            access.store_snapshot(coord, token, state);
        }
    }

    if constexpr (Mode == RecurrentMode::Fold) {
        access.store_final_state(coord, state);
        access.publish_final_conv_history(coord, valid);
    } else {
        if (coord.lane < kDvPerWarp) {
            for (std::int32_t token = valid; token < width; ++token) {
                access.output_ptr(coord, token)[coord.dv_base + coord.lane] =
                    __float2bfloat16(0.0f);
            }
        }
    }
}

template <bool NormalizeInputs, bool Batched, bool Masked, ForgetGate Gate = ForgetGate::Scalar>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_snapshot_kernel(SnapshotAccess<Batched, Masked, Gate> access) {
    static_assert(!Masked || Batched);
    const RecurrentCoordinates coord = access.coordinates();
    recurrent_bf16_body<RecurrentMode::Snapshot, NormalizeInputs>(access, coord, access.width,
                                                                  access.active_columns(coord));
}

template <bool Masked, ForgetGate Gate = ForgetGate::Scalar>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_record_kernel(RecordAccess<Masked, Gate> access) {
    const RecurrentCoordinates coord = access.coordinates();
    recurrent_bf16_body<RecurrentMode::Record, true>(access, coord, access.width,
                                                     access.active_columns(coord));
}

template <ForgetGate Gate = ForgetGate::Scalar>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_fold_kernel(const __grid_constant__ FoldAccess<Gate> access) {
    const RecurrentCoordinates coord = access.coordinates();
    recurrent_bf16_body<RecurrentMode::Fold, true>(access, coord, access.width,
                                                   access.active_columns(coord));
}

} // namespace sinfer::ops::detail::gated_delta_net
