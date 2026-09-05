// Kimi Delta Attention's recurrence: the gated delta net's, with the forget gate per key
// channel instead of per head.
//
// The tiling is the delta net's, deliberately: a warp owns `kDvPerWarp` rows of the state and a
// lane owns `kQkPerLane` key channels of them, and the whole state tile lives in registers for
// the length of the round. What changes is one function. There, `alpha` is a scalar and can be
// pulled out of the dot product; here each key channel has decayed by its own amount, so it
// belongs inside -- and the delta corrects the prediction the *decayed* state makes.

#include "ops/linear_attention/kimi_delta_net/launch.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

// The kernel lives in the delta net's own namespace: every helper it reuses -- the lane loads,
// the l2 normalisation, the readout -- is declared there, and reaching them from elsewhere would
// mean qualifying each one or re-declaring it. Only the launcher is this op's.
namespace sinfer::ops::detail::gated_delta_net {
namespace {

/// One token of the recurrence, with a diagonal forget gate.
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

template <bool NormalizeQK>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    kda_recurrent_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                         const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                         const float* __restrict__ beta,
                         const GdnStateStorage* __restrict__ state_read,
                         GdnStateStorage* __restrict__ state_write, __nv_bfloat16* __restrict__ out,
                         std::int32_t width, head_map heads, float scale) {
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
        load_qk_lane(state[r],
                           read_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                           dqk_base);
    }

    RawQkLane key =
        load_raw_qk_lane(k + static_cast<std::int64_t>(h_qk) * kStateDim, dqk_base);
    normalize_qk_lane<NormalizeQK>(key.value, lane);

    for (std::int32_t token = 0; token < width; ++token) {
        const std::int64_t column = token;
        const float beta_value    = beta[column * heads.H_v + h_v];
        // The gate is laid out like v -- one value per channel per head per token -- so a lane
        // reads the same four channels of it that it owns of the key.
        __align__(16) float gate[kQkPerLane];
        load_qk_lane(gate, g + (column * heads.H_v + h_v) * kStateDim, dqk_base);
        const RawValueLane value = load_value_lane(
            v + (column * heads.H_v + h_v) * kStateDim, lane, dv_base);
        apply_kda_transition(state, key.value, gate, value.value, beta_value);

        if (token + 1 < width) {
            key = load_raw_qk_lane(
                k + ((column + 1) * heads.H_qk + h_qk) * kStateDim, dqk_base);
            normalize_qk_lane<NormalizeQK>(key.value, lane);
        }

        readout_and_store<NormalizeQK>(
            state, q + (column * heads.H_qk + h_qk) * kStateDim,
            out + (column * heads.H_v + h_v) * kStateDim, dqk_base, dv_base, lane, scale);
    }

    GdnStateStorage* write_h =
        state_write + static_cast<std::int64_t>(h_v) * kStateDim * kStateDim;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_qk_lane(state[r],
                            write_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                            dqk_base);
    }
}

} // namespace

// Not in the anonymous namespace above: the op's own entry point forwards to it.
void kda_launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk,
                      const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                      cudaStream_t stream) {
    const std::int32_t qk_heads = k.ne[1];
    const std::int32_t v_heads  = v.ne[1];
    const std::int32_t width    = v.ne[2];
    const dim3 grid(static_cast<unsigned>(v_heads), 1u,
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(static_cast<unsigned>(kWarpSize), static_cast<unsigned>(kNumWarps));
    const auto heads = head_map::of(qk_heads, v_heads);
    const auto* qp   = static_cast<const __nv_bfloat16*>(q.data);
    const auto* kp   = static_cast<const __nv_bfloat16*>(k.data);
    const auto* vp   = static_cast<const __nv_bfloat16*>(v.data);
    const auto* gp   = static_cast<const float*>(g.data);
    const auto* bp   = static_cast<const float*>(beta.data);
    const auto* sr   = static_cast<const GdnStateStorage*>(ssm_state_in.data);
    auto* sw         = static_cast<GdnStateStorage*>(ssm_state_out.data);
    auto* op         = static_cast<__nv_bfloat16*>(out.data);

    if (normalize_qk) {
        kda_recurrent_kernel<true><<<grid, block, 0, stream>>>(qp, kp, vp, gp, bp, sr, sw, op,
                                                               width, heads, scale);
    } else {
        kda_recurrent_kernel<false><<<grid, block, 0, stream>>>(qp, kp, vp, gp, bp, sr, sw, op,
                                                                width, heads, scale);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail::gated_delta_net

namespace sinfer::ops::detail::kimi_delta_net {

void launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk,
                      const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                      cudaStream_t stream) {
    ::sinfer::ops::detail::gated_delta_net::kda_launch_recurrent(
        q, k, v, g, beta, scale, normalize_qk, ssm_state_in, ssm_state_out, out, stream);
}

} // namespace sinfer::ops::detail::kimi_delta_net
