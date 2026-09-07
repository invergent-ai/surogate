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

template <bool Masked>
void kda_launch_recurrent_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, float scale,
                                       const Tensor& ssm_states, const Tensor& valid_columns,
                                       const Tensor& initial_state_slots, Tensor& key_record,
                                       Tensor& value_record, Tensor& gate_record,
                                       Tensor& beta_record, Tensor& out, cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    const RecordAccess<Masked, ForgetGate::Diagonal> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const GdnStateStorage*>(ssm_states.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<__nv_bfloat16*>(value_record.data),
        static_cast<float*>(gate_record.data),
        static_cast<float*>(beta_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        state_slot_stride,
        scale,
    };
    recurrent_record_kernel<Masked, ForgetGate::Diagonal><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void kda_launch_replay_fold_fixed(const GdnReplayRecords& records,
                                  LinearAttentionStateAllLayersView states,
                                  const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                                  cudaStream_t stream) {
    const FoldAccess<Geometry, ForgetGate::Diagonal> access{
        static_cast<const __nv_bfloat16*>(records.key.data),
        static_cast<const __nv_bfloat16*>(records.value.data),
        static_cast<const float*>(records.gate.data),
        static_cast<const float*>(records.beta.data),
        static_cast<const __nv_bfloat16*>(records.conv.data),
        static_cast<GdnStateStorage*>(states.recurrent_layer0.data),
        static_cast<__nv_bfloat16*>(states.conv_layer0.data),
        states.recurrent_layer_stride_bytes / static_cast<std::int64_t>(sizeof(GdnStateStorage)),
        states.conv_layer_stride_bytes / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        records.spec.record_capacity,
        records.spec.width,
        rows,
    };
    const dim3 grid(static_cast<unsigned>(Geometry::kValueHeads),
                    static_cast<unsigned>(active_rows),
                    static_cast<unsigned>(Geometry::kLayers * (kStateDim / kBlockDv)));
    const dim3 block(kWarpSize, kNumWarps, 1);
    recurrent_fold_kernel<Geometry, ForgetGate::Diagonal><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
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

/// The replay-record form: the delta net's record kernel with the diagonal gate, so a lane
/// records the four channels of g it owns beside the key and value it read.
void kda_launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& g, const Tensor& beta, float scale,
                                 const Tensor& ssm_states, const Tensor& valid_columns,
                                 const Tensor& initial_state_slots, Tensor& key_record,
                                 Tensor& value_record, Tensor& gate_record, Tensor& beta_record,
                                 Tensor& out, cudaStream_t stream) {
    if (valid_columns.data == nullptr) {
        kda_launch_recurrent_record_fixed<false>(q, k, v, g, beta, scale, ssm_states,
                                                 valid_columns, initial_state_slots, key_record,
                                                 value_record, gate_record, beta_record, out,
                                                 stream);
    } else {
        kda_launch_recurrent_record_fixed<true>(q, k, v, g, beta, scale, ssm_states,
                                                valid_columns, initial_state_slots, key_record,
                                                value_record, gate_record, beta_record, out,
                                                stream);
    }
}

/// The fold over every registered diagonal-gate geometry. One today: GLM-5.3-Flash's.
void kda_launch_replay_fold(const GdnReplayRecords& records,
                            LinearAttentionStateAllLayersView states,
                            const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                            cudaStream_t stream) {
    if (records.spec.layers == FoldGeometry34x64::kLayers &&
        records.spec.qk_heads == FoldGeometry34x64::kQkHeads &&
        records.spec.value_heads == FoldGeometry34x64::kValueHeads &&
        records.spec.conv_channels == FoldGeometry34x64::kConvChannels) {
        kda_launch_replay_fold_fixed<FoldGeometry34x64>(records, states, rows, active_rows,
                                                        stream);
        return;
    }
    throw std::invalid_argument(
        "Kimi Delta Attention replay fold launcher received an unregistered geometry");
}

} // namespace sinfer::ops::detail::gated_delta_net

namespace sinfer::ops::detail::kimi_delta_net {

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& beta_record, Tensor& out, cudaStream_t stream) {
    ::sinfer::ops::detail::gated_delta_net::kda_launch_recurrent_record(
        q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, key_record,
        value_record, gate_record, beta_record, out, stream);
}

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const gated_delta_net::GdnReplayFoldKernelRows& rows,
                        std::int32_t active_rows, cudaStream_t stream) {
    ::sinfer::ops::detail::gated_delta_net::kda_launch_replay_fold(records, states, rows,
                                                                   active_rows, stream);
}

void launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk,
                      const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                      cudaStream_t stream) {
    ::sinfer::ops::detail::gated_delta_net::kda_launch_recurrent(
        q, k, v, g, beta, scale, normalize_qk, ssm_state_in, ssm_state_out, out, stream);
}

void launch_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                               const Tensor& beta, float scale, bool normalize_qk,
                               Tensor& ssm_states, const Tensor& valid_columns,
                               const Tensor& initial_state_slots,
                               const Tensor& snapshot_base_slots, Tensor& out,
                               cudaStream_t stream) {
    ::sinfer::ops::detail::gated_delta_net::kda_launch_recurrent_snapshot(
        q, k, v, g, beta, scale, normalize_qk, ssm_states, valid_columns, initial_state_slots,
        snapshot_base_slots, out, stream);
}

} // namespace sinfer::ops::detail::kimi_delta_net
