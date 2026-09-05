#pragma once

#include "api/ops/sparse_moe.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace sinfer::ops::detail {

using ::sinfer::ops::SparseMoeGating;

struct SparseMoeRankedValue {
    float value;
    int id;
    int origin;
};

__device__ __forceinline__ bool sparse_moe_ranked_better(const SparseMoeRankedValue& a,
                                                         const SparseMoeRankedValue& b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}

__device__ __forceinline__ SparseMoeRankedValue sparse_moe_warp_best(SparseMoeRankedValue value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        SparseMoeRankedValue other;
        other.value  = __shfl_down_sync(kFullWarpMask, value.value, offset);
        other.id     = __shfl_down_sync(kFullWarpMask, value.id, offset);
        other.origin = __shfl_down_sync(kFullWarpMask, value.origin, offset);
        if (sparse_moe_ranked_better(other, value)) { value = other; }
    }
    value.value  = __shfl_sync(kFullWarpMask, value.value, 0);
    value.id     = __shfl_sync(kFullWarpMask, value.id, 0);
    value.origin = __shfl_sync(kFullWarpMask, value.origin, 0);
    return value;
}

// One warp selects the top-k of `Experts` router logits (each lane owns Experts/32 of them),
// renormalises them with a softmax, and -- where the mixture has an always-on expert -- reads
// its gate from logit `Experts`. A routed-only router has exactly `Experts` rows, so there is no
// such logit to read and `HasShared` is what says so; `shared_scale` is then untouched.
template <int Experts, int TopK, bool HasShared = true,
          SparseMoeGating Gating = SparseMoeGating::SoftmaxTopK, bool SharedGated = true>
__device__ __forceinline__ void sparse_moe_select_top_k_warp(const float* scores, int* ids,
                                                             float* alpha, float* shared_scale,
                                                             float* selected_logits,
                                                             const float* router_bias = nullptr,
                                                             float routed_scale = 1.0f) {
    static_assert(Experts % 32 == 0 && TopK >= 1 && TopK <= 32);
    constexpr int kPerLane = Experts / 32;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    SparseMoeRankedValue local[kPerLane];
#pragma unroll
    for (int item = 0; item < kPerLane; ++item) {
        const int id = lane + item * 32;
        // What the ranking sees. A softmax router ranks the logits themselves; a sigmoid one
        // ranks the score plus a learned per-expert bias, and the bias is dropped again once
        // the winners are known -- it steers which experts are chosen, not what they are worth.
        const float ranked = Gating == SparseMoeGating::SigmoidBiasTopK
                                 ? sigmoid(scores[id]) + router_bias[id]
                                 : scores[id];
        local[item]  = {ranked, id, lane};
    }
#pragma unroll
    for (int i = 1; i < kPerLane; ++i) {
        const SparseMoeRankedValue value = local[i];
        int position                     = i;
        while (position > 0 && sparse_moe_ranked_better(value, local[position - 1])) {
            local[position] = local[position - 1];
            --position;
        }
        local[position] = value;
    }

    int cursor = 0;
#pragma unroll
    for (int rank = 0; rank < TopK; ++rank) {
        SparseMoeRankedValue candidate = cursor < kPerLane
                                             ? local[cursor]
                                             : SparseMoeRankedValue{-CUDART_INF_F, 0x7fffffff, lane};
        const SparseMoeRankedValue winner = sparse_moe_warp_best(candidate);
        if (lane == 0) {
            ids[rank]             = winner.id;
            selected_logits[rank] = winner.value;
        }
        if (lane == winner.origin) { ++cursor; }
        __syncwarp();
    }

    if constexpr (Gating == SparseMoeGating::SigmoidBiasTopK) {
        // The weight is the winner's own sigmoid score, without the bias that selected it,
        // renormalised over the winners and scaled.
        float weight = 0.0f;
        if (lane < TopK) { weight = sigmoid(scores[ids[lane]]); }
        float denominator = warp_reduce_sum(weight);
        denominator       = __shfl_sync(kFullWarpMask, denominator, 0);
        if (lane < TopK) { alpha[lane] = routed_scale * weight / denominator; }
    } else {
        float exponential = 0.0f;
        if (lane < TopK) { exponential = expf(selected_logits[lane] - selected_logits[0]); }
        float denominator = warp_reduce_sum(exponential);
        denominator       = __shfl_sync(kFullWarpMask, denominator, 0);
        if (lane < TopK) { alpha[lane] = exponential / denominator; }
    }
    if constexpr (HasShared) {
        // An ungated shared expert is added with weight one, and its router has no row to read:
        // `scores[Experts]` would be one past the end of a table sized to the experts alone.
        if (lane == 0) { *shared_scale = SharedGated ? sigmoid(scores[Experts]) : 1.0f; }
    }
}

} // namespace sinfer::ops::detail
