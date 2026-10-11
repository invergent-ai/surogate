#pragma once

#include <cuda_bf16.h>

#include <climits>
#include <cmath>
#include <cstdint>

namespace sinfer::ops {

// DFlash 2's dynamic convolution. One thread writes eight channels of one block row: the eight
// share a channel group (H/G is a multiple of 8), so one pair of dynamic weights per thread.
__global__ void dflash_conv_kernel(const __nv_bfloat16* __restrict__ input,
                                   const __nv_bfloat16* __restrict__ dynamic,
                                   const __nv_bfloat16* __restrict__ base,
                                   const __nv_bfloat16* residual, __nv_bfloat16* out,
                                   int hidden, int columns, int groups, int stage, int width) {
    const int vectors = hidden / 8;
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<long long>(vectors) * columns) return;
    const int column  = static_cast<int>(index / vectors);
    const int channel = static_cast<int>(index % vectors) * 8;
    const int row     = column % width;
    const int group   = channel / (hidden / groups);
    const __nv_bfloat16* weights = dynamic + static_cast<long long>(column) * 4 * groups;
    const float current_dynamic  = __bfloat162float(weights[(stage * 2) * groups + group]);
    const float previous_dynamic = __bfloat162float(weights[(stage * 2 + 1) * groups + group]);

    const auto load = [](const __nv_bfloat16* source, float (&values)[8]) {
        const uint4 packed = *reinterpret_cast<const uint4*>(source);
        const auto* pairs  = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 pair = __bfloat1622float2(pairs[i]);
            values[2 * i]     = pair.x;
            values[2 * i + 1] = pair.y;
        }
    };
    const long long offset = static_cast<long long>(column) * hidden + channel;
    float x[8], current[8], value[8];
    load(input + offset, x);
    load(base + (stage * 2) * hidden + channel, current);
#pragma unroll
    for (int i = 0; i < 8; ++i) { value[i] = x[i] * (current[i] + current_dynamic); }
    if (row > 0) {
        float before[8], previous[8];
        load(input + offset - hidden, before);
        load(base + (stage * 2 + 1) * hidden + channel, previous);
#pragma unroll
        for (int i = 0; i < 8; ++i) { value[i] += before[i] * (previous[i] + previous_dynamic); }
    }
    if (residual != nullptr) {
        float stream[8];
        load(residual + offset, stream);
#pragma unroll
        for (int i = 0; i < 8; ++i) { value[i] += stream[i]; }
    }
    uint4 packed;
    auto* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) { pairs[i] = __floats2bfloat162_rn(value[2 * i], value[2 * i + 1]); }
    *reinterpret_cast<uint4*>(out + offset) = packed;
}

namespace dflash2 {

/// The selector's total order: a larger value first, then the lower id.
__device__ __forceinline__ bool beats(float value, int id, float other, int other_id) {
    return value > other || (value == other && id < other_id);
}

template <int MaxK>
__device__ __forceinline__ void insert(float (&values)[MaxK], int (&ids)[MaxK], float value, int id) {
    if (!beats(value, id, values[MaxK - 1], ids[MaxK - 1])) return;
    bool placed = false;
#pragma unroll
    for (int i = MaxK - 1; i > 0; --i) {
        const bool shift = !placed && beats(value, id, values[i - 1], ids[i - 1]);
        if (!placed) {
            values[i] = shift ? values[i - 1] : value;
            ids[i]    = shift ? ids[i - 1] : id;
            placed    = !shift;
        }
    }
    if (!placed) {
        values[0] = value;
        ids[0]    = id;
    }
}

template <int MaxK>
__device__ __forceinline__ void drop_head(float (&values)[MaxK], int (&ids)[MaxK]) {
#pragma unroll
    for (int i = 0; i + 1 < MaxK; ++i) {
        values[i] = values[i + 1];
        ids[i]    = ids[i + 1];
    }
    values[MaxK - 1] = -INFINITY;
    ids[MaxK - 1]    = INT_MAX;
}

/// `k` rounds of a warp-wide merge of each lane's sorted list; round r's winner is passed to
/// `emit(r, value, id)` on lane 0.
template <int MaxK, class Emit>
__device__ __forceinline__ void warp_merge(float (&values)[MaxK], int (&ids)[MaxK], int k, Emit emit) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    for (int round = 0; round < k; ++round) {
        float best = values[0];
        int best_id = ids[0], best_lane = lane;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other     = __shfl_xor_sync(0xffffffffu, best, offset);
            const int other_id    = __shfl_xor_sync(0xffffffffu, best_id, offset);
            const int other_lane  = __shfl_xor_sync(0xffffffffu, best_lane, offset);
            if (beats(other, other_id, best, best_id) ||
                (other == best && other_id == best_id && other_lane < best_lane)) {
                best = other;
                best_id = other_id;
                best_lane = other_lane;
            }
        }
        if (lane == 0) emit(round, best, best_id);
        if (lane == best_lane) drop_head(values, ids);
    }
}

} // namespace dflash2

template <int MaxK>
__global__ void __launch_bounds__(256)
    dflash_top_k_kernel(const __nv_bfloat16* __restrict__ logits, int rows, int domain, int k,
                        int* __restrict__ ids_out, float* __restrict__ values_out) {
    constexpr int kWarps = 8;
    __shared__ float warp_values[kWarps][MaxK];
    __shared__ int warp_ids[kWarps][MaxK];
    const int column = static_cast<int>(blockIdx.x);
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const __nv_bfloat16* source = logits + static_cast<long long>(column) * rows;

    float values[MaxK];
    int ids[MaxK];
#pragma unroll
    for (int i = 0; i < MaxK; ++i) {
        values[i] = -INFINITY;
        ids[i]    = INT_MAX;
    }
    const auto offer = [&](float value, int row) {
        dflash2::insert(values, ids, value != value ? -INFINITY : value, row);
    };
    // Eight rows a load when a column starts on 16 bytes; the tail and odd heights go one by one.
    const int vector_end = (rows % 8 == 0) ? (domain / 8) * 8 : 0;
    for (int row = static_cast<int>(threadIdx.x) * 8; row < vector_end; row += 256 * 8) {
        const uint4 packed = *reinterpret_cast<const uint4*>(source + row);
        const auto* pairs  = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 pair = __bfloat1622float2(pairs[i]);
            offer(pair.x, row + 2 * i);
            offer(pair.y, row + 2 * i + 1);
        }
    }
    for (int row = vector_end + static_cast<int>(threadIdx.x); row < domain; row += 256) {
        offer(__bfloat162float(source[row]), row);
    }

    dflash2::warp_merge(values, ids, k, [&](int round, float value, int id) {
        warp_values[warp][round] = value;
        warp_ids[warp][round]    = id;
    });
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int i = 0; i < MaxK; ++i) {
        const bool mine = lane < kWarps && i < k;
        values[i] = mine ? warp_values[lane < kWarps ? lane : 0][i] : -INFINITY;
        ids[i]    = mine ? warp_ids[lane < kWarps ? lane : 0][i] : INT_MAX;
    }
    dflash2::warp_merge(values, ids, k, [&](int round, float value, int id) {
        const long long at = static_cast<long long>(column) * k + round;
        ids_out[at]    = id == INT_MAX ? 0 : id;
        values_out[at] = value;
    });
}

// One block per lane, one warp per candidate: a position's K edge scores in parallel, then the
// greedy choice, which the next position's edges depend on.
__global__ void dflash_walk_kernel(const std::int32_t* __restrict__ anchors,
                                   const std::int32_t* __restrict__ ids,
                                   const float* __restrict__ values,
                                   const __nv_bfloat16* __restrict__ hidden,
                                   const __nv_bfloat16* __restrict__ predecessor,
                                   const __nv_bfloat16* __restrict__ successor, int k, int positions,
                                   int dims, int vocab, std::int32_t* __restrict__ drafts) {
    __shared__ float scores[32];
    __shared__ int previous;
    const int lane_index = static_cast<int>(blockIdx.x);
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const auto clamp     = [vocab](int token) { return token < 0 ? 0 : (token >= vocab ? vocab - 1 : token); };
    if (threadIdx.x == 0) previous = anchors[lane_index];
    __syncthreads();
    for (int position = 0; position < positions; ++position) {
        const int column = lane_index * positions + position;
        if (warp < k) {
            const int before    = clamp(previous);
            const int candidate = clamp(ids[static_cast<long long>(column) * k + warp]);
            const __nv_bfloat16* left  = predecessor + static_cast<long long>(before) * dims;
            const __nv_bfloat16* mid   = hidden + static_cast<long long>(column) * dims;
            const __nv_bfloat16* right = successor + static_cast<long long>(candidate) * dims;
            float sum = 0.0F;
            for (int d = lane; d < dims; d += 32) {
                sum += __bfloat162float(left[d]) * __bfloat162float(mid[d]) * __bfloat162float(right[d]);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, offset); }
            if (lane == 0) scores[warp] = values[static_cast<long long>(column) * k + warp] + sum;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int best = 0;
            for (int j = 1; j < k; ++j) {
                if (scores[j] > scores[best]) best = j;
            }
            const int token  = ids[static_cast<long long>(column) * k + best];
            drafts[column]   = token;
            previous         = token;
        }
        __syncthreads();
    }
}

} // namespace sinfer::ops
