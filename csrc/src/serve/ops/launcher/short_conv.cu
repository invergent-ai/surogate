#include "ops/launcher/short_conv.h"

#include "core/device.h"

#include <cuda_bf16.h>

namespace sinfer::ops::detail {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxTaps = 4;

// Every tensor here is channel-fastest, which is this engine's layout for an activation: a
// column's channels are contiguous and columns follow one another. So [channels, T] indexes as
// `t * channels + c`, [3*channels, T] as `t * 3 * channels + r`, and a state pool
// [channels, K-1, slots] as `slot * channels * (K-1) + i * channels + c`. The taps are
// [channels, K], one tap plane after another, exactly as the linear-attention convolution
// stores its own.

/// One thread per output element, walking (column, channel) in a grid stride.
///
/// The convolution is a stencil rather than a recurrence -- every output depends on the taps
/// and the inputs alone -- so nothing here is sequential in t. Each output reads its K inputs
/// directly: at width three or four that is a handful of loads against a row the projection
/// just wrote, and it keeps the kernel one pass with no shared-memory staging to get wrong.
__global__ void __launch_bounds__(kThreads)
short_conv_kernel(const __nv_bfloat16* __restrict__ bcx, const __nv_bfloat16* __restrict__ taps,
                  const __nv_bfloat16* __restrict__ state_in, __nv_bfloat16* __restrict__ out,
                  int channels, int columns, int width) {
    const long long total = static_cast<long long>(channels) * columns;
    const int history     = width - 1;
    const long long parts = 3LL * channels;
    for (long long index = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         index < total; index += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int column  = static_cast<int>(index / channels);
        const int channel = static_cast<int>(index % channels);

        float accumulated = 0.0f;
        for (int tap = 0; tap < width; ++tap) {
            const int source = column - history + tap;
            float value;
            if (source >= 0) {
                const long long base = static_cast<long long>(source) * parts + channel;
                value = __bfloat162float(bcx[base]) *
                        __bfloat162float(bcx[base + 2LL * channels]);
            } else {
                // Before column zero the history is whatever the previous round left, already
                // gated: the state holds u, not B and x apart.
                value = __bfloat162float(
                    state_in[static_cast<long long>(source + history) * channels + channel]);
            }
            accumulated +=
                __bfloat162float(taps[static_cast<long long>(tap) * channels + channel]) * value;
        }
        const float gate =
            __bfloat162float(bcx[static_cast<long long>(column) * parts + channels + channel]);
        out[index] = __float2bfloat16_rn(accumulated * gate);
    }
}

/// The trailing `K-1` columns of u, for the round that follows.
///
/// One thread owns a channel's whole window rather than a slot of it. A round narrower than the
/// history keeps part of the old state, so some slots are read from the very buffer being
/// written; with one owner per channel those reads all happen into registers before any write,
/// and there is no ordering to get wrong between threads that never touch the same bytes.
__global__ void __launch_bounds__(kThreads)
short_conv_snapshot_kernel(const __nv_bfloat16* __restrict__ bcx,
                           __nv_bfloat16* __restrict__ state, const int* __restrict__ valid,
                           int channels, int columns, int width) {
    const int history     = width - 1;
    const long long parts = 3LL * channels;
    // The window ends at the last column the round actually carries, which under graph bucket
    // padding is fewer than the captured width.
    const int live = valid == nullptr ? columns : min(*valid, columns);
    for (int channel = blockIdx.x * blockDim.x + threadIdx.x; channel < channels;
         channel += gridDim.x * blockDim.x) {
        float window[kMaxTaps];
        for (int i = 0; i < history; ++i) {
            const int source = live - history + i;
            if (source >= 0) {
                const long long base = static_cast<long long>(source) * parts + channel;
                window[i] = __bfloat162float(bcx[base]) *
                            __bfloat162float(bcx[base + 2LL * channels]);
            } else {
                window[i] = __bfloat162float(
                    state[static_cast<long long>(source + history) * channels + channel]);
            }
        }
        for (int i = 0; i < history; ++i) {
            state[static_cast<long long>(i) * channels + channel] =
                __float2bfloat16_rn(window[i]);
        }
    }
}

/// Compute for B independent rows, each starting from its own slot of the state pool.
///
/// A column past the row's valid count is exact zero: it is a lane with nothing to say this
/// round, and its consumers read the zero rather than a stale product.
__global__ void __launch_bounds__(kThreads)
short_conv_rows_kernel(const __nv_bfloat16* __restrict__ bcx,
                       const __nv_bfloat16* __restrict__ taps,
                       const __nv_bfloat16* __restrict__ states,
                       const int* __restrict__ initial_slots, const int* __restrict__ valid,
                       __nv_bfloat16* __restrict__ out, int channels, int columns, int rows,
                       int width) {
    const long long per_row = static_cast<long long>(channels) * columns;
    const long long total   = per_row * rows;
    const int history       = width - 1;
    const long long parts   = 3LL * channels;
    const long long slot_stride = static_cast<long long>(channels) * history;
    for (long long index = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         index < total; index += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int row        = static_cast<int>(index / per_row);
        const long long off  = index - static_cast<long long>(row) * per_row;
        const int column     = static_cast<int>(off / channels);
        const int channel    = static_cast<int>(off % channels);

        const int live = valid == nullptr ? columns : valid[row];
        if (column >= live) {
            out[index] = __float2bfloat16_rn(0.0f);
            continue;
        }
        const __nv_bfloat16* row_base = bcx + static_cast<long long>(row) * parts * columns;
        const __nv_bfloat16* window   = states + initial_slots[row] * slot_stride + channel;

        float accumulated = 0.0f;
        for (int tap = 0; tap < width; ++tap) {
            const int source = column - history + tap;
            float value;
            if (source >= 0) {
                const long long base = static_cast<long long>(source) * parts + channel;
                value = __bfloat162float(row_base[base]) *
                        __bfloat162float(row_base[base + 2LL * channels]);
            } else {
                value = __bfloat162float(
                    window[static_cast<long long>(source + history) * channels]);
            }
            accumulated +=
                __bfloat162float(taps[static_cast<long long>(tap) * channels + channel]) * value;
        }
        const float gate = __bfloat162float(
            row_base[static_cast<long long>(column) * parts + channels + channel]);
        out[index] = __float2bfloat16_rn(accumulated * gate);
    }
}

/// One checkpoint per valid column, per row.
///
/// A thread owns a (row, channel) pair and walks that pair's columns in order, carrying the
/// running window in registers. It reads the initial window once, before it has written
/// anything, which is what makes it safe for a row whose initial slot lies inside its own
/// destination reservation -- the ordinary decode case, where a lane overwrites the very window
/// it started from.
__global__ void __launch_bounds__(kThreads)
short_conv_rows_snapshot_kernel(const __nv_bfloat16* __restrict__ bcx,
                                __nv_bfloat16* __restrict__ states,
                                const int* __restrict__ initial_slots,
                                const int* __restrict__ base_slots, const int* __restrict__ valid,
                                int channels, int columns, int rows, int width) {
    const int history           = width - 1;
    const long long parts       = 3LL * channels;
    const long long slot_stride = static_cast<long long>(channels) * history;
    const long long total       = static_cast<long long>(channels) * rows;
    for (long long index = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         index < total; index += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int row     = static_cast<int>(index / channels);
        const int channel = static_cast<int>(index % channels);
        const int live    = valid == nullptr ? columns : valid[row];

        const __nv_bfloat16* row_base = bcx + static_cast<long long>(row) * parts * columns;

        float window[kMaxTaps];
        for (int i = 0; i < history; ++i) {
            window[i] = __bfloat162float(
                states[initial_slots[row] * slot_stride +
                       static_cast<long long>(i) * channels + channel]);
        }
        for (int column = 0; column < live; ++column) {
            // Slide, then append this column's gated input: the window that follows column j.
            for (int i = 0; i + 1 < history; ++i) { window[i] = window[i + 1]; }
            const long long base = static_cast<long long>(column) * parts + channel;
            window[history - 1]  = __bfloat162float(row_base[base]) *
                                  __bfloat162float(row_base[base + 2LL * channels]);
            const long long destination =
                (static_cast<long long>(base_slots[row]) + column) * slot_stride;
            for (int i = 0; i < history; ++i) {
                states[destination + static_cast<long long>(i) * channels + channel] =
                    __float2bfloat16_rn(window[i]);
            }
        }
    }
}

} // namespace

void short_conv_launch(const Tensor& bcx, const Tensor& taps, Tensor& state, Tensor& out,
                       std::int32_t channels, const Tensor& valid_columns, cudaStream_t stream) {
    const int columns = bcx.ne[1];
    const int width   = taps.ne[1];
    if (columns <= 0 || channels <= 0) { return; }

    const long long elements = static_cast<long long>(channels) * columns;
    const unsigned blocks    = static_cast<unsigned>((elements + kThreads - 1) / kThreads);
    short_conv_kernel<<<blocks == 0 ? 1U : blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bcx.data),
        static_cast<const __nv_bfloat16*>(taps.data),
        static_cast<const __nv_bfloat16*>(state.data),
        static_cast<__nv_bfloat16*>(out.data), channels, columns, width);
    CUDA_CHECK(cudaGetLastError());

    if (width > 1) {
        const unsigned snapshot_blocks =
            static_cast<unsigned>((channels + kThreads - 1) / kThreads);
        short_conv_snapshot_kernel<<<snapshot_blocks == 0 ? 1U : snapshot_blocks, kThreads, 0,
                                     stream>>>(
            static_cast<const __nv_bfloat16*>(bcx.data),
            static_cast<__nv_bfloat16*>(state.data),
            valid_columns.data == nullptr ? nullptr : static_cast<const int*>(valid_columns.data),
            channels, columns, width);
        CUDA_CHECK(cudaGetLastError());
    }
}

void short_conv_snapshot_launch(const Tensor& bcx, const Tensor& taps, Tensor& conv_states,
                                const Tensor& initial_state_slots,
                                const Tensor& snapshot_base_slots, const Tensor& valid_columns,
                                Tensor& out, std::int32_t channels, cudaStream_t stream) {
    const int columns = bcx.ne[1];
    const int rows    = bcx.ne[2];
    const int width   = taps.ne[1];
    if (columns <= 0 || channels <= 0 || rows <= 0) { return; }
    const int* valid =
        valid_columns.data == nullptr ? nullptr : static_cast<const int*>(valid_columns.data);

    const long long elements = static_cast<long long>(channels) * columns * rows;
    const unsigned blocks    = static_cast<unsigned>((elements + kThreads - 1) / kThreads);
    short_conv_rows_kernel<<<blocks == 0 ? 1U : blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bcx.data),
        static_cast<const __nv_bfloat16*>(taps.data),
        static_cast<const __nv_bfloat16*>(conv_states.data),
        static_cast<const int*>(initial_state_slots.data), valid,
        static_cast<__nv_bfloat16*>(out.data), channels, columns, rows, width);
    CUDA_CHECK(cudaGetLastError());

    // Strictly after the compute pass, which reads windows this one may overwrite. Same stream,
    // so the ordering is the launch order and needs nothing else to say it.
    const long long pairs          = static_cast<long long>(channels) * rows;
    const unsigned snapshot_blocks = static_cast<unsigned>((pairs + kThreads - 1) / kThreads);
    short_conv_rows_snapshot_kernel<<<snapshot_blocks == 0 ? 1U : snapshot_blocks, kThreads, 0,
                                      stream>>>(
        static_cast<const __nv_bfloat16*>(bcx.data),
        static_cast<__nv_bfloat16*>(conv_states.data),
        static_cast<const int*>(initial_state_slots.data),
        static_cast<const int*>(snapshot_base_slots.data), valid, channels, columns, rows, width);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
