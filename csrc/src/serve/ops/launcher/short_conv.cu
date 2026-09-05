#include "ops/launcher/short_conv.h"

#include "core/device.h"

#include <cuda_bf16.h>

namespace sinfer::ops::detail {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxTaps = 4;

/// One thread per output element, walking (channel, column) in a grid stride.
///
/// The convolution is a stencil rather than a recurrence -- every output depends on
/// the taps and the inputs alone -- so nothing here is sequential in t. Each output
/// reads its K inputs directly: at width three or four that is a handful of loads
/// against a row the projection just wrote, and it keeps the kernel one pass with
/// no shared-memory staging to get wrong.
__global__ void __launch_bounds__(kThreads)
short_conv_kernel(const __nv_bfloat16* __restrict__ bcx, const __nv_bfloat16* __restrict__ taps,
                  const __nv_bfloat16* __restrict__ state_in, __nv_bfloat16* __restrict__ out,
                  int channels, int columns, int width) {
    const long long total = static_cast<long long>(channels) * columns;
    const int history    = width - 1;
    for (long long index = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         index < total; index += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int channel = static_cast<int>(index / columns);
        const int column  = static_cast<int>(index % columns);

        const __nv_bfloat16* b_row = bcx + static_cast<long long>(channel) * columns;
        const __nv_bfloat16* c_row = b_row + static_cast<long long>(channels) * columns;
        const __nv_bfloat16* x_row = c_row + static_cast<long long>(channels) * columns;

        float accumulated = 0.0f;
        for (int tap = 0; tap < width; ++tap) {
            const int source = column - history + tap;
            float value;
            if (source >= 0) {
                value = __bfloat162float(b_row[source]) * __bfloat162float(x_row[source]);
            } else {
                // Before column zero the history is whatever the previous round left,
                // already gated: the state holds u, not B and x apart.
                value = __bfloat162float(state_in[static_cast<long long>(channel) * history +
                                                  source + history]);
            }
            accumulated += __bfloat162float(taps[static_cast<long long>(tap) * channels + channel]) *
                           value;
        }
        out[index] = __float2bfloat16_rn(accumulated * __bfloat162float(c_row[column]));
    }
}

/// The trailing `K-1` columns of u, for the round that follows.
///
/// One thread owns a channel's whole window rather than a slot of it. A round
/// narrower than the history keeps part of the old state, so some slots are read
/// from the very buffer being written; with one owner per channel those reads all
/// happen into registers before any write, and there is no ordering to get wrong
/// between threads that never touch the same bytes.
__global__ void __launch_bounds__(kThreads)
short_conv_snapshot_kernel(const __nv_bfloat16* __restrict__ bcx,
                           __nv_bfloat16* __restrict__ state, int channels, int columns,
                           int width) {
    const int history = width - 1;
    for (int channel = blockIdx.x * blockDim.x + threadIdx.x; channel < channels;
         channel += gridDim.x * blockDim.x) {
        float window[kMaxTaps];
        const __nv_bfloat16* b_row = bcx + static_cast<long long>(channel) * columns;
        const __nv_bfloat16* x_row = b_row + 2LL * channels * columns;
        __nv_bfloat16* slot        = state + static_cast<long long>(channel) * history;
        for (int i = 0; i < history; ++i) {
            const int source = columns - history + i;
            window[i] = source >= 0
                            ? __bfloat162float(b_row[source]) * __bfloat162float(x_row[source])
                            : __bfloat162float(slot[source + history]);
        }
        for (int i = 0; i < history; ++i) { slot[i] = __float2bfloat16_rn(window[i]); }
    }
}

} // namespace

void short_conv_launch(const Tensor& bcx, const Tensor& taps, Tensor& state, Tensor& out,
                       std::int32_t channels, cudaStream_t stream) {
    const int columns = bcx.ne[1];
    const int width   = taps.ne[0];
    if (columns <= 0 || channels <= 0) { return; }

    const long long elements = static_cast<long long>(channels) * columns;
    const unsigned blocks = static_cast<unsigned>((elements + kThreads - 1) / kThreads);
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
            static_cast<__nv_bfloat16*>(state.data), channels, columns, width);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace sinfer::ops::detail
