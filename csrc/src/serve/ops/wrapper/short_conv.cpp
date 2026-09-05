#include "api/ops/short_conv.h"

#include "ops/launcher/short_conv.h" // detail::short_conv_launch

#include <stdexcept>
#include <string>

namespace sinfer::ops {

void short_conv(const Tensor& bcx, const Tensor& taps, Tensor& state, Tensor& out,
                std::int32_t channels, cudaStream_t stream) {
    if (channels <= 0) { throw std::invalid_argument("short_conv: channels must be positive"); }
    if (bcx.dtype != DType::BF16 || bcx.data == nullptr || bcx.ne[1] <= 0 || bcx.ne[2] != 1 ||
        bcx.ne[3] != 1) {
        throw std::invalid_argument("short_conv: bcx must be BF16 [3*channels, T]");
    }
    if (bcx.ne[0] != 3 * channels) {
        throw std::invalid_argument("short_conv: bcx has " + std::to_string(bcx.ne[0]) +
                                    " rows, three parts of " + std::to_string(channels) +
                                    " would be " + std::to_string(3 * channels));
    }
    const std::int32_t columns = bcx.ne[1];
    if (taps.dtype != DType::BF16 || taps.data == nullptr || taps.ne[1] != channels ||
        taps.ne[2] != 1) {
        throw std::invalid_argument("short_conv: taps must be BF16 [K, channels]");
    }
    const std::int32_t width = taps.ne[0];
    if (width < 2 || width > 4) {
        throw std::invalid_argument("short_conv: tap width " + std::to_string(width) +
                                    " is outside [2,4]");
    }
    if (state.dtype != DType::BF16 || state.data == nullptr || state.ne[0] != channels ||
        state.ne[1] != width - 1) {
        throw std::invalid_argument("short_conv: state must be BF16 [channels, K-1]");
    }
    if (out.dtype != DType::BF16 || out.data == nullptr || out.ne[0] != channels ||
        out.ne[1] != columns) {
        throw std::invalid_argument("short_conv: out must be BF16 [channels, T]");
    }
    detail::short_conv_launch(bcx, taps, state, out, channels, stream);
}

} // namespace sinfer::ops
