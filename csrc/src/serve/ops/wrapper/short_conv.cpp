#include "api/ops/short_conv.h"

#include "ops/launcher/short_conv.h" // detail::short_conv_launch

#include <stdexcept>
#include <string>
#include <utility>

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
    if (taps.dtype != DType::BF16 || taps.data == nullptr || taps.ne[0] != channels ||
        taps.ne[2] != 1) {
        throw std::invalid_argument("short_conv: taps must be BF16 [channels, K]");
    }
    const std::int32_t width = taps.ne[1];
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

void short_conv_snapshot(const Tensor& bcx, const Tensor& taps, Tensor& conv_states,
                         const Tensor& initial_state_slots, const Tensor& snapshot_base_slots,
                         const Tensor& valid_columns, Tensor& out, std::int32_t channels,
                         cudaStream_t stream) {
    if (channels <= 0) {
        throw std::invalid_argument("short_conv_snapshot: channels must be positive");
    }
    if (bcx.dtype != DType::BF16 || bcx.data == nullptr || bcx.ne[0] != 3 * channels ||
        bcx.ne[1] <= 0 || bcx.ne[2] <= 0 || bcx.ne[3] != 1 || !bcx.is_contiguous()) {
        throw std::invalid_argument("short_conv_snapshot: bcx must be BF16 [3*channels, W, B]");
    }
    const std::int32_t columns = bcx.ne[1];
    const std::int32_t rows    = bcx.ne[2];
    if (taps.dtype != DType::BF16 || taps.data == nullptr || taps.ne[0] != channels ||
        taps.ne[2] != 1) {
        throw std::invalid_argument("short_conv_snapshot: taps must be BF16 [channels, K]");
    }
    const std::int32_t width = taps.ne[1];
    if (width < 2 || width > 4) {
        throw std::invalid_argument("short_conv_snapshot: tap width " + std::to_string(width) +
                                    " is outside [2,4]");
    }
    if (conv_states.dtype != DType::BF16 || conv_states.data == nullptr ||
        conv_states.ne[0] != channels || conv_states.ne[1] != width - 1 ||
        conv_states.ne[2] <= 0 || !conv_states.is_contiguous()) {
        throw std::invalid_argument(
            "short_conv_snapshot: conv_states must be BF16 [channels, K-1, slots]");
    }
    for (const auto& [slots, label] :
         {std::pair{&initial_state_slots, "initial_state_slots"},
          std::pair{&snapshot_base_slots, "snapshot_base_slots"}}) {
        if (slots->dtype != DType::I32 || slots->data == nullptr || slots->ne[0] != rows ||
            slots->ne[1] != 1 || !slots->is_contiguous()) {
            throw std::invalid_argument(std::string("short_conv_snapshot: ") + label +
                                        " must be contiguous I32 [B]");
        }
    }
    if (valid_columns.data != nullptr &&
        (valid_columns.dtype != DType::I32 || valid_columns.ne[0] != rows ||
         valid_columns.ne[1] != 1 || !valid_columns.is_contiguous())) {
        throw std::invalid_argument(
            "short_conv_snapshot: valid_columns must be empty or contiguous I32 [B]");
    }
    if (out.dtype != DType::BF16 || out.data == nullptr || out.ne[0] != channels ||
        out.ne[1] != columns || out.ne[2] != rows || !out.is_contiguous()) {
        throw std::invalid_argument("short_conv_snapshot: out must be BF16 [channels, W, B]");
    }
    detail::short_conv_snapshot_launch(bcx, taps, conv_states, initial_state_slots,
                                       snapshot_base_slots, valid_columns, out, channels, stream);
}

} // namespace sinfer::ops
