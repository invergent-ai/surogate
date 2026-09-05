// Public-contract qualification for short_conv().
//
// The oracle evaluates the documented math in FP64 from the represented BF16
// inputs. It never reproduces the kernel's loop structure: it builds the full
// input sequence -- the state's columns followed by B*x -- and convolves that, so
// a kernel that mixed up tap order, dropped the gate, or read the wrong end of the
// history disagrees with it rather than with a copy of itself.
//
// Two cases matter beyond the ordinary one. A round exactly as wide as the history
// leaves nothing of the old state, and a round narrower than the history keeps part
// of it -- which is the decode step, and the case where the snapshot both reads and
// writes the state buffer.
#include "api/ops/short_conv.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr PointwiseCriterion short_conv_criterion() {
    return {/*absolute*/ 4.0e-5, /*relative*/ 6.0e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

struct Case {
    std::int32_t channels;
    std::int32_t columns;
    std::int32_t width;
    std::uint32_t seed;
    const char* label;
};

/// out[c,t] = C[c,t] * sum_j taps[j,c] * u[c, t-(K-1)+j], u the state then B*x.
std::vector<double> oracle(const Case& item, const std::vector<float>& bcx,
                           const std::vector<float>& taps, const std::vector<float>& state) {
    const int history = item.width - 1;
    std::vector<double> expected(static_cast<std::size_t>(item.channels) * item.columns);
    for (int channel = 0; channel < item.channels; ++channel) {
        const std::size_t b_base = static_cast<std::size_t>(channel) * item.columns;
        const std::size_t c_base = b_base + static_cast<std::size_t>(item.channels) * item.columns;
        const std::size_t x_base = c_base + static_cast<std::size_t>(item.channels) * item.columns;
        // The whole sequence this channel convolves: history first, then the round.
        std::vector<double> u(static_cast<std::size_t>(history + item.columns));
        for (int i = 0; i < history; ++i) {
            u[static_cast<std::size_t>(i)] =
                state[static_cast<std::size_t>(channel) * history + i];
        }
        for (int t = 0; t < item.columns; ++t) {
            u[static_cast<std::size_t>(history + t)] =
                static_cast<double>(bcx[b_base + t]) * static_cast<double>(bcx[x_base + t]);
        }
        for (int t = 0; t < item.columns; ++t) {
            double sum = 0.0;
            for (int tap = 0; tap < item.width; ++tap) {
                sum += static_cast<double>(
                           taps[static_cast<std::size_t>(tap) * item.channels + channel]) *
                       u[static_cast<std::size_t>(t + tap)];
            }
            expected[b_base + t] = sum * static_cast<double>(bcx[c_base + t]);
        }
    }
    return expected;
}

/// What the state must hold afterwards: the trailing history columns of u.
std::vector<double> state_oracle(const Case& item, const std::vector<float>& bcx,
                                 const std::vector<float>& state) {
    const int history = item.width - 1;
    std::vector<double> expected(static_cast<std::size_t>(item.channels) * history);
    for (int channel = 0; channel < item.channels; ++channel) {
        const std::size_t b_base = static_cast<std::size_t>(channel) * item.columns;
        const std::size_t x_base = b_base + 2ULL * item.channels * item.columns;
        for (int slot = 0; slot < history; ++slot) {
            const int source = item.columns - history + slot;
            expected[static_cast<std::size_t>(channel) * history + slot] =
                source >= 0 ? static_cast<double>(bcx[b_base + source]) *
                                  static_cast<double>(bcx[x_base + source])
                            : static_cast<double>(
                                  state[static_cast<std::size_t>(channel) * history + source +
                                        history]);
        }
    }
    return expected;
}

int run_case(const Case& item) {
    const int history       = item.width - 1;
    const std::size_t cells = static_cast<std::size_t>(item.channels) * item.columns;
    std::vector<float> bcx(cells * 3), taps(static_cast<std::size_t>(item.width) * item.channels),
        state(static_cast<std::size_t>(item.channels) * history);
    fill_uniform(bcx, item.seed, -3.0f, 3.0f);
    fill_uniform(taps, item.seed + 1, -1.0f, 1.0f);
    fill_uniform(state, item.seed + 2, -3.0f, 3.0f);
    round_to_bf16(bcx);
    round_to_bf16(taps);
    round_to_bf16(state);

    const auto expected       = oracle(item, bcx, taps, state);
    const auto expected_state = state_oracle(item, bcx, state);
    const auto bcx_bits = encode_bf16(bcx), taps_bits = encode_bf16(taps),
               state_bits = encode_bf16(state);

    GuardedDeviceBuffer device_bcx(bcx_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_taps(taps_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_state(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(cells * sizeof(std::uint16_t));
    device_bcx.copy_from_host(bcx_bits.data(), device_bcx.bytes());
    device_taps.copy_from_host(taps_bits.data(), device_taps.bytes());
    device_state.copy_from_host(state_bits.data(), device_state.bytes());

    Tensor bcx_tensor(device_bcx.data(), DType::BF16, {3 * item.channels, item.columns});
    Tensor taps_tensor(device_taps.data(), DType::BF16, {item.width, item.channels});
    Tensor state_tensor(device_state.data(), DType::BF16, {item.channels, history});
    Tensor out_tensor(device_out.data(), DType::BF16, {item.channels, item.columns});
    ops::short_conv(bcx_tensor, taps_tensor, state_tensor, out_tensor, item.channels, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(item.label, from_device_bf16(device_out.data(), cells),
                                    expected, short_conv_criterion());
    failures += verify_pointwise((std::string(item.label) + " state").c_str(),
                                 from_device_bf16(device_state.data(), expected_state.size()),
                                 expected_state, short_conv_criterion());
    failures += verify_exact((std::string(item.label) + " taps unchanged").c_str(),
                             from_device<std::uint16_t>(device_taps.data(), taps_bits.size()),
                             taps_bits);
    failures += verify_exact((std::string(item.label) + " input unchanged").c_str(),
                             from_device<std::uint16_t>(device_bcx.data(), bcx_bits.size()),
                             bcx_bits);
    failures += device_bcx.verify_guards("short_conv bcx");
    failures += device_taps.verify_guards("short_conv taps");
    failures += device_state.verify_guards("short_conv state");
    failures += device_out.verify_guards("short_conv out");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    const Case cases[] = {
        {2048, 37, 3, 11u, "short_conv prefill K=3"},
        {2048, 1, 3, 23u, "short_conv decode K=3 (keeps part of the old state)"},
        {2048, 2, 3, 31u, "short_conv width exactly the history"},
        {512, 128, 4, 41u, "short_conv K=4"},
        {320, 5, 2, 53u, "short_conv K=2"},
    };
    for (const Case& item : cases) { failures += run_case(item); }
    if (failures != 0) {
        std::cerr << failures << " short_conv check(s) failed\n";
        return 1;
    }
    std::cout << "short_conv: PASS\n";
    return 0;
}
