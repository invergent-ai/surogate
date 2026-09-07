// Public-contract qualification for short_conv() and short_conv_snapshot().
//
// The oracle evaluates the documented math in FP64 from the represented BF16 inputs, indexing
// the tensors the way the *contract* says they are laid out -- first dimension fastest, a
// column's channels contiguous -- rather than the way the kernel happens to walk them. An
// earlier version of this test agreed with an earlier version of the kernel about a layout
// that was neither the engine's nor the contract's, and passed; nothing else would have caught
// that until the first served token came out wrong.
//
// Two rounds matter beyond the ordinary one: a round exactly as wide as the history retains
// none of the old state, and a round narrower than it retains most -- which is decode, and the
// case where the snapshot both reads and writes the same slot.
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

/// The layout the contract states, spelled once so no reader has to re-derive it.
struct Layout {
    int channels;
    int columns;
    int width;

    [[nodiscard]] int history() const { return width - 1; }
    /// bcx [3*channels, W, (B)]: B at row c, C at row channels+c, x at row 2*channels+c.
    [[nodiscard]] std::size_t b(int channel, int column, int row = 0) const {
        return static_cast<std::size_t>(row) * 3 * channels * columns +
               static_cast<std::size_t>(column) * 3 * channels + channel;
    }
    [[nodiscard]] std::size_t gate(int channel, int column, int row = 0) const {
        return b(channel, column, row) + channels;
    }
    [[nodiscard]] std::size_t x(int channel, int column, int row = 0) const {
        return b(channel, column, row) + 2 * channels;
    }
    /// taps [channels, K]: one tap plane after another.
    [[nodiscard]] std::size_t tap(int channel, int index) const {
        return static_cast<std::size_t>(index) * channels + channel;
    }
    /// out [channels, W, (B)].
    [[nodiscard]] std::size_t out(int channel, int column, int row = 0) const {
        return (static_cast<std::size_t>(row) * columns + column) * channels + channel;
    }
    /// state [channels, K-1, (slots)].
    [[nodiscard]] std::size_t state(int channel, int index, int slot = 0) const {
        return (static_cast<std::size_t>(slot) * history() + index) * channels + channel;
    }
};

/// The whole sequence a channel convolves: its history, then the round's gated inputs.
std::vector<double> sequence(const Layout& layout, const std::vector<float>& bcx,
                             const std::vector<float>& state, int channel, int row, int slot) {
    std::vector<double> u(static_cast<std::size_t>(layout.history() + layout.columns));
    for (int i = 0; i < layout.history(); ++i) {
        u[static_cast<std::size_t>(i)] = state[layout.state(channel, i, slot)];
    }
    for (int t = 0; t < layout.columns; ++t) {
        u[static_cast<std::size_t>(layout.history() + t)] =
            static_cast<double>(bcx[layout.b(channel, t, row)]) *
            static_cast<double>(bcx[layout.x(channel, t, row)]);
    }
    return u;
}

// ---------------------------------------------------------------------------------------
// The one-sequence form.
// ---------------------------------------------------------------------------------------

struct Case {
    int channels;
    int columns;
    int width;
    std::uint32_t seed;
    const char* label;
};

int run_case(const Case& item) {
    const Layout layout{item.channels, item.columns, item.width};
    const std::size_t cells = static_cast<std::size_t>(item.channels) * item.columns;
    std::vector<float> bcx(cells * 3), taps(static_cast<std::size_t>(item.width) * item.channels),
        state(static_cast<std::size_t>(item.channels) * layout.history());
    fill_uniform(bcx, item.seed, -3.0f, 3.0f);
    fill_uniform(taps, item.seed + 1, -1.0f, 1.0f);
    fill_uniform(state, item.seed + 2, -3.0f, 3.0f);
    round_to_bf16(bcx);
    round_to_bf16(taps);
    round_to_bf16(state);

    std::vector<double> expected(cells), expected_state(state.size());
    for (int channel = 0; channel < item.channels; ++channel) {
        const std::vector<double> u = sequence(layout, bcx, state, channel, 0, 0);
        for (int t = 0; t < item.columns; ++t) {
            double sum = 0.0;
            for (int tap = 0; tap < item.width; ++tap) {
                sum += static_cast<double>(taps[layout.tap(channel, tap)]) *
                       u[static_cast<std::size_t>(t + tap)];
            }
            expected[layout.out(channel, t)] =
                sum * static_cast<double>(bcx[layout.gate(channel, t)]);
        }
        for (int i = 0; i < layout.history(); ++i) {
            expected_state[layout.state(channel, i)] =
                u[static_cast<std::size_t>(item.columns + i)];
        }
    }

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
    Tensor taps_tensor(device_taps.data(), DType::BF16, {item.channels, item.width});
    Tensor state_tensor(device_state.data(), DType::BF16, {item.channels, layout.history()});
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

// ---------------------------------------------------------------------------------------
// The slotted form: B rows that share no history.
// ---------------------------------------------------------------------------------------

struct RowCase {
    int channels;
    int columns;
    int rows;
    int width;
    int slots;
    bool mask_tail;      ///< give the rows differing valid counts
    bool overlap_initial;///< let each row's base reservation start at its own initial slot
    std::uint32_t seed;
    const char* label;
};

int run_rows(const RowCase& item) {
    const Layout layout{item.channels, item.columns, item.width};
    const std::size_t cells =
        static_cast<std::size_t>(item.channels) * item.columns * item.rows;
    std::vector<float> bcx(cells * 3), taps(static_cast<std::size_t>(item.width) * item.channels),
        states(static_cast<std::size_t>(item.channels) * layout.history() * item.slots);
    fill_uniform(bcx, item.seed, -3.0f, 3.0f);
    fill_uniform(taps, item.seed + 1, -1.0f, 1.0f);
    fill_uniform(states, item.seed + 2, -3.0f, 3.0f);
    round_to_bf16(bcx);
    round_to_bf16(taps);
    round_to_bf16(states);

    // Every row gets its own [base, base+W) reservation, disjoint from every other row's.
    // `overlap_initial` puts a row's initial slot at the head of its own reservation, which is
    // the ordinary decode case: the lane overwrites the window it just read.
    std::vector<std::int32_t> initial(item.rows), base(item.rows), valid(item.rows);
    for (int row = 0; row < item.rows; ++row) {
        base[row]    = row * item.columns;
        initial[row] = item.overlap_initial ? base[row] : item.slots - 1 - row;
        valid[row]   = item.mask_tail ? 1 + (row % item.columns) : item.columns;
    }

    std::vector<double> expected(cells);
    std::vector<double> expected_states(states.begin(), states.end());
    for (int row = 0; row < item.rows; ++row) {
        for (int channel = 0; channel < item.channels; ++channel) {
            const std::vector<double> u =
                sequence(layout, bcx, states, channel, row, initial[row]);
            for (int t = 0; t < item.columns; ++t) {
                if (t >= valid[row]) {
                    expected[layout.out(channel, t, row)] = 0.0;
                    continue;
                }
                double sum = 0.0;
                for (int tap = 0; tap < item.width; ++tap) {
                    sum += static_cast<double>(taps[layout.tap(channel, tap)]) *
                           u[static_cast<std::size_t>(t + tap)];
                }
                expected[layout.out(channel, t, row)] =
                    sum * static_cast<double>(bcx[layout.gate(channel, t, row)]);
            }
            // After valid column j the window that follows it lands in base+j.
            for (int j = 0; j < valid[row]; ++j) {
                for (int i = 0; i < layout.history(); ++i) {
                    expected_states[layout.state(channel, i, base[row] + j)] =
                        u[static_cast<std::size_t>(j + 1 + i)];
                }
            }
        }
    }

    const auto bcx_bits = encode_bf16(bcx), taps_bits = encode_bf16(taps),
               state_bits = encode_bf16(states);
    GuardedDeviceBuffer device_bcx(bcx_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_taps(taps_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_states(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(cells * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_initial(initial.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_base(base.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_valid(valid.size() * sizeof(std::int32_t));
    device_bcx.copy_from_host(bcx_bits.data(), device_bcx.bytes());
    device_taps.copy_from_host(taps_bits.data(), device_taps.bytes());
    device_states.copy_from_host(state_bits.data(), device_states.bytes());
    device_initial.copy_from_host(initial.data(), device_initial.bytes());
    device_base.copy_from_host(base.data(), device_base.bytes());
    device_valid.copy_from_host(valid.data(), device_valid.bytes());

    Tensor bcx_tensor(device_bcx.data(), DType::BF16,
                      {3 * item.channels, item.columns, item.rows});
    Tensor taps_tensor(device_taps.data(), DType::BF16, {item.channels, item.width});
    Tensor states_tensor(device_states.data(), DType::BF16,
                         {item.channels, layout.history(), item.slots});
    Tensor out_tensor(device_out.data(), DType::BF16,
                      {item.channels, item.columns, item.rows});
    Tensor initial_tensor(device_initial.data(), DType::I32, {item.rows});
    Tensor base_tensor(device_base.data(), DType::I32, {item.rows});
    Tensor valid_tensor = item.mask_tail
                              ? Tensor(device_valid.data(), DType::I32, {item.rows})
                              : Tensor{};
    ops::short_conv_snapshot(bcx_tensor, taps_tensor, states_tensor, initial_tensor, base_tensor,
                             valid_tensor, out_tensor, item.channels, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(item.label, from_device_bf16(device_out.data(), cells),
                                    expected, short_conv_criterion());
    failures += verify_pointwise((std::string(item.label) + " states").c_str(),
                                 from_device_bf16(device_states.data(), expected_states.size()),
                                 expected_states, short_conv_criterion());
    failures += verify_exact((std::string(item.label) + " input unchanged").c_str(),
                             from_device<std::uint16_t>(device_bcx.data(), bcx_bits.size()),
                             bcx_bits);
    failures += device_bcx.verify_guards("short_conv_snapshot bcx");
    failures += device_taps.verify_guards("short_conv_snapshot taps");
    failures += device_states.verify_guards("short_conv_snapshot states");
    failures += device_out.verify_guards("short_conv_snapshot out");
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

    const RowCase rows[] = {
        // The ordinary decode round: one column per lane, each lane overwriting the window it
        // just read.
        {2048, 1, 8, 3, 8, false, true, 61u, "short_conv_snapshot decode 8 lanes"},
        // Wider rounds, disjoint slots, and a checkpoint after every column.
        {512, 4, 3, 3, 16, false, false, 71u, "short_conv_snapshot 3 rows x 4 columns"},
        // Rows that stop at different columns: the tail is exact zero and changes no state.
        {512, 4, 3, 3, 16, true, false, 83u, "short_conv_snapshot ragged rows"},
        {256, 3, 2, 4, 12, true, true, 97u, "short_conv_snapshot K=4, overlapping reservations"},
        {256, 2, 2, 2, 8, false, true, 101u, "short_conv_snapshot K=2"},
    };
    for (const RowCase& item : rows) { failures += run_rows(item); }

    if (failures != 0) {
        std::cerr << failures << " short_conv check(s) failed\n";
        return 1;
    }
    std::cout << "short_conv: PASS\n";
    return 0;
}
