#include "ninfer/ops/causal_conv1d_silu.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// CausalConv1dSiLU has one A16 convolution-reduction profile. Decode, small-T, sequence, prefill,
// snapshot, and state-alias choices all qualify against this single normwise criterion.
constexpr ReductionCriterion kCausalConvA16Criterion{
    /*relative_l2*/ 1.85e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 3.7e-3,
};

constexpr std::uint8_t kOutputPoison = 0xff;

std::size_t offset(std::int32_t c, std::int32_t column, std::int32_t C) {
    return static_cast<std::size_t>(column) * static_cast<std::size_t>(C) +
           static_cast<std::size_t>(c);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { result[i] = f32_to_bf16(values[i]); }
    return result;
}

std::vector<float> make_values(std::size_t count, std::uint32_t seed, float low, float high) {
    std::vector<float> values(count);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return values;
}

struct LogicalInput {
    std::vector<float> x;
    std::vector<float> weight;
};

LogicalInput make_input(std::int32_t C, std::int32_t T, std::uint32_t seed) {
    LogicalInput input{
        make_values(static_cast<std::size_t>(C) * static_cast<std::size_t>(T), seed, -3.0F, 3.0F),
        make_values(static_cast<std::size_t>(C) * 4U, seed + 1U, -1.0F, 1.0F),
    };

    // Deterministic cancellation and sign cases remain logical inputs to the same oracle. They
    // catch tap-order mistakes without introducing a route-specific reference.
    const std::array<std::int32_t, 3> channels{0, C / 2, C - 1};
    for (const std::int32_t c : channels) {
        input.weight[offset(c, 0, C)] = 1.0F;
        input.weight[offset(c, 1, C)] = -1.0F;
        input.weight[offset(c, 2, C)] = 1.0F;
        input.weight[offset(c, 3, C)] = -1.0F;
        for (std::int32_t t = 0; t < T; ++t) {
            input.x[offset(c, t, C)] = (t & 1) == 0 ? 0.75F : -0.75F;
        }
    }
    return input;
}

std::vector<float> make_state(std::int32_t C, std::uint32_t seed) {
    std::vector<float> state = make_values(static_cast<std::size_t>(C) * 3U, seed, -3.0F, 3.0F);
    for (const std::int32_t c : {0, C / 2, C - 1}) {
        state[offset(c, 0, C)] = 0.75F;
        state[offset(c, 1, C)] = 0.75F;
        state[offset(c, 2, C)] = 0.75F;
    }
    return state;
}

struct OracleResult {
    std::vector<double> output;
    std::vector<float> final_state;
    std::vector<float> snapshots;
};

// The one Op oracle: evaluate the complete convolution and SiLU in FP64 over the logical BF16
// inputs, and separately apply the specified width-3 state transition. It has no production
// staging, accumulator, route, or output-rounding behavior.
OracleResult causal_conv_oracle(const std::vector<float>& x, const std::vector<float>& weight,
                                const std::vector<float>& initial_state, std::int32_t C,
                                std::int32_t T, bool record_snapshots) {
    OracleResult result;
    result.output.resize(static_cast<std::size_t>(C) * static_cast<std::size_t>(T));
    result.final_state = initial_state;
    if (record_snapshots) {
        result.snapshots.resize(static_cast<std::size_t>(C) * 3U * static_cast<std::size_t>(T));
    }

    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t c = 0; c < C; ++c) {
            const double x0 = static_cast<double>(result.final_state[offset(c, 0, C)]);
            const double x1 = static_cast<double>(result.final_state[offset(c, 1, C)]);
            const double x2 = static_cast<double>(result.final_state[offset(c, 2, C)]);
            const double x3 = static_cast<double>(x[offset(c, t, C)]);
            double sum      = 0.0;
            sum += static_cast<double>(weight[offset(c, 0, C)]) * x0;
            sum += static_cast<double>(weight[offset(c, 1, C)]) * x1;
            sum += static_cast<double>(weight[offset(c, 2, C)]) * x2;
            sum += static_cast<double>(weight[offset(c, 3, C)]) * x3;
            result.output[offset(c, t, C)] = sum / (1.0 + std::exp(-sum));
        }

        for (std::int32_t c = 0; c < C; ++c) {
            result.final_state[offset(c, 0, C)] = result.final_state[offset(c, 1, C)];
            result.final_state[offset(c, 1, C)] = result.final_state[offset(c, 2, C)];
            result.final_state[offset(c, 2, C)] = x[offset(c, t, C)];
        }

        if (record_snapshots) {
            std::copy(result.final_state.begin(), result.final_state.end(),
                      result.snapshots.begin() +
                          static_cast<std::size_t>(t) * result.final_state.size());
        }
    }
    return result;
}

int verify_bits(const std::string& label, const void* device,
                const std::vector<std::uint16_t>& ref) {
    return verify_exact(label.c_str(), from_device<std::uint16_t>(device, ref.size()), ref);
}

int verify_buffer_guards(const std::string& label, const GuardedDeviceBuffer& buffer) {
    return buffer.verify_guards(label.c_str());
}

int verify_output(const std::string& label, const std::vector<double>& got,
                  const std::vector<double>& reference) {
    return verify_reduction(label.c_str(), got, reference, kCausalConvA16Criterion);
}

enum class StateCall {
    InPlaceEntry,
    DistinctEntry,
    DistinctEntryExactAlias,
};

const char* call_name(StateCall call) {
    switch (call) {
    case StateCall::InPlaceEntry:
        return "in-place-entry";
    case StateCall::DistinctEntry:
        return "distinct-entry";
    case StateCall::DistinctEntryExactAlias:
        return "distinct-entry-exact-alias";
    }
    return "unknown";
}

int ordinary_case(std::int32_t C, std::int32_t T, StateCall call, std::uint32_t seed) {
    const LogicalInput input       = make_input(C, T, seed);
    const std::vector<float> state = make_state(C, seed + 2U);
    const OracleResult oracle      = causal_conv_oracle(input.x, input.weight, state, C, T, false);
    const std::vector<std::uint16_t> x_bits      = bf16_bits(input.x);
    const std::vector<std::uint16_t> weight_bits = bf16_bits(input.weight);
    const std::vector<std::uint16_t> state_bits  = bf16_bits(state);
    const std::vector<std::uint16_t> final_bits  = bf16_bits(oracle.final_state);

    GuardedDeviceBuffer x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state_in(state_bits.size() * sizeof(std::uint16_t));
    std::unique_ptr<GuardedDeviceBuffer> state_out;
    if (call == StateCall::DistinctEntry) {
        state_out =
            std::make_unique<GuardedDeviceBuffer>(state_bits.size() * sizeof(std::uint16_t));
        state_out->fill(0x5a);
    }
    GuardedDeviceBuffer output(x_bits.size() * sizeof(std::uint16_t));
    x.copy_from_host(x_bits.data(), x.bytes());
    weight.copy_from_host(weight_bits.data(), weight.bytes());
    state_in.copy_from_host(state_bits.data(), state_in.bytes());
    output.fill(kOutputPoison);

    Tensor tx(x.data(), DType::BF16, {C, T});
    Tensor tw(weight.data(), DType::BF16, {C, 4});
    Tensor ts_in(state_in.data(), DType::BF16, {C, 3});
    Tensor ts_out(state_out == nullptr ? state_in.data() : state_out->data(), DType::BF16, {C, 3});
    Tensor tout(output.data(), DType::BF16, {C, T});

    if (call == StateCall::InPlaceEntry) {
        ops::causal_conv1d_silu(tx, tw, ts_in, tout, nullptr);
    } else if (call == StateCall::DistinctEntry) {
        ops::causal_conv1d_silu(tx, tw, ts_in, ts_out, tout, nullptr);
    } else {
        ops::causal_conv1d_silu(tx, tw, ts_in, ts_in, tout, nullptr);
    }
    cuda_synchronize();

    const std::string tag = "causal_conv1d_silu C=" + std::to_string(C) +
                            " T=" + std::to_string(T) + " " + call_name(call);
    int failures = 0;
    failures += verify_output(tag + " output", from_device_bf16(output.data(), x_bits.size()),
                              oracle.output);
    failures += verify_bits(tag + " final state",
                            state_out == nullptr ? state_in.data() : state_out->data(), final_bits);
    failures += verify_bits(tag + " x preserved", x.data(), x_bits);
    failures += verify_bits(tag + " weight preserved", weight.data(), weight_bits);
    if (call == StateCall::DistinctEntry) {
        failures += verify_bits(tag + " initial state preserved", state_in.data(), state_bits);
    }
    failures += verify_buffer_guards(tag + " x", x);
    failures += verify_buffer_guards(tag + " weight", weight);
    failures += verify_buffer_guards(tag + " state input", state_in);
    if (state_out != nullptr) {
        failures += verify_buffer_guards(tag + " state output", *state_out);
    }
    failures += verify_buffer_guards(tag + " output", output);
    return failures;
}

// Continuation prefill reads a selected committed slot and publishes the new running state to slot
// 0. The Op sees two valid disjoint [C,3] tensors; checking their common backing allocation also
// proves that every surrounding state slot remains untouched.
int continuation_slot_case(std::int32_t C, std::int32_t T, std::int32_t slots,
                           std::int32_t read_slot, std::uint32_t seed) {
    const LogicalInput input        = make_input(C, T, seed);
    const std::size_t slot_elements = static_cast<std::size_t>(C) * 3U;
    std::vector<float> states(slot_elements * static_cast<std::size_t>(slots));
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        std::vector<float> one = make_state(C, seed + 10U + static_cast<std::uint32_t>(slot));
        std::copy(one.begin(), one.end(),
                  states.begin() + static_cast<std::size_t>(slot) * slot_elements);
    }

    const auto selected_begin =
        states.begin() + static_cast<std::size_t>(read_slot) * slot_elements;
    const std::vector<float> selected_state(selected_begin, selected_begin + slot_elements);
    const OracleResult oracle =
        causal_conv_oracle(input.x, input.weight, selected_state, C, T, false);
    std::vector<float> expected_states = states;
    std::copy(oracle.final_state.begin(), oracle.final_state.end(), expected_states.begin());

    const std::vector<std::uint16_t> x_bits        = bf16_bits(input.x);
    const std::vector<std::uint16_t> weight_bits   = bf16_bits(input.weight);
    const std::vector<std::uint16_t> state_bits    = bf16_bits(states);
    const std::vector<std::uint16_t> expected_bits = bf16_bits(expected_states);

    GuardedDeviceBuffer x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer output(x_bits.size() * sizeof(std::uint16_t));
    x.copy_from_host(x_bits.data(), x.bytes());
    weight.copy_from_host(weight_bits.data(), weight.bytes());
    state.copy_from_host(state_bits.data(), state.bytes());
    output.fill(kOutputPoison);

    auto* selected_device =
        static_cast<std::uint8_t*>(state.data()) +
        static_cast<std::size_t>(read_slot) * slot_elements * sizeof(std::uint16_t);
    Tensor tx(x.data(), DType::BF16, {C, T});
    Tensor tw(weight.data(), DType::BF16, {C, 4});
    Tensor ts_in(selected_device, DType::BF16, {C, 3});
    Tensor ts_out(state.data(), DType::BF16, {C, 3});
    Tensor tout(output.data(), DType::BF16, {C, T});
    ops::causal_conv1d_silu(tx, tw, ts_in, ts_out, tout, nullptr);
    cuda_synchronize();

    const std::string tag = "causal_conv1d_silu continuation C=" + std::to_string(C) +
                            " T=" + std::to_string(T) + " read_slot=" + std::to_string(read_slot);
    int failures = 0;
    failures += verify_output(tag + " output", from_device_bf16(output.data(), x_bits.size()),
                              oracle.output);
    failures += verify_bits(tag + " all state slots", state.data(), expected_bits);
    failures += verify_bits(tag + " x preserved", x.data(), x_bits);
    failures += verify_bits(tag + " weight preserved", weight.data(), weight_bits);
    failures += verify_buffer_guards(tag + " x", x);
    failures += verify_buffer_guards(tag + " weight", weight);
    failures += verify_buffer_guards(tag + " states", state);
    failures += verify_buffer_guards(tag + " output", output);
    return failures;
}

int snapshot_case(std::int32_t C, std::int32_t T, std::int32_t slots, std::int32_t initial_slot,
                  std::int32_t snapshot_base_slot, std::uint32_t seed) {
    const LogicalInput input        = make_input(C, T, seed);
    const std::size_t slot_elements = static_cast<std::size_t>(C) * 3U;
    std::vector<float> states(slot_elements * static_cast<std::size_t>(slots));
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        std::vector<float> one = make_state(C, seed + 10U + static_cast<std::uint32_t>(slot));
        std::copy(one.begin(), one.end(),
                  states.begin() + static_cast<std::size_t>(slot) * slot_elements);
    }

    const auto selected_begin =
        states.begin() + static_cast<std::size_t>(initial_slot) * slot_elements;
    const std::vector<float> selected_state(selected_begin, selected_begin + slot_elements);
    const OracleResult oracle =
        causal_conv_oracle(input.x, input.weight, selected_state, C, T, true);

    std::vector<float> expected_states = states;
    std::copy(oracle.snapshots.begin(), oracle.snapshots.end(),
              expected_states.begin() +
                  static_cast<std::size_t>(snapshot_base_slot) * slot_elements);

    const std::vector<std::uint16_t> x_bits        = bf16_bits(input.x);
    const std::vector<std::uint16_t> weight_bits   = bf16_bits(input.weight);
    const std::vector<std::uint16_t> state_bits    = bf16_bits(states);
    const std::vector<std::uint16_t> expected_bits = bf16_bits(expected_states);
    const std::int32_t initial_slot_host           = initial_slot;
    const std::int32_t snapshot_base_slot_host     = snapshot_base_slot;

    GuardedDeviceBuffer x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer slot(sizeof(std::int32_t));
    GuardedDeviceBuffer snapshot_base(sizeof(std::int32_t));
    GuardedDeviceBuffer output(x_bits.size() * sizeof(std::uint16_t));
    x.copy_from_host(x_bits.data(), x.bytes());
    weight.copy_from_host(weight_bits.data(), weight.bytes());
    state.copy_from_host(state_bits.data(), state.bytes());
    slot.copy_from_host(&initial_slot_host, sizeof(initial_slot_host));
    snapshot_base.copy_from_host(&snapshot_base_slot_host, sizeof(snapshot_base_slot_host));
    output.fill(kOutputPoison);

    Tensor tx(x.data(), DType::BF16, {C, T});
    Tensor tw(weight.data(), DType::BF16, {C, 4});
    Tensor tstate(state.data(), DType::BF16, {C, 3, slots});
    Tensor tslot(slot.data(), DType::I32, {1});
    Tensor tsnapshot_base(snapshot_base.data(), DType::I32, {1});
    Tensor tout(output.data(), DType::BF16, {C, T});
    ops::causal_conv1d_silu_snapshot(tx, tw, tstate, Tensor{}, tslot, tsnapshot_base, tout,
                                     nullptr);
    cuda_synchronize();

    const std::string tag = "causal_conv1d_silu snapshot C=" + std::to_string(C) +
                            " T=" + std::to_string(T) + " slots=" + std::to_string(slots) +
                            " initial_slot=" + std::to_string(initial_slot) +
                            " snapshot_base_slot=" + std::to_string(snapshot_base_slot);
    int failures = 0;
    failures += verify_output(tag + " output", from_device_bf16(output.data(), x_bits.size()),
                              oracle.output);
    failures += verify_bits(tag + " all state slots", state.data(), expected_bits);
    failures += verify_bits(tag + " x preserved", x.data(), x_bits);
    failures += verify_bits(tag + " weight preserved", weight.data(), weight_bits);
    failures += verify_exact((tag + " initial slot preserved").c_str(),
                             from_device<std::int32_t>(slot.data(), 1),
                             std::vector<std::int32_t>{initial_slot});
    failures += verify_exact((tag + " snapshot base slot preserved").c_str(),
                             from_device<std::int32_t>(snapshot_base.data(), 1),
                             std::vector<std::int32_t>{snapshot_base_slot});
    failures += verify_buffer_guards(tag + " x", x);
    failures += verify_buffer_guards(tag + " weight", weight);
    failures += verify_buffer_guards(tag + " states", state);
    failures += verify_buffer_guards(tag + " initial slot", slot);
    failures += verify_buffer_guards(tag + " snapshot base slot", snapshot_base);
    failures += verify_buffer_guards(tag + " output", output);
    return failures;
}

int batched_snapshot_case(std::int32_t C, std::int32_t width,
                          const std::vector<std::int32_t>& initial_slots,
                          const std::vector<std::int32_t>& snapshot_bases,
                          const std::vector<std::int32_t>& valid_columns, std::int32_t slots,
                          std::uint32_t seed) {
    const std::int32_t batch        = static_cast<std::int32_t>(initial_slots.size());
    const bool masked               = !valid_columns.empty();
    const LogicalInput input        = make_input(C, width * batch, seed);
    const std::size_t slot_elements = static_cast<std::size_t>(C) * 3U;
    const std::size_t row_elements  = static_cast<std::size_t>(C) * width;

    std::vector<float> states(slot_elements * static_cast<std::size_t>(slots));
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        std::vector<float> one = make_state(C, seed + 10U + static_cast<std::uint32_t>(slot));
        std::copy(one.begin(), one.end(),
                  states.begin() + static_cast<std::size_t>(slot) * slot_elements);
    }

    std::vector<double> expected_output(row_elements * static_cast<std::size_t>(batch), 0.0);
    std::vector<float> expected_states = states;
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid = masked ? valid_columns[static_cast<std::size_t>(row)] : width;
        const std::size_t input_begin = static_cast<std::size_t>(row) * row_elements;
        std::vector<float> row_input(input.x.begin() + input_begin,
                                     input.x.begin() + input_begin +
                                         static_cast<std::size_t>(valid) * C);
        const std::size_t initial_begin =
            static_cast<std::size_t>(initial_slots[static_cast<std::size_t>(row)]) * slot_elements;
        std::vector<float> initial_state(states.begin() + initial_begin,
                                         states.begin() + initial_begin + slot_elements);
        const OracleResult oracle =
            causal_conv_oracle(row_input, input.weight, initial_state, C, valid, true);
        std::copy(oracle.output.begin(), oracle.output.end(),
                  expected_output.begin() + input_begin);
        const std::size_t destination_begin =
            static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(row)]) * slot_elements;
        std::copy(oracle.snapshots.begin(), oracle.snapshots.end(),
                  expected_states.begin() + destination_begin);
    }

    const std::vector<std::uint16_t> x_bits        = bf16_bits(input.x);
    const std::vector<std::uint16_t> weight_bits   = bf16_bits(input.weight);
    const std::vector<std::uint16_t> state_bits    = bf16_bits(states);
    const std::vector<std::uint16_t> expected_bits = bf16_bits(expected_states);

    GuardedDeviceBuffer x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer initial(initial_slots.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer bases(snapshot_bases.size() * sizeof(std::int32_t));
    std::unique_ptr<GuardedDeviceBuffer> valid;
    if (masked) {
        valid = std::make_unique<GuardedDeviceBuffer>(valid_columns.size() * sizeof(std::int32_t));
        valid->copy_from_host(valid_columns.data(), valid->bytes());
    }
    GuardedDeviceBuffer output(x_bits.size() * sizeof(std::uint16_t));
    x.copy_from_host(x_bits.data(), x.bytes());
    weight.copy_from_host(weight_bits.data(), weight.bytes());
    state.copy_from_host(state_bits.data(), state.bytes());
    initial.copy_from_host(initial_slots.data(), initial.bytes());
    bases.copy_from_host(snapshot_bases.data(), bases.bytes());
    output.fill(kOutputPoison);

    Tensor tx(x.data(), DType::BF16, {C, width, batch});
    Tensor tw(weight.data(), DType::BF16, {C, 4});
    Tensor tstate(state.data(), DType::BF16, {C, 3, slots});
    Tensor tvalid;
    if (masked) { tvalid = Tensor(valid->data(), DType::I32, {batch}); }
    Tensor tinitial(initial.data(), DType::I32, {batch});
    Tensor tbases(bases.data(), DType::I32, {batch});
    Tensor tout(output.data(), DType::BF16, {C, width, batch});
    ops::causal_conv1d_silu_snapshot(tx, tw, tstate, tvalid, tinitial, tbases, tout, nullptr);
    cuda_synchronize();

    const std::string tag = "causal_conv1d_silu batched snapshot C=" + std::to_string(C) +
                            " W=" + std::to_string(width) + " B=" + std::to_string(batch) +
                            (masked ? " masked" : " dense");
    int failures = 0;
    failures += verify_output(tag + " output", from_device_bf16(output.data(), x_bits.size()),
                              expected_output);
    failures += verify_bits(tag + " all state slots", state.data(), expected_bits);
    failures += verify_bits(tag + " x preserved", x.data(), x_bits);
    failures += verify_bits(tag + " weight preserved", weight.data(), weight_bits);
    failures += verify_exact((tag + " initial selectors preserved").c_str(),
                             from_device<std::int32_t>(initial.data(), initial_slots.size()),
                             initial_slots);
    failures += verify_exact((tag + " snapshot bases preserved").c_str(),
                             from_device<std::int32_t>(bases.data(), snapshot_bases.size()),
                             snapshot_bases);
    if (masked) {
        failures += verify_exact((tag + " valid columns preserved").c_str(),
                                 from_device<std::int32_t>(valid->data(), valid_columns.size()),
                                 valid_columns);
        failures += verify_buffer_guards(tag + " valid columns", *valid);
    }
    failures += verify_buffer_guards(tag + " x", x);
    failures += verify_buffer_guards(tag + " weight", weight);
    failures += verify_buffer_guards(tag + " states", state);
    failures += verify_buffer_guards(tag + " initial selectors", initial);
    failures += verify_buffer_guards(tag + " snapshot bases", bases);
    failures += verify_buffer_guards(tag + " output", output);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    // The 27B geometry exercises every ordinary positive-T route boundary and one interior point
    // per route. Every case is compared directly with the same complete oracle.
    constexpr std::int32_t kQwen27Channels = 10240;
    for (const std::int32_t T : {1, 2, 7, 15, 16, 17, 32, 63, 64, 65, 257}) {
        failures += ordinary_case(kQwen27Channels, T, StateCall::InPlaceEntry,
                                  1000U + static_cast<std::uint32_t>(T));
    }

    // Qualify the second public entry directly, including its disjoint and exact-alias state forms.
    failures += ordinary_case(kQwen27Channels, 1, StateCall::DistinctEntry, 2001U);
    failures += ordinary_case(kQwen27Channels, 7, StateCall::DistinctEntry, 2007U);
    failures += ordinary_case(kQwen27Channels, 17, StateCall::DistinctEntryExactAlias, 2017U);
    failures += ordinary_case(kQwen27Channels, 32, StateCall::DistinctEntry, 2032U);
    failures += continuation_slot_case(kQwen27Channels, 65, 6, 4, 2065U);

    // The peer 35B-A3B geometry is a separate real channel extent.
    constexpr std::int32_t kQwen35Channels = 8192;
    failures += ordinary_case(kQwen35Channels, 257, StateCall::InPlaceEntry, 3257U);

    // Snapshot decode, small-T boundary/interior, sequence route, slot 0 initialization, and
    // continuation from selected slots. A nonzero destination base proves that snapshots are not
    // hard-wired to physical slots [0,T); selected source slots may overlap the destination range.
    failures += snapshot_case(kQwen27Channels, 1, 4, 0, 1, 4001U);
    failures += snapshot_case(kQwen27Channels, 2, 5, 4, 1, 4002U);
    failures += snapshot_case(kQwen27Channels, 7, 9, 3, 1, 4007U);
    failures += snapshot_case(kQwen27Channels, 15, 18, 14, 1, 4015U);
    failures += snapshot_case(kQwen27Channels, 16, 18, 17, 1, 4016U);
    failures += snapshot_case(kQwen35Channels, 17, 20, 16, 1, 4017U);

    failures += batched_snapshot_case(kQwen35Channels, 1, {8, 9, 10, 11, 12, 13, 14, 15},
                                      {0, 1, 2, 3, 4, 5, 6, 7}, {}, 16, 5001U);
    failures +=
        batched_snapshot_case(kQwen27Channels, 6, {18, 19, 20}, {0, 6, 12}, {6, 3, 1}, 21, 5006U);
    // Row 0 reads slot 15 and overwrites it only after the final valid column. This is the
    // production same-row alias pattern; row 1 remains fully disjoint.
    failures += batched_snapshot_case(kQwen35Channels, 16, {15, 33}, {0, 16}, {16, 7}, 34, 5016U);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " causal_conv1d_silu\n";
    return failures == 0 ? 0 : 1;
}
