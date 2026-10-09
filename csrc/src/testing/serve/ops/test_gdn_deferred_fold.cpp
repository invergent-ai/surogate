// A deferred fold (gdn_replay_stash, then the next replay record applying it) must leave every
// output, record and state bit exactly where folding at once (gdn_replay_fold, then the plain
// record) leaves them; and gdn_replay_fold_pending must equal the fold it stood in for.
#include "api/ops/gated_delta_net.h"
#include "api/ops/gdn_replay.h"

#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr std::int32_t kStateDim = 128;
constexpr std::int32_t kQkHeads  = 16;
constexpr std::int32_t kLayers   = 2;
constexpr std::int32_t kCapacity = 4;
constexpr std::int32_t kLanes    = 6;
constexpr float kScale           = 0.08838834764831845F; // 1/sqrt(128)

std::vector<std::uint16_t> bf16_values(std::size_t count, std::uint32_t seed, float magnitude) {
    std::vector<float> values(count);
    fill_uniform(values, seed, -magnitude, magnitude);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t i = 0; i < count; ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<std::uint32_t> gate_pairs(std::size_t pairs, std::uint32_t seed) {
    std::vector<float> g(pairs);
    std::vector<float> beta(pairs);
    fill_uniform(g, seed, -1.0F, -0.02F);
    fill_uniform(beta, seed + 1, 0.05F, 0.95F);
    std::vector<std::uint32_t> bits(2 * pairs);
    for (std::size_t i = 0; i < pairs; ++i) {
        bits[2 * i]     = std::bit_cast<std::uint32_t>(g[i]);
        bits[2 * i + 1] = std::bit_cast<std::uint32_t>(beta[i]);
    }
    return bits;
}

template <class T>
void upload(const Tensor& tensor, const std::vector<T>& host, const char* label) {
    cuda_check(cudaMemcpy(tensor.data, host.data(), tensor.bytes(), cudaMemcpyHostToDevice), label);
}

/// One engine's worth of device state: records (with deferred fold planes) and a state pool.
struct Engine {
    DeviceBuffer record_storage;
    DeviceBuffer state_storage;
    GdnReplayRecords records;
    LinearAttentionStatePool states;

    Engine(const GdnReplayRecordLayout& record_layout, std::size_t record_bytes,
           const LinearAttentionStatePoolLayout& state_layout, std::size_t state_bytes)
        : record_storage(record_bytes), state_storage(state_bytes),
          records({record_storage.p, record_bytes}, record_layout),
          states({state_storage.p, state_bytes}, state_layout) {
        record_storage.fill(0xff);
        cuda_check(cudaMemset(records.pending_columns.data, 0, records.pending_columns.bytes()),
                   "zero pending columns");
    }
};

int expect_same(const std::string& label, const void* lhs, const void* rhs, std::size_t bytes) {
    const auto a = from_device<std::uint8_t>(lhs, bytes);
    const auto b = from_device<std::uint8_t>(rhs, bytes);
    if (a == b) { return 0; }
    std::size_t first = 0;
    while (first < bytes && a[first] == b[first]) { ++first; }
    std::cerr << label << ": bytes differ from offset " << first << " of " << bytes << '\n';
    return 1;
}

int run_case(std::int32_t value_heads, std::uint32_t seed) {
    const std::int32_t width         = 4;
    const std::int32_t conv_channels = (2 * kQkHeads + value_heads) * kStateDim;
    const std::int32_t slot_count    = kLanes + 1;
    // Round one: three rows on lanes 2, 0, 5 committing 3, 1 and 4 transitions.
    const std::vector<std::int32_t> first_slots{2, 0, 5};
    const std::vector<std::int32_t> commits{3, 1, 4};
    // Round two verifies the same lanes in another row order, plus lane 4 with nothing pending.
    const std::vector<std::int32_t> second_slots{5, 4, 2, 0};
    const std::int32_t second_rows = static_cast<std::int32_t>(second_slots.size());

    const GdnReplayRecordSpec spec{
        .layers          = kLayers,
        .record_capacity = kCapacity,
        .width           = width,
        .conv_channels   = conv_channels,
        .qk_heads        = kQkHeads,
        .value_heads     = value_heads,
        .key_dim         = kStateDim,
        .value_dim       = kStateDim,
        .pending_slots   = kLanes,
    };
    LayoutBuilder record_builder;
    const GdnReplayRecordLayout record_layout = plan_gdn_replay_records(record_builder, spec);
    const std::size_t record_bytes            = record_builder.finish(256);
    LayoutBuilder state_builder;
    const LinearAttentionStatePoolLayout state_layout = plan_linear_attention_state_pool(
        state_builder, {.layers         = static_cast<std::uint32_t>(kLayers),
                        .conv_channels  = conv_channels,
                        .conv_width     = 3,
                        .value_heads    = value_heads,
                        .value_head_dim = kStateDim,
                        .key_head_dim   = kStateDim,
                        .slot_count     = slot_count,
                        .conv_dtype     = DType::BF16});
    const std::size_t state_bytes = state_builder.finish(256);

    Engine now(record_layout, record_bytes, state_layout, state_bytes);   // fold at once
    Engine later(record_layout, record_bytes, state_layout, state_bytes); // stash, then verify
    Engine settled(record_layout, record_bytes, state_layout, state_bytes); // stash, then settle

    // Round one's records and the states it started from, identical in all three.
    const auto conv_records  = bf16_values(now.records.conv.numel(), seed, 0.08F);
    const auto key_records   = bf16_values(now.records.key.numel(), seed + 1, 0.08F);
    const auto value_records = bf16_values(now.records.value.numel(), seed + 2, 0.08F);
    const auto gate_records  = gate_pairs(now.records.gate.numel() / 2, seed + 3);
    const std::size_t recurrent_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * value_heads * slot_count;
    const std::size_t conv_elements = static_cast<std::size_t>(conv_channels) * 3 * slot_count;
    std::vector<std::vector<std::uint16_t>> recurrent_init, conv_init;
    for (std::int32_t layer = 0; layer < kLayers; ++layer) {
        recurrent_init.push_back(bf16_values(recurrent_elements, seed + 10 + layer, 0.03F));
        conv_init.push_back(bf16_values(conv_elements, seed + 20 + layer, 0.05F));
    }
    for (Engine* engine : {&now, &later, &settled}) {
        upload(engine->records.conv, conv_records, "conv records");
        upload(engine->records.key, key_records, "key records");
        upload(engine->records.value, value_records, "value records");
        upload(engine->records.gate, gate_records, "gate records");
        for (std::int32_t layer = 0; layer < kLayers; ++layer) {
            upload(engine->states.recurrent.at(layer), recurrent_init[layer], "recurrent state");
            upload(engine->states.conv.at(layer), conv_init[layer], "conv state");
        }
    }

    std::vector<ops::GdnReplayFoldRow> rows;
    std::vector<std::int32_t> counts(kLanes, 0);
    for (std::size_t row = 0; row < first_slots.size(); ++row) {
        rows.push_back({first_slots[row], commits[row]});
        counts[static_cast<std::size_t>(first_slots[row])] = commits[row];
    }
    ops::gdn_replay_fold(now.records, now.states.all_layers_view(), rows, nullptr);
    ops::gdn_replay_stash(later.records, later.states.all_layers_view(), rows, nullptr);
    ops::gdn_replay_stash(settled.records, settled.states.all_layers_view(), rows, nullptr);
    upload(later.records.pending_columns, counts, "pending columns");
    upload(settled.records.pending_columns, counts, "pending columns");

    std::vector<ops::GdnReplayFoldRow> pending_rows;
    for (std::size_t row = 0; row < first_slots.size(); ++row) {
        pending_rows.push_back({first_slots[row], commits[row]});
    }
    ops::gdn_replay_fold_pending(settled.records, settled.states.all_layers_view(), pending_rows,
                                 nullptr);
    cuda_synchronize();

    int failures             = 0;
    const std::string suffix = " Hv=" + std::to_string(value_heads);
    for (std::int32_t layer = 0; layer < kLayers; ++layer) {
        failures += expect_same("conv history after stash" + suffix,
                                now.states.conv.at(layer).data, later.states.conv.at(layer).data,
                                now.states.conv.at(layer).bytes());
        failures += expect_same("state after settling" + suffix,
                                now.states.recurrent.at(layer).data,
                                settled.states.recurrent.at(layer).data,
                                now.states.recurrent.at(layer).bytes());
    }

    // Round two: the same inputs through the plain record (after the fold) and through the
    // record that applies the stash.
    const std::size_t columns    = static_cast<std::size_t>(width) * second_rows;
    const std::size_t qk_count   = static_cast<std::size_t>(kStateDim) * kQkHeads * columns;
    const std::size_t v_count    = static_cast<std::size_t>(kStateDim) * value_heads * columns;
    const std::size_t gate_count = static_cast<std::size_t>(value_heads) * columns;
    DeviceBuffer slots_device      = to_device(second_slots);
    const std::vector<std::int32_t> valid_host{4, 2, 4, 3};
    DeviceBuffer valid_device      = to_device(valid_host);
    const Tensor initial(slots_device.p, DType::I32, {second_rows});
    const Tensor valid(valid_device.p, DType::I32, {second_rows});
    DeviceBuffer out_now(v_count * sizeof(std::uint16_t));
    DeviceBuffer out_later(v_count * sizeof(std::uint16_t));
    for (std::int32_t layer = 0; layer < kLayers; ++layer) {
        DeviceBuffer q = to_device(bf16_values(qk_count, seed + 100 + layer, 0.5F));
        DeviceBuffer k = to_device(bf16_values(qk_count, seed + 110 + layer, 0.5F));
        DeviceBuffer v = to_device(bf16_values(v_count, seed + 120 + layer, 0.2F));
        std::vector<float> g(gate_count), beta(gate_count);
        fill_uniform(g, seed + 130 + layer, -1.0F, -0.02F);
        fill_uniform(beta, seed + 140 + layer, 0.05F, 0.95F);
        DeviceBuffer g_device    = to_device(g);
        DeviceBuffer beta_device = to_device(beta);
        const Tensor q_t(q.p, DType::BF16, {kStateDim, kQkHeads, width, second_rows});
        const Tensor k_t(k.p, DType::BF16, {kStateDim, kQkHeads, width, second_rows});
        const Tensor v_t(v.p, DType::BF16, {kStateDim, value_heads, width, second_rows});
        const Tensor g_t(g_device.p, DType::FP32, {value_heads, width, second_rows});
        const Tensor beta_t(beta_device.p, DType::FP32, {value_heads, width, second_rows});
        out_now.fill(0xff);
        out_later.fill(0xff);
        Tensor o_now(out_now.p, DType::BF16, {kStateDim, value_heads, width, second_rows});
        Tensor o_later(out_later.p, DType::BF16, {kStateDim, value_heads, width, second_rows});

        GdnReplayRecordLayer records_now   = now.records.layer(layer, second_rows);
        GdnReplayRecordLayer records_later = later.records.layer(layer, second_rows);
        ops::gated_delta_net_replay_record(q_t, k_t, v_t, g_t, beta_t, kScale,
                                           now.states.recurrent.at(layer), valid, initial,
                                           records_now.key, records_now.value, records_now.gate,
                                           o_now, nullptr);
        ops::gated_delta_net_replay_record(q_t, k_t, v_t, g_t, beta_t, kScale,
                                           later.states.recurrent.at(layer), valid, initial,
                                           records_later.key, records_later.value,
                                           records_later.gate, records_later.pending, o_later,
                                           nullptr);
        cuda_synchronize();
        const std::string at = suffix + " layer " + std::to_string(layer);
        failures += expect_same("verify output" + at, out_now.p, out_later.p, out_now.bytes);
        failures += expect_same("key records" + at, records_now.key.data, records_later.key.data,
                                records_now.key.bytes());
        failures += expect_same("value records" + at, records_now.value.data,
                                records_later.value.data, records_now.value.bytes());
        failures += expect_same("gate records" + at, records_now.gate.data,
                                records_later.gate.data, records_now.gate.bytes());
        failures += expect_same("state after the deferred fold" + at,
                                now.states.recurrent.at(layer).data,
                                later.states.recurrent.at(layer).data,
                                now.states.recurrent.at(layer).bytes());
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case(32, 4401U);
    failures += run_case(48, 4421U);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn deferred fold\n";
    return failures == 0 ? 0 : 1;
}
