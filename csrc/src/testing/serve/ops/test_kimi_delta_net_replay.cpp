// The replay forms of Kimi Delta Attention: the record and the fold that speculative decoding
// verifies a round with, over the mixer whose forget gate is one value per key channel.
//
// Two checks, each against the mixer's own snapshot form rather than against the delta net's
// replay: (1) the record form's output is bit-identical to the snapshot form's, and what it
// records is bit-identical to what it was given; (2) folding a record prefix into the all-layer
// state lands on the same state the snapshot form reaches after that many tokens, with the
// convolution history advanced by the same prefix. The fold's geometry is GLM-5.3-Flash's --
// 34 layers of 64 symmetric heads over three planes of 8192 channels -- because that is the
// registered one, and a fold instantiated for another shape would prove nothing about it.

#include "api/ops/gdn_replay.h"
#include "api/ops/kimi_delta_net.h"

#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
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

constexpr std::int32_t kStateDim    = 128;
constexpr std::uint16_t kBf16Poison = 0xffffU;
constexpr std::uint32_t kFp32Poison = 0xffffffffU;
constexpr float kScale              = 1.0F / std::sqrt(128.0F);

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float signed_pattern(std::uint32_t key, float magnitude) {
    const std::int32_t centered = static_cast<std::int32_t>(mix(key) % 2001U) - 1000;
    return static_cast<float>(centered) * (magnitude / 1000.0F);
}

std::uint16_t bf16_pattern(std::uint32_t key, float magnitude) {
    return f32_to_bf16(signed_pattern(key, magnitude));
}

/// A log-decay per channel, strictly negative so the state contracts, and a beta in (0,1).
float gate_pattern(std::uint32_t key) { return -0.02F - static_cast<float>(mix(key) % 900U) / 1000.0F; }
float beta_pattern(std::uint32_t key) { return 0.05F + static_cast<float>(mix(key) % 900U) / 1000.0F; }

std::vector<std::uint16_t> make_bf16(std::size_t count, std::uint32_t seed, float magnitude) {
    std::vector<std::uint16_t> bits(count);
    for (std::size_t index = 0; index < count; ++index) {
        bits[index] = bf16_pattern(seed + static_cast<std::uint32_t>(index), magnitude);
    }
    return bits;
}

// ---- (1) the record form against the snapshot form ---------------------------------------

int run_record_case(std::int32_t heads, std::int32_t width, std::int32_t batch,
                    std::vector<std::int32_t> valid_columns, std::uint32_t seed) {
    const bool dense = valid_columns.empty();
    if (dense) { valid_columns.assign(static_cast<std::size_t>(batch), width); }
    const std::int32_t columns    = width * batch;
    const std::int32_t slots      = columns + batch;
    const std::size_t qk_elements = static_cast<std::size_t>(kStateDim) * heads * columns;
    const std::size_t gate_elements = static_cast<std::size_t>(heads) * columns;
    const std::size_t state_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * heads * slots;

    const std::vector<std::uint16_t> q_bits = make_bf16(qk_elements, seed, 0.9F);
    const std::vector<std::uint16_t> k_bits = make_bf16(qk_elements, seed + 1000003U, 0.9F);
    const std::vector<std::uint16_t> v_bits = make_bf16(qk_elements, seed + 2000003U, 0.9F);
    std::vector<float> g(qk_elements);
    std::vector<float> beta(gate_elements);
    for (std::size_t index = 0; index < g.size(); ++index) {
        g[index] = gate_pattern(seed + 3000003U + static_cast<std::uint32_t>(index));
    }
    for (std::size_t index = 0; index < beta.size(); ++index) {
        beta[index] = beta_pattern(seed + 4000003U + static_cast<std::uint32_t>(index));
    }
    // Sign and subnormal edge cases: a record that rounds or canonicalises would show here.
    g[0]    = std::bit_cast<float>(0x80000000U);
    beta[0] = std::bit_cast<float>(0x00000001U);
    std::vector<std::uint16_t> state_bits(state_elements);
    for (std::size_t index = 0; index < state_elements; ++index) {
        state_bits[index] = bf16_pattern(seed + 5000003U + static_cast<std::uint32_t>(index), 0.4F);
    }
    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> snapshot_bases(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        snapshot_bases[static_cast<std::size_t>(row)] = row * width;
        initial_slots[static_cast<std::size_t>(row)]  = columns + row;
    }

    DeviceBuffer device_q       = to_device(q_bits);
    DeviceBuffer device_k       = to_device(k_bits);
    DeviceBuffer device_v       = to_device(v_bits);
    DeviceBuffer device_g       = to_device(g);
    DeviceBuffer device_beta    = to_device(beta);
    DeviceBuffer snapshot_state = to_device(state_bits);
    DeviceBuffer record_state   = to_device(state_bits);
    DeviceBuffer device_initial = to_device(initial_slots);
    DeviceBuffer device_bases   = to_device(snapshot_bases);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }
    DeviceBuffer snapshot_out(qk_elements * sizeof(std::uint16_t));
    DeviceBuffer record_out(qk_elements * sizeof(std::uint16_t));
    DeviceBuffer key_record(qk_elements * sizeof(std::uint16_t));
    DeviceBuffer value_record(qk_elements * sizeof(std::uint16_t));
    DeviceBuffer gate_record(qk_elements * sizeof(float));
    DeviceBuffer beta_record(gate_elements * sizeof(float));
    for (DeviceBuffer* buffer : {&snapshot_out, &record_out, &key_record, &value_record,
                                 &gate_record, &beta_record}) {
        buffer->fill(0xff);
    }

    Tensor q(device_q.p, DType::BF16, {kStateDim, heads, width, batch});
    Tensor k(device_k.p, DType::BF16, {kStateDim, heads, width, batch});
    Tensor v(device_v.p, DType::BF16, {kStateDim, heads, width, batch});
    Tensor g_tensor(device_g.p, DType::FP32, {kStateDim, heads, width, batch});
    Tensor beta_tensor(device_beta.p, DType::FP32, {heads, width, batch});
    Tensor snapshot_states(snapshot_state.p, DType::BF16, {kStateDim, kStateDim, heads, slots});
    Tensor record_states(record_state.p, DType::BF16, {kStateDim, kStateDim, heads, slots});
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
    Tensor initial(device_initial.p, DType::I32, {batch});
    Tensor bases(device_bases.p, DType::I32, {batch});
    Tensor snapshot_output(snapshot_out.p, DType::BF16, {kStateDim, heads, width, batch});
    Tensor record_output(record_out.p, DType::BF16, {kStateDim, heads, width, batch});
    Tensor key_record_tensor(key_record.p, DType::BF16, {kStateDim, heads, width, batch});
    Tensor value_record_tensor(value_record.p, DType::BF16, {kStateDim, heads, width, batch});
    Tensor gate_record_tensor(gate_record.p, DType::FP32, {kStateDim, heads, width, batch});
    Tensor beta_record_tensor(beta_record.p, DType::FP32, {heads, width, batch});

    ops::kimi_delta_net_snapshot(q, k, v, g_tensor, beta_tensor, kScale, true, snapshot_states,
                                 valid, initial, bases, snapshot_output, nullptr);
    ops::kimi_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, record_states, valid,
                                      initial, key_record_tensor, value_record_tensor,
                                      gate_record_tensor, beta_record_tensor, record_output,
                                      nullptr);
    cuda_synchronize();

    const std::string suffix = " H=" + std::to_string(heads) + " T=" + std::to_string(width) +
                               " B=" + std::to_string(batch);
    if (from_device<std::uint16_t>(snapshot_out, qk_elements) !=
        from_device<std::uint16_t>(record_out, qk_elements)) {
        std::cerr << "record output differs from the snapshot form's" << suffix << "\n";
        return 1;
    }
    const auto key_after   = from_device<std::uint16_t>(key_record, qk_elements);
    const auto value_after = from_device<std::uint16_t>(value_record, qk_elements);
    const auto gate_after  = from_device<std::uint32_t>(gate_record, qk_elements);
    const auto beta_after  = from_device<std::uint32_t>(beta_record, gate_elements);
    const auto output_bits = from_device<std::uint16_t>(record_out, qk_elements);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t extent = valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < width; ++token) {
            const std::int64_t column = static_cast<std::int64_t>(row) * width + token;
            const bool active         = token < extent;
            for (std::int32_t head = 0; head < heads; ++head) {
                const std::size_t base = static_cast<std::size_t>((column * heads + head) * kStateDim);
                const std::size_t scalar = static_cast<std::size_t>(column * heads + head);
                for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                    if (key_after[base + dim] != (active ? k_bits[base + dim] : kBf16Poison)) {
                        std::cerr << "key record mismatch" << suffix << "\n";
                        return 1;
                    }
                    if (value_after[base + dim] != (active ? v_bits[base + dim] : kBf16Poison)) {
                        std::cerr << "value record mismatch" << suffix << "\n";
                        return 1;
                    }
                    const std::uint32_t expected_g =
                        active ? std::bit_cast<std::uint32_t>(g[base + dim]) : kFp32Poison;
                    if (gate_after[base + dim] != expected_g) {
                        std::cerr << "gate record mismatch" << suffix << " row=" << row
                                  << " token=" << token << " head=" << head << " dim=" << dim
                                  << "\n";
                        return 1;
                    }
                }
                const std::uint32_t expected_beta =
                    active ? std::bit_cast<std::uint32_t>(beta[scalar]) : kFp32Poison;
                if (beta_after[scalar] != expected_beta) {
                    std::cerr << "beta record mismatch" << suffix << "\n";
                    return 1;
                }
            }
            if (!active) {
                const std::size_t output_base = static_cast<std::size_t>(column) * heads * kStateDim;
                for (std::int32_t index = 0; index < heads * kStateDim; ++index) {
                    if (output_bits[output_base + index] != 0) {
                        std::cerr << "record invalid output is not zero" << suffix << "\n";
                        return 1;
                    }
                }
            }
        }
    }
    if (from_device<std::uint16_t>(record_state, state_elements) != state_bits) {
        std::cerr << "replay record modified the source state" << suffix << "\n";
        return 1;
    }
    return 0;
}

// ---- (2) the fold against the snapshot form ----------------------------------------------

struct FoldProfile {
    std::int32_t layers;
    std::int32_t heads;
    std::int32_t conv_channels;
};

constexpr FoldProfile kGlm{34, 64, 24576};
constexpr std::int32_t kRecordCapacity = 4;
constexpr std::size_t kGuardBytes      = 256;

void* offset_pointer(void* pointer, std::size_t bytes) {
    return static_cast<void*>(static_cast<std::byte*>(pointer) + bytes);
}

int run_fold_case(const FoldProfile profile, std::int32_t width, std::int32_t rows,
                  const std::vector<std::int32_t>& commits, std::uint32_t seed) {
    // The rows' slots, scattered so a fold that indexed by row rather than by slot shows.
    std::vector<std::int32_t> slots{5, 1, 3, 0};
    slots.resize(static_cast<std::size_t>(rows));
    const std::int32_t slot_count = 6;
    const std::int32_t outer      = profile.layers * kRecordCapacity;
    const std::size_t recurrent_slot_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * profile.heads;
    const std::size_t recurrent_slot_bytes = recurrent_slot_elements * sizeof(std::uint16_t);
    const std::size_t conv_slot_elements   = static_cast<std::size_t>(profile.conv_channels) * 3;
    const std::size_t conv_slot_bytes      = conv_slot_elements * sizeof(std::uint16_t);

    const GdnReplayRecordSpec record_spec{
        .layers          = profile.layers,
        .record_capacity = kRecordCapacity,
        .width           = width,
        .conv_channels   = profile.conv_channels,
        .qk_heads        = profile.heads,
        .value_heads     = profile.heads,
        .key_dim         = kStateDim,
        .value_dim       = kStateDim,
        .diagonal_gate   = true,
    };
    LayoutBuilder record_builder;
    const GdnReplayRecordLayout record_layout =
        plan_gdn_replay_records(record_builder, record_spec);
    const std::size_t record_bytes = record_builder.finish(256);
    DeviceBuffer record_storage(record_bytes + 2 * kGuardBytes);
    record_storage.fill(0xa5);
    void* record_base = offset_pointer(record_storage.p, kGuardBytes);
    cuda_check(cudaMemset(record_base, 0xff, record_bytes), "initialize replay records");
    const GdnReplayRecords records({record_base, record_bytes}, record_layout);

    // Records: poison everywhere, patterns on the committed prefix of every active row.
    std::vector<std::uint16_t> conv_records(records.conv.numel(), 0xffffU);
    std::vector<std::uint16_t> key_records(records.key.numel(), 0xffffU);
    std::vector<std::uint16_t> value_records(records.value.numel(), 0xffffU);
    std::vector<float> gate_records(records.gate.numel(), std::bit_cast<float>(kFp32Poison));
    std::vector<float> beta_records(records.beta.numel(), std::bit_cast<float>(kFp32Poison));
    for (std::int32_t layer = 0; layer < profile.layers; ++layer) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::int32_t commit = commits[static_cast<std::size_t>(row)];
            const std::int64_t record_outer =
                static_cast<std::int64_t>(layer) * kRecordCapacity + row;
            for (std::int32_t token = 0; token < commit; ++token) {
                const std::int64_t column = record_outer * width + token;
                const std::uint32_t key   = seed + static_cast<std::uint32_t>(column) * 7919U;
                for (std::int32_t channel = 0; channel < profile.conv_channels; ++channel) {
                    conv_records[static_cast<std::size_t>(column) * profile.conv_channels + channel] =
                        bf16_pattern(key + 11U * channel, 0.06F);
                }
                for (std::int32_t head = 0; head < profile.heads; ++head) {
                    const std::size_t base =
                        static_cast<std::size_t>((column * profile.heads + head) * kStateDim);
                    for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                        key_records[base + dim]   = bf16_pattern(key + 100003U + head * 131U + dim, 0.9F);
                        value_records[base + dim] = bf16_pattern(key + 200003U + head * 137U + dim, 0.9F);
                        gate_records[base + dim]  = gate_pattern(key + 300007U + head * 139U + dim);
                    }
                    beta_records[static_cast<std::size_t>(column * profile.heads + head)] =
                        beta_pattern(key + 400009U + head);
                }
            }
        }
    }
    cuda_check(cudaMemcpy(records.conv.data, conv_records.data(), records.conv.bytes(),
                          cudaMemcpyHostToDevice), "upload conv records");
    cuda_check(cudaMemcpy(records.key.data, key_records.data(), records.key.bytes(),
                          cudaMemcpyHostToDevice), "upload key records");
    cuda_check(cudaMemcpy(records.value.data, value_records.data(), records.value.bytes(),
                          cudaMemcpyHostToDevice), "upload value records");
    cuda_check(cudaMemcpy(records.gate.data, gate_records.data(), records.gate.bytes(),
                          cudaMemcpyHostToDevice), "upload gate records");
    cuda_check(cudaMemcpy(records.beta.data, beta_records.data(), records.beta.bytes(),
                          cudaMemcpyHostToDevice), "upload beta records");

    LayoutBuilder state_builder;
    const LinearAttentionStatePoolLayout state_layout = plan_linear_attention_state_pool(
        state_builder, {.layers         = static_cast<std::uint32_t>(profile.layers),
                        .conv_channels  = profile.conv_channels,
                        .conv_width     = 3,
                        .value_heads    = profile.heads,
                        .value_head_dim = kStateDim,
                        .key_head_dim   = kStateDim,
                        .slot_count     = slot_count,
                        .conv_dtype     = DType::BF16});
    const std::size_t state_bytes = state_builder.finish(256);
    DeviceBuffer state_storage(state_bytes + 2 * kGuardBytes);
    state_storage.fill(0xa5);
    void* state_base = offset_pointer(state_storage.p, kGuardBytes);
    cuda_check(cudaMemset(state_base, 0, state_bytes), "initialize all-layer state");
    LinearAttentionStatePool state_pool({state_base, state_bytes}, state_layout);

    // Expected states, from the snapshot form over the recorded prefix; expected conv history
    // by hand. Both are computed before the fold from the same initial state it will read.
    std::vector<std::vector<std::uint16_t>> expected_recurrent(
        static_cast<std::size_t>(profile.layers * rows));
    std::vector<std::vector<std::uint16_t>> expected_conv(
        static_cast<std::size_t>(profile.layers * rows));
    DeviceBuffer local_states(static_cast<std::size_t>(width + 1) * recurrent_slot_bytes);
    DeviceBuffer q(static_cast<std::size_t>(kStateDim) * profile.heads * width * sizeof(std::uint16_t));
    DeviceBuffer out(q.bytes);
    q.fill(0);
    DeviceBuffer valid_device(sizeof(std::int32_t));
    DeviceBuffer initial_device(sizeof(std::int32_t));
    DeviceBuffer base_device(sizeof(std::int32_t));
    const std::int32_t local_initial = width;
    const std::int32_t local_base    = 0;
    initial_device.copy_from_host(&local_initial, sizeof(local_initial));
    base_device.copy_from_host(&local_base, sizeof(local_base));
    const Tensor q_tensor(q.p, DType::BF16, {kStateDim, profile.heads, width, 1});
    Tensor local_state_tensor(local_states.p, DType::BF16,
                              {kStateDim, kStateDim, profile.heads, width + 1});
    Tensor output(out.p, DType::BF16, {kStateDim, profile.heads, width, 1});
    Tensor initial_selector(initial_device.p, DType::I32, {1});
    Tensor base_selector(base_device.p, DType::I32, {1});

    for (std::int32_t layer = 0; layer < profile.layers; ++layer) {
        const GdnReplayRecordLayer layer_records = records.layer(layer, rows);
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::size_t at      = static_cast<std::size_t>(layer * rows + row);
            const std::int32_t commit = commits[static_cast<std::size_t>(row)];
            const std::int32_t slot   = slots[static_cast<std::size_t>(row)];
            std::vector<std::uint16_t> initial_recurrent(recurrent_slot_elements);
            for (std::size_t index = 0; index < recurrent_slot_elements; ++index) {
                initial_recurrent[index] =
                    bf16_pattern(seed + 500009U + layer * 227U + row * 43U + static_cast<std::uint32_t>(index), 0.3F);
            }
            std::vector<std::uint16_t> initial_conv(conv_slot_elements);
            for (std::size_t index = 0; index < conv_slot_elements; ++index) {
                initial_conv[index] =
                    bf16_pattern(seed + 600011U + layer * 229U + row * 47U + static_cast<std::uint32_t>(index), 0.05F);
            }
            const Tensor recurrent_slot = state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), slot);
            const Tensor conv_slot      = state_pool.conv_slot(static_cast<std::uint32_t>(layer), slot);
            cuda_check(cudaMemcpy(recurrent_slot.data, initial_recurrent.data(), recurrent_slot_bytes,
                                  cudaMemcpyHostToDevice), "upload initial recurrent state");
            cuda_check(cudaMemcpy(conv_slot.data, initial_conv.data(), conv_slot_bytes,
                                  cudaMemcpyHostToDevice), "upload initial conv state");

            // Conv history: tail_3(old || record[0:commit]).
            expected_conv[at] = initial_conv;
            if (commit > 0) {
                const std::int64_t record_outer = static_cast<std::int64_t>(layer) * kRecordCapacity + row;
                const auto record_value = [&](std::int32_t token, std::int32_t channel) {
                    return conv_records[static_cast<std::size_t>((record_outer * width + token) * profile.conv_channels) + channel];
                };
                for (std::int32_t channel = 0; channel < profile.conv_channels; ++channel) {
                    std::uint16_t history[3];
                    for (std::int32_t h = 0; h < 3; ++h) {
                        // Position h of the new history is old position h + commit, or a record.
                        const std::int32_t source = h + commit;
                        history[h] = source < 3
                                         ? initial_conv[static_cast<std::size_t>(source) * profile.conv_channels + channel]
                                         : record_value(source - 3, channel);
                    }
                    for (std::int32_t h = 0; h < 3; ++h) {
                        expected_conv[at][static_cast<std::size_t>(h) * profile.conv_channels + channel] = history[h];
                    }
                }
            }

            if (commit == 0) {
                expected_recurrent[at] = initial_recurrent;
                continue;
            }
            cuda_check(cudaMemcpy(offset_pointer(local_states.p, static_cast<std::size_t>(local_initial) * recurrent_slot_bytes),
                                  initial_recurrent.data(), recurrent_slot_bytes,
                                  cudaMemcpyHostToDevice), "upload local snapshot initial state");
            valid_device.copy_from_host(&commit, sizeof(commit));
            Tensor key   = layer_records.key.slice(3, row, 1);
            Tensor value = layer_records.value.slice(3, row, 1);
            Tensor gate  = layer_records.gate.slice(3, row, 1);
            Tensor beta  = layer_records.beta.slice(2, row, 1);
            Tensor valid(valid_device.p, DType::I32, {1});
            ops::kimi_delta_net_snapshot(q_tensor, key, value, gate, beta, kScale, true,
                                         local_state_tensor, valid, initial_selector,
                                         base_selector, output, nullptr);
            expected_recurrent[at] = from_device<std::uint16_t>(
                offset_pointer(local_states.p, static_cast<std::size_t>(commit - 1) * recurrent_slot_bytes),
                recurrent_slot_elements);
        }
    }

    const auto records_before = from_device<std::uint8_t>(record_storage, record_storage.bytes);
    std::vector<ops::GdnReplayFoldRow> fold_rows(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        fold_rows[static_cast<std::size_t>(row)] = {slots[static_cast<std::size_t>(row)],
                                                    commits[static_cast<std::size_t>(row)]};
    }
    ops::gdn_replay_fold(records, state_pool.all_layers_view(), fold_rows, nullptr);
    cuda_synchronize();

    const std::string suffix = " L=" + std::to_string(profile.layers) + " H=" +
                               std::to_string(profile.heads) + " T=" + std::to_string(width) +
                               " B=" + std::to_string(rows);
    for (std::int32_t layer = 0; layer < profile.layers; ++layer) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::size_t at    = static_cast<std::size_t>(layer * rows + row);
            const std::int32_t slot = slots[static_cast<std::size_t>(row)];
            const Tensor actual_state = state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), slot);
            if (from_device<std::uint16_t>(actual_state.data, recurrent_slot_elements) != expected_recurrent[at]) {
                std::cerr << "fold recurrent state differs from the snapshot form's" << suffix
                          << " layer=" << layer << " row=" << row << "\n";
                return 1;
            }
            const Tensor actual_conv = state_pool.conv_slot(static_cast<std::uint32_t>(layer), slot);
            if (from_device<std::uint16_t>(actual_conv.data, conv_slot_elements) != expected_conv[at]) {
                std::cerr << "fold conv history differs" << suffix << " layer=" << layer
                          << " row=" << row << "\n";
                return 1;
            }
        }
        for (std::int32_t slot = 0; slot < slot_count; ++slot) {
            if (std::find(slots.begin(), slots.end(), slot) != slots.end()) { continue; }
            const auto recurrent = from_device<std::uint16_t>(
                state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), slot).data,
                recurrent_slot_elements);
            const auto conv = from_device<std::uint16_t>(
                state_pool.conv_slot(static_cast<std::uint32_t>(layer), slot).data, conv_slot_elements);
            const auto zero = [](std::uint16_t value) { return value == 0; };
            if (!std::all_of(recurrent.begin(), recurrent.end(), zero) ||
                !std::all_of(conv.begin(), conv.end(), zero)) {
                std::cerr << "fold modified an inactive slot" << suffix << " layer=" << layer
                          << " slot=" << slot << "\n";
                return 1;
            }
        }
    }
    if (from_device<std::uint8_t>(record_storage, record_storage.bytes) != records_before) {
        std::cerr << "fold modified record storage" << suffix << "\n";
        return 1;
    }
    const auto state_after = from_device<std::uint8_t>(state_storage, state_storage.bytes);
    const auto guard       = [](std::uint8_t byte) { return byte == 0xa5; };
    if (!std::all_of(state_after.begin(), state_after.begin() + static_cast<std::ptrdiff_t>(kGuardBytes), guard) ||
        !std::all_of(state_after.end() - static_cast<std::ptrdiff_t>(kGuardBytes), state_after.end(), guard)) {
        std::cerr << "fold modified the state's outer guard" << suffix << "\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    // The record form, at the registered head count and at a small one the kernel is generic
    // over: dense rows, a partial row, and a batch of mixed extents.
    failures += run_record_case(64, 2, 1, {}, 2101U);
    failures += run_record_case(64, 6, 4, {6, 3, 1, 5}, 2111U);
    failures += run_record_case(8, 16, 2, {16, 9}, 2121U);
    // The fold, at GLM-5.3-Flash's geometry: one row, then several with the whole spread of
    // commit extents including zero (a strict no-op) and the full window.
    failures += run_fold_case(kGlm, 2, 1, {2}, 2201U);
    failures += run_fold_case(kGlm, 6, 4, {6, 0, 1, 4}, 2211U);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " kimi_delta_net_replay\n";
    return failures == 0 ? 0 : 1;
}
