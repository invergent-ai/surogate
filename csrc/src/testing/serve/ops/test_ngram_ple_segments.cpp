#include "api/ops/ngram_ple.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/ngram_ple_state.h"
#include "ops/op_tester.h"

#include <iostream>

using namespace sinfer;
using namespace sinfer::test;

namespace {
Weight projection(const Tensor& tensor) {
    Weight weight{};
    weight.qtype = QType::BF16_CTRL;
    weight.layout = QuantLayout::Contiguous;
    weight.payload = weight.qdata = tensor.data;
    weight.payload_bytes = tensor.bytes();
    weight.n = weight.shape[0] = weight.padded_shape[0] = tensor.ne[0];
    weight.k = weight.shape[1] = weight.padded_shape[1] = tensor.ne[1];
    weight.ndim = 2;
    return weight;
}

int exercise(int width) {
    constexpr int batch = 3, slots = 4, hidden = 32, streams = 2, channels = hidden * streams;
    constexpr int kernel = 4, dilation = 3, history = (kernel - 1) * dilation;
    const int tokens = batch * width;
    DeviceArena arena(4U << 20);
    const auto patterned = [&](std::initializer_list<std::int32_t> shape, int salt) {
        auto tensor = arena.alloc(DType::BF16, shape);
        std::vector<std::uint16_t> values(tensor.numel());
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = f32_to_bf16(float(int((i * 7 + salt) % 29) - 14) / 64.0F);
        }
        CUDA_CHECK(cudaMemcpy(tensor.data, values.data(), tensor.bytes(), cudaMemcpyHostToDevice));
        return tensor;
    };
    Tensor key = patterned({channels, 64}, 1), value = patterned({hidden, 64}, 3);
    Tensor conv = patterned({channels, kernel}, 5);
    Tensor norm = arena.alloc(DType::FP32, {channels});
    const std::vector<float> norms(channels, 1.0F);
    CUDA_CHECK(cudaMemcpy(norm.data, norms.data(), norm.bytes(), cudaMemcpyHostToDevice));
    ops::NgramPleWeights weights{projection(key), projection(value), norm, norm, norm, conv};
    // Small, distinct IQ4_NL table rows; both bigram and trigram heads are used.
    std::vector<std::uint8_t> table_bytes(64 * 18);
    for (int row = 0; row < 64; ++row) {
        table_bytes[row * 18] = 0;
        table_bytes[row * 18 + 1] = 0x20; // FP16 scale 1/128
        for (int d = 0; d < 16; ++d) {
            table_bytes[row * 18 + 2 + d] = ((row + d * 3) % 16) | (((row * 3 + d) % 16) << 4);
        }
    }
    auto table_storage = to_device(table_bytes);
    const ops::NgramPleTable table{table_storage.p, 64, 18, 32};
    ops::NgramPleHash hash;
    hash.ngram = 3;
    hash.heads = 2;
    hash.eos_token = 0;
    hash.multipliers[0] = 17; hash.multipliers[1] = 29; hash.multipliers[2] = 43;
    hash.head_vocab_sizes[0] = 31; hash.head_vocab_sizes[1] = 33; hash.head_offsets[1] = 31;
    ops::NgramPleState batched{arena.alloc(DType::I32, {2, slots}), patterned({history, channels, slots}, 7)};
    ops::NgramPleState separate{arena.alloc(DType::I32, {2, slots}), patterned({history, channels, slots}, 7)};
    const std::vector<int> previous{7, 9, 11, 13, 17, 19, 23, 29};
    CUDA_CHECK(cudaMemcpy(batched.history.data, previous.data(), batched.history.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(separate.history.data, previous.data(), separate.history.bytes(), cudaMemcpyHostToDevice));
    auto slot_storage = to_device(std::vector<int>{2, 0, 3});
    Tensor lane_slots(slot_storage.p, DType::I32, {batch});
    ops::NgramPleColumns columns{arena.alloc(DType::I32, {tokens}), arena.alloc(DType::I32, {tokens}),
                                 arena.alloc(DType::I32, {tokens}), arena.alloc(DType::I32, {tokens})};
    ops::ngram_ple_expand_columns(lane_slots, width, columns.slots, columns.segment_begin,
                                  columns.segment_last, nullptr);
    Tensor single_begin = arena.alloc(DType::I32, {width});
    CUDA_CHECK(cudaMemset(single_begin.data, 0, single_begin.bytes()));
    WorkspaceArena workspace(ops::ngram_ple_workspace_capacity_bytes(streams, hidden, 64, 2, 1, tokens));
    int failures = 0;
    for (int round = 0; round < 2; ++round) {
        auto actual = patterned({channels, tokens}, 11 + round);
        auto expected = patterned({channels, tokens}, 11 + round);
        std::vector<int> ids(tokens);
        for (int i = 0; i < tokens; ++i) { ids[i] = i + 31 + 17 * round; }
        if (width > 2) { ids[width + 1] = 0; } // EOS cuts predecessor history.
        CUDA_CHECK(cudaMemcpy(columns.ids.data, ids.data(), columns.ids.bytes(), cudaMemcpyHostToDevice));
        ops::ngram_ple_forward(actual, columns, hash, table, weights, batched,
                               streams, kernel, dilation, 1e-6F, workspace, nullptr);
        for (int row = 0; row < batch; ++row) {
            const int begin = row * width;
            ops::NgramPleColumns single{columns.ids.slice(0, begin, width), single_begin,
                                        columns.slots.slice(0, begin, width),
                                        columns.segment_last.slice(0, begin, width)};
            auto residual = expected.slice(1, begin, width);
            ops::ngram_ple_forward(residual, single, hash, table, weights, separate,
                                   streams, kernel, dilation, 1e-6F, workspace, nullptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        failures += verify_exact("PLE batched vs independent sequence outputs",
                                  from_device<std::uint16_t>(actual.data, actual.numel()),
                                  from_device<std::uint16_t>(expected.data, expected.numel()));
        failures += verify_exact("PLE independent token histories",
                                  from_device<int>(batched.history.data, batched.history.numel()),
                                  from_device<int>(separate.history.data, separate.history.numel()));
        failures += verify_exact("PLE independent convolution histories",
                                  from_device<std::uint16_t>(batched.conv_state.data, batched.conv_state.numel()),
                                  from_device<std::uint16_t>(separate.conv_state.data, separate.conv_state.numel()));
    }
    return failures;
}

// A speculative round commits only the columns it accepted. The pool's snapshot of the last
// accepted column must leave a slot exactly where an ordinary forward over just those columns
// leaves it, and the round's outputs for them must match that forward's.
int exercise_commit(int width) {
    constexpr int batch = 3, slots = 4, hidden = 32, streams = 2, channels = hidden * streams;
    constexpr int kernel = 4, dilation = 3, history = (kernel - 1) * dilation;
    const int tokens = batch * width;
    DeviceArena arena(8U << 20);
    const auto patterned = [&](std::initializer_list<std::int32_t> shape, int salt) {
        auto tensor = arena.alloc(DType::BF16, shape);
        std::vector<std::uint16_t> values(tensor.numel());
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = f32_to_bf16(float(int((i * 7 + salt) % 29) - 14) / 64.0F);
        }
        CUDA_CHECK(cudaMemcpy(tensor.data, values.data(), tensor.bytes(), cudaMemcpyHostToDevice));
        return tensor;
    };
    Tensor key = patterned({channels, 64}, 1), value = patterned({hidden, 64}, 3);
    Tensor conv = patterned({channels, kernel}, 5);
    Tensor norm = arena.alloc(DType::FP32, {channels});
    const std::vector<float> norms(channels, 1.0F);
    CUDA_CHECK(cudaMemcpy(norm.data, norms.data(), norm.bytes(), cudaMemcpyHostToDevice));
    ops::NgramPleWeights weights{projection(key), projection(value), norm, norm, norm, conv};
    std::vector<std::uint8_t> table_bytes(64 * 18);
    for (int row = 0; row < 64; ++row) {
        table_bytes[row * 18] = 0;
        table_bytes[row * 18 + 1] = 0x20;
        for (int d = 0; d < 16; ++d) {
            table_bytes[row * 18 + 2 + d] = ((row + d * 3) % 16) | (((row * 3 + d) % 16) << 4);
        }
    }
    auto table_storage = to_device(table_bytes);
    const ops::NgramPleTable table{table_storage.p, 64, 18, 32};
    ops::NgramPleHash hash;
    hash.ngram = 3;
    hash.heads = 2;
    hash.eos_token = 0;
    hash.multipliers[0] = 17; hash.multipliers[1] = 29; hash.multipliers[2] = 43;
    hash.head_vocab_sizes[0] = 31; hash.head_vocab_sizes[1] = 33; hash.head_offsets[1] = 31;

    const NgramPleStatePoolSpec spec{.history_tokens = 2, .conv_history = history,
                                     .channels = channels, .slot_count = slots, .eos_token = 0,
                                     .snapshot_width = width};
    LayoutBuilder builder;
    const NgramPleStatePoolLayout layout = plan_ngram_ple_state_pool(builder, spec);
    NgramPleStatePool pool(arena.alloc_bytes(builder.finish(256)), layout);
    ops::NgramPleState reference{arena.alloc(DType::I32, {2, slots}),
                                 patterned({history, channels, slots}, 7)};
    const std::vector<int> previous{7, 9, 11, 13, 17, 19, 23, 29};
    CUDA_CHECK(cudaMemcpy(pool.history.data, previous.data(), pool.history.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reference.history.data, previous.data(), reference.history.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pool.conv_state.data, reference.conv_state.data, reference.conv_state.bytes(),
                          cudaMemcpyDeviceToDevice));

    const std::vector<int> lane_slot_values{2, 0, 3};
    auto slot_storage = to_device(lane_slot_values);
    Tensor lane_slots(slot_storage.p, DType::I32, {batch});
    ops::NgramPleColumns columns{arena.alloc(DType::I32, {tokens}), arena.alloc(DType::I32, {tokens}),
                                 arena.alloc(DType::I32, {tokens}), arena.alloc(DType::I32, {tokens})};
    ops::ngram_ple_expand_columns(lane_slots, width, columns.slots, columns.segment_begin,
                                  columns.segment_last, nullptr);
    Tensor single_begin = arena.alloc(DType::I32, {width});
    CUDA_CHECK(cudaMemset(single_begin.data, 0, single_begin.bytes()));
    Tensor single_last = arena.alloc(DType::I32, {width});
    WorkspaceArena workspace(ops::ngram_ple_workspace_capacity_bytes(streams, hidden, 64, 2, 1, tokens));
    ops::NgramPleState round_state{pool.history, pool.conv_state, pool.history_snapshots,
                                   pool.conv_snapshots};
    int failures = 0;
    for (int round = 0; round < 3; ++round) {
        auto actual   = patterned({channels, tokens}, 11 + round);
        auto expected = patterned({channels, tokens}, 11 + round);
        std::vector<int> ids(tokens);
        for (int i = 0; i < tokens; ++i) { ids[i] = i + 31 + 17 * round; }
        if (width > 2) { ids[width + 1] = 0; } // EOS cuts predecessor history.
        CUDA_CHECK(cudaMemcpy(columns.ids.data, ids.data(), columns.ids.bytes(), cudaMemcpyHostToDevice));
        ops::ngram_ple_forward(actual, columns, hash, table, weights, round_state, streams, kernel,
                               dilation, 1e-6F, workspace, nullptr);
        for (int row = 0; row < batch; ++row) {
            // The first column only, a prefix, and every column, rotating over the rounds.
            const int accepted = 1 + ((row + round) % 3 == 0 ? 0
                                      : (row + round) % 3 == 1 ? width / 2
                                                               : width - 1);
            pool.commit_snapshot(lane_slot_values[row], accepted - 1, nullptr);
            const int begin = row * width;
            CUDA_CHECK(cudaMemset(single_last.data, 0, single_last.bytes()));
            const int one = 1;
            CUDA_CHECK(cudaMemcpy(static_cast<int*>(single_last.data) + accepted - 1, &one, sizeof(int),
                                  cudaMemcpyHostToDevice));
            ops::NgramPleColumns single{columns.ids.slice(0, begin, accepted),
                                        single_begin.slice(0, 0, accepted),
                                        columns.slots.slice(0, begin, accepted),
                                        single_last.slice(0, 0, accepted)};
            auto residual = expected.slice(1, begin, accepted);
            ops::ngram_ple_forward(residual, single, hash, table, weights, reference, streams,
                                   kernel, dilation, 1e-6F, workspace, nullptr);
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto got  = from_device<std::uint16_t>(actual.data, actual.numel());
            const auto want = from_device<std::uint16_t>(expected.data, expected.numel());
            failures += verify_exact(
                "PLE speculative columns vs the committed prefix alone",
                std::vector<std::uint16_t>(got.begin() + std::int64_t(begin) * channels,
                                           got.begin() + std::int64_t(begin + accepted) * channels),
                std::vector<std::uint16_t>(want.begin() + std::int64_t(begin) * channels,
                                           want.begin() + std::int64_t(begin + accepted) * channels));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        failures += verify_exact("PLE committed token histories",
                                  from_device<int>(pool.history.data, pool.history.numel()),
                                  from_device<int>(reference.history.data, reference.history.numel()));
        failures += verify_exact("PLE committed convolution histories",
                                  from_device<std::uint16_t>(pool.conv_state.data, pool.conv_state.numel()),
                                  from_device<std::uint16_t>(reference.conv_state.data, reference.conv_state.numel()));
    }
    return failures;
}
} // namespace

int main() {
    if (cuda_unavailable()) { return 77; }
    int failures = 0;
    for (const int width : {1, 4, 8}) { failures += exercise(width); }
    for (const int width : {1, 4, 8}) { failures += exercise_commit(width); }
    return failures != 0;
}
