#include "ninfer/ops/swa.h"

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kD       = 128;
constexpr int kQHeads  = 32;
constexpr int kKVHeads = 8;
constexpr int kGroup   = 4;
constexpr int kWindow  = 4096;
constexpr float kScale = 0.08838834764831844055f;

constexpr ReductionCriterion kSwaBf16Criterion{
    .relative_l2                     = 3.95e-3,
    .gross_absolute                  = 3e-4,
    .gross_relative_to_max_reference = 3.0e-3,
};

std::size_t q_index(int d, int q_head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(q_head) +
                static_cast<std::size_t>(kQHeads) * static_cast<std::size_t>(token));
}

std::size_t query_kv_index(int d, int kv_head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(kv_head) +
                static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(token));
}

std::size_t context_index(int d, int kv_head, int slot) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(slot) +
                static_cast<std::size_t>(kWindow) * static_cast<std::size_t>(kv_head));
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

void swa_oracle(const std::vector<float>& q, const std::vector<float>& query_k,
                const std::vector<float>& query_v, const std::vector<float>& context_k,
                const std::vector<float>& context_v, const std::vector<int>& positions,
                int context_length, std::vector<double>& out) {
    const int tokens = static_cast<int>(positions.size());
    out.assign(static_cast<std::size_t>(kD) * kQHeads * tokens, 0.0);
    std::vector<double> scores(static_cast<std::size_t>(kWindow - 1 + tokens));

    for (int token = 0; token < tokens; ++token) {
        const int query_position = positions[static_cast<std::size_t>(token)];
        const int context_begin  = std::max(0, query_position - (kWindow - 1));
        const int context_keys   = context_length - context_begin;
        const int key_count      = context_keys + tokens;

        for (int q_head = 0; q_head < kQHeads; ++q_head) {
            const int kv_head = q_head / kGroup;
            double max_score  = -std::numeric_limits<double>::infinity();
            for (int key = 0; key < key_count; ++key) {
                double dot = 0.0;
                for (int d = 0; d < kD; ++d) {
                    const double k_value =
                        key < context_keys
                            ? static_cast<double>(context_k[context_index(
                                  d, kv_head, (context_begin + key) & (kWindow - 1))])
                            : static_cast<double>(
                                  query_k[query_kv_index(d, kv_head, key - context_keys)]);
                    dot += static_cast<double>(q[q_index(d, q_head, token)]) * k_value;
                }
                const double score                    = dot * static_cast<double>(kScale);
                scores[static_cast<std::size_t>(key)] = score;
                max_score                             = std::max(max_score, score);
            }

            double denominator = 0.0;
            for (int key = 0; key < key_count; ++key) {
                double& score = scores[static_cast<std::size_t>(key)];
                score         = std::exp(score - max_score);
                denominator += score;
            }
            for (int d = 0; d < kD; ++d) {
                double numerator = 0.0;
                for (int key = 0; key < key_count; ++key) {
                    const double v_value =
                        key < context_keys
                            ? static_cast<double>(context_v[context_index(
                                  d, kv_head, (context_begin + key) & (kWindow - 1))])
                            : static_cast<double>(
                                  query_v[query_kv_index(d, kv_head, key - context_keys)]);
                    numerator += scores[static_cast<std::size_t>(key)] * v_value;
                }
                out[q_index(d, q_head, token)] = numerator / denominator;
            }
        }
    }
}

CyclicKVCacheLayerView make_context_view(DeviceBuffer& k, DeviceBuffer& v, int lane_capacity = 1) {
    return {
        .k               = Tensor(k.p, DType::BF16, {kD, kWindow, kKVHeads, lane_capacity}),
        .v               = Tensor(v.p, DType::BF16, {kD, kWindow, kKVHeads, lane_capacity}),
        .capacity        = kWindow,
        .padded_capacity = kWindow,
        .num_kv_heads    = kKVHeads,
        .head_dim        = kD,
        .lane_capacity   = lane_capacity,
    };
}

enum class InputProfile {
    Random,
    WindowBoundary,
};

int run_case(int tokens, int context_length, InputProfile profile = InputProfile::Random,
             int envelope_max = -1) {
    if (envelope_max < 0) envelope_max = context_length;
    const std::size_t q_count        = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t query_kv_count = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t context_count  = static_cast<std::size_t>(kD) * kWindow * kKVHeads;

    std::vector<float> q(q_count);
    std::vector<float> query_k(query_kv_count);
    std::vector<float> query_v(query_kv_count);
    std::vector<float> context_k(context_count);
    std::vector<float> context_v(context_count);
    const auto seed = static_cast<unsigned>(tokens * 131 + context_length * 17);
    fill_uniform(q, 101u + seed, -0.35f, 0.35f);
    fill_uniform(query_k, 211u + seed, -0.4f, 0.4f);
    fill_uniform(query_v, 307u + seed, -0.8f, 0.8f);
    fill_uniform(context_k, 401u + seed, -0.4f, 0.4f);
    fill_uniform(context_v, 503u + seed, -0.8f, 0.8f);

    if (profile == InputProfile::WindowBoundary) {
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(query_k.begin(), query_k.end(), 0.0f);
        std::fill(query_v.begin(), query_v.end(), 0.0f);
        std::fill(context_k.begin(), context_k.end(), 0.0f);
        std::fill(context_v.begin(), context_v.end(), 0.0f);
        for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
            for (int d = 0; d < kD; ++d) {
                context_v[context_index(d, kv_head, 0)]         = 512.0f;
                context_v[context_index(d, kv_head, 1)]         = 256.0f;
                query_v[query_kv_index(d, kv_head, tokens - 1)] = 1.0f;
            }
        }
    }

    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(context_k);
    round_to_bf16(context_v);

    std::vector<int> positions(static_cast<std::size_t>(tokens));
    for (int token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = context_length + token;
    }
    std::vector<double> reference;
    swa_oracle(q, query_k, query_v, context_k, context_v, positions, context_length, reference);

    const auto q_expected         = bf16_bits(q);
    const auto query_k_expected   = bf16_bits(query_k);
    const auto query_v_expected   = bf16_bits(query_v);
    const auto context_k_expected = bf16_bits(context_k);
    const auto context_v_expected = bf16_bits(context_v);

    DeviceBuffer d_q         = to_device(q_expected);
    DeviceBuffer d_query_k   = to_device(query_k_expected);
    DeviceBuffer d_query_v   = to_device(query_v_expected);
    DeviceBuffer d_context_k = to_device(context_k_expected);
    DeviceBuffer d_context_v = to_device(context_v_expected);
    DeviceBuffer d_positions = to_device_i32(positions);
    DeviceBuffer d_valid     = to_device<std::int32_t>({tokens});
    DeviceBuffer d_lane      = to_device<std::int32_t>({0});
    GuardedDeviceBuffer d_out(q_count * sizeof(std::uint16_t));
    d_out.fill(0x7f);

    Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens, 1});
    Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor positions_tensor(d_positions.p, DType::I32, {tokens, 1});
    Tensor valid_tensor(d_valid.p, DType::I32, {1});
    Tensor lane_tensor(d_lane.p, DType::I32, {1});
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens, 1});
    CyclicKVCacheLayerView context = make_context_view(d_context_k, d_context_v);
    const ops::SwaContextExecutionEnvelope envelope{0, static_cast<std::uint32_t>(envelope_max)};
    const std::size_t workspace_bytes =
        ops::swa_workspace_capacity_bytes(envelope, tokens, tokens, 1);
    DeviceArena workspace(workspace_bytes);

    ops::swa(q_tensor, query_k_tensor, query_v_tensor, positions_tensor, valid_tensor, lane_tensor,
             kScale, context, envelope, workspace, out_tensor, nullptr);
    cuda_synchronize();

    std::string label = "swa T=" + std::to_string(tokens) + " L=" + std::to_string(context_length);
    if (envelope_max != context_length) {
        label += " envelope=[0," + std::to_string(envelope_max) + "]";
    }
    if (profile == InputProfile::WindowBoundary) label += " window-boundary";

    int failures = verify_reduction(label.c_str(), from_device_bf16(d_out.data(), q_count),
                                    reference, kSwaBf16Criterion);
    failures += d_out.verify_guards((label + " output guards").c_str());
    failures += verify_exact((label + " q unchanged").c_str(),
                             from_device<std::uint16_t>(d_q, q_count), q_expected);
    failures +=
        verify_exact((label + " query k unchanged").c_str(),
                     from_device<std::uint16_t>(d_query_k, query_kv_count), query_k_expected);
    failures +=
        verify_exact((label + " query v unchanged").c_str(),
                     from_device<std::uint16_t>(d_query_v, query_kv_count), query_v_expected);
    failures += verify_exact((label + " positions unchanged").c_str(),
                             from_device<int>(d_positions, positions.size()), positions);
    failures +=
        verify_exact((label + " context k unchanged").c_str(),
                     from_device<std::uint16_t>(d_context_k, context_count), context_k_expected);
    failures +=
        verify_exact((label + " context v unchanged").c_str(),
                     from_device<std::uint16_t>(d_context_v, context_count), context_v_expected);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_batch_case() {
    constexpr int tokens                 = 2;
    constexpr int batch                  = 2;
    const std::size_t row_q_count        = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t row_kv_count       = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t lane_context_count = static_cast<std::size_t>(kD) * kWindow * kKVHeads;
    std::vector<float> q(row_q_count * batch);
    std::vector<float> query_k(row_kv_count * batch);
    std::vector<float> query_v(row_kv_count * batch);
    std::vector<float> context_k(lane_context_count * batch);
    std::vector<float> context_v(lane_context_count * batch);
    fill_uniform(q, 1709u, -0.35f, 0.35f);
    fill_uniform(query_k, 1801u, -0.4f, 0.4f);
    fill_uniform(query_v, 1907u, -0.8f, 0.8f);
    fill_uniform(context_k, 2003u, -0.4f, 0.4f);
    fill_uniform(context_v, 2111u, -0.8f, 0.8f);
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(context_k);
    round_to_bf16(context_v);

    const std::vector<int> positions{4096, 4097, 65, 65};
    const std::vector<std::int32_t> valid{2, 1};
    const std::vector<std::int32_t> lanes{1, 0};

    DeviceBuffer d_q         = to_device(bf16_bits(q));
    DeviceBuffer d_query_k   = to_device(bf16_bits(query_k));
    DeviceBuffer d_query_v   = to_device(bf16_bits(query_v));
    DeviceBuffer d_context_k = to_device(bf16_bits(context_k));
    DeviceBuffer d_context_v = to_device(bf16_bits(context_v));
    DeviceBuffer d_positions = to_device_i32(positions);
    DeviceBuffer d_valid     = to_device(valid);
    DeviceBuffer d_lanes     = to_device(lanes);
    GuardedDeviceBuffer d_out(row_q_count * batch * sizeof(std::uint16_t));
    d_out.fill(0x7f);

    Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens, batch});
    Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor positions_tensor(d_positions.p, DType::I32, {tokens, batch});
    Tensor valid_tensor(d_valid.p, DType::I32, {batch});
    Tensor lanes_tensor(d_lanes.p, DType::I32, {batch});
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens, batch});
    constexpr ops::SwaContextExecutionEnvelope envelope{0, 4096};
    DeviceArena workspace(ops::swa_workspace_capacity_bytes(envelope, tokens, tokens, batch));
    auto context = make_context_view(d_context_k, d_context_v, batch);

    std::vector<std::uint16_t> expected(row_q_count * batch);
    DeviceArena single_workspace(ops::swa_workspace_capacity_bytes(envelope, tokens, tokens, 1));
    for (int b = 0; b < batch; ++b) {
        GuardedDeviceBuffer single_out(row_q_count * sizeof(std::uint16_t));
        Tensor single_out_tensor(single_out.data(), DType::BF16, {kD, kQHeads, tokens, 1});
        Tensor q_row         = q_tensor.slice(3, b, 1);
        Tensor query_k_row   = query_k_tensor.slice(3, b, 1);
        Tensor query_v_row   = query_v_tensor.slice(3, b, 1);
        Tensor positions_row = positions_tensor.slice(1, b, 1);
        Tensor valid_row     = valid_tensor.slice(0, b, 1);
        Tensor lane_row      = lanes_tensor.slice(0, b, 1);
        ops::swa(q_row, query_k_row, query_v_row, positions_row, valid_row, lane_row, kScale,
                 context, envelope, single_workspace, single_out_tensor, nullptr);
        cuda_synchronize();
        const auto row = from_device<std::uint16_t>(single_out.data(), row_q_count);
        std::copy(row.begin(), row.end(),
                  expected.begin() + static_cast<std::ptrdiff_t>(b * row_q_count));
    }

    ops::swa(q_tensor, query_k_tensor, query_v_tensor, positions_tensor, valid_tensor, lanes_tensor,
             kScale, context, envelope, workspace, out_tensor, nullptr);
    cuda_synchronize();

    int failures =
        verify_exact("swa B=2 mixed lengths and lanes",
                     from_device<std::uint16_t>(d_out.data(), row_q_count * batch), expected);
    failures += d_out.verify_guards("swa B=2 output guards");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }

    int failures = 0;
    constexpr ops::SwaContextExecutionEnvelope capacity_envelope{0, 8194};
    const std::size_t interval = ops::swa_workspace_capacity_bytes(capacity_envelope, 1, 16, 1);
    const std::size_t endpoint = ops::swa_workspace_capacity_bytes(capacity_envelope, 16, 16, 1);
    if (interval != endpoint) {
        std::cerr << "swa interval capacity did not resolve to its monotonic endpoint\n";
        ++failures;
    }
    try {
        (void)ops::swa_workspace_capacity_bytes(capacity_envelope, 0, 16, 1);
        std::cerr << "swa accepted an invalid token interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += run_case(1, 0);
    failures += run_case(16, 1);
    failures += run_case(8, 96, InputProfile::Random, 4096);
    failures += run_case(16, 4096);
    failures += run_case(2, 4096, InputProfile::WindowBoundary);
    failures += run_case(2, 8194);
    failures += run_batch_case();

    if (failures != 0) {
        std::cerr << "swa failures=" << failures << '\n';
        return 1;
    }
    std::cout << "swa: PASS\n";
    return 0;
}
