#include "ninfer/ops/bidirectional_gqa_attention.h"

#include "core/arena.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
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
constexpr int kPage    = 64;
constexpr float kScale = 0.08838834764831844055f;

constexpr ReductionCriterion kBidirectionalGqaBf16Criterion{
    .relative_l2                     = 2.95e-3,
    .gross_absolute                  = 3e-4,
    .gross_relative_to_max_reference = 5.7e-3,
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

std::size_t context_index(int d, int kv_head, int position, int padded_context) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(kv_head));
}

enum class MappingPattern {
    Identity,
    Offset,
    Fragmented,
};

std::vector<std::int32_t> page_mapping(int logical_pages, MappingPattern pattern) {
    std::vector<std::int32_t> mapping(static_cast<std::size_t>(logical_pages));
    for (int page = 0; page < logical_pages; ++page) {
        switch (pattern) {
        case MappingPattern::Identity:
            mapping[static_cast<std::size_t>(page)] = page;
            break;
        case MappingPattern::Offset:
            mapping[static_cast<std::size_t>(page)] = logical_pages + page;
            break;
        case MappingPattern::Fragmented:
            mapping[static_cast<std::size_t>(page)] = 2 * page + 1;
            break;
        }
    }
    return mapping;
}

const char* mapping_name(MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return "identity";
    case MappingPattern::Offset:
        return "offset";
    case MappingPattern::Fragmented:
        return "fragmented";
    }
    return "unknown";
}

std::size_t physical_context_index(int d, int kv_head, int position, int physical_page,
                                   int physical_pages) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(position % kPage) +
                static_cast<std::size_t>(kPage) *
                    (static_cast<std::size_t>(physical_page) +
                     static_cast<std::size_t>(physical_pages) * static_cast<std::size_t>(kv_head)));
}

void scatter_context_mapping(std::vector<float>& physical, const std::vector<float>& logical,
                             int logical_stride, int storage_capacity,
                             const std::vector<std::int32_t>& mapping, int physical_pages) {
    for (int position = 0; position < storage_capacity; ++position) {
        const int physical_page = mapping[static_cast<std::size_t>(position / kPage)];
        for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
            for (int d = 0; d < kD; ++d) {
                physical[physical_context_index(d, kv_head, position, physical_page,
                                                physical_pages)] =
                    logical[context_index(d, kv_head, position, logical_stride)];
            }
        }
    }
}

std::vector<float> scatter_context(const std::vector<float>& logical, int logical_stride,
                                   int storage_capacity, int physical_pages,
                                   const std::vector<std::int32_t>& mapping, std::uint32_t seed) {
    std::vector<float> physical(static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages);
    fill_uniform(physical, seed, -0.9f, 0.9f);
    round_to_bf16(physical);
    scatter_context_mapping(physical, logical, logical_stride, storage_capacity, mapping,
                            physical_pages);
    return physical;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

void bidirectional_gqa_oracle(const std::vector<float>& q, const std::vector<float>& query_k,
                              const std::vector<float>& query_v,
                              const std::vector<float>& context_k,
                              const std::vector<float>& context_v, int tokens, int context_length,
                              int padded_context, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kD) * kQHeads * tokens, 0.0);
    const int key_count = context_length + tokens;
    std::vector<double> scores(static_cast<std::size_t>(key_count));

    for (int token = 0; token < tokens; ++token) {
        for (int q_head = 0; q_head < kQHeads; ++q_head) {
            const int kv_head = q_head / kGroup;
            double max_score  = -std::numeric_limits<double>::infinity();
            for (int key = 0; key < key_count; ++key) {
                double dot = 0.0;
                for (int d = 0; d < kD; ++d) {
                    const double k_value =
                        key < context_length
                            ? static_cast<double>(
                                  context_k[context_index(d, kv_head, key, padded_context)])
                            : static_cast<double>(
                                  query_k[query_kv_index(d, kv_head, key - context_length)]);
                    dot += static_cast<double>(q[q_index(d, q_head, token)]) * k_value;
                }
                const double score                    = dot * static_cast<double>(kScale);
                scores[static_cast<std::size_t>(key)] = score;
                max_score                             = std::max(max_score, score);
            }

            double denominator = 0.0;
            for (double& score : scores) {
                score = std::exp(score - max_score);
                denominator += score;
            }
            for (int d = 0; d < kD; ++d) {
                double numerator = 0.0;
                for (int key = 0; key < key_count; ++key) {
                    const double v_value =
                        key < context_length
                            ? static_cast<double>(
                                  context_v[context_index(d, kv_head, key, padded_context)])
                            : static_cast<double>(
                                  query_v[query_kv_index(d, kv_head, key - context_length)]);
                    numerator += scores[static_cast<std::size_t>(key)] * v_value;
                }
                out[q_index(d, q_head, token)] = numerator / denominator;
            }
        }
    }
}

PagedKVBatchLayerView make_context_view(DeviceBuffer& k, DeviceBuffer& v,
                                        DeviceBuffer& block_tables, int logical_pages,
                                        int physical_pages, int table_rows = 1) {
    return {
        .k_pages      = Tensor(k.p, DType::BF16, {kD, kPage, physical_pages, kKVHeads}),
        .v_pages      = Tensor(v.p, DType::BF16, {kD, kPage, physical_pages, kKVHeads}),
        .block_tables = Tensor(block_tables.p, DType::I32, {logical_pages, table_rows}),
        .head_dim     = kD,
        .num_kv_heads = kKVHeads,
        .dtype        = DType::BF16,
        .quant_group  = 0,
    };
}

enum class InputProfile {
    Random,
    QueryVisibility,
};

int run_case(int tokens, int context_length, InputProfile profile = InputProfile::Random,
             int envelope_max = -1, MappingPattern mapping_pattern = MappingPattern::Identity) {
    if (envelope_max < 0) envelope_max = context_length;
    const int max_context            = std::max({context_length, envelope_max, 1});
    const int logical_pages          = (max_context + kPage - 1) / kPage;
    const int logical_capacity       = logical_pages * kPage;
    const int logical_stride         = ((max_context + 127) / 128) * 128;
    const int physical_pages         = 2 * logical_pages + 1;
    const std::size_t q_count        = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t query_kv_count = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t logical_context_count =
        static_cast<std::size_t>(kD) * logical_stride * kKVHeads;
    const std::size_t physical_context_count =
        static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages;

    std::vector<float> q(q_count);
    std::vector<float> query_k(query_kv_count);
    std::vector<float> query_v(query_kv_count);
    std::vector<float> context_k(logical_context_count);
    std::vector<float> context_v(logical_context_count);

    const auto seed = static_cast<unsigned>(tokens * 131 + context_length * 17);
    fill_uniform(q, 1001u + seed, -0.35f, 0.35f);
    fill_uniform(query_k, 2003u + seed, -0.4f, 0.4f);
    fill_uniform(query_v, 3001u + seed, -0.8f, 0.8f);
    fill_uniform(context_k, 4001u + seed, -0.4f, 0.4f);
    fill_uniform(context_v, 5003u + seed, -0.8f, 0.8f);

    if (profile == InputProfile::QueryVisibility) {
        std::fill(q.begin(), q.end(), 0.0f);
        std::fill(query_k.begin(), query_k.end(), 0.0f);
        std::fill(query_v.begin(), query_v.end(), 0.0f);
        std::fill(context_k.begin(), context_k.end(), 0.0f);
        std::fill(context_v.begin(), context_v.end(), 0.0f);
        for (int kv_head = 0; kv_head < kKVHeads; ++kv_head) {
            for (int d = 0; d < kD; ++d) { query_v[query_kv_index(d, kv_head, tokens - 1)] = 1.0f; }
        }
    }

    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(context_k);
    round_to_bf16(context_v);

    std::vector<double> reference;
    bidirectional_gqa_oracle(q, query_k, query_v, context_k, context_v, tokens, context_length,
                             logical_stride, reference);

    std::vector<std::int32_t> mapping = page_mapping(logical_pages, mapping_pattern);
    if (context_length == 0 && envelope_max == 0) {
        mapping[0] = std::numeric_limits<std::int32_t>::max();
    }
    const std::vector<float> physical_k =
        context_length == 0 && envelope_max == 0
            ? std::vector<float>(static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages,
                                 0.25f)
            : scatter_context(context_k, logical_stride, logical_capacity, physical_pages, mapping,
                              6007u + seed);
    const std::vector<float> physical_v =
        context_length == 0 && envelope_max == 0
            ? std::vector<float>(static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages,
                                 -0.5f)
            : scatter_context(context_v, logical_stride, logical_capacity, physical_pages, mapping,
                              7001u + seed);

    const auto q_expected         = bf16_bits(q);
    const auto query_k_expected   = bf16_bits(query_k);
    const auto query_v_expected   = bf16_bits(query_v);
    const auto context_k_expected = bf16_bits(physical_k);
    const auto context_v_expected = bf16_bits(physical_v);
    const std::vector<int> length_expected{context_length};

    DeviceBuffer d_q         = to_device(q_expected);
    DeviceBuffer d_query_k   = to_device(query_k_expected);
    DeviceBuffer d_query_v   = to_device(query_v_expected);
    DeviceBuffer d_context_k = to_device(context_k_expected);
    DeviceBuffer d_context_v = to_device(context_v_expected);
    DeviceBuffer d_table     = to_device(mapping);
    DeviceBuffer d_length    = to_device_i32(length_expected);
    DeviceBuffer d_valid     = to_device_i32({tokens});
    DeviceBuffer d_table_row = to_device_i32({0});
    GuardedDeviceBuffer d_out(q_count * sizeof(std::uint16_t));
    d_out.fill(0x7f);

    Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens, 1});
    Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor length_tensor(d_length.p, DType::I32, {1});
    Tensor valid_tensor(d_valid.p, DType::I32, {1});
    Tensor table_row_tensor(d_table_row.p, DType::I32, {1});
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens, 1});
    PagedKVBatchLayerView context =
        make_context_view(d_context_k, d_context_v, d_table, logical_pages, physical_pages);
    const ops::GqaContextExecutionEnvelope envelope{0, static_cast<std::uint32_t>(envelope_max)};
    const std::size_t workspace_bytes =
        ops::bidirectional_gqa_attention_workspace_capacity_bytes(envelope, tokens, tokens, 1);
    DeviceArena workspace(workspace_bytes);

    ops::bidirectional_gqa_attention(q_tensor, query_k_tensor, query_v_tensor, length_tensor,
                                     valid_tensor, table_row_tensor, kScale, context, envelope,
                                     workspace, out_tensor, nullptr);
    cuda_synchronize();

    std::string label = "bidirectional_gqa_attention T=" + std::to_string(tokens) +
                        " L=" + std::to_string(context_length) +
                        " mapping=" + mapping_name(mapping_pattern);
    if (envelope_max != context_length) {
        label += " envelope=[0," + std::to_string(envelope_max) + "]";
    }
    if (profile == InputProfile::QueryVisibility) label += " query-visibility";

    int failures = verify_reduction(label.c_str(), from_device_bf16(d_out.data(), q_count),
                                    reference, kBidirectionalGqaBf16Criterion);
    failures += d_out.verify_guards((label + " output guards").c_str());
    failures += verify_exact((label + " q unchanged").c_str(),
                             from_device<std::uint16_t>(d_q, q_count), q_expected);
    failures +=
        verify_exact((label + " query k unchanged").c_str(),
                     from_device<std::uint16_t>(d_query_k, query_kv_count), query_k_expected);
    failures +=
        verify_exact((label + " query v unchanged").c_str(),
                     from_device<std::uint16_t>(d_query_v, query_kv_count), query_v_expected);
    failures += verify_exact((label + " context k unchanged").c_str(),
                             from_device<std::uint16_t>(d_context_k, physical_context_count),
                             context_k_expected);
    failures += verify_exact((label + " context v unchanged").c_str(),
                             from_device<std::uint16_t>(d_context_v, physical_context_count),
                             context_v_expected);
    failures += verify_exact((label + " block table unchanged").c_str(),
                             from_device<std::int32_t>(d_table, mapping.size()), mapping);
    failures += verify_exact((label + " context length unchanged").c_str(),
                             from_device<int>(d_length, 1), length_expected);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int graph_mapping_replay_case() {
    constexpr int tokens            = 1;
    constexpr int context_length    = 65;
    constexpr int logical_pages     = 2;
    constexpr int logical_capacity  = logical_pages * kPage;
    constexpr int physical_pages    = 6;
    constexpr int logical_stride    = 128;
    const std::size_t q_count       = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t query_count   = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t context_count = static_cast<std::size_t>(kD) * logical_stride * kKVHeads;
    const std::size_t physical_count =
        static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages;

    std::vector<float> q(q_count);
    std::vector<float> query_k(query_count);
    std::vector<float> query_v(query_count);
    std::vector<float> context_k(context_count);
    std::vector<float> context_v(context_count);
    fill_uniform(q, 11003u, -0.35f, 0.35f);
    fill_uniform(query_k, 12007u, -0.4f, 0.4f);
    fill_uniform(query_v, 13001u, -0.8f, 0.8f);
    fill_uniform(context_k, 14009u, -0.4f, 0.4f);
    fill_uniform(context_v, 15013u, -0.8f, 0.8f);
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(context_k);
    round_to_bf16(context_v);

    const std::array<std::array<std::int32_t, logical_pages>, 3> replay_mappings{
        std::array<std::int32_t, logical_pages>{0, 1},
        std::array<std::int32_t, logical_pages>{2, 3},
        std::array<std::int32_t, logical_pages>{5, 4},
    };
    std::vector<float> physical_k(physical_count);
    std::vector<float> physical_v(physical_count);
    for (const auto& replay_mapping : replay_mappings) {
        const std::vector<std::int32_t> mapping(replay_mapping.begin(), replay_mapping.end());
        scatter_context_mapping(physical_k, context_k, logical_stride, logical_capacity, mapping,
                                physical_pages);
        scatter_context_mapping(physical_v, context_v, logical_stride, logical_capacity, mapping,
                                physical_pages);
    }

    std::vector<double> reference;
    bidirectional_gqa_oracle(q, query_k, query_v, context_k, context_v, tokens, context_length,
                             logical_stride, reference);

    DeviceBuffer d_q         = to_device(bf16_bits(q));
    DeviceBuffer d_query_k   = to_device(bf16_bits(query_k));
    DeviceBuffer d_query_v   = to_device(bf16_bits(query_v));
    DeviceBuffer d_context_k = to_device(bf16_bits(physical_k));
    DeviceBuffer d_context_v = to_device(bf16_bits(physical_v));
    DeviceBuffer d_table     = to_device<std::int32_t>({0, 1});
    DeviceBuffer d_length    = to_device_i32({context_length});
    DeviceBuffer d_valid     = to_device_i32({tokens});
    DeviceBuffer d_table_row = to_device_i32({0});
    GuardedDeviceBuffer d_out(q_count * sizeof(std::uint16_t));
    Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens, 1});
    Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens, 1});
    Tensor length_tensor(d_length.p, DType::I32, {1});
    Tensor valid_tensor(d_valid.p, DType::I32, {1});
    Tensor table_row_tensor(d_table_row.p, DType::I32, {1});
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens, 1});
    PagedKVBatchLayerView context =
        make_context_view(d_context_k, d_context_v, d_table, logical_pages, physical_pages);
    constexpr ops::GqaContextExecutionEnvelope envelope{0, context_length};
    DeviceArena workspace(
        ops::bidirectional_gqa_attention_workspace_capacity_bytes(envelope, tokens, tokens, 1));

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create bidirectional GQA stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin bidirectional GQA capture");
    ops::bidirectional_gqa_attention(q_tensor, query_k_tensor, query_v_tensor, length_tensor,
                                     valid_tensor, table_row_tensor, kScale, context, envelope,
                                     workspace, out_tensor, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end bidirectional GQA capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "instantiate bidirectional GQA graph");

    int failures = 0;
    for (const auto& replay_mapping : replay_mappings) {
        d_table.copy_from_host(replay_mapping.data(), sizeof(replay_mapping));
        cuda_check(cudaGraphLaunch(executable, stream), "launch bidirectional GQA graph");
        cuda_synchronize(stream);
        const std::vector<std::int32_t> mapping(replay_mapping.begin(), replay_mapping.end());
        const std::string label =
            "bidirectional_gqa_attention graph mapping=" + std::to_string(mapping[0]) + "," +
            std::to_string(mapping[1]);
        failures += verify_reduction(label.c_str(), from_device_bf16(d_out.data(), q_count),
                                     reference, kBidirectionalGqaBf16Criterion);
        failures += verify_exact((label + " block table unchanged").c_str(),
                                 from_device<std::int32_t>(d_table, mapping.size()), mapping);
    }

    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    failures += d_out.verify_guards("bidirectional GQA graph output guards");
    return failures;
}

int batch_table_case() {
    constexpr int tokens            = 2;
    constexpr int batch             = 2;
    constexpr int logical_pages     = 2;
    constexpr int logical_capacity  = logical_pages * kPage;
    constexpr int physical_pages    = 6;
    const std::size_t row_q_count   = static_cast<std::size_t>(kD) * kQHeads * tokens;
    const std::size_t row_kv_count  = static_cast<std::size_t>(kD) * kKVHeads * tokens;
    const std::size_t logical_count = static_cast<std::size_t>(kD) * logical_capacity * kKVHeads;
    const std::size_t physical_count =
        static_cast<std::size_t>(kD) * kPage * kKVHeads * physical_pages;

    std::vector<float> q(row_q_count * batch);
    std::vector<float> query_k(row_kv_count * batch);
    std::vector<float> query_v(row_kv_count * batch);
    std::vector<float> logical_k(logical_count * batch);
    std::vector<float> logical_v(logical_count * batch);
    std::vector<float> physical_k(physical_count);
    std::vector<float> physical_v(physical_count);
    fill_uniform(q, 16001u, -0.35f, 0.35f);
    fill_uniform(query_k, 17011u, -0.4f, 0.4f);
    fill_uniform(query_v, 18013u, -0.8f, 0.8f);
    fill_uniform(logical_k, 19001u, -0.4f, 0.4f);
    fill_uniform(logical_v, 20011u, -0.8f, 0.8f);
    fill_uniform(physical_k, 21001u, -0.9f, 0.9f);
    fill_uniform(physical_v, 22003u, -0.9f, 0.9f);
    round_to_bf16(q);
    round_to_bf16(query_k);
    round_to_bf16(query_v);
    round_to_bf16(logical_k);
    round_to_bf16(logical_v);
    round_to_bf16(physical_k);
    round_to_bf16(physical_v);

    const std::vector<std::int32_t> block_tables{0, 2, 5, 3};
    for (int row = 0; row < batch; ++row) {
        const std::vector<float> row_k(
            logical_k.begin() + static_cast<std::ptrdiff_t>(row * logical_count),
            logical_k.begin() + static_cast<std::ptrdiff_t>((row + 1) * logical_count));
        const std::vector<float> row_v(
            logical_v.begin() + static_cast<std::ptrdiff_t>(row * logical_count),
            logical_v.begin() + static_cast<std::ptrdiff_t>((row + 1) * logical_count));
        const std::vector<std::int32_t> mapping(
            block_tables.begin() + static_cast<std::ptrdiff_t>(row * logical_pages),
            block_tables.begin() + static_cast<std::ptrdiff_t>((row + 1) * logical_pages));
        scatter_context_mapping(physical_k, row_k, logical_capacity, logical_capacity, mapping,
                                physical_pages);
        scatter_context_mapping(physical_v, row_v, logical_capacity, logical_capacity, mapping,
                                physical_pages);
    }

    const std::vector<std::int32_t> lengths{65, 1};
    const std::vector<std::int32_t> valid{2, 1};
    const std::vector<std::int32_t> table_rows{1, 0};
    DeviceBuffer d_q          = to_device(bf16_bits(q));
    DeviceBuffer d_query_k    = to_device(bf16_bits(query_k));
    DeviceBuffer d_query_v    = to_device(bf16_bits(query_v));
    DeviceBuffer d_context_k  = to_device(bf16_bits(physical_k));
    DeviceBuffer d_context_v  = to_device(bf16_bits(physical_v));
    DeviceBuffer d_tables     = to_device(block_tables);
    DeviceBuffer d_lengths    = to_device(lengths);
    DeviceBuffer d_valid      = to_device(valid);
    DeviceBuffer d_table_rows = to_device(table_rows);
    GuardedDeviceBuffer d_out(row_q_count * batch * sizeof(std::uint16_t));
    d_out.fill(0x7f);

    Tensor q_tensor(d_q.p, DType::BF16, {kD, kQHeads, tokens, batch});
    Tensor query_k_tensor(d_query_k.p, DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor query_v_tensor(d_query_v.p, DType::BF16, {kD, kKVHeads, tokens, batch});
    Tensor length_tensor(d_lengths.p, DType::I32, {batch});
    Tensor valid_tensor(d_valid.p, DType::I32, {batch});
    Tensor table_row_tensor(d_table_rows.p, DType::I32, {batch});
    Tensor out_tensor(d_out.data(), DType::BF16, {kD, kQHeads, tokens, batch});
    auto context =
        make_context_view(d_context_k, d_context_v, d_tables, logical_pages, physical_pages, batch);
    constexpr ops::GqaContextExecutionEnvelope envelope{0, 65};

    std::vector<std::uint16_t> expected(row_q_count * batch);
    DeviceArena single_workspace(
        ops::bidirectional_gqa_attention_workspace_capacity_bytes(envelope, tokens, tokens, 1));
    for (int b = 0; b < batch; ++b) {
        GuardedDeviceBuffer single_out(row_q_count * sizeof(std::uint16_t));
        Tensor single_out_tensor(single_out.data(), DType::BF16, {kD, kQHeads, tokens, 1});
        Tensor q_row       = q_tensor.slice(3, b, 1);
        Tensor query_k_row = query_k_tensor.slice(3, b, 1);
        Tensor query_v_row = query_v_tensor.slice(3, b, 1);
        Tensor length_row  = length_tensor.slice(0, b, 1);
        Tensor valid_row   = valid_tensor.slice(0, b, 1);
        Tensor table_row   = table_row_tensor.slice(0, b, 1);
        ops::bidirectional_gqa_attention(q_row, query_k_row, query_v_row, length_row, valid_row,
                                         table_row, kScale, context, envelope, single_workspace,
                                         single_out_tensor, nullptr);
        cuda_synchronize();
        const auto row = from_device<std::uint16_t>(single_out.data(), row_q_count);
        std::copy(row.begin(), row.end(),
                  expected.begin() + static_cast<std::ptrdiff_t>(b * row_q_count));
    }

    DeviceArena workspace(
        ops::bidirectional_gqa_attention_workspace_capacity_bytes(envelope, tokens, tokens, batch));
    ops::bidirectional_gqa_attention(q_tensor, query_k_tensor, query_v_tensor, length_tensor,
                                     valid_tensor, table_row_tensor, kScale, context, envelope,
                                     workspace, out_tensor, nullptr);
    cuda_synchronize();

    int failures =
        verify_exact("bidirectional_gqa_attention B=2 mixed lengths and table rows",
                     from_device<std::uint16_t>(d_out.data(), row_q_count * batch), expected);
    failures += d_out.verify_guards("bidirectional GQA B=2 output guards");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }

    int failures = 0;
    constexpr ops::GqaContextExecutionEnvelope capacity_envelope{0, 196609};
    const std::size_t interval =
        ops::bidirectional_gqa_attention_workspace_capacity_bytes(capacity_envelope, 1, 16, 1);
    const std::size_t witness = std::max(
        ops::bidirectional_gqa_attention_workspace_capacity_bytes(capacity_envelope, 8, 8, 1),
        ops::bidirectional_gqa_attention_workspace_capacity_bytes(capacity_envelope, 16, 16, 1));
    if (interval != witness) {
        std::cerr << "bidirectional GQA interval capacity missed a token-band endpoint\n";
        ++failures;
    }
    try {
        (void)ops::bidirectional_gqa_attention_workspace_capacity_bytes(capacity_envelope, 9, 8, 1);
        std::cerr << "bidirectional GQA accepted an invalid token interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += run_case(1, 0);
    failures += run_case(2, 1);
    failures += run_case(1, 63);
    failures += run_case(2, 64, InputProfile::Random, -1, MappingPattern::Offset);
    failures += run_case(4, 65, InputProfile::Random, -1, MappingPattern::Fragmented);
    failures += run_case(8, 95, InputProfile::Random, 4096, MappingPattern::Fragmented);
    failures += run_case(16, 257);
    failures += run_case(1, 4096, InputProfile::Random, -1, MappingPattern::Fragmented);
    failures += run_case(4, 0, InputProfile::QueryVisibility);
    failures += graph_mapping_replay_case();
    failures += batch_table_case();

    if (failures != 0) {
        std::cerr << "bidirectional_gqa_attention failures=" << failures << '\n';
        return 1;
    }
    std::cout << "bidirectional_gqa_attention: PASS\n";
    return 0;
}
