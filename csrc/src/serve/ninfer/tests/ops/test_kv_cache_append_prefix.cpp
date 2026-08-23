#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHeadDim       = 128;
constexpr int kKVHeads       = 8;
constexpr int kPage          = 64;
constexpr int kLogicalPages  = 3;
constexpr int kPhysicalPages = 6;
constexpr int kWindow        = 4096;

std::size_t input_index(int d, int head, int token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kKVHeads) * static_cast<std::size_t>(token));
}

std::size_t cyclic_cache_index(int d, int head, int slot) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(slot) +
                static_cast<std::size_t>(kWindow) * static_cast<std::size_t>(head));
}

std::size_t paged_cache_index(int d, int head, int position,
                              const std::vector<std::int32_t>& mapping) {
    const int physical_page = mapping.at(static_cast<std::size_t>(position / kPage));
    const int page_offset   = position % kPage;
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(page_offset) +
                static_cast<std::size_t>(kPage) *
                    (static_cast<std::size_t>(physical_page) +
                     static_cast<std::size_t>(kPhysicalPages) * static_cast<std::size_t>(head)));
}

std::vector<std::uint16_t> patterned_bits(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(count);
    std::uint32_t state = seed;
    for (auto& bit : bits) {
        state = state * 1664525u + 1013904223u;
        bit   = static_cast<std::uint16_t>(state >> 16);
    }
    return bits;
}

void append_oracle(std::vector<std::uint16_t>& cache_k, std::vector<std::uint16_t>& cache_v,
                   const std::vector<std::uint16_t>& input_k,
                   const std::vector<std::uint16_t>& input_v,
                   const std::vector<std::int32_t>& positions, int commit_count, bool cyclic,
                   const std::vector<std::int32_t>& mapping) {
    for (int token = 0; token < commit_count; ++token) {
        const int position = positions[static_cast<std::size_t>(token)];
        const int slot     = cyclic ? position % kWindow : 0;
        for (int head = 0; head < kKVHeads; ++head) {
            for (int d = 0; d < kHeadDim; ++d) {
                const auto src = input_index(d, head, token);
                const auto dst = cyclic ? cyclic_cache_index(d, head, slot)
                                        : paged_cache_index(d, head, position, mapping);
                cache_k[dst]   = input_k[src];
                cache_v[dst]   = input_v[src];
            }
        }
    }
}

PagedKVBatchLayerView paged_view(GuardedDeviceBuffer& k, GuardedDeviceBuffer& v,
                                 DeviceBuffer& block_table, int table_rows = 1) {
    return {
        .k_pages      = Tensor(k.data(), DType::BF16, {kHeadDim, kPage, kPhysicalPages, kKVHeads}),
        .v_pages      = Tensor(v.data(), DType::BF16, {kHeadDim, kPage, kPhysicalPages, kKVHeads}),
        .block_tables = Tensor(block_table.p, DType::I32, {kLogicalPages, table_rows}),
        .head_dim     = kHeadDim,
        .num_kv_heads = kKVHeads,
        .dtype        = DType::BF16,
        .quant_group  = 0,
    };
}

CyclicKVCacheLayerView cyclic_view(GuardedDeviceBuffer& k, GuardedDeviceBuffer& v,
                                   int lane_capacity = 1) {
    return {
        .k        = Tensor(k.data(), DType::BF16, {kHeadDim, kWindow, kKVHeads, lane_capacity}),
        .v        = Tensor(v.data(), DType::BF16, {kHeadDim, kWindow, kKVHeads, lane_capacity}),
        .capacity = kWindow,
        .padded_capacity = kWindow,
        .num_kv_heads    = kKVHeads,
        .head_dim        = kHeadDim,
        .lane_capacity   = lane_capacity,
    };
}

int run_case(int tokens, int commit_count, int first_position, bool cyclic,
             std::vector<std::int32_t> mapping = {}, int min_count = 0) {
    if (!cyclic && mapping.size() != kLogicalPages) {
        throw std::invalid_argument("paged prefix case requires a complete mapping");
    }
    const std::size_t input_count = static_cast<std::size_t>(kHeadDim) * kKVHeads * tokens;
    const std::size_t cache_count =
        static_cast<std::size_t>(kHeadDim) * kKVHeads * (cyclic ? kWindow : kPage * kPhysicalPages);
    const auto host_k = patterned_bits(input_count, 0x10203040u + static_cast<unsigned>(tokens));
    const auto host_v =
        patterned_bits(input_count, 0x50607080u + static_cast<unsigned>(commit_count));
    const auto initial_k = patterned_bits(cache_count, 0x90a0b0c0u);
    const auto initial_v = patterned_bits(cache_count, 0xd0e0f001u);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) {
        positions[static_cast<std::size_t>(i)] = first_position + i;
    }
    auto expected_k = initial_k;
    auto expected_v = initial_v;
    append_oracle(expected_k, expected_v, host_k, host_v, positions, commit_count, cyclic, mapping);

    DeviceBuffer d_k         = to_device(host_k);
    DeviceBuffer d_v         = to_device(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_count     = to_device<std::int32_t>({commit_count});
    DeviceBuffer d_selector  = to_device<std::int32_t>({0});
    DeviceBuffer d_table     = cyclic ? DeviceBuffer(1) : to_device(mapping);
    GuardedDeviceBuffer cache_k(cache_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(cache_count * sizeof(std::uint16_t));
    cache_k.copy_from_host(initial_k.data(), cache_k.bytes());
    cache_v.copy_from_host(initial_v.data(), cache_v.bytes());

    Tensor k(d_k.p, DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor v(d_v.p, DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor position_tensor(d_positions.p, DType::I32, {tokens, 1});
    Tensor count_tensor(d_count.p, DType::I32, {1});
    Tensor selector_tensor(d_selector.p, DType::I32, {1});
    const ops::KVCacheAppendPrefixExecutionEnvelope envelope{
        .min_count = static_cast<std::uint32_t>(min_count),
        .max_count = static_cast<std::uint32_t>(tokens),
    };
    if (cyclic) {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, selector_tensor, envelope,
                                    cyclic_view(cache_k, cache_v), nullptr);
    } else {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, selector_tensor, envelope,
                                    paged_view(cache_k, cache_v, d_table), nullptr);
    }
    cuda_synchronize();

    const std::string label = std::string("kv_cache_append_prefix ") +
                              (cyclic ? "cyclic" : "paged") + " T=" + std::to_string(tokens) +
                              " C=" + std::to_string(commit_count) +
                              " min=" + std::to_string(min_count);
    int failures =
        verify_exact((label + " cache k").c_str(),
                     from_device<std::uint16_t>(cache_k.data(), cache_count), expected_k);
    failures += verify_exact((label + " cache v").c_str(),
                             from_device<std::uint16_t>(cache_v.data(), cache_count), expected_v);
    failures += verify_exact((label + " input k unchanged").c_str(),
                             from_device<std::uint16_t>(d_k, input_count), host_k);
    failures += verify_exact((label + " input v unchanged").c_str(),
                             from_device<std::uint16_t>(d_v, input_count), host_v);
    failures += verify_exact((label + " positions unchanged").c_str(),
                             from_device<std::int32_t>(d_positions, positions.size()), positions);
    failures += verify_exact((label + " count unchanged").c_str(),
                             from_device<std::int32_t>(d_count, 1), {commit_count});
    failures += cache_k.verify_guards((label + " cache k guards").c_str());
    failures += cache_v.verify_guards((label + " cache v guards").c_str());
    return failures;
}

int cyclic_graph_replay_case() {
    constexpr int tokens          = 16;
    constexpr int first_position  = 2 * kWindow - 4;
    const std::size_t input_count = static_cast<std::size_t>(kHeadDim) * kKVHeads * tokens;
    const std::size_t cache_count = static_cast<std::size_t>(kHeadDim) * kWindow * kKVHeads;
    const auto host_k             = patterned_bits(input_count, 0x11223344u);
    const auto host_v             = patterned_bits(input_count, 0x55667788u);
    const auto initial_k          = patterned_bits(cache_count, 0x99aabbccu);
    const auto initial_v          = patterned_bits(cache_count, 0xddeeff01u);
    std::vector<std::int32_t> positions(tokens);
    for (int i = 0; i < tokens; ++i) positions[static_cast<std::size_t>(i)] = first_position + i;

    DeviceBuffer d_k         = to_device(host_k);
    DeviceBuffer d_v         = to_device(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_count     = to_device<std::int32_t>({0});
    DeviceBuffer d_lane      = to_device<std::int32_t>({0});
    GuardedDeviceBuffer cache_k(cache_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(cache_count * sizeof(std::uint16_t));
    Tensor k(d_k.p, DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor v(d_v.p, DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor position_tensor(d_positions.p, DType::I32, {tokens, 1});
    Tensor count_tensor(d_count.p, DType::I32, {1});
    Tensor lane_tensor(d_lane.p, DType::I32, {1});
    auto cache = cyclic_view(cache_k, cache_v);

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create kv append stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin kv append capture");
    ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, lane_tensor, {0, tokens},
                                cache, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end kv append capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "instantiate kv append graph");

    int failures = 0;
    for (const int commit_count : std::array{0, 7, tokens}) {
        cache_k.copy_from_host(initial_k.data(), cache_k.bytes());
        cache_v.copy_from_host(initial_v.data(), cache_v.bytes());
        d_count.copy_from_host(&commit_count, sizeof(commit_count));
        cuda_check(cudaGraphLaunch(executable, stream), "launch kv append graph");
        cuda_synchronize(stream);

        auto expected_k = initial_k;
        auto expected_v = initial_v;
        append_oracle(expected_k, expected_v, host_k, host_v, positions, commit_count, true, {});
        const std::string label =
            "kv_cache_append_prefix cyclic graph C=" + std::to_string(commit_count);
        failures +=
            verify_exact((label + " cache k").c_str(),
                         from_device<std::uint16_t>(cache_k.data(), cache_count), expected_k);
        failures +=
            verify_exact((label + " cache v").c_str(),
                         from_device<std::uint16_t>(cache_v.data(), cache_count), expected_v);
        failures += verify_exact((label + " count unchanged").c_str(),
                                 from_device<std::int32_t>(d_count, 1), {commit_count});
    }

    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    failures += verify_exact("kv append graph input k unchanged",
                             from_device<std::uint16_t>(d_k, input_count), host_k);
    failures += verify_exact("kv append graph input v unchanged",
                             from_device<std::uint16_t>(d_v, input_count), host_v);
    failures += verify_exact("kv append graph positions unchanged",
                             from_device<std::int32_t>(d_positions, positions.size()), positions);
    failures += cache_k.verify_guards("kv append graph cache k guards");
    failures += cache_v.verify_guards("kv append graph cache v guards");
    return failures;
}

int paged_graph_replay_case() {
    constexpr int tokens          = 16;
    constexpr int first_position  = 60;
    const std::size_t input_count = static_cast<std::size_t>(kHeadDim) * kKVHeads * tokens;
    const std::size_t cache_count =
        static_cast<std::size_t>(kHeadDim) * kPage * kKVHeads * kPhysicalPages;
    const auto host_k    = patterned_bits(input_count, 0x12345678u);
    const auto host_v    = patterned_bits(input_count, 0x87654321u);
    const auto initial_k = patterned_bits(cache_count, 0xabcdef01u);
    const auto initial_v = patterned_bits(cache_count, 0x10fedcbau);
    std::vector<std::int32_t> positions(tokens);
    for (int i = 0; i < tokens; ++i) positions[static_cast<std::size_t>(i)] = first_position + i;

    DeviceBuffer d_k         = to_device(host_k);
    DeviceBuffer d_v         = to_device(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_count     = to_device<std::int32_t>({0});
    DeviceBuffer d_row       = to_device<std::int32_t>({0});
    DeviceBuffer d_table     = to_device<std::int32_t>({0, 1, 2});
    GuardedDeviceBuffer cache_k(cache_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(cache_count * sizeof(std::uint16_t));
    Tensor k(d_k.p, DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor v(d_v.p, DType::BF16, {kHeadDim, kKVHeads, tokens, 1});
    Tensor position_tensor(d_positions.p, DType::I32, {tokens, 1});
    Tensor count_tensor(d_count.p, DType::I32, {1});
    Tensor row_tensor(d_row.p, DType::I32, {1});
    auto cache = paged_view(cache_k, cache_v, d_table);

    cudaStream_t stream        = nullptr;
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create paged kv append stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "begin paged kv append capture");
    ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, row_tensor, {0, tokens}, cache,
                                stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end paged kv append capture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
               "instantiate paged kv append graph");

    const std::array<int, 3> counts{0, 7, tokens};
    const std::array<std::array<std::int32_t, kLogicalPages>, 3> mappings{
        std::array<std::int32_t, kLogicalPages>{0, 1, 2},
        std::array<std::int32_t, kLogicalPages>{2, 3, 4},
        std::array<std::int32_t, kLogicalPages>{5, 1, 4},
    };
    int failures = 0;
    for (std::size_t replay = 0; replay < counts.size(); ++replay) {
        const int commit_count = counts[replay];
        const std::vector<std::int32_t> mapping(mappings[replay].begin(), mappings[replay].end());
        cache_k.copy_from_host(initial_k.data(), cache_k.bytes());
        cache_v.copy_from_host(initial_v.data(), cache_v.bytes());
        d_count.copy_from_host(&commit_count, sizeof(commit_count));
        d_table.copy_from_host(mapping.data(), mapping.size() * sizeof(std::int32_t));
        cuda_check(cudaGraphLaunch(executable, stream), "launch paged kv append graph");
        cuda_synchronize(stream);

        auto expected_k = initial_k;
        auto expected_v = initial_v;
        append_oracle(expected_k, expected_v, host_k, host_v, positions, commit_count, false,
                      mapping);
        const std::string label =
            "kv_cache_append_prefix paged graph C=" + std::to_string(commit_count);
        failures +=
            verify_exact((label + " cache k").c_str(),
                         from_device<std::uint16_t>(cache_k.data(), cache_count), expected_k);
        failures +=
            verify_exact((label + " cache v").c_str(),
                         from_device<std::uint16_t>(cache_v.data(), cache_count), expected_v);
        failures += verify_exact((label + " block table unchanged").c_str(),
                                 from_device<std::int32_t>(d_table, mapping.size()), mapping);
    }

    cudaGraphExecDestroy(executable);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    failures += cache_k.verify_guards("paged kv append graph cache k guards");
    failures += cache_v.verify_guards("paged kv append graph cache v guards");
    return failures;
}

int batch_selector_case(bool cyclic) {
    constexpr int tokens = 3;
    constexpr int batch  = 2;
    const std::vector<std::int32_t> counts{1, 3};
    const std::vector<std::int32_t> selectors{1, 0};
    const std::vector<std::int32_t> positions =
        cyclic ? std::vector<std::int32_t>{kWindow - 1, kWindow, kWindow + 1, 5, 6, 7}
               : std::vector<std::int32_t>{63, 64, 65, 5, 6, 7};
    const std::size_t row_input_count = static_cast<std::size_t>(kHeadDim) * kKVHeads * tokens;
    const std::size_t lane_cache_count =
        static_cast<std::size_t>(kHeadDim) * kKVHeads * (cyclic ? kWindow : kPage * kPhysicalPages);
    const auto host_k    = patterned_bits(row_input_count * batch, 0x31415926u);
    const auto host_v    = patterned_bits(row_input_count * batch, 0x27182818u);
    const auto initial_k = patterned_bits(lane_cache_count * (cyclic ? batch : 1), 0x16180339u);
    const auto initial_v = patterned_bits(lane_cache_count * (cyclic ? batch : 1), 0x57721566u);
    const std::vector<std::int32_t> tables{0, 1, 2, 3, 4, 5};
    auto expected_k = initial_k;
    auto expected_v = initial_v;

    for (int b = 0; b < batch; ++b) {
        const std::vector<std::int32_t> mapping(
            tables.begin() +
                static_cast<std::ptrdiff_t>(selectors[static_cast<std::size_t>(b)] * kLogicalPages),
            tables.begin() + static_cast<std::ptrdiff_t>(
                                 (selectors[static_cast<std::size_t>(b)] + 1) * kLogicalPages));
        for (int token = 0; token < counts[static_cast<std::size_t>(b)]; ++token) {
            const int position = positions[static_cast<std::size_t>(b * tokens + token)];
            for (int head = 0; head < kKVHeads; ++head) {
                for (int d = 0; d < kHeadDim; ++d) {
                    const std::size_t src =
                        static_cast<std::size_t>(b) * row_input_count + input_index(d, head, token);
                    const std::size_t dst =
                        cyclic ? static_cast<std::size_t>(selectors[static_cast<std::size_t>(b)]) *
                                         lane_cache_count +
                                     cyclic_cache_index(d, head, position % kWindow)
                               : paged_cache_index(d, head, position, mapping);
                    expected_k[dst] = host_k[src];
                    expected_v[dst] = host_v[src];
                }
            }
        }
    }

    DeviceBuffer d_k         = to_device(host_k);
    DeviceBuffer d_v         = to_device(host_v);
    DeviceBuffer d_positions = to_device(positions);
    DeviceBuffer d_counts    = to_device(counts);
    DeviceBuffer d_selectors = to_device(selectors);
    DeviceBuffer d_tables    = cyclic ? DeviceBuffer(1) : to_device(tables);
    GuardedDeviceBuffer cache_k(initial_k.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer cache_v(initial_v.size() * sizeof(std::uint16_t));
    cache_k.copy_from_host(initial_k.data(), cache_k.bytes());
    cache_v.copy_from_host(initial_v.data(), cache_v.bytes());

    Tensor k(d_k.p, DType::BF16, {kHeadDim, kKVHeads, tokens, batch});
    Tensor v(d_v.p, DType::BF16, {kHeadDim, kKVHeads, tokens, batch});
    Tensor position_tensor(d_positions.p, DType::I32, {tokens, batch});
    Tensor count_tensor(d_counts.p, DType::I32, {batch});
    Tensor selector_tensor(d_selectors.p, DType::I32, {batch});
    constexpr ops::KVCacheAppendPrefixExecutionEnvelope envelope{0, tokens};
    if (cyclic) {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, selector_tensor, envelope,
                                    cyclic_view(cache_k, cache_v, batch), nullptr);
    } else {
        ops::kv_cache_append_prefix(k, v, position_tensor, count_tensor, selector_tensor, envelope,
                                    paged_view(cache_k, cache_v, d_tables, batch), nullptr);
    }
    cuda_synchronize();

    const std::string label =
        std::string("kv_cache_append_prefix B=2 ") + (cyclic ? "cyclic lanes" : "paged rows");
    int failures =
        verify_exact((label + " k").c_str(),
                     from_device<std::uint16_t>(cache_k.data(), expected_k.size()), expected_k);
    failures +=
        verify_exact((label + " v").c_str(),
                     from_device<std::uint16_t>(cache_v.data(), expected_v.size()), expected_v);
    failures += cache_k.verify_guards((label + " k guards").c_str());
    failures += cache_v.verify_guards((label + " v guards").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "kv_cache_append_prefix: SKIP (CUDA unavailable)\n";
        return 77;
    }

    int failures = 0;
    failures += run_case(1, 0, 0, false, {0, 1, 2});
    failures += run_case(1, 1, 63, false, {2, 3, 4});
    failures += run_case(16, 7, 60, false, {5, 1, 4}, 5);
    failures += run_case(16, 16, 120, false, {2, 5, 0});
    failures += run_case(1, 0, kWindow - 1, true);
    failures += run_case(1, 1, 2 * kWindow - 1, true);
    failures += run_case(16, 7, 2 * kWindow - 2, true);
    failures += run_case(16, 16, 3 * kWindow - 8, true, {}, 16);
    failures += cyclic_graph_replay_case();
    failures += paged_graph_replay_case();
    failures += batch_selector_case(true);
    failures += batch_selector_case(false);

    if (failures != 0) {
        std::cerr << "kv_cache_append_prefix failures=" << failures << '\n';
        return 1;
    }
    std::cout << "kv_cache_append_prefix: PASS\n";
    return 0;
}
