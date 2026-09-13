// Independent FP64 oracle for GLM's learned pooling and signed head scoring.
#include "api/ops/glm_indexer.h"
#include "api/ops/qsa_indexer.h"
#include "ops/op_tester.h"
#include <numeric>
using namespace sinfer;
using namespace sinfer::test;
namespace {
void expect(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void exercise(int dim, int heads, int pool, int budget, int length) {
    const ops::GlmIndexerGeometry g{dim, heads, pool, budget, 1e-6F};
    const int sequences = 2, pages = (length + 63) / 64, count = length * sequences;
    std::vector<int> tables(pages * sequences), rows{1, 0}, positions(count);
    std::iota(tables.rbegin(), tables.rend(), 0);
    for (int t = 0; t < count; ++t) positions[t] = t % length;
    std::vector<float> raw(count * dim), gates(count * dim), gain(dim), bias(dim), ape(pool * dim);
    fill_uniform(raw, 19, -2, 2); fill_uniform(gates, 29, -3, 3);
    fill_uniform(gain, 39, .3, 1.2); fill_uniform(bias, 49, -.2, .4); fill_uniform(ape, 59, -2, 2);
    auto dk = to_device(raw), dg = to_device(gates), dn = to_device(gain), db = to_device(bias), da = to_device(ape);
    auto dp = to_device(positions), dt = to_device(tables), dr = to_device(rows);
    DeviceBuffer cache_bytes(std::size_t(pages) * sequences * 64 * 3 * dim * sizeof(float));
    cuda_check(cudaMemset(cache_bytes.p, 0, cache_bytes.bytes), "clear cache");
    PagedKVBatchLayerView cache;
    cache.indexer_pages = Tensor(cache_bytes.p, DType::FP32, {3 * dim, 64, 1, pages * sequences});
    cache.block_tables = Tensor(dt.p, DType::I32, {pages, sequences});
    const auto at = [&](int seq, int pos) {
        return (std::size_t(tables[rows[seq] * pages + pos / 64]) * 64 + pos % 64) * 3 * dim;
    };
    Tensor key(dk.p, DType::FP32, {dim, count}), gate(dg.p, DType::FP32, {dim, count});
    Tensor norm(dn.p, DType::FP32, {dim}), offset(db.p, DType::FP32, {dim}), positional(da.p, DType::FP32, {dim, pool});
    Tensor pos(dp.p, DType::I32, {count}), row(dr.p, DType::I32, {sequences});
    auto append = [&](int first, int n, int seq) {
        ops::glm_indexer_append(key.slice(1, seq * length + first, n),
            gate.slice(1, seq * length + first, n), norm, offset, positional,
            pos.slice(0, seq * length + first, n), row.slice(0, seq, 1), Tensor{}, n, g, cache, nullptr);
    };
    // Chunks cut through pooling groups and physical page boundaries.
    for (int seq = 0; seq < sequences; ++seq) {
        append(0, 61, seq);
        for (int t = 61; t < 69; ++t) append(t, 1, seq);
        append(69, length - 69, seq);
    }
    auto reference = [&]() {
        std::vector<double> normalized(count * dim), pooled(sequences * (length / pool) * dim);
        for (int t = 0; t < count; ++t) {
            double mean = 0, var = 0;
            for (int d = 0; d < dim; ++d) mean += raw[t * dim + d];
            mean /= dim;
            for (int d = 0; d < dim; ++d) var += std::pow(raw[t * dim + d] - mean, 2);
            for (int d = 0; d < dim; ++d)
                normalized[t * dim + d] = (raw[t * dim + d] - mean) / std::sqrt(var / dim + g.norm_epsilon) * gain[d] + bias[d];
        }
        for (int seq = 0; seq < sequences; ++seq) for (int b = 0; b < length / pool; ++b) {
            for (int d = 0; d < dim; ++d) {
                double sum = 0, weight = 0;
                for (int r = 0; r < pool; ++r) {
                    const int t = seq * length + b * pool + r;
                    const double w = std::exp(double(gates[t * dim + d]) + ape[r * dim + d]);
                    sum += w * normalized[t * dim + d]; weight += w;
                }
                pooled[(seq * (length / pool) + b) * dim + d] = sum / weight;
            }
        }
        const auto actual = from_device<float>(cache_bytes, cache_bytes.bytes / 4);
        for (int seq = 0; seq < sequences; ++seq) {
            for (int t = 0; t < length; ++t) for (int d = 0; d < dim; ++d) {
                expect(std::abs(actual[at(seq, t) + d] - normalized[(seq * length + t) * dim + d]) < 2e-6,
                    "raw normalized key changed on pool completion or replay");
                expect(actual[at(seq, t) + dim + d] == gates[(seq * length + t) * dim + d], "raw gate changed");
            }
            for (int b = 0; b < length / pool; ++b) for (int d = 0; d < dim; ++d)
                expect(std::abs(actual[at(seq, b * pool) + 2 * dim + d] - pooled[(seq * (length / pool) + b) * dim + d]) < 2e-6,
                    "learned pooled key differs from FP64 reference");
        }
        return pooled;
    };
    auto pooled = reference();
    // Invalid padded positions must never address the page table, including during append.
    auto padded_positions = positions;
    for (int t = count - 3; t < count; ++t) padded_positions[t] = -1;
    auto padded = to_device(padded_positions), append_valid = to_device(std::vector<int>{length, length - 3});
    Tensor padded_pos(padded.p, DType::I32, {count}), append_valid_t(append_valid.p, DType::I32, {2});
    ops::glm_indexer_append(key, gate, norm, offset, positional, padded_pos, row, append_valid_t,
                           length, g, cache, nullptr);
    pooled = reference();
    // Replace a suffix beginning INSIDE an already completed pool. Earlier raw keys must survive.
    const int rewind = (length / pool - 1) * pool + 1;
    for (int seq = 0; seq < sequences; ++seq) for (int t = rewind; t < length; ++t)
        for (int d = 0; d < dim; ++d) { raw[(seq * length + t) * dim + d] *= -.7F; gates[(seq * length + t) * dim + d] += .8F; }
    dk.copy_from_host(raw.data(), dk.bytes); dg.copy_from_host(gates.data(), dg.bytes);
    for (int seq = 0; seq < sequences; ++seq) append(rewind, length - rewind, seq);
    pooled = reference();

    // Two lanes, ragged padding, earlier causal queries, exact ties, and all-negative scores.
    const int width = 5, queries = width * sequences;
    std::vector<float> q(queries * heads * dim), weights(queries * heads);
    fill_uniform(q, 79, -1, 1); fill_uniform(weights, 89, -2, 1);
    for (int h = 0; h < heads; ++h) { weights[h] = 0; weights[heads + h] = -std::abs(weights[heads + h]); }
    std::vector<int> qp{length - 1, length - 2, budget - 1, budget + pool - 2, length - 3,
                        length - 1, length - 2, budget + pool - 1, -1, -1}, valid{5, 3};
    auto dq = to_device(q), dw = to_device(weights), dqp = to_device(qp), dv = to_device(valid);
    Tensor qt(dq.p, DType::FP32, {dim, heads, queries}), wt(dw.p, DType::FP32, {heads, queries});
    Tensor qpos(dqp.p, DType::I32, {queries}), vt(dv.p, DType::I32, {sequences});
    const int words = ops::qsa_block_mask_words(length, pool);
    DeviceBuffer dm(words * queries * 4);
    Tensor mask(dm.p, DType::I32, {words, queries});
    WorkspaceArena workspace(ops::glm_indexer_select_workspace_capacity_bytes(queries, length, g));
    cudaStream_t stream; cuda_check(cudaStreamCreate(&stream), "create stream");
    auto run = [&]() { ops::glm_indexer_select(qt, wt, qpos, row, vt, width, g, cache, length, workspace, mask, stream); };
    run(); cuda_synchronize(stream);
    cudaGraph_t graph; cudaGraphExec_t exec;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture"); run();
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
    cuda_check(cudaGraphLaunch(exec, stream), "replay"); cuda_synchronize(stream);
    const auto got = from_device<unsigned>(dm, words * queries);
    for (int r = 0; r < queries; ++r) {
        std::vector<bool> wanted(words * 32, false);
        const int seq = r / width;
        if (r % width < valid[seq]) {
            int complete = (qp[r] + 1) / pool;
            std::vector<std::pair<double, int>> scores;
            for (int b = 0; b < complete; ++b) {
                double score = 0;
                for (int h = 0; h < heads; ++h) {
                    double dot = 0;
                    for (int d = 0; d < dim; ++d) dot += q[(r * heads + h) * dim + d] * pooled[(seq * (length / pool) + b) * dim + d];
                    score += std::max(dot, 0.) * weights[r * heads + h] / std::sqrt(double(dim * heads));
                }
                scores.emplace_back(-score, b);
            }
            std::sort(scores.begin(), scores.end());
            for (int b = 0; b < std::min(complete, budget / pool); ++b) wanted[scores[b].second] = true;
            if ((qp[r] + 1) % pool) wanted[complete] = true;
        }
        for (int b = 0; b < words * 32; ++b)
            expect(bool((got[r * words + b / 32] >> (b % 32)) & 1) == wanted[b], "pool selection differs from causal FP64 reference");
    }
    cuda_check(cudaGraphExecDestroy(exec), "destroy graph exec"); cuda_check(cudaGraphDestroy(graph), "destroy graph");
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
}
void tiled_selection() {
    const int length = 4096, queries = 17000, dim = 32, words = 32;
    const ops::GlmIndexerGeometry g{dim, 1, 4, 32, 1e-6F};
    std::vector<int> tables(64); std::iota(tables.begin(), tables.end(), 0);
    auto dt = to_device(tables), dp = to_device(std::vector<int>(queries, length - 1)),
         dr = to_device(std::vector<int>(queries, 0));
    auto dq = to_device(std::vector<float>(dim * queries, 0)),
         dw = to_device(std::vector<float>(queries, 0)),
         dc = to_device(std::vector<float>(3 * dim * length, 0));
    DeviceBuffer dm(words * queries * 4);
    PagedKVBatchLayerView cache;
    cache.indexer_pages = Tensor(dc.p, DType::FP32, {3 * dim, 64, 1, 64});
    cache.block_tables = Tensor(dt.p, DType::I32, {64, 1});
    Tensor q(dq.p, DType::FP32, {dim, 1, queries}), weights(dw.p, DType::FP32, {1, queries}),
           positions(dp.p, DType::I32, {queries}), rows(dr.p, DType::I32, {queries}),
           mask(dm.p, DType::I32, {words, queries});
    WorkspaceArena workspace(ops::glm_indexer_select_workspace_capacity_bytes(queries, length, g));
    ops::glm_indexer_select(q, weights, positions, rows, Tensor{}, 1, g, cache, length, workspace, mask, nullptr);
    const auto result = from_device<unsigned>(dm, words * queries);
    for (int r = 0; r < queries; ++r) for (int w = 0; w < words; ++w)
            expect(result[r * words + w] == (w == 0 ? 255U : 0U), "selection scratch tile boundary changed ties");
}
void decode_history() {
    // Decode's parallel history scoring must produce exactly the prefill path's masks,
    // including when a captured 1M-token profile replays at a much shorter position.
    constexpr int dim = 128, heads = 32, keys = 1048576, length = 32768, reference_rows = 33;
    const ops::GlmIndexerGeometry g{dim, heads, 4, 2048, 1e-6F};
    const int words = ops::qsa_block_mask_words(keys, g.block);
    std::vector<int> tables(keys / 64, -1);
    for (int p = 0; p < length / 64; ++p) { tables[p] = (p * 17) % (length / 64); }
    std::vector<float> cache(length * 3 * dim), q(dim * heads * reference_rows), weights(heads * reference_rows);
    fill_uniform(cache, 119, -1, 1);
    std::vector<float> query(dim * heads), signed_weights(heads);
    fill_uniform(query, 129, -1, 1); fill_uniform(signed_weights, 139, -1, 1);
    for (int r = 0; r < reference_rows; ++r) {
        std::copy(query.begin(), query.end(), q.begin() + r * dim * heads);
    }
    auto dc = to_device(cache), dt = to_device(tables), dq = to_device(q), dw = to_device(weights),
         dp = to_device(std::vector<int>(reference_rows, 0)),
         dr = to_device(std::vector<int>(reference_rows, 0)),
         dv = to_device(std::vector<int>(reference_rows, 1));
    DeviceBuffer decoded(words * 4), reference(words * reference_rows * 4);
    PagedKVBatchLayerView kv;
    kv.indexer_pages = Tensor(dc.p, DType::FP32, {3 * dim, 64, 1, length / 64});
    kv.block_tables = Tensor(dt.p, DType::I32, {keys / 64, 1});
    Tensor qt(dq.p, DType::FP32, {dim, heads, reference_rows}), wt(dw.p, DType::FP32, {heads, reference_rows}),
           pos(dp.p, DType::I32, {reference_rows}), rows(dr.p, DType::I32, {reference_rows}),
           valid(dv.p, DType::I32, {reference_rows}), mask(decoded.p, DType::I32, {words, 1}),
           expected(reference.p, DType::I32, {words, reference_rows});
    WorkspaceArena workspace(ops::glm_indexer_select_workspace_capacity_bytes(reference_rows, keys, g));
    cudaStream_t stream; cuda_check(cudaStreamCreate(&stream), "decode stream");
    auto run = [&] {
        ops::glm_indexer_select(qt.slice(2, 0, 1), wt.slice(1, 0, 1), pos.slice(0, 0, 1),
            rows.slice(0, 0, 1), valid.slice(0, 0, 1), 1, g, kv, keys, workspace, mask, stream);
    };
    run(); cuda_synchronize(stream);
    cudaGraph_t graph; cudaGraphExec_t exec;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "decode capture"); run();
    cuda_check(cudaStreamEndCapture(stream, &graph), "end decode capture");
    cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate decode");
    for (int mode = 0; mode < 3; ++mode) {
        for (int r = 0; r < reference_rows; ++r) for (int h = 0; h < heads; ++h) {
            weights[r * heads + h] = mode == 0 ? signed_weights[h]
                : mode == 1 ? 0.F : -std::abs(signed_weights[h]);
        }
        dw.copy_from_host(weights.data(), dw.bytes);
        for (int visible : {127, 2048, 2051, 21003, 23100, length, 0, 4099}) {
            std::vector<int> positions(reference_rows, visible - 1), validity(reference_rows, visible != 0);
            dp.copy_from_host(positions.data(), dp.bytes); dv.copy_from_host(validity.data(), dv.bytes);
            cuda_check(cudaGraphLaunch(exec, stream), "decode replay");
            ops::glm_indexer_select(qt, wt, pos, rows, valid, 1, g, kv, keys, workspace, expected, stream);
            cuda_synchronize(stream);
            const auto got = from_device<unsigned>(decoded, words);
            const auto ref = from_device<unsigned>(reference, words * reference_rows);
            expect(std::equal(got.begin(), got.end(), ref.begin()), "decode and prefill pool masks differ");
            unsigned selected = 0;
            for (unsigned word : got) { selected += __builtin_popcount(word); }
            expect(selected == std::min(visible / g.block, g.top_k / g.block) + (visible % g.block != 0),
                   "decode changed selection budget or incomplete tail");
            if (mode == 1) {
                for (int b = 0; b < words * 32; ++b) {
                    const bool wanted = b < std::min(visible / g.block, g.top_k / g.block) ||
                        (visible % g.block && b == visible / g.block);
                    expect(bool((got[b / 32] >> (b % 32)) & 1) == wanted, "decode changed exact-tie ordering");
                }
            }
        }
    }
    cuda_check(cudaGraphExecDestroy(exec), "destroy decode exec");
    cuda_check(cudaGraphDestroy(graph), "destroy decode graph");
    cuda_check(cudaStreamDestroy(stream), "destroy decode stream");
}
void projection(int k = 96, int n = 39, int tokens = 7) {
    std::vector<float> w(k * n), x(k * tokens);
    fill_uniform(w, 4, -1, 1); fill_uniform(x, 5, -2, 2); round_to_bf16(x);
    auto dw = to_device(w), dx = to_device_bf16(x); DeviceBuffer dout(n * tokens * 4);
    Tensor wt(dw.p, DType::FP32, {k, n}), xt(dx.p, DType::BF16, {k, tokens}), out(dout.p, DType::FP32, {n, tokens});
    ops::glm_indexer_project(wt, xt, out, nullptr);
    const auto actual = from_device<float>(dout, n * tokens);
    for (int t = 0; t < tokens; ++t) for (int r = 0; r < n; ++r) {
        double expected = 0; for (int c = 0; c < k; ++c) expected += double(w[r * k + c]) * x[t * k + c];
        expect(std::abs(actual[t * n + r] - expected) < 3e-6, "FP32 control projection differs from oracle");
    }
}
}
int main() {
    if (cuda_unavailable()) return 77;
    projection();
    projection(32, 1, 65536);
    tiled_selection();
    exercise(64, 4, 8, 128, 301);
    exercise(128, 32, 4, 2048, 4099);
    decode_history();
    const ops::GlmIndexerGeometry g{128, 32, 4, 2048, 1e-6F};
    expect(ops::glm_indexer_select_workspace_capacity_bytes(4096, 1048576, g) <= (64ULL << 20) + 256,
           "long-context scratch exceeded bound");
    std::cout << "GLM indexer: projection, causal selection, learned pooling, rollback, batching and graph replay passed\n";
}
