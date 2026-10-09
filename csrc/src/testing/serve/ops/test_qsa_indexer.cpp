// QSA sparse indexer: block-key folding (mean, RMSNorm, rope at the block's first position) and
// block selection (rectified per-head scores, always-visible tail, budget cut on a block
// boundary) against a scalar reference, on a small paged cache and on a 40k-cell history whose
// scores spread over many CTAs, and the per-tile block lists the prompt kernel walks. `--bench`
// times one layer's selection at long histories; SUROGATE_SERVE_QSA_SELECT=bisect times the
// one-CTA-per-row control instead.
#include "api/ops/qsa_indexer.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr int kHeadDim   = 128;
constexpr int kBlock     = 4;
constexpr int kHeads     = 4;
constexpr int kRotary    = 64;
constexpr float kTheta   = 1.0e7F;
constexpr float kEps     = 1.0e-6F;
constexpr int kPageSize  = 64; // kPagedKVPageSize

sinfer::ops::QsaIndexerGeometry geometry(int top_k) {
    return sinfer::ops::QsaIndexerGeometry{.head_dim   = kHeadDim,
                                           .heads      = kHeads,
                                           .block      = kBlock,
                                           .top_k      = top_k,
                                           .rotary_dim = kRotary,
                                           .rope_theta = kTheta,
                                           .rms_eps    = kEps};
}

float bf16_round(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7FFFU + lsb;
    bits &= 0xFFFF0000U;
    float out = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// The block key the kernel must produce: mean of the block's raw keys, RMS-normalised with the
// gain, then split-half NeoX rope at the block's first position.
std::vector<float> reference_block_key(const std::vector<float>& raw, int first,
                                       const std::vector<float>& gain, bool mrope = false) {
    std::vector<float> value(kHeadDim, 0.0F);
    for (int c = 0; c < kBlock; ++c) {
        for (int d = 0; d < kHeadDim; ++d) {
            value[d] += bf16_round(raw[static_cast<std::size_t>(first + c) * kHeadDim + d]);
        }
    }
    double square = 0.0;
    for (int d = 0; d < kHeadDim; ++d) {
        value[d] /= static_cast<float>(kBlock);
        square += static_cast<double>(value[d]) * value[d];
    }
    const float scale = static_cast<float>(1.0 / std::sqrt(square / kHeadDim + kEps));
    for (int d = 0; d < kHeadDim; ++d) { value[d] *= scale * gain[d]; }
    std::vector<float> out(value);
    const int half = kRotary / 2;
    for (int d = 0; d < half; ++d) {
        const int axis = d % 3 == 1 && d < 33 ? 1 : d % 3 == 2 && d < 30 ? 2 : 0;
        const int coordinate = !mrope ? first : axis == 0 ? 17 + first / 40 : axis == 1 ? 23 + first / 7 : 31 + first % 19;
        const float angle = static_cast<float>(coordinate) *
                            std::pow(kTheta, -2.0F * static_cast<float>(d) / kRotary);
        const float c = std::cos(angle), s = std::sin(angle);
        out[d]        = value[d] * c - value[d + half] * s;
        out[d + half] = value[d + half] * c + value[d] * s;
    }
    return out;
}

struct Cache {
    int pages;
    DeviceBuffer plane;
    DeviceBuffer table;
    std::vector<int> table_host;
    explicit Cache(int pages_in) : pages(pages_in), plane(0), table(0) {
        plane = DeviceBuffer(static_cast<std::size_t>(pages) * kPageSize * ops::kQsaIndexerStorageHeadDim * 2);
        cudaMemset(plane.p, 0, plane.bytes);
        table_host.resize(static_cast<std::size_t>(pages));
        std::iota(table_host.begin(), table_host.end(), 0);
        // Shuffle the pages so the test exercises the block table rather than an identity map.
        std::reverse(table_host.begin(), table_host.end());
        table = to_device_i32(table_host);
    }
    PagedKVLayerView layer_view() const {
        PagedKVLayerView view;
        view.indexer_pages = Tensor(const_cast<void*>(plane.p), DType::BF16,
                                    {ops::kQsaIndexerStorageHeadDim, kPageSize, pages});
        view.block_table   = Tensor(const_cast<void*>(table.p), DType::I32, {pages});
        view.head_dim      = kHeadDim;
        view.num_kv_heads  = 1;
        return view;
    }
    PagedKVBatchLayerView batch_view() const {
        PagedKVBatchLayerView view;
        view.indexer_pages = Tensor(const_cast<void*>(plane.p), DType::BF16,
                                    {ops::kQsaIndexerStorageHeadDim, kPageSize, pages});
        view.block_tables  = Tensor(const_cast<void*>(table.p), DType::I32, {pages, 1});
        view.head_dim      = kHeadDim;
        view.num_kv_heads  = 1;
        return view;
    }
    std::size_t cell_offset(int position) const {
        const int page = table_host[static_cast<std::size_t>(position / kPageSize)];
        return static_cast<std::size_t>(page) * kPageSize * ops::kQsaIndexerStorageHeadDim +
               kPageSize * kHeadDim + ((position % kPageSize) / kBlock) * kHeadDim;
    }
};

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

} // namespace

int run_case(bool mrope) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    const int tokens = 300; // 75 complete blocks
    const int pages  = (tokens + kPageSize - 1) / kPageSize + 1;
    Cache cache(pages);

    std::vector<float> raw(static_cast<std::size_t>(tokens) * kHeadDim);
    for (auto& v : raw) { v = dist(rng); }
    std::vector<float> gain(kHeadDim);
    for (auto& v : gain) { v = 0.5F + 0.5F * dist(rng); }
    std::vector<int> positions(tokens);
    std::iota(positions.begin(), positions.end(), 0);

    // Append in two rounds: a prompt chunk and then single columns, as the engine does.
    DeviceBuffer d_gain = to_device_bf16(gain);
    const auto append   = [&](int begin, int count) {
        std::vector<float> chunk(raw.begin() + static_cast<std::size_t>(begin) * kHeadDim,
                                 raw.begin() + static_cast<std::size_t>(begin + count) * kHeadDim);
        std::vector<int> pos(positions.begin() + begin, positions.begin() + begin + count);
        DeviceBuffer d_keys = to_device_bf16(chunk);
        DeviceBuffer d_pos  = to_device_i32(pos);
        Tensor keys(d_keys.p, DType::BF16, {kHeadDim, count});
        Tensor pos_t(d_pos.p, DType::I32, {count});
        Tensor gain_t(d_gain.p, DType::BF16, {kHeadDim});
        std::vector<int> rows(1, 0);
        DeviceBuffer d_rows = to_device_i32(rows);
        Tensor rows_t(d_rows.p, DType::I32, {1});
        auto g = geometry(2048);
        g.mrope_height = mrope ? 11 : 0;
        g.mrope_width = mrope ? 10 : 0;
        std::vector<int> rope(3 * (count + 3), -999);
        for (int i = 0; i < count; ++i) {
            rope[i] = 17 + (begin + i) / 40;
            rope[count + 3 + i] = 23 + (begin + i) / 7;
            rope[2 * (count + 3) + i] = 31 + (begin + i) % 19;
        }
        DeviceBuffer d_rope = to_device_i32(rope);
        Tensor rope_t(d_rope.p, DType::I32, {count, 3});
        rope_t.nb[1] = (count + 3) * sizeof(int); // slice of a larger packed round
        sinfer::ops::qsa_indexer_append(keys, pos_t, rows_t, count, gain_t, g,
                                        cache.batch_view(), nullptr, mrope ? rope_t : Tensor{});
        cudaStreamSynchronize(nullptr);
    };
    append(0, 257);
    for (int t = 257; t < tokens; ++t) { append(t, 1); }
    expect(cudaGetLastError() == cudaSuccess, "append launched cleanly");

    // A rejected speculative suffix can start inside an already folded block.
    // Replace its keys and replay it, including suffixes crossing physical pages.
    for (const auto [begin, count] : {std::pair{13, 3}, {62, 6}, {127, 5}, {254, 2}}) {
        for (int token = begin; token < begin + count; ++token) {
            for (int d = 0; d < kHeadDim; ++d) { raw[token * kHeadDim + d] = dist(rng); }
        }
        append(begin, count);
        const auto first = from_device_bf16(cache.plane, cache.plane.bytes / 2);
        append(begin, count);
        const auto replay = from_device_bf16(cache.plane, cache.plane.bytes / 2);
        expect(first == replay, "re-appending a speculative suffix must be idempotent");
    }

    // Every complete block has a pooled key separate from its raw cells.
    const std::vector<double> plane_host =
        from_device_bf16(cache.plane, cache.plane.bytes / 2);
    // The plane stores BF16, so a component is exact only to one BF16 ulp (~0.4 % relative,
    // and a fixed floor where the value is near zero).
    double worst = 0.0;
    for (int b = 0; b * kBlock + kBlock <= tokens; ++b) {
        const std::vector<float> want = reference_block_key(raw, b * kBlock, gain, mrope);
        const std::size_t base        = cache.cell_offset(b * kBlock);
        for (int d = 0; d < kHeadDim; ++d) {
            const double tolerance = 0.008 * std::abs(want[d]) + 0.004;
            worst = std::max(worst, std::abs(plane_host[base + d] - want[d]) / tolerance);
        }
    }
    if (worst >= 1.0) { // report where it diverges before failing
        for (int b = 0; b * kBlock + kBlock <= tokens && b < 3; ++b) {
            const std::vector<float> want = reference_block_key(raw, b * kBlock, gain, mrope);
            const std::size_t base        = cache.cell_offset(b * kBlock);
            std::cerr << "block " << b << " (first " << b * kBlock << "):";
            for (int d = 0; d < 6; ++d) {
                std::cerr << " " << plane_host[base + d] << "/" << want[d];
            }
            std::cerr << " | d64:" << plane_host[base + 64] << "/" << want[64] << "\n";
        }
    }
    expect(worst < 1.0, "block keys match the reference (worst error " +
                            std::to_string(worst) + " of one BF16 tolerance)");

    // Selection with a budget that covers everything: every touched block is visible.
    const int rows = 3;
    std::vector<float> q(static_cast<std::size_t>(rows) * kHeads * kHeadDim);
    for (auto& v : q) { v = dist(rng); }
    std::vector<int> q_pos{tokens - 1, 200, 99};
    std::vector<int> q_rows(rows, 0);
    DeviceBuffer d_q    = to_device_bf16(q);
    DeviceBuffer d_qpos = to_device_i32(q_pos);
    DeviceBuffer d_qrow = to_device_i32(q_rows);
    const int words     = sinfer::ops::qsa_block_mask_words(tokens, kBlock);
    DeviceBuffer d_mask(static_cast<std::size_t>(words) * rows * sizeof(std::uint32_t));
    WorkspaceArena arena(1 << 20);
    Tensor q_t(d_q.p, DType::BF16, {kHeadDim, kHeads, rows});
    Tensor qpos_t(d_qpos.p, DType::I32, {rows});
    Tensor qrow_t(d_qrow.p, DType::I32, {rows});
    Tensor mask_t(d_mask.p, DType::I32, {words, rows});

    sinfer::ops::qsa_indexer_select(q_t, qpos_t, qrow_t, 1, geometry(2048), cache.batch_view(),
                                    tokens, arena, mask_t, nullptr);
    cudaStreamSynchronize(nullptr);
    expect(cudaGetLastError() == cudaSuccess, "select launched cleanly");
    {
        const std::vector<int> mask = from_device_i32(d_mask, static_cast<std::size_t>(words) * rows);
        for (int r = 0; r < rows; ++r) {
            const int touched = (q_pos[r] + 1 + kBlock - 1) / kBlock;
            int visible       = 0;
            for (int b = 0; b < words * 32; ++b) {
                const bool bit = (static_cast<std::uint32_t>(mask[r * words + (b >> 5)]) >>
                                  (b & 31)) & 1U;
                if (bit) { ++visible; }
                if (bit && b >= touched) { expect(false, "a block past the query is visible"); }
            }
            expect(visible == touched, "a covering budget selects every touched block (row " +
                                           std::to_string(r) + ": " + std::to_string(visible) +
                                           " of " + std::to_string(touched) + ")");
        }
    }

    // A budget of 32 cells = 8 complete blocks, plus the always-visible tail.
    arena.reset();
    sinfer::ops::qsa_indexer_select(q_t, qpos_t, qrow_t, 1, geometry(32), cache.batch_view(),
                                    tokens, arena, mask_t, nullptr);
    cudaStreamSynchronize(nullptr);
    const std::vector<int> mask = from_device_i32(d_mask, static_cast<std::size_t>(words) * rows);
    for (int r = 0; r < rows; ++r) {
        const int scored  = (q_pos[r] + 1) / kBlock;
        const int touched = (q_pos[r] + 1 + kBlock - 1) / kBlock;
        // Reference scores: rectified per-head dot products against the stored block keys.
        std::vector<std::pair<float, int>> ranked;
        for (int b = 0; b < scored; ++b) {
            const std::size_t base = cache.cell_offset(b * kBlock);
            float total            = 0.0F;
            for (int h = 0; h < kHeads; ++h) {
                float dot = 0.0F;
                for (int d = 0; d < kHeadDim; ++d) {
                    dot += bf16_round(q[(static_cast<std::size_t>(r) * kHeads + h) * kHeadDim + d]) *
                           static_cast<float>(plane_host[base + d]);
                }
                total += std::max(dot, 0.0F);
            }
            ranked.emplace_back(total, b);
        }
        std::stable_sort(ranked.begin(), ranked.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        const int budget = 32 / kBlock;
        std::vector<bool> want(static_cast<std::size_t>(touched), false);
        for (int b = scored; b < touched; ++b) { want[static_cast<std::size_t>(b)] = true; }
        for (int i = 0; i < budget && i < static_cast<int>(ranked.size()); ++i) {
            want[static_cast<std::size_t>(ranked[static_cast<std::size_t>(i)].second)] = true;
        }
        int selected = 0, mismatched = 0;
        for (int b = 0; b < touched; ++b) {
            const bool bit = (static_cast<std::uint32_t>(mask[r * words + (b >> 5)]) >> (b & 31)) & 1U;
            if (bit) { ++selected; }
            if (bit != want[static_cast<std::size_t>(b)]) { ++mismatched; }
        }
        const int expected = std::min(budget, scored) + (touched - scored);
        expect(selected == expected, "the budget selects " + std::to_string(expected) +
                                         " blocks (row " + std::to_string(r) + " got " +
                                         std::to_string(selected) + ")");
        expect(mismatched == 0, "the selected blocks are the highest scoring (row " +
                                    std::to_string(r) + ", " + std::to_string(mismatched) +
                                    " differ)");
    }

    if (failures == 0) { std::cout << "qsa_indexer: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}

// Random pooled keys (or zeros) written straight into the plane for every complete block of
// `keys` cells, the same keys for every sequence but on disjoint pages. `host` receives the
// BF16-rounded keys, block by block.
struct LongCache {
    int pages_per_sequence;
    int sequences;
    DeviceBuffer plane;
    DeviceBuffer table;
    LongCache(int keys, int sequences_in, std::uint32_t seed, bool zero, std::vector<float>* host)
        : pages_per_sequence(keys / kPageSize + 1), sequences(sequences_in) {
        const std::size_t page_elements = static_cast<std::size_t>(kPageSize) * ops::kQsaIndexerStorageHeadDim;
        const std::size_t sequence_elements = page_elements * pages_per_sequence;
        std::vector<std::uint16_t> bits(sequence_elements, 0);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
        const int blocks = keys / kBlock;
        if (host != nullptr) { host->assign(static_cast<std::size_t>(blocks) * kHeadDim, 0.0F); }
        for (int b = 0; b < blocks && !zero; ++b) {
            const int position = b * kBlock;
            const std::size_t base = static_cast<std::size_t>(position / kPageSize) * page_elements +
                                     kPageSize * kHeadDim + ((position % kPageSize) / kBlock) * kHeadDim;
            for (int d = 0; d < kHeadDim; ++d) {
                const std::uint16_t value = f32_to_bf16(dist(rng));
                bits[base + d] = value;
                if (host != nullptr) { (*host)[static_cast<std::size_t>(b) * kHeadDim + d] = bf16_to_f32(value); }
            }
        }
        plane = DeviceBuffer(sequence_elements * sizeof(std::uint16_t) * sequences);
        for (int s = 0; s < sequences; ++s) {
            plane.copy_from_host(bits.data(), sequence_elements * sizeof(std::uint16_t),
                                 sequence_elements * sizeof(std::uint16_t) * s);
        }
        std::vector<int> table_host(static_cast<std::size_t>(pages_per_sequence) * sequences);
        std::iota(table_host.begin(), table_host.end(), 0);
        table = to_device_i32(table_host);
    }
    PagedKVBatchLayerView view() const {
        PagedKVBatchLayerView view;
        view.indexer_pages = Tensor(const_cast<void*>(plane.p), DType::BF16,
                                    {ops::kQsaIndexerStorageHeadDim, kPageSize, pages_per_sequence * sequences});
        view.block_tables  = Tensor(const_cast<void*>(table.p), DType::I32, {pages_per_sequence, sequences});
        view.head_dim      = kHeadDim;
        view.num_kv_heads  = 1;
        return view;
    }
};

// A long history: rows past the budget, one exactly at it and one in the open tail, scored by
// several CTAs each. Then every key zero, so every score ties and the budget must take the
// earliest blocks.
void run_long_case() {
    const int keys = 40003; // 10,000 complete blocks and an open tail
    const auto g   = geometry(2048);
    const int budget = 2048 / kBlock;
    std::vector<int> q_pos{keys - 1, 30001, 2100, 2047, 5002, 39999};
    const int rows = static_cast<int>(q_pos.size());
    std::mt19937 rng(29);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> q(static_cast<std::size_t>(rows) * kHeads * kHeadDim);
    for (auto& v : q) { v = bf16_round(dist(rng)); }
    DeviceBuffer d_q    = to_device_bf16(q);
    DeviceBuffer d_qpos = to_device_i32(q_pos);
    DeviceBuffer d_qrow = to_device_i32(std::vector<int>(static_cast<std::size_t>(rows), 0));
    const int words     = sinfer::ops::qsa_block_mask_words(keys, kBlock);
    DeviceBuffer d_mask(static_cast<std::size_t>(words) * rows * sizeof(std::uint32_t));
    WorkspaceArena arena(sinfer::ops::qsa_indexer_select_workspace_capacity_bytes(rows, keys, g));
    Tensor q_t(d_q.p, DType::BF16, {kHeadDim, kHeads, rows});
    Tensor qpos_t(d_qpos.p, DType::I32, {rows});
    Tensor qrow_t(d_qrow.p, DType::I32, {rows});
    Tensor mask_t(d_mask.p, DType::I32, {words, rows});

    for (const bool zero : {false, true}) {
        std::vector<float> host;
        LongCache cache(keys, 1, 31, zero, &host);
        d_mask.fill(0xA5); // every word must be written
        arena.reset();
        sinfer::ops::qsa_indexer_select(q_t, qpos_t, qrow_t, 1, g, cache.view(), keys, arena,
                                        mask_t, nullptr);
        cudaStreamSynchronize(nullptr);
        expect(cudaGetLastError() == cudaSuccess, "long select launched cleanly");
        const std::vector<int> mask = from_device_i32(d_mask, static_cast<std::size_t>(words) * rows);
        for (int r = 0; r < rows; ++r) {
            const int scored  = (q_pos[r] + 1) / kBlock;
            const int touched = (q_pos[r] + 1 + kBlock - 1) / kBlock;
            std::vector<double> score(static_cast<std::size_t>(scored));
            for (int b = 0; b < scored; ++b) {
                double total = 0.0;
                for (int h = 0; h < kHeads; ++h) {
                    double dot = 0.0;
                    for (int d = 0; d < kHeadDim; ++d) {
                        dot += static_cast<double>(q[(static_cast<std::size_t>(r) * kHeads + h) * kHeadDim + d]) *
                               host[static_cast<std::size_t>(b) * kHeadDim + d];
                    }
                    total += std::max(dot, 0.0);
                }
                score[static_cast<std::size_t>(b)] = total;
            }
            std::vector<double> sorted(score);
            std::sort(sorted.begin(), sorted.end(), std::greater<>());
            const bool covered = scored <= budget;
            const double threshold = covered ? 0.0 : sorted[static_cast<std::size_t>(budget - 1)];
            const std::string where = std::string(zero ? "tied" : "random") + " row " + std::to_string(r);
            int selected = 0, wrong = 0, past = 0;
            for (int b = 0; b < words * 32; ++b) {
                const bool bit = (static_cast<std::uint32_t>(mask[static_cast<std::size_t>(r) * words + (b >> 5)]) >> (b & 31)) & 1U;
                selected += bit ? 1 : 0;
                if (b >= touched) { past += bit ? 1 : 0; continue; }
                if (b >= scored || covered) { wrong += bit ? 0 : 1; continue; }
                if (zero) { // every score is zero: the budget takes the earliest blocks
                    wrong += bit != (b < budget) ? 1 : 0;
                    continue;
                }
                const double value = score[static_cast<std::size_t>(b)];
                // The kernel sums in FP32 in its own order, so a block within rounding of the
                // threshold may fall either way.
                const bool near = std::abs(value - threshold) <= 1.0e-5 * threshold + 1.0e-6;
                if (!near && bit != (value > threshold)) { ++wrong; }
            }
            const int expected = std::min(budget, scored) + (touched - scored);
            expect(selected == expected, "long " + where + " selects " + std::to_string(expected) +
                                             " blocks (got " + std::to_string(selected) + ")");
            expect(wrong == 0, "long " + where + ": " + std::to_string(wrong) + " blocks differ from the reference");
            expect(past == 0, "long " + where + ": a block past the query is visible");
        }
    }
    if (failures == 0) { std::cout << "qsa_indexer: long-history selection matches\n"; }
}

// One layer's selection: a decode round of 1 and 8 sequences (one query each, on disjoint
// histories) and a prompt chunk of 2,048 queries over one history.
void run_bench() {
    const auto g = geometry(2048);
    const char* path = std::getenv("SUROGATE_SERVE_QSA_SELECT");
    std::cout << "qsa select bench (" << (path != nullptr ? path : "split") << ")\n";
    for (const int keys : {32768, 131072, 262144}) {
        for (const auto [rows, sequences] : {std::pair{1, 1}, {8, 8}, {32, 8}, {2048, 1}}) {
            LongCache cache(keys, sequences, 37, false, nullptr);
            std::mt19937 rng(41);
            std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
            std::vector<float> q(static_cast<std::size_t>(rows) * kHeads * kHeadDim);
            for (auto& v : q) { v = dist(rng); }
            const int per_sequence = rows / sequences;
            std::vector<int> positions(static_cast<std::size_t>(rows));
            for (int r = 0; r < rows; ++r) { positions[static_cast<std::size_t>(r)] = keys - per_sequence + r % per_sequence; }
            DeviceBuffer d_q    = to_device_bf16(q);
            DeviceBuffer d_pos  = to_device_i32(positions);
            std::vector<int> table_rows(static_cast<std::size_t>(sequences));
            std::iota(table_rows.begin(), table_rows.end(), 0);
            DeviceBuffer d_rows = to_device_i32(table_rows);
            const int words = sinfer::ops::qsa_block_mask_words(keys, kBlock);
            DeviceBuffer d_mask(static_cast<std::size_t>(words) * rows * sizeof(std::uint32_t));
            WorkspaceArena arena(sinfer::ops::qsa_indexer_select_workspace_capacity_bytes(rows, keys, g));
            Tensor q_t(d_q.p, DType::BF16, {kHeadDim, kHeads, rows});
            Tensor pos_t(d_pos.p, DType::I32, {rows});
            Tensor rows_t(d_rows.p, DType::I32, {sequences});
            Tensor mask_t(d_mask.p, DType::I32, {words, rows});
            const auto once = [&] {
                arena.reset();
                sinfer::ops::qsa_indexer_select(q_t, pos_t, rows_t, per_sequence, g, cache.view(),
                                                keys, arena, mask_t, nullptr);
            };
            for (int i = 0; i < 3; ++i) { once(); }
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            const int iterations = rows >= 2048 ? 5 : 50;
            cudaEventRecord(start);
            for (int i = 0; i < iterations; ++i) { once(); }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0.0F;
            cudaEventElapsedTime(&ms, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            expect(cudaGetLastError() == cudaSuccess, "bench select launched cleanly");
            std::cout << "  keys " << keys << " rows " << rows << " sequences " << sequences << ": "
                      << 1000.0F * ms / iterations << " us per call\n";
        }
    }
}

// Per-tile block lists: 130 rows (two full tiles and a two-row tail) of a 131,072-key mask, each
// row a sparse random selection, against the union of each tile's rows in ascending order.
void run_tile_union_case() {
    const int keys  = 131072;
    const int rows  = 130;
    const int words = sinfer::ops::qsa_block_mask_words(keys, kBlock);
    const int tiles = (rows + sinfer::ops::kQsaTileRows - 1) / sinfer::ops::kQsaTileRows;
    const int stride = sinfer::ops::qsa_tile_union_stride(keys, kBlock);
    std::mt19937 rng(23);
    std::vector<int> mask(static_cast<std::size_t>(words) * rows, 0);
    for (auto& word : mask) {
        // About one bit in 64, and some words left empty.
        word = static_cast<int>(rng() & rng() & rng() & rng() & rng() & rng());
    }
    DeviceBuffer d_mask   = to_device_i32(mask);
    DeviceBuffer d_blocks(static_cast<std::size_t>(stride) * tiles * sizeof(std::int32_t));
    DeviceBuffer d_counts(static_cast<std::size_t>(tiles) * sizeof(std::int32_t));
    Tensor mask_t(d_mask.p, DType::I32, {words, rows});
    Tensor blocks_t(d_blocks.p, DType::I32, {stride, tiles});
    Tensor counts_t(d_counts.p, DType::I32, {tiles});
    sinfer::ops::qsa_tile_union(mask_t, blocks_t, counts_t, nullptr);
    cudaStreamSynchronize(nullptr);
    expect(cudaGetLastError() == cudaSuccess, "tile union launched cleanly");
    const std::vector<int> blocks = from_device_i32(d_blocks, static_cast<std::size_t>(stride) * tiles);
    const std::vector<int> counts = from_device_i32(d_counts, static_cast<std::size_t>(tiles));
    for (int tile = 0; tile < tiles; ++tile) {
        std::vector<int> expected;
        for (int w = 0; w < words; ++w) {
            std::uint32_t bits = 0U;
            for (int r = tile * sinfer::ops::kQsaTileRows;
                 r < std::min(rows, (tile + 1) * sinfer::ops::kQsaTileRows); ++r) {
                bits |= static_cast<std::uint32_t>(mask[static_cast<std::size_t>(r) * words + w]);
            }
            for (int b = 0; b < 32; ++b) {
                if ((bits >> b) & 1U) { expected.push_back(w * 32 + b); }
            }
        }
        expect(counts[tile] == static_cast<int>(expected.size()),
               "tile " + std::to_string(tile) + " lists " + std::to_string(counts[tile]) +
                   " blocks, expected " + std::to_string(expected.size()));
        const auto first = blocks.begin() + static_cast<std::ptrdiff_t>(tile) * stride;
        expect(counts[tile] == static_cast<int>(expected.size()) &&
                   std::equal(expected.begin(), expected.end(), first),
               "tile " + std::to_string(tile) + " lists its rows' blocks in ascending order");
    }
}

int main(int argc, char** argv) {
    if (cuda_unavailable()) { return 77; }
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        run_bench();
        return failures ? 1 : 0;
    }
    run_case(false);
    run_case(true);
    run_long_case();
    run_tile_union_case();
    return failures ? 1 : 0;
}
