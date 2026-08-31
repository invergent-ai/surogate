// QSA sparse indexer: block-key folding (mean, RMSNorm, rope at the block's first position) and
// block selection (rectified per-head scores, always-visible tail, budget cut on a block
// boundary) against a scalar reference, on a small paged cache.
#include "api/ops/qsa_indexer.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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
                                       const std::vector<float>& gain) {
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
        const float angle = static_cast<float>(first) *
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
        plane = DeviceBuffer(static_cast<std::size_t>(pages) * kPageSize * kHeadDim * 2);
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
                                    {kHeadDim, kPageSize, pages});
        view.block_table   = Tensor(const_cast<void*>(table.p), DType::I32, {pages});
        view.head_dim      = kHeadDim;
        view.num_kv_heads  = 1;
        return view;
    }
    PagedKVBatchLayerView batch_view() const {
        PagedKVBatchLayerView view;
        view.indexer_pages = Tensor(const_cast<void*>(plane.p), DType::BF16,
                                    {kHeadDim, kPageSize, pages});
        view.block_tables  = Tensor(const_cast<void*>(table.p), DType::I32, {pages, 1});
        view.head_dim      = kHeadDim;
        view.num_kv_heads  = 1;
        return view;
    }
    std::size_t cell_offset(int position) const {
        const int page = table_host[static_cast<std::size_t>(position / kPageSize)];
        return (static_cast<std::size_t>(page) * kPageSize + (position % kPageSize)) * kHeadDim;
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

int main() {
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
        sinfer::ops::qsa_indexer_append(keys, pos_t, rows_t, count, gain_t, geometry(2048),
                                        cache.batch_view(), nullptr);
        cudaStreamSynchronize(nullptr);
    };
    append(0, 256);
    for (int t = 256; t < tokens; ++t) { append(t, 1); }
    expect(cudaGetLastError() == cudaSuccess, "append launched cleanly");

    // Every complete block's first cell holds its block key.
    const std::vector<double> plane_host =
        from_device_bf16(cache.plane, cache.plane.bytes / 2);
    // The plane stores BF16, so a component is exact only to one BF16 ulp (~0.4 % relative,
    // and a fixed floor where the value is near zero).
    double worst = 0.0;
    for (int b = 0; b * kBlock + kBlock <= tokens; ++b) {
        const std::vector<float> want = reference_block_key(raw, b * kBlock, gain);
        const std::size_t base        = cache.cell_offset(b * kBlock);
        for (int d = 0; d < kHeadDim; ++d) {
            const double tolerance = 0.008 * std::abs(want[d]) + 0.004;
            worst = std::max(worst, std::abs(plane_host[base + d] - want[d]) / tolerance);
        }
    }
    if (worst >= 1.0) { // report where it diverges before failing
        for (int b = 0; b * kBlock + kBlock <= tokens && b < 3; ++b) {
            const std::vector<float> want = reference_block_key(raw, b * kBlock, gain);
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
