// The router's top-k on non-finite logits.
//
// One non-finite value in a token's hidden state makes every router logit of that token NaN.
// NaN compares neither better nor worse than anything, so the warp reduction kept lane 0's
// end-of-list sentinel once lane 0's own values were taken: ranks 4..7 of Gemma 4's top 8 of 128
// came out as expert id 0x7fffffff, and the prefill route and the decode path then indexed
// their tables with it -- the serve engine's cudaErrorIllegalAddress (Xid 31 MMU faults far
// outside any allocation). This checks, for several mixture shapes, that every selected id is a
// valid distinct expert whatever the logits hold, and that finite logits still select exactly
// the reference top-k (value descending, lower id first on a tie).

#include "ops/sparse_moe/sparse_moe_route.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

using sinfer::ops::SparseMoeGating;

bool cuda_unavailable() {
    int count = 0;
    return cudaGetDeviceCount(&count) != cudaSuccess || count == 0;
}

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        const cudaError_t status_ = (call);                                                        \
        if (status_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status_));  \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

template <int Experts, int TopK, SparseMoeGating Gating>
__global__ void route_rows_kernel(const float* scores, const float* bias, int* ids, float* alpha,
                                  int rows) {
    __shared__ float selected[TopK];
    __shared__ float shared_scale;
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) { return; }
    sinfer::ops::detail::sparse_moe_select_top_k_warp<Experts, TopK, /*HasShared=*/false, Gating>(
        scores + static_cast<std::int64_t>(row) * Experts, ids + row * TopK, alpha + row * TopK,
        &shared_scale, selected, bias);
}

float host_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

template <int Experts, int TopK, SparseMoeGating Gating>
int run_shape(const char* name) {
    constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
    constexpr float kInf = std::numeric_limits<float>::infinity();
    std::mt19937 rng(0x5eed + Experts * 31 + TopK);
    std::normal_distribution<float> normal(0.0f, 2.0f);

    std::vector<std::vector<float>> rows;
    std::vector<bool> finite_row;
    const auto add = [&](std::vector<float> row, bool finite) {
        rows.push_back(std::move(row));
        finite_row.push_back(finite);
    };
    const auto random_row = [&] {
        std::vector<float> row(Experts);
        for (float& value : row) { value = normal(rng); }
        return row;
    };
    // Non-finite rows: all NaN (what a NaN hidden state gives), lane 0's values NaN, one NaN,
    // every other one NaN, NaN beside infinities, and all -inf.
    add(std::vector<float>(Experts, kNaN), false);
    {
        auto row = random_row();
        for (int item = 0; item < Experts / 32; ++item) { row[item * 32] = kNaN; }
        add(row, false);
    }
    {
        auto row = random_row();
        row[Experts / 2] = kNaN;
        add(row, false);
    }
    {
        auto row = random_row();
        for (int id = 0; id < Experts; id += 2) { row[id] = kNaN; }
        add(row, false);
    }
    {
        std::vector<float> row(Experts, kNaN);
        row[3] = kInf;
        row[Experts - 1] = -kInf;
        add(row, false);
    }
    add(std::vector<float>(Experts, -kInf), false);
    // Finite rows, with ties, which must keep the reference order exactly.
    for (int i = 0; i < 256; ++i) {
        auto row = random_row();
        if (i % 4 == 0) {
            for (int id = 0; id < Experts; id += 3) { row[id] = std::round(row[id]); }
        }
        add(row, true);
    }

    const int count = static_cast<int>(rows.size());
    std::vector<float> flat;
    for (const auto& row : rows) { flat.insert(flat.end(), row.begin(), row.end()); }
    std::vector<float> bias(Experts);
    for (float& value : bias) { value = 0.05f * normal(rng); }

    float* d_scores = nullptr;
    float* d_bias   = nullptr;
    int* d_ids      = nullptr;
    float* d_alpha  = nullptr;
    CHECK_CUDA(cudaMalloc(&d_scores, flat.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, bias.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ids, static_cast<std::size_t>(count) * TopK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_alpha, static_cast<std::size_t>(count) * TopK * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_scores, flat.data(), flat.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    route_rows_kernel<Experts, TopK, Gating><<<count, 32>>>(d_scores, d_bias, d_ids, d_alpha, count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<int> ids(static_cast<std::size_t>(count) * TopK);
    CHECK_CUDA(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_ids));
    CHECK_CUDA(cudaFree(d_alpha));

    int failures = 0;
    for (int r = 0; r < count; ++r) {
        const int* row_ids = ids.data() + static_cast<std::size_t>(r) * TopK;
        std::set<int> distinct;
        bool valid = true;
        for (int k = 0; k < TopK; ++k) {
            valid = valid && row_ids[k] >= 0 && row_ids[k] < Experts;
            distinct.insert(row_ids[k]);
        }
        if (!valid || static_cast<int>(distinct.size()) != TopK) {
            std::string list;
            for (int k = 0; k < TopK; ++k) { list += " " + std::to_string(row_ids[k]); }
            std::printf("FAIL %s row %d (%s): ids%s\n", name, r,
                        finite_row[r] ? "finite" : "non-finite", list.c_str());
            ++failures;
            continue;
        }
        if (!finite_row[r]) { continue; }
        // The reference ranking: value descending, the lower id first on a tie.
        std::vector<float> ranked(Experts);
        for (int id = 0; id < Experts; ++id) {
            const float score = rows[r][id];
            ranked[id] = Gating == SparseMoeGating::SigmoidBiasTopK ? host_sigmoid(score) + bias[id]
                                                                     : score;
        }
        std::vector<int> order(Experts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return ranked[a] > ranked[b]; });
        for (int k = 0; k < TopK; ++k) {
            // The device's sigmoid may round differently from the host's, so a sigmoid router
            // is held to the ranking only where the host values are clearly apart.
            const bool tight = Gating != SparseMoeGating::SigmoidBiasTopK ||
                               (k + 1 < Experts && ranked[order[k]] - ranked[order[k + 1]] > 1e-5f &&
                                (k == 0 || ranked[order[k - 1]] - ranked[order[k]] > 1e-5f));
            if (tight && row_ids[k] != order[k]) {
                std::printf("FAIL %s row %d rank %d: id %d, reference %d\n", name, r, k, row_ids[k],
                            order[k]);
                ++failures;
                break;
            }
        }
    }
    std::printf("%s %s: %d rows\n", failures == 0 ? "ok  " : "FAIL", name, count);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    int failures = 0;
    failures += run_shape<128, 8, SparseMoeGating::SoftmaxTopK>("softmax 8 of 128 (Gemma 4)");
    failures += run_shape<256, 8, SparseMoeGating::SoftmaxTopK>("softmax 8 of 256");
    failures += run_shape<64, 6, SparseMoeGating::SoftmaxTopK>("softmax 6 of 64");
    failures += run_shape<32, 4, SparseMoeGating::SoftmaxTopK>("softmax 4 of 32");
    failures += run_shape<256, 8, SparseMoeGating::SigmoidBiasTopK>("sigmoid+bias 8 of 256");
    std::printf("%s sparse_moe route top-k\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
