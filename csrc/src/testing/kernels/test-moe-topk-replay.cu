// Rollout Routing Replay (R3): moe_topk_replay must (1) leave a selection unchanged when replaying that same
// selection, (2) install the rollout's experts with weights recomputed from the current scores by the same rule as
// moe_topk_forward, (3) skip rows marked -1 and reject out-of-range ids, and (4) keep moe_topk_backward exact, i.e.
// gradients reach exactly the replayed experts.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/kernels.h"

namespace {

template <class T>
struct Device {
    T* ptr = nullptr;
    explicit Device(const std::vector<T>& values) {
        REQUIRE(cudaMalloc(&ptr, std::max<std::size_t>(values.size(), 1) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMemcpy(ptr, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    ~Device() {
        cudaFree(ptr);
    }
    std::vector<T> read(std::size_t n) const {
        std::vector<T> result(n);
        REQUIRE(cudaMemcpy(result.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
        return result;
    }
};

std::vector<float> random_probs(int tokens, int experts, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.5f);
    std::vector<float> probs(static_cast<std::size_t>(tokens) * experts);
    for (int t = 0; t < tokens; ++t) {
        float maxv = -INFINITY;
        for (int e = 0; e < experts; ++e) {
            probs[t * experts + e] = dist(rng);
            maxv = std::max(maxv, probs[t * experts + e]);
        }
        float sum = 0.0f;
        for (int e = 0; e < experts; ++e) {
            probs[t * experts + e] = std::exp(probs[t * experts + e] - maxv);
            sum += probs[t * experts + e];
        }
        for (int e = 0; e < experts; ++e)
            probs[t * experts + e] /= sum;
    }
    return probs;
}

// Distinct random experts per replayed token; rows with token % 3 == 2 are marked -1 (not replayed).
std::vector<std::int16_t> random_replay(int tokens, int experts, int top_k, std::mt19937& rng) {
    std::vector<std::int16_t> rows(static_cast<std::size_t>(tokens) * top_k, -1);
    std::vector<int> ids(experts);
    for (int t = 0; t < tokens; ++t) {
        if (t % 3 == 2) continue;
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        for (int k = 0; k < top_k; ++k)
            rows[t * top_k + k] = static_cast<std::int16_t>(ids[k]);
    }
    return rows;
}

}  // namespace

TEST_CASE("moe_topk_replay of the engine's own selection changes nothing", "[moe][replay]") {
    const int tokens = 37, experts = 64, top_k = 8;
    std::mt19937 rng(7);
    auto probs = random_probs(tokens, experts, rng);
    Device<float> d_probs(probs);
    Device<int> d_idx(std::vector<int>(tokens * top_k, 0));
    Device<float> d_w(std::vector<float>(tokens * top_k, 0.0f));
    moe_topk_forward(d_idx.ptr, d_w.ptr, d_probs.ptr, nullptr, tokens, experts, top_k, true, false, false, false, 0.0f, 0);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    auto idx0 = d_idx.read(tokens * top_k);
    auto w0 = d_w.read(tokens * top_k);

    std::vector<std::int16_t> rows(idx0.begin(), idx0.end());
    Device<std::int16_t> d_rows(rows);
    Device<int> d_bad(std::vector<int>(1, 0));
    moe_topk_replay(d_idx.ptr, d_w.ptr, d_probs.ptr, d_rows.ptr, tokens, experts, top_k, true, false, false, d_bad.ptr, 0);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    auto idx1 = d_idx.read(tokens * top_k);
    auto w1 = d_w.read(tokens * top_k);
    REQUIRE(idx1 == idx0);
    for (int i = 0; i < tokens * top_k; ++i)
        REQUIRE(w1[i] == Catch::Approx(w0[i]).epsilon(1e-6));
    REQUIRE(d_bad.read(1)[0] == 0);
}

TEST_CASE("moe_topk_replay installs the rollout's experts with recomputed weights", "[moe][replay]") {
    const int tokens = 41, experts = 256, top_k = 8;
    std::mt19937 rng(11);
    auto probs = random_probs(tokens, experts, rng);
    auto rows = random_replay(tokens, experts, top_k, rng);

    for (const bool softmax : {false, true}) {
        Device<float> d_probs(probs);
        Device<int> d_idx(std::vector<int>(tokens * top_k, 0));
        Device<float> d_w(std::vector<float>(tokens * top_k, 0.0f));
        // softmax=true treats the scores as logits and normalizes over the selection (as moe_topk_forward does).
        moe_topk_forward(d_idx.ptr, d_w.ptr, d_probs.ptr, nullptr, tokens, experts, top_k, true, softmax, false, false,
                         0.0f, 0);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        auto idx_fwd = d_idx.read(tokens * top_k);
        auto w_fwd = d_w.read(tokens * top_k);

        Device<std::int16_t> d_rows(rows);
        Device<int> d_bad(std::vector<int>(1, 0));
        moe_topk_replay(d_idx.ptr, d_w.ptr, d_probs.ptr, d_rows.ptr, tokens, experts, top_k, true, softmax, false,
                        d_bad.ptr, 0);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        auto idx = d_idx.read(tokens * top_k);
        auto w = d_w.read(tokens * top_k);
        REQUIRE(d_bad.read(1)[0] == 0);

        for (int t = 0; t < tokens; ++t) {
            if (rows[t * top_k] < 0) {  // not replayed: the computed selection is kept
                for (int k = 0; k < top_k; ++k) {
                    REQUIRE(idx[t * top_k + k] == idx_fwd[t * top_k + k]);
                    REQUIRE(w[t * top_k + k] == Catch::Approx(w_fwd[t * top_k + k]).epsilon(1e-6));
                }
                continue;
            }
            std::vector<double> v(top_k);
            for (int k = 0; k < top_k; ++k) {
                REQUIRE(idx[t * top_k + k] == rows[t * top_k + k]);
                v[k] = probs[t * experts + rows[t * top_k + k]];
            }
            if (softmax) {
                const double m = *std::max_element(v.begin(), v.end());
                double s = 0.0;
                for (auto& x : v) s += std::exp(x - m);
                for (auto& x : v) x = std::exp(x - m) / s;
            } else {
                const double s = std::accumulate(v.begin(), v.end(), 0.0);
                for (auto& x : v) x /= s;
            }
            for (int k = 0; k < top_k; ++k)
                REQUIRE(w[t * top_k + k] == Catch::Approx(v[k]).epsilon(1e-5));
        }
    }
}

TEST_CASE("moe_topk_replay rejects out-of-range expert ids", "[moe][replay]") {
    const int tokens = 4, experts = 16, top_k = 2;
    std::mt19937 rng(3);
    auto probs = random_probs(tokens, experts, rng);
    Device<float> d_probs(probs);
    Device<int> d_idx(std::vector<int>(tokens * top_k, 0));
    Device<float> d_w(std::vector<float>(tokens * top_k, 0.0f));
    moe_topk_forward(d_idx.ptr, d_w.ptr, d_probs.ptr, nullptr, tokens, experts, top_k, true, false, false, false, 0.0f, 0);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    auto idx_fwd = d_idx.read(tokens * top_k);

    std::vector<std::int16_t> rows = {3, 5, 1, 300, -1, -1, 7, 2};  // token 1 has id 300 >= experts
    Device<std::int16_t> d_rows(rows);
    Device<int> d_bad(std::vector<int>(1, 0));
    moe_topk_replay(d_idx.ptr, d_w.ptr, d_probs.ptr, d_rows.ptr, tokens, experts, top_k, true, false, false, d_bad.ptr, 0);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    auto idx = d_idx.read(tokens * top_k);
    REQUIRE(d_bad.read(1)[0] == 1);
    REQUIRE(idx[0] == 3);
    REQUIRE(idx[1] == 5);
    REQUIRE(idx[2] == idx_fwd[2]);  // bad row untouched
    REQUIRE(idx[3] == idx_fwd[3]);
    REQUIRE(idx[4] == idx_fwd[4]);  // -1 row untouched
    REQUIRE(idx[6] == 7);
    REQUIRE(idx[7] == 2);
}

TEST_CASE("moe_topk_backward routes gradient to exactly the replayed experts", "[moe][replay]") {
    const int tokens = 9, experts = 32, top_k = 4;
    std::mt19937 rng(5);
    auto probs = random_probs(tokens, experts, rng);
    auto rows = random_replay(tokens, experts, top_k, rng);
    Device<float> d_probs(probs);
    Device<int> d_idx(std::vector<int>(tokens * top_k, 0));
    Device<float> d_w(std::vector<float>(tokens * top_k, 0.0f));
    moe_topk_forward(d_idx.ptr, d_w.ptr, d_probs.ptr, nullptr, tokens, experts, top_k, true, false, false, false, 0.0f, 0);
    Device<std::int16_t> d_rows(rows);
    moe_topk_replay(d_idx.ptr, d_w.ptr, d_probs.ptr, d_rows.ptr, tokens, experts, top_k, true, false, false, nullptr, 0);

    std::vector<float> d_weights(tokens * top_k);
    for (int i = 0; i < tokens * top_k; ++i)
        d_weights[i] = 0.1f * static_cast<float>((i % 7) - 3);
    Device<float> d_dw(d_weights);
    Device<float> d_dprobs(std::vector<float>(tokens * experts, 0.0f));
    moe_topk_backward(d_dprobs.ptr, d_dw.ptr, d_probs.ptr, d_idx.ptr, tokens, experts, top_k, true, false, false, 0);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    auto idx = d_idx.read(tokens * top_k);
    auto dprobs = d_dprobs.read(tokens * experts);

    for (int t = 0; t < tokens; ++t) {
        double S = 0.0, dot = 0.0;
        for (int k = 0; k < top_k; ++k) {
            S += probs[t * experts + idx[t * top_k + k]];
        }
        for (int k = 0; k < top_k; ++k) {
            dot += d_weights[t * top_k + k] * probs[t * experts + idx[t * top_k + k]];
        }
        std::vector<double> expected(experts, 0.0);
        for (int k = 0; k < top_k; ++k) {
            expected[idx[t * top_k + k]] = (d_weights[t * top_k + k] * S - dot) / (S * S);
        }
        if (rows[t * top_k] >= 0) {
            for (int k = 0; k < top_k; ++k)
                REQUIRE(idx[t * top_k + k] == rows[t * top_k + k]);
        }
        for (int e = 0; e < experts; ++e)
            REQUIRE(dprobs[t * experts + e] == Catch::Approx(expected[e]).margin(1e-5));
    }
}
