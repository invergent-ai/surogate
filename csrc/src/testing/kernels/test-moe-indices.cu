// MoE dispatch bookkeeping kernels: the parallel gather/scatter index
// construction must reproduce the single-thread reference bit for bit, and the
// warp-per-token routing statistics must match a double-precision host model.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

struct IndexOutputs {
    std::vector<int> gather, scatter, positions;
};

// Runs either construction from the same initial state the runtime prepares:
// gather zero-filled, scatter 0xFF-filled, positions zero-filled.
IndexOutputs build(const std::vector<int>& expert_indices, int num_tokens, int top_k, int num_experts, bool serial) {
    const int total = num_tokens * top_k;
    std::vector<int> counts(num_experts, 0);
    for (int e : expert_indices) {
        if (e >= 0 && e < num_experts) counts[e]++;
    }
    std::vector<int> offsets(num_experts + 1, 0);
    for (int e = 0; e < num_experts; ++e)
        offsets[e + 1] = offsets[e] + counts[e];

    Device<int> d_indices(expert_indices);
    Device<int> d_offsets(offsets);
    Device<int> d_positions(std::vector<int>(num_experts, 0));
    Device<int> d_gather(std::vector<int>(total, 0));
    Device<int> d_scatter(std::vector<int>(total, -1));

    if (serial) {
        moe_build_indices_serial(d_gather.ptr,
                                 d_scatter.ptr,
                                 d_indices.ptr,
                                 d_offsets.ptr,
                                 d_positions.ptr,
                                 num_tokens,
                                 top_k,
                                 num_experts,
                                 0);
    } else {
        moe_build_indices(d_gather.ptr,
                          d_scatter.ptr,
                          d_indices.ptr,
                          d_offsets.ptr,
                          d_positions.ptr,
                          num_tokens,
                          top_k,
                          num_experts,
                          0);
    }
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    return {d_gather.read(total), d_scatter.read(total), d_positions.read(num_experts)};
}

void check_indices(int num_tokens, int top_k, int num_experts, double invalid_fraction, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> expert(0, num_experts - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::vector<int> expert_indices(static_cast<std::size_t>(num_tokens) * top_k);
    for (int& e : expert_indices) {
        e = expert(rng);
        if (coin(rng) < invalid_fraction) e = (coin(rng) < 0.5) ? -1 : num_experts + expert(rng);
    }
    const IndexOutputs reference = build(expert_indices, num_tokens, top_k, num_experts, /*serial=*/true);
    const IndexOutputs parallel = build(expert_indices, num_tokens, top_k, num_experts, /*serial=*/false);
    REQUIRE(parallel.gather == reference.gather);
    REQUIRE(parallel.scatter == reference.scatter);
    REQUIRE(parallel.positions == reference.positions);

    // The reference itself: expert-major, assignment-index-minor, invalid skipped.
    int valid = 0;
    for (int e : expert_indices)
        valid += (e >= 0 && e < num_experts);
    for (int pos = 1; pos < valid; ++pos) {
        const int a = reference.gather[pos - 1], b = reference.gather[pos];
        const int ea = expert_indices[a], eb = expert_indices[b];
        REQUIRE((ea < eb || (ea == eb && a < b)));
    }
}

}  // namespace

TEST_CASE("moe_build_indices reproduces the single-thread ordering", "[moe][indices]") {
    check_indices(/*num_tokens=*/1, /*top_k=*/1, /*num_experts=*/1, 0.0, 1);
    check_indices(7, 2, 4, 0.0, 2);
    check_indices(300, 8, 128, 0.0, 3);
    check_indices(4352, 8, 128, 0.0, 4);      // one production microbatch
    check_indices(4 * 4352, 8, 128, 0.0, 5);  // H200 microbatch of four sequences
    check_indices(2048, 8, 128, 0.05, 6);     // invalid ids stay zero / -1
    check_indices(1000, 6, 257, 0.0, 7);      // experts above 256, non-power-of-two
    check_indices(513, 8, 128, 1.0, 8);       // every assignment invalid
}

TEST_CASE("moe_build_indices workspace query is stable", "[moe][indices]") {
    REQUIRE(moe_build_indices_workspace_bytes(0) == 0);
    REQUIRE(moe_build_indices_workspace_bytes(4352 * 8) > 0);
    REQUIRE(moe_build_indices_workspace_bytes(4352 * 8) == moe_build_indices_workspace_bytes(4352 * 8));
}

namespace {

// Host model of the statistics in double precision (stats layout of DslRunState).
std::vector<double> host_stats(const std::vector<float>& logits,
                               const std::vector<int>& expert_indices,
                               int num_tokens,
                               int num_experts,
                               int top_k,
                               float aux_loss_coef,
                               float z_loss_coef) {
    const int total = num_tokens * top_k;
    std::vector<double> counts(num_experts, 0.0), probs(num_experts, 0.0);
    for (int e : expert_indices) {
        if (e >= 0 && e < num_experts) counts[e] += 1.0;
    }
    double entropy_sum = 0.0, confidence_sum = 0.0, z_sum = 0.0;
    const double entropy_denom = (num_experts > 1) ? std::log(static_cast<double>(num_experts)) : 1.0;
    for (int t = 0; t < num_tokens; ++t) {
        const float* row = logits.data() + static_cast<std::size_t>(t) * num_experts;
        double max_logit = -INFINITY;
        for (int e = 0; e < num_experts; ++e)
            max_logit = std::max(max_logit, static_cast<double>(row[e]));
        double denom = 0.0;
        for (int e = 0; e < num_experts; ++e)
            denom += std::exp(row[e] - max_logit);
        const double lse = max_logit + std::log(denom);
        double entropy = 0.0, max_prob = 0.0;
        for (int e = 0; e < num_experts; ++e) {
            const double p = std::exp(row[e] - max_logit) / denom;
            if (p > 0.0) entropy -= p * std::log(p);
            max_prob = std::max(max_prob, p);
            probs[e] += p / num_tokens;
        }
        entropy_sum += entropy / entropy_denom;
        confidence_sum += max_prob;
        z_sum += z_loss_coef * lse * lse / std::max(num_tokens, 1);
    }
    double aux = 0.0, max_count = 0.0, min_active = total + 1.0, sq = 0.0;
    int active = 0;
    const double mean_count = static_cast<double>(total) / num_experts;
    for (int e = 0; e < num_experts; ++e) {
        aux += (counts[e] / total) * probs[e];
        max_count = std::max(max_count, counts[e]);
        sq += counts[e] * counts[e];
        if (counts[e] > 0.0) {
            min_active = std::min(min_active, counts[e]);
            active++;
        }
    }
    aux *= num_experts * aux_loss_coef;
    const double variance = std::max(sq / num_experts - mean_count * mean_count, 0.0);
    std::vector<double> stats(11, 0.0);
    stats[0] = aux;
    stats[1] = z_sum;
    stats[2] = static_cast<double>(active) / num_experts;
    stats[3] = max_count / mean_count;
    stats[4] = 1.0;
    stats[5] = active;
    stats[6] = max_count / total;
    stats[7] = (active > 0) ? min_active / total : 0.0;
    stats[8] = std::sqrt(variance) / mean_count;
    stats[9] = entropy_sum / num_tokens;
    stats[10] = confidence_sum / num_tokens;
    return stats;
}

template <class T>
void check_stats(int num_tokens, int num_experts, int top_k, float aux_loss_coef, float z_loss_coef, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> logit(0.0f, 2.0f);
    std::uniform_int_distribution<int> expert(0, num_experts - 1);
    std::vector<float> logits_f(static_cast<std::size_t>(num_tokens) * num_experts);
    std::vector<T> logits_t(logits_f.size());
    for (std::size_t i = 0; i < logits_f.size(); ++i) {
        logits_t[i] = T(logit(rng));
        logits_f[i] = static_cast<float>(logits_t[i]);  // the exact values the kernel sees
    }
    std::vector<int> expert_indices(static_cast<std::size_t>(num_tokens) * top_k);
    for (int& e : expert_indices)
        e = expert(rng);

    Device<T> d_logits(logits_t);
    Device<int> d_indices(expert_indices);
    Device<float> d_stats(std::vector<float>(16, 0.0f));
    moe_compute_routing_stats_from_logits(d_stats.ptr,
                                          d_logits.ptr,
                                          d_indices.ptr,
                                          num_tokens,
                                          num_experts,
                                          top_k,
                                          aux_loss_coef,
                                          z_loss_coef,
                                          0);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const std::vector<float> got = d_stats.read(11);
    const std::vector<double> want =
        host_stats(logits_f, expert_indices, num_tokens, num_experts, top_k, aux_loss_coef, z_loss_coef);
    for (int i = 0; i < 11; ++i) {
        INFO("stat " << i);
        REQUIRE(got[i] == Catch::Approx(want[i]).epsilon(2e-3).margin(2e-4));
    }
}

}  // namespace

TEST_CASE("moe routing statistics match the host model", "[moe][stats]") {
    check_stats<float>(64, 8, 2, 0.01f, 0.001f, 11);
    check_stats<float>(4352, 128, 8, 0.0f, 0.0f, 12);  // production: coefficients zero
    check_stats<nv_bfloat16>(4352, 128, 8, 0.02f, 0.001f, 13);
    check_stats<nv_bfloat16>(1000, 257, 4, 0.02f, 0.0f, 14);  // 16-experts-per-lane path
    check_stats<float>(3, 600, 2, 0.02f, 0.001f, 15);         // fallback path (> 512 experts)
}
