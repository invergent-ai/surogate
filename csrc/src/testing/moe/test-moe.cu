// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for Mixture of Experts (MoE) CUDA kernels

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "kernels/kernels.h"
#include "utilities/utils.h"
#include "../utilities/test_config.h"
#include "../utilities/test_utils.h"

using namespace testing_utils;

namespace {

// ============================================================================
// CPU Reference Implementations
// ============================================================================

// CPU softmax over last dimension
static void softmax_cpu(float* out, const float* inp, int num_tokens, int num_experts) {
    for (int t = 0; t < num_tokens; ++t) {
        const float* row_in = inp + t * num_experts;
        float* row_out = out + t * num_experts;

        // Find max for numerical stability
        float max_val = row_in[0];
        for (int e = 1; e < num_experts; ++e) {
            max_val = std::max(max_val, row_in[e]);
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            row_out[e] = std::exp(row_in[e] - max_val);
            sum += row_out[e];
        }

        // Normalize
        for (int e = 0; e < num_experts; ++e) {
            row_out[e] /= sum;
        }
    }
}

// CPU softmax backward
static void softmax_backward_cpu(float* d_logits, const float* d_probs,
                                  const float* softmax_probs, int num_tokens, int num_experts) {
    for (int t = 0; t < num_tokens; ++t) {
        const float* d_p = d_probs + t * num_experts;
        const float* p = softmax_probs + t * num_experts;
        float* d_l = d_logits + t * num_experts;

        // Compute sum(d_probs * softmax_probs)
        float dot = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            dot += d_p[e] * p[e];
        }

        // d_logits = softmax_probs * (d_probs - dot)
        for (int e = 0; e < num_experts; ++e) {
            d_l[e] = p[e] * (d_p[e] - dot);
        }
    }
}

// CPU sigmoid
static void sigmoid_cpu(float* out, const float* inp, int num_elements) {
    for (int i = 0; i < num_elements; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-inp[i]));
    }
}

// CPU top-K selection per row
static void topk_cpu(int* indices, float* weights, const float* scores,
                     int num_tokens, int num_experts, int top_k, bool normalize) {
    for (int t = 0; t < num_tokens; ++t) {
        const float* row = scores + t * num_experts;
        int* idx_out = indices + t * top_k;
        float* w_out = weights + t * top_k;

        // Create index-value pairs
        std::vector<std::pair<float, int>> pairs(num_experts);
        for (int e = 0; e < num_experts; ++e) {
            pairs[e] = {row[e], e};
        }

        // Partial sort to get top-k
        std::partial_sort(pairs.begin(), pairs.begin() + top_k, pairs.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        // Extract top-k
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            idx_out[k] = pairs[k].second;
            w_out[k] = pairs[k].first;
            sum += w_out[k];
        }

        // Normalize if requested
        if (normalize && sum > 0.0f) {
            for (int k = 0; k < top_k; ++k) {
                w_out[k] /= sum;
            }
        }
    }
}

// CPU expert counts
static void compute_expert_counts_cpu(int* counts, const int* expert_indices,
                                       int num_tokens, int top_k, int num_experts) {
    std::fill(counts, counts + num_experts, 0);
    for (int i = 0; i < num_tokens * top_k; ++i) {
        int expert_id = expert_indices[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            counts[expert_id]++;
        }
    }
}

// CPU token permutation
static void permute_tokens_cpu(float* out, const float* inp, const int* gather_indices,
                                int total_tokens, int num_tokens, int hidden_size, int top_k) {
    for (int out_idx = 0; out_idx < total_tokens; ++out_idx) {
        int token_assignment_idx = gather_indices[out_idx];
        int token_idx = token_assignment_idx / top_k;

        const float* src = inp + token_idx * hidden_size;
        float* dst = out + out_idx * hidden_size;

        for (int d = 0; d < hidden_size; ++d) {
            dst[d] = src[d];
        }
    }
}

// CPU unpermute and combine
static void unpermute_and_combine_cpu(float* out, const float* expert_out,
                                       const float* routing_weights, const int* scatter_indices,
                                       int num_tokens, int total_tokens, int hidden_size, int top_k) {
    // Zero output
    std::fill(out, out + num_tokens * hidden_size, 0.0f);

    for (int t = 0; t < num_tokens; ++t) {
        float* dst = out + t * hidden_size;
        const float* weights = routing_weights + t * top_k;

        for (int k = 0; k < top_k; ++k) {
            int assignment_idx = t * top_k + k;
            int expert_pos = scatter_indices[assignment_idx];
            const float* exp_out = expert_out + expert_pos * hidden_size;
            float weight = weights[k];

            for (int d = 0; d < hidden_size; ++d) {
                dst[d] += weight * exp_out[d];
            }
        }
    }
}

// CPU auxiliary loss
static float compute_aux_loss_cpu(const float* routing_probs, const int* expert_indices,
                                   int num_tokens, int num_experts, int top_k, float aux_loss_coef) {
    // Compute expert fractions (tokens assigned / total assignments)
    std::vector<float> expert_fractions(num_experts, 0.0f);
    int total_assignments = num_tokens * top_k;
    for (int i = 0; i < total_assignments; ++i) {
        int expert_id = expert_indices[i];
        if (expert_id >= 0 && expert_id < num_experts) {
            expert_fractions[expert_id] += 1.0f / total_assignments;
        }
    }

    // Compute average routing probability per expert
    std::vector<float> expert_probs(num_experts, 0.0f);
    for (int t = 0; t < num_tokens; ++t) {
        for (int e = 0; e < num_experts; ++e) {
            expert_probs[e] += routing_probs[t * num_experts + e] / num_tokens;
        }
    }

    // Load balancing loss
    float loss = 0.0f;
    for (int e = 0; e < num_experts; ++e) {
        loss += expert_fractions[e] * expert_probs[e];
    }
    loss *= num_experts * aux_loss_coef;

    return loss;
}

// Create simple gather indices for testing (sequential assignment)
static void create_gather_indices(int* gather_indices, int* scatter_indices,
                                   const int* expert_indices, const int* expert_counts,
                                   int num_tokens, int top_k, int num_experts) {
    // Compute expert offsets (cumsum of counts)
    std::vector<int> expert_offsets(num_experts + 1, 0);
    for (int e = 0; e < num_experts; ++e) {
        expert_offsets[e + 1] = expert_offsets[e] + expert_counts[e];
    }

    // Track current position within each expert's region
    std::vector<int> expert_positions(num_experts, 0);

    // Build indices
    int total_tokens = num_tokens * top_k;
    for (int idx = 0; idx < total_tokens; ++idx) {
        int expert_id = expert_indices[idx];
        if (expert_id >= 0 && expert_id < num_experts) {
            int dest_idx = expert_offsets[expert_id] + expert_positions[expert_id];
            gather_indices[dest_idx] = idx;
            scatter_indices[idx] = dest_idx;
            expert_positions[expert_id]++;
        }
    }
}

static float max_abs_diff(const float* a, const float* b, size_t n) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

} // namespace

// ============================================================================
// Test Cases
// ============================================================================

TEST_CASE("moe_softmax_forward fp32 matches CPU", "[moe][softmax]") {
    const int num_tokens = 64;
    const int num_experts = 8;
    const size_t n = num_tokens * num_experts;

    // Generate input data
    std::vector<float> h_inp = uniform_host(n, -2.0f, 2.0f, 1234ull);

    // CPU reference
    std::vector<float> h_out_cpu(n);
    softmax_cpu(h_out_cpu.data(), h_inp.data(), num_tokens, num_experts);

    // GPU computation
    thrust::device_vector<float> d_inp = to_device(h_inp);
    thrust::device_vector<float> d_out(n);

    moe_softmax_forward(
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_inp.data()),
        num_tokens, num_experts, 0
    );

    std::vector<float> h_out = from_device(d_out);

    // Verify
    for (size_t i = 0; i < n; ++i) {
        REQUIRE(h_out[i] == Catch::Approx(h_out_cpu[i]).margin(1e-5f));
    }

    // Verify each row sums to 1
    for (int t = 0; t < num_tokens; ++t) {
        float sum = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            sum += h_out[t * num_experts + e];
        }
        REQUIRE(sum == Catch::Approx(1.0f).margin(1e-5f));
    }
}

TEST_CASE("moe_softmax_forward bf16 matches CPU", "[moe][softmax]") {
    const int num_tokens = 64;
    const int num_experts = 8;
    const size_t n = num_tokens * num_experts;

    // Generate input data
    std::vector<float> h_inp_f = uniform_host(n, -2.0f, 2.0f, 1234ull);
    std::vector<nv_bfloat16> h_inp_bf16 = to_bf16(h_inp_f);
    std::vector<float> h_inp_q = round_bf16(h_inp_f);

    // CPU reference (on quantized input)
    std::vector<float> h_out_cpu(n);
    softmax_cpu(h_out_cpu.data(), h_inp_q.data(), num_tokens, num_experts);
    std::vector<float> h_out_cpu_q = round_bf16(h_out_cpu);

    // GPU computation
    thrust::device_vector<nv_bfloat16> d_inp = to_device(h_inp_bf16);
    thrust::device_vector<nv_bfloat16> d_out(n);

    moe_softmax_forward(
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_inp.data()),
        num_tokens, num_experts, 0
    );

    std::vector<nv_bfloat16> h_out_bf16 = from_device(d_out);

    // Verify with bf16 tolerance
    for (size_t i = 0; i < n; ++i) {
        uint16_t bits;
        std::memcpy(&bits, &h_out_bf16[i], sizeof(bits));
        float h_out_val = bf16_bits_to_float(bits);
        REQUIRE(h_out_val == Catch::Approx(h_out_cpu_q[i]).margin(5e-2f));
    }
}

TEST_CASE("moe_sigmoid_forward fp32 matches CPU", "[moe][sigmoid]") {
    const int num_elements = 512;

    // Generate input data
    std::vector<float> h_inp = uniform_host(num_elements, -5.0f, 5.0f, 5678ull);

    // CPU reference
    std::vector<float> h_out_cpu(num_elements);
    sigmoid_cpu(h_out_cpu.data(), h_inp.data(), num_elements);

    // GPU computation
    thrust::device_vector<float> d_inp = to_device(h_inp);
    thrust::device_vector<float> d_out(num_elements);

    moe_sigmoid_forward(
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_inp.data()),
        num_elements, 0
    );

    std::vector<float> h_out = from_device(d_out);

    // Verify
    for (int i = 0; i < num_elements; ++i) {
        REQUIRE(h_out[i] == Catch::Approx(h_out_cpu[i]).margin(1e-5f));
    }
}

TEST_CASE("moe_topk_forward fp32 matches CPU", "[moe][topk]") {
    const int num_tokens = 32;
    const int num_experts = 8;
    const int top_k = 2;
    const bool normalize = true;
    const size_t n_scores = num_tokens * num_experts;
    const size_t n_topk = num_tokens * top_k;

    // Generate softmax-like scores (positive, sum to ~1)
    std::vector<float> h_scores = uniform_host(n_scores, 0.0f, 1.0f, 9012ull);
    // Normalize each row to sum to 1 (like softmax output)
    for (int t = 0; t < num_tokens; ++t) {
        float sum = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            sum += h_scores[t * num_experts + e];
        }
        for (int e = 0; e < num_experts; ++e) {
            h_scores[t * num_experts + e] /= sum;
        }
    }

    // CPU reference
    std::vector<int> h_indices_cpu(n_topk);
    std::vector<float> h_weights_cpu(n_topk);
    topk_cpu(h_indices_cpu.data(), h_weights_cpu.data(), h_scores.data(),
             num_tokens, num_experts, top_k, normalize);

    // GPU computation
    thrust::device_vector<float> d_scores = to_device(h_scores);
    thrust::device_vector<int> d_indices(n_topk);
    thrust::device_vector<float> d_weights(n_topk);

    moe_topk_forward(
        thrust::raw_pointer_cast(d_indices.data()),
        thrust::raw_pointer_cast(d_weights.data()),
        thrust::raw_pointer_cast(d_scores.data()),
        nullptr,  // no correction bias
        num_tokens, num_experts, top_k, normalize, 0
    );

    std::vector<int> h_indices = from_device(d_indices);
    std::vector<float> h_weights = from_device(d_weights);

    // Verify indices match (order within top-k may vary if tied, so check set membership)
    for (int t = 0; t < num_tokens; ++t) {
        std::vector<int> cpu_set(h_indices_cpu.begin() + t * top_k,
                                  h_indices_cpu.begin() + (t + 1) * top_k);
        std::vector<int> gpu_set(h_indices.begin() + t * top_k,
                                  h_indices.begin() + (t + 1) * top_k);
        std::sort(cpu_set.begin(), cpu_set.end());
        std::sort(gpu_set.begin(), gpu_set.end());
        REQUIRE(cpu_set == gpu_set);
    }

    // Verify weights are normalized (sum to 1) if normalize=true
    if (normalize) {
        for (int t = 0; t < num_tokens; ++t) {
            float sum = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                sum += h_weights[t * top_k + k];
            }
            REQUIRE(sum == Catch::Approx(1.0f).margin(1e-4f));
        }
    }
}

TEST_CASE("moe_compute_expert_counts matches CPU", "[moe][counts]") {
    const int num_tokens = 64;
    const int num_experts = 8;
    const int top_k = 2;
    const size_t n_topk = num_tokens * top_k;

    // Generate random expert indices
    std::vector<int> h_indices(n_topk);
    std::mt19937 gen(3456);
    std::uniform_int_distribution<int> dist(0, num_experts - 1);
    for (auto& idx : h_indices) {
        idx = dist(gen);
    }

    // CPU reference
    std::vector<int> h_counts_cpu(num_experts);
    compute_expert_counts_cpu(h_counts_cpu.data(), h_indices.data(),
                               num_tokens, top_k, num_experts);

    // GPU computation
    thrust::device_vector<int> d_indices = to_device(h_indices);
    thrust::device_vector<int> d_counts(num_experts);

    moe_compute_expert_counts(
        thrust::raw_pointer_cast(d_counts.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        num_tokens, top_k, num_experts, 0
    );

    std::vector<int> h_counts = from_device(d_counts);

    // Verify
    for (int e = 0; e < num_experts; ++e) {
        REQUIRE(h_counts[e] == h_counts_cpu[e]);
    }

    // Verify total count
    int total_gpu = std::accumulate(h_counts.begin(), h_counts.end(), 0);
    REQUIRE(total_gpu == num_tokens * top_k);
}

TEST_CASE("moe_permute_tokens fp32 matches CPU", "[moe][permute]") {
    const int num_tokens = 32;
    const int num_experts = 4;
    const int top_k = 2;
    const int hidden_size = 64;
    const int total_tokens = num_tokens * top_k;

    // Generate input hidden states
    std::vector<float> h_inp = uniform_host(num_tokens * hidden_size, -1.0f, 1.0f, 7890ull);

    // Generate random expert assignments
    std::vector<int> h_expert_indices(total_tokens);
    std::mt19937 gen(1111);
    std::uniform_int_distribution<int> dist(0, num_experts - 1);
    for (auto& idx : h_expert_indices) {
        idx = dist(gen);
    }

    // Compute expert counts and create gather/scatter indices
    std::vector<int> h_expert_counts(num_experts, 0);
    compute_expert_counts_cpu(h_expert_counts.data(), h_expert_indices.data(),
                               num_tokens, top_k, num_experts);

    std::vector<int> h_gather_indices(total_tokens);
    std::vector<int> h_scatter_indices(total_tokens);
    create_gather_indices(h_gather_indices.data(), h_scatter_indices.data(),
                           h_expert_indices.data(), h_expert_counts.data(),
                           num_tokens, top_k, num_experts);

    // CPU reference
    std::vector<float> h_out_cpu(total_tokens * hidden_size);
    permute_tokens_cpu(h_out_cpu.data(), h_inp.data(), h_gather_indices.data(),
                        total_tokens, num_tokens, hidden_size, top_k);

    // GPU computation
    thrust::device_vector<float> d_inp = to_device(h_inp);
    thrust::device_vector<int> d_gather_indices = to_device(h_gather_indices);
    thrust::device_vector<float> d_out(total_tokens * hidden_size);

    moe_permute_tokens(
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_inp.data()),
        thrust::raw_pointer_cast(d_gather_indices.data()),
        total_tokens, num_tokens, hidden_size, top_k, 0
    );

    std::vector<float> h_out = from_device(d_out);

    // Verify
    float max_diff = max_abs_diff(h_out.data(), h_out_cpu.data(), total_tokens * hidden_size);
    REQUIRE(max_diff < 1e-5f);
}

TEST_CASE("moe_unpermute_and_combine fp32 matches CPU", "[moe][combine]") {
    const int num_tokens = 32;
    const int num_experts = 4;
    const int top_k = 2;
    const int hidden_size = 64;
    const int total_tokens = num_tokens * top_k;

    // Generate expert outputs
    std::vector<float> h_expert_out = uniform_host(total_tokens * hidden_size, -1.0f, 1.0f, 2222ull);

    // Generate routing weights (normalized)
    std::vector<float> h_routing_weights = uniform_host(num_tokens * top_k, 0.1f, 0.9f, 3333ull);
    // Normalize per token
    for (int t = 0; t < num_tokens; ++t) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            sum += h_routing_weights[t * top_k + k];
        }
        for (int k = 0; k < top_k; ++k) {
            h_routing_weights[t * top_k + k] /= sum;
        }
    }

    // Generate random expert assignments
    std::vector<int> h_expert_indices(total_tokens);
    std::mt19937 gen(4444);
    std::uniform_int_distribution<int> dist(0, num_experts - 1);
    for (auto& idx : h_expert_indices) {
        idx = dist(gen);
    }

    // Compute expert counts and create indices
    std::vector<int> h_expert_counts(num_experts, 0);
    compute_expert_counts_cpu(h_expert_counts.data(), h_expert_indices.data(),
                               num_tokens, top_k, num_experts);

    std::vector<int> h_gather_indices(total_tokens);
    std::vector<int> h_scatter_indices(total_tokens);
    create_gather_indices(h_gather_indices.data(), h_scatter_indices.data(),
                           h_expert_indices.data(), h_expert_counts.data(),
                           num_tokens, top_k, num_experts);

    // CPU reference
    std::vector<float> h_out_cpu(num_tokens * hidden_size);
    unpermute_and_combine_cpu(h_out_cpu.data(), h_expert_out.data(),
                               h_routing_weights.data(), h_scatter_indices.data(),
                               num_tokens, total_tokens, hidden_size, top_k);

    // GPU computation
    thrust::device_vector<float> d_expert_out = to_device(h_expert_out);
    thrust::device_vector<float> d_routing_weights = to_device(h_routing_weights);
    thrust::device_vector<int> d_scatter_indices = to_device(h_scatter_indices);
    thrust::device_vector<float> d_out(num_tokens * hidden_size);

    moe_unpermute_and_combine(
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_expert_out.data()),
        thrust::raw_pointer_cast(d_routing_weights.data()),
        thrust::raw_pointer_cast(d_scatter_indices.data()),
        num_tokens, total_tokens, hidden_size, top_k, 0
    );

    std::vector<float> h_out = from_device(d_out);

    // Verify
    float max_diff = max_abs_diff(h_out.data(), h_out_cpu.data(), num_tokens * hidden_size);
    REQUIRE(max_diff < 1e-4f);
}

TEST_CASE("moe_compute_aux_loss fp32 matches CPU", "[moe][auxloss]") {
    const int num_tokens = 64;
    const int num_experts = 8;
    const int top_k = 2;
    const float aux_loss_coef = 0.01f;
    const size_t n_probs = num_tokens * num_experts;
    const size_t n_topk = num_tokens * top_k;

    // Generate routing probabilities (normalized per token)
    std::vector<float> h_probs = uniform_host(n_probs, 0.0f, 1.0f, 5555ull);
    for (int t = 0; t < num_tokens; ++t) {
        float sum = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            sum += h_probs[t * num_experts + e];
        }
        for (int e = 0; e < num_experts; ++e) {
            h_probs[t * num_experts + e] /= sum;
        }
    }

    // Generate expert indices
    std::vector<int> h_indices(n_topk);
    std::mt19937 gen(6666);
    std::uniform_int_distribution<int> dist(0, num_experts - 1);
    for (auto& idx : h_indices) {
        idx = dist(gen);
    }

    // CPU reference
    float cpu_loss = compute_aux_loss_cpu(h_probs.data(), h_indices.data(),
                                           num_tokens, num_experts, top_k, aux_loss_coef);

    // GPU computation
    thrust::device_vector<float> d_probs = to_device(h_probs);
    thrust::device_vector<int> d_indices = to_device(h_indices);
    thrust::device_vector<float> d_loss(1);

    moe_compute_aux_loss(
        thrust::raw_pointer_cast(d_loss.data()),
        thrust::raw_pointer_cast(d_probs.data()),
        thrust::raw_pointer_cast(d_indices.data()),
        num_tokens, num_experts, top_k, aux_loss_coef, 0
    );

    cudaDeviceSynchronize();
    std::vector<float> h_loss = from_device(d_loss);

    // Verify (allow some tolerance due to atomics and float accumulation)
    REQUIRE(h_loss[0] == Catch::Approx(cpu_loss).margin(1e-4f));
}

TEST_CASE("moe_softmax_backward fp32 matches CPU", "[moe][softmax]") {
    const int num_tokens = 64;
    const int num_experts = 8;
    const size_t n = num_tokens * num_experts;

    // Generate input logits
    std::vector<float> h_logits = uniform_host(n, -2.0f, 2.0f, 7777ull);

    // Compute softmax for forward
    std::vector<float> h_probs(n);
    softmax_cpu(h_probs.data(), h_logits.data(), num_tokens, num_experts);

    // Generate upstream gradient
    std::vector<float> h_d_probs = uniform_host(n, -0.5f, 0.5f, 8888ull);

    // CPU backward reference
    std::vector<float> h_d_logits_cpu(n);
    softmax_backward_cpu(h_d_logits_cpu.data(), h_d_probs.data(), h_probs.data(),
                          num_tokens, num_experts);

    // GPU computation
    thrust::device_vector<float> d_probs = to_device(h_probs);
    thrust::device_vector<float> d_d_probs = to_device(h_d_probs);
    thrust::device_vector<float> d_d_logits(n);

    moe_softmax_backward(
        thrust::raw_pointer_cast(d_d_logits.data()),
        thrust::raw_pointer_cast(d_d_probs.data()),
        thrust::raw_pointer_cast(d_probs.data()),
        num_tokens, num_experts, 0
    );

    std::vector<float> h_d_logits = from_device(d_d_logits);

    // Verify
    float max_diff = max_abs_diff(h_d_logits.data(), h_d_logits_cpu.data(), n);
    REQUIRE(max_diff < 1e-5f);
}

TEST_CASE("moe full forward pass integration", "[moe][integration]") {
    // Test the full MoE forward pass: input -> router -> permute -> (expert placeholder) -> combine -> output
    const int num_tokens = 32;
    const int num_experts = 4;
    const int top_k = 2;
    const int hidden_size = 64;
    const int total_tokens = num_tokens * top_k;

    // Step 1: Generate input hidden states
    std::vector<float> h_input = uniform_host(num_tokens * hidden_size, -1.0f, 1.0f, 1001ull);

    // Step 2: Generate routing logits and compute softmax
    std::vector<float> h_logits = uniform_host(num_tokens * num_experts, -2.0f, 2.0f, 1002ull);
    std::vector<float> h_probs(num_tokens * num_experts);
    softmax_cpu(h_probs.data(), h_logits.data(), num_tokens, num_experts);

    // Step 3: Top-K selection
    std::vector<int> h_expert_indices(total_tokens);
    std::vector<float> h_routing_weights(total_tokens);
    topk_cpu(h_expert_indices.data(), h_routing_weights.data(), h_probs.data(),
             num_tokens, num_experts, top_k, true);

    // Step 4: Compute expert counts and create indices
    std::vector<int> h_expert_counts(num_experts);
    compute_expert_counts_cpu(h_expert_counts.data(), h_expert_indices.data(),
                               num_tokens, top_k, num_experts);

    std::vector<int> h_gather_indices(total_tokens);
    std::vector<int> h_scatter_indices(total_tokens);
    create_gather_indices(h_gather_indices.data(), h_scatter_indices.data(),
                           h_expert_indices.data(), h_expert_counts.data(),
                           num_tokens, top_k, num_experts);

    // Step 5: Permute tokens
    std::vector<float> h_permuted(total_tokens * hidden_size);
    permute_tokens_cpu(h_permuted.data(), h_input.data(), h_gather_indices.data(),
                        total_tokens, num_tokens, hidden_size, top_k);

    // Step 6: Simulate expert computation (identity for now)
    std::vector<float> h_expert_out = h_permuted;  // Identity expert

    // Step 7: Unpermute and combine
    std::vector<float> h_output_cpu(num_tokens * hidden_size);
    unpermute_and_combine_cpu(h_output_cpu.data(), h_expert_out.data(),
                               h_routing_weights.data(), h_scatter_indices.data(),
                               num_tokens, total_tokens, hidden_size, top_k);

    // Now run the GPU version
    thrust::device_vector<float> d_input = to_device(h_input);
    thrust::device_vector<float> d_logits = to_device(h_logits);
    thrust::device_vector<float> d_probs(num_tokens * num_experts);
    thrust::device_vector<int> d_expert_indices(total_tokens);
    thrust::device_vector<float> d_routing_weights(total_tokens);
    thrust::device_vector<int> d_expert_counts(num_experts);
    thrust::device_vector<int> d_gather_indices = to_device(h_gather_indices);
    thrust::device_vector<int> d_scatter_indices = to_device(h_scatter_indices);
    thrust::device_vector<float> d_permuted(total_tokens * hidden_size);
    thrust::device_vector<float> d_expert_out(total_tokens * hidden_size);
    thrust::device_vector<float> d_output(num_tokens * hidden_size);

    // GPU softmax
    moe_softmax_forward(
        thrust::raw_pointer_cast(d_probs.data()),
        thrust::raw_pointer_cast(d_logits.data()),
        num_tokens, num_experts, 0
    );

    // GPU top-k
    moe_topk_forward(
        thrust::raw_pointer_cast(d_expert_indices.data()),
        thrust::raw_pointer_cast(d_routing_weights.data()),
        thrust::raw_pointer_cast(d_probs.data()),
        nullptr,  // no correction bias
        num_tokens, num_experts, top_k, true, 0
    );

    // GPU permute
    moe_permute_tokens(
        thrust::raw_pointer_cast(d_permuted.data()),
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_gather_indices.data()),
        total_tokens, num_tokens, hidden_size, top_k, 0
    );

    // Copy permuted as expert output (identity)
    thrust::copy(d_permuted.begin(), d_permuted.end(), d_expert_out.begin());

    // GPU combine
    moe_unpermute_and_combine(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_expert_out.data()),
        thrust::raw_pointer_cast(d_routing_weights.data()),
        thrust::raw_pointer_cast(d_scatter_indices.data()),
        num_tokens, total_tokens, hidden_size, top_k, 0
    );

    std::vector<float> h_output = from_device(d_output);

    // Verify
    float max_diff = max_abs_diff(h_output.data(), h_output_cpu.data(), num_tokens * hidden_size);
    INFO("Max difference in full forward pass: " << max_diff);
    REQUIRE(max_diff < 1e-3f);
}

// ============================================================================
// Router Z-Loss Tests
// ============================================================================

// CPU reference for logsumexp
static float logsumexp_cpu(const float* logits, int num_experts) {
    float max_val = logits[0];
    for (int e = 1; e < num_experts; ++e) {
        max_val = std::max(max_val, logits[e]);
    }
    float sum = 0.0f;
    for (int e = 0; e < num_experts; ++e) {
        sum += std::exp(logits[e] - max_val);
    }
    return max_val + std::log(sum);
}

// CPU reference for z-loss
static float z_loss_cpu(const float* router_logits, int num_tokens, int num_experts, float z_loss_coef) {
    float z_loss = 0.0f;
    for (int t = 0; t < num_tokens; ++t) {
        float lse = logsumexp_cpu(router_logits + t * num_experts, num_experts);
        z_loss += lse * lse;
    }
    return z_loss_coef * z_loss / num_tokens;
}

// CPU reference for z-loss backward
static void z_loss_backward_cpu(float* d_logits, const float* router_logits,
                                 int num_tokens, int num_experts, float z_loss_coef) {
    for (int t = 0; t < num_tokens; ++t) {
        const float* logits = router_logits + t * num_experts;
        float* d_l = d_logits + t * num_experts;

        // Compute logsumexp
        float lse = logsumexp_cpu(logits, num_experts);

        // Compute softmax
        float max_val = logits[0];
        for (int e = 1; e < num_experts; ++e) {
            max_val = std::max(max_val, logits[e]);
        }
        float sum = 0.0f;
        for (int e = 0; e < num_experts; ++e) {
            sum += std::exp(logits[e] - max_val);
        }

        // d_logits = coef * (2 * lse / num_tokens) * softmax
        float scale = z_loss_coef * 2.0f * lse / num_tokens;
        for (int e = 0; e < num_experts; ++e) {
            float softmax_val = std::exp(logits[e] - max_val) / sum;
            d_l[e] = scale * softmax_val;
        }
    }
}

TEST_CASE("moe_router_z_loss_forward FP32", "[moe][z_loss]") {
    const int num_tokens = 32;
    const int num_experts = 8;
    const float z_loss_coef = 0.01f;

    // Create random router logits
    std::vector<float> h_logits(num_tokens * num_experts);
    for (int i = 0; i < num_tokens * num_experts; ++i) {
        h_logits[i] = 2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f;  // [-1, 1]
    }

    // CPU reference
    float z_loss_cpu_val = z_loss_cpu(h_logits.data(), num_tokens, num_experts, z_loss_coef);

    // GPU
    thrust::device_vector<float> d_logits = to_device(h_logits);
    thrust::device_vector<float> d_z_loss(1, 0.0f);

    moe_router_z_loss_forward(
        thrust::raw_pointer_cast(d_z_loss.data()),
        thrust::raw_pointer_cast(d_logits.data()),
        num_tokens, num_experts, z_loss_coef, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    float z_loss_gpu;
    CUDA_CHECK(cudaMemcpy(&z_loss_gpu, thrust::raw_pointer_cast(d_z_loss.data()),
                          sizeof(float), cudaMemcpyDeviceToHost));

    INFO("CPU z-loss: " << z_loss_cpu_val);
    INFO("GPU z-loss: " << z_loss_gpu);
    REQUIRE(std::abs(z_loss_gpu - z_loss_cpu_val) < 1e-5f);
}

TEST_CASE("moe_router_z_loss_forward BF16", "[moe][z_loss]") {
    const int num_tokens = 64;
    const int num_experts = 16;
    const float z_loss_coef = 0.001f;

    std::vector<float> h_logits(num_tokens * num_experts);
    for (int i = 0; i < num_tokens * num_experts; ++i) {
        h_logits[i] = 2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f;
    }

    // Convert to BF16
    std::vector<nv_bfloat16> h_logits_bf16(num_tokens * num_experts);
    for (int i = 0; i < num_tokens * num_experts; ++i) {
        h_logits_bf16[i] = __float2bfloat16(h_logits[i]);
    }

    // CPU reference (using FP32 logits for reference)
    float z_loss_cpu_val = z_loss_cpu(h_logits.data(), num_tokens, num_experts, z_loss_coef);

    // GPU with BF16
    thrust::device_vector<nv_bfloat16> d_logits(h_logits_bf16.begin(), h_logits_bf16.end());
    thrust::device_vector<float> d_z_loss(1, 0.0f);

    moe_router_z_loss_forward(
        thrust::raw_pointer_cast(d_z_loss.data()),
        thrust::raw_pointer_cast(d_logits.data()),
        num_tokens, num_experts, z_loss_coef, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    float z_loss_gpu;
    CUDA_CHECK(cudaMemcpy(&z_loss_gpu, thrust::raw_pointer_cast(d_z_loss.data()),
                          sizeof(float), cudaMemcpyDeviceToHost));

    INFO("CPU z-loss: " << z_loss_cpu_val);
    INFO("GPU z-loss (BF16): " << z_loss_gpu);
    // BF16 has lower precision, allow larger tolerance
    REQUIRE(std::abs(z_loss_gpu - z_loss_cpu_val) < 1e-2f);
}

TEST_CASE("moe_router_z_loss_backward FP32", "[moe][z_loss]") {
    const int num_tokens = 32;
    const int num_experts = 8;
    const float z_loss_coef = 0.01f;

    std::vector<float> h_logits(num_tokens * num_experts);
    for (int i = 0; i < num_tokens * num_experts; ++i) {
        h_logits[i] = 2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f;
    }

    // CPU reference
    std::vector<float> h_d_logits_cpu(num_tokens * num_experts);
    z_loss_backward_cpu(h_d_logits_cpu.data(), h_logits.data(),
                        num_tokens, num_experts, z_loss_coef);

    // GPU
    thrust::device_vector<float> d_logits = to_device(h_logits);
    thrust::device_vector<float> d_d_logits(num_tokens * num_experts, 0.0f);

    moe_router_z_loss_backward(
        thrust::raw_pointer_cast(d_d_logits.data()),
        thrust::raw_pointer_cast(d_logits.data()),
        num_tokens, num_experts, z_loss_coef, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_d_logits_gpu = from_device(d_d_logits);

    float max_diff = max_abs_diff(h_d_logits_gpu.data(), h_d_logits_cpu.data(), num_tokens * num_experts);
    INFO("Max gradient difference: " << max_diff);
    REQUIRE(max_diff < 1e-5f);
}

TEST_CASE("moe_router_z_loss large logits", "[moe][z_loss]") {
    // Test numerical stability with large logits
    const int num_tokens = 16;
    const int num_experts = 8;
    const float z_loss_coef = 0.01f;

    std::vector<float> h_logits(num_tokens * num_experts);
    for (int i = 0; i < num_tokens * num_experts; ++i) {
        // Large range: [-50, 50] to test numerical stability
        h_logits[i] = 100.0f * (static_cast<float>(rand()) / RAND_MAX) - 50.0f;
    }

    // CPU reference
    float z_loss_cpu_val = z_loss_cpu(h_logits.data(), num_tokens, num_experts, z_loss_coef);

    // GPU
    thrust::device_vector<float> d_logits = to_device(h_logits);
    thrust::device_vector<float> d_z_loss(1, 0.0f);

    moe_router_z_loss_forward(
        thrust::raw_pointer_cast(d_z_loss.data()),
        thrust::raw_pointer_cast(d_logits.data()),
        num_tokens, num_experts, z_loss_coef, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    float z_loss_gpu;
    CUDA_CHECK(cudaMemcpy(&z_loss_gpu, thrust::raw_pointer_cast(d_z_loss.data()),
                          sizeof(float), cudaMemcpyDeviceToHost));

    INFO("CPU z-loss (large logits): " << z_loss_cpu_val);
    INFO("GPU z-loss (large logits): " << z_loss_gpu);
    // Should be numerically stable - relative tolerance
    float rel_diff = std::abs(z_loss_gpu - z_loss_cpu_val) / std::abs(z_loss_cpu_val);
    REQUIRE(rel_diff < 1e-4f);
}

// ============================================================================
// Grouped GEMM Tests
// ============================================================================

TEST_CASE("moe_grouped_gemm_gate_up FP32", "[moe][grouped_gemm]") {
    const int num_experts = 4;
    const int hidden_size = 64;     // C
    const int intermediate_size = 128;  // D
    const int total_tokens = 32;

    // Create expert offsets (uniform distribution for simplicity)
    std::vector<int> h_offsets(num_experts + 1);
    h_offsets[0] = 0;
    for (int e = 0; e < num_experts; ++e) {
        h_offsets[e + 1] = h_offsets[e] + total_tokens / num_experts;
    }
    h_offsets[num_experts] = total_tokens;  // Ensure last offset is total

    // Create random input (total_tokens, C)
    std::vector<float> h_input(total_tokens * hidden_size);
    for (auto& v : h_input) v = 0.1f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);

    // Create random weights (num_experts, 2*D, C)
    std::vector<float> h_weights(num_experts * 2 * intermediate_size * hidden_size);
    for (auto& v : h_weights) v = 0.1f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);

    // CPU reference: sequential per-expert matmul
    std::vector<float> h_output_cpu(total_tokens * 2 * intermediate_size, 0.0f);
    for (int e = 0; e < num_experts; ++e) {
        int start = h_offsets[e];
        int end = h_offsets[e + 1];
        int tokens_e = end - start;
        if (tokens_e == 0) continue;

        const float* w = h_weights.data() + e * (2 * intermediate_size) * hidden_size;

        // output[start:end] = input[start:end] @ weight^T
        for (int t = 0; t < tokens_e; ++t) {
            for (int d = 0; d < 2 * intermediate_size; ++d) {
                float sum = 0.0f;
                for (int c = 0; c < hidden_size; ++c) {
                    sum += h_input[(start + t) * hidden_size + c] * w[d * hidden_size + c];
                }
                h_output_cpu[(start + t) * 2 * intermediate_size + d] = sum;
            }
        }
    }

    // GPU
    thrust::device_vector<float> d_input = to_device(h_input);
    thrust::device_vector<float> d_weights = to_device(h_weights);
    thrust::device_vector<int> d_offsets = to_device(h_offsets);
    thrust::device_vector<float> d_output(total_tokens * 2 * intermediate_size, 0.0f);

    // Create cuBLAS handle
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    moe_grouped_gemm_gate_up(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_weights.data()),
        thrust::raw_pointer_cast(d_offsets.data()),
        num_experts, hidden_size, intermediate_size,
        cublas_handle, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_output_gpu = from_device(d_output);

    float max_diff = max_abs_diff(h_output_gpu.data(), h_output_cpu.data(),
                                   total_tokens * 2 * intermediate_size);
    INFO("Max difference in grouped gate_up GEMM: " << max_diff);
    REQUIRE(max_diff < 1e-4f);

    cublasDestroy(cublas_handle);
}

TEST_CASE("moe_grouped_gemm_down FP32", "[moe][grouped_gemm]") {
    const int num_experts = 4;
    const int hidden_size = 64;     // C
    const int intermediate_size = 128;  // D
    const int total_tokens = 32;

    // Create expert offsets
    std::vector<int> h_offsets(num_experts + 1);
    h_offsets[0] = 0;
    for (int e = 0; e < num_experts; ++e) {
        h_offsets[e + 1] = h_offsets[e] + total_tokens / num_experts;
    }
    h_offsets[num_experts] = total_tokens;

    // Create random input (total_tokens, D)
    std::vector<float> h_input(total_tokens * intermediate_size);
    for (auto& v : h_input) v = 0.1f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);

    // Create random weights (num_experts, C, D)
    std::vector<float> h_weights(num_experts * hidden_size * intermediate_size);
    for (auto& v : h_weights) v = 0.1f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);

    // CPU reference
    std::vector<float> h_output_cpu(total_tokens * hidden_size, 0.0f);
    for (int e = 0; e < num_experts; ++e) {
        int start = h_offsets[e];
        int end = h_offsets[e + 1];
        int tokens_e = end - start;
        if (tokens_e == 0) continue;

        const float* w = h_weights.data() + e * hidden_size * intermediate_size;

        // output[start:end] = input[start:end] @ weight^T
        for (int t = 0; t < tokens_e; ++t) {
            for (int c = 0; c < hidden_size; ++c) {
                float sum = 0.0f;
                for (int d = 0; d < intermediate_size; ++d) {
                    sum += h_input[(start + t) * intermediate_size + d] * w[c * intermediate_size + d];
                }
                h_output_cpu[(start + t) * hidden_size + c] = sum;
            }
        }
    }

    // GPU
    thrust::device_vector<float> d_input = to_device(h_input);
    thrust::device_vector<float> d_weights = to_device(h_weights);
    thrust::device_vector<int> d_offsets = to_device(h_offsets);
    thrust::device_vector<float> d_output(total_tokens * hidden_size, 0.0f);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    moe_grouped_gemm_down(
        thrust::raw_pointer_cast(d_output.data()),
        thrust::raw_pointer_cast(d_input.data()),
        thrust::raw_pointer_cast(d_weights.data()),
        thrust::raw_pointer_cast(d_offsets.data()),
        num_experts, hidden_size, intermediate_size,
        cublas_handle, 0
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_output_gpu = from_device(d_output);

    float max_diff = max_abs_diff(h_output_gpu.data(), h_output_cpu.data(),
                                   total_tokens * hidden_size);
    INFO("Max difference in grouped down GEMM: " << max_diff);
    REQUIRE(max_diff < 1e-4f);

    cublasDestroy(cublas_handle);
}
