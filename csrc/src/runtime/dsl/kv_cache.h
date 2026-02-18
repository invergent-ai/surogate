// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// KV-cache for inference (used by GRPO online generation).

#ifndef SUROGATE_SRC_DSL_KV_CACHE_H
#define SUROGATE_SRC_DSL_KV_CACHE_H

#include <cstddef>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "utilities/utils.h"

namespace dsl {

/// Per-layer key/value cache for autoregressive inference.
///
/// Layout per layer: (B, max_seq_len, Hkv, HS) — time-major, BF16.
/// This matches the (B, Hkv, T_kv, HS) view used by cuDNN SDPA via
/// non-contiguous strides: batch_stride = max_seq_len * Hkv * HS.
///
/// Memory is allocated via raw cudaMalloc (not TensorAllocator) so that
/// it can be freed independently of the training allocator.
struct KVCache {
    int num_layers  = 0;
    int B           = 0;
    int max_seq_len = 0;
    int Hkv         = 0;
    int HS          = 0;
    int current_pos = 0;   ///< number of tokens currently in cache
    bool is_decode  = false; ///< true once prefill is done, set before decode steps

    /// Per-layer GPU pointers. k_layers[l] / v_layers[l] point to
    /// a (B, max_seq_len, Hkv, HS) BF16 buffer, or nullptr if the layer
    /// has no attention (never written to).
    std::vector<nv_bfloat16*> k_layers;
    std::vector<nv_bfloat16*> v_layers;

    /// Allocate GPU memory for all layers. Previous allocations are freed first.
    void allocate(int num_layers, int B, int max_seq, int Hkv, int HS) {
        free_memory();
        this->num_layers  = num_layers;
        this->B           = B;
        this->max_seq_len = max_seq;
        this->Hkv         = Hkv;
        this->HS          = HS;
        this->current_pos = 0;
        this->is_decode   = false;

        k_layers.resize(num_layers, nullptr);
        v_layers.resize(num_layers, nullptr);

        const std::size_t layer_bytes =
            static_cast<std::size_t>(B) * max_seq * Hkv * HS * sizeof(nv_bfloat16);

        for (int l = 0; l < num_layers; ++l) {
            CUDA_CHECK(cudaMalloc(&k_layers[l], layer_bytes));
            CUDA_CHECK(cudaMalloc(&v_layers[l], layer_bytes));
            // Zero-initialize so unused positions are harmless.
            CUDA_CHECK(cudaMemset(k_layers[l], 0, layer_bytes));
            CUDA_CHECK(cudaMemset(v_layers[l], 0, layer_bytes));
        }
    }

    /// Reset position counter (reuse memory for a new sequence).
    void reset() {
        current_pos = 0;
        is_decode   = false;
    }

    /// Set current position without clearing the cache contents.
    ///
    /// Used to restart decode from an earlier position, e.g. to generate G
    /// completions from the same prompt: after prefill stores the prompt KV,
    /// call set_current_pos(prompt_len) before each new completion so the
    /// prompt's cached K/V is reused while the completion positions are
    /// overwritten.
    void set_current_pos(int pos) {
        current_pos = pos;
        is_decode   = (pos > 0);
    }

    /// Free all GPU allocations.
    void free_memory() {
        for (auto* p : k_layers) { if (p) cudaFree(p); }
        for (auto* p : v_layers) { if (p) cudaFree(p); }
        k_layers.clear();
        v_layers.clear();
        num_layers  = 0;
        B           = 0;
        max_seq_len = 0;
        Hkv         = 0;
        HS          = 0;
        current_pos = 0;
        is_decode   = false;
    }

    ~KVCache() { free_memory(); }

    /// Bytes per layer per K (or V) buffer.
    std::size_t layer_bytes() const {
        return static_cast<std::size_t>(B) * max_seq_len * Hkv * HS * sizeof(nv_bfloat16);
    }
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_KV_CACHE_H
