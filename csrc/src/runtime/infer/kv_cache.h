// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// KV-Cache for autoregressive generation.
//
// Backed by DeviceMemoryStack (the mutable arena) — no separate allocation.
// Supports BF16 (all GPUs) and FP8 E4M3 (SM89+) KV storage.
// Static contiguous layout: no PagedAttention needed for closed-loop GRPO.

#ifndef SUROGATE_SRC_RUNTIME_INFER_KV_CACHE_H
#define SUROGATE_SRC_RUNTIME_INFER_KV_CACHE_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "utilities/stack.h"
#include "utilities/tensor.h"
#include "utilities/dtype.h"

namespace infer {

/// KV-cache data type for decode generation.
enum class KVDType {
    BF16,       // nv_bfloat16 — works on all GPUs (SM80+)
    FP8_E4M3,   // __nv_fp8_e4m3 — halves memory, SM89+ only
};

/// Static contiguous KV-cache backed by arena memory.
///
/// Two separate buffers for K and V, each with layout:
///   [num_layers, batch_size, max_seq_len, num_kv_heads, head_dim]
///
/// This layout makes per-layer K and V contiguous across the batch dimension,
/// which is what FlashAttention decode expects. Each batch item's K/V occupies
/// [max_seq_len, num_kv_heads, head_dim] contiguous memory.
///
/// FP8 path: K/V stored as FP8 E4M3 with per-head per-position scales.
///   Scale layout: [num_layers, batch_size, max_seq_len, num_kv_heads] per K/V
struct KVCache {
    // ========================================================================
    // Configuration (set at construction, immutable)
    // ========================================================================
    int num_layers = 0;
    int batch_size = 0;
    int max_seq_len = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    KVDType dtype = KVDType::BF16;

    // ========================================================================
    // Arena-backed storage (populated by allocate())
    // ========================================================================
    std::byte* k_data = nullptr;        // K cache: [num_layers, batch_size, max_seq_len, Hkv, Hs]
    std::byte* v_data = nullptr;        // V cache: [num_layers, batch_size, max_seq_len, Hkv, Hs]
    float* k_scales = nullptr;          // FP8 K scales (nullptr for BF16)
    float* v_scales = nullptr;          // FP8 V scales (nullptr for BF16)
    std::size_t per_buffer_bytes = 0;   // Bytes for one K or V buffer
    std::size_t per_scale_bytes = 0;    // Bytes for one scale buffer

    // ========================================================================
    // Per-sequence state
    // ========================================================================
    std::vector<int> seq_lens;          // Current length per sequence [batch_size]

    // ========================================================================
    // Construction
    // ========================================================================

    KVCache() = default;

    /// Create a KV-cache with given configuration.
    /// Does NOT allocate — call allocate() with an arena.
    KVCache(int num_layers_, int batch_size_, int max_seq_len_,
            int num_kv_heads_, int head_dim_, KVDType dtype_)
        : num_layers(num_layers_)
        , batch_size(batch_size_)
        , max_seq_len(max_seq_len_)
        , num_kv_heads(num_kv_heads_)
        , head_dim(head_dim_)
        , dtype(dtype_)
        , seq_lens(static_cast<std::size_t>(batch_size_), 0) {}

    // ========================================================================
    // Memory management
    // ========================================================================

    /// Bytes per KV element (1 for FP8, 2 for BF16).
    int elem_bytes() const {
        return dtype == KVDType::FP8_E4M3 ? 1 : 2;
    }

    /// Bytes per single K or V buffer.
    std::size_t compute_per_buffer_bytes() const {
        return static_cast<std::size_t>(num_layers)
             * static_cast<std::size_t>(batch_size)
             * static_cast<std::size_t>(max_seq_len)
             * static_cast<std::size_t>(num_kv_heads)
             * static_cast<std::size_t>(head_dim)
             * static_cast<std::size_t>(elem_bytes());
    }

    /// Bytes for one FP8 scale buffer (0 for BF16).
    std::size_t compute_per_scale_bytes() const {
        if (dtype != KVDType::FP8_E4M3) return 0;
        return static_cast<std::size_t>(num_layers)
             * static_cast<std::size_t>(batch_size)
             * static_cast<std::size_t>(max_seq_len)
             * static_cast<std::size_t>(num_kv_heads)
             * sizeof(float);
    }

    /// Total arena bytes needed (K + V data + scales).
    std::size_t total_bytes() const {
        return 2 * compute_per_buffer_bytes() + 2 * compute_per_scale_bytes();
    }

    /// Allocate from arena. Does NOT zero the memory.
    void allocate(DeviceMemoryStack& arena) {
        per_buffer_bytes = compute_per_buffer_bytes();
        k_data = arena.allocate(per_buffer_bytes, "kv_cache_k");
        v_data = arena.allocate(per_buffer_bytes, "kv_cache_v");

        per_scale_bytes = compute_per_scale_bytes();
        if (per_scale_bytes > 0) {
            k_scales = reinterpret_cast<float*>(arena.allocate(per_scale_bytes, "kv_cache_k_scales"));
            v_scales = reinterpret_cast<float*>(arena.allocate(per_scale_bytes, "kv_cache_v_scales"));
        }
    }

    /// Reset all sequence lengths to 0 (reuse the same memory).
    void reset() {
        std::fill(seq_lens.begin(), seq_lens.end(), 0);
    }

    // ========================================================================
    // Addressing — O(1) pointer arithmetic
    // ========================================================================

    /// Elements per batch item: max_seq_len * num_kv_heads * head_dim
    std::size_t batch_stride_elems() const {
        return static_cast<std::size_t>(max_seq_len)
             * static_cast<std::size_t>(num_kv_heads)
             * static_cast<std::size_t>(head_dim);
    }

    /// Elements per layer: batch_size * batch_stride_elems
    std::size_t layer_stride_elems() const {
        return static_cast<std::size_t>(batch_size) * batch_stride_elems();
    }

    /// Byte offset for K[layer][batch_idx][0][0][0]
    std::size_t k_offset_bytes(int layer, int batch_idx = 0) const {
        return (static_cast<std::size_t>(layer) * layer_stride_elems()
              + static_cast<std::size_t>(batch_idx) * batch_stride_elems())
             * static_cast<std::size_t>(elem_bytes());
    }

    /// Byte offset for V[layer][batch_idx][0][0][0]
    std::size_t v_offset_bytes(int layer, int batch_idx = 0) const {
        return k_offset_bytes(layer, batch_idx);  // Same layout, different base pointer
    }

    /// K cache pointer for (layer, batch_idx=0): [batch_size, max_seq_len, Hkv, Hs]
    void* k_layer_ptr(int layer) { return k_data + k_offset_bytes(layer); }
    void* v_layer_ptr(int layer) { return v_data + v_offset_bytes(layer); }

    /// K cache pointer for (layer, batch_idx): [max_seq_len, Hkv, Hs]
    void* k_ptr(int layer, int batch_idx) { return k_data + k_offset_bytes(layer, batch_idx); }
    void* v_ptr(int layer, int batch_idx) { return v_data + v_offset_bytes(layer, batch_idx); }

    /// K/V scales for layer (nullptr for BF16)
    float* k_scale_layer(int layer) {
        if (!k_scales) return nullptr;
        std::size_t scale_layer_stride = static_cast<std::size_t>(batch_size)
                                       * static_cast<std::size_t>(max_seq_len)
                                       * static_cast<std::size_t>(num_kv_heads);
        return k_scales + static_cast<std::size_t>(layer) * scale_layer_stride;
    }
    float* v_scale_layer(int layer) {
        if (!v_scales) return nullptr;
        std::size_t scale_layer_stride = static_cast<std::size_t>(batch_size)
                                       * static_cast<std::size_t>(max_seq_len)
                                       * static_cast<std::size_t>(num_kv_heads);
        return v_scales + static_cast<std::size_t>(layer) * scale_layer_stride;
    }
};

// ============================================================================
// Paged KV-cache for GRPO multi-prompt batching with shared prefix pages.
//
// Instead of allocating [batch_size, max_seq_len] per sequence, memory is
// divided into fixed-size pages. A block table maps each sequence's virtual
// page indices to physical pages in a flat page pool. Multiple sequences
// can share the same physical prefix pages (pointer aliasing).
//
// Memory savings for M prompts × N completions:
//   Contiguous: M*N * max_total_len * Hkv * Hs * elem_bytes  (per K or V)
//   Paged:      (M * prefix_pages + M*N * suffix_pages) * page_block_size * Hkv * Hs * elem_bytes
//   Savings:    M*(N-1) * prefix_pages * page_block_size * Hkv * Hs * elem_bytes
// ============================================================================

/// Paged KV-cache backed by arena memory.
///
/// Page pool layout per buffer (K or V):
///   [num_layers, total_pages, page_block_size, num_kv_heads, head_dim]
///
/// Block table layout:
///   [batch_size, max_pages_per_seq]  — on GPU
///   Maps virtual page index → physical page index for each sequence.
///
/// For GRPO: sequences in the same prompt group share prefix pages.
///   block_table[m*N+n][0..prefix_pages-1] all point to the same physical pages.
struct PagedKVCache {
    // Configuration
    int num_layers = 0;
    int batch_size = 0;          // M*N total sequences
    int num_kv_heads = 0;
    int head_dim = 0;
    int page_block_size = 256;   // Tokens per page (aligned to Flash Attention tile)
    int max_pages_per_seq = 0;   // ceil(max_total_len / page_block_size)
    int total_pages = 0;         // Total physical pages allocated
    KVDType dtype = KVDType::BF16;

    // GRPO structure
    int num_prompts = 0;         // M
    int num_completions = 0;     // N

    // Arena-backed storage
    std::byte* k_pages = nullptr;   // K page pool [num_layers, total_pages, page_block_size, Hkv, Hs]
    std::byte* v_pages = nullptr;   // V page pool
    float* k_scales = nullptr;      // FP8 K scales [num_layers, total_pages, page_block_size, Hkv]
    float* v_scales = nullptr;      // FP8 V scales [num_layers, total_pages, page_block_size, Hkv]
    int* block_table_gpu = nullptr; // [batch_size, max_pages_per_seq] on GPU

    // Host state
    std::vector<int> block_table_host;  // Host mirror of block table
    int next_free_page = 0;             // Next unallocated page index
    std::vector<int> seq_lens;          // Current length per sequence [batch_size]

    PagedKVCache() = default;

    /// Bytes per KV element.
    int elem_bytes() const {
        return dtype == KVDType::FP8_E4M3 ? 1 : 2;
    }

    /// Elements per page: page_block_size * Hkv * Hs
    std::size_t page_elems() const {
        return static_cast<std::size_t>(page_block_size)
             * static_cast<std::size_t>(num_kv_heads)
             * static_cast<std::size_t>(head_dim);
    }

    /// Bytes per page pool (one K or V): num_layers * total_pages * page_elems * elem_bytes
    std::size_t per_pool_bytes() const {
        return static_cast<std::size_t>(num_layers)
             * static_cast<std::size_t>(total_pages)
             * page_elems()
             * static_cast<std::size_t>(elem_bytes());
    }

    /// Bytes for one physical page in one layer.
    std::size_t page_bytes() const {
        return page_elems() * static_cast<std::size_t>(elem_bytes());
    }

    /// Bytes for block table on GPU.
    std::size_t block_table_bytes() const {
        return static_cast<std::size_t>(batch_size)
             * static_cast<std::size_t>(max_pages_per_seq)
             * sizeof(int);
    }

    /// Bytes for one FP8 scale pool (0 for BF16).
    std::size_t per_scale_pool_bytes() const {
        if (dtype != KVDType::FP8_E4M3) return 0;
        return static_cast<std::size_t>(num_layers)
             * static_cast<std::size_t>(total_pages)
             * static_cast<std::size_t>(page_block_size)
             * static_cast<std::size_t>(num_kv_heads)
             * sizeof(float);
    }

    /// Total arena bytes needed (K pool + V pool + block table).
    std::size_t total_bytes() const {
        return 2 * per_pool_bytes() + 2 * per_scale_pool_bytes() + block_table_bytes();
    }

    /// Configure for GRPO: M prompts × N completions, with given prompt/gen lengths.
    void configure(int M, int N, int num_layers_, int max_prefix_len, int max_gen_len,
                   int num_kv_heads_, int head_dim_, int page_size, KVDType dtype_) {
        num_prompts = M;
        num_completions = N;
        batch_size = M * N;
        num_layers = num_layers_;
        num_kv_heads = num_kv_heads_;
        head_dim = head_dim_;
        page_block_size = page_size;
        dtype = dtype_;

        const int max_total_len = max_prefix_len + max_gen_len;
        max_pages_per_seq = (max_total_len + page_block_size - 1) / page_block_size;

        // Page budget:
        // - Prefix: full pages are shared across N completions.
        // - Partial last prefix page (if any) is private per completion to
        //   avoid cross-completion writes when generation continues in-page.
        // Reserve this private-page overhead pessimistically for every prompt
        // because prompt lengths are ragged and some prompts may end on a
        // partial page even when max_prefix_len is page-aligned.
        const int shared_prefix_pages_per_prompt = (max_prefix_len + page_block_size - 1) / page_block_size;
        const int private_partial_extra_per_prompt =
            (max_prefix_len > 0 && N > 1) ? (N - 1) : 0;
        const int prefix_pages_per_prompt = shared_prefix_pages_per_prompt + private_partial_extra_per_prompt;
        const int suffix_pages_per_seq = (max_gen_len + page_block_size - 1) / page_block_size;
        total_pages = M * prefix_pages_per_prompt + M * N * suffix_pages_per_seq;

        // Initialize host state
        block_table_host.assign(
            static_cast<std::size_t>(batch_size) * max_pages_per_seq, -1);
        seq_lens.assign(static_cast<std::size_t>(batch_size), 0);
        next_free_page = 0;
    }

    /// Allocate page pool and block table from arena.
    void allocate(DeviceMemoryStack& arena) {
        k_pages = arena.allocate(per_pool_bytes(), "paged_kv_k_pool");
        v_pages = arena.allocate(per_pool_bytes(), "paged_kv_v_pool");
        if (dtype == KVDType::FP8_E4M3) {
            k_scales = reinterpret_cast<float*>(arena.allocate(per_scale_pool_bytes(), "paged_kv_k_scales"));
            v_scales = reinterpret_cast<float*>(arena.allocate(per_scale_pool_bytes(), "paged_kv_v_scales"));
        } else {
            k_scales = nullptr;
            v_scales = nullptr;
        }
        block_table_gpu = reinterpret_cast<int*>(
            arena.allocate(block_table_bytes(), "paged_kv_block_table"));
    }

    /// Allocate a physical page. Returns page index.
    int alloc_page() {
        if (next_free_page >= total_pages) {
            throw std::runtime_error("PagedKVCache: out of pages");
        }
        return next_free_page++;
    }

    /// Set block table entry: block_table[seq_idx][virtual_page] = physical_page.
    void set_block_table(int seq_idx, int virtual_page, int physical_page) {
        if (seq_idx < 0 || seq_idx >= batch_size) {
            throw std::runtime_error("PagedKVCache: seq_idx out of range");
        }
        if (virtual_page < 0 || virtual_page >= max_pages_per_seq) {
            throw std::runtime_error("PagedKVCache: virtual_page out of range");
        }
        block_table_host[static_cast<std::size_t>(seq_idx) * max_pages_per_seq + virtual_page] = physical_page;
    }

    /// Get block table entry.
    int get_block_table(int seq_idx, int virtual_page) const {
        if (seq_idx < 0 || seq_idx >= batch_size) {
            throw std::runtime_error("PagedKVCache: seq_idx out of range");
        }
        if (virtual_page < 0 || virtual_page >= max_pages_per_seq) {
            throw std::runtime_error("PagedKVCache: virtual_page out of range");
        }
        return block_table_host[static_cast<std::size_t>(seq_idx) * max_pages_per_seq + virtual_page];
    }

    /// Allocate prefix pages for a prompt and assign to source slot.
    /// Returns number of pages allocated.
    int alloc_prefix_pages(int prompt_idx, int prefix_len) {
        const int npages = (prefix_len + page_block_size - 1) / page_block_size;
        const int src_slot = prompt_idx * num_completions;
        for (int p = 0; p < npages; ++p) {
            int phys = alloc_page();
            set_block_table(src_slot, p, phys);
        }
        return npages;
    }

    /// Share prefix pages: point all N sequences in a prompt group to the same prefix pages.
    void share_prefix_pages(int prompt_idx, int prefix_pages) {
        const int src_slot = prompt_idx * num_completions;
        for (int n = 1; n < num_completions; ++n) {
            const int dst_slot = src_slot + n;
            for (int p = 0; p < prefix_pages; ++p) {
                set_block_table(dst_slot, p, get_block_table(src_slot, p));
            }
        }
    }

    /// Allocate a suffix page for a sequence and append to its block table.
    /// virtual_page is the next page index after prefix pages.
    void alloc_suffix_page(int seq_idx, int virtual_page) {
        int phys = alloc_page();
        set_block_table(seq_idx, virtual_page, phys);
    }

    /// Upload block table to GPU (call after all pages are allocated/shared).
    void upload_block_table(cudaStream_t stream) {
        CUDA_CHECK(cudaMemcpyAsync(
            block_table_gpu, block_table_host.data(),
            block_table_bytes(), cudaMemcpyHostToDevice, stream));
    }

    /// Copy the prefix token region from one physical page to another for all layers.
    /// Used to materialize private copies of a partially-filled shared prefix page.
    void copy_prefix_tokens_between_pages(
        int src_physical_page, int dst_physical_page, int prefix_tokens, cudaStream_t stream) {
        if (prefix_tokens <= 0) return;
        if (src_physical_page < 0 || dst_physical_page < 0) {
            throw std::runtime_error("PagedKVCache: invalid page index for copy");
        }
        if (prefix_tokens > page_block_size) {
            throw std::runtime_error("PagedKVCache: prefix_tokens exceeds page_block_size");
        }

        const std::size_t tokens_bytes =
            static_cast<std::size_t>(prefix_tokens)
            * static_cast<std::size_t>(num_kv_heads)
            * static_cast<std::size_t>(head_dim)
            * static_cast<std::size_t>(elem_bytes());
        const std::size_t per_layer_bytes =
            static_cast<std::size_t>(total_pages) * page_bytes();
        const std::size_t src_off = static_cast<std::size_t>(src_physical_page) * page_bytes();
        const std::size_t dst_off = static_cast<std::size_t>(dst_physical_page) * page_bytes();
        const std::size_t scale_tokens_bytes =
            static_cast<std::size_t>(prefix_tokens)
            * static_cast<std::size_t>(num_kv_heads)
            * sizeof(float);
        const std::size_t per_layer_scale_bytes =
            static_cast<std::size_t>(total_pages)
            * static_cast<std::size_t>(page_block_size)
            * static_cast<std::size_t>(num_kv_heads)
            * sizeof(float);
        const std::size_t src_scale_off =
            static_cast<std::size_t>(src_physical_page)
            * static_cast<std::size_t>(page_block_size)
            * static_cast<std::size_t>(num_kv_heads)
            * sizeof(float);
        const std::size_t dst_scale_off =
            static_cast<std::size_t>(dst_physical_page)
            * static_cast<std::size_t>(page_block_size)
            * static_cast<std::size_t>(num_kv_heads)
            * sizeof(float);

        for (int layer = 0; layer < num_layers; ++layer) {
            auto* k_layer = k_pages + static_cast<std::size_t>(layer) * per_layer_bytes;
            auto* v_layer = v_pages + static_cast<std::size_t>(layer) * per_layer_bytes;

            CUDA_CHECK(cudaMemcpyAsync(
                k_layer + dst_off, k_layer + src_off, tokens_bytes,
                cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                v_layer + dst_off, v_layer + src_off, tokens_bytes,
                cudaMemcpyDeviceToDevice, stream));

            if (dtype == KVDType::FP8_E4M3 && k_scales && v_scales) {
                auto* ks_layer = reinterpret_cast<std::byte*>(k_scales)
                    + static_cast<std::size_t>(layer) * per_layer_scale_bytes;
                auto* vs_layer = reinterpret_cast<std::byte*>(v_scales)
                    + static_cast<std::size_t>(layer) * per_layer_scale_bytes;
                CUDA_CHECK(cudaMemcpyAsync(
                    ks_layer + dst_scale_off, ks_layer + src_scale_off, scale_tokens_bytes,
                    cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(
                    vs_layer + dst_scale_off, vs_layer + src_scale_off, scale_tokens_bytes,
                    cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

    /// Get K page pool pointer for a specific layer.
    /// Returns pointer to [total_pages, page_block_size, Hkv, Hs].
    void* k_pool_layer(int layer) {
        return k_pages + static_cast<std::size_t>(layer)
                       * static_cast<std::size_t>(total_pages)
                       * page_elems()
                       * static_cast<std::size_t>(elem_bytes());
    }

    void* v_pool_layer(int layer) {
        return v_pages + static_cast<std::size_t>(layer)
                       * static_cast<std::size_t>(total_pages)
                       * page_elems()
                       * static_cast<std::size_t>(elem_bytes());
    }

    void reset() {
        std::fill(seq_lens.begin(), seq_lens.end(), 0);
        std::fill(block_table_host.begin(), block_table_host.end(), -1);
        next_free_page = 0;
    }
};

}  // namespace infer

#endif  // SUROGATE_SRC_RUNTIME_INFER_KV_CACHE_H
