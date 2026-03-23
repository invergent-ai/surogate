// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Page pool with free-list for continuous batching KV-cache management.
//
// Unlike the GRPO PagedKVCache (monotonic allocation, freed at session close),
// this pool supports O(1) per-page allocation and deallocation via a stack-based
// free-list.  Finished sequences return their pages immediately so new requests
// can reuse them.

#ifndef SUROGATE_SRC_RUNTIME_INFER_PAGE_POOL_H
#define SUROGATE_SRC_RUNTIME_INFER_PAGE_POOL_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/infer/kv_cache.h"   // KVDType
#include "utilities/stack.h"

namespace infer {

/// Page pool with free-list for continuous batching.
///
/// GPU memory layout (identical to PagedKVCache):
///   K pool: [num_layers, total_pages, page_block_size, Hkv, Hs]
///   V pool: [num_layers, total_pages, page_block_size, Hkv, Hs]
///   FP8 scales: [num_layers, total_pages, page_block_size, Hkv] per K/V
///
/// The pool is allocated once from the arena.  Individual pages are handed out
/// and returned via the free-list (stack-based, O(1) alloc/free).
class PagePool {
public:
    PagePool() = default;

    /// Configure and allocate the pool.
    ///
    /// @param total_pages      Number of physical pages in the pool.
    /// @param page_block_size  Tokens per page (typically 256).
    /// @param num_layers       Number of model layers with KV-cache.
    /// @param num_kv_heads     Number of KV heads.
    /// @param head_dim         Head dimension.
    /// @param dtype            BF16 or FP8_E4M3.
    /// @param arena            Arena to allocate GPU memory from.
    void init(int total_pages, int page_block_size, int num_layers,
              int num_kv_heads, int head_dim, KVDType dtype,
              DeviceMemoryStack& arena) {
        total_pages_ = total_pages;
        page_block_size_ = page_block_size;
        num_layers_ = num_layers;
        num_kv_heads_ = num_kv_heads;
        head_dim_ = head_dim;
        dtype_ = dtype;

        const int elem_bytes = (dtype == KVDType::FP8_E4M3) ? 1 : 2;
        const std::size_t page_elems =
            static_cast<std::size_t>(page_block_size) * num_kv_heads * head_dim;
        per_layer_pool_bytes_ =
            static_cast<std::size_t>(total_pages) * page_elems * elem_bytes;
        const std::size_t pool_bytes =
            static_cast<std::size_t>(num_layers) * per_layer_pool_bytes_;

        k_pages_ = arena.allocate(pool_bytes, "page_pool_k");
        v_pages_ = arena.allocate(pool_bytes, "page_pool_v");

        if (dtype == KVDType::FP8_E4M3) {
            const std::size_t scale_elems =
                static_cast<std::size_t>(num_layers) * total_pages
                * page_block_size * num_kv_heads;
            k_scales_ = reinterpret_cast<float*>(
                arena.allocate(scale_elems * sizeof(float), "page_pool_k_scales"));
            v_scales_ = reinterpret_cast<float*>(
                arena.allocate(scale_elems * sizeof(float), "page_pool_v_scales"));
        }

        // Initialize free-list: all pages available, in reverse order so that
        // page 0 is popped first (nicer for debugging).
        free_list_.resize(total_pages);
        for (int i = 0; i < total_pages; ++i) {
            free_list_[i] = total_pages - 1 - i;
        }
    }

    /// Allocate a single page.  Returns page index, or -1 if exhausted.
    int allocate_page() {
        if (free_list_.empty()) return -1;
        const int page_id = free_list_.back();
        free_list_.pop_back();
        return page_id;
    }

    /// Return a single page to the pool.
    void free_page(int page_id) {
        free_list_.push_back(page_id);
    }

    /// Allocate `count` pages.  Returns the page IDs.
    /// Throws if not enough pages available.
    std::vector<int> allocate_pages(int count) {
        if (count <= 0) return {};
        if (static_cast<int>(free_list_.size()) < count) {
            throw std::runtime_error(
                "PagePool: not enough free pages (" +
                std::to_string(free_list_.size()) + " available, " +
                std::to_string(count) + " requested)");
        }
        std::vector<int> result(count);
        for (int i = 0; i < count; ++i) {
            result[i] = free_list_.back();
            free_list_.pop_back();
        }
        return result;
    }

    /// Return multiple pages to the pool.
    void free_pages(const std::vector<int>& page_ids) {
        for (int id : page_ids) {
            free_list_.push_back(id);
        }
    }

    int num_free() const { return static_cast<int>(free_list_.size()); }
    int total_pages() const { return total_pages_; }
    int page_block_size() const { return page_block_size_; }
    int num_layers() const { return num_layers_; }
    int num_kv_heads() const { return num_kv_heads_; }
    int head_dim() const { return head_dim_; }
    KVDType dtype() const { return dtype_; }

    std::byte* k_pages() { return k_pages_; }
    std::byte* v_pages() { return v_pages_; }
    float* k_scales() { return k_scales_; }
    float* v_scales() { return v_scales_; }

    /// Bytes for one layer's page pool (one K or V buffer).
    std::size_t per_layer_pool_bytes() const { return per_layer_pool_bytes_; }

    /// Total bytes for one full pool (K or V, all layers).
    std::size_t full_pool_bytes() const {
        return static_cast<std::size_t>(num_layers_) * per_layer_pool_bytes_;
    }

private:
    int total_pages_ = 0;
    int page_block_size_ = 256;
    int num_layers_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    KVDType dtype_ = KVDType::BF16;
    std::size_t per_layer_pool_bytes_ = 0;

    std::byte* k_pages_ = nullptr;
    std::byte* v_pages_ = nullptr;
    float* k_scales_ = nullptr;
    float* v_scales_ = nullptr;

    std::vector<int> free_list_;
};

}  // namespace infer

#endif  // SUROGATE_SRC_RUNTIME_INFER_PAGE_POOL_H
