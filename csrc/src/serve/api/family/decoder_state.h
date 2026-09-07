#pragma once

#include "core/ngram_ple_state.h"
#include "core/linear_attention_state.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace sinfer::family {

inline constexpr std::int32_t kKvQuantGroup = 64;

struct DecoderStateSpec {
    std::uint32_t full_attention_layers     = 0;
    std::uint32_t mtp_layers                = 0;
    std::uint32_t capacity                  = 0;
    std::int32_t kv_heads                   = 0;
    std::int32_t attention_head_dim         = 0;
    // The second attention geometry, for a family whose global layers are shaped
    // differently from its windowed ones (Gemma 4: 8 key/value heads of 256 through the
    // window, 1 head of 512 over the whole context). Zero leaves the cache homogeneous,
    // which is every other family. `global_geometry_layers` indexes the full-attention
    // layers, the same numbering `kv_skip_layers` uses.
    std::int32_t global_kv_heads            = 0;
    std::int32_t global_attention_head_dim  = 0;
    std::vector<std::uint32_t> global_geometry_layers;
    DType kv_dtype                          = DType::BF16;
    std::int32_t kv_quant_group             = 0;
    // QSA indexer keys per full-attention layer (0 = none): one BF16 plane of this width
    // beside the layer's K/V, sharing the pages and block tables (phase 4).
    std::int32_t indexer_head_dim           = 0;
    // Full-attention layer indices held at the model dtype while the rest of
    // the cache is quantized. Linear-attention layers never appear here: they
    // hold no KV planes at all, so a quantized cache cannot reach them.
    std::vector<std::uint32_t> kv_skip_layers;
    bool enable_mtp                         = false;
    // The Main pool's planes follow demand (core/elastic_kv_region.h) instead of sitting in
    // the arena. The MTP pool stays in the arena: it is small and per-lane.
    bool elastic_kv                         = false;
    std::int32_t kv_table_rows              = 1;
    std::uint32_t text_physical_page_groups = 0;
    // Elastic Main pool: pages that may be physical at once (0 = all of the above).
    std::uint32_t text_physical_page_cap    = 0;
    // Elastic Main pool: the cap is a guaranteed floor and pages past it are gated on the
    // device's free memory at admission (core/elastic_kv_region.h).
    bool elastic_kv_overcommit              = false;
    std::uint32_t mtp_physical_page_groups  = 0;
    LinearAttentionStatePoolSpec linear_attention;
    // Per-slot state of a layer prologue (n-gram memory); absent for targets without one.
    std::optional<NgramPleStatePoolSpec> ple;
};

struct PagedKVCacheLayout {
    PagedKVPoolLayout pool;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;
    std::int32_t kv_heads     = 0;
    std::int32_t head_dim     = 0;
    DType dtype               = DType::BF16;
    std::int32_t quant_group  = 0;
    std::int32_t indexer_head_dim = 0;
    // Storage dtype per full-attention layer. A quantized cache may keep some
    // layers at the model dtype (--kv-cache-dtype-skip-layers), so the pool is
    // not necessarily homogeneous; dtype above is the cache's nominal setting.
    std::vector<DType> layer_dtypes;
    // Head geometry per full-attention layer, for the same reason and in the same shape:
    // a family may attend at two geometries, so `kv_heads`/`head_dim` above are the
    // nominal (windowed) setting and these are what each layer's planes were sized to.
    // Empty means homogeneous, which is what every family but Gemma 4 leaves it.
    std::vector<std::int32_t> layer_kv_heads;
    std::vector<std::int32_t> layer_head_dim;

    [[nodiscard]] std::size_t payload_bytes() const noexcept { return pool.payload_bytes(); }
};

class PagedKVCache;

class PagedKVCacheView {
public:
    PagedKVCacheView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return cache_ != nullptr; }

    [[nodiscard]] std::uint32_t max_context() const noexcept;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCache;
    PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept;

    const PagedKVCache* cache_ = nullptr;
    Tensor block_table_;
};

class PagedKVCache {
public:
    PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout,
                 const PagedKVElasticOptions* elastic = nullptr);

    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&)                 = delete;
    PagedKVCache& operator=(PagedKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] PagedKVPool& pool() noexcept { return pool_; }

    [[nodiscard]] const PagedKVPool& pool() const noexcept { return pool_; }

    [[nodiscard]] PagedKVCacheView execution_view(const PagedKVAllocation& allocation) const;

    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, Tensor block_table) const;

    PagedKVPool pool_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    std::int32_t head_dim_     = 0;
    DType dtype_               = DType::BF16;
    std::int32_t quant_group_  = 0;
    std::int32_t indexer_head_dim_ = 0;
    std::vector<DType> layer_dtypes_;
    std::vector<std::int32_t> layer_kv_heads_;
    std::vector<std::int32_t> layer_head_dim_;

    [[nodiscard]] DType layer_dtype(std::uint32_t layer) const noexcept {
        return layer < layer_dtypes_.size() ? layer_dtypes_[layer] : dtype_;
    }

    /// This layer's head geometry. Falls back to the nominal setting, so a homogeneous
    /// cache -- which is every family but Gemma 4 -- answers exactly as it did before the
    /// vectors existed.
    [[nodiscard]] std::int32_t layer_kv_heads(std::uint32_t layer) const noexcept {
        return layer < layer_kv_heads_.size() ? layer_kv_heads_[layer] : kv_heads_;
    }
    [[nodiscard]] std::int32_t layer_head_dim(std::uint32_t layer) const noexcept {
        return layer < layer_head_dim_.size() ? layer_head_dim_[layer] : head_dim_;
    }
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;
    LinearAttentionStatePoolLayout linear_attention;
    std::optional<NgramPleStatePoolLayout> ple;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;
    LinearAttentionStatePool linear_attention;
    NgramPleStatePool ple; ///< empty unless the layout planned one

    /// `elastic` supplies the device and fence stream an elastic Main pool maps with; it is
    /// ignored by layouts that keep their planes in the arena.
    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout,
                 const PagedKVElasticOptions* elastic = nullptr);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;

    /// Slot lifecycle across every per-slot pool (linear attention and, when present, PLE).
    void copy_state_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream);
    void reset_state_slot(std::int32_t slot, cudaStream_t stream);
};

} // namespace sinfer::family
