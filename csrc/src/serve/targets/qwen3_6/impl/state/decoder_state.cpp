#include <api/targets/qwen3_6/decoder_state.h>

#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, DType dtype,
                              std::int32_t quant_group, std::int32_t table_rows,
                              std::uint32_t physical_page_groups,
                              const std::vector<std::uint32_t>& skip_layers,
                              std::int32_t indexer_head_dim) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    // I8 carries a per-group scale plane pair; FP8_E4M3FN stores raw e4m3 codes
    // with no side planes, so it has the same plane count as BF16 and can be
    // mixed with it layer by layer.
    const bool grouped = dtype == DType::I8;
    const bool codes   = dtype == DType::FP8_E4M3FN;
    if ((!grouped && !codes && (dtype != DType::BF16 || quant_group != 0)) ||
        (codes && quant_group != 0) ||
        (grouped && (quant_group != kKvQuantGroup || head_dim % quant_group != 0))) {
        throw std::invalid_argument("Paged KV cache dtype or quantization is invalid");
    }
    if (grouped && !skip_layers.empty()) {
        // The int8 pool is homogeneous: its layers own four planes each, so a
        // skipped layer would change the plane stride the views index with.
        throw std::invalid_argument(
            "kv-cache-dtype-skip-layers requires an fp8 cache (int8 is all-or-nothing)");
    }
    for (const std::uint32_t skipped : skip_layers) {
        if (skipped >= layers) {
            throw std::invalid_argument("kv-cache-dtype-skip-layers names layer " +
                                        std::to_string(skipped) + ", but the model has " +
                                        std::to_string(layers) + " full-attention layers");
        }
    }

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    PagedKVPoolSpec pool_spec;
    pool_spec.page_group_count      = physical_page_groups;
    pool_spec.logical_page_capacity = logical_pages;
    pool_spec.table_rows            = table_rows;
    const std::size_t planes_per_layer =
        (grouped ? 4ULL : 2ULL) + (indexer_head_dim > 0 ? 1ULL : 0ULL);
    pool_spec.planes.reserve(static_cast<std::size_t>(layers) * planes_per_layer);
    std::vector<DType> layer_dtypes(layers, dtype);
    for (const std::uint32_t skipped : skip_layers) { layer_dtypes[skipped] = DType::BF16; }
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const DType layer_dtype = layer_dtypes[layer];
        pool_spec.planes.push_back({layer_dtype, head_dim, kv_heads, 256});
        pool_spec.planes.push_back({layer_dtype, head_dim, kv_heads, 256});
        if (grouped) {
            pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
            pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
        }
        // The indexer plane is always BF16 and one head: the selection reads raw keys and a
        // quantized cache would change which cells the model attends to, not just their values.
        if (indexer_head_dim > 0) {
            pool_spec.planes.push_back({DType::BF16, indexer_head_dim, 1, 256});
        }
    }
    return PagedKVCacheLayout{
        .pool        = plan_paged_kv_pool(builder, pool_spec),
        .layers      = layers,
        .max_context = capacity,
        .kv_heads    = kv_heads,
        .head_dim    = head_dim,
        .dtype        = dtype,
        .quant_group  = quant_group,
        .indexer_head_dim = indexer_head_dim,
        .layer_dtypes = std::move(layer_dtypes),
    };
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                spec.kv_table_rows, spec.text_physical_page_groups,
                                spec.kv_skip_layers, spec.indexer_head_dim);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                   spec.kv_table_rows, spec.mtp_physical_page_groups,
                                   spec.kv_skip_layers, 0);
    }
    layout.linear_attention = plan_linear_attention_state_pool(builder, spec.linear_attention);
    if (spec.ple) { layout.ple = plan_ngram_ple_state_pool(builder, *spec.ple); }
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pool_(backing, layout.pool), layers_(layout.layers), max_context_(layout.max_context),
      kv_heads_(layout.kv_heads), head_dim_(layout.head_dim), dtype_(layout.dtype),
      quant_group_(layout.quant_group), indexer_head_dim_(layout.indexer_head_dim),
      layer_dtypes_(layout.layer_dtypes) {}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const PagedKVAllocation& allocation) const {
    if (!allocation.belongs_to(pool_)) {
        throw std::invalid_argument("Paged KV allocation belongs to another cache pool");
    }
    return PagedKVCacheView(*this, allocation.block_table());
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    // Only the int8 cache adds scale planes, so it alone widens the stride; an
    // fp8 layer occupies the same two planes a bf16 layer does, which is what
    // lets the two be mixed within one pool.
    const bool grouped       = dtype_ == DType::I8;
    const std::size_t stride =
        (grouped ? 4ULL : 2ULL) + (indexer_head_dim_ > 0 ? 1ULL : 0ULL);
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = grouped ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = grouped ? pool_.plane(base + 3) : Tensor(),
        .indexer_pages = indexer_head_dim_ > 0 ? pool_.plane(base + (grouped ? 4ULL : 2ULL)) : Tensor(),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = layer_dtype(layer),
        .quant_group   = quant_group_,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    // Only the int8 cache adds scale planes, so it alone widens the stride; an
    // fp8 layer occupies the same two planes a bf16 layer does, which is what
    // lets the two be mixed within one pool.
    const bool grouped       = dtype_ == DType::I8;
    const std::size_t stride =
        (grouped ? 4ULL : 2ULL) + (indexer_head_dim_ > 0 ? 1ULL : 0ULL);
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVBatchLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = grouped ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = grouped ? pool_.plane(base + 3) : Tensor(),
        .indexer_pages = indexer_head_dim_ > 0 ? pool_.plane(base + (grouped ? 4ULL : 2ULL)) : Tensor(),
        .block_tables  = pool_.block_tables(),
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = layer_dtype(layer),
        .quant_group   = quant_group_,
    };
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv), linear_attention(backing, layout.linear_attention) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
    if (layout.ple) { ple = NgramPleStatePool(backing, *layout.ple); }
}

void DecoderState::copy_state_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream) {
    linear_attention.copy_slot(src, dst, stream);
    if (!ple.empty()) { ple.copy_slot(src, dst, stream); }
}

void DecoderState::reset_state_slot(std::int32_t slot, cudaStream_t stream) {
    linear_attention.zero_slot(slot, stream);
    if (!ple.empty()) { ple.reset_slot(slot, stream); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
