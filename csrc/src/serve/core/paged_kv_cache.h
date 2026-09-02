#pragma once

#include "core/elastic_kv_region.h"
#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace sinfer {

inline constexpr std::int32_t kPagedKVPageSize = 64;
// Mapping quantum of an elastic pool's planes: the CUDA VMM minimum on every device we serve.
// Fixed at plan time so the layout is a pure host computation; the region checks it against
// the device's actual granularity when it reserves.
inline constexpr std::size_t kElasticKvGranuleBytes = std::size_t{2} << 20;

/**
 * Non-owning, single-sequence view consumed by growing-cache Ops.
 *
 * Physical plane order is fixed by the owning homogeneous pool and validated by the consuming
 * Op. block_table is one contiguous I32 logical-block row.
 */
struct PagedKVLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    // QSA indexer keys of this layer, raw and unrotated, one head of `indexer_head_dim`
    // (design/INFERENCE.md, phase 4). Empty when the cache carries no indexer plane.
    Tensor indexer_pages;
    Tensor block_table;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    DType dtype               = DType::BF16;
    std::int32_t quant_group  = 0;
};

/**
 * Non-owning multi-sequence view consumed by batched growing-cache Ops.
 *
 * Physical planes and the complete block-table matrix are shared by every logical row in one
 * invocation. block_tables is contiguous I32 [logical_pages, table_rows]; the consuming Op
 * receives its per-row table selectors separately.
 */
struct PagedKVBatchLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor indexer_pages; // see PagedKVLayerView
    Tensor block_tables;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    DType dtype               = DType::BF16;
    std::int32_t quant_group  = 0;
};

// A pool plane is storage-only. Consumers assign K/V/layer meaning to plane indices.
struct PagedKVPlaneSpec {
    DType dtype                 = DType::BF16;
    std::int32_t leading_extent = 0;
    std::int32_t head_extent    = 0;
    std::size_t alignment       = 256;
};

enum class PagedKVPlaneOrder : std::uint8_t {
    PageMajor,
    HeadMajor,
};

struct PagedKVPoolSpec {
    std::uint32_t page_group_count      = 0;
    std::uint32_t logical_page_capacity = 0;
    std::int32_t table_rows             = 0;
    PagedKVPlaneOrder plane_order       = PagedKVPlaneOrder::PageMajor;
    // Planes live in their own demand-mapped region (core/elastic_kv_region.h) rather than
    // the backing arena; physical memory then follows the pages in use. PageMajor only.
    bool elastic                        = false;
    // Elastic pools: the pages that may hold physical memory at once. Admission is bounded by
    // this, the page count above being only the virtual span any one sequence may reach into.
    // Zero means the whole pool.
    std::uint32_t physical_page_cap     = 0;
    std::vector<PagedKVPlaneSpec> planes;
};

struct PagedKVPlaneLayout {
    PagedKVPlaneSpec spec;
    TensorRegion storage;
};

struct PagedKVPoolLayout {
    PagedKVPoolSpec spec;
    std::vector<PagedKVPlaneLayout> planes;
    TensorRegion block_tables;
    // Elastic pools: the planes' offsets are relative to a separate region of this many
    // bytes, mapped in granules of this many pages. Zero for pools inside the arena.
    std::size_t elastic_plane_bytes     = 0;
    std::uint32_t elastic_granule_pages = 0;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
};

/// What an elastic pool needs from its engine to map on demand.
struct PagedKVElasticOptions {
    int device                     = 0;
    cudaStream_t fence_stream      = nullptr;
    std::uint32_t reserve_granules = 4;
};

[[nodiscard]] PagedKVPoolLayout plan_paged_kv_pool(LayoutBuilder& builder,
                                                   const PagedKVPoolSpec& spec);

class PagedKVAllocation;
struct PagedKVResize;

/**
 * Physical occupancy of one pool at an instant, for demand-mapped-KV sizing.
 *
 * The pool's pages are all physically resident today, so `pages_in_use` is the demand a
 * VMM-backed pool would have had to map. `resident_pages_at_granule` is what such a pool
 * could not have unmapped: pages are mapped in granules (the CUDA VMM minimum is 2 MiB per
 * mapping, and a plane-major pool must map that much in every plane), and a granule stays
 * resident while any single page in it is in use. The gap between the two is what page
 * layout costs; the gap between `page_group_count` and `pages_in_use` is the whole prize.
 */
struct PagedKVOccupancy {
    std::uint32_t page_group_count          = 0;
    std::uint32_t entitled_pages            = 0;
    std::uint32_t pages_in_use              = 0;
    std::uint32_t granule_pages             = 0; ///< 0 when the plane order is not page-mappable
    std::uint32_t resident_pages_at_granule = 0;
    /// Pages whose granule is physically mapped right now: every page for a pool inside the
    /// arena, the demand-mapped set (in use plus the reserve) for an elastic one.
    std::uint32_t mapped_pages              = 0;
    std::uint32_t capacity_pages            = 0; ///< pages admission may commit (see PagedKVPoolSpec)
    std::size_t page_bytes                  = 0; ///< bytes one page occupies across every plane
};

class PagedKVPool {
public:
    /// `elastic` is consulted only when the layout's spec is elastic, and is then required.
    PagedKVPool(DeviceSpan backing, const PagedKVPoolLayout& layout,
                const PagedKVElasticOptions* elastic = nullptr);

    PagedKVPool(const PagedKVPool&)            = delete;
    PagedKVPool& operator=(const PagedKVPool&) = delete;
    PagedKVPool(PagedKVPool&&)                 = delete;
    PagedKVPool& operator=(PagedKVPool&&)      = delete;

    [[nodiscard]] std::uint32_t page_group_count() const noexcept;
    /// Pages admission may commit: the physical cap of an elastic pool, else every page.
    [[nodiscard]] std::uint32_t capacity_pages() const noexcept;
    [[nodiscard]] std::uint32_t logical_page_capacity() const noexcept;
    [[nodiscard]] std::int32_t table_row_count() const noexcept;
    [[nodiscard]] std::size_t plane_count() const noexcept;
    [[nodiscard]] const Tensor& plane(std::size_t index) const;
    [[nodiscard]] const Tensor& block_tables() const noexcept;
    [[nodiscard]] Tensor block_table_row(std::int32_t row) const;

    [[nodiscard]] std::uint32_t entitled_pages() const noexcept;
    [[nodiscard]] std::uint32_t mapped_pages() const noexcept;
    [[nodiscard]] std::uint32_t free_pages() const noexcept;
    [[nodiscard]] bool can_reserve(std::uint32_t page_entitlement) const noexcept;
    [[nodiscard]] PagedKVAllocation reserve(std::uint32_t page_entitlement);

    /// Occupancy snapshot. `granule_bytes` is the mapping quantum a demand-mapped pool would
    /// use per plane (the CUDA VMM minimum allocation granularity). Allocation-free, so callers
    /// holding the execution lock in a noexcept path can use it.
    [[nodiscard]] PagedKVOccupancy
    occupancy(std::size_t granule_bytes = 2ULL * 1024ULL * 1024ULL) const noexcept;

    // Zeros only the named physical page groups across every storage plane.
    void zero_pages(std::span<const std::int32_t> page_ids, cudaStream_t stream = nullptr,
                    int byte = 0);

    /// The demand-mapped plane region, or null for a pool inside the arena.
    [[nodiscard]] ElasticKvRegion* elastic_region() noexcept { return elastic_.get(); }

private:
    friend class PagedKVAllocation;
    friend void resize_paged_kv_bundle(std::span<const PagedKVResize> changes);

    [[nodiscard]] bool can_replace_entitlement(std::uint32_t old_pages,
                                               std::uint32_t new_pages) const noexcept;
    [[nodiscard]] std::vector<std::int32_t> take_pages(std::uint32_t count,
                                                       std::int32_t preferred_first);
    void return_pages(std::span<const std::int32_t> pages) noexcept;
    void add_entitlement(std::uint32_t pages) noexcept;
    void replace_entitlement(std::uint32_t old_pages, std::uint32_t new_pages) noexcept;
    void acquire_row(std::int32_t row);
    void release_row(std::int32_t row) noexcept;

    PagedKVPoolSpec spec_;
    std::unique_ptr<ElasticKvRegion> elastic_; ///< owns the planes' memory when elastic
    std::vector<Tensor> planes_;
    Tensor block_tables_;
    std::vector<std::int32_t> free_page_ids_;

    std::vector<bool> row_in_use_;
    std::uint32_t entitled_pages_ = 0;
    std::uint32_t mapped_pages_   = 0;
};

class PagedKVAllocation {
public:
    PagedKVAllocation() noexcept = default;
    ~PagedKVAllocation();

    PagedKVAllocation(const PagedKVAllocation&)            = delete;
    PagedKVAllocation& operator=(const PagedKVAllocation&) = delete;
    PagedKVAllocation(PagedKVAllocation&& other) noexcept;
    PagedKVAllocation& operator=(PagedKVAllocation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint32_t page_entitlement() const noexcept;
    [[nodiscard]] std::uint32_t mapped_page_count() const noexcept;
    [[nodiscard]] std::uint32_t mapped_token_capacity() const noexcept;
    [[nodiscard]] std::int32_t bound_row() const noexcept;
    [[nodiscard]] std::span<const std::int32_t> page_ids() const noexcept;
    [[nodiscard]] bool belongs_to(const PagedKVPool& pool) const noexcept;

    void set_page_entitlement(std::uint32_t pages);
    void cancel_unmapped_entitlement() noexcept;
    void materialize_pages(std::uint32_t pages, cudaStream_t stream = nullptr);
    void materialize_tokens(std::uint32_t tokens, cudaStream_t stream = nullptr);
    /// With a stream, the trimmed block-table entries are overwritten on it with an
    /// out-of-range page id, so a kernel that still addresses a trimmed page faults instead of
    /// reading whatever lane owns that page next. Without one the entries go stale.
    void trim_pages(std::uint32_t pages, cudaStream_t stream = nullptr);
    void trim_tokens(std::uint32_t tokens, cudaStream_t stream = nullptr);

    void bind_row(std::int32_t row, cudaStream_t stream = nullptr);
    void publish_mapping(cudaStream_t stream = nullptr) const;
    void unbind_row() noexcept;
    [[nodiscard]] Tensor block_table() const;

    void release() noexcept;

private:
    friend class PagedKVPool;
    friend void resize_paged_kv_bundle(std::span<const PagedKVResize> changes);

    PagedKVAllocation(PagedKVPool& pool, std::uint32_t page_entitlement);
    void publish_range(std::uint32_t first_page, std::uint32_t page_count,
                       cudaStream_t stream) const;

    PagedKVPool* pool_ = nullptr;
    std::vector<std::int32_t> page_ids_;
    std::uint32_t page_entitlement_ = 0;
    std::int32_t bound_row_         = -1;
};

struct PagedKVReservation {
    PagedKVPool* pool              = nullptr;
    std::uint32_t page_entitlement = 0;
};

// Reserves every requested pool or leaves all pools unchanged.
[[nodiscard]] std::vector<PagedKVAllocation>
reserve_paged_kv_bundle(std::span<const PagedKVReservation> reservations);

struct PagedKVResize {
    PagedKVAllocation* allocation  = nullptr;
    std::uint32_t mapped_pages     = 0;
    std::uint32_t page_entitlement = 0;
};

// Atomically validates a retained-claim/truncate resize vector, then applies it at a GPU boundary.
void resize_paged_kv_bundle(std::span<const PagedKVResize> changes);

} // namespace sinfer
