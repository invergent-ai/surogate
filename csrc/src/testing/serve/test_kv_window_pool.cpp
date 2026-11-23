// Window-bounded KV pages (window-KV design): the
// paged pool's window class on a device -- layout, zeroed sentinel, block-table rows, page
// recycling, images and forks, window page up/download. Exits 77 without a usable device.

#include "core/device.h"
#include "core/elastic_kv_region.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        ++g_failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

std::vector<std::int32_t> device_row(const sinfer::Tensor& row, std::size_t count) {
    std::vector<std::int32_t> out(count);
    if (cudaMemcpy(out.data(), row.data, count * sizeof(std::int32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        expect(false, "row copy");
    }
    return out;
}

/// Bytes of the given window pages, every windowed plane, in one host vector.
std::vector<std::byte> window_bytes(const sinfer::PagedKVPool& pool, const std::vector<std::int32_t>& pages,
                                    cudaStream_t stream) {
    std::vector<std::vector<std::byte>> planes(pool.window_plane_count());
    std::vector<std::byte*> pointers;
    for (std::size_t k = 0; k < planes.size(); ++k) {
        planes[k].resize(pool.window_image_plane_bytes(k, pages.size()));
        pointers.push_back(planes[k].data());
    }
    pool.download_window_pages_to(pages, pointers, stream);
    cudaStreamSynchronize(stream);
    std::vector<std::byte> out;
    for (const auto& plane : planes) { out.insert(out.end(), plane.begin(), plane.end()); }
    return out;
}

void fill_window_pages(const sinfer::PagedKVPool& pool, const std::vector<std::int32_t>& pages, int seed,
                       cudaStream_t stream) {
    std::vector<std::vector<std::byte>> planes(pool.window_plane_count());
    std::vector<const std::byte*> pointers;
    for (std::size_t k = 0; k < planes.size(); ++k) {
        planes[k].resize(pool.window_image_plane_bytes(k, pages.size()));
        for (std::size_t i = 0; i < planes[k].size(); ++i) {
            planes[k][i] = static_cast<std::byte>((seed * 31 + static_cast<int>(k) * 7 + static_cast<int>(i)) & 0x7f);
        }
        pointers.push_back(planes[k].data());
    }
    pool.upload_window_pages_from(pointers, static_cast<std::uint32_t>(pages.size()), pages, stream);
    cudaStreamSynchronize(stream);
}

void check(sinfer::PagedKVPlaneOrder order, cudaStream_t stream) {
    const std::string tag = order == sinfer::PagedKVPlaneOrder::PageMajor ? "page-major" : "head-major";
    constexpr std::int32_t kWindow = 128;
    sinfer::LayoutBuilder builder;
    sinfer::PagedKVPoolSpec spec{.page_group_count      = 8,
                                 .logical_page_capacity = 8,
                                 .table_rows            = 2,
                                 .plane_order           = order,
                                 .planes = {{sinfer::DType::BF16, 16, 2, 256, 0},
                                            {sinfer::DType::BF16, 16, 2, 256, 0},
                                            {sinfer::DType::BF16, 16, 2, 256, kWindow},
                                            {sinfer::DType::BF16, 16, 2, 256, kWindow}},
                                 .window_lane_pages     = 6,
                                 .window_image_pages    = 2,
                                 .window_forward_tokens = 192};
    const auto layout = sinfer::plan_paged_kv_pool(builder, spec);
    sinfer::DeviceArena arena(builder.finish(256));
    sinfer::PagedKVPool pool({arena.base(), arena.capacity()}, layout);

    expect(layout.window_page_count == 9, tag + ": window pages = lanes + images + sentinel");
    expect(pool.has_window_class() && pool.window_tokens() == kWindow, tag + ": window class present");
    expect(pool.full_plane_count() == 2 && pool.window_plane_count() == 2, tag + ": plane classes");
    expect(pool.window_sentinel_page() == 8, tag + ": sentinel is the last window page");
    const int page_axis = order == sinfer::PagedKVPlaneOrder::PageMajor ? 3 : 2;
    expect(pool.plane(0).ne[page_axis] == 8 && pool.plane(2).ne[page_axis] == 9, tag + ": plane page extents");
    {
        const auto sentinel = window_bytes(pool, {8}, stream);
        bool zero = true;
        for (const std::byte b : sentinel) { zero = zero && b == std::byte{0}; }
        expect(zero, tag + ": sentinel page is zeroed");
    }

    auto a = pool.reserve(4);
    a.materialize_tokens(256, stream);
    a.bind_row(0, stream);
    cudaStreamSynchronize(stream);
    expect(device_row(a.window_block_table(), 8) == std::vector<std::int32_t>(8, 8), tag + ": fresh row is all sentinel");

    // A forward at 0 writing 256 tokens maps only the forward reach (192 tokens = 3 pages).
    a.window_prepare(0, 256, {}, stream);
    cudaStreamSynchronize(stream);
    expect(a.window_slots().size() == 3, tag + ": prepare maps the forward reach");
    std::vector<std::int32_t> live;
    for (const auto& slot : a.window_slots()) { live.push_back(slot.physical); }
    auto row = device_row(a.window_block_table(), 8);
    expect(row[0] == live[0] && row[1] == live[1] && row[2] == live[2] && row[3] == 8 && row[7] == 8,
           tag + ": row publishes live pages and the sentinel");
    fill_window_pages(pool, live, 5, stream);

    // Image of the window behind 192 tokens: keep_first(192, 128) = 1, so pages 1 and 2.
    auto image = pool.capture_window_image(a, 192, stream);
    expect(image && image->slots().size() == 2 && image->slots()[0].logical == 1, tag + ": image of pages 1-2");
    expect(pool.window_occupancy().image_in_use == 2, tag + ": image region in use");

    // Settle at 192: page 0 falls behind the window and goes back; its entry becomes the sentinel.
    a.window_settle(192, {}, stream);
    cudaStreamSynchronize(stream);
    row = device_row(a.window_block_table(), 8);
    expect(a.window_slots().size() == 2 && row[0] == 8 && row[1] == live[1], tag + ": settle recycles page 0");
    expect(a.window_covers(192) && a.window_covers(191) && !a.window_covers(64), tag + ": covers");

    // A fork restores the image into its own pages, with the same bytes.
    auto b = pool.reserve(4);
    b.bind_row(1, stream);
    b.window_restore_from_image(*image, stream);
    cudaStreamSynchronize(stream);
    std::vector<std::int32_t> b_pages;
    for (const auto& slot : b.window_slots()) { b_pages.push_back(slot.physical); }
    expect(b_pages.size() == 2 && b.window_covers(192), tag + ": fork covers the boundary");
    expect(window_bytes(pool, b_pages, stream) == window_bytes(pool, {live[1], live[2]}, stream),
           tag + ": fork pages equal the source window");
    row = device_row(b.window_block_table(), 8);
    expect(row[0] == 8 && row[1] == b_pages[0] && row[2] == b_pages[1] && row[3] == 8, tag + ": fork row");

    // Rewind (trim) gives back pages at and past the trim; release gives back the rest.
    a.trim_tokens(128, stream);
    expect(a.window_slots().size() == 1, tag + ": trim rewinds the window class");
    image.reset();
    a.release();
    b.release();
    const auto occupancy = pool.window_occupancy();
    expect(occupancy.lane_in_use == 0 && occupancy.image_in_use == 0, tag + ": every window page returned");
    // a held 3 pages, settled to 2, then b restored 2 (images live in their own region): peak 4.
    expect(occupancy.lane_peak == 4, tag + ": peak recorded");
}

void check_without_windows(cudaStream_t stream) {
    sinfer::LayoutBuilder builder;
    const auto layout = sinfer::plan_paged_kv_pool(
        builder, {.page_group_count = 4, .logical_page_capacity = 4, .table_rows = 1,
                  .planes = {{sinfer::DType::BF16, 16, 2}, {sinfer::DType::BF16, 16, 2}}});
    sinfer::DeviceArena arena(builder.finish(256));
    sinfer::PagedKVPool pool({arena.base(), arena.capacity()}, layout);
    expect(!pool.has_window_class() && layout.window_page_count == 0, "no window class without windows");
    expect(pool.full_plane_count() == pool.plane_count(), "every plane is a full plane");
    auto a = pool.reserve(2);
    a.materialize_tokens(100, stream);
    a.bind_row(0, stream);
    a.window_prepare(0, 100, {}, stream);
    a.window_settle(100, {}, stream);
    expect(a.window_slots().empty() && a.window_covers(5), "window calls are no-ops");
}

/// M3-C: an elastic window class -- its own demand-mapped region, the sentinel first and zeroed,
/// window charges booked in the main ledger (reserve, images, release), lowest-first pages,
/// compaction moves pages down bitwise and rewrites the owners' rows.
void check_elastic(const sinfer::DeviceContext& device) {
    const std::string tag = "elastic";
    cudaStream_t stream = device.stream;
    constexpr std::int32_t kWindow = 128;
    // 128 x 8 BF16 per token row: 128 KiB per page per plane -> 16 pages per 2 MiB granule.
    sinfer::LayoutBuilder builder;
    sinfer::PagedKVPoolSpec spec{.page_group_count      = 64,
                                 .logical_page_capacity = 16,
                                 .table_rows            = 3,
                                 .plane_order           = sinfer::PagedKVPlaneOrder::PageMajor,
                                 .elastic               = true,
                                 .physical_page_cap     = 64,
                                 .planes = {{sinfer::DType::BF16, 128, 8, 256, 0},
                                            {sinfer::DType::BF16, 128, 8, 256, 0},
                                            {sinfer::DType::BF16, 128, 8, 256, kWindow},
                                            {sinfer::DType::BF16, 128, 8, 256, kWindow}},
                                 .window_lane_pages     = 40,
                                 .window_image_pages    = 8,
                                 .window_forward_tokens = 1024,
                                 .window_elastic        = true,
                                 .window_floor_pages    = 10, // transient: rounded to 16, + 16 slack
                                 .window_lane_charge    = 7};
    const auto layout = sinfer::plan_paged_kv_pool(builder, spec);
    expect(layout.window_elastic_bytes != 0 && layout.window_granule_pages == 16, tag + ": own region, 16-page granules");
    expect(layout.spec.window_floor_pages == 32, tag + ": floor = transient in granules + one granule");
    sinfer::DeviceArena arena(builder.finish(256));
    const sinfer::PagedKVElasticOptions options{.device = device.device, .fence_stream = stream, .reserve_granules = 0};
    sinfer::PagedKVPool pool({arena.base(), arena.capacity()}, layout, &options);
    expect(pool.window_elastic() && pool.window_sentinel_page() == 0, tag + ": sentinel is page 0");
    {
        const auto sentinel = window_bytes(pool, {0}, stream);
        bool zero = true;
        for (const std::byte b : sentinel) { zero = zero && b == std::byte{0}; }
        expect(zero, tag + ": sentinel page is zeroed");
    }
    // Equal page bytes in both classes: one window page = one main page, so a charge of 7 = 7 pages.
    expect(pool.window_lane_charge_pages() == 7, tag + ": lane charge in main pages");
    auto a = pool.reserve(4);
    expect(pool.entitled_pages() == 11 && pool.window_occupancy().charged_pages == 7, tag + ": reserve charges");
    auto capture = pool.reserve(1, 1);
    expect(pool.entitled_pages() == 13, tag + ": a capture row is charged one window page");
    capture.release();
    expect(pool.entitled_pages() == 11, tag + ": release returns the charge");
    auto b = pool.reserve(4);
    auto c = pool.reserve(4);
    a.bind_row(0, stream);
    b.bind_row(1, stream);
    c.bind_row(2, stream);
    {
        // 33 entitled; a 16-page reservation books 23 -> 56; then 1 more page books 8 (64 = the cap), 2 do not.
        auto x = pool.reserve(16);
        expect(pool.entitled_pages() == 56 && pool.can_reserve(1) && !pool.can_reserve(2),
               tag + ": can_reserve books the window charge");
    }
    // a and b: long forwards map 16 pages each (lowest first: 1..16, 17..32); c then 2 (33, 34).
    a.window_prepare(0, 1024, {}, stream);
    b.window_prepare(0, 1024, {}, stream);
    c.window_prepare(0, 128, {}, stream);
    const auto pages_of = [](const sinfer::PagedKVAllocation& allocation) {
        std::vector<std::int32_t> out;
        for (const auto& slot : allocation.window_slots()) { out.push_back(slot.physical); }
        return out;
    };
    const auto a_pages = pages_of(a), b_pages = pages_of(b), c_pages = pages_of(c);
    expect(a_pages.size() == 16 && a_pages.front() == 1 && a_pages.back() == 16 && b_pages.front() == 17 &&
               c_pages == std::vector<std::int32_t>({33, 34}), tag + ": lowest pages first");
    fill_window_pages(pool, c_pages, 9, stream);
    const auto c_bytes = window_bytes(pool, c_pages, stream);
    // a and b settle at 1024: each keeps its window (logical 14, 15 = pages 15, 16 and 31, 32).
    a.window_settle(1024, {}, stream);
    b.window_settle(1024, {}, stream);
    const auto b_kept = pages_of(b);
    expect(b_kept == std::vector<std::int32_t>({31, 32}), tag + ": b keeps its window");
    fill_window_pages(pool, b_kept, 11, stream);
    const auto b_bytes = window_bytes(pool, b_kept, stream);
    // 7 pages in use over 3 granules: compaction empties granules 2 and 1 into granule 0.
    const std::uint32_t moved = pool.compact_window_class({}, stream);
    cudaStreamSynchronize(stream);
    const auto a_after = pages_of(a), b_after = pages_of(b), c_after = pages_of(c);
    bool low = true;
    for (const auto* pages : {&a_after, &b_after, &c_after}) {
        for (const std::int32_t page : *pages) { low = low && page < 16; }
    }
    expect(moved == 5 && low, tag + ": compaction moved every page into granule 0 (" + std::to_string(moved) + " moves)");
    expect(window_bytes(pool, c_after, stream) == c_bytes && window_bytes(pool, b_after, stream) == b_bytes,
           tag + ": moved pages are bitwise the same");
    auto row = device_row(c.window_block_table(), 16);
    expect(row[0] == c_after[0] && row[1] == c_after[1] && row[2] == 0, tag + ": c's row follows the move");
    row = device_row(b.window_block_table(), 16);
    expect(row[13] == 0 && row[14] == b_after[0] && row[15] == b_after[1], tag + ": b's row follows the move");
    expect(pool.compact_window_class({}, stream) == 0, tag + ": nothing left to compact");
    // An image is charged while it lives.
    const std::uint32_t before = pool.entitled_pages();
    {
        auto image = pool.capture_window_image(b, 1024, stream);
        expect(image != nullptr && pool.entitled_pages() == before + 2, tag + ": image charged");
    }
    expect(pool.entitled_pages() == before, tag + ": image charge returned");
    a.release();
    b.release();
    c.release();
    expect(pool.entitled_pages() == 0 && pool.window_occupancy().charged_pages == 0 &&
               pool.window_occupancy().lane_in_use == 0, tag + ": everything returned");
}

/// Out-of-memory fix F1 (2026-11-23): the window pages charged in the main ledger are mapped by the window
/// region, so the main region must not map those bytes too. Before the fix the main region could
/// map its whole cap while the window region mapped the charged pages beside it: on an Inkling-2-Mini card both
/// filled together, 4 MiB were left and the engine aborted. At a round boundary the main region's
/// limit is its cap less the charged window bytes (rounded up to a granule), and what it maps
/// stays inside that limit.
void check_elastic_physical_bound(const sinfer::DeviceContext& device) {
    const std::string tag = "elastic bound";
    cudaStream_t stream = device.stream;
    constexpr std::int32_t kWindow = 128;
    // The shape of check_elastic: 256 KiB pages in both classes (one window page = one main page),
    // 16 pages = 4 MiB per granule; main cap 64 pages = 4 granules; a lane is charged 12 pages.
    sinfer::LayoutBuilder builder;
    sinfer::PagedKVPoolSpec spec{.page_group_count      = 64,
                                 .logical_page_capacity = 16,
                                 .table_rows            = 3,
                                 .plane_order           = sinfer::PagedKVPlaneOrder::PageMajor,
                                 .elastic               = true,
                                 .physical_page_cap     = 64,
                                 .planes = {{sinfer::DType::BF16, 128, 8, 256, 0},
                                            {sinfer::DType::BF16, 128, 8, 256, 0},
                                            {sinfer::DType::BF16, 128, 8, 256, kWindow},
                                            {sinfer::DType::BF16, 128, 8, 256, kWindow}},
                                 .window_lane_pages     = 40,
                                 .window_image_pages    = 8,
                                 .window_forward_tokens = 1024,
                                 .window_elastic        = true,
                                 .window_floor_pages    = 10,
                                 .window_lane_charge    = 12};
    const auto layout = sinfer::plan_paged_kv_pool(builder, spec);
    sinfer::DeviceArena arena(builder.finish(256));
    // The engine's default reserve of four granules: before the fix it kept the whole cap mapped.
    const sinfer::PagedKVElasticOptions options{.device = device.device, .fence_stream = stream, .reserve_granules = 4};
    sinfer::PagedKVPool pool({arena.base(), arena.capacity()}, layout, &options);
    sinfer::ElasticKvRegion* main_region   = pool.elastic_region();
    sinfer::ElasticKvRegion* window_region = pool.window_elastic_region();
    expect(main_region != nullptr && window_region != nullptr, tag + ": both classes are demand-mapped");
    if (main_region == nullptr || window_region == nullptr) { return; }
    main_region->wait_idle();
    auto a = pool.reserve(8);
    auto b = pool.reserve(8);
    auto c = pool.reserve(8);
    const auto charged = pool.window_occupancy();
    expect(charged.charged_pages == 36 && pool.entitled_pages() == 60, tag + ": three lanes charged 12 window pages each");
    const std::size_t granule       = main_region->granule_bytes();
    const std::size_t cap_bytes     = 4 * granule;
    const std::size_t charged_bytes = std::size_t{charged.charged_pages} * charged.page_bytes;
    const auto allowed = static_cast<std::uint32_t>((cap_bytes - charged_bytes + granule - 1) / granule);
    expect(main_region->granule_limit() == allowed,
           tag + ": the main limit leaves the charged window bytes out (limit " +
               std::to_string(main_region->granule_limit()) + " granules, " + std::to_string(allowed) + " allowed)");
    // A round boundary, then the lanes take their pages and the long forwards their window pages.
    main_region->flush_reserve_release();
    main_region->wait_idle();
    a.materialize_tokens(512, stream);
    b.materialize_tokens(512, stream);
    c.materialize_tokens(512, stream);
    a.bind_row(0, stream);
    b.bind_row(1, stream);
    c.bind_row(2, stream);
    a.window_prepare(0, 1024, {}, stream);
    b.window_prepare(0, 1024, {}, stream);
    c.window_prepare(0, 128, {}, stream);
    cudaStreamSynchronize(stream);
    main_region->flush_reserve_release();
    main_region->wait_idle();
    window_region->wait_idle();
    const std::size_t main_mapped = main_region->mapped_bytes(), window_mapped = window_region->mapped_bytes();
    expect(main_mapped <= allowed * granule,
           tag + ": main mapped " + std::to_string(main_mapped >> 20) + " MiB, at most " +
               std::to_string((allowed * granule) >> 20) + " MiB (cap less the charged window bytes)");
    // The plan in whole granules: main cap + window floor, + one window granule of rounding.
    const std::size_t floor_bytes = std::size_t{layout.spec.window_floor_pages} * charged.page_bytes;
    const std::size_t window_granule = std::size_t{layout.window_granule_pages} * charged.page_bytes;
    expect(main_mapped + window_mapped <= cap_bytes + floor_bytes + window_granule,
           tag + ": main + window mapped " + std::to_string((main_mapped + window_mapped) >> 20) + " MiB within the plan");
    // Two holders leave: the main limit rises at once, so the window region must give its emptied
    // granules back down to its own (lower) limit at the next boundary (review of 75204b7d, MEDIUM 1).
    a.release();
    b.release();
    window_region->flush_reserve_release();
    window_region->wait_idle();
    main_region->wait_idle();
    expect(window_region->mapped_granules() <= window_region->granule_limit(),
           tag + ": after holders leave the window region maps " + std::to_string(window_region->mapped_granules()) +
               " granules, its limit is " + std::to_string(window_region->granule_limit()));
    c.release();
}

/// Holders leave while the window region still maps granules past its lowered limit: granules that
/// still hold another holder's pages (a lane the compaction may not move this round, an image,
/// which never moves). Before the fix the main region's limit rose by the whole charge at once, so
/// the two limits together promised more than the plan. Now the main limit also leaves out what
/// the window region maps past its own limit, and the sum stays inside main cap + window floor
/// (+ one window granule of rounding, as check_elastic_physical_bound allows).
void check_elastic_overshoot(const sinfer::DeviceContext& device) {
    const std::string tag = "elastic overshoot";
    cudaStream_t stream = device.stream;
    constexpr std::int32_t kWindow = 128;
    // check_elastic_physical_bound's shape: 256 KiB pages in both classes, 16 pages a granule,
    // main cap 64 pages; window pages: sentinel 0, lanes 1..40, images 41..48 (granules 0..3).
    sinfer::LayoutBuilder builder;
    sinfer::PagedKVPoolSpec spec{.page_group_count      = 64,
                                 .logical_page_capacity = 16,
                                 .table_rows            = 3,
                                 .plane_order           = sinfer::PagedKVPlaneOrder::PageMajor,
                                 .elastic               = true,
                                 .physical_page_cap     = 64,
                                 .planes = {{sinfer::DType::BF16, 128, 8, 256, 0},
                                            {sinfer::DType::BF16, 128, 8, 256, 0},
                                            {sinfer::DType::BF16, 128, 8, 256, kWindow},
                                            {sinfer::DType::BF16, 128, 8, 256, kWindow}},
                                 .window_lane_pages     = 40,
                                 .window_image_pages    = 8,
                                 .window_forward_tokens = 1024,
                                 .window_elastic        = true,
                                 .window_floor_pages    = 10,
                                 .window_lane_charge    = 12};
    const auto layout = sinfer::plan_paged_kv_pool(builder, spec);
    sinfer::DeviceArena arena(builder.finish(256));
    const sinfer::PagedKVElasticOptions options{.device = device.device, .fence_stream = stream, .reserve_granules = 4};
    sinfer::PagedKVPool pool({arena.base(), arena.capacity()}, layout, &options);
    sinfer::ElasticKvRegion* main_region   = pool.elastic_region();
    sinfer::ElasticKvRegion* window_region = pool.window_elastic_region();
    expect(main_region != nullptr && window_region != nullptr, tag + ": both classes are demand-mapped");
    if (main_region == nullptr || window_region == nullptr) { return; }
    const auto pages_of = [](const sinfer::PagedKVAllocation& allocation) {
        std::vector<std::int32_t> out;
        for (const auto& slot : allocation.window_slots()) { out.push_back(slot.physical); }
        return out;
    };
    auto a = pool.reserve(4);
    auto b = pool.reserve(4);
    auto c = pool.reserve(4);
    a.bind_row(0, stream);
    b.bind_row(1, stream);
    c.bind_row(2, stream);
    // a: pages 1..16, c: 17..32, b: 33, 34 (before the settles give 1..14 and 17..30 back); a and c
    // keep their window (a: 15, 16; c: 31, 32).
    a.window_prepare(0, 1024, {}, stream);
    c.window_prepare(0, 1024, {}, stream);
    b.window_prepare(0, 128, {}, stream);
    a.window_settle(1024, {}, stream);
    c.window_settle(1024, {}, stream);
    expect(pages_of(c) == std::vector<std::int32_t>({31, 32}) && pages_of(b) == std::vector<std::int32_t>({33, 34}),
           tag + ": c keeps 31, 32 (granules 1, 2); b holds 33, 34 (granule 2)");
    // Four images of c's window: 41-42, 43-44, 45-46, 47-48; the last one straddles granules 2 and 3.
    std::vector<std::shared_ptr<const sinfer::PagedKVWindowImage>> images;
    for (int i = 0; i < 4; ++i) { images.push_back(pool.capture_window_image(c, 1024, stream)); }
    expect(images.back() != nullptr, tag + ": four images captured");
    cudaStreamSynchronize(stream);
    // a, b and the first three images leave; c (not compacted this round) and the last image stay.
    std::shared_ptr<const sinfer::PagedKVWindowImage> last_image = images.back();
    images.clear();
    a.release();
    b.release();
    expect(pool.window_occupancy().charged_pages == 14, tag + ": c (12) and one image (2) stay charged");
    // The next round boundary: both regions give back what they may.
    main_region->flush_reserve_release();
    window_region->flush_reserve_release();
    window_region->wait_idle();
    main_region->wait_idle();
    // Granules 0 (sentinel), 1 (c), 2 (c, image), 3 (image) hold pages; the window limit is
    // ceil((floor 32 + charged 14) / 16) = 3 granules.
    const std::uint32_t window_granules = window_region->mapped_granules();
    expect(window_granules == 4 && window_region->granule_limit() == 3,
           tag + ": the window region maps " + std::to_string(window_granules) + " granules past its limit of " +
               std::to_string(window_region->granule_limit()));
    const std::size_t granule        = main_region->granule_bytes();
    const std::size_t page_bytes     = pool.window_occupancy().page_bytes;
    const std::size_t cap_bytes      = 4 * granule;
    const std::size_t floor_bytes    = std::size_t{layout.spec.window_floor_pages} * page_bytes;
    const std::size_t window_granule = std::size_t{layout.window_granule_pages} * page_bytes;
    const std::size_t promised       = std::size_t{main_region->granule_limit()} * granule + window_region->mapped_bytes();
    expect(promised <= cap_bytes + floor_bytes + window_granule,
           tag + ": main limit + window mapped " + std::to_string(promised >> 20) + " MiB within the plan " +
               std::to_string((cap_bytes + floor_bytes + window_granule) >> 20) + " MiB (main limit " +
               std::to_string(main_region->granule_limit()) + " granules)");
    // The overshoot gone (c compacts into granule 0's holes, granule 1 goes back), the limit comes back.
    expect(pool.compact_window_class({}, stream) == 2, tag + ": c's two pages move into granule 0");
    cudaStreamSynchronize(stream);
    window_region->flush_reserve_release();
    window_region->wait_idle();
    const auto restored = static_cast<std::uint32_t>((cap_bytes - 14 * page_bytes + granule - 1) / granule);
    expect(window_region->mapped_granules() == 3 && main_region->granule_limit() == restored,
           tag + ": window back at its limit (" + std::to_string(window_region->mapped_granules()) +
               " granules), main limit " + std::to_string(main_region->granule_limit()) + " = cap less the charge (" +
               std::to_string(restored) + ")");
    last_image.reset();
    c.release();
}

} // namespace

int main() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver || (err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        sinfer::DeviceContext ctx(0);
        check(sinfer::PagedKVPlaneOrder::PageMajor, ctx.stream);
        check(sinfer::PagedKVPlaneOrder::HeadMajor, ctx.stream);
        check_without_windows(ctx.stream);
        check_elastic(ctx);
        check_elastic_physical_bound(ctx);
        check_elastic_overshoot(ctx);
    } catch (const std::exception& error) {
        std::cerr << "FAIL (exception): " << error.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << "test_kv_window_pool: " << g_failures << " failures\n";
        return 1;
    }
    std::cout << "test_kv_window_pool: PASS\n";
    return 0;
}
