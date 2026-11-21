// Window-bounded KV pages (lane f1, rl/engine-inkling-2026-11-21/lanes/f1/DESIGN-window-kv.md): the
// paged pool's window class on a device -- layout, zeroed sentinel, block-table rows, page
// recycling, images and forks, window page up/download. Exits 77 without a usable device.

#include "core/device.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
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
