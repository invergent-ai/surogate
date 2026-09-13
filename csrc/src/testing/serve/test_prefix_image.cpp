#include "core/device.h"
#include "core/paged_kv_cache.h"
#include <array>
#include <cassert>

using namespace sinfer;

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) { return 77; }
    DeviceContext device(0);
    for (auto order : {PagedKVPlaneOrder::PageMajor, PagedKVPlaneOrder::HeadMajor}) {
        PagedKVPoolSpec spec{.page_group_count = 32, .logical_page_capacity = 16,
            .table_rows = 2, .plane_order = order,
            .planes = {{DType::BF16, 16, 3}, {DType::FP8_E4M3FN, 32, 2},
                       {DType::FP32, 3, 2}, {DType::I32, 1, 1}}};
        LayoutBuilder layout;
        const auto plan = plan_paged_kv_pool(layout, spec);
        DeviceArena memory(layout.finish());
        PagedKVPool pool({memory.base(), memory.capacity()}, plan);
        auto allocation = pool.reserve(8);
        allocation.materialize_pages(8, device.stream);
        const std::array<int, 8> scrambled{7,1,4,2,0,6,3,5};
        for (unsigned i = 0; i < scrambled.size(); ++i) {
            pool.zero_pages(std::span(&scrambled[i], 1), device.stream, 17 + i);
        }
        const auto image = pool.download_pages(scrambled, device.stream);
        // Reject invalid input before touching any destination plane.
        auto invalid = image;
        invalid.planes.back().pop_back();
        bool refused = false;
        try { pool.upload_pages(invalid, scrambled, device.stream); }
        catch (const std::invalid_argument&) { refused = true; }
        assert(refused);
        auto invalid_ids = scrambled;
        invalid_ids.back() = 32;
        refused = false;
        try { (void)pool.download_pages(invalid_ids, device.stream); }
        catch (const std::out_of_range&) { refused = true; }
        assert(refused);
        allocation.release();
        auto occupied = pool.reserve(8);
        occupied.materialize_pages(8, device.stream);
        auto destination = pool.reserve(8);
        destination.materialize_pages(8, device.stream);
        assert(destination.page_ids().front() >= 8);
        pool.upload_pages(image, destination.page_ids(), device.stream);
        const auto restored = pool.download_pages(destination.page_ids(), device.stream);
        assert(restored.planes == image.planes);
        // Prefix restore can stop before the saved frontier, including with head-major planes.
        pool.zero_pages(destination.page_ids(), device.stream);
        pool.upload_pages(image, destination.page_ids().first(3), device.stream);
        const auto prefix = pool.download_pages(destination.page_ids().first(3), device.stream);
        for (std::size_t p = 0; p < spec.planes.size(); ++p) {
            const auto heads = order == PagedKVPlaneOrder::PageMajor ? 1 : spec.planes[p].head_extent;
            const auto bytes = prefix.planes[p].size() / heads;
            for (int head = 0; head < heads; ++head) {
                assert(std::equal(prefix.planes[p].begin() + head * bytes,
                                  prefix.planes[p].begin() + (head + 1) * bytes,
                                  image.planes[p].begin() + head * image.planes[p].size() / heads));
            }
        }
    }
}
