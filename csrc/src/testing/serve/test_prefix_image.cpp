#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "family/impl/archive_storage.h"
#include <array>
#include <cassert>
#include <cstring>
#include <vector>

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
        // The same image into page-locked memory, without a wait per plane.
        {
            auto arena = sinfer::family::detail::PinnedArchiveArena::create(1U << 20);
            assert(arena);
            std::vector<std::optional<sinfer::family::detail::ArchiveBlock>> blocks;
            std::vector<std::byte*> planes;
            for (std::size_t p = 0; p < pool.plane_count(); ++p) {
                blocks.push_back(arena->allocate(pool.image_plane_bytes(p, scrambled.size())));
                assert(blocks.back() && blocks.back()->pinned);
                planes.push_back(blocks.back()->data);
            }
            pool.download_pages_to(scrambled, planes, device.stream);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            for (std::size_t p = 0; p < pool.plane_count(); ++p) {
                assert(image.planes[p].size() == pool.image_plane_bytes(p, scrambled.size()));
                assert(std::memcmp(planes[p], image.planes[p].data(), image.planes[p].size()) == 0);
            }
        }
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
    // The archive arena hands out first fit and coalesces what comes back.
    {
        auto arena = sinfer::family::detail::PinnedArchiveArena::create(4096);
        assert(arena);
        auto a = arena->allocate(1000);
        auto b = arena->allocate(1000);
        auto c = arena->allocate(2000);
        assert(a && b && c && a->pinned && a->size == 1000);
        assert(b->data == a->data + 1024 && c->data == b->data + 1024);
        assert(!arena->allocate(1));
        std::byte* const base = a->data;
        a.reset();
        b.reset();
        auto d = arena->allocate(2048);
        assert(d && d->data == base);
        c.reset();
        d.reset();
        auto whole = arena->allocate(4096);
        assert(whole && whole->data == base);
    }
}
