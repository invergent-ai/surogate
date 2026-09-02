// The demand-mapped KV plane region: a granule maps when its first page is taken and goes
// back once its last is returned, addresses hold across an unmap/remap so a captured graph
// keeps replaying into it, VRAM actually drops, and sleep/wake restore what was in use.
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/elastic_kv_region.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t kMiB       = std::size_t{1} << 20;
constexpr std::size_t kPageBytes = std::size_t{64} << 10; // 64 KiB per page per plane
constexpr std::uint32_t kPages   = 128;                    // 4 granules of 32 pages at 2 MiB
constexpr std::uint32_t kGranule = 32;

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect(bool ok, const char* label) {
    if (ok) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

std::size_t device_free() {
    std::size_t free_bytes = 0, total = 0;
    (void)cudaMemGetInfo(&free_bytes, &total);
    return free_bytes;
}

unsigned char* page_address(sinfer::ElasticKvRegion& region, std::size_t plane_offset,
                            std::int32_t page) {
    return static_cast<unsigned char*>(region.base()) + plane_offset + kPageBytes * page;
}

int read_byte(const void* device_pointer, unsigned char& out) {
    return expect(cudaMemcpy(&out, device_pointer, 1, cudaMemcpyDeviceToHost) == cudaSuccess,
                  "device read");
}

int region_cases(sinfer::DeviceContext& device) {
    int failures = 0;
    sinfer::ElasticKvRegionSpec spec;
    spec.device           = device.device;
    spec.fence_stream     = device.stream;
    spec.page_count       = kPages;
    spec.granule_pages    = kGranule;
    spec.reserve_granules = 0; // map on demand only, so every transition is deterministic
    // Two planes, each 128 pages of 64 KiB = 8 MiB, on 2 MiB boundaries.
    spec.planes = {{0, kPageBytes}, {8 * kMiB, kPageBytes}};
    spec.bytes  = 16 * kMiB;
    sinfer::ElasticKvRegion region(spec);
    region.wait_idle();

    failures += expect(region.granule_count() == 4, "four granules");
    failures += expect(region.granule_bytes() == 4 * kMiB, "granule spans 2 MiB in each plane");
    failures += expect(region.mapped_granules() == 0, "nothing mapped before demand");
    failures += expect(region.mapped_bytes() == 0, "no footprint before demand");

    // Page 40 lives in granule 1: acquiring it maps on the spot.
    const std::size_t free_before = device_free();
    region.acquire_page(40);
    region.wait_idle();
    failures += expect(region.mapped_granules() == 1, "granule 1 mapped on demand");
    failures += expect(free_before - device_free() >= 4 * kMiB, "mapping took real VRAM");

    // A captured graph writes into page 40 of plane 0.
    unsigned char* target = page_address(region, spec.planes[0].offset, 40);
    sinfer::DecodeGraphDefinition definition;
    definition.capture(device.stream, [&] {
        CUDA_CHECK(cudaMemsetAsync(target, 0x5A, kPageBytes, device.stream));
    });
    sinfer::DecodeGraphExecutable executable;
    executable.instantiate(definition);
    executable.upload(device.stream);
    executable.launch(device.stream);
    device.synchronize();
    unsigned char value = 0;
    failures += read_byte(target, value);
    failures += expect(value == 0x5A, "graph wrote the mapped page");

    // Release: the granule empties, the fence passes, the worker unmaps it, and the VRAM
    // comes back.
    const std::size_t free_mapped = device_free();
    region.release_page(40);
    region.wait_idle();
    failures += expect(region.mapped_granules() == 0, "empty granule unmapped after the fence");
    failures += expect(device_free() - free_mapped >= 4 * kMiB, "unmapping returned the VRAM");

    // Remap by taking the page again, replay the same executable: the addresses it baked
    // are the same addresses, now backed by fresh pages.
    region.acquire_page(40);
    region.wait_idle();
    executable.launch(device.stream);
    device.synchronize();
    value = 0;
    failures += read_byte(target, value);
    failures += expect(value == 0x5A, "graph replays into the remapped page");

    // Sleep backs up the granule in use, drops everything; wake brings the contents back.
    CUDA_CHECK(cudaMemset(target, 0x77, kPageBytes));
    device.synchronize();
    const std::size_t released = region.sleep();
    failures += expect(released == region.granule_bytes(), "sleep released the mapped granule");
    failures += expect(region.mapped_granules() == 0, "nothing mapped while asleep");
    const std::size_t woken = region.wake();
    region.wait_idle();
    failures += expect(woken == region.granule_bytes(), "wake remapped the granule in use");
    value = 0;
    failures += read_byte(target, value);
    failures += expect(value == 0x77, "wake restored the page contents");
    region.release_page(40);
    region.wait_idle();
    return failures;
}

int pool_cases(sinfer::DeviceContext& device) {
    int failures = 0;
    // Two I8 planes of {64, 64, 2}: 8 KiB per page per plane, so a 2 MiB granule is 256 pages;
    // 1024 pages make four granules.
    sinfer::LayoutBuilder builder;
    sinfer::PagedKVPoolLayout layout =
        sinfer::plan_paged_kv_pool(builder, {.page_group_count      = 1024,
                                             .logical_page_capacity = 1024,
                                             .table_rows            = 2,
                                             .elastic               = true,
                                             .planes = {{sinfer::DType::I8, 64, 2},
                                                        {sinfer::DType::I8, 64, 2}}});
    failures += expect(layout.elastic_granule_pages == 256, "granule pages from the stride");
    failures += expect(layout.elastic_plane_bytes == 16 * kMiB, "planes span two 8 MiB regions");
    const std::size_t arena_bytes = builder.finish(256, "test arena");
    failures += expect(arena_bytes < kMiB, "only the block tables stay in the arena");

    sinfer::DeviceArena arena(arena_bytes);
    const sinfer::PagedKVElasticOptions options{
        .device = device.device, .fence_stream = device.stream, .reserve_granules = 0};
    sinfer::PagedKVPool pool({arena.base(), arena.capacity()}, layout, &options);
    sinfer::ElasticKvRegion* region = pool.elastic_region();
    failures += expect(region != nullptr, "the pool owns a region");
    if (region == nullptr) { return failures + 1; }
    region->wait_idle();
    failures += expect(pool.occupancy().mapped_pages == 0, "nothing mapped before demand");

    // 300 pages straddle granules 0 and 1.
    auto allocation = pool.reserve(300);
    allocation.materialize_pages(300);
    allocation.bind_row(0, device.stream);
    region->wait_idle();
    const sinfer::PagedKVOccupancy busy = pool.occupancy();
    failures += expect(busy.pages_in_use == 300, "pages in use");
    failures += expect(busy.mapped_pages == 512, "the two granules in use are mapped");

    // Zeroing the pages it holds touches only mapped memory.
    const std::int32_t ids[] = {0, 299};
    pool.zero_pages(ids, device.stream);
    device.synchronize();

    allocation.release();
    region->wait_idle();
    failures += expect(pool.occupancy().pages_in_use == 0, "pages returned");
    failures += expect(pool.occupancy().mapped_pages == 0, "released granules unmapped");
    return failures;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    try {
        sinfer::DeviceContext device(0);
        int failures = region_cases(device);
        failures += pool_cases(device);
        if (failures == 0) { std::cout << "OK elastic kv region\n"; }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "elastic kv region test failed: " << error.what() << '\n';
        return 1;
    }
}
