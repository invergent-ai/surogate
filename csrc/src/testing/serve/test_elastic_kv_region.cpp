// The demand-mapped KV plane region: a granule maps when its first page is taken and goes
// back once its last is returned, addresses hold across an unmap/remap so a captured graph
// keeps replaying into it, VRAM actually drops, and sleep/wake restore what was in use.
#include "core/device_footprint.h"
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

// What this process holds on the device.
//
// The two assertions below are about this region's own mapping and unmapping,
// and `cudaMemGetInfo` answers for the whole device: run beside anything else
// that allocates -- the rest of the suite under `ctest -j`, say -- a neighbour
// taking memory between the two samples masks the VRAM this region gave back,
// and "unmapping returned the VRAM" fails for a region that unmapped correctly.
// The per-process figure cancels that traffic. Where the driver will not
// attribute it (no NVML, a PID namespace), the caller below skips the
// comparison rather than making a device-wide claim about one process.
struct DeviceHeld {
    std::size_t bytes = 0;
    bool attributed   = false;
};

DeviceHeld device_held() {
    const sinfer::DeviceFootprint sample = sinfer::sample_device_footprint();
    if (sample.attributed) { return {sample.process_used_bytes, true}; }
    return {0, false};
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
    const DeviceHeld held_before = device_held();
    region.acquire_page(40);
    region.wait_idle();
    failures += expect(region.mapped_granules() == 1, "granule 1 mapped on demand");
    const DeviceHeld held_mapped_probe = device_held();
    if (held_before.attributed && held_mapped_probe.attributed) {
        failures += expect(held_mapped_probe.bytes - held_before.bytes >= 4 * kMiB,
                           "mapping took real VRAM");
    }

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
    const DeviceHeld held_mapped = device_held();
    region.release_page(40);
    region.wait_idle();
    failures += expect(region.mapped_granules() == 0, "empty granule unmapped after the fence");
    const DeviceHeld held_unmapped = device_held();
    if (held_mapped.attributed && held_unmapped.attributed) {
        failures += expect(held_mapped.bytes - held_unmapped.bytes >= 4 * kMiB,
                           "unmapping returned the VRAM");
    }

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

// Overcommit: the cap is a floor, growth past it is gated on what the device has free, and a
// refusal marks the device and asks the other regions there for their reserves.
std::size_t g_fake_free = 0;
std::size_t fake_free() { return g_fake_free; }

sinfer::ElasticKvRegionSpec overcommit_spec(sinfer::DeviceContext& device, std::uint32_t reserve) {
    sinfer::ElasticKvRegionSpec spec;
    spec.device           = device.device;
    spec.fence_stream     = device.stream;
    spec.page_count       = kPages;   // 4 granules of 32 pages
    spec.granule_pages    = kGranule;
    spec.reserve_granules = reserve;
    spec.cap_pages        = kGranule; // one granule guaranteed
    spec.overcommit       = true;
    spec.headroom_bytes   = 0; // the arithmetic below counts granules, not an engine's headroom
    spec.free_bytes_probe = &fake_free;
    spec.planes           = {{0, kPageBytes}, {8 * kMiB, kPageBytes}};
    spec.bytes            = 16 * kMiB;
    return spec;
}

int overcommit_cases(sinfer::DeviceContext& device) {
    int failures = 0;
    const int dev = device.device;
    g_fake_free   = 64 * kMiB; // construction charges the floor against this
    // A reserve of two, then a granule mapped and released: the reserve keeps it mapped until
    // a release_reserve (what a refused neighbour sends) unmaps it.
    {
        sinfer::ElasticKvRegion a(overcommit_spec(device, 2));
        a.wait_idle();
        a.acquire_page(0);
        a.release_page(0);
        a.wait_idle();
        failures += expect(a.mapped_granules() >= 1, "reserve keeps the freed granule mapped");
        a.release_reserve();          // any thread asks
        a.flush_reserve_release();    // the engine thread performs it at its round boundary
        a.wait_idle();
        failures += expect(a.mapped_granules() == 0, "release_reserve unmapped the reserve");
    }
    sinfer::ElasticKvRegion a(overcommit_spec(device, 0));
    sinfer::ElasticKvRegion b(overcommit_spec(device, 0));
    a.wait_idle();
    b.wait_idle();
    const std::size_t granule = a.granule_bytes(); // 4 MiB: 2 MiB in each plane
    failures += expect(sinfer::elastic_kv_unmapped_commitment(dev) == 2 * granule,
                       "two floors outstanding before any entitlement");

    // Within the floor: no probe consulted.
    g_fake_free = 0;
    failures += expect(a.try_entitle(kGranule), "the floor is always granted");
    failures += expect(!sinfer::elastic_kv_device_pressure(dev), "no pressure within the floor");

    // Past the floor: others' outstanding (b's floor, 1 granule) + growth (2) + slack (2) = 5
    // granules must fit.
    g_fake_free = 4 * granule;
    failures += expect(!a.try_entitle(2 * kGranule), "gate refuses when the device is short");
    failures += expect(sinfer::elastic_kv_device_pressure(dev), "a refusal marks pressure");
    g_fake_free = 5 * granule;
    failures += expect(a.try_entitle(2 * kGranule), "gate admits when it fits exactly");
    failures += expect(sinfer::elastic_kv_unmapped_commitment(dev) == 3 * granule,
                       "a's entitlement and b's floor are outstanding");

    // Mapping reduces what is outstanding: a maps its first granule.
    a.acquire_page(0);
    a.wait_idle();
    failures += expect(sinfer::elastic_kv_unmapped_commitment(dev) == 2 * granule,
                       "a mapped granule is no longer outstanding");
    // b growing to 3 granules: a's outstanding (1) + growth (3) + slack (2) = 6.
    g_fake_free = 5 * granule;
    failures += expect(!b.try_entitle(3 * kGranule), "b refused at five granules free");
    g_fake_free = 6 * granule;
    failures += expect(b.try_entitle(3 * kGranule), "b admitted at six granules free");
    // Shrinking never asks.
    g_fake_free = 0;
    failures += expect(b.try_entitle(kGranule), "shrinking is always granted");
    b.set_entitled_pages(0);
    failures += expect(sinfer::elastic_kv_unmapped_commitment(dev) == 2 * granule,
                       "b back to its floor, a's second granule outstanding");
    a.release_page(0);
    a.wait_idle();
    return failures;
}

int pool_overcommit_cases(sinfer::DeviceContext& device) {
    int failures = 0;
    sinfer::LayoutBuilder builder;
    sinfer::PagedKVPoolLayout layout =
        sinfer::plan_paged_kv_pool(builder, {.page_group_count      = 1024,
                                             .logical_page_capacity = 1024,
                                             .table_rows            = 2,
                                             .elastic               = true,
                                             .physical_page_cap     = 256, // one granule
                                             .overcommit            = true,
                                             .planes = {{sinfer::DType::I8, 64, 2},
                                                        {sinfer::DType::I8, 64, 2}}});
    const std::size_t arena_bytes = builder.finish(256, "test arena");
    sinfer::DeviceArena arena(arena_bytes);
    g_fake_free = 64 * kMiB;
    const sinfer::PagedKVElasticOptions options{.device           = device.device,
                                                .free_bytes_probe = &fake_free,
                                                .fence_stream     = device.stream,
                                                .reserve_granules = 0,
                                                .headroom_bytes   = 0};
    sinfer::PagedKVPool pool({arena.base(), arena.capacity()}, layout, &options);
    pool.elastic_region()->wait_idle();
    failures += expect(pool.capacity_pages() == 1024, "overcommit admits against the whole span");
    g_fake_free = 0;
    failures += expect(pool.can_reserve(200), "a reservation within the floor needs no free memory");
    failures += expect(!pool.can_reserve(300), "a reservation past the floor is gated");
    g_fake_free = 64 * kMiB;
    failures += expect(pool.can_reserve(300), "and admitted when the device has room");
    auto allocation = pool.reserve(300);
    failures += expect(pool.occupancy().entitled_pages == 300, "entitlement recorded");
    failures += expect(sinfer::elastic_kv_unmapped_commitment(device.device) ==
                           2 * pool.elastic_region()->granule_bytes(),
                       "two granules outstanding for 300 pages");
    allocation.release();
    pool.elastic_region()->wait_idle();
    failures += expect(pool.occupancy().entitled_pages == 0, "entitlement released");
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
        failures += overcommit_cases(device);
        failures += pool_overcommit_cases(device);
        if (failures == 0) { std::cout << "OK elastic kv region\n"; }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "elastic kv region test failed: " << error.what() << '\n';
        return 1;
    }
}
