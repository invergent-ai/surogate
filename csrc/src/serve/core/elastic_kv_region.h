#pragma once

// A KV plane region whose physical memory follows demand.
//
// The pool's planes are reserved as one virtual span and backed granule by
// granule through CUDA VMM (cuMemAddressReserve + cuMemCreate/cuMemMap), the
// same primitive sleep mode uses for whole arenas. A granule is the run of
// pages whose bytes, in every plane, fill a whole mapping quantum; it is
// mapped when the first of its pages is taken and eligible to go back once
// the last is returned. Addresses never move, so tensors, block tables and
// captured graphs keep pointing where they always did; a kernel only ever
// reaches a page a published block table names, and those are mapped.
//
// Two policies keep the driver calls off the round loop. A reserve of the
// lowest unmapped granules is mapped ahead on a worker thread, because the
// pool hands out low page ids first; and a granule that empties is unmapped
// on that thread only after an event recorded on the engine's stream has
// passed, so nothing in flight can still touch it. The executor thread maps
// synchronously only when demand outruns the reserve.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sinfer {

struct ElasticKvPlane {
    std::size_t offset     = 0; ///< byte offset of the plane within the region
    std::size_t page_bytes = 0; ///< bytes one page occupies in this plane (its page stride)
};

struct ElasticKvRegionSpec {
    int device                 = 0;
    cudaStream_t fence_stream  = nullptr; ///< the stream whose completion retires a granule
    std::size_t bytes          = 0;       ///< virtual span covering every plane
    std::uint32_t page_count   = 0;
    std::uint32_t granule_pages = 0; ///< pages per mapping granule, every plane's span a quantum multiple
    std::uint32_t reserve_granules = 4; ///< mapped-but-free granules kept ahead of demand; trimming starts at twice this
    /// Pages that may be physically mapped at once (0 = all). Its bytes are committed to the
    /// process-wide elastic budget at construction: a later engine on the same device sees
    /// this region's unmapped remainder as spoken for, so co-resident pools that could not
    /// all fill at once are refused at startup, as arena pools are, not mid-round.
    std::uint32_t cap_pages        = 0;
    /// Overcommit: `cap_pages` is a guaranteed floor rather than a ceiling. Entitlements past
    /// it are admitted through `try_entitle`, a device-wide gate over every region's
    /// outstanding (entitled or guaranteed, not yet mapped) bytes against the memory actually
    /// free, so co-resident pools share the device's idle KV instead of each holding its own.
    bool overcommit                = false;
    /// Bytes the gate never entitles into: CUDA graphs are captured lazily per bucket and a
    /// capture that cannot instantiate is fatal, and workspaces grow with the widest round.
    std::size_t headroom_bytes     = 0;
    /// Tests: what the gate reads as free device bytes (default cudaMemGetInfo).
    std::size_t (*free_bytes_probe)() = nullptr;
    std::vector<ElasticKvPlane> planes;
};

/// Bytes every elastic region on `device` could still map that are spoken for — its guaranteed
/// cap or its entitlement, whichever is larger, less what it has mapped: what a new engine must
/// leave free. Zero when no region is registered there.
[[nodiscard]] std::size_t elastic_kv_unmapped_commitment(int device) noexcept;

/// True while a region on `device` was recently refused an entitlement by the gate: regions
/// there give up their reserve and executors give up retained lanes until it clears.
[[nodiscard]] bool elastic_kv_device_pressure(int device) noexcept;

class ElasticKvRegion {
public:
    explicit ElasticKvRegion(ElasticKvRegionSpec spec);
    ~ElasticKvRegion();

    ElasticKvRegion(const ElasticKvRegion&)            = delete;
    ElasticKvRegion& operator=(const ElasticKvRegion&) = delete;

    [[nodiscard]] void* base() const noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept;
    [[nodiscard]] std::uint32_t granule_count() const noexcept;
    [[nodiscard]] std::uint32_t granule_pages() const noexcept;
    /// Physical bytes one granule costs across every plane.
    [[nodiscard]] std::size_t granule_bytes() const noexcept;

    /// The page is now in use: its granule is mapped before this returns.
    void acquire_page(std::int32_t page);
    /// The page is free again. A granule with no pages left is unmapped on the worker once
    /// the fence stream has drained, unless it is inside the reserve.
    void release_page(std::int32_t page) noexcept;

    [[nodiscard]] std::size_t mapped_bytes() const noexcept;
    [[nodiscard]] std::uint32_t mapped_granules() const noexcept;

    /// Sleep mode: back up every granule holding pages, unmap everything. Returns bytes released.
    std::size_t sleep();
    /// Map the granules that held pages, restore them, and let the reserve refill. Returns bytes mapped.
    std::size_t wake();

    /// Overcommit gate: may the pool's entitlement become `pages`? Within the guaranteed cap
    /// always; beyond it when every region's outstanding bytes plus this growth still fit the
    /// device's free memory. Records the entitlement when it says yes; a no flags pressure on
    /// the device and asks the other regions there to release their reserves.
    [[nodiscard]] bool try_entitle(std::uint32_t pages) noexcept;
    /// Records the pool's entitlement without gating (decreases, and the per-round resync).
    void set_entitled_pages(std::uint32_t pages) noexcept;
    /// Another region on the device is short: give back every free granule, reserve included.
    /// Any thread may ask; the engine's own thread performs it at its next round boundary
    /// through `flush_reserve_release`, since only that thread may fence the engine's stream.
    void release_reserve() noexcept;
    void flush_reserve_release() noexcept;

    /// Blocks until the worker has no pending map or unmap work (tests, teardown).
    void wait_idle();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sinfer
