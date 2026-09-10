#pragma once

// Sleep mode's memory machinery: CUDA VMM-backed regions whose virtual
// addresses outlive their physical pages.
//
// The engine's estate is a handful of owning DeviceArenas, and everything else
// -- tensors, weights, the addresses baked into captured CUDA graphs -- is a
// non-owning view into them. Sleeping therefore cannot move anything: a freed
// and re-cudaMalloc'd arena would land elsewhere and every captured graph would
// be garbage. Instead, when sleep mode is enabled the owning arenas allocate
// through cuMemAddressReserve + cuMemCreate/cuMemMap: sleep unmaps and releases
// the physical pages (VRAM actually drops) while the reservation keeps the
// addresses; wake maps fresh pages at the same addresses. Verified on the
// RTX 5090: a graph captured before sleep replays correctly after wake.
//
// Regions are tagged. Offload regions are copied to pinned host on sleep and
// restored on wake -- byte-identical, so weights, KV values and every
// init-once device fill survive. Discard regions (per-round and per-request
// scratch, rewritten before every read) come back as garbage. The default is
// Offload: an arena nobody thought about costs host memory, never corruption.
//
// Registration happens at startup on one thread; sleep_device/wake_device are
// serialized by the engine, which parks its worker loop first.

#include <cstddef>
#include <functional>

namespace sinfer {

enum class SleepTag {
    Offload, ///< backed up to pinned host on sleep, restored on wake
    Discard, ///< physical pages dropped; contents are garbage after wake
};

/// When true, owning DeviceArenas allocate through the VMM path and join the
/// registry. Set by the Engine constructor around target construction.
[[nodiscard]] bool sleepable_allocations_enabled() noexcept;
void set_sleepable_allocations(bool enabled) noexcept;

/// Reserve + map a sleepable region on `device`, registered with
/// SleepTag::Offload and owned by the thread's bound ops context, so one
/// engine's sleep leaves its neighbours' memory mapped. A null `owner` filter
/// on the operations below matches every region (tools, single-engine paths).
[[nodiscard]] void* sleep_alloc(std::size_t bytes, int device);

/// Tear down a sleepable region (any state). Returns false when `base` is not a
/// registered region -- the caller then owns freeing it the ordinary way.
bool sleep_free(void* base) noexcept;

/// Change a registered region's tag (startup only, before the first sleep).
void sleep_tag_region(const void* base, SleepTag tag);

/// A region that maps itself granule by granule (core/elastic_kv_region.h) joins the
/// registry through hooks instead of a fixed span: sleep and wake are its own, and its
/// footprint is whatever it has mapped right now. Offload semantics are the region's job.
struct SparseRegionHooks {
    const void* owner = nullptr;
    int device        = 0;
    std::function<std::size_t()> sleep_fn;     ///< back up, unmap; returns bytes released
    std::function<std::size_t()> wake_fn;      ///< remap, restore; returns bytes mapped
    std::function<std::size_t()> mapped_bytes; ///< current physical footprint
};
void sleep_register_sparse(const void* key, SparseRegionHooks hooks);
void sleep_unregister_sparse(const void* key) noexcept;

/// Back up Offload regions to pinned host, then unmap and release the physical
/// pages of every region on `device`. Returns bytes released. The caller must
/// have quiesced all work on the device.
std::size_t sleep_device(int device, const void* owner = nullptr);

/// Map fresh physical pages at the original addresses and restore Offload
/// regions. Returns bytes mapped. Throws on allocation failure (VRAM taken by
/// another process while asleep); regions already woken stay woken, so a retry
/// finishes the job.
std::size_t wake_device(int device, const void* owner = nullptr);

[[nodiscard]] bool device_asleep(int device, const void* owner = nullptr) noexcept;

/// Pinned-host bytes currently held as sleep backups for `device`.
[[nodiscard]] std::size_t sleep_backup_bytes(int device) noexcept;

/// Mapped bytes of the regions owned by `owner` (their VRAM footprint while
/// awake). The scheduler sizes resident sets with this.
[[nodiscard]] std::size_t sleep_owned_bytes(const void* owner, int device = -1) noexcept;

/// Allocate the pinned host backups for `owner`'s Offload regions without
/// sleeping anything. First-time pinning runs at ~2 GiB/s, so a model's first
/// eviction would otherwise stall a *different* model's requester by many
/// seconds; the scheduler pays this at startup instead.
void sleep_prepare_backups(const void* owner);

/// Free VRAM on `device` right now (cudaMemGetInfo), for budget planning by
/// callers that are not CUDA translation units themselves.
[[nodiscard]] std::size_t device_free_bytes(int device) noexcept;

} // namespace sinfer
