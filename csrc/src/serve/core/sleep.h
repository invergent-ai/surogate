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

namespace sinfer {

enum class SleepTag {
    Offload, ///< backed up to pinned host on sleep, restored on wake
    Discard, ///< physical pages dropped; contents are garbage after wake
};

/// When true, owning DeviceArenas allocate through the VMM path and join the
/// registry. Set by the Engine constructor around target construction.
[[nodiscard]] bool sleepable_allocations_enabled() noexcept;
void set_sleepable_allocations(bool enabled) noexcept;

/// Reserve + map a sleepable region on `device`. Registered with SleepTag::Offload.
[[nodiscard]] void* sleep_alloc(std::size_t bytes, int device);

/// Tear down a sleepable region (any state). Returns false when `base` is not a
/// registered region -- the caller then owns freeing it the ordinary way.
bool sleep_free(void* base) noexcept;

/// Change a registered region's tag (startup only, before the first sleep).
void sleep_tag_region(const void* base, SleepTag tag);

/// Back up Offload regions to pinned host, then unmap and release the physical
/// pages of every region on `device`. Returns bytes released. The caller must
/// have quiesced all work on the device.
std::size_t sleep_device(int device);

/// Map fresh physical pages at the original addresses and restore Offload
/// regions. Returns bytes mapped. Throws on allocation failure (VRAM taken by
/// another process while asleep); regions already woken stay woken, so a retry
/// finishes the job.
std::size_t wake_device(int device);

[[nodiscard]] bool device_asleep(int device) noexcept;

/// Pinned-host bytes currently held as sleep backups for `device`.
[[nodiscard]] std::size_t sleep_backup_bytes(int device) noexcept;

} // namespace sinfer
