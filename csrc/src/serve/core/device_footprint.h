#pragma once

#include <cstddef>

namespace sinfer {

/// How much device memory is in use, measured two ways at one instant.
///
/// `cudaMemGetInfo` reports the *device's* free memory, so a delta across it
/// counts every process on the card. That is the right number for "is there
/// room left" and the wrong one for "how much did this engine allocate": a
/// second engine loading its weights inside the window is charged to the first,
/// which once aborted a startup with an allowance error naming bytes it had
/// never allocated.
///
/// NVML reports per-process usage, so a delta across `process_used_bytes`
/// cancels foreign traffic. It is not always available -- no driver library, no
/// permission, a PID namespace that hides this process from the device's list --
/// and `attributed` says whether the number means anything.
struct DeviceFootprint {
    std::size_t device_free_bytes  = 0;
    std::size_t process_used_bytes = 0;
    bool attributed                = false;
};

/// Samples both counters for the current CUDA device. Never throws; on any
/// failure the NVML half is simply not attributed.
[[nodiscard]] DeviceFootprint sample_device_footprint() noexcept;

/// What was allocated between two samples.
struct DeviceFootprintDelta {
    /// Bytes allocated. Attributed to this process where possible, otherwise
    /// the device-wide figure, which is an upper bound that includes whatever
    /// else ran on the card.
    std::size_t bytes = 0;
    /// True when `bytes` counts only this process. A caller that refuses to
    /// serve on the strength of this number must not do so when it is false:
    /// it cannot tell its own allocation from a neighbour's.
    bool attributed = false;
};

[[nodiscard]] DeviceFootprintDelta device_footprint_delta(const DeviceFootprint& before,
                                                          const DeviceFootprint& after) noexcept;

/// One line naming why attribution is unavailable, for a diagnostic. Empty
/// while attribution is working.
[[nodiscard]] const char* device_footprint_attribution_note() noexcept;

} // namespace sinfer
