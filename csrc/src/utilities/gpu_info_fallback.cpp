// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_info.h"

#include "utils.h"
#include "serve/core/unified_memory.h"

/**
 * @brief Fallback implementation of IGPUUtilTracker when no vendor/OS-specific
 *        telemetry backend is available.
 *
 * This implementation only reports values obtainable via the CUDA Runtime API.
 * Most utilization/telemetry fields are filled with sentinel values (e.g. -1)
 * and the throttle reason is set to "not supported".
 */
class GPUUtilTrackerFallback : public IGPUUtilTracker {
public:
    /**
     * @brief Construct the fallback tracker.
     *
     * Side effects:
     * - Queries and stores the current CUDA device ID via cudaGetDevice().
     *
     * @throws On CUDA errors via CUDA_CHECK.
     */
    GPUUtilTrackerFallback();

    /**
     * @brief Destroy the fallback tracker.
     *
     * No resources are owned beyond simple POD state.
     */
    ~GPUUtilTrackerFallback();

    /**
     * @brief Update and return the latest GPU utilization/telemetry snapshot.
     *
     * Populates supported fields using CUDA Runtime queries and sets unsupported
     * fields to sentinel values.
     *
     * @return Reference to an internally stored GPUUtilInfo snapshot that remains
     *         valid until the next call to update() or destruction.
     *
     * @throws On CUDA errors via CUDA_CHECK.
     */
    const GPUUtilInfo& update() override;

private:
    /// Cached output structure returned by update().
    GPUUtilInfo mInfo;

    /// CUDA device ID captured at construction time.
    int mDeviceID;
};

/**
 * @brief Factory for an IGPUUtilTracker instance.
 *
 * @return A tracker instance that provides a CUDA-only fallback implementation.
 */
std::unique_ptr<IGPUUtilTracker> IGPUUtilTracker::create() {
    return std::make_unique<GPUUtilTrackerFallback>();
}

GPUUtilTrackerFallback::GPUUtilTrackerFallback() {
    CUDA_CHECK(cudaGetDevice(&mDeviceID));
}

GPUUtilTrackerFallback::~GPUUtilTrackerFallback() {
}

const GPUUtilInfo& GPUUtilTrackerFallback::update() {
    // just return fallback values
    mInfo.clock = 0;
    mInfo.max_clock = 0;
    mInfo.power_limit = 0;
    mInfo.temperature = 0;
    mInfo.temp_slowdown = 0;
    mInfo.fan = 0;
    mInfo.gpu_utilization = -1.f;
    mInfo.mem_utilization = -1.f;
    mInfo.throttle_reason = "not supported";
    mInfo.pcie_rx = -1;
    mInfo.pcie_tx = -1;
    mInfo.power = -1;

    CUDA_CHECK(sinfer::device_mem_get_info(&mInfo.mem_free, &mInfo.mem_total));
    mInfo.mem_reserved = -1;
    int clockRateKHz;
    CUDA_CHECK(cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, mDeviceID));
    mInfo.max_clock = clockRateKHz / 1000;
    return mInfo;
}

/**
 * @brief Retrieve the amount of CUDA memory reserved by the runtime/allocator.
 *
 * Fallback implementation: not supported.
 *
 * @return Always 0 in this fallback.
 */
std::size_t get_mem_reserved() {
    return 0;
}

/**
 * @brief Get the name of the currently selected CUDA device.
 *
 * @return Device name as reported by cudaGetDeviceProperties().
 *
 * @throws On CUDA errors via CUDA_CHECK.
 */
std::string get_gpu_name() {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    return prop.name;
}

/**
 * @brief Attempt to set CPU affinity for the current process/thread.
 *
 * Fallback implementation: not supported.
 *
 * @return Always false in this fallback, indicating no affinity was applied.
 */
bool set_cpu_affinity() {
    return false;
}
