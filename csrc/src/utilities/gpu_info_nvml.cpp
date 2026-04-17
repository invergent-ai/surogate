// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "gpu_info.h"

#include <type_traits>

#include <fmt/core.h>
#include <nvml.h>

#include "utils.h"

/**
 * @brief Check an NVML API call result and handle errors/warnings.
 *
 * If @p allow_unsupported is true, NVML "not supported" and "no permission"
 * errors are downgraded to a one-time warning; otherwise, non-success results
 * throw an exception.
 *
 * @param status NVML return code from an NVML call.
 * @param file Source file where the check is performed (typically __FILE__).
 * @param line Source line where the check is performed (typically __LINE__).
 * @param allow_unsupported If true, tolerate unsupported/permission errors with a warning.
 * @return The original @p status (useful for conditional handling by caller).
 * @throws std::runtime_error For non-success errors when not tolerated.
 */
inline nvmlReturn_t nvml_check(nvmlReturn_t status, const char* file, int line, bool allow_unsupported) {
    if (status == NVML_ERROR_NOT_SUPPORTED && allow_unsupported) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[NVML WARNING] NVML_ERROR_NOT_SUPPORTED\n");
        }
    } else if (status == NVML_ERROR_NO_PERMISSION && allow_unsupported) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[NVML WARNING] NVML_ERROR_NO_PERMISSION\n");
        }
    } else if (status != NVML_SUCCESS) {
        throw std::runtime_error(fmt::format("[NVML ERROR] at file {}:{}:\n{}\n", file, line, nvmlErrorString(status)));
    }
    return status;
};

// the _U version of the macro does not throw an exception on unsupported features
#define NVML_CHECK(err) (nvml_check(err, __FILE__, __LINE__, false))
#define NVML_CHECK_U(err) (nvml_check(err, __FILE__, __LINE__, true))

/**
 * @brief Get the NVML device handle corresponding to the current CUDA device.
 *
 * Initializes NVML once per thread (thread-local) and resolves the current CUDA device
 * to an NVML device handle via PCI bus ID.
 *
 * @return nvmlDevice_t Handle for the active CUDA device in the calling thread.
 * @throws std::runtime_error If NVML initialization/lookup fails.
 */
inline nvmlDevice_t nvml_get_device() {
    thread_local bool needs_init = true;
    thread_local nvmlDevice_t device;
    if (needs_init) {
        needs_init = false;
        NVML_CHECK(nvmlInit());
        char bus_id[256];
        int did;
        CUDA_CHECK(cudaGetDevice(&did));
        CUDA_CHECK(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), did));
        NVML_CHECK(nvmlDeviceGetHandleByPciBusId(bus_id, &device));
    }
    return device;
}

/**
 * @brief Convert NVML throttle reason bitmask into a user-facing string.
 *
 * @param bits Bitmask as returned by nvmlDeviceGetCurrentClocksThrottleReasons().
 * @return A static string describing the dominant throttle reason category.
 */
inline const char* get_throttle_reason(unsigned long long bits) {
    if (bits & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)) {
        return "power cap";
    } else if (bits & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)) {
        return "thermal cap";
    } else if (bits & (nvmlClocksThrottleReasonGpuIdle)) {
        return "idle";
    } else if (bits & (nvmlClocksThrottleReasonAll)) {
        return "other cap";
    } else {
        return "no cap";
    }
}

/**
 * @brief NVML-based GPU utilization tracker.
 *
 * Uses direct NVML queries for instantaneous values (clocks, temperatures, etc.)
 * and a background sampling thread for PCIe RX/TX counters and energy deltas,
 * which are converted into rates during update().
 */
class GPUUtilTrackerNVML : public IGPUUtilTracker {
public:
    /**
     * @brief Construct the tracker and start background sampling if supported.
     *
     * Attempts to query PCIe counters and total energy consumption. If unsupported,
     * the tracker still functions but will not report PCIe/energy-rate metrics.
     */
    GPUUtilTrackerNVML();

    /**
     * @brief Stop background sampling (if running) and shut down NVML.
     */
    ~GPUUtilTrackerNVML();

    /**
     * @brief Refresh tracked GPU metrics and return the latest snapshot.
     *
     * @return Reference to an internally-stored GPUUtilInfo struct updated with the latest values.
     */
    const GPUUtilInfo& update() override;

private:
    GPUUtilInfo mInfo;
    nvmlDevice_t mDevice;

    /**
     * @brief Start the background thread that periodically samples PCIe and energy counters.
     *
     * The thread accumulates deltas into atomic interval counters, which are converted to
     * rates (bytes/sec, energy units/sec) by update().
     */
    void setup_tracking_thread();

    long long mLastTimestamp;  // µs
    std::size_t mLastPCIeRX;
    std::size_t mLastPCIeTX;
    unsigned long long mLastEnergy;

    std::atomic<std::size_t> mIntervalPCIeRX{0};
    std::atomic<std::size_t> mIntervalPCIeTX{0};
    std::atomic<std::size_t> mIntervalEnergy{0};

    std::size_t mTotalPCIeRX;
    std::size_t mTotalPCIeTX;
    std::size_t mTotalEnergy;

    std::jthread mThread;
};

/**
 * @brief Factory for creating the default GPU utilization tracker implementation.
 *
 * @return A new tracker instance as a polymorphic IGPUUtilTracker.
 */
std::unique_ptr<IGPUUtilTracker> IGPUUtilTracker::create() {
    return std::make_unique<GPUUtilTrackerNVML>();
}

GPUUtilTrackerNVML::GPUUtilTrackerNVML()
    : mDevice(nvml_get_device()) {
    nvmlFieldValue_t fields[] = {{NVML_FI_DEV_PCIE_COUNT_RX_BYTES},
                                 {NVML_FI_DEV_PCIE_COUNT_TX_BYTES},
                                 {NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION, 0}};
    NVML_CHECK(nvmlDeviceGetFieldValues(mDevice, 3, fields));
    if (fields[0].nvmlReturn == NVML_ERROR_NOT_SUPPORTED || fields[1].nvmlReturn == NVML_ERROR_NOT_SUPPORTED ||
        fields[2].nvmlReturn == NVML_ERROR_NOT_SUPPORTED) {
        fprintf(stderr, "[NVML WARNING] PCIe counters not supported\n");
    } else {
        NVML_CHECK(fields[0].nvmlReturn);
        NVML_CHECK(fields[1].nvmlReturn);
        NVML_CHECK(fields[2].nvmlReturn);
        mLastPCIeRX = fields[0].value.uiVal;
        mLastPCIeTX = fields[1].value.uiVal;
        mLastEnergy = fields[2].value.ullVal;

        mLastTimestamp = std::chrono::steady_clock::now().time_since_epoch().count();

        setup_tracking_thread();
    }
}

GPUUtilTrackerNVML::~GPUUtilTrackerNVML() {
    if (mThread.joinable()) {
        mThread.request_stop();
        mThread.join();
    }
    NVML_CHECK(nvmlShutdown());
}

void GPUUtilTrackerNVML::setup_tracking_thread() {
    // TODO should this be one thread for all devices?
    mThread = std::jthread([this](std::stop_token stop_token) {
        NVML_CHECK(nvmlDeviceSetCpuAffinity(mDevice));

        nvmlFieldValue_t fields[] = {{NVML_FI_DEV_PCIE_COUNT_RX_BYTES},
                                     {NVML_FI_DEV_PCIE_COUNT_TX_BYTES},
                                     {NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION, 0}};
        while (true) {
            NVML_CHECK(nvmlDeviceGetFieldValues(mDevice, 3, fields));
            NVML_CHECK(fields[0].nvmlReturn);
            NVML_CHECK(fields[1].nvmlReturn);
            NVML_CHECK(fields[2].nvmlReturn);
            unsigned pcie_rx, pcie_tx;
            unsigned long long energy;
            if (mLastPCIeRX <= fields[0].value.uiVal) {
                pcie_rx = fields[0].value.uiVal - mLastPCIeRX;
            } else {
                pcie_rx = std::numeric_limits<unsigned>::max() - mLastPCIeRX + fields[0].value.uiVal;
            }

            if (mLastPCIeTX <= fields[1].value.uiVal) {
                pcie_tx = fields[1].value.uiVal - mLastPCIeTX;
            } else {
                pcie_tx = std::numeric_limits<unsigned>::max() - mLastPCIeTX + fields[1].value.uiVal;
            }

            if (mLastEnergy <= fields[2].value.ullVal) {
                energy = fields[2].value.ullVal - mLastEnergy;
            } else {
                energy = (std::numeric_limits<unsigned long long>::max() - mLastEnergy) + fields[2].value.ullVal;
            }

            mIntervalPCIeRX += pcie_rx;
            mIntervalPCIeTX += pcie_tx;
            mIntervalEnergy += energy;
            mLastPCIeRX = fields[0].value.uiVal;
            mLastPCIeTX = fields[1].value.uiVal;
            mLastEnergy = fields[2].value.ullVal;

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (stop_token.stop_requested()) {
                break;
            }
        }
    });
}

/**
 * @brief Query extended NVML memory information for a device.
 *
 * Uses nvmlDeviceGetMemoryInfo_v2 when available; otherwise falls back to CUDA's
 * cudaMemGetInfo() and sets reserved/used to best-effort defaults.
 *
 * @param device NVML device handle to query.
 * @return nvmlMemory_v2_t Filled memory information struct (version set by callee).
 */
nvmlMemory_v2_t get_mem_info(nvmlDevice_t device);

const GPUUtilInfo& GPUUtilTrackerNVML::update() {
    // set default values to indicate not-supported operations
    mInfo.clock = 0;
    mInfo.max_clock = 0;
    mInfo.power_limit = 0;
    mInfo.temperature = 0;
    mInfo.temp_slowdown = 0;
    mInfo.fan = 0;
    mInfo.gpu_utilization = -1.f;
    mInfo.mem_utilization = -1.f;
    mInfo.throttle_reason = "not supported";

    // query different infos directly
    NVML_CHECK_U(nvmlDeviceGetClockInfo(mDevice, NVML_CLOCK_SM, &mInfo.clock));
    NVML_CHECK_U(nvmlDeviceGetMaxClockInfo(mDevice, NVML_CLOCK_SM, &mInfo.max_clock));
    NVML_CHECK_U(nvmlDeviceGetPowerManagementLimit(mDevice, &mInfo.power_limit));
    NVML_CHECK_U(nvmlDeviceGetTemperature(mDevice, NVML_TEMPERATURE_GPU, &mInfo.temperature));
    NVML_CHECK_U(nvmlDeviceGetTemperatureThreshold(mDevice, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &mInfo.temp_slowdown));
    unsigned long long throttle;
    if (NVML_CHECK_U(nvmlDeviceGetCurrentClocksThrottleReasons(mDevice, &throttle)) == NVML_SUCCESS) {
        mInfo.throttle_reason = get_throttle_reason(throttle);
    }
    NVML_CHECK_U(nvmlDeviceGetFanSpeed(mDevice, &mInfo.fan));

    // For "utilization", we look at recorded samples. In principle, we could query the driver for how many samples
    // to request, but then we'd need to dynamically allocate sufficient space. Let's just hard-code a limit of 128,
    // and have no memory management required
    constexpr const int BUFFER_LIMIT = 128;
    nvmlSample_t buffer[BUFFER_LIMIT];
    nvmlValueType_t v_type;
    unsigned int sample_count = BUFFER_LIMIT;
    if (NVML_CHECK_U(nvmlDeviceGetSamples(mDevice, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_count, buffer)) ==
        NVML_SUCCESS) {
        float gpu_utilization = 0.f;
        for (unsigned i = 0; i < sample_count; ++i) {
            gpu_utilization += (float)buffer[i].sampleValue.uiVal;
        }
        mInfo.gpu_utilization = gpu_utilization / (float)sample_count;
    }

    // sample count may have been modified by the query above; reset back to buffer size
    sample_count = BUFFER_LIMIT;
    if (NVML_CHECK_U(
            nvmlDeviceGetSamples(mDevice, NVML_MEMORY_UTILIZATION_SAMPLES, 0, &v_type, &sample_count, buffer)) ==
        NVML_SUCCESS) {
        float mem_utilization = 0.f;
        for (unsigned i = 0; i < sample_count; ++i) {
            mem_utilization += (float)buffer[i].sampleValue.uiVal;
        }
        mInfo.mem_utilization = mem_utilization / (float)sample_count;
    }

    nvmlMemory_v2_t mem_info = get_mem_info(mDevice);
    mInfo.mem_free = mem_info.free;
    mInfo.mem_total = mem_info.total;
    mInfo.mem_reserved = mem_info.reserved;
    mInfo.mem_used = mem_info.used;

    // query PCIe info, if available
    if (mThread.joinable()) {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        auto interval = std::chrono::duration_cast<std::chrono::microseconds>(
                            now - std::chrono::steady_clock::duration{mLastTimestamp})
                            .count();

        std::size_t int_rx = mIntervalPCIeRX.exchange(0);
        std::size_t int_tx = mIntervalPCIeTX.exchange(0);
        std::size_t int_eg = mIntervalEnergy.exchange(0);

        mInfo.pcie_rx = (1'000'000ull * int_rx) / interval;
        mInfo.pcie_tx = (1'000'000ull * int_tx) / interval;
        // not using nvmlDeviceGetPowerUsage, because that is the past 1sec average, so it might not be representative
        mInfo.power = (1'000'000ull * int_eg) / interval;

        mTotalPCIeRX += int_rx;
        mTotalPCIeTX += int_tx;
        mTotalEnergy += int_eg;
        mLastTimestamp = now.count();
    }

    return mInfo;
}

nvmlMemory_v2_t get_mem_info(nvmlDevice_t device) {
    static bool has_printed_warning = false;
    nvmlMemory_v2_t mem_info;
    mem_info.version = nvmlMemory_v2;
    auto status = nvmlDeviceGetMemoryInfo_v2(device, &mem_info);
    if (status == NVML_ERROR_NOT_SUPPORTED) {
        fprintf(stderr, "[NVML WARNING] nvmlDeviceGetMemoryInfo not supported.");
        has_printed_warning = true;
        std::size_t free, total;
        // hail mary -- use cuda's basic interface instead
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        mem_info.reserved = 0;
        mem_info.free = free;
        mem_info.total = total;
        mem_info.used = 0;
    } else {
        NVML_CHECK(status);
    }
    return mem_info;
}

/**
 * @brief Get the amount of reserved GPU memory in bytes (best-effort).
 *
 * @return Reserved memory in bytes, or 0 if not supported/available via fallback.
 */
std::size_t get_mem_reserved() {
    return get_mem_info(nvml_get_device()).reserved;
}

/**
 * @brief Get the human-readable name of the current CUDA/NVML GPU.
 *
 * @return Device name string as reported by NVML.
 * @throws std::runtime_error If NVML lookup fails.
 */
std::string get_gpu_name() {
    char name[256];
    NVML_CHECK(nvmlDeviceGetName(nvml_get_device(), name, 256));
    return name;
}

/**
 * @brief Pin the current process/thread CPU affinity to the GPU's preferred CPUs.
 *
 * @return True if the NVML affinity call succeeded; false otherwise.
 */
bool set_cpu_affinity() {
    auto status = nvmlDeviceSetCpuAffinity(nvml_get_device());
    return status == NVML_SUCCESS;
}
