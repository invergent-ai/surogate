#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace sinfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::sinfer::cuda_check((expr), #expr, __FILE__, __LINE__)

/// Whether this build holds block-scaled FP4 tensor-core code that a device of compute
/// capability `cc` (major * 10 + minor) can run. The FP4 families are compiled only for the
/// sm_12x entries of SUROGATE_SERVE_CUDA_ARCHS: `120a` loads on exactly sm_120 (RTX 50, RTX PRO
/// 6000), `121a` on exactly sm_121 (GB10: DGX Spark), and the family target `120f` on both. Any
/// other pairing falls back to the fatbin's compute_89 PTX, whose FP4 bodies are __trap(), so
/// every NVFP4 route asks this rather than comparing the capability with 120.
bool fp4_tensor_cores(int cc) noexcept;

/// Streaming multiprocessors on the current device: 48 on GB10, 132 on an H100, 170 on an RTX
/// 5090. Read once per device; `fallback` when the device cannot be queried. For the launch
/// policies that size a grid in waves, which must not assume one card's SM count.
int current_device_sm_count(int fallback) noexcept;

struct DeviceContext {
    int device               = 0;
    cudaStream_t stream      = nullptr;
    cudaStream_t load_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    int sm() const noexcept;
    std::size_t total_vram() const noexcept;
    void synchronize() const;
};

/// Binds the calling thread to a CUDA device for a scope, restoring whatever it
/// was bound to before.
///
/// The current device is a property of the thread, and the runtime resolves every
/// pointer against it. A thread that never bound the engine's device -- an HTTP
/// handler thread, say -- fails every copy and every set against that engine's
/// memory, and fails with an error about an invalid argument rather than one
/// about the device, which is a long way from the cause. Any thread that touches
/// an engine's memory outside its executor needs one of these.
class ScopedDevice {
public:
    // restore_always also restores scopes whose body selects other devices.
    explicit ScopedDevice(int device, bool restore_always = false);
    ~ScopedDevice();

    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;
    ScopedDevice(ScopedDevice&&)                 = delete;
    ScopedDevice& operator=(ScopedDevice&&)      = delete;

private:
    int previous_ = 0;
    bool restore_ = false;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

} // namespace sinfer
