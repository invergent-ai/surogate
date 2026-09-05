#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace sinfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::sinfer::cuda_check((expr), #expr, __FILE__, __LINE__)

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
    explicit ScopedDevice(int device);
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
