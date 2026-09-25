#include "family/impl/runtime/target_support.h"

#include "core/device.h"

#include <cstdio>
#include <atomic>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace sinfer::family {
namespace {

const char* probe_directory() {
    static const char* dir = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_DUMP_RESIDUAL");
        return (raw != nullptr && *raw != '\0') ? raw : nullptr;
    }();
    return dir;
}

std::int32_t probe_columns() {
    static const std::int32_t columns = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_DUMP_COLUMNS");
        return (raw != nullptr && *raw != '\0') ? std::atoi(raw) : 0;
    }();
    return columns;
}

std::map<std::string, int>& probe_counts() {
    static std::map<std::string, int> counts;
    return counts;
}

} // namespace

bool take_injected_fault(InjectedFault fault) noexcept {
    const auto armed = [](const char* name) {
        const char* raw = std::getenv(name);
        return raw != nullptr && *raw != '\0' ? std::atoll(raw) : 0LL;
    };
    static std::atomic<long long> poison_gpu_prefix{armed("SUROGATE_SERVE_FAULT_POISON_GPU_PREFIX")};
    static std::atomic<long long> nan_readout{armed("SUROGATE_SERVE_FAULT_NAN_READOUT")};
    std::atomic<long long>& left = fault == InjectedFault::PoisonGpuPrefix ? poison_gpu_prefix : nan_readout;
    long long current = left.load(std::memory_order_relaxed);
    while (current > 0) {
        if (left.compare_exchange_weak(current, current - 1, std::memory_order_relaxed)) { return true; }
    }
    return false;
}

void debug_probe_dump(std::int32_t magic, const char* tag, const Tensor& tensor,
                      std::int32_t layer_count, cudaStream_t stream) {
    const char* dir = probe_directory();
    if (dir == nullptr || tensor.data == nullptr) { return; }
    if (tensor.ne[1] > 64) { return; }
    if (probe_columns() > 0 && tensor.ne[1] != probe_columns()) { return; }
    const std::string key = std::string(tag) + "@" + std::to_string(tensor.ne[1]);
    const int occurrence  = probe_counts()[key]++;
    if (occurrence >= layer_count) { return; }

    const std::size_t bytes = tensor.bytes();
    std::vector<std::byte> host(bytes);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), tensor.data, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const std::string path =
        std::string(dir) + "/" + tag + "_" + std::to_string(occurrence) + ".bin";
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) { return; }
    const std::int32_t header[4] = {magic, tensor.ne[0], tensor.ne[1], occurrence};
    std::fwrite(header, sizeof(header), 1, file);
    std::fwrite(host.data(), 1, bytes, file);
    std::fclose(file);
}

} // namespace sinfer::family
