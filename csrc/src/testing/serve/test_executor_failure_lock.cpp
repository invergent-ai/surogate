#include "core/device.h"
#include "targets/registry.h"
#include "runtime/engine/concurrent_executor.h"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>

namespace sinfer::runtime {
struct ConcurrentExecutorTestAccess {
    template <class Instance>
    static void fail(ConcurrentExecutor<Instance>& executor) {
        executor.fail_all(std::make_exception_ptr(std::runtime_error("injected worker failure")));
    }
};
}

int main() {
    const auto* artifact = std::getenv("SUROGATE_EXECUTOR_LOCK_ARTIFACT");
    if (!artifact) { return 77; }
    try {
        sinfer::DeviceContext device(0);
        sinfer::EngineOptions options;
        options.artifact_path = artifact;
        options.max_context = 256;
        options.kv_capacity = sinfer::KvCapacityPolicy::explicit_capacity(512);
        options.max_concurrency = 2;
        options.prefill_chunk = 128;
        options.use_cuda_graph = false;
        auto target = sinfer::targets::construct_target(options, device);
        auto& instance = *std::get<std::unique_ptr<sinfer::targets::Qwen3_5Instance>>(target.active);
        sinfer::runtime::ConcurrentExecutor executor(instance, options);
        auto paused = executor.pause_execution();
        std::promise<void> started;
        auto failure = std::async(std::launch::async, [&] {
            CUDA_CHECK(cudaSetDevice(device.device));
            started.set_value();
            sinfer::runtime::ConcurrentExecutorTestAccess::fail(executor);
        });
        started.get_future().wait();
        const bool blocked = failure.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
        paused.unlock();
        failure.get();
        if (!blocked) {
            std::cerr << "worker failure cleanup bypassed the execution lock\n";
            return 1;
        }
        std::cout << "Worker failure cleanup waits for exclusive program access\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
