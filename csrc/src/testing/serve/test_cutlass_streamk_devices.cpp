#include "core/arena.h"
#include "core/device.h"
#include "core/engine_context.h"
#include "core/sleep.h"
#include "ops/linear/w8a8/w4fp4_cutlass_gemm.h"

#include <bit>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace sinfer;
namespace detail = sinfer::ops::detail;
namespace {
void require(bool value, const char* message) { if (!value) { throw std::runtime_error(message); } }

void run_device(int device, const void* owner, bool cold, bool test_sleep) {
    CUDA_CHECK(cudaSetDevice(device));
    constexpr int tokens = 33, n = 256, k = 4096;
    DeviceBuffer a(tokens * k / 2), b(n * k / 2),
        as(detail::w4fp4_sf_atom_bytes(tokens, k)), bs(detail::w4fp4_sf_atom_bytes(n, k)),
        out(tokens * n * 2);
    a.fill(0x22); b.fill(0x22); // two E2M1 values of 1 per byte
    as.fill(0x38); bs.fill(0x38); // UE4M3 scale 1
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto launch = [&](int tile, bool residual) {
        return detail::nvfp4_cutlass_gemm(static_cast<const std::uint8_t*>(a.p),
            static_cast<const std::uint8_t*>(as.p), static_cast<const std::uint8_t*>(b.p),
            static_cast<const std::uint8_t*>(bs.p), 0.5F, residual ? out.p : nullptr, out.p,
            tokens, n, k, tile, stream);
    };
    if (cold) {
        cudaGraph_t empty;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        const bool accepted = launch(2, false);
        CUDA_CHECK(cudaStreamEndCapture(stream, &empty));
        CUDA_CHECK(cudaGraphDestroy(empty));
        require(!accepted, "cold capture inherited Stream-K workspace from another device or engine");
    }
    std::vector<std::uint16_t> initial(tokens * n, 0x4200); // BF16 32
    std::vector<std::uint16_t> result(tokens * n);
    const auto check = [&](bool residual) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        out.copy_to_host(result.data(), out.bytes);
        for (const auto bits : result) {
            const float value = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
            require(value == k * 0.5F + (residual ? 32.F : 0.F), "Stream-K GEMM used invalid workspace or produced incorrect output");
        }
    };
    for (int tile : {2, 3}) {
        for (bool residual : {false, true}) {
            out.copy_from_host(initial.data(), out.bytes);
            require(launch(tile, residual), "eager Stream-K GEMM declined");
            check(residual);
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            const bool captured = launch(tile, residual);
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            require(captured, "warmed Stream-K GEMM declined capture");
            CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            if (test_sleep) {
                const auto bytes = sleep_owned_bytes(owner, device);
                require(bytes >= (64U << 20), "Stream-K workspace is not owned by its engine/device");
                require(sleep_device(device, owner) == bytes && wake_device(device, owner) == bytes,
                        "Stream-K workspace did not sleep and wake");
            }
            for (int repeat = 0; repeat < 3; ++repeat) {
                out.copy_from_host(initial.data(), out.bytes);
                CUDA_CHECK(cudaGraphLaunch(executable, stream));
                check(residual);
            }
            CUDA_CHECK(cudaGraphExecDestroy(executable));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}
} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return 77; }
    try {
        DeviceContext device(0);
        for (int gpu : {0, 1}) {
            int major = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, gpu));
            if (major != 12) { return 77; }
        }
        auto context = std::make_unique<ops::EngineOpsContext>();
        const void* owner = context.get();
        ops::bind_ops_context(context.get());
        set_sleepable_allocations(true);
        // One host thread switches between physical GPUs, as the pipeline executor does.
        for (int gpu : {0, 1}) { run_device(gpu, owner, true, false); }
        std::exception_ptr worker_error;
        std::thread worker([&] {
            ops::bind_ops_context(context.get());
            try { run_device(0, owner, true, false); }
            catch (...) { worker_error = std::current_exception(); }
            ops::bind_ops_context(nullptr);
        });
        worker.join();
        if (worker_error) { std::rethrow_exception(worker_error); }
        require(sleep_owned_bytes(owner, 0) >= (128U << 20),
                "a worker reused another caller's workspace or freed it on thread exit");
        for (int gpu : {0, 1}) { run_device(gpu, owner, false, true); }
        {
            ops::EngineOpsContext independent;
            ops::bind_ops_context(&independent);
            set_sleepable_allocations(true);
            run_device(0, &independent, true, false);
            ops::bind_ops_context(nullptr);
        }
        context.reset();
        for (int gpu : {0, 1}) {
            require(sleep_owned_bytes(owner, gpu) == 0, "Stream-K workspace survived its engine");
        }
        std::cout << "Stream-K device ownership, eager/captured output and sleep: OK\n";
    } catch (const std::exception& error) {
        ops::bind_ops_context(nullptr);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
