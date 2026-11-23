// lanes/OOM F3 (rl/engine-inkling-2026-11-21/lanes/OOM/DIAGNOSIS.md, 2026-11-23): log-probability
// scoring on a full device fails with a DeviceOutOfMemory, which the executor survives (the round's
// requests get a retryable 429), instead of aborting the process. A W5 engine with 4 MiB left died
// at sampled_logprob.cu:254 (CUDA_CHECK(cudaMallocAsync) -> std::abort). Exits 77 without a device.
//
// The test fills the device with plain allocations, calls score_logprobs_batch, and requires the
// typed exception; then frees the memory and requires that scoring works again (no sticky error).

#include "api/ops/sampled_logprob.h"
#include "core/device_memory_error.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

bool check(bool condition, const char* what) {
    if (!condition) { std::cerr << "FAIL: " << what << '\n'; }
    return condition;
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (cudaSetDevice(0) != cudaSuccess) { return 77; }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) { return 1; }

    constexpr int kVocab = 1024, kColumns = 4;
    std::vector<float> host(static_cast<std::size_t>(kVocab) * kColumns);
    for (std::size_t i = 0; i < host.size(); ++i) { host[i] = static_cast<float>(static_cast<int>(i % 97) - 48) / 8.0f; }
    float* device_logits = nullptr;
    if (cudaMalloc(&device_logits, host.size() * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(device_logits, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return 1;
    }
    const sinfer::Tensor logits(device_logits, sinfer::DType::FP32, {kVocab, kColumns});
    const std::vector<sinfer::TokenId> tokens{1, 2, 3, 4};

    bool ok = true;
    const auto before = sinfer::ops::score_logprobs_batch(logits, tokens, kVocab, 0, stream);
    ok &= check(before.size() == tokens.size() && std::isfinite(before[0].selected.logprob), "scoring works with memory free");

    // Nothing kept in the stream-ordered pool, then nothing left on the device.
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess) { (void)cudaMemPoolTrimTo(pool, 0); }
    (void)cudaDeviceSynchronize();
    std::vector<void*> hog;
    for (std::size_t chunk = std::size_t{1} << 30; chunk >= 4096; chunk >>= 1) {
        for (;;) {
            void* pointer = nullptr;
            if (cudaMalloc(&pointer, chunk) != cudaSuccess) {
                (void)cudaGetLastError();
                break;
            }
            hog.push_back(pointer);
        }
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    (void)cudaMemGetInfo(&free_bytes, &total_bytes);
    std::cout << "device filled: " << hog.size() << " allocations, " << (free_bytes >> 10) << " KiB free\n";

    bool typed = false;
    try {
        (void)sinfer::ops::score_logprobs_batch(logits, tokens, kVocab, 0, stream);
    } catch (const sinfer::DeviceOutOfMemory& error) {
        typed = true;
        std::cout << "typed: " << error.what() << '\n';
    }
    ok &= check(typed, "scoring on a full device must throw DeviceOutOfMemory (and must not abort)");

    for (void* pointer : hog) { (void)cudaFree(pointer); }
    hog.clear();
    ok &= check(cudaGetLastError() == cudaSuccess, "no sticky CUDA error after the refusal");
    const auto after = sinfer::ops::score_logprobs_batch(logits, tokens, kVocab, 0, stream);
    ok &= check(after.size() == tokens.size() && after[0].selected.logprob == before[0].selected.logprob,
                "scoring works again, with the same result, once memory is back");

    (void)cudaFree(device_logits);
    (void)cudaStreamDestroy(stream);
    std::cout << (ok ? "test_sampled_logprob_oom: PASS\n" : "test_sampled_logprob_oom: FAIL\n");
    return ok ? 0 : 1;
}
