#include "ninfer/ops/scatter.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kBlock = 128;

__global__ void scatter_payload_control_x8(const uint4* source, uint4* destination,
                                           std::int32_t vectors_per_column) {
    const std::int32_t source_column = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t source_base = static_cast<std::int64_t>(source_column) * vectors_per_column;
    const std::int64_t destination_base =
        static_cast<std::int64_t>(source_column + 1) * vectors_per_column;
    for (std::int32_t vector = static_cast<std::int32_t>(threadIdx.x); vector < vectors_per_column;
         vector += static_cast<std::int32_t>(blockDim.x)) {
        destination[destination_base + vector] = source[source_base + vector];
    }
}

void run(std::int32_t d, std::int32_t vision_tokens, bool control, bool profile_once) {
    const std::int32_t prompt_tokens = vision_tokens + 2;
    const std::size_t n              = static_cast<std::size_t>(d) * vision_tokens;
    DeviceBuffer source              = make_bf16(n);
    DeviceBuffer destination         = make_bf16(static_cast<std::size_t>(d) * prompt_tokens);
    std::vector<std::int32_t> indices(static_cast<std::size_t>(vision_tokens));
    std::iota(indices.begin(), indices.end(), 1);
    DeviceBuffer device_indices(indices.size() * sizeof(std::int32_t));
    cudaMemcpy(device_indices.p, indices.data(), device_indices.bytes, cudaMemcpyHostToDevice);
    Tensor source_tensor(source.p, DType::BF16, {d, vision_tokens});
    Tensor destination_tensor(destination.p, DType::BF16, {d, prompt_tokens});
    Tensor indices_tensor(device_indices.p, DType::I32, {vision_tokens});

    const auto launch = [&](cudaStream_t stream) {
        if (control) {
            scatter_payload_control_x8<<<static_cast<unsigned>(vision_tokens), kBlock, 0, stream>>>(
                static_cast<const uint4*>(source.p), static_cast<uint4*>(destination.p), d / 8);
        } else {
            ops::scatter(source_tensor, indices_tensor, destination_tensor, stream);
        }
    };

    if (profile_once) {
        launch(nullptr);
        cudaDeviceSynchronize();
        std::printf("scatter %s profile D=%d V=%d grid=%d block=%d\n",
                    control ? "control" : "production", d, vision_tokens, vision_tokens, kBlock);
        return;
    }

    const double bytes  = static_cast<double>(vision_tokens) * (d * 4.0 + sizeof(std::int32_t));
    const Result result = bench_loop(launch, bytes);
    char tag[96];
    std::snprintf(tag, sizeof(tag), "scatter%s [%d,%d]", control ? " control" : "", d,
                  vision_tokens);
    print_result(tag, result);
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    std::int32_t d    = 2048;
    bool control      = false;
    bool profile_once = false;
    std::vector<std::int32_t> vision_tokens;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--d") && i + 1 < argc) {
            d = static_cast<std::int32_t>(std::strtol(argv[++i], nullptr, 10));
        } else if (!std::strcmp(argv[i], "--vision") && i + 1 < argc) {
            vision_tokens.push_back(static_cast<std::int32_t>(std::strtol(argv[++i], nullptr, 10)));
        } else if (!std::strcmp(argv[i], "--control")) {
            control = true;
        } else if (!std::strcmp(argv[i], "--profile-once")) {
            profile_once = true;
        } else {
            std::fprintf(stderr, "usage: %s [--d D] [--vision V] [--control] [--profile-once]\n",
                         argv[0]);
            return 2;
        }
    }
    if (vision_tokens.empty()) { vision_tokens = {1, 2, 64, 1024, 12288, 16384}; }
    for (const std::int32_t v : vision_tokens) { run(d, v, control, profile_once); }
    return 0;
}
