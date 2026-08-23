#include "ninfer/ops/cast.h"
#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kD = 1536;
constexpr int kBlock      = 256;
constexpr int kGridCap    = 4096;

struct alignas(8) PackedBf16Bytes {
    std::uint32_t lo;
    std::uint32_t hi;
};

__global__ void cast_payload_control_x4_kernel(const uint4* source, PackedBf16Bytes* destination,
                                               std::int64_t vectors) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < vectors; i += stride) {
        const uint4 value = source[i];
        destination[i]    = {value.x ^ value.y, value.z ^ value.w};
    }
}

int cast_grid(std::int64_t vectors) {
    return static_cast<int>(std::max<std::int64_t>(
        1, std::min<std::int64_t>((vectors + kBlock - 1) / kBlock, kGridCap)));
}

void run(std::int32_t patches, bool control, bool profile_once) {
    const std::int64_t count   = static_cast<std::int64_t>(kD) * patches;
    const std::int64_t vectors = count / 4;
    DeviceBuffer source(static_cast<std::size_t>(count) * sizeof(float));
    DeviceBuffer destination(static_cast<std::size_t>(count) * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemset(source.p, 0x3c, source.bytes));
    CUDA_CHECK(cudaMemset(destination.p, 0, destination.bytes));
    Tensor source_tensor(source.p, DType::FP32, {kD, patches});
    Tensor destination_tensor(destination.p, DType::BF16, {kD, patches});

    const auto launch = [&](cudaStream_t stream) {
        if (!control) {
            ops::cast_fp32_to_bf16(source_tensor, destination_tensor, stream);
        } else {
            cast_payload_control_x4_kernel<<<cast_grid(vectors), kBlock, 0, stream>>>(
                static_cast<const uint4*>(source.p), static_cast<PackedBf16Bytes*>(destination.p),
                vectors);
        }
    };

    if (profile_once) {
        launch(nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("cast %s profile P=%d grid=%d block=%d\n", control ? "control" : "fp32-bf16",
                    patches, cast_grid(vectors), kBlock);
        return;
    }

    const double bytes  = static_cast<double>(count) * 6.0;
    const Result result = bench_loop(launch, bytes);
    char tag[96];
    std::snprintf(tag, sizeof(tag), "cast fp32-bf16%s [1536,%d]", control ? " control" : "",
                  patches);
    print_result(tag, result);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool control      = false;
    bool profile_once = false;
    std::vector<std::int32_t> patches;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--control")) {
            control = true;
        } else if (!std::strcmp(argv[i], "--profile-once")) {
            profile_once = true;
        } else if (!std::strcmp(argv[i], "--patch") && i + 1 < argc) {
            patches.push_back(static_cast<std::int32_t>(std::strtol(argv[++i], nullptr, 10)));
        }
    }
    if (patches.empty()) { patches = {8, 256, 4096, 49152, 65536}; }
    for (const std::int32_t p : patches) { run(p, control, profile_once); }
    return 0;
}
