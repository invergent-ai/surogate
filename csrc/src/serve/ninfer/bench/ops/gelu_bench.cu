#include "ninfer/ops/gelu.h"
#include "ninfer_bench_common.h"
#include "ops/common/bf16_vector.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

__global__ void gelu_payload_control(uint4* x, std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        uint4 v = x[i];
        v.x ^= 0x00010001u;
        v.y ^= 0x00010001u;
        v.z ^= 0x00010001u;
        v.w ^= 0x00010001u;
        x[i] = v;
    }
}

__global__ void gelu_pair_payload_control(__nv_bfloat162* x, std::int64_t pairs) {
    constexpr int pairsPerThread = 4;
    const std::int64_t first =
        (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) * pairsPerThread;
#pragma unroll
    for (int item = 0; item < pairsPerThread; ++item) {
        const std::int64_t pair = first + item;
        if (pair < pairs) {
            auto* bits = reinterpret_cast<unsigned int*>(x + pair);
            *bits ^= 0x00010001u;
        }
    }
}

void run(ops::GeluMode mode, int d, int columns, bool control) {
    const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(columns);
    DeviceBuffer x      = make_bf16(n);
    Tensor tx(x.p, DType::BF16, {d, columns});

    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            if (control) {
                constexpr int block   = 256;
                constexpr int maxGrid = 16384;
                if (static_cast<std::int64_t>(n) <= ops::kBf16x8CacheSizedMaxElements) {
                    const auto packs = static_cast<std::int64_t>(n / 8);
                    const int grid   = static_cast<int>(std::min<std::int64_t>(
                        maxGrid, std::max<std::int64_t>(1, (packs + block - 1) / block)));
                    gelu_payload_control<<<grid, block, 0, stream>>>(static_cast<uint4*>(x.p),
                                                                     packs);
                } else {
                    constexpr int pairsPerThread = 4;
                    const auto pairs             = static_cast<std::int64_t>(n / 2);
                    const int grid = static_cast<int>((pairs + block * pairsPerThread - 1) /
                                                      (block * pairsPerThread));
                    gelu_pair_payload_control<<<grid, block, 0, stream>>>(
                        static_cast<__nv_bfloat162*>(x.p), pairs);
                }
            } else {
                ops::gelu(tx, mode, stream);
            }
        },
        static_cast<double>(n) * 4.0);

    char tag[96];
    const char* name = mode == ops::GeluMode::Tanh ? "gelu_tanh" : "gelu_exact";
    std::snprintf(tag, sizeof(tag), "%s%s [%d,%-5d]", control ? "control_" : "", name, d, columns);
    print_result(tag, result);
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    int selected_columns        = 0;
    ops::GeluMode selected_mode = ops::GeluMode::Tanh;
    bool mode_selected          = false;
    bool control                = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--columns") && i + 1 < argc) {
            selected_columns = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char* mode = argv[++i];
            if (!std::strcmp(mode, "tanh")) {
                selected_mode = ops::GeluMode::Tanh;
            } else if (!std::strcmp(mode, "exact")) {
                selected_mode = ops::GeluMode::Exact;
            } else {
                std::fprintf(stderr, "mode must be tanh or exact\n");
                return 2;
            }
            mode_selected = true;
        } else if (!std::strcmp(argv[i], "--control")) {
            control = true;
        } else {
            std::fprintf(stderr, "usage: %s [--mode tanh|exact --columns C] [--control]\n",
                         argv[0]);
            return 2;
        }
    }
    if (selected_columns < 0 || (selected_columns > 0 && !mode_selected)) {
        std::fprintf(stderr, "columns requires an explicit mode and must be positive\n");
        return 2;
    }

    if (selected_columns > 0) {
        const int d = selected_mode == ops::GeluMode::Tanh ? 4304 : 4608;
        run(selected_mode, d, selected_columns, control);
        return 0;
    }
    for (const int patches : {8, 256, 4096, 49152, 65536}) {
        run(ops::GeluMode::Tanh, 4304, patches, control);
    }
    for (const int merged : {2, 64, 1024, 12288, 16384}) {
        run(ops::GeluMode::Exact, 4608, merged, control);
    }
    return 0;
}
