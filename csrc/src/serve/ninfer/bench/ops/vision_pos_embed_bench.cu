#include "ninfer/ops/vision_pos_embed.h"
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

constexpr int kD     = 1152;
constexpr int kRows  = 2304;
constexpr int kBlock = 128;

int small_p_tiles(int patches) {
    constexpr int pairs = kD / 2;
    return std::min((pairs + 31) / 32, std::max(1, (4096 + patches - 1) / patches));
}

__global__ void vision_pos_embed_payload_control_warp(const std::uint32_t* table,
                                                      const std::int32_t* indices,
                                                      const float* weights, std::uint32_t* x,
                                                      std::int32_t patches,
                                                      std::int32_t tiles_per_patch) {
    constexpr int pairs = kD / 2;
    const int lane      = static_cast<int>(threadIdx.x);
    const int patch     = static_cast<int>(blockIdx.x) / tiles_per_patch;
    const int tile      = static_cast<int>(blockIdx.x) - patch * tiles_per_patch;
    if (patch >= patches) { return; }

    const std::int64_t control = static_cast<std::int64_t>(patch) * 4 + lane;
    const int lane_index       = lane < 4 ? indices[control] : 0;
    const float lane_weight    = lane < 4 ? weights[control] : 0.0f;
    int corner_indices[4];
    std::uint32_t corner_weights[4];
#pragma unroll
    for (int corner = 0; corner < 4; ++corner) {
        corner_indices[corner] = __shfl_sync(0xffffffffu, lane_index, corner);
        corner_weights[corner] = __float_as_uint(__shfl_sync(0xffffffffu, lane_weight, corner));
    }

    const std::int64_t x_base = static_cast<std::int64_t>(patch) * pairs;
    for (int pair = tile * 32 + lane; pair < pairs; pair += tiles_per_patch * 32) {
        std::uint32_t value = x[x_base + pair];
#pragma unroll
        for (int corner = 0; corner < 4; ++corner) {
            value ^= table[static_cast<std::int64_t>(corner_indices[corner]) * pairs + pair];
            value ^= corner_weights[corner];
        }
        x[x_base + pair] = value;
    }
}

template <int Block>
__global__ void
vision_pos_embed_payload_control_cta(const std::uint32_t* table, const std::int32_t* indices,
                                     const float* weights, std::uint32_t* x, std::int32_t patches) {
    constexpr int pairs = kD / 2;
    const int patch     = static_cast<int>(blockIdx.x);
    if (patch >= patches) { return; }
    __shared__ std::int32_t corner_indices[4];
    __shared__ float corner_weights[4];
    if (threadIdx.x < 4) {
        const std::int64_t control  = static_cast<std::int64_t>(patch) * 4 + threadIdx.x;
        corner_indices[threadIdx.x] = indices[control];
        corner_weights[threadIdx.x] = weights[control];
    }
    __syncthreads();

    const std::int64_t x_base = static_cast<std::int64_t>(patch) * pairs;
    for (int pair = static_cast<int>(threadIdx.x); pair < pairs; pair += Block) {
        std::uint32_t value = x[x_base + pair];
#pragma unroll
        for (int corner = 0; corner < 4; ++corner) {
            value ^= table[static_cast<std::int64_t>(corner_indices[corner]) * pairs + pair];
            value ^= __float_as_uint(corner_weights[corner]);
        }
        x[x_base + pair] = value;
    }
}

void run(std::int32_t patches, bool control, bool profile_once) {
    const std::size_t n = static_cast<std::size_t>(kD) * patches;
    DeviceBuffer table  = make_bf16(static_cast<std::size_t>(kD) * kRows);
    DeviceBuffer x      = make_bf16(n);
    std::vector<int> indices(static_cast<std::size_t>(patches) * 4);
    std::vector<float> weights(static_cast<std::size_t>(patches) * 4, 0.25f);
    for (int patch = 0; patch < patches; ++patch) {
        for (int corner = 0; corner < 4; ++corner) {
            indices[static_cast<std::size_t>(patch) * 4 + corner] =
                (patch * 17 + corner * 49) % kRows;
        }
    }
    DeviceBuffer di(indices.size() * sizeof(std::int32_t));
    DeviceBuffer dw(weights.size() * sizeof(float));
    cudaMemcpy(di.p, indices.data(), di.bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dw.p, weights.data(), dw.bytes, cudaMemcpyHostToDevice);
    Tensor ttable(table.p, DType::BF16, {kD, kRows});
    Tensor tx(x.p, DType::BF16, {kD, patches});
    Tensor ti(di.p, DType::I32, {4, patches});
    Tensor tw(dw.p, DType::FP32, {4, patches});

    const auto launch = [&](cudaStream_t stream) {
        if (control) {
            if (patches < 1024) {
                const int tiles = small_p_tiles(patches);
                vision_pos_embed_payload_control_warp<<<static_cast<unsigned>(patches * tiles), 32,
                                                        0, stream>>>(
                    static_cast<const std::uint32_t*>(table.p),
                    static_cast<const std::int32_t*>(di.p), static_cast<const float*>(dw.p),
                    static_cast<std::uint32_t*>(x.p), patches, tiles);
            } else {
                vision_pos_embed_payload_control_cta<kBlock>
                    <<<static_cast<unsigned>(patches), kBlock, 0, stream>>>(
                        static_cast<const std::uint32_t*>(table.p),
                        static_cast<const std::int32_t*>(di.p), static_cast<const float*>(dw.p),
                        static_cast<std::uint32_t*>(x.p), patches);
            }
        } else {
            ops::vision_pos_embed_add(ttable, ti, tw, tx, stream);
        }
    };

    if (profile_once) {
        launch(nullptr);
        cudaDeviceSynchronize();
        const int tiles = patches < 1024 ? small_p_tiles(patches) : 1;
        std::printf("vision_pos_embed %s profile P=%d grid=%d block=%d\n",
                    control ? "control" : "production", patches, patches * tiles,
                    patches < 1024 ? 32 : kBlock);
        return;
    }

    // The table is 5.1 MiB and remains resident after warmup. Count the minimum external traffic:
    // in-place x read/write plus the per-patch indices and weights. Production/control comparison
    // accounts for the required four gathered table rows served by L2.
    const double external_bytes = static_cast<double>(n) * 4.0 + patches * 32.0;
    const Result result         = bench_loop(launch, external_bytes);
    char tag[96];
    std::snprintf(tag, sizeof(tag), "vision_pos_embed%s [1152,%d]", control ? " control" : "",
                  patches);
    print_result(tag, result);
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
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
        } else {
            std::fprintf(stderr, "usage: %s [--patch P] [--control] [--profile-once]\n", argv[0]);
            return 2;
        }
    }
    if (patches.empty()) { patches = {8, 256, 4096, 49152, 65536}; }
    for (const std::int32_t p : patches) { run(p, control, profile_once); }
    return 0;
}
