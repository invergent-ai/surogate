#include "ninfer/ops/add_bias.h"
#include "ninfer_bench_common.h"
#include "ops/common/bf16_vector.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

template <int RowsPerBlock>
__global__ void add_bias_payload_control(const uint4* bias, uint4* x, std::int32_t packs,
                                         std::int32_t rows) {
    const int pack = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pack >= packs) { return; }
    const uint4 b       = bias[pack];
    const int first_row = static_cast<int>(blockIdx.y) * RowsPerBlock;
#pragma unroll
    for (int item = 0; item < RowsPerBlock; ++item) {
        const int row = first_row + item;
        if (row >= rows) { return; }
        uint4 value = x[static_cast<std::int64_t>(row) * packs + pack];
        value.x ^= b.x;
        value.y ^= b.y;
        value.z ^= b.z;
        value.w ^= b.w;
        x[static_cast<std::int64_t>(row) * packs + pack] = value;
    }
}

template <int Block>
__global__ void add_bias_pair_payload_control(const unsigned int* bias, unsigned int* x,
                                              std::int32_t pairs, std::int32_t rows) {
    constexpr int pairsPerThread = 4;
    const int first =
        (static_cast<int>(blockIdx.x) * Block + static_cast<int>(threadIdx.x)) * pairsPerThread;
    for (int row = static_cast<int>(blockIdx.y); row < rows; row += static_cast<int>(gridDim.y)) {
        const std::int64_t base = static_cast<std::int64_t>(row) * pairs;
#pragma unroll
        for (int item = 0; item < pairsPerThread; ++item) {
            const int pair = first + item;
            if (pair < pairs) { x[base + pair] ^= bias[pair]; }
        }
    }
}

void run(int d, int columns, bool control) {
    const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(columns);
    DeviceBuffer x      = make_bf16(n);
    DeviceBuffer bias   = make_bf16(d);
    Tensor tx(x.p, DType::BF16, {d, columns});
    Tensor tb(bias.p, DType::BF16, {d});

    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            if (control) {
                constexpr int block   = 256;
                const int packs       = d / 8;
                const unsigned grid_x = static_cast<unsigned>((packs + block - 1) / block);
                if (static_cast<std::int64_t>(n) <= ops::kBf16x8CacheSizedMaxElements &&
                    columns >= 1024) {
                    constexpr int rowsPerBlock = 4;
                    const unsigned grid_y =
                        static_cast<unsigned>((columns + rowsPerBlock - 1) / rowsPerBlock);
                    add_bias_payload_control<rowsPerBlock>
                        <<<dim3(grid_x, grid_y), block, 0, stream>>>(
                            static_cast<const uint4*>(bias.p), static_cast<uint4*>(x.p), packs,
                            columns);
                } else if (static_cast<std::int64_t>(n) <= ops::kBf16x8CacheSizedMaxElements) {
                    add_bias_payload_control<1><<<dim3(grid_x, columns), block, 0, stream>>>(
                        static_cast<const uint4*>(bias.p), static_cast<uint4*>(x.p), packs,
                        columns);
                } else {
                    constexpr int pairsPerThread = 4;
                    const int pairs              = d / 2;
                    const unsigned pair_grid_x   = static_cast<unsigned>(
                        (pairs + block * pairsPerThread - 1) / (block * pairsPerThread));
                    const unsigned pair_grid_y = static_cast<unsigned>(std::min(columns, 65535));
                    add_bias_pair_payload_control<block>
                        <<<dim3(pair_grid_x, pair_grid_y), block, 0, stream>>>(
                            static_cast<const unsigned int*>(bias.p),
                            static_cast<unsigned int*>(x.p), pairs, columns);
                }
            } else {
                ops::add_bias(tb, tx, stream);
            }
        },
        static_cast<double>(n) * 4.0);

    char tag[80];
    std::snprintf(tag, sizeof(tag), "%s [%d,%-5d]", control ? "control" : "add_bias", d, columns);
    print_result(tag, result);
}

void run_matrix(bool control) {
    constexpr std::array<int, 5> patches{8, 256, 4096, 49152, 65536};
    constexpr std::array<int, 5> merged{2, 64, 1024, 12288, 16384};
    for (int d : {1152, 3456, 4304}) {
        for (int p : patches) { run(d, p, control); }
    }
    for (int d : {2048, 4608}) {
        for (int v : merged) { run(d, v, control); }
    }
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    int selected_d       = 0;
    int selected_columns = 0;
    bool control         = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--d") && i + 1 < argc) {
            selected_d = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--columns") && i + 1 < argc) {
            selected_columns = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--control")) {
            control = true;
        } else {
            std::fprintf(stderr, "usage: %s [--d D --columns C] [--control]\n", argv[0]);
            return 2;
        }
    }
    if ((selected_d == 0) != (selected_columns == 0) || selected_d < 0 || selected_columns < 0 ||
        (selected_d > 0 && selected_d % 8 != 0)) {
        std::fprintf(stderr, "d and columns must be supplied together; d must be divisible by 8\n");
        return 2;
    }

    if (selected_d > 0) {
        run(selected_d, selected_columns, control);
        return 0;
    }
    run_matrix(control);
    return 0;
}
