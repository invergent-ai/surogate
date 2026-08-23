#include "ninfer/ops/residual_add.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

__global__ void residual_add_payload_control(const uint4* y, uint4* x, std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        const uint4 a = y[i];
        uint4 b       = x[i];
        b.x ^= a.x;
        b.y ^= a.y;
        b.z ^= a.z;
        b.w ^= a.w;
        x[i] = b;
    }
}

std::vector<int> parse_tokens(const char* raw) {
    std::vector<int> result;
    const std::string text(raw);
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end  = text.find(',', begin);
        const std::string item = text.substr(begin, end == std::string::npos ? end : end - begin);
        const int value        = std::stoi(item);
        if (value <= 0) { throw std::invalid_argument("tokens must be positive"); }
        result.push_back(value);
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument("tokens must not be empty"); }
    return result;
}

void run(int d, int tokens, bool control) {
    const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(tokens);
    DeviceBuffer y      = make_bf16(n);
    DeviceBuffer x      = make_bf16(n);
    Tensor ty(y.p, DType::BF16, {d, tokens});
    Tensor tx(x.p, DType::BF16, {d, tokens});

    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            if (control) {
                constexpr int block   = 256;
                constexpr int maxGrid = 4096;
                const auto packs      = static_cast<std::int64_t>(n / 8);
                const int grid        = static_cast<int>(std::min<std::int64_t>(
                    maxGrid, std::max<std::int64_t>(1, (packs + block - 1) / block)));
                residual_add_payload_control<<<grid, block, 0, stream>>>(
                    static_cast<const uint4*>(y.p), static_cast<uint4*>(x.p), packs);
            } else {
                ops::residual_add(ty, tx, stream);
            }
        },
        static_cast<double>(n) * 6.0);

    char tag[96];
    std::snprintf(tag, sizeof(tag), "%s [%d,%-5d]", control ? "control" : "residual_add", d,
                  tokens);
    print_result(tag, result);
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    int d = 1152;
    std::vector<int> tokens;
    bool control = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--patches") && i + 1 < argc) {
            if (!tokens.empty()) {
                std::fprintf(stderr, "--patches and --tokens are mutually exclusive\n");
                return 2;
            }
            d      = 1152;
            tokens = {std::atoi(argv[++i])};
        } else if (!std::strcmp(argv[i], "--d") && i + 1 < argc) {
            d = std::atoi(argv[++i]);
            if (d != 1152 && d != 2048 && d != 5120) {
                std::fprintf(stderr, "D must be 1152, 2048, or 5120\n");
                return 2;
            }
        } else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            if (!tokens.empty()) {
                std::fprintf(stderr, "token extents were specified more than once\n");
                return 2;
            }
            tokens = parse_tokens(argv[++i]);
        } else if (!std::strcmp(argv[i], "--control")) {
            control = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--d 1152|2048|5120] [--tokens T[,T...] | --patches P] "
                         "[--control]\n",
                         argv[0]);
            return 2;
        }
    }
    for (const int tokens_value : tokens) {
        if (tokens_value <= 0) {
            std::fprintf(stderr, "tokens must be positive\n");
            return 2;
        }
    }
    if (tokens.empty()) { tokens = {8, 256, 4096, 49152, 65536}; }
    for (const int tokens_value : tokens) { run(d, tokens_value, control); }
    return 0;
}
