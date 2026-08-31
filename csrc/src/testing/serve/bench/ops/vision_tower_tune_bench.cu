// Which launcher is fastest for the vision tower's GEMMs, at each patch count.
//
// The other benches in this directory deliberately measure *through* the dispatch,
// leaving implementation selection behind the op contract. This one is the
// maintainer's counterpart: it calls each candidate launcher directly, so the
// dispatch tables can be tuned from measurement rather than guessed. The 64-wide
// tower's entries were added by mirroring the 72-wide tower's schedule, which is
// correctness-complete and says nothing about speed; this is what replaces that
// mirroring with numbers.
//
// Shapes, per tower width (hidden H, intermediate I, 16 heads):
//   patch embedding  q6  n=H          k=1536
//   attention qkv    q4  n=3H         k=H
//   attention output q5  n=H          k=H
//   mlp fc1          q4  n=I          k=H
//   mlp fc2          q5  n=H          k=I
//   merger fc1       w8  n=4H         k=4H
//
//   ninfer_vision_tower_tune_bench [--hidden 1024|1152] [--t-sweep 196,256,...]

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q6/q6_launch.h"
#include "ops/linear/w8/w8_launch.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::size_t kFlushBytes = 256ULL << 20;

struct Candidate {
    const char* name;
    void (*launch)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
};

struct Shape {
    const char* label;
    QType qtype;
    std::int32_t n;
    std::int32_t k;
    const Candidate* candidates;
    std::size_t candidate_count;
};

const Candidate kQ4[] = {
    {"q4_simt_r8_c4", ops::detail::launch_q4_simt_r8_c4},
    {"q4_simt_r8_c8", ops::detail::launch_q4_simt_r8_c8},
    {"q4_mma_r64_c64", ops::detail::launch_q4_mma_r64_c64},
    {"q4_mma_r64_c96", ops::detail::launch_q4_mma_r64_c96},
    {"q4_mma_r64_c128", ops::detail::launch_q4_mma_r64_c128},
};
const Candidate kQ5[] = {
    {"q5_simt_r8_c4", ops::detail::launch_q5_simt_r8_c4},
    {"q5_simt_r8_c8", ops::detail::launch_q5_simt_r8_c8},
    {"q5_mma_r64_c64", ops::detail::launch_q5_mma_r64_c64},
    {"q5_mma_r64_c128", ops::detail::launch_q5_mma_r64_c128},
};
const Candidate kQ6[] = {
    {"q6_simt_r8_c4", ops::detail::launch_q6_simt_r8_c4},
    {"q6_simt_r8_c8", ops::detail::launch_q6_simt_r8_c8},
    {"q6_mma_r64_c64", ops::detail::launch_q6_mma_r64_c64},
    {"q6_mma_r64_c96", ops::detail::launch_q6_mma_r64_c96},
    {"q6_mma_r64_c128", ops::detail::launch_q6_mma_r64_c128},
};
const Candidate kW8[] = {
    {"w8_simt_r8_c4", ops::detail::launch_w8_simt_r8_c4},
    {"w8_simt_r8_c8", ops::detail::launch_w8_simt_r8_c8},
    {"w8_mma_r32_c128", ops::detail::launch_w8_mma_r32_c128},
    {"w8_mma_r64_c96", ops::detail::launch_w8_mma_r64_c96},
    {"w8_mma_r64_c128", ops::detail::launch_w8_mma_r64_c128},
};

std::vector<std::int32_t> parse_list(std::string_view raw) {
    std::vector<std::int32_t> out;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string piece(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        out.push_back(static_cast<std::int32_t>(std::stol(piece)));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    if (out.empty()) { throw std::invalid_argument("--t-sweep must not be empty"); }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::int32_t hidden = 1024;
        // A 224x224 image is 196 patches; larger inputs and batches run longer.
        std::vector<std::int32_t> sweep{64, 128, 196, 256, 512, 1024, 2048};
        int warmup = 3, repeat = 20;
        for (int i = 1; i < argc; ++i) {
            const std::string_view a(argv[i]);
            const auto next = [&](const char* what) -> std::string_view {
                if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + what); }
                return argv[i];
            };
            if (a == "--hidden") {
                hidden = std::stoi(std::string(next("--hidden")));
            } else if (a == "--t-sweep") {
                sweep = parse_list(next("--t-sweep"));
            } else if (a == "--warmup") {
                warmup = std::stoi(std::string(next("--warmup")));
            } else if (a == "--repeat") {
                repeat = std::stoi(std::string(next("--repeat")));
            } else {
                throw std::invalid_argument("unknown argument: " + std::string(a));
            }
        }
        // 16 heads either way; the intermediate follows the tower.
        const std::int32_t intermediate = hidden == 1152 ? 4304 : 4 * hidden;

        const Shape shapes[] = {
            {"patch_embedding", QType::Q6G64_F16S, hidden, 1536, kQ6, std::size(kQ6)},
            {"attention/qkv", QType::Q4G64_F16S, 3 * hidden, hidden, kQ4, std::size(kQ4)},
            {"attention/output", QType::Q5G64_F16S, hidden, hidden, kQ5, std::size(kQ5)},
            {"mlp/fc1", QType::Q4G64_F16S, intermediate, hidden, kQ4, std::size(kQ4)},
            {"mlp/fc2", QType::Q5G64_F16S, hidden, intermediate, kQ5, std::size(kQ5)},
            {"merger/fc1", QType::W8G32_F16S, 4 * hidden, 4 * hidden, kW8, std::size(kW8)},
        };

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        std::int32_t max_t = 0;
        for (const std::int32_t t : sweep) { max_t = std::max(max_t, t); }

        std::printf("vision tower tuning, hidden=%d intermediate=%d\n", hidden, intermediate);
        for (const Shape& shape : shapes) {
            bench::PackedQuantizedWeight packed =
                bench::make_row_split_weight(shape.qtype, shape.n, shape.k, shape.k);
            DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(shape.k) * max_t);
            DeviceBuffer output = bench::make_bf16(static_cast<std::size_t>(shape.n) * max_t);

            std::printf("\n%-18s n=%-6d k=%-6d\n", shape.label, shape.n, shape.k);
            for (const std::int32_t t : sweep) {
                const char* best_name = nullptr;
                double best_us        = 0.0;
                for (std::size_t c = 0; c < shape.candidate_count; ++c) {
                    const Candidate& candidate = shape.candidates[c];
                    double median              = 0.0;
                    try {
                        const auto timing = bench::measure_cold_launch(
                            [&](cudaStream_t s) {
                                Tensor x(input.p, DType::BF16, {shape.k, t});
                                Tensor out(output.p, DType::BF16, {shape.n, t});
                                candidate.launch(x, packed.weight, out, s);
                            },
                            flush, stream, warmup, repeat);
                        median = timing.median_us;
                    } catch (const std::exception&) {
                        continue; // this launcher does not accept the shape
                    }
                    if (median <= 0.0) { continue; }
                    if (best_name == nullptr || median < best_us) {
                        best_name = candidate.name;
                        best_us   = median;
                    }
                }
                if (best_name == nullptr) {
                    std::printf("  T=%-5d  (no candidate accepted this shape)\n", t);
                } else {
                    std::printf("  T=%-5d  fastest=%-18s %8.3f us\n", t, best_name, best_us);
                }
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_vision_tower_tune_bench: %s\n", error.what());
        return 1;
    }
}
