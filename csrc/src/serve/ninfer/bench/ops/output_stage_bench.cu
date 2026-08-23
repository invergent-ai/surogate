// Captured Qwen3.6-35B-A3B target-verification output-stage benchmark.
//
// The timed graph is final RMSNorm -> Q6 full-vocabulary head -> argmax.

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"
#include "ops/linear/q6/q6_dispatch.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHidden          = 2048;
constexpr std::int32_t kVocab           = 248320;
constexpr std::int32_t kValidVocab      = 248077;
constexpr std::int32_t kMaxTokens       = 16;
constexpr std::size_t kDefaultFlushSize = 256ULL << 20;
constexpr float kRmsEps                 = 1.0e-6F;

struct Options {
    std::vector<std::int32_t> t_sweep{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int warmup             = 5;
    int repeat             = 50;
    std::size_t flush_size = kDefaultFlushSize;
};

std::vector<std::int32_t> parse_t_sweep(std::string_view raw) {
    std::vector<std::int32_t> result;
    for (std::size_t begin = 0; begin < raw.size();) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        const long value = std::stol(token);
        if (value < 1 || value > kMaxTokens) {
            throw std::invalid_argument("--t-sweep values must be in [1,16]");
        }
        result.push_back(static_cast<std::int32_t>(value));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument("--t-sweep must not be empty"); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[i];
        };
        if (arg == "--t-sweep") {
            options.t_sweep = parse_t_sweep(next("--t-sweep value"));
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (arg == "--flush-mib") {
            const long mib = std::stol(std::string(next("--flush-mib value")));
            if (mib <= 0) { throw std::invalid_argument("--flush-mib must be positive"); }
            options.flush_size = static_cast<std::size_t>(mib) << 20;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--t-sweep 1,2,...,16] "
                        "[--warmup N] [--repeat N] [--flush-mib N]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    return options;
}

const char* q6_launch_name(ops::detail::Q6Launch launch) {
    if (launch == ops::detail::launch_q6_simt_r8_c4) { return "q6.simt.r8_c4"; }
    if (launch == ops::detail::launch_q6_simt_r8_c8) { return "q6.simt.r8_c8"; }
    if (launch == ops::detail::launch_q6_mma_r64_c64) { return "q6.mma.r64_c64"; }
    if (launch == ops::detail::launch_q6_mma_r64_c128) { return "q6.mma.r64_c128"; }
    throw std::invalid_argument("output-stage benchmark received an unknown Q6 launcher");
}

int run(const Options& options) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    bench::PackedQuantizedWeight head = bench::make_row_split_weight(
        QType::Q6G64_F16S, kVocab, kHidden, kHidden, {0x35, 0x12, 0x3c00});
    ninfer::DeviceBuffer residual =
        bench::make_bf16(static_cast<std::size_t>(kHidden) * kMaxTokens);
    std::vector<std::uint16_t> norm_host(kHidden, bench::f32_to_bf16(0.0F));
    ninfer::DeviceBuffer norm(norm_host.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(norm.p, norm_host.data(), norm.bytes, cudaMemcpyHostToDevice));
    ninfer::DeviceBuffer hidden(static_cast<std::size_t>(kHidden) * kMaxTokens *
                                sizeof(std::uint16_t));
    ninfer::DeviceBuffer logits(static_cast<std::size_t>(kVocab) * kMaxTokens *
                                sizeof(std::uint16_t));
    ninfer::DeviceBuffer tokens(static_cast<std::size_t>(kMaxTokens) * sizeof(std::int32_t));
    ninfer::DeviceBuffer flush(options.flush_size);
    WorkspaceArena workspace(1);

    std::printf("# gpu=RTX_5090 cuda=13.1 sm=120a flush_mib=%zu warmup=%d repeat=%d\n",
                options.flush_size >> 20, options.warmup, options.repeat);
    std::printf("%4s %5s %10s %10s %10s %-24s %s\n", "T", "nodes", "median_us", "min_us", "p95_us",
                "head_route", "argmax_route");

    for (const std::int32_t t : options.t_sweep) {
        Tensor x(residual.p, DType::BF16, {kHidden, t});
        Tensor norm_weight(norm.p, DType::BF16, {kHidden});
        Tensor normalized(hidden.p, DType::BF16, {kHidden, t});
        Tensor output(logits.p, DType::BF16, {kVocab, t});
        Tensor selected(tokens.p, DType::I32, {t});
        const ops::detail::Q6Launch head_launch =
            ops::detail::select_q6_a16_launch(kVocab, kHidden, t);

        const auto body = [&](cudaStream_t body_stream) {
            ops::rmsnorm(x, norm_weight, kRmsEps, true, normalized, body_stream);
            ops::linear(normalized, head.weight, output, body_stream);
            ops::argmax(output, selected, kValidVocab, body_stream);
        };

        body(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        bench::TimedGraph graph;
        graph.capture(stream, body);
        const bench::ColdTiming timing =
            bench::measure_cold_graph(graph, flush, stream, options.warmup, options.repeat);
        std::printf("%4d %5zu %10.3f %10.3f %10.3f %-24s %s\n", t, graph.nodes(), timing.median_us,
                    timing.min_us, timing.p95_us, q6_launch_name(head_launch), "public");
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_output_stage_bench: %s\n", error.what());
        return 2;
    }
}
