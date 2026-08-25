// Cold-cache CUDA Graph benchmark for the public DFlash LinearPair contract.

#include "ninfer/ops/linear_pair.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kParentRows  = 6144;
constexpr std::int32_t kFirstRow    = 4096;
constexpr std::int32_t kSecondRow   = 5120;
constexpr std::int32_t kRows        = 1024;
constexpr std::int32_t kHidden      = 2048;
constexpr double kRtx5090ReadGBs    = 1674.5;
constexpr double kRtx5090Bf16Tflops = 209.5;

struct Options {
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 16, 32, 48, 64, 96, 128};
    int warmup                = 5;
    int repeat                = 30;
    std::uint64_t flush_bytes = 256ULL << 20;
};

std::uint64_t parse_u64(std::string_view text, const char* label) {
    const std::string value(text);
    char* end                       = nullptr;
    errno                           = 0;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + value);
    }
    return static_cast<std::uint64_t>(parsed);
}

std::int32_t parse_positive_i32(std::string_view text, const char* label) {
    const std::uint64_t value = parse_u64(text, label);
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(label) + " must be a positive int32");
    }
    return static_cast<std::int32_t>(value);
}

int parse_nonnegative_int(std::string_view text, const char* label) {
    const std::uint64_t value = parse_u64(text, label);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(label) + " is too large");
    }
    return static_cast<int>(value);
}

std::vector<std::int32_t> parse_tokens(std::string_view text) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end = text.find(',', begin);
        result.push_back(parse_positive_i32(
            text.substr(begin, end == std::string_view::npos ? text.size() - begin : end - begin),
            "token extent"));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument("--tokens must not be empty"); }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::int32_t> parse_sweep(std::string_view text) {
    const std::size_t first = text.find(':');
    const std::size_t second =
        first == std::string_view::npos ? std::string_view::npos : text.find(':', first + 1);
    if (first == std::string_view::npos || (second != std::string_view::npos &&
                                            text.find(':', second + 1) != std::string_view::npos)) {
        throw std::invalid_argument("--sweep must be START:END[:STEP]");
    }
    const std::int32_t begin = parse_positive_i32(text.substr(0, first), "sweep start");
    const std::int32_t end   = parse_positive_i32(
        text.substr(first + 1, second == std::string_view::npos ? text.size() - first - 1
                                                                  : second - first - 1),
        "sweep end");
    const std::int32_t step = second == std::string_view::npos
                                  ? 1
                                  : parse_positive_i32(text.substr(second + 1), "sweep step");
    if (begin > end) { throw std::invalid_argument("sweep start must not exceed end"); }
    std::vector<std::int32_t> result;
    for (std::int64_t token = begin; token <= end; token += step) {
        result.push_back(static_cast<std::int32_t>(token));
        if (token > static_cast<std::int64_t>(end) - step) { break; }
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool have_tokens = false;
    bool have_sweep  = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--tokens") {
            options.tokens = parse_tokens(next("--tokens value"));
            have_tokens    = true;
        } else if (argument == "--sweep") {
            options.tokens = parse_sweep(next("--sweep value"));
            have_sweep     = true;
        } else if (argument == "--warmup") {
            options.warmup = parse_nonnegative_int(next("--warmup value"), "warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_nonnegative_int(next("--repeat value"), "repeat");
        } else if (argument == "--flush-mib") {
            const std::uint64_t mib = parse_u64(next("--flush-mib value"), "flush MiB");
            if (mib > (std::numeric_limits<std::uint64_t>::max() >> 20)) {
                throw std::overflow_error("flush size overflows");
            }
            options.flush_bytes = mib << 20;
        } else if (argument == "--help" || argument == "-h") {
            std::printf("Usage: %s [--tokens 1,2,... | --sweep START:END[:STEP]] "
                        "[--warmup N] [--repeat N] [--flush-mib N]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (have_tokens && have_sweep) {
        throw std::invalid_argument("--tokens and --sweep are mutually exclusive");
    }
    if (options.repeat <= 0) { throw std::invalid_argument("--repeat must be positive"); }
    return options;
}

std::uint64_t pair_weight_bytes() {
    constexpr std::uint64_t kCodesPerWeight = static_cast<std::uint64_t>(kRows) * kHidden;
    constexpr std::uint64_t kScalesPerWeight =
        static_cast<std::uint64_t>(kRows) * (kHidden / 32) * sizeof(std::uint16_t);
    return 2 * (kCodesPerWeight + kScalesPerWeight);
}

double useful_bytes(std::int32_t tokens) {
    return static_cast<double>(pair_weight_bytes()) +
           2.0 * static_cast<double>((kHidden + 2 * kRows) * tokens);
}

double useful_flops(std::int32_t tokens) {
    return 4.0 * static_cast<double>(kRows) * kHidden * tokens;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::int32_t maximum_tokens =
            *std::max_element(options.tokens.begin(), options.tokens.end());

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        DeviceBuffer flush(options.flush_bytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * maximum_tokens);
        DeviceBuffer first_output(static_cast<std::size_t>(kRows) * maximum_tokens * 2);
        DeviceBuffer second_output(static_cast<std::size_t>(kRows) * maximum_tokens * 2);
        bench::PackedQuantizedWeight parent = bench::make_row_split_weight(
            QType::W8G32_F16S, kParentRows, kHidden, kHidden, {0x31, 0x00, 0x3c00});
        const Weight first_weight  = bench::row_view(parent.weight, kFirstRow, kRows);
        const Weight second_weight = bench::row_view(parent.weight, kSecondRow, kRows);

        std::printf("# gpu=%s public=linear_pair shape=two_adjacent_W8[1024,2048] "
                    "execution=graph_replay cache=cold read_reference=%.1f_GB/s "
                    "bf16_tc_reference=%.1f_TFLOP/s\n",
                    properties.name, kRtx5090ReadGBs, kRtx5090Bf16Tflops);

        double t1_median = std::numeric_limits<double>::quiet_NaN();
        for (const std::int32_t tokens : options.tokens) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor first(first_output.p, DType::BF16, {kRows, tokens});
            Tensor second(second_output.p, DType::BF16, {kRows, tokens});
            bench::TimedGraph graph;
            graph.capture(stream, [&](cudaStream_t capture_stream) {
                ops::linear_pair(x, first_weight, second_weight, first, second, capture_stream);
            });
            graph.launch(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const bench::ColdTiming timing =
                bench::measure_cold_graph(graph, flush, stream, options.warmup, options.repeat);
            if (tokens == 1) { t1_median = timing.median_us; }

            const double seconds       = timing.median_us * 1.0e-6;
            const double gbs           = useful_bytes(tokens) / seconds / 1.0e9;
            const double tflops        = useful_flops(tokens) / seconds / 1.0e12;
            const double extrapolation = std::isnan(t1_median)
                                             ? std::numeric_limits<double>::quiet_NaN()
                                             : tokens * t1_median / timing.median_us;
            std::printf("T=%-3d median=%8.3f us min=%8.3f us p95=%8.3f us "
                        "effective=%7.1f GB/s READ=%5.1f%% logical=%6.2f TFLOP/s ",
                        tokens, timing.median_us, timing.min_us, timing.p95_us, gbs,
                        100.0 * gbs / kRtx5090ReadGBs, tflops);
            if (tokens == 1) {
                std::printf("TC=n/a ");
            } else {
                std::printf("TC=%5.1f%% ", 100.0 * tflops / kRtx5090Bf16Tflops);
            }
            if (std::isnan(extrapolation)) {
                std::printf("T1_linear=n/a\n");
            } else {
                std::printf("T1_linear=%5.2fx\n", extrapolation);
            }
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_linear_pair_bench: %s\n", error.what());
        return 1;
    }
}
