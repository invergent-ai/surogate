// Qualification benchmark for the Qwen3.6-35B G1 argmax domains.
//
//   ./ninfer_argmax_bench
//   ./ninfer_argmax_bench --shape full --cols 1
//   ./ninfer_argmax_bench --shape shortlist --cols 120
#include "ninfer/ops/argmax.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kFullPhysicalRows = 248320;
constexpr std::int32_t kFullValidRows    = 248077;
constexpr std::int32_t kShortlistRows    = 131072;
constexpr int kLogitSlots                = 256;

struct Options {
    std::string shape;
    int cols = 0;
};

void usage(const char* argv0) {
    std::printf("usage: %s [--shape full|shortlist --cols N]\n", argv0);
}

int parse_int(std::string_view value, const char* name) {
    try {
        std::size_t parsed = 0;
        const int out      = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) { throw std::invalid_argument("trailing"); }
        return out;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " expects an integer");
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto need_value = [&](const char* name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " needs a value");
            }
            return argv[++i];
        };
        if (arg == "--shape") {
            options.shape = need_value("--shape");
        } else if (arg == "--cols") {
            options.cols = parse_int(need_value("--cols"), "--cols");
        } else if (arg == "-h" || arg == "--help") {
            usage(argc > 0 ? argv[0] : "ninfer_argmax_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.shape.empty()) {
        if (options.cols != 0) { throw std::invalid_argument("--cols requires --shape"); }
        return options;
    }
    if (options.shape != "full" && options.shape != "shortlist") {
        throw std::invalid_argument("--shape must be full or shortlist");
    }
    if (options.cols == 0) { options.cols = 1; }
    const int maximum_cols = options.shape == "full" ? 128 : 120;
    if (options.cols < 1 || options.cols > maximum_cols) {
        throw std::invalid_argument("--cols exceeds the production aggregate domain");
    }
    return options;
}

void run_shape(std::int32_t physical_rows, std::int32_t valid_rows, int cols, const char* shape) {
    DeviceBuffer logits = make_bf16(static_cast<std::size_t>(physical_rows) * kLogitSlots);
    DeviceBuffer out    = make_zeros(static_cast<std::size_t>(cols) * sizeof(std::int32_t));
    auto* logits_base   = static_cast<std::uint16_t*>(logits.p);
    Tensor tout(out.p, DType::I32, {cols});

    const double bytes     = static_cast<double>(valid_rows) * 2.0 * static_cast<double>(cols);
    int launch             = 0;
    const int window_count = kLogitSlots / cols;
    const Result result    = bench_loop(
        [&](cudaStream_t stream) {
            const int slot = (launch++ % window_count) * cols;
            auto* window   = logits_base + static_cast<std::size_t>(slot) * physical_rows;
            Tensor tlogits(window, DType::BF16, {physical_rows, cols});
            ops::argmax(tlogits, tout, valid_rows, stream);
        },
        bytes);

    const std::string label = std::string("argmax ") + shape +
                              " rows=" + std::to_string(valid_rows) + " C=" + std::to_string(cols) +
                              " route=public";
    print_result(label.c_str(), result);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    try {
        const Options options = parse_args(argc, argv);
        if (options.shape.empty()) {
            for (int cols = 1; cols <= 16; ++cols) {
                run_shape(kFullPhysicalRows, kFullValidRows, cols, "full");
            }
            run_shape(kFullPhysicalRows, kFullValidRows, 128, "full");
            for (const int cols : {1, 120}) {
                run_shape(kShortlistRows, kShortlistRows, cols, "shortlist");
            }
        } else if (options.shape == "full") {
            run_shape(kFullPhysicalRows, kFullValidRows, options.cols, "full");
        } else {
            run_shape(kShortlistRows, kShortlistRows, options.cols, "shortlist");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_argmax_bench: %s\n", e.what());
        return 2;
    }
    return 0;
}
