// Performance bench for the BF16 depthwise causal width-4 convolution + SiLU Op.
//
// Examples:
//   ./ninfer_causal_conv1d_silu_bench --decode --channels 8192
//   ./ninfer_causal_conv1d_silu_bench --prefill --channels 8192 --tokens 1024
//   ./ninfer_causal_conv1d_silu_bench --distinct --channels 8192 --tokens 6
//   ./ninfer_causal_conv1d_silu_bench --snapshot --channels 8192 --tokens 6 --slots 7
// Printed logical GB/s is informational; NCU determines the applicable resource roofline.
#include "ninfer/ops/causal_conv1d_silu.h"
#include "core/device.h"
#include "ninfer_bench_common.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

struct Options {
    std::int32_t channels     = 8192;
    std::int32_t tokens       = 1024;
    std::int32_t slots        = 7;
    std::int32_t initial_slot = 6;
    std::int32_t batch        = 1;
    std::vector<std::int32_t> valid_columns;
    bool decode   = false;
    bool prefill  = false;
    bool distinct = false;
    bool snapshot = false;
};

__global__ void copy_u128_kernel(const uint4* src, uint4* dst, std::size_t n4) {
    const std::size_t start  = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = start; i < n4; i += stride) { dst[i] = src[i]; }
}

DeviceBuffer make_varied_bf16(std::size_t n, std::uint32_t seed) {
    std::vector<std::uint16_t> h(n);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        state         = state * 1664525u + 1013904223u;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
        h[i]          = f32_to_bf16(2.0f * u - 1.0f);
    }
    DeviceBuffer d(n * 2u);
    cudaMemcpy(d.p, h.data(), n * 2u, cudaMemcpyHostToDevice);
    return d;
}

void copy_bytes_launch(const DeviceBuffer& src, DeviceBuffer& dst, std::size_t copy_bytes,
                       cudaStream_t stream) {
    constexpr int kBlock            = 256;
    constexpr std::size_t kVecBytes = sizeof(uint4);
    const std::size_t n4            = (copy_bytes + kVecBytes - 1u) / kVecBytes;
    const int grid                  = static_cast<int>((n4 + kBlock - 1) / kBlock);
    copy_u128_kernel<<<grid, kBlock, 0, stream>>>(static_cast<const uint4*>(src.p),
                                                  static_cast<uint4*>(dst.p), n4);
    CUDA_CHECK(cudaGetLastError());
}

void run_copy_baseline(double bytes, const char* tag) {
    const auto copy_bytes   = static_cast<std::size_t>(bytes / 2.0);
    const auto padded_bytes = (copy_bytes + sizeof(uint4) - 1u) & ~(sizeof(uint4) - 1u);
    DeviceBuffer src        = make_varied_bf16(padded_bytes / 2u, 0xc001c0deU);
    DeviceBuffer dst        = make_zeros(padded_bytes);

    const Result r =
        bench_loop([&](cudaStream_t s) { copy_bytes_launch(src, dst, copy_bytes, s); }, bytes);
    print_result(tag, r);
}

std::string shape_tag(const char* mode, std::int32_t channels, std::int32_t tokens,
                      std::int32_t batch = 1) {
    return std::string("causal_conv1d ") + mode + " [" + std::to_string(channels) + "," +
           std::to_string(tokens) + "," + std::to_string(batch) + "]";
}

void run_prefill(const Options& options, bool distinct) {
    const std::size_t n       = static_cast<std::size_t>(options.channels) * options.tokens;
    const std::size_t state_n = static_cast<std::size_t>(options.channels) * 3u;

    DeviceBuffer x = make_varied_bf16(n, 0x12345678U);
    DeviceBuffer weight =
        make_varied_bf16(static_cast<std::size_t>(options.channels) * 4u, 0x87654321U);
    DeviceBuffer state_in  = make_varied_bf16(state_n, 0x31415926U);
    DeviceBuffer state_out = make_zeros(state_n * 2u);
    DeviceBuffer out       = make_zeros(n * 2u);

    Tensor tx(x.p, DType::BF16, {options.channels, options.tokens});
    Tensor tw(weight.p, DType::BF16, {options.channels, 4});
    Tensor tin(state_in.p, DType::BF16, {options.channels, 3});
    Tensor tout_state(state_out.p, DType::BF16, {options.channels, 3});
    Tensor tout(out.p, DType::BF16, {options.channels, options.tokens});

    // Informational compulsory traffic: x/out, the four-tap weight, and state read/write. NCU
    // counters remain the performance authority.
    const double bytes = 4.0 * static_cast<double>(n) + 20.0 * options.channels;
    const Result r     = bench_loop(
        [&](cudaStream_t s) {
            if (distinct) {
                ops::causal_conv1d_silu(tx, tw, tin, tout_state, tout, s);
            } else {
                ops::causal_conv1d_silu(tx, tw, tin, tout, s);
            }
        },
        bytes);
    const std::string tag =
        shape_tag(distinct ? "distinct" : "prefill", options.channels, options.tokens);
    print_result(tag.c_str(), r);
}

void run_decode(const Options& options) {
    const std::size_t channels = static_cast<std::size_t>(options.channels);

    DeviceBuffer x      = make_varied_bf16(channels, 0x12345678U);
    DeviceBuffer weight = make_varied_bf16(channels * 4u, 0x87654321U);
    DeviceBuffer state  = make_varied_bf16(channels * 3u, 0x31415926U);
    DeviceBuffer out    = make_zeros(channels * 2u);

    Tensor tx(x.p, DType::BF16, {options.channels, 1});
    Tensor tw(weight.p, DType::BF16, {options.channels, 4});
    Tensor ts(state.p, DType::BF16, {options.channels, 3});
    Tensor tout(out.p, DType::BF16, {options.channels, 1});

    const double bytes = 24.0 * options.channels;
    const Result r =
        bench_loop([&](cudaStream_t s) { ops::causal_conv1d_silu(tx, tw, ts, tout, s); }, bytes);
    const std::string tag = shape_tag("decode", options.channels, 1);
    print_result(tag.c_str(), r);
    run_copy_baseline(bytes, "copy same-byte decode baseline");
}

void run_snapshot(const Options& options) {
    const std::size_t n =
        static_cast<std::size_t>(options.channels) * options.tokens * options.batch;
    const std::int32_t required_slots =
        options.batch == 1 ? options.slots : options.batch * options.tokens + options.batch;
    const std::int32_t slots  = std::max(options.slots, required_slots);
    const std::size_t state_n = static_cast<std::size_t>(options.channels) * 3u * slots;

    DeviceBuffer x = make_varied_bf16(n, 0x12345678U);
    DeviceBuffer weight =
        make_varied_bf16(static_cast<std::size_t>(options.channels) * 4u, 0x87654321U);
    DeviceBuffer states = make_varied_bf16(state_n, 0x31415926U);
    DeviceBuffer initial_slot =
        make_zeros(static_cast<std::size_t>(options.batch) * sizeof(std::int32_t));
    DeviceBuffer snapshot_base_slot =
        make_zeros(static_cast<std::size_t>(options.batch) * sizeof(std::int32_t));
    DeviceBuffer valid_columns;
    DeviceBuffer out = make_zeros(n * 2u);
    std::vector<std::int32_t> initial_host(static_cast<std::size_t>(options.batch));
    std::vector<std::int32_t> base_host(static_cast<std::size_t>(options.batch));
    if (options.batch == 1) {
        initial_host[0] = options.initial_slot;
    } else {
        for (std::int32_t row = 0; row < options.batch; ++row) {
            base_host[static_cast<std::size_t>(row)]    = row * options.tokens;
            initial_host[static_cast<std::size_t>(row)] = options.batch * options.tokens + row;
        }
    }
    CUDA_CHECK(cudaMemcpy(initial_slot.p, initial_host.data(), initial_slot.bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(snapshot_base_slot.p, base_host.data(), snapshot_base_slot.bytes,
                          cudaMemcpyHostToDevice));
    if (!options.valid_columns.empty()) {
        valid_columns = make_zeros(options.valid_columns.size() * sizeof(std::int32_t));
        CUDA_CHECK(cudaMemcpy(valid_columns.p, options.valid_columns.data(), valid_columns.bytes,
                              cudaMemcpyHostToDevice));
    }

    Tensor tx(x.p, DType::BF16, {options.channels, options.tokens, options.batch});
    Tensor tw(weight.p, DType::BF16, {options.channels, 4});
    Tensor ts(states.p, DType::BF16, {options.channels, 3, slots});
    Tensor tvalid;
    if (!options.valid_columns.empty()) {
        tvalid = Tensor(valid_columns.p, DType::I32, {options.batch});
    }
    Tensor tslot(initial_slot.p, DType::I32, {options.batch});
    Tensor tsnapshot_base(snapshot_base_slot.p, DType::I32, {options.batch});
    Tensor tout(out.p, DType::BF16, {options.channels, options.tokens, options.batch});

    // The selected history and four weights are reusable across T; each token reads x, writes out,
    // and publishes three BF16 state columns.
    const double bytes = 8.0 * options.channels + 6.0 * options.channels * options.batch +
                         10.0 * static_cast<double>(n);
    const Result r = bench_loop(
        [&](cudaStream_t s) {
            ops::causal_conv1d_silu_snapshot(tx, tw, ts, tvalid, tslot, tsnapshot_base, tout, s);
        },
        bytes);
    const std::string tag =
        shape_tag(options.valid_columns.empty() ? "snapshot dense" : "snapshot masked",
                  options.channels, options.tokens, options.batch);
    print_result(tag.c_str(), r);
}

void print_usage(const char* program) {
    std::fprintf(stderr,
                 "usage: %s [--decode] [--prefill] [--distinct] [--snapshot] "
                 "[--channels C] [--tokens T] [--batch B] [--valid-columns V0,V1,...] "
                 "[--slots S] [--initial-slot I]\n",
                 program);
}

bool parse_positive(const char* text, std::int32_t& value) {
    const long parsed = std::strtol(text, nullptr, 10);
    if (parsed <= 0 || parsed > INT32_MAX) { return false; }
    value = static_cast<std::int32_t>(parsed);
    return true;
}

bool parse_valid_columns(const char* text, std::vector<std::int32_t>& values) {
    values.clear();
    const char* cursor = text;
    while (*cursor != '\0') {
        char* end        = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (end == cursor || value <= 0 || value > INT32_MAX) { return false; }
        values.push_back(static_cast<std::int32_t>(value));
        if (*end == '\0') return true;
        if (*end != ',') return false;
        cursor = end + 1;
    }
    return false;
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--decode")) {
            options.decode = true;
        } else if (!std::strcmp(argv[i], "--prefill")) {
            options.prefill = true;
        } else if (!std::strcmp(argv[i], "--distinct")) {
            options.distinct = true;
        } else if (!std::strcmp(argv[i], "--snapshot")) {
            options.snapshot = true;
        } else if (!std::strcmp(argv[i], "--valid-columns") && i + 1 < argc) {
            if (!parse_valid_columns(argv[++i], options.valid_columns)) { return false; }
        } else if ((!std::strcmp(argv[i], "--channels") || !std::strcmp(argv[i], "--tokens") ||
                    !std::strcmp(argv[i], "--slots") || !std::strcmp(argv[i], "--initial-slot") ||
                    !std::strcmp(argv[i], "--batch")) &&
                   i + 1 < argc) {
            const char* flag = argv[i++];
            if (!std::strcmp(flag, "--initial-slot")) {
                const long parsed = std::strtol(argv[i], nullptr, 10);
                if (parsed < 0 || parsed > INT32_MAX) { return false; }
                options.initial_slot = static_cast<std::int32_t>(parsed);
            } else {
                std::int32_t* destination = !std::strcmp(flag, "--channels") ? &options.channels
                                            : !std::strcmp(flag, "--tokens") ? &options.tokens
                                            : !std::strcmp(flag, "--batch")  ? &options.batch
                                                                             : &options.slots;
                if (!parse_positive(argv[i], *destination)) { return false; }
            }
        } else {
            return false;
        }
    }
    if (!options.decode && !options.prefill && !options.distinct && !options.snapshot) {
        options.decode = options.prefill = true;
    }
    if (options.batch > 8 || (options.batch > 1 && options.tokens > 16) ||
        ((!options.snapshot) && (options.batch != 1 || !options.valid_columns.empty())) ||
        (!options.valid_columns.empty() &&
         options.valid_columns.size() != static_cast<std::size_t>(options.batch))) {
        return false;
    }
    for (const std::int32_t valid : options.valid_columns) {
        if (valid > options.tokens) return false;
    }
    return !options.snapshot || options.batch > 1 ||
           (options.initial_slot < options.slots && options.tokens <= options.slots);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 2;
    }

    if (options.decode) run_decode(options);
    if (options.prefill) run_prefill(options, false);
    if (options.distinct) run_prefill(options, true);
    if (options.snapshot) run_snapshot(options);
    return 0;
}
