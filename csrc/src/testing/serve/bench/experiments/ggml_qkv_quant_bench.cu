// Isolated Q/K/V measurement: production projection launchers, either quantizing
// before each projection or sharing one quantization. No engine dispatch changes.
#include "ops/linear/ggml/ggml_linear.h"
#include "ops/linear/ggml/ggml_q8_1.h"
#include "sinfer_bench_common.h"

#include <array>
#include <random>
#include <string>

using namespace sinfer;
namespace gg = sinfer::ops::detail::ggml;

namespace {

struct Shape {
    const char* name;
    int hidden;
    std::array<int, 3> rows;
};

// Valid, finite native blocks with deterministic varied codes and scales. These
// are synthetic operator inputs, not checkpoint weights or model TTFT results.
template <class Block>
DeviceBuffer make_blocks(std::size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<Block> blocks(count);
    for (auto& block : blocks) {
        auto* bytes = reinterpret_cast<unsigned char*>(&block);
        for (std::size_t i = 0; i < sizeof(Block); ++i) { bytes[i] = rng() & 255; }
        const auto scale = __float2half_rn(0.0001f * (1 + rng() % 9));
        if constexpr (requires { block.dm; }) {
            block.dm = __halves2half2(scale, scale);
        } else {
            block.d = scale;
        }
    }
    DeviceBuffer result(count * sizeof(Block));
    result.copy_from_host(blocks.data(), result.bytes);
    return result;
}

DeviceBuffer make_weight(gg::GgmlType type, int rows, int hidden, unsigned seed) {
    const auto count = std::size_t(rows) * (hidden / gg::block_values(type));
    switch (type) {
    case gg::GgmlType::Q4_K: return make_blocks<gg::block_q4_K>(count, seed);
    case gg::GgmlType::Q5_K: return make_blocks<gg::block_q5_K>(count, seed);
    case gg::GgmlType::Q6_K: return make_blocks<gg::block_q6_K>(count, seed);
    case gg::GgmlType::Q8_0: return make_blocks<gg::block_q8_0>(count, seed);
    case gg::GgmlType::IQ4_NL: return make_blocks<gg::block_iq4_nl>(count, seed);
    default: throw std::invalid_argument("unsupported benchmark format");
    }
}

double median(std::vector<double> values) {
    return bench::summarize_timings(std::move(values)).median_us;
}

void measure(const Shape& shape, const std::array<gg::GgmlType, 3>& types,
             const std::array<DeviceBuffer, 3>& weights, int tokens, int repeat,
             DeviceBuffer& flush, cudaStream_t stream) {
    const int hidden = shape.hidden;
    auto x = bench::make_bf16(std::size_t(hidden) * tokens);
    std::array<DeviceBuffer, 3> out;
    for (int i = 0; i < 3; ++i) { out[i] = DeviceBuffer(std::size_t(shape.rows[i]) * tokens * 2); }
    // Both paths use this same allocation; the current path reuses it sequentially.
    DeviceBuffer scratch(gg::linear_workspace_bytes(shape.rows[0], hidden, tokens));
    const auto launch = [&](bool shared) {
        if (shared) {
            auto* codes = static_cast<std::int8_t*>(scratch.p);
            auto* ds = reinterpret_cast<__half2*>(codes + std::size_t(hidden) * tokens);
            gg::quantize_q8_1_planes_launch(static_cast<const __nv_bfloat16*>(x.p),
                                            hidden, tokens, codes, ds, stream);
        }
        for (int i = 0; i < 3; ++i) {
            if (shared) {
                gg::linear_prequantized_launch(types[i], weights[i].p, shape.rows[i], hidden,
                    tokens, static_cast<__nv_bfloat16*>(out[i].p), scratch.p, scratch.bytes, stream);
            } else {
                gg::linear_launch(types[i], weights[i].p, shape.rows[i], hidden,
                    static_cast<const __nv_bfloat16*>(x.p), tokens,
                    static_cast<__nv_bfloat16*>(out[i].p), scratch.p, scratch.bytes, stream);
            }
        }
    };

    launch(false);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::array<std::vector<std::uint16_t>, 3> expected;
    for (int i = 0; i < 3; ++i) {
        expected[i].resize(out[i].bytes / 2);
        out[i].copy_to_host(expected[i].data(), out[i].bytes);
        out[i].fill(0xff);
    }
    launch(true);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto verify = [&] {
        for (int i = 0; i < 3; ++i) {
            std::vector<std::uint16_t> actual(expected[i].size());
            out[i].copy_to_host(actual.data(), out[i].bytes);
            if (actual != expected[i]) { throw std::runtime_error("Q/K/V bitwise mismatch"); }
            for (auto value : actual) {
                if ((value & 0x7f80) == 0x7f80) { throw std::runtime_error("non-finite output"); }
            }
        }
    };
    verify();

    for (bool cold : {false, true}) {
        // Cold cache: evict before each measured graph, outside the timed region.
        // Warm cache: batch 20 operations to amortize graph launch/event overhead.
        const int inner = cold ? 1 : 20;
        std::array<bench::TimedGraph, 2> graphs;
        for (int variant = 0; variant < 2; ++variant) {
            graphs[variant].capture(stream, [&](cudaStream_t) {
                for (int j = 0; j < inner; ++j) { launch(variant == 1); }
            });
            if (graphs[variant].nodes() != std::size_t(inner * (variant == 0 ? 6 : 4))) {
                throw std::runtime_error("unexpected graph node count");
            }
        }
        for (int j = 0; j < 8; ++j) {
            graphs[0].launch(stream);
            graphs[1].launch(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        verify();
        std::array<std::vector<double>, 2> samples;
        std::vector<double> savings;
        for (int j = 0; j < repeat; ++j) {
            // Alternate A/B order to avoid always favoring one path as clocks drift.
            for (int pass = 0; pass < 2; ++pass) {
                const int variant = (j + pass) % 2;
                if (cold) { bench::flush_l2(flush, stream); }
                samples[variant].push_back(graphs[variant].launch_timed(stream) / inner);
            }
            savings.push_back(samples[0].back() - samples[1].back());
        }
        const double before = median(samples[0]);
        const double after = median(samples[1]);
        const auto deltas = bench::summarize_timings(savings);
        std::printf("%s,%s,%s,%s,%d,%d,%d,%d,%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%zu,6,4,1\n",
            shape.name, gg::type_name(types[0]), gg::type_name(types[1]), gg::type_name(types[2]),
            hidden, shape.rows[0], shape.rows[1], shape.rows[2], tokens,
            cold ? "cold" : "warm", repeat, before, after, (before - after) / before * 100,
            deltas.median_us, deltas.min_us, scratch.bytes);
        std::fflush(stdout);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        int repeat = 31;
        int only_tokens = 0;
        std::string only_shape;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (i + 1 == argc) { throw std::invalid_argument("missing option value"); }
            const std::string value(argv[++i]);
            if (arg == "--repeat") { repeat = std::stoi(value); }
            else if (arg == "--tokens") { only_tokens = std::stoi(value); }
            else if (arg == "--shape") { only_shape = value; }
            else { throw std::invalid_argument("unknown option: " + arg); }
        }
        if (repeat < 1 || repeat > 10000 || only_tokens < 0 || only_tokens > 8192) {
            throw std::invalid_argument("invalid repeat or token count");
        }
        DeviceContext device;
        char pci[32]{};
        CUDA_CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), 0));
        std::fprintf(stderr, "GPU=%s PCI=%s L2=%d repeat=%d; synthetic weights; exact parity required\n",
                     device.props.name, pci, device.props.l2CacheSize, repeat);
        DeviceBuffer flush(std::size_t{256} << 20);
        using T = gg::GgmlType;
        const std::array<std::array<T, 3>, 7> formats{{
            {T::Q4_K, T::Q4_K, T::Q4_K}, {T::Q5_K, T::Q5_K, T::Q5_K},
            {T::Q6_K, T::Q6_K, T::Q6_K}, {T::Q8_0, T::Q8_0, T::Q8_0},
            {T::IQ4_NL, T::IQ4_NL, T::IQ4_NL}, {T::Q5_K, T::Q5_K, T::Q6_K},
            {T::Q8_0, T::Q5_K, T::Q6_K}}};
        std::vector<int> tokens{1, 8, 16, 32, 128, 512, 2048};
        if (only_tokens) { tokens = {only_tokens}; }
        const std::array<Shape, 4> shapes{{
            {"tinyllama", 2048, {2048, 256, 256}},
            {"qwen3_06b", 1024, {2048, 1024, 1024}},
            {"llama_8b", 4096, {4096, 1024, 1024}},
            {"llama_70b", 8192, {8192, 1024, 1024}}}};
        if (!only_shape.empty() && std::none_of(shapes.begin(), shapes.end(), [&](const Shape& s) {
                return only_shape == s.name;
            })) { throw std::invalid_argument("unknown shape: " + only_shape); }
        std::puts("shape,q_type,k_type,v_type,hidden,q_rows,k_rows,v_rows,tokens,cache,repeat,"
                  "baseline_us,shared_us,saved_pct,paired_saved_us,min_paired_saved_us,"
                  "scratch_bytes,baseline_kernels,shared_kernels,bitwise_equal");
        for (const auto& shape : shapes) {
            if (!only_shape.empty() && only_shape != shape.name) { continue; }
            for (const auto& types : formats) {
                std::array<DeviceBuffer, 3> weights;
                for (int i = 0; i < 3; ++i) {
                    weights[i] = make_weight(types[i], shape.rows[i], shape.hidden, 41 + i);
                }
                for (int t : tokens) { measure(shape, types, weights, t, repeat, flush, device.stream); }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
