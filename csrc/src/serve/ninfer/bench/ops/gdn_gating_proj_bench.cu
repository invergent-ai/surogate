// Cold-cache qualification rig for the exact fused BF16 GDN-control projections.
//
// Examples:
//   ./build/bench/ninfer_gdn_gating_proj_bench --35b --candidate auto
//   ./build/bench/ninfer_gdn_gating_proj_bench --35b \
//     --candidate mma-split16 -p 128,512,1024
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer_bench_common.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::size_t kDefaultFlushBytes = 256ULL << 20;

struct Options {
    bool geometry35            = false;
    bool norm_control          = false;
    bool auto_route            = true;
    bool composed_norm_control = false;
    ops::detail::Bf16GdnGatingScheduleId candidate =
        ops::detail::Bf16GdnGatingScheduleId::SimtWarpRowC4;
    std::vector<std::int32_t> tokens{1, 2, 4, 6, 8, 9, 16, 32, 64, 128, 256, 512, 1024};
    int warmup              = 8;
    int repeat              = 40;
    std::size_t flush_bytes = kDefaultFlushBytes;
};

DeviceBuffer make_f32(std::size_t n, std::uint32_t seed) {
    std::vector<float> h(n);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        state         = state * 1664525u + 1013904223u;
        const float u = static_cast<float>((state >> 8) & 0x00ffffffu) * (1.0f / 16777216.0f);
        h[i]          = 2.0f * u - 1.0f;
    }
    DeviceBuffer d(n * sizeof(float));
    cudaMemcpy(d.p, h.data(), d.bytes, cudaMemcpyHostToDevice);
    return d;
}

Weight bf16_weight(const void* data, std::int32_t rows, std::int32_t hidden) {
    Weight w{};
    w.qtype           = QType::BF16_CTRL;
    w.layout          = QuantLayout::Contiguous;
    w.payload         = data;
    w.payload_bytes   = static_cast<std::uint64_t>(rows) * hidden * sizeof(std::uint16_t);
    w.qdata           = data;
    w.ndim            = 2;
    w.shape[0]        = rows;
    w.shape[1]        = hidden;
    w.padded_shape[0] = rows;
    w.padded_shape[1] = hidden;
    w.n               = rows;
    w.k               = hidden;
    return w;
}

Weight bf16_row_view(const Weight& parent, std::int32_t row_begin, std::int32_t rows) {
    const std::size_t row_bytes = static_cast<std::size_t>(parent.k) * sizeof(std::uint16_t);
    const auto* data            = static_cast<const std::uint8_t*>(parent.qdata) +
                       static_cast<std::size_t>(row_begin) * row_bytes;
    Weight view          = parent;
    view.payload         = data;
    view.payload_bytes   = static_cast<std::uint64_t>(rows) * row_bytes;
    view.qdata           = data;
    view.shape[0]        = rows;
    view.padded_shape[0] = rows;
    view.n               = rows;
    return view;
}

std::vector<std::int32_t> parse_tokens(const char* csv) {
    std::vector<std::int32_t> out;
    const char* p = csv;
    while (*p != '\0') {
        char* end    = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || v <= 0 || v > 1'000'000) {
            throw std::invalid_argument(std::string("invalid token list: ") + csv);
        }
        out.push_back(static_cast<std::int32_t>(v));
        p = (*end == ',') ? end + 1 : end;
    }
    return out;
}

ops::detail::Bf16GdnGatingScheduleId parse_candidate(std::string_view raw) {
    using S = ops::detail::Bf16GdnGatingScheduleId;
    if (raw == "simt-c4") { return S::SimtWarpRowC4; }
    if (raw == "simt-c8") { return S::SimtWarpRowC8; }
    if (raw == "mma-split32") { return S::MmaCooperativeSplit32; }
    if (raw == "mma-split16") { return S::MmaCooperativeSplit16; }
    if (raw == "mma-split8") { return S::MmaCooperativeSplit8; }
    if (raw == "mma-split4") { return S::MmaCooperativeSplit4; }
    if (raw == "mma-split2") { return S::MmaCooperativeSplit2; }
    if (raw == "mma-unsplit") { return S::MmaUnsplit; }
    throw std::invalid_argument("unknown candidate: " + std::string(raw));
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* label) -> const char* {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--35b")) {
            opt.geometry35 = true;
        } else if (!std::strcmp(argv[i], "--norm-control")) {
            opt.norm_control = true;
        } else if (!std::strcmp(argv[i], "--candidate")) {
            const std::string_view raw = next("candidate");
            opt.auto_route             = raw == "auto";
            opt.composed_norm_control  = raw == "composed";
            if (!opt.auto_route && !opt.composed_norm_control) {
                opt.candidate = parse_candidate(raw);
            }
        } else if (!std::strcmp(argv[i], "-p") || !std::strcmp(argv[i], "--tokens")) {
            opt.tokens = parse_tokens(next("tokens"));
        } else if (!std::strcmp(argv[i], "--warmup")) {
            opt.warmup = std::atoi(next("warmup"));
        } else if (!std::strcmp(argv[i], "--repeat")) {
            opt.repeat = std::atoi(next("repeat"));
        } else if (!std::strcmp(argv[i], "--flush-mib")) {
            const long mib = std::strtol(next("flush-mib"), nullptr, 10);
            if (mib <= 0) { throw std::invalid_argument("flush MiB must be positive"); }
            opt.flush_bytes = static_cast<std::size_t>(mib) << 20;
        } else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            std::printf("usage: %s [--35b] [--norm-control] "
                        "[--candidate auto|composed|simt-c4|simt-c8|mma-split32|"
                        "mma-split16|mma-split8|mma-split4|mma-split2|mma-unsplit] "
                        "[-p 1,2,...] [--warmup N] [--repeat N] [--flush-mib N]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argv[i]));
        }
    }
    if (!opt.geometry35 && !opt.auto_route) {
        throw std::invalid_argument("fixed candidate screening is supported only for --35b");
    }
    if (opt.norm_control && !opt.auto_route && !opt.composed_norm_control) {
        throw std::invalid_argument("--norm-control supports only --candidate auto or composed");
    }
    if (!opt.norm_control && opt.composed_norm_control) {
        throw std::invalid_argument("--candidate composed requires --norm-control");
    }
    if (opt.warmup < 0 || opt.repeat <= 0) {
        throw std::invalid_argument("warmup must be nonnegative and repeat positive");
    }
    return opt;
}

void run(const Options& opt, std::int32_t tokens, std::size_t interval_capacity,
         DeviceBuffer& flush) {
    const std::int32_t heads  = opt.geometry35 ? 32 : 48;
    const std::int32_t hidden = opt.geometry35 ? 2048 : 5120;
    const std::size_t x_elems = static_cast<std::size_t>(hidden) * tokens;
    const std::size_t weight_elems =
        static_cast<std::size_t>(2 * heads) * static_cast<std::size_t>(hidden);
    const std::size_t out_elems = static_cast<std::size_t>(heads) * tokens;

    DeviceBuffer x           = make_bf16(x_elems);
    DeviceBuffer weights     = make_bf16(weight_elems);
    DeviceBuffer norm_weight = make_bf16(hidden);
    DeviceBuffer A_log       = make_f32(heads, 0x1234abcdU);
    DeviceBuffer dt_bias     = make_f32(heads, 0x9876fedcU);
    DeviceBuffer h           = make_zeros(x_elems * sizeof(std::uint16_t));
    DeviceBuffer g           = make_zeros(out_elems * sizeof(float));
    DeviceBuffer beta        = make_zeros(out_elems * sizeof(float));

    Tensor tx(x.p, DType::BF16, {hidden, tokens});
    Tensor tnorm_weight(norm_weight.p, DType::BF16, {hidden});
    Tensor th(h.p, DType::BF16, {hidden, tokens});
    Tensor tA_log(A_log.p, DType::FP32, {heads});
    Tensor tdt_bias(dt_bias.p, DType::FP32, {heads});
    Tensor tg(g.p, DType::FP32, {heads, tokens});
    Tensor tbeta(beta.p, DType::FP32, {heads, tokens});
    const Weight parent = bf16_weight(weights.p, 2 * heads, hidden);
    const Weight wa     = bf16_row_view(parent, 0, heads);
    const Weight wb     = bf16_row_view(parent, heads, heads);

    const ops::detail::Bf16GdnGatingProblem problem{heads, hidden, tokens};
    const auto plan                   = opt.auto_route || opt.composed_norm_control
                                            ? ops::detail::bf16_gdn_gating_resolve_plan(problem)
                                            : ops::detail::bf16_gdn_gating_resolve_candidate(opt.candidate, problem);
    const auto norm_plan              = ops::detail::bf16_gdn_norm_gating_resolve_plan(problem);
    const std::size_t workspace_bytes = std::max(interval_capacity, plan.workspace_bytes);
    WorkspaceArena ws(std::max<std::size_t>(1, workspace_bytes));
    const auto launch = [&](cudaStream_t stream) {
        if (opt.norm_control && opt.composed_norm_control) {
            ops::rmsnorm(tx, tnorm_weight, 1.0e-6F, true, th, stream);
            if (opt.geometry35) {
                ops::gdn_gating_proj(th, parent, tA_log, tdt_bias, ws, tg, tbeta, stream);
            } else {
                ops::gdn_gating_proj(th, wa, wb, tA_log, tdt_bias, ws, tg, tbeta, stream);
            }
        } else if (opt.norm_control) {
            if (opt.geometry35) {
                ops::gdn_norm_gating_proj(tx, tnorm_weight, 1.0e-6F, parent, tA_log, tdt_bias, ws,
                                          th, tg, tbeta, stream);
            } else {
                ops::gdn_norm_gating_proj(tx, tnorm_weight, 1.0e-6F, wa, wb, tA_log, tdt_bias, ws,
                                          th, tg, tbeta, stream);
            }
        } else if (opt.auto_route) {
            if (opt.geometry35) {
                ops::gdn_gating_proj(tx, parent, tA_log, tdt_bias, ws, tg, tbeta, stream);
            } else {
                ops::gdn_gating_proj(tx, wa, wb, tA_log, tdt_bias, ws, tg, tbeta, stream);
            }
        } else {
            ops::detail::bf16_gdn_gating_execute_candidate(opt.candidate, tx, wa, wb, tA_log,
                                                           tdt_bias, ws, tg, tbeta, stream);
        }
    };
    const bench::ColdTiming timing =
        bench::measure_cold_launch(launch, flush, nullptr, opt.warmup, opt.repeat);
    const double sec = timing.median_us * 1e-6;
    const double useful_flops =
        2.0 * 2.0 * static_cast<double>(heads) * hidden * static_cast<double>(tokens);
    const bool mma = plan.token_variant != ops::detail::Bf16GdnGatingTokenVariant::None;
    const std::int32_t mma_tile = opt.geometry35 ? 64 : 128;
    const double executed_cols =
        mma ? static_cast<double>(((tokens + mma_tile - 1) / mma_tile) * mma_tile) : tokens;
    const double executed_flops  = 2.0 * 2.0 * static_cast<double>(heads) * hidden * executed_cols;
    const double useful_tflops   = useful_flops / sec / 1e12;
    const double executed_tflops = executed_flops / sec / 1e12;
    double useful_bytes =
        static_cast<double>(weight_elems * sizeof(std::uint16_t) + x_elems * sizeof(std::uint16_t) +
                            2 * out_elems * sizeof(float) + 2 * heads * sizeof(float));
    if (opt.norm_control) {
        useful_bytes += static_cast<double>((hidden + x_elems) * sizeof(std::uint16_t));
    }
    const double useful_gbs = useful_bytes / sec / 1e9;

    const char* route =
        opt.norm_control
            ? (opt.composed_norm_control
                   ? "gdn_norm_gating_proj.bf16.composed_control"
                   : ops::detail::bf16_gdn_norm_gating_schedule_name(norm_plan.schedule))
            : ops::detail::bf16_gdn_gating_schedule_name(plan.schedule);
    const std::size_t reported_workspace = opt.norm_control && !opt.composed_norm_control
                                               ? norm_plan.workspace_bytes
                                               : plan.workspace_bytes;
    std::printf("%s,%s,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%zu\n", opt.geometry35 ? "35b" : "27b",
                opt.norm_control ? "norm_control" : "control", tokens, route, timing.median_us,
                timing.min_us, timing.p95_us, useful_tflops, executed_tflops, useful_gbs,
                reported_workspace);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    try {
        const Options opt = parse_args(argc, argv);
        DeviceBuffer flush(opt.flush_bytes);
        const std::int32_t heads      = opt.geometry35 ? 32 : 48;
        const std::int32_t hidden     = opt.geometry35 ? 2048 : 5120;
        const std::int32_t min_tokens = *std::min_element(opt.tokens.begin(), opt.tokens.end());
        const std::int32_t max_tokens = *std::max_element(opt.tokens.begin(), opt.tokens.end());
        const std::size_t interval_capacity =
            opt.norm_control
                ? ops::gdn_norm_gating_proj_workspace_capacity_bytes(heads, hidden, min_tokens,
                                                                     max_tokens)
                : ops::gdn_gating_proj_workspace_capacity_bytes(heads, hidden, min_tokens,
                                                                max_tokens);
        std::printf("geometry,operation,T,route,median_us,min_us,p95_us,useful_tflops,"
                    "executed_tflops,useful_gbps,workspace_bytes\n");
        for (const std::int32_t tokens : opt.tokens) { run(opt, tokens, interval_capacity, flush); }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    }
}
