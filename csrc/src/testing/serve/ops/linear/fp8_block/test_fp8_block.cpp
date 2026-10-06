// Block-scaled FP8 linears against a double-precision reference: the decode GEMV on exact
// activations, and the tensor-core tile on activations quantised per token per 128 -- the
// reference quantises the same way, so what remains is accumulation order and BF16 output.
#include "api/ops/linear.h"
#include "api/ops/linear_add.h"
#include "api/ops/linear_swiglu_down_add.h"
#include "api/ops/silu_mul.h"
#include "ops/linear/fp8_block/fp8_block.h"
#include "ops/linear/fp8_block/fp8_block_sm90_gemm.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                                        \
    do {                                                                                        \
        const cudaError_t status_ = (call);                                                     \
        if (status_ != cudaSuccess) {                                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(status_),      \
                         __FILE__, __LINE__);                                                   \
            std::exit(2);                                                                       \
        }                                                                                       \
    } while (0)

namespace {
using namespace sinfer;

float e4m3_to_float(std::uint8_t code) {
    __nv_fp8_e4m3 v;
    v.__x = code;
    return static_cast<float>(v);
}
std::uint8_t float_to_e4m3(float value) { return __nv_fp8_e4m3(value).__x; }

struct Case {
    std::int32_t rows, k, tokens;
    bool rows_view;       // exercise linear_rows on the second half
    bool per_row = false; // one scale per row (the compressed-tensors kind) instead of per 128x128
};

int run(const Case& c) {
    std::mt19937 rng(static_cast<unsigned>(1234 + c.rows + 7 * c.k + 13 * c.tokens));
    std::uniform_real_distribution<float> uw(-1.0f, 1.0f), us(0.5f, 2.0f), ux(-3.0f, 3.0f);
    const std::int32_t kb = c.k / 128;
    const std::int32_t k_per = c.per_row ? c.k : 128, rows_per = c.per_row ? 1 : 128;
    const std::int32_t scale_cols = c.k / k_per, scale_rows = c.rows / rows_per;
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(c.rows) * c.k);
    std::vector<float> scales(static_cast<std::size_t>(scale_rows) * scale_cols), x(static_cast<std::size_t>(c.k) * c.tokens);
    for (auto& v : codes) { v = float_to_e4m3(uw(rng)); }
    for (auto& v : scales) { v = us(rng) * 0.01f; }
    for (auto& v : x) { v = __bfloat162float(__float2bfloat16(ux(rng))); }
    const std::int32_t row_begin = c.rows_view ? 128 * (c.rows / 256) : 0; // a 128-block boundary
    const std::int32_t out_rows  = c.rows - row_begin;
    // The weight the launch sees: its rows and scale cell decide whether the GEMV takes it.
    Weight launch_shape{};
    launch_shape.n = out_rows; launch_shape.scale_ne[0] = k_per; launch_shape.scale_ne[1] = rows_per;
    // the reference activation: exact at the GEMV's widths, quantised per token per 128 past them
    std::vector<double> xr(x.begin(), x.end());
    if (ops::detail::fp8_block::quantizes_activations(launch_shape, c.tokens)) {
        for (std::int32_t t = 0; t < c.tokens; ++t) {
            for (std::int32_t b = 0; b < kb; ++b) {
                float amax = 0.0f;
                for (int i = 0; i < 128; ++i) { amax = std::fmax(amax, std::fabs(x[static_cast<std::size_t>(t) * c.k + b * 128 + i])); }
                const float scale = amax > 0.0f ? amax / 448.0f : 1.0f;
                for (int i = 0; i < 128; ++i) {
                    const std::size_t at = static_cast<std::size_t>(t) * c.k + b * 128 + i;
                    xr[at] = static_cast<double>(e4m3_to_float(float_to_e4m3(x[at] / scale))) * scale;
                }
            }
        }
    }
    std::vector<double> ref(static_cast<std::size_t>(out_rows) * c.tokens, 0.0);
    for (std::int32_t t = 0; t < c.tokens; ++t) {
        for (std::int32_t r = 0; r < out_rows; ++r) {
            const std::int32_t row = row_begin + r;
            double acc = 0.0;
            for (std::int32_t b = 0; b < kb; ++b) {
                double part = 0.0;
                for (int i = 0; i < 128; ++i) {
                    part += static_cast<double>(e4m3_to_float(codes[static_cast<std::size_t>(row) * c.k + b * 128 + i])) *
                            xr[static_cast<std::size_t>(t) * c.k + b * 128 + i];
                }
                acc += part * scales[static_cast<std::size_t>(row / rows_per) * scale_cols + (b * 128) / k_per];
            }
            ref[static_cast<std::size_t>(t) * out_rows + r] = acc;
        }
    }
    // device weight in the artifact's own layout: codes, then the scale grid 256-aligned
    const std::size_t scale_off = (codes.size() + 255) / 256 * 256;
    std::vector<std::uint8_t> payload(scale_off + scales.size() * 4);
    std::copy(codes.begin(), codes.end(), payload.begin());
    std::memcpy(payload.data() + scale_off, scales.data(), scales.size() * 4);
    void* d_payload = nullptr;
    CHECK_CUDA(cudaMalloc(&d_payload, payload.size()));
    CHECK_CUDA(cudaMemcpy(d_payload, payload.data(), payload.size(), cudaMemcpyHostToDevice));
    Weight w{};
    w.payload = d_payload; w.payload_bytes = payload.size(); w.qtype = c.per_row ? QType::FP8_E4M3FN_ROW_F32S : QType::FP8_E4M3FN_BLK128_F32S;
    w.layout = QuantLayout::Fp8Block128; w.group_size = 128; w.group = 128; w.ndim = 2;
    w.qdata = d_payload; w.scales = static_cast<std::uint8_t*>(d_payload) + scale_off; w.scale_dtype = DType::FP32;
    w.scale_ne[0] = k_per; w.scale_ne[1] = rows_per;
    w.n = c.rows; w.k = c.k; w.shape[0] = c.rows; w.shape[1] = c.k; w.padded_shape[0] = c.rows; w.padded_shape[1] = c.k;
    std::vector<__nv_bfloat16> hx(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { hx[i] = __float2bfloat16(x[i]); }
    __nv_bfloat16* d_x = nullptr; __nv_bfloat16* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, hx.size() * 2));
    CHECK_CUDA(cudaMalloc(&d_out, static_cast<std::size_t>(out_rows) * c.tokens * 2));
    CHECK_CUDA(cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    Tensor xt(d_x, DType::BF16, {c.k, c.tokens});
    Tensor out(d_out, DType::BF16, {out_rows, c.tokens});
    if (c.rows_view) {
        ops::linear_rows(xt, w, row_begin, out, nullptr, nullptr);
    } else {
        ops::linear(xt, w, out, nullptr);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> got(ref.size());
    CHECK_CUDA(cudaMemcpy(got.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost));
    double err = 0.0, norm = 0.0, max_abs = 0.0, max_ref = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = static_cast<double>(__bfloat162float(got[i])) - ref[i];
        err += d * d; norm += ref[i] * ref[i]; max_abs = std::fmax(max_abs, std::fabs(d)); max_ref = std::fmax(max_ref, std::fabs(ref[i]));
    }
    const double rel = std::sqrt(err / std::fmax(norm, 1e-30));
    bool ok = rel <= 6e-3 && max_abs <= 1.5e-2 * max_ref;
    // Whole matrices and row ranges share the same quantization for either scale grid.
    DeviceBuffer second(std::size_t(out_rows) * c.tokens * 2);
    Tensor second_out(second.p, DType::BF16, {out_rows, c.tokens});
    Weight companion = w;
    if (c.per_row) {
        // The row-scale payload also has enough entries for this smaller block grid.
        companion.qtype = QType::FP8_E4M3FN_BLK128_F32S;
        companion.scale_ne[0] = companion.scale_ne[1] = 128;
    }
    WorkspaceArena workspace(std::max(std::size_t{256}, ops::detail::fp8_block::workspace_bytes(c.k, c.tokens)));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (bool arena : {false, true}) {
        const auto grouped = [&] {
            ops::linear_projections(xt, {{w, out, ops::LinearPolicy::A16Only, c.rows_view ? row_begin : -1},
                                         {companion, second_out, ops::LinearPolicy::A16Only, row_begin}},
                                    arena ? &workspace : nullptr, stream);
        };
        grouped(); CHECK_CUDA(cudaStreamSynchronize(stream));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        grouped();
        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        std::size_t nodes = 0;
        CHECK_CUDA(cudaGraphGetNodes(graph, nullptr, &nodes));
        // Both run apart; a quantise first when either leaves the GEMV.
        Weight block_shape = launch_shape;
        block_shape.scale_ne[0] = block_shape.scale_ne[1] = 128;
        const bool planes = ops::detail::fp8_block::quantizes_activations(launch_shape, c.tokens) ||
                            ops::detail::fp8_block::quantizes_activations(block_shape, c.tokens);
        ok &= nodes == std::size_t(planes ? 3 : 2);
        for (int round = 0; round < 3; ++round) {
            for (auto& value : hx) { value = __float2bfloat16(ux(rng)); }
            CHECK_CUDA(cudaMemcpyAsync(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice, stream));
            ops::linear_rows(xt, w, row_begin, out, nullptr, stream);
            ops::linear_rows(xt, companion, row_begin, second_out, nullptr, stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            std::vector<std::uint16_t> expected(got.size()), expected_second(got.size()), actual(got.size());
            CHECK_CUDA(cudaMemcpy(expected.data(), d_out, got.size() * 2, cudaMemcpyDeviceToHost));
            second.copy_to_host(expected_second.data(), second.bytes);
            CHECK_CUDA(cudaGraphLaunch(exec, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaMemcpy(actual.data(), d_out, actual.size() * 2, cudaMemcpyDeviceToHost));
            ok &= actual == expected;
            second.copy_to_host(actual.data(), second.bytes);
            ok &= actual == expected_second;
        }
        CHECK_CUDA(cudaGraphExecDestroy(exec)); CHECK_CUDA(cudaGraphDestroy(graph));
    }
    // linear_add on the whole weight: residual += W . x in place, which on Hopper is the CUTLASS
    // kernel's residual epilogue (C = D). Checked as the delta it added against the reference.
    double add_rel = 0.0;
    if (!c.rows_view) {
        for (std::size_t i = 0; i < x.size(); ++i) { hx[i] = __float2bfloat16(x[i]); }
        CHECK_CUDA(cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        std::uniform_real_distribution<float> ur(-0.1f, 0.1f);
        std::vector<__nv_bfloat16> res0(ref.size());
        for (auto& v : res0) { v = __float2bfloat16(ur(rng)); }
        DeviceBuffer residual(res0.size() * 2);
        CHECK_CUDA(cudaMemcpy(residual.p, res0.data(), residual.bytes, cudaMemcpyHostToDevice));
        // A pageable cudaMemcpy may return before its DMA lands, and `stream` does not wait for
        // the legacy stream: finish the uploads before the launch reads them.
        CHECK_CUDA(cudaDeviceSynchronize());
        Tensor residual_t(residual.p, DType::BF16, {c.rows, c.tokens});
        ops::linear_add(xt, w, residual_t, workspace, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::vector<__nv_bfloat16> summed(res0.size());
        CHECK_CUDA(cudaMemcpy(summed.data(), residual.p, residual.bytes, cudaMemcpyDeviceToHost));
        double aerr = 0.0, anorm = 0.0, amax = 0.0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            const double delta = static_cast<double>(__bfloat162float(summed[i])) - __bfloat162float(res0[i]);
            const double d = delta - ref[i];
            aerr += d * d; anorm += ref[i] * ref[i]; amax = std::fmax(amax, std::fabs(d));
        }
        add_rel = std::sqrt(aerr / std::fmax(anorm, 1e-30));
        ok &= add_rel <= 8e-3 && amax <= 1.5e-2 * max_ref;
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
    std::printf("  [%5d, %5d] T=%-4d %-12s %-8s rel_l2=%.2e max_abs=%.2e of max add_rel=%.2e  %s\n", c.rows, c.k, c.tokens,
                c.rows_view ? "linear_rows" : "linear", c.per_row ? "per-row" : "blk128", rel, max_abs / max_ref,
                add_rel, ok ? "ok" : "FAIL");
    CHECK_CUDA(cudaFree(d_payload)); CHECK_CUDA(cudaFree(d_x)); CHECK_CUDA(cudaFree(d_out));
    return ok ? 0 : 1;
}
// Consecutive row ranges of one parent (q/k/v, q/k/gate/v) through linear_projections, which
// runs them as one launch: each output must hold exactly the rows the whole-parent linear writes,
// eagerly, replayed from a graph, with no arena and in arenas with and without the staging room.
int run_chain(std::vector<std::int32_t> parts, std::int32_t k, std::int32_t tokens) {
    std::int32_t rows = 0;
    for (auto r : parts) { rows += r; }
    std::mt19937 rng(static_cast<unsigned>(99 + rows + 7 * k + 13 * tokens));
    std::uniform_real_distribution<float> uw(-1.0f, 1.0f), us(0.5f, 2.0f), ux(-3.0f, 3.0f);
    const std::size_t scale_off = (static_cast<std::size_t>(rows) * k + 255) / 256 * 256;
    std::vector<std::uint8_t> payload(scale_off + static_cast<std::size_t>(rows / 128) * (k / 128) * 4);
    for (std::size_t i = 0; i < static_cast<std::size_t>(rows) * k; ++i) { payload[i] = float_to_e4m3(uw(rng)); }
    for (std::size_t i = 0; i < static_cast<std::size_t>(rows / 128) * (k / 128); ++i) {
        const float v = us(rng) * 0.01f;
        std::memcpy(payload.data() + scale_off + 4 * i, &v, 4);
    }
    DeviceBuffer d_payload(payload.size());
    d_payload.copy_from_host(payload.data(), payload.size());
    Weight w{};
    w.payload = d_payload.p; w.payload_bytes = payload.size(); w.qtype = QType::FP8_E4M3FN_BLK128_F32S;
    w.layout = QuantLayout::Fp8Block128; w.group_size = 128; w.group = 128; w.ndim = 2;
    w.qdata = d_payload.p; w.scales = static_cast<std::uint8_t*>(d_payload.p) + scale_off; w.scale_dtype = DType::FP32;
    w.scale_ne[0] = 128; w.scale_ne[1] = 128;
    w.n = rows; w.k = k; w.shape[0] = rows; w.shape[1] = k; w.padded_shape[0] = rows; w.padded_shape[1] = k;
    std::vector<__nv_bfloat16> hx(static_cast<std::size_t>(k) * tokens);
    for (auto& v : hx) { v = __float2bfloat16(ux(rng)); }
    DeviceBuffer d_x(hx.size() * 2), d_whole(static_cast<std::size_t>(rows) * tokens * 2);
    d_x.copy_from_host(hx.data(), d_x.bytes);
    Tensor xt(d_x.p, DType::BF16, {k, tokens});
    Tensor whole(d_whole.p, DType::BF16, {rows, tokens});
    ops::linear(xt, w, whole, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<std::uint16_t> want(static_cast<std::size_t>(rows) * tokens);
    d_whole.copy_to_host(want.data(), d_whole.bytes);

    std::vector<DeviceBuffer> outs;
    std::vector<Tensor> views;
    std::vector<ops::LinearProjection> projections;
    for (auto r : parts) { outs.emplace_back(static_cast<std::size_t>(r) * tokens * 2); }
    std::int32_t begin = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        views.emplace_back(outs[i].p, DType::BF16, std::initializer_list<std::int32_t>{parts[i], tokens});
    }
    for (std::size_t i = 0; i < parts.size(); ++i) {
        projections.push_back({w, views[i], ops::LinearPolicy::A16Only, begin});
        begin += parts[i];
    }
    // The chained launch's bits are the whole linear's: the same tile over the same rows.
    const auto matches = [&] {
        std::int32_t at = 0;
        bool same = true;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            std::vector<std::uint16_t> got(static_cast<std::size_t>(parts[i]) * tokens);
            outs[i].copy_to_host(got.data(), outs[i].bytes);
            for (std::int32_t t = 0; t < tokens; ++t) {
                for (std::int32_t r = 0; r < parts[i]; ++r) {
                    same &= got[static_cast<std::size_t>(t) * parts[i] + r] ==
                            want[static_cast<std::size_t>(t) * rows + at + r];
                }
            }
            at += parts[i];
        }
        return same;
    };
    const bool hopper  = ops::detail::fp8_block::sm90_gemm_available();
    const bool gemv    = !ops::detail::fp8_block::quantizes_activations(w, tokens);
    const bool chained = gemv || !hopper || tokens <= 128;
    // GEMV: one launch; the tile: quantize + one launch; Hopper: quantize + GEMM + split
    const std::size_t want_nodes = gemv ? 1 : !chained ? 1 + parts.size() : hopper ? 3 : 2;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    bool ok = true;
    const std::size_t roomy = std::max(std::size_t{256}, ops::detail::fp8_block::projections_workspace_capacity_bytes(rows, k, tokens));
    const std::size_t tight = std::max(std::size_t{256}, ops::detail::fp8_block::workspace_bytes(k, tokens));
    std::size_t nodes_seen = 0;
    for (int arena = 0; arena < 3; ++arena) {
        WorkspaceArena workspace(arena == 2 ? tight : roomy);
        for (auto& o : outs) { o.fill(0); }
        CHECK_CUDA(cudaDeviceSynchronize()); // the fills run on the legacy stream
        ops::linear_projections(xt, projections, arena == 0 ? nullptr : &workspace, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        ok &= matches();
        for (auto& o : outs) { o.fill(0); }
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        ops::linear_projections(xt, projections, arena == 0 ? nullptr : &workspace, stream);
        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        std::size_t nodes = 0;
        CHECK_CUDA(cudaGraphGetNodes(graph, nullptr, &nodes));
        // The tight arena has no staging room: Hopper's chained launch falls back to one GEMM
        // per range.
        const std::size_t expect = arena == 2 && hopper && !gemv ? 1 + parts.size() : want_nodes;
        ok &= nodes == expect;
        if (arena == 0) { nodes_seen = nodes; }
        CHECK_CUDA(cudaGraphLaunch(exec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        ok &= matches();
        CHECK_CUDA(cudaGraphExecDestroy(exec)); CHECK_CUDA(cudaGraphDestroy(graph));
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
    std::printf("  chain [");
    for (std::size_t i = 0; i < parts.size(); ++i) { std::printf(i ? " %d" : "%d", parts[i]); }
    std::printf("] k=%d T=%-4d nodes=%zu  %s\n", k, tokens, nodes_seen, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
// A random block-FP8 weight [rows, k] in the artifact's layout, its payload held by `storage`.
Weight random_weight(std::int32_t rows, std::int32_t k, std::mt19937& rng, DeviceBuffer& storage) {
    std::uniform_real_distribution<float> uw(-1.0f, 1.0f), us(0.5f, 2.0f);
    const std::size_t scale_off = (static_cast<std::size_t>(rows) * k + 255) / 256 * 256;
    const std::size_t cells     = static_cast<std::size_t>(rows / 128) * (k / 128);
    std::vector<std::uint8_t> payload(scale_off + cells * 4);
    for (std::size_t i = 0; i < static_cast<std::size_t>(rows) * k; ++i) { payload[i] = float_to_e4m3(uw(rng)); }
    for (std::size_t i = 0; i < cells; ++i) {
        const float v = us(rng) * 0.01f;
        std::memcpy(payload.data() + scale_off + 4 * i, &v, 4);
    }
    storage = DeviceBuffer(payload.size());
    storage.copy_from_host(payload.data(), payload.size());
    Weight w{};
    w.payload = storage.p; w.payload_bytes = payload.size(); w.qtype = QType::FP8_E4M3FN_BLK128_F32S;
    w.layout = QuantLayout::Fp8Block128; w.group_size = 128; w.group = 128; w.ndim = 2;
    w.qdata = storage.p; w.scales = static_cast<std::uint8_t*>(storage.p) + scale_off; w.scale_dtype = DType::FP32;
    w.scale_ne[0] = 128; w.scale_ne[1] = 128;
    w.n = rows; w.k = k; w.shape[0] = rows; w.shape[1] = k; w.padded_shape[0] = rows; w.padded_shape[1] = k;
    return w;
}

// The SwiGLU MLP with the activation quantised as it is formed (linear_swiglu_down_add) against
// the unfused pair -- gate/up linear, silu_mul, linear_add -- bit for bit, eagerly and replayed
// from a graph; at decode GEMV widths the fused route must decline.
int run_swiglu(std::int32_t intermediate, std::int32_t hidden, std::int32_t tokens, float limit) {
    std::mt19937 rng(static_cast<unsigned>(7 + intermediate + 3 * hidden + 11 * tokens));
    DeviceBuffer gate_up_storage(1), down_storage(1);
    const Weight gate_up = random_weight(2 * intermediate, hidden, rng, gate_up_storage);
    const Weight down    = random_weight(hidden, intermediate, rng, down_storage);
    std::uniform_real_distribution<float> ux(-3.0f, 3.0f), ur(-0.5f, 0.5f);
    std::vector<__nv_bfloat16> hx(static_cast<std::size_t>(hidden) * tokens), hres(hx.size());
    for (auto& v : hx) { v = __float2bfloat16(ux(rng)); }
    for (auto& v : hres) { v = __float2bfloat16(ur(rng)); }
    DeviceBuffer d_x(hx.size() * 2), d_want(hres.size() * 2), d_got(hres.size() * 2);
    DeviceBuffer d_packed(static_cast<std::size_t>(2) * intermediate * tokens * 2),
        d_act(static_cast<std::size_t>(intermediate) * tokens * 2);
    d_x.copy_from_host(hx.data(), d_x.bytes);
    d_want.copy_from_host(hres.data(), d_want.bytes);
    Tensor xt(d_x.p, DType::BF16, {hidden, tokens});
    Tensor want(d_want.p, DType::BF16, {hidden, tokens});
    Tensor got(d_got.p, DType::BF16, {hidden, tokens});
    Tensor packed(d_packed.p, DType::BF16, {2 * intermediate, tokens});
    Tensor act(d_act.p, DType::BF16, {intermediate, tokens});
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaDeviceSynchronize());
    const auto policy = ops::LinearPolicy::A16Only;
    WorkspaceArena workspace(std::size_t{64} << 20);
    ops::linear(xt, gate_up, packed, policy, workspace, stream);
    ops::silu_mul(packed.slice(0, 0, intermediate), packed.slice(0, intermediate, intermediate), act, limit, stream);
    ops::linear_add(act, down, want, policy, workspace, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<std::uint16_t> expected(hres.size()), actual(hres.size());
    d_want.copy_to_host(expected.data(), d_want.bytes);

    const bool admits = ops::linear_swiglu_down_add_admits(gate_up, down, policy, tokens);
    bool ok = admits == ops::detail::fp8_block::quantizes_activations(down, tokens);
    std::size_t nodes = 0;
    if (admits) {
        // The plan's own capacity, not the roomy arena: the route must fit what it declares.
        WorkspaceArena planned(ops::linear_swiglu_down_add_workspace_capacity_bytes(
            gate_up.qtype, intermediate, hidden, policy, tokens, tokens));
        d_got.copy_from_host(hres.data(), d_got.bytes);
        CHECK_CUDA(cudaDeviceSynchronize());
        ops::linear_swiglu_down_add(xt, gate_up, down, got, policy, limit, planned, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        d_got.copy_to_host(actual.data(), d_got.bytes);
        ok &= actual == expected;
        d_got.copy_from_host(hres.data(), d_got.bytes);
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        ops::linear_swiglu_down_add(xt, gate_up, down, got, policy, limit, planned, stream);
        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        CHECK_CUDA(cudaGraphGetNodes(graph, nullptr, &nodes));
        CHECK_CUDA(cudaGraphLaunch(exec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        d_got.copy_to_host(actual.data(), d_got.bytes);
        ok &= actual == expected;
        // gate/up: quantize + GEMM (or the GEMV); down: SwiGLU quantize + GEMM
        ok &= nodes == (ops::detail::fp8_block::quantizes_activations(gate_up, tokens) ? 4u : 3u);
        CHECK_CUDA(cudaGraphExecDestroy(exec)); CHECK_CUDA(cudaGraphDestroy(graph));
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
    std::printf("  swiglu down_add I=%d H=%d T=%-4d limit=%.1f %s nodes=%zu  %s\n", intermediate, hidden,
                tokens, limit, admits ? "fused " : "paired", nodes, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int failures = 0;
    for (const Case& c : {Case{256, 512, 1, false}, Case{256, 512, 3, false}, Case{256, 512, 4, true},
                          Case{256, 512, 5, false}, Case{384, 1024, 64, false}, Case{384, 1024, 200, true},
                          Case{512, 1024, 130, false}, Case{1024, 3584, 33, true},
                          // Hopper's CUTLASS tiles (fp8_block_sm90_gemm.cu), as an H100's 132 SMs
                          // pick them: the swapped 16-token tile at 5 tokens; the swapped 32-token one
                          // at 33 to 96, and the 64-token one at 100 over 4352 rows; 128 x 128 at
                          // 130 to 1027, and 256 x 128 at 300 over 5760 rows -- over activation
                          // scales padded to a multiple of four tokens.
                          Case{1024, 2048, 66, false}, Case{1536, 1024, 96, false},
                          Case{4352, 1024, 100, false}, Case{4352, 1024, 100, true},
                          Case{2048, 2048, 256, false}, Case{2048, 1024, 256, true},
                          Case{5760, 512, 300, false},
                          Case{2048, 2048, 573, false}, Case{1536, 1024, 1027, true},
                          Case{256, 512, 1, false, true}, Case{384, 1024, 64, true, true},
                          Case{512, 1024, 200, false, true},
                          // On Hopper, 3 and 4 tokens over 64 and 40 row blocks leave the GEMV
                          // for the narrow tile (gemv_serves); per-row scales keep the GEMV.
                          Case{8192, 1024, 3, false}, Case{5120, 2048, 4, false},
                          Case{10240, 1024, 4, true}, Case{5120, 1024, 4, false, true}}) {
        failures += run(c);
    }
    // q/k/v of a 4:1:1 head layout, q/k/gate/v of a gated one, and a qkv/z pair: decode GEMV
    // widths, Hopper's staged widths (narrow and swapped tiles), and wide rounds it runs apart.
    for (std::int32_t tokens : {1, 3, 20, 64, 100, 200}) {
        failures += run_chain({512, 128, 128}, 1024, tokens);
    }
    failures += run_chain({256, 128, 256, 128}, 512, 48);
    failures += run_chain({768, 256}, 512, 128);
    failures += run_chain({4096, 1024, 1024}, 1024, 4);
    // Decode GEMV widths keep the pair; the narrow, swapped and wide tiles take the fused route.
    for (std::int32_t tokens : {3, 5, 64, 100, 300}) { failures += run_swiglu(512, 256, tokens, 0.0f); }
    failures += run_swiglu(384, 512, 40, 1.5f);
    failures += run_swiglu(512, 5120, 4, 0.0f);
    std::printf("%s\n", failures == 0 ? "fp8 block: all cases ok" : "fp8 block: FAILURES");
    return failures == 0 ? 0 : 1;
}
