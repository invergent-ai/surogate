// Block-scaled FP8 linears against a double-precision reference: the decode GEMV on exact
// activations, and the tensor-core tile on activations quantised per token per 128 -- the
// reference quantises the same way, so what remains is accumulation order and BF16 output.
#include "api/ops/linear.h"
#include "ops/linear/fp8_block/fp8_block.h"
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
    // the reference activation: exact below the GEMV width, quantised per token per 128 above
    std::vector<double> xr(x.begin(), x.end());
    if (c.tokens > 4) {
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
    const std::int32_t row_begin = c.rows_view ? 128 * (c.rows / 256) : 0; // a 128-block boundary
    const std::int32_t out_rows  = c.rows - row_begin;
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
    const bool ok = rel <= 6e-3 && max_abs <= 1.5e-2 * max_ref;
    std::printf("  [%5d, %5d] T=%-4d %-12s %-8s rel_l2=%.2e max_abs=%.2e of max  %s\n", c.rows, c.k, c.tokens,
                c.rows_view ? "linear_rows" : "linear", c.per_row ? "per-row" : "blk128", rel, max_abs / max_ref, ok ? "ok" : "FAIL");
    CHECK_CUDA(cudaFree(d_payload)); CHECK_CUDA(cudaFree(d_x)); CHECK_CUDA(cudaFree(d_out));
    return ok ? 0 : 1;
}
} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int failures = 0;
    for (const Case& c : {Case{256, 512, 1, false}, Case{256, 512, 3, false}, Case{256, 512, 4, true},
                          Case{256, 512, 5, false}, Case{384, 1024, 64, false}, Case{384, 1024, 200, true},
                          Case{512, 1024, 130, false}, Case{1024, 3584, 33, true},
                          Case{256, 512, 1, false, true}, Case{384, 1024, 64, true, true},
                          Case{512, 1024, 200, false, true}}) {
        failures += run(c);
    }
    std::printf("%s\n", failures == 0 ? "fp8 block: all cases ok" : "fp8 block: FAILURES");
    return failures == 0 ? 0 : 1;
}
