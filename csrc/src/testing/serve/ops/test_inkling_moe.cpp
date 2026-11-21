// sinfer_inkling_moe_test: Inkling-2-Mini's mixture (api/ops/inkling_moe.h) against host ports of
// the reference semantics (rl/engine-inkling-2026-11-21/ref/inkling_ref.py, which follows the vLLM
// image's sm_120 kernels). Design and criteria: rl/engine-inkling-2026-11-21/lanes/d/DESIGN-moe.md 9.
//
//   quantisers (NVFP4 four-over-six, MXFP8): BIT-EXACT (every step is an IEEE _rn op or a cvt)
//   SiTU: within 1 bf16 ulp (the device runs the reference's PTX approximations, the host exact math)
//   router: logits within the fp32 bound, ids exact away from near-ties, weights rel 1e-5
//   MoE (synthetic weights, both formats): cosine / rel-L2 per token vs a host pipeline with fp64
//   dots; width invariance BIT-EXACT (T = 1 vs 24 vs 300 rounds, padding via live_tokens);
//   capture rows; expert LoRA: zero adapter bit-equal, base tokens bit-equal, delta vs host.
// Capture mode (`sinfer_inkling_moe_test --capture <moe_capture_v*.safetensors>`, written by
// lanes/d/scripts/extract_moe_capture.py from lane a's eager capture of the vLLM image's own kernels): every stage
// compared BIT FOR BIT against the reference GPU kernels (routing, x quantisation, SiTU + requantisation on the
// captured gate_up, the full op's y_q / down rows / output).
// Golden-vector mode (lane a's vectors/) is added once their format is published (M3).

#include "api/ops/inkling_moe.h"
#include "core/arena.h"
#include "ops/op_tester.h"

#include <nlohmann/json.hpp>

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr int H = 2304, E = 128, TOPK = 8, I = 1152, N = 2304;
constexpr float kInv1536  = 0x1.555556p-11F;
constexpr float kOneSixth = 0x1.555556p-3F;
constexpr float kInv448   = 0x1.24924Ap-9F;

// ---- small host numerics -------------------------------------------------------------------
std::uint8_t e4m3(float v) { return static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)); }
float e4m3_value(std::uint8_t c) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(c), __NV_E4M3)));
}
float e8m0_value(std::uint32_t e) {
    const std::uint32_t bits = e == 0 ? 0x00400000U : (e << 23);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
float bf(float v) { return bf16_to_f32(f32_to_bf16(v)); }
std::uint32_t bits_of(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
}

/// `ops/nvfp4.py` `_e2m1` (the reference's software path; the device uses the hardware cvt).
std::uint32_t e2m1(float x, float& value) {
    static const float kValues[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float a = std::fabs(x);
    const std::uint32_t code = a <= 0.25F ? 0U : a < 0.75F ? 1U : a <= 1.25F ? 2U : a < 1.75F ? 3U
                               : a <= 2.5F ? 4U : a < 3.5F ? 5U : a <= 5.0F ? 6U : 7U;
    const std::uint32_t sign = bits_of(x) >> 31;
    value = sign ? -kValues[code] : kValues[code];
    return code | (sign << 3);
}

struct Nvfp4Row {
    std::vector<std::uint8_t> codes, scales;
    std::vector<float> values; ///< e2m1 * block scale
    float amax = 0.0F;
};

/// `inkling_ref.inkling_quant_nvfp4_4o6` for one row.
Nvfp4Row quant_nvfp4(const float* x, int k) {
    Nvfp4Row r;
    r.codes.assign(k / 2, 0);
    r.scales.assign(k / 16, 0);
    r.values.assign(k, 0.0F);
    float amax = 0.0F;
    for (int i = 0; i < k; ++i) { amax = std::fmax(amax, std::fabs(x[i])); }
    r.amax          = amax;
    const float enc = std::fmin(1536.0F / amax, FLT_MAX);
    const float dec = 1.0F / enc;
    const float outer = amax * kInv1536;
    for (int b = 0; b < k / 16; ++b) {
        const float* xb = x + 16 * b;
        float bmax = 0.0F;
        for (int j = 0; j < 16; ++j) { bmax = std::fmax(bmax, std::fabs(xb[j])); }
        const std::uint8_t c6 = e4m3(std::fmin(bmax * (enc * kOneSixth), 448.0F));
        const std::uint8_t c4 = e4m3(std::fmin(bmax * (enc * 0.25F), 448.0F));
        const float s6 = e4m3_value(c6), s4 = e4m3_value(c4);
        const float inv6 = std::fmin(1.0F / (s6 * dec), FLT_MAX);
        const float inv4 = std::fmin(1.0F / (s4 * dec), FLT_MAX);
        std::uint32_t code6[16], code4[16];
        float v6[16], v4[16], err6[16], err4[16];
        for (int j = 0; j < 16; ++j) {
            code6[j] = e2m1(xb[j] * inv6, v6[j]);
            code4[j] = e2m1(xb[j] * inv4, v4[j]);
            err6[j]  = std::fabs((v6[j] * s6) * outer - xb[j]);
            err4[j]  = std::fabs((v4[j] * s4) * outer - xb[j]);
        }
        for (int w = 16; w > 1; w /= 2) {
            for (int j = 0; j < w / 2; ++j) {
                err6[j] = err6[2 * j] + err6[2 * j + 1];
                err4[j] = err4[2 * j] + err4[2 * j + 1];
            }
        }
        const bool four = err4[0] < err6[0];
        r.scales[b] = four ? c4 : c6;
        const float s = four ? s4 : s6;
        for (int j = 0; j < 16; ++j) {
            const std::uint32_t code = four ? code4[j] : code6[j];
            r.values[16 * b + j]     = (four ? v4[j] : v6[j]) * s;
            r.codes[8 * b + j / 2] |= static_cast<std::uint8_t>(code << (4 * (j & 1)));
        }
    }
    return r;
}

struct Mxfp8Row {
    std::vector<std::uint8_t> codes, exps;
    std::vector<float> values; ///< dequantised (bf16-exact)
};

/// `inkling_ref.inkling_quant_mxfp8` for one row.
Mxfp8Row quant_mxfp8(const float* x, int k) {
    Mxfp8Row r;
    r.codes.assign(k, 0);
    r.exps.assign(k / 32, 0);
    r.values.assign(k, 0.0F);
    for (int b = 0; b < k / 32; ++b) {
        float amax = 0.0F;
        for (int j = 0; j < 32; ++j) { amax = std::fmax(amax, std::fabs(x[32 * b + j])); }
        const float raw = amax * kInv448;
        std::int64_t exponent = ((static_cast<std::int64_t>(bits_of(raw)) + 0x007FFFFF) & 0x7F800000) >> 23;
        exponent = raw <= 0x1p-127F ? 0 : std::min<std::int64_t>(exponent, 254);
        float scale_bits_f;
        const std::uint32_t sb = static_cast<std::uint32_t>(exponent) << 23;
        std::memcpy(&scale_bits_f, &sb, 4);
        const float scale   = std::fmax(scale_bits_f, 0x1p-127F);
        const float inverse = exponent == 0 ? 0x1p127F : 1.0F / scale;
        r.exps[b] = static_cast<std::uint8_t>(exponent);
        for (int j = 0; j < 32; ++j) {
            const float v = std::fmin(std::fmax(x[32 * b + j] * inverse, -448.0F), 448.0F);
            const std::uint8_t c = e4m3(v);
            r.codes[32 * b + j]  = c;
            r.values[32 * b + j] = bf(e4m3_value(c) * e8m0_value(static_cast<std::uint32_t>(exponent)));
        }
    }
    return r;
}

/// The SiTU with exact transcendentals (the device's PTX approximations differ by < 1 bf16 ulp).
float situ_exact(float g, float u, float gamma) {
    const float sig = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(g))));
    const float gc  = static_cast<float>(4.0 * std::tanh(static_cast<double>(g) / 4.0));
    const float uc  = static_cast<float>(25.0 * std::tanh(static_cast<double>(u) / 25.0));
    return bf(((gc * sig) * uc) * gamma);
}

int bf16_ulps(float a, float b) {
    return std::abs(static_cast<int>(static_cast<std::int16_t>(f32_to_bf16(a))) -
                    static_cast<int>(static_cast<std::int16_t>(f32_to_bf16(b))));
}

struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ULL + 1) {}
    std::uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * static_cast<float>(next() >> 40) / 16777216.0F; }
};

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& v) {
    std::vector<std::uint16_t> b(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { b[i] = f32_to_bf16(v[i]); }
    return b;
}

int report(const char* label, int failures) {
    std::printf("%-58s %s\n", label, failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}

// ---- quantisers ----------------------------------------------------------------------------
std::vector<float> quant_rows(int rows, int k, std::uint64_t seed) {
    Rng rng(seed);
    std::vector<float> x(static_cast<std::size_t>(rows) * k);
    for (int r = 0; r < rows; ++r) {
        const float magnitude = std::pow(10.0F, rng.uniform(-3.0F, 2.0F));
        for (int c = 0; c < k; ++c) { x[static_cast<std::size_t>(r) * k + c] = magnitude * rng.uniform(-1.0F, 1.0F); }
    }
    // Edge rows: zero, one nonzero, tiny, huge, constant.
    std::fill(x.begin(), x.begin() + k, 0.0F);
    std::fill(x.begin() + k, x.begin() + 2 * k, 0.0F);
    x[k + 5] = -3.0F;
    for (int c = 0; c < k; ++c) {
        x[2 * static_cast<std::size_t>(k) + c] = 1e-30F * rng.uniform(-1.0F, 1.0F);
        x[3 * static_cast<std::size_t>(k) + c] = 1e30F * rng.uniform(-1.0F, 1.0F);
        x[4 * static_cast<std::size_t>(k) + c] = 0.75F;
    }
    round_to_bf16(x);
    return x;
}

int test_quant_nvfp4(int k) {
    const int rows = 41;
    const auto x   = quant_rows(rows, k, 11 + k);
    DeviceBuffer dx = to_device(to_bf16_bits(x));
    DeviceBuffer codes(static_cast<std::size_t>(rows) * k / 2), scales(static_cast<std::size_t>(rows) * k / 16),
        amax(rows * sizeof(float));
    Tensor tx(dx.p, DType::BF16, {k, rows}), tc(codes.p, DType::U8, {k / 2, rows}),
        ts(scales.p, DType::U8, {k / 16, rows}), ta(amax.p, DType::FP32, {rows});
    ops::inkling_quant_nvfp4_4o6(tx, tc, ts, ta, nullptr);
    cuda_synchronize();
    const auto gc = from_device<std::uint8_t>(codes, codes.bytes);
    const auto gs = from_device<std::uint8_t>(scales, scales.bytes);
    const auto ga = from_device<float>(amax, rows);
    int bad_codes = 0, bad_scales = 0, bad_amax = 0;
    for (int r = 0; r < rows; ++r) {
        const Nvfp4Row ref = quant_nvfp4(x.data() + static_cast<std::size_t>(r) * k, k);
        for (int i = 0; i < k / 2; ++i) { bad_codes += gc[static_cast<std::size_t>(r) * k / 2 + i] != ref.codes[i]; }
        for (int i = 0; i < k / 16; ++i) { bad_scales += gs[static_cast<std::size_t>(r) * k / 16 + i] != ref.scales[i]; }
        bad_amax += bits_of(ga[r]) != bits_of(ref.amax);
    }
    std::printf("  nvfp4 k=%d: code bytes off %d, scales off %d, amax off %d\n", k, bad_codes, bad_scales, bad_amax);
    char label[96];
    std::snprintf(label, sizeof label, "quant_nvfp4_4o6 k=%d bit-exact", k);
    return report(label, bad_codes + bad_scales + bad_amax);
}

int test_quant_mxfp8(int k) {
    const int rows = 41;
    const auto x   = quant_rows(rows, k, 23 + k);
    DeviceBuffer dx = to_device(to_bf16_bits(x));
    DeviceBuffer codes(static_cast<std::size_t>(rows) * k), exps(static_cast<std::size_t>(rows) * k / 32);
    Tensor tx(dx.p, DType::BF16, {k, rows}), tc(codes.p, DType::U8, {k, rows}), te(exps.p, DType::U8, {k / 32, rows});
    ops::inkling_quant_mxfp8(tx, tc, te, nullptr);
    cuda_synchronize();
    const auto gc = from_device<std::uint8_t>(codes, codes.bytes);
    const auto ge = from_device<std::uint8_t>(exps, exps.bytes);
    int bad_codes = 0, bad_exps = 0;
    for (int r = 0; r < rows; ++r) {
        const Mxfp8Row ref = quant_mxfp8(x.data() + static_cast<std::size_t>(r) * k, k);
        for (int i = 0; i < k; ++i) { bad_codes += gc[static_cast<std::size_t>(r) * k + i] != ref.codes[i]; }
        for (int i = 0; i < k / 32; ++i) { bad_exps += ge[static_cast<std::size_t>(r) * k / 32 + i] != ref.exps[i]; }
    }
    std::printf("  mxfp8 k=%d: codes off %d, exponents off %d\n", k, bad_codes, bad_exps);
    char label[96];
    std::snprintf(label, sizeof label, "quant_mxfp8 k=%d bit-exact", k);
    return report(label, bad_codes + bad_exps);
}

int test_softcap_swiglu() {
    const int rows = 64;
    Rng rng(5);
    std::vector<float> gu(static_cast<std::size_t>(rows) * 2 * I), gamma(rows);
    for (std::size_t i = 0; i < gu.size(); ++i) { gu[i] = (i & 1) ? rng.uniform(-40.0F, 40.0F) : rng.uniform(-9.0F, 9.0F); }
    for (auto& g : gamma) { g = rng.uniform(0.0F, 3.0F); }
    round_to_bf16(gu);
    DeviceBuffer dgu = to_device(to_bf16_bits(gu)), dg = to_device(gamma), dy(static_cast<std::size_t>(rows) * I * 2);
    Tensor tgu(dgu.p, DType::BF16, {2 * I, rows}), tg(dg.p, DType::FP32, {rows}), ty(dy.p, DType::BF16, {I, rows});
    ops::inkling_softcap_swiglu(tgu, tg, ty, nullptr);
    cuda_synchronize();
    const auto y = from_device<std::uint16_t>(dy, static_cast<std::size_t>(rows) * I);
    int over = 0, exact = 0;
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < I; ++j) {
            const float ref = situ_exact(gu[(static_cast<std::size_t>(r) * I + j) * 2], gu[(static_cast<std::size_t>(r) * I + j) * 2 + 1], gamma[r]);
            const float got = bf16_to_f32(y[static_cast<std::size_t>(r) * I + j]);
            const int ulps  = bf16_ulps(got, ref);
            exact += ulps == 0;
            over += ulps > 1;
        }
    }
    std::printf("  situ: exact %.4f, over 1 ulp %d of %d\n", static_cast<double>(exact) / (rows * I), over, rows * I);
    return report("softcap_swiglu within 1 bf16 ulp of exact math", over);
}

// ---- synthetic layer -----------------------------------------------------------------------
struct Layer {
    ops::InklingExpertFormat format;
    std::vector<float> router, bias;                 // bf16 values / fp32
    std::vector<std::uint8_t> codes[2], scales[2];   // w13, w2
    std::vector<float> amax[2];
    float den = 2688.0F;
    DeviceBuffer d_router, d_bias, d_codes[2], d_scales[2], d_amax[2], d_den;
    ops::InklingMoeWeights view{};
};

std::size_t nvfp4_scale_offset(std::size_t row, int group, int groups_per_row) {
    return ((row / 128) * (groups_per_row / 4) + group / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
}

void build_layer(Layer& l, ops::InklingExpertFormat format, std::uint64_t seed) {
    Rng rng(seed);
    l.format = format;
    l.router.resize(static_cast<std::size_t>(E) * H);
    for (auto& w : l.router) { w = rng.uniform(-0.05F, 0.05F); }
    round_to_bf16(l.router);
    l.bias.resize(E);
    for (auto& b : l.bias) { b = rng.uniform(-0.02F, 0.02F); }
    const int ks[2] = {H, I};
    for (int p = 0; p < 2; ++p) {
        const std::size_t rows = static_cast<std::size_t>(E) * N;
        const int k = ks[p];
        if (format == ops::InklingExpertFormat::Nvfp4) {
            l.codes[p].resize(rows * k / 2);
            for (auto& c : l.codes[p]) { c = static_cast<std::uint8_t>(rng.next()); }
            l.scales[p].resize(rows * k / 16);
            for (std::size_t r = 0; r < rows; ++r) {
                for (int g = 0; g < k / 16; ++g) {
                    l.scales[p][nvfp4_scale_offset(r, g, k / 16)] = static_cast<std::uint8_t>(0x38 + rng.next() % 0x20);
                }
            }
            l.amax[p].resize(E);
            for (auto& a : l.amax[p]) { a = rng.uniform(0.3F, 0.7F); }
        } else {
            l.codes[p].resize(rows * k);
            for (auto& c : l.codes[p]) {
                c = static_cast<std::uint8_t>((rng.next() % 0x70) | ((rng.next() & 1) << 7));
            }
            l.scales[p].resize(rows * k / 32);
            for (auto& s : l.scales[p]) { s = static_cast<std::uint8_t>(113 + rng.next() % 6); }
        }
    }
    l.d_router = to_device(to_bf16_bits(l.router));
    l.d_bias   = to_device(l.bias);
    l.view.format      = format;
    l.view.router      = l.d_router.p;
    l.view.router_bias = static_cast<const float*>(l.d_bias.p);
    ops::InklingExpertBank* banks[2] = {&l.view.w13, &l.view.w2};
    for (int p = 0; p < 2; ++p) {
        l.d_codes[p]  = to_device(l.codes[p]);
        l.d_scales[p] = to_device(l.scales[p]);
        banks[p]->codes  = static_cast<const std::uint8_t*>(l.d_codes[p].p);
        banks[p]->scales = static_cast<const std::uint8_t*>(l.d_scales[p].p);
        if (format == ops::InklingExpertFormat::Nvfp4) {
            l.d_amax[p] = to_device(l.amax[p]);
            banks[p]->amax = static_cast<const float*>(l.d_amax[p].p);
        }
    }
    if (format == ops::InklingExpertFormat::Nvfp4) {
        l.d_den = to_device(std::vector<float>{l.den});
        l.view.w13.outer_den = static_cast<const float*>(l.d_den.p);
        l.view.w2.outer_den  = static_cast<const float*>(l.d_den.p);
    }
}

/// Unit weight (NVFP4: e2m1 * block scale, without amax / den; MXFP8: the bf16-exact value).
void decode_expert(const Layer& l, int p, int e, std::vector<float>& w) {
    const int k = p == 0 ? H : I;
    w.resize(static_cast<std::size_t>(N) * k);
    for (int r = 0; r < N; ++r) {
        const std::size_t row = static_cast<std::size_t>(e) * N + r;
        for (int c = 0; c < k; ++c) {
            float v;
            if (l.format == ops::InklingExpertFormat::Nvfp4) {
                static const float kValues[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
                const std::uint8_t byte = l.codes[p][row * k / 2 + c / 2];
                const std::uint32_t code = (c & 1) ? (byte >> 4) : (byte & 15);
                v = (code & 8 ? -1.0F : 1.0F) * kValues[code & 7] *
                    e4m3_value(l.scales[p][nvfp4_scale_offset(row, c / 16, k / 16)]);
            } else {
                v = bf(e4m3_value(l.codes[p][row * k + c]) * e8m0_value(l.scales[p][row * (k / 32) + c / 32]));
            }
            w[static_cast<std::size_t>(r) * k + c] = v;
        }
    }
}

/// Host adapter: A [E][slots][rank][k], B [E][slots][n][rank] per port (gate, up, down).
struct Adapter {
    int rank = 16, slots = 2;
    std::vector<float> a[3], b[3];
    DeviceBuffer d_a[3], d_b[3], d_table;
};

void build_adapter(Adapter& ad, std::uint64_t seed, bool zero_b) {
    Rng rng(seed);
    const int ks[3] = {H, H, I}, ns[3] = {I, I, H};
    std::vector<ops::LoraBank> table(2 + 3 * E);
    for (int p = 0; p < 3; ++p) {
        ad.a[p].resize(static_cast<std::size_t>(E) * ad.slots * ad.rank * ks[p]);
        ad.b[p].resize(static_cast<std::size_t>(E) * ad.slots * ns[p] * ad.rank);
        for (auto& v : ad.a[p]) { v = rng.uniform(-0.02F, 0.02F); }
        for (auto& v : ad.b[p]) { v = zero_b ? 0.0F : rng.uniform(-0.02F, 0.02F); }
        round_to_bf16(ad.a[p]);
        round_to_bf16(ad.b[p]);
        ad.d_a[p] = to_device(to_bf16_bits(ad.a[p]));
        ad.d_b[p] = to_device(to_bf16_bits(ad.b[p]));
        for (int e = 0; e < E; ++e) {
            ops::LoraBank bank;
            bank.a = static_cast<const std::uint16_t*>(ad.d_a[p].p) + static_cast<std::size_t>(e) * ad.slots * ad.rank * ks[p];
            bank.b = static_cast<const std::uint16_t*>(ad.d_b[p].p) + static_cast<std::size_t>(e) * ad.slots * ns[p] * ad.rank;
            bank.a_stride = static_cast<std::int64_t>(ad.rank) * ks[p];
            bank.b_stride = static_cast<std::int64_t>(ns[p]) * ad.rank;
            bank.rank = ad.rank;
            bank.n    = ns[p];
            bank.k    = ks[p];
            table[2 + 3 * e + p] = bank;
        }
    }
    ad.d_table = to_device(table);
}

/// Host pipeline for one token given its routing (ids / gammas in slot order).
std::vector<float> host_moe_token(const Layer& l, const std::vector<std::vector<float>>& w13,
                                  const std::vector<std::vector<float>>& w2, const float* x, const int* ids,
                                  const float* gammas, const Adapter* ad, int adapter_slot) {
    const bool nv = l.format == ops::InklingExpertFormat::Nvfp4;
    std::vector<float> xv(H);
    float x_amax = 0.0F;
    if (nv) {
        const Nvfp4Row q = quant_nvfp4(x, H);
        xv = q.values;
        x_amax = q.amax;
    } else {
        xv = quant_mxfp8(x, H).values;
    }
    std::vector<double> acc(H, 0.0);
    for (int s = 0; s < TOPK; ++s) {
        const int e = ids[s];
        const std::vector<float>& W13 = w13[e];
        const std::vector<float>& W2  = w2[e];
        std::vector<float> lg(16), lu(16);
        if (ad != nullptr && adapter_slot >= 0) {
            for (int r = 0; r < ad->rank; ++r) {
                double sg = 0.0, su = 0.0;
                const std::size_t base = ((static_cast<std::size_t>(e) * ad->slots + adapter_slot) * ad->rank + r) * H;
                for (int k = 0; k < H; ++k) { sg += double(ad->a[0][base + k]) * x[k]; su += double(ad->a[1][base + k]) * x[k]; }
                lg[r] = static_cast<float>(sg);
                lu[r] = static_cast<float>(su);
            }
        }
        const float rcp13 = 1.0F / (1536.0F * l.den);
        const float rs13  = nv ? (x_amax * l.amax[0][e]) * rcp13 : 1.0F;
        std::vector<float> y(I);
        for (int j = 0; j < I; ++j) {
            float gu[2];
            for (int h = 0; h < 2; ++h) {
                double d = 0.0;
                const float* row = W13.data() + static_cast<std::size_t>(2 * j + h) * H;
                for (int k = 0; k < H; ++k) { d += double(xv[k]) * row[k]; }
                float v = nv ? static_cast<float>(d) * rs13 : static_cast<float>(d);
                if (ad != nullptr && adapter_slot >= 0) {
                    double delta = 0.0;
                    const std::size_t bb = ((static_cast<std::size_t>(e) * ad->slots + adapter_slot) * I + j) * ad->rank;
                    for (int r = 0; r < ad->rank; ++r) { delta += double(ad->b[h][bb + r]) * (h == 0 ? lg[r] : lu[r]); }
                    v += static_cast<float>(delta);
                }
                gu[h] = bf(v);
            }
            y[j] = situ_exact(gu[0], gu[1], gammas[s]);
        }
        std::vector<float> yv;
        float y_amax = 0.0F;
        if (nv) {
            const Nvfp4Row q = quant_nvfp4(y.data(), I);
            yv = q.values;
            y_amax = q.amax;
        } else {
            yv = quant_mxfp8(y.data(), I).values;
        }
        std::vector<float> ld(16);
        if (ad != nullptr && adapter_slot >= 0) {
            for (int r = 0; r < ad->rank; ++r) {
                double sd = 0.0;
                const std::size_t base = ((static_cast<std::size_t>(e) * ad->slots + adapter_slot) * ad->rank + r) * I;
                for (int k = 0; k < I; ++k) { sd += double(ad->a[2][base + k]) * y[k]; }
                ld[r] = static_cast<float>(sd);
            }
        }
        const float rs2 = nv ? (y_amax * l.amax[1][e]) * (1.0F / (1536.0F * l.den)) : 1.0F;
        for (int n = 0; n < H; ++n) {
            double d = 0.0;
            const float* row = W2.data() + static_cast<std::size_t>(n) * I;
            for (int k = 0; k < I; ++k) { d += double(yv[k]) * row[k]; }
            float v = nv ? static_cast<float>(d) * rs2 : static_cast<float>(d);
            if (ad != nullptr && adapter_slot >= 0) {
                double delta = 0.0;
                const std::size_t bb = ((static_cast<std::size_t>(e) * ad->slots + adapter_slot) * H + n) * ad->rank;
                for (int r = 0; r < ad->rank; ++r) { delta += double(ad->b[2][bb + r]) * ld[r]; }
                v += static_cast<float>(delta);
            }
            acc[n] = static_cast<float>(acc[n] + bf(v));
        }
    }
    std::vector<float> out(H);
    for (int n = 0; n < H; ++n) { out[n] = bf(static_cast<float>(acc[n])); }
    return out;
}

struct DeviceRun {
    std::vector<std::uint16_t> out;
    std::vector<int> capture;
};

DeviceRun run_moe(const Layer& l, const std::vector<float>& x, int tokens, const int* live = nullptr,
                  const ops::InklingMoeAdapters* adapters = nullptr, bool capture = false) {
    DeviceBuffer dx = to_device(to_bf16_bits(std::vector<float>(x.begin(), x.begin() + static_cast<std::size_t>(tokens) * H)));
    DeviceBuffer dout(static_cast<std::size_t>(tokens) * H * 2);
    DeviceArena arena(ops::inkling_moe_workspace_bytes(l.format, tokens, adapters != nullptr));
    Tensor tx(dx.p, DType::BF16, {H, tokens}), tout(dout.p, DType::BF16, {H, tokens});
    ops::InklingMoeOptions options;
    DeviceBuffer dlive, dcap;
    if (live != nullptr) {
        dlive = to_device(std::vector<int>{*live});
        options.live_tokens = static_cast<const std::int32_t*>(dlive.p);
    }
    ops::InklingMoeCapture cap;
    constexpr int kStride = 16, kFirst = 3;
    if (capture) {
        dcap = to_device(std::vector<int>((tokens + kFirst) * kStride, -7));
        cap.ids = static_cast<std::int32_t*>(dcap.p);
        cap.capacity_tokens = tokens + kFirst;
        cap.first_token = kFirst;
        cap.token_stride = kStride;
        options.capture = &cap;
    }
    options.adapters = adapters;
    ops::inkling_moe(tx, l.view, tout, arena, nullptr, options);
    cuda_synchronize();
    DeviceRun run;
    run.out = from_device<std::uint16_t>(dout, static_cast<std::size_t>(tokens) * H);
    if (capture) { run.capture = from_device<int>(dcap, static_cast<std::size_t>(tokens + kFirst) * kStride); }
    return run;
}

void route_on_device(const Layer& l, const std::vector<float>& x, int tokens, std::vector<int>& ids,
                     std::vector<float>& gammas, std::vector<float>& logits) {
    DeviceBuffer dx = to_device(to_bf16_bits(std::vector<float>(x.begin(), x.begin() + static_cast<std::size_t>(tokens) * H)));
    DeviceBuffer dl(static_cast<std::size_t>(tokens) * E * 4), di(static_cast<std::size_t>(tokens) * TOPK * 4),
        dw(static_cast<std::size_t>(tokens) * TOPK * 4);
    Tensor tx(dx.p, DType::BF16, {H, tokens}), tl(dl.p, DType::FP32, {E, tokens}), ti(di.p, DType::I32, {TOPK, tokens}),
        tw(dw.p, DType::FP32, {TOPK, tokens});
    ops::inkling_route(tx, l.view, tl, ti, tw, nullptr);
    cuda_synchronize();
    ids    = from_device<int>(di, static_cast<std::size_t>(tokens) * TOPK);
    gammas = from_device<float>(dw, static_cast<std::size_t>(tokens) * TOPK);
    logits = from_device<float>(dl, static_cast<std::size_t>(tokens) * E);
}

int test_route(const Layer& l, const std::vector<float>& x, int tokens) {
    std::vector<int> ids;
    std::vector<float> gammas, logits;
    route_on_device(l, x, tokens, ids, gammas, logits);
    int bad_logits = 0, bad_ids = 0, near_ties = 0, bad_weights = 0;
    for (int t = 0; t < tokens; ++t) {
        const float* xt = x.data() + static_cast<std::size_t>(t) * H;
        for (int e = 0; e < E; ++e) {
            double dot = 0.0, mag = 0.0;
            for (int k = 0; k < H; ++k) {
                dot += double(xt[k]) * l.router[static_cast<std::size_t>(e) * H + k];
                mag += std::fabs(double(xt[k]) * l.router[static_cast<std::size_t>(e) * H + k]);
            }
            // fp32 chains of up to 64 MMAs per K group: a few roundings of 2^-24 each, relative to sum |x w|.
            bad_logits += std::fabs(logits[static_cast<std::size_t>(t) * E + e] - dot) > mag * 0x1p-17 + 1e-30;
        }
        // Selection from the device's own logits (host fp32 sigmoid; near-ties excused).
        std::vector<std::pair<float, int>> sel(E);
        for (int e = 0; e < E; ++e) {
            const float lg = logits[static_cast<std::size_t>(t) * E + e];
            sel[e] = {1.0F / (1.0F + std::exp(-lg)) + l.bias[e], e};
        }
        std::stable_sort(sel.begin(), sel.end(), [](auto a, auto b) { return a.first > b.first; });
        if (sel[7].first - sel[8].first < 1e-6F) { ++near_ties; continue; }
        double lp[8], m = -1e300;
        for (int s = 0; s < TOPK; ++s) {
            bad_ids += ids[static_cast<std::size_t>(t) * TOPK + s] != sel[s].second;
            const double lg = logits[static_cast<std::size_t>(t) * E + sel[s].second];
            lp[s] = std::min(lg, 0.0) - std::log1p(std::exp(-std::fabs(lg)));
            m = std::max(m, lp[s]);
        }
        double total = 0.0;
        for (double v : lp) { total += std::exp(v - m); }
        for (int s = 0; s < TOPK; ++s) {
            const double w = std::exp(lp[s] - (m + std::log(total))) * 8.0;
            bad_weights += std::fabs(gammas[static_cast<std::size_t>(t) * TOPK + s] - w) > 1e-5 * w;
        }
    }
    std::printf("  route T=%d: logits off %d, ids off %d, weights off %d, near-ties skipped %d\n", tokens, bad_logits,
                bad_ids, bad_weights, near_ties);
    return report("inkling_route vs host", bad_logits + bad_ids + bad_weights);
}

int test_moe(ops::InklingExpertFormat format) {
    const bool nv = format == ops::InklingExpertFormat::Nvfp4;
    std::printf("[%s layer]\n", nv ? "NVFP4" : "MXFP8");
    Layer l;
    build_layer(l, format, nv ? 101 : 202);
    constexpr int kT = 24, kWide = 300;
    Rng rng(nv ? 7 : 8);
    std::vector<float> x(static_cast<std::size_t>(kWide) * H);
    for (auto& v : x) { v = rng.uniform(-2.0F, 2.0F); }
    round_to_bf16(x);
    int failures = 0;
    failures += test_route(l, x, kT);

    // Round of 24 (narrow) with capture.
    const DeviceRun base = run_moe(l, x, kT, nullptr, nullptr, true);
    std::vector<int> ids;
    std::vector<float> gammas, logits;
    route_on_device(l, x, kT, ids, gammas, logits);
    int bad_capture = 0;
    for (int t = 0; t < kT; ++t) {
        for (int s = 0; s < TOPK; ++s) { bad_capture += base.capture[(t + 3) * 16 + s] != ids[t * TOPK + s]; }
        for (int s = TOPK; s < 16; ++s) { bad_capture += base.capture[(t + 3) * 16 + s] != -7; }
    }
    for (int s = 0; s < 3 * 16; ++s) { bad_capture += base.capture[s] != -7; }
    failures += report("capture row = routing ids (stride / first / capacity)", bad_capture);

    // Host reference on the 24 tokens.
    std::vector<std::vector<float>> w13(E), w2(E);
    std::vector<bool> used(E, false);
    for (int i : ids) { used[i] = true; }
    for (int e = 0; e < E; ++e) {
        if (used[e]) { decode_expert(l, 0, e, w13[e]); decode_expert(l, 1, e, w2[e]); }
    }
    double worst_cos = 1.0, worst_rel = 0.0;
    for (int t = 0; t < kT; ++t) {
        const auto ref = host_moe_token(l, w13, w2, x.data() + static_cast<std::size_t>(t) * H, ids.data() + t * TOPK,
                                        gammas.data() + t * TOPK, nullptr, -1);
        double dot = 0, nr = 0, ng = 0, diff = 0;
        for (int n = 0; n < H; ++n) {
            const double g = bf16_to_f32(base.out[static_cast<std::size_t>(t) * H + n]);
            dot += g * ref[n]; nr += double(ref[n]) * ref[n]; ng += g * g; diff += (g - ref[n]) * (g - ref[n]);
        }
        worst_cos = std::min(worst_cos, dot / std::sqrt(nr * ng + 1e-300));
        worst_rel = std::max(worst_rel, std::sqrt(diff / (nr + 1e-300)));
    }
    std::printf("  moe T=%d vs host: worst cosine %.6f, worst rel-L2 %.3e\n", kT, worst_cos, worst_rel);
    failures += report("inkling_moe vs host pipeline (cos >= 0.9999, rel-L2 <= 1e-2)",
                       (worst_cos >= 0.9999 && worst_rel <= 1e-2) ? 0 : 1);

    // Width invariance: each token alone, and inside a wide round, and with padding.
    int width_bad = 0;
    for (int t = 0; t < kT; t += 5) {
        std::vector<float> one(x.begin() + static_cast<std::size_t>(t) * H, x.begin() + static_cast<std::size_t>(t + 1) * H);
        const DeviceRun single = run_moe(l, one, 1);
        width_bad += !std::equal(single.out.begin(), single.out.end(), base.out.begin() + static_cast<std::size_t>(t) * H);
    }
    const DeviceRun wide = run_moe(l, x, kWide);
    width_bad += !std::equal(base.out.begin(), base.out.end(), wide.out.begin());
    const int live = kT;
    std::vector<float> padded(x.begin(), x.begin() + static_cast<std::size_t>(32) * H);
    const DeviceRun pad = run_moe(l, padded, 32, &live);
    width_bad += !std::equal(base.out.begin(), base.out.end(), pad.out.begin());
    for (std::size_t i = static_cast<std::size_t>(kT) * H; i < pad.out.size(); ++i) { width_bad += pad.out[i] != 0; }
    failures += report("width invariance T=1 / 24 / 300 / padded 32 (bit-exact)", width_bad);

    // Expert LoRA: zero adapter neutral; base tokens untouched; delta vs host.
    Adapter zero, adapter;
    build_adapter(zero, 31, true);
    build_adapter(adapter, 37, false);
    std::vector<int> slots(kT);
    for (int t = 0; t < kT; ++t) { slots[t] = t % 3 == 2 ? -1 : t % 2; }
    DeviceBuffer dslots = to_device(slots);
    ops::InklingMoeAdapters az{static_cast<const ops::LoraBank*>(zero.d_table.p), static_cast<const std::int32_t*>(dslots.p), nullptr, 16};
    ops::InklingMoeAdapters aa{static_cast<const ops::LoraBank*>(adapter.d_table.p), static_cast<const std::int32_t*>(dslots.p), nullptr, 16};
    const DeviceRun zero_run = run_moe(l, x, kT, nullptr, &az);
    failures += report("LoRA: zero adapter output == base (bit-exact)", zero_run.out == base.out ? 0 : 1);
    const DeviceRun lora_run = run_moe(l, x, kT, nullptr, &aa);
    int base_tokens_bad = 0;
    double worst_delta_cos = 1.0;
    for (int t = 0; t < kT; ++t) {
        const auto begin = static_cast<std::size_t>(t) * H;
        if (slots[t] < 0) {
            base_tokens_bad += !std::equal(lora_run.out.begin() + begin, lora_run.out.begin() + begin + H, base.out.begin() + begin);
            continue;
        }
        const auto ref_base = host_moe_token(l, w13, w2, x.data() + begin, ids.data() + t * TOPK, gammas.data() + t * TOPK, nullptr, -1);
        const auto ref_lora = host_moe_token(l, w13, w2, x.data() + begin, ids.data() + t * TOPK, gammas.data() + t * TOPK, &adapter, slots[t]);
        double dot = 0, na = 0, nb = 0;
        for (int n = 0; n < H; ++n) {
            const double dg = double(bf16_to_f32(lora_run.out[begin + n])) - bf16_to_f32(base.out[begin + n]);
            const double dr = double(ref_lora[n]) - ref_base[n];
            dot += dg * dr; na += dg * dg; nb += dr * dr;
        }
        worst_delta_cos = std::min(worst_delta_cos, dot / std::sqrt(na * nb + 1e-300));
    }
    std::printf("  lora: base tokens changed %d, worst delta cosine %.5f\n", base_tokens_bad, worst_delta_cos);
    failures += report("LoRA: base tokens bit-exact in an adapter round", base_tokens_bad);
    failures += report("LoRA: delta vs host (cos >= 0.999)", worst_delta_cos >= 0.999 ? 0 : 1);
    return failures;
}


// ---- capture mode: bit test against the vLLM image's kernels -----------------------------------
struct SafeTensors {
    std::vector<char> data;
    std::map<std::string, std::pair<std::size_t, std::size_t>> spans; // name -> (offset, bytes)
    std::map<std::string, std::string> meta;
    const void* get(const std::string& name, std::size_t* bytes = nullptr) const {
        const auto it = spans.find(name);
        if (it == spans.end()) { return nullptr; }
        if (bytes != nullptr) { *bytes = it->second.second; }
        return data.data() + it->second.first;
    }
    template <class T>
    std::vector<T> vec(const std::string& name) const {
        std::size_t bytes = 0;
        const void* p = get(name, &bytes);
        if (p == nullptr) { throw std::runtime_error("capture file lacks " + name); }
        std::vector<T> out(bytes / sizeof(T));
        std::memcpy(out.data(), p, bytes);
        return out;
    }
};

SafeTensors read_safetensors(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error(std::string("cannot open ") + path); }
    std::uint64_t header_bytes = 0;
    in.read(reinterpret_cast<char*>(&header_bytes), 8);
    std::string header(header_bytes, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_bytes));
    const auto json = nlohmann::json::parse(header);
    SafeTensors st;
    in.seekg(0, std::ios::end);
    const std::size_t total = static_cast<std::size_t>(in.tellg()) - 8 - header_bytes;
    in.seekg(static_cast<std::streamoff>(8 + header_bytes));
    st.data.resize(total);
    in.read(st.data.data(), static_cast<std::streamsize>(total));
    for (const auto& [name, value] : json.items()) {
        if (name == "__metadata__") {
            for (const auto& [k, v] : value.items()) { st.meta[k] = v.get<std::string>(); }
            continue;
        }
        const auto begin = value["data_offsets"][0].get<std::size_t>();
        const auto end   = value["data_offsets"][1].get<std::size_t>();
        st.spans[name]   = {begin, end - begin};
    }
    return st;
}

template <class T>
int count_mismatch(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) { return static_cast<int>(std::max(a.size(), b.size())); }
    int bad = 0;
    for (std::size_t i = 0; i < a.size(); ++i) { bad += std::memcmp(&a[i], &b[i], sizeof(T)) != 0; }
    return bad;
}

int run_capture(const char* path) {
    const SafeTensors st = read_safetensors(path);
    const bool nv   = st.meta.at("format") == "nvfp4";
    const int tokens = std::stoi(st.meta.at("tokens"));
    std::printf("[capture %s: layer %s visit %s, %s, T=%d]\n", path, st.meta.at("layer").c_str(),
                st.meta.at("visit").c_str(), nv ? "NVFP4" : "MXFP8", tokens);
    const auto format = nv ? ops::InklingExpertFormat::Nvfp4 : ops::InklingExpertFormat::Mxfp8;
    Layer l;
    l.format = format;
    l.d_router = to_device(st.vec<std::uint16_t>("router"));
    l.d_bias   = to_device(st.vec<float>("router_bias"));
    l.view.format      = format;
    l.view.router      = l.d_router.p;
    l.view.router_bias = static_cast<const float*>(l.d_bias.p);
    const char* names[2] = {"w13", "w2"};
    const int ks[2]      = {H, I};
    ops::InklingExpertBank* banks[2] = {&l.view.w13, &l.view.w2};
    for (int p = 0; p < 2; ++p) {
        l.codes[p] = st.vec<std::uint8_t>(std::string(names[p]) + "_codes");
        auto scales = st.vec<std::uint8_t>(std::string(names[p]) + "_scales");
        if (nv) {
            // checkpoint row-major [E * N][K/16] -> blockscale-k16-m128x4 (the artifact layout)
            const int groups = ks[p] / 16;
            l.scales[p].assign(scales.size(), 0);
            for (std::size_t r = 0; r < static_cast<std::size_t>(E) * N; ++r) {
                for (int g = 0; g < groups; ++g) { l.scales[p][nvfp4_scale_offset(r, g, groups)] = scales[r * groups + g]; }
            }
            l.amax[p] = st.vec<float>(std::string(names[p]) + "_amax");
            l.den     = st.vec<float>(std::string(names[p]) + "_den")[0];
            l.d_amax[p] = to_device(l.amax[p]);
            banks[p]->amax = static_cast<const float*>(l.d_amax[p].p);
        } else {
            l.scales[p] = std::move(scales);
        }
        l.d_codes[p]  = to_device(l.codes[p]);
        l.d_scales[p] = to_device(l.scales[p]);
        banks[p]->codes  = static_cast<const std::uint8_t*>(l.d_codes[p].p);
        banks[p]->scales = static_cast<const std::uint8_t*>(l.d_scales[p].p);
    }
    if (nv) {
        l.d_den = to_device(std::vector<float>{l.den});
        l.view.w13.outer_den = static_cast<const float*>(l.d_den.p);
        l.view.w2.outer_den  = static_cast<const float*>(l.d_den.p);
    }
    int failures = 0;
    const auto x_bits = st.vec<std::uint16_t>("x");
    DeviceBuffer dx = to_device(x_bits);
    Tensor tx(dx.p, DType::BF16, {H, tokens});
    const auto ref_ids    = st.vec<int>("ids");
    const auto ref_gammas = st.vec<float>("gammas");

    // 1. routing
    {
        DeviceBuffer dl(static_cast<std::size_t>(tokens) * E * 4), di(static_cast<std::size_t>(tokens) * TOPK * 4),
            dw(static_cast<std::size_t>(tokens) * TOPK * 4);
        Tensor tl(dl.p, DType::FP32, {E, tokens}), ti(di.p, DType::I32, {TOPK, tokens}), tw(dw.p, DType::FP32, {TOPK, tokens});
        ops::inkling_route(tx, l.view, tl, ti, tw, nullptr);
        cuda_synchronize();
        const int bad_ids = count_mismatch(from_device<int>(di, ref_ids.size()), ref_ids);
        const int bad_w   = count_mismatch(from_device<float>(dw, ref_gammas.size()), ref_gammas);
        std::printf("  route: ids off %d, gammas off %d (of %zu)\n", bad_ids, bad_w, ref_ids.size());
        failures += report("capture: routing ids + weights bit-exact", bad_ids + bad_w);
    }
    // 2. x quantisation (+ the NVFP4 gate_up row scale)
    {
        const auto ref_codes  = st.vec<std::uint8_t>("x_codes");
        const auto ref_scales = st.vec<std::uint8_t>("x_scales");
        int bad = 0;
        if (nv) {
            DeviceBuffer c(static_cast<std::size_t>(tokens) * H / 2), s(static_cast<std::size_t>(tokens) * H / 16), a(tokens * 4);
            Tensor tc(c.p, DType::U8, {H / 2, tokens}), ts(s.p, DType::U8, {H / 16, tokens}), ta(a.p, DType::FP32, {tokens});
            ops::inkling_quant_nvfp4_4o6(tx, tc, ts, ta, nullptr);
            cuda_synchronize();
            bad += count_mismatch(from_device<std::uint8_t>(c, c.bytes), ref_codes);
            bad += count_mismatch(from_device<std::uint8_t>(s, s.bytes), ref_scales);
            const auto amax = from_device<float>(a, tokens);
            const auto ref_rs = st.vec<float>("x_rowscale");
            const float rcp = 1.0F / (1536.0F * l.den);
            int bad_rs = 0;
            for (int t = 0; t < tokens; ++t) {
                for (int k = 0; k < TOPK; ++k) {
                    const float rs = (amax[t] * l.amax[0][ref_ids[t * TOPK + k]]) * rcp;
                    bad_rs += bits_of(rs) != bits_of(ref_rs[t * TOPK + k]);
                }
            }
            std::printf("  x quant: code/scale bytes off %d, gate_up row scales off %d\n", bad, bad_rs);
            bad += bad_rs;
        } else {
            DeviceBuffer c(static_cast<std::size_t>(tokens) * H), s(static_cast<std::size_t>(tokens) * H / 32);
            Tensor tc(c.p, DType::U8, {H, tokens}), ts(s.p, DType::U8, {H / 32, tokens});
            ops::inkling_quant_mxfp8(tx, tc, ts, nullptr);
            cuda_synchronize();
            bad += count_mismatch(from_device<std::uint8_t>(c, c.bytes), ref_codes);
            bad += count_mismatch(from_device<std::uint8_t>(s, s.bytes), ref_scales);
            std::printf("  x quant: code/scale bytes off %d\n", bad);
        }
        failures += report("capture: x quantisation bit-exact", bad);
    }
    // 3. SiTU + requantisation on the CAPTURED gate_up rows (isolates them from the GEMM)
    const std::size_t slots = static_cast<std::size_t>(tokens) * TOPK;
    {
        DeviceBuffer gu = to_device(st.vec<std::uint16_t>("gate_up")), g = to_device(ref_gammas), y(slots * I * 2);
        Tensor tgu(gu.p, DType::BF16, {2 * I, static_cast<int>(slots)}), tg(g.p, DType::FP32, {static_cast<int>(slots)}),
            ty(y.p, DType::BF16, {I, static_cast<int>(slots)});
        ops::inkling_softcap_swiglu(tgu, tg, ty, nullptr);
        int bad = 0;
        const auto ref_codes  = st.vec<std::uint8_t>("y_codes");
        const auto ref_scales = st.vec<std::uint8_t>("y_scales");
        if (nv) {
            DeviceBuffer c(slots * I / 2), s(slots * I / 16), a(slots * 4);
            Tensor tc(c.p, DType::U8, {I / 2, static_cast<int>(slots)}), ts(s.p, DType::U8, {I / 16, static_cast<int>(slots)}),
                ta(a.p, DType::FP32, {static_cast<int>(slots)});
            ops::inkling_quant_nvfp4_4o6(ty, tc, ts, ta, nullptr);
            cuda_synchronize();
            bad += count_mismatch(from_device<std::uint8_t>(c, c.bytes), ref_codes);
            bad += count_mismatch(from_device<std::uint8_t>(s, s.bytes), ref_scales);
            const auto amax   = from_device<float>(a, slots);
            const auto ref_rs = st.vec<float>("y_rowscale");
            const float rcp   = 1.0F / (1536.0F * l.den);
            for (std::size_t i = 0; i < slots; ++i) {
                bad += bits_of((amax[i] * l.amax[1][ref_ids[i]]) * rcp) != bits_of(ref_rs[i]);
            }
        } else {
            DeviceBuffer c(slots * I), s(slots * I / 32);
            Tensor tc(c.p, DType::U8, {I, static_cast<int>(slots)}), ts(s.p, DType::U8, {I / 32, static_cast<int>(slots)});
            ops::inkling_quant_mxfp8(ty, tc, ts, nullptr);
            cuda_synchronize();
            bad += count_mismatch(from_device<std::uint8_t>(c, c.bytes), ref_codes);
            bad += count_mismatch(from_device<std::uint8_t>(s, s.bytes), ref_scales);
        }
        std::printf("  SiTU + requant on captured gate_up: bytes / row scales off %d\n", bad);
        failures += report("capture: SiTU + requantisation bit-exact", bad);
    }
    // 4. the whole op with taps: y_q, down rows, output
    {
        DeviceBuffer out(static_cast<std::size_t>(tokens) * H * 2), y_codes(slots * (nv ? I / 2 : I)),
            y_scales(slots * (nv ? I / 16 : I / 32)), y_amax(slots * 4), down(slots * H * 2);
        ops::InklingMoeTaps taps;
        taps.y_codes  = static_cast<std::uint8_t*>(y_codes.p);
        taps.y_scales = static_cast<std::uint8_t*>(y_scales.p);
        taps.y_amax   = static_cast<float*>(y_amax.p);
        taps.down     = down.p;
        ops::InklingMoeOptions options;
        options.taps = &taps;
        DeviceArena arena(ops::inkling_moe_workspace_bytes(format, tokens, false));
        Tensor tout(out.p, DType::BF16, {H, tokens});
        ops::inkling_moe(tx, l.view, tout, arena, nullptr, options);
        cuda_synchronize();
        const int bad_y = count_mismatch(from_device<std::uint8_t>(y_codes, y_codes.bytes), st.vec<std::uint8_t>("y_codes")) +
                          count_mismatch(from_device<std::uint8_t>(y_scales, y_scales.bytes), st.vec<std::uint8_t>("y_scales"));
        const int bad_down = count_mismatch(from_device<std::uint16_t>(down, slots * H), st.vec<std::uint16_t>("down"));
        const int bad_out  = count_mismatch(from_device<std::uint16_t>(out, static_cast<std::size_t>(tokens) * H), st.vec<std::uint16_t>("out"));
        std::printf("  inkling_moe: y_q bytes off %d, down elements off %d, output elements off %d\n", bad_y, bad_down, bad_out);
        failures += report("capture: gate_up GEMM + SiTU + requant (y_q) bit-exact", bad_y);
        failures += report("capture: down GEMM rows bit-exact", bad_down);
        failures += report("capture: combined output bit-exact", bad_out);
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    int device = 0, major = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cuda_check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "cc");
    if (major < 12) {
        std::printf("Inkling's W4A4 experts need sm_120+: skipped\n");
        return 77;
    }
    int failures = 0;
    if (argc == 3 && std::string(argv[1]) == "--capture") {
        failures = run_capture(argv[2]);
        std::printf("%s (%d failing checks)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
        return failures == 0 ? 0 : 1;
    }
    failures += test_quant_nvfp4(H);
    failures += test_quant_nvfp4(I);
    failures += test_quant_mxfp8(H);
    failures += test_quant_mxfp8(I);
    failures += test_softcap_swiglu();
    failures += test_moe(ops::InklingExpertFormat::Nvfp4);
    failures += test_moe(ops::InklingExpertFormat::Mxfp8);
    std::printf("%s (%d failing checks)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
