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
#include "api/ops/lora_store.h"
#include "core/arena.h"
#include "core/engine_context.h"
#include "family/impl/lora_bind_inkling.h"
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
#include <cstdlib>
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
        int head_cols = 1;
        while (head_cols * 2 <= k) { head_cols *= 2; }
        const bool head = 16 * b < head_cols;
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
            // as the vLLM image's compiled kernels (measured on lane a's capture): fused in the row's power-of-two
            // head, unfused in its tail
            if (head) {
                err6[j] = std::fabs(std::fma(v6[j] * s6, outer, -xb[j]));
                err4[j] = std::fabs(std::fma(v4[j] * s4, outer, -xb[j]));
            } else {
                err6[j] = std::fabs((v6[j] * s6) * outer - xb[j]);
                err4[j] = std::fabs((v4[j] * s4) * outer - xb[j]);
            }
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

void upload_adapter(Adapter& ad);

void build_adapter(Adapter& ad, std::uint64_t seed, bool zero_b) {
    Rng rng(seed);
    const int ks[3] = {H, H, I}, ns[3] = {I, I, H};
    for (int p = 0; p < 3; ++p) {
        ad.a[p].resize(static_cast<std::size_t>(E) * ad.slots * ad.rank * ks[p]);
        ad.b[p].resize(static_cast<std::size_t>(E) * ad.slots * ns[p] * ad.rank);
        for (auto& v : ad.a[p]) { v = rng.uniform(-0.02F, 0.02F); }
        for (auto& v : ad.b[p]) { v = zero_b ? 0.0F : rng.uniform(-0.02F, 0.02F); }
        round_to_bf16(ad.a[p]);
        round_to_bf16(ad.b[p]);
    }
    upload_adapter(ad);
}

/// Device copies of a host adapter's banks and a device table in the `bind_lora_moe` order.
void upload_adapter(Adapter& ad) {
    const int ks[3] = {H, H, I}, ns[3] = {I, I, H};
    std::vector<ops::LoraBank> table(2 + 3 * E);
    for (int p = 0; p < 3; ++p) {
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

/// The persistent zeroed state (InklingMoeOptions::state) every test call shares, as one engine stream
/// would: allocated and zeroed ONCE; each call must leave it zero.
DeviceBuffer& state_buffer() {
    static DeviceBuffer* state =
        new DeviceBuffer(to_device(std::vector<std::int32_t>(ops::inkling_moe_state_bytes() / 4, 0)));
    return *state;
}
std::int32_t* persistent_state() { return static_cast<std::int32_t*>(state_buffer().p); }
int state_nonzero_words() {
    const auto words = from_device<std::int32_t>(state_buffer(), ops::inkling_moe_state_bytes() / 4);
    return static_cast<int>(std::count_if(words.begin(), words.end(), [](std::int32_t w) { return w != 0; }));
}

DeviceRun run_moe(const Layer& l, const std::vector<float>& x, int tokens, const int* live = nullptr,
                  const ops::InklingMoeAdapters* adapters = nullptr, bool capture = false, bool use_state = true) {
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
    options.state    = use_state ? persistent_state() : nullptr;
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

/// `inkling_decode_expert_units` (the trainer's dX operand) vs the host decode, bit for bit, on experts at
/// both ends and in the middle of the bank, both projections; the factor vs amax / den.
int test_decode_units(const Layer& l, const char* label) {
    const bool nv = l.format == ops::InklingExpertFormat::Nvfp4;
    const ops::InklingExpertBank* banks[2] = {&l.view.w13, &l.view.w2};
    const int ks[2] = {H, I};
    const int ranges[3][2] = {{0, 2}, {61, 3}, {126, 2}};
    int bad_unit = 0, bad_factor = 0;
    for (int p = 0; p < 2; ++p) {
        for (const auto& range : ranges) {
            const int first = range[0], count = range[1], k = ks[p];
            DeviceBuffer du(static_cast<std::size_t>(count) * N * k * 2), df(static_cast<std::size_t>(count) * 4);
            Tensor tu(du.p, DType::BF16, {k, count * N}), tf(df.p, DType::FP32, {count});
            ops::inkling_decode_expert_units(*banks[p], l.format, k, first, count, tu, tf, nullptr);
            cuda_synchronize();
            const auto unit   = from_device<std::uint16_t>(du, static_cast<std::size_t>(count) * N * k);
            const auto factor = from_device<float>(df, count);
            for (int i = 0; i < count; ++i) {
                std::vector<float> w;
                decode_expert(l, p, first + i, w);
                for (std::size_t j = 0; j < w.size(); ++j) {
                    // the host decode is exact in fp32 and in bf16: compare bits (incl. the sign of zero)
                    bad_unit += unit[static_cast<std::size_t>(i) * N * k + j] != f32_to_bf16(w[j]) ||
                                bf16_to_f32(f32_to_bf16(w[j])) != w[j];
                }
                const float expect = nv ? l.amax[p][first + i] / l.den : 1.0F;
                bad_factor += bits_of(factor[i]) != bits_of(expect);
            }
        }
    }
    std::printf("  decode units (%s): unit bf16 off %d, factor off %d\n", label, bad_unit, bad_factor);
    return report((std::string("decode units ") + label + ": bit-exact vs host (unit, amax/den)").c_str(),
                  bad_unit + bad_factor);
}

/// Fused-gate_up LoRA in PEFT form for every expert: A [E][rank][2304], B [E][2304 interleaved][rank],
/// down A [E][rank][1152], B [E][2304][rank] (bf16 values); or split gate / up (B [E][1152][rank] each).
struct PeftAdapter {
    int rank = 16;
    float scale = 2.0F;
    bool fused = true;
    std::vector<float> a13, b13, a_up, b_up, a2, b2; // a_up / b_up: split form only (a13 / b13 = gate)
};

PeftAdapter make_peft(std::uint64_t seed, int rank, float scale, bool fused) {
    Rng rng(seed);
    PeftAdapter p;
    p.rank = rank, p.scale = scale, p.fused = fused;
    const std::size_t e = E, r = rank;
    const auto fill = [&](std::vector<float>& v, std::size_t n) {
        v.resize(n);
        for (auto& x : v) { x = rng.uniform(-0.03F, 0.03F); }
        round_to_bf16(v);
    };
    fill(p.a13, e * r * H);
    fill(p.b13, e * (fused ? 2 * I : I) * r);
    if (!fused) { fill(p.a_up, e * r * H); fill(p.b_up, e * I * r); }
    fill(p.a2, e * r * I);
    fill(p.b2, e * H * r);
    return p;
}

std::vector<std::uint16_t> bits_slice(const std::vector<float>& v, std::size_t begin, std::size_t n) {
    std::vector<std::uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = f32_to_bf16(v[begin + i]); }
    return out;
}

/// Loads `p` into `slot` of the store by module name, as the engine's adapter loader does.
void load_peft(ops::LoraStore& store, const PeftAdapter& p, int slot) {
    const std::size_t r = p.rank;
    for (int e = 0; e < E; ++e) {
        const std::string prefix = "mlp.experts." + std::to_string(e) + ".";
        if (p.fused) {
            store.set_module_slot(0, prefix + "gate_up_proj", slot, bits_slice(p.a13, e * r * H, r * H),
                                  bits_slice(p.b13, e * 2 * I * r, 2 * I * r), p.rank, H, 2 * I, p.scale);
        } else {
            store.set_module_slot(0, prefix + "gate_proj", slot, bits_slice(p.a13, e * r * H, r * H),
                                  bits_slice(p.b13, e * I * r, I * r), p.rank, H, I, p.scale);
            store.set_module_slot(0, prefix + "up_proj", slot, bits_slice(p.a_up, e * r * H, r * H),
                                  bits_slice(p.b_up, e * I * r, I * r), p.rank, H, I, p.scale);
        }
        store.set_module_slot(0, prefix + "down_proj", slot, bits_slice(p.a2, e * r * I, r * I),
                              bits_slice(p.b2, e * H * r, H * r), p.rank, I, H, p.scale);
    }
}

/// What the store must hold for `p` in `slot` of a host Adapter (bank layout, padded to ad.rank, alpha/r
/// folded into A with one bf16 rounding, gate = even / up = odd rows of a fused B).
void expect_peft(Adapter& ad, const PeftAdapter& p, int slot) {
    const int ks[3] = {H, H, I}, ns[3] = {I, I, H};
    for (int port = 0; port < 3; ++port) {
        for (int e = 0; e < E; ++e) {
            const std::vector<float>& a = port == 2 ? p.a2 : (port == 1 && !p.fused ? p.a_up : p.a13);
            const int k = ks[port], n = ns[port];
            float* da = ad.a[port].data() + ((static_cast<std::size_t>(e) * ad.slots + slot) * ad.rank) * k;
            float* db = ad.b[port].data() + ((static_cast<std::size_t>(e) * ad.slots + slot) * n) * ad.rank;
            std::fill(da, da + static_cast<std::size_t>(ad.rank) * k, 0.0F);
            std::fill(db, db + static_cast<std::size_t>(n) * ad.rank, 0.0F);
            for (int r = 0; r < p.rank; ++r) {
                for (int i = 0; i < k; ++i) {
                    da[static_cast<std::size_t>(r) * k + i] = bf(a[(static_cast<std::size_t>(e) * p.rank + r) * k + i] * p.scale);
                }
            }
            for (int row = 0; row < n; ++row) {
                float const* source;
                if (port == 2) {
                    source = p.b2.data() + (static_cast<std::size_t>(e) * H + row) * p.rank;
                } else if (p.fused) {
                    source = p.b13.data() + (static_cast<std::size_t>(e) * 2 * I + 2 * row + port) * p.rank;
                } else {
                    source = (port == 0 ? p.b13 : p.b_up).data() + (static_cast<std::size_t>(e) * I + row) * p.rank;
                }
                std::copy(source, source + p.rank, db + static_cast<std::size_t>(row) * ad.rank);
            }
        }
    }
}

/// Expert LoRA through the engine's slot store (family/impl/lora_bind_inkling.h): binding, bank bytes,
/// zero / base rows bit-exact, store table == a hand-built table bit for bit, vs the host pipeline, uniform
/// slot, width, hot swap (staged commit) under a captured graph, unload, the round helper, refusals.
int test_lora_store(const Layer& l, const std::vector<float>& x, const DeviceRun& base, const std::vector<int>& ids,
                    const std::vector<float>& gammas, const std::vector<std::vector<float>>& w13,
                    const std::vector<std::vector<float>>& w2) {
    constexpr int kT = 24, kSlots = 3, kRank = 16;
    int failures = 0;
    struct ScopedContext {
        ops::EngineOpsContext context;
        ScopedContext() { ops::bind_ops_context(&context); }
        ~ScopedContext() { ops::lora_clear_round(); ops::bind_ops_context(nullptr); }
    } scoped;
    ops::LoraStore& store = ops::lora_store_for_current_device();
    store.configure(kSlots, kRank, 300);
    family::bind_lora_inkling_moe(store, 0, l.view);
    store.ensure_banks();
    store.set_active(true);
    const auto* table = store.bank_table(l.view.router);
    failures += report("LoRA store: bank table registered under the router key", table != nullptr ? 0 : 1);
    if (table == nullptr) { return failures; }

    std::vector<int> slots(kT);
    for (int t = 0; t < kT; ++t) { slots[t] = t % 4 - 1; } // base, empty slot 0, slot 1, slot 2
    DeviceBuffer dslots = to_device(slots);
    const ops::InklingMoeAdapters store_round{table, static_cast<const std::int32_t*>(dslots.p), nullptr, kRank};
    const auto rows_equal = [&](const DeviceRun& a, const DeviceRun& b, int t) {
        return std::equal(a.out.begin() + static_cast<std::size_t>(t) * H, a.out.begin() + static_cast<std::size_t>(t + 1) * H,
                          b.out.begin() + static_cast<std::size_t>(t) * H);
    };

    // 1. Fresh banks are zero: every token (any slot) == base, bit for bit.
    const DeviceRun empty = run_moe(l, x, kT, nullptr, &store_round);
    failures += report("LoRA store: empty slots == base (bit-exact)", empty.out == base.out ? 0 : 1);

    // 2. Load P (fused gate_up_proj r16 alpha 32 = the kit's recipe) into slot 1 and Q (split gate / up, r8,
    //    scale 1.5: the fold rounds) into slot 2, by module name.
    const PeftAdapter P = make_peft(41, 16, 2.0F, true), Q = make_peft(43, 8, 1.5F, false);
    load_peft(store, P, 1);
    load_peft(store, Q, 2);
    Adapter expected;
    expected.rank = kRank, expected.slots = kSlots;
    const int ks[3] = {H, H, I}, ns[3] = {I, I, H};
    for (int port = 0; port < 3; ++port) {
        expected.a[port].assign(static_cast<std::size_t>(E) * kSlots * kRank * ks[port], 0.0F);
        expected.b[port].assign(static_cast<std::size_t>(E) * kSlots * ns[port] * kRank, 0.0F);
    }
    expect_peft(expected, P, 1);
    expect_peft(expected, Q, 2);
    upload_adapter(expected);

    // 2a. The store's device banks hold exactly the expected bytes (layout, padding, fold, row map).
    {
        std::vector<ops::LoraBank> views(2 + 3 * E);
        cuda_check(cudaMemcpy(views.data(), table, views.size() * sizeof(ops::LoraBank), cudaMemcpyDeviceToHost), "table");
        int bad_bytes = 0, bad_shape = 0;
        bad_shape += views[0].a != nullptr || views[1].a != nullptr;
        for (int e = 0; e < E; ++e) {
            for (int port = 0; port < 3; ++port) {
                const ops::LoraBank& bank = views[2 + 3 * e + port];
                bad_shape += bank.rank != kRank || bank.k != ks[port] || bank.n != ns[port] ||
                             bank.a_stride != static_cast<std::int64_t>(kRank) * ks[port] ||
                             bank.b_stride != static_cast<std::int64_t>(ns[port]) * kRank;
                std::vector<std::uint16_t> a(static_cast<std::size_t>(kSlots) * bank.a_stride),
                    b(static_cast<std::size_t>(kSlots) * bank.b_stride);
                cuda_check(cudaMemcpy(a.data(), bank.a, a.size() * 2, cudaMemcpyDeviceToHost), "bank a");
                cuda_check(cudaMemcpy(b.data(), bank.b, b.size() * 2, cudaMemcpyDeviceToHost), "bank b");
                const std::size_t ea = static_cast<std::size_t>(e) * kSlots * bank.a_stride;
                const std::size_t eb = static_cast<std::size_t>(e) * kSlots * bank.b_stride;
                for (std::size_t i = 0; i < a.size(); ++i) { bad_bytes += a[i] != f32_to_bf16(expected.a[port][ea + i]); }
                for (std::size_t i = 0; i < b.size(); ++i) { bad_bytes += b[i] != f32_to_bf16(expected.b[port][eb + i]); }
            }
        }
        std::printf("  lora store: bank bytes off %d, bank shapes off %d\n", bad_bytes, bad_shape);
        failures += report("LoRA store: device banks == expected bytes (fold, padding, interleaved rows)",
                           bad_bytes + bad_shape);
    }

    // 2b. The op through the store's table == through a hand-built table of the same bytes; base and
    //     empty-slot rows == base; adapted rows vs the host pipeline.
    const DeviceRun loaded = run_moe(l, x, kT, nullptr, &store_round);
    const ops::InklingMoeAdapters direct_round{static_cast<const ops::LoraBank*>(expected.d_table.p),
                                               static_cast<const std::int32_t*>(dslots.p), nullptr, kRank};
    const DeviceRun direct = run_moe(l, x, kT, nullptr, &direct_round);
    failures += report("LoRA store: store table == hand-built table (bit-exact)", loaded.out == direct.out ? 0 : 1);
    int base_bad = 0;
    double worst_cos = 1.0, worst_rel = 0.0, worst_delta = 1.0;
    for (int t = 0; t < kT; ++t) {
        if (slots[t] <= 0) { base_bad += !rows_equal(loaded, base, t); continue; }
        const std::size_t begin = static_cast<std::size_t>(t) * H;
        const auto ref_base = host_moe_token(l, w13, w2, x.data() + begin, ids.data() + t * TOPK, gammas.data() + t * TOPK, nullptr, -1);
        const auto ref = host_moe_token(l, w13, w2, x.data() + begin, ids.data() + t * TOPK, gammas.data() + t * TOPK,
                                        &expected, slots[t]);
        double fd = 0, fr = 0, fg = 0, fdiff = 0, dd = 0, na = 0, nb = 0;
        for (int n = 0; n < H; ++n) {
            const double g = bf16_to_f32(loaded.out[begin + n]);
            const double dg = g - bf16_to_f32(base.out[begin + n]), dr = double(ref[n]) - ref_base[n];
            fd += g * ref[n]; fr += double(ref[n]) * ref[n]; fg += g * g; fdiff += (g - ref[n]) * (g - ref[n]);
            dd += dg * dr; na += dg * dg; nb += dr * dr;
        }
        worst_cos = std::min(worst_cos, fd / std::sqrt(fr * fg + 1e-300));
        worst_rel = std::max(worst_rel, std::sqrt(fdiff / (fr + 1e-300)));
        worst_delta = std::min(worst_delta, dd / std::sqrt(na * nb + 1e-300));
    }
    std::printf("  lora store: base / empty-slot rows changed %d; adapted vs host worst cos %.6f rel-L2 %.3e, delta cos %.5f\n",
                base_bad, worst_cos, worst_rel, worst_delta);
    failures += report("LoRA store: base and empty-slot rows bit-exact in an adapter round", base_bad);
    failures += report("LoRA store: adapted rows vs host pipeline (cos >= 0.9999, rel-L2 <= 1e-2, delta cos >= 0.99)",
                       (worst_cos >= 0.9999 && worst_rel <= 1e-2 && worst_delta >= 0.99) ? 0 : 1);

    // 3. Uniform slot (a prefill chunk): == every token at that slot by token slots == the mixed round's rows.
    {
        store.write_uniform_slot(1, nullptr);
        const ops::InklingMoeAdapters uniform{table, nullptr, store.uniform_cell(), kRank};
        const DeviceRun by_cell = run_moe(l, x, kT, nullptr, &uniform);
        DeviceBuffer ones = to_device(std::vector<int>(kT, 1));
        const ops::InklingMoeAdapters all_one{table, static_cast<const std::int32_t*>(ones.p), nullptr, kRank};
        const DeviceRun by_tokens = run_moe(l, x, kT, nullptr, &all_one);
        int bad = by_cell.out == by_tokens.out ? 0 : 1;
        for (int t = 0; t < kT; ++t) { bad += slots[t] == 1 && !rows_equal(by_cell, loaded, t); }
        // width: an adapted token alone (T = 1, both slot forms) == its row in the round
        for (int t : {2, 3, 10, 11}) {
            std::vector<float> one(x.begin() + static_cast<std::size_t>(t) * H, x.begin() + static_cast<std::size_t>(t + 1) * H);
            DeviceBuffer s1 = to_device(std::vector<int>{slots[t]});
            const ops::InklingMoeAdapters a1{table, static_cast<const std::int32_t*>(s1.p), nullptr, kRank};
            const DeviceRun single = run_moe(l, one, 1, nullptr, &a1);
            bad += !std::equal(single.out.begin(), single.out.end(), loaded.out.begin() + static_cast<std::size_t>(t) * H);
        }
        store.write_uniform_slot(-1, nullptr);
        failures += report("LoRA store: uniform slot == token slots == mixed rows; T=1 rows (bit-exact)", bad);
    }

    // 4. Hot swap under a captured graph: R replaces P in slot 1 by a staged in-place commit between replays.
    {
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        DeviceBuffer dx = to_device(to_bf16_bits(std::vector<float>(x.begin(), x.begin() + static_cast<std::size_t>(kT) * H)));
        DeviceBuffer dout(static_cast<std::size_t>(kT) * H * 2);
        DeviceArena arena(ops::inkling_moe_workspace_bytes(l.format, kT, true));
        Tensor tx(dx.p, DType::BF16, {H, kT}), tout(dout.p, DType::BF16, {H, kT});
        ops::InklingMoeOptions options;
        options.adapters = &store_round;
        options.state    = persistent_state();
        ops::inkling_moe(tx, l.view, tout, arena, stream, options); // warm (attributes) outside the capture
        arena.reset();
        cuda_synchronize(stream);
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        ops::inkling_moe(tx, l.view, tout, arena, stream, options);
        arena.reset();
        cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
        cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        const auto replay = [&]() {
            cuda_check(cudaMemsetAsync(dout.p, 0xFF, dout.bytes, stream), "poison");
            cuda_check(cudaGraphLaunch(exec, stream), "launch");
            cuda_synchronize(stream);
            DeviceRun run;
            run.out = from_device<std::uint16_t>(dout, static_cast<std::size_t>(kT) * H);
            return run;
        };
        const DeviceRun before = replay();
        int bad = before.out == loaded.out ? 0 : 1;
        // stage while "serving" (host only), commit at a round boundary
        const PeftAdapter R = make_peft(47, 16, 2.0F, true);
        std::vector<std::string> names;
        std::vector<std::vector<std::uint16_t>> tensors;
        names.reserve(2 * E);
        tensors.reserve(4 * E);
        std::vector<ops::LoraStore::StagedModule> modules;
        for (int e = 0; e < E; ++e) {
            const std::size_t r = R.rank;
            names.push_back("mlp.experts." + std::to_string(e) + ".gate_up_proj");
            tensors.push_back(bits_slice(R.a13, e * r * H, r * H));
            tensors.push_back(bits_slice(R.b13, e * 2 * I * r, 2 * I * r));
            names.push_back("mlp.experts." + std::to_string(e) + ".down_proj");
            tensors.push_back(bits_slice(R.a2, e * r * I, r * I));
            tensors.push_back(bits_slice(R.b2, e * H * r, H * r));
        }
        for (int e = 0; e < E; ++e) {
            modules.push_back({0, &names[2 * e], &tensors[4 * e], &tensors[4 * e + 1], R.rank, H, 2 * I, R.scale});
            modules.push_back({0, &names[2 * e + 1], &tensors[4 * e + 2], &tensors[4 * e + 3], R.rank, I, H, R.scale});
        }
        const auto staged = store.stage_modules(1, modules);
        const DeviceRun still = replay(); // staging alone changes nothing
        bad += still.out == before.out ? 0 : 1;
        store.commit_staged(*staged);
        const DeviceRun after = replay();
        expect_peft(expected, R, 1);
        upload_adapter(expected);
        const ops::InklingMoeAdapters direct_swapped{static_cast<const ops::LoraBank*>(expected.d_table.p),
                                                     static_cast<const std::int32_t*>(dslots.p), nullptr, kRank};
        const DeviceRun direct_after = run_moe(l, x, kT, nullptr, &direct_swapped);
        int base_changed = 0, kept_changed = 0, swapped_same = 0;
        for (int t = 0; t < kT; ++t) {
            if (slots[t] <= 0) { base_changed += !rows_equal(after, base, t); }
            if (slots[t] == 2) { kept_changed += !rows_equal(after, before, t); }
            if (slots[t] == 1) { swapped_same += rows_equal(after, before, t); }
        }
        bad += after.out == direct_after.out ? 0 : 1;
        std::printf("  lora hot swap (graph): base rows changed %d, slot-2 rows changed %d, slot-1 rows unchanged %d, "
                    "replay == eager with the new bytes %s\n", base_changed, kept_changed, swapped_same,
                    after.out == direct_after.out ? "yes" : "NO");
        failures += report("LoRA store: hot swap between graph replays (base / other slots bit-exact, new adapter live)",
                           bad + base_changed + kept_changed + swapped_same);
        // unload: slot 1's rows return to base (clear_slot's memsets run on the legacy stream; the graph's
        // stream is non-blocking -- the engine drains before an unload, the test synchronises)
        store.clear_slot(1);
        cuda_check(cudaDeviceSynchronize(), "clear");
        const DeviceRun cleared = replay();
        int clear_bad = 0;
        for (int t = 0; t < kT; ++t) {
            if (slots[t] <= 1) { clear_bad += !rows_equal(cleared, base, t); }
            if (slots[t] == 2) { clear_bad += !rows_equal(cleared, before, t); }
        }
        failures += report("LoRA store: clear_slot -> the slot's rows == base (bit-exact)", clear_bad);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    }

    // 5. The round helper the target calls per visit.
    {
        int bad = 0;
        ops::InklingMoeAdapters storage;
        bad += family::inkling_moe_round_adapters(l.view, storage) != nullptr; // no round published
        Tensor slot_tensor(dslots.p, DType::I32, {kT});
        ops::LoraRound round;
        round.slots        = &slot_tensor;
        round.scratch      = store.scratch(kT);
        round.uniform_cell = store.uniform_cell();
        ops::lora_set_round(round);
        const auto* got = family::inkling_moe_round_adapters(l.view, storage, 5);
        bad += got == nullptr || got->table != table || got->max_rank != kRank || got->uniform_slot != nullptr ||
               got->token_slot != static_cast<const std::int32_t*>(dslots.p) + 5;
        {
            ops::ScopedLoraBaseRound base_round(true);
            bad += family::inkling_moe_round_adapters(l.view, storage) != nullptr;
        }
        round.slots   = nullptr;
        round.uniform = true;
        ops::lora_set_round(round);
        got = family::inkling_moe_round_adapters(l.view, storage);
        bad += got == nullptr || got->uniform_slot != store.uniform_cell() || got->token_slot != nullptr;
        ops::InklingMoeWeights other = l.view;
        other.router = dslots.p; // a layer without a table
        bad += family::inkling_moe_round_adapters(other, storage) != nullptr;
        ops::lora_clear_round();
        failures += report("LoRA store: round helper (token / uniform / base round / no table)", bad);
    }

    // 6. Refusals.
    {
        int bad = 0;
        const auto throws = [](auto&& f) {
            try { f(); } catch (const std::exception&) { return true; }
            return false;
        };
        bad += !throws([&] { store.set_module_slot(0, "mlp.gate", 0, std::vector<std::uint16_t>(16 * H), std::vector<std::uint16_t>(E * 16), 16, H, E, 2.0F); });
        bad += !throws([&] {
            store.set_module_slot(0, "mlp.experts.3.down_proj", 0, std::vector<std::uint16_t>(16 * I), std::vector<std::uint16_t>(H * 16),
                                  16, I, H, 2.0F, std::vector<float>(H, 1.0F));
        });
        bad += !throws([&] { store.set_module_slot(0, "mlp.experts.3.gate_up_proj", 0, std::vector<std::uint16_t>(16 * H), std::vector<std::uint16_t>(I * 16), 16, H, I, 2.0F); });
        ops::LoraStore wide;
        wide.configure(1, 128, 8);
        bad += !throws([&] { family::bind_lora_inkling_moe(wide, 0, l.view); });
        failures += report("LoRA store: refusals (router, DoRA magnitude, wrong gate_up shape, rank > 64)", bad);
    }
    return failures;
}

/// The zeroed-counter contract of the fused narrow route (InklingMoeOptions::state): the state is zero after
/// every call (self-cleaning); bits == the call without state at T = 1 / 8 / 24 / 64 (and == the wide path's
/// rows); back-to-back calls; a captured graph that is discarded unrun ("aborted") and a graph replayed 3x;
/// a call whose live count is 0; a call that throws on the host before launching.
int test_state_contract(const Layer& l, const std::vector<float>& x, const DeviceRun& base, const DeviceRun& wide) {
    int bad = 0, dirty = 0;
    constexpr int kT = 24;
    dirty += state_nonzero_words();
    const DeviceRun no_state = run_moe(l, x, kT, nullptr, nullptr, false, false);
    bad += no_state.out != base.out;
    dirty += state_nonzero_words();
    for (int t : {1, 8, 64}) {
        const DeviceRun with  = run_moe(l, x, t);
        dirty += state_nonzero_words();
        const DeviceRun plain = run_moe(l, x, t, nullptr, nullptr, false, false);
        bad += with.out != plain.out;
        bad += !std::equal(with.out.begin(), with.out.end(), wide.out.begin());
    }
    const DeviceRun again = run_moe(l, x, kT);
    bad += again.out != base.out;
    dirty += state_nonzero_words();
    {
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        DeviceBuffer dx = to_device(to_bf16_bits(std::vector<float>(x.begin(), x.begin() + static_cast<std::size_t>(kT) * H)));
        DeviceBuffer dout(static_cast<std::size_t>(kT) * H * 2);
        DeviceArena arena(ops::inkling_moe_workspace_bytes(l.format, kT, false));
        Tensor tx(dx.p, DType::BF16, {H, kT}), tout(dout.p, DType::BF16, {H, kT});
        ops::InklingMoeOptions options;
        options.state = persistent_state();
        // "aborted": captured, never launched, discarded
        cudaGraph_t discarded = nullptr;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        ops::inkling_moe(tx, l.view, tout, arena, stream, options);
        arena.reset();
        cuda_check(cudaStreamEndCapture(stream, &discarded), "end capture");
        cudaGraphDestroy(discarded);
        const DeviceRun after_discard = run_moe(l, x, kT);
        bad += after_discard.out != base.out;
        dirty += state_nonzero_words();
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        ops::inkling_moe(tx, l.view, tout, arena, stream, options);
        arena.reset();
        cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
        cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        for (int replay = 0; replay < 3; ++replay) {
            cuda_check(cudaMemsetAsync(dout.p, 0xFF, dout.bytes, stream), "poison");
            cuda_check(cudaGraphLaunch(exec, stream), "launch");
            cuda_synchronize(stream);
            bad += from_device<std::uint16_t>(dout, static_cast<std::size_t>(kT) * H) != base.out;
            dirty += state_nonzero_words();
        }
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    }
    // every token padding: output rows zero, nothing routed
    const int none = 0;
    const DeviceRun empty = run_moe(l, x, kT, &none);
    bad += std::any_of(empty.out.begin(), empty.out.end(), [](std::uint16_t v) { return v != 0; });
    dirty += state_nonzero_words();
    // a call that throws on the host (bad adapter rank) before any launch
    try {
        const ops::InklingMoeAdapters broken{reinterpret_cast<const ops::LoraBank*>(persistent_state()), nullptr, nullptr, 0};
        (void)run_moe(l, x, kT, nullptr, &broken);
        ++bad;
    } catch (const std::exception&) {}
    const DeviceRun last = run_moe(l, x, kT);
    bad += last.out != base.out;
    dirty += state_nonzero_words();
    std::printf("  state contract: rows off %d, non-zero state words after calls %d\n", bad, dirty);
    return report("fused narrow route: state self-cleaning, bits == no-state / wide, graphs, aborted capture",
                  bad + dirty);
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
    // A prefill-sized round (several tiles per expert; rows land at atomic-order positions).
    constexpr int kHuge = 4096;
    std::vector<float> huge_x(static_cast<std::size_t>(kHuge) * H);
    std::copy(x.begin(), x.end(), huge_x.begin());
    Rng huge_rng(nv ? 17 : 18);
    for (std::size_t i = x.size(); i < huge_x.size(); ++i) { huge_x[i] = huge_rng.uniform(-2.0F, 2.0F); }
    round_to_bf16(huge_x);
    const DeviceRun huge = run_moe(l, huge_x, kHuge);
    width_bad += !std::equal(base.out.begin(), base.out.end(), huge.out.begin());
    width_bad += !std::equal(wide.out.begin(), wide.out.end(), huge.out.begin());
    failures += report("width invariance T=1 / 24 / 300 / 4096 / padded 32 (bit-exact)", width_bad);

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
    double worst_delta_cos = 1.0, worst_lora_cos = 1.0, worst_lora_rel = 0.0;
    for (int t = 0; t < kT; ++t) {
        const auto begin = static_cast<std::size_t>(t) * H;
        if (slots[t] < 0) {
            base_tokens_bad += !std::equal(lora_run.out.begin() + begin, lora_run.out.begin() + begin + H, base.out.begin() + begin);
            continue;
        }
        const auto ref_base = host_moe_token(l, w13, w2, x.data() + begin, ids.data() + t * TOPK, gammas.data() + t * TOPK, nullptr, -1);
        const auto ref_lora = host_moe_token(l, w13, w2, x.data() + begin, ids.data() + t * TOPK, gammas.data() + t * TOPK, &adapter, slots[t]);
        double dot = 0, na = 0, nb = 0, fd = 0, fr = 0, fg = 0, fdiff = 0;
        for (int n = 0; n < H; ++n) {
            const double g  = bf16_to_f32(lora_run.out[begin + n]);
            const double dg = g - bf16_to_f32(base.out[begin + n]);
            const double dr = double(ref_lora[n]) - ref_base[n];
            dot += dg * dr; na += dg * dg; nb += dr * dr;
            fd += g * ref_lora[n]; fr += double(ref_lora[n]) * ref_lora[n]; fg += g * g;
            fdiff += (g - ref_lora[n]) * (g - ref_lora[n]);
        }
        worst_delta_cos = std::min(worst_delta_cos, dot / std::sqrt(na * nb + 1e-300));
        worst_lora_cos  = std::min(worst_lora_cos, fd / std::sqrt(fr * fg + 1e-300));
        worst_lora_rel  = std::max(worst_lora_rel, std::sqrt(fdiff / (fr + 1e-300)));
    }
    std::printf("  lora: base tokens changed %d, adapted outputs vs host: worst cos %.6f rel-L2 %.3e; worst delta cos %.5f\n",
                base_tokens_bad, worst_lora_cos, worst_lora_rel, worst_delta_cos);
    failures += report("LoRA: base tokens bit-exact in an adapter round", base_tokens_bad);
    // Adapter rows are width invariant too: an adapted token alone (T = 1) == its row in the 24-token round.
    int adapted_width_bad = 0;
    for (int t = 0; t < kT; ++t) {
        if (slots[t] < 0 || t % 4 != 0) { continue; }
        DeviceBuffer one_slot = to_device(std::vector<int>{slots[t]});
        ops::InklingMoeAdapters a1{static_cast<const ops::LoraBank*>(adapter.d_table.p),
                                   static_cast<const std::int32_t*>(one_slot.p), nullptr, 16};
        std::vector<float> one(x.begin() + static_cast<std::size_t>(t) * H, x.begin() + static_cast<std::size_t>(t + 1) * H);
        const DeviceRun single = run_moe(l, one, 1, nullptr, &a1);
        adapted_width_bad += !std::equal(single.out.begin(), single.out.end(),
                                         lora_run.out.begin() + static_cast<std::size_t>(t) * H);
    }
    failures += report("LoRA: adapted rows width invariant (T=1 vs 24, bit-exact)", adapted_width_bad);
    failures += report("LoRA: adapted output vs host pipeline (cos >= 0.9999, rel-L2 <= 1e-2)",
                       (worst_lora_cos >= 0.9999 && worst_lora_rel <= 1e-2) ? 0 : 1);
    // The delta is the difference of two bf16-rounded outputs: an indexing check (a wrong expert / slot / port gives
    // cos ~ 0), not a precision one.
    failures += report("LoRA: delta direction vs host (cos >= 0.99)", worst_delta_cos >= 0.99 ? 0 : 1);
    failures += test_state_contract(l, x, base, wide);
    failures += test_lora_store(l, x, base, ids, gammas, w13, w2);
    failures += test_decode_units(l, nv ? "NVFP4 synthetic" : "MXFP8 synthetic");
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

void load_capture_layer(const SafeTensors& st, bool nv, Layer& l) {
    const auto format = nv ? ops::InklingExpertFormat::Nvfp4 : ops::InklingExpertFormat::Mxfp8;
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
}

int run_capture(const char* path, const char* dump_dir) {
    const SafeTensors st = read_safetensors(path);
    const bool nv   = st.meta.at("format") == "nvfp4";
    const int tokens = std::stoi(st.meta.at("tokens"));
    std::printf("[capture %s: layer %s visit %s, %s, T=%d]\n", path, st.meta.at("layer").c_str(),
                st.meta.at("visit").c_str(), nv ? "NVFP4" : "MXFP8", tokens);
    const auto format = nv ? ops::InklingExpertFormat::Nvfp4 : ops::InklingExpertFormat::Mxfp8;
    Layer l;
    load_capture_layer(st, nv, l);
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
    std::vector<std::uint16_t> y_sub;
    {
        DeviceBuffer gu = to_device(st.vec<std::uint16_t>("gate_up")), g = to_device(ref_gammas), y(slots * I * 2);
        Tensor tgu(gu.p, DType::BF16, {2 * I, static_cast<int>(slots)}), tg(g.p, DType::FP32, {static_cast<int>(slots)}),
            ty(y.p, DType::BF16, {I, static_cast<int>(slots)});
        ops::inkling_softcap_swiglu(tgu, tg, ty, nullptr);
        cuda_synchronize();
        y_sub = from_device<std::uint16_t>(y, slots * I);
        if (dump_dir != nullptr) {
            const std::string out = std::string(dump_dir) + "/y_sub_v" + st.meta.at("visit") + ".bf16";
            std::ofstream f(out, std::ios::binary);
            f.write(reinterpret_cast<const char*>(y_sub.data()), static_cast<std::streamsize>(y_sub.size() * 2));
            std::printf("  dumped GPU SiTU of the captured gate_up -> %s\n", out.c_str());
        }
        int bad = 0;
        const auto ref_codes  = st.vec<std::uint8_t>("y_codes");
        const auto ref_scales = st.vec<std::uint8_t>("y_scales");
        if (nv) {
            DeviceBuffer c(slots * I / 2), s(slots * I / 16), a(slots * 4);
            Tensor tc(c.p, DType::U8, {I / 2, static_cast<int>(slots)}), ts(s.p, DType::U8, {I / 16, static_cast<int>(slots)}),
                ta(a.p, DType::FP32, {static_cast<int>(slots)});
            ops::inkling_quant_nvfp4_4o6(ty, tc, ts, ta, nullptr);
            cuda_synchronize();
            const auto got_codes  = from_device<std::uint8_t>(c, c.bytes);
            const auto got_scales = from_device<std::uint8_t>(s, s.bytes);
            bad += count_mismatch(got_codes, ref_codes);
            bad += count_mismatch(got_scales, ref_scales);
            int shown = 0;
            for (std::size_t i = 0; i < got_scales.size() && shown < 8; ++i) {
                if (got_scales[i] != ref_scales[i]) {
                    std::printf("    y scale differs: slot %zu block %zu ours 0x%02x ref 0x%02x\n", i / (I / 16), i % (I / 16),
                                got_scales[i], ref_scales[i]);
                    ++shown;
                }
            }
            for (std::size_t i = 0, shown_codes = 0; i < got_codes.size() && shown_codes < 8; ++i) {
                if (got_codes[i] != ref_codes[i] && got_scales[i / 8] == ref_scales[i / 8]) {
                    std::printf("    y code byte differs with equal scale: slot %zu byte %zu ours 0x%02x ref 0x%02x\n",
                                i / (I / 2), i % (I / 2), got_codes[i], ref_codes[i]);
                    ++shown_codes;
                }
            }
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
        DeviceBuffer gate_up_tap(slots * 2 * I * 2), y_tap(slots * I * 2);
        taps.gate_up = gate_up_tap.p;
        taps.y       = y_tap.p;
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
        // Stage localisation: gate_up vs the capture, y vs the SiTU of the captured gate_up, down on equal y_q rows.
        const auto got_gu = from_device<std::uint16_t>(gate_up_tap, slots * 2 * I);
        const auto ref_gu = st.vec<std::uint16_t>("gate_up");
        int gu_off = 0, gu_shown = 0, gu_max_ulp = 0;
        for (std::size_t i = 0; i < got_gu.size(); ++i) {
            if (got_gu[i] != ref_gu[i]) {
                ++gu_off;
                const int ulp = std::abs(static_cast<int>(static_cast<std::int16_t>(got_gu[i])) -
                                         static_cast<int>(static_cast<std::int16_t>(ref_gu[i])));
                gu_max_ulp = std::max(gu_max_ulp, ulp);
                if (gu_shown++ < 6) {
                    std::printf("    gate_up differs: slot %zu col %zu ours %.8g ref %.8g\n", i / (2 * I), i % (2 * I),
                                bf16_to_f32(got_gu[i]), bf16_to_f32(ref_gu[i]));
                }
            }
        }
        const auto got_y = from_device<std::uint16_t>(y_tap, slots * I);
        const int y_off  = count_mismatch(got_y, y_sub);
        const auto got_yc = from_device<std::uint8_t>(y_codes, y_codes.bytes);
        const auto ref_yc = st.vec<std::uint8_t>("y_codes");
        const auto got_down = from_device<std::uint16_t>(down, slots * H);
        const auto ref_down = st.vec<std::uint16_t>("down");
        const std::size_t code_row = nv ? I / 2 : I;
        int down_off_clean = 0;
        std::size_t clean_rows = 0;
        for (std::size_t r = 0; r < slots; ++r) {
            if (!std::equal(got_yc.begin() + r * code_row, got_yc.begin() + (r + 1) * code_row, ref_yc.begin() + r * code_row)) { continue; }
            ++clean_rows;
            for (int n = 0; n < H; ++n) { down_off_clean += got_down[r * H + n] != ref_down[r * H + n]; }
        }
        std::printf("  stages: gate_up elements off %d (max %d bf16 ulp); y (op) vs SiTU(captured gate_up) off %d; "
                    "down elements off on the %zu rows with equal y_q codes: %d\n",
                    gu_off, gu_max_ulp, y_off, clean_rows, down_off_clean);
        failures += report("capture: gate_up GEMM rows bit-exact", gu_off);
        failures += report("capture: down GEMM bit-exact on rows with equal y_q", down_off_clean);
        failures += report("capture: gate_up GEMM + SiTU + requant (y_q) bit-exact", bad_y);
        failures += report("capture: down GEMM rows bit-exact", bad_down);
        failures += report("capture: combined output bit-exact", bad_out);
    }
    // 5. Narrow rounds on the real weights (fused route with the persistent state, and without): every row ==
    //    our T = 256 row; NVFP4 also == vLLM's captured output rows.
    {
        DeviceBuffer full(static_cast<std::size_t>(tokens) * H * 2);
        DeviceArena arena(ops::inkling_moe_workspace_bytes(format, tokens, false));
        Tensor tfull(full.p, DType::BF16, {H, tokens});
        ops::inkling_moe(tx, l.view, tfull, arena, nullptr);
        cuda_synchronize();
        const auto ours = from_device<std::uint16_t>(full, static_cast<std::size_t>(tokens) * H);
        const auto ref  = st.vec<std::uint16_t>("out");
        int off_ours = 0, off_ref = 0, dirty = 0;
        for (const auto& [first, count] : std::vector<std::pair<int, int>>{{0, 1}, {100, 1}, {tokens - 1, 1}, {8, 8}, {64, 64}, {190, 24}}) {
            for (const bool state : {true, false}) {
                DeviceBuffer out(static_cast<std::size_t>(count) * H * 2);
                DeviceArena small(ops::inkling_moe_workspace_bytes(format, count, false));
                Tensor tin(static_cast<std::uint16_t*>(dx.p) + static_cast<std::size_t>(first) * H, DType::BF16, {H, count});
                Tensor tout(out.p, DType::BF16, {H, count});
                ops::InklingMoeOptions options;
                options.state = state ? persistent_state() : nullptr;
                ops::inkling_moe(tin, l.view, tout, small, nullptr, options);
                cuda_synchronize();
                const auto rows = from_device<std::uint16_t>(out, static_cast<std::size_t>(count) * H);
                for (std::size_t i = 0; i < rows.size(); ++i) {
                    off_ours += rows[i] != ours[static_cast<std::size_t>(first) * H + i];
                    off_ref += rows[i] != ref[static_cast<std::size_t>(first) * H + i];
                }
                dirty += state_nonzero_words();
            }
        }
        std::printf("  narrow rows (T = 1 / 8 / 24 / 64, fused route and not) vs our T = 256: off %d; vs vLLM: off %d; "
                    "state words left %d\n", off_ours, off_ref, dirty);
        failures += report("capture: narrow rows (fused route + plain) == T=256 rows (bit-exact), state clean",
                           off_ours + dirty);
        if (nv) { failures += report("capture: narrow rows == vLLM output rows (bit-exact)", off_ref); }
    }
    // The trainer's dX operand decoded from the REAL bank (checkpoint scales / amax) vs the host decode.
    failures += test_decode_units(l, nv ? "NVFP4 real layer" : "MXFP8 real layer");
    return failures;
}

/// Timing on real weights (M3, after parity): one CUDA-graph-captured call per T (PDL edges inside the graph),
/// replayed; reports us / call, distinct experts touched, the bytes they imply and the share of 1,707 GB/s (lane h).
int run_bench(const char* path) {
    const SafeTensors st = read_safetensors(path);
    const bool nv = st.meta.at("format") == "nvfp4";
    const int captured = std::stoi(st.meta.at("tokens"));
    Layer l;
    load_capture_layer(st, nv, l);
    const auto x_bits = st.vec<std::uint16_t>("x");
    // INKLING_BENCH_COPIES=n (default 1): n device copies of the layer's weights, one graph each, replayed
    // round-robin, so a decode call streams its experts from DRAM as in serving (one copy stays L2-resident:
    // 96 MB L2 vs 36 MB per T = 1 call).
    int copies = 1;
    if (const char* value = std::getenv("INKLING_BENCH_COPIES"); value != nullptr && std::atoi(value) > 1) {
        copies = std::atoi(value);
    }
    std::vector<DeviceBuffer> copy_buffers;
    std::vector<ops::InklingMoeWeights> views{l.view};
    const auto router_bits = st.vec<std::uint16_t>("router");
    for (int c = 1; c < copies; ++c) {
        ops::InklingMoeWeights v = l.view;
        copy_buffers.push_back(to_device(router_bits));
        v.router = copy_buffers.back().p;
        for (int p = 0; p < 2; ++p) {
            ops::InklingExpertBank& bank = p == 0 ? v.w13 : v.w2;
            copy_buffers.push_back(to_device(l.codes[p]));
            bank.codes = static_cast<const std::uint8_t*>(copy_buffers.back().p);
            copy_buffers.push_back(to_device(l.scales[p]));
            bank.scales = static_cast<const std::uint8_t*>(copy_buffers.back().p);
        }
        views.push_back(v);
    }
    const double expert_bytes = nv ? 4478976.0 : 8211456.0;
    const double router_bytes = 589824.0;
    std::printf("[bench %s: layer %s %s, %d weight cop%s]\n", path, st.meta.at("layer").c_str(), nv ? "NVFP4" : "MXFP8",
                copies, copies == 1 ? "y (L2-warm at small T)" : "ies (DRAM-cold)");
    std::printf("  %6s %10s %9s %10s %9s %7s\n", "T", "us/call", "distinct", "MB", "GB/s", "of BW");
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    // INKLING_BENCH_T="1,8,4096" selects widths (nsys sessions); default: the full sweep.
    std::vector<int> widths{1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096};
    if (const char* list = std::getenv("INKLING_BENCH_T"); list != nullptr && *list != '\0') {
        widths.clear();
        for (const char* p = list; *p != '\0';) {
            widths.push_back(std::atoi(p));
            while (*p != '\0' && *p != ',') { ++p; }
            if (*p == ',') { ++p; }
        }
    }
    for (const int tokens : widths) {
        std::vector<std::uint16_t> xs(static_cast<std::size_t>(tokens) * H);
        for (int t = 0; t < tokens; ++t) {
            std::copy(x_bits.begin() + static_cast<std::size_t>(t % captured) * H,
                      x_bits.begin() + static_cast<std::size_t>(t % captured + 1) * H, xs.begin() + static_cast<std::size_t>(t) * H);
        }
        DeviceBuffer dx = to_device(xs), dout(static_cast<std::size_t>(tokens) * H * 2);
        DeviceArena arena(ops::inkling_moe_workspace_bytes(l.format, tokens, false));
        Tensor tx(dx.p, DType::BF16, {H, tokens}), tout(dout.p, DType::BF16, {H, tokens});
        // distinct experts of this round
        DeviceBuffer dl(static_cast<std::size_t>(tokens) * E * 4), di(static_cast<std::size_t>(tokens) * TOPK * 4),
            dw(static_cast<std::size_t>(tokens) * TOPK * 4);
        Tensor tl(dl.p, DType::FP32, {E, tokens}), ti(di.p, DType::I32, {TOPK, tokens}), tw(dw.p, DType::FP32, {TOPK, tokens});
        ops::inkling_route(tx, l.view, tl, ti, tw, stream);
        cuda_synchronize(stream);
        const auto ids = from_device<int>(di, static_cast<std::size_t>(tokens) * TOPK);
        std::vector<bool> used(E, false);
        for (int id : ids) { used[id] = true; }
        const int distinct = static_cast<int>(std::count(used.begin(), used.end(), true));
        // graph of one call (with the persistent state unless INKLING_BENCH_NOSTATE=1: the A/B of the fused route)
        ops::InklingMoeOptions bench_options;
        bench_options.state = std::getenv("INKLING_BENCH_NOSTATE") != nullptr ? nullptr : persistent_state();
        // INKLING_BENCH_PRED_US=n: each graph starts with a 1-CTA kernel that triggers its PDL dependents at entry
        // and spins n us (the engine's predecessors trigger early since e1 f9ebe0eb), so the route launches early
        // as in serving. Reported: us/call - n.
        const char* pred_text  = std::getenv("INKLING_BENCH_PRED_US");
        const double pred_us   = pred_text != nullptr ? std::atof(pred_text) : 0.0;
        const std::int32_t pred_ns = static_cast<std::int32_t>(pred_us * 1000.0);
        ops::inkling_moe(tx, l.view, tout, arena, stream, bench_options);
        arena.reset();
        cuda_synchronize(stream);
        std::vector<cudaGraph_t> graphs(views.size(), nullptr);
        std::vector<cudaGraphExec_t> execs(views.size(), nullptr);
        for (std::size_t c = 0; c < views.size(); ++c) {
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
            if (pred_text != nullptr) { ops::inkling_bench_predecessor(pred_ns, stream); }
            ops::inkling_moe(tx, views[c], tout, arena, stream, bench_options);
            arena.reset();
            cuda_check(cudaStreamEndCapture(stream, &graphs[c]), "end capture");
            cuda_check(cudaGraphInstantiate(&execs[c], graphs[c], 0), "instantiate");
        }
        const int iterations = tokens <= 64 ? 400 : (tokens <= 1024 ? 50 : 10);
        for (int i = 0; i < 10; ++i) { cuda_check(cudaGraphLaunch(execs[i % execs.size()], stream), "warmup"); }
        cudaEvent_t start, stop;
        cuda_check(cudaEventCreate(&start), "event");
        cuda_check(cudaEventCreate(&stop), "event");
        cuda_check(cudaEventRecord(start, stream), "record");
        for (int i = 0; i < iterations; ++i) { cuda_check(cudaGraphLaunch(execs[i % execs.size()], stream), "launch"); }
        cuda_check(cudaEventRecord(stop, stream), "record");
        cuda_check(cudaEventSynchronize(stop), "sync");
        float ms = 0.0F;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        const double us    = 1000.0 * ms / iterations - pred_us;
        const double bytes = distinct * expert_bytes + router_bytes;
        std::printf("  %6d %10.2f %9d %10.2f %9.1f %6.1f%%\n", tokens, us, distinct, bytes / 1e6, bytes / us / 1e3,
                    100.0 * (bytes / us / 1e3) / 1707.0);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        for (std::size_t c = 0; c < views.size(); ++c) {
            cudaGraphExecDestroy(execs[c]);
            cudaGraphDestroy(graphs[c]);
        }
    }
    cudaStreamDestroy(stream);
    return 0;
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
    if (argc == 3 && std::string(argv[1]) == "--bench") { return run_bench(argv[2]); }
    if ((argc == 3 || argc == 5) && std::string(argv[1]) == "--capture") {
        const char* dump = argc == 5 && std::string(argv[3]) == "--dump" ? argv[4] : nullptr;
        failures = run_capture(argv[2], dump);
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
