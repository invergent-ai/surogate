// CPU expert compute: the planar-W8 gated FFN on the host against a double-precision reference
// of the same quantised arithmetic, single job and pooled rounds (no GPU needed).
#include "api/ops/cpu_expert_compute.h"

#include "ops/linear/ggml/ggml_host_decode.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace sinfer;

namespace {

constexpr ops::SparseMoeGeometry kGeometry{128, 4, 2, 128}; // hidden 128, 4 experts, top-2, ffn 128 (16-row chunks: the tile path runs)

std::uint16_t float_to_fp16(float f) {
    std::uint32_t w;
    std::memcpy(&w, &f, sizeof(w));
    const std::uint32_t sign = (w >> 16) & 0x8000U;
    const int exp            = static_cast<int>((w >> 23) & 0xFFU) - 127 + 15;
    std::uint32_t mant       = w & 0x7FFFFFU;
    if (exp <= 0) { return static_cast<std::uint16_t>(sign); }
    if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7C00U); }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
}
float fp16_to_float(std::uint16_t h) {
    const std::uint32_t sign = (h >> 15) & 1U, exp = (h >> 10) & 0x1FU, mant = h & 0x3FFU;
    if (exp == 0) { return sign ? -0.0F : 0.0F; }
    std::uint32_t w = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &w, sizeof(f));
    return f;
}
std::uint16_t float_to_bf16(float f) {
    std::uint32_t w;
    std::memcpy(&w, &f, sizeof(w));
    return static_cast<std::uint16_t>((w + 0x8000U) >> 16);
}
float bf16_to_float(std::uint16_t b) {
    std::uint32_t w = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &w, sizeof(f));
    return f;
}

struct Bank {
    std::vector<std::int8_t> gate_codes, down_codes;
    std::vector<std::uint16_t> gate_scales, down_scales;
    ops::CpuExpertBank view() const {
        ops::CpuExpertBank bank;
        bank.gate_up_codes  = reinterpret_cast<const std::byte*>(gate_codes.data());
        bank.gate_up_scales = reinterpret_cast<const std::byte*>(gate_scales.data());
        bank.down_codes     = reinterpret_cast<const std::byte*>(down_codes.data());
        bank.down_scales    = reinterpret_cast<const std::byte*>(down_scales.data());
        return bank;
    }
};

Bank make_bank(std::mt19937& rng) {
    const int H = kGeometry.hidden, I = kGeometry.intermediate, E = kGeometry.experts;
    Bank b;
    b.gate_codes.resize(static_cast<std::size_t>(E) * 2 * I * H);
    b.gate_scales.resize(static_cast<std::size_t>(E) * 2 * I * (H / 32));
    b.down_codes.resize(static_cast<std::size_t>(E) * H * I);
    b.down_scales.resize(static_cast<std::size_t>(E) * H * (I / 32));
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> scale(0.002F, 0.02F);
    for (auto& c : b.gate_codes) { c = static_cast<std::int8_t>(code(rng)); }
    for (auto& c : b.down_codes) { c = static_cast<std::int8_t>(code(rng)); }
    for (auto& s : b.gate_scales) { s = float_to_fp16(scale(rng)); }
    for (auto& s : b.down_scales) { s = float_to_fp16(scale(rng)); }
    return b;
}

// The clamped SwiGLU, as the kernels apply it: both halves bounded before the product where
// the geometry states a limit (GLM-5.3: 10), the plain product otherwise.
double swiglu_ref(double g, double u, double limit) {
    if (limit > 0.0) {
        g = std::min(g, limit);
        u = std::min(std::max(u, -limit), limit);
    }
    return (g / (1.0 + std::exp(-g))) * u;
}

// Reference: identical quantisation rules in double precision.
void quantise_ref(const std::vector<double>& v, std::vector<int>& q, std::vector<double>& s) {
    const int groups = static_cast<int>(v.size()) / 32;
    q.assign(v.size(), 0);
    s.assign(static_cast<std::size_t>(groups), 0.0);
    for (int g = 0; g < groups; ++g) {
        double amax = 0.0;
        for (int i = 0; i < 32; ++i) { amax = std::max(amax, std::fabs(v[g * 32 + i])); }
        const float scale = static_cast<float>(amax) / 127.0F; // float, as the op computes it
        const float inv   = scale > 0.0F ? 1.0F / scale : 0.0F;
        s[g]              = scale;
        for (int i = 0; i < 32; ++i) {
            const float r = std::nearbyint(static_cast<float>(v[g * 32 + i]) * inv);
            q[g * 32 + i]  = static_cast<int>(std::max(-127.0F, std::min(127.0F, r)));
        }
    }
}
double dot_ref(const std::int8_t* codes, const std::uint16_t* scales, const std::vector<int>& q,
               const std::vector<double>& s, int k) {
    double acc = 0.0;
    for (int g = 0; g < k / 32; ++g) {
        long dot = 0;
        for (int i = 0; i < 32; ++i) { dot += static_cast<long>(codes[g * 32 + i]) * q[g * 32 + i]; }
        acc += static_cast<double>(dot) * fp16_to_float(scales[g]) * s[g];
    }
    return acc;
}
std::vector<double> reference_job(const Bank& b, int expert, const std::vector<std::uint16_t>& x, float weight,
                                  double limit = 0.0) {
    const int H = kGeometry.hidden, I = kGeometry.intermediate;
    std::vector<double> xf(H);
    for (int i = 0; i < H; ++i) { xf[i] = bf16_to_float(x[i]); }
    std::vector<int> xq;
    std::vector<double> xs;
    quantise_ref(xf, xq, xs);
    const std::int8_t* gc     = b.gate_codes.data() + static_cast<std::size_t>(expert) * 2 * I * H;
    const std::uint16_t* gs   = b.gate_scales.data() + static_cast<std::size_t>(expert) * 2 * I * (H / 32);
    std::vector<double> h(I);
    for (int j = 0; j < I; ++j) {
        const double g = dot_ref(gc + j * H, gs + j * (H / 32), xq, xs, H);
        const double u = dot_ref(gc + (I + j) * H, gs + (I + j) * (H / 32), xq, xs, H);
        h[j]           = swiglu_ref(g, u, limit);
    }
    std::vector<int> hq;
    std::vector<double> hs;
    quantise_ref(h, hq, hs);
    const std::int8_t* dc   = b.down_codes.data() + static_cast<std::size_t>(expert) * H * I;
    const std::uint16_t* ds = b.down_scales.data() + static_cast<std::size_t>(expert) * H * (I / 32);
    std::vector<double> y(H);
    for (int r = 0; r < H; ++r) { y[r] = weight * dot_ref(dc + r * I, ds + r * (I / 32), hq, hs, I); }
    return y;
}

// --- Q4G32AM bank: requantised from the W8 bank; the reference decodes the affine grid. ---
struct Q4Bank {
    std::vector<std::uint8_t> gate_codes, down_codes;
    std::vector<std::uint16_t> gate_scales, gate_mins, down_scales, down_mins;
    ops::CpuExpertBank view() const {
        ops::CpuExpertBank bank;
        bank.gate_up_format = ops::ExpertBankFormat::Q4G32AM;
        bank.down_format    = ops::ExpertBankFormat::Q4G32AM;
        bank.gate_up_codes  = reinterpret_cast<const std::byte*>(gate_codes.data());
        bank.gate_up_scales = reinterpret_cast<const std::byte*>(gate_scales.data());
        bank.gate_up_mins   = reinterpret_cast<const std::byte*>(gate_mins.data());
        bank.down_codes     = reinterpret_cast<const std::byte*>(down_codes.data());
        bank.down_scales    = reinterpret_cast<const std::byte*>(down_scales.data());
        bank.down_mins      = reinterpret_cast<const std::byte*>(down_mins.data());
        return bank;
    }
};

Q4Bank requantise_bank(const Bank& b) {
    Q4Bank q;
    const auto gate_groups = static_cast<std::int64_t>(b.gate_scales.size());
    const auto down_groups = static_cast<std::int64_t>(b.down_scales.size());
    q.gate_codes.resize(static_cast<std::size_t>(gate_groups) * 16);
    q.gate_scales.resize(b.gate_scales.size());
    q.gate_mins.resize(b.gate_scales.size());
    q.down_codes.resize(static_cast<std::size_t>(down_groups) * 16);
    q.down_scales.resize(b.down_scales.size());
    q.down_mins.resize(b.down_scales.size());
    ops::requantise_w8_expert_groups_to_q4(b.gate_codes.data(), b.gate_scales.data(), gate_groups,
                                           q.gate_codes.data(), q.gate_scales.data(),
                                           q.gate_mins.data());
    ops::requantise_w8_expert_groups_to_q4(b.down_codes.data(), b.down_scales.data(), down_groups,
                                           q.down_codes.data(), q.down_scales.data(),
                                           q.down_mins.data());
    return q;
}

double dot_ref_q4(const std::uint8_t* q4, const std::uint16_t* scales, const std::uint16_t* mins,
                  const std::vector<int>& xq, const std::vector<double>& xs, int k) {
    double acc = 0.0;
    for (int g = 0; g < k / 32; ++g) {
        long dot = 0, sum = 0;
        for (int i = 0; i < 16; ++i) {
            const int q0 = q4[g * 16 + i] & 0x0F;
            const int q1 = q4[g * 16 + i] >> 4;
            dot += static_cast<long>(q0) * xq[g * 32 + 2 * i] +
                   static_cast<long>(q1) * xq[g * 32 + 2 * i + 1];
            sum += xq[g * 32 + 2 * i] + xq[g * 32 + 2 * i + 1];
        }
        acc += xs[g] * (static_cast<double>(fp16_to_float(scales[g])) * static_cast<double>(dot) +
                        static_cast<double>(fp16_to_float(mins[g])) * static_cast<double>(sum));
    }
    return acc;
}

std::vector<double> reference_job_q4(const Q4Bank& b, int expert,
                                     const std::vector<std::uint16_t>& x, float weight,
                                     double limit = 0.0) {
    const int H = kGeometry.hidden, I = kGeometry.intermediate;
    std::vector<double> xf(H);
    for (int i = 0; i < H; ++i) { xf[i] = bf16_to_float(x[i]); }
    std::vector<int> xq;
    std::vector<double> xs;
    quantise_ref(xf, xq, xs);
    const std::uint8_t* gc  = b.gate_codes.data() + static_cast<std::size_t>(expert) * 2 * I * (H / 2);
    const std::uint16_t* gs = b.gate_scales.data() + static_cast<std::size_t>(expert) * 2 * I * (H / 32);
    const std::uint16_t* gm = b.gate_mins.data() + static_cast<std::size_t>(expert) * 2 * I * (H / 32);
    std::vector<double> h(I);
    for (int j = 0; j < I; ++j) {
        const double g = dot_ref_q4(gc + j * (H / 2), gs + j * (H / 32), gm + j * (H / 32), xq, xs, H);
        const double u = dot_ref_q4(gc + (I + j) * (H / 2), gs + (I + j) * (H / 32),
                                    gm + (I + j) * (H / 32), xq, xs, H);
        h[j] = swiglu_ref(g, u, limit);
    }
    std::vector<int> hq;
    std::vector<double> hs;
    quantise_ref(h, hq, hs);
    const std::uint8_t* dc  = b.down_codes.data() + static_cast<std::size_t>(expert) * H * (I / 2);
    const std::uint16_t* ds = b.down_scales.data() + static_cast<std::size_t>(expert) * H * (I / 32);
    const std::uint16_t* dm = b.down_mins.data() + static_cast<std::size_t>(expert) * H * (I / 32);
    std::vector<double> y(H);
    for (int r = 0; r < H; ++r) {
        y[r] = weight * dot_ref_q4(dc + r * (I / 2), ds + r * (I / 32), dm + r * (I / 32), hq, hs, I);
    }
    return y;
}

// A bank whose halves differ: gate/up as Q4G32AM, down as W8 -- the shape a K_XL mixture's
// bank takes when only its 4-bit halves are kept as 4-bit planes.
std::vector<double> reference_job_mixed(const Q4Bank& q, const Bank& b, int expert,
                                        const std::vector<std::uint16_t>& x, float weight,
                                        double limit = 0.0) {
    const int H = kGeometry.hidden, I = kGeometry.intermediate;
    std::vector<double> xf(H);
    for (int i = 0; i < H; ++i) { xf[i] = bf16_to_float(x[i]); }
    std::vector<int> xq;
    std::vector<double> xs;
    quantise_ref(xf, xq, xs);
    const std::uint8_t* gc  = q.gate_codes.data() + static_cast<std::size_t>(expert) * 2 * I * (H / 2);
    const std::uint16_t* gs = q.gate_scales.data() + static_cast<std::size_t>(expert) * 2 * I * (H / 32);
    const std::uint16_t* gm = q.gate_mins.data() + static_cast<std::size_t>(expert) * 2 * I * (H / 32);
    std::vector<double> h(I);
    for (int j = 0; j < I; ++j) {
        const double g = dot_ref_q4(gc + j * (H / 2), gs + j * (H / 32), gm + j * (H / 32), xq, xs, H);
        const double u = dot_ref_q4(gc + (I + j) * (H / 2), gs + (I + j) * (H / 32),
                                    gm + (I + j) * (H / 32), xq, xs, H);
        h[j] = swiglu_ref(g, u, limit);
    }
    std::vector<int> hq;
    std::vector<double> hs;
    quantise_ref(h, hq, hs);
    const std::int8_t* dc   = b.down_codes.data() + static_cast<std::size_t>(expert) * H * I;
    const std::uint16_t* ds = b.down_scales.data() + static_cast<std::size_t>(expert) * H * (I / 32);
    std::vector<double> y(H);
    for (int r = 0; r < H; ++r) { y[r] = weight * dot_ref(dc + r * I, ds + r * (I / 32), hq, hs, I); }
    return y;
}

// --- Q5G32AM bank: random planes (the format is its own source of truth here; the repack from
// GGML blocks is checked separately against the codec). Per group: 16 bytes of pairwise
// nibbles, 4 bytes whose bit v is value v's fifth bit, then FP16 scale and min.
struct Q5Bank {
    std::vector<std::uint8_t> gate_codes, down_codes; // 20 bytes per group
    std::vector<std::uint16_t> gate_scales, gate_mins, down_scales, down_mins;
    ops::CpuExpertBank view() const {
        ops::CpuExpertBank bank;
        bank.gate_up_format = ops::ExpertBankFormat::Q5G32AM;
        bank.down_format    = ops::ExpertBankFormat::Q5G32AM;
        bank.gate_up_codes  = reinterpret_cast<const std::byte*>(gate_codes.data());
        bank.gate_up_scales = reinterpret_cast<const std::byte*>(gate_scales.data());
        bank.gate_up_mins   = reinterpret_cast<const std::byte*>(gate_mins.data());
        bank.down_codes     = reinterpret_cast<const std::byte*>(down_codes.data());
        bank.down_scales    = reinterpret_cast<const std::byte*>(down_scales.data());
        bank.down_mins      = reinterpret_cast<const std::byte*>(down_mins.data());
        return bank;
    }
};

Q5Bank make_q5_bank(std::mt19937& rng) {
    const int H = kGeometry.hidden, I = kGeometry.intermediate, E = kGeometry.experts;
    const std::size_t gate_groups = static_cast<std::size_t>(E) * 2 * I * (H / 32);
    const std::size_t down_groups = static_cast<std::size_t>(E) * H * (I / 32);
    Q5Bank b;
    b.gate_codes.resize(gate_groups * 20);
    b.down_codes.resize(down_groups * 20);
    b.gate_scales.resize(gate_groups);
    b.gate_mins.resize(gate_groups);
    b.down_scales.resize(down_groups);
    b.down_mins.resize(down_groups);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002F, 0.02F);
    std::uniform_real_distribution<float> minimum(-0.3F, 0.0F);
    for (auto& c : b.gate_codes) { c = static_cast<std::uint8_t>(byte(rng)); }
    for (auto& c : b.down_codes) { c = static_cast<std::uint8_t>(byte(rng)); }
    for (auto& v : b.gate_scales) { v = float_to_fp16(scale(rng)); }
    for (auto& v : b.down_scales) { v = float_to_fp16(scale(rng)); }
    for (auto& v : b.gate_mins) { v = float_to_fp16(minimum(rng)); }
    for (auto& v : b.down_mins) { v = float_to_fp16(minimum(rng)); }
    return b;
}

int q5_code(const std::uint8_t* group, int v) {
    std::uint32_t high;
    std::memcpy(&high, group + 16, sizeof(high));
    const int low = (group[v / 2] >> (4 * (v % 2))) & 0xF;
    return low | static_cast<int>((high >> v) & 1U) << 4;
}

double dot_ref_q5(const std::uint8_t* q5, const std::uint16_t* scales, const std::uint16_t* mins,
                  const std::vector<int>& xq, const std::vector<double>& xs, int k) {
    double acc = 0.0;
    for (int g = 0; g < k / 32; ++g) {
        long dot = 0, sum = 0;
        for (int v = 0; v < 32; ++v) {
            dot += static_cast<long>(q5_code(q5 + g * 20, v)) * xq[g * 32 + v];
            sum += xq[g * 32 + v];
        }
        acc += xs[g] * (static_cast<double>(fp16_to_float(scales[g])) * static_cast<double>(dot) +
                        static_cast<double>(fp16_to_float(mins[g])) * static_cast<double>(sum));
    }
    return acc;
}

std::vector<double> reference_job_q5(const Q5Bank& b, int expert,
                                     const std::vector<std::uint16_t>& x, float weight,
                                     double limit = 0.0) {
    const int H = kGeometry.hidden, I = kGeometry.intermediate;
    std::vector<double> xf(H);
    for (int i = 0; i < H; ++i) { xf[i] = bf16_to_float(x[i]); }
    std::vector<int> xq;
    std::vector<double> xs;
    quantise_ref(xf, xq, xs);
    const std::size_t gh = static_cast<std::size_t>(H / 32), gi = static_cast<std::size_t>(I / 32);
    const std::uint8_t* gc  = b.gate_codes.data() + static_cast<std::size_t>(expert) * 2 * I * gh * 20;
    const std::uint16_t* gs = b.gate_scales.data() + static_cast<std::size_t>(expert) * 2 * I * gh;
    const std::uint16_t* gm = b.gate_mins.data() + static_cast<std::size_t>(expert) * 2 * I * gh;
    std::vector<double> h(I);
    for (int j = 0; j < I; ++j) {
        const double g = dot_ref_q5(gc + j * gh * 20, gs + j * gh, gm + j * gh, xq, xs, H);
        const double u = dot_ref_q5(gc + (I + j) * gh * 20, gs + (I + j) * gh, gm + (I + j) * gh, xq, xs, H);
        h[j] = swiglu_ref(g, u, limit);
    }
    std::vector<int> hq;
    std::vector<double> hs;
    quantise_ref(h, hq, hs);
    const std::uint8_t* dc  = b.down_codes.data() + static_cast<std::size_t>(expert) * H * gi * 20;
    const std::uint16_t* ds = b.down_scales.data() + static_cast<std::size_t>(expert) * H * gi;
    const std::uint16_t* dm = b.down_mins.data() + static_cast<std::size_t>(expert) * H * gi;
    std::vector<double> y(H);
    for (int r = 0; r < H; ++r) {
        y[r] = weight * dot_ref_q5(dc + r * gi * 20, ds + r * gi, dm + r * gi, hq, hs, I);
    }
    return y;
}

int compare(const std::string& label, const std::vector<float>& got, const std::vector<double>& want) {
    double max_abs = 0.0, max_ref = 0.0, err2 = 0.0, ref2 = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double d = static_cast<double>(got[i]) - want[i];
        max_abs        = std::max(max_abs, std::fabs(d));
        max_ref        = std::max(max_ref, std::fabs(want[i]));
        err2 += d * d;
        ref2 += want[i] * want[i];
    }
    const double rel = std::sqrt(err2) / std::max(std::sqrt(ref2), 1e-30);
    const bool ok    = rel < 2.0e-5 && max_abs <= 1.0e-4 * std::max(max_ref, 1.0);
    std::cout << (ok ? "ok   " : "FAIL ") << label << " rel-L2 " << rel << " max|err| " << max_abs << "\n";
    return ok ? 0 : 1;
}

} // namespace

int main() {
    std::mt19937 rng(20260828U);
    const Bank bank = make_bank(rng);
    const int H     = kGeometry.hidden;
    std::uniform_real_distribution<float> act(-2.0F, 2.0F);
    int failures = 0;
    std::cout << "avx512 path: " << (ops::cpu_expert_compute_has_avx512() ? "yes" : "no (scalar)") << ", vnni: " << (ops::cpu_expert_compute_has_vnni() ? "yes" : "no") << ", tile: " << (ops::cpu_expert_compute_has_tile() ? "yes" : "no") << "\n";

    // Single job vs reference.
    std::vector<std::uint16_t> x(H);
    for (auto& v : x) { v = float_to_bf16(act(rng)); }
    std::vector<float> out(H, 0.0F);
    std::vector<std::byte> scratch(ops::cpu_expert_scratch_bytes(kGeometry) + 64);
    const ops::CpuExpertJob job{0, 2, 0.75F};
    ops::cpu_expert_compute_job(kGeometry, bank.view(), job, x.data(), out.data(), scratch.data());
    failures += compare("single job expert 2", out, reference_job(bank, 2, x, 0.75F));

    // A pooled round: 3 tokens, top-2 each (6 jobs), accumulated per token; compare each column.
    const int tokens = 3;
    std::vector<std::uint16_t> xs(static_cast<std::size_t>(H) * tokens);
    for (auto& v : xs) { v = float_to_bf16(act(rng)); }
    std::vector<ops::CpuExpertJob> jobs = {{0, 0, 0.6F}, {0, 3, 0.4F}, {1, 1, 0.5F},
                                           {1, 2, 0.5F}, {2, 3, 0.9F}, {2, 0, 0.1F}};
    std::vector<float> outs(static_cast<std::size_t>(H) * tokens, 0.0F);
    ops::CpuExpertPool pool(kGeometry, {.threads = 4, .pin_threads = false});
    ops::CpuExpertRound round{xs.data(), outs.data(), tokens, jobs};
    pool.run(bank.view(), round);
    pool.run(bank.view(), round); // second run doubles every column (accumulation semantics)
    for (int t = 0; t < tokens; ++t) {
        std::vector<std::uint16_t> column(xs.begin() + t * H, xs.begin() + (t + 1) * H);
        std::vector<double> want(H, 0.0);
        for (const auto& j : jobs) {
            if (j.token != t) { continue; }
            const std::vector<double> y = reference_job(bank, j.expert, column, j.weight);
            for (int i = 0; i < H; ++i) { want[i] += 2.0 * y[i]; }
        }
        std::vector<float> got(outs.begin() + t * H, outs.begin() + (t + 1) * H);
        failures += compare("pooled round token " + std::to_string(t), got, want);
    }
    // A prefill-shaped round: 13 tokens over the 4 experts with uneven, odd-sized groups (the
    // grouped path pairs tokens two at a time and must handle the trailing one).
    {
        const int many = 13;
        std::vector<std::uint16_t> xm(static_cast<std::size_t>(H) * many);
        for (auto& v : xm) { v = float_to_bf16(act(rng)); }
        std::vector<ops::CpuExpertJob> jm;
        std::uniform_real_distribution<float> wdist(0.05F, 0.95F);
        for (int t2 = 0; t2 < many; ++t2) {
            jm.push_back({t2, t2 % 4, wdist(rng)});
            jm.push_back({t2, (t2 * 7 + 1) % 4, wdist(rng)});
            if (t2 % 3 == 0) { jm.push_back({t2, 3, wdist(rng)}); } // expert 3 gets extra, odd count
        }
        std::vector<float> om(static_cast<std::size_t>(H) * many, 0.0F);
        ops::CpuExpertPool pool6(kGeometry, {.threads = 6, .pin_threads = false});
        ops::CpuExpertRound rm{xm.data(), om.data(), many, jm};
        pool6.run(bank.view(), rm);
        int bad = 0;
        for (int t2 = 0; t2 < many; ++t2) {
            std::vector<std::uint16_t> column(xm.begin() + t2 * H, xm.begin() + (t2 + 1) * H);
            std::vector<double> want(H, 0.0);
            for (const auto& j : jm) {
                if (j.token != t2) { continue; }
                const std::vector<double> y = reference_job(bank, j.expert, column, j.weight);
                for (int i = 0; i < H; ++i) { want[i] += y[i]; }
            }
            std::vector<float> got(om.begin() + t2 * H, om.begin() + (t2 + 1) * H);
            bad += compare("grouped round token " + std::to_string(t2), got, want);
        }
        failures += bad;
    }
    // --- GGML-block bank -------------------------------------------------------------------
    // The blocks are the bank; the CPU path decodes a chunk of rows at a time into W8 staging
    // and reads them at an offset. That offset arithmetic, not the decoders (which
    // test-ggml-k owns), is what this checks: the single-job reference walks whole rows and
    // the pool walks chunks, so agreeing means the chunking is right. Random bytes are a legal
    // block of every format, which is what lets one loop cover the vocabulary.
    {
        // A K-quant superblock is 256 values, so this section needs a geometry whose rows are
        // a whole number of blocks -- the 128-wide one above cannot hold one.
        constexpr ops::SparseMoeGeometry kBlockGeometry{256, 4, 2, 256};
        const int H = kBlockGeometry.hidden, I = kBlockGeometry.intermediate,
                  E = kBlockGeometry.experts;
        std::vector<std::byte> gscratch(ops::cpu_expert_scratch_bytes(kBlockGeometry) + 64);
        for (const auto& [name, type] : std::vector<std::pair<const char*, QType>>{
                 {"Q4_K", QType::Q4_K}, {"Q5_K", QType::Q5_K}, {"Q6_K", QType::Q6_K},
                 {"Q2_K", QType::Q2_K}, {"Q3_K", QType::Q3_K}, {"Q8_0", QType::Q8_0},
                 {"Q4_1", QType::Q4_1}, {"Q5_1", QType::Q5_1}, {"IQ4_NL", QType::IQ4_NL},
                 {"Q4_0", QType::Q4_0}, {"Q5_0", QType::Q5_0},
          {"IQ2_XXS", QType::IQ2_XXS},
          {"IQ2_XS", QType::IQ2_XS},
          {"IQ2_S", QType::IQ2_S},
          {"IQ3_XXS", QType::IQ3_XXS},
          {"IQ3_S", QType::IQ3_S},
          {"IQ1_S", QType::IQ1_S},
          {"IQ1_M", QType::IQ1_M},
          {"IQ4_XS", QType::IQ4_XS},
          {"TQ1_0", QType::TQ1_0},
          {"TQ2_0", QType::TQ2_0},
          {"MXFP4", QType::MXFP4},
          {"NVFP4_GGML", QType::NVFP4_GGML},
          {"Q1_0", QType::Q1_0},
          {"Q2_0", QType::Q2_0}}) {
            const std::int64_t gate_row = ops::ggml_row_bytes(type, H);
            const std::int64_t down_row = ops::ggml_row_bytes(type, I);
            std::vector<std::byte> gate(static_cast<std::size_t>(E) * 2 * I * gate_row);
            std::vector<std::byte> down(static_cast<std::size_t>(E) * H * down_row);
            std::uniform_int_distribution<int> byte(0, 255);
            // Exponent bytes kept small so the fp16 scales stay in a sane range: a random
            // half is as likely to be an inf as anything else, and inf weights compare as NaN.
            for (auto& b : gate) { b = static_cast<std::byte>(byte(rng) & 0x3F); }
            for (auto& b : down) { b = static_cast<std::byte>(byte(rng) & 0x3F); }
            ops::CpuExpertBank gbank;
            gbank.gate_up_format = ops::ExpertBankFormat::GgmlBlocks;
            gbank.down_format    = ops::ExpertBankFormat::GgmlBlocks;
            gbank.gate_up_ggml  = type;
            gbank.down_ggml     = type;
            gbank.gate_up_codes = gate.data();
            gbank.down_codes    = down.data();

            const int gt = 5;
            std::vector<std::uint16_t> gx(static_cast<std::size_t>(H) * gt);
            for (auto& v : gx) { v = float_to_bf16(act(rng)); }
            std::vector<ops::CpuExpertJob> gj;
            std::uniform_real_distribution<float> wd(0.05F, 0.95F);
            for (int t2 = 0; t2 < gt; ++t2) {
                gj.push_back({t2, t2 % E, wd(rng)});
                gj.push_back({t2, (t2 * 3 + 1) % E, wd(rng)});
            }
            std::vector<float> gpool(static_cast<std::size_t>(H) * gt, 0.0F);
            ops::CpuExpertPool gp(kBlockGeometry, {.threads = 5, .pin_threads = false});
            ops::CpuExpertRound gr{gx.data(), gpool.data(), gt, gj};
            gp.run(gbank, gr);
            // The single-job entry point is the oracle: same weights, whole rows, no chunking.
            std::vector<double> want(static_cast<std::size_t>(H) * gt, 0.0);
            std::vector<float> one(H, 0.0F);
            for (const auto& j : gj) {
                std::fill(one.begin(), one.end(), 0.0F);
                ops::cpu_expert_compute_job(kBlockGeometry, gbank, j, gx.data() + j.token * H,
                                            one.data(), gscratch.data());
                for (int i = 0; i < H; ++i) { want[static_cast<std::size_t>(j.token) * H + i] += one[i]; }
            }
            failures += compare(std::string("ggml ") + name + " pooled vs single job", gpool, want);
        }
    }

    // --- 4-bit affine blocks straight to Q4G32AM ---
    // Each converted group, reconstructed as scale*q + min, must match the codec's float decode
    // of the same block to FP16 rounding of the endpoints: the conversion is a repack, not a
    // requantisation. Random bytes are legal blocks of every format; the fp16 exponents are
    // kept small as above.
    {
        for (const auto& [name, type] : std::vector<std::pair<const char*, QType>>{
                 {"Q4_K", QType::Q4_K}, {"Q4_0", QType::Q4_0}, {"Q4_1", QType::Q4_1}}) {
            const int k = 512;
            std::vector<std::byte> row(static_cast<std::size_t>(ops::ggml_row_bytes(type, k)));
            std::uniform_int_distribution<int> byte(0, 255);
            for (auto& b : row) { b = static_cast<std::byte>(byte(rng) & 0x3F); }
            std::vector<float> want(k);
            std::vector<std::uint8_t> codes(static_cast<std::size_t>(k) / 2);
            std::vector<std::uint16_t> scales(static_cast<std::size_t>(k) / 32), mins(static_cast<std::size_t>(k) / 32);
            const bool a = ops::ggml_decode_row_float(type, row.data(), k, want.data());
            const bool b = ops::ggml_row_to_q4g32am(type, row.data(), k, codes.data(), scales.data(), mins.data());
            double worst = 0.0, range = 1e-30;
            for (int g = 0; g < k / 32 && a && b; ++g) {
                const double sc = fp16_to_float(scales[static_cast<std::size_t>(g)]);
                const double mn = fp16_to_float(mins[static_cast<std::size_t>(g)]);
                for (int v = 0; v < 32; ++v) {
                    const int q = (codes[static_cast<std::size_t>(g * 16 + v / 2)] >> (4 * (v % 2))) & 0xF;
                    const double got  = sc * q + mn;
                    const double wref = want[static_cast<std::size_t>(g * 32 + v)];
                    worst = std::max(worst, std::fabs(got - wref));
                    range = std::max(range, std::fabs(wref));
                }
            }
            // Two FP16 roundings (scale and min), each within 2^-11 relative, on values up to
            // `range`: 2e-3 of the range is a generous bound for an exact repack and far below a
            // requantisation's half-step (3 % of the range at sixteen levels).
            const bool ok = a && b && worst <= 2e-3 * range;
            std::cout << (ok ? "ok   " : "FAIL ") << name << " -> Q4G32AM exact repack: max|err| " << worst
                      << " over range " << range << "\n";
            failures += ok ? 0 : 1;
        }
        // And a type the repack does not cover says so rather than guessing.
        std::vector<std::byte> q6(static_cast<std::size_t>(ops::ggml_row_bytes(QType::Q6_K, 256)));
        std::vector<std::uint8_t> c6(128);
        std::vector<std::uint16_t> s6(8), m6(8);
        const bool refused = !ops::ggml_row_to_q4g32am(QType::Q6_K, q6.data(), 256, c6.data(), s6.data(), m6.data());
        std::cout << (refused ? "ok   " : "FAIL ") << "Q6_K -> Q4G32AM refused\n";
        failures += refused ? 0 : 1;
    }

    // Stress: many tiny rounds on a full-width pool exercise the wake-up/barrier protocol; a
    // lost wake-up shows up as a hang, so the test runs it with a watchdog thread.
    {
        std::atomic<bool> finished{false};
        std::thread watchdog([&] {
            for (int i = 0; i < 600 && !finished.load(); ++i) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
            if (!finished.load()) { std::cout << "FAIL pool stress hung\n"; std::_Exit(2); }
        });
        ops::CpuExpertPool wide(kGeometry, {.threads = 32, .pin_threads = false});
        std::vector<ops::CpuExpertJob> tiny = {{0, 1, 1.0F}};
        std::vector<float> tiny_out(static_cast<std::size_t>(H), 0.0F);
        ops::CpuExpertRound tiny_round{x.data(), tiny_out.data(), 1, tiny};
        for (int r = 0; r < 3000; ++r) {
            std::fill(tiny_out.begin(), tiny_out.end(), 0.0F);
            wide.run(bank.view(), tiny_round);
            if (r % 7 == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); } // let workers sleep
        }
        failures += compare("stress round (last)", tiny_out, reference_job(bank, 1, x, 1.0F));
        finished.store(true);
        watchdog.join();
    }
    // --- Q4G32AM bank ---
    {
        // Requantiser fidelity on an exact 16-level affine grid (a Q4_K sub-block's shape):
        // c = -60 + 8q with q in [0,15], so max-min = 15 steps and the affine fit reproduces
        // every decoded value to FP16 endpoint rounding.
        std::vector<std::int8_t> grid_codes(32);
        std::vector<std::uint16_t> grid_scale = {float_to_fp16(0.01F)};
        for (int i = 0; i < 32; ++i) {
            grid_codes[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(-60 + 8 * (i % 16));
        }
        std::vector<std::uint8_t> gq(16);
        std::vector<std::uint16_t> gs(1), gm(1);
        ops::requantise_w8_expert_groups_to_q4(grid_codes.data(), grid_scale.data(), 1, gq.data(),
                                               gs.data(), gm.data());
        double worst = 0.0;
        for (int i = 0; i < 32; ++i) {
            const int q = (gq[static_cast<std::size_t>(i / 2)] >> (4 * (i % 2))) & 0x0F;
            const double got  = static_cast<double>(fp16_to_float(gs[0])) * q + fp16_to_float(gm[0]);
            const double want = static_cast<double>(grid_codes[static_cast<std::size_t>(i)]) *
                                fp16_to_float(grid_scale[0]);
            worst = std::max(worst, std::fabs(got - want));
        }
        const bool grid_ok = worst < 2.0e-3 * 1.2; // 1.2 = the grid's value range
        std::cout << (grid_ok ? "ok   " : "FAIL ") << "q4 affine grid max|err| " << worst << "\n";
        failures += grid_ok ? 0 : 1;

        const Q4Bank q4 = requantise_bank(bank);
        // Scalar oracle path (cpu_expert_compute_job).
        std::vector<float> outq(H, 0.0F);
        ops::cpu_expert_compute_job(kGeometry, q4.view(), job, x.data(), outq.data(), scratch.data());
        failures += compare("q4 single job expert 2", outq, reference_job_q4(q4, 2, x, 0.75F));

        // Pooled round through the VNNI kernels (paired tokens, odd tails, one-token groups).
        const int many = 13;
        std::vector<std::uint16_t> xm(static_cast<std::size_t>(H) * many);
        for (auto& v : xm) { v = float_to_bf16(act(rng)); }
        std::vector<ops::CpuExpertJob> jm;
        std::uniform_real_distribution<float> wdist(0.05F, 0.95F);
        for (int t2 = 0; t2 < many; ++t2) {
            jm.push_back({t2, t2 % 4, wdist(rng)});
            if (t2 % 2 == 0) { jm.push_back({t2, (t2 * 7 + 1) % 4, wdist(rng)}); }
        }
        std::vector<float> om(static_cast<std::size_t>(H) * many, 0.0F);
        ops::CpuExpertPool poolq(kGeometry, {.threads = 6, .pin_threads = false});
        ops::CpuExpertRound rq{xm.data(), om.data(), many, jm};
        poolq.run(q4.view(), rq);
        int bad = 0;
        for (int t2 = 0; t2 < many; ++t2) {
            std::vector<std::uint16_t> column(xm.begin() + t2 * H, xm.begin() + (t2 + 1) * H);
            std::vector<double> want(H, 0.0);
            for (const auto& j2 : jm) {
                if (j2.token != t2) { continue; }
                const std::vector<double> y = reference_job_q4(q4, j2.expert, column, j2.weight);
                for (int i = 0; i < H; ++i) { want[i] += y[i]; }
            }
            std::vector<float> got(om.begin() + t2 * H, om.begin() + (t2 + 1) * H);
            bad += compare("q4 pooled token " + std::to_string(t2), got, want);
        }
        failures += bad;

        // The requantised bank must stay close to the W8 bank it came from: same job, W8
        // reference, loose bound (this is the 4-bit quantisation error itself, on random,
        // non-affine synthetic codes — the worst case the real Q4_K-derived weights never hit).
        std::vector<double> w8_want = reference_job(bank, 2, x, 0.75F);
        double err2 = 0.0, ref2 = 0.0;
        for (int i = 0; i < H; ++i) {
            const double d = static_cast<double>(outq[static_cast<std::size_t>(i)]) - w8_want[static_cast<std::size_t>(i)];
            err2 += d * d;
            ref2 += w8_want[static_cast<std::size_t>(i)] * w8_want[static_cast<std::size_t>(i)];
        }
        const double rel = std::sqrt(err2 / std::max(ref2, 1e-30));
        std::cout << "info q4-vs-w8 rel-L2 " << rel << " (random codes; affine sources are near-exact)\n";
        failures += rel < 0.2 ? 0 : 1;
    }
    // --- The clamped SwiGLU ---
    // The same banks under a geometry that states a limit: every path (single job, the paired
    // and odd-tail pool paths, W8 and Q4) must bound both halves before the product. The limit
    // is the one GLM-5.3 states, and the random codes drive the dot products well past it, so
    // the case first checks the clamp actually engages -- a reference that agreed with the
    // kernel because neither clamped would prove nothing.
    {
        constexpr ops::SparseMoeGeometry kClamped{.hidden            = 128,
                                                  .experts           = 4,
                                                  .experts_per_token = 2,
                                                  .intermediate      = 128,
                                                  .swiglu_limit      = 10.0F};
        const double limit = kClamped.swiglu_limit;
        {
            const std::vector<double> plain   = reference_job(bank, 2, x, 0.75F);
            const std::vector<double> clamped = reference_job(bank, 2, x, 0.75F, limit);
            double diff2 = 0.0, ref2 = 0.0;
            for (int i = 0; i < H; ++i) {
                diff2 += (plain[i] - clamped[i]) * (plain[i] - clamped[i]);
                ref2 += plain[i] * plain[i];
            }
            const double rel     = std::sqrt(diff2 / std::max(ref2, 1e-30));
            const bool engaged   = rel > 1e-3;
            std::cout << (engaged ? "ok   " : "FAIL ") << "clamp engages (rel-L2 vs unclamped " << rel << ")\n";
            failures += engaged ? 0 : 1;
        }
        std::vector<float> outc(H, 0.0F);
        std::vector<std::byte> cscratch(ops::cpu_expert_scratch_bytes(kClamped) + 64);
        ops::cpu_expert_compute_job(kClamped, bank.view(), job, x.data(), outc.data(), cscratch.data());
        failures += compare("clamped single job expert 2", outc, reference_job(bank, 2, x, 0.75F, limit));
        const Q4Bank q4c = requantise_bank(bank);
        std::fill(outc.begin(), outc.end(), 0.0F);
        ops::cpu_expert_compute_job(kClamped, q4c.view(), job, x.data(), outc.data(), cscratch.data());
        failures += compare("clamped q4 single job expert 2", outc, reference_job_q4(q4c, 2, x, 0.75F, limit));

        const int many = 13;
        std::vector<std::uint16_t> xm(static_cast<std::size_t>(H) * many);
        for (auto& v : xm) { v = float_to_bf16(act(rng)); }
        std::vector<ops::CpuExpertJob> jm;
        std::uniform_real_distribution<float> wdist(0.05F, 0.95F);
        for (int t2 = 0; t2 < many; ++t2) {
            jm.push_back({t2, t2 % 4, wdist(rng)});
            jm.push_back({t2, (t2 * 7 + 1) % 4, wdist(rng)});
            if (t2 % 3 == 0) { jm.push_back({t2, 3, wdist(rng)}); }
        }
        ops::CpuExpertPool poolc(kClamped, {.threads = 6, .pin_threads = false});
        for (const bool q4_bank : {false, true}) {
            std::vector<float> om(static_cast<std::size_t>(H) * many, 0.0F);
            ops::CpuExpertRound rm{xm.data(), om.data(), many, jm};
            poolc.run(q4_bank ? q4c.view() : bank.view(), rm);
            int bad = 0;
            for (int t2 = 0; t2 < many; ++t2) {
                std::vector<std::uint16_t> column(xm.begin() + t2 * H, xm.begin() + (t2 + 1) * H);
                std::vector<double> want(H, 0.0);
                for (const auto& j2 : jm) {
                    if (j2.token != t2) { continue; }
                    const std::vector<double> y = q4_bank
                                                      ? reference_job_q4(q4c, j2.expert, column, j2.weight, limit)
                                                      : reference_job(bank, j2.expert, column, j2.weight, limit);
                    for (int i = 0; i < H; ++i) { want[i] += y[i]; }
                }
                std::vector<float> got(om.begin() + t2 * H, om.begin() + (t2 + 1) * H);
                bad += compare(std::string(q4_bank ? "clamped q4 pooled token " : "clamped pooled token ") +
                                   std::to_string(t2),
                               got, want);
            }
            failures += bad;
        }
    }
    // --- The mixed bank: Q4G32AM gate/up over W8 down ---
    // Each half is read by its own format; the reference composes the two formats' oracles.
    {
        const Q4Bank q4m = requantise_bank(bank);
        ops::CpuExpertBank mixed = q4m.view();
        const ops::CpuExpertBank w8 = bank.view();
        mixed.down_format = ops::ExpertBankFormat::W8G32;
        mixed.down_codes  = w8.down_codes;
        mixed.down_scales = w8.down_scales;
        mixed.down_mins   = nullptr;
        std::vector<float> outm(H, 0.0F);
        ops::cpu_expert_compute_job(kGeometry, mixed, job, x.data(), outm.data(), scratch.data());
        failures += compare("mixed single job expert 2", outm, reference_job_mixed(q4m, bank, 2, x, 0.75F));
        const int many = 13;
        std::vector<std::uint16_t> xm(static_cast<std::size_t>(H) * many);
        for (auto& v : xm) { v = float_to_bf16(act(rng)); }
        std::vector<ops::CpuExpertJob> jm;
        std::uniform_real_distribution<float> wdist(0.05F, 0.95F);
        for (int t2 = 0; t2 < many; ++t2) {
            jm.push_back({t2, t2 % 4, wdist(rng)});
            if (t2 % 2 == 0) { jm.push_back({t2, (t2 * 7 + 1) % 4, wdist(rng)}); }
        }
        std::vector<float> om(static_cast<std::size_t>(H) * many, 0.0F);
        ops::CpuExpertPool poolm(kGeometry, {.threads = 6, .pin_threads = false});
        ops::CpuExpertRound rm{xm.data(), om.data(), many, jm};
        poolm.run(mixed, rm);
        int bad = 0;
        for (int t2 = 0; t2 < many; ++t2) {
            std::vector<std::uint16_t> column(xm.begin() + t2 * H, xm.begin() + (t2 + 1) * H);
            std::vector<double> want(H, 0.0);
            for (const auto& j2 : jm) {
                if (j2.token != t2) { continue; }
                const std::vector<double> y = reference_job_mixed(q4m, bank, j2.expert, column, j2.weight);
                for (int i = 0; i < H; ++i) { want[i] += y[i]; }
            }
            std::vector<float> got(om.begin() + t2 * H, om.begin() + (t2 + 1) * H);
            bad += compare("mixed pooled token " + std::to_string(t2), got, want);
        }
        failures += bad;
    }
    // --- The Q5G32AM bank: single job (scalar oracle path), the pooled paths, and Q5 gate/up
    // over W8 down, which is the shape a checkpoint with Q5 gate/up and Q8_0 down takes ---
    {
        const Q5Bank q5 = make_q5_bank(rng);
        std::vector<float> out5(H, 0.0F);
        ops::cpu_expert_compute_job(kGeometry, q5.view(), job, x.data(), out5.data(), scratch.data());
        failures += compare("q5 single job expert 2", out5, reference_job_q5(q5, 2, x, 0.75F));
        const int many = 13;
        std::vector<std::uint16_t> xm(static_cast<std::size_t>(H) * many);
        for (auto& v : xm) { v = float_to_bf16(act(rng)); }
        std::vector<ops::CpuExpertJob> jm;
        std::uniform_real_distribution<float> wdist(0.05F, 0.95F);
        for (int t2 = 0; t2 < many; ++t2) {
            jm.push_back({t2, t2 % 4, wdist(rng)});
            if (t2 % 2 == 0) { jm.push_back({t2, (t2 * 7 + 1) % 4, wdist(rng)}); }
            if (t2 % 3 == 0) { jm.push_back({t2, 3, wdist(rng)}); }
        }
        ops::CpuExpertPool pool5(kGeometry, {.threads = 6, .pin_threads = false});
        for (const bool mixed : {false, true}) {
            ops::CpuExpertBank view = q5.view();
            if (mixed) {
                const ops::CpuExpertBank w8 = bank.view();
                view.down_format = ops::ExpertBankFormat::W8G32;
                view.down_codes  = w8.down_codes;
                view.down_scales = w8.down_scales;
                view.down_mins   = nullptr;
            }
            std::vector<float> om(static_cast<std::size_t>(H) * many, 0.0F);
            ops::CpuExpertRound rm{xm.data(), om.data(), many, jm};
            pool5.run(view, rm);
            int bad = 0;
            for (int t2 = 0; t2 < many; ++t2) {
                std::vector<std::uint16_t> column(xm.begin() + t2 * H, xm.begin() + (t2 + 1) * H);
                std::vector<double> want(H, 0.0);
                for (const auto& j2 : jm) {
                    if (j2.token != t2) { continue; }
                    std::vector<double> y;
                    if (mixed) {
                        // Q5 gate/up, W8 down: compose the two oracles through the same h.
                        const int I = kGeometry.intermediate;
                        std::vector<double> xf(H);
                        for (int i = 0; i < H; ++i) { xf[i] = bf16_to_float(column[i]); }
                        std::vector<int> xq; std::vector<double> xs;
                        quantise_ref(xf, xq, xs);
                        const std::size_t gh = static_cast<std::size_t>(H / 32);
                        const std::uint8_t* gc  = q5.gate_codes.data() + static_cast<std::size_t>(j2.expert) * 2 * I * gh * 20;
                        const std::uint16_t* gs = q5.gate_scales.data() + static_cast<std::size_t>(j2.expert) * 2 * I * gh;
                        const std::uint16_t* gm = q5.gate_mins.data() + static_cast<std::size_t>(j2.expert) * 2 * I * gh;
                        std::vector<double> hh(I);
                        for (int j = 0; j < I; ++j) {
                            const double g = dot_ref_q5(gc + j * gh * 20, gs + j * gh, gm + j * gh, xq, xs, H);
                            const double u = dot_ref_q5(gc + (I + j) * gh * 20, gs + (I + j) * gh, gm + (I + j) * gh, xq, xs, H);
                            hh[j] = swiglu_ref(g, u, 0.0);
                        }
                        std::vector<int> hq; std::vector<double> hs;
                        quantise_ref(hh, hq, hs);
                        const std::int8_t* dc   = bank.down_codes.data() + static_cast<std::size_t>(j2.expert) * H * I;
                        const std::uint16_t* ds = bank.down_scales.data() + static_cast<std::size_t>(j2.expert) * H * (I / 32);
                        y.resize(H);
                        for (int r = 0; r < H; ++r) { y[r] = j2.weight * dot_ref(dc + r * I, ds + r * (I / 32), hq, hs, I); }
                    } else {
                        y = reference_job_q5(q5, j2.expert, column, j2.weight);
                    }
                    for (int i = 0; i < H; ++i) { want[i] += y[i]; }
                }
                std::vector<float> got(om.begin() + t2 * H, om.begin() + (t2 + 1) * H);
                bad += compare(std::string(mixed ? "q5 gate/up over w8 down token " : "q5 pooled token ") + std::to_string(t2), got, want);
            }
            failures += bad;
        }
    }
    // --- 5-bit affine blocks straight to Q5G32AM: exact against the codec's float decode ---
    {
        for (const auto& [name, type] : std::vector<std::pair<const char*, QType>>{
                 {"Q5_K", QType::Q5_K}, {"Q5_0", QType::Q5_0}, {"Q5_1", QType::Q5_1}}) {
            const int k = 512;
            std::vector<std::byte> row(static_cast<std::size_t>(ops::ggml_row_bytes(type, k)));
            std::uniform_int_distribution<int> byte(0, 255);
            for (auto& b : row) { b = static_cast<std::byte>(byte(rng) & 0x3F); }
            std::vector<float> want(k);
            std::vector<std::uint8_t> codes(static_cast<std::size_t>(k) / 32 * 20);
            std::vector<std::uint16_t> scales(static_cast<std::size_t>(k) / 32), mins(static_cast<std::size_t>(k) / 32);
            const bool a = ops::ggml_decode_row_float(type, row.data(), k, want.data());
            const bool b = ops::ggml_row_to_q5g32am(type, row.data(), k, codes.data(), scales.data(), mins.data());
            double worst = 0.0, range = 1e-30;
            for (int g = 0; g < k / 32 && a && b; ++g) {
                const double sc = fp16_to_float(scales[static_cast<std::size_t>(g)]);
                const double mn = fp16_to_float(mins[static_cast<std::size_t>(g)]);
                for (int v = 0; v < 32; ++v) {
                    const double got  = sc * q5_code(codes.data() + static_cast<std::size_t>(g) * 20, v) + mn;
                    const double wref = want[static_cast<std::size_t>(g * 32 + v)];
                    worst = std::max(worst, std::fabs(got - wref));
                    range = std::max(range, std::fabs(wref));
                }
            }
            const bool ok = a && b && worst <= 2e-3 * range;
            std::cout << (ok ? "ok   " : "FAIL ") << name << " -> Q5G32AM exact repack: max|err| " << worst
                      << " over range " << range << "\n";
            failures += ok ? 0 : 1;
        }
        std::vector<std::byte> q4(static_cast<std::size_t>(ops::ggml_row_bytes(QType::Q4_K, 256)));
        std::vector<std::uint8_t> c5(8 * 20);
        std::vector<std::uint16_t> s5(8), m5(8);
        const bool refused = !ops::ggml_row_to_q5g32am(QType::Q4_K, q4.data(), 256, c5.data(), s5.data(), m5.data());
        std::cout << (refused ? "ok   " : "FAIL ") << "Q4_K -> Q5G32AM refused\n";
        failures += refused ? 0 : 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " cpu_expert_compute\n";
    return failures ? 1 : 0;
}
