// CPU expert compute: the planar-W8 gated FFN on the host against a double-precision reference
// of the same quantised arithmetic, single job and pooled rounds (no GPU needed).
#include "api/ops/cpu_expert_compute.h"

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

using namespace ninfer;

namespace {

constexpr ops::SparseMoeGeometry kGeometry{128, 4, 2, 64}; // hidden 128, 4 experts, top-2, ffn 64

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
        return {reinterpret_cast<const std::byte*>(gate_codes.data()),
                reinterpret_cast<const std::byte*>(gate_scales.data()),
                reinterpret_cast<const std::byte*>(down_codes.data()),
                reinterpret_cast<const std::byte*>(down_scales.data())};
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
std::vector<double> reference_job(const Bank& b, int expert, const std::vector<std::uint16_t>& x, float weight) {
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
        h[j]           = (g / (1.0 + std::exp(-g))) * u;
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
    std::cout << "avx512 path: " << (ops::cpu_expert_compute_has_avx512() ? "yes" : "no (scalar)") << "\n";

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
    std::cout << (failures ? "FAIL" : "OK") << " cpu_expert_compute\n";
    return failures ? 1 : 0;
}
