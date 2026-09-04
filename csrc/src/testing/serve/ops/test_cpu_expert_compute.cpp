// CPU expert compute: the planar-W8 gated FFN on the host against a double-precision reference
// of the same quantised arithmetic, single job and pooled rounds (no GPU needed).
#include "api/ops/cpu_expert_compute.h"

#include "ops/linear/ggml/ggml_host_decode.h"

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

// --- Q4G32AM bank: requantised from the W8 bank; the reference decodes the affine grid. ---
struct Q4Bank {
    std::vector<std::uint8_t> gate_codes, down_codes;
    std::vector<std::uint16_t> gate_scales, gate_mins, down_scales, down_mins;
    ops::CpuExpertBank view() const {
        ops::CpuExpertBank bank;
        bank.format         = ops::ExpertBankFormat::Q4G32AM;
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
                                     const std::vector<std::uint16_t>& x, float weight) {
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
        h[j] = (g / (1.0 + std::exp(-g))) * u;
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
                 {"Q4_0", QType::Q4_0}, {"Q5_0", QType::Q5_0}}) {
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
            gbank.format        = ops::ExpertBankFormat::GgmlBlocks;
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
    std::cout << (failures ? "FAIL" : "OK") << " cpu_expert_compute\n";
    return failures ? 1 : 0;
}
