// Inkling-2-Mini lane e1: BITWISE batch- and path-invariance of the e1 ops (agent-EN 10:2x decision, item 3).
//
// Every check compares bits, never a tolerance: the same column computed
//   - in a round of 300 columns, alone, in a window of 8 / 37 / 128 / 257, or beside other problems of a batch
//     (I1 invariant GEMM at the inkling2 shapes, I2 AttnRes, I3 pair / blend norms);
//   - by one prefill, chunked prefill (cuts at 1, 63, 64, 65, 200), token-by-token decode, MTP-style verify rounds
//     (W = 4, backup + record + fold, commits cycling 4 / 2 / 0 / 1 / 3), and decode beside another sequence in one
//     block (I4 hidden short conv: outputs AND the carried state).
// I1 also checks the GEMM against a float64 host product (a loose bound: a wrong chain is off by O(1)).

#include "api/ops/inkling_mix.h"
#include "api/ops/sampled_logprob.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

using G            = ops::InklingMixGeometry;
constexpr int kD   = G::kHidden;
constexpr int kAll = 300; // columns of the reference round

float rb(float x) { return bf16_to_f32(f32_to_bf16(x)); }

std::vector<float> randn_bf16(std::size_t n, std::uint32_t seed, float scale) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) { x = rb(d(g)); }
    return v;
}

std::vector<float> randn_f32(std::size_t n, std::uint32_t seed, float scale) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) { x = d(g); }
    return v;
}

/// A cheap uniform bf16 fill for the large weights (the 201,024-row head): an integer hash, no RNG state.
std::vector<std::uint16_t> hash_bf16(std::size_t n, std::uint32_t seed, float scale) {
    std::vector<std::uint16_t> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 2654435761U ^ (seed * 2246822519U) ^ static_cast<std::uint32_t>(i >> 32);
        h ^= h >> 15;
        h *= 2246822519U;
        h ^= h >> 13;
        h *= 3266489917U;
        h ^= h >> 16;
        const float u = static_cast<float>(h & 0xFFFFFFU) / 16777216.0F * 2.0F - 1.0F;
        v[i]          = f32_to_bf16(u * scale);
    }
    return v;
}

std::vector<std::uint16_t> bits(const std::vector<float>& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { out[i] = f32_to_bf16(v[i]); }
    return out;
}

DeviceBuffer dev_bf16(const std::vector<float>& v) { return to_device(bits(v)); }

std::vector<std::uint8_t> raw(const void* p, std::size_t bytes) { return from_device<std::uint8_t>(p, bytes); }

/// Bitwise: element i of `got` (element size `esize`) against `want`.
int same_bits(const std::string& label, const std::vector<std::uint8_t>& got, const std::vector<std::uint8_t>& want,
              std::size_t esize) {
    if (got.size() != want.size()) {
        std::cout << "FAIL " << label << ": sizes " << got.size() << " vs " << want.size() << "\n";
        return 1;
    }
    std::size_t bad = 0, first = 0;
    for (std::size_t i = 0; i < got.size(); i += esize) {
        if (std::memcmp(&got[i], &want[i], esize) != 0) {
            if (bad == 0) { first = i / esize; }
            ++bad;
        }
    }
    std::cout << (bad ? "FAIL " : "ok   ") << label << ": " << bad << " of " << got.size() / esize << " differ";
    if (bad) { std::cout << " (first element " << first << ")"; }
    std::cout << "\n";
    return bad ? 1 : 0;
}

/// Columns [c0, c0 + n) of every plane of a [rows, T, planes] buffer (element size esize), compacted.
std::vector<std::uint8_t> gather(const std::vector<std::uint8_t>& full, std::size_t rows, int T, int planes, int c0,
                                 int n, std::size_t esize) {
    std::vector<std::uint8_t> out;
    out.reserve(rows * n * planes * esize);
    for (int p = 0; p < planes; ++p) {
        const auto* base = full.data() + (static_cast<std::size_t>(p) * T + c0) * rows * esize;
        out.insert(out.end(), base, base + static_cast<std::size_t>(n) * rows * esize);
    }
    return out;
}

// ------------------------------------------------------------------------------------- I1 GEMM

struct GemmShape {
    const char* name;
    int n, k, batch;
    DType out;
};

int test_gemm(const GemmShape& g, std::uint32_t seed) {
    int failures = 0;
    const std::size_t esize = g.out == DType::FP32 ? 4 : 2;
    const auto w_bits = hash_bf16(static_cast<std::size_t>(g.batch) * g.n * g.k, seed, 0.06F);
    const auto x_bits = hash_bf16(static_cast<std::size_t>(g.batch) * kAll * g.k, seed + 7, 1.0F);
    DeviceBuffer d_w = to_device(w_bits), d_x = to_device(x_bits);
    const auto run = [&](int b0, int batch, int c0, int T) {
        // problems [b0, b0 + batch), columns [c0, c0 + T) of each; out [batch][T][n] compact
        DeviceBuffer d_out(static_cast<std::size_t>(batch) * T * g.n * esize);
        const std::size_t ws_bytes = ops::inkling_gemm_inv_workspace_bytes(g.n, g.k, T, batch);
        DeviceBuffer d_ws(ws_bytes == 0 ? 16 : ws_bytes);
        const ops::InklingGemmBatch bt{batch, static_cast<std::int64_t>(g.n) * g.k, static_cast<std::int64_t>(kAll) * g.k,
                                       static_cast<std::int64_t>(T) * g.n};
        const auto* w = static_cast<const std::uint16_t*>(d_w.p) + static_cast<std::size_t>(b0) * g.n * g.k;
        const auto* x = static_cast<const std::uint16_t*>(d_x.p) + (static_cast<std::size_t>(b0) * kAll + c0) * g.k;
        ops::inkling_gemm_inv(w, g.n, g.k, x, T, d_out.p, g.out, g.n, bt, d_ws.p, ws_bytes, nullptr);
        cuda_synchronize();
        return raw(d_out.p, d_out.bytes);
    };
    const auto full = run(0, g.batch, 0, kAll);
    const std::string tag = std::string("I1 ") + g.name + " (split " + std::to_string(ops::inkling_gemm_split(g.n, g.k)) + ")";
    const int windows[][2] = {{0, 1}, {1, 1}, {150, 1}, {299, 1}, {64, 8}, {200, 37}, {100, 128}, {20, 257}};
    for (const auto& win : windows) {
        const auto sub = run(0, g.batch, win[0], win[1]);
        failures += same_bits(tag + " T=" + std::to_string(win[1]) + " at " + std::to_string(win[0]) + " == T=300", sub,
                              gather(full, g.n, kAll, g.batch, win[0], win[1], esize), esize);
    }
    if (g.batch > 1) {
        const auto alone = run(1, 1, 0, kAll);
        const auto plane = static_cast<std::ptrdiff_t>(kAll) * g.n * static_cast<std::ptrdiff_t>(esize);
        failures += same_bits(tag + " problem 1 alone == in a batch of " + std::to_string(g.batch), alone,
                              std::vector<std::uint8_t>(full.begin() + plane, full.begin() + 2 * plane), esize);
    }
    // float64 sanity on problem 0, columns 0 and 299, the first min(n, 4096) rows
    {
        const int rows = std::min(g.n, 4096);
        std::size_t bad = 0;
        double worst   = 0;
        for (const int c : {0, 299}) {
            for (int r = 0; r < rows; ++r) {
                double ref = 0, mag = 0;
                for (int i = 0; i < g.k; ++i) {
                    const double t = static_cast<double>(bf16_to_f32(w_bits[static_cast<std::size_t>(r) * g.k + i])) *
                                     bf16_to_f32(x_bits[static_cast<std::size_t>(c) * g.k + i]);
                    ref += t;
                    mag += std::fabs(t);
                }
                const std::size_t at = (static_cast<std::size_t>(c) * g.n + r) * esize;
                float got            = 0;
                if (esize == 4) {
                    std::memcpy(&got, &full[at], 4);
                } else {
                    std::uint16_t h;
                    std::memcpy(&h, &full[at], 2);
                    got = bf16_to_f32(h);
                }
                const double limit = (esize == 2 ? std::fabs(ref) / 128.0 : std::fabs(ref) * 1e-6) +
                                     4.0 * std::sqrt(static_cast<double>(g.k)) * mag / 16777216.0;
                const double err = std::fabs(got - ref);
                if (!(err <= limit)) { ++bad; }
                worst = std::max(worst, err / limit);
            }
        }
        std::cout << (bad ? "FAIL " : "ok   ") << tag << " vs float64 (2 columns x " << rows << " rows): worst err/limit "
                  << worst << ", " << bad << " over\n";
        failures += bad ? 1 : 0;
    }
    return failures;
}

/// The 3-input qkvr of a global visit with its dynamic taps: one grouped launch of 5 == the 3 row ranges as separate
/// invariant calls + the 2 taps as a group of their own (every group problem runs unsplit), and a window of the
/// grouped launch == the full round.
int test_gemm_group(std::uint32_t seed) {
    constexpr int kRows = 8600, kQ = 6552, kK = 6552, kV = 7576, kTap = 32;
    const auto w_bits = hash_bf16(static_cast<std::size_t>(kRows) * kD, seed, 0.06F);
    const auto t_bits = hash_bf16(static_cast<std::size_t>(2) * kTap * kD, seed + 2, 0.06F);
    DeviceBuffer d_t = to_device(t_bits);
    const auto* tk = static_cast<const std::uint16_t*>(d_t.p);
    const auto* tv = tk + static_cast<std::size_t>(kTap) * kD;
    const auto x_bits = hash_bf16(static_cast<std::size_t>(3) * kAll * kD, seed + 1, 1.0F);
    DeviceBuffer d_w = to_device(w_bits), d_x = to_device(x_bits);
    const auto* w = static_cast<const std::uint16_t*>(d_w.p);
    const auto x  = [&](int stream, int c0) {
        return static_cast<const std::uint16_t*>(d_x.p) + (static_cast<std::size_t>(stream) * kAll + c0) * kD;
    };
    // out: [T][kRows] qkvr, then [T][32] k taps, then [T][32] v taps
    const auto grouped = [&](int c0, int T) {
        DeviceBuffer out(static_cast<std::size_t>(T) * (kRows + 2 * kTap) * 2);
        out.fill(0);
        auto* o  = static_cast<std::uint16_t*>(out.p);
        auto* ok = o + static_cast<std::size_t>(T) * kRows;
        auto* ov = ok + static_cast<std::size_t>(T) * kTap;
        const ops::InklingGemmProblem pr[] = {{w, kQ, x(0, c0), o, kRows},
                                              {w + static_cast<std::size_t>(kK) * kD, kV - kK, x(1, c0), o + kK, kRows},
                                              {w + static_cast<std::size_t>(kV) * kD, kRows - kV, x(2, c0), o + kV, kRows},
                                              {tk, kTap, x(1, c0), ok, kTap},
                                              {tv, kTap, x(2, c0), ov, kTap}};
        ops::inkling_gemm_inv_group(pr, kD, T, DType::BF16, nullptr);
        cuda_synchronize();
        return raw(out.p, out.bytes);
    };
    int failures = 0;
    const auto full = grouped(0, kAll);
    {
        DeviceBuffer out(static_cast<std::size_t>(kAll) * (kRows + 2 * kTap) * 2);
        out.fill(0);
        auto* o  = static_cast<std::uint16_t*>(out.p);
        auto* ok = o + static_cast<std::size_t>(kAll) * kRows;
        auto* ov = ok + static_cast<std::size_t>(kAll) * kTap;
        const int begin[] = {0, kK, kV}, rows[] = {kQ, kV - kK, kRows - kV};
        for (int i = 0; i < 3; ++i) {
            ops::inkling_gemm_inv(w + static_cast<std::size_t>(begin[i]) * kD, rows[i], kD, x(i, 0), kAll, o + begin[i],
                                  DType::BF16, kRows, {}, nullptr, 0, nullptr);
        }
        const ops::InklingGemmProblem taps[] = {{tk, kTap, x(1, 0), ok, kTap}, {tv, kTap, x(2, 0), ov, kTap}};
        ops::inkling_gemm_inv_group(taps, kD, kAll, DType::BF16, nullptr);
        cuda_synchronize();
        failures += same_bits("I1 qkvr + taps grouped (5) == 3 separate calls + the taps grouped alone (T=300)", full,
                              raw(out.p, out.bytes), 2);
    }
    // windows: the qkvr part and the two tap parts of the compact outputs
    const auto part = [](const std::vector<std::uint8_t>& all, int T, std::size_t off_rows, int rows) {
        return std::vector<std::uint8_t>(all.begin() + static_cast<std::ptrdiff_t>(T) * off_rows * 2,
                                         all.begin() + static_cast<std::ptrdiff_t>(T) * (off_rows + rows) * 2);
    };
    for (const auto& win : {std::pair<int, int>{0, 1}, std::pair<int, int>{299, 1}, std::pair<int, int>{100, 37}}) {
        const auto sub          = grouped(win.first, win.second);
        const std::string where = " T=" + std::to_string(win.second) + " at " + std::to_string(win.first) + " == T=300";
        failures += same_bits("I1 grouped qkvr" + where, part(sub, win.second, 0, kRows),
                              gather(part(full, kAll, 0, kRows), kRows, kAll, 1, win.first, win.second, 2), 2);
        failures += same_bits("I1 grouped k taps" + where, part(sub, win.second, kRows, kTap),
                              gather(part(full, kAll, kRows, kTap), kTap, kAll, 1, win.first, win.second, 2), 2);
        failures += same_bits("I1 grouped v taps" + where, part(sub, win.second, kRows + kTap, kTap),
                              gather(part(full, kAll, kRows + kTap, kTap), kTap, kAll, 1, win.first, win.second, 2), 2);
    }
    return failures;
}

// ------------------------------------------------------------------------------------- I2 AttnRes

int test_attn_res(std::uint32_t seed) {
    // The real layout: stack slots 0..2 and the partial (slot 6) in the [7 x 2304] residual, an addend, Q = 3, norm.
    constexpr int kSlots = 7, Q = 3;
    const auto residual = randn_bf16(static_cast<std::size_t>(kAll) * kSlots * kD, seed, 0.8F);
    const auto addend   = randn_bf16(static_cast<std::size_t>(kAll) * kD, seed + 1, 0.3F);
    const auto q        = randn_f32(static_cast<std::size_t>(Q) * kD, seed + 2, 1.0F / 48.0F);
    const auto w        = randn_bf16(static_cast<std::size_t>(Q) * kD, seed + 3, 1.0F);
    DeviceBuffer d_res = dev_bf16(residual), d_add = dev_bf16(addend), d_q = to_device(q), d_w = dev_bf16(w);
    Tensor q_t(d_q.p, DType::FP32, {kD, Q});
    Tensor w_t(d_w.p, DType::BF16, {kD, Q});
    const auto run = [&](int c0, int T) {
        ops::InklingAttnResSources s;
        s.stack_count = 3;
        auto* base    = static_cast<std::uint16_t*>(d_res.p) + static_cast<std::size_t>(c0) * kSlots * kD;
        for (int n = 0; n < 3; ++n) { s.stack[n] = ops::InklingPlane{base + n * kD, DType::BF16, kSlots * kD}; }
        s.last   = ops::InklingPlane{base + 6 * kD, DType::BF16, kSlots * kD};
        s.addend = ops::InklingPlane{static_cast<std::uint16_t*>(d_add.p) + static_cast<std::size_t>(c0) * kD, DType::BF16, kD};
        DeviceBuffer d_out(static_cast<std::size_t>(T) * kD * Q * 2);
        Tensor out(d_out.p, DType::BF16, {kD, T, Q});
        ops::inkling_attn_res(s, T, q_t, &w_t, 1e-6F, out, nullptr);
        cuda_synchronize();
        return raw(d_out.p, d_out.bytes);
    };
    const auto full = run(0, kAll);
    int failures    = 0;
    const int windows[][2] = {{0, 1}, {299, 1}, {64, 8}, {100, 37}, {20, 257}};
    for (const auto& win : windows) {
        failures += same_bits("I2 inkling_attn_res N=3+last+addend Q=3 norm T=" + std::to_string(win[1]) + " at " +
                                  std::to_string(win[0]) + " == T=300",
                              run(win[0], win[1]), gather(full, kD, kAll, Q, win[0], win[1], 2), 2);
    }
    return failures;
}

// ------------------------------------------------------------------------------------- I3 pair / blend

int test_blend(std::uint32_t seed) {
    constexpr int S = 3;
    const auto dest = randn_f32(static_cast<std::size_t>(kD) * kAll * S, seed, 0.7F);
    const auto msg  = randn_f32(static_cast<std::size_t>(kD) * kAll * S, seed + 1, 0.7F);
    const auto gf   = randn_f32(static_cast<std::size_t>(2 * kD) * kAll * S, seed + 2, 1.0F);
    const auto w    = randn_bf16(static_cast<std::size_t>(kD) * S, seed + 3, 1.0F);
    DeviceBuffer d_w = dev_bf16(w);
    Tensor w_t(d_w.p, DType::BF16, {kD, S});
    const auto bytes_of = [](const std::vector<float>& v) {
        std::vector<std::uint8_t> out(v.size() * 4);
        std::memcpy(out.data(), v.data(), out.size());
        return out;
    };
    const auto dest_b = bytes_of(dest), msg_b = bytes_of(msg), gf_b = bytes_of(gf);
    const auto run = [&](int c0, int T) {
        const auto sub_d = gather(dest_b, kD, kAll, S, c0, T, 4), sub_m = gather(msg_b, kD, kAll, S, c0, T, 4);
        const auto sub_g = gather(gf_b, 2 * kD, kAll, S, c0, T, 4);
        DeviceBuffer dd = to_device(sub_d), dm = to_device(sub_m), dg = to_device(sub_g);
        DeviceBuffer d_pair(static_cast<std::size_t>(2 * kD) * T * S * 2), d_out(static_cast<std::size_t>(kD) * T * S * 2);
        Tensor td(dd.p, DType::FP32, {kD, T, S}), tm(dm.p, DType::FP32, {kD, T, S}), tg(dg.p, DType::FP32, {2 * kD, T, S});
        Tensor tp(d_pair.p, DType::BF16, {2 * kD, T, S}), to(d_out.p, DType::BF16, {kD, T, S});
        ops::inkling_pair_rms_norm(td, tm, tp, nullptr);
        ops::inkling_blend_rms_norm(td, tm, tg, w_t, 1e-6F, to, nullptr);
        cuda_synchronize();
        return std::make_pair(raw(d_pair.p, d_pair.bytes), raw(d_out.p, d_out.bytes));
    };
    const auto full = run(0, kAll);
    int failures    = 0;
    const int windows[][2] = {{0, 1}, {299, 1}, {100, 37}, {20, 257}};
    for (const auto& win : windows) {
        const auto sub          = run(win[0], win[1]);
        const std::string where = " T=" + std::to_string(win[1]) + " at " + std::to_string(win[0]) + " == T=300";
        failures += same_bits("I3 inkling_pair_rms_norm S=3" + where, sub.first,
                              gather(full.first, 2 * kD, kAll, S, win[0], win[1], 2), 2);
        failures += same_bits("I3 inkling_blend_rms_norm S=3" + where, sub.second,
                              gather(full.second, kD, kAll, S, win[0], win[1], 2), 2);
    }
    return failures;
}

/// The fused controller inputs (inkling_blend_inputs) == attn_res(dest) + attn_res(message) + pair_rms_norm bit for
/// bit, on the real layout (stack and bank slots in the [7 x 2304] residual), and a window == the full round.
int test_blend_inputs(std::uint32_t seed, int stack, bool addend, int dq, int streams, int first_stream) {
    constexpr int kSlots = 7;
    const auto residual = randn_bf16(static_cast<std::size_t>(kAll) * kSlots * kD, seed, 0.8F);
    const auto add      = randn_bf16(static_cast<std::size_t>(kAll) * kD, seed + 1, 0.3F);
    const auto dqv      = randn_f32(static_cast<std::size_t>(dq) * kD, seed + 2, 1.0F / 48.0F);
    const auto mqv      = randn_f32(static_cast<std::size_t>(4) * kD, seed + 3, 1.0F / 48.0F);
    DeviceBuffer d_res = dev_bf16(residual), d_add = dev_bf16(add), d_dq = to_device(dqv), d_mq = to_device(mqv);
    Tensor dq_t(d_dq.p, DType::FP32, {kD, dq});
    Tensor mq_t(static_cast<float*>(d_mq.p) + static_cast<std::size_t>(first_stream) * kD, DType::FP32, {kD, streams});
    const auto sources = [&](int c0, ops::InklingAttnResSources& dst, ops::InklingAttnResSources& bank) {
        auto* base      = static_cast<std::uint16_t*>(d_res.p) + static_cast<std::size_t>(c0) * kSlots * kD;
        dst.stack_count = stack;
        for (int n = 0; n < stack; ++n) { dst.stack[n] = ops::InklingPlane{base + n * kD, DType::BF16, kSlots * kD}; }
        dst.last = ops::InklingPlane{base + 6 * kD, DType::BF16, kSlots * kD};
        if (addend) {
            dst.addend = ops::InklingPlane{static_cast<std::uint16_t*>(d_add.p) + static_cast<std::size_t>(c0) * kD,
                                           DType::BF16, kD};
        }
        bank.stack_count = 1;
        bank.stack[0]    = ops::InklingPlane{base + 2 * kD, DType::BF16, kSlots * kD};
        bank.last        = ops::InklingPlane{base + 3 * kD, DType::BF16, kSlots * kD};
    };
    struct Out {
        std::vector<std::uint8_t> dest, msg, pair;
    };
    const auto run = [&](int c0, int T, bool fused) {
        ops::InklingAttnResSources dst, bank;
        sources(c0, dst, bank);
        DeviceBuffer dd(static_cast<std::size_t>(kD) * T * dq * 4), dm(static_cast<std::size_t>(kD) * T * streams * 4),
            dp(static_cast<std::size_t>(2 * kD) * T * streams * 2);
        Tensor td(dd.p, DType::FP32, {kD, T, dq}), tm(dm.p, DType::FP32, {kD, T, streams}),
            tp(dp.p, DType::BF16, {2 * kD, T, streams});
        if (fused) {
            ops::inkling_blend_inputs(dst, dq_t, bank, mq_t, T, td, tm, tp, nullptr);
        } else {
            ops::inkling_attn_res(dst, T, dq_t, nullptr, 0.0F, td, nullptr);
            ops::inkling_attn_res(bank, T, mq_t, nullptr, 0.0F, tm, nullptr);
            ops::inkling_pair_rms_norm(td, tm, tp, nullptr);
        }
        cuda_synchronize();
        return Out{raw(dd.p, dd.bytes), raw(dm.p, dm.bytes), raw(dp.p, dp.bytes)};
    };
    const std::string tag = "I3 inkling_blend_inputs stack " + std::to_string(stack) + (addend ? " +addend" : "") +
                            " DQ=" + std::to_string(dq) + " S=" + std::to_string(streams);
    const Out f = run(0, kAll, true), u = run(0, kAll, false);
    int failures = 0;
    failures += same_bits(tag + " fused == unfused: dest", f.dest, u.dest, 4);
    failures += same_bits(tag + " fused == unfused: message", f.msg, u.msg, 4);
    failures += same_bits(tag + " fused == unfused: pair", f.pair, u.pair, 2);
    const Out w = run(100, 37, true);
    failures += same_bits(tag + " T=37 at 100 == T=300: pair", w.pair, gather(f.pair, 2 * kD, kAll, streams, 100, 37, 2), 2);
    failures += same_bits(tag + " T=37 at 100 == T=300: dest", w.dest, gather(f.dest, kD, kAll, dq, 100, 37, 4), 4);
    return failures;
}

// ------------------------------------------------------------------------------------- I4 conv paths

int test_conv_paths(std::uint32_t seed) {
    constexpr int C = kD, kSlotsN = 8;
    const std::size_t slot_elems = static_cast<std::size_t>(G::kConvChannels) * 3;
    const auto w = randn_bf16(static_cast<std::size_t>(C) * 4, seed, 0.4F);
    DeviceBuffer d_w = dev_bf16(w);
    Tensor w_t(d_w.p, DType::BF16, {4, C});
    DeviceBuffer d_pool(slot_elems * kSlotsN * 2);
    d_pool.fill(0); // every sequence starts at position 0 with an empty window
    const ops::InklingConvStateView view{d_pool.p, static_cast<std::int64_t>(slot_elems), G::kConvChannels};
    const ops::InklingConvPool pool{d_pool.p, static_cast<std::int64_t>(slot_elems * kSlotsN),
                                    static_cast<std::int64_t>(slot_elems), 1};
    const auto x = randn_bf16(static_cast<std::size_t>(kAll) * C, seed + 1, 1.0F);
    const auto y = randn_bf16(static_cast<std::size_t>(kAll) * C, seed + 2, 1.0F); // the co-scheduled sequence
    const auto res = randn_bf16(static_cast<std::size_t>(kAll) * C, seed + 3, 1.0F);
    const auto add = randn_bf16(static_cast<std::size_t>(kAll) * C, seed + 4, 1.0F);
    DeviceBuffer d_x = dev_bf16(x), d_y = dev_bf16(y);
    const auto col = [&](const DeviceBuffer& b, int c) {
        return ops::InklingPlane{static_cast<std::uint16_t*>(b.p) + static_cast<std::size_t>(c) * C, DType::BF16, C};
    };
    std::vector<int> pos(kAll);
    std::iota(pos.begin(), pos.end(), 0);
    DeviceBuffer d_pos = to_device(pos);
    // A segment / single-row block of `n` columns from column c0 of `src` (positions c0..), state slot `slot`.
    const auto conv = [&](const DeviceBuffer& src, int c0, int n, int slot, DeviceBuffer& out, const ops::InklingPlane* r,
                          const ops::InklingPlane* a, bool in_place) {
        std::vector<int> sl{slot};
        DeviceBuffer d_sl = to_device(sl);
        const ops::InklingConvRows rows{n, 1, Tensor(static_cast<int*>(d_pos.p) + c0, DType::I32, {n, 1}), Tensor{},
                                        Tensor(d_sl.p, DType::I32, {1})};
        const ops::InklingPlane o = in_place ? *r : col(out, c0);
        ops::inkling_short_conv(col(src, c0), C, w_t, rows, view, G::kStreamAttn, r, a, o, nullptr);
        cuda_synchronize();
    };
    const auto slot_bytes = [&](int slot) {
        const auto all = raw(d_pool.p, d_pool.bytes);
        return std::vector<std::uint8_t>(all.begin() + static_cast<std::ptrdiff_t>(slot) * slot_elems * 2,
                                         all.begin() + static_cast<std::ptrdiff_t>(slot + 1) * slot_elems * 2);
    };
    const std::size_t out_bytes = static_cast<std::size_t>(kAll) * C * 2;
    int failures                = 0;

    // P1: one prefill segment of 300 columns (slot 0) -- the reference
    DeviceBuffer o1(out_bytes);
    conv(d_x, 0, kAll, 0, o1, nullptr, nullptr, false);
    const auto ref_out = raw(o1.p, out_bytes), ref_state = slot_bytes(0);

    // P2: chunked prefill, cuts at 1, 63, 64, 65, 200 (slot 1)
    {
        DeviceBuffer o(out_bytes);
        const int cuts[] = {0, 1, 63, 64, 65, 200, kAll};
        for (int i = 0; i + 1 < 7; ++i) { conv(d_x, cuts[i], cuts[i + 1] - cuts[i], 1, o, nullptr, nullptr, false); }
        failures += same_bits("I4 conv chunked (cuts 1/63/64/65/200) == one prefill: outputs", raw(o.p, out_bytes), ref_out, 2);
        failures += same_bits("I4 conv chunked == one prefill: state", slot_bytes(1), ref_state, 2);
    }
    // P3: token-by-token decode (slot 2)
    {
        DeviceBuffer o(out_bytes);
        for (int c = 0; c < kAll; ++c) { conv(d_x, c, 1, 2, o, nullptr, nullptr, false); }
        failures += same_bits("I4 conv decode x300 == one prefill: outputs", raw(o.p, out_bytes), ref_out, 2);
        failures += same_bits("I4 conv decode == one prefill: state", slot_bytes(2), ref_state, 2);
    }
    // P4: verify rounds W = 4: backup, in-place conv over the 4 drafted columns, record, fold with the committed count;
    //     the rejected columns are drafted again next round (commits cycle 4, 2, 0, 1, 3).
    {
        DeviceBuffer o(out_bytes), scratch(static_cast<std::size_t>(4) * C * 2);
        DeviceBuffer d_old(static_cast<std::size_t>(1) * 3 * G::kConvChannels * 2);
        DeviceBuffer d_in(static_cast<std::size_t>(1) * 4 * G::kConvChannels * 2);
        d_in.fill(0);
        const ops::InklingConvRecords rec{d_old.p, d_in.p, 1, 4, 1};
        std::vector<int> sl{3};
        DeviceBuffer d_sl = to_device(sl);
        Tensor sl_t(d_sl.p, DType::I32, {1});
        const int commits[] = {4, 2, 0, 1, 3};
        int p = 0, round = 0, rounds = 0;
        std::vector<std::uint8_t> got(out_bytes);
        while (p < kAll && rounds < 2000) {
            const int W = std::min(4, kAll - p);
            const int c = std::min(commits[round++ % 5], W);
            ops::inkling_conv_backup(pool, sl_t, 1, rec, nullptr);
            const ops::InklingConvRows rows{W, 1, Tensor(static_cast<int*>(d_pos.p) + p, DType::I32, {W, 1}), Tensor{}, sl_t};
            ops::inkling_short_conv(col(d_x, p), C, w_t, rows, view, G::kStreamAttn, nullptr, nullptr,
                                    ops::InklingPlane{scratch.p, DType::BF16, C}, nullptr);
            ops::inkling_conv_record(col(d_x, p), C, W, 1, 0, G::kStreamAttn, rec, nullptr);
            const std::vector<ops::InklingConvFoldRow> fold{{0, 3, c}};
            ops::inkling_conv_fold(pool, fold, rec, nullptr);
            cuda_synchronize();
            const auto s = raw(scratch.p, static_cast<std::size_t>(W) * C * 2);
            std::memcpy(got.data() + static_cast<std::size_t>(p) * C * 2, s.data(), static_cast<std::size_t>(c) * C * 2);
            p += c;
            ++rounds;
        }
        failures += same_bits("I4 conv verify W=4 + fold (commits 4/2/0/1/3, " + std::to_string(rounds) +
                                  " rounds) == one prefill: committed outputs",
                              got, ref_out, 2);
        // the attn range of slot 3 (the fold rewrites every channel; K / V / mlp records are zero inputs here)
        const auto s3 = slot_bytes(3), s0 = ref_state;
        std::vector<std::uint8_t> a3, a0;
        for (int j = 0; j < 3; ++j) {
            const std::size_t at = (static_cast<std::size_t>(j) * G::kConvChannels + G::kStreamAttn) * 2;
            a3.insert(a3.end(), s3.begin() + static_cast<std::ptrdiff_t>(at), s3.begin() + static_cast<std::ptrdiff_t>(at + C * 2));
            a0.insert(a0.end(), s0.begin() + static_cast<std::ptrdiff_t>(at), s0.begin() + static_cast<std::ptrdiff_t>(at + C * 2));
        }
        failures += same_bits("I4 conv verify + fold == one prefill: state (attn range)", a3, a0, 2);
    }
    // P5: decode in a block of B = 2 beside another sequence (x in slot 4, y in slot 5, positions differ by 1000);
    //     y alone as one prefill in slot 6.
    {
        DeviceBuffer o(out_bytes), oy(out_bytes), oy_ref(out_bytes), both(static_cast<std::size_t>(2) * C * 2);
        DeviceBuffer d_xy(static_cast<std::size_t>(2) * C * 2);
        std::vector<int> sl{4, 5};
        DeviceBuffer d_sl = to_device(sl);
        for (int c = 0; c < kAll; ++c) {
            // block columns: row 0 = x[c], row 1 = y[c]
            CUDA_CHECK(cudaMemcpy(d_xy.p, static_cast<std::uint16_t*>(d_x.p) + static_cast<std::size_t>(c) * C, C * 2,
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(static_cast<std::uint16_t*>(d_xy.p) + C, static_cast<std::uint16_t*>(d_y.p) + static_cast<std::size_t>(c) * C,
                                  C * 2, cudaMemcpyDeviceToDevice));
            std::vector<int> pc{c, 1000 + c};
            DeviceBuffer d_pc = to_device(pc);
            const ops::InklingConvRows rows{1, 2, Tensor(d_pc.p, DType::I32, {1, 2}), Tensor{}, Tensor(d_sl.p, DType::I32, {2})};
            ops::inkling_short_conv(ops::InklingPlane{d_xy.p, DType::BF16, C}, C, w_t, rows, view, G::kStreamAttn, nullptr,
                                    nullptr, ops::InklingPlane{both.p, DType::BF16, C}, nullptr);
            CUDA_CHECK(cudaMemcpy(static_cast<std::uint16_t*>(o.p) + static_cast<std::size_t>(c) * C, both.p, C * 2,
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(static_cast<std::uint16_t*>(oy.p) + static_cast<std::size_t>(c) * C,
                                  static_cast<std::uint16_t*>(both.p) + C, C * 2, cudaMemcpyDeviceToDevice));
        }
        // y alone: positions 1000.. (the kernel masks taps before position 0 only, so 1000 + c reads the slot's zeros
        // exactly like position c would)
        std::vector<int> py(kAll);
        std::iota(py.begin(), py.end(), 1000);
        DeviceBuffer d_py = to_device(py);
        std::vector<int> s6{6};
        DeviceBuffer d_s6 = to_device(s6);
        const ops::InklingConvRows rows_y{kAll, 1, Tensor(d_py.p, DType::I32, {kAll, 1}), Tensor{}, Tensor(d_s6.p, DType::I32, {1})};
        ops::inkling_short_conv(col(d_y, 0), C, w_t, rows_y, view, G::kStreamAttn, nullptr, nullptr, col(oy_ref, 0), nullptr);
        cuda_synchronize();
        failures += same_bits("I4 conv decode in a B=2 block == one prefill (sequence x)", raw(o.p, out_bytes), ref_out, 2);
        failures += same_bits("I4 conv decode in a B=2 block == alone (sequence y)", raw(oy.p, out_bytes),
                              raw(oy_ref.p, out_bytes), 2);
        failures += same_bits("I4 conv B=2 block: state of x == one prefill", slot_bytes(4), ref_state, 2);
        failures += same_bits("I4 conv B=2 block: state of y == y alone", slot_bytes(5), slot_bytes(6), 2);
    }
    // P6: the attention-output conv as the visit runs it: residual + addend, written in place over the residual;
    //     one prefill (slot 7) vs token-by-token decode (slot 0 reused after a reset of its window).
    {
        DeviceBuffer r1 = dev_bf16(res), r2 = dev_bf16(res), a1 = dev_bf16(add);
        const ops::InklingPlane ap{a1.p, DType::BF16, C};
        ops::InklingPlane rp1{r1.p, DType::BF16, C};
        conv(d_x, 0, kAll, 7, r1, &rp1, &ap, true);
        CUDA_CHECK(cudaMemset(static_cast<std::uint16_t*>(d_pool.p), 0, slot_elems * 2)); // slot 0 empty again
        DeviceBuffer dummy(16);
        for (int c = 0; c < kAll; ++c) {
            const ops::InklingPlane rc = col(r2, c), acol = col(a1, c);
            conv(d_x, c, 1, 0, dummy, &rc, &acol, true);
        }
        failures += same_bits("I4 conv residual + addend in place: decode x300 == one prefill", raw(r2.p, out_bytes),
                              raw(r1.p, out_bytes), 2);
    }
    return failures;
}

// ------------------------------------------------------------------------------------- I5 trainer log-prob

/// inkling_head_logprob == the engine's own path (invariant FP32 head + softcap, then the scoring kernel through
/// score_logprobs_batch) bit for bit; one column alone == in the round; an id outside the domain gives NaN.
int test_head_logprob(std::uint32_t seed) {
    constexpr int kRows = 32768, kDomain = 32000, T = 64;
    const auto w_bits = hash_bf16(static_cast<std::size_t>(kRows) * kD, seed, 0.06F);
    const auto h_bits = hash_bf16(static_cast<std::size_t>(T) * kD, seed + 1, 1.0F);
    std::vector<int> ids(T);
    std::mt19937 g(seed + 2);
    for (auto& id : ids) { id = static_cast<int>(g() % kDomain); }
    ids[T - 1] = kDomain + 5; // outside the domain
    DeviceBuffer d_w = to_device(w_bits), d_h = to_device(h_bits), d_ids = to_device(ids);
    const auto run = [&](int c0, int n) {
        const std::size_t ws = ops::inkling_head_logprob_workspace_bytes(kRows, n);
        DeviceBuffer d_ws(ws), d_out(static_cast<std::size_t>(n) * 4);
        ops::inkling_head_logprob(d_w.p, kRows, kD, kDomain, 30.0F,
                                  static_cast<const std::uint16_t*>(d_h.p) + static_cast<std::size_t>(c0) * kD, n,
                                  static_cast<const std::int32_t*>(d_ids.p) + c0, static_cast<float*>(d_out.p), d_ws.p,
                                  ws, nullptr);
        cuda_synchronize();
        return from_device<float>(d_out.p, static_cast<std::size_t>(n));
    };
    const auto full = run(0, T);
    // the engine's path for the T - 1 in-domain columns
    DeviceBuffer d_logits(static_cast<std::size_t>(kRows) * T * 4);
    ops::inkling_gemm_inv(d_w.p, kRows, kD, d_h.p, T, d_logits.p, DType::FP32, kRows, {}, nullptr, 0, nullptr);
    ops::inkling_softcap_f32(static_cast<float*>(d_logits.p), kRows, T, kRows, 30.0F, nullptr);
    cuda_synchronize();
    const Tensor logits(d_logits.p, DType::FP32, {kRows, T});
    std::vector<TokenId> valid(ids.begin(), ids.end() - 1);
    const auto ref = ops::score_logprobs_batch(logits, valid, kDomain, 0, nullptr);
    std::size_t bad = 0;
    for (int t = 0; t + 1 < T; ++t) { bad += std::memcmp(&full[t], &ref[t].selected.logprob, 4) != 0; }
    int failures = 0;
    std::cout << (bad ? "FAIL " : "ok   ") << "I5 inkling_head_logprob == head + score_logprobs_batch (" << T - 1
              << " columns): " << bad << " differ\n";
    failures += bad ? 1 : 0;
    const bool nan_last = std::isnan(full[T - 1]);
    std::cout << (nan_last ? "ok   " : "FAIL ") << "I5 an id outside the domain gives NaN\n";
    failures += nan_last ? 0 : 1;
    std::size_t bad_w = 0;
    for (const int c : {0, 17, T - 2}) {
        const auto one = run(c, 1);
        bad_w += std::memcmp(&one[0], &full[c], 4) != 0;
    }
    std::cout << (bad_w ? "FAIL " : "ok   ") << "I5 one column alone == in the round (3 columns): " << bad_w << " differ\n";
    failures += bad_w ? 1 : 0;
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    const GemmShape shapes[] = {
        {"qkvr q-side 6552 x 2304 bf16", 6552, kD, 1, DType::BF16},
        {"wo_ud 2304 x 3072 bf16", kD, 3072, 1, DType::BF16},
        {"dynamic taps 32 x 2304 bf16", 32, kD, 1, DType::BF16},
        {"controller down 64 x 4608 bf16, 3 streams", 64, 2 * kD, 3, DType::BF16},
        {"controller gain/forget 4608 x 64 fp32, 3 streams", 2 * kD, 64, 3, DType::FP32},
        {"LM head 201024 x 2304 fp32", 201024, kD, 1, DType::FP32},
    };
    std::uint32_t seed = 500;
    for (const auto& g : shapes) { failures += test_gemm(g, seed += 100); }
    failures += test_gemm_group(2900);
    failures += test_attn_res(3001);
    failures += test_blend(3002);
    failures += test_blend_inputs(3101, 2, false, 1, 3, 0); // looped attention inputs, local layer (dest broadcast)
    failures += test_blend_inputs(3102, 4, false, 3, 3, 0); // looped global layer: 3 dest queries
    failures += test_blend_inputs(3103, 3, true, 1, 1, 3);  // MLP blend: partial + attn_out addend, mlp stream
    failures += test_conv_paths(3003);
    failures += test_head_logprob(3201);
    std::cout << (failures ? "FAIL" : "PASS") << " inkling_invariance (" << failures << " failures)\n";
    return failures ? 1 : 0;
}
