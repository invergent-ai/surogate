// Inkling-2-Mini hidden-stream ops, tests T1-T5 of the target design (section 8) against
// double-precision CPU ports of the `inkling_ref.py` functions of the same names. The inputs are on
// the bf16 grid the kernels read; outputs rounded to bf16 by the reference are compared within one
// bf16 ulp, fp32 outputs within a few fp32 ulps of the reduction, conv states exactly.
// When the golden vectors exist (vectors/<op>/), a second test reads them; this one needs none.

#include "api/ops/inkling_mix.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr int kD = 2304;

float rb(float x) { return bf16_to_f32(f32_to_bf16(x)); }

std::vector<float> randn_bf16(std::size_t n, std::uint32_t seed, float scale) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) { x = rb(d(g)); }
    return v;
}

std::vector<std::uint16_t> bits(const std::vector<float>& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { out[i] = f32_to_bf16(v[i]); }
    return out;
}

DeviceBuffer dev_bf16(const std::vector<float>& v) { return to_device(bits(v)); }

std::vector<float> host_bf16(const void* p, std::size_t n) {
    const auto raw = from_device<std::uint16_t>(p, n);
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = bf16_to_f32(raw[i]); }
    return out;
}

/// |got - ref| <= atol + rtol * |ref| for every element; prints the worst one.
int compare(const std::string& label, const std::vector<float>& got, const std::vector<double>& ref,
            double atol, double rtol) {
    if (got.size() != ref.size()) {
        std::cerr << label << ": size " << got.size() << " vs " << ref.size() << "\n";
        return 1;
    }
    double worst = 0.0;
    std::size_t at = 0, bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double err   = std::fabs(static_cast<double>(got[i]) - ref[i]);
        const double limit = atol + rtol * std::fabs(ref[i]);
        if (!(err <= limit)) { ++bad; }
        const double ratio = err / limit;
        if (!(ratio <= worst)) { worst = ratio; at = i; }
    }
    std::cout << (bad ? "FAIL " : "ok   ") << label << ": worst err/limit " << worst << " at " << at
              << " (got " << got[at] << " ref " << ref[at] << "), " << bad << " over\n";
    return bad ? 1 : 0;
}

int compare_exact(const std::string& label, const std::vector<float>& got, const std::vector<float>& ref) {
    std::size_t bad = 0, first = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (std::memcmp(&got[i], &ref[i], sizeof(float)) != 0) {
            if (bad == 0) { first = i; }
            ++bad;
        }
    }
    std::cout << (bad ? "FAIL " : "ok   ") << label << ": " << bad << " differ";
    if (bad) { std::cout << " (first " << first << ": " << got[first] << " vs " << ref[first] << ")"; }
    std::cout << "\n";
    return bad ? 1 : 0;
}

constexpr double kBf16Rel = 1.0 / 128.0; // one bf16 ulp at the bottom of a binade

/// |got - ref| <= limit[i] for every element (a limit per element: see test_attn_res).
int compare_limits(const std::string& label, const std::vector<float>& got, const std::vector<double>& ref,
                   const std::vector<double>& limit) {
    double worst = 0.0;
    std::size_t at = 0, bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double err   = std::fabs(static_cast<double>(got[i]) - ref[i]);
        if (!(err <= limit[i])) { ++bad; }
        const double ratio = err / limit[i];
        if (!(ratio <= worst)) { worst = ratio; at = i; }
    }
    std::cout << (bad ? "FAIL " : "ok   ") << label << ": worst err/limit " << worst << " at " << at
              << " (got " << got[at] << " ref " << ref[at] << "), " << bad << " over\n";
    return bad ? 1 : 0;
}

// ------------------------------------------------------------------------------------- T1 embed

int test_embed() {
    const int rows = 97, T = 33;
    const auto table = randn_bf16(static_cast<std::size_t>(rows) * kD, 11, 0.02F);
    const auto w     = randn_bf16(kD, 12, 1.0F);
    std::vector<int> ids(T);
    for (int t = 0; t < T; ++t) { ids[t] = (t * 37 + 5) % rows; }
    const int slots = 7; // write into a strided plane, as slot 0 of the stack
    DeviceBuffer d_table = dev_bf16(table), d_w = dev_bf16(w), d_ids = to_device(ids);
    DeviceBuffer d_out(static_cast<std::size_t>(T) * slots * kD * 2);
    d_out.fill(0);
    Tensor ids_t(d_ids.p, DType::I32, {T});
    Tensor w_t(d_w.p, DType::BF16, {kD});
    ops::inkling_embed(ids_t, d_table.p, rows, w_t, 1e-6F,
                       ops::InklingPlane{d_out.p, DType::BF16, slots * kD}, nullptr);
    cuda_synchronize();
    const auto out = host_bf16(d_out.p, static_cast<std::size_t>(T) * slots * kD);
    std::vector<float> got;
    std::vector<double> ref;
    for (int t = 0; t < T; ++t) {
        const float* row = table.data() + static_cast<std::size_t>(ids[t]) * kD;
        double ss = 0;
        for (int d = 0; d < kD; ++d) { ss += static_cast<double>(row[d]) * row[d]; }
        const double rstd = 1.0 / std::sqrt(ss / kD + 1e-6);
        for (int d = 0; d < kD; ++d) {
            got.push_back(out[static_cast<std::size_t>(t) * slots * kD + d]);
            ref.push_back(row[d] * rstd * w[d]);
        }
    }
    return compare("T1 inkling_embed (strided slot 0)", got, ref, 1e-30, kBf16Rel);
}

// ---------------------------------------------------------------------------------- T2 attn_res

struct AttnResCase {
    const char* name;
    int stack, queries, columns;
    bool addend, fp32_last, norm, strided;
};

int test_attn_res(const AttnResCase& c, std::uint32_t seed) {
    const int N = c.stack + 1, T = c.columns, Q = c.queries;
    const int slots = 7, stride = c.strided ? slots * kD : kD;
    // Stack sources live in one buffer: strided = slot s of [7 x 2304] per column, else packed planes.
    std::vector<float> stack_host(static_cast<std::size_t>(T) * slots * kD, 0.0F);
    std::vector<std::vector<float>> src(N, std::vector<float>(static_cast<std::size_t>(T) * kD));
    for (int n = 0; n < c.stack; ++n) {
        const auto v = randn_bf16(static_cast<std::size_t>(T) * kD, seed + 10 * n, 0.5F + 0.4F * n);
        src[n]       = v;
    }
    auto last = randn_bf16(static_cast<std::size_t>(T) * kD, seed + 99, 0.7F);
    if (c.fp32_last) {
        std::mt19937 g(seed + 5);
        std::normal_distribution<float> d(0.0F, 0.7F);
        for (auto& x : last) { x = d(g); }
    }
    const auto addend = randn_bf16(static_cast<std::size_t>(T) * kD, seed + 77, 0.3F);
    // fp32 sum of last + addend, exactly as the kernel forms it
    src[N - 1] = last;
    if (c.addend) {
        for (std::size_t i = 0; i < last.size(); ++i) { src[N - 1][i] = last[i] + addend[i]; }
    }
    std::vector<DeviceBuffer> planes;
    ops::InklingAttnResSources s;
    s.stack_count = c.stack;
    DeviceBuffer d_stack;
    if (c.strided) {
        for (int n = 0; n < c.stack; ++n) {
            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < kD; ++d) {
                    stack_host[(static_cast<std::size_t>(t) * slots + n) * kD + d] = src[n][static_cast<std::size_t>(t) * kD + d];
                }
            }
        }
        d_stack = dev_bf16(stack_host);
        for (int n = 0; n < c.stack; ++n) {
            s.stack[n] = ops::InklingPlane{static_cast<std::uint16_t*>(d_stack.p) + n * kD, DType::BF16, stride};
        }
    } else {
        for (int n = 0; n < c.stack; ++n) {
            planes.push_back(dev_bf16(src[n]));
            s.stack[n] = ops::InklingPlane{planes.back().p, DType::BF16, kD};
        }
    }
    DeviceBuffer d_last = c.fp32_last ? to_device(last) : dev_bf16(last);
    s.last              = ops::InklingPlane{d_last.p, c.fp32_last ? DType::FP32 : DType::BF16, kD};
    DeviceBuffer d_add  = dev_bf16(addend);
    if (c.addend) { s.addend = ops::InklingPlane{d_add.p, DType::BF16, kD}; }
    // queries fp32, pre-scaled
    std::vector<float> q(static_cast<std::size_t>(Q) * kD);
    {
        std::mt19937 g(seed + 3);
        std::normal_distribution<float> d(0.0F, 1.0F);
        for (auto& x : q) { x = d(g) * (1.0F / 48.0F); }
    }
    const auto w = randn_bf16(static_cast<std::size_t>(Q) * kD, seed + 4, 1.0F);
    DeviceBuffer d_q = to_device(q), d_w = dev_bf16(w);
    Tensor q_t(d_q.p, DType::FP32, {kD, Q});
    Tensor w_t(d_w.p, DType::BF16, {kD, Q});
    const std::size_t out_n = static_cast<std::size_t>(Q) * T * kD;
    DeviceBuffer d_out(out_n * (c.norm ? 2 : 4));
    Tensor out_t(d_out.p, c.norm ? DType::BF16 : DType::FP32, {kD, T, Q});
    ops::inkling_attn_res(s, T, q_t, c.norm ? &w_t : nullptr, 1e-6F, out_t, nullptr);
    cuda_synchronize();
    std::vector<float> got = c.norm ? host_bf16(d_out.p, out_n) : from_device<float>(d_out.p, out_n);
    std::vector<double> ref(out_n), limit(out_n);
    // The kernel sums the N mix terms in fp32: where they cancel (|mix| << sum |term|) its value carries an absolute
    // error of a few fp32 ulps of the terms, which no relative bf16 bound on |ref| covers (5 of 592k elements at
    // T = 257 in one session). Limit = the output's own rounding (1 bf16 ulp, or the fp32 bound) + 8 fp32 ulps
    // of sum |term|, carried through the norm.
    constexpr double kF32Ulp = 1.0 / 16777216.0;
    for (int t = 0; t < T; ++t) {
        std::vector<double> rstd(N);
        for (int n = 0; n < N; ++n) {
            double ss = 0;
            for (int d = 0; d < kD; ++d) {
                const double x = src[n][static_cast<std::size_t>(t) * kD + d];
                ss += x * x;
            }
            rstd[n] = 1.0 / std::sqrt(ss / kD + 1e-6);
        }
        for (int qq = 0; qq < Q; ++qq) {
            std::vector<double> logit(N);
            double mx = -1e300;
            for (int n = 0; n < N; ++n) {
                double dot = 0;
                for (int d = 0; d < kD; ++d) {
                    dot += static_cast<double>(q[static_cast<std::size_t>(qq) * kD + d]) *
                           src[n][static_cast<std::size_t>(t) * kD + d];
                }
                logit[n] = dot * rstd[n];
                mx       = std::max(mx, logit[n]);
            }
            double tot = 0;
            std::vector<double> p(N);
            for (int n = 0; n < N; ++n) { tot += (p[n] = std::exp(logit[n] - mx)); }
            std::vector<double> mix(kD, 0.0), mag(kD, 0.0);
            for (int n = 0; n < N; ++n) {
                for (int d = 0; d < kD; ++d) {
                    const double term = p[n] / tot * rstd[n] * src[n][static_cast<std::size_t>(t) * kD + d];
                    mix[d] += term;
                    mag[d] += std::fabs(term);
                }
            }
            double ss = 0;
            for (double m : mix) { ss += m * m; }
            const double orstd = 1.0 / std::sqrt(ss / kD + 1e-6);
            for (int d = 0; d < kD; ++d) {
                const std::size_t i = (static_cast<std::size_t>(qq) * T + t) * kD + d;
                const double wd     = w[static_cast<std::size_t>(qq) * kD + d];
                ref[i]              = c.norm ? mix[d] * orstd * wd : mix[d];
                const double floor  = 8.0 * kF32Ulp * mag[d] * (c.norm ? orstd * std::fabs(wd) : 1.0);
                limit[i] = (c.norm ? 1e-30 + kBf16Rel * std::fabs(ref[i]) : 2e-6 + 3e-5 * std::fabs(ref[i])) + floor;
            }
        }
    }
    return compare_limits(std::string("T2 inkling_attn_res ") + c.name, got, ref, limit);
}

// ------------------------------------------------------------------------- T3 pair / blend norms

int test_blend(int S, bool broadcast, int T, std::uint32_t seed) {
    const int SD = broadcast ? 1 : S;
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<float> dest(static_cast<std::size_t>(SD) * T * kD), msg(static_cast<std::size_t>(S) * T * kD);
    for (auto& x : dest) { x = nd(g) * 0.8F; }
    for (auto& x : msg) { x = nd(g) * 1.3F; }
    std::vector<float> gf(static_cast<std::size_t>(S) * T * 2 * kD);
    for (auto& x : gf) { x = nd(g) * 1.5F; }
    const auto w = randn_bf16(static_cast<std::size_t>(broadcast ? 1 : S) * kD, seed + 1, 1.0F);
    DeviceBuffer d_dest = to_device(dest), d_msg = to_device(msg), d_gf = to_device(gf), d_w = dev_bf16(w);
    DeviceBuffer d_pair(static_cast<std::size_t>(S) * T * 2 * kD * 2), d_out(static_cast<std::size_t>(S) * T * kD * 2);
    Tensor dest_t(d_dest.p, DType::FP32, {kD, T, SD}), msg_t(d_msg.p, DType::FP32, {kD, T, S});
    Tensor gf_t(d_gf.p, DType::FP32, {2 * kD, T, S}), w_t(d_w.p, DType::BF16, {kD, broadcast ? 1 : S});
    Tensor pair_t(d_pair.p, DType::BF16, {2 * kD, T, S}), out_t(d_out.p, DType::BF16, {kD, T, S});
    ops::inkling_pair_rms_norm(dest_t, msg_t, pair_t, nullptr);
    ops::inkling_blend_rms_norm(dest_t, msg_t, gf_t, w_t, 1e-6F, out_t, nullptr);
    cuda_synchronize();
    const auto pair = host_bf16(d_pair.p, static_cast<std::size_t>(S) * T * 2 * kD);
    const auto out  = host_bf16(d_out.p, static_cast<std::size_t>(S) * T * kD);
    std::vector<double> ref_pair(pair.size()), ref_out(out.size());
    for (int s = 0; s < S; ++s) {
        for (int t = 0; t < T; ++t) {
            const float* dp = dest.data() + (static_cast<std::size_t>(broadcast ? 0 : s) * T + t) * kD;
            const float* mp = msg.data() + (static_cast<std::size_t>(s) * T + t) * kD;
            const float* gp = gf.data() + (static_cast<std::size_t>(s) * T + t) * 2 * kD;
            double sd = 0, sm = 0;
            for (int d = 0; d < kD; ++d) { sd += double(dp[d]) * dp[d]; sm += double(mp[d]) * mp[d]; }
            const double rd = 1.0 / std::sqrt(sd / kD + 1e-6), rm = 1.0 / std::sqrt(sm / kD + 1e-6);
            std::vector<double> x(kD);
            double sx = 0;
            for (int d = 0; d < kD; ++d) {
                ref_pair[(static_cast<std::size_t>(s) * T + t) * 2 * kD + d]      = dp[d] * rd;
                ref_pair[(static_cast<std::size_t>(s) * T + t) * 2 * kD + kD + d] = mp[d] * rm;
                x[d] = (1.0 - std::tanh(double(gp[kD + d]))) * dp[d] + std::tanh(double(gp[d])) * mp[d];
                sx += x[d] * x[d];
            }
            const double rx = 1.0 / std::sqrt(sx / kD + 1e-6);
            for (int d = 0; d < kD; ++d) {
                ref_out[(static_cast<std::size_t>(s) * T + t) * kD + d] =
                    x[d] * rx * w[static_cast<std::size_t>(broadcast ? 0 : s) * kD + d];
            }
        }
    }
    const std::string tag = " S=" + std::to_string(S) + (broadcast ? " (dest broadcast)" : "") + " T=" + std::to_string(T);
    int f = compare("T3 inkling_pair_rms_norm" + tag, pair, ref_pair, 1e-30, kBf16Rel);
    f += compare("T3 inkling_blend_rms_norm" + tag, out, ref_out, 1e-30, kBf16Rel);
    return f;
}

// ------------------------------------------------------------------------------- T4 short conv

/// The CPU port of `inkling_short_conv` for one row: returns outputs and the new 3-input state.
void conv_ref(const std::vector<float>& x, int channels, int W, const std::vector<float>& w,
              std::vector<float> state, int pos0, const std::vector<float>* residual,
              const std::vector<float>* addend, std::vector<double>& out, std::vector<float>& new_state) {
    out.assign(static_cast<std::size_t>(W) * channels, 0.0);
    std::vector<float> seq(state);
    seq.insert(seq.end(), x.begin(), x.end()); // [3 + W][channels]
    const float scale = 0.44721359549995793F;
    for (int j = 0; j < W; ++j) {
        for (int c = 0; c < channels; ++c) {
            float acc = 0.0F;
            for (int k = 0; k < 4; ++k) {
                const int lag = 3 - k;
                const float v = (pos0 + j - lag >= 0) ? seq[static_cast<std::size_t>(j + k) * channels + c] : 0.0F;
                acc = std::fma(v, w[static_cast<std::size_t>(c) * 4 + k], acc);
            }
            float y = rb((acc + x[static_cast<std::size_t>(j) * channels + c]) * scale);
            if (addend != nullptr) {
                float base = (*addend)[static_cast<std::size_t>(j) * channels + c];
                if (residual != nullptr) { base = (*residual)[static_cast<std::size_t>(j) * channels + c] + base; }
                y = rb(base + y);
            }
            out[static_cast<std::size_t>(j) * channels + c] = y;
        }
    }
    new_state.assign(seq.end() - 3 * channels, seq.end());
}

int test_short_conv(std::uint32_t seed) {
    using G = ops::InklingMixGeometry;
    int failures = 0;
    const int C = kD; // the attn-out stream
    const int slots = 6;
    const auto w = randn_bf16(static_cast<std::size_t>(C) * 4, seed, 0.4F);
    DeviceBuffer d_w = dev_bf16(w);
    Tensor w_t(d_w.p, DType::BF16, {4, C});
    // Pool: [6656, 3, slots]
    const std::size_t slot_elems = static_cast<std::size_t>(G::kConvChannels) * 3;
    std::vector<float> pool(slot_elems * slots);
    {
        const auto r = randn_bf16(pool.size(), seed + 1, 0.9F);
        pool         = r;
    }
    DeviceBuffer d_pool = dev_bf16(pool);
    ops::InklingConvStateView view{d_pool.p, static_cast<std::int64_t>(slot_elems), G::kConvChannels};
    const auto slot_state = [&](const std::vector<float>& p, int slot) {
        std::vector<float> s(static_cast<std::size_t>(3) * C);
        for (int i = 0; i < 3; ++i) {
            for (int c = 0; c < C; ++c) {
                s[static_cast<std::size_t>(i) * C + c] = p[slot * slot_elems + static_cast<std::size_t>(i) * G::kConvChannels + G::kStreamAttn + c];
            }
        }
        return s;
    };

    // (a) one row, W = 70, starting at position 1 (taps at position -2.. read zero), with residual + addend
    //     written in place over the residual (the stack's partial slot), and a capture at column 41.
    {
        const int W = 70, pos0 = 1, slot = 2, cap_col = 41, cap_slot = 4;
        const auto x   = randn_bf16(static_cast<std::size_t>(W) * C, seed + 2, 1.0F);
        const auto res = randn_bf16(static_cast<std::size_t>(W) * C, seed + 3, 1.0F);
        const auto add = randn_bf16(static_cast<std::size_t>(W) * C, seed + 4, 1.0F);
        DeviceBuffer d_x = dev_bf16(x), d_io = dev_bf16(res), d_add = dev_bf16(add);
        std::vector<int> pos(W), slot_v{slot};
        std::iota(pos.begin(), pos.end(), pos0);
        DeviceBuffer d_pos = to_device(pos), d_slot = to_device(slot_v);
        ops::InklingConvRows rows{W, 1, Tensor(d_pos.p, DType::I32, {W, 1}), Tensor{},
                                  Tensor(d_slot.p, DType::I32, {1})};
        ops::InklingPlane xp{d_x.p, DType::BF16, C}, io{d_io.p, DType::BF16, C}, ap{d_add.p, DType::BF16, C};
        ops::inkling_short_conv(xp, C, w_t, rows, view, G::kStreamAttn, &io, &ap, io, nullptr, cap_col, cap_slot);
        cuda_synchronize();
        std::vector<double> ref;
        std::vector<float> new_state;
        conv_ref(x, C, W, w, slot_state(pool, slot), pos0, &res, &add, ref, new_state);
        failures += compare("T4 inkling_short_conv W=70 residual+addend in place", host_bf16(d_io.p, x.size()), ref,
                            1e-30, kBf16Rel);
        const auto after = host_bf16(d_pool.p, pool.size());
        failures += compare_exact("T4 conv state after W=70 (in place)", slot_state(after, slot), new_state);
        // capture: the 3 inputs before column 41 = x columns 38..40
        std::vector<float> cap_ref(x.begin() + static_cast<std::size_t>(cap_col - 3) * C,
                                   x.begin() + static_cast<std::size_t>(cap_col) * C);
        failures += compare_exact("T4 conv capture window at column 41", slot_state(after, cap_slot), cap_ref);
        pool = after;
    }

    // (b) chunked == one segment: 2 chunks of 23 + 17 on slot 1 == one chunk of 40 on slot 3 (same start state)
    {
        const int W = 40, slot_a = 1, slot_b = 3;
        // make the two slots' attn ranges equal
        for (int i = 0; i < 3; ++i) {
            for (int c = 0; c < C; ++c) {
                pool[slot_b * slot_elems + static_cast<std::size_t>(i) * G::kConvChannels + G::kStreamAttn + c] =
                    pool[slot_a * slot_elems + static_cast<std::size_t>(i) * G::kConvChannels + G::kStreamAttn + c];
            }
        }
        d_pool.copy_from_host(bits(pool).data(), d_pool.bytes);
        const auto x = randn_bf16(static_cast<std::size_t>(W) * C, seed + 5, 1.0F);
        DeviceBuffer d_x = dev_bf16(x), d_out1(x.size() * 2), d_out2(x.size() * 2);
        std::vector<int> pos(W);
        std::iota(pos.begin(), pos.end(), 500);
        DeviceBuffer d_pos = to_device(pos);
        std::vector<int> sa{slot_a}, sb{slot_b};
        DeviceBuffer d_sa = to_device(sa), d_sb = to_device(sb);
        const auto rows_of = [&](int w0, int n, DeviceBuffer& s) {
            return ops::InklingConvRows{n, 1, Tensor(static_cast<int*>(d_pos.p) + w0, DType::I32, {n, 1}),
                                        Tensor{}, Tensor(s.p, DType::I32, {1})};
        };
        const auto plane = [&](void* base, int col) {
            return ops::InklingPlane{static_cast<std::uint16_t*>(base) + static_cast<std::size_t>(col) * C, DType::BF16, C};
        };
        ops::inkling_short_conv(plane(d_x.p, 0), C, w_t, rows_of(0, 23, d_sa), view, G::kStreamAttn, nullptr, nullptr,
                                plane(d_out1.p, 0), nullptr);
        ops::inkling_short_conv(plane(d_x.p, 23), C, w_t, rows_of(23, 17, d_sa), view, G::kStreamAttn, nullptr,
                                nullptr, plane(d_out1.p, 23), nullptr);
        ops::inkling_short_conv(plane(d_x.p, 0), C, w_t, rows_of(0, 40, d_sb), view, G::kStreamAttn, nullptr, nullptr,
                                plane(d_out2.p, 0), nullptr);
        cuda_synchronize();
        const auto o1 = host_bf16(d_out1.p, x.size()), o2 = host_bf16(d_out2.p, x.size());
        failures += compare_exact("T4 chunked (23 + 17) == one segment (40): outputs", o1, o2);
        const auto after = host_bf16(d_pool.p, pool.size());
        failures += compare_exact("T4 chunked == one segment: states", slot_state(after, slot_a), slot_state(after, slot_b));
        std::vector<double> ref;
        std::vector<float> ns;
        conv_ref(x, C, W, w, slot_state(pool, slot_a), 500, nullptr, nullptr, ref, ns);
        failures += compare("T4 one segment W=40 vs CPU port", o2, ref, 1e-30, kBf16Rel);
        pool = after;
    }

    // (c) verify B=2 W=4 with records and the fold: run in place, record, then fold committed {2, 0};
    //     == the state after running only the committed columns (and the state before for 0).
    {
        const int W = 4, B = 2;
        const int slot0 = 0, slot1 = 5;
        const auto x = randn_bf16(static_cast<std::size_t>(W) * B * C, seed + 6, 1.0F);
        DeviceBuffer d_x = dev_bf16(x), d_out(x.size() * 2);
        std::vector<int> pos{10, 11, 12, 13, 3, 4, 5, 6}, sl{slot0, slot1};
        DeviceBuffer d_pos = to_device(pos), d_sl = to_device(sl);
        // Records for a 1-visit pool view: visit stride irrelevant (one visit).
        ops::InklingConvPool pv{d_pool.p, static_cast<std::int64_t>(slot_elems * slots), static_cast<std::int64_t>(slot_elems), 1};
        DeviceBuffer d_old(static_cast<std::size_t>(B) * 1 * 3 * G::kConvChannels * 2);
        DeviceBuffer d_in(static_cast<std::size_t>(B) * 1 * W * G::kConvChannels * 2);
        d_in.fill(0);
        ops::InklingConvRecords rec{d_old.p, d_in.p, 1, W, B};
        Tensor sl_t(d_sl.p, DType::I32, {B});
        const auto before = pool;
        ops::inkling_conv_backup(pv, sl_t, B, rec, nullptr);
        ops::InklingConvRows rows{W, B, Tensor(d_pos.p, DType::I32, {W, B}), Tensor{}, sl_t};
        ops::InklingPlane xp{d_x.p, DType::BF16, C}, op{d_out.p, DType::BF16, C};
        ops::inkling_short_conv(xp, C, w_t, rows, view, G::kStreamAttn, nullptr, nullptr, op, nullptr);
        ops::inkling_conv_record(xp, C, W, B, 0, G::kStreamAttn, rec, nullptr);
        cuda_synchronize();
        const auto o = host_bf16(d_out.p, x.size());
        for (int b = 0; b < B; ++b) {
            std::vector<float> xb(x.begin() + static_cast<std::size_t>(b) * W * C, x.begin() + static_cast<std::size_t>(b + 1) * W * C);
            std::vector<double> ref;
            std::vector<float> ns;
            conv_ref(xb, C, W, w, slot_state(before, sl[b]), pos[b * W], nullptr, nullptr, ref, ns);
            std::vector<float> ob(o.begin() + static_cast<std::size_t>(b) * W * C, o.begin() + static_cast<std::size_t>(b + 1) * W * C);
            failures += compare("T4 verify row " + std::to_string(b) + " outputs", ob, ref, 1e-30, kBf16Rel);
        }
        const std::vector<ops::InklingConvFoldRow> fold{{0, slot0, 2}, {1, slot1, 0}};
        ops::inkling_conv_fold(pv, fold, rec, nullptr);
        cuda_synchronize();
        const auto after = host_bf16(d_pool.p, pool.size());
        // row 0, committed 2: [old0 old1 old2 x0 x1] -> last 3 = old2, x0, x1 (other channels untouched: full slot compare
        // on the attn range; K/V/mlp ranges were not in the records' inputs but the fold writes old for them --
        // their inputs are zero, so check only the attn range here)
        std::vector<float> want0 = slot_state(before, slot0);
        std::vector<float> seq(want0);
        seq.insert(seq.end(), x.begin(), x.begin() + static_cast<std::size_t>(2) * C);
        want0.assign(seq.end() - 3 * C, seq.end());
        failures += compare_exact("T4 fold committed 2", slot_state(after, slot0), want0);
        failures += compare_exact("T4 fold committed 0 restores", slot_state(after, slot1), slot_state(before, slot1));
        pool = after;
    }
    return failures;
}

// ----------------------------------------------------------------------------------- T5 lm head

int test_lm_head(std::uint32_t seed) {
    const int n = 1032, k = kD, T = 5;
    const auto w = randn_bf16(static_cast<std::size_t>(n) * k, seed, 0.05F);
    const auto x = randn_bf16(static_cast<std::size_t>(k) * T, seed + 1, 1.0F);
    DeviceBuffer d_w = dev_bf16(w), d_x = dev_bf16(x), d_out(static_cast<std::size_t>(n) * T * 4);
    ops::inkling_gemm_f32_prepare(n, k, T, n);
    ops::inkling_gemm_bf16_f32(d_w.p, n, k, d_x.p, T, static_cast<float*>(d_out.p), n, nullptr);
    const auto raw = [&] {
        cuda_synchronize();
        return from_device<float>(d_out.p, static_cast<std::size_t>(n) * T);
    }();
    ops::inkling_softcap_f32(static_cast<float*>(d_out.p), n, T, n, 30.0F, nullptr);
    cuda_synchronize();
    const auto capped = from_device<float>(d_out.p, static_cast<std::size_t>(n) * T);
    std::vector<double> ref_raw(raw.size()), ref_cap(raw.size());
    double scale = 0;
    for (int t = 0; t < T; ++t) {
        for (int r = 0; r < n; ++r) {
            double s = 0, a = 0;
            for (int i = 0; i < k; ++i) {
                const double p = double(w[static_cast<std::size_t>(r) * k + i]) * x[static_cast<std::size_t>(t) * k + i];
                s += p;
                a += std::fabs(p);
            }
            ref_raw[static_cast<std::size_t>(t) * n + r] = s;
            scale = std::max(scale, a);
        }
    }
    for (std::size_t i = 0; i < raw.size(); ++i) {
        // the softcap of the kernel's own fp32 logit, the reference's order (divide, tanh, multiply)
        ref_cap[i] = std::tanh(double(raw[i]) / 30.0) * 30.0;
    }
    int f = compare("T5 inkling_gemm_bf16_f32 (fp32 out)", raw, ref_raw, 1e-5 * scale, 0.0);
    f += compare("T5 inkling_softcap_f32", capped, ref_cap, 1e-6, 1e-6);
    return f;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += test_embed();
    const AttnResCase cases[] = {
        {"N=1 Q=1 norm (layer-1 attn input)", 0, 1, 7, false, false, true, false},
        {"N=2 Q=3 norm strided (global attn input)", 1, 3, 64, false, false, true, true},
        {"N=3 Q=1 addend norm (mlp input)", 2, 1, 33, true, false, true, true},
        {"N=5 Q=1 addend no-norm fp32 (looped mlp)", 4, 1, 9, true, false, false, true},
        {"N=2 Q=4 fp32 out (messages)", 1, 4, 16, false, false, false, false},
        {"N=6 Q=1 norm (final output mix)", 5, 1, 5, false, false, true, true},
        {"N=4 Q=3 fp32 last + addend", 3, 3, 3, true, true, false, false},
        {"N=3 Q=1 norm T=257", 2, 1, 257, false, false, true, true},
    };
    std::uint32_t seed = 1000;
    for (const auto& c : cases) { failures += test_attn_res(c, seed += 100); }
    failures += test_blend(3, true, 11, 2001);
    failures += test_blend(1, false, 37, 2002);
    failures += test_blend(3, false, 5, 2003);
    failures += test_short_conv(3001);
    failures += test_lm_head(4001);
    std::cout << (failures ? "FAIL" : "OK") << " inkling_mix (" << failures << " failures)\n";
    return failures ? 1 : 0;
}
