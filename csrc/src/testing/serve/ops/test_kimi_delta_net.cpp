// Public-contract qualification for kimi_delta_net().
//
// The oracle evaluates the recurrence in FP64 from the represented BF16 inputs and the represented BF16
// initial state, indexing every tensor the way the *contract* says it is laid out. It shares no
// code with the kernel and none with the gated delta net's reference: the two differ in exactly
// the place this op exists for, and a reference that reached for the neighbouring one would
// agree with the wrong recurrence.
//
// Uniform-gate cases also run both ops against one independent FP64 reference. Factoring a
// scalar decay out of a reduction changes FP32 accumulation order; final BF16 values can land
// on opposite sides of a rounding midpoint, so cross-op agreement accounts for that spacing.
#include "api/ops/gated_delta_net.h"
#include "api/ops/kimi_delta_net.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr int kStateDim = 128;

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

int verify_recurrence(const std::string& label, const std::vector<double>& got,
                      const std::vector<double>& expected, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), got, expected, criterion);
}

// The output is BF16 and the state is stored BF16, so both carry that storage quantisation and
// are held to the floor the gated delta net's own outputs are.
constexpr ReductionCriterion output_criterion() {
    return {/*relative_l2=*/4.1e-3, /*gross_absolute=*/5.0e-6,
            /*gross_relative_to_max_reference=*/5.5e-3};
}

constexpr ReductionCriterion state_criterion() {
    return {/*relative_l2=*/4.1e-3, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/6.0e-3};
}

struct Case {
    const char* name;
    int qk_heads;
    int value_heads;
    int tokens;
    bool normalize_qk;
    /// Every channel of a head shares one gate value, which makes this op's recurrence the
    /// gated delta net's exactly.
    bool uniform_gate = false;
};

struct Inputs {
    int qk_heads;
    int value_heads;
    int tokens;
    std::vector<float> q, k, v, g, beta, state;
};

Inputs make_inputs(const Case& item, std::uint32_t seed) {
    Inputs in{item.qk_heads, item.value_heads, item.tokens, {}, {}, {}, {}, {}, {}};
    const std::size_t qk = static_cast<std::size_t>(kStateDim) * item.qk_heads * item.tokens;
    const std::size_t vv = static_cast<std::size_t>(kStateDim) * item.value_heads * item.tokens;
    in.q.resize(qk);
    in.k.resize(qk);
    in.v.resize(vv);
    in.g.resize(vv);
    in.beta.resize(static_cast<std::size_t>(item.value_heads) * item.tokens);
    in.state.resize(static_cast<std::size_t>(kStateDim) * kStateDim * item.value_heads);
    fill_uniform(in.q, seed, -1.0f, 1.0f);
    fill_uniform(in.k, seed + 1, -1.0f, 1.0f);
    fill_uniform(in.v, seed + 2, -1.0f, 1.0f);
    fill_uniform(in.state, seed + 3, -0.5f, 0.5f);
    // The gate is a log-decay: strictly negative, so alpha is in (0,1) and the state contracts.
    fill_uniform(in.g, seed + 4, -0.75f, -0.02f);
    fill_uniform(in.beta, seed + 5, 0.05f, 0.95f);
    if (item.uniform_gate) {
        for (int t = 0; t < item.tokens; ++t) {
            for (int h = 0; h < item.value_heads; ++h) {
                const std::size_t base =
                    (static_cast<std::size_t>(t) * item.value_heads + h) * kStateDim;
                for (int c = 1; c < kStateDim; ++c) { in.g[base + c] = in.g[base]; }
            }
        }
    }
    if (item.uniform_gate && !item.normalize_qk) {
        // Keep long raw-Q/K recurrences stable without asking either op to normalize them.
        for (float& value : in.q) { value *= 0.125f; }
        for (float& value : in.k) { value *= 0.125f; }
    }
    // Activations and the initial state reach the op as BF16.
    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    round_to_bf16(in.state);
    return in;
}

struct Reference {
    std::vector<double> out;
    std::vector<double> final_state;
    std::vector<double> snapshots;
};

/// The documented recurrence, in FP64, with the gate applied per key channel.
Reference evaluate(const Inputs& in, double scale, bool normalize_qk,
                   bool record_snapshots = false, bool bf16_normalization = false) {
    const std::int64_t S = kStateDim, Hq = in.qk_heads, Hv = in.value_heads, T = in.tokens;
    std::vector<double> q(in.q.begin(), in.q.end()), k(in.k.begin(), in.k.end());
    if (normalize_qk) {
        for (std::int64_t t = 0; t < T; ++t) {
            for (std::int64_t h = 0; h < Hq; ++h) {
                const std::size_t base = static_cast<std::size_t>((t * Hq + h) * S);
                double qs = 0.0, ks = 0.0;
                for (std::int64_t d = 0; d < S; ++d) {
                    qs += q[base + d] * q[base + d];
                    ks += k[base + d] * k[base + d];
                }
                const double qi = 1.0 / std::sqrt(qs + 1.0e-6);
                const double ki = 1.0 / std::sqrt(ks + 1.0e-6);
                for (std::int64_t d = 0; d < S; ++d) {
                    q[base + d] *= qi;
                    k[base + d] *= ki;
                    if (bf16_normalization) {
                        q[base + d] = bf16_to_f32(f32_to_bf16(static_cast<float>(q[base + d])));
                        k[base + d] = bf16_to_f32(f32_to_bf16(static_cast<float>(k[base + d])));
                    }
                }
            }
        }
    }

    Reference out;
    out.out.assign(static_cast<std::size_t>(S * Hv * T), 0.0);
    out.final_state.assign(static_cast<std::size_t>(S * S * Hv), 0.0);
    if (record_snapshots) { out.snapshots.resize(static_cast<std::size_t>(S * S * Hv * T)); }
    std::vector<double> state(static_cast<std::size_t>(S * S));
    std::vector<double> delta(static_cast<std::size_t>(S));
    std::vector<double> alpha(static_cast<std::size_t>(S));

    for (std::int64_t h = 0; h < Hv; ++h) {
        const std::size_t sbase = static_cast<std::size_t>(h * S * S);
        for (std::int64_t i = 0; i < S * S; ++i) { state[i] = in.state[sbase + i]; }
        const std::int64_t qh = h / (Hv / Hq);
        for (std::int64_t t = 0; t < T; ++t) {
            const std::size_t qk_base = static_cast<std::size_t>((t * Hq + qh) * S);
            const std::size_t v_base  = static_cast<std::size_t>((t * Hv + h) * S);
            const double beta         = in.beta[static_cast<std::size_t>(t * Hv + h)];
            for (std::int64_t c = 0; c < S; ++c) {
                alpha[c] = std::exp(static_cast<double>(in.g[v_base + c]));
            }
            // The delta corrects the prediction the *decayed* state makes, and each channel
            // has decayed by its own amount -- so alpha sits inside the dot product.
            for (std::int64_t row = 0; row < S; ++row) {
                double dot = 0.0;
                for (std::int64_t c = 0; c < S; ++c) {
                    dot += state[static_cast<std::size_t>(row * S + c)] * alpha[c] * k[qk_base + c];
                }
                delta[row] = beta * (static_cast<double>(in.v[v_base + row]) - dot);
            }
            for (std::int64_t row = 0; row < S; ++row) {
                for (std::int64_t c = 0; c < S; ++c) {
                    const std::size_t at = static_cast<std::size_t>(row * S + c);
                    state[at] = alpha[c] * state[at] + delta[row] * k[qk_base + c];
                }
            }
            for (std::int64_t row = 0; row < S; ++row) {
                double dot = 0.0;
                for (std::int64_t c = 0; c < S; ++c) {
                    dot += state[static_cast<std::size_t>(row * S + c)] * q[qk_base + c];
                }
                out.out[v_base + row] = scale * dot;
            }
            if (record_snapshots) {
                std::copy(state.begin(), state.end(),
                          out.snapshots.begin() + static_cast<std::size_t>((t * Hv + h) * S * S));
            }
        }
        for (std::int64_t i = 0; i < S * S; ++i) { out.final_state[sbase + i] = state[i]; }
    }
    return out;
}

int run_case(const Case& item, std::uint32_t seed) {
    const Inputs in    = make_inputs(item, seed);
    const float scale  = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const Reference ref = evaluate(in, static_cast<double>(scale), item.normalize_qk);

    GuardedDeviceBuffer q(in.q.size() * 2), k(in.k.size() * 2), v(in.v.size() * 2);
    GuardedDeviceBuffer g(in.g.size() * 4), beta(in.beta.size() * 4);
    GuardedDeviceBuffer state(in.state.size() * 2), out(in.v.size() * 2);
    const auto qb = bf16_bits(in.q), kb = bf16_bits(in.k), vb = bf16_bits(in.v);
    const auto sb = bf16_bits(in.state);
    q.copy_from_host(qb.data(), q.bytes());
    k.copy_from_host(kb.data(), k.bytes());
    v.copy_from_host(vb.data(), v.bytes());
    g.copy_from_host(in.g.data(), g.bytes());
    beta.copy_from_host(in.beta.data(), beta.bytes());
    state.copy_from_host(sb.data(), state.bytes());

    Tensor qt(q.data(), DType::BF16, {kStateDim, item.qk_heads, item.tokens});
    Tensor kt(k.data(), DType::BF16, {kStateDim, item.qk_heads, item.tokens});
    Tensor vt(v.data(), DType::BF16, {kStateDim, item.value_heads, item.tokens});
    Tensor gt(g.data(), DType::FP32, {kStateDim, item.value_heads, item.tokens});
    Tensor bt(beta.data(), DType::FP32, {item.value_heads, item.tokens});
    Tensor st(state.data(), DType::BF16, {kStateDim, kStateDim, item.value_heads});
    Tensor ot(out.data(), DType::BF16, {kStateDim, item.value_heads, item.tokens});
    ops::kimi_delta_net(qt, kt, vt, gt, bt, scale, item.normalize_qk, st, ot, nullptr);
    cuda_synchronize();

    const std::string label = std::string("kimi_delta_net ") + item.name;
    int failures = verify_recurrence(label, from_device_bf16(out.data(), in.v.size()), ref.out,
                                     output_criterion());
    failures += verify_recurrence(label + " state",
                                  from_device_bf16(state.data(), in.state.size()),
                                  ref.final_state, state_criterion());
    failures += q.verify_guards((label + " q").c_str());
    failures += g.verify_guards((label + " g").c_str());
    failures += state.verify_guards((label + " state").c_str());
    failures += out.verify_guards((label + " out").c_str());
    return failures;
}

// One BF16 spacing per value, plus FP32 cancellation noise near zero. The relative-L2 bound
// remains 1e-3: allowing adjacent rounded values must not admit a systematic change in output.
int verify_bf16_agreement(const std::string& label, const std::vector<double>& a,
                          const std::vector<double>& b) {
    constexpr double fp32_floor = 1e-6;
    constexpr ReductionCriterion agreement{1e-3, 0.0, 1.0 / 128.0 + fp32_floor};
    int failures = verify_recurrence(label, a, b, agreement);
    double maximum = 0.0;
    for (double value : b) { maximum = std::max(maximum, std::abs(value)); }
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double magnitude = std::max(std::abs(a[i]), std::abs(b[i]));
        if (!std::isfinite(magnitude)) { continue; } // reduction check reports non-finite values
        const int exponent = magnitude == 0.0 ? -126 : std::max(-126, std::ilogb(magnitude));
        const double spacing = std::ldexp(1.0, exponent - 7);
        if (std::abs(a[i] - b[i]) > spacing + fp32_floor * maximum) {
            std::cerr << label << ": exceeds one BF16 spacing at " << i
                      << " actual=" << a[i] << " reference=" << b[i] << '\n';
            ++failures;
            break;
        }
    }
    return failures;
}

/// A uniform gate makes both recurrences mathematically identical, including the final state.
int run_against_gated_delta_net(const Case& item, std::uint32_t seed) {
    const Inputs in   = make_inputs(item, seed);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));

    // The gated delta net's gate is one value per head per token: this input's first channel.
    std::vector<float> scalar_g(static_cast<std::size_t>(item.value_heads) * item.tokens);
    for (int t = 0; t < item.tokens; ++t) {
        for (int h = 0; h < item.value_heads; ++h) {
            scalar_g[static_cast<std::size_t>(t) * item.value_heads + h] =
                in.g[(static_cast<std::size_t>(t) * item.value_heads + h) * kStateDim];
        }
    }

    const auto qb = bf16_bits(in.q), kb = bf16_bits(in.k), vb = bf16_bits(in.v);
    const auto sb = bf16_bits(in.state);
    GuardedDeviceBuffer q(qb.size() * 2), k(kb.size() * 2), v(vb.size() * 2);
    GuardedDeviceBuffer g(in.g.size() * 4), gs(scalar_g.size() * 4), beta(in.beta.size() * 4);
    GuardedDeviceBuffer s1(sb.size() * 2), s2(sb.size() * 2);
    GuardedDeviceBuffer o1(vb.size() * 2), o2(vb.size() * 2);
    q.copy_from_host(qb.data(), q.bytes());
    k.copy_from_host(kb.data(), k.bytes());
    v.copy_from_host(vb.data(), v.bytes());
    g.copy_from_host(in.g.data(), g.bytes());
    gs.copy_from_host(scalar_g.data(), gs.bytes());
    beta.copy_from_host(in.beta.data(), beta.bytes());
    s1.copy_from_host(sb.data(), s1.bytes());
    s2.copy_from_host(sb.data(), s2.bytes());

    Tensor qt(q.data(), DType::BF16, {kStateDim, item.qk_heads, item.tokens});
    Tensor kt(k.data(), DType::BF16, {kStateDim, item.qk_heads, item.tokens});
    Tensor vt(v.data(), DType::BF16, {kStateDim, item.value_heads, item.tokens});
    Tensor gt(g.data(), DType::FP32, {kStateDim, item.value_heads, item.tokens});
    Tensor gst(gs.data(), DType::FP32, {item.value_heads, item.tokens});
    Tensor bt(beta.data(), DType::FP32, {item.value_heads, item.tokens});
    Tensor st1(s1.data(), DType::BF16, {kStateDim, kStateDim, item.value_heads});
    Tensor st2(s2.data(), DType::BF16, {kStateDim, kStateDim, item.value_heads});
    Tensor ot1(o1.data(), DType::BF16, {kStateDim, item.value_heads, item.tokens});
    Tensor ot2(o2.data(), DType::BF16, {kStateDim, item.value_heads, item.tokens});

    ops::kimi_delta_net(qt, kt, vt, gt, bt, scale, item.normalize_qk, st1, ot1, nullptr);

    const std::size_t ws_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        item.qk_heads, item.value_heads, item.normalize_qk, item.tokens, item.tokens);
    GuardedDeviceBuffer ws(std::max<std::size_t>(ws_bytes, 256));
    WorkspaceArena arena(DeviceSpan{ws.data(), ws.bytes()});
    ops::gated_delta_net(qt, kt, vt, gst, bt, scale, item.normalize_qk, arena, st2, ot2, nullptr);
    cuda_synchronize();

    const auto a = from_device_bf16(o1.data(), vb.size());
    const auto b = from_device_bf16(o2.data(), vb.size());
    const auto as = from_device_bf16(s1.data(), sb.size());
    const auto bs = from_device_bf16(s2.data(), sb.size());
    // Account for the normalized BF16 staging shared by both implementations. The recurrence
    // itself stays FP64 and independent of either GPU reduction or gate implementation.
    const auto ref = evaluate(in, scale, item.normalize_qk, false, true);
    const std::string label = std::string("uniform gate ") + item.name;
    int failures = verify_recurrence(label + " KDA vs FP64", a, ref.out, output_criterion());
    failures += verify_recurrence(label + " GDN vs FP64", b, ref.out, output_criterion());
    failures += verify_recurrence(label + " KDA state vs FP64", as, ref.final_state, state_criterion());
    failures += verify_recurrence(label + " GDN state vs FP64", bs, ref.final_state, state_criterion());
    failures += verify_bf16_agreement(label + " KDA/GDN output", a, b);
    failures += verify_bf16_agreement(label + " KDA/GDN state", as, bs);
    failures += s1.verify_guards((label + " KDA state").c_str());
    failures += s2.verify_guards((label + " GDN state").c_str());
    failures += o1.verify_guards((label + " KDA output").c_str());
    failures += o2.verify_guards((label + " GDN output").c_str());
    return failures;
}

/// The decode round's shape: B independent lanes, each starting from its own state slot and
/// checkpointing the state after every valid column into its own reserved interval.
struct SnapshotCase {
    const char* name;
    int qk_heads;
    int value_heads;
    int width;
    int batch;
    /// Per-lane valid column counts, or empty for dense rows.
    std::vector<int> valid;
};

int run_snapshot_case(const SnapshotCase& item, std::uint32_t seed) {
    const int S = kStateDim, Hq = item.qk_heads, Hv = item.value_heads;
    const int W = item.width, B = item.batch;
    const float scale = 1.0f / std::sqrt(static_cast<float>(S));
    // Lane b reads slot b and writes W slots starting at B + b*W, so no lane's interval touches
    // another's initial slot and the intervals are disjoint.
    const int slots = B + B * W;

    const std::size_t qk = static_cast<std::size_t>(S) * Hq * W * B;
    const std::size_t vv = static_cast<std::size_t>(S) * Hv * W * B;
    std::vector<float> q(qk), k(qk), v(vv), g(vv);
    std::vector<float> beta(static_cast<std::size_t>(Hv) * W * B);
    std::vector<float> pool(static_cast<std::size_t>(S) * S * Hv * slots);
    fill_uniform(q, seed, -1.0f, 1.0f);
    fill_uniform(k, seed + 1, -1.0f, 1.0f);
    fill_uniform(v, seed + 2, -1.0f, 1.0f);
    fill_uniform(pool, seed + 3, -0.5f, 0.5f);
    fill_uniform(g, seed + 4, -0.75f, -0.02f);
    fill_uniform(beta, seed + 5, 0.05f, 0.95f);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);
    round_to_bf16(pool);

    std::vector<int> initial(B), base(B), valid(item.valid);
    for (int b = 0; b < B; ++b) {
        initial[static_cast<std::size_t>(b)] = b;
        base[static_cast<std::size_t>(b)]    = B + b * W;
    }

    // The oracle runs each lane through the single-sequence form it already qualifies, then
    // says where every checkpoint landed. The addressing is the contract's, not the kernel's.
    std::vector<double> expected_out(vv, 0.0);
    std::vector<double> expected_pool(pool.begin(), pool.end());
    for (int b = 0; b < B; ++b) {
        const int columns = valid.empty() ? W : valid[static_cast<std::size_t>(b)];
        Inputs lane{Hq, Hv, columns, {}, {}, {}, {}, {}, {}};
        lane.q.resize(static_cast<std::size_t>(S) * Hq * columns);
        lane.k.resize(lane.q.size());
        lane.v.resize(static_cast<std::size_t>(S) * Hv * columns);
        lane.g.resize(lane.v.size());
        lane.beta.resize(static_cast<std::size_t>(Hv) * columns);
        lane.state.resize(static_cast<std::size_t>(S) * S * Hv);
        for (int t = 0; t < columns; ++t) {
            const std::size_t qk_from = (static_cast<std::size_t>(b) * W + t) * S * Hq;
            const std::size_t v_from  = (static_cast<std::size_t>(b) * W + t) * S * Hv;
            std::copy_n(q.begin() + qk_from, static_cast<std::size_t>(S) * Hq,
                        lane.q.begin() + static_cast<std::size_t>(t) * S * Hq);
            std::copy_n(k.begin() + qk_from, static_cast<std::size_t>(S) * Hq,
                        lane.k.begin() + static_cast<std::size_t>(t) * S * Hq);
            std::copy_n(v.begin() + v_from, static_cast<std::size_t>(S) * Hv,
                        lane.v.begin() + static_cast<std::size_t>(t) * S * Hv);
            std::copy_n(g.begin() + v_from, static_cast<std::size_t>(S) * Hv,
                        lane.g.begin() + static_cast<std::size_t>(t) * S * Hv);
            std::copy_n(beta.begin() + (static_cast<std::size_t>(b) * W + t) * Hv,
                        static_cast<std::size_t>(Hv),
                        lane.beta.begin() + static_cast<std::size_t>(t) * Hv);
        }
        std::copy_n(pool.begin() + static_cast<std::size_t>(initial[static_cast<std::size_t>(b)]) *
                        S * S * Hv,
                    lane.state.size(), lane.state.begin());
        // Snapshots are BF16 writes; within this call the next column continues
        // from the unrounded register state, rather than reloading the snapshot.
        const auto ref = evaluate(lane, scale, true, true);
        std::copy(ref.out.begin(), ref.out.end(), expected_out.begin() + static_cast<std::size_t>(b) * W * S * Hv);
        for (int t = 0; t < columns; ++t) {
            const auto from = ref.snapshots.begin() + static_cast<std::size_t>(t) * S * S * Hv;
            const auto to = expected_pool.begin() +
                static_cast<std::size_t>(base[static_cast<std::size_t>(b)] + t) * S * S * Hv;
            std::copy_n(from, static_cast<std::size_t>(S) * S * Hv, to);
        }
        // An invalid tail is exact zero and leaves its reserved slots untouched.
        for (int t = columns; t < W; ++t) {
            for (std::size_t i = 0; i < static_cast<std::size_t>(S) * Hv; ++i) {
                expected_out[(static_cast<std::size_t>(b) * W + t) * S * Hv + i] = 0.0;
            }
        }
    }

    GuardedDeviceBuffer dq(q.size() * 2), dk(k.size() * 2), dv(v.size() * 2);
    GuardedDeviceBuffer dg(g.size() * 4), db(beta.size() * 4);
    GuardedDeviceBuffer dpool(pool.size() * 2), dout(v.size() * 2);
    GuardedDeviceBuffer dinitial(initial.size() * 4), dbase(base.size() * 4);
    GuardedDeviceBuffer dvalid(std::max<std::size_t>(valid.size(), 1) * 4);
    const auto qb = bf16_bits(q), kb = bf16_bits(k), vb = bf16_bits(v), pb = bf16_bits(pool);
    dq.copy_from_host(qb.data(), dq.bytes());
    dk.copy_from_host(kb.data(), dk.bytes());
    dv.copy_from_host(vb.data(), dv.bytes());
    dg.copy_from_host(g.data(), dg.bytes());
    db.copy_from_host(beta.data(), db.bytes());
    dpool.copy_from_host(pb.data(), dpool.bytes());
    dinitial.copy_from_host(initial.data(), dinitial.bytes());
    dbase.copy_from_host(base.data(), dbase.bytes());
    if (!valid.empty()) { dvalid.copy_from_host(valid.data(), valid.size() * 4); }

    Tensor qt(dq.data(), DType::BF16, {S, Hq, W, B});
    Tensor kt(dk.data(), DType::BF16, {S, Hq, W, B});
    Tensor vt(dv.data(), DType::BF16, {S, Hv, W, B});
    Tensor gt(dg.data(), DType::FP32, {S, Hv, W, B});
    Tensor bt(db.data(), DType::FP32, {Hv, W, B});
    Tensor pt(dpool.data(), DType::BF16, {S, S, Hv, slots});
    Tensor ot(dout.data(), DType::BF16, {S, Hv, W, B});
    Tensor it(dinitial.data(), DType::I32, {B});
    Tensor bs(dbase.data(), DType::I32, {B});
    Tensor vc = valid.empty() ? Tensor{} : Tensor(dvalid.data(), DType::I32, {B});
    ops::kimi_delta_net_snapshot(qt, kt, vt, gt, bt, scale, true, pt, vc, it, bs, ot, nullptr);
    cuda_synchronize();

    const std::string label = std::string("kimi_delta_net_snapshot ") + item.name;
    int failures = verify_recurrence(label, from_device_bf16(dout.data(), v.size()), expected_out,
                                     output_criterion());
    failures += verify_recurrence(label + " pool",
                                  from_device_bf16(dpool.data(), pool.size()), expected_pool,
                                  state_criterion());
    failures += dq.verify_guards((label + " q").c_str());
    failures += dg.verify_guards((label + " g").c_str());
    failures += dpool.verify_guards((label + " pool").c_str());
    failures += dout.verify_guards((label + " out").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    const Case cases[] = {
        {"decode T=1", 4, 4, 1, true},
        {"T=2", 4, 8, 2, true},
        {"grouped heads T=17", 2, 8, 17, true},
        {"unnormalised q/k", 4, 4, 5, false},
        {"prefill T=64", 8, 8, 64, true},
    };
    std::uint32_t seed = 7u;
    for (const Case& item : cases) { failures += run_case(item, seed += 17u); }
    // Retain the original failing seed/shape, plus decode, grouping and raw-Q/K coverage.
    failures += run_against_gated_delta_net({"original T=24", 4, 8, 24, true, true}, seed + 31u);
    const Case uniform_cases[] = {
        {"decode T=1", 4, 4, 1, true, true},
        {"grouped T=2", 2, 8, 2, true, true},
        {"recurrent T=63", 4, 8, 63, true, true},
        {"raw Q/K T=24", 2, 8, 24, false, true},
    };
    auto uniform_seed = seed;
    for (const auto& item : uniform_cases) {
        failures += run_against_gated_delta_net(item, uniform_seed += 13u);
    }

    // The decode round's shape. Every lane owns a state slot and a reserved interval, and the
    // pool is checked whole: a lane that wrote outside its interval would show up as a
    // difference in a slot the oracle left alone.
    const SnapshotCase snapshots[] = {
        {"decode 4 lanes", 4, 4, 1, 4, {}},
        {"8 lanes x 3 columns", 2, 8, 3, 8, {}},
        {"ragged lanes", 4, 8, 4, 4, {4, 1, 3, 2}},
        {"one lane, wide round", 8, 8, 12, 1, {}},
        {"grouped heads, ragged", 2, 8, 5, 3, {5, 2, 4}},
    };
    for (const SnapshotCase& item : snapshots) { failures += run_snapshot_case(item, seed += 23u); }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " kimi_delta_net correctness\n";
    return failures == 0 ? 0 : 1;
}
