// Public-contract qualification for kimi_delta_net().
//
// The oracle evaluates the recurrence in FP64 from the represented BF16 inputs and the FP32
// initial state, indexing every tensor the way the *contract* says it is laid out. It shares no
// code with the kernel and none with the gated delta net's reference: the two differ in exactly
// the place this op exists for, and a reference that reached for the neighbouring one would
// agree with the wrong recurrence.
//
// The check that matters most is the last case. Give every key channel the same gate and this
// op must compute what the gated delta net computes -- a scalar decay is a diagonal one whose
// entries happen to be equal -- so the two are run against each other on identical inputs. If
// the per-channel alpha were folded in at the wrong point, that case is where it shows.
#include "api/ops/gated_delta_net.h"
#include "api/ops/kimi_delta_net.h"

#include "ops/op_tester.h"

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
    // q/k/v reach the op as BF16, so the oracle must see exactly what it will.
    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    return in;
}

struct Reference {
    std::vector<double> out;
    std::vector<double> final_state;
};

/// The documented recurrence, in FP64, with the gate applied per key channel.
Reference evaluate(const Inputs& in, double scale, bool normalize_qk) {
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
                }
            }
        }
    }

    Reference out;
    out.out.assign(static_cast<std::size_t>(S * Hv * T), 0.0);
    out.final_state.assign(static_cast<std::size_t>(S * S * Hv), 0.0);
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

/// A uniform gate makes this recurrence the gated delta net's, so the two must agree on the
/// same inputs -- the one comparison that pins where the per-channel alpha belongs.
int run_against_gated_delta_net(std::uint32_t seed) {
    const Case item{"uniform gate vs gated_delta_net", 4, 8, 24, true, true};
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

    ops::kimi_delta_net(qt, kt, vt, gt, bt, scale, true, st1, ot1, nullptr);

    const std::size_t ws_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        item.qk_heads, item.value_heads, true, item.tokens, item.tokens);
    GuardedDeviceBuffer ws(std::max<std::size_t>(ws_bytes, 256));
    WorkspaceArena arena(DeviceSpan{ws.data(), ws.bytes()});
    ops::gated_delta_net(qt, kt, vt, gst, bt, scale, true, arena, st2, ot2, nullptr);
    cuda_synchronize();

    const std::string label = "kimi_delta_net uniform gate matches gated_delta_net";
    // Two kernels, the same mathematics: they need not be bit-identical, but they must agree
    // far inside the bound either is held to against its own oracle.
    constexpr ReductionCriterion kAgreement{/*relative_l2=*/1.0e-3, /*gross_absolute=*/1.0e-5,
                                            /*gross_relative_to_max_reference=*/1.0e-3};
    return verify_recurrence(label, from_device_bf16(o1.data(), vb.size()),
                             from_device_bf16(o2.data(), vb.size()), kAgreement);
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
    failures += run_against_gated_delta_net(seed + 31u);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " kimi_delta_net correctness\n";
    return failures == 0 ? 0 : 1;
}
