// ops::qk_norm_rope against the two ops it fuses: for every geometry in its domain the fused
// launch must leave exactly the bits ops::rmsnorm on the queries and keys followed by ops::rope
// (or ops::rope_interleaved) leave, and outside it it must decline without writing anything.
#include "api/ops/qk_norm_rope.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/rope.h"
#include "ops/op_tester.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

struct Case {
    const char* label;
    int head_dim;
    int rotary_dim;
    int q_heads;
    int k_heads;
    int axes;
    float theta;
    bool unit_offset;
    int active_pairs = 0;          ///< 0: every pair
    std::array<int, 3> sections{}; ///< nonzero: rope_interleaved
    bool keys = true;
    bool expect_fused = true;
};

std::vector<std::uint16_t> random_bf16(std::size_t n, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<std::uint16_t> out(n);
    for (auto& v : out) { v = f32_to_bf16(u(rng)); }
    return out;
}

/// Exact comparison that says where and by how much, so a contraction difference reads as one.
int compare(const std::string& label, const std::vector<std::uint16_t>& got,
            const std::vector<std::uint16_t>& want, int head_dim) {
    std::size_t mismatches = 0, first = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i] && mismatches++ == 0) { first = i; }
    }
    if (mismatches == 0) { return 0; }
    std::cerr << label << ": " << mismatches << " of " << got.size() << " differ; first at channel "
              << first % head_dim << " of row " << first / head_dim << ": " << bf16_to_f32(got[first])
              << " vs " << bf16_to_f32(want[first]) << '\n';
    return 1;
}

int run(const Case& c, int tokens, int first_position, std::uint32_t seed) {
    const std::string label = std::string(c.label) + " T=" + std::to_string(tokens) + " pos=" +
                              std::to_string(first_position);
    std::mt19937 rng(seed);
    const std::size_t qn = static_cast<std::size_t>(c.head_dim) * c.q_heads * tokens;
    const std::size_t kn = static_cast<std::size_t>(c.head_dim) * c.k_heads * tokens;
    auto q_host = random_bf16(qn, rng, -4.0F, 4.0F);
    auto k_host = random_bf16(kn, rng, -4.0F, 4.0F);
    // A few massive activations, as real query/key heads carry.
    for (std::size_t i = 0; i < qn; i += 997) { q_host[i] = f32_to_bf16(bf16_to_f32(q_host[i]) * 600.0F); }
    for (std::size_t i = 0; i < kn; i += 389) { k_host[i] = f32_to_bf16(bf16_to_f32(k_host[i]) * 600.0F); }
    const float gain_lo = c.unit_offset ? -0.6F : 0.4F, gain_hi = c.unit_offset ? 0.6F : 1.6F;
    const auto qw_host = random_bf16(c.head_dim, rng, gain_lo, gain_hi);
    const auto kw_host = random_bf16(c.head_dim, rng, gain_lo, gain_hi);
    std::vector<int> pos_host(static_cast<std::size_t>(c.axes) * tokens);
    for (int axis = 0; axis < c.axes; ++axis) {
        for (int t = 0; t < tokens; ++t) { pos_host[static_cast<std::size_t>(axis) * tokens + t] = first_position + 31 * axis + t; }
    }

    DeviceBuffer q_dev = to_device(q_host), k_dev = to_device(k_host);
    DeviceBuffer qw_dev = to_device(qw_host), kw_dev = to_device(kw_host);
    DeviceBuffer pos_dev = to_device(pos_host);
    DeviceBuffer q_ref(qn * 2), k_ref(kn * 2), q_got(qn * 2), k_got(kn * 2);
    q_got.fill(0x5a), k_got.fill(0x5a);
    q_ref.fill(0x5a), k_ref.fill(0x5a);

    Tensor q(q_dev.p, DType::BF16, {c.head_dim, c.q_heads, tokens});
    Tensor k(k_dev.p, DType::BF16, {c.head_dim, c.k_heads, tokens});
    Tensor qw(qw_dev.p, DType::BF16, {c.head_dim});
    Tensor kw(kw_dev.p, DType::BF16, {c.head_dim});
    Tensor positions(pos_dev.p, DType::I32, {tokens, c.axes});
    Tensor qr(q_ref.p, DType::BF16, {c.head_dim, c.q_heads, tokens});
    Tensor kr(k_ref.p, DType::BF16, {c.head_dim, c.k_heads, tokens});
    Tensor qg(q_got.p, DType::BF16, {c.head_dim, c.q_heads, tokens});
    Tensor kg(k_got.p, DType::BF16, {c.head_dim, c.k_heads, tokens});
    const float eps          = 1.0e-6F;
    const int active_pairs   = c.active_pairs > 0 ? c.active_pairs : c.rotary_dim / 2;

    // The separate ops, as the layer runs them. A layer without keys still rotates the key plane
    // it was handed (here the reference's own normalised keys).
    ops::rmsnorm(q, qw, eps, c.unit_offset, qr, nullptr);
    ops::rmsnorm(k, kw, eps, c.unit_offset, kr, nullptr);
    if (c.sections[0] != 0) {
        ops::rope_interleaved(positions, c.rotary_dim, c.theta, c.sections, qr, kr, nullptr);
    } else {
        ops::rope(positions, c.rotary_dim, active_pairs, c.theta, qr, kr, nullptr);
    }

    ops::QkNormRope args;
    args.q = &q, args.q_norm = &qw, args.q_out = &qg;
    if (c.keys) { args.k = &k, args.k_norm = &kw, args.k_out = &kg; }
    args.k_heads = c.k_heads, args.eps = eps, args.unit_offset = c.unit_offset;
    args.positions = &positions, args.rotary_dim = c.rotary_dim, args.active_pairs = active_pairs;
    args.theta = c.theta, args.sections = c.sections;
    const bool fused = ops::qk_norm_rope(args, nullptr);
    cuda_synchronize();

    if (fused != c.expect_fused) {
        std::cerr << label << ": fused " << fused << ", expected " << c.expect_fused << '\n';
        return 1;
    }
    const auto qg_host = from_device<std::uint16_t>(q_got, qn);
    const auto kg_host = from_device<std::uint16_t>(k_got, kn);
    if (!fused) {
        // Declined: nothing written.
        for (auto v : qg_host) { if (v != 0x5a5a) { std::cerr << label << ": declined but wrote queries\n"; return 1; } }
        for (auto v : kg_host) { if (v != 0x5a5a) { std::cerr << label << ": declined but wrote keys\n"; return 1; } }
        std::cout << "PASS " << label << " (declined)\n";
        return 0;
    }
    const auto qr_host = from_device<std::uint16_t>(q_ref, qn);
    const auto kr_host = from_device<std::uint16_t>(k_ref, kn);
    int failures = compare(label + " queries", qg_host, qr_host, c.head_dim);
    if (c.keys) {
        failures += compare(label + " keys", kg_host, kr_host, c.head_dim);
    } else {
        for (auto v : kg_host) {
            if (v != 0x5a5a) { std::cerr << label << ": wrote keys it was not given\n"; return failures + 1; }
        }
    }
    if (failures == 0) { std::cout << "PASS " << label << '\n'; }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const Case cases[] = {
        // Qwen3 / Llama-style heads: 128 wide, rotated in full (partner in the same lane).
        {"qwen3-8b d128/r128 32q8k theta 1e6", 128, 128, 32, 8, 1, 1.0e6F, false},
        // The DFlash shape, whose angles the rope kernels take from a fixed table.
        {"dflash d128/r128 32q8k theta 1e7", 128, 128, 32, 8, 1, 1.0e7F, false},
        {"d128/r128 offset gain", 128, 128, 16, 4, 1, 1.0e6F, true},
        // Qwen3.5/3.6 heads: 256 wide, 64 rotated (partner 16 lanes away), fixed and generic angles.
        {"qwen3.6-27b d256/r64 24q4k text1d", 256, 64, 24, 4, 1, 1.0e7F, true},
        {"qwen3.6-35b d256/r64 16q2k mrope", 256, 64, 16, 2, 3, 1.0e7F, true},
        {"qwen3.6 d256/r64 16q2k interleaved", 256, 64, 16, 2, 3, 1.0e7F, true, 0, {11, 11, 10}},
        {"d256/r64 8q2k generic angles", 256, 64, 8, 2, 1, 1.0e7F, true},
        // Gemma-style heads: 256 wide rotated in full (two registers apart), and partially.
        {"gemma d256/r256 8q4k", 256, 256, 8, 4, 1, 1.0e4F, true},
        {"d256/r256 64 active pairs", 256, 256, 8, 4, 1, 1.0e6F, true, 64},
        {"d256/r128 plain", 256, 128, 8, 2, 1, 1.0e6F, false},
        // Narrow heads and partial rotations.
        {"d64/r64 12q4k", 64, 64, 12, 4, 1, 1.0e4F, false},
        {"d128/r64 partial", 128, 64, 16, 8, 1, 1.0e4F, false},
        {"d128/r32 partial", 128, 32, 16, 8, 1, 1.0e4F, false},
        // Queries alone: a layer that reads an earlier layer's keys.
        {"queries only d128", 128, 128, 32, 8, 1, 1.0e6F, false, 0, {}, false},
        {"queries only d256/r64", 256, 64, 24, 4, 1, 1.0e7F, true, 0, {}, false},
        // Outside the domain: declined.
        {"d128/r96 declined", 128, 96, 8, 2, 1, 1.0e4F, false, 0, {}, true, false},
        {"d512/r512 declined", 512, 512, 4, 1, 1, 1.0e6F, true, 64, {}, true, false},
    };
    int failures = 0;
    std::uint32_t seed = 11;
    for (const Case& c : cases) {
        for (const int tokens : {1, 3, 64, 333}) {
            for (const int first : {0, 9000, 250000}) { failures += run(c, tokens, first, seed++); }
        }
    }
    if (failures != 0) {
        std::cerr << failures << " qk_norm_rope check(s) failed\n";
        return 1;
    }
    std::cout << "qk_norm_rope: all cases bit-identical to rmsnorm + rope\n";
    return 0;
}
