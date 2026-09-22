// The tiled prefill GEMMs against the small-T slices, for every routed pair the GGUF artifacts
// of a 704-wide mixture actually hold.
//
// `llama-quantize` cannot K-quant a tensor whose reduction axis is not a multiple of 256, so a
// Gemma 4 GGUF holds `ffn_gate_up_exps` in a K-quant (or Q8_0) and `ffn_down_exps` in Q8_0,
// Q5_0 or Q5_1 -- 704 is not a whole superblock. Those pairs take the prefill family's wide
// kernels: the int8 tensor-core route where both sides have an int8 codec (Q4_K/Q5_K/Q6_K,
// Q8_0, Q5_0, Q5_1 and Q3_K on the gate/up side), the BF16-activation route otherwise. Before,
// all but the K-quant/Q8_0 pair took none and every prompt ran the decode kernels in 46-token
// slices.
//
// The oracle here is the engine's own other path, not an fp64 reference: a mixture-of-experts
// round is token-independent, so the same tokens pushed through in slices of at most 46 take
// the small-T kernels and must produce, token for token, what one wide call produces on the
// prefill kernels. That is the comparison that says the new route did not change the answer,
// and it is the one that would catch a mis-strided row, a dropped tail column, a scale (or a
// min) read from the wrong block or a fifth bit put on the wrong value.
//
// Every fixture gives each block its OWN random scale (and min, where the format has one),
// because a constant scale would make the codec's scale indexing unobservable.
//
// The token counts are chosen so nothing divides evenly. 47 is the first prefill width; 65 and
// 130 leave a column tail inside the 32-wide job; 768 is the wide-plan boundary and 801 is one
// past it with a 33-column tail in a 64-wide job. The K tail is inherent and always exercised:
// 704 values is eleven 64-wide tiles, which is two whole 256-value superblocks and three tiles
// of a third.
#include "api/ops/sparse_moe.h"

#include "ops/op_tester.h"
#include "ops/linear/ggml/ggml_blocks.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

namespace gg = sinfer::ops::detail::ggml;

/// Which registration this is: the BF16-activation prefill route under
/// `SUROGATE_SERVE_MOE_INT8=0`, the int8 tensor-core route otherwise. The plan reads the same
/// variable, and the route assertion in `run_mixture` checks the two agree.
bool bf16_route() {
    static const bool value = [] {
        const char* env = std::getenv("SUROGATE_SERVE_MOE_INT8");
        return env != nullptr && env[0] == '0';
    }();
    return value;
}

/// A cheap reproducible byte source. The weights are not a quantisation of anything -- both
/// paths read the same bytes and must agree on them -- so this is deliberately not a quantiser.
struct Rng {
    std::uint64_t state;
    explicit Rng(std::uint64_t seed) : state(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    std::uint32_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return static_cast<std::uint32_t>(state >> 32);
    }
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * (static_cast<float>(next() >> 8) * (1.0F / 16777216.0F));
    }
};

/// Each block's own factor on the nominal scale, spread over [0.35, 1.65). A constant scale
/// would make a codec's scale indexing unobservable: reading the scale of the wrong block would
/// still produce the right answer, which is precisely the bug these fixtures have to be able to
/// fail on.
float spread(Rng& rng) {
    return 0.35F + 1.30F * (static_cast<float>(rng.next() % 1024) / 1023.0F);
}

/// Q6_K superblocks: six-bit codes take whatever bytes come out of the generator, the
/// sub-scales stay inside +-8 and the super-scale is the block's own, so a decoded value is at
/// most `d * 8 * 32`.
std::vector<std::uint8_t> make_q6_k_blocks(std::int64_t rows, std::int32_t k, std::uint64_t seed,
                                           float d) {
    const std::int64_t blocks = rows * (k / gg::QK_K);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(blocks) * sizeof(gg::block_q6_K));
    auto* b = reinterpret_cast<gg::block_q6_K*>(out.data());
    Rng rng(seed);
    for (std::int64_t i = 0; i < blocks; ++i) {
        for (int j = 0; j < gg::QK_K / 2; ++j) { b[i].ql[j] = static_cast<std::uint8_t>(rng.next()); }
        for (int j = 0; j < gg::QK_K / 4; ++j) { b[i].qh[j] = static_cast<std::uint8_t>(rng.next()); }
        for (int j = 0; j < gg::QK_K / 16; ++j) {
            b[i].scales[j] = static_cast<std::int8_t>(static_cast<int>(rng.next() % 17) - 8);
        }
        b[i].d = __float2half_rn(d * spread(rng));
    }
    return out;
}

/// Q4_K superblocks: random nibbles, random six-bit sub-scales and mins (any twelve bytes are
/// a valid packing), and the block's own `d` and `dmin`. A value is at most `d * 63 * 15`.
std::vector<std::uint8_t> make_q4_k_blocks(std::int64_t rows, std::int32_t k, std::uint64_t seed,
                                           float d) {
    const std::int64_t blocks = rows * (k / gg::QK_K);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(blocks) * sizeof(gg::block_q4_K));
    auto* b = reinterpret_cast<gg::block_q4_K*>(out.data());
    Rng rng(seed);
    for (std::int64_t i = 0; i < blocks; ++i) {
        for (int j = 0; j < gg::K_SCALE_SIZE; ++j) { b[i].scales[j] = static_cast<std::uint8_t>(rng.next()); }
        for (int j = 0; j < gg::QK_K / 2; ++j) { b[i].qs[j] = static_cast<std::uint8_t>(rng.next()); }
        b[i].dm = __floats2half2_rn(d * spread(rng), d * spread(rng));
    }
    return out;
}

/// Q5_K superblocks: Q4_K plus the fifth-bit plane. A value is at most `d * 63 * 31`.
std::vector<std::uint8_t> make_q5_k_blocks(std::int64_t rows, std::int32_t k, std::uint64_t seed,
                                           float d) {
    const std::int64_t blocks = rows * (k / gg::QK_K);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(blocks) * sizeof(gg::block_q5_K));
    auto* b = reinterpret_cast<gg::block_q5_K*>(out.data());
    Rng rng(seed);
    for (std::int64_t i = 0; i < blocks; ++i) {
        for (int j = 0; j < gg::K_SCALE_SIZE; ++j) { b[i].scales[j] = static_cast<std::uint8_t>(rng.next()); }
        for (int j = 0; j < gg::QK_K / 8; ++j) { b[i].qh[j] = static_cast<std::uint8_t>(rng.next()); }
        for (int j = 0; j < gg::QK_K / 2; ++j) { b[i].qs[j] = static_cast<std::uint8_t>(rng.next()); }
        b[i].dm = __floats2half2_rn(d * spread(rng), d * spread(rng));
    }
    return out;
}

/// Q3_K superblocks: two-bit codes in `qs`, the inverted third bit in `hmask`, sixteen six-bit
/// scales packed over twelve bytes (any bytes are a valid packing; a scale decodes to -32..31),
/// and the block's own `d`. A value is at most `d * 32 * 4`.
std::vector<std::uint8_t> make_q3_k_blocks(std::int64_t rows, std::int32_t k, std::uint64_t seed,
                                           float d) {
    const std::int64_t blocks = rows * (k / gg::QK_K);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(blocks) * sizeof(gg::block_q3_K));
    auto* b = reinterpret_cast<gg::block_q3_K*>(out.data());
    Rng rng(seed);
    for (std::int64_t i = 0; i < blocks; ++i) {
        for (int j = 0; j < gg::QK_K / 8; ++j) { b[i].hmask[j] = static_cast<std::uint8_t>(rng.next()); }
        for (int j = 0; j < gg::QK_K / 4; ++j) { b[i].qs[j] = static_cast<std::uint8_t>(rng.next()); }
        for (int j = 0; j < 12; ++j) { b[i].scales[j] = static_cast<std::uint8_t>(rng.next()); }
        b[i].d = __float2half_rn(d * spread(rng));
    }
    return out;
}

/// Q8_0 blocks: signed int8 codes under one FP16 scale, which is the format verbatim.
std::vector<std::uint8_t> make_q8_0_blocks(std::int64_t rows, std::int32_t k, std::uint64_t seed,
                                           float d) {
    const std::int64_t blocks = rows * (k / gg::QK8_0);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(blocks) * sizeof(gg::block_q8_0));
    auto* b = reinterpret_cast<gg::block_q8_0*>(out.data());
    Rng rng(seed);
    for (std::int64_t i = 0; i < blocks; ++i) {
        for (int j = 0; j < gg::QK8_0; ++j) {
            b[i].qs[j] = static_cast<std::int8_t>(static_cast<int>(rng.next() % 255) - 127);
        }
        b[i].d = __float2half_rn(d * spread(rng));
    }
    return out;
}

/// Q5_0 blocks: sixteen nibble bytes, a four-byte fifth-bit plane, one scale each;
/// `w = d * (q - 16)`, so a value is at most `16 d`.
std::vector<std::uint8_t> make_q5_0_blocks(std::int64_t rows, std::int32_t k, std::uint64_t seed,
                                           float d) {
    const std::int64_t blocks = rows * (k / gg::QK5_0);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(blocks) * sizeof(gg::block_q5_0));
    auto* b = reinterpret_cast<gg::block_q5_0*>(out.data());
    Rng rng(seed);
    for (std::int64_t i = 0; i < blocks; ++i) {
        for (int j = 0; j < 4; ++j) { b[i].qh[j] = static_cast<std::uint8_t>(rng.next()); }
        for (int j = 0; j < gg::QK5_0 / 2; ++j) { b[i].qs[j] = static_cast<std::uint8_t>(rng.next()); }
        b[i].d = __float2half_rn(d * spread(rng));
    }
    return out;
}

/// Q5_1 blocks: as Q5_0 with `w = d * q + m`, q unsigned. Each block's min is its own, drawn
/// from `[m_lo, m_hi)`, so a min read from the wrong block (or with the wrong sign) is visible.
std::vector<std::uint8_t> make_q5_1_blocks(std::int64_t rows, std::int32_t k, std::uint64_t seed,
                                           float d, float m_lo, float m_hi) {
    const std::int64_t blocks = rows * (k / gg::QK5_1);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(blocks) * sizeof(gg::block_q5_1));
    auto* b = reinterpret_cast<gg::block_q5_1*>(out.data());
    Rng rng(seed);
    for (std::int64_t i = 0; i < blocks; ++i) {
        for (int j = 0; j < 4; ++j) { b[i].qh[j] = static_cast<std::uint8_t>(rng.next()); }
        for (int j = 0; j < gg::QK5_1 / 2; ++j) { b[i].qs[j] = static_cast<std::uint8_t>(rng.next()); }
        b[i].dm = __floats2half2_rn(d * spread(rng), rng.uniform(m_lo, m_hi));
    }
    return out;
}

/// Values per stored block: 32 for the plain formats, 256 for a K-quant superblock.
std::int32_t values_per_block(QType qtype) {
    return qtype == QType::Q8_0 || qtype == QType::Q5_0 || qtype == QType::Q5_1 ? 32 : 256;
}

/// A tensor of `rows` x `k` in `qtype`, with the nominal scale chosen so a decoded value is
/// about a quarter at most whatever the format, since the comparison is stated relative to the
/// round's largest output and every mixture should land in the same band.
std::vector<std::uint8_t> make_blocks(QType qtype, std::int64_t rows, std::int32_t k,
                                      std::uint64_t seed) {
    switch (qtype) {
    case QType::Q3_K: return make_q3_k_blocks(rows, k, seed, 0.0030F);   // 128 d
    case QType::Q4_K: return make_q4_k_blocks(rows, k, seed, 0.00040F);  // 945 d
    case QType::Q5_K: return make_q5_k_blocks(rows, k, seed, 0.00020F);  // 1953 d
    case QType::Q6_K: return make_q6_k_blocks(rows, k, seed, 0.0015F);   // 256 d
    case QType::Q8_0: return make_q8_0_blocks(rows, k, seed, 0.0020F);   // 127 d
    case QType::Q5_0: return make_q5_0_blocks(rows, k, seed, 0.016F);    // 16 d
    case QType::Q5_1: return make_q5_1_blocks(rows, k, seed, 0.016F, -0.35F, -0.15F); // 31 d + m
    default: throw std::invalid_argument("no fixture for this format");
    }
}

Weight ggml_blocks_weight(const void* device, std::size_t bytes, QType qtype, std::int32_t n,
                          std::int32_t k, std::int32_t block_values) {
    Weight w{};
    w.payload         = device;
    w.payload_bytes   = bytes;
    w.qtype           = qtype;
    w.group_size      = static_cast<std::uint32_t>(block_values);
    w.ndim            = 2;
    w.qdata           = device;
    w.n               = n;
    w.k               = k;
    w.group           = block_values;
    w.layout          = QuantLayout::GgmlBlocks;
    w.scale_dtype     = DType::FP16;
    w.shape[0]        = n;
    w.shape[1]        = k;
    w.padded_shape[0] = n;
    w.padded_shape[1] = k;
    return w;
}

Weight dense_bf16_weight(const void* device, std::int32_t rows, std::int32_t columns) {
    Weight w{};
    w.payload         = device;
    w.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * sizeof(std::uint16_t);
    w.qtype           = QType::BF16_CTRL;
    w.layout          = QuantLayout::Contiguous;
    w.ndim            = 2;
    w.qdata           = device;
    w.n               = rows;
    w.k               = columns;
    w.shape[0]        = rows;
    w.shape[1]        = columns;
    w.padded_shape[0] = rows;
    w.padded_shape[1] = columns;
    return w;
}

struct Mixture {
    const char* name;
    ops::SparseMoeGeometry geometry;
    QType gate_up;
    QType down;
    ops::GatedActivation activation;
    bool per_expert_scaled;
    /// The bound the two paths must agree within, as a fraction of the round's largest output
    /// magnitude. Gross rather than pointwise: an output channel is a sum over 704 products
    /// over eight experts, so a channel that lands near zero did so by cancellation and its
    /// own magnitude says nothing about how much the sum may move. Stated, not derived -- see
    /// the note where each mixture is registered.
    double relative_to_max;
};

/// One mixture's weights, held on the device for the whole run.
class Fixture {
public:
    explicit Fixture(const Mixture& mixture) : mixture_(mixture) {
        const auto& g              = mixture.geometry;
        const std::int32_t gate_n  = g.experts * 2 * g.intermediate;
        const std::int32_t down_n  = g.experts * g.hidden;

        const std::vector<std::uint8_t> gate_up_host = make_blocks(mixture.gate_up, gate_n, g.hidden, 0x51D3u);
        const std::vector<std::uint8_t> down_host =
            mixture.down == QType::Q6_K
                ? make_q6_k_blocks(down_n, g.intermediate, 0x9E37u, 0.00025F)
                : make_blocks(mixture.down, down_n, g.intermediate, 0x9E37u);
        gate_up_ = DeviceBuffer(gate_up_host.size());
        gate_up_.copy_from_host(gate_up_host.data(), gate_up_host.size());
        down_ = DeviceBuffer(down_host.size());
        down_.copy_from_host(down_host.data(), down_host.size());

        // A router that cannot route differently between two kernels. Row e picks out column e
        // of the router input and nothing else, so a token's logit for expert e is one product
        // and 2,815 exact zeros -- the same number whatever order a kernel sums it in. With
        // random router weights the two paths would agree to a rounding error, and a token
        // whose eighth and ninth experts sat within that error would be routed differently and
        // the comparison would be measuring the tie, not the GEMMs.
        std::vector<std::uint16_t> router(static_cast<std::size_t>(g.router_rows()) * g.hidden, 0);
        for (std::int32_t e = 0; e < g.router_rows(); ++e) {
            router[static_cast<std::size_t>(e) * g.hidden + e] = f32_to_bf16(1.0F);
        }
        router_ = to_device(router);

        if (mixture.per_expert_scaled) {
            std::vector<float> scale(static_cast<std::size_t>(g.experts));
            Rng rng(0x2C1Bu);
            for (auto& v : scale) { v = rng.uniform(0.6F, 1.4F); }
            per_expert_scale_ = to_device(scale);
        }
    }

    ops::SparseMoeWeights weights() const {
        const auto& g = mixture_.geometry;
        ops::SparseMoeWeights w{
            .router_shared_gate = dense_bf16_weight(router_.p, g.router_rows(), g.hidden),
            .router_bias        = nullptr,
            .routed_scale       = g.routed_scale,
            .shared_gated       = g.shared_gated,
            .swiglu_limit       = g.swiglu_limit,
            .activation         = mixture_.activation,
            .per_expert_scale   = mixture_.per_expert_scaled
                                      ? static_cast<const float*>(per_expert_scale_.p)
                                      : nullptr,
            .routed_gate_up     = ggml_blocks_weight(gate_up_.p, gate_up_.bytes, mixture_.gate_up,
                                                     g.experts * 2 * g.intermediate, g.hidden,
                                                     values_per_block(mixture_.gate_up)),
            .routed_down        = ggml_blocks_weight(down_.p, down_.bytes, mixture_.down,
                                                     g.experts * g.hidden, g.intermediate,
                                                     values_per_block(mixture_.down)),
            .shared_gate_up     = Weight{},
            .shared_down        = Weight{},
            .experts_per_token  = g.experts_per_token,
        };
        return w;
    }

private:
    Mixture mixture_;
    DeviceBuffer gate_up_;
    DeviceBuffer down_;
    DeviceBuffer router_;
    DeviceBuffer per_expert_scale_;
};

/// Runs the op over `tokens` columns in chunks of `chunk`, into a destination seeded with zero.
/// `chunk == tokens` is the one wide call; a chunk of at most 46 is what forces the small-T
/// slices, because that is the bound the wrapper itself uses.
std::vector<double> run_moe(const ops::SparseMoeWeights& weights,
                            const ops::SparseMoeGeometry& geometry, const DeviceBuffer& x,
                            const DeviceBuffer& router_x, std::int32_t tokens,
                            std::int32_t chunk) {
    const std::size_t values = static_cast<std::size_t>(geometry.hidden) * tokens;
    DeviceBuffer out(values * sizeof(std::uint16_t));
    out.fill(0);
    const std::size_t workspace_bytes = ops::sparse_moe_workspace_capacity_bytes(
        geometry, weights.routed_gate_up.qtype, weights.routed_down.qtype, 1, chunk);
    WorkspaceArena workspace(workspace_bytes);
    const Tensor x_all(x.p, DType::BF16, {geometry.hidden, tokens});
    const Tensor router_all(router_x.p, DType::BF16, {geometry.hidden, tokens});
    Tensor out_all(out.p, DType::BF16, {geometry.hidden, tokens});
    for (std::int32_t offset = 0; offset < tokens; offset += chunk) {
        const std::int32_t width = std::min(chunk, tokens - offset);
        const Tensor xs          = x_all.slice(1, offset, width);
        const Tensor rs          = router_all.slice(1, offset, width);
        Tensor os                = out_all.slice(1, offset, width);
        ops::sparse_moe(xs, rs, weights, ops::SparseMoeEpilogue::AddResidual, os, workspace,
                        nullptr, ops::SparseMoeRoundHook{});
    }
    cuda_synchronize();
    return from_device_bf16(out.p, values);
}

int compare(const std::string& label, const std::vector<double>& actual,
            const std::vector<double>& reference, double relative_to_max) {
    double worst_abs  = 0.0;
    double magnitude  = 0.0;
    double rms        = 0.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double diff = std::abs(actual[i] - reference[i]);
        magnitude         = std::max(magnitude, std::abs(reference[i]));
        rms += diff * diff;
        if (diff > worst_abs) {
            worst_abs = diff;
            worst     = i;
        }
    }
    rms                = std::sqrt(rms / static_cast<double>(actual.size()));
    const double ratio = magnitude > 0.0 ? worst_abs / magnitude : worst_abs;
    std::printf("  %-34s max|d| %.3e (%.3f%% of |ref|max %.4f), rms|d| %.2e, ref %.6f got %.6f\n",
                label.c_str(), worst_abs, 100.0 * ratio, magnitude, rms, reference[worst],
                actual[worst]);
    // Two different kernel families never agree bitwise here (the prefill kernels round the
    // gated product through BF16 between the GEMMs; the small-T kernels keep it in
    // registers). An exact match means the wide call fell to the same slices as the
    // reference, and a comparison of a path with itself proves nothing.
    if (rms == 0.0) {
        std::cerr << label << ": the two paths agree bitwise, so the wide call did not take the "
                     "prefill route and the comparison is vacuous\n";
        return 1;
    }
    if (!(ratio <= relative_to_max)) {
        std::cerr << label << ": prefill and small-T disagree beyond the stated tolerance ("
                  << 100.0 * ratio << "% of the largest output, bound "
                  << 100.0 * relative_to_max << "%)\n";
        return 1;
    }
    return 0;
}

int run_mixture(const Mixture& mixture, const std::vector<std::int32_t>& token_cases) {
    const auto& g = mixture.geometry;
    std::printf("%s (hidden %d, experts %d, top-%d, intermediate %d)\n", mixture.name, g.hidden,
                g.experts, g.experts_per_token, g.intermediate);
    Fixture fixture(mixture);
    const ops::SparseMoeWeights weights = fixture.weights();
    if (!(ops::sparse_moe_geometry(weights) == g)) {
        std::cerr << mixture.name << ": the weights do not describe the intended geometry\n";
        return 1;
    }
    // The comparison is worth something only if the wide call takes the route this
    // registration tests. A pair the plan refuses falls to the same slices as the reference
    // (caught again by the bitwise check in `compare`); a pair the plan sends to the other
    // route would pass this registration's bound without exercising the codec under test.
    if (!ops::detail::sparse_moe_uses_prefill(token_cases.front(), mixture.gate_up, mixture.down)) {
        std::cerr << mixture.name << ": the plan refuses this pair; the comparison would be vacuous\n";
        return 1;
    }
    if (ops::detail::sparse_moe_routed_int8_profile(mixture.gate_up, mixture.down) == bf16_route()) {
        std::cerr << mixture.name << ": the plan's route does not match this registration\n";
        return 1;
    }
    int failures = 0;
    for (const std::int32_t tokens : token_cases) {
        // The expert input is small so the gated product stays inside a few units; the router
        // input is one distinct value per expert per token and zero elsewhere, which fixes the
        // routing exactly (see the router weight above).
        std::vector<float> x(static_cast<std::size_t>(g.hidden) * tokens);
        std::vector<float> router_x(static_cast<std::size_t>(g.hidden) * tokens, 0.0F);
        Rng rng(0xA53Fu + static_cast<std::uint64_t>(tokens));
        for (auto& v : x) { v = rng.uniform(-0.06F, 0.06F); }
        for (std::int32_t t = 0; t < tokens; ++t) {
            for (std::int32_t e = 0; e < g.experts; ++e) {
                router_x[static_cast<std::size_t>(t) * g.hidden + e] = rng.uniform(-4.0F, 4.0F);
            }
        }
        const DeviceBuffer dx  = to_device_bf16(x);
        const DeviceBuffer drx = to_device_bf16(router_x);

        const std::vector<double> reference =
            run_moe(weights, g, dx, drx, tokens, ops::detail::kSparseMoeSmallTMax);
        const std::vector<double> prefill = run_moe(weights, g, dx, drx, tokens, tokens);
        failures += compare("T=" + std::to_string(tokens) + " prefill vs small-T", prefill,
                            reference, mixture.relative_to_max);
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    // Gemma 4 26B-A4B as its Q6_K artifact stores it: Q6_K gate/up beside a Q8_0 down, because
    // the expert width of 704 is not a multiple of 256 and the quantiser could not K-quant the
    // down tensor at all. This is the pair the change exists for.
    //
    // The tolerance is not a rounding bound. The prefill family runs this pair on the int8
    // tensor-core route, which quantises the expert activations to int8 per 32 values the way
    // llama.cpp's MMQ does, while the small-T kernels multiply the BF16 activations directly --
    // so the two differ by the activation quantisation, not only by summation order, and the
    // bound has to admit that. It is the same trade the Q4_K/Q5_K/Q6_K pairs already take, and
    // `SUROGATE_SERVE_MOE_INT8=0` takes the BF16-activation prefill kernels instead; the second
    // registered run of this binary does exactly that and holds the tighter bound below.
    // Measured on an RTX 5090 with the fixtures below, as a fraction of the round's largest
    // output. Worst case seen over every mixture below: 0.71 % on the BF16-activation route
    // (Q4_K + Q5_0, T = 47) and 1.24 % on the int8 one (Q3_K + Q5_1, T = 801); the control
    // pair -- which none of the codecs here touches -- sits at 0.41 % and 1.20 %, so every
    // new codec lands inside the band the pre-existing ones occupy. These are the prefill
    // family's own distance from the small-T kernels (the gated product round-trips through
    // BF16 between the two GEMMs) plus, on the int8 route, the activation quantisation. The
    // bounds are the measured worst case with room, not a derived error bound; the kernels are
    // deterministic, so the room is against a future codec, not against noise.
    const double relative_to_max = bf16_route() ? 1.0e-2 : 1.5e-2;
    std::printf("route: %s (bound %.2f%% of the largest output)\n",
                bf16_route() ? "BF16 activations" : "int8 tensor core", 100.0 * relative_to_max);

    // Every routed pair the five published Gemma 4 artifacts hold, on the geometry they hold it
    // for. `llama-quantize` wrote (per artifact, per layer): Q3_K_M = Q3_K over Q5_0 x29 and
    // Q5_1 x1; Q4_K_M = Q4_K over Q8_0 x14 and Q5_0 x16; Q5_K_M = Q5_K over Q8_0 x14 and
    // Q5_1 x16; Q6_K = Q6_K over Q8_0 x30; Q8_0 = Q8_0 over Q8_0 x30.
    const auto gemma4 = [&](const char* name, QType gate_up, QType down) {
        return Mixture{name, ops::kSparseMoeGemma4Geometry, gate_up, down,
                       ops::GatedActivation::GeluTanh, /*per_expert_scaled=*/true, relative_to_max};
    };
    const std::vector<std::int32_t> all_cases{47, 65, 130, 768, 801};
    // Serving prefills prompts up to the model length in one round (the plan slices at 4,096),
    // so the widest cases run where the served rounds do: 2,048 and one past a 64-column
    // boundary at 3,009 (47 whole 64-wide jobs and a 1-column tail), through the same
    // persistent-block work loop the 801 case only starts.
    const std::vector<std::int32_t> wide_cases{47, 65, 130, 768, 801, 2048, 3009};
    int failures = 0;
    // The pair the Q8_0 codec was written for, and the two other K-quant/Q8_0 pairs.
    failures += run_mixture(gemma4("gemma4 Q6_K gate/up + Q8_0 down", QType::Q6_K, QType::Q8_0), all_cases);
    failures += run_mixture(gemma4("gemma4 Q4_K gate/up + Q8_0 down", QType::Q4_K, QType::Q8_0), {47, 130, 801});
    failures += run_mixture(gemma4("gemma4 Q5_K gate/up + Q8_0 down", QType::Q5_K, QType::Q8_0), {47, 130, 801});
    // The Q5_0 and Q5_1 down codecs (int8 route), beside the hand K-quant codecs.
    failures += run_mixture(gemma4("gemma4 Q4_K gate/up + Q5_0 down", QType::Q4_K, QType::Q5_0), all_cases);
    failures += run_mixture(gemma4("gemma4 Q5_K gate/up + Q5_1 down", QType::Q5_K, QType::Q5_1), all_cases);
    // Q8_0 as the gate/up side: the same codec over a 2,816-wide row, eleven superblocks' worth
    // of 64-wide tiles with the scales in the tile.
    failures += run_mixture(gemma4("gemma4 Q8_0 gate/up + Q8_0 down", QType::Q8_0, QType::Q8_0), wide_cases);
    // Q3_K gate/up: the generic BF16 codec until it has an int8 one, then that.
    failures += run_mixture(gemma4("gemma4 Q3_K gate/up + Q5_0 down", QType::Q3_K, QType::Q5_0), wide_cases);
    failures += run_mixture(gemma4("gemma4 Q3_K gate/up + Q5_1 down", QType::Q3_K, QType::Q5_1), {47, 130, 801});

    // A second mixture whose expert width *is* a whole superblock, so the only thing new about
    // it is the 32-value down codec. It separates "the down tensor is Q8_0" from "the reduction
    // is 704 long", which the Gemma 4 case tests together.
    const Mixture qwen3{"qwen3-moe Q6_K gate/up + Q8_0 down",
                        ops::kSparseMoeQwen3MoeGeometry,
                        QType::Q6_K,
                        QType::Q8_0,
                        ops::GatedActivation::Silu,
                        /*per_expert_scaled=*/false,
                        relative_to_max};

    // The control: the same mixture with a Q6_K down, which is a pair the prefill family
    // already served before any of this and which none of it touches. Whatever the two paths
    // disagree by here is the prefill design's own distance from the small-T kernels -- the
    // gated product round-trips through BF16 between the two GEMMs where the small-T kernels
    // keep it in registers -- and every case above has to be no worse.
    const Mixture control{"qwen3-moe Q6_K gate/up + Q6_K down (pre-existing route)",
                          ops::kSparseMoeQwen3MoeGeometry,
                          QType::Q6_K,
                          QType::Q6_K,
                          ops::GatedActivation::Silu,
                          /*per_expert_scaled=*/false,
                          relative_to_max};

    failures += run_mixture(qwen3, {47, 130, 801});
    failures += run_mixture(control, {47, 130, 801});

    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_moe ggml prefill\n";
    return failures == 0 ? 0 : 1;
}
