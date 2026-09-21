// The tiled prefill GEMMs against the small-T slices, for a routed pair whose down tensor is a
// plain 32-value block format.
//
// `llama-quantize` cannot K-quant a tensor whose reduction axis is not a multiple of 256, so a
// Gemma 4 GGUF holds `ffn_gate_up_exps` in Q6_K and `ffn_down_exps` in Q8_0 -- 704 is not a
// whole superblock. That pair now takes the prefill family's int8 tensor-core route; before it
// took none and every prompt ran the decode kernels in 46-token slices.
//
// The oracle here is the engine's own other path, not an fp64 reference: a mixture-of-experts
// round is token-independent, so the same tokens pushed through in slices of at most 46 take
// the small-T kernels and must produce, token for token, what one wide call produces on the
// prefill kernels. That is the comparison that says the new route did not change the answer,
// and it is the one that would catch a mis-strided row, a dropped tail column or a scale read
// from the wrong block.
//
// The token counts are chosen so nothing divides evenly. 47 is the first prefill width; 65 and
// 130 leave a column tail inside the 32-wide job; 768 is the wide-plan boundary and 801 is one
// past it with a 33-column tail in a 64-wide job. The K tail is inherent and always exercised:
// 704 values is eleven 64-wide tiles, which is two whole 256-value superblocks and three tiles
// of a third.
#include "api/ops/sparse_moe.h"

#include "ops/op_tester.h"
#include "ops/linear/ggml/ggml_blocks.h"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

namespace gg = sinfer::ops::detail::ggml;

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

/// Q6_K superblocks with a modest dynamic range: six-bit codes take whatever bytes come out of
/// the generator, the sub-scales stay inside +-8 and the super-scale is fixed, so a decoded
/// value is at most `d * 8 * 32`.
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
        b[i].d = __float2half_rn(d);
    }
    return out;
}

/// Q8_0 blocks: signed int8 codes under one FP16 scale, which is the format verbatim.
/// Each block gets its OWN scale, spread around `d`. A constant scale would make the codec's
/// scale indexing unobservable: reading the scale of the wrong block would still produce the
/// right answer, which is precisely the bug this fixture has to be able to fail on.
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
        b[i].d = __float2half_rn(d * (0.35f + 1.30f * (static_cast<float>(rng.next() % 1024) / 1023.0f)));
    }
    return out;
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

        const std::vector<std::uint8_t> gate_up_host =
            mixture.gate_up == QType::Q6_K
                ? make_q6_k_blocks(gate_n, g.hidden, 0x51D3u, 0.0015F)
                : make_q8_0_blocks(gate_n, g.hidden, 0x51D3u, 0.0015F);
        const std::vector<std::uint8_t> down_host =
            mixture.down == QType::Q6_K
                ? make_q6_k_blocks(down_n, g.intermediate, 0x9E37u, 0.00025F)
                : make_q8_0_blocks(down_n, g.intermediate, 0x9E37u, 0.002F);
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
                                                     mixture_.gate_up == QType::Q8_0 ? 32 : 256),
            .routed_down        = ggml_blocks_weight(down_.p, down_.bytes, mixture_.down,
                                                     g.experts * g.hidden, g.intermediate,
                                                     mixture_.down == QType::Q8_0 ? 32 : 256),
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
    const bool bf16_route = [] {
        const char* env = std::getenv("SUROGATE_SERVE_MOE_INT8");
        return env != nullptr && env[0] == '0';
    }();
    // Measured on an RTX 5090 with the fixtures below, as a fraction of the round's largest
    // output. Worst case seen: 0.67 % on the BF16-activation route and 1.08 % on the int8 one,
    // and the control pair -- which this change does not touch -- sits at 0.54 % and 1.08 %.
    // So these are the prefill family's own distance from the small-T kernels (the gated
    // product round-trips through BF16 between the two GEMMs) plus, on the int8 route, the
    // activation quantisation; the new codec adds nothing measurable to either. The bounds are
    // the measured worst case with room, not a derived error bound.
    const double relative_to_max = bf16_route ? 1.0e-2 : 1.5e-2;
    std::printf("route: %s (bound %.2f%% of the largest output)\n",
                bf16_route ? "BF16 activations" : "int8 tensor core", 100.0 * relative_to_max);

    const Mixture gemma4{"gemma4 Q6_K gate/up + Q8_0 down",
                         ops::kSparseMoeGemma4Geometry,
                         QType::Q6_K,
                         QType::Q8_0,
                         ops::GatedActivation::GeluTanh,
                         /*per_expert_scaled=*/true,
                         relative_to_max};
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
    // already served before this change and which this change does not touch. Whatever the two
    // paths disagree by here is the prefill design's own distance from the small-T kernels --
    // the gated product round-trips through BF16 between the two GEMMs where the small-T
    // kernels keep it in registers -- and the Q8_0 cases above have to be no worse.
    const Mixture control{"qwen3-moe Q6_K gate/up + Q6_K down (pre-existing route)",
                          ops::kSparseMoeQwen3MoeGeometry,
                          QType::Q6_K,
                          QType::Q6_K,
                          ops::GatedActivation::Silu,
                          /*per_expert_scaled=*/false,
                          relative_to_max};

    int failures = run_mixture(gemma4, {47, 65, 130, 768, 801});
    failures += run_mixture(qwen3, {47, 130, 801});
    failures += run_mixture(control, {47, 130, 801});

    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_moe ggml prefill\n";
    return failures == 0 ? 0 : 1;
}
