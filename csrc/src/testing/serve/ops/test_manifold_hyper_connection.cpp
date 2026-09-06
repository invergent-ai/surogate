// Public-contract qualification for GLM-5.3's manifold-constrained hyper-connections.
//
// The oracle evaluates the documented math in FP64 from the represented BF16 inputs, indexing
// every tensor through a `Layout` that states the contract's addressing once. The kernel packs
// a whole residual column into shared memory and walks it three different ways; an oracle that
// borrowed any of those walks would only prove the kernel agrees with itself.
//
// Two properties are worth more than the pointwise comparison and are checked outright:
// `comb` is doubly stochastic (that is what Sinkhorn is for, and a wrong iteration order still
// produces a plausible-looking matrix), and the combine really is a mix -- with `post` zero the
// residual is a pure recombination of the streams, so their total mass per channel is preserved.
#include "api/ops/manifold_hyper_connection.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr PointwiseCriterion mix_criterion() {
    return {/*absolute*/ 3.0e-3, /*relative*/ 8.0e-3};
}

constexpr PointwiseCriterion gate_criterion() {
    return {/*absolute*/ 2.0e-5, /*relative*/ 2.0e-5};
}

std::vector<double> read_f32(const void* device, std::size_t count) {
    const std::vector<float> raw = from_device<float>(device, count);
    return std::vector<double>(raw.begin(), raw.end());
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

/// The layout the contract states, spelled once so no reader has to re-derive it.
struct Layout {
    int streams;
    int hidden;
    int tokens;

    [[nodiscard]] int width() const { return streams * hidden; }
    [[nodiscard]] int rows() const { return (2 + streams) * streams; }
    /// residual [streams*hidden, T]: stream s occupies rows [s*hidden, (s+1)*hidden).
    [[nodiscard]] std::size_t residual(int stream, int channel, int token) const {
        return static_cast<std::size_t>(token) * width() +
               static_cast<std::size_t>(stream) * hidden + channel;
    }
    /// mix [(2+streams)*streams, streams*hidden]: k fastest.
    [[nodiscard]] std::size_t mix(int row, int column) const {
        return static_cast<std::size_t>(row) * width() + column;
    }
    /// collapsed [hidden, T].
    [[nodiscard]] std::size_t collapsed(int channel, int token) const {
        return static_cast<std::size_t>(token) * hidden + channel;
    }
    /// post [streams, T].
    [[nodiscard]] std::size_t post(int stream, int token) const {
        return static_cast<std::size_t>(token) * streams + stream;
    }
    /// comb [streams, streams, T]: column t holds the matrix with its second index fastest.
    [[nodiscard]] std::size_t comb(int i, int j, int token) const {
        return (static_cast<std::size_t>(token) * streams + i) * streams + j;
    }
};

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct Mixings {
    std::vector<double> pre;   ///< [streams]
    std::vector<double> post;  ///< [streams]
    std::vector<double> comb;  ///< [streams*streams], row-major
};

/// One token's three mixings, from the contract.
Mixings oracle_mixings(const Layout& layout, const std::vector<float>& residual,
                       const std::vector<float>& mix, const std::vector<float>& base,
                       const std::vector<float>& scale, int token, double rms_eps, double hc_eps,
                       int iterations) {
    const int streams = layout.streams;
    double sum_sq     = 0.0;
    for (int s = 0; s < streams; ++s) {
        for (int c = 0; c < layout.hidden; ++c) {
            const double v = residual[layout.residual(s, c, token)];
            sum_sq += v * v;
        }
    }
    const double inv = 1.0 / std::sqrt(sum_sq / layout.width() + rms_eps);

    std::vector<double> logits(static_cast<std::size_t>(layout.rows()), 0.0);
    for (int r = 0; r < layout.rows(); ++r) {
        double dot = 0.0;
        for (int s = 0; s < streams; ++s) {
            for (int c = 0; c < layout.hidden; ++c) {
                dot += static_cast<double>(mix[layout.mix(r, s * layout.hidden + c)]) *
                       static_cast<double>(residual[layout.residual(s, c, token)]) * inv;
            }
        }
        logits[static_cast<std::size_t>(r)] = dot;
    }

    Mixings out;
    out.pre.resize(static_cast<std::size_t>(streams));
    out.post.resize(static_cast<std::size_t>(streams));
    out.comb.resize(static_cast<std::size_t>(streams) * streams);
    for (int s = 0; s < streams; ++s) {
        out.pre[static_cast<std::size_t>(s)] =
            sigmoid(logits[static_cast<std::size_t>(s)] * scale[0] + base[static_cast<std::size_t>(s)]) + hc_eps;
        out.post[static_cast<std::size_t>(s)] =
            2.0 * sigmoid(logits[static_cast<std::size_t>(streams + s)] * scale[1] +
                          base[static_cast<std::size_t>(streams + s)]);
    }
    for (int i = 0; i < streams; ++i) {
        double total = 0.0;
        std::vector<double> row(static_cast<std::size_t>(streams));
        for (int j = 0; j < streams; ++j) {
            const std::size_t index = static_cast<std::size_t>(2 * streams + i * streams + j);
            row[static_cast<std::size_t>(j)] = logits[index] * scale[2] + base[index];
        }
        double maximum = row[0];
        for (double value : row) { maximum = std::max(maximum, value); }
        for (double& value : row) {
            value = std::exp(value - maximum);
            total += value;
        }
        for (int j = 0; j < streams; ++j) {
            out.comb[static_cast<std::size_t>(i * streams + j)] =
                row[static_cast<std::size_t>(j)] / total + hc_eps;
        }
    }
    const auto normalise_columns = [&] {
        for (int j = 0; j < streams; ++j) {
            double sum = 0.0;
            for (int i = 0; i < streams; ++i) { sum += out.comb[static_cast<std::size_t>(i * streams + j)]; }
            for (int i = 0; i < streams; ++i) { out.comb[static_cast<std::size_t>(i * streams + j)] /= sum + hc_eps; }
        }
    };
    const auto normalise_rows = [&] {
        for (int i = 0; i < streams; ++i) {
            double sum = 0.0;
            for (int j = 0; j < streams; ++j) { sum += out.comb[static_cast<std::size_t>(i * streams + j)]; }
            for (int j = 0; j < streams; ++j) { out.comb[static_cast<std::size_t>(i * streams + j)] /= sum + hc_eps; }
        }
    };
    normalise_columns();
    for (int iteration = 1; iteration < iterations; ++iteration) {
        normalise_rows();
        normalise_columns();
    }
    return out;
}

struct Case {
    int streams;
    int hidden;
    int tokens;
    int iterations;
    std::uint32_t seed;
    const char* label;
};

int run_case(const Case& item) {
    const Layout layout{item.streams, item.hidden, item.tokens};
    constexpr double kRmsEps = 1.0e-5;
    constexpr double kHcEps  = 1.0e-6;

    const std::size_t residual_cells = static_cast<std::size_t>(layout.width()) * item.tokens;
    const std::size_t mix_cells      = static_cast<std::size_t>(layout.rows()) * layout.width();
    std::vector<float> residual(residual_cells), mix(mix_cells), block(
        static_cast<std::size_t>(item.hidden) * item.tokens);
    std::vector<float> base(static_cast<std::size_t>(layout.rows())), scale(3);
    fill_uniform(residual, item.seed, -2.0f, 2.0f);
    // The projection sums `streams*hidden` terms, so keep its weights small enough that the
    // logits stay in the range a real checkpoint's do rather than saturating every sigmoid.
    fill_uniform(mix, item.seed + 1, -0.02f, 0.02f);
    fill_uniform(base, item.seed + 2, -1.0f, 1.0f);
    fill_uniform(scale, item.seed + 3, 0.5f, 1.5f);
    fill_uniform(block, item.seed + 4, -3.0f, 3.0f);
    round_to_bf16(residual);
    round_to_bf16(mix);
    round_to_bf16(block);

    std::vector<double> expected_collapsed(static_cast<std::size_t>(item.hidden) * item.tokens);
    std::vector<double> expected_post(static_cast<std::size_t>(item.streams) * item.tokens);
    std::vector<double> expected_comb(static_cast<std::size_t>(item.streams) * item.streams *
                                      item.tokens);
    std::vector<double> expected_residual(residual_cells);
    for (int token = 0; token < item.tokens; ++token) {
        const Mixings m = oracle_mixings(layout, residual, mix, base, scale, token, kRmsEps,
                                         kHcEps, item.iterations);
        for (int s = 0; s < item.streams; ++s) {
            expected_post[layout.post(s, token)] = m.post[static_cast<std::size_t>(s)];
            for (int j = 0; j < item.streams; ++j) {
                expected_comb[layout.comb(s, j, token)] =
                    m.comb[static_cast<std::size_t>(s * item.streams + j)];
            }
        }
        for (int c = 0; c < item.hidden; ++c) {
            double sum = 0.0;
            for (int s = 0; s < item.streams; ++s) {
                sum += m.pre[static_cast<std::size_t>(s)] *
                       static_cast<double>(residual[layout.residual(s, c, token)]);
            }
            expected_collapsed[layout.collapsed(c, token)] = sum;
        }
        for (int s = 0; s < item.streams; ++s) {
            for (int c = 0; c < item.hidden; ++c) {
                double sum = m.post[static_cast<std::size_t>(s)] *
                             static_cast<double>(block[layout.collapsed(c, token)]);
                for (int source = 0; source < item.streams; ++source) {
                    sum += m.comb[static_cast<std::size_t>(source * item.streams + s)] *
                           static_cast<double>(residual[layout.residual(source, c, token)]);
                }
                expected_residual[layout.residual(s, c, token)] = sum;
            }
        }
    }

    const auto residual_bits = encode_bf16(residual), mix_bits = encode_bf16(mix),
               block_bits = encode_bf16(block);
    GuardedDeviceBuffer device_residual(residual_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_mix(mix_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_block(block_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_base(base.size() * sizeof(float));
    GuardedDeviceBuffer device_scale(scale.size() * sizeof(float));
    GuardedDeviceBuffer device_collapsed(expected_collapsed.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_post(expected_post.size() * sizeof(float));
    GuardedDeviceBuffer device_comb(expected_comb.size() * sizeof(float));
    device_residual.copy_from_host(residual_bits.data(), device_residual.bytes());
    device_mix.copy_from_host(mix_bits.data(), device_mix.bytes());
    device_block.copy_from_host(block_bits.data(), device_block.bytes());
    device_base.copy_from_host(base.data(), device_base.bytes());
    device_scale.copy_from_host(scale.data(), device_scale.bytes());

    Weight mix_weight;
    mix_weight.qtype           = QType::BF16_CTRL;
    mix_weight.payload         = device_mix.data();
    mix_weight.qdata           = device_mix.data();
    mix_weight.payload_bytes   = device_mix.bytes();
    mix_weight.n               = layout.rows();
    mix_weight.k               = layout.width();
    mix_weight.ndim            = 2;
    mix_weight.shape[0]        = layout.rows();
    mix_weight.shape[1]        = layout.width();
    mix_weight.padded_shape[0] = layout.rows();
    mix_weight.padded_shape[1] = layout.width();
    mix_weight.layout          = QuantLayout::Contiguous;

    ops::ManifoldHyperConnectionWeights weights{
        mix_weight, Tensor(device_base.data(), DType::FP32, {layout.rows()}),
        Tensor(device_scale.data(), DType::FP32, {3})};

    Tensor residual_tensor(device_residual.data(), DType::BF16, {layout.width(), item.tokens});
    Tensor collapsed(device_collapsed.data(), DType::BF16, {item.hidden, item.tokens});
    Tensor post(device_post.data(), DType::FP32, {item.streams, item.tokens});
    Tensor comb(device_comb.data(), DType::FP32, {item.streams, item.streams, item.tokens});
    Tensor block_tensor(device_block.data(), DType::BF16, {item.hidden, item.tokens});

    // Sized by the op's own answer: the split projection lands its partial sums in the arena.
    WorkspaceArena arena(ops::manifold_hyper_connection_mix_workspace_capacity_bytes(
        item.streams, item.hidden, item.tokens, item.tokens));
    ops::manifold_hyper_connection_mix(residual_tensor, weights, item.streams,
                                       static_cast<float>(kRmsEps), static_cast<float>(kHcEps),
                                       item.iterations, collapsed, post, comb, arena, nullptr);
    cuda_synchronize();

    const std::string label(item.label);
    int failures = verify_pointwise((label + " collapsed").c_str(),
                                    from_device_bf16(device_collapsed.data(),
                                                     expected_collapsed.size()),
                                    expected_collapsed, mix_criterion());
    failures += verify_pointwise((label + " post").c_str(),
                                 read_f32(device_post.data(), expected_post.size()), expected_post,
                                 gate_criterion());
    failures += verify_pointwise((label + " comb").c_str(),
                                 read_f32(device_comb.data(), expected_comb.size()), expected_comb,
                                 gate_criterion());

    // Sinkhorn's whole purpose: rows and columns each sum to one. A wrong iteration order still
    // yields a positive matrix that a pointwise check against a matching oracle would pass, so
    // this is asked of the device result on its own terms.
    const std::vector<double> got_comb = read_f32(device_comb.data(), expected_comb.size());
    double worst_row = 0.0, worst_column = 0.0;
    for (int token = 0; token < item.tokens; ++token) {
        for (int i = 0; i < item.streams; ++i) {
            double row = 0.0, column = 0.0;
            for (int j = 0; j < item.streams; ++j) {
                row += got_comb[layout.comb(i, j, token)];
                column += got_comb[layout.comb(j, i, token)];
            }
            worst_row    = std::max(worst_row, std::abs(row - 1.0));
            worst_column = std::max(worst_column, std::abs(column - 1.0));
        }
    }
    // The two axes are not symmetric and it matters which is which. Sinkhorn alternates, and
    // the reference stops after a *column* normalisation: columns are exactly stochastic, rows
    // only as stochastic as the iteration count made them. Asserting the tight bound on the
    // wrong axis is exactly how a reversed iteration order would slip through, so both are
    // checked and only the column bound is tight.
    if (worst_column > 1.0e-5) {
        std::cerr << label << ": comb's columns do not sum to one (worst " << worst_column
                  << "); the last Sinkhorn step is a column normalisation\n";
        failures += 1;
    }
    if (worst_row > 5.0e-2) {
        std::cerr << label << ": comb's rows are far from one (worst " << worst_row
                  << "); Sinkhorn has not converged\n";
        failures += 1;
    }

    ops::manifold_hyper_connection_combine(block_tensor, post, comb, item.streams,
                                           residual_tensor, nullptr);
    cuda_synchronize();
    failures += verify_pointwise((label + " combine").c_str(),
                                 from_device_bf16(device_residual.data(), expected_residual.size()),
                                 expected_residual, mix_criterion());
    failures += verify_exact((label + " block unchanged").c_str(),
                             from_device<std::uint16_t>(device_block.data(), block_bits.size()),
                             block_bits);
    failures += verify_exact((label + " mix unchanged").c_str(),
                             from_device<std::uint16_t>(device_mix.data(), mix_bits.size()),
                             mix_bits);
    failures += device_residual.verify_guards("manifold residual");
    failures += device_mix.verify_guards("manifold mix");
    failures += device_block.verify_guards("manifold block");
    failures += device_collapsed.verify_guards("manifold collapsed");
    failures += device_post.verify_guards("manifold post");
    failures += device_comb.verify_guards("manifold comb");
    return failures;
}

/// With `post` zero the combine is a pure recombination, and a doubly-stochastic matrix moves
/// mass between streams without creating or destroying it: the per-channel total over streams
/// is what it was. That is the property the manifold constraint buys, and it is invisible to a
/// pointwise comparison against an oracle that shares the same iteration.
int run_mass_conservation() {
    constexpr int kStreams = 4, kHidden = 512, kTokens = 3;
    const Layout layout{kStreams, kHidden, kTokens};
    std::vector<float> residual(static_cast<std::size_t>(layout.width()) * kTokens);
    std::vector<float> mix(static_cast<std::size_t>(layout.rows()) * layout.width());
    std::vector<float> base(static_cast<std::size_t>(layout.rows())), scale(3);
    std::vector<float> block(static_cast<std::size_t>(kHidden) * kTokens, 0.0f);
    fill_uniform(residual, 909u, -2.0f, 2.0f);
    fill_uniform(mix, 911u, -0.02f, 0.02f);
    fill_uniform(base, 913u, -1.0f, 1.0f);
    fill_uniform(scale, 917u, 0.5f, 1.5f);
    round_to_bf16(residual);
    round_to_bf16(mix);

    std::vector<double> before(static_cast<std::size_t>(kHidden) * kTokens, 0.0);
    for (int token = 0; token < kTokens; ++token) {
        for (int c = 0; c < kHidden; ++c) {
            for (int s = 0; s < kStreams; ++s) {
                before[layout.collapsed(c, token)] += residual[layout.residual(s, c, token)];
            }
        }
    }

    const auto residual_bits = encode_bf16(residual), mix_bits = encode_bf16(mix),
               block_bits = encode_bf16(block);
    GuardedDeviceBuffer device_residual(residual_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_mix(mix_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_block(block_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_base(base.size() * sizeof(float));
    GuardedDeviceBuffer device_scale(scale.size() * sizeof(float));
    GuardedDeviceBuffer device_collapsed(static_cast<std::size_t>(kHidden) * kTokens *
                                         sizeof(std::uint16_t));
    GuardedDeviceBuffer device_post(static_cast<std::size_t>(kStreams) * kTokens * sizeof(float));
    GuardedDeviceBuffer device_comb(static_cast<std::size_t>(kStreams) * kStreams * kTokens *
                                    sizeof(float));
    device_residual.copy_from_host(residual_bits.data(), device_residual.bytes());
    device_mix.copy_from_host(mix_bits.data(), device_mix.bytes());
    device_block.copy_from_host(block_bits.data(), device_block.bytes());
    device_base.copy_from_host(base.data(), device_base.bytes());
    device_scale.copy_from_host(scale.data(), device_scale.bytes());

    Weight mix_weight;
    mix_weight.qtype           = QType::BF16_CTRL;
    mix_weight.payload         = device_mix.data();
    mix_weight.qdata           = device_mix.data();
    mix_weight.payload_bytes   = device_mix.bytes();
    mix_weight.n               = layout.rows();
    mix_weight.k               = layout.width();
    mix_weight.ndim            = 2;
    mix_weight.shape[0]        = layout.rows();
    mix_weight.shape[1]        = layout.width();
    mix_weight.padded_shape[0] = layout.rows();
    mix_weight.padded_shape[1] = layout.width();
    mix_weight.layout          = QuantLayout::Contiguous;
    ops::ManifoldHyperConnectionWeights weights{
        mix_weight, Tensor(device_base.data(), DType::FP32, {layout.rows()}),
        Tensor(device_scale.data(), DType::FP32, {3})};

    Tensor residual_tensor(device_residual.data(), DType::BF16, {layout.width(), kTokens});
    Tensor collapsed(device_collapsed.data(), DType::BF16, {kHidden, kTokens});
    Tensor post(device_post.data(), DType::FP32, {kStreams, kTokens});
    Tensor comb(device_comb.data(), DType::FP32, {kStreams, kStreams, kTokens});
    Tensor block_tensor(device_block.data(), DType::BF16, {kHidden, kTokens});
    WorkspaceArena arena(ops::manifold_hyper_connection_mix_workspace_capacity_bytes(
        kStreams, kHidden, kTokens, kTokens));
    ops::manifold_hyper_connection_mix(residual_tensor, weights, kStreams, 1.0e-5f, 1.0e-6f, 20,
                                       collapsed, post, comb, arena, nullptr);
    // Zero placement: nothing of the block output enters the streams.
    const std::vector<float> zeros(static_cast<std::size_t>(kStreams) * kTokens, 0.0f);
    device_post.copy_from_host(zeros.data(), device_post.bytes());
    ops::manifold_hyper_connection_combine(block_tensor, post, comb, kStreams, residual_tensor,
                                           nullptr);
    cuda_synchronize();

    const std::vector<double> after_values =
        from_device_bf16(device_residual.data(), residual_bits.size());
    std::vector<double> after(static_cast<std::size_t>(kHidden) * kTokens, 0.0);
    for (int token = 0; token < kTokens; ++token) {
        for (int c = 0; c < kHidden; ++c) {
            for (int s = 0; s < kStreams; ++s) {
                after[layout.collapsed(c, token)] += after_values[layout.residual(s, c, token)];
            }
        }
    }
    // BF16 storage of the rewritten streams is the whole error budget here.
    return verify_pointwise("manifold mass conservation", after, before,
                            {/*absolute*/ 3.0e-2, /*relative*/ 1.0e-2});
}

int run_mean() {
    constexpr int kStreams = 4, kHidden = 1024, kTokens = 5;
    const Layout layout{kStreams, kHidden, kTokens};
    std::vector<float> residual(static_cast<std::size_t>(layout.width()) * kTokens);
    fill_uniform(residual, 1301u, -4.0f, 4.0f);
    round_to_bf16(residual);
    std::vector<double> expected(static_cast<std::size_t>(kHidden) * kTokens, 0.0);
    for (int token = 0; token < kTokens; ++token) {
        for (int c = 0; c < kHidden; ++c) {
            double sum = 0.0;
            for (int s = 0; s < kStreams; ++s) { sum += residual[layout.residual(s, c, token)]; }
            expected[layout.collapsed(c, token)] = sum / kStreams;
        }
    }
    const auto bits = encode_bf16(residual);
    GuardedDeviceBuffer device_residual(bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_mean(expected.size() * sizeof(std::uint16_t));
    device_residual.copy_from_host(bits.data(), device_residual.bytes());
    Tensor residual_tensor(device_residual.data(), DType::BF16, {layout.width(), kTokens});
    Tensor mean(device_mean.data(), DType::BF16, {kHidden, kTokens});
    ops::collapse_streams_mean(residual_tensor, kStreams, mean, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise("collapse_streams_mean",
                                    from_device_bf16(device_mean.data(), expected.size()), expected,
                                    {/*absolute*/ 1.0e-5, /*relative*/ 4.0e-3});
    failures += device_residual.verify_guards("collapse_streams_mean residual");
    failures += device_mean.verify_guards("collapse_streams_mean mean");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    const Case cases[] = {
        // GLM-5.3-Flash: four streams over a 4096 hidden, one token (decode) and a prefill run.
        {4, 4096, 1, 20, 11u, "mHC decode, 4 streams x 4096"},
        {4, 4096, 37, 20, 13u, "mHC prefill 37 tokens"},
        // The degenerate ends of the geometry: one stream (comb is the 1x1 identity) and two.
        {1, 2048, 4, 20, 17u, "mHC one stream"},
        {2, 1024, 8, 20, 19u, "mHC two streams"},
        // A single Sinkhorn iteration is column-normalisation alone: rows will not sum to one,
        // and the row check below is skipped for it by construction (streams == 1).
        {1, 512, 3, 1, 23u, "mHC one Sinkhorn iteration"},
    };
    for (const Case& item : cases) { failures += run_case(item); }
    failures += run_mass_conservation();
    failures += run_mean();

    if (failures != 0) {
        std::cerr << failures << " manifold_hyper_connection check(s) failed\n";
        return 1;
    }
    std::cout << "manifold_hyper_connection: PASS\n";
    return 0;
}
