// The encoder's host kernels, against double-precision references.
//
// No GPU and no artifact: these are pure arithmetic, so they are checked where
// they can be checked cheaply and deterministically. The properties that matter
// are the ones the GPU path already learned the hard way -- the symmetric window
// (not the causal half-window), the unit-offset norm, and a mean that does not
// lose precision over a long sequence.

#include "encoder/cpu/cpu_ops.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace sinfer::encoder;

std::vector<float> random_floats(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> distribution(lo, hi);
    std::vector<float> out(count);
    for (float& value : out) { value = distribution(engine); }
    return out;
}

int check(const std::string& label, const std::vector<float>& got, const std::vector<double>& want,
          double tolerance) {
    if (got.size() != want.size()) {
        std::cerr << label << ": size " << got.size() << " != " << want.size() << '\n';
        return 1;
    }
    double worst = 0.0;
    std::size_t worst_index = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) {
            std::cerr << label << ": non-finite at " << i << '\n';
            return 1;
        }
        const double error = std::abs(static_cast<double>(got[i]) - want[i]);
        const double scale = std::max(1.0, std::abs(want[i]));
        if (error / scale > worst) {
            worst       = error / scale;
            worst_index = i;
        }
    }
    if (worst > tolerance) {
        std::cerr << label << ": relative error " << worst << " at " << worst_index << " (got "
                  << got[worst_index] << ", want " << want[worst_index] << ")\n";
        return 1;
    }
    std::printf("  %-44s max relative error %.2e\n", label.c_str(), worst);
    return 0;
}

std::uint16_t to_bf16(float value) {
    std::uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    word += 0x7FFFU + ((word >> 16U) & 1U);
    return static_cast<std::uint16_t>(word >> 16U);
}

float from_bf16(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value              = 0.0F;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

int test_gemm(cpu::ThreadPool& pool) {
    // Gemma's own shapes, and a token count that is not a multiple of anything.
    // The weight is BF16 as stored; the reference reads the same rounded values,
    // so the tolerance stays about arithmetic rather than storage.
    const std::int32_t n = 1152, k = 768, tokens = 37;
    const std::vector<float> raw = random_floats(static_cast<std::size_t>(n) * k, 1, -0.3F, 0.3F);
    std::vector<std::uint16_t> w(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) { w[i] = to_bf16(raw[i]); }
    const std::vector<float> raw_x =
        random_floats(static_cast<std::size_t>(k) * tokens, 2, -2.0F, 2.0F);
    std::vector<std::uint16_t> x(raw_x.size());
    for (std::size_t i = 0; i < raw_x.size(); ++i) { x[i] = to_bf16(raw_x[i]); }
    std::vector<float> out(static_cast<std::size_t>(n) * tokens);
    cpu::gemm(w.data(), x.data(), out.data(), n, k, tokens, pool);

    std::vector<double> want(out.size());
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t row = 0; row < n; ++row) {
            double sum = 0.0;
            for (std::int32_t i = 0; i < k; ++i) {
                sum += static_cast<double>(from_bf16(w[static_cast<std::size_t>(row) * k + i])) *
                       static_cast<double>(from_bf16(x[static_cast<std::size_t>(t) * k + i]));
            }
            want[static_cast<std::size_t>(t) * n + row] = sum;
        }
    }
    return check("gemm bf16 x bf16 [1152,768] x 37", out, want, 2e-5);
}

int test_rmsnorm(cpu::ThreadPool& pool) {
    const std::int32_t rows = 768, tokens = 5;
    const std::vector<float> x = random_floats(static_cast<std::size_t>(rows) * tokens, 3, -3.F, 3.F);
    const std::vector<float> weight = random_floats(rows, 4, -0.5F, 0.5F);
    const float epsilon = 1e-6F;
    std::vector<float> out(x.size());
    cpu::rmsnorm(x.data(), weight.data(), epsilon, true, out.data(), rows, tokens, pool);

    std::vector<double> want(x.size());
    for (std::int32_t t = 0; t < tokens; ++t) {
        double squares = 0.0;
        for (std::int32_t i = 0; i < rows; ++i) {
            const double value = x[static_cast<std::size_t>(t) * rows + i];
            squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(squares / rows + epsilon);
        for (std::int32_t i = 0; i < rows; ++i) {
            want[static_cast<std::size_t>(t) * rows + i] =
                x[static_cast<std::size_t>(t) * rows + i] * inverse * (1.0 + weight[i]);
        }
    }
    return check("rmsnorm unit_offset [768] x 5", out, want, 1e-5);
}

/// The window is symmetric, and a window wider than the sequence is no window at
/// all. Both are properties the GPU op is tested for; a second implementation
/// that disagrees would be worse than none.
int test_attention(cpu::ThreadPool& pool) {
    const std::int32_t heads = 3, head_dim = 256, tokens = 40;
    const auto raw_q =
        random_floats(static_cast<std::size_t>(heads) * head_dim * tokens, 5, -1.F, 1.F);
    const auto raw_k = random_floats(static_cast<std::size_t>(head_dim) * tokens, 6, -1.F, 1.F);
    const auto raw_v = random_floats(static_cast<std::size_t>(head_dim) * tokens, 7, -1.F, 1.F);
    std::vector<std::uint16_t> q(raw_q.size()), k(raw_k.size()), v(raw_v.size());
    for (std::size_t i = 0; i < raw_q.size(); ++i) { q[i] = to_bf16(raw_q[i]); }
    for (std::size_t i = 0; i < raw_k.size(); ++i) { k[i] = to_bf16(raw_k[i]); }
    for (std::size_t i = 0; i < raw_v.size(); ++i) { v[i] = to_bf16(raw_v[i]); }
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scratch(cpu::attention_scratch(heads, tokens));

    int failures = 0;
    for (const std::int32_t window : {0, 8, 1000}) {
        std::vector<float> out(static_cast<std::size_t>(heads) * head_dim * tokens);
        cpu::attention(q.data(), k.data(), v.data(), out.data(), heads, head_dim, tokens, window,
                       scale, scratch.data(), pool);

        std::vector<double> want(out.size());
        const std::int32_t rows = heads * head_dim;
        for (std::int32_t head = 0; head < heads; ++head) {
            for (std::int32_t query = 0; query < tokens; ++query) {
                const std::int32_t lo = window > 0 ? std::max(0, query - window + 1) : 0;
                const std::int32_t hi = window > 0 ? std::min(tokens, query + window) : tokens;
                std::vector<double> weights(static_cast<std::size_t>(hi - lo));
                double maximum = -1e300;
                for (std::int32_t key = lo; key < hi; ++key) {
                    double dot = 0.0;
                    for (std::int32_t d = 0; d < head_dim; ++d) {
                        dot += static_cast<double>(from_bf16(
                                   q[static_cast<std::size_t>(query) * rows + head * head_dim +
                                     d])) *
                               static_cast<double>(
                                   from_bf16(k[static_cast<std::size_t>(key) * head_dim + d]));
                    }
                    weights[static_cast<std::size_t>(key - lo)] = dot * scale;
                    maximum = std::max(maximum, dot * scale);
                }
                double sum = 0.0;
                for (double& value : weights) {
                    value = std::exp(value - maximum);
                    sum += value;
                }
                for (std::int32_t d = 0; d < head_dim; ++d) {
                    double accumulated = 0.0;
                    for (std::int32_t key = lo; key < hi; ++key) {
                        accumulated += weights[static_cast<std::size_t>(key - lo)] *
                                       from_bf16(v[static_cast<std::size_t>(key) * head_dim + d]);
                    }
                    want[static_cast<std::size_t>(query) * rows + head * head_dim + d] =
                        accumulated / sum;
                }
            }
        }
        failures += check("attention window=" + std::to_string(window), out, want, 2e-5);
    }
    return failures;
}

/// A 2048-token mean is where FP32 accumulation would start to show; this
/// accumulates in double, so it should not.
int test_mean_pool() {
    const std::int32_t hidden = 768, tokens = 2048;
    const auto x = random_floats(static_cast<std::size_t>(hidden) * tokens, 8, -4.F, 4.F);
    std::vector<float> out(hidden);
    cpu::mean_pool(x.data(), out.data(), hidden, tokens);

    std::vector<double> want(hidden, 0.0);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t h = 0; h < hidden; ++h) {
            want[static_cast<std::size_t>(h)] += x[static_cast<std::size_t>(t) * hidden + h];
        }
    }
    for (double& value : want) { value /= tokens; }
    return check("mean_pool 2048 tokens", out, want, 1e-6);
}

int test_gelu_mul(cpu::ThreadPool& pool) {
    const std::size_t count = 4097; // odd, so any pair-wise tail is exercised
    const auto gate = random_floats(count, 9, -6.F, 6.F);
    const auto up   = random_floats(count, 10, -3.F, 3.F);
    std::vector<float> out(count);
    cpu::gelu_mul(gate.data(), up.data(), out.data(), static_cast<std::int64_t>(count), pool);

    std::vector<double> want(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double z = gate[i];
        want[i] = 0.5 * z * (1.0 + std::tanh(0.79788456080286535588 * (z + 0.044715 * z * z * z))) *
                  up[i];
    }
    return check("gelu_mul tanh, odd length", out, want, 1e-5);
}

int test_l2norm() {
    const std::int32_t rows = 768, columns = 4;
    auto x = random_floats(static_cast<std::size_t>(rows) * columns, 11, -2.F, 2.F);
    const std::vector<float> before = x;
    cpu::l2norm(x.data(), rows, columns, 1e-12F);

    std::vector<double> want(x.size());
    for (std::int32_t c = 0; c < columns; ++c) {
        double squares = 0.0;
        for (std::int32_t i = 0; i < rows; ++i) {
            const double value = before[static_cast<std::size_t>(c) * rows + i];
            squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(squares + 1e-12);
        for (std::int32_t i = 0; i < rows; ++i) {
            want[static_cast<std::size_t>(c) * rows + i] =
                before[static_cast<std::size_t>(c) * rows + i] * inverse;
        }
    }
    return check("l2norm 4 columns", x, want, 1e-6);
}

} // namespace

int main() {
    const cpu::ThreadPlan plan = cpu::ThreadPlan::detect();
    cpu::ThreadPool pool(plan);
    std::printf("cpu ops: %d threads on %zu pinned cpus\n", pool.threads(), plan.cpus.size());

    int failures = 0;
    failures += test_gemm(pool);
    failures += test_rmsnorm(pool);
    failures += test_attention(pool);
    failures += test_mean_pool();
    failures += test_gelu_mul(pool);
    failures += test_l2norm();

    if (failures != 0) {
        std::cerr << "cpu ops: " << failures << " case(s) failed\n";
        return 1;
    }
    std::printf("cpu ops: all cases passed\n");
    return 0;
}
