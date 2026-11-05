// The e4m3 vision attention (ops::vision_attention_fp8) against an FP64 oracle and beside the
// BF16 kernel, with Q, K, V the strided views a tower makes of its packed QKV projection.
//
//   sinfer_vision_attention_fp8_test                 correctness (one and several segments)
//   sinfer_vision_attention_fp8_test bench [heads]   both kernels' times at the tower's widths

#include "api/ops/vision_attention.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace sinfer;
using namespace sinfer::test;

struct Problem {
    int dim, heads, patches;
    std::vector<int> cu;
    std::vector<std::uint16_t> qkv; // [3 * heads * dim, patches], token-major
};

Problem make_problem(int dim, int heads, std::vector<int> cu, std::uint32_t seed) {
    Problem p{dim, heads, cu.back(), std::move(cu), {}};
    const int width = 3 * heads * dim;
    p.qkv.resize(static_cast<std::size_t>(width) * p.patches);
    std::uint32_t state = seed;
    for (int t = 0; t < p.patches; ++t) {
        for (int i = 0; i < width; ++i) {
            state = state * 1664525U + 1013904223U;
            float x = (static_cast<float>(state >> 8) / 16777216.0F - 0.5F) * 4.0F;
            // An outlier channel per head, as trained towers have, so per-token scaling matters.
            if ((i % dim) == 5) { x *= 6.0F; }
            p.qkv[static_cast<std::size_t>(t) * width + i] = f32_to_bf16(x);
        }
    }
    return p;
}

double value(const Problem& p, int which, int head, int token, int d) {
    const int width = 3 * p.heads * p.dim;
    return bf16_to_f32(p.qkv[static_cast<std::size_t>(token) * width + which * p.heads * p.dim + head * p.dim + d]);
}

/// FP64 attention for (token, head): the segment's keys, softmax(q.k / sqrt(D)) v.
std::vector<double> oracle(const Problem& p, int token, int head) {
    int begin = 0, end = p.patches;
    for (std::size_t s = 0; s + 1 < p.cu.size(); ++s) {
        if (token >= p.cu[s] && token < p.cu[s + 1]) { begin = p.cu[s]; end = p.cu[s + 1]; }
    }
    std::vector<double> scores(static_cast<std::size_t>(end - begin));
    double peak = -1e300;
    for (int j = begin; j < end; ++j) {
        double dot = 0;
        for (int d = 0; d < p.dim; ++d) { dot += value(p, 0, head, token, d) * value(p, 1, head, j, d); }
        scores[j - begin] = dot / std::sqrt(static_cast<double>(p.dim));
        peak = std::max(peak, scores[j - begin]);
    }
    double sum = 0;
    for (double& s : scores) { s = std::exp(s - peak); sum += s; }
    std::vector<double> out(static_cast<std::size_t>(p.dim), 0.0);
    for (int j = begin; j < end; ++j) {
        for (int d = 0; d < p.dim; ++d) { out[d] += scores[j - begin] / sum * value(p, 2, head, j, d); }
    }
    return out;
}

/// e4m3, round to nearest even, saturating at 448 -- the device conversion's arithmetic.
double e4m3(double x) {
    const double a = std::fabs(x);
    if (a >= 448.0) { return std::copysign(448.0, x); }
    double step;
    if (a < std::ldexp(1.0, -6)) {
        step = std::ldexp(1.0, -9); // subnormals
    } else {
        int e = 0;
        (void)std::frexp(a, &e); // a = m 2^e, m in [0.5, 1)
        step = std::ldexp(1.0, e - 1 - 3);
    }
    return std::copysign(std::min(448.0, std::nearbyint(a / step) * step), x);
}

/// The kernel's own arithmetic in FP64: Q and K in e4m3 per (token, head) at amax / 448, V per
/// (head, channel) over all tokens, and the online softmax in 64-key blocks from the segment's
/// start with P * 448 rounded to e4m3 against the running maximum and l summed unrounded. The
/// kernel must track this closely whatever FP8's own error on the input is.
std::vector<double> oracle_fp8(const Problem& p, int token, int head) {
    int begin = 0, end = p.patches;
    for (std::size_t s = 0; s + 1 < p.cu.size(); ++s) {
        if (token >= p.cu[s] && token < p.cu[s + 1]) { begin = p.cu[s]; end = p.cu[s + 1]; }
    }
    const auto row = [&](int which, int t) {
        std::vector<double> r(static_cast<std::size_t>(p.dim));
        double amax = 0;
        for (int d = 0; d < p.dim; ++d) { r[d] = value(p, which, head, t, d); amax = std::max(amax, std::fabs(r[d])); }
        // The device's float arithmetic, so a code lands where the kernel's does.
        const float scale   = amax > 0 ? static_cast<float>(amax) * (1.0F / 448.0F) : 1.0F;
        const float inverse = 1.0F / scale;
        for (double& x : r) { x = e4m3(static_cast<float>(x) * inverse) * static_cast<double>(scale); }
        return r;
    };
    std::vector<double> v_scale(static_cast<std::size_t>(p.dim), 0.0);
    for (int t = 0; t < p.patches; ++t) {
        for (int d = 0; d < p.dim; ++d) { v_scale[d] = std::max(v_scale[d], std::fabs(value(p, 2, head, t, d))); }
    }
    std::vector<float> v_inverse(static_cast<std::size_t>(p.dim), 0.0F);
    for (int d = 0; d < p.dim; ++d) {
        v_inverse[d] = v_scale[d] > 0 ? 448.0F / static_cast<float>(v_scale[d]) : 0.0F;
        v_scale[d]   = static_cast<float>(v_scale[d]) * (1.0F / 448.0F);
    }
    const std::vector<double> q = row(0, token);
    const double inv_sqrt = 1.0 / std::sqrt(static_cast<double>(p.dim));
    std::vector<double> acc(static_cast<std::size_t>(p.dim), 0.0);
    double m = -1e300, l = 0;
    for (int b0 = begin; b0 < end; b0 += 64) {
        const int b1 = std::min(end, b0 + 64);
        std::vector<double> s;
        double block_max = -1e300;
        for (int j = b0; j < b1; ++j) {
            const std::vector<double> k = row(1, j);
            double dot = 0;
            for (int d = 0; d < p.dim; ++d) { dot += q[d] * k[d]; }
            s.push_back(dot);
            block_max = std::max(block_max, dot);
        }
        const double next  = std::max(m, block_max);
        const double alpha = m == -1e300 ? 0.0 : std::exp((m - next) * inv_sqrt);
        for (double& a : acc) { a *= alpha; }
        l *= alpha;
        for (int j = b0; j < b1; ++j) {
            const double pj = std::exp((s[j - b0] - next) * inv_sqrt);
            l += pj;
            const double p8 = e4m3(static_cast<float>(pj) * 448.0F) / 448.0;
            for (int d = 0; d < p.dim; ++d) {
                const double vd = v_scale[d] > 0
                                      ? e4m3(static_cast<float>(value(p, 2, head, j, d)) * v_inverse[d]) * v_scale[d]
                                      : 0.0;
                acc[d] += p8 * vd;
            }
        }
        m = next;
    }
    for (double& a : acc) { a /= l; }
    return acc;
}

struct Device {
    DeviceBuffer qkv, out_bf16, out_fp8, cu, scratch;
    Tensor q, k, v, o_bf16, o_fp8, cu_tensor;
    DeviceArena workspace;
    DeviceSpan scratch_span;
};

Device upload(const Problem& p) {
    const int plane = p.heads * p.dim;
    Device d{to_device(p.qkv), DeviceBuffer(static_cast<std::size_t>(plane) * p.patches * 2),
             DeviceBuffer(static_cast<std::size_t>(plane) * p.patches * 2), to_device_i32(p.cu),
             DeviceBuffer(std::max<std::size_t>(256, ops::vision_attention_fp8_scratch_bytes(p.dim, p.heads, p.patches))),
             {}, {}, {}, {}, {}, {},
             DeviceArena(std::max<std::size_t>(256, ops::vision_attention_workspace_capacity_bytes(
                                                        p.patches, p.patches, 1, static_cast<int>(p.cu.size()) - 1))),
             {}};
    const auto view = [&](int which) {
        Tensor t(static_cast<unsigned char*>(d.qkv.p) + static_cast<std::size_t>(which) * plane * 2, DType::BF16,
                 {p.dim, p.heads, p.patches});
        t.nb[2] = static_cast<std::int64_t>(3) * plane * 2;
        return t;
    };
    d.q = view(0);
    d.k = view(1);
    d.v = view(2);
    d.o_bf16    = Tensor(d.out_bf16.p, DType::BF16, {p.dim, p.heads, p.patches});
    d.o_fp8     = Tensor(d.out_fp8.p, DType::BF16, {p.dim, p.heads, p.patches});
    d.cu_tensor = Tensor(d.cu.p, DType::I32, {static_cast<std::int32_t>(p.cu.size())});
    d.scratch_span = DeviceSpan{d.scratch.p, d.scratch.bytes};
    return d;
}

int run_case(int dim, int heads, std::vector<int> cu, std::uint32_t seed) {
    const Problem p = make_problem(dim, heads, std::move(cu), seed);
    Device d        = upload(p);
    ops::vision_attention(d.q, d.k, d.v, d.cu_tensor, d.workspace, d.o_bf16, nullptr);
    if (!ops::vision_attention_fp8(d.q, d.k, d.v, d.cu_tensor, d.workspace, d.scratch_span, d.o_fp8, nullptr)) {
        std::cerr << "vision_attention_fp8 declined D=" << dim << '\n';
        return 1;
    }
    cuda_synchronize();
    const auto bf16 = from_device<std::uint16_t>(d.out_bf16.p, static_cast<std::size_t>(heads) * dim * p.patches);
    const auto fp8  = from_device<std::uint16_t>(d.out_fp8.p, static_cast<std::size_t>(heads) * dim * p.patches);
    double err_bf16 = 0, err_fp8 = 0, ref_norm = 0, err_emul = 0, emul_norm = 0;
    int non_finite  = 0;
    std::vector<int> tokens{0, 1, p.patches / 3, p.patches / 2, p.patches - 2, p.patches - 1};
    for (std::size_t s = 1; s + 1 < p.cu.size(); ++s) { tokens.push_back(p.cu[s]); tokens.push_back(p.cu[s] - 1); }
    for (const int t : tokens) {
        for (const int h : {0, heads / 2, heads - 1}) {
            const auto ref  = oracle(p, t, h);
            const auto emul = oracle_fp8(p, t, h);
            for (int i = 0; i < dim; ++i) {
                const std::size_t at = (static_cast<std::size_t>(t) * heads + h) * dim + i;
                const double a = bf16_to_f32(bf16[at]), b = bf16_to_f32(fp8[at]);
                if (!std::isfinite(b)) { ++non_finite; }
                err_bf16 += (a - ref[i]) * (a - ref[i]);
                err_fp8 += (b - ref[i]) * (b - ref[i]);
                ref_norm += ref[i] * ref[i];
                err_emul += (b - emul[i]) * (b - emul[i]);
                emul_norm += emul[i] * emul[i];
            }
        }
    }
    const double rel_bf16 = std::sqrt(err_bf16 / ref_norm), rel_fp8 = std::sqrt(err_fp8 / ref_norm);
    const double rel_emul = std::sqrt(err_emul / emul_norm);
    // Against its own arithmetic in FP64 the kernel differs by FP32 accumulation, exp2.approx
    // and the odd tie the device and host round apart: a misplaced fragment, key, or scale is
    // orders of magnitude past 5e-3. Against the exact result the gap is FP8's own rounding,
    // which this input (an outlier channel per head) makes coarse; the bound is a sanity check.
    const bool ok = non_finite == 0 && rel_bf16 < 4e-3 && rel_emul < 5e-3 && rel_fp8 < 0.1;
    std::cout << (ok ? "ok  " : "FAIL") << " D=" << dim << " H=" << heads << " P=" << p.patches
              << " S=" << p.cu.size() - 1 << "  rel L2: bf16 vs FP64 " << rel_bf16
              << " | fp8 vs its FP64 emulation " << rel_emul << " | fp8 vs FP64 " << rel_fp8
              << (non_finite ? "  NON-FINITE" : "") << '\n';
    return ok ? 0 : 1;
}

double time_us(const std::function<void()>& launch) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) { launch(); }
    std::vector<float> samples;
    for (int r = 0; r < 15; ++r) {
        cudaEventRecord(a);
        launch();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        samples.push_back(ms * 1000.0F);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(samples.begin(), samples.end());
    return samples.front(); // fastest: a co-tenant process time-slices the GPU
}

int bench(int heads) {
    std::cout << "vision attention, D=72 H=" << heads << " one segment (fastest of 15)\n";
    for (const int patches : {784, 1600, 3136, 4624, 7056, 8224, 12544, 16384}) {
        const Problem p = make_problem(72, heads, {0, patches}, 7U);
        Device d        = upload(p);
        const double bf16 = time_us([&] { ops::vision_attention(d.q, d.k, d.v, d.cu_tensor, d.workspace, d.o_bf16, nullptr); });
        const double fp8  = time_us([&] {
            (void)ops::vision_attention_fp8(d.q, d.k, d.v, d.cu_tensor, d.workspace, d.scratch_span, d.o_fp8, nullptr);
        });
        const double flops = 4.0 * patches * static_cast<double>(patches) * 72 * heads;
        std::printf("P=%-6d bf16 %9.1f us %6.1f TFLOP/s | fp8 (incl. prepare) %9.1f us %6.1f TFLOP/s | %.2fx\n",
                    patches, bf16, flops / bf16 / 1e6, fp8, flops / fp8 / 1e6, bf16 / fp8);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        {
            // The route serves compute capability 12.x only.
            const Problem probe = make_problem(72, 1, {0, 64}, 1U);
            Device d            = upload(probe);
            if (!ops::vision_attention_fp8(d.q, d.k, d.v, d.cu_tensor, d.workspace, d.scratch_span, d.o_fp8, nullptr)) {
                std::cout << "SKIP: the e4m3 vision attention needs compute capability 12.x\n";
                return 77;
            }
        }
        if (argc > 1 && std::strcmp(argv[1], "bench") == 0) { return bench(argc > 2 ? std::atoi(argv[2]) : 16); }
        int failures = 0;
        for (const int dim : {72, 64}) {
            failures += run_case(dim, 16, {0, 64}, 11U);
            failures += run_case(dim, 16, {0, 300}, 12U);
            failures += run_case(dim, 16, {0, 3136}, 13U);
            failures += run_case(dim, 4, {0, 100, 164, 900, 1296}, 14U); // unequal segments
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " vision attention fp8\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "vision attention fp8: " << error.what() << '\n';
        return 1;
    }
}
