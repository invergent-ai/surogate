// K-quant routes against an independent reference. Fixtures (gen_fixture.py) hold each
// type's weight in native GGML blocks plus gguf-py's exact dequantisation. The reference
// quantises the activation exactly as the kernel does (per 32: d = amax/127 in fp32,
// q = round(x/d), d kept as fp16) and takes the fp64 dot of the dequantised weight with that
// quantised activation, so the only divergence left is fp32 accumulation order -- a tight
// bound, not a statistical one. Beyond the launchers, the public wrappers are driven too:
// ops::linear without a workspace (the engine-slot scratch the lm_head takes) and
// ops::embedding (the row gather).
#include "api/ops/embedding.h"
#include "api/ops/linear.h"
#include "ops/linear/ggml/ggml_dispatch.h"
#include "ops/linear/ggml/ggml_embedding.h"
#include "ops/linear/ggml/ggml_linear.h"
#include "ops/linear/ggml/ggml_moe.h"
#include "ops/linear/ggml/ggml_q8_1.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace {

namespace gg = sinfer::ops::detail::ggml;
using sinfer::DType;
using sinfer::QType;
using sinfer::QuantLayout;
using sinfer::Tensor;
using sinfer::Weight;

#define CHECK_CUDA(call)                                                                        \
    do {                                                                                        \
        const cudaError_t error_ = (call);                                                      \
        if (error_ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,                  \
                         cudaGetErrorString(error_));                                           \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

struct Fixture {
    gg::GgmlType type;
    std::string label;
    int n = 0, k = 0;
    std::vector<std::uint8_t> blocks;
    std::vector<float> dequant; // [n][k]
};

bool read_file(const std::string& path, std::vector<std::uint8_t>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { return false; }
    in.seekg(0, std::ios::end);
    out.resize(static_cast<std::size_t>(in.tellg()));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    return static_cast<bool>(in);
}

bool load_fixture(const std::string& dir, gg::GgmlType type, const std::string& label, Fixture& f) {
    const std::string stem = dir + "/" + gg::type_name(type) + "_" + label;
    std::ifstream meta(stem + ".meta");
    if (!(meta >> f.n >> f.k)) { return false; }
    std::vector<std::uint8_t> raw;
    if (!read_file(stem + ".blocks", f.blocks) || !read_file(stem + ".f32", raw)) { return false; }
    f.type  = type;
    f.label = label;
    f.dequant.resize(raw.size() / sizeof(float));
    std::memcpy(f.dequant.data(), raw.data(), raw.size());
    const std::size_t expect_blocks =
        static_cast<std::size_t>(f.n) * (f.k / gg::QK_K) * gg::block_bytes(type);
    if (f.blocks.size() != expect_blocks || f.dequant.size() != static_cast<std::size_t>(f.n) * f.k) {
        std::fprintf(stderr, "fixture %s: unexpected sizes\n", stem.c_str());
        return false;
    }
    return true;
}

QType qtype_of(gg::GgmlType type) {
    switch (type) {
    case gg::GgmlType::Q2_K: return QType::Q2_K;
    case gg::GgmlType::Q3_K: return QType::Q3_K;
    case gg::GgmlType::Q4_K: return QType::Q4_K;
    case gg::GgmlType::Q5_K: return QType::Q5_K;
    case gg::GgmlType::Q6_K: return QType::Q6_K;
    }
    return QType::Q4_K;
}

// The Weight the artifact loader builds for a GgmlBlocks object (typed_binding.cpp).
Weight make_weight(const Fixture& f, void* d_blocks) {
    Weight w{};
    w.payload         = d_blocks;
    w.payload_bytes   = f.blocks.size();
    w.qtype           = qtype_of(f.type);
    w.group_size      = 256;
    w.ndim            = 2;
    w.qdata           = d_blocks;
    w.n               = f.n;
    w.k               = f.k;
    w.group           = 256;
    w.layout          = QuantLayout::GgmlBlocks;
    w.scale_dtype     = DType::FP16;
    w.shape[0]        = f.n;
    w.shape[1]        = f.k;
    w.padded_shape[0] = f.n;
    w.padded_shape[1] = f.k;
    return w;
}

// A linear takes one of two routes and they do different arithmetic, so each is checked
// against the arithmetic it actually performs: below the threshold the activation is
// quantised to int8 per 32 (the GEMV route), above it the BF16 activation goes to the tensor
// cores untouched (the dequantise-once route, which is the more accurate of the two).
bool takes_wide_route(int rows, int k, int tokens) {
    return tokens >= gg::wide_min_tokens() && (rows % 8) == 0 && (k % 8) == 0;
}

std::vector<double> exact_activation(const std::vector<__nv_bfloat16>& x) {
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { y[i] = __bfloat162float(x[i]); }
    return y;
}

// The kernel's own activation quantisation, on the host, in the same arithmetic.
std::vector<double> quantised_activation(const std::vector<__nv_bfloat16>& x, int k, int tokens) {
    std::vector<double> y(static_cast<std::size_t>(k) * tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int b = 0; b < k / 32; ++b) {
            float amax = 0.0F;
            float vals[32];
            for (int i = 0; i < 32; ++i) {
                vals[i] = __bfloat162float(x[static_cast<std::size_t>(t) * k + b * 32 + i]);
                amax    = std::fmax(amax, std::fabs(vals[i]));
            }
            const float d      = amax / 127.0F;
            const float d_half = __half2float(__float2half(d));
            for (int i = 0; i < 32; ++i) {
                const int q = amax == 0.0F ? 0 : static_cast<int>(std::round(vals[i] / d));
                y[static_cast<std::size_t>(t) * k + b * 32 + i] = static_cast<double>(d_half) * q;
            }
        }
    }
    return y;
}

std::vector<__nv_bfloat16> random_activation(int k, int tokens, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<__nv_bfloat16> x(static_cast<std::size_t>(k) * tokens);
    for (auto& v : x) { v = __float2bfloat16(normal(rng)); }
    return x;
}

// ref[t][row] = base[t][row] + sum_k W[row][k] * yq[t][k]
std::vector<double> reference(const Fixture& f, const std::vector<double>& y, int tokens,
                              const std::vector<double>* base) {
    const int n = f.n, k = f.k;
    std::vector<double> ref(static_cast<std::size_t>(n) * tokens, 0.0);
    for (int t = 0; t < tokens; ++t) {
        const double* yt = &y[static_cast<std::size_t>(t) * k];
        for (int row = 0; row < n; ++row) {
            double acc = base != nullptr ? (*base)[static_cast<std::size_t>(t) * n + row] : 0.0;
            const float* w = &f.dequant[static_cast<std::size_t>(row) * k];
            for (int i = 0; i < k; ++i) { acc += static_cast<double>(w[i]) * yt[i]; }
            ref[static_cast<std::size_t>(t) * n + row] = acc;
        }
    }
    return ref;
}

struct Score {
    double rel_l2 = 0.0, max_abs = 0.0, max_ref = 0.0;
};

Score score(const std::vector<float>& got, const std::vector<double>& ref) {
    Score s;
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double diff = static_cast<double>(got[i]) - ref[i];
        num += diff * diff;
        den += ref[i] * ref[i];
        s.max_abs = std::fmax(s.max_abs, std::fabs(diff));
        s.max_ref = std::fmax(s.max_ref, std::fabs(ref[i]));
    }
    s.rel_l2 = std::sqrt(num / std::fmax(den, 1e-300));
    return s;
}

int report(const Fixture& f, const char* what, int tokens, const Score& s, double bound_l2,
           double bound_abs_of_max) {
    const bool ok = s.rel_l2 <= bound_l2 && s.max_abs <= bound_abs_of_max * s.max_ref;
    std::printf("  %-5s %-10s [%6d,%5d] %-22s T=%-2d rel_l2=%.2e max_abs=%.1e of max  %s\n",
                gg::type_name(f.type), f.label.c_str(), f.n, f.k, what, tokens, s.rel_l2,
                s.max_abs / std::fmax(s.max_ref, 1e-300), ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

// linear_launch_f32 / linear_launch (bf16) against the reference.
int run_case(const Fixture& f, int tokens, bool bf16_out, void* d_blocks, void* d_scratch,
             std::size_t scratch_bytes) {
    const int n = f.n, k = f.k;
    const auto x = random_activation(k, tokens, static_cast<unsigned>(1234 + tokens + 7 * static_cast<int>(f.type)));
    const bool wide = bf16_out && takes_wide_route(n, k, tokens);
    const std::vector<double> ref =
        reference(f, wide ? exact_activation(x) : quantised_activation(x, k, tokens), tokens, nullptr);
    __nv_bfloat16* d_x = nullptr;
    void* d_out        = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_out, static_cast<std::size_t>(n) * tokens * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    std::vector<float> got(static_cast<std::size_t>(n) * tokens);
    if (bf16_out) {
        gg::linear_launch(f.type, d_blocks, n, k, d_x, tokens, static_cast<__nv_bfloat16*>(d_out),
                          d_scratch, scratch_bytes, nullptr);
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> raw(got.size());
        CHECK_CUDA(cudaMemcpy(raw.data(), d_out, raw.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < got.size(); ++i) { got[i] = __bfloat162float(raw[i]); }
    } else {
        gg::linear_launch_f32(f.type, d_blocks, n, k, d_x, tokens, static_cast<float*>(d_out),
                              d_scratch, scratch_bytes, nullptr);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_out));
    return report(f, wide ? "launch bf16 (wide)" : (bf16_out ? "launch bf16" : "launch f32"), tokens,
                  score(got, ref), bf16_out ? 4e-3 : 2e-5, bf16_out ? 8e-3 : 2e-4);
}

// linear_add_launch: residual += W x, BF16 residual.
int run_accumulate(const Fixture& f, int tokens, void* d_blocks, void* d_scratch, std::size_t scratch_bytes) {
    const int n = f.n, k = f.k;
    const auto x = random_activation(k, tokens, static_cast<unsigned>(777 + tokens));
    std::mt19937 rng(static_cast<unsigned>(99 + tokens));
    std::normal_distribution<float> normal(0.0F, 4.0F);
    std::vector<__nv_bfloat16> residual(static_cast<std::size_t>(n) * tokens);
    std::vector<double> base(residual.size());
    for (std::size_t i = 0; i < residual.size(); ++i) {
        residual[i] = __float2bfloat16(normal(rng));
        base[i]     = __bfloat162float(residual[i]);
    }
    const std::vector<double> ref =
        reference(f, takes_wide_route(n, k, tokens) ? exact_activation(x)
                                                    : quantised_activation(x, k, tokens),
                  tokens, &base);
    __nv_bfloat16* d_x   = nullptr;
    __nv_bfloat16* d_res = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_res, residual.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_res, residual.data(), residual.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    gg::linear_add_launch(f.type, d_blocks, n, k, d_x, tokens, d_res, d_scratch, scratch_bytes, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> raw(residual.size());
    CHECK_CUDA(cudaMemcpy(raw.data(), d_res, raw.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_res));
    std::vector<float> got(raw.size());
    for (std::size_t i = 0; i < got.size(); ++i) { got[i] = __bfloat162float(raw[i]); }
    return report(f, takes_wide_route(n, k, tokens) ? "linear_add (wide)" : "linear_add_launch",
                  tokens, score(got, ref), 4e-3, 8e-3);
}

// ops::linear with no workspace: the wrapper, the QType dispatch and the engine-slot scratch.
int run_wrapper_linear(const Fixture& f, int tokens, void* d_blocks) {
    const int n = f.n, k = f.k;
    const auto x = random_activation(k, tokens, static_cast<unsigned>(4242 + tokens));
    const std::vector<double> ref =
        reference(f, takes_wide_route(n, k, tokens) ? exact_activation(x)
                                                    : quantised_activation(x, k, tokens),
                  tokens, nullptr);
    __nv_bfloat16* d_x   = nullptr;
    __nv_bfloat16* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_out, static_cast<std::size_t>(n) * tokens * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    const Weight w = make_weight(f, d_blocks);
    const Tensor xt(d_x, DType::BF16, {k, tokens});
    Tensor out(d_out, DType::BF16, {n, tokens});
    sinfer::ops::linear(xt, w, out, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> raw(static_cast<std::size_t>(n) * tokens);
    CHECK_CUDA(cudaMemcpy(raw.data(), d_out, raw.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_out));
    std::vector<float> got(raw.size());
    for (std::size_t i = 0; i < got.size(); ++i) { got[i] = __bfloat162float(raw[i]); }
    return report(f, "ops::linear (no ws)", tokens, score(got, ref), 4e-3, 8e-3);
}

// The row gather, through the launcher and through ops::embedding, against the oracle rows.
int run_gather(const Fixture& f, bool through_wrapper, void* d_blocks) {
    const int n = f.n, k = f.k, tokens = 64;
    std::mt19937 rng(through_wrapper ? 98 : 99);
    std::vector<std::int32_t> ids(tokens);
    for (auto& id : ids) { id = static_cast<std::int32_t>(rng() % static_cast<unsigned>(n)); }
    std::int32_t* d_ids  = nullptr;
    __nv_bfloat16* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ids, ids.size() * sizeof(std::int32_t)));
    CHECK_CUDA(cudaMalloc(&d_out, static_cast<std::size_t>(k) * tokens * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    if (through_wrapper) {
        const Weight w = make_weight(f, d_blocks);
        const Tensor idt(d_ids, DType::I32, {tokens});
        Tensor out(d_out, DType::BF16, {k, tokens});
        sinfer::ops::embedding(idt, w, out, nullptr);
    } else {
        gg::embedding_gather_launch(f.type, d_blocks, n, k, d_ids, tokens, d_out, nullptr);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> got(static_cast<std::size_t>(k) * tokens);
    CHECK_CUDA(cudaMemcpy(got.data(), d_out, got.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_ids));
    CHECK_CUDA(cudaFree(d_out));
    double max_rel = 0.0;
    for (int t = 0; t < tokens; ++t) {
        for (int i = 0; i < k; ++i) {
            const float ref    = f.dequant[static_cast<std::size_t>(ids[t]) * k + i];
            const float expect = __bfloat162float(__float2bfloat16(ref)); // exact up to the BF16 store
            const float val    = __bfloat162float(got[static_cast<std::size_t>(t) * k + i]);
            max_rel = std::fmax(max_rel, std::fabs(static_cast<double>(val) - expect) / std::fmax(std::fabs(expect), 1e-6));
        }
    }
    const bool ok = max_rel <= 1e-6;
    std::printf("  %-5s %-10s [%6d,%5d] %-22s 64 rows: max rel %.1e  %s\n", gg::type_name(f.type),
                f.label.c_str(), n, k, through_wrapper ? "ops::embedding" : "gather launch", max_rel,
                ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

// The routed-expert GEMV: the same weight read as [experts, rows, k], each token routed to
// `slots` experts by an id table, against the same reference the plain GEMV is held to.
int run_moe(const Fixture& f, void* d_blocks, void* d_scratch, std::size_t scratch_bytes) {
    constexpr int kExperts = 4, kTokens = 3, kSlots = 2;
    const int k = f.k, rows = f.n / kExperts;
    if (rows <= 0 || f.n % kExperts != 0) { return 0; }
    const auto x = random_activation(k, kTokens, 31337u);
    const std::vector<double> y = quantised_activation(x, k, kTokens);
    std::mt19937 rng(5150);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(kTokens) * kSlots);
    for (auto& id : ids) { id = static_cast<std::int32_t>(rng() % kExperts); }

    std::vector<double> ref(static_cast<std::size_t>(kTokens) * kSlots * rows, 0.0);
    for (int t = 0; t < kTokens; ++t) {
        for (int slot = 0; slot < kSlots; ++slot) {
            const int expert = ids[static_cast<std::size_t>(t) * kSlots + slot];
            for (int row = 0; row < rows; ++row) {
                const float* w  = &f.dequant[(static_cast<std::size_t>(expert) * rows + row) * k];
                const double* yt = &y[static_cast<std::size_t>(t) * k];
                double acc = 0.0;
                for (int i = 0; i < k; ++i) { acc += static_cast<double>(w[i]) * yt[i]; }
                ref[(static_cast<std::size_t>(t) * kSlots + slot) * rows + row] = acc;
            }
        }
    }
    __nv_bfloat16* d_x  = nullptr;
    std::int32_t* d_ids = nullptr;
    float* d_out        = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_ids, ids.size() * sizeof(std::int32_t)));
    CHECK_CUDA(cudaMalloc(&d_out, ref.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    gg::quantize_q8_1_launch(d_x, k, kTokens, static_cast<gg::block_q8_1*>(d_scratch), nullptr);
    gg::moe_gemv_launch(f.type, d_blocks, rows, k, static_cast<gg::block_q8_1*>(d_scratch), d_ids,
                        kTokens, kSlots, kSlots, d_out, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> got(ref.size());
    CHECK_CUDA(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_ids));
    CHECK_CUDA(cudaFree(d_out));
    const Score sc = score(got, ref);
    const bool ok  = sc.rel_l2 <= 2e-5 && sc.max_abs <= 2e-4 * sc.max_ref;
    std::printf("  %-5s %-10s [%6d,%5d] %-22s %dx%d routed: rel_l2=%.2e  %s\n", gg::type_name(f.type),
                f.label.c_str(), rows, k, "moe_gemv (4 experts)", kTokens, kSlots, sc.rel_l2,
                ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

// The sparse-MoE codec decodes eight consecutive values per lane instead of a whole block per
// CTA; it must agree exactly with the reference dequantisation, or a routed expert would be
// read differently from every other weight of the same type.
int run_codec(const Fixture& f, void* d_blocks) {
    if (f.type == gg::GgmlType::Q2_K || f.type == gg::GgmlType::Q3_K) { return 0; }
    const std::int64_t superblocks = static_cast<std::int64_t>(f.n) * (f.k / gg::QK_K);
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, superblocks * gg::QK_K * sizeof(float)));
    gg::moe_codec_decode_launch(f.type, d_blocks, superblocks, d_out, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> got(static_cast<std::size_t>(superblocks) * gg::QK_K);
    CHECK_CUDA(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_out));
    double worst = 0.0;
    std::size_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(got[i]) - f.dequant[i]);
        worst = std::fmax(worst, diff / std::fmax(std::fabs(f.dequant[i]), 1e-6));
        bad += diff > 1e-6 * std::fmax(std::fabs(f.dequant[i]), 1e-3);
    }
    const bool ok = bad == 0;
    std::printf("  %-5s %-10s [%6d,%5d] %-22s %zu of %zu values differ (worst rel %.1e)  %s\n",
                gg::type_name(f.type), f.label.c_str(), f.n, f.k, "moe codec vs oracle", bad,
                got.size(), worst, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

} // namespace

int main() {
    const char* env       = std::getenv("SINFER_GGML_FIXTURE_DIR");
    const std::string dir = env != nullptr ? env : "/tmp/surogate_ggml_test";
    int device_count      = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "SKIP: no CUDA device\n");
        return 77;
    }
    int failures = 0, cases = 0;
    const gg::GgmlType types[] = {gg::GgmlType::Q2_K, gg::GgmlType::Q3_K, gg::GgmlType::Q4_K,
                                  gg::GgmlType::Q5_K, gg::GgmlType::Q6_K};
    for (const gg::GgmlType type : types) {
        for (const char* label : {"synthetic", "odd", "real", "big"}) {
            Fixture f;
            if (!load_fixture(dir, type, label, f)) { continue; }
            const bool big = f.label == "big";
            void* d_blocks = nullptr;
            CHECK_CUDA(cudaMalloc(&d_blocks, f.blocks.size()));
            CHECK_CUDA(cudaMemcpy(d_blocks, f.blocks.data(), f.blocks.size(), cudaMemcpyHostToDevice));
            // The GEMV route needs its int8 planes, the wide route its dequantisation tile;
            // one buffer covers whichever the widest case picks.
            const int widest = big ? 2 : 512;
            const std::size_t scratch_bytes =
                std::max(gg::linear_workspace_bytes(f.n, f.k, widest),
                         gg::linear_workspace_bytes(f.n, f.k, 1));
            void* d_scratch = nullptr;
            CHECK_CUDA(cudaMalloc(&d_scratch, scratch_bytes));
            for (const int tokens : {1, 2, 3, 5, 8, 17}) {
                if (big && tokens > 2) { continue; }
                failures += run_case(f, tokens, false, d_blocks, d_scratch, scratch_bytes);
                ++cases;
            }
            failures += run_case(f, 1, true, d_blocks, d_scratch, scratch_bytes);
            ++cases;
            // Wide batches: above the threshold these run the dequantise-once tensor-core
            // route, so the same reference now covers both of a linear's two shapes.
            for (const int tokens : {64, 65, 512}) {
                if (big) { continue; }
                failures += run_case(f, tokens, true, d_blocks, d_scratch, scratch_bytes);
                ++cases;
            }
            for (const int tokens : {1, 3, 128}) {
                if (big && tokens > 1) { continue; }
                failures += run_accumulate(f, tokens, d_blocks, d_scratch, scratch_bytes);
                ++cases;
                failures += run_wrapper_linear(f, tokens, d_blocks);
                ++cases;
            }
            failures += run_gather(f, false, d_blocks);
            failures += run_gather(f, true, d_blocks);
            cases += 2;
            if (!big) {
                failures += run_moe(f, d_blocks, d_scratch, scratch_bytes);
                failures += run_codec(f, d_blocks);
                cases += 2;
            }
            CHECK_CUDA(cudaFree(d_scratch));
            CHECK_CUDA(cudaFree(d_blocks));
        }
    }
    if (cases == 0) {
        std::fprintf(stderr, "SKIP: no fixtures in %s (run gen_fixture.py)\n", dir.c_str());
        return 77;
    }
    std::printf("%d cases, %d failures\n", cases, failures);
    return failures == 0 ? 0 : 1;
}
