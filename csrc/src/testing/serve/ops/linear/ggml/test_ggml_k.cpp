// GGML projection routes against an independent reference. Fixtures (gen_fixture.py) hold each
// type's weight in native GGML blocks plus gguf-py's exact dequantisation. The reference
// quantises the activation exactly as the kernel does (per 32: d = amax/127 in fp32,
// q = round(x/d), d kept as fp16) and takes the fp64 dot of the dequantised weight with that
// quantised activation (the F16 projection keeps the original activation), leaving only
// fp32 accumulation order and output rounding -- a tight
// bound, not a statistical one. Beyond the launchers, the public wrappers are driven too:
// ops::linear without a workspace (the engine-slot scratch the lm_head takes) and
// ops::embedding (the row gather).
#include "api/ops/embedding.h"
#include "api/ops/linear.h"
#include "api/ops/linear_swiglu.h"
#include "api/ops/silu_mul.h"
#include "ops/linear/ggml/ggml_dispatch.h"
#include "ops/linear/ggml/ggml_embedding.h"
#include "ops/linear/ggml/ggml_linear.h"
#include "ops/linear/ggml/ggml_moe.h"
#include "ops/linear/ggml/ggml_q8_1.h"
#include "ops/linear/ggml/ggml_swiglu.h"

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
#include <type_traits>
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
    // Not every GGML type is a superblock: Q8_0, Q4_1 and Q5_1 hold 32 values.
    const std::size_t expect_blocks = static_cast<std::size_t>(f.n) *
                                      (f.k / gg::block_values(type)) * gg::block_bytes(type);
    if (f.blocks.size() != expect_blocks || f.dequant.size() != static_cast<std::size_t>(f.n) * f.k) {
        std::fprintf(stderr, "fixture %s: unexpected sizes\n", stem.c_str());
        return false;
    }
    return true;
}

QType qtype_of(gg::GgmlType type) {
    switch (type) {
#define SINFER_TEST_MAP(NAME) case gg::GgmlType::NAME: return QType::NAME;
        SINFER_GGML_FOR_EACH_TYPE(SINFER_TEST_MAP)
#undef SINFER_TEST_MAP
    }
    return QType::Q4_K;
}

// The Weight the artifact loader builds for a GgmlBlocks object (typed_binding.cpp).
Weight make_weight(const Fixture& f, void* d_blocks) {
    Weight w{};
    w.payload         = d_blocks;
    w.payload_bytes   = f.blocks.size();
    w.qtype           = qtype_of(f.type);
    // The group is the block's value count, which is 256 only for the superblock types.
    const std::int32_t values = gg::block_values(f.type);
    w.group_size      = values;
    w.ndim            = 2;
    w.qdata           = d_blocks;
    w.n               = f.n;
    w.k               = f.k;
    w.group           = values;
    w.layout          = QuantLayout::GgmlBlocks;
    w.scale_dtype     = DType::FP16;
    w.shape[0]        = f.n;
    w.shape[1]        = f.k;
    w.padded_shape[0] = f.n;
    w.padded_shape[1] = f.k;
    return w;
}

// All quantized formats use the same activation at every batch width. F16
// preserves the BF16 activation without additional quantization.
bool takes_wide_route(int tokens) {
    return tokens >= gg::wide_min_tokens();
}

bool uses_exact_activation(const Fixture& f) {
    return f.type == gg::GgmlType::F16;
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
    const bool wide = bf16_out && takes_wide_route(tokens);
    const std::vector<double> ref =
        reference(f, bf16_out && uses_exact_activation(f) ? exact_activation(x)
                                                                : quantised_activation(x, k, tokens), tokens, nullptr);
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
    if (const char* dump = std::getenv("SINFER_GGML_DUMP_DIR"); dump != nullptr && bf16_out) {
        // got / ref as f64, x as f32, for offline analysis of a route's error structure
        const std::string stem = std::string(dump) + "/" + gg::type_name(f.type) + "_" + f.label +
                                 "_T" + std::to_string(tokens);
        std::vector<double> g64(got.begin(), got.end());
        std::vector<float> x32(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) { x32[i] = __bfloat162float(x[i]); }
        std::ofstream(stem + ".got", std::ios::binary).write(reinterpret_cast<const char*>(g64.data()), g64.size() * 8);
        std::ofstream(stem + ".ref", std::ios::binary).write(reinterpret_cast<const char*>(ref.data()), ref.size() * 8);
        std::ofstream(stem + ".x", std::ios::binary).write(reinterpret_cast<const char*>(x32.data()), x32.size() * 4);
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
        reference(f, uses_exact_activation(f) ? exact_activation(x)
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
    return report(f, takes_wide_route(tokens) ? "linear_add (wide)" : "linear_add_launch",
                  tokens, score(got, ref), 4e-3, 8e-3);
}

// ops::linear with no workspace: the wrapper, the QType dispatch and the engine-slot scratch.
int run_wrapper_linear(const Fixture& f, int tokens, void* d_blocks) {
    const int n = f.n, k = f.k;
    const auto x = random_activation(k, tokens, static_cast<unsigned>(4242 + tokens));
    const std::vector<double> ref =
        reference(f, uses_exact_activation(f) ? exact_activation(x)
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

// One input prefix evaluated at several widths must retain its represented
// outputs. This crosses both the decode and prefill tile boundaries.
int run_consistent_columns(const Fixture& f, void* d_blocks, void* scratch,
                           std::size_t scratch_bytes) {
    constexpr int columns = 129;
    auto x = random_activation(f.k, columns, 1773);
    std::fill_n(x.begin() + 2 * f.k, f.k, __float2bfloat16(0.0f));
    std::fill_n(x.begin() + f.k, f.k, __float2bfloat16(-0.003f));
    __nv_bfloat16 *d_x = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_out, std::size_t(f.n) * columns * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    std::vector<__nv_bfloat16> reference(std::size_t(f.n) * columns), got(reference.size());
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int failures = 0;
    for (bool accumulate : {false, true}) {
        const auto run = [&](int tokens, bool capture) {
            CHECK_CUDA(cudaMemsetAsync(d_out, 0x3e, got.size() * sizeof(__nv_bfloat16), stream));
            if (capture) { CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); }
            if (accumulate) {
                gg::linear_add_launch(f.type, d_blocks, f.n, f.k, d_x, tokens, d_out,
                                      scratch, scratch_bytes, stream);
            } else {
                gg::linear_launch(f.type, d_blocks, f.n, f.k, d_x, tokens, d_out,
                                  scratch, scratch_bytes, stream);
            }
            if (capture) {
                cudaGraph_t graph;
                cudaGraphExec_t executable;
                CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
                CHECK_CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                CHECK_CUDA(cudaGraphLaunch(executable, stream));
                CHECK_CUDA(cudaStreamSynchronize(stream));
                CHECK_CUDA(cudaGraphExecDestroy(executable));
                CHECK_CUDA(cudaGraphDestroy(graph));
            }
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaMemcpy(got.data(), d_out, got.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        };
        run(columns, false);
        reference = got;
        for (bool capture : {false, true}) {
            for (int tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 32, 33, 64, 65, 128}) {
                run(tokens, capture);
                const auto count = std::size_t(f.n) * tokens;
                const bool equal = std::memcmp(reference.data(), got.data(), count * sizeof(__nv_bfloat16)) == 0;
                const auto* tail = reinterpret_cast<const unsigned char*>(got.data() + count);
                const auto tail_bytes = (got.size() - count) * sizeof(__nv_bfloat16);
                const bool guarded = std::all_of(tail, tail + tail_bytes, [](unsigned char v) { return v == 0x3e; });
                failures += !(equal && guarded);
                if (!equal || !guarded) {
                    std::printf("  %s %s batch invariant accumulate=%d graph=%d T=%d: FAIL (equal=%d guarded=%d)\n",
                                gg::type_name(f.type), f.label.c_str(), accumulate, capture, tokens, equal, guarded);
                }
            }
        }
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_x));
    std::printf("  %s %s batch invariant projection/residual: %s\n",
                gg::type_name(f.type), f.label.c_str(), failures ? "FAIL" : "ok");
    return failures;
}

// Compare the fused public operation with independent projection + activation
// calls at a prefill width. Minimal decode workspace also proves that gate/up
// intermediate buffers are not allocated, including during graph capture.
int run_swiglu(const Fixture& gate_f, const Fixture& up_f, bool mapped, bool segmented,
               int& cases, int columns = 129) {
    const int n = gate_f.n, k = gate_f.k;
    auto up_bytes = up_f.blocks;
    std::rotate(up_bytes.begin(), up_bytes.begin() + gg::block_bytes(up_f.type), up_bytes.end());
    sinfer::DeviceBuffer gate_storage(gate_f.blocks.size() + (segmented ? 0 : up_bytes.size()));
    sinfer::DeviceBuffer up_storage(segmented ? up_bytes.size() : 0);
    gate_storage.copy_from_host(gate_f.blocks.data(), gate_f.blocks.size());
    void* up_data;
    if (segmented) {
        up_storage.copy_from_host(up_bytes.data(), up_bytes.size());
        up_data = up_storage.p;
    } else {
        gate_storage.copy_from_host(up_bytes.data(), up_bytes.size(), gate_f.blocks.size());
        up_data = static_cast<std::uint8_t*>(gate_storage.p) + gate_f.blocks.size();
    }
    Weight parent = make_weight(gate_f, gate_storage.p);
    parent.n = parent.shape[0] = parent.padded_shape[0] = 2 * n;
    parent.payload_bytes = gate_f.blocks.size() + up_bytes.size();
    sinfer::WeightSegment segments[] = {
        {0, n, qtype_of(gate_f.type), gate_storage.p, gate_f.blocks.size()},
        {n, n, qtype_of(up_f.type), up_data, up_bytes.size()}};
    if (segmented) { parent.segments = segments; parent.segment_count = 2; }
    std::vector<std::int32_t> map(k / 32);
    for (int i = 0; i < k / 32; ++i) { map[i] = k / 32 - 1 - i; }
    sinfer::DeviceBuffer map_storage(map.size() * sizeof(std::int32_t));
    map_storage.copy_from_host(map.data(), map_storage.bytes);
    if (mapped) { parent.input_group_map = static_cast<const std::int32_t*>(map_storage.p); }

    auto x = random_activation(k, columns, 9127);
    std::fill_n(x.begin() + k, k, __float2bfloat16(-0.003f));
    std::fill_n(x.begin() + 2 * k, k, __float2bfloat16(0.0f));
    sinfer::DeviceBuffer input(x.size() * sizeof(__nv_bfloat16));
    input.copy_from_host(x.data(), input.bytes);
    const std::size_t output_bytes = std::size_t(n) * columns * sizeof(__nv_bfloat16);
    sinfer::DeviceBuffer gate_ref(output_bytes), up_ref(output_bytes), expected_gpu(output_bytes), output(output_bytes);
    Tensor activation(input.p, DType::BF16, {k, columns});
    Tensor gate_tensor(gate_ref.p, DType::BF16, {n, columns});
    Tensor up_tensor(up_ref.p, DType::BF16, {n, columns});
    Tensor reference_tensor(expected_gpu.p, DType::BF16, {n, columns});
    sinfer::WorkspaceArena full(sinfer::ops::linear_swiglu_workspace_capacity_bytes(
        parent.qtype, parent.n, k, 1, columns));
    const auto planned = sinfer::ops::linear_swiglu_workspace_capacity_bytes(
        parent.qtype, qtype_of(up_f.type), parent.n, k, sinfer::ops::LinearPolicy::A16Only, 1, columns);
    sinfer::WorkspaceArena selected(planned);
    sinfer::WorkspaceArena small(gg::linear_workspace_bytes(n, k, columns) + 256);
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    gg::ggml_project_rows(activation, parent, 0, gate_tensor, &full, stream);
    gg::ggml_project_rows(activation, parent, n, up_tensor, &full, stream);
    sinfer::ops::silu_mul(gate_tensor, up_tensor, reference_tensor, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<__nv_bfloat16> expected(std::size_t(n) * columns), got(expected.size());
    expected_gpu.copy_to_host(expected.data(), output_bytes);
    int failures = 0;
    for (bool capture : {false, true}) {
        for (int tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 31, 32, 33, 63, 64, 65, 128, 129, 257, 513, 2048, 2053}) {
            if (tokens > columns) { continue; }
            const bool fused = gg::swiglu_decode_admits(gate_f.type, up_f.type, n, k, tokens) ||
                               gg::swiglu_prefill_admits(gate_f.type, up_f.type, n, k, tokens);
            auto& workspace = fused ? small : selected;
            workspace.reset_peak();
            Tensor in(input.p, DType::BF16, {k, tokens});
            Tensor out(output.p, DType::BF16, {n, tokens});
            CHECK_CUDA(cudaMemsetAsync(output.p, 0x3e, output_bytes, stream));
            if (capture) { CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)); }
            sinfer::ops::linear_swiglu(in, parent, out, workspace, stream);
            if (capture) {
                cudaGraph_t graph;
                cudaGraphExec_t executable;
                CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
                CHECK_CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                CHECK_CUDA(cudaGraphLaunch(executable, stream));
                CHECK_CUDA(cudaStreamSynchronize(stream));
                CHECK_CUDA(cudaGraphExecDestroy(executable));
                CHECK_CUDA(cudaGraphDestroy(graph));
            }
            CHECK_CUDA(cudaStreamSynchronize(stream));
            output.copy_to_host(got.data(), output_bytes);
            const std::size_t count = std::size_t(n) * tokens;
            const bool equal = std::memcmp(got.data(), expected.data(), count * sizeof(__nv_bfloat16)) == 0;
            const auto* tail = reinterpret_cast<const unsigned char*>(got.data() + count);
            const bool guarded = std::all_of(tail, tail + (got.size() - count) * sizeof(__nv_bfloat16),
                                             [](unsigned char v) { return v == 0x3e; });
            const auto interval_capacity = sinfer::ops::linear_swiglu_workspace_capacity_bytes(
                parent.qtype, qtype_of(up_f.type), parent.n, k, sinfer::ops::LinearPolicy::A16Only, 1, tokens);
            const auto call_capacity = sinfer::ops::linear_swiglu_workspace_capacity_bytes(
                parent.qtype, qtype_of(up_f.type), parent.n, k, sinfer::ops::LinearPolicy::A16Only, tokens, tokens);
            const bool bounded = workspace.used() == 0 && workspace.peak_used() <= planned &&
                workspace.peak_used() <= interval_capacity && workspace.peak_used() <= call_capacity &&
                (!fused || workspace.peak_used() <= gg::linear_workspace_bytes(n, k, tokens) + 255);
            const bool ok = equal && guarded && bounded;
            ++cases;
            failures += !ok;
            if (!ok) {
                std::printf("  SwiGLU %s/%s %s map=%d segments=%d T=%d graph=%d: FAIL (equal=%d guard=%d workspace=%zu)\n",
                            gg::type_name(gate_f.type), gg::type_name(up_f.type), gate_f.label.c_str(),
                            mapped, segmented, tokens, capture, equal, guarded, workspace.peak_used());
            }
        }
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
    std::printf("  SwiGLU %s/%s %s map=%d segments=%d: %s\n", gg::type_name(gate_f.type),
                gg::type_name(up_f.type), gate_f.label.c_str(), mapped, segmented, failures ? "FAIL" : "ok");
    return failures;
}

// Exercise large matrices on both sides of the bandwidth fallback, including
// switching between scalar fusion, batched decode and prefill for one weight.
int run_large_swiglu(const std::string& dir, int& cases) {
    std::vector<Fixture> weights;
    for (auto type : {gg::GgmlType::Q8_0, gg::GgmlType::IQ4_NL, gg::GgmlType::Q4_K,
                      gg::GgmlType::Q5_K, gg::GgmlType::Q6_K}) {
        Fixture f;
        if (!load_fixture(dir, type, "synthetic", f)) { continue; }
        const auto original = f.blocks;
        f.n = f.k = 8192;
        f.label = "large_swiglu";
        f.dequant.clear();
        f.blocks.resize(std::size_t(f.n) * (f.k / gg::block_values(type)) * gg::block_bytes(type));
        for (std::size_t offset = 0; offset < f.blocks.size(); offset += original.size()) {
            std::memcpy(f.blocks.data() + offset, original.data(),
                        std::min(original.size(), f.blocks.size() - offset));
        }
        weights.push_back(std::move(f));
    }
    int failures = 0;
    for (const auto& gate : weights) {
        for (const auto& up : weights) {
            failures += run_swiglu(gate, up, false, true, cases);
        }
    }
    return failures;
}

// A zero gate must stay zero after affine cancellation, even with nonzero up
// weights and activation scales whose sum is not exactly representable in FP16.
template <class Block>
int run_zero_swiglu(const Fixture& up, int& cases) {
    Fixture gate = up;
    gate.label = "zero_gate";
    Block b{};
    if constexpr (std::is_same_v<Block, gg::block_q6_K>) {
        b.d = __float2half(1.0f);
        std::fill_n(b.scales, gg::QK_K / 16, 1);
        std::fill_n(b.qh, gg::QK_K / 4, 0xaa);
    } else {
        b.dm = __floats2half2_rn(1.0f, 1.0f);
        std::fill_n(b.scales, 8, 1);
        std::fill_n(b.scales + 8, 4, 0x11);
        std::fill_n(b.qs, gg::QK_K / 2, 0x11);
    }
    for (std::size_t offset = 0; offset < gate.blocks.size(); offset += sizeof(Block)) {
        std::memcpy(gate.blocks.data() + offset, &b, sizeof(Block));
    }
    return run_swiglu(gate, up, false, true, cases);
}

// Every stored value is d*1 - dmin*1 == 0. An affine projection must stay zero
// even when the activation scale/sum cannot be represented exactly as half2.
// Using a rounded sum with an independently rounded scale breaks that identity.
// Q6_K stores signed zeros in an unpadded buffer, exercising the last header load.
template <class Block>
int run_affine_zero(gg::GgmlType type, int tokens) {
    constexpr int n = std::is_same_v<Block, gg::block_q6_K> ? 63 : 72;
    // An odd number of superblocks also exercises the unrolled decode tail.
    constexpr int k = 768, guard = 16;
    std::vector<Block> blocks(n * k / gg::block_values(type));
    for (auto& block : blocks) {
        if constexpr (std::is_same_v<Block, gg::block_q6_K>) {
            block.d = __float2half(1.0f);
            std::fill_n(block.scales, gg::QK_K / 16, 1);
            std::fill_n(block.qh, gg::QK_K / 4, 0xaa);
        } else if constexpr (std::is_same_v<Block, gg::block_q2_K>) {
            block.dm = __floats2half2_rn(1.0f, 1.0f);
            std::fill_n(block.scales, gg::QK_K / 16, 0x11);
            std::fill_n(block.qs, gg::QK_K / 4, 0x55);
        } else if constexpr (std::is_same_v<Block, gg::block_q4_1> ||
                             std::is_same_v<Block, gg::block_q5_1>) {
            block.dm = __floats2half2_rn(1.0f, -1.0f);
            std::fill_n(block.qs, 16, 0x11);
        } else {
            block.dm = __floats2half2_rn(1.0f, 1.0f);
            std::fill_n(block.scales, 8, 1);
            std::fill_n(block.scales + 8, 4, 0x11);
            std::fill_n(block.qs, gg::QK_K / 2, 0x11);
        }
    }
    std::vector<__nv_bfloat16> x(k * tokens, __float2bfloat16(0.003f));
    std::vector<__nv_bfloat16> got(n * tokens + 2 * guard, __float2bfloat16(7.0f));
    Block* d_blocks = nullptr;
    __nv_bfloat16 *d_x = nullptr, *d_out = nullptr;
    void* scratch = nullptr;
    const auto bytes = gg::linear_workspace_bytes(n, k, tokens);
    CHECK_CUDA(cudaMalloc(&d_blocks, blocks.size() * sizeof(Block)));
    CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_out, got.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&scratch, bytes));
    CHECK_CUDA(cudaMemcpy(d_blocks, blocks.data(), blocks.size() * sizeof(Block), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_out, got.data(), got.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto run = [&] {
        gg::linear_launch(type, d_blocks, n, k, d_x, tokens, d_out + guard, scratch, bytes, stream);
    };
    run();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    run();
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    int failures = 0;
    for (float value : {0.003f, -0.0173f, 0.0f}) {
        std::fill(x.begin(), x.end(), __float2bfloat16(value));
        CHECK_CUDA(cudaMemcpyAsync(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaGraphLaunch(executable, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaMemcpy(got.data(), d_out, got.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < int(got.size()); ++i) {
            const float expected = i < guard || i >= guard + n * tokens ? 7.0f : 0.0f;
            ok &= __bfloat162float(got[i]) == expected;
        }
        failures += !ok;
        std::printf("  %-5s zero-weight graph replay T=%d x=%g: %s\n", gg::type_name(type), tokens, value, ok ? "ok" : "FAIL");
    }
    CHECK_CUDA(cudaGraphExecDestroy(executable));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(scratch));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_blocks));
    return failures;
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
    for (int tokens : {1, 3, 8, 128, 8193}) {
        failures += run_affine_zero<gg::block_q4_K>(gg::GgmlType::Q4_K, tokens);
        failures += run_affine_zero<gg::block_q5_K>(gg::GgmlType::Q5_K, tokens);
        failures += run_affine_zero<gg::block_q6_K>(gg::GgmlType::Q6_K, tokens);
        failures += run_affine_zero<gg::block_q2_K>(gg::GgmlType::Q2_K, tokens);
        failures += run_affine_zero<gg::block_q4_1>(gg::GgmlType::Q4_1, tokens);
        failures += run_affine_zero<gg::block_q5_1>(gg::GgmlType::Q5_1, tokens);
        cases += 18;
    }
    failures += run_affine_zero<gg::block_q2_K>(gg::GgmlType::Q2_K, 65536);
    cases += 3;
    const int built_in_cases = cases;
    // every format the list names, so a new one cannot ship untested
    const gg::GgmlType types[] = {
#define SINFER_TEST_TYPE(NAME) gg::GgmlType::NAME,
        SINFER_GGML_FOR_EACH_TYPE(SINFER_TEST_TYPE)
#undef SINFER_TEST_TYPE
    };
    for (const gg::GgmlType type : types) {
        for (const char* label : {"synthetic", "odd", "tail", "decode_rows", "real", "big"}) {
            Fixture f;
            if (!load_fixture(dir, type, label, f)) { continue; }
            const bool big = f.label == "big";
            void* d_blocks = nullptr;
            CHECK_CUDA(cudaMalloc(&d_blocks, f.blocks.size()));
            CHECK_CUDA(cudaMemcpy(d_blocks, f.blocks.data(), f.blocks.size(), cudaMemcpyHostToDevice));
            // Reserve activation planes for every width before graph capture.
            std::size_t scratch_bytes = 0;
            for (const int tokens : {1, 2, 3, 5, 8, 17, 64, 65, 128, 512}) {
                if (big && tokens > 2) { continue; }
                scratch_bytes = std::max(scratch_bytes, gg::linear_workspace_bytes(f.n, f.k, tokens));
            }
            scratch_bytes = std::max(scratch_bytes, gg::linear_workspace_bytes(f.n, f.k, 129));
            void* d_scratch = nullptr;
            CHECK_CUDA(cudaMalloc(&d_scratch, scratch_bytes));
            for (const int tokens : {1, 2, 3, 5, 8, 17}) {
                if (big && tokens > 2) { continue; }
                failures += run_case(f, tokens, false, d_blocks, d_scratch, scratch_bytes);
                ++cases;
            }
            failures += run_case(f, 1, true, d_blocks, d_scratch, scratch_bytes);
            ++cases;
            // Wide batches cover the integer and F16 tensor-core routes.
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
                ++cases;
                if (f.k % gg::QK_K == 0) {
                    failures += run_codec(f, d_blocks);
                    ++cases;
                }
            }
            failures += run_consistent_columns(f, d_blocks, d_scratch, scratch_bytes);
            cases += 60;
            if ((type == gg::GgmlType::Q8_0 || type == gg::GgmlType::IQ4_NL ||
                 type == gg::GgmlType::Q4_K || type == gg::GgmlType::Q5_K || type == gg::GgmlType::Q6_K) &&
                (f.label == "synthetic" || f.label == "odd" || f.label == "tail" || f.label == "decode_rows")) {
                for (auto up_type : {gg::GgmlType::Q8_0, gg::GgmlType::IQ4_NL, gg::GgmlType::Q4_K,
                                     gg::GgmlType::Q5_K, gg::GgmlType::Q6_K, gg::GgmlType::F16}) {
                    Fixture up;
                    if (!load_fixture(dir, up_type, f.label, up)) { continue; }
                    if (up.n != f.n || up.k != f.k) { continue; }
                    for (bool mapped : {false, true}) {
                        failures += run_swiglu(f, up, mapped, true, cases);
                        if (type == up_type) { failures += run_swiglu(f, up, mapped, false, cases); }
                    }
                    if (f.label == "synthetic" &&
                        (type == gg::GgmlType::Q4_K || type == gg::GgmlType::Q5_K || type == gg::GgmlType::Q6_K) &&
                        (up_type == gg::GgmlType::Q4_K || up_type == gg::GgmlType::Q5_K || up_type == gg::GgmlType::Q6_K)) {
                        failures += run_swiglu(f, up, true, true, cases, 2053);
                    }
                }
            }
            if (f.label == "tail") {
                if (type == gg::GgmlType::Q4_K) { failures += run_zero_swiglu<gg::block_q4_K>(f, cases); }
                if (type == gg::GgmlType::Q5_K) { failures += run_zero_swiglu<gg::block_q5_K>(f, cases); }
                if (type == gg::GgmlType::Q6_K) { failures += run_zero_swiglu<gg::block_q6_K>(f, cases); }
            }
            CHECK_CUDA(cudaFree(d_scratch));
            CHECK_CUDA(cudaFree(d_blocks));
        }
    }
    failures += run_large_swiglu(dir, cases);
    if (cases == built_in_cases) {
        std::fprintf(stderr, "Only built-in zero-weight cases ran: no fixtures in %s (run gen_fixture.py)\n", dir.c_str());
    }
    std::printf("%d cases, %d failures\n", cases, failures);
    return failures == 0 ? 0 : 1;
}
