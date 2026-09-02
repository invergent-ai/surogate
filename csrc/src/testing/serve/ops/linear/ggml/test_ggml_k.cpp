// K-quant GEMV against an independent reference. Fixtures (gen_fixture.py) hold each type's
// weight in native GGML blocks plus gguf-py's exact dequantisation. The reference quantises the
// activation exactly as the kernel does (per 32: d = amax/127 in fp32, q = round(x/d), d kept
// as fp16) and takes the fp64 dot of the dequantised weight with that quantised activation, so
// the only divergence left is fp32 accumulation order -- a tight bound, not a statistical one.
#include "ops/linear/ggml/ggml_linear.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

namespace gg = sinfer::ops::detail::ggml;

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
            const float d_half = __half2float(__float2half(d)); // stored as half, read back
            for (int i = 0; i < 32; ++i) {
                const int q = amax == 0.0F ? 0 : static_cast<int>(std::round(vals[i] / d));
                y[static_cast<std::size_t>(t) * k + b * 32 + i] = static_cast<double>(d_half) * q;
            }
        }
    }
    return y;
}

int run_case(const Fixture& f, int tokens, bool bf16_out, void* d_blocks, void* d_scratch,
             std::size_t scratch_bytes) {
    const int n = f.n, k = f.k;
    std::mt19937 rng(static_cast<unsigned>(1234 + tokens + 7 * static_cast<int>(f.type)));
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<__nv_bfloat16> x(static_cast<std::size_t>(k) * tokens);
    for (auto& v : x) { v = __float2bfloat16(normal(rng)); }

    const std::vector<double> y = quantised_activation(x, k, tokens);
    std::vector<double> ref(static_cast<std::size_t>(n) * tokens, 0.0);
    for (int t = 0; t < tokens; ++t) {
        for (int row = 0; row < n; ++row) {
            double acc = 0.0;
            const float* w = &f.dequant[static_cast<std::size_t>(row) * k];
            const double* yt = &y[static_cast<std::size_t>(t) * k];
            for (int i = 0; i < k; ++i) { acc += static_cast<double>(w[i]) * yt[i]; }
            ref[static_cast<std::size_t>(t) * n + row] = acc;
        }
    }

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

    if (std::getenv("SINFER_GGML_DEBUG") != nullptr && tokens <= 3 && !bf16_out) {
        for (int t = 0; t < tokens; ++t) {
            for (int row = 0; row < 4; ++row) {
                const std::size_t at = static_cast<std::size_t>(t) * n + row;
                std::printf("      col %d row %d: got %12.5f ref %12.5f  ratio %.4f\n", t, row, got[at],
                            ref[at], got[at] / ref[at]);
            }
        }
    }
    double num = 0.0, den = 0.0, max_abs = 0.0, max_ref = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double diff = static_cast<double>(got[i]) - ref[i];
        num += diff * diff;
        den += ref[i] * ref[i];
        max_abs = std::fmax(max_abs, std::fabs(diff));
        max_ref = std::fmax(max_ref, std::fabs(ref[i]));
    }
    const double rel_l2 = std::sqrt(num / std::fmax(den, 1e-300));
    // fp32 accumulation over k <= 2048 terms: ~1e-6; BF16 output adds its 2^-9 per element.
    const double bound_l2  = bf16_out ? 4e-3 : 2e-5;
    const double bound_abs = bf16_out ? 8e-3 * max_ref : 2e-4 * max_ref;
    const bool ok = rel_l2 <= bound_l2 && max_abs <= bound_abs;
    std::printf("  %-5s %-10s [%5d,%5d] T=%-2d %-4s rel_l2=%.2e max_abs=%.2e (%.1e of max)  %s\n",
                gg::type_name(f.type), f.label.c_str(), n, k, tokens, bf16_out ? "bf16" : "f32",
                rel_l2, max_abs, max_abs / std::fmax(max_ref, 1e-300), ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

} // namespace

int main() {
    const char* env = std::getenv("SINFER_GGML_FIXTURE_DIR");
    const std::string dir = env != nullptr ? env : "/tmp/surogate_ggml_test";
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "SKIP: no CUDA device\n");
        return 77;
    }
    int failures = 0, cases = 0;
    const gg::GgmlType types[] = {gg::GgmlType::Q2_K, gg::GgmlType::Q3_K, gg::GgmlType::Q4_K,
                                  gg::GgmlType::Q5_K, gg::GgmlType::Q6_K};
    for (const gg::GgmlType type : types) {
        for (const char* label : {"synthetic", "odd", "real"}) {
            Fixture f;
            if (!load_fixture(dir, type, label, f)) { continue; }
            void* d_blocks = nullptr;
            CHECK_CUDA(cudaMalloc(&d_blocks, f.blocks.size()));
            CHECK_CUDA(cudaMemcpy(d_blocks, f.blocks.data(), f.blocks.size(), cudaMemcpyHostToDevice));
            const std::size_t scratch_bytes = gg::linear_workspace_bytes(f.k, 17);
            void* d_scratch = nullptr;
            CHECK_CUDA(cudaMalloc(&d_scratch, scratch_bytes));
            for (const int tokens : {1, 2, 3, 5, 8, 17}) {
                failures += run_case(f, tokens, false, d_blocks, d_scratch, scratch_bytes);
                ++cases;
            }
            failures += run_case(f, 1, true, d_blocks, d_scratch, scratch_bytes);
            ++cases;
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
