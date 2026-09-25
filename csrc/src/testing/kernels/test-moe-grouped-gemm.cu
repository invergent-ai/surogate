// MoE expert GEMMs (moe_grouped_gemm*, kernels/moe/): every bf16 entry point against a double-precision
// per-expert reference; a token's output independent of how many tokens its expert received (row
// packing compares a packed row with the row alone bit for bit); and no lock that a second GPU's
// worker thread could wait for while the first is blocked in a launch (the multi-GPU training hang
// that cublasGemmGroupedBatchedEx caused).
#include <catch2/catch_test_macros.hpp>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <thread>
#include <vector>

#include "kernels/kernels.h"
#include "recipes/recipe.h"
#include "recipes/recipe_factory.h"
#include "runtime/core/matmul_context.h"

cudnnHandle_t create_cudnn_handle();  // runtime/attention/attention_cudnn.cpp
void destroy_cudnn_handle(cudnnHandle_t handle) noexcept;

namespace {

constexpr int E = 16;   // experts (two of them empty)
constexpr int C = 256;  // hidden
constexpr int D = 128;  // expert intermediate
constexpr int R = 8;    // LoRA rank

std::vector<int> token_counts(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(1, 70);
    std::vector<int> t(E);
    for (int e = 0; e < E; ++e)
        t[e] = (e == 3 || e == 11) ? 0 : dist(rng);
    return t;
}
std::vector<int> offsets_of(const std::vector<int>& t) {
    std::vector<int> off(t.size() + 1, 0);
    for (std::size_t e = 0; e < t.size(); ++e)
        off[e + 1] = off[e] + t[e];
    return off;
}

struct Bf16 {
    std::vector<float> host;  // the bf16-rounded values
    nv_bfloat16* dev = nullptr;
    Bf16(std::size_t n, float scale, std::mt19937& rng)
        : host(n) {
        std::uniform_real_distribution<float> u(-scale, scale);
        std::vector<nv_bfloat16> h(n);
        for (std::size_t i = 0; i < n; ++i) {
            h[i] = __float2bfloat16(scale == 0.0f ? 0.0f : u(rng));
            host[i] = __bfloat162float(h[i]);
        }
        REQUIRE(cudaMalloc(&dev, std::max<std::size_t>(n, 1) * sizeof(nv_bfloat16)) == cudaSuccess);
        REQUIRE(cudaMemcpy(dev, h.data(), n * sizeof(nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    Bf16(const Bf16&) = delete;
    ~Bf16() {
        cudaFree(dev);
    }
    std::vector<float> read() const {
        std::vector<nv_bfloat16> h(host.size());
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        REQUIRE(cudaMemcpy(h.data(), dev, h.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost) == cudaSuccess);
        std::vector<float> out(h.size());
        for (std::size_t i = 0; i < h.size(); ++i)
            out[i] = __bfloat162float(h[i]);
        return out;
    }
};

struct Offsets {
    std::vector<int> host;
    int* dev = nullptr;
    explicit Offsets(std::vector<int> off)
        : host(std::move(off)) {
        REQUIRE(cudaMalloc(&dev, host.size() * sizeof(int)) == cudaSuccess);
        REQUIRE(cudaMemcpy(dev, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    ~Offsets() {
        cudaFree(dev);
    }
    int total() const {
        return host.back();
    }
};

// Relative L2 distance of `got` from a double-precision reference.
double rel_l2(const std::vector<float>& got, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        num += (got[i] - ref[i]) * (got[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / den);
}

// Per expert e and each of its tokens t: out[t, :] = f(e, t) (row-major, width `cols`).
std::vector<double> per_token(const Offsets& off, int cols, const std::function<double(int, int, int)>& f) {
    std::vector<double> out(static_cast<std::size_t>(off.total()) * cols, 0.0);
    for (int e = 0; e < E; ++e)
        for (int t = off.host[e]; t < off.host[e + 1]; ++t)
            for (int j = 0; j < cols; ++j)
                out[static_cast<std::size_t>(t) * cols + j] = f(e, t, j);
    return out;
}

struct Handles {
    cublasHandle_t cublas{};
    cudaStream_t stream{};
    Handles() {
        REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
        REQUIRE(cublasCreate(&cublas) == CUBLAS_STATUS_SUCCESS);
    }
    ~Handles() {
        cublasDestroy(cublas);
        cudaStreamDestroy(stream);
    }
};

// bf16 output rounding alone gives ~1.6e-3; the kernels add only fp32 accumulation error.
constexpr double kTol = 4e-3;

}  // namespace

TEST_CASE("MoE grouped GEMM bf16 entry points match a double-precision reference", "[moe][gemm]") {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) SKIP("no CUDA device");
    Handles h;
    std::mt19937 rng(20260925);
    Offsets off(offsets_of(token_counts(7)));
    const int T = off.total();
    Bf16 x_c(T * C, 1.0f, rng), x_d(T * D, 1.0f, rng), x_2d(T * 2 * D, 1.0f, rng), x_r(T * R, 1.0f, rng);
    Bf16 w_gu(E * 2 * D * C, 0.05f, rng), w_dn(E * C * D, 0.05f, rng), a_rc(E * R * C, 0.05f, rng),
        b_dr(E * D * R, 0.05f, rng);
    auto at = [](const std::vector<float>& v, std::size_t i) {
        return static_cast<double>(v[i]);
    };

    SECTION("LoRA A forward (TN): out = x @ A_e^T") {
        Bf16 out(T * R, 0.0f, rng);
        moe_grouped_gemm(out.dev,
                         x_c.dev,
                         a_rc.dev,
                         off.dev,
                         E,
                         R,
                         C,
                         h.cublas,
                         h.stream,
                         off.host.data(),
                         1.0f,
                         0.0f,
                         EMMTranspose::TN,
                         nullptr,
                         false,
                         -1);
        auto ref = per_token(off, R, [&](int e, int t, int j) {
            double s = 0;
            for (int k = 0; k < C; ++k)
                s += at(x_c.host, (std::size_t)t * C + k) * at(a_rc.host, ((std::size_t)e * R + j) * C + k);
            return s;
        });
        CHECK(rel_l2(out.read(), ref) < kTol);
    }
    SECTION("LoRA B forward (TN), scaled and accumulated: out = 0.5 x @ B_e^T + out") {
        Bf16 out(T * D, 1.0f, rng);
        const std::vector<float> init = out.host;
        moe_grouped_gemm(out.dev,
                         x_r.dev,
                         b_dr.dev,
                         off.dev,
                         E,
                         D,
                         R,
                         h.cublas,
                         h.stream,
                         off.host.data(),
                         0.5f,
                         1.0f,
                         EMMTranspose::TN,
                         nullptr,
                         false,
                         -1);
        auto ref = per_token(off, D, [&](int e, int t, int j) {
            double s = 0;
            for (int k = 0; k < R; ++k)
                s += at(x_r.host, (std::size_t)t * R + k) * at(b_dr.host, ((std::size_t)e * D + j) * R + k);
            return 0.5 * s + init[(std::size_t)t * D + j];
        });
        CHECK(rel_l2(out.read(), ref) < kTol);
    }
    SECTION("LoRA dX (NN): out = d @ A_e") {
        Bf16 out(T * C, 0.0f, rng);
        moe_grouped_gemm(out.dev,
                         x_r.dev,
                         a_rc.dev,
                         off.dev,
                         E,
                         C,
                         R,
                         h.cublas,
                         h.stream,
                         off.host.data(),
                         1.0f,
                         0.0f,
                         EMMTranspose::NN,
                         nullptr,
                         false,
                         -1);
        auto ref = per_token(off, C, [&](int e, int t, int j) {
            double s = 0;
            for (int k = 0; k < R; ++k)
                s += at(x_r.host, (std::size_t)t * R + k) * at(a_rc.host, ((std::size_t)e * R + k) * C + j);
            return s;
        });
        CHECK(rel_l2(out.read(), ref) < kTol);
    }
    SECTION("weight gradient, accumulated and overwritten") {
        Bf16 da(E * R * C, 0.01f, rng), db(E * D * R, 0.0f, rng);
        const std::vector<float> init = da.host;
        moe_grouped_gemm_weight_grad(da.dev,
                                     x_r.dev,
                                     x_c.dev,
                                     off.dev,
                                     E,
                                     R,
                                     C,
                                     h.cublas,
                                     h.stream,
                                     off.host.data(),
                                     1.0f,
                                     1.0f,
                                     nullptr,
                                     false,
                                     -1);
        moe_grouped_gemm_weight_grad(db.dev,
                                     x_d.dev,
                                     x_r.dev,
                                     off.dev,
                                     E,
                                     D,
                                     R,
                                     h.cublas,
                                     h.stream,
                                     off.host.data(),
                                     1.0f,
                                     0.0f,
                                     nullptr,
                                     false,
                                     -1);
        std::vector<double> ra(init.begin(), init.end()), rb(E * D * R, 0.0);
        for (int e = 0; e < E; ++e)
            for (int t = off.host[e]; t < off.host[e + 1]; ++t) {
                for (int i = 0; i < R; ++i)
                    for (int j = 0; j < C; ++j)
                        ra[((std::size_t)e * R + i) * C + j] +=
                            at(x_r.host, (std::size_t)t * R + i) * at(x_c.host, (std::size_t)t * C + j);
                for (int i = 0; i < D; ++i)
                    for (int j = 0; j < R; ++j)
                        rb[((std::size_t)e * D + i) * R + j] +=
                            at(x_d.host, (std::size_t)t * D + i) * at(x_r.host, (std::size_t)t * R + j);
            }
        CHECK(rel_l2(da.read(), ra) < kTol);
        CHECK(rel_l2(db.read(), rb) < kTol);
        // Experts without tokens keep their gradient untouched (no GEMM runs for them).
        const auto got = da.read();
        for (int j = 0; j < R * C; ++j)
            REQUIRE(got[(std::size_t)3 * R * C + j] == init[(std::size_t)3 * R * C + j]);
    }
    SECTION("base forward and dX backward") {
        Bf16 gu(T * 2 * D, 0.0f, rng), dn(T * C, 0.0f, rng), dnb(T * D, 0.0f, rng), gub(T * C, 0.0f, rng);
        moe_grouped_gemm_gate_up(gu.dev,
                                 x_c.dev,
                                 w_gu.dev,
                                 off.dev,
                                 E,
                                 C,
                                 D,
                                 h.cublas,
                                 h.stream,
                                 off.host.data(),
                                 nullptr,
                                 false,
                                 -1);
        moe_grouped_gemm_down(dn.dev,
                              x_d.dev,
                              w_dn.dev,
                              off.dev,
                              E,
                              C,
                              D,
                              h.cublas,
                              h.stream,
                              off.host.data(),
                              nullptr,
                              false,
                              -1);
        moe_grouped_gemm_down_backward(dnb.dev,
                                       x_c.dev,
                                       w_dn.dev,
                                       off.dev,
                                       E,
                                       C,
                                       D,
                                       h.cublas,
                                       h.stream,
                                       off.host.data(),
                                       nullptr,
                                       false,
                                       -1);
        moe_grouped_gemm_gate_up_backward(gub.dev,
                                          x_2d.dev,
                                          w_gu.dev,
                                          off.dev,
                                          E,
                                          C,
                                          D,
                                          h.cublas,
                                          h.stream,
                                          off.host.data(),
                                          nullptr,
                                          false,
                                          -1);
        CHECK(rel_l2(gu.read(), per_token(off, 2 * D, [&](int e, int t, int j) {
                         double s = 0;
                         for (int k = 0; k < C; ++k)
                             s += at(x_c.host, (std::size_t)t * C + k) *
                                  at(w_gu.host, ((std::size_t)e * 2 * D + j) * C + k);
                         return s;
                     })) < kTol);
        CHECK(rel_l2(dn.read(), per_token(off, C, [&](int e, int t, int j) {
                         double s = 0;
                         for (int k = 0; k < D; ++k)
                             s +=
                                 at(x_d.host, (std::size_t)t * D + k) * at(w_dn.host, ((std::size_t)e * C + j) * D + k);
                         return s;
                     })) < kTol);
        CHECK(rel_l2(dnb.read(), per_token(off, D, [&](int e, int t, int j) {
                         double s = 0;
                         for (int k = 0; k < C; ++k)
                             s +=
                                 at(x_c.host, (std::size_t)t * C + k) * at(w_dn.host, ((std::size_t)e * C + k) * D + j);
                         return s;
                     })) < kTol);
        CHECK(rel_l2(gub.read(), per_token(off, C, [&](int e, int t, int j) {
                         double s = 0;
                         for (int k = 0; k < 2 * D; ++k)
                             s += at(x_2d.host, (std::size_t)t * 2 * D + k) *
                                  at(w_gu.host, ((std::size_t)e * 2 * D + k) * C + j);
                         return s;
                     })) < kTol);
    }
}

TEST_CASE("A token's MoE GEMM output does not depend on its expert's token count", "[moe][gemm]") {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) SKIP("no CUDA device");
    Handles h;
    std::mt19937 rng(11);
    // Layout 1: every expert holds its own tokens. Layout 2: the same tokens, but each expert has
    // up to 300 extra tokens appended after them (as when a neighbouring packed row routes there).
    const auto t1 = token_counts(3);
    std::vector<int> t2(t1);
    for (int e = 0; e < E; ++e)
        t2[e] += (e * 97) % 301;
    Offsets o1(offsets_of(t1)), o2(offsets_of(t2));
    Bf16 x2(o2.total() * C, 1.0f, rng), w(E * 2 * D * C, 0.05f, rng), a(E * R * C, 0.05f, rng);
    // Layout 1's input: each expert's first t1[e] rows of layout 2.
    std::vector<nv_bfloat16> hx1(static_cast<std::size_t>(o1.total()) * C);
    for (int e = 0; e < E; ++e)
        for (int i = 0; i < t1[e]; ++i)
            for (int k = 0; k < C; ++k)
                hx1[((std::size_t)o1.host[e] + i) * C + k] =
                    __float2bfloat16(x2.host[((std::size_t)o2.host[e] + i) * C + k]);
    nv_bfloat16* x1 = nullptr;
    REQUIRE(cudaMalloc(&x1, hx1.size() * sizeof(nv_bfloat16)) == cudaSuccess);
    REQUIRE(cudaMemcpy(x1, hx1.data(), hx1.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice) == cudaSuccess);

    auto same_rows = [&](const Bf16& y1, const Bf16& y2, int cols) {
        const auto r1 = y1.read(), r2 = y2.read();
        std::size_t differ = 0;
        for (int e = 0; e < E; ++e)
            for (int i = 0; i < t1[e]; ++i)
                for (int j = 0; j < cols; ++j)
                    differ +=
                        r1[((std::size_t)o1.host[e] + i) * cols + j] != r2[((std::size_t)o2.host[e] + i) * cols + j];
        return differ;
    };
    SECTION("LoRA A forward (TN, K = hidden)") {
        Bf16 y1(o1.total() * R, 0.0f, rng), y2(o2.total() * R, 0.0f, rng);
        moe_grouped_gemm(y1.dev,
                         x1,
                         a.dev,
                         o1.dev,
                         E,
                         R,
                         C,
                         h.cublas,
                         h.stream,
                         o1.host.data(),
                         1.0f,
                         0.0f,
                         EMMTranspose::TN,
                         nullptr,
                         false,
                         -1);
        moe_grouped_gemm(y2.dev,
                         x2.dev,
                         a.dev,
                         o2.dev,
                         E,
                         R,
                         C,
                         h.cublas,
                         h.stream,
                         o2.host.data(),
                         1.0f,
                         0.0f,
                         EMMTranspose::TN,
                         nullptr,
                         false,
                         -1);
        CHECK(same_rows(y1, y2, R) == 0);
    }
    SECTION("base gate_up forward") {
        Bf16 y1(o1.total() * 2 * D, 0.0f, rng), y2(o2.total() * 2 * D, 0.0f, rng);
        moe_grouped_gemm_gate_up(y1.dev,
                                 x1,
                                 w.dev,
                                 o1.dev,
                                 E,
                                 C,
                                 D,
                                 h.cublas,
                                 h.stream,
                                 o1.host.data(),
                                 nullptr,
                                 false,
                                 -1);
        moe_grouped_gemm_gate_up(y2.dev,
                                 x2.dev,
                                 w.dev,
                                 o2.dev,
                                 E,
                                 C,
                                 D,
                                 h.cublas,
                                 h.stream,
                                 o2.host.data(),
                                 nullptr,
                                 false,
                                 -1);
        CHECK(same_rows(y1, y2, 2 * D) == 0);
    }
    cudaFree(x1);
}

namespace {
__global__ void gate_kernel(volatile int* flag) {
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    while (*flag == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (t - t0 > 60ull * 1000000000ull) break;  // never wedge the GPU, whatever happens
    }
}
__global__ void filler_kernel() {
}
}  // namespace

TEST_CASE("MoE grouped GEMM holds no lock another GPU's worker waits for", "[moe][gemm][multigpu]") {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 2) SKIP("needs two CUDA devices");
    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point t) {
        return std::chrono::duration<double, std::milli>(clk::now() - t).count();
    };

    // One LoRA-B-shaped problem per device, as the multi-GPU trainer runs one worker thread per GPU.
    struct Gpu {
        int dev;
        cudaStream_t stream{};
        cublasHandle_t cublas{};
        std::vector<int> off;
        int* d_off = nullptr;
        nv_bfloat16 *w = nullptr, *x = nullptr, *y = nullptr;
        void setup(int d) {
            dev = d;
            REQUIRE(cudaSetDevice(dev) == cudaSuccess);
            REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
            REQUIRE(cublasCreate(&cublas) == CUBLAS_STATUS_SUCCESS);
            off.assign(E + 1, 0);
            for (int e = 0; e < E; ++e)
                off[e + 1] = off[e] + 1 + (e * 37) % 64;
            REQUIRE(cudaMalloc(&d_off, off.size() * sizeof(int)) == cudaSuccess);
            REQUIRE(cudaMemcpy(d_off, off.data(), off.size() * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
            REQUIRE(cudaMalloc(&w, sizeof(nv_bfloat16) * E * D * R) == cudaSuccess);
            REQUIRE(cudaMalloc(&x, sizeof(nv_bfloat16) * off.back() * R) == cudaSuccess);
            REQUIRE(cudaMalloc(&y, sizeof(nv_bfloat16) * off.back() * D) == cudaSuccess);
            REQUIRE(cudaMemset(w, 0, sizeof(nv_bfloat16) * E * D * R) == cudaSuccess);
            REQUIRE(cudaMemset(x, 0, sizeof(nv_bfloat16) * off.back() * R) == cudaSuccess);
        }
        void run() {
            cudaSetDevice(dev);
            moe_grouped_gemm(y,
                             x,
                             w,
                             d_off,
                             E,
                             D,
                             R,
                             cublas,
                             stream,
                             off.data(),
                             1.0f,
                             0.0f,
                             EMMTranspose::TN,
                             nullptr,
                             false,
                             -1);
        }
        void teardown() {
            cudaSetDevice(dev);
            cudaStreamSynchronize(stream);
            cudaFree(w);
            cudaFree(x);
            cudaFree(y);
            cudaFree(d_off);
            cublasDestroy(cublas);
            cudaStreamDestroy(stream);
        }
    } a, b;
    a.setup(0);
    b.setup(1);

    int* flag = nullptr;
    REQUIRE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess);
    int* dflag = nullptr;
    REQUIRE(cudaHostGetDevicePointer(&dflag, flag, 0) == cudaSuccess);
    auto set_flag = [&](int v) {
        *reinterpret_cast<volatile int*>(flag) = v;
    };

    // How many launches fit behind a kernel that cannot finish: fill until a launch blocks.
    long capacity = 0;
    {
        set_flag(0);
        std::atomic<long> n{0};
        std::atomic<bool> stop{false};
        std::thread t([&] {
            cudaSetDevice(0);
            gate_kernel<<<1, 1, 0, a.stream>>>(dflag);
            while (!stop.load() && n.load() < 200000) {
                filler_kernel<<<1, 1, 0, a.stream>>>();
                n.fetch_add(1);
            }
        });
        long last = -1;
        auto changed = clk::now();
        while (ms_since(changed) < 500) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (n.load() != last) {
                last = n.load();
                changed = clk::now();
            }
        }
        capacity = n.load();
        stop.store(true);
        set_flag(1);
        t.join();
        REQUIRE(cudaStreamSynchronize(a.stream) == cudaSuccess);
    }

    // One round: worker A queues `capacity - margin` fillers behind the gate, then GEMMs until one
    // blocks inside the path; worker B then runs one GEMM on GPU 1 and opens the gate. Where A blocks
    // (in the staging copy or in the GEMM launch) depends on the margin, so several margins are tried;
    // with cuBLAS grouped GEMM, margins 1, 3 and 5 caught the lock on an RTX 5090.
    auto round = [&](long margin, bool& a_blocked) {
        std::atomic<bool> b_ready{false}, b_go{false}, b_done{false};
        std::thread tb([&] {
            b.run();  // B warms up on its own thread
            cudaStreamSynchronize(b.stream);
            b_ready.store(true);
            while (!b_go.load())
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            b.run();  // B's GEMM on GPU 1 ...
            b_done.store(true);
            set_flag(1);  // ... then B "joins the collective" GPU 0 waits for
        });
        while (!b_ready.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        set_flag(0);
        std::atomic<long> a_calls{0};
        std::atomic<int> a_inside{0};
        std::atomic<bool> a_stop{false};
        std::thread ta([&] {
            a.run();
            cudaStreamSynchronize(a.stream);
            gate_kernel<<<1, 1, 0, a.stream>>>(dflag);
            for (long i = 0; i + margin < capacity; ++i)
                filler_kernel<<<1, 1, 0, a.stream>>>();
            while (!a_stop.load() && a_calls.load() < 100000) {
                a_inside.store(1);
                a.run();
                a_inside.store(0);
                a_calls.fetch_add(1);
            }
        });
        long last = -1;
        auto changed = clk::now(), started = clk::now();
        a_blocked = false;
        while (ms_since(started) < 30000) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (a_calls.load() != last) {
                last = a_calls.load();
                changed = clk::now();
            } else if (a_inside.load() && ms_since(changed) > 1000) {
                a_blocked = true;
                break;
            }
        }
        b_go.store(true);
        const auto b_start = clk::now();
        while (a_blocked && !b_done.load() && ms_since(b_start) < 10000)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const bool cycle = a_blocked && !b_done.load();
        set_flag(1);  // releases everything if B could not
        a_stop.store(true);
        tb.join();
        ta.join();
        cudaSetDevice(0);
        cudaStreamSynchronize(a.stream);
        cudaSetDevice(1);
        cudaStreamSynchronize(b.stream);
        return cycle;
    };
    int blocked_rounds = 0, cycles = 0;
    for (long margin = 1; margin <= 6; ++margin) {
        bool a_blocked = false;
        if (round(margin, a_blocked)) ++cycles;
        blocked_rounds += a_blocked;
    }
    a.teardown();
    b.teardown();
    cudaFreeHost(flag);
    INFO("launches behind the gate: " << capacity << ", rounds with A blocked: " << blocked_rounds);
    REQUIRE(blocked_rounds > 0);  // the setup reached the state it tests
    CHECK(cycles == 0);           // B never waited for A
}

TEST_CASE("The bf16 MoE forward neither waits for its GPU nor holds a lock another GPU's worker needs",
          "[moe][gemm][multigpu]") {
    // #212: cuDNN's grouped matmul, once the bf16 recipe's MoE forward, calls cuKernelSetAttribute on a
    // plan's first execute, and the driver then waits for the GPU to go idle while holding a process-wide
    // library lock. On a GPU waiting in a collective, the first forward for a new routed-token count
    // stalled; a second GPU's worker building its own plan waited for the lock, and neither GPU moved.
    // Here GPU 0 runs a kernel that only finishes when the host says so. Worker A runs the recipe's MoE
    // forward on GPU 0 and worker B on GPU 1, each with a shape it never ran before; both must return
    // while GPU 0 is still busy.
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 2) SKIP("needs two CUDA devices");
    using clk = std::chrono::steady_clock;
    auto ms_since = [](clk::time_point t) {
        return std::chrono::duration<double, std::milli>(clk::now() - t).count();
    };
    auto recipe = recipes::RecipeFactory::create("bf16");

    struct Gpu {
        int dev = 0;
        cudaStream_t stream{};
        cublasHandle_t cublas{};
        cudnnHandle_t cudnn{};
        std::vector<int> off;
        int* d_off = nullptr;
        nv_bfloat16 *w = nullptr, *x = nullptr, *y = nullptr;
        std::byte* ws = nullptr;
        const int n = 1408, k = 2816;  // Gemma 4 26B-A4B gate_up: N = 2 x 704, K = hidden
        void setup(int d, int tokens_per_expert) {
            dev = d;
            REQUIRE(cudaSetDevice(dev) == cudaSuccess);
            REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
            REQUIRE(cublasCreate(&cublas) == CUBLAS_STATUS_SUCCESS);
            cudnn = create_cudnn_handle();  // only the pre-#212 cuDNN forward used it
            off.assign(E + 1, 0);
            for (int e = 0; e < E; ++e)
                off[e + 1] = off[e] + tokens_per_expert;
            REQUIRE(cudaMalloc(&d_off, off.size() * sizeof(int)) == cudaSuccess);
            REQUIRE(cudaMemcpy(d_off, off.data(), off.size() * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
            REQUIRE(cudaMalloc(&w, sizeof(nv_bfloat16) * E * n * k) == cudaSuccess);
            REQUIRE(cudaMalloc(&x, sizeof(nv_bfloat16) * off.back() * k) == cudaSuccess);
            REQUIRE(cudaMalloc(&y, sizeof(nv_bfloat16) * off.back() * n) == cudaSuccess);
            REQUIRE(cudaMemset(w, 0, sizeof(nv_bfloat16) * E * n * k) == cudaSuccess);
            REQUIRE(cudaMemset(x, 0, sizeof(nv_bfloat16) * off.back() * k) == cudaSuccess);
            REQUIRE(cudaMalloc(&ws, 64 << 20) == cudaSuccess);
            REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        }
        void forward(const recipes::Recipe& r) {
            cudaSetDevice(dev);
            modules::MoeMatmulContext ctx;
            ctx.out = y;
            ctx.inp = x;
            ctx.weights = w;
            ctx.expert_offsets = d_off;
            ctx.num_experts = E;
            ctx.N = n;
            ctx.K = k;
            ctx.total_tokens = off.back();
            ctx.cudnn_handle = cudnn;
            ctx.cublas_handle = cublas;
            ctx.workspace = ws;
            ctx.workspace_size = 64 << 20;
            ctx.stream = stream;
            ctx.host_offsets = off.data();
            cublasSetStream(cublas, stream);
            r.forward_moe_matmul(ctx);
        }
        void teardown() {
            cudaSetDevice(dev);
            cudaStreamSynchronize(stream);
            cudaFree(w);
            cudaFree(x);
            cudaFree(y);
            cudaFree(d_off);
            cudaFree(ws);
            destroy_cudnn_handle(cudnn);
            cublasDestroy(cublas);
            cudaStreamDestroy(stream);
        }
    } a, b;
    // Token counts no other test uses, so neither worker has run these shapes before.
    a.setup(0, 37);
    b.setup(1, 41);

    int* flag = nullptr;
    REQUIRE(cudaHostAlloc(&flag, sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess);
    int* dflag = nullptr;
    REQUIRE(cudaHostGetDevicePointer(&dflag, flag, 0) == cudaSuccess);
    *reinterpret_cast<volatile int*>(flag) = 1;
    REQUIRE(cudaSetDevice(0) == cudaSuccess);
    gate_kernel<<<1, 1, 0, a.stream>>>(dflag);  // load the gate kernel while GPU 0 is idle
    REQUIRE(cudaStreamSynchronize(a.stream) == cudaSuccess);
    *reinterpret_cast<volatile int*>(flag) = 0;
    gate_kernel<<<1, 1, 0, a.stream>>>(dflag);  // GPU 0 busy until the flag is set

    std::atomic<bool> a_done{false}, b_done{false};
    std::thread ta([&] {
        a.forward(*recipe);
        a_done.store(true);
    });
    const auto t0 = clk::now();
    while (!a_done.load() && ms_since(t0) < 5000)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const bool a_returned_while_busy = a_done.load();
    std::thread tb([&] {
        b.forward(*recipe);
        b_done.store(true);
    });
    const auto t1 = clk::now();
    while (!b_done.load() && ms_since(t1) < 10000)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    const bool b_returned_while_busy = b_done.load();
    *reinterpret_cast<volatile int*>(flag) = 1;  // releases everything if either could not return
    ta.join();
    tb.join();
    a.teardown();
    b.teardown();
    cudaFreeHost(flag);
    CHECK(a_returned_while_busy);  // the forward did not wait for GPU 0's pending work
    CHECK(b_returned_while_busy);  // and held nothing GPU 1's worker needed
}
