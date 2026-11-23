// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// The 8-bit AdamW kernels against plain FP32 Adam on the same gradients. The scale type is read from the kernel's own
// signature, so this file also builds against the kernels from before the FP32 scales: there the [small] cases fail,
// because FP16 scales round to 0 at per-coordinate gradients of 1e-8 to 1e-6 and the group loses its moments.

#include <catch2/catch_test_macros.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "runtime/optimizers/flash_adamw_8bit.h"

namespace {

template <typename S>
S scale_of(void (*)(signed char*, unsigned char*, S*, S*, size_t, cudaStream_t));
using Scale = decltype(scale_of(&optimizers::init_flash_adamw8bit_state));

constexpr float kLr = 2.5e-5f, kBeta1 = 0.9f, kBeta2 = 0.999f, kEps = 1e-8f;
constexpr int kGroup = optimizers::FLASH_ADAMW8BIT_GROUP_SIZE;

template <typename T>
struct Device {
    T* ptr = nullptr;
    size_t n = 0;
    explicit Device(size_t count)
        : n(count) {
        REQUIRE(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)) == cudaSuccess);
    }
    ~Device() {
        cudaFree(ptr);
    }
    Device(const Device&) = delete;
    void put(const std::vector<T>& v) {
        REQUIRE(v.size() == n);
        REQUIRE(cudaMemcpy(ptr, v.data(), n * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    std::vector<T> get() const {
        std::vector<T> v(n);
        REQUIRE(cudaMemcpy(v.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
        return v;
    }
};

double cosine(const std::vector<double>& a, const std::vector<double>& b) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ab += a[i] * b[i];
        aa += a[i] * a[i];
        bb += b[i] * b[i];
    }
    return ab / std::sqrt(aa * bb);
}

double norm_ratio(const std::vector<double>& a, const std::vector<double>& b) {
    double aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        aa += a[i] * a[i];
        bb += b[i] * b[i];
    }
    return std::sqrt(aa / bb);
}

// Gradient size log-uniform in [lo, hi], fixed per 32-element group, or per coordinate when `spread` (up to hi / lo between
// the neighbours of one group); each step that size times (mean + noise).
struct Gradients {
    std::vector<float> size;
    std::mt19937 rng;
    Gradients(size_t n, float lo, float hi, bool spread, unsigned seed)
        : size(n), rng(seed) {
        std::uniform_real_distribution<float> u(std::log(lo), std::log(hi));
        float group_size = 0.f;
        for (size_t i = 0; i < n; ++i) {
            if (spread || i % kGroup == 0)
                group_size = std::exp(u(rng));
            size[i] = group_size;
        }
    }
    std::vector<float> next() {
        std::normal_distribution<float> noise(0.3f, 1.0f);
        std::vector<float> g(size.size());
        for (size_t i = 0; i < g.size(); ++i)
            g[i] = size[i] * noise(rng);
        return g;
    }
};

struct Tolerance {
    double cos_step, ratio_step, moment_rel;
};

// Runs T updates through the kernel (one tensor or three, as a LoRA state does) and through plain FP32 Adam.
void compare_with_fp32_adam(float lo, float hi, const Tolerance& tol, bool multi_tensor, bool spread, int steps = 100) {
    const std::vector<size_t> sizes = {8192 * 4, 8192, 8192 * 2};
    size_t n = 0;
    for (size_t s : sizes)
        n += s;
    const size_t groups = optimizers::flash_adamw8bit_num_scales(n);

    std::vector<float> p_host(n, 0.01f), m_ref(n, 0.f), v_ref(n, 0.f);
    std::vector<double> disp_kernel(n, 0.), disp_ref(n, 0.);
    Device<float> p(n), g(n), one(1);
    Device<signed char> s1(n);
    Device<unsigned char> s2(n);
    Device<Scale> sc1(groups), sc2(groups);
    p.put(p_host);
    one.put({1.f});
    optimizers::init_flash_adamw8bit_state(s1.ptr, s2.ptr, sc1.ptr, sc2.ptr, n, nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    Device<float*> param_ptrs(sizes.size()), grad_ptrs(sizes.size());
    Device<int> dev_sizes(sizes.size()), offsets(sizes.size());
    {
        std::vector<float*> pp, gp;
        std::vector<int> sz, off;
        size_t at = 0;
        for (size_t s : sizes) {
            pp.push_back(p.ptr + at);
            gp.push_back(g.ptr + at);
            sz.push_back(static_cast<int>(s));
            off.push_back(static_cast<int>(at));
            at += s;
        }
        param_ptrs.put(pp);
        grad_ptrs.put(gp);
        dev_sizes.put(sz);
        offsets.put(off);
    }

    Gradients gradients(n, lo, hi, spread, 20261101);
    std::vector<float> before = p_host;
    for (int t = 1; t <= steps; ++t) {
        const std::vector<float> grad = gradients.next();
        g.put(grad);
        if (multi_tensor) {
            optimizers::flash_adamw_update_8bit_multi_tensor(param_ptrs.ptr, grad_ptrs.ptr, dev_sizes.ptr,
                                                             static_cast<int>(sizes.size()), s1.ptr, s2.ptr, sc1.ptr, sc2.ptr,
                                                             offsets.ptr, n, kLr, kBeta1, kBeta2, t, kEps, 0.f, one.ptr, nullptr,
                                                             nullptr, nullptr);
        } else {
            optimizers::flash_adamw_update_8bit(p.ptr, g.ptr, s1.ptr, s2.ptr, sc1.ptr, sc2.ptr, n, kLr, kBeta1, kBeta2, t, kEps,
                                                0.f, one.ptr, nullptr, nullptr, nullptr);
        }
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

        const float c1 = 1.f - std::pow(kBeta1, static_cast<float>(t)), c2 = 1.f - std::pow(kBeta2, static_cast<float>(t));
        std::vector<double> u_ref(n), u_kernel(n);
        const std::vector<float> after = p.get();
        for (size_t i = 0; i < n; ++i) {
            m_ref[i] = kBeta1 * m_ref[i] + (1.f - kBeta1) * grad[i];
            v_ref[i] = kBeta2 * v_ref[i] + (1.f - kBeta2) * grad[i] * grad[i];
            u_ref[i] = kLr * (m_ref[i] / c1) / (std::sqrt(v_ref[i] / c2) + kEps);
            u_kernel[i] = static_cast<double>(before[i]) - static_cast<double>(after[i]);
            disp_ref[i] += u_ref[i];
            disp_kernel[i] += u_kernel[i];
        }
        before = after;

        const std::vector<Scale> scales1 = sc1.get(), scales2 = sc2.get();
        const std::vector<signed char> m8 = s1.get();
        const std::vector<unsigned char> v8 = s2.get();
        size_t zero1 = 0, zero2 = 0;
        for (size_t k = 0; k < groups; ++k) {
            zero1 += static_cast<float>(scales1[k]) == 0.f;
            zero2 += static_cast<float>(scales2[k]) == 0.f;
        }
        double m_err = 0, m_all = 0, v_err = 0, v_all = 0;
        for (size_t i = 0; i < n; ++i) {
            const float y = static_cast<float>(m8[i]) / 127.f;
            const float dec_m = y / (2.f - std::fabs(y)) * static_cast<float>(scales1[i / kGroup]);
            const float dec_v = static_cast<float>(v8[i]) / 255.f * static_cast<float>(scales2[i / kGroup]);
            const float ref_v = std::sqrt(v_ref[i]);
            m_err += (dec_m - m_ref[i]) * (dec_m - m_ref[i]);
            m_all += m_ref[i] * m_ref[i];
            v_err += (dec_v - ref_v) * (dec_v - ref_v);
            v_all += ref_v * ref_v;
        }
        const double cos_u = cosine(u_kernel, u_ref), ratio_u = norm_ratio(u_kernel, u_ref);
        const double m_rel = std::sqrt(m_err / m_all), v_rel = std::sqrt(v_err / v_all);
        if (t == 1 || t == 2 || t == 5 || t == 10 || t == 20 || t == 50 || t == steps) {
            std::printf("  [%g, %g] %s %s t %3d: cos(u, ref) %.5f, ||u|| / ||ref|| %.4f, m error %.4f, sqrt v error %.4f, zero scales %zu, %zu of %zu\n",
                        lo, hi, spread ? "per coordinate" : "per group", multi_tensor ? "multi " : "single", t, cos_u, ratio_u, m_rel, v_rel,
                        zero1, zero2, groups);
        }
        INFO("step " << t << ": cos " << cos_u << ", ratio " << ratio_u << ", m error " << m_rel << ", sqrt v error " << v_rel
                     << ", zero scales " << zero1 << ", " << zero2);
        CHECK(zero1 == 0);
        CHECK(zero2 == 0);
        CHECK(cos_u >= tol.cos_step);
        CHECK(std::fabs(ratio_u - 1.0) <= tol.ratio_step);
        CHECK(m_rel <= tol.moment_rel);
        CHECK(v_rel <= tol.moment_rel);
    }
    const double cos_d = cosine(disp_kernel, disp_ref), ratio_d = norm_ratio(disp_kernel, disp_ref);
    std::printf("  [%g, %g] %s %s after %d steps: cos(displacement, ref) %.5f, ratio %.4f\n", lo, hi, spread ? "per coordinate" : "per group",
                multi_tensor ? "multi " : "single", steps, cos_d, ratio_d);
    CHECK(cos_d >= tol.cos_step);
    CHECK(std::fabs(ratio_d - 1.0) <= tol.ratio_step);
}

// Gradients whose size belongs to the group: every update within cos 0.999 and 1 % in norm of plain FP32 Adam, each moment
// within 5 % rms.
constexpr Tolerance kTight{0.999, 0.01, 0.05};
// Up to 100x between the neighbours of one group, the 8-bit moments' limit: a group encodes its small coordinates in few
// levels of its largest one's scale (sqrt v linear, 255 levels), so this holds them to cos 0.98 and 4 % in norm.
constexpr Tolerance kSpread{0.98, 0.04, 0.05};

// Raw files of real gradients (optional, see the [real] case). Groups of 32, flat, little endian.
template <typename T>
std::vector<T> read_raw(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    REQUIRE(in.good());
    const size_t bytes = static_cast<size_t>(in.tellg());
    std::vector<T> v(bytes / sizeof(T));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
    return v;
}

// One sampled LoRA tensor set: dot products weighted by each sampled group's weight, on the groups of `mask`.
struct Weighted {
    const std::vector<float>& weight;
    const std::vector<unsigned char>& mask;
    double dot(const std::vector<double>& a, const std::vector<double>& b) const {
        double out = 0;
        for (size_t k = 0; k < weight.size(); ++k) {
            if (!mask[k])
                continue;
            double e = 0;
            for (int j = 0; j < kGroup; ++j)
                e += a[k * kGroup + j] * b[k * kGroup + j];
            out += e * weight[k];
        }
        return out;
    }
    double norm(const std::vector<double>& a) const {
        return std::sqrt(dot(a, a));
    }
    double cos(const std::vector<double>& a, const std::vector<double>& b) const {
        return dot(a, b) / (norm(a) * norm(b));
    }
};

}  // namespace

TEST_CASE("flash AdamW 8-bit follows FP32 Adam at small gradients, LoRA path", "[optim][flash][small]") {
    compare_with_fp32_adam(1e-8f, 1e-6f, kTight, /*multi_tensor=*/true, /*spread=*/false);
    compare_with_fp32_adam(1e-8f, 1e-6f, kSpread, /*multi_tensor=*/true, /*spread=*/true);
}

TEST_CASE("flash AdamW 8-bit follows FP32 Adam at small gradients, full-model path", "[optim][flash][small]") {
    compare_with_fp32_adam(1e-8f, 1e-6f, kTight, /*multi_tensor=*/false, /*spread=*/false);
    compare_with_fp32_adam(1e-8f, 1e-6f, kSpread, /*multi_tensor=*/false, /*spread=*/true);
}

TEST_CASE("flash AdamW 8-bit follows FP32 Adam at large gradients", "[optim][flash][large]") {
    compare_with_fp32_adam(1e-3f, 1e-1f, kTight, /*multi_tensor=*/true, /*spread=*/false);
    compare_with_fp32_adam(1e-3f, 1e-1f, kTight, /*multi_tensor=*/false, /*spread=*/false);
    compare_with_fp32_adam(1e-3f, 1e-1f, kSpread, /*multi_tensor=*/true, /*spread=*/true);
    compare_with_fp32_adam(1e-3f, 1e-1f, kSpread, /*multi_tensor=*/false, /*spread=*/true);
}

// Real LoRA gradients through the kernel, against plain FP32 Adam on the same sequence: each step sums the base gradients,
// each with a random sign. FLASH_ADAMW_REAL_DIR holds meta.txt (groups, base gradients, gradient scale), grads.f32 (bases one
// after the other), weight.f32 and is_b.u8 (per group: how many groups it stands for, lora_B or not). Tolerance on lora_B for
// the FP32 scales: as kTight. FLASH_ADAMW_REPORT_ONLY=1 only prints, for the kernels from before them.
TEST_CASE("flash AdamW 8-bit on real LoRA gradients", "[.][optim][flash][real]") {
    const char* dir_env = std::getenv("FLASH_ADAMW_REAL_DIR");
    if (dir_env == nullptr)
        SKIP("FLASH_ADAMW_REAL_DIR is not set");
    const std::string dir(dir_env);
    const bool report_only = std::getenv("FLASH_ADAMW_REPORT_ONLY") != nullptr;
    size_t groups = 0, bases = 0;
    float gscale = 1.f;
    {
        std::ifstream meta(dir + "/meta.txt");
        meta >> groups >> bases >> gscale;
        REQUIRE(meta.good());
    }
    const size_t n = groups * kGroup;
    const std::vector<float> all = read_raw<float>(dir + "/grads.f32"), weight = read_raw<float>(dir + "/weight.f32");
    const std::vector<unsigned char> is_b = read_raw<unsigned char>(dir + "/is_b.u8");
    REQUIRE(all.size() == n * bases);
    REQUIRE(weight.size() == groups);
    REQUIRE(is_b.size() == groups);
    const Weighted on_b{weight, is_b};

    const std::vector<float> start(n, 0.f);
    std::vector<float> m_ref(n, 0.f), v_ref(n, 0.f);
    std::vector<double> disp_kernel(n, 0.), disp_ref(n, 0.);
    Device<float> p(n), g(n), one(1);
    Device<signed char> s1(n);
    Device<unsigned char> s2(n);
    Device<Scale> sc1(groups), sc2(groups);
    p.put(start);
    one.put({1.f});
    optimizers::init_flash_adamw8bit_state(s1.ptr, s2.ptr, sc1.ptr, sc2.ptr, n, nullptr);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    auto kernel_step = [&](int t, const std::vector<float>& grad) {
        g.put(grad);
        optimizers::flash_adamw_update_8bit(p.ptr, g.ptr, s1.ptr, s2.ptr, sc1.ptr, sc2.ptr, n, kLr, kBeta1, kBeta2, t, kEps, 0.f,
                                            one.ptr, nullptr, nullptr, nullptr);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    };
    std::mt19937 rng(20261101);
    auto draw = [&] {
        std::vector<float> grad(n, 0.f);
        for (size_t b = 0; b < bases; ++b) {
            const float sign = (rng() & 1) ? gscale : -gscale;
            for (size_t i = 0; i < n; ++i)
                grad[i] += sign * all[b * n + i];
        }
        return grad;
    };
    auto as_double = [](const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); };

    std::vector<float> before = start;
    std::printf("  t | cos(u, ref) on B | ||u|| / ||ref|| on B | ||u|| in lr | ||ref|| in lr | cos(u, g) | ref cos(u, g) | cos(disp, ref) | ||disp|| / ||ref||\n");
    for (int t = 1; t <= 100; ++t) {
        const std::vector<float> grad = draw();
        kernel_step(t, grad);
        const float c1 = 1.f - std::pow(kBeta1, static_cast<float>(t)), c2 = 1.f - std::pow(kBeta2, static_cast<float>(t));
        const std::vector<float> after = p.get();
        std::vector<double> u_ref(n), u_kernel(n);
        for (size_t i = 0; i < n; ++i) {
            m_ref[i] = kBeta1 * m_ref[i] + (1.f - kBeta1) * grad[i];
            v_ref[i] = kBeta2 * v_ref[i] + (1.f - kBeta2) * grad[i] * grad[i];
            u_ref[i] = kLr * (m_ref[i] / c1) / (std::sqrt(v_ref[i] / c2) + kEps);
            u_kernel[i] = static_cast<double>(before[i]) - static_cast<double>(after[i]);
            disp_ref[i] += u_ref[i];
            disp_kernel[i] += u_kernel[i];
        }
        before = after;
        const double cos_u = on_b.cos(u_kernel, u_ref), ratio_u = on_b.norm(u_kernel) / on_b.norm(u_ref);
        const double cos_d = on_b.cos(disp_kernel, disp_ref), ratio_d = on_b.norm(disp_kernel) / on_b.norm(disp_ref);
        if (t == 1 || t == 2 || t == 5 || t == 10 || t == 20 || t == 30 || t == 50 || t == 100) {
            const std::vector<double> gd = as_double(grad);
            std::printf("%4d | %.4f | %.3f | %.0f | %.0f | %.3f | %.3f | %.4f | %.3f\n", t, cos_u, ratio_u, on_b.norm(u_kernel) / kLr,
                        on_b.norm(u_ref) / kLr, on_b.cos(u_kernel, gd), on_b.cos(u_ref, gd), cos_d, ratio_d);
        }
        if (!report_only) {
            INFO("step " << t);
            CHECK(cos_u >= kTight.cos_step);
            CHECK(std::fabs(ratio_u - 1.0) <= kTight.ratio_step);
            CHECK(cos_d >= kTight.cos_step);
            CHECK(std::fabs(ratio_d - 1.0) <= kTight.ratio_step);
        }
        if (t == 20 || t == 100) {
            // How the next step answers to the size of its gradient, from this state; the state and the weights stay as they are.
            const std::vector<float> snap_p = p.get(), probe = draw();
            const std::vector<signed char> snap_s1 = s1.get();
            const std::vector<unsigned char> snap_s2 = s2.get();
            const std::vector<Scale> snap_sc1 = sc1.get(), snap_sc2 = sc2.get();
            double unit = 0;
            for (const float c : {1.f, 0.25f, 0.5f, 1.5f, 2.f, 4.f}) {
                std::vector<float> scaled(probe);
                for (auto& x : scaled)
                    x *= c;
                kernel_step(t + 1, scaled);
                const std::vector<float> stepped = p.get();
                std::vector<double> u(n);
                for (size_t i = 0; i < n; ++i)
                    u[i] = static_cast<double>(snap_p[i]) - static_cast<double>(stepped[i]);
                if (c == 1.f)
                    unit = on_b.norm(u);
                std::printf("    from t %d, next gradient x %.2f: step x %.3f, cos(u, g) %.3f\n", t, c, on_b.norm(u) / unit,
                            on_b.cos(u, as_double(scaled)));
                p.put(snap_p);
                s1.put(snap_s1);
                s2.put(snap_s2);
                sc1.put(snap_sc1);
                sc2.put(snap_sc2);
            }
        }
    }
}
