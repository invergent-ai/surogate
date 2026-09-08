#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include "kernels/kernels.h"
#include "utilities/tensor.h"

namespace {
struct DeviceTensor {
    Tensor tensor;
    DeviceTensor(ETensorDType dtype, std::vector<long> shape) {
        long n = 1;
        for (long d : shape)
            n *= d;
        void* ptr = nullptr;
        REQUIRE(cudaMalloc(&ptr, n * get_dtype_size(dtype)) == cudaSuccess);
        tensor = Tensor::from_pointer(static_cast<std::byte*>(ptr), 0, dtype, shape);
        REQUIRE(cudaMemset(ptr, 0, tensor.bytes()) == cudaSuccess);
    }
    ~DeviceTensor() {
        cudaFree(tensor.Data);
    }
    template <class T>
    void put(const std::vector<T>& values) {
        REQUIRE(values.size() * sizeof(T) == tensor.bytes());
        REQUIRE(cudaMemcpy(tensor.Data, values.data(), tensor.bytes(), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    template <class T>
    std::vector<T> get() {
        std::vector<T> out(tensor.nelem());
        REQUIRE(cudaMemcpy(out.data(), tensor.Data, tensor.bytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
        return out;
    }
};
bool have_gpu() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count;
}
}  // namespace

TEST_CASE("Exact GELU forward and derivative", "[spark][kernels]") {
    if (!have_gpu()) SKIP("CUDA device required");
    constexpr int N = 517;
    DeviceTensor x(ETensorDType::FP32, {N}), y(ETensorDType::FP32, {N}), dy(ETensorDType::FP32, {N}),
        dx(ETensorDType::FP32, {N});
    std::vector<float> values(N), upstream(N);
    for (int i = 0; i < N; ++i) {
        values[i] = (i - 258) * 0.025f;
        upstream[i] = std::sin(i * 0.21f);
    }
    x.put(values);
    dy.put(upstream);
    gelu_exact_forward(y.tensor, x.tensor, N, nullptr);
    gelu_exact_backward(dx.tensor, x.tensor, dy.tensor, N, nullptr);
    auto actual = y.get<float>(), grad = dx.get<float>();
    auto gelu = [](double v) {
        return v * 0.5 * (1 + std::erf(v / std::sqrt(2.0)));
    };
    for (int i = 0; i < N; ++i) {
        const double v = values[i], step = 1e-5;
        const double derivative = (gelu(v + step) - gelu(v - step)) / (2 * step);
        REQUIRE(actual[i] == Catch::Approx(gelu(v)).margin(1e-6));
        REQUIRE(grad[i] == Catch::Approx(derivative * upstream[i]).margin(1e-6));
    }
}

TEST_CASE("FP32 residual norm keeps precision and separate gradient dtypes", "[spark][kernels]") {
    if (!have_gpu()) SKIP("CUDA device required");
    constexpr int R = 35, C = 259, N = R * C;
    DeviceTensor rin(ETensorDType::FP32, {R, C}), branch(ETensorDType::BF16, {R, C}), weight(ETensorDType::BF16, {C}),
        residual(ETensorDType::FP32, {R, C}), y(ETensorDType::BF16, {R, C}), rstd(ETensorDType::FP32, {R}),
        dy(ETensorDType::BF16, {R, C}), dn(ETensorDType::FP32, {R, C}), dr(ETensorDType::FP32, {R, C}),
        db(ETensorDType::BF16, {R, C}), dw(ETensorDType::FP32, {C}), partials(ETensorDType::FP32, {(R + 31) / 32, C});
    std::vector<float> rv(N), nv(N);
    std::vector<nv_bfloat16> bv(N), uv(N), wv(C);
    for (int i = 0; i < N; ++i) {
        rv[i] = std::sin(i * .031f) + .000173f;
        bv[i] = nv_bfloat16(std::cos(i * .071f) * .002f);
        uv[i] = nv_bfloat16(std::sin(i * .053f));
        nv[i] = std::cos(i * .029f) * .037f;
    }
    for (int c = 0; c < C; ++c)
        wv[c] = nv_bfloat16(1 + std::sin(c * .043f) * .1f);
    rin.put(rv);
    branch.put(bv);
    weight.put(wv);
    dy.put(uv);
    dn.put(nv);
    fused_residual_rmsnorm_fp32_forward(residual.tensor,
                                        y.tensor,
                                        rstd.tensor,
                                        rin.tensor,
                                        branch.tensor,
                                        weight.tensor,
                                        1e-6f,
                                        R,
                                        C,
                                        nullptr);
    fused_residual_rmsnorm_fp32_backward(dr.tensor,
                                         db.tensor,
                                         &dw.tensor,
                                         partials.tensor,
                                         dy.tensor,
                                         &dn.tensor,
                                         residual.tensor,
                                         weight.tensor,
                                         rstd.tensor,
                                         R,
                                         C,
                                         nullptr);
    auto res = residual.get<float>(), scales = rstd.get<float>(), gr = dr.get<float>(), gw = dw.get<float>();
    auto out = y.get<nv_bfloat16>(), gb = db.get<nv_bfloat16>();
    std::vector<double> expected_dw(C);
    for (int row = 0; row < R; ++row) {
        double square = 0;
        for (int c = 0; c < C; ++c) {
            const int i = row * C + c;
            REQUIRE(res[i] == rv[i] + float(bv[i]));
            square += double(res[i]) * res[i];
        }
        const double scale = 1 / std::sqrt(square / C + 1e-6);
        REQUIRE(scales[row] == Catch::Approx(scale).epsilon(2e-6));
        double mean = 0;
        for (int c = 0; c < C; ++c) {
            const int i = row * C + c;
            mean += double(float(uv[i])) * float(wv[c]) * res[i] * scale / C;
        }
        for (int c = 0; c < C; ++c) {
            const int i = row * C + c;
            const float expected_y = nv_bfloat16(float(res[i] * scale * float(wv[c])));
            const float expected_grad =
                float(scale * (float(uv[i]) * double(float(wv[c])) - res[i] * scale * mean) + nv[i]);
            REQUIRE(float(out[i]) == Catch::Approx(expected_y).margin(.0079));
            REQUIRE(gr[i] == Catch::Approx(expected_grad).margin(2e-6));
            REQUIRE(float(gb[i]) == float(nv_bfloat16(gr[i])));
            expected_dw[c] += double(float(uv[i])) * res[i] * scale;
        }
    }
    for (int c = 0; c < C; ++c)
        REQUIRE(gw[c] == Catch::Approx(expected_dw[c]).margin(1e-5));
    // A second microbatch must accumulate the norm-weight gradient.
    fused_residual_rmsnorm_fp32_backward(dr.tensor,
                                         db.tensor,
                                         &dw.tensor,
                                         partials.tensor,
                                         dy.tensor,
                                         &dn.tensor,
                                         residual.tensor,
                                         weight.tensor,
                                         rstd.tensor,
                                         R,
                                         C,
                                         nullptr);
    gw = dw.get<float>();
    for (int c = 0; c < C; ++c)
        REQUIRE(gw[c] == Catch::Approx(2 * expected_dw[c]).margin(2e-5));
}
