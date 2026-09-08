#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cmath>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/column_scale.h"
#include "runtime/lora/lora_model_utils.h"
#include "utilities/tensor.h"

namespace {
struct DeviceTensor {
    Tensor tensor;
    DeviceTensor(ETensorDType dtype, std::vector<long> shape) {
        long count = 1;
        for (long n : shape) count *= n;
        void* data = nullptr;
        REQUIRE(cudaMalloc(&data, count * get_dtype_size(dtype)) == cudaSuccess);
        tensor = Tensor::from_pointer(static_cast<std::byte*>(data), 0, dtype, shape);
        REQUIRE(cudaMemset(data, 0, tensor.bytes()) == cudaSuccess);
    }
    ~DeviceTensor() { cudaFree(tensor.Data); }
    template <typename T> void put(const std::vector<T>& values) {
        REQUIRE(values.size() * sizeof(T) == tensor.bytes());
        REQUIRE(cudaMemcpy(tensor.Data, values.data(), tensor.bytes(), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    template <typename T> std::vector<T> get() {
        std::vector<T> result(tensor.nelem());
        REQUIRE(cudaMemcpy(result.data(), tensor.Data, tensor.bytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
        return result;
    }
};
bool have_gpu() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
}

TEST_CASE("Random initialization preserves storage after a scalar or odd-length parameter", "[shared-generation][kernels]") {
    if (!have_gpu()) SKIP("CUDA device required");
    for (int count : {0, 1, 3, 5, 13}) {
        DeviceTensor data(ETensorDType::BF16, {count + 8});
        data.put(std::vector<nv_bfloat16>(count + 8, nv_bfloat16(42.f)));
        fill_normal(data.tensor, count, 0.f, .02f, 42, 0, nullptr);
        const auto values = data.get<nv_bfloat16>();
        for (int i = 0; i < count; ++i) REQUIRE(std::isfinite(float(values[i])));
        for (int i = count; i < values.size(); ++i) REQUIRE(float(values[i]) == 42.f);
    }
}

TEST_CASE("Per-channel scales and small embedding tables support arbitrary widths", "[shared-generation][kernels]") {
    if (!have_gpu()) SKIP("CUDA device required");
    for (int C : {1, 3, 32, 128}) {
        constexpr int N = 13, V = 7;
        INFO("channels=" << C);
        DeviceTensor data(ETensorDType::BF16, {N, C}), scale(ETensorDType::BF16, {C});
        DeviceTensor out(ETensorDType::BF16, {N, C}), grad(ETensorDType::BF16, {C});
        std::vector<nv_bfloat16> x(N * C), s(C);
        for (int i = 0; i < x.size(); ++i) x[i] = static_cast<nv_bfloat16>((i % 17 - 8) * .0625f);
        for (int i = 0; i < C; ++i) s[i] = static_cast<nv_bfloat16>((i % 5 + 1) * .25f);
        data.put(x); scale.put(s);
        column_scale(out.tensor, data.tensor, scale.tensor, N, C, nullptr);
        const auto y = out.get<nv_bfloat16>();
        for (int i = 0; i < y.size(); ++i) REQUIRE(float(y[i]) == float(nv_bfloat16(float(x[i]) * float(s[i % C]))));
        column_scale_gradient(grad.tensor, data.tensor, out.tensor, N, C, nullptr);
        const auto ds = grad.get<nv_bfloat16>();
        for (int c = 0; c < C; ++c) {
            float expected = 0;
            for (int row = 0; row < N; ++row) expected += float(x[row * C + c]) * float(y[row * C + c]);
            REQUIRE(float(ds[c]) == float(nv_bfloat16(expected)));
        }
        DeviceTensor table(ETensorDType::BF16, {V, C}), indices(ETensorDType::INT32, {N});
        std::vector<nv_bfloat16> weights(V * C);
        std::vector<int> ids(N);
        for (int i = 0; i < weights.size(); ++i) weights[i] = static_cast<nv_bfloat16>(i * .03125f);
        for (int i = 0; i < N; ++i) ids[i] = (i * 3) % V;
        table.put(weights); indices.put(ids);
        encoder_forward(out.tensor, indices.tensor, table.tensor, std::nullopt, 1, N, C, V, nullptr);
        const auto embedded = out.get<nv_bfloat16>();
        for (int i = 0; i < N * C; ++i) REQUIRE(float(embedded[i]) == float(weights[ids[i / C] * C + i % C]));
    }
}

TEST_CASE("Router bias gradients accumulate with narrow and odd channel counts", "[shared-generation][kernels]") {
    if (!have_gpu()) SKIP("CUDA device required");
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
    for (int C : {1, 3, 4, 7, 32, 64}) {
        INFO("channels=" << C);
        constexpr int N = 17;
        DeviceTensor input(ETensorDType::BF16, {N, C}), bias(ETensorDType::BF16, {C});
        DeviceTensor scratch(ETensorDType::FP32, {get_bias_backward_scratch_size(ETensorDType::BF16, C, prop) / 4});
        std::vector<nv_bfloat16> x(N * C), initial(C, nv_bfloat16(.5f));
        for (int i = 0; i < x.size(); ++i) x[i] = static_cast<nv_bfloat16>((i % 19 - 9) * .0625f);
        input.put(x); bias.put(initial);
        backward_bias(bias.tensor, input.tensor, nullptr, nullptr, scratch.tensor, 1, N, C, prop, nullptr);
        const auto actual = bias.get<nv_bfloat16>();
        for (int c = 0; c < C; ++c) {
            float sum = .5f;
            for (int row = 0; row < N; ++row) sum += float(x[row * C + c]);
            REQUIRE(float(actual[c]) == float(nv_bfloat16(sum)));
        }
    }
}

TEST_CASE("MRoPE rotates the prefix and preserves its tail in forward and backward", "[qwen35][kernels]") {
    if (!have_gpu()) SKIP("CUDA device required");
    constexpr int B = 2, T = 7, HQ = 2, HK = 1, HEADS = HQ + 2 * HK, D = 64, POS = 32;
    for (int rotary : {16, 64}) {
        for (int planes : {1, 3}) {
            for (bool inplace : {false, true}) {
                INFO("rotary=" << rotary << " planes=" << planes << " inplace=" << inplace);
                DeviceTensor input(ETensorDType::FP32, {B, T, HEADS, D});
                DeviceTensor output(ETensorDType::FP32, {B, T, HEADS, D});
                DeviceTensor freq(ETensorDType::FP32, {POS, rotary});
                DeviceTensor position(ETensorDType::INT32, {planes, B, T});
                std::vector<float> x(input.tensor.nelem()), f(freq.tensor.nelem());
                std::vector<int> p(planes * B * T);
                for (int i = 0; i < x.size(); ++i) x[i] = std::sin(i * .173f);
                for (int i = 0; i < p.size(); ++i) p[i] = (i * 3 + i / T) % POS;
                for (int t = 0; t < POS; ++t) {
                    for (int d = 0; d < rotary / 2; ++d) {
                        const float angle = t * std::pow(10000.f, -2.f * d / rotary);
                        f[t * rotary + 2 * d] = std::cos(angle);
                        f[t * rotary + 2 * d + 1] = std::sin(angle);
                    }
                }
                freq.put(f); position.put(p);
                for (bool backward : {false, true}) {
                    INFO("backward=" << backward);
                    input.put(x);
                    Tensor& dst = inplace ? input.tensor : output.tensor;
                    auto kernel = backward ? mrope_backward : mrope_forward;
                    kernel(dst, input.tensor, freq.tensor, position.tensor.get<int>(), planes,
                           3, 2, 2, nullptr, B, T, HQ, HK, D, rotary, nullptr);
                    const auto actual = inplace ? input.get<float>() : output.get<float>();
                    for (int bt = 0; bt < B * T; ++bt) {
                        for (int h = 0; h < HEADS; ++h) {
                            const int base = (bt * HEADS + h) * D;
                            for (int j = 0; j < D; ++j) {
                                double expected = x[base + j];
                                if (h < HQ + HK && j < rotary) {
                                    const int d = j % (rotary / 2);
                                    int axis = 0;
                                    if (planes == 3 && d < 6 && d % 3 != 0) axis = d % 3;
                                    const int fi = p[axis * B * T + bt] * rotary + 2 * d;
                                    const double cosine = f[fi], sine = f[fi + 1] * (backward ? -1 : 1);
                                    const double real = x[base + d], imag = x[base + d + rotary / 2];
                                    expected = j < rotary / 2 ? real * cosine - imag * sine
                                                               : real * sine + imag * cosine;
                                }
                                REQUIRE(actual[base + j] == Catch::Approx(expected).margin(2e-7));
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("LoRA preserves alpha over rank for packed and sliced projections", "[qwen35][lora][kernels]") {
    if (!have_gpu()) SKIP("CUDA device required");
    constexpr int BT = 3, IN = 64, RANK = 8;
    cublasLtHandle_t handle;
    REQUIRE(cublasLtCreate(&handle) == CUBLAS_STATUS_SUCCESS);
    DeviceTensor workspace(ETensorDType::BYTE, {32 << 20});
    for (int out : {1024, 3584}) {
        for (bool fused : {false, true}) {
            for (float scale : {.5f, 2.f}) {
                INFO("out=" << out << " fused=" << fused << " scale=" << scale);
                const int total = fused ? 2 * out : out;
                const int offset = fused ? out : 0;
                DeviceTensor input(ETensorDType::BF16, {BT, IN}), a(ETensorDType::BF16, {RANK, IN}),
                    b(ETensorDType::BF16, {out, RANK}), output(ETensorDType::BF16, {BT, total}),
                    intermediate(ETensorDType::BF16, {BT, RANK}), scratch(ETensorDType::BF16, {BT, total});
                std::vector<nv_bfloat16> xv(BT * IN), av(RANK * IN), bv(out * RANK), yv(BT * total);
                for (int i = 0; i < xv.size(); ++i) xv[i] = nv_bfloat16(std::sin(i * .13f) * .2f);
                for (int i = 0; i < av.size(); ++i) av[i] = nv_bfloat16(std::cos(i * .03f) * .1f);
                for (int i = 0; i < bv.size(); ++i) bv[i] = nv_bfloat16(std::sin(i * .09f) * .3f);
                for (int i = 0; i < yv.size(); ++i) yv[i] = nv_bfloat16(.125f);
                input.put(xv); a.put(av); b.put(bv); output.put(yv);
                modules::LoRALayerWeights<Tensor> weights;
                weights.A = a.tensor; weights.B = b.tensor;
                modules::detail::apply_lora_contribution(output.tensor, offset, input.tensor, weights,
                    intermediate.tensor, scratch.tensor, scale, 0.f, 0, false, BT, IN, out, RANK,
                    handle, workspace.tensor, nullptr);
                const auto actual = output.get<nv_bfloat16>();
                for (int t = 0; t < BT; ++t) {
                    float low[RANK];
                    for (int r = 0; r < RANK; ++r) {
                        double sum = 0;
                        for (int i = 0; i < IN; ++i) sum += float(xv[t * IN + i]) * float(av[r * IN + i]);
                        low[r] = float(nv_bfloat16(sum));
                    }
                    for (int j = 0; j < total; ++j) {
                        double expected = .125;
                        if (j >= offset && j < offset + out) {
                            double delta = 0;
                            for (int r = 0; r < RANK; ++r) delta += low[r] * float(bv[(j - offset) * RANK + r]);
                            expected += scale * delta;
                        }
                        REQUIRE(float(actual[t * total + j]) == Catch::Approx(expected).margin(.002));
                    }
                }
            }
        }
    }
    REQUIRE(cublasLtDestroy(handle) == CUBLAS_STATUS_SUCCESS);
}
