// mean_pool: out[h] = mean over the token axis of a [hidden, tokens] matrix.
//
// Two properties beyond the arithmetic. `count` may be fewer than the columns
// present, so a padded slot pools only its own tokens; and `accumulate` adds the
// chunk's *sum*, so a prompt spanning several prefill chunks pools across calls
// with one divide at the end. The chunked case is checked against the same
// oracle as the single-shot one, since they must agree exactly in intent.

#include "api/ops/mean_pool.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr ReductionCriterion mean_pool_bf16_criterion() {
    // FP32 output, so BF16 input rounding is the only floor left.
    return {/*relative_l2*/ 2.0e-4, /*gross_absolute*/ 2.0e-4,
            /*gross_relative_to_max_reference*/ 4.0e-4};
}

std::vector<double> oracle(const std::vector<float>& x, std::int32_t hidden, std::int32_t count) {
    std::vector<double> out(static_cast<std::size_t>(hidden), 0.0);
    for (std::int32_t t = 0; t < count; ++t) {
        for (std::int32_t h = 0; h < hidden; ++h) {
            out[h] += static_cast<double>(x[static_cast<std::size_t>(t) * hidden + h]);
        }
    }
    for (double& value : out) { value /= static_cast<double>(count); }
    return out;
}

int run(std::int32_t hidden, std::int32_t tokens, std::int32_t count, std::uint32_t seed) {
    const auto n = static_cast<std::size_t>(hidden) * tokens;
    std::vector<float> x(n);
    fill_uniform(x, seed, -3.0F, 3.0F);
    round_to_bf16(x);

    const std::vector<double> reference = oracle(x, hidden, count);

    DeviceBuffer dx = to_device_bf16(x);
    DeviceBuffer dout(static_cast<std::size_t>(hidden) * sizeof(float));
    cuda_check(cudaMemset(dout.p, 0, static_cast<std::size_t>(hidden) * sizeof(float)),
               "memset");
    Tensor tx(dx.p, DType::BF16, {hidden, tokens});
    Tensor to(dout.p, DType::FP32, {hidden});

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    ops::mean_pool(tx, count, /*accumulate*/ false, to, stream);
    cuda_synchronize(stream);
    cuda_check(cudaStreamDestroy(stream), "stream destroy");

    const std::string label = "mean_pool hidden=" + std::to_string(hidden) + " count=" +
                              std::to_string(count) + "/" + std::to_string(tokens);
    return verify_reduction(label, from_device_f32(dout, static_cast<std::size_t>(hidden)),
                            reference, mean_pool_bf16_criterion());
}

/// Pool the same columns in two calls and divide once; must match one call.
int run_chunked(std::int32_t hidden, std::int32_t tokens, std::uint32_t seed) {
    const auto n = static_cast<std::size_t>(hidden) * tokens;
    std::vector<float> x(n);
    fill_uniform(x, seed, -3.0F, 3.0F);
    round_to_bf16(x);
    const std::vector<double> reference = oracle(x, hidden, tokens);

    DeviceBuffer dx = to_device_bf16(x);
    DeviceBuffer dout(static_cast<std::size_t>(hidden) * sizeof(float));
    cuda_check(cudaMemset(dout.p, 0, static_cast<std::size_t>(hidden) * sizeof(float)),
               "memset");
    Tensor to(dout.p, DType::FP32, {hidden});

    const std::int32_t split = tokens / 2;
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    {
        Tensor first(dx.p, DType::BF16, {hidden, split});
        ops::mean_pool(first, split, /*accumulate*/ true, to, stream);
        auto* second_data =
            static_cast<std::uint16_t*>(dx.p) + static_cast<std::size_t>(split) * hidden;
        Tensor second(second_data, DType::BF16, {hidden, tokens - split});
        ops::mean_pool(second, tokens - split, /*accumulate*/ true, to, stream);
    }
    cuda_synchronize(stream);
    cuda_check(cudaStreamDestroy(stream), "stream destroy");

    std::vector<double> got = from_device_f32(dout, static_cast<std::size_t>(hidden));
    for (double& value : got) { value /= static_cast<double>(tokens); }
    return verify_reduction("mean_pool chunked hidden=" + std::to_string(hidden), got, reference,
                            mean_pool_bf16_criterion());
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "mean_pool: no CUDA device, skipping\n";
        return 0;
    }
    int failures = 0;
    failures += run(768, 512, 512, 31);  // EmbeddingGemma's width, full sequence
    failures += run(768, 512, 137, 32);  // a padded slot pooling only its own tokens
    failures += run(768, 1, 1, 33);      // one token
    failures += run(64, 2048, 2048, 34); // long sequence, where FP32 accumulation earns its keep
    failures += run_chunked(768, 512, 35);
    if (failures != 0) {
        std::cerr << "mean_pool: " << failures << " case(s) failed\n";
        return 1;
    }
    std::cout << "mean_pool: all cases passed\n";
    return 0;
}
