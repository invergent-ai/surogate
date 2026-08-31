// encoder_attention: non-causal GQA over a whole sequence, with a symmetric window.
//
// The op is two batched products around a masked softmax, so the things that can
// be wrong are orientation and masking rather than arithmetic. The cases below
// are chosen for exactly that: a global layer, a narrow window, the real
// EmbeddingGemma geometry at a length that straddles the window, and a window
// wider than the sequence -- which must come out identical to the global case,
// since a mask that admits everything is no mask.

#include "api/ops/encoder_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr ReductionCriterion encoder_attention_bf16_criterion() {
    return {/*relative_l2*/ 4.0e-3, /*gross_absolute*/ 4.0e-3,
            /*gross_relative_to_max_reference*/ 8.0e-3};
}

struct Case {
    const char* name;
    std::int32_t q_heads;
    std::int32_t head_dim;
    std::int32_t tokens;
    std::int32_t window; // 0 = global
};

/// The op's contract, in double: softmax over the admitted band, then P V.
std::vector<double> oracle(const std::vector<float>& q, const std::vector<float>& k,
                           const std::vector<float>& v, const Case& shape, float scale) {
    const std::int64_t out_rows = static_cast<std::int64_t>(shape.q_heads) * shape.head_dim;

    std::vector<double> out(static_cast<std::size_t>(out_rows) * shape.tokens, 0.0);
    std::vector<double> probability(shape.tokens);

    for (std::int32_t head = 0; head < shape.q_heads; ++head) {
        for (std::int32_t query = 0; query < shape.tokens; ++query) {
            const std::int32_t lo =
                shape.window > 0 ? std::max(0, query - shape.window + 1) : 0;
            const std::int32_t hi =
                shape.window > 0 ? std::min(shape.tokens, query + shape.window) : shape.tokens;

            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int32_t key = lo; key < hi; ++key) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < shape.head_dim; ++d) {
                    dot += static_cast<double>(q[query * out_rows + head * shape.head_dim + d]) *
                           static_cast<double>(k[key * shape.head_dim + d]);
                }
                probability[key] = dot * scale;
                maximum          = std::max(maximum, probability[key]);
            }
            double sum = 0.0;
            for (std::int32_t key = lo; key < hi; ++key) {
                probability[key] = std::exp(probability[key] - maximum);
                sum += probability[key];
            }
            for (std::int32_t d = 0; d < shape.head_dim; ++d) {
                double accumulated = 0.0;
                for (std::int32_t key = lo; key < hi; ++key) {
                    accumulated += probability[key] *
                                   static_cast<double>(v[key * shape.head_dim + d]);
                }
                out[static_cast<std::size_t>(query) * out_rows + head * shape.head_dim + d] =
                    accumulated / sum;
            }
        }
    }
    return out;
}

int run(const Case& shape, std::uint32_t seed) {
    const std::int64_t out_rows = static_cast<std::int64_t>(shape.q_heads) * shape.head_dim;
    const auto q_elements       = static_cast<std::size_t>(out_rows) * shape.tokens;
    const auto kv_elements      = static_cast<std::size_t>(shape.head_dim) * shape.tokens;
    const auto out_elements     = q_elements;
    const float scale = 1.0F / std::sqrt(static_cast<float>(shape.head_dim));

    std::vector<float> q(q_elements);
    std::vector<float> k(kv_elements);
    std::vector<float> v(kv_elements);
    fill_uniform(q, seed, -2.0F, 2.0F);
    fill_uniform(k, seed + 101, -2.0F, 2.0F);
    fill_uniform(v, seed + 202, -2.0F, 2.0F);
    round_to_bf16(q); // what the kernel will actually read
    round_to_bf16(k);
    round_to_bf16(v);

    const std::vector<double> reference = oracle(q, k, v, shape, scale);

    DeviceBuffer device_q = to_device_bf16(q);
    DeviceBuffer device_k = to_device_bf16(k);
    DeviceBuffer device_v = to_device_bf16(v);
    DeviceBuffer device_out(out_elements * sizeof(std::uint16_t));
    const std::size_t workspace_bytes =
        ops::encoder_attention_workspace_bytes(shape.q_heads, shape.tokens);
    DeviceBuffer workspace(workspace_bytes);

    Tensor tq(device_q.p, DType::BF16, {out_rows, shape.tokens});
    Tensor tk(device_k.p, DType::BF16, {shape.head_dim, shape.tokens});
    Tensor tv(device_v.p, DType::BF16, {shape.head_dim, shape.tokens});
    Tensor output(device_out.p, DType::BF16, {out_rows, shape.tokens});

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    ops::encoder_attention(tq, tk, tv, shape.window, scale, output, workspace.p, workspace_bytes,
                           stream);
    cuda_synchronize(stream);
    cuda_check(cudaStreamDestroy(stream), "stream destroy");

    const std::vector<double> got = from_device_bf16(device_out, out_elements);
    const std::string label = std::string("encoder_attention ") + shape.name;
    return verify_reduction(label, got, reference, encoder_attention_bf16_criterion());
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "encoder_attention: no CUDA device, skipping\n";
        return 0;
    }
    ops::encoder_attention_prewarm();

    int failures = 0;
    // A global layer: every row sees every column.
    failures += run({"global t=64", 3, 256, 64, 0}, 11);
    // A narrow window, where most of the matrix is masked away.
    failures += run({"window=8 t=64", 3, 256, 64, 8}, 12);
    // EmbeddingGemma's own geometry at a length that straddles the window, so
    // some rows are clipped at both ends and some at neither.
    failures += run({"window=512 t=600", 3, 256, 600, 512}, 13);
    // A window wider than the sequence admits everything, so it must agree with
    // the global case to the last bit of the same inputs.
    failures += run({"window>tokens t=64", 3, 256, 64, 4096}, 11);
    // More than one key head is not this op's shape, but more query heads is.
    failures += run({"8 heads t=128", 8, 128, 128, 0}, 14);

    if (failures != 0) {
        std::cerr << "encoder_attention: " << failures << " case(s) failed\n";
        return 1;
    }
    std::cout << "encoder_attention: all cases passed\n";
    return 0;
}
