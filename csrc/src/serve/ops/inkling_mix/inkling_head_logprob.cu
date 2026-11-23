// The trainer's log-prob of a token through the inkling2 head (the engine-op trainer's IR): the bits every
// log-prob the engine returns carries (api/ops/inkling_mix.h).

#include "api/ops/inkling_mix.h"
#include "api/ops/sampled_logprob.h"
#include "api/ops/sampling.h"
#include "api/types.h"

#include "core/device.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace sinfer::ops {
namespace {

constexpr std::size_t kAlign = 256;

std::size_t up(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

__global__ void selected_kernel(const RawTokenScores* scores, const std::int32_t* tokens, int count, int domain,
                                float* out) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count) { return; }
    const int token = tokens[t];
    out[t]          = token >= 0 && token < domain ? scores[t].selected.logprob : NAN;
}

} // namespace

std::size_t inkling_head_logprob_workspace_bytes(std::int32_t vocab_rows, std::int32_t tokens) noexcept {
    if (vocab_rows <= 0 || tokens <= 0) { return 0; }
    return up(static_cast<std::size_t>(vocab_rows) * tokens * sizeof(float)) +
           up(static_cast<std::size_t>(tokens) * sizeof(RawTokenScores)) + up(sizeof(SamplingConfig));
}

void inkling_head_logprob(const void* head_weight, std::int32_t vocab_rows, std::int32_t hidden_size,
                          std::int32_t token_domain, float softcap, const void* hidden, std::int32_t tokens,
                          const std::int32_t* token_ids, float* out, void* workspace, std::size_t workspace_bytes,
                          cudaStream_t stream) {
    if (tokens <= 0) { return; }
    if (head_weight == nullptr || hidden == nullptr || token_ids == nullptr || out == nullptr || vocab_rows <= 0 ||
        token_domain <= 0 || token_domain > vocab_rows || hidden_size <= 0 || workspace == nullptr ||
        workspace_bytes < inkling_head_logprob_workspace_bytes(vocab_rows, tokens) ||
        (reinterpret_cast<std::uintptr_t>(workspace) & (kAlign - 1)) != 0) {
        throw std::invalid_argument("inkling_head_logprob: weights, hidden, ids, out and a 256-byte aligned workspace of "
                                    "inkling_head_logprob_workspace_bytes are required; 0 < domain <= rows");
    }
    auto* base     = static_cast<std::byte*>(workspace);
    auto* logits   = reinterpret_cast<float*>(base);
    auto* scores   = reinterpret_cast<RawTokenScores*>(base + up(static_cast<std::size_t>(vocab_rows) * tokens * sizeof(float)));
    auto* config   = reinterpret_cast<SamplingConfig*>(reinterpret_cast<std::byte*>(scores) +
                                                     up(static_cast<std::size_t>(tokens) * sizeof(RawTokenScores)));
    // 1. the head: the invariant GEMM, FP32 out, then the softcap (Variant::lm_head)
    inkling_gemm_inv(head_weight, vocab_rows, hidden_size, hidden, tokens, logits, DType::FP32, vocab_rows, {}, nullptr,
                     0, stream);
    inkling_softcap_f32(logits, vocab_rows, tokens, vocab_rows, softcap, stream);
    // 2. the engine's scoring kernel (score_logprobs_device: raw logits, temperature 1)
    // One lane's config: the engine's defaults with top_logprobs = 0 (score the selected token only). A copy from
    // pageable memory is staged before the call returns, so the local may go (not capturable: a trainer-side op).
    SamplingConfig host_config{};
    host_config.top_logprobs = 0;
    CUDA_CHECK(cudaMemcpyAsync(config, &host_config, sizeof(SamplingConfig), cudaMemcpyHostToDevice, stream));
    const Tensor logits_t(logits, DType::FP32, {vocab_rows, tokens});
    const Tensor tokens_t(const_cast<std::int32_t*>(token_ids), DType::I32, {tokens});
    score_logprobs_device(logits_t, tokens_t, token_domain, config, scores, stream, tokens, nullptr);
    selected_kernel<<<(tokens + 255) / 256, 256, 0, stream>>>(scores, token_ids, tokens, token_domain, out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops
