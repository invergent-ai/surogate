#include "api/ops/sampled_logprob.h"

#include "ops/launcher/sampled_logprob.h" // detail::sampled_logprob_launch

#include <stdexcept>
#include <string>

namespace sinfer::ops {

void sampled_logprob(const Tensor& logits, const Tensor& tokens, Tensor& out,
                     std::int32_t token_domain, const SamplingConfig* configs,
                     cudaStream_t stream) {
    if (logits.dtype != DType::BF16 || logits.data == nullptr || logits.ne[0] <= 0 ||
        logits.ne[1] <= 0 || logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("sampled_logprob: logits must be BF16 [vocab, n]");
    }
    const std::int32_t columns = logits.ne[1];
    if (tokens.dtype != DType::I32 || tokens.data == nullptr || tokens.ne[0] != columns ||
        tokens.ne[1] != 1) {
        throw std::invalid_argument("sampled_logprob: tokens must be I32 [n]");
    }
    if (out.dtype != DType::FP32 || out.data == nullptr || out.ne[0] != columns || out.ne[1] != 1) {
        throw std::invalid_argument("sampled_logprob: out must be FP32 [n]");
    }
    if (configs == nullptr) {
        throw std::invalid_argument("sampled_logprob: configs must be a device SamplingConfig[n]");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument("sampled_logprob: token_domain " + std::to_string(token_domain) +
                                    " must be in [1, " + std::to_string(logits.ne[0]) + "]");
    }
    detail::sampled_logprob_launch(logits, tokens, out, token_domain, configs, stream);
}

} // namespace sinfer::ops
