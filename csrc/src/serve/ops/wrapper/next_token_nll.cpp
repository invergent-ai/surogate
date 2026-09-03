#include "api/ops/next_token_nll.h"

#include "ops/launcher/next_token_nll.h" // detail::next_token_nll_launch

#include <stdexcept>

namespace sinfer::ops {

void next_token_nll(const Tensor& logits, const Tensor& targets, Tensor& out, Tensor* argmax,
                    std::int32_t token_domain, cudaStream_t stream) {
    if (logits.dtype != DType::BF16 || logits.data == nullptr || logits.ne[0] <= 0 ||
        logits.ne[1] <= 0 || logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("next_token_nll: logits must be BF16 [vocab, n]");
    }
    const std::int32_t columns = logits.ne[1];
    if (targets.dtype != DType::I32 || targets.data == nullptr || targets.ne[0] != columns ||
        targets.ne[1] != 1) {
        throw std::invalid_argument("next_token_nll: targets must be I32 [n]");
    }
    if (out.dtype != DType::FP32 || out.data == nullptr || out.ne[0] != columns ||
        out.ne[1] != 1) {
        throw std::invalid_argument("next_token_nll: out must be FP32 [n]");
    }
    if (argmax != nullptr && (argmax->dtype != DType::I32 || argmax->data == nullptr ||
                              argmax->ne[0] != columns || argmax->ne[1] != 1)) {
        throw std::invalid_argument("next_token_nll: argmax must be I32 [n]");
    }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        throw std::invalid_argument("next_token_nll: token_domain must be in [1, logits.ne[0]]");
    }
    detail::next_token_nll_launch(logits, targets, out, argmax, token_domain, stream);
}

} // namespace sinfer::ops
