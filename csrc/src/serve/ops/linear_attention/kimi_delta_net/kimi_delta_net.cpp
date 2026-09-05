#include "api/ops/kimi_delta_net.h"

#include "ops/linear_attention/gated_delta_net/common.h"
#include "ops/linear_attention/kimi_delta_net/launch.h"

#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

using detail::gated_delta_net::are_head_counts_valid;
using detail::gated_delta_net::kStateDim;

void require(bool condition, const char* what) {
    if (!condition) { throw std::invalid_argument(std::string("kimi_delta_net: ") + what); }
}

void require_matrix(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t heads,
                    std::int32_t tokens, const char* name) {
    require(tensor.dtype == dtype && tensor.data != nullptr && tensor.is_contiguous(),
            (std::string(name) + " must be contiguous and of the declared dtype").c_str());
    require(tensor.ne[0] == rows && tensor.ne[1] == heads && tensor.ne[2] == tokens &&
                tensor.ne[3] == 1,
            (std::string(name) + " has the wrong shape").c_str());
}

void validate(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
              const Tensor& beta, float scale, const Tensor& state_in, const Tensor& state_out,
              const Tensor& out) {
    const std::int32_t qk_heads = k.ne[1];
    const std::int32_t v_heads  = v.ne[1];
    const std::int32_t tokens   = v.ne[2];
    require(tokens > 0, "T must be positive");
    require(are_head_counts_valid(qk_heads, v_heads),
            "value heads must be a positive multiple of the qk heads");
    require_matrix(q, DType::BF16, kStateDim, qk_heads, tokens, "q");
    require_matrix(k, DType::BF16, kStateDim, qk_heads, tokens, "k");
    require_matrix(v, DType::BF16, kStateDim, v_heads, tokens, "v");
    // The gate is per key channel, which is what separates this op from the gated delta net:
    // there it is one FP32 value per head per token, here one per channel of every head.
    require_matrix(g, DType::FP32, kStateDim, v_heads, tokens, "g");
    require(beta.dtype == DType::FP32 && beta.data != nullptr && beta.is_contiguous() &&
                beta.ne[0] == v_heads && beta.ne[1] == tokens && beta.ne[2] == 1,
            "beta must be contiguous FP32 [value heads, T]");
    require_matrix(out, DType::BF16, kStateDim, v_heads, tokens, "out");
    for (const Tensor* state : {&state_in, &state_out}) {
        require(state->data != nullptr && state->is_contiguous() && state->ne[0] == kStateDim &&
                    state->ne[1] == kStateDim && state->ne[2] == v_heads && state->ne[3] == 1,
                "the state must be contiguous [128, 128, value heads]");
    }
    require(scale > 0.0F, "scale must be positive");
}

} // namespace

void kimi_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                    const Tensor& beta, float scale, bool normalize_qk,
                    const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                    cudaStream_t stream) {
    validate(q, k, v, g, beta, scale, ssm_state_in, ssm_state_out, out);
    detail::kimi_delta_net::launch_recurrent(q, k, v, g, beta, scale, normalize_qk, ssm_state_in,
                                             ssm_state_out, out, stream);
}

void kimi_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                    const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_state,
                    Tensor& out, cudaStream_t stream) {
    kimi_delta_net(q, k, v, g, beta, scale, normalize_qk, ssm_state, ssm_state, out, stream);
}

} // namespace sinfer::ops
