#include "api/ops/kimi_delta_net.h"

#include "core/limits.h"
#include "ops/linear_attention/gated_delta_net/common.h"
#include "ops/linear_attention/kimi_delta_net/launch.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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

void require_batched(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t heads,
                     std::int32_t width, std::int32_t batch, const char* name) {
    require(tensor.dtype == dtype && tensor.data != nullptr && tensor.is_contiguous(),
            (std::string(name) + " must be contiguous and of the declared dtype").c_str());
    require(tensor.ne[0] == rows && tensor.ne[1] == heads && tensor.ne[2] == width &&
                tensor.ne[3] == batch,
            (std::string(name) + " has the wrong shape").c_str());
}

void require_slots(const Tensor& tensor, std::int32_t batch, const char* name) {
    require(tensor.dtype == DType::I32 && tensor.data != nullptr && tensor.is_contiguous() &&
                tensor.ne[0] == batch && tensor.ne[1] == 1,
            (std::string(name) + " must be contiguous I32 [B]").c_str());
}

void validate_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                       const Tensor& beta, float scale, const Tensor& states,
                       const Tensor& valid_columns, const Tensor& initial_state_slots,
                       const Tensor& snapshot_base_slots, const Tensor& out) {
    const std::int32_t qk_heads = k.ne[1];
    const std::int32_t v_heads  = v.ne[1];
    const std::int32_t width    = v.ne[2];
    const std::int32_t batch    = v.ne[3];
    require(width > 0 && batch > 0, "W and B must be positive");
    require(are_head_counts_valid(qk_heads, v_heads),
            "value heads must be a positive multiple of the qk heads");
    require_batched(q, DType::BF16, kStateDim, qk_heads, width, batch, "q");
    require_batched(k, DType::BF16, kStateDim, qk_heads, width, batch, "k");
    require_batched(v, DType::BF16, kStateDim, v_heads, width, batch, "v");
    require_batched(g, DType::FP32, kStateDim, v_heads, width, batch, "g");
    require(beta.dtype == DType::FP32 && beta.data != nullptr && beta.is_contiguous() &&
                beta.ne[0] == v_heads && beta.ne[1] == width && beta.ne[2] == batch &&
                beta.ne[3] == 1,
            "beta must be contiguous FP32 [value heads, W, B]");
    require_batched(out, DType::BF16, kStateDim, v_heads, width, batch, "out");
    require(states.data != nullptr && states.is_contiguous() && states.ne[0] == kStateDim &&
                states.ne[1] == kStateDim && states.ne[2] == v_heads && states.ne[3] > 0,
            "the state pool must be contiguous [128, 128, value heads, slots]");
    require_slots(initial_state_slots, batch, "initial_state_slots");
    require_slots(snapshot_base_slots, batch, "snapshot_base_slots");
    if (valid_columns.data != nullptr) { require_slots(valid_columns, batch, "valid_columns"); }
    require(scale > 0.0F, "scale must be positive");
}

struct Range {
    std::uintptr_t begin;
    std::uintptr_t end;
};

Range range_of(const Tensor& tensor) {
    const auto begin = reinterpret_cast<std::uintptr_t>(tensor.data);
    return {begin, begin + tensor.bytes()};
}

void validate_replay_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, const Tensor& states,
                            const Tensor& valid_columns, const Tensor& initial_state_slots,
                            const Tensor& key_record, const Tensor& value_record,
                            const Tensor& gate_record, const Tensor& beta_record,
                            const Tensor& out) {
    const std::int32_t qk_heads = k.ne[1];
    const std::int32_t v_heads  = v.ne[1];
    const std::int32_t width    = v.ne[2];
    const std::int32_t batch    = v.ne[3];
    // The record domain is a speculative round's: the draft window plus its anchor, over the
    // engine's lanes. The delta net's record form has the same bounds.
    require(width >= 2 && width <= 16, "the record width T must be in [2,16]");
    require(batch > 0 && batch <= kMaximumBatchColumns,
            "the record batch B must be in [1, the lane cap]");
    require(are_head_counts_valid(qk_heads, v_heads),
            "value heads must be a positive multiple of the qk heads");
    require_batched(q, DType::BF16, kStateDim, qk_heads, width, batch, "q");
    require_batched(k, DType::BF16, kStateDim, qk_heads, width, batch, "k");
    require_batched(v, DType::BF16, kStateDim, v_heads, width, batch, "v");
    require_batched(g, DType::FP32, kStateDim, v_heads, width, batch, "g");
    require(beta.dtype == DType::FP32 && beta.data != nullptr && beta.is_contiguous() &&
                beta.ne[0] == v_heads && beta.ne[1] == width && beta.ne[2] == batch &&
                beta.ne[3] == 1,
            "beta must be contiguous FP32 [value heads, W, B]");
    require_batched(out, DType::BF16, kStateDim, v_heads, width, batch, "out");
    require(states.data != nullptr && states.is_contiguous() && states.ne[0] == kStateDim &&
                states.ne[1] == kStateDim && states.ne[2] == v_heads && states.ne[3] > 0,
            "the state pool must be contiguous [128, 128, value heads, slots]");
    require_slots(initial_state_slots, batch, "initial_state_slots");
    if (valid_columns.data != nullptr) { require_slots(valid_columns, batch, "valid_columns"); }
    require_batched(key_record, DType::BF16, kStateDim, qk_heads, width, batch, "key record");
    require_batched(value_record, DType::BF16, kStateDim, v_heads, width, batch, "value record");
    require_batched(gate_record, DType::FP32, kStateDim, v_heads, width, batch, "gate record");
    require(beta_record.dtype == DType::FP32 && beta_record.data != nullptr &&
                beta_record.is_contiguous() && beta_record.ne[0] == v_heads &&
                beta_record.ne[1] == width && beta_record.ne[2] == batch &&
                beta_record.ne[3] == 1,
            "the beta record must be contiguous FP32 [value heads, W, B]");
    const float expected_scale = 1.0F / std::sqrt(static_cast<float>(kStateDim));
    require(std::isfinite(scale) && std::abs(scale - expected_scale) <= 1.0e-6F,
            "scale must be 1/sqrt(128)");

    std::vector<Range> ranges;
    for (const Tensor* tensor : {&q, &k, &v, &g, &beta, &states, &valid_columns,
                                 &initial_state_slots, &key_record, &value_record, &gate_record,
                                 &beta_record, &out}) {
        if (tensor->data != nullptr) { ranges.push_back(range_of(*tensor)); }
    }
    for (std::size_t lhs = 0; lhs < ranges.size(); ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < ranges.size(); ++rhs) {
            require(!(ranges[lhs].begin < ranges[rhs].end && ranges[rhs].begin < ranges[lhs].end),
                    "replay-record tensors must not overlap");
        }
    }
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

void kimi_delta_net_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, bool normalize_qk,
                             Tensor& ssm_states, const Tensor& valid_columns,
                             const Tensor& initial_state_slots,
                             const Tensor& snapshot_base_slots, Tensor& out,
                             cudaStream_t stream) {
    validate_snapshot(q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots,
                      snapshot_base_slots, out);
    detail::kimi_delta_net::launch_recurrent_snapshot(q, k, v, g, beta, scale, normalize_qk,
                                                      ssm_states, valid_columns,
                                                      initial_state_slots, snapshot_base_slots,
                                                      out, stream);
}

void kimi_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& g, const Tensor& beta, float scale,
                                  const Tensor& ssm_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots, Tensor& key_record,
                                  Tensor& value_record, Tensor& gate_record, Tensor& beta_record,
                                  Tensor& out, cudaStream_t stream) {
    validate_replay_record(q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots,
                           key_record, value_record, gate_record, beta_record, out);
    detail::kimi_delta_net::launch_recurrent_record(q, k, v, g, beta, scale, ssm_states,
                                                    valid_columns, initial_state_slots, key_record,
                                                    value_record, gate_record, beta_record, out,
                                                    stream);
}

} // namespace sinfer::ops
