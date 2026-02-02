#include "dsl/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_moe_softmax(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    int layer_idx = op.attrs.layer_idx;
    std::string field;
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        parse_block_param(name, layer_idx, field);
    }

    log_moe_gate_up_weight_sample("PRE_MOE_SOFTMAX", layer_idx, mMicroStep, mWeights, mConfig);

    const int num_tokens = static_cast<int>(inp.Sizes[0]);
    const int num_experts = static_cast<int>(inp.Sizes[1]);

    // Allocate output with same shape as input (softmax doesn't change shape)
    std::vector<long> out_shape = {static_cast<long>(num_tokens), static_cast<long>(num_experts)};
    Tensor out = mRunState.temp_alloc(inp.DType, out_shape);
    mTemps.push_back(out);

    if (inp.DType == ETensorDType::BF16) {
        moe_softmax_forward(out.get<nv_bfloat16>(),
                            inp.get<nv_bfloat16>(),
                            num_tokens, num_experts, mRunState.MainStream);
    } else {
        moe_softmax_forward(out.get<float>(),
                            inp.get<float>(),
                            num_tokens, num_experts, mRunState.MainStream);
    }

    const bool moe_route_trace = (std::getenv("SUROGATE_MOE_ROUTE_TRACE") != nullptr);
    static int moe_route_trace_count = 0;
    if (moe_route_trace && moe_route_trace_count < 4) {
        const std::size_t total = static_cast<std::size_t>(out.nelem());
        const std::size_t sample = std::min<std::size_t>(4096, total);
        log_tensor_sample_stats("MOE_ROUTER_PROBS", out, 0, sample);
        moe_route_trace_count++;
    }

    mTensorMap[op.outputs[0].name] = out;
}

void CompiledExecutor::dispatch_moe_softmax_backward(const CompiledOp& op) {
    Tensor& d_probs = resolve_tensor(op.inputs[0]);
    Tensor& softmax_probs = resolve_tensor(op.inputs[1]);
    Tensor& d_logits = ensure_output_tensor(op.outputs[0]);

    const int num_tokens = static_cast<int>(d_probs.Sizes[0]);
    const int num_experts = static_cast<int>(d_probs.Sizes[1]);
    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("d_", 0) == 0) {
            name.remove_prefix(2);
        }
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        std::string field;
        parse_block_param(name, layer_idx, field);
    }

    static int moe_softmax_mag = 0;
    if (moe_softmax_mag < 8 && layer_idx <= 2) {
        fprintf(stderr,
                "[MOE_SOFTMAX_BWD] layer=%d tokens=%d experts=%d\n",
                layer_idx, num_tokens, num_experts);
        log_tensor_mag("MOE_SOFTMAX_BWD_DPROBS", layer_idx, op.inputs[0].name, d_probs, 4096);
        log_tensor_mag("MOE_SOFTMAX_BWD_PROBS", layer_idx, op.inputs[1].name, softmax_probs, 4096);
        moe_softmax_mag++;
    }

    if (d_probs.DType == ETensorDType::BF16) {
        moe_softmax_backward(d_logits.get<nv_bfloat16>(),
                             d_probs.get<nv_bfloat16>(),
                             softmax_probs.get<nv_bfloat16>(),
                             num_tokens, num_experts, mRunState.MainStream);
    } else {
        moe_softmax_backward(d_logits.get<float>(),
                             d_probs.get<float>(),
                             softmax_probs.get<float>(),
                             num_tokens, num_experts, mRunState.MainStream);
    }

    mTensorMap[op.outputs[0].name] = d_logits;
}


}  // namespace dsl
