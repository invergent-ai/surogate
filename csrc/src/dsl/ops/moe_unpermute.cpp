#include "dsl/compiled_ops.h"

#include <algorithm>
#include <vector>

#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
namespace dsl {

void CompiledExecutor::dispatch_moe_unpermute(const CompiledOp& op) {
    Tensor& expert_out = resolve_tensor(op.inputs[0]);
    Tensor& routing_weights = resolve_tensor(op.inputs[1]);
    Tensor& scatter_indices = resolve_tensor(op.inputs[2]);

    const int top_k = op.attrs.top_k;
    const int num_tokens = static_cast<int>(routing_weights.Sizes[0]);
    const int total_tokens = num_tokens * top_k;
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);

    // MoE output shape is dynamic: [num_tokens, hidden_size]
    // Use the preallocated mlp_down buffer to avoid stack allocation issues.
    // The mlp_down buffer has shape (B, T, C) which equals [num_tokens, hidden_size]
    // when viewed as 2D. This buffer survives layer boundary cleanup.
    int layer_idx = mCurrentLayer >= 0 ? mCurrentLayer : 0;
    auto& acts = mRunState.simplified_acts(layer_idx);
    Tensor out = view_tensor(acts.mlp_down, {static_cast<long>(num_tokens), static_cast<long>(hidden_size)});

    if (expert_out.DType == ETensorDType::BF16) {
        moe_unpermute_and_combine(out.get<nv_bfloat16>(),
                                  expert_out.get<nv_bfloat16>(),
                                  routing_weights.get<nv_bfloat16>(),
                                  scatter_indices.get<int>(),
                                  num_tokens, total_tokens, hidden_size, top_k,
                                  mRunState.MainStream);
    } else {
        moe_unpermute_and_combine(out.get<float>(),
                                  expert_out.get<float>(),
                                  routing_weights.get<float>(),
                                  scatter_indices.get<int>(),
                                  num_tokens, total_tokens, hidden_size, top_k,
                                  mRunState.MainStream);
    }

    mTensorMap[op.outputs[0].name] = out;
}

void CompiledExecutor::dispatch_moe_unpermute_backward(const CompiledOp& op) {
    Tensor& d_output = resolve_tensor(op.inputs[0]);
    Tensor& expert_out = resolve_tensor(op.inputs[1]);
    Tensor& routing_weights = resolve_tensor(op.inputs[2]);
    Tensor& scatter_indices = resolve_tensor(op.inputs[3]);

    Tensor& d_expert_out = ensure_output_tensor(op.outputs[0]);
    Tensor& d_routing_weights = ensure_output_tensor(op.outputs[1]);

    const int top_k = op.attrs.top_k;
    const int num_tokens = static_cast<int>(routing_weights.Sizes[0]);
    const int total_tokens = num_tokens * top_k;
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);

    if (d_output.DType == ETensorDType::BF16) {
        moe_combine_backward(d_expert_out.get<nv_bfloat16>(),
                             d_routing_weights.get<nv_bfloat16>(),
                             d_output.get<nv_bfloat16>(),
                             expert_out.get<nv_bfloat16>(),
                             routing_weights.get<nv_bfloat16>(),
                             scatter_indices.get<int>(),
                             num_tokens, total_tokens, hidden_size, top_k,
                             mRunState.MainStream);
    } else {
        moe_combine_backward(d_expert_out.get<float>(),
                             d_routing_weights.get<float>(),
                             d_output.get<float>(),
                             expert_out.get<float>(),
                             routing_weights.get<float>(),
                             scatter_indices.get<int>(),
                             num_tokens, total_tokens, hidden_size, top_k,
                             mRunState.MainStream);
    }

    mTensorMap[op.outputs[0].name] = d_expert_out;
    mTensorMap[op.outputs[1].name] = d_routing_weights;
}


}  // namespace dsl
