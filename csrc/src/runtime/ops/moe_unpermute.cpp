#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
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
        if (routing_weights.DType == ETensorDType::FP32) {
            moe_unpermute_and_combine(out.get<nv_bfloat16>(),
                                      expert_out.get<nv_bfloat16>(),
                                      routing_weights.get<float>(),
                                      scatter_indices.get<int>(),
                                      num_tokens, total_tokens, hidden_size, top_k,
                                      mRunState.MainStream);
        } else if (routing_weights.DType == ETensorDType::BF16) {
            moe_unpermute_and_combine(out.get<nv_bfloat16>(),
                                      expert_out.get<nv_bfloat16>(),
                                      routing_weights.get<nv_bfloat16>(),
                                      scatter_indices.get<int>(),
                                      num_tokens, total_tokens, hidden_size, top_k,
                                      mRunState.MainStream);
        } else {
            throw std::logic_error("moe_unpermute: unsupported routing_weights dtype");
        }
    } else if (expert_out.DType == ETensorDType::FP32) {
        if (routing_weights.DType != ETensorDType::FP32) {
            throw std::logic_error("moe_unpermute: routing_weights dtype must match FP32 expert_out");
        }
        moe_unpermute_and_combine(out.get<float>(),
                                  expert_out.get<float>(),
                                  routing_weights.get<float>(),
                                  scatter_indices.get<int>(),
                                  num_tokens, total_tokens, hidden_size, top_k,
                                  mRunState.MainStream);
    } else {
        throw std::logic_error("moe_unpermute: unsupported expert_out dtype");
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

    // Validate expert_out shape: must be [total_tokens, hidden_size].
    // A shape mismatch (e.g., [1, hidden_size] from a stale/incorrectly saved tensor)
    // causes the kernel to read out of bounds, producing NaN in routing weight gradients.
    if (expert_out.nelem() != static_cast<long>(total_tokens) * hidden_size) {
        std::ostringstream oss;
        oss << "moe_unpermute_backward: expert_out shape mismatch. "
            << "Expected [" << total_tokens << ", " << hidden_size << "] ("
            << static_cast<long>(total_tokens) * hidden_size << " elems), got [";
        for (int d = 0; d < expert_out.Rank; ++d) {
            if (d > 0) oss << ", ";
            oss << expert_out.Sizes[d];
        }
        oss << "] (" << expert_out.nelem() << " elems). "
            << "ref_name='" << op.inputs[1].name << "' ref_slot="
            << static_cast<int>(op.inputs[1].slot);
        throw std::runtime_error(oss.str());
    }

    if (d_output.DType == ETensorDType::BF16) {
        if (routing_weights.DType == ETensorDType::FP32) {
            moe_combine_backward(d_expert_out.get<nv_bfloat16>(),
                                 d_routing_weights.get<float>(),
                                 d_output.get<nv_bfloat16>(),
                                 expert_out.get<nv_bfloat16>(),
                                 routing_weights.get<float>(),
                                 scatter_indices.get<int>(),
                                 num_tokens, total_tokens, hidden_size, top_k,
                                 mRunState.MainStream);
        } else if (routing_weights.DType == ETensorDType::BF16) {
            moe_combine_backward(d_expert_out.get<nv_bfloat16>(),
                                 d_routing_weights.get<nv_bfloat16>(),
                                 d_output.get<nv_bfloat16>(),
                                 expert_out.get<nv_bfloat16>(),
                                 routing_weights.get<nv_bfloat16>(),
                                 scatter_indices.get<int>(),
                                 num_tokens, total_tokens, hidden_size, top_k,
                                 mRunState.MainStream);
        } else {
            throw std::logic_error("moe_unpermute_backward: unsupported routing_weights dtype");
        }
    } else if (d_output.DType == ETensorDType::FP32) {
        if (routing_weights.DType != ETensorDType::FP32) {
            throw std::logic_error("moe_unpermute_backward: routing_weights dtype must match FP32 d_output");
        }
        moe_combine_backward(d_expert_out.get<float>(),
                             d_routing_weights.get<float>(),
                             d_output.get<float>(),
                             expert_out.get<float>(),
                             routing_weights.get<float>(),
                             scatter_indices.get<int>(),
                             num_tokens, total_tokens, hidden_size, top_k,
                             mRunState.MainStream);
    } else {
        throw std::logic_error("moe_unpermute_backward: unsupported d_output dtype");
    }

    mTensorMap[op.outputs[0].name] = d_expert_out;
    mTensorMap[op.outputs[1].name] = d_routing_weights;
}


}  // namespace dsl
