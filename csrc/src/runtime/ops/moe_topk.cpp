#include "runtime/dsl/compiled_ops.h"

#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_moe_topk(const CompiledOp& op) {
    Tensor& probs = resolve_tensor(op.inputs[0]);
    Tensor& weights = ensure_output_tensor(op.outputs[0]);
    Tensor& indices = ensure_output_tensor(op.outputs[1]);

    const int num_tokens = static_cast<int>(probs.Sizes[0]);
    const int num_experts = static_cast<int>(probs.Sizes[1]);
    const int top_k = op.attrs.top_k;
    const bool normalize = op.attrs.normalize_weights;

    if (probs.DType == ETensorDType::BF16) {
        moe_topk_forward(indices.get<int>(),
                         weights.get<nv_bfloat16>(),
                         probs.get<nv_bfloat16>(),
                         num_tokens, num_experts, top_k, normalize, mRunState.MainStream);
    } else {
        moe_topk_forward(indices.get<int>(),
                         weights.get<float>(),
                         probs.get<float>(),
                         num_tokens, num_experts, top_k, normalize, mRunState.MainStream);
    }

    mTensorMap[op.outputs[0].name] = weights;
    mTensorMap[op.outputs[1].name] = indices;

}

void CompiledExecutor::dispatch_moe_topk_backward(const CompiledOp& op) {
    Tensor& d_routing_weights = resolve_tensor(op.inputs[0]);
    Tensor& probs = resolve_tensor(op.inputs[1]);
    Tensor& expert_indices = resolve_tensor(op.inputs[2]);

    const int num_tokens = static_cast<int>(probs.Sizes[0]);
    const int num_experts = static_cast<int>(probs.Sizes[1]);
    const int top_k = op.attrs.top_k;
    const bool normalize = op.attrs.normalize_weights;
    // Allocate output with correct shape derived from probs (not from compile-time inference)
    // d_probs must have shape [num_tokens, num_experts] matching probs
    std::vector<long> d_probs_shape = {static_cast<long>(num_tokens), static_cast<long>(num_experts)};
    Tensor d_probs = mRunState.temp_alloc(d_routing_weights.DType, d_probs_shape);
    mTemps.push_back(d_probs);

    // TopK backward kernel only supports FP32
    // If inputs are BF16, cast to FP32 temporaries and cast output back
    if (probs.DType == ETensorDType::BF16) {
        // Allocate FP32 temporaries
        Tensor d_weights_f32 = mRunState.Stack.allocate(ETensorDType::FP32,
            {static_cast<long>(num_tokens), static_cast<long>(top_k)}, "d_weights_f32");
        Tensor probs_f32 = mRunState.Stack.allocate(ETensorDType::FP32,
            {static_cast<long>(num_tokens), static_cast<long>(num_experts)}, "probs_f32");
        Tensor d_probs_f32 = mRunState.Stack.allocate(ETensorDType::FP32,
            {static_cast<long>(num_tokens), static_cast<long>(num_experts)}, "d_probs_f32");

        // Cast inputs to FP32
        convert_dtype(d_weights_f32.get<float>(), d_routing_weights.get<nv_bfloat16>(),
                      d_routing_weights.nelem(), mRunState.MainStream);
        convert_dtype(probs_f32.get<float>(), probs.get<nv_bfloat16>(),
                      probs.nelem(), mRunState.MainStream);

        // Run backward in FP32
        moe_topk_backward(d_probs_f32.get<float>(),
                          d_weights_f32.get<float>(),
                          probs_f32.get<float>(),
                          expert_indices.get<int>(),
                          num_tokens, num_experts, top_k, normalize, mRunState.MainStream);

        // Cast output back to BF16
        convert_dtype(d_probs.get<nv_bfloat16>(), d_probs_f32.get<float>(),
                      d_probs.nelem(), mRunState.MainStream);
    } else {
        // FP32 path
        moe_topk_backward(d_probs.get<float>(),
                          d_routing_weights.get<float>(),
                          probs.get<float>(),
                          expert_indices.get<int>(),
                          num_tokens, num_experts, top_k, normalize, mRunState.MainStream);
    }

    mTensorMap[op.outputs[0].name] = d_probs;
}


}  // namespace dsl
