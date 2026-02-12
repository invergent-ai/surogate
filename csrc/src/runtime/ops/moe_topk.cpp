#include "runtime/dsl/compiled_ops.h"

#include <vector>
#include <cstdlib>
#include <cmath>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_moe_topk(const CompiledOp& op) {
    Tensor& probs = resolve_tensor(op.inputs[0]);

    int layer_idx_any = op.attrs.layer_idx;
    if (layer_idx_any < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        std::string field;
        parse_block_param(name, layer_idx_any, field);
    }

    // Optional correction bias (e.g., e_score_correction_bias from DeepSeek V3 / Nemotron-H routing)
    const float* correction_bias = nullptr;
    if (op.inputs.size() > 1 && !op.inputs[1].name.empty()) {
        Tensor& bias = resolve_tensor(op.inputs[1]);
        correction_bias = bias.get<float>();
    }

    const int num_tokens = static_cast<int>(probs.Sizes[0]);
    const int num_experts = static_cast<int>(probs.Sizes[1]);
    const int top_k = op.attrs.top_k;
    const bool normalize = op.attrs.normalize_weights;
    const bool softmax = op.attrs.topk_softmax;
    const float scaling_factor = op.attrs.scaling_factor;
    const float rounding_scale = op.attrs.topk_rounding_scale;
    const bool sort_by_index = op.attrs.topk_sort_by_index;

    // MoE topk outputs have dynamic shapes depending on num_tokens and top_k.
    // The compiled graph may have empty shapes for these intermediates, so we
    // allocate directly with the correct dimensions.
    Tensor weights = mRunState.temp_alloc(probs.DType,
        {static_cast<long>(num_tokens), static_cast<long>(top_k)});
    mTemps.push_back(weights);
    Tensor indices = mRunState.temp_alloc(ETensorDType::INT32,
        {static_cast<long>(num_tokens), static_cast<long>(top_k)});
    mTemps.push_back(indices);

    if (probs.DType == ETensorDType::BF16) {
        moe_topk_forward(indices.get<int>(),
                         weights.get<nv_bfloat16>(),
                         probs.get<nv_bfloat16>(),
                         correction_bias,
                         num_tokens, num_experts, top_k, normalize, softmax,
                         sort_by_index, rounding_scale, mRunState.MainStream);
    } else {
        moe_topk_forward(indices.get<int>(),
                         weights.get<float>(),
                         probs.get<float>(),
                         correction_bias,
                         num_tokens, num_experts, top_k, normalize, softmax,
                         sort_by_index, rounding_scale, mRunState.MainStream);
    }

    // Apply routed_scaling_factor to weights after top-k selection + normalization
    if (scaling_factor != 1.0f) {
        const int n = num_tokens * top_k;
        if (weights.DType == ETensorDType::BF16) {
            moe_scale_forward(weights.get<nv_bfloat16>(), weights.get<nv_bfloat16>(),
                              scaling_factor, n, mRunState.MainStream);
        } else {
            moe_scale_forward(weights.get<float>(), weights.get<float>(),
                              scaling_factor, n, mRunState.MainStream);
        }
    }

    store_tensor(op.outputs[0], weights);
    store_tensor(op.outputs[1], indices);

}

void CompiledExecutor::dispatch_moe_topk_backward(const CompiledOp& op) {
    Tensor& d_routing_weights = resolve_tensor(op.inputs[0]);
    Tensor& probs = resolve_tensor(op.inputs[1]);
    Tensor& expert_indices = resolve_tensor(op.inputs[2]);

    const int num_tokens = static_cast<int>(probs.Sizes[0]);
    const int num_experts = static_cast<int>(probs.Sizes[1]);
    const int top_k = op.attrs.top_k;
    const bool normalize = op.attrs.normalize_weights;
    const bool softmax = op.attrs.topk_softmax;
    const float scaling_factor = op.attrs.scaling_factor;

    // If scaling_factor != 1.0, scale d_routing_weights by the factor before backward.
    // Forward was: output = topk(probs) * sf, so d_topk = d_output * sf
    Tensor* d_weights_ptr = &d_routing_weights;
    Tensor d_weights_scaled;
    if (scaling_factor != 1.0f) {
        d_weights_scaled = mRunState.temp_alloc(d_routing_weights.DType,
            {static_cast<long>(num_tokens), static_cast<long>(top_k)});
        mTemps.push_back(d_weights_scaled);
        const int n = num_tokens * top_k;
        if (d_routing_weights.DType == ETensorDType::BF16) {
            moe_scale_forward(d_weights_scaled.get<nv_bfloat16>(),
                              d_routing_weights.get<nv_bfloat16>(),
                              scaling_factor, n, mRunState.MainStream);
        } else {
            moe_scale_forward(d_weights_scaled.get<float>(),
                              d_routing_weights.get<float>(),
                              scaling_factor, n, mRunState.MainStream);
        }
        d_weights_ptr = &d_weights_scaled;
    }

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
        fill_zero(d_probs_f32, mRunState.MainStream);

        // Cast inputs to FP32
        convert_dtype(d_weights_f32.get<float>(), d_weights_ptr->get<nv_bfloat16>(),
                      d_weights_ptr->nelem(), mRunState.MainStream);
        convert_dtype(probs_f32.get<float>(), probs.get<nv_bfloat16>(),
                      probs.nelem(), mRunState.MainStream);

        // Run backward in FP32
        moe_topk_backward(d_probs_f32.get<float>(),
                          d_weights_f32.get<float>(),
                          probs_f32.get<float>(),
                          expert_indices.get<int>(),
                          num_tokens, num_experts, top_k, normalize, softmax, mRunState.MainStream);

        // Cast output back to BF16
        convert_dtype(d_probs.get<nv_bfloat16>(), d_probs_f32.get<float>(),
                      d_probs.nelem(), mRunState.MainStream);
    } else {
        // FP32 path
        fill_zero(d_probs, mRunState.MainStream);
        moe_topk_backward(d_probs.get<float>(),
                          d_weights_ptr->get<float>(),
                          probs.get<float>(),
                          expert_indices.get<int>(),
                          num_tokens, num_experts, top_k, normalize, softmax, mRunState.MainStream);
    }

    store_tensor(op.outputs[0], d_probs);
}


}  // namespace dsl
