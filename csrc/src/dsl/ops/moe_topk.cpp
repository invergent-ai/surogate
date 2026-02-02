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

    const bool moe_route_trace = (std::getenv("SUROGATE_MOE_ROUTE_TRACE") != nullptr);
    static int moe_route_topk_trace = 0;
    if (moe_route_trace && moe_route_topk_trace < 4) {
        const std::size_t total = static_cast<std::size_t>(weights.nelem());
        const std::size_t sample = std::min<std::size_t>(4096, total);
        log_tensor_sample_stats("MOE_TOPK_WEIGHTS", weights, 0, sample);
        moe_route_topk_trace++;
    }

    // Resolve layer index for debugging and selective expert refresh.
    int layer_idx = op.attrs.layer_idx;
    std::string field;
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        parse_block_param(name, layer_idx, field);
    }

    log_moe_gate_up_weight_sample("PRE_MOE_TOPK", layer_idx, mMicroStep, mWeights, mConfig);

    // Force selective expert dequantization for this layer based on top-k indices.
    if (layer_idx >= 0) {
        if (auto* provider = mWeights.qlora_provider()) {
            modules::SelectiveExpertInfo selection;
            selection.build_from_router_output(indices, num_experts, mRunState.MainStream);
            (void)provider->refresh_moe_experts(layer_idx, selection, mRunState.MainStream);
            log_moe_gate_up_weight_sample("POST_MOE_TOPK_REFRESH",
                                          layer_idx, mMicroStep, mWeights, mConfig);
        }
    }

    // DEBUG: Verify routing indices are within [0, num_experts).
    static int moe_topk_trace = 0;
    if (layer_idx < 0 || layer_idx < 4 || layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
        const int total = num_tokens * top_k;
        std::vector<int> host_idx(total, -1);
        CUDA_CHECK(cudaMemcpy(host_idx.data(), indices.get<int>(),
                              static_cast<std::size_t>(total) * sizeof(int),
                              cudaMemcpyDeviceToHost));
        int min_idx = std::numeric_limits<int>::max();
        int max_idx = std::numeric_limits<int>::min();
        int oob = 0;
        for (int v : host_idx) {
            min_idx = std::min(min_idx, v);
            max_idx = std::max(max_idx, v);
            if (v < 0 || v >= num_experts) {
                oob++;
            }
        }
        if (moe_topk_trace < 8 || oob > 0 || min_idx < 0) {
            fprintf(stderr,
                    "[MOE_TOPK_IDX] layer=%d shape=[%d,%d] num_experts=%d min=%d max=%d oob=%d\n",
                    layer_idx, num_tokens, top_k, num_experts, min_idx, max_idx, oob);
        }
        if (oob > 0 || min_idx < 0) {
            log_nan_sample("MOE_PROBS_NAN", layer_idx, op.inputs[0].name, probs, 3);
            log_tensor_stats_ex("MOE_PROBS_NAN", layer_idx, op.inputs[0].name, probs, 4096, true);
        }
        moe_topk_trace++;
    }
}

void CompiledExecutor::dispatch_moe_topk_backward(const CompiledOp& op) {
    Tensor& d_routing_weights = resolve_tensor(op.inputs[0]);
    Tensor& probs = resolve_tensor(op.inputs[1]);
    Tensor& expert_indices = resolve_tensor(op.inputs[2]);

    const int num_tokens = static_cast<int>(probs.Sizes[0]);
    const int num_experts = static_cast<int>(probs.Sizes[1]);
    const int top_k = op.attrs.top_k;
    const bool normalize = op.attrs.normalize_weights;
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

    static int moe_topk_mag = 0;
    if (moe_topk_mag < 8 && layer_idx <= 2) {
        fprintf(stderr,
                "[MOE_TOPK_BWD] layer=%d tokens=%d experts=%d top_k=%d normalize=%d\n",
                layer_idx, num_tokens, num_experts, top_k, normalize ? 1 : 0);
        log_tensor_mag("MOE_TOPK_BWD_DROUTING", layer_idx, op.inputs[0].name, d_routing_weights, 4096);
        log_tensor_mag("MOE_TOPK_BWD_PROBS", layer_idx, op.inputs[1].name, probs, 4096);
        moe_topk_mag++;
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

    if (moe_topk_mag < 12 && layer_idx <= 2) {
        log_tensor_mag("MOE_TOPK_BWD_DPROBS", layer_idx, op.outputs[0].name, d_probs, 4096);
        moe_topk_mag++;
    }

    mTensorMap[op.outputs[0].name] = d_probs;
}


}  // namespace dsl
