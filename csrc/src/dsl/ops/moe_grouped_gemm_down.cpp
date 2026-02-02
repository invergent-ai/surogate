#include "dsl/compiled_ops.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "modules/lora/lora_config.h"
#include "modules/lora/lora_weights_manager.h"

namespace dsl {

void CompiledExecutor::dispatch_moe_grouped_gemm_down(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& weights = resolve_tensor(op.inputs[1]);  // Parameter name resolved by graph compiler
    Tensor& scatter_indices = resolve_tensor(op.inputs[2]);
    (void)scatter_indices;  // Used by kernel through expert_offsets
    const int num_tokens = static_cast<int>(mB * mT);
    int top_k = op.attrs.top_k;
    if (top_k <= 0 && num_tokens > 0 && inp.Rank == 2) {
        top_k = static_cast<int>(inp.Sizes[0] / num_tokens);
    }
    if (top_k <= 0) {
        top_k = 1;
    }

    const int num_experts = static_cast<int>(mConfig.NumExperts);
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
    // Use MoeIntermediateSize for MoE models (may differ from IntermediateSize)
    const int intermediate_size = (mConfig.MoeIntermediateSize > 0)
        ? static_cast<int>(mConfig.MoeIntermediateSize)
        : static_cast<int>(mConfig.IntermediateSize);
    const int weight_experts = (weights.Rank > 0) ? static_cast<int>(weights.Sizes[0]) : num_experts;
    int layer_idx_any = op.attrs.layer_idx;
    if (layer_idx_any < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        std::string field;
        parse_block_param(name, layer_idx_any, field);
    }
    // Get expert offsets from per-layer saved buffers when available.
    Tensor expert_offsets_view;
    Tensor* expert_offsets_ptr = nullptr;
    if (layer_idx_any >= 0) {
        const std::string key = "blocks[" + std::to_string(layer_idx_any) + "].moe_expert_offsets";
        auto it_saved = mMoESavedBuffers.find(key);
        if (it_saved != mMoESavedBuffers.end() && it_saved->second != nullptr) {
            expert_offsets_view.DType = ETensorDType::INT32;
            expert_offsets_view.Rank = 1;
            expert_offsets_view.Sizes[0] = static_cast<long>(mConfig.NumExperts + 1);
            expert_offsets_view.Data = static_cast<std::byte*>(it_saved->second);
            expert_offsets_ptr = &expert_offsets_view;
        }
    }
    if (!expert_offsets_ptr) {
        auto it = mTensorMap.find("moe_expert_offsets");
        if (it == mTensorMap.end()) {
            throw std::runtime_error("moe_grouped_gemm_down: expert_offsets not found");
        }
        expert_offsets_ptr = &it->second;
    }
    Tensor& expert_offsets = *expert_offsets_ptr;
    std::vector<int> host_offsets_local;
    const int* host_offsets_ptr = nullptr;
    if (num_experts > 0 && expert_offsets.Data) {
        host_offsets_local.resize(static_cast<std::size_t>(num_experts + 1), 0);
        CUDA_CHECK(cudaMemcpyAsync(host_offsets_local.data(),
                                   expert_offsets.get<int>(),
                                   static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        host_offsets_ptr = host_offsets_local.data();
    }

    if (host_offsets_ptr) {
        int bad = 0;
        int last = host_offsets_ptr[num_experts];
        for (int e = 1; e <= num_experts; ++e) {
            if (host_offsets_ptr[e] < host_offsets_ptr[e - 1]) {
                bad++;
            }
        }
        (void)bad;
        (void)last;
    }

    MoeCompactInfo compact = host_offsets_ptr
        ? build_moe_compact_info_from_host(host_offsets_ptr,
                                           num_experts,
                                           weight_experts,
                                           layer_idx_any,
                                           "moe_grouped_gemm_down")
        : build_moe_compact_info(expert_offsets.get<int>(),
                                 num_experts,
                                 weight_experts,
                                 mRunState.MainStream,
                                 layer_idx_any,
                                 "moe_grouped_gemm_down");
    if (!host_offsets_ptr && !compact.host_offsets.empty()) {
        host_offsets_ptr = compact.host_offsets.data();
    }
    const int* active_ptr = compact.active_experts.empty() ? nullptr : compact.active_experts.data();
    const int num_active = compact.active_experts.empty() ? -1 : compact.num_active;
    const bool weight_is_compact = compact.weight_is_compact;

    // MoE output shape is dynamic: [total_tokens, hidden_size]
    // total_tokens = inp.Sizes[0] (permuted token count)
    const long total_tokens = inp.Sizes[0];
    std::vector<long> out_shape = {total_tokens, static_cast<long>(hidden_size)};
    Tensor out = mRunState.temp_alloc(inp.DType, out_shape);
    mTemps.push_back(out);

    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(out, mRunState.MainStream);
    } else if (inp.DType == ETensorDType::BF16) {
        moe_grouped_gemm_down(out.get<nv_bfloat16>(),
                              inp.get<nv_bfloat16>(),
                              weights.get<nv_bfloat16>(),
                              expert_offsets.get<int>(),
                              num_experts, hidden_size, intermediate_size,
                              mRunState.cublas_handle(), mRunState.MainStream,
                              host_offsets_ptr,
                              active_ptr,
                              weight_is_compact,
                              num_active);
    } else {
        moe_grouped_gemm_down(out.get<float>(),
                              inp.get<float>(),
                              weights.get<float>(),
                              expert_offsets.get<int>(),
                              num_experts, hidden_size, intermediate_size,
                              mRunState.cublas_handle(), mRunState.MainStream,
                              host_offsets_ptr,
                              active_ptr,
                              weight_is_compact,
                              num_active);
    }

    mTensorMap[op.outputs[0].name] = out;
}

void CompiledExecutor::dispatch_moe_grouped_gemm_down_backward(const CompiledOp& op) {
    Tensor& d_output = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& weights = resolve_tensor(op.inputs[2]);
    Tensor& d_input = ensure_output_tensor(op.outputs[0]);
    (void)inp;  // Used by weight gradient computation if enabled
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


    // Use per-layer expert_offsets when available; fall back to global buffer.
    const int* expert_offsets_ptr = nullptr;
    Tensor expert_offsets_view;
    if (layer_idx >= 0) {
        const std::string key = "blocks[" + std::to_string(layer_idx) + "].moe_expert_offsets";
        auto it = mMoESavedBuffers.find(key);
        if (it != mMoESavedBuffers.end() && it->second != nullptr) {
            expert_offsets_view.DType = ETensorDType::INT32;
            expert_offsets_view.Rank = 1;
            expert_offsets_view.Sizes[0] = static_cast<long>(mConfig.NumExperts + 1);
            expert_offsets_view.Data = static_cast<std::byte*>(it->second);
            expert_offsets_ptr = expert_offsets_view.get<int>();
        }
    }
    if (!expert_offsets_ptr) {
        if (mMoEExpertOffsetsGPU == nullptr) {
            throw std::runtime_error("moe_grouped_gemm_down_backward: mMoEExpertOffsetsGPU not allocated");
        }
        expert_offsets_ptr = static_cast<const int*>(mMoEExpertOffsetsGPU);
    }

    const int num_experts = static_cast<int>(mConfig.NumExperts);
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
    // Use MoeIntermediateSize for MoE models (may differ from IntermediateSize)
    const int intermediate_size = (mConfig.MoeIntermediateSize > 0)
        ? static_cast<int>(mConfig.MoeIntermediateSize)
        : static_cast<int>(mConfig.IntermediateSize);

    const int weight_experts = static_cast<int>(weights.Sizes[0]);
    MoeCompactInfo compact = build_moe_compact_info(expert_offsets_ptr,
                                                    num_experts,
                                                    weight_experts,
                                                    mRunState.MainStream,
                                                    layer_idx,
                                                    "moe_grouped_gemm_down_backward");
    const bool weight_is_compact = compact.weight_is_compact;
    const int* host_offsets_ptr = compact.host_offsets.empty() ? nullptr : compact.host_offsets.data();
    const int* active_ptr = compact.active_experts.empty() ? nullptr : compact.active_experts.data();
    const int num_active = compact.num_active;

    // Refresh MoE experts for this layer (selective dequant) before using weights in backward.
    if (!compact.host_offsets.empty()) {
        (void)refresh_moe_experts_if_needed(layer_idx,
                                            compact.host_offsets.data(),
                                            num_experts,
                                            mWeights,
                                            mRunState.MainStream);
    } else {
        std::vector<int> host_offsets_fallback(static_cast<std::size_t>(num_experts + 1), 0);
        CUDA_CHECK(cudaMemcpyAsync(host_offsets_fallback.data(),
                                   expert_offsets_ptr,
                                   static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        (void)refresh_moe_experts_if_needed(layer_idx,
                                            host_offsets_fallback.data(),
                                            num_experts,
                                            mWeights,
                                            mRunState.MainStream);
    }

    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(d_input, mRunState.MainStream);
    } else if (d_output.DType == ETensorDType::BF16) {
        moe_grouped_gemm_down_backward(d_input.get<nv_bfloat16>(),
                                       d_output.get<nv_bfloat16>(),
                                       weights.get<nv_bfloat16>(),
                                       expert_offsets_ptr,
                                       num_experts, hidden_size, intermediate_size,
                                       mRunState.cublas_handle(), mRunState.MainStream,
                                       host_offsets_ptr,
                                       active_ptr,
                                       weight_is_compact,
                                       num_active);
    } else {
        moe_grouped_gemm_down_backward(d_input.get<float>(),
                                       d_output.get<float>(),
                                       weights.get<float>(),
                                       expert_offsets_ptr,
                                       num_experts, hidden_size, intermediate_size,
                                       mRunState.cublas_handle(), mRunState.MainStream,
                                       host_offsets_ptr,
                                       active_ptr,
                                       weight_is_compact,
                                       num_active);
    }

    mTensorMap[op.outputs[0].name] = d_input;

    // Weight gradient computation would go here if needed (for fine-tuning experts)
}


}  // namespace dsl
