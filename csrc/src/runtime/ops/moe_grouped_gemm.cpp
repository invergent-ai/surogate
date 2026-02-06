// MoE Grouped GEMM operation (generic version without fused activation)
// Used for Nemotron-H MoE blocks that use relu2 activation instead of swiglu

#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_moe_grouped_gemm(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& weights = resolve_tensor(op.inputs[1]);
    Tensor& scatter_indices = resolve_tensor(op.inputs[2]);
    (void)scatter_indices;

    const int num_tokens = static_cast<int>(mB * mT);
    int top_k = op.attrs.top_k;
    if (top_k <= 0 && num_tokens > 0 && inp.Rank == 2) {
        top_k = static_cast<int>(inp.Sizes[0] / num_tokens);
    }
    if (top_k <= 0) {
        top_k = 1;
    }

    const int num_experts = static_cast<int>(mConfig.NumExperts);
    // Weight dimensions: [num_experts, out_features, in_features]
    // Output size: out_features (second dimension of weights)
    const int out_features = (weights.Rank >= 2) ? static_cast<int>(weights.Sizes[1]) : 0;
    const int in_features = (weights.Rank >= 3) ? static_cast<int>(weights.Sizes[2]) : 0;
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

    // Get expert offsets
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
            throw std::runtime_error("moe_grouped_gemm: expert_offsets not found");
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

    MoeCompactInfo compact = host_offsets_ptr
        ? build_moe_compact_info_from_host(host_offsets_ptr, num_experts, weight_experts,
                                           layer_idx_any, "moe_grouped_gemm")
        : build_moe_compact_info(expert_offsets.get<int>(), num_experts, weight_experts,
                                 mRunState.MainStream, layer_idx_any, "moe_grouped_gemm");
    if (!host_offsets_ptr && !compact.host_offsets.empty()) {
        host_offsets_ptr = compact.host_offsets.data();
    }
    const int* active_ptr = compact.active_experts.empty() ? nullptr : compact.active_experts.data();
    const int num_active = compact.active_experts.empty() ? -1 : compact.num_active;
    const bool weight_is_compact = compact.weight_is_compact;

    // Output shape: [total_tokens, out_features]
    const long total_tokens = inp.Sizes[0];
    std::vector<long> out_shape = {total_tokens, static_cast<long>(out_features)};
    Tensor out = mRunState.temp_alloc(inp.DType, out_shape);
    mTemps.push_back(out);

    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(out, mRunState.MainStream);
    } else if (inp.DType == ETensorDType::BF16) {
        moe_grouped_gemm(out.get<nv_bfloat16>(),
                         inp.get<nv_bfloat16>(),
                         weights.get<nv_bfloat16>(),
                         expert_offsets.get<int>(),
                         num_experts, out_features, in_features,
                         mRunState.cublas_handle(), mRunState.MainStream,
                         host_offsets_ptr,
                         /*alpha=*/1.0f, /*beta=*/0.0f, EMMTranspose::TN,
                         active_ptr, weight_is_compact, num_active);
    } else {
        moe_grouped_gemm(out.get<float>(),
                         inp.get<float>(),
                         weights.get<float>(),
                         expert_offsets.get<int>(),
                         num_experts, out_features, in_features,
                         mRunState.cublas_handle(), mRunState.MainStream,
                         host_offsets_ptr,
                         /*alpha=*/1.0f, /*beta=*/0.0f, EMMTranspose::TN,
                         active_ptr, weight_is_compact, num_active);
    }

    mTensorMap[op.outputs[0].name] = out;
}

void CompiledExecutor::dispatch_moe_grouped_gemm_backward(const CompiledOp& op) {
    // Backward pass: compute input gradient
    // Inputs: d_out, inp, weights, scatter_indices
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& weights = resolve_tensor(op.inputs[2]);
    Tensor& scatter_indices = resolve_tensor(op.inputs[3]);
    (void)scatter_indices;
    (void)inp;

    const int num_experts = static_cast<int>(mConfig.NumExperts);
    const int out_features = (weights.Rank >= 2) ? static_cast<int>(weights.Sizes[1]) : 0;
    const int in_features = (weights.Rank >= 3) ? static_cast<int>(weights.Sizes[2]) : 0;
    const int weight_experts = (weights.Rank > 0) ? static_cast<int>(weights.Sizes[0]) : num_experts;

    int layer_idx_any = op.attrs.layer_idx;
    if (layer_idx_any < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[1].name;  // inp
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        std::string field;
        parse_block_param(name, layer_idx_any, field);
    }

    // Get expert offsets
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
            throw std::runtime_error("moe_grouped_gemm_backward: expert_offsets not found");
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

    MoeCompactInfo compact = host_offsets_ptr
        ? build_moe_compact_info_from_host(host_offsets_ptr, num_experts, weight_experts,
                                           layer_idx_any, "moe_grouped_gemm_backward")
        : build_moe_compact_info(expert_offsets.get<int>(), num_experts, weight_experts,
                                 mRunState.MainStream, layer_idx_any, "moe_grouped_gemm_backward");
    if (!host_offsets_ptr && !compact.host_offsets.empty()) {
        host_offsets_ptr = compact.host_offsets.data();
    }
    const int* active_ptr = compact.active_experts.empty() ? nullptr : compact.active_experts.data();
    const int num_active = compact.active_experts.empty() ? -1 : compact.num_active;
    const bool weight_is_compact = compact.weight_is_compact;

    // Input gradient shape: same as inp
    const long total_tokens = d_out.Sizes[0];
    std::vector<long> d_inp_shape = {total_tokens, static_cast<long>(in_features)};
    Tensor d_inp = mRunState.temp_alloc(d_out.DType, d_inp_shape);
    mTemps.push_back(d_inp);

    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(d_inp, mRunState.MainStream);
    } else if (d_out.DType == ETensorDType::BF16) {
        // Backward: d_inp = d_out @ weights (using NN mode for transposed operation)
        // Use moe_grouped_gemm_up_backward which computes: d_input = d_up @ weights^T
        // For generic case: d_inp[tokens, in_features] = d_out[tokens, out_features] @ W[E, out, in]
        // moe_grouped_gemm_up_backward expects: hidden_size=in_features, intermediate_size=out_features
        moe_grouped_gemm_up_backward(d_inp.get<nv_bfloat16>(),
                                     d_out.get<nv_bfloat16>(),
                                     weights.get<nv_bfloat16>(),
                                     expert_offsets.get<int>(),
                                     num_experts, in_features, out_features,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr,
                                     active_ptr, weight_is_compact, num_active);
    } else {
        moe_grouped_gemm_up_backward(d_inp.get<float>(),
                                     d_out.get<float>(),
                                     weights.get<float>(),
                                     expert_offsets.get<int>(),
                                     num_experts, in_features, out_features,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr,
                                     active_ptr, weight_is_compact, num_active);
    }

    mTensorMap[op.outputs[0].name] = d_inp;
}

}  // namespace dsl
