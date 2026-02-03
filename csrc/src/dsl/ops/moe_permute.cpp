#include "dsl/compiled_ops.h"

#include <vector>

#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_moe_permute(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& routing_indices = resolve_tensor(op.inputs[1]);
    Tensor& permuted = ensure_output_tensor(op.outputs[0]);
    Tensor& scatter_indices = ensure_output_tensor(op.outputs[1]);

    const int num_tokens = static_cast<int>(inp.Sizes[0]);
    const int hidden_size = static_cast<int>(inp.Sizes[1]);
    const int top_k = op.attrs.top_k;
    const int total_tokens = num_tokens * top_k;
    const int num_experts = static_cast<int>(mConfig.NumExperts);
    int layer_idx_any = op.attrs.layer_idx;
    std::string field_any;
    if (layer_idx_any < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        parse_block_param(name, layer_idx_any, field_any);
    }
    // Allocate temporary buffers for permutation indices
    // Use Stack.allocate for small buffers that can be freed at layer boundaries
    Tensor expert_counts = mRunState.Stack.allocate(ETensorDType::INT32, {num_experts}, "moe_expert_counts");
    Tensor expert_offsets = mRunState.Stack.allocate(ETensorDType::INT32, {num_experts + 1}, "moe_expert_offsets");
    Tensor expert_positions = mRunState.Stack.allocate(ETensorDType::INT32, {num_experts}, "moe_expert_positions");
    Tensor gather_indices = mRunState.Stack.allocate(ETensorDType::INT32, {total_tokens}, "moe_gather_indices");

    // Zero-initialize expert_positions before atomicAdd in build_indices
    // Stack memory is reused across forward passes and contains stale values
    fill_zero(expert_positions, mRunState.MainStream);

    // Compute expert counts
    moe_compute_expert_counts(expert_counts.get<int>(),
                              routing_indices.get<int>(),
                              num_tokens, top_k, num_experts, mRunState.MainStream);

    // Compute expert offsets (prefix sum)
    moe_compute_expert_offsets(expert_offsets.get<int>(),
                               expert_counts.get<int>(),
                               num_experts, mRunState.MainStream);

    // Build gather and scatter indices
    moe_build_indices(gather_indices.get<int>(),
                      scatter_indices.get<int>(),
                      routing_indices.get<int>(),
                      expert_offsets.get<int>(),
                      expert_positions.get<int>(),
                      num_tokens, top_k, num_experts, mRunState.MainStream);

    // Cache expert offsets on host for grouped GEMM fast path.
    if (num_experts > 0) {
        mMoEExpertOffsetsData.resize(static_cast<std::size_t>(num_experts + 1));
        CUDA_CHECK(cudaMemcpyAsync(mMoEExpertOffsetsData.data(),
                                   expert_offsets.get<int>(),
                                   static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
    }

    // Permute tokens
    if (inp.DType == ETensorDType::BF16) {
        moe_permute_tokens(permuted.get<nv_bfloat16>(),
                           inp.get<nv_bfloat16>(),
                           gather_indices.get<int>(),
                           total_tokens, num_tokens, hidden_size, top_k, mRunState.MainStream);
    } else {
        moe_permute_tokens(permuted.get<float>(),
                           inp.get<float>(),
                           gather_indices.get<int>(),
                           total_tokens, num_tokens, hidden_size, top_k, mRunState.MainStream);
    }

    // Persist per-layer routing buffers for backward (expert_offsets + gather_indices).
    {
        int layer_idx = op.attrs.layer_idx;
        std::string field;
        if (layer_idx < 0 && !op.inputs.empty()) {
            std::string_view name = op.inputs[0].name;
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx, field);
        }
        if (layer_idx >= 0) {
            auto save_buffer = [&](const std::string& suffix, const Tensor& src) {
                if (!src.Data) {
                    return;
                }
                const std::string key = "blocks[" + std::to_string(layer_idx) + "]." + suffix;
                const size_t bytes = src.bytes();
                if (bytes == 0) {
                    return;
                }
                auto buf_it = mMoESavedBuffers.find(key);
                if (buf_it == mMoESavedBuffers.end() || mMoESavedSizes[key] < bytes) {
                    if (buf_it != mMoESavedBuffers.end() && buf_it->second != nullptr) {
                        CUDA_CHECK(cudaFree(buf_it->second));
                    }
                    void* new_buffer = nullptr;
                    CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
                    mMoESavedBuffers[key] = new_buffer;
                    mMoESavedSizes[key] = bytes;
                }
                void* dst_buffer = mMoESavedBuffers[key];
                CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src.Data, bytes,
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));
            };
            save_buffer("moe_expert_offsets", expert_offsets);
            save_buffer("moe_gather_indices", gather_indices);
        }
    }

    // Store expert_offsets in scatter_indices output for later use
    // Note: scatter_indices tensor is already populated by moe_build_indices

    mTensorMap[op.outputs[0].name] = permuted;
    mTensorMap[op.outputs[1].name] = scatter_indices;
    // Store expert_offsets for use by grouped GEMM and unpermute
    // Note: expert_offsets lives on the stack; store for this layer in case we need it,
    // but grouped GEMM should prefer host offsets to avoid touching possibly-stale device memory.
    mTensorMap["moe_expert_offsets"] = expert_offsets;
    mTensorMap["moe_gather_indices"] = gather_indices;

    // Keep temps for later use
    mTemps.push_back(expert_counts);
    mTemps.push_back(expert_offsets);
    mTemps.push_back(expert_positions);
    mTemps.push_back(gather_indices);
}

void CompiledExecutor::dispatch_moe_permute_backward(const CompiledOp& op) {
    Tensor& d_permuted = resolve_tensor(op.inputs[0]);
    Tensor& gather_indices_saved = resolve_tensor(op.inputs[1]);  // Saved from forward
    Tensor& d_input = ensure_output_tensor(op.outputs[0]);

    // Prefer per-layer saved gather indices when available.
    Tensor* gather_indices = nullptr;
    Tensor gather_indices_view;
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
    if (layer_idx >= 0) {
        const std::string key = "blocks[" + std::to_string(layer_idx) + "].moe_gather_indices";
        auto it = mMoESavedBuffers.find(key);
        if (it != mMoESavedBuffers.end() && it->second != nullptr) {
            const int top_k = op.attrs.top_k > 0 ? op.attrs.top_k : 1;
            const int num_tokens = static_cast<int>(d_input.Sizes[0]);
            const int total_tokens = num_tokens * top_k;
            gather_indices_view.DType = ETensorDType::INT32;
            gather_indices_view.Rank = 1;
            gather_indices_view.Sizes[0] = total_tokens;
            gather_indices_view.Data = static_cast<std::byte*>(it->second);
            gather_indices = &gather_indices_view;
        }
    }
    if (!gather_indices) {
        auto it = mTensorMap.find("moe_gather_indices");
        gather_indices = (it != mTensorMap.end()) ? &it->second : &gather_indices_saved;
    }

    const int top_k = op.attrs.top_k;
    const int num_tokens = static_cast<int>(d_input.Sizes[0]);
    const int hidden_size = static_cast<int>(d_input.Sizes[1]);
    const int total_tokens = num_tokens * top_k;
    if (d_permuted.DType == ETensorDType::BF16) {
        fill_zero(d_input, mRunState.MainStream);
        moe_permute_backward(d_input.get<nv_bfloat16>(),
                             d_permuted.get<nv_bfloat16>(),
                             gather_indices->get<int>(),
                             total_tokens, num_tokens, hidden_size, top_k,
                             mRunState.MainStream);
    } else {
        fill_zero(d_input, mRunState.MainStream);
        moe_permute_backward(d_input.get<float>(),
                             d_permuted.get<float>(),
                             gather_indices->get<int>(),
                             total_tokens, num_tokens, hidden_size, top_k,
                             mRunState.MainStream);
    }

    mTensorMap[op.outputs[0].name] = d_input;
}


}  // namespace dsl
