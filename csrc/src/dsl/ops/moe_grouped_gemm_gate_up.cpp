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

void CompiledExecutor::dispatch_moe_grouped_gemm_gate_up(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& weights = resolve_tensor(op.inputs[1]);  // Parameter name resolved by graph compiler
    Tensor& scatter_indices = resolve_tensor(op.inputs[2]);
    const int num_tokens = static_cast<int>(mB * mT);
    int top_k = op.attrs.top_k;
    if (top_k <= 0 && num_tokens > 0 && inp.Rank == 2) {
        top_k = static_cast<int>(inp.Sizes[0] / num_tokens);
    }
    if (top_k <= 0) {
        top_k = 1;
    }

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
            throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets not found");
        }
        expert_offsets_ptr = &it->second;
    }
    Tensor& expert_offsets = *expert_offsets_ptr;
    if (expert_offsets.DType != ETensorDType::INT32) {
        throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets dtype is not INT32");
    }
    if (!expert_offsets.Data) {
        throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets has null data");
    }
    auto ensure_device_ptr = [&](const Tensor& t, const char* name) {
        if (!t.Data) {
            return;
        }
        cudaPointerAttributes attr{};
        cudaError_t err = cudaPointerGetAttributes(&attr, t.Data);
        if (err != cudaSuccess) {
            cudaGetLastError();
            throw std::runtime_error("moe_grouped_gemm_gate_up: pointer attributes unavailable");
        }
#if CUDART_VERSION >= 10000
        const int mem_type = static_cast<int>(attr.type);
#else
        const int mem_type = static_cast<int>(attr.memoryType);
#endif
        if (mem_type == cudaMemoryTypeHost) {
            throw std::runtime_error("moe_grouped_gemm_gate_up: pointer on host memory");
        }
    };

    const int num_experts = static_cast<int>(mConfig.NumExperts);
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
    // Use MoeIntermediateSize for MoE models (may differ from IntermediateSize)
    const int intermediate_size = (mConfig.MoeIntermediateSize > 0)
        ? static_cast<int>(mConfig.MoeIntermediateSize)
        : static_cast<int>(mConfig.IntermediateSize);
    const int weight_experts = (weights.Rank > 0) ? static_cast<int>(weights.Sizes[0]) : num_experts;
    const bool offsets_owned = mRunState.Stack.owns(expert_offsets.Data);
    if (offsets_owned && !mRunState.Stack.is_live(expert_offsets.Data)) {
        throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets pointer is not live");
    }
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
        int oob = 0;
        const int total_tokens = static_cast<int>(inp.Sizes[0]);
        for (int e = 1; e <= num_experts; ++e) {
            if (host_offsets_ptr[e] < host_offsets_ptr[e - 1]) {
                bad++;
            }
            if (host_offsets_ptr[e] < 0 || host_offsets_ptr[e] > total_tokens) {
                oob++;
            }
        }
        if (bad > 0 || oob > 0 || last != total_tokens) {
            throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets invalid");
        }
    }

    // Refresh MoE expert weights for this layer using the current routing offsets.
    if (host_offsets_ptr && layer_idx_any >= 0) {
        (void)refresh_moe_experts_if_needed(layer_idx_any,
                                            host_offsets_ptr,
                                            num_experts,
                                            mWeights,
                                            mRunState.MainStream);
    }

    MoeCompactInfo compact = host_offsets_ptr
        ? build_moe_compact_info_from_host(host_offsets_ptr,
                                           num_experts,
                                           weight_experts,
                                           layer_idx_any,
                                           "moe_grouped_gemm_gate_up")
        : build_moe_compact_info(expert_offsets.get<int>(),
                                 num_experts,
                                 weight_experts,
                                 mRunState.MainStream,
                                 layer_idx_any,
                                 "moe_grouped_gemm_gate_up");
    if (!host_offsets_ptr && !compact.host_offsets.empty()) {
        host_offsets_ptr = compact.host_offsets.data();
    }
    const int* active_ptr = compact.active_experts.empty() ? nullptr : compact.active_experts.data();
    const int num_active = compact.active_experts.empty() ? -1 : compact.num_active;
    const bool weight_is_compact = compact.weight_is_compact;

    // MoE output shape is dynamic: [total_tokens, 2 * intermediate_size]
    // total_tokens = inp.Sizes[0] (permuted token count)
    // Allocate output with correct runtime shape
    const long total_tokens = inp.Sizes[0];
    const long gate_up_dim = 2 * intermediate_size;
    std::vector<long> out_shape = {total_tokens, gate_up_dim};
    Tensor out = mRunState.temp_alloc(inp.DType, out_shape);
    mTemps.push_back(out);
    ensure_device_ptr(inp, "moe_input");
    ensure_device_ptr(weights, "moe_gate_up_weights");
    ensure_device_ptr(out, "moe_gate_up_out");
    ensure_device_ptr(expert_offsets, "moe_expert_offsets");

    if (host_offsets_ptr) {
        const std::size_t in_elems = static_cast<std::size_t>(inp.nelem());
        const std::size_t out_elems = static_cast<std::size_t>(out.nelem());
        const std::size_t w_elems = static_cast<std::size_t>(weights.nelem());
        const std::size_t in_stride = static_cast<std::size_t>(hidden_size);
        const std::size_t out_stride = static_cast<std::size_t>(gate_up_dim);
        const std::size_t w_stride = static_cast<std::size_t>(gate_up_dim) * static_cast<std::size_t>(hidden_size);
        const int n_active = (num_active > 0) ? num_active : num_experts;
        for (int e = 0; e < n_active; ++e) {
            const int global_idx = active_ptr ? active_ptr[e] : e;
            if (global_idx < 0 || global_idx >= num_experts) {
                throw std::runtime_error("moe_grouped_gemm_gate_up: active expert index out of range");
            }
            const int start = host_offsets_ptr[global_idx];
            const int end = host_offsets_ptr[global_idx + 1];
            const int tokens_e = end - start;
            if (tokens_e <= 0) {
                continue;
            }
            const std::size_t in_offset = static_cast<std::size_t>(start) * in_stride;
            const std::size_t out_offset = static_cast<std::size_t>(start) * out_stride;
            if (in_offset + static_cast<std::size_t>(tokens_e) * in_stride > in_elems) {
                throw std::runtime_error("moe_grouped_gemm_gate_up: input pointer out of range");
            }
            if (out_offset + static_cast<std::size_t>(tokens_e) * out_stride > out_elems) {
                throw std::runtime_error("moe_grouped_gemm_gate_up: output pointer out of range");
            }
            const int weight_idx = weight_is_compact ? e : global_idx;
            if (weight_idx < 0 || weight_idx >= weight_experts) {
                throw std::runtime_error("moe_grouped_gemm_gate_up: weight index out of range");
            }
            const std::size_t w_offset = static_cast<std::size_t>(weight_idx) * w_stride;
            if (w_offset + w_stride > w_elems) {
                throw std::runtime_error("moe_grouped_gemm_gate_up: weight pointer out of range");
            }
        }
    }
    // Optional debug: zero output to detect unwritten rows.
    static int moe_zero_gate_up = -1;
    if (moe_zero_gate_up < 0) {
        moe_zero_gate_up = (std::getenv("SUROGATE_MOE_ZERO_OUT") != nullptr) ? 1 : 0;
    }
    if (moe_zero_gate_up) {
        fill_zero(out, mRunState.MainStream);
    }

    // Use weights dtype to determine compute precision (QLoRA may return FP32 dequantized weights)
    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(out, mRunState.MainStream);
    } else if (weights.DType == ETensorDType::BF16) {
        moe_grouped_gemm_gate_up(out.get<nv_bfloat16>(),
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
        moe_grouped_gemm_gate_up(out.get<float>(),
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

void CompiledExecutor::dispatch_moe_grouped_gemm_gate_up_backward(const CompiledOp& op) {
    Tensor& d_gate_up = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& weights = resolve_tensor(op.inputs[2]);
    Tensor& d_input = ensure_output_tensor(op.outputs[0]);

    // Get expert offsets from stored state (per-layer when available).
    Tensor* expert_offsets_ptr = nullptr;
    Tensor expert_offsets_view;
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
        const std::string key = "blocks[" + std::to_string(layer_idx) + "].moe_expert_offsets";
        auto it = mMoESavedBuffers.find(key);
        if (it != mMoESavedBuffers.end() && it->second != nullptr) {
            expert_offsets_view.DType = ETensorDType::INT32;
            expert_offsets_view.Rank = 1;
            expert_offsets_view.Sizes[0] = static_cast<long>(mConfig.NumExperts + 1);
            expert_offsets_view.Data = static_cast<std::byte*>(it->second);
            expert_offsets_ptr = &expert_offsets_view;
        }
    }
    if (!expert_offsets_ptr) {
        auto it = mTensorMap.find("moe_expert_offsets");
        if (it == mTensorMap.end()) {
            throw std::runtime_error("moe_grouped_gemm_gate_up_backward: expert_offsets not found");
        }
        expert_offsets_ptr = &it->second;
    }

    // Use the persistent buffer directly instead of tensorMap
    const int* offsets_ptr = static_cast<const int*>(mMoEExpertOffsetsGPU);
    (void)offsets_ptr;  // Used by kernel through expert_offsets

    const int num_experts = static_cast<int>(mConfig.NumExperts);
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
    // Use MoeIntermediateSize for MoE models (may differ from IntermediateSize)
    const int intermediate_size = (mConfig.MoeIntermediateSize > 0)
        ? static_cast<int>(mConfig.MoeIntermediateSize)
        : static_cast<int>(mConfig.IntermediateSize);

    const int weight_experts = static_cast<int>(weights.Sizes[0]);
    MoeCompactInfo compact = build_moe_compact_info(expert_offsets_ptr->get<int>(),
                                                    num_experts,
                                                    weight_experts,
                                                    mRunState.MainStream,
                                                    layer_idx,
                                                    "moe_grouped_gemm_gate_up_backward");
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
        // Fallback: copy offsets to host for selection.
        std::vector<int> host_offsets_fallback(static_cast<std::size_t>(num_experts + 1), 0);
        CUDA_CHECK(cudaMemcpyAsync(host_offsets_fallback.data(),
                                   expert_offsets_ptr->get<int>(),
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
    } else if (d_gate_up.DType == ETensorDType::BF16) {
        moe_grouped_gemm_gate_up_backward(d_input.get<nv_bfloat16>(),
                                          d_gate_up.get<nv_bfloat16>(),
                                          weights.get<nv_bfloat16>(),
                                          expert_offsets_ptr->get<int>(),
                                          num_experts, hidden_size, intermediate_size,
                                          mRunState.cublas_handle(), mRunState.MainStream,
                                          host_offsets_ptr,
                                          active_ptr,
                                          weight_is_compact,
                                          num_active);
    } else {
        moe_grouped_gemm_gate_up_backward(d_input.get<float>(),
                                          d_gate_up.get<float>(),
                                          weights.get<float>(),
                                          expert_offsets_ptr->get<int>(),
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
