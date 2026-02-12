#include "runtime/dsl/compiled_ops.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

Tensor CompiledExecutor::resolve_moe_expert_offsets(const CompiledOp& op) {
    Tensor expert_offsets_view;
    Tensor* expert_offsets_ptr = nullptr;
    int layer_idx_any = op.attrs.layer_idx;
    if (layer_idx_any < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("d_", 0) == 0) {
            name.remove_prefix(2);
        }
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        std::string field;
        parse_block_param(name, layer_idx_any, field);
    }

    if (layer_idx_any >= 0) {
        // For the last MoE layer, prefer the global expert_offsets restored for backward.
        if (layer_idx_any == static_cast<int>(mConfig.NumLayers) - 1) {
            // Try flat vector first via pre-resolved offsets tensor ID, fall back to mirror
            Tensor* global_offsets = nullptr;
            if (op.attrs.moe_offsets_tensor_id >= 0 &&
                static_cast<std::size_t>(op.attrs.moe_offsets_tensor_id) < mTensors.size() &&
                mTensors[op.attrs.moe_offsets_tensor_id].Data) {
                global_offsets = &mTensors[op.attrs.moe_offsets_tensor_id];
            }
            // mTensors lookup is the only path (no mirror fallback)
            if (global_offsets && global_offsets->Data) {
                cudaPointerAttributes attr{};
                cudaError_t err = cudaPointerGetAttributes(&attr, global_offsets->Data);
                if (err == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
                    return *global_offsets;
                }
                cudaGetLastError();
            }
        }
        const std::string key = "blocks[" + std::to_string(layer_idx_any) + "].moe_expert_offsets";
        auto it_saved = mMoeSavedBuffers.find(key);
        if (it_saved != mMoeSavedBuffers.end() && it_saved->second != nullptr) {
            cudaPointerAttributes attr{};
            cudaError_t err = cudaPointerGetAttributes(&attr, it_saved->second);
            if (err == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
            expert_offsets_view.DType = ETensorDType::INT32;
            expert_offsets_view.Rank = 1;
            expert_offsets_view.Sizes[0] = static_cast<long>(mConfig.NumExperts + 1);
            expert_offsets_view.Data = static_cast<std::byte*>(it_saved->second);
            expert_offsets_ptr = &expert_offsets_view;
            } else {
                cudaGetLastError();
            }
        }
    }
    if (!expert_offsets_ptr) {
        Tensor* moe_offsets_ptr = nullptr;
        if (op.attrs.moe_offsets_tensor_id >= 0 &&
            static_cast<std::size_t>(op.attrs.moe_offsets_tensor_id) < mTensors.size() &&
            mTensors[op.attrs.moe_offsets_tensor_id].Data) {
            moe_offsets_ptr = &mTensors[op.attrs.moe_offsets_tensor_id];
        }
        if (!moe_offsets_ptr) {
            throw std::runtime_error("moe_expert_bias_add: expert_offsets not found");
        }
        if (moe_offsets_ptr->Data) {
            cudaPointerAttributes attr{};
            cudaError_t err = cudaPointerGetAttributes(&attr, moe_offsets_ptr->Data);
            if (err != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
                cudaGetLastError();
                throw std::runtime_error("moe_expert_bias_add: expert_offsets is not device memory");
            }
        }
        expert_offsets_ptr = moe_offsets_ptr;
    }
    if (expert_offsets_ptr->DType != ETensorDType::INT32) {
        throw std::runtime_error("moe_expert_bias_add: expert_offsets dtype is not INT32");
    }
    if (!expert_offsets_ptr->Data) {
        throw std::runtime_error("moe_expert_bias_add: expert_offsets has null data");
    }
    return *expert_offsets_ptr;
}

void CompiledExecutor::dispatch_moe_expert_bias_add(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& bias = resolve_tensor(op.inputs[1]);

    Tensor expert_offsets_tensor = resolve_moe_expert_offsets(op);
    const int* expert_offsets = expert_offsets_tensor.get<int>();

    const int num_experts = static_cast<int>(mConfig.NumExperts);
    const int hidden_size = static_cast<int>(inp.Sizes[1]);
    const int total_tokens = static_cast<int>(inp.Sizes[0]);

    std::vector<long> out_shape = {static_cast<long>(total_tokens), static_cast<long>(hidden_size)};
    Tensor out = mRunState.temp_alloc(inp.DType, out_shape);
    mTemps.push_back(out);

    if (inp.DType == ETensorDType::BF16) {
        moe_expert_bias_add_forward(out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(), bias.get<nv_bfloat16>(),
                                    expert_offsets, num_experts, hidden_size, total_tokens, mRunState.MainStream);
    } else if (inp.DType == ETensorDType::FP32) {
        moe_expert_bias_add_forward(out.get<float>(), inp.get<float>(), bias.get<float>(),
                                    expert_offsets, num_experts, hidden_size, total_tokens, mRunState.MainStream);
    } else {
        throw std::logic_error("moe_expert_bias_add: unsupported input dtype");
    }

    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_moe_expert_bias_add_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    // d_input = d_out (pass through)
    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        store_tensor(op.outputs[0], d_out);
    }

    if (op.outputs.size() < 2 || op.outputs[1].name.empty()) {
        return;
    }

    // GPT-OSS uses per-expert bias tensors; skip backward to avoid unstable CUDA errors for now.
    if (op.outputs[1].name.find("experts_") != std::string::npos &&
        op.outputs[1].name.find("_bias") != std::string::npos) {
        if (auto base = base_param_from_grad(op.outputs[1].name)) {
            bool grad_accum = false;
            if (Tensor* grad_tensor = mGrads.get_param_grad(*base, grad_accum)) {
                if (grad_tensor->Data) {
                    store_tensor(op.outputs[1], *grad_tensor);
                }
            }
        }
        return;
    }

    const int num_experts = static_cast<int>(mConfig.NumExperts);
    const int top_k = (mConfig.NumExpertsPerTok > 0) ? static_cast<int>(mConfig.NumExpertsPerTok) : 1;
    const int num_tokens = static_cast<int>(mB * mT);
    const int total_tokens = num_tokens * (top_k > 0 ? top_k : 1);
    if (total_tokens <= 0) {
        throw std::runtime_error("moe_expert_bias_add_backward: invalid total_tokens");
    }
    const long d_out_elems = static_cast<long>(d_out.nelem());
    if (d_out_elems % total_tokens != 0) {
        throw std::runtime_error("moe_expert_bias_add_backward: d_out shape mismatch vs total_tokens");
    }
    const int hidden_size = static_cast<int>(d_out_elems / total_tokens);

    if (mRunState.Stack.owns(d_out.Data) && !mRunState.Stack.is_live(d_out.Data)) {
        throw std::runtime_error("moe_expert_bias_add_backward: d_out points to dead stack memory");
    }

    Tensor expert_offsets_tensor = resolve_moe_expert_offsets(op);
    const int* expert_offsets = expert_offsets_tensor.get<int>();

    Tensor* grad_tensor = nullptr;
    bool grad_accum = false;
    if (!op.outputs[1].name.empty()) {
        if (auto base = base_param_from_grad(op.outputs[1].name)) {
            grad_tensor = mGrads.get_param_grad(*base, grad_accum);
        }
    }
    Tensor& d_bias = (grad_tensor && grad_tensor->Data) ? *grad_tensor : ensure_output_tensor(op.outputs[1]);
    if (grad_tensor && grad_tensor->Data) {
        store_tensor(op.outputs[1], d_bias);
    }
    bool accumulate = mAccumulateTensors.count(op.outputs[1].name) > 0;
    if (!accumulate && grad_accum) {
        accumulate = true;
    }
    if (!accumulate && !op.outputs[1].name.empty()) {
        if (auto base = base_param_from_grad(op.outputs[1].name)) {
            accumulate = mAccumulateTensors.count("d_" + *base) > 0;
        }
    }

    const long expected_bias_elems = static_cast<long>(num_experts) * hidden_size;
    if (d_bias.nelem() != expected_bias_elems) {
        throw std::runtime_error("moe_expert_bias_add_backward: d_bias shape mismatch");
    }

    // Note: expert_offsets are produced by moe_permute and expected to be valid here.

    const bool need_temp = (d_bias.DType != ETensorDType::FP32) || accumulate;
    Tensor d_bias_f32;
    if (need_temp) {
        d_bias_f32 = mRunState.temp_alloc(ETensorDType::FP32,
                                          {static_cast<long>(num_experts), static_cast<long>(hidden_size)});
        mTemps.push_back(d_bias_f32);
    }

    float* d_bias_ptr = need_temp ? d_bias_f32.get<float>() : d_bias.get<float>();

    if (d_out.DType == ETensorDType::BF16) {
        moe_expert_bias_add_backward(nullptr, d_bias_ptr, d_out.get<nv_bfloat16>(),
                                     expert_offsets, num_experts, hidden_size, total_tokens, mRunState.MainStream);
    } else if (d_out.DType == ETensorDType::FP32) {
        moe_expert_bias_add_backward(nullptr, d_bias_ptr, d_out.get<float>(),
                                     expert_offsets, num_experts, hidden_size, total_tokens, mRunState.MainStream);
    } else {
        throw std::logic_error("moe_expert_bias_add_backward: unsupported d_out dtype");
    }

    if (d_bias.DType == ETensorDType::FP32) {
        if (accumulate) {
            vector_add_sr(d_bias, d_bias, d_bias_f32, 1.0f,
                          static_cast<long>(d_bias.nelem()), 0, mRunState.MainStream);
        }
        return;
    }

    // Convert to output dtype (BF16) and accumulate if needed.
    auto shape_vec = [](const Tensor& t) {
        return std::vector<long>(t.Sizes.begin(), t.Sizes.begin() + t.Rank);
    };
    Tensor d_bias_cast = mRunState.temp_alloc(d_bias.DType, shape_vec(d_bias));
    mTemps.push_back(d_bias_cast);
    convert_dtype(d_bias_cast.get<nv_bfloat16>(), d_bias_f32.get<float>(),
                  d_bias_f32.nelem(), mRunState.MainStream);
    if (accumulate) {
        vector_add_sr(d_bias, d_bias, d_bias_cast, 1.0f,
                      static_cast<long>(d_bias.nelem()), 0, mRunState.MainStream);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_bias.Data, d_bias_cast.Data, d_bias.bytes(),
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }
}

}  // namespace dsl
