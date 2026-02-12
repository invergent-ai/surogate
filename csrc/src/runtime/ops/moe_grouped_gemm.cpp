// MoE Grouped GEMM operation (generic version without fused activation)
// Used for Nemotron-H MoE blocks that use relu2 activation instead of swiglu

#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "runtime/lora/lora_config.h"
#include "runtime/lora/lora_grads_manager.h"
#include "runtime/lora/lora_run_state.h"
#include "runtime/lora/lora_weights_manager.h"

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
        auto it_saved = mMoeSavedBuffers.find(key);
        if (it_saved != mMoeSavedBuffers.end() && it_saved->second != nullptr) {
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
    const int* expert_offsets_data = expert_offsets.get<int>();

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
        : build_moe_compact_info(expert_offsets_data, num_experts, weight_experts,
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
                         expert_offsets_data,
                         num_experts, out_features, in_features,
                         mRunState.cublas_handle(), mRunState.MainStream,
                         host_offsets_ptr,
                         /*alpha=*/1.0f, /*beta=*/0.0f, EMMTranspose::TN,
                         active_ptr, weight_is_compact, num_active);
    } else {
        moe_grouped_gemm(out.get<float>(),
                         inp.get<float>(),
                         weights.get<float>(),
                         expert_offsets_data,
                         num_experts, out_features, in_features,
                         mRunState.cublas_handle(), mRunState.MainStream,
                         host_offsets_ptr,
                         /*alpha=*/1.0f, /*beta=*/0.0f, EMMTranspose::TN,
                         active_ptr, weight_is_compact, num_active);
    }

    // Apply grouped MoE LoRA (up projection) when enabled.
    if (mLoRAConfig && mLoRAWeights && mLoRARunState &&
        mLoRAConfig->enabled() && mLoRAWeights->enabled() &&
        layer_idx_any >= 0) {
        auto& lora_block = mLoRAWeights->get_block(layer_idx_any, mRunState.MainStream);
        if (lora_block.moe.use_grouped && lora_block.moe.grouped.up.has_value() &&
            lora_block.moe.grouped.up->has_value()) {
            const auto& lora_up = *lora_block.moe.grouped.up;
            const int rank = mLoRAConfig->rank;
            const float scaling = mLoRAConfig->scaling();
            const float dropout = mLoRAConfig->dropout;
            const bool training = mLoRARunState->is_training;
            const long total_tokens_l = total_tokens;
            const int total_tokens_i = static_cast<int>(total_tokens_l);

            auto get_dropout_seed = [&](int proj_type) -> unsigned int {
                return mLoRARunState->dropout_base_seed
                       + static_cast<unsigned int>(layer_idx_any) * 1000000u
                       + static_cast<unsigned int>(proj_type) * 100000u
                       + static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;
            };

            auto view_or_temp = [&](Tensor& buf, long rows, long cols) -> Tensor {
                const long need = rows * cols;
                if (!buf.Data || buf.DType != out.DType || buf.nelem() < need) {
                    Tensor tmp = mRunState.temp_alloc(out.DType, {rows, cols});
                    mTemps.push_back(tmp);
                    return tmp;
                }
                Tensor view = buf;
                view.DType = out.DType;
                view.Rank = 2;
                view.Sizes[0] = rows;
                view.Sizes[1] = cols;
                for (int i = 2; i < MAX_TENSOR_DIM; ++i) view.Sizes[i] = 1;
                return view;
            };

            auto dispatch_grouped_gemm = [&](Tensor& out_t, const Tensor& in_t, const Tensor& weight_t,
                                             int M, int K, float alpha, float beta, EMMTranspose mode) {
                if (in_t.DType != weight_t.DType || in_t.DType != out_t.DType) {
                    std::string msg = "MoE LoRA: dtype mismatch between activation and LoRA weights. "
                                      "Set lora_dtype='bf16' in your config to match activation dtype.";
                    throw std::runtime_error(msg);
                }
                if (in_t.DType == ETensorDType::BF16) {
                    moe_grouped_gemm(out_t.get<nv_bfloat16>(), in_t.get<nv_bfloat16>(), weight_t.get<nv_bfloat16>(),
                                     expert_offsets_data, num_experts, M, K,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr, alpha, beta, mode, active_ptr,
                                     /*weight_is_compact=*/false, num_active);
                } else {
                    moe_grouped_gemm(out_t.get<float>(), in_t.get<float>(), weight_t.get<float>(),
                                     expert_offsets_data, num_experts, M, K,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr, alpha, beta, mode, active_ptr,
                                     /*weight_is_compact=*/false, num_active);
                }
            };

            auto scale_and_dropout = [&](Tensor& t, unsigned int seed) {
                if (training && dropout > 0.0f) {
                    lora_dropout_scale(t, dropout, seed, mRunState.MainStream);
                }
                if (scaling != 1.0f) {
                    vector_add_sr(t, t, t, 0.5f * scaling, t.nelem(), /*seed=*/0, mRunState.MainStream);
                }
            };

            if (total_tokens_i > 0 && rank > 0) {
                Tensor lora_intermediate = view_or_temp(mLoRARunState->moe_lora_intermediate1, total_tokens_l, rank);
                dispatch_grouped_gemm(lora_intermediate, inp, lora_up.A,
                                      rank, in_features, 1.0f, 0.0f, EMMTranspose::TN);
                scale_and_dropout(lora_intermediate, get_dropout_seed(4));
                dispatch_grouped_gemm(out, lora_intermediate, lora_up.B,
                                      out_features, rank, 1.0f, 1.0f, EMMTranspose::TN);
            }
        }
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
        auto it_saved = mMoeSavedBuffers.find(key);
        if (it_saved != mMoeSavedBuffers.end() && it_saved->second != nullptr) {
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
    const int* expert_offsets_data = expert_offsets.get<int>();

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
        : build_moe_compact_info(expert_offsets_data, num_experts, weight_experts,
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
                                     expert_offsets_data,
                                     num_experts, in_features, out_features,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr,
                                     active_ptr, weight_is_compact, num_active);
    } else {
        moe_grouped_gemm_up_backward(d_inp.get<float>(),
                                     d_out.get<float>(),
                                     weights.get<float>(),
                                     expert_offsets_data,
                                     num_experts, in_features, out_features,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr,
                                     active_ptr, weight_is_compact, num_active);
    }

    // Apply grouped MoE LoRA backward (up projection) when enabled.
    if (mLoRAConfig && mLoRAWeights && mLoRARunState &&
        mLoRAConfig->enabled() && mLoRAWeights->enabled() &&
        layer_idx_any >= 0) {
        auto& lora_block = mLoRAWeights->get_block(layer_idx_any, mRunState.MainStream);
        if (lora_block.moe.use_grouped && lora_block.moe.grouped.up.has_value() &&
            lora_block.moe.grouped.up->has_value()) {
            const auto& lora_up = *lora_block.moe.grouped.up;
            const int rank = mLoRAConfig->rank;
            const float scaling = mLoRAConfig->scaling();
            const float dropout = mLoRAConfig->dropout;
            const bool training = mLoRARunState->is_training;
            const long total_tokens_l = d_out.Sizes[0];
            const int total_tokens_i = static_cast<int>(total_tokens_l);

            auto get_dropout_seed = [&](int proj_type) -> unsigned int {
                return mLoRARunState->dropout_base_seed
                       + static_cast<unsigned int>(layer_idx_any) * 1000000u
                       + static_cast<unsigned int>(proj_type) * 100000u
                       + static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;
            };

            auto view_or_temp = [&](Tensor& buf, long rows, long cols) -> Tensor {
                const long need = rows * cols;
                if (!buf.Data || buf.DType != d_out.DType || buf.nelem() < need) {
                    Tensor tmp = mRunState.temp_alloc(d_out.DType, {rows, cols});
                    mTemps.push_back(tmp);
                    return tmp;
                }
                Tensor view = buf;
                view.DType = d_out.DType;
                view.Rank = 2;
                view.Sizes[0] = rows;
                view.Sizes[1] = cols;
                for (int i = 2; i < MAX_TENSOR_DIM; ++i) view.Sizes[i] = 1;
                return view;
            };

            auto dispatch_grouped_gemm = [&](Tensor& out_t, const Tensor& in_t, const Tensor& weight_t,
                                             int M, int K, float alpha, float beta, EMMTranspose mode) {
                if (in_t.DType != weight_t.DType || in_t.DType != out_t.DType) {
                    std::string msg = "MoE LoRA backward: dtype mismatch between activation and LoRA weights. "
                                      "Set lora_dtype='bf16' in your config to match activation dtype.";
                    throw std::runtime_error(msg);
                }
                if (in_t.DType == ETensorDType::BF16) {
                    moe_grouped_gemm(out_t.get<nv_bfloat16>(), in_t.get<nv_bfloat16>(), weight_t.get<nv_bfloat16>(),
                                     expert_offsets_data, num_experts, M, K,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr, alpha, beta, mode, active_ptr,
                                     /*weight_is_compact=*/false, num_active);
                } else {
                    moe_grouped_gemm(out_t.get<float>(), in_t.get<float>(), weight_t.get<float>(),
                                     expert_offsets_data, num_experts, M, K,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr, alpha, beta, mode, active_ptr,
                                     /*weight_is_compact=*/false, num_active);
                }
            };

            auto dispatch_weight_grad = [&](Tensor& d_weight, const Tensor& grad_output, const Tensor& in,
                                            int M, int N, float beta) {
                if (grad_output.DType != in.DType) {
                    throw std::runtime_error("MoE LoRA backward: grad/output dtype mismatch.");
                }
                if (grad_output.DType == ETensorDType::BF16) {
                    if (d_weight.DType != ETensorDType::BF16) {
                        throw std::runtime_error("MoE LoRA backward: lora_dtype=fp32 with bf16 activations not supported. "
                                                 "Set lora_dtype='bf16' in your config.");
                    }
                    moe_grouped_gemm_weight_grad(d_weight.get<nv_bfloat16>(),
                                                 grad_output.get<nv_bfloat16>(),
                                                 in.get<nv_bfloat16>(),
                                                 expert_offsets_data, num_experts, M, N,
                                                 mRunState.cublas_handle(), mRunState.MainStream,
                                                 host_offsets_ptr, /*alpha=*/1.0f, beta,
                                                 active_ptr, /*weight_is_compact=*/false, num_active);
                } else {
                    if (d_weight.DType != ETensorDType::FP32) {
                        throw std::runtime_error("MoE LoRA backward: dtype mismatch in weight gradients.");
                    }
                    moe_grouped_gemm_weight_grad(d_weight.get<float>(),
                                                 grad_output.get<float>(),
                                                 in.get<float>(),
                                                 expert_offsets_data, num_experts, M, N,
                                                 mRunState.cublas_handle(), mRunState.MainStream,
                                                 host_offsets_ptr, /*alpha=*/1.0f, beta,
                                                 active_ptr, /*weight_is_compact=*/false, num_active);
                }
            };

            auto scale_and_dropout = [&](Tensor& t, unsigned int seed) {
                if (training && dropout > 0.0f) {
                    lora_dropout_scale(t, dropout, seed, mRunState.MainStream);
                }
                if (scaling != 1.0f) {
                    vector_add_sr(t, t, t, 0.5f * scaling, t.nelem(), /*seed=*/0, mRunState.MainStream);
                }
            };

            modules::LoRABlockWeights<Tensor>* lora_grads = nullptr;
            bool lora_accum = false;
            if (mLoRAGrads && mComm) {
                lora_grads = &mLoRAGrads->get_block_full(layer_idx_any, mRunState.MainStream, *mComm, lora_accum);
            }
            const float grad_beta = lora_accum ? 1.0f : 0.0f;

            if (total_tokens_i > 0 && rank > 0) {
                Tensor lora_intermediate = view_or_temp(mLoRARunState->moe_lora_intermediate1, total_tokens_l, rank);
                const unsigned int seed_up = get_dropout_seed(4);

                // dB: intermediate = x @ A^T
                dispatch_grouped_gemm(lora_intermediate, inp, lora_up.A,
                                      rank, in_features, 1.0f, 0.0f, EMMTranspose::TN);
                scale_and_dropout(lora_intermediate, seed_up);
                if (lora_grads && lora_grads->moe.grouped.up.has_value()) {
                    dispatch_weight_grad(lora_grads->moe.grouped.up->B, d_out, lora_intermediate,
                                         out_features, rank, grad_beta);
                }

                // intermediate = d_out @ B
                dispatch_grouped_gemm(lora_intermediate, d_out, lora_up.B,
                                      rank, out_features, 1.0f, 0.0f, EMMTranspose::NN);
                scale_and_dropout(lora_intermediate, seed_up);
                if (lora_grads && lora_grads->moe.grouped.up.has_value()) {
                    dispatch_weight_grad(lora_grads->moe.grouped.up->A, lora_intermediate, inp,
                                         rank, in_features, grad_beta);
                }

                // d_input += intermediate @ A
                dispatch_grouped_gemm(d_inp, lora_intermediate, lora_up.A,
                                      in_features, rank, 1.0f, 1.0f, EMMTranspose::NN);
            }
        }
    }

    mTensorMap[op.outputs[0].name] = d_inp;
}

}  // namespace dsl
