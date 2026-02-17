#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "recipes/recipe.h"
#include "utilities/dtype.h"
#include "runtime/lora/lora_config.h"
#include "runtime/lora/lora_grads_manager.h"
#include "runtime/lora/lora_run_state.h"
#include "runtime/lora/lora_weights_manager.h"
#include "utilities/tensor.h"

namespace dsl {

void CompiledExecutor::dispatch_moe_grouped_gemm_down(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor weights = resolve_tensor(op.inputs[1]);  // Parameter name resolved by graph compiler (copy for LLEP override)
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

    int num_experts = static_cast<int>(mConfig.NumLocalExperts);
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
    // Use MoeIntermediateSize for MoE models (may differ from IntermediateSize)
    const int intermediate_size = (mConfig.MoeIntermediateSize > 0)
        ? static_cast<int>(mConfig.MoeIntermediateSize)
        : static_cast<int>(mConfig.IntermediateSize);
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
        auto it_saved = mMoeSavedBuffers.find(key);
        if (it_saved != mMoeSavedBuffers.end() && it_saved->second != nullptr) {
            expert_offsets_view.DType = ETensorDType::INT32;
            expert_offsets_view.Rank = 1;
            expert_offsets_view.Sizes[0] = static_cast<long>(mConfig.NumLocalExperts + 1);
            expert_offsets_view.Data = static_cast<std::byte*>(it_saved->second);
            expert_offsets_ptr = &expert_offsets_view;
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
            throw std::runtime_error("moe_grouped_gemm_down: expert_offsets not found");
        }
        expert_offsets_ptr = moe_offsets_ptr;
    }
    Tensor& expert_offsets = *expert_offsets_ptr;

    // LLEP per-expert weight pointer override: when LLEP is active, use per-expert
    // pointers (native dequant buffer + foreign P2P receive) instead of contiguous weights.
    bool is_llep_active = false;
    const void* const* llep_weight_ptrs = nullptr;
    {
        auto llep_it = mLLEPStates.find(layer_idx_any);
        if (llep_it != mLLEPStates.end() && llep_it->second.active) {
            auto& llep = llep_it->second;
            num_experts = llep.num_merged_experts;
            expert_offsets_view.Sizes[0] = static_cast<long>(num_experts + 1);
            llep_weight_ptrs = llep.down_weight_ptrs.data();
            is_llep_active = true;
        }
    }

    const int weight_experts = llep_weight_ptrs ? num_experts
        : ((weights.Rank > 0) ? static_cast<int>(weights.Sizes[0]) : num_experts);
    const int* host_offsets_ptr = nullptr;
    if (num_experts > 0 && expert_offsets.Data) {
        // Use cached host offsets (populated by dispatch_moe_permute or ep_dispatch for this layer).
        host_offsets_ptr = get_or_sync_moe_host_offsets(
            layer_idx_any, expert_offsets.get<int>(), num_experts);
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
    } else if (mRecipe && inp.DType == ETensorDType::BF16 && !weight_is_compact && !is_llep_active) {
        // Recipe-driven MoE GEMM via cuDNN FE (skip when LLEP active — cuDNN
        // crashes with variable merged expert counts; cuBLAS per-expert is safe)
        // down weight is (E, C, D) → N=C, K=D
        modules::MoeMatmulContext ctx;
        ctx.out = out.get<nv_bfloat16>();
        ctx.inp = inp.get<nv_bfloat16>();
        ctx.weights = weights.get<nv_bfloat16>();
        ctx.expert_offsets = expert_offsets.get<int>();
        ctx.num_experts = num_experts;
        ctx.N = hidden_size;
        ctx.K = intermediate_size;
        ctx.total_tokens = static_cast<int>(total_tokens);
        ctx.run_state = &mRunState;
        ctx.cudnn_handle = mRunState.CudnnHandle;
        ctx.workspace = mRunState.CuBlasWorkspace.get<std::byte>();
        ctx.workspace_size = mRunState.CuBlasWorkspace.bytes();
        ctx.stream = mRunState.MainStream;
        mRecipe->forward_moe_matmul(ctx);
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
                              num_active,
                              llep_weight_ptrs);
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
                              num_active,
                              llep_weight_ptrs);
    }

    // Apply grouped MoE LoRA (down projection) when enabled.
    if (mLoRAConfig && mLoRAWeights && mLoRARunState &&
        mLoRAConfig->enabled() && mLoRAWeights->enabled() &&
        layer_idx_any >= 0) {
        auto& lora_block = mLoRAWeights->get_block(layer_idx_any, mRunState.MainStream);
        if (lora_block.moe.use_grouped) {
            // When LLEP is active, use merged LoRA tensors
            const auto* down_ptr = lora_block.moe.grouped.down.has_value()
                ? &(*lora_block.moe.grouped.down) : nullptr;
            {
                auto llep_lora_it = mLLEPStates.find(layer_idx_any);
                if (llep_lora_it != mLLEPStates.end() && llep_lora_it->second.active
                    && llep_lora_it->second.has_merged_lora
                    && llep_lora_it->second.merged_lora.down.has_value()) {
                    down_ptr = &(*llep_lora_it->second.merged_lora.down);
                }
            }
            if (down_ptr && down_ptr->has_value()) {
            const auto& lora_down = *down_ptr;
            const int rank = mLoRAConfig->rank;
            const float scaling = mLoRAConfig->scaling();
            const float dropout = mLoRAConfig->dropout;
            const bool training = mLoRARunState->is_training;
            const long total_tokens_l = total_tokens;
            const int total_tokens_i = static_cast<int>(total_tokens_l);
            const int micro_step = mLoRARunState->micro_step;

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
                                     expert_offsets.get<int>(), num_experts, M, K,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr, alpha, beta, mode, active_ptr,
                                     /*weight_is_compact=*/false, num_active);
                } else {
                    moe_grouped_gemm(out_t.get<float>(), in_t.get<float>(), weight_t.get<float>(),
                                     expert_offsets.get<int>(), num_experts, M, K,
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
                dispatch_grouped_gemm(lora_intermediate, inp, lora_down.A,
                                      rank, intermediate_size, 1.0f, 0.0f, EMMTranspose::TN);
                scale_and_dropout(lora_intermediate, get_dropout_seed(6));
                dispatch_grouped_gemm(out, lora_intermediate, lora_down.B,
                                      hidden_size, rank, 1.0f, 1.0f, EMMTranspose::TN);
            }
            }  // if (down_ptr && down_ptr->has_value())
        }  // if (use_grouped)
    }

    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_moe_grouped_gemm_down_backward(const CompiledOp& op) {
    Tensor& d_output = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor weights = resolve_tensor(op.inputs[2]);  // copy for LLEP override
    // Output shape for MoE backward can be dynamic (B*T*K, M). Use input shape when
    // compiled shape is missing or mismatched to avoid incorrect allocations.
    auto needs_dynamic = [&](const TensorRef& out_ref, const Tensor& in) -> bool {
        if (out_ref.shape.empty()) return true;
        if (in.Rank <= 0) return false;
        if (static_cast<int>(out_ref.shape.size()) != in.Rank) return true;
        for (int i = 0; i < in.Rank; ++i) {
            if (out_ref.shape[i] != in.Sizes[i]) return true;
        }
        return false;
    };
    Tensor d_input_local;
    Tensor* d_input_ptr = nullptr;
    if (needs_dynamic(op.outputs[0], inp)) {
        std::vector<long> shape(inp.Sizes.begin(), inp.Sizes.begin() + inp.Rank);
        d_input_local = mRunState.temp_alloc(inp.DType, shape);
        mTemps.push_back(d_input_local);
        store_tensor(op.outputs[0], d_input_local);
        d_input_ptr = &mTensors[op.outputs[0].tensor_id];
    } else {
        d_input_ptr = &ensure_output_tensor(op.outputs[0]);
    }
    Tensor& d_input = *d_input_ptr;
    if (d_input_ptr->Device == -1 && mRunState.Stack.owns(d_input_ptr->Data)) {
        d_input_ptr->Device = mRunState.Stack.device_id();
    }
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


    // For EP, derive num_experts from the forward-cached offsets.
    // LLEP may change num_merged per layer.
    int num_experts_for_offsets = static_cast<int>(mConfig.NumLocalExperts);
    if (mOptions.EPSize > 1 && layer_idx >= 0) {
        auto ci = mMoEHostOffsetsCache.find(layer_idx);
        if (ci != mMoEHostOffsetsCache.end() && ci->second.size() > 1) {
            num_experts_for_offsets = static_cast<int>(ci->second.size()) - 1;
        }
    }

    // Use per-layer expert_offsets when available; fall back to global buffer.
    const int* expert_offsets_ptr = nullptr;
    Tensor expert_offsets_view;
    if (layer_idx >= 0) {
        const std::string key = "blocks[" + std::to_string(layer_idx) + "].moe_expert_offsets";
        auto it = mMoeSavedBuffers.find(key);
        if (it != mMoeSavedBuffers.end() && it->second != nullptr) {
            expert_offsets_view.DType = ETensorDType::INT32;
            expert_offsets_view.Rank = 1;
            expert_offsets_view.Sizes[0] = static_cast<long>(num_experts_for_offsets + 1);
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

    int num_experts = num_experts_for_offsets;
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
    // Use MoeIntermediateSize for MoE models (may differ from IntermediateSize)
    const int intermediate_size = (mConfig.MoeIntermediateSize > 0)
        ? static_cast<int>(mConfig.MoeIntermediateSize)
        : static_cast<int>(mConfig.IntermediateSize);

    // LLEP per-expert weight pointer override for backward.
    // Case 1: Full LLEP state (last MoE layer) — use directly.
    // Case 2: No LLEP state but EP metadata (earlier layers) — reconstruct.
    const void* const* llep_weight_ptrs = nullptr;
    bool is_llep_active = false;
    std::vector<const void*> reconstructed_weight_ptrs;
    {
        auto llep_it = mLLEPStates.find(layer_idx);
        if (llep_it != mLLEPStates.end() && llep_it->second.active) {
            auto& llep = llep_it->second;
            is_llep_active = true;
            num_experts = llep.num_merged_experts;
            expert_offsets_view.Sizes[0] = static_cast<long>(num_experts + 1);
            llep_weight_ptrs = llep.down_weight_ptrs.data();
        } else if (mOptions.EPSize > 1 && layer_idx >= 0) {
            auto meta_it = mEPLayerMeta.find(layer_idx);
            if (meta_it != mEPLayerMeta.end() && meta_it->second.num_merged != meta_it->second.num_local) {
                const auto& meta = meta_it->second;
                is_llep_active = true;
                num_experts = meta.num_merged;
                expert_offsets_view.Sizes[0] = static_cast<long>(num_experts + 1);

                const size_t elem_sz = get_dtype_size(weights.DType);
                const size_t expert_bytes = static_cast<size_t>(hidden_size) * intermediate_size * elem_sz;
                std::vector<long> zw_shape = {1L, static_cast<long>(hidden_size), static_cast<long>(intermediate_size)};
                Tensor zero_weight = mRunState.temp_alloc(weights.DType, zw_shape);
                fill_zero(zero_weight, mRunState.MainStream);
                mTemps.push_back(zero_weight);

                reconstructed_weight_ptrs.resize(meta.num_merged);
                for (int m = 0; m < meta.num_merged; ++m) {
                    const int global_e = meta.merged_to_global[m];
                    const int local_idx = global_e - meta.native_start;
                    if (local_idx >= 0 && local_idx < meta.num_local) {
                        reconstructed_weight_ptrs[m] = static_cast<const std::byte*>(weights.Data)
                            + static_cast<size_t>(local_idx) * expert_bytes;
                    } else {
                        reconstructed_weight_ptrs[m] = zero_weight.Data;
                    }
                }
                llep_weight_ptrs = reconstructed_weight_ptrs.data();
            }
        }
    }

    const int weight_experts = llep_weight_ptrs ? num_experts
        : static_cast<int>(weights.Sizes[0]);

    // Get host offsets from cache (populates on first backward access for this layer).
    const int* cached_host_offsets = get_or_sync_moe_host_offsets(
        layer_idx, expert_offsets_ptr, num_experts);

    MoeCompactInfo compact = cached_host_offsets
        ? build_moe_compact_info_from_host(cached_host_offsets,
                                           num_experts,
                                           weight_experts,
                                           layer_idx,
                                           "moe_grouped_gemm_down_backward")
        : build_moe_compact_info(expert_offsets_ptr,
                                 num_experts,
                                 weight_experts,
                                 mRunState.MainStream,
                                 layer_idx,
                                 "moe_grouped_gemm_down_backward");
    const bool weight_is_compact = compact.weight_is_compact;
    const int* host_offsets_ptr = cached_host_offsets;
    if (!host_offsets_ptr && !compact.host_offsets.empty()) {
        host_offsets_ptr = compact.host_offsets.data();
    }
    std::vector<int> host_offsets_sanitized;
    if (host_offsets_ptr && num_experts > 0) {
        const long total_tokens = d_output.Sizes[0];
        bool valid = (host_offsets_ptr[0] == 0);
        int last = host_offsets_ptr[0];
        for (int e = 1; e <= num_experts && valid; ++e) {
            int v = host_offsets_ptr[e];
            if (v < last || v < 0 || v > total_tokens) {
                valid = false;
                break;
            }
            last = v;
        }
        if (valid && last != total_tokens) {
            valid = false;
        }
        if (!valid) {
            host_offsets_sanitized.assign(static_cast<std::size_t>(num_experts + 1), 0);
            const int clamped_total = static_cast<int>(total_tokens);
            host_offsets_sanitized[1] = clamped_total;
            for (int e = 2; e <= num_experts; ++e) {
                host_offsets_sanitized[e] = clamped_total;
            }
            host_offsets_ptr = host_offsets_sanitized.data();
        }
    }
    const int* active_ptr = compact.active_experts.empty() ? nullptr : compact.active_experts.data();
    const int num_active = compact.num_active;

    // Refresh MoE experts for this layer (selective dequant) before using weights in backward.
    auto* qlora_provider = mWeights.qlora_provider();
    if (qlora_provider && qlora_provider->supports_selective_moe()) {
        const int* refresh_offsets = host_offsets_ptr;
        if (refresh_offsets) {
            (void)refresh_moe_experts_if_needed(layer_idx,
                                                refresh_offsets,
                                                num_experts,
                                                mWeights,
                                                mRunState.MainStream);
        }
    }

    const bool lora_enabled = mLoRAConfig && mLoRAWeights && mLoRARunState &&
                              mLoRAConfig->enabled() && mLoRAWeights->enabled() &&
                              layer_idx >= 0;
    bool skip_base_backward = false;
    if (lora_enabled &&
        mRunState.is_lora_only_mode() &&
        mRunState.is_prequantized() &&
        mConfig.Architecture == PretrainedConfig::GPT_OSS) {
        const char* env = std::getenv("SUROGATE_PREQUANT_BASE_GRAD");
        const bool allow_base_grad = (env && *env && std::string(env) != "0");
        skip_base_backward = !allow_base_grad;
    }

    auto zero_d_input = [&]() {
        if (!d_input_ptr->Data || d_input_ptr->bytes() == 0) return;
        if (d_input_ptr->Device == -1 && mRunState.Stack.owns(d_input_ptr->Data)) {
            CUDA_CHECK(cudaMemsetAsync(d_input_ptr->Data, 0, d_input_ptr->bytes(), mRunState.MainStream));
        } else {
            fill_zero(*d_input_ptr, mRunState.MainStream);
        }
    };

    if (skip_base_backward || (weight_is_compact && compact.active_experts.empty())) {
        zero_d_input();
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
                                       num_active,
                                       llep_weight_ptrs);
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
                                       num_active,
                                       llep_weight_ptrs);
    }

    // Apply grouped MoE LoRA backward (down projection) when enabled.
    if (lora_enabled) {
        auto& lora_block = mLoRAWeights->get_block(layer_idx, mRunState.MainStream);
        if (lora_block.moe.use_grouped) {
            // When LLEP is active, use merged LoRA tensors
            const auto* down_ptr = lora_block.moe.grouped.down.has_value()
                ? &(*lora_block.moe.grouped.down) : nullptr;
            bool llep_lora_active = false;
            {
                auto llep_lora_it = mLLEPStates.find(layer_idx);
                if (llep_lora_it != mLLEPStates.end() && llep_lora_it->second.active
                    && llep_lora_it->second.has_merged_lora
                    && llep_lora_it->second.merged_lora.down.has_value()) {
                    down_ptr = &(*llep_lora_it->second.merged_lora.down);
                    llep_lora_active = true;
                }
            }
            if (down_ptr && down_ptr->has_value()) {
            const auto& lora_down = *down_ptr;
            const int rank = mLoRAConfig->rank;
            const float scaling = mLoRAConfig->scaling();
            const float dropout = mLoRAConfig->dropout;
            const bool training = mLoRARunState->is_training;
            const long total_tokens_l = d_output.Sizes[0];
            const int total_tokens_i = static_cast<int>(total_tokens_l);

            auto get_dropout_seed = [&](int proj_type) -> unsigned int {
                return mLoRARunState->dropout_base_seed
                       + static_cast<unsigned int>(layer_idx) * 1000000u
                       + static_cast<unsigned int>(proj_type) * 100000u
                       + static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;
            };

            auto view_or_temp = [&](Tensor& buf, long rows, long cols) -> Tensor {
                const long need = rows * cols;
                if (!buf.Data || buf.DType != d_output.DType || buf.nelem() < need) {
                    Tensor tmp = mRunState.temp_alloc(d_output.DType, {rows, cols});
                    mTemps.push_back(tmp);
                    return tmp;
                }
                Tensor view = buf;
                view.DType = d_output.DType;
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
                                     expert_offsets_ptr, num_experts, M, K,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr, alpha, beta, mode, active_ptr,
                                     /*weight_is_compact=*/false, num_active);
                } else {
                    moe_grouped_gemm(out_t.get<float>(), in_t.get<float>(), weight_t.get<float>(),
                                     expert_offsets_ptr, num_experts, M, K,
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
                                                 expert_offsets_ptr, num_experts, M, N,
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
                                                 expert_offsets_ptr, num_experts, M, N,
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
            if (mLoRAGrads && mComm && !llep_lora_active) {
                // Skip LoRA weight grads when LLEP is active — grad storage is
                // [num_local, ...] which doesn't match num_merged.
                lora_grads = &mLoRAGrads->get_block_full(layer_idx, mRunState.MainStream, *mComm, lora_accum);
            }
            const float grad_beta = lora_accum ? 1.0f : 0.0f;

            if (total_tokens_i > 0 && rank > 0) {
                Tensor lora_intermediate = view_or_temp(mLoRARunState->moe_lora_intermediate1, total_tokens_l, rank);
                const unsigned int seed_down = get_dropout_seed(6);

                // dB: intermediate = x @ A^T
                dispatch_grouped_gemm(lora_intermediate, inp, lora_down.A,
                                      rank, intermediate_size, 1.0f, 0.0f, EMMTranspose::TN);
                scale_and_dropout(lora_intermediate, seed_down);
                if (lora_grads && lora_grads->moe.grouped.down.has_value()) {
                    dispatch_weight_grad(lora_grads->moe.grouped.down->B, d_output, lora_intermediate,
                                         hidden_size, rank, grad_beta);
                }

                // intermediate = d_output @ B
                dispatch_grouped_gemm(lora_intermediate, d_output, lora_down.B,
                                      rank, hidden_size, 1.0f, 0.0f, EMMTranspose::NN);
                scale_and_dropout(lora_intermediate, seed_down);
                if (lora_grads && lora_grads->moe.grouped.down.has_value()) {
                    dispatch_weight_grad(lora_grads->moe.grouped.down->A, lora_intermediate, inp,
                                         rank, intermediate_size, grad_beta);
                }

                // d_input += intermediate @ A
                dispatch_grouped_gemm(d_input, lora_intermediate, lora_down.A,
                                      intermediate_size, rank, 1.0f, 1.0f, EMMTranspose::NN);
            }
            }  // if (down_ptr && down_ptr->has_value())
        }  // if (use_grouped)
    }

    store_tensor(op.outputs[0], d_input);

    // Weight gradient computation would go here if needed (for fine-tuning experts)
}


}  // namespace dsl
