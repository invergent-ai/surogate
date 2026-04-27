#include "runtime/executor/compiled_ops.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
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
    Tensor weights =
        resolve_tensor(op.inputs[1]);  // Parameter name resolved by graph compiler (copy for LLEP override)
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

    int layer_idx_any = op.attrs.layer_idx;
    if (layer_idx_any < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        std::string field;
        parse_block_param(name, layer_idx_any, field);
    }
    const int ep_key_any = ep_state_key(layer_idx_any);

    // For EP, derive num_experts from the forward-cached offsets.
    // LLEP may change num_merged per layer.
    int num_experts_for_offsets = static_cast<int>(mConfig.NumLocalExperts);
    if (mOptions.EPSize > 1 && layer_idx_any >= 0) {
        auto ci = mMoEHostOffsetsCache.find(ep_key_any);
        if (ci == mMoEHostOffsetsCache.end()) {
            ci = mMoEHostOffsetsCache.find(layer_idx_any);
        }
        if (ci != mMoEHostOffsetsCache.end() && ci->second.size() > 1) {
            num_experts_for_offsets = static_cast<int>(ci->second.size()) - 1;
        }
    }
    int num_experts = num_experts_for_offsets;
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
    // Use MoeIntermediateSize for MoE models (may differ from IntermediateSize)
    const int intermediate_size = (mConfig.MoeIntermediateSize > 0) ? static_cast<int>(mConfig.MoeIntermediateSize)
                                                                    : static_cast<int>(mConfig.IntermediateSize);
    // Get expert offsets from per-layer saved buffers when available.
    Tensor expert_offsets_view;
    Tensor* expert_offsets_ptr = nullptr;
    if (layer_idx_any >= 0) {
        const std::string base_key = "blocks[" + std::to_string(layer_idx_any) + "].moe_expert_offsets";
        std::vector<std::string> candidate_keys;
        if (mOptions.EPSize > 1) {
            candidate_keys.push_back(base_key + (mInReplay ? "#r1" : "#r0"));
            candidate_keys.push_back(base_key + (mInReplay ? "#r0" : "#r1"));
        }
        candidate_keys.push_back(base_key);

        void* saved_ptr = nullptr;
        for (const auto& key : candidate_keys) {
            auto it_saved = mMoeSavedBuffers.find(key);
            if (it_saved != mMoeSavedBuffers.end() && it_saved->second != nullptr) {
                saved_ptr = it_saved->second;
                break;
            }
        }
        if (saved_ptr != nullptr) {
            expert_offsets_view.DType = ETensorDType::INT32;
            expert_offsets_view.Rank = 1;
            expert_offsets_view.Sizes[0] = static_cast<long>(num_experts_for_offsets + 1);
            expert_offsets_view.Data = static_cast<std::byte*>(saved_ptr);
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
    std::vector<const void*> refreshed_native_weight_ptrs;
    {
        auto llep_it = mLLEPStates.find(ep_key_any);
        if (llep_it != mLLEPStates.end() && llep_it->second.active) {
            auto& llep = llep_it->second;
            num_experts = llep.num_merged_experts;
            expert_offsets_view.Sizes[0] = static_cast<long>(num_experts + 1);
            const auto meta_it = mEPLayerMeta.find(ep_key_any);
            const bool refresh_native_only_ptrs = llep.owned_foreign_ptrs.empty() && meta_it != mEPLayerMeta.end() &&
                                                  meta_it->second.num_merged == meta_it->second.num_local &&
                                                  llep.num_merged_experts == meta_it->second.num_local &&
                                                  weights.Rank >= 3;
            if (refresh_native_only_ptrs) {
                const auto& meta = meta_it->second;
                const std::size_t elem_sz = get_dtype_size(weights.DType);
                const std::size_t expert_elems =
                    static_cast<std::size_t>(weights.Sizes[1]) * static_cast<std::size_t>(weights.Sizes[2]);
                const std::size_t expert_bytes = expert_elems * elem_sz;
                refreshed_native_weight_ptrs.resize(llep.num_merged_experts);
                for (int m = 0; m < llep.num_merged_experts; ++m) {
                    const int global_e = llep.merged_to_global[m];
                    const int local_idx = global_e - meta.native_start;
                    if (local_idx < 0 || local_idx >= meta.num_local) {
                        throw std::runtime_error(
                            "moe_grouped_gemm_down: invalid native expert refresh mapping at layer " +
                            std::to_string(layer_idx_any) + " global_e=" + std::to_string(global_e) +
                            " local_idx=" + std::to_string(local_idx));
                    }
                    refreshed_native_weight_ptrs[m] = static_cast<const std::byte*>(weights.Data) +
                                                      static_cast<std::size_t>(local_idx) * expert_bytes;
                }
                llep_weight_ptrs = refreshed_native_weight_ptrs.data();
            } else {
                llep_weight_ptrs = llep.down_weight_ptrs.data();
            }
            is_llep_active = true;
        }
    }
    if (!llep_weight_ptrs && mOptions.EPSize > 1 && layer_idx_any >= 0 && weights.Rank >= 3) {
        const int weight_rows = static_cast<int>(weights.Sizes[0]);
        if (weight_rows > num_experts) {
            auto meta_it = mEPLayerMeta.find(ep_key_any);
            if (meta_it != mEPLayerMeta.end()) {
                const auto& meta = meta_it->second;
                if (meta.num_local == num_experts && meta.native_start >= 0 &&
                    (meta.native_start + num_experts) <= weight_rows) {
                    const std::size_t elem_sz = get_dtype_size(weights.DType);
                    const std::size_t expert_elems =
                        static_cast<std::size_t>(weights.Sizes[1]) * static_cast<std::size_t>(weights.Sizes[2]);
                    weights.Data = static_cast<std::byte*>(weights.Data) +
                                   static_cast<std::size_t>(meta.native_start) * expert_elems * elem_sz;
                    weights.Sizes[0] = num_experts;
                }
            }
        }
    }

    const int weight_experts =
        llep_weight_ptrs ? num_experts : ((weights.Rank > 0) ? static_cast<int>(weights.Sizes[0]) : num_experts);
    const int* host_offsets_ptr = nullptr;
    if (num_experts > 0 && expert_offsets.Data) {
        // Use cached host offsets (populated by dispatch_moe_permute or ep_dispatch for this layer).
        host_offsets_ptr = get_or_sync_moe_host_offsets(ep_key_any, expert_offsets.get<int>(), num_experts);
    }

    MoeCompactInfo compact = host_offsets_ptr ? build_moe_compact_info_from_host(host_offsets_ptr,
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
    Tensor out = mRunState.temp_alloc(inp.DType, out_shape, "moe_grouped_gemm_down_out");
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
        ctx.cublas_handle = mRunState.cublas_handle();
        ctx.cublaslt_handle = mRunState.CublasLtHandle;
        ctx.workspace = mRunState.CuBlasWorkspace.get<std::byte>();
        ctx.workspace_size = mRunState.CuBlasWorkspace.bytes();
        ctx.stream = mRunState.MainStream;
        ctx.op_caps = op.default_caps;
        ctx.moe_caps = op.moe_caps;
        ctx.epilogue_support = op.epilogue_support;
        ctx.storage_compat = op.storage_compat;
        attach_moe_fp8_cache(ctx, op.inputs[1].name);
        mRecipe->forward_moe_matmul(ctx);
    } else if (inp.DType == ETensorDType::BF16) {
        moe_grouped_gemm_down(out.get<nv_bfloat16>(),
                              inp.get<nv_bfloat16>(),
                              weights.get<nv_bfloat16>(),
                              expert_offsets.get<int>(),
                              num_experts,
                              hidden_size,
                              intermediate_size,
                              mRunState.cublas_handle(),
                              mRunState.MainStream,
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
                              num_experts,
                              hidden_size,
                              intermediate_size,
                              mRunState.cublas_handle(),
                              mRunState.MainStream,
                              host_offsets_ptr,
                              active_ptr,
                              weight_is_compact,
                              num_active,
                              llep_weight_ptrs);
    }

    // Apply grouped MoE LoRA (down projection) when enabled.
    if (mLoRAConfig && mLoRAWeights && mLoRARunState && mLoRAConfig->enabled() && mLoRAWeights->enabled() &&
        layer_idx_any >= 0) {
        auto& lora_block = mLoRAWeights->get_block(layer_idx_any, mRunState.MainStream);
        if (lora_block.moe.use_grouped) {
            // When LLEP is active, use merged LoRA tensors
            const auto* down_ptr = lora_block.moe.grouped.down.has_value() ? &(*lora_block.moe.grouped.down) : nullptr;
            {
                auto llep_lora_it = mLLEPStates.find(ep_key_any);
                if (llep_lora_it != mLLEPStates.end() && llep_lora_it->second.active &&
                    llep_lora_it->second.has_merged_lora && llep_lora_it->second.merged_lora.down.has_value()) {
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
                    return mLoRARunState->dropout_base_seed + static_cast<unsigned int>(layer_idx_any) * 1000000u +
                           static_cast<unsigned int>(proj_type) * 100000u +
                           static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;
                };

                auto view_or_temp = [&](Tensor& buf, long rows, long cols) -> Tensor {
                    const long need = rows * cols;
                    if (!buf.Data || buf.DType != out.DType || buf.nelem() < need) {
                        Tensor tmp = mRunState.temp_alloc(out.DType, {rows, cols}, "moe_grouped_gemm_down_temp");
                        mTemps.push_back(tmp);
                        return tmp;
                    }
                    Tensor view = buf;
                    view.DType = out.DType;
                    view.Rank = 2;
                    view.Sizes[0] = rows;
                    view.Sizes[1] = cols;
                    for (int i = 2; i < MAX_TENSOR_DIM; ++i)
                        view.Sizes[i] = 1;
                    return view;
                };

                auto dispatch_grouped_gemm = [&](Tensor& out_t,
                                                 const Tensor& in_t,
                                                 const Tensor& weight_t,
                                                 int M,
                                                 int K,
                                                 float alpha,
                                                 float beta,
                                                 EMMTranspose mode) {
                    if (in_t.DType != weight_t.DType || in_t.DType != out_t.DType) {
                        std::string msg = "MoE LoRA: dtype mismatch between activation and LoRA weights. "
                                          "Set lora_dtype='bf16' in your config to match activation dtype.";
                        throw std::runtime_error(msg);
                    }
                    Tensor weight_view = weight_t;
                    int weight_rows = (weight_view.Rank > 0) ? static_cast<int>(weight_view.Sizes[0]) : num_experts;
                    if (mOptions.EPSize > 1 && layer_idx_any >= 0 && weight_view.Rank >= 3 &&
                        weight_rows > num_experts) {
                        auto meta_it = mEPLayerMeta.find(ep_key_any);
                        if (meta_it != mEPLayerMeta.end()) {
                            const auto& meta = meta_it->second;
                            if (meta.num_local == num_experts && meta.native_start >= 0 &&
                                (meta.native_start + num_experts) <= weight_rows) {
                                const std::size_t elem_sz = get_dtype_size(weight_view.DType);
                                const std::size_t expert_elems = static_cast<std::size_t>(weight_view.Sizes[1]) *
                                                                 static_cast<std::size_t>(weight_view.Sizes[2]);
                                weight_view.Data = static_cast<std::byte*>(weight_view.Data) +
                                                   static_cast<std::size_t>(meta.native_start) * expert_elems * elem_sz;
                                weight_view.Sizes[0] = num_experts;
                                weight_rows = num_experts;
                            }
                        }
                    }
                    const bool lora_weight_is_compact = (weight_rows != num_experts);
                    const int* lora_active_ptr = active_ptr;
                    int lora_num_active = num_active;
                    std::vector<int> fallback_active;
                    if (lora_weight_is_compact &&
                        (lora_active_ptr == nullptr || lora_num_active <= 0 || lora_num_active > weight_rows)) {
                        const int fallback_count = std::max(0, std::min(weight_rows, num_experts));
                        fallback_active.resize(static_cast<std::size_t>(fallback_count));
                        for (int i = 0; i < fallback_count; ++i) {
                            fallback_active[static_cast<std::size_t>(i)] = i;
                        }
                        lora_active_ptr = fallback_active.empty() ? nullptr : fallback_active.data();
                        lora_num_active = fallback_count;
                    }
                    if (in_t.DType == ETensorDType::BF16) {
                        moe_grouped_gemm(out_t.get<nv_bfloat16>(),
                                         in_t.get<nv_bfloat16>(),
                                         weight_view.get<nv_bfloat16>(),
                                         expert_offsets.get<int>(),
                                         num_experts,
                                         M,
                                         K,
                                         mRunState.cublas_handle(),
                                         mRunState.MainStream,
                                         host_offsets_ptr,
                                         alpha,
                                         beta,
                                         mode,
                                         lora_active_ptr,
                                         lora_weight_is_compact,
                                         lora_num_active);
                    } else {
                        moe_grouped_gemm(out_t.get<float>(),
                                         in_t.get<float>(),
                                         weight_view.get<float>(),
                                         expert_offsets.get<int>(),
                                         num_experts,
                                         M,
                                         K,
                                         mRunState.cublas_handle(),
                                         mRunState.MainStream,
                                         host_offsets_ptr,
                                         alpha,
                                         beta,
                                         mode,
                                         lora_active_ptr,
                                         lora_weight_is_compact,
                                         lora_num_active);
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
                    Tensor lora_intermediate =
                        view_or_temp(mLoRARunState->moe_lora_intermediate1, total_tokens_l, rank);
                    dispatch_grouped_gemm(lora_intermediate,
                                          inp,
                                          lora_down.A,
                                          rank,
                                          intermediate_size,
                                          1.0f,
                                          0.0f,
                                          EMMTranspose::TN);
                    scale_and_dropout(lora_intermediate, get_dropout_seed(6));
                    dispatch_grouped_gemm(out,
                                          lora_intermediate,
                                          lora_down.B,
                                          hidden_size,
                                          rank,
                                          1.0f,
                                          1.0f,
                                          EMMTranspose::TN);
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
        d_input_local = mRunState.temp_alloc(inp.DType, shape, "moe_grouped_gemm_down_d_input_local");
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
    const int input_total_recv = static_cast<int>(inp.Sizes[0]);
    int ep_key = ep_state_key(layer_idx);
    if (mOptions.EPSize > 1 && layer_idx >= 0) {
        const int ep_key_r0 = (layer_idx << 1);
        const int ep_key_r1 = ep_key_r0 | 1;
        auto it_r0 = mEpStates.find(ep_key_r0);
        auto it_r1 = mEpStates.find(ep_key_r1);
        if (it_r1 != mEpStates.end() && it_r1->second.total_recv == input_total_recv) {
            ep_key = ep_key_r1;
        } else if (it_r0 != mEpStates.end() && it_r0->second.total_recv == input_total_recv) {
            ep_key = ep_key_r0;
        } else if (it_r1 != mEpStates.end()) {
            ep_key = ep_key_r1;
        } else if (it_r0 != mEpStates.end()) {
            ep_key = ep_key_r0;
        }
    }
    // For EP, derive num_experts from the forward-cached offsets.
    // LLEP may change num_merged per layer.
    int num_experts_for_offsets = static_cast<int>(mConfig.NumLocalExperts);
    if (mOptions.EPSize > 1 && layer_idx >= 0) {
        auto ci = mMoEHostOffsetsCache.find(ep_key);
        if (ci == mMoEHostOffsetsCache.end()) {
            ci = mMoEHostOffsetsCache.find(layer_idx);
        }
        if (ci != mMoEHostOffsetsCache.end() && ci->second.size() > 1) {
            num_experts_for_offsets = static_cast<int>(ci->second.size()) - 1;
        }
    }

    // Use per-layer expert_offsets when available; fall back to global buffer.
    const int* expert_offsets_ptr = nullptr;
    Tensor expert_offsets_view;
    if (layer_idx >= 0) {
        std::vector<std::string> candidate_keys;
        if (mOptions.EPSize > 1) {
            const std::string base_key = "blocks[" + std::to_string(layer_idx) + "].moe_expert_offsets";
            candidate_keys.push_back(base_key + "#r1");
            candidate_keys.push_back(base_key + "#r0");
        } else {
            candidate_keys.push_back(moe_saved_key(layer_idx, "moe_expert_offsets"));
        }
        candidate_keys.push_back("blocks[" + std::to_string(layer_idx) + "].moe_expert_offsets");
        void* saved_ptr = nullptr;
        for (const auto& key : candidate_keys) {
            auto it = mMoeSavedBuffers.find(key);
            if (it != mMoeSavedBuffers.end() && it->second != nullptr) {
                saved_ptr = it->second;
                break;
            }
        }
        if (saved_ptr != nullptr) {
            expert_offsets_view.DType = ETensorDType::INT32;
            expert_offsets_view.Rank = 1;
            expert_offsets_view.Sizes[0] = static_cast<long>(num_experts_for_offsets + 1);
            expert_offsets_view.Data = static_cast<std::byte*>(saved_ptr);
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
    const int intermediate_size = (mConfig.MoeIntermediateSize > 0) ? static_cast<int>(mConfig.MoeIntermediateSize)
                                                                    : static_cast<int>(mConfig.IntermediateSize);

    // LLEP per-expert weight pointer override for backward.
    // Case 1: Full LLEP state (last MoE layer) — use directly.
    // Case 2: No LLEP state but EP metadata (earlier layers) — reconstruct.
    const void* const* llep_weight_ptrs = nullptr;
    bool is_llep_active = false;
    std::vector<const void*> reconstructed_weight_ptrs;
    std::vector<const void*> refreshed_native_weight_ptrs;
    {
        auto llep_it = mLLEPStates.find(ep_key);
        if (llep_it != mLLEPStates.end() && llep_it->second.active) {
            auto& llep = llep_it->second;
            is_llep_active = true;
            num_experts = llep.num_merged_experts;
            expert_offsets_view.Sizes[0] = static_cast<long>(num_experts + 1);
            const auto meta_it = mEPLayerMeta.find(ep_key);
            const bool refresh_native_only_ptrs = llep.owned_foreign_ptrs.empty() && meta_it != mEPLayerMeta.end() &&
                                                  meta_it->second.num_merged == meta_it->second.num_local &&
                                                  llep.num_merged_experts == meta_it->second.num_local &&
                                                  weights.Rank >= 3;
            if (refresh_native_only_ptrs) {
                const auto& meta = meta_it->second;
                const std::size_t elem_sz = get_dtype_size(weights.DType);
                const std::size_t expert_elems =
                    static_cast<std::size_t>(weights.Sizes[1]) * static_cast<std::size_t>(weights.Sizes[2]);
                const std::size_t expert_bytes = expert_elems * elem_sz;
                refreshed_native_weight_ptrs.resize(llep.num_merged_experts);
                for (int m = 0; m < llep.num_merged_experts; ++m) {
                    const int global_e = llep.merged_to_global[m];
                    const int local_idx = global_e - meta.native_start;
                    if (local_idx < 0 || local_idx >= meta.num_local) {
                        throw std::runtime_error(
                            "moe_grouped_gemm_down_backward: invalid native expert refresh mapping at layer " +
                            std::to_string(layer_idx) + " global_e=" + std::to_string(global_e) +
                            " local_idx=" + std::to_string(local_idx));
                    }
                    refreshed_native_weight_ptrs[m] = static_cast<const std::byte*>(weights.Data) +
                                                      static_cast<std::size_t>(local_idx) * expert_bytes;
                }
                llep_weight_ptrs = refreshed_native_weight_ptrs.data();
            } else {
                llep_weight_ptrs = llep.down_weight_ptrs.data();
            }
        } else if (mOptions.EPSize > 1 && layer_idx >= 0) {
            auto meta_it = mEPLayerMeta.find(ep_key);
            if (meta_it != mEPLayerMeta.end() && meta_it->second.num_merged != meta_it->second.num_local) {
                const auto& meta = meta_it->second;
                is_llep_active = true;
                num_experts = meta.num_merged;
                expert_offsets_view.Sizes[0] = static_cast<long>(num_experts + 1);

                const size_t elem_sz = get_dtype_size(weights.DType);
                const size_t expert_bytes = static_cast<size_t>(hidden_size) * intermediate_size * elem_sz;
                std::vector<long> zw_shape = {1L, static_cast<long>(hidden_size), static_cast<long>(intermediate_size)};
                Tensor zero_weight = mRunState.temp_alloc(weights.DType, zw_shape, "moe_grouped_gemm_down_zero_weight");
                fill_zero(zero_weight, mRunState.MainStream);
                mTemps.push_back(zero_weight);

                reconstructed_weight_ptrs.resize(meta.num_merged);
                for (int m = 0; m < meta.num_merged; ++m) {
                    const int global_e = meta.merged_to_global[m];
                    const int local_idx = global_e - meta.native_start;
                    if (local_idx >= 0 && local_idx < meta.num_local) {
                        reconstructed_weight_ptrs[m] =
                            static_cast<const std::byte*>(weights.Data) + static_cast<size_t>(local_idx) * expert_bytes;
                    } else {
                        reconstructed_weight_ptrs[m] = zero_weight.Data;
                    }
                }
                llep_weight_ptrs = reconstructed_weight_ptrs.data();
            }
        }
    }
    if (!llep_weight_ptrs && mOptions.EPSize > 1 && layer_idx >= 0 && weights.Rank >= 3) {
        const int weight_rows = static_cast<int>(weights.Sizes[0]);
        if (weight_rows > num_experts) {
            auto meta_it = mEPLayerMeta.find(ep_key);
            if (meta_it != mEPLayerMeta.end()) {
                const auto& meta = meta_it->second;
                if (meta.num_local == num_experts && meta.native_start >= 0 &&
                    (meta.native_start + num_experts) <= weight_rows) {
                    const std::size_t elem_sz = get_dtype_size(weights.DType);
                    const std::size_t expert_elems =
                        static_cast<std::size_t>(weights.Sizes[1]) * static_cast<std::size_t>(weights.Sizes[2]);
                    weights.Data = static_cast<std::byte*>(weights.Data) +
                                   static_cast<std::size_t>(meta.native_start) * expert_elems * elem_sz;
                    weights.Sizes[0] = num_experts;
                }
            }
        }
    }

    const int weight_experts = llep_weight_ptrs ? num_experts : static_cast<int>(weights.Sizes[0]);

    // Get host offsets from cache (populates on first backward access for this layer).
    const int* cached_host_offsets = get_or_sync_moe_host_offsets(ep_key, expert_offsets_ptr, num_experts);

    MoeCompactInfo compact = cached_host_offsets ? build_moe_compact_info_from_host(cached_host_offsets,
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
            const bool refreshed =
                refresh_moe_experts_if_needed(layer_idx, refresh_offsets, num_experts, mWeights, mRunState.MainStream);
            if (refreshed) {
                invalidate_moe_fp8_cache(op.inputs[2].name);
            }
        }
    }

    const bool lora_enabled = mLoRAConfig && mLoRAWeights && mLoRARunState && mLoRAConfig->enabled() &&
                              mLoRAWeights->enabled() && layer_idx >= 0;
    const bool skip_base_backward = lora_enabled && mRunState.is_lora_only_mode() && mRunState.is_prequantized() &&
                                    mConfig.Architecture == PretrainedConfig::GPT_OSS;

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
                                       num_experts,
                                       hidden_size,
                                       intermediate_size,
                                       mRunState.cublas_handle(),
                                       mRunState.MainStream,
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
                                       num_experts,
                                       hidden_size,
                                       intermediate_size,
                                       mRunState.cublas_handle(),
                                       mRunState.MainStream,
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
            const auto* down_ptr = lora_block.moe.grouped.down.has_value() ? &(*lora_block.moe.grouped.down) : nullptr;
            bool llep_lora_active = false;
            {
                auto llep_lora_it = mLLEPStates.find(ep_key);
                if (llep_lora_it != mLLEPStates.end() && llep_lora_it->second.active &&
                    llep_lora_it->second.has_merged_lora && llep_lora_it->second.merged_lora.down.has_value()) {
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
                    return mLoRARunState->dropout_base_seed + static_cast<unsigned int>(layer_idx) * 1000000u +
                           static_cast<unsigned int>(proj_type) * 100000u +
                           static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;
                };

                auto view_or_temp = [&](Tensor& buf, long rows, long cols) -> Tensor {
                    const long need = rows * cols;
                    if (!buf.Data || buf.DType != d_output.DType || buf.nelem() < need) {
                        Tensor tmp = mRunState.temp_alloc(d_output.DType, {rows, cols}, "moe_grouped_gemm_down_temp");
                        mTemps.push_back(tmp);
                        return tmp;
                    }
                    Tensor view = buf;
                    view.DType = d_output.DType;
                    view.Rank = 2;
                    view.Sizes[0] = rows;
                    view.Sizes[1] = cols;
                    for (int i = 2; i < MAX_TENSOR_DIM; ++i)
                        view.Sizes[i] = 1;
                    return view;
                };

                auto dispatch_grouped_gemm = [&](Tensor& out_t,
                                                 const Tensor& in_t,
                                                 const Tensor& weight_t,
                                                 int M,
                                                 int K,
                                                 float alpha,
                                                 float beta,
                                                 EMMTranspose mode) {
                    if (in_t.DType != weight_t.DType || in_t.DType != out_t.DType) {
                        std::string msg = "MoE LoRA backward: dtype mismatch between activation and LoRA weights. "
                                          "Set lora_dtype='bf16' in your config to match activation dtype.";
                        throw std::runtime_error(msg);
                    }
                    Tensor weight_view = weight_t;
                    int weight_rows = (weight_view.Rank > 0) ? static_cast<int>(weight_view.Sizes[0]) : num_experts;
                    if (mOptions.EPSize > 1 && layer_idx >= 0 && weight_view.Rank >= 3 && weight_rows > num_experts) {
                        auto meta_it = mEPLayerMeta.find(ep_key);
                        if (meta_it != mEPLayerMeta.end()) {
                            const auto& meta = meta_it->second;
                            if (meta.num_local == num_experts && meta.native_start >= 0 &&
                                (meta.native_start + num_experts) <= weight_rows) {
                                const std::size_t elem_sz = get_dtype_size(weight_view.DType);
                                const std::size_t expert_elems = static_cast<std::size_t>(weight_view.Sizes[1]) *
                                                                 static_cast<std::size_t>(weight_view.Sizes[2]);
                                weight_view.Data = static_cast<std::byte*>(weight_view.Data) +
                                                   static_cast<std::size_t>(meta.native_start) * expert_elems * elem_sz;
                                weight_view.Sizes[0] = num_experts;
                                weight_rows = num_experts;
                            }
                        }
                    }
                    const bool lora_weight_is_compact = (weight_rows != num_experts);
                    const int* lora_active_ptr = active_ptr;
                    int lora_num_active = num_active;
                    std::vector<int> fallback_active;
                    if (lora_weight_is_compact &&
                        (lora_active_ptr == nullptr || lora_num_active <= 0 || lora_num_active > weight_rows)) {
                        const int fallback_count = std::max(0, std::min(weight_rows, num_experts));
                        fallback_active.resize(static_cast<std::size_t>(fallback_count));
                        for (int i = 0; i < fallback_count; ++i) {
                            fallback_active[static_cast<std::size_t>(i)] = i;
                        }
                        lora_active_ptr = fallback_active.empty() ? nullptr : fallback_active.data();
                        lora_num_active = fallback_count;
                    }
                    const int* lora_host_offsets_ptr = host_offsets_ptr;
                    std::vector<int> lora_host_offsets_sanitized;
                    if (host_offsets_ptr && num_experts > 0 && in_t.Rank >= 2 && out_t.Rank >= 2) {
                        const int total_tokens_lora = static_cast<int>(std::min(in_t.Sizes[0], out_t.Sizes[0]));
                        bool valid = (host_offsets_ptr[0] == 0);
                        int prev = host_offsets_ptr[0];
                        int last = prev;
                        for (int e = 1; e <= num_experts && valid; ++e) {
                            const int v = host_offsets_ptr[e];
                            if (v < prev || v < 0 || v > total_tokens_lora) {
                                valid = false;
                                break;
                            }
                            prev = v;
                            last = v;
                        }
                        if (valid && last != total_tokens_lora) {
                            valid = false;
                        }
                        if (!valid) {
                            lora_host_offsets_sanitized.assign(static_cast<std::size_t>(num_experts + 1), 0);
                            int cur = 0;
                            for (int e = 1; e <= num_experts; ++e) {
                                int v = host_offsets_ptr[e];
                                if (v < cur) v = cur;
                                if (v < 0) v = 0;
                                if (v > total_tokens_lora) v = total_tokens_lora;
                                lora_host_offsets_sanitized[static_cast<std::size_t>(e)] = v;
                                cur = v;
                            }
                            lora_host_offsets_sanitized[static_cast<std::size_t>(num_experts)] = total_tokens_lora;
                            lora_host_offsets_ptr = lora_host_offsets_sanitized.data();
                        }
                    }
                    if (in_t.DType == ETensorDType::BF16) {
                        moe_grouped_gemm(out_t.get<nv_bfloat16>(),
                                         in_t.get<nv_bfloat16>(),
                                         weight_view.get<nv_bfloat16>(),
                                         expert_offsets_ptr,
                                         num_experts,
                                         M,
                                         K,
                                         mRunState.cublas_handle(),
                                         mRunState.MainStream,
                                         lora_host_offsets_ptr,
                                         alpha,
                                         beta,
                                         mode,
                                         lora_active_ptr,
                                         lora_weight_is_compact,
                                         lora_num_active);
                    } else {
                        moe_grouped_gemm(out_t.get<float>(),
                                         in_t.get<float>(),
                                         weight_view.get<float>(),
                                         expert_offsets_ptr,
                                         num_experts,
                                         M,
                                         K,
                                         mRunState.cublas_handle(),
                                         mRunState.MainStream,
                                         lora_host_offsets_ptr,
                                         alpha,
                                         beta,
                                         mode,
                                         lora_active_ptr,
                                         lora_weight_is_compact,
                                         lora_num_active);
                    }
                };

                auto dispatch_weight_grad =
                    [&](Tensor& d_weight, const Tensor& grad_output, const Tensor& in, int M, int N, float beta) {
                        if (grad_output.DType != in.DType) {
                            throw std::runtime_error("MoE LoRA backward: grad/output dtype mismatch.");
                        }
                        Tensor d_weight_view = d_weight;
                        int weight_rows =
                            (d_weight_view.Rank > 0) ? static_cast<int>(d_weight_view.Sizes[0]) : num_experts;
                        if (mOptions.EPSize > 1 && layer_idx >= 0 && d_weight_view.Rank >= 3 &&
                            weight_rows > num_experts) {
                            auto meta_it = mEPLayerMeta.find(ep_key);
                            if (meta_it != mEPLayerMeta.end()) {
                                const auto& meta = meta_it->second;
                                if (meta.num_local == num_experts && meta.native_start >= 0 &&
                                    (meta.native_start + num_experts) <= weight_rows) {
                                    const std::size_t elem_sz = get_dtype_size(d_weight_view.DType);
                                    const std::size_t expert_elems = static_cast<std::size_t>(d_weight_view.Sizes[1]) *
                                                                     static_cast<std::size_t>(d_weight_view.Sizes[2]);
                                    d_weight_view.Data =
                                        static_cast<std::byte*>(d_weight_view.Data) +
                                        static_cast<std::size_t>(meta.native_start) * expert_elems * elem_sz;
                                    d_weight_view.Sizes[0] = num_experts;
                                    weight_rows = num_experts;
                                }
                            }
                        }
                        const bool lora_weight_is_compact = (weight_rows != num_experts);
                        const int* lora_active_ptr = active_ptr;
                        int lora_num_active = num_active;
                        std::vector<int> fallback_active;
                        if (lora_weight_is_compact &&
                            (lora_active_ptr == nullptr || lora_num_active <= 0 || lora_num_active > weight_rows)) {
                            const int fallback_count = std::max(0, std::min(weight_rows, num_experts));
                            fallback_active.resize(static_cast<std::size_t>(fallback_count));
                            for (int i = 0; i < fallback_count; ++i) {
                                fallback_active[static_cast<std::size_t>(i)] = i;
                            }
                            lora_active_ptr = fallback_active.empty() ? nullptr : fallback_active.data();
                            lora_num_active = fallback_count;
                        }
                        const int* lora_host_offsets_ptr = host_offsets_ptr;
                        std::vector<int> lora_host_offsets_sanitized;
                        if (host_offsets_ptr && num_experts > 0 && grad_output.Rank >= 2 && in.Rank >= 2) {
                            const int total_tokens_lora = static_cast<int>(std::min(grad_output.Sizes[0], in.Sizes[0]));
                            bool valid = (host_offsets_ptr[0] == 0);
                            int prev = host_offsets_ptr[0];
                            int last = prev;
                            for (int e = 1; e <= num_experts && valid; ++e) {
                                const int v = host_offsets_ptr[e];
                                if (v < prev || v < 0 || v > total_tokens_lora) {
                                    valid = false;
                                    break;
                                }
                                prev = v;
                                last = v;
                            }
                            if (valid && last != total_tokens_lora) {
                                valid = false;
                            }
                            if (!valid) {
                                lora_host_offsets_sanitized.assign(static_cast<std::size_t>(num_experts + 1), 0);
                                int cur = 0;
                                for (int e = 1; e <= num_experts; ++e) {
                                    int v = host_offsets_ptr[e];
                                    if (v < cur) v = cur;
                                    if (v < 0) v = 0;
                                    if (v > total_tokens_lora) v = total_tokens_lora;
                                    lora_host_offsets_sanitized[static_cast<std::size_t>(e)] = v;
                                    cur = v;
                                }
                                lora_host_offsets_sanitized[static_cast<std::size_t>(num_experts)] = total_tokens_lora;
                                lora_host_offsets_ptr = lora_host_offsets_sanitized.data();
                            }
                        }
                        if (grad_output.DType == ETensorDType::BF16) {
                            if (try_moe_fp8_weight_grad(d_weight_view,
                                                        grad_output,
                                                        in,
                                                        expert_offsets_ptr,
                                                        num_experts,
                                                        M,
                                                        N,
                                                        lora_host_offsets_ptr,
                                                        lora_active_ptr,
                                                        lora_weight_is_compact,
                                                        lora_num_active,
                                                        beta,
                                                        "moe_grouped_gemm_down_lora")) {
                                return;
                            }
                            if (d_weight_view.DType != ETensorDType::BF16) {
                                throw std::runtime_error(
                                    "MoE LoRA backward: lora_dtype=fp32 with bf16 activations not supported. "
                                    "Set lora_dtype='bf16' in your config.");
                            }
                            moe_grouped_gemm_weight_grad(d_weight_view.get<nv_bfloat16>(),
                                                         grad_output.get<nv_bfloat16>(),
                                                         in.get<nv_bfloat16>(),
                                                         expert_offsets_ptr,
                                                         num_experts,
                                                         M,
                                                         N,
                                                         mRunState.cublas_handle(),
                                                         mRunState.MainStream,
                                                         lora_host_offsets_ptr,
                                                         /*alpha=*/1.0f,
                                                         beta,
                                                         lora_active_ptr,
                                                         lora_weight_is_compact,
                                                         lora_num_active);
                        } else {
                            if (d_weight_view.DType != ETensorDType::FP32) {
                                throw std::runtime_error("MoE LoRA backward: dtype mismatch in weight gradients.");
                            }
                            moe_grouped_gemm_weight_grad(d_weight_view.get<float>(),
                                                         grad_output.get<float>(),
                                                         in.get<float>(),
                                                         expert_offsets_ptr,
                                                         num_experts,
                                                         M,
                                                         N,
                                                         mRunState.cublas_handle(),
                                                         mRunState.MainStream,
                                                         lora_host_offsets_ptr,
                                                         /*alpha=*/1.0f,
                                                         beta,
                                                         lora_active_ptr,
                                                         lora_weight_is_compact,
                                                         lora_num_active);
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
                    Tensor lora_intermediate =
                        view_or_temp(mLoRARunState->moe_lora_intermediate1, total_tokens_l, rank);
                    const unsigned int seed_down = get_dropout_seed(6);

                    // dB: intermediate = x @ A^T
                    dispatch_grouped_gemm(lora_intermediate,
                                          inp,
                                          lora_down.A,
                                          rank,
                                          intermediate_size,
                                          1.0f,
                                          0.0f,
                                          EMMTranspose::TN);
                    scale_and_dropout(lora_intermediate, seed_down);
                    if (lora_grads && lora_grads->moe.grouped.down.has_value()) {
                        dispatch_weight_grad(lora_grads->moe.grouped.down->B,
                                             d_output,
                                             lora_intermediate,
                                             hidden_size,
                                             rank,
                                             grad_beta);
                    }

                    // intermediate = d_output @ B
                    dispatch_grouped_gemm(lora_intermediate,
                                          d_output,
                                          lora_down.B,
                                          rank,
                                          hidden_size,
                                          1.0f,
                                          0.0f,
                                          EMMTranspose::NN);
                    scale_and_dropout(lora_intermediate, seed_down);
                    if (lora_grads && lora_grads->moe.grouped.down.has_value()) {
                        dispatch_weight_grad(lora_grads->moe.grouped.down->A,
                                             lora_intermediate,
                                             inp,
                                             rank,
                                             intermediate_size,
                                             grad_beta);
                    }

                    // d_input += intermediate @ A
                    dispatch_grouped_gemm(d_input,
                                          lora_intermediate,
                                          lora_down.A,
                                          intermediate_size,
                                          rank,
                                          1.0f,
                                          1.0f,
                                          EMMTranspose::NN);
                }
            }  // if (down_ptr && down_ptr->has_value())
        }  // if (use_grouped)
    }

    store_tensor(op.outputs[0], d_input);

    // Weight gradient computation would go here if needed (for fine-tuning experts)
}

namespace {

// -----------------------------------------------------------------------------
// MoE Grouped GEMM Down backward rule
// Forward: out = moe_grouped_gemm_down(inp, weights, scatter_indices)
// Backward: d_inp = moe_grouped_gemm_down_backward(d_out, inp, weights, scatter_indices)
// Note: weights gradient is computed but not propagated (frozen expert weights)
// -----------------------------------------------------------------------------
std::vector<Operation> moe_grouped_gemm_down_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string inp = fwd.inputs[0];
        std::string weights = fwd.inputs[1];
        std::string scatter_indices = fwd.inputs[2];

        std::string inp_ref = ctx.is_param(inp) ? inp : saved_ref(inp);
        std::string weights_ref = ctx.is_param(weights) ? weights : saved_ref(weights);
        std::string scatter_ref = saved_ref(scatter_indices);

        ops.push_back(make_operation("moe_grouped_gemm_down_backward_" + std::to_string(ctx.op_counter++),
                                     "moe_grouped_gemm_down_backward",
                                     "moe_grouped_gemm_down_backward",
                                     {ctx.d_output, inp_ref, weights_ref, scatter_ref},
                                     {ctx.d_inputs[0]}));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("moe_grouped_gemm_down", ::dsl::moe_grouped_gemm_down_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// MoE Grouped GEMM Down
// Input[0]: x [total_tokens, intermediate]
// Input[1]: weights [num_experts, hidden_size, intermediate]
// Input[2]: scatter_indices
// Output: [total_tokens, hidden_size]
// ------------------------------------------------------------------------
const int _moe_grouped_gemm_down_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "moe_grouped_gemm_down";
    sig.min_inputs = 3;
    sig.max_inputs = 3;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
        // Complex grouped GEMM shapes; accept for now
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

const int _moe_grouped_gemm_down_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "moe_grouped_gemm_down_backward";
    sig.min_inputs = 4;
    sig.max_inputs = 4;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
