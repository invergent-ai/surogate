// MoE Grouped GEMM operation (generic version without fused activation)
// Used for Nemotron-H MoE blocks that use relu2 activation instead of swiglu

#include "runtime/executor/compiled_ops.h"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/buffer_plan.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "recipes/recipe.h"
#include "utilities/dtype.h"
#include "utilities/stack.h"
#include "runtime/lora/lora_config.h"
#include "runtime/lora/lora_grads_manager.h"
#include "runtime/lora/lora_run_state.h"
#include "runtime/lora/lora_weights_manager.h"

namespace dsl {

void CompiledExecutor::dispatch_moe_grouped_gemm(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor weights = resolve_tensor(op.inputs[1]);  // copy for LLEP override
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

    // Get expert offsets
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
        Tensor* moe_offsets_fwd_ptr = nullptr;
        if (op.attrs.moe_offsets_tensor_id >= 0 &&
            static_cast<std::size_t>(op.attrs.moe_offsets_tensor_id) < mTensors.size() &&
            mTensors[op.attrs.moe_offsets_tensor_id].Data) {
            moe_offsets_fwd_ptr = &mTensors[op.attrs.moe_offsets_tensor_id];
        }
        if (!moe_offsets_fwd_ptr) {
            throw std::runtime_error("moe_grouped_gemm: expert_offsets not found");
        }
        expert_offsets_ptr = moe_offsets_fwd_ptr;
    }
    Tensor& expert_offsets = *expert_offsets_ptr;

    // LLEP per-expert weight pointer override: when LLEP is active, use per-expert
    // pointers instead of contiguous weight tensors.
    bool is_llep_active = false;
    const void* const* llep_weight_ptrs = nullptr;
    std::vector<const void*> refreshed_native_weight_ptrs;
    {
        auto llep_it = mLLEPStates.find(ep_key_any);
        if (llep_it != mLLEPStates.end() && llep_it->second.active) {
            auto& llep = llep_it->second;
            is_llep_active = true;
            std::string_view wname = op.inputs[1].name;
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
                        throw std::runtime_error("moe_grouped_gemm: invalid native expert refresh mapping at layer " +
                                                 std::to_string(layer_idx_any) + " global_e=" +
                                                 std::to_string(global_e) + " local_idx=" + std::to_string(local_idx));
                    }
                    refreshed_native_weight_ptrs[m] = static_cast<const std::byte*>(weights.Data) +
                                                      static_cast<std::size_t>(local_idx) * expert_bytes;
                }
                llep_weight_ptrs = refreshed_native_weight_ptrs.data();
            } else if (wname.find("gate_up") != std::string_view::npos && !llep.gate_up_weight_ptrs.empty()) {
                llep_weight_ptrs = llep.gate_up_weight_ptrs.data();
            } else if (wname.find("down") != std::string_view::npos && !llep.down_weight_ptrs.empty()) {
                llep_weight_ptrs = llep.down_weight_ptrs.data();
            } else if (wname.find("_up") != std::string_view::npos && !llep.gate_up_weight_ptrs.empty()) {
                // Separate up projection (Nemotron-H / Nemotron Nano "experts_up") — reuses
                // gate_up_weight_ptrs since ep_dispatch populates it from whichever up-projection
                // weight exists.
                llep_weight_ptrs = llep.gate_up_weight_ptrs.data();
            }
            num_experts = llep.num_merged_experts;
            expert_offsets_view.Sizes[0] = static_cast<long>(num_experts + 1);
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

    // Weight dimensions: [num_experts, out_features, in_features]
    const int out_features = (weights.Rank >= 2) ? static_cast<int>(weights.Sizes[1]) : 0;
    const int in_features = (weights.Rank >= 3) ? static_cast<int>(weights.Sizes[2]) : 0;
    const int weight_experts =
        llep_weight_ptrs ? num_experts : ((weights.Rank > 0) ? static_cast<int>(weights.Sizes[0]) : num_experts);
    const int* expert_offsets_data = expert_offsets.get<int>();

    const int* host_offsets_ptr = nullptr;
    if (num_experts > 0 && expert_offsets.Data) {
        // Use cached host offsets (populated by dispatch_moe_permute or ep_dispatch for this layer).
        host_offsets_ptr = get_or_sync_moe_host_offsets(ep_key_any, expert_offsets.get<int>(), num_experts);
    }

    // Refresh MoE expert weights for this layer using the current routing offsets.
    auto* qlora_provider = mWeights.qlora_provider();
    if (host_offsets_ptr && layer_idx_any >= 0 && qlora_provider && qlora_provider->supports_selective_moe()) {
        const bool refreshed =
            refresh_moe_experts_if_needed(layer_idx_any, host_offsets_ptr, num_experts, mWeights, mRunState.MainStream);
        if (refreshed) {
            invalidate_moe_fp8_cache(op.inputs[1].name);
        }
    }

    MoeCompactInfo compact = host_offsets_ptr ? build_moe_compact_info_from_host(host_offsets_ptr,
                                                                                 num_experts,
                                                                                 weight_experts,
                                                                                 layer_idx_any,
                                                                                 "moe_grouped_gemm")
                                              : build_moe_compact_info(expert_offsets_data,
                                                                       num_experts,
                                                                       weight_experts,
                                                                       mRunState.MainStream,
                                                                       layer_idx_any,
                                                                       "moe_grouped_gemm");
    if (!host_offsets_ptr && !compact.host_offsets.empty()) {
        host_offsets_ptr = compact.host_offsets.data();
    }
    const int* active_ptr = compact.active_experts.empty() ? nullptr : compact.active_experts.data();
    const int num_active = compact.active_experts.empty() ? -1 : compact.num_active;
    const bool weight_is_compact = compact.weight_is_compact;

    // Output shape: [total_tokens, out_features]
    const long total_tokens = inp.Sizes[0];
    std::vector<long> out_shape = {total_tokens, static_cast<long>(out_features)};
    Tensor out = mRunState.temp_alloc(inp.DType, out_shape, "moe_grouped_gemm_out");
    mTemps.push_back(out);

    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(out, mRunState.MainStream);
    } else if (mRecipe && inp.DType == ETensorDType::BF16 && !weight_is_compact && !is_llep_active) {
        // Recipe-driven MoE GEMM via cuDNN FE or FP8 (skip when LLEP active — cuDNN
        // crashes with variable merged expert counts; cuBLAS per-expert is safe)
        modules::MoeMatmulContext ctx;
        ctx.out = out.get<nv_bfloat16>();
        ctx.inp = inp.get<nv_bfloat16>();
        ctx.weights = weights.get<nv_bfloat16>();
        ctx.expert_offsets = expert_offsets_data;
        ctx.num_experts = num_experts;
        ctx.N = out_features;
        ctx.K = in_features;
        ctx.total_tokens = static_cast<int>(total_tokens);
        ctx.layer_idx = layer_idx_any;
        ctx.run_state = &mRunState;
        ctx.cudnn_handle = mRunState.CudnnHandle;
        ctx.cublas_handle = mRunState.cublas_handle();
        ctx.cublaslt_handle = mRunState.CublasLtHandle;
        ctx.workspace = mRunState.CuBlasWorkspace.get<std::byte>();
        ctx.workspace_size = mRunState.CuBlasWorkspace.bytes();
        ctx.stream = mRunState.MainStream;
        ctx.op_caps = op.default_caps;
        ctx.epilogue_support = op.epilogue_support;
        ctx.storage_compat = op.storage_compat;
        ctx.host_offsets = host_offsets_ptr;
        ctx.active_experts = active_ptr;
        ctx.num_active = num_active;
        ctx.weight_is_compact = weight_is_compact;
        ctx.allow_fp8 = op.attrs.allow_quant;
        attach_moe_fp8_cache(ctx, op.inputs[1].name);

        // Allocate FP8 buffers if using FP8 hybrid recipe
        Tensor inp_quant_buf, inp_stats_buf;
        if (mRecipe->is_fp8_hybrid() && ctx.allow_fp8) {
            const long num_elements = total_tokens * in_features;
            inp_quant_buf = mRunState.temp_alloc(ETensorDType::FP8_E4M3,
                                                 {total_tokens, static_cast<long>(in_features)},
                                                 "moe_grouped_gemm_inp_quant_buf");
            inp_stats_buf =
                mRunState.temp_alloc(ETensorDType::FP32, {2}, "moe_grouped_gemm_inp_stats_buf");  // abs_max, scale
            inp_quant_buf.Stats = inp_stats_buf.get<float>();
            ctx.inp_quant = &inp_quant_buf;
            mTemps.push_back(inp_quant_buf);
            mTemps.push_back(inp_stats_buf);
        }

        mRecipe->forward_moe_matmul(ctx);
    } else if (inp.DType == ETensorDType::BF16) {
        moe_grouped_gemm(out.get<nv_bfloat16>(),
                         inp.get<nv_bfloat16>(),
                         weights.get<nv_bfloat16>(),
                         expert_offsets_data,
                         num_experts,
                         out_features,
                         in_features,
                         mRunState.cublas_handle(),
                         mRunState.MainStream,
                         host_offsets_ptr,
                         /*alpha=*/1.0f,
                         /*beta=*/0.0f,
                         EMMTranspose::TN,
                         active_ptr,
                         weight_is_compact,
                         num_active,
                         llep_weight_ptrs);
    } else {
        moe_grouped_gemm(out.get<float>(),
                         inp.get<float>(),
                         weights.get<float>(),
                         expert_offsets_data,
                         num_experts,
                         out_features,
                         in_features,
                         mRunState.cublas_handle(),
                         mRunState.MainStream,
                         host_offsets_ptr,
                         /*alpha=*/1.0f,
                         /*beta=*/0.0f,
                         EMMTranspose::TN,
                         active_ptr,
                         weight_is_compact,
                         num_active,
                         llep_weight_ptrs);
    }

    // Apply grouped MoE LoRA (up projection) when enabled.
    if (mLoRAConfig && mLoRAWeights && mLoRARunState && mLoRAConfig->enabled() && mLoRAWeights->enabled() &&
        layer_idx_any >= 0) {
        auto& lora_block = mLoRAWeights->get_block(layer_idx_any, mRunState.MainStream);
        if (lora_block.moe.use_grouped) {
            // When LLEP is active, use merged LoRA tensors
            const auto* up_ptr = lora_block.moe.grouped.up.has_value() ? &(*lora_block.moe.grouped.up) : nullptr;
            {
                auto llep_lora_it = mLLEPStates.find(ep_key_any);
                if (llep_lora_it != mLLEPStates.end() && llep_lora_it->second.active &&
                    llep_lora_it->second.has_merged_lora && llep_lora_it->second.merged_lora.up.has_value()) {
                    up_ptr = &(*llep_lora_it->second.merged_lora.up);
                }
            }
            if (up_ptr && up_ptr->has_value()) {
                const auto& lora_up = *up_ptr;
                const int rank = mLoRAConfig->rank;
                const float scaling = mLoRAConfig->scaling();
                const float dropout = mLoRAConfig->dropout;
                const bool training = mLoRARunState->is_training;
                const long total_tokens_l = total_tokens;
                const int total_tokens_i = static_cast<int>(total_tokens_l);

                auto get_dropout_seed = [&](int proj_type) -> unsigned int {
                    return mLoRARunState->dropout_base_seed + static_cast<unsigned int>(layer_idx_any) * 1000000u +
                           static_cast<unsigned int>(proj_type) * 100000u +
                           static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;
                };

                auto view_or_temp = [&](Tensor& buf, long rows, long cols) -> Tensor {
                    const long need = rows * cols;
                    if (!buf.Data || buf.DType != out.DType || buf.nelem() < need) {
                        Tensor tmp = mRunState.temp_alloc(out.DType, {rows, cols}, "moe_grouped_gemm_lora_temp");
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
                                         expert_offsets_data,
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
                                         expert_offsets_data,
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
                                          lora_up.A,
                                          rank,
                                          in_features,
                                          1.0f,
                                          0.0f,
                                          EMMTranspose::TN);
                    scale_and_dropout(lora_intermediate, get_dropout_seed(4));
                    dispatch_grouped_gemm(out,
                                          lora_intermediate,
                                          lora_up.B,
                                          out_features,
                                          rank,
                                          1.0f,
                                          1.0f,
                                          EMMTranspose::TN);
                }
            }  // if (up_ptr && up_ptr->has_value())
        }  // if (use_grouped)
    }

    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_moe_grouped_gemm_backward(const CompiledOp& op) {
    // Backward pass: compute input gradient
    // Inputs: d_out, inp, weights, scatter_indices
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor weights = resolve_tensor(op.inputs[2]);  // copy for LLEP override
    Tensor& scatter_indices = resolve_tensor(op.inputs[3]);
    (void)scatter_indices;

    int num_experts = static_cast<int>(mConfig.NumLocalExperts);

    int layer_idx_any = op.attrs.layer_idx;
    if (layer_idx_any < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[1].name;  // inp
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        std::string field;
        parse_block_param(name, layer_idx_any, field);
    }
    const int input_total_recv = static_cast<int>(inp.Sizes[0]);
    int ep_key_any = ep_state_key(layer_idx_any);
    if (mOptions.EPSize > 1 && layer_idx_any >= 0) {
        const int ep_key_r0 = (layer_idx_any << 1);
        const int ep_key_r1 = ep_key_r0 | 1;
        auto it_r0 = mEpStates.find(ep_key_r0);
        auto it_r1 = mEpStates.find(ep_key_r1);
        if (it_r1 != mEpStates.end() && it_r1->second.total_recv == input_total_recv) {
            ep_key_any = ep_key_r1;
        } else if (it_r0 != mEpStates.end() && it_r0->second.total_recv == input_total_recv) {
            ep_key_any = ep_key_r0;
        } else if (it_r1 != mEpStates.end()) {
            ep_key_any = ep_key_r1;
        } else if (it_r0 != mEpStates.end()) {
            ep_key_any = ep_key_r0;
        }
    }

    // For EP, derive num_experts from the forward-cached offsets.
    // LLEP may change num_merged per layer (transferring experts between GPUs),
    // so mConfig.NumLocalExperts may not match the actual merged expert count.
    if (mOptions.EPSize > 1 && layer_idx_any >= 0) {
        auto ci = mMoEHostOffsetsCache.find(ep_key_any);
        if (ci == mMoEHostOffsetsCache.end()) {
            ci = mMoEHostOffsetsCache.find(layer_idx_any);
        }
        if (ci != mMoEHostOffsetsCache.end() && ci->second.size() > 1) {
            num_experts = static_cast<int>(ci->second.size()) - 1;
        }
    }

    // Get expert offsets
    Tensor expert_offsets_view;
    Tensor* expert_offsets_ptr = nullptr;
    if (layer_idx_any >= 0) {
        const std::string base_key = "blocks[" + std::to_string(layer_idx_any) + "].moe_expert_offsets";
        std::vector<std::string> candidate_keys;
        if (mOptions.EPSize > 1) {
            candidate_keys.push_back(base_key + "#r1");
            candidate_keys.push_back(base_key + "#r0");
        } else {
            candidate_keys.push_back(moe_saved_key(layer_idx_any, "moe_expert_offsets"));
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
            expert_offsets_view.Sizes[0] = static_cast<long>(num_experts + 1);
            expert_offsets_view.Data = static_cast<std::byte*>(saved_ptr);
            expert_offsets_ptr = &expert_offsets_view;
        }
    }
    if (!expert_offsets_ptr) {
        Tensor* moe_offsets_bwd_ptr = nullptr;
        if (op.attrs.moe_offsets_tensor_id >= 0 &&
            static_cast<std::size_t>(op.attrs.moe_offsets_tensor_id) < mTensors.size() &&
            mTensors[op.attrs.moe_offsets_tensor_id].Data) {
            moe_offsets_bwd_ptr = &mTensors[op.attrs.moe_offsets_tensor_id];
        }
        if (!moe_offsets_bwd_ptr) {
            throw std::runtime_error("moe_grouped_gemm_backward: expert_offsets not found");
        }
        expert_offsets_ptr = moe_offsets_bwd_ptr;
    }
    Tensor& expert_offsets = *expert_offsets_ptr;

    // LLEP per-expert weight pointer override for backward.
    // Case 1: Full LLEP state exists (last MoE layer) — use directly.
    // Case 2: No LLEP state but EP metadata exists (earlier layers) — reconstruct
    //         native-only weight pointers (foreign expert tokens get zero gradient).
    const void* const* llep_weight_ptrs = nullptr;
    bool is_llep_active = false;
    std::vector<const void*> reconstructed_weight_ptrs;  // local storage for case 2
    std::vector<const void*> refreshed_native_weight_ptrs;
    {
        auto llep_it = mLLEPStates.find(ep_key_any);
        if (llep_it != mLLEPStates.end() && llep_it->second.active) {
            // Case 1: full LLEP state with foreign weight pointers
            is_llep_active = true;
            auto& llep = llep_it->second;
            std::string_view wname = op.inputs[2].name;
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
                            "moe_grouped_gemm_backward: invalid native expert refresh mapping at layer " +
                            std::to_string(layer_idx_any) + " global_e=" + std::to_string(global_e) +
                            " local_idx=" + std::to_string(local_idx));
                    }
                    refreshed_native_weight_ptrs[m] = static_cast<const std::byte*>(weights.Data) +
                                                      static_cast<std::size_t>(local_idx) * expert_bytes;
                }
                llep_weight_ptrs = refreshed_native_weight_ptrs.data();
            } else if (wname.find("gate_up") != std::string_view::npos && !llep.gate_up_weight_ptrs.empty()) {
                llep_weight_ptrs = llep.gate_up_weight_ptrs.data();
            } else if (wname.find("down") != std::string_view::npos && !llep.down_weight_ptrs.empty()) {
                llep_weight_ptrs = llep.down_weight_ptrs.data();
            } else if (wname.find("_up") != std::string_view::npos && !llep.gate_up_weight_ptrs.empty()) {
                llep_weight_ptrs = llep.gate_up_weight_ptrs.data();
            }
            num_experts = llep.num_merged_experts;
            expert_offsets_view.Sizes[0] = static_cast<long>(num_experts + 1);
        } else if (mOptions.EPSize > 1 && layer_idx_any >= 0) {
            // Case 2: LLEP state cleared but metadata survives
            auto meta_it = mEPLayerMeta.find(ep_key_any);
            if (meta_it != mEPLayerMeta.end() && meta_it->second.num_merged != meta_it->second.num_local) {
                // num_merged > num_local means foreign experts were present in forward.
                // Reconstruct weight pointers: native experts use dequantized weights,
                // foreign experts use a zero buffer (their token gradients become zero).
                const auto& meta = meta_it->second;
                is_llep_active = true;
                num_experts = meta.num_merged;
                expert_offsets_view.Sizes[0] = static_cast<long>(num_experts + 1);

                const size_t elem_sz = get_dtype_size(weights.DType);
                const long w_rows = (weights.Rank >= 2) ? weights.Sizes[1] : 0;
                const long w_cols = (weights.Rank >= 3) ? weights.Sizes[2] : 0;
                const size_t expert_bytes = static_cast<size_t>(w_rows * w_cols) * elem_sz;
                // Allocate zero buffer for foreign expert weight pointers
                std::vector<long> zw_shape = {1L, w_rows, w_cols};
                Tensor zero_weight = mRunState.temp_alloc(weights.DType, zw_shape, "moe_grouped_gemm_zero_weight");
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

    const int out_features = (weights.Rank >= 2) ? static_cast<int>(weights.Sizes[1]) : 0;
    const int in_features = (weights.Rank >= 3) ? static_cast<int>(weights.Sizes[2]) : 0;
    const int weight_experts =
        llep_weight_ptrs ? num_experts : ((weights.Rank > 0) ? static_cast<int>(weights.Sizes[0]) : num_experts);
    const int* expert_offsets_data = expert_offsets.get<int>();

    const int* host_offsets_ptr = nullptr;
    if (num_experts > 0 && expert_offsets.Data) {
        // Use cached host offsets (populates on first backward access for this layer).
        host_offsets_ptr = get_or_sync_moe_host_offsets(ep_key_any, expert_offsets.get<int>(), num_experts);
    }

    MoeCompactInfo compact = host_offsets_ptr ? build_moe_compact_info_from_host(host_offsets_ptr,
                                                                                 num_experts,
                                                                                 weight_experts,
                                                                                 layer_idx_any,
                                                                                 "moe_grouped_gemm_backward")
                                              : build_moe_compact_info(expert_offsets_data,
                                                                       num_experts,
                                                                       weight_experts,
                                                                       mRunState.MainStream,
                                                                       layer_idx_any,
                                                                       "moe_grouped_gemm_backward");
    if (!host_offsets_ptr && !compact.host_offsets.empty()) {
        host_offsets_ptr = compact.host_offsets.data();
    }
    const int* active_ptr = compact.active_experts.empty() ? nullptr : compact.active_experts.data();
    const int num_active = compact.active_experts.empty() ? -1 : compact.num_active;
    const bool weight_is_compact = compact.weight_is_compact;

    auto* qlora_provider = mWeights.qlora_provider();
    if (qlora_provider && qlora_provider->supports_selective_moe()) {
        const int* refresh_offsets = host_offsets_ptr;
        if (refresh_offsets) {
            const bool refreshed = refresh_moe_experts_if_needed(layer_idx_any,
                                                                 refresh_offsets,
                                                                 num_experts,
                                                                 mWeights,
                                                                 mRunState.MainStream);
            if (refreshed) {
                invalidate_moe_fp8_cache(op.inputs[2].name);
            }
        }
    }

    // Input gradient shape: same as inp
    const long total_tokens = d_out.Sizes[0];
    std::vector<long> d_inp_shape = {total_tokens, static_cast<long>(in_features)};
    Tensor d_inp = mRunState.temp_alloc(d_out.DType, d_inp_shape, "moe_grouped_gemm_backward_d_input");
    mTemps.push_back(d_inp);

    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(d_inp, mRunState.MainStream);
    } else if (mRecipe && d_out.DType == ETensorDType::BF16 && !is_llep_active && mOptions.EPSize <= 1) {
        // Recipe-driven backward MoE GEMM (skip when EP active: gradient checkpointing
        // may clear LLEP state during multi-layer recompute, leaving intermediate layers
        // without state; direct kernel path handles this correctly)
        modules::MoeMatmulContext ctx;
        ctx.dinp = d_inp.get<nv_bfloat16>();
        ctx.dout = d_out.get<nv_bfloat16>();
        ctx.weights = weights.get<nv_bfloat16>();
        ctx.expert_offsets = expert_offsets_data;
        ctx.num_experts = num_experts;
        ctx.N = out_features;
        ctx.K = in_features;
        ctx.total_tokens = static_cast<int>(total_tokens);
        ctx.layer_idx = layer_idx_any;
        ctx.run_state = &mRunState;
        ctx.cublas_handle = mRunState.cublas_handle();
        ctx.cublaslt_handle = mRunState.CublasLtHandle;
        ctx.stream = mRunState.MainStream;
        ctx.op_caps = op.default_caps;
        ctx.epilogue_support = op.epilogue_support;
        ctx.storage_compat = op.storage_compat;
        ctx.host_offsets = host_offsets_ptr;
        ctx.active_experts = active_ptr;
        ctx.num_active = num_active;
        ctx.weight_is_compact = weight_is_compact;
        ctx.skip_weight_grad = false;
        ctx.allow_fp8 = op.attrs.allow_quant;
        attach_moe_fp8_cache(ctx, op.inputs[2].name);

        // Allocate FP8 buffers if using FP8 hybrid recipe
        Tensor dout_quant_buf, dout_stats_buf, inp_quant_buf, inp_stats_buf;
        if (mRecipe->is_fp8_hybrid() && ctx.allow_fp8) {
            const long num_elements = total_tokens * out_features;
            dout_quant_buf = mRunState.temp_alloc(ETensorDType::FP8_E5M2,
                                                  {total_tokens, static_cast<long>(out_features)},
                                                  "moe_grouped_gemm_dout_quant_buf");
            dout_stats_buf =
                mRunState.temp_alloc(ETensorDType::FP32, {2}, "moe_grouped_gemm_dout_stats_buf");  // abs_max, scale
            dout_quant_buf.Stats = dout_stats_buf.get<float>();
            ctx.dout_quant = &dout_quant_buf;
            mTemps.push_back(dout_quant_buf);
            mTemps.push_back(dout_stats_buf);

            inp_quant_buf = mRunState.temp_alloc(ETensorDType::FP8_E4M3,
                                                 {total_tokens, static_cast<long>(in_features)},
                                                 "moe_grouped_gemm_bwd_inp_quant_buf");
            inp_stats_buf = mRunState.temp_alloc(ETensorDType::FP32, {2}, "moe_grouped_gemm_bwd_inp_stats_buf");
            inp_quant_buf.Stats = inp_stats_buf.get<float>();
            ctx.inp_quant = &inp_quant_buf;
            mTemps.push_back(inp_quant_buf);
            mTemps.push_back(inp_stats_buf);
        }

        mRecipe->backward_moe_matmul(ctx);
    } else if (d_out.DType == ETensorDType::BF16) {
        // Direct kernel call (also used as LLEP fallback when recipe path is skipped)
        moe_grouped_gemm_up_backward(d_inp.get<nv_bfloat16>(),
                                     d_out.get<nv_bfloat16>(),
                                     weights.get<nv_bfloat16>(),
                                     expert_offsets_data,
                                     num_experts,
                                     in_features,
                                     out_features,
                                     mRunState.cublas_handle(),
                                     mRunState.MainStream,
                                     host_offsets_ptr,
                                     active_ptr,
                                     weight_is_compact,
                                     num_active,
                                     llep_weight_ptrs);
    } else {
        moe_grouped_gemm_up_backward(d_inp.get<float>(),
                                     d_out.get<float>(),
                                     weights.get<float>(),
                                     expert_offsets_data,
                                     num_experts,
                                     in_features,
                                     out_features,
                                     mRunState.cublas_handle(),
                                     mRunState.MainStream,
                                     host_offsets_ptr,
                                     active_ptr,
                                     weight_is_compact,
                                     num_active,
                                     llep_weight_ptrs);
    }

    // Apply grouped MoE LoRA backward (up projection) when enabled.
    if (mLoRAConfig && mLoRAWeights && mLoRARunState && mLoRAConfig->enabled() && mLoRAWeights->enabled() &&
        layer_idx_any >= 0) {
        auto& lora_block = mLoRAWeights->get_block(layer_idx_any, mRunState.MainStream);
        if (lora_block.moe.use_grouped) {
            // When LLEP is active, use merged LoRA tensors
            const auto* up_ptr = lora_block.moe.grouped.up.has_value() ? &(*lora_block.moe.grouped.up) : nullptr;
            bool llep_lora_active = false;
            {
                auto llep_lora_it = mLLEPStates.find(ep_key_any);
                if (llep_lora_it != mLLEPStates.end() && llep_lora_it->second.active &&
                    llep_lora_it->second.has_merged_lora && llep_lora_it->second.merged_lora.up.has_value()) {
                    up_ptr = &(*llep_lora_it->second.merged_lora.up);
                    llep_lora_active = true;
                }
            }
            if (up_ptr && up_ptr->has_value()) {
                const auto& lora_up = *up_ptr;
                const int rank = mLoRAConfig->rank;
                const float scaling = mLoRAConfig->scaling();
                const float dropout = mLoRAConfig->dropout;
                const bool training = mLoRARunState->is_training;
                const long total_tokens_l = d_out.Sizes[0];
                const int total_tokens_i = static_cast<int>(total_tokens_l);

                auto get_dropout_seed = [&](int proj_type) -> unsigned int {
                    return mLoRARunState->dropout_base_seed + static_cast<unsigned int>(layer_idx_any) * 1000000u +
                           static_cast<unsigned int>(proj_type) * 100000u +
                           static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;
                };

                auto view_or_temp = [&](Tensor& buf, long rows, long cols) -> Tensor {
                    const long need = rows * cols;
                    if (!buf.Data || buf.DType != d_out.DType || buf.nelem() < need) {
                        Tensor tmp = mRunState.temp_alloc(d_out.DType, {rows, cols}, "moe_grouped_gemm_lora_temp");
                        mTemps.push_back(tmp);
                        return tmp;
                    }
                    Tensor view = buf;
                    view.DType = d_out.DType;
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
                                         expert_offsets_data,
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
                                         expert_offsets_data,
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

                auto dispatch_weight_grad =
                    [&](Tensor& d_weight, const Tensor& grad_output, const Tensor& in, int M, int N, float beta) {
                        if (grad_output.DType != in.DType) {
                            throw std::runtime_error("MoE LoRA backward: grad/output dtype mismatch.");
                        }
                        Tensor d_weight_view = d_weight;
                        int weight_rows =
                            (d_weight_view.Rank > 0) ? static_cast<int>(d_weight_view.Sizes[0]) : num_experts;
                        if (mOptions.EPSize > 1 && layer_idx_any >= 0 && d_weight_view.Rank >= 3 &&
                            weight_rows > num_experts) {
                            auto meta_it = mEPLayerMeta.find(ep_key_any);
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
                        if (grad_output.DType == ETensorDType::BF16) {
                            if (try_moe_fp8_weight_grad(d_weight_view,
                                                        grad_output,
                                                        in,
                                                        expert_offsets_data,
                                                        num_experts,
                                                        M,
                                                        N,
                                                        host_offsets_ptr,
                                                        lora_active_ptr,
                                                        lora_weight_is_compact,
                                                        lora_num_active,
                                                        beta,
                                                        "moe_grouped_gemm_lora")) {
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
                                                         expert_offsets_data,
                                                         num_experts,
                                                         M,
                                                         N,
                                                         mRunState.cublas_handle(),
                                                         mRunState.MainStream,
                                                         host_offsets_ptr,
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
                                                         expert_offsets_data,
                                                         num_experts,
                                                         M,
                                                         N,
                                                         mRunState.cublas_handle(),
                                                         mRunState.MainStream,
                                                         host_offsets_ptr,
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
                    lora_grads = &mLoRAGrads->get_block_full(layer_idx_any, mRunState.MainStream, *mComm, lora_accum);
                }
                const float grad_beta = lora_accum ? 1.0f : 0.0f;

                if (total_tokens_i > 0 && rank > 0) {
                    Tensor lora_intermediate =
                        view_or_temp(mLoRARunState->moe_lora_intermediate1, total_tokens_l, rank);
                    const unsigned int seed_up = get_dropout_seed(4);

                    // dB: intermediate = x @ A^T
                    dispatch_grouped_gemm(lora_intermediate,
                                          inp,
                                          lora_up.A,
                                          rank,
                                          in_features,
                                          1.0f,
                                          0.0f,
                                          EMMTranspose::TN);
                    scale_and_dropout(lora_intermediate, seed_up);
                    if (lora_grads && lora_grads->moe.grouped.up.has_value()) {
                        dispatch_weight_grad(lora_grads->moe.grouped.up->B,
                                             d_out,
                                             lora_intermediate,
                                             out_features,
                                             rank,
                                             grad_beta);
                    }

                    // intermediate = d_out @ B
                    dispatch_grouped_gemm(lora_intermediate,
                                          d_out,
                                          lora_up.B,
                                          rank,
                                          out_features,
                                          1.0f,
                                          0.0f,
                                          EMMTranspose::NN);
                    scale_and_dropout(lora_intermediate, seed_up);
                    if (lora_grads && lora_grads->moe.grouped.up.has_value()) {
                        dispatch_weight_grad(lora_grads->moe.grouped.up->A,
                                             lora_intermediate,
                                             inp,
                                             rank,
                                             in_features,
                                             grad_beta);
                    }

                    // d_input += intermediate @ A
                    dispatch_grouped_gemm(d_inp,
                                          lora_intermediate,
                                          lora_up.A,
                                          in_features,
                                          rank,
                                          1.0f,
                                          1.0f,
                                          EMMTranspose::NN);
                }
            }  // if (up_ptr && up_ptr->has_value())
        }  // if (use_grouped)
    }

    store_tensor(op.outputs[0], d_inp);
}

namespace {

// -----------------------------------------------------------------------------
// MoE Grouped GEMM backward rule (generic version without fused activation)
// Forward: out = moe_grouped_gemm(inp, weights, scatter_indices)
// Backward: d_inp = moe_grouped_gemm_backward(d_out, inp, weights, scatter_indices)
// Used for Nemotron-H MoE blocks that use relu2 activation instead of swiglu
// -----------------------------------------------------------------------------
std::vector<Operation> moe_grouped_gemm_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string inp = fwd.inputs[0];
        std::string weights = fwd.inputs[1];
        std::string scatter_indices = fwd.inputs[2];

        std::string inp_ref = ctx.is_param(inp) ? inp : saved_ref(inp);
        std::string weights_ref = ctx.is_param(weights) ? weights : saved_ref(weights);
        std::string scatter_ref = saved_ref(scatter_indices);

        ops.push_back(make_operation("moe_grouped_gemm_backward_" + std::to_string(ctx.op_counter++),
                                     "moe_grouped_gemm_backward",
                                     "moe_grouped_gemm_backward",
                                     {ctx.d_output, inp_ref, weights_ref, scatter_ref},
                                     {ctx.d_inputs[0]}));
    }

    return ops;
}

}  // namespace

// Upper bound for dispatch_moe_grouped_gemm_backward. Dominant temp is
// `d_inp` = [total_tokens, in_features] in the d_out dtype; the FP8-hybrid
// recipe path also stages a quantized d_out + scalar stats which is smaller
// than d_inp and so rolled into a 1.5x factor.
//
// Shapes taken from op.inputs: d_out is inputs[0] of rank 2
// [total_tokens, out_features]; inp is inputs[1] of rank 2
// [total_tokens, in_features].
long moe_grouped_gemm_backward_stack_bound(const CompiledOp& op, const BufferPlan& plan) {
    if (op.inputs.size() < 2) return 0;
    const auto& dout_shape = op.inputs[0].shape;
    const auto& inp_shape = op.inputs[1].shape;
    if (dout_shape.size() < 2 || inp_shape.size() < 2) return 0;
    const long total_tokens = dout_shape[0];
    const long in_features = inp_shape[1];
    const long out_features = dout_shape[1];
    const long dtype_bytes = static_cast<long>(get_dtype_size(op.inputs[0].dtype));
    if (total_tokens <= 0 || in_features <= 0 || dtype_bytes <= 0) return 0;

    long bytes = align_stack_bytes(total_tokens * in_features * dtype_bytes);  // d_inp
    // FP8 hybrid: add dout_quant (E5M2, 1 byte/elem) and 2 FP32 scalars.
    // Gated by recipe at runtime; bound unconditionally to keep the upper
    // bound valid regardless of recipe selection.
    bytes += align_stack_bytes(total_tokens * out_features);
    bytes += align_stack_bytes(2 * 4);
    (void)plan;
    return bytes;
}

}  // namespace dsl

REGISTER_AUTODIFF("moe_grouped_gemm", ::dsl::moe_grouped_gemm_backward);
REGISTER_STACK_BOUND("moe_grouped_gemm_backward", MoEGroupedGemmBackward, ::dsl::moe_grouped_gemm_backward_stack_bound);
