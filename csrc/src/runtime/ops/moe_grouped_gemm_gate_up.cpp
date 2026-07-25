#include "runtime/executor/compiled_ops.h"

#include <algorithm>
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
#include "runtime/dsl/dsl_param_store.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "recipes/recipe.h"
#include "utilities/dtype.h"
#include "runtime/lora/lora_config.h"
#include "runtime/lora/lora_grads_manager.h"
#include "runtime/lora/lora_run_state.h"
#include "runtime/lora/lora_weights_manager.h"
#include "runtime/lora/lora_model_utils.h"

namespace dsl {
namespace {

void attach_token_role(modules::MoeMatmulContext& ctx, const CompiledGraph* graph, const TensorRef& ref) {
    if (!graph) {
        return;
    }
    if (const TensorRole* role = graph->role_for_tensor_id(ref.tensor_id)) {
        ctx.token_role = *role;
        ctx.has_token_role = true;
    }
}

}  // namespace

void CompiledExecutor::dispatch_moe_grouped_gemm_gate_up(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor weights =
        resolve_tensor(op.inputs[1]);  // Parameter name resolved by graph compiler (copy for LLEP override)
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
            auto it_saved = mSavedCache.buffers().find(key);
            if (it_saved != mSavedCache.buffers().end() && it_saved->second != nullptr) {
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
            throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets not found");
        }
        expert_offsets_ptr = moe_offsets_fwd_ptr;
    }
    Tensor& expert_offsets = *expert_offsets_ptr;
    if (expert_offsets.DType != ETensorDType::INT32) {
        throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets dtype is not INT32");
    }
    if (!expert_offsets.Data) {
        throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets has null data");
    }

    int num_experts = num_experts_for_offsets;
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
    // Use MoeIntermediateSize for MoE models (may differ from IntermediateSize)
    const int intermediate_size = (mConfig.MoeIntermediateSize > 0) ? static_cast<int>(mConfig.MoeIntermediateSize)
                                                                    : static_cast<int>(mConfig.IntermediateSize);
    const bool gate_up_interleaved = op.attrs.gate_up_interleaved;

    // LLEP per-expert weight pointer override: when LLEP is active, use per-expert
    // pointers (native dequant buffer + foreign P2P receive) instead of contiguous weights.
    bool is_llep_active = false;
    const void* const* llep_weight_ptrs = nullptr;
    {
        auto llep_it = mLLEPStates.find(ep_key_any);
        if (llep_it != mLLEPStates.end() && llep_it->second.active) {
            auto& llep = llep_it->second;
            num_experts = llep.num_merged_experts;
            expert_offsets_view.Sizes[0] = static_cast<long>(num_experts + 1);
            llep_weight_ptrs = llep.gate_up_weight_ptrs.data();
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

    // When LLEP is active, per-expert weight pointers handle indexing — use merged expert count.
    // Otherwise use the weight tensor's leading dimension (local expert count).
    const int weight_experts =
        llep_weight_ptrs ? num_experts : ((weights.Rank > 0) ? static_cast<int>(weights.Sizes[0]) : num_experts);
    const bool offsets_owned = mRunState.Stack.owns(expert_offsets.Data);
    if (offsets_owned && !mRunState.Stack.is_live(expert_offsets.Data)) {
        throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets pointer is not live");
    }
    const int* host_offsets_ptr = nullptr;
    if (num_experts > 0 && expert_offsets.Data) {
        // Use cached host offsets (populated by dispatch_moe_permute for this layer).
        host_offsets_ptr = get_or_sync_moe_host_offsets(ep_key_any, expert_offsets.get<int>(), num_experts);
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
            std::string msg = "moe_grouped_gemm_gate_up: expert_offsets invalid (layer=";
            msg += std::to_string(layer_idx_any);
            msg += ", bad=" + std::to_string(bad);
            msg += ", oob=" + std::to_string(oob);
            msg += ", last=" + std::to_string(last);
            msg += ", total_tokens=" + std::to_string(total_tokens);
            msg += ", num_experts=" + std::to_string(num_experts);
            msg += ")";
            throw std::runtime_error(msg);
        }
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
    Tensor out = mRunState.temp_alloc(inp.DType, out_shape, "moe_grouped_gemm_gate_up_out");
    mTemps.push_back(out);

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
            // Skip weight bounds check when LLEP per-expert pointers are active:
            // per-expert pointers address separate memory regions (native dequant buffer
            // or foreign P2P receive buffer), not offsets into the contiguous weight tensor.
            if (!llep_weight_ptrs) {
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
    }
    // Use weights dtype to determine compute precision (QLoRA may return FP32 dequantized weights)
    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(out, mRunState.MainStream);
    } else if (mRecipe && weights.DType == ETensorDType::BF16 && !weight_is_compact && !is_llep_active) {
        // Recipe-driven MoE GEMM via cuDNN FE (skip when LLEP active — cuDNN
        // crashes with variable merged expert counts; cuBLAS per-expert is safe)
        // gate_up weight is (E, 2*D, C) → N=2*D, K=C
        modules::MoeMatmulContext ctx;
        ctx.out = out.get<nv_bfloat16>();
        ctx.inp = inp.get<nv_bfloat16>();
        ctx.weights = weights.get<nv_bfloat16>();
        ctx.expert_offsets = expert_offsets.get<int>();
        ctx.num_experts = num_experts;
        ctx.N = 2 * intermediate_size;
        ctx.K = hidden_size;
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
        attach_token_role(ctx, mCurrentGraph, op.inputs[0]);
        attach_moe_fp8_cache(ctx, op.inputs[1].name);
        mRecipe->forward_moe_matmul(ctx);
    } else if (weights.DType == ETensorDType::BF16) {
        moe_grouped_gemm_gate_up(out.get<nv_bfloat16>(),
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
        moe_grouped_gemm_gate_up(out.get<float>(),
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

    auto apply_grouped_moe_gate_up_lora = [&]() {
        // Apply grouped MoE LoRA (gate/up) when enabled.
        if (!(mLoRAConfig && mLoRAWeights && mLoRARunState && mLoRAConfig->enabled() && mLoRAWeights->enabled() &&
              layer_idx_any >= 0)) {
            return;
        }
        auto& lora_block = mLoRAWeights->get_block(layer_idx_any, mRunState.MainStream);
        if (lora_block.moe.use_grouped) {
            // When LLEP is active, use merged LoRA tensors [num_merged, ...]
            const auto* lora_grouped_ptr = &lora_block.moe.grouped;
            {
                auto llep_lora_it = mLLEPStates.find(ep_key_any);
                if (llep_lora_it != mLLEPStates.end() && llep_lora_it->second.has_merged_lora) {
                    lora_grouped_ptr = &llep_lora_it->second.merged_lora;
                }
            }
            const auto& grouped = *lora_grouped_ptr;
            const int rank = mLoRAConfig->rank;
            const float scaling = mLoRAConfig->scaling();
            const float dropout = mLoRAConfig->dropout;
            const bool training = mLoRARunState->is_training;
            const long total_tokens_l = total_tokens;
            const int total_tokens_i = static_cast<int>(total_tokens_l);
            const int micro_step = mLoRARunState->micro_step;
            bool lora_applied = false;

            auto get_dropout_seed = [&](int proj_type) -> unsigned int {
                return mLoRARunState->dropout_base_seed + static_cast<unsigned int>(layer_idx_any) * 1000000u +
                       static_cast<unsigned int>(proj_type) * 100000u +
                       static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;
            };

            auto view_or_temp = [&](Tensor& buf, long rows, long cols) -> Tensor {
                const long need = rows * cols;
                if (!buf.Data || buf.DType != out.DType || buf.nelem() < need) {
                    Tensor tmp = mRunState.temp_alloc(out.DType, {rows, cols}, "moe_grouped_gemm_gate_up_temp");
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
                if (mOptions.EPSize > 1 && layer_idx_any >= 0 && weight_view.Rank >= 3 && weight_rows > num_experts) {
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
                Tensor lora_intermediate = view_or_temp(mLoRARunState->moe_lora_intermediate1, total_tokens_l, rank);
                const bool has_gate_up = grouped.gate_up.has_value() && grouped.gate_up->has_value();

                if (has_gate_up) {
                    Tensor lora_gate_up = view_or_temp(mLoRARunState->moe_lora_gate_up, total_tokens_l, gate_up_dim);
                    dispatch_grouped_gemm(lora_intermediate,
                                          inp,
                                          grouped.gate_up->A,
                                          rank,
                                          hidden_size,
                                          1.0f,
                                          0.0f,
                                          EMMTranspose::TN);
                    scale_and_dropout(lora_intermediate, get_dropout_seed(7));
                    dispatch_grouped_gemm(lora_gate_up,
                                          lora_intermediate,
                                          grouped.gate_up->B,
                                          gate_up_dim,
                                          rank,
                                          1.0f,
                                          0.0f,
                                          EMMTranspose::TN);
                    vector_add_sr(out, out, lora_gate_up, 1.0f, out.nelem(), /*seed=*/0, mRunState.MainStream);
                    lora_applied = true;
                } else {
                    Tensor lora_gate = view_or_temp(mLoRARunState->moe_lora_gate, total_tokens_l, intermediate_size);
                    Tensor lora_up = view_or_temp(mLoRARunState->moe_lora_up, total_tokens_l, intermediate_size);

                    bool has_gate = false;
                    bool has_up = false;

                    if (grouped.gate.has_value() && grouped.gate->has_value()) {
                        dispatch_grouped_gemm(lora_intermediate,
                                              inp,
                                              grouped.gate->A,
                                              rank,
                                              hidden_size,
                                              1.0f,
                                              0.0f,
                                              EMMTranspose::TN);
                        scale_and_dropout(lora_intermediate, get_dropout_seed(5));
                        dispatch_grouped_gemm(lora_gate,
                                              lora_intermediate,
                                              grouped.gate->B,
                                              intermediate_size,
                                              rank,
                                              1.0f,
                                              0.0f,
                                              EMMTranspose::TN);
                        has_gate = true;
                        if (!gate_up_interleaved) {
                            add_2d_slice(out,
                                         lora_gate,
                                         total_tokens_l,
                                         gate_up_dim,
                                         intermediate_size,
                                         /*dst_col_offset=*/intermediate_size,
                                         mRunState.MainStream);
                        }
                    }

                    if (grouped.up.has_value() && grouped.up->has_value()) {
                        dispatch_grouped_gemm(lora_intermediate,
                                              inp,
                                              grouped.up->A,
                                              rank,
                                              hidden_size,
                                              1.0f,
                                              0.0f,
                                              EMMTranspose::TN);
                        scale_and_dropout(lora_intermediate, get_dropout_seed(4));
                        dispatch_grouped_gemm(lora_up,
                                              lora_intermediate,
                                              grouped.up->B,
                                              intermediate_size,
                                              rank,
                                              1.0f,
                                              0.0f,
                                              EMMTranspose::TN);
                        has_up = true;
                        if (!gate_up_interleaved) {
                            add_2d_slice(out,
                                         lora_up,
                                         total_tokens_l,
                                         gate_up_dim,
                                         intermediate_size,
                                         /*dst_col_offset=*/0,
                                         mRunState.MainStream);
                        }
                    }

                    if (gate_up_interleaved) {
                        if (has_gate && has_up) {
                            add_gate_up_interleaved(out,
                                                    lora_up,
                                                    lora_gate,
                                                    total_tokens_i,
                                                    intermediate_size,
                                                    mRunState.MainStream);
                        } else if (has_gate) {
                            add_gate_up_interleaved_gate(out,
                                                         lora_gate,
                                                         total_tokens_i,
                                                         intermediate_size,
                                                         mRunState.MainStream);
                        } else if (has_up) {
                            add_gate_up_interleaved_up(out,
                                                       lora_up,
                                                       total_tokens_i,
                                                       intermediate_size,
                                                       mRunState.MainStream);
                        }
                    }
                    lora_applied = has_gate || has_up;
                }
            }
        }
    };

    AfterProduceHookPayload after_produce_payload;
    std::string_view after_produce_slot =
        grouped_lora_after_produce_slot(op, modules::LoRATargetId::ExpertGateUp, "expert_gate_up");
    if (after_produce_slot.empty()) {
        after_produce_slot = grouped_lora_after_produce_slot(op, modules::LoRATargetId::ExpertGate, "expert_gate_up");
    }
    if (after_produce_slot.empty()) {
        after_produce_slot = grouped_lora_after_produce_slot(op, modules::LoRATargetId::ExpertUp, "expert_gate_up");
    }
    if (!after_produce_slot.empty()) {
        after_produce_payload.apply_lora_action = apply_grouped_moe_gate_up_lora;
        dispatch_schema_hook(HookEventKind::AfterProduce,
                             layer_idx_any,
                             op.attrs.hook_schema_id,
                             after_produce_slot,
                             &after_produce_payload);
    }
    if (!after_produce_payload.lora_applied) {
        apply_grouped_moe_gate_up_lora();
    }

    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_moe_grouped_gemm_gate_up_backward(const CompiledOp& op) {
    Tensor& d_gate_up = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor weights = resolve_tensor(op.inputs[2]);  // copy for LLEP override
    Tensor& d_input = ensure_output_tensor(op.outputs[0]);
    Tensor* d_input_ptr = &d_input;
    const long expected_nelem = static_cast<long>(inp.nelem());
    if (d_input_ptr->nelem() != expected_nelem) {
        std::vector<long> shape(inp.Sizes.begin(), inp.Sizes.begin() + inp.Rank);
        Tensor tmp = mRunState.temp_alloc(inp.DType, shape, "moe_grouped_gemm_gate_up_d_input_temp");
        mTemps.push_back(tmp);
        store_tensor(op.outputs[0], tmp);
        d_input_ptr = &mTensors[op.outputs[0].tensor_id];
    }
    if (d_input_ptr->Device == -1 && mRunState.Stack.owns(d_input_ptr->Data)) {
        d_input_ptr->Device = mRunState.Stack.device_id();
    }

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
            auto it = mSavedCache.buffers().find(key);
            if (it != mSavedCache.buffers().end() && it->second != nullptr) {
                saved_ptr = it->second;
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
        Tensor* moe_offsets_bwd_ptr = nullptr;
        if (op.attrs.moe_offsets_tensor_id >= 0 &&
            static_cast<std::size_t>(op.attrs.moe_offsets_tensor_id) < mTensors.size() &&
            mTensors[op.attrs.moe_offsets_tensor_id].Data) {
            moe_offsets_bwd_ptr = &mTensors[op.attrs.moe_offsets_tensor_id];
        }
        if (!moe_offsets_bwd_ptr) {
            throw std::runtime_error("moe_grouped_gemm_gate_up_backward: expert_offsets not found");
        }
        expert_offsets_ptr = moe_offsets_bwd_ptr;
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
    {
        auto llep_it = mLLEPStates.find(ep_key);
        // cpu_training: stored native pointers reference a forward-time prefetch
        // buffer that has been recycled — always reconstruct with fresh pointers.
        if (llep_it != mLLEPStates.end() && llep_it->second.active && !mOptions.CpuTraining) {
            auto& llep = llep_it->second;
            is_llep_active = true;
            num_experts = llep.num_merged_experts;
            expert_offsets_view.Sizes[0] = static_cast<long>(num_experts + 1);
            llep_weight_ptrs = llep.gate_up_weight_ptrs.data();
        } else if (mOptions.EPSize > 1 && layer_idx >= 0) {
            auto meta_it = mEPLayerMeta.find(ep_key);
            auto merged_set_differs = [](const auto& meta) {
                if (meta.num_merged != meta.num_local) return true;
                for (int m = 0; m < meta.num_merged; ++m) {
                    const int li = meta.merged_to_global[m] - meta.native_start;
                    if (li < 0 || li >= meta.num_local) return true;
                }
                return false;
            };
            if (meta_it != mEPLayerMeta.end() && merged_set_differs(meta_it->second)) {
                const auto& meta = meta_it->second;
                is_llep_active = true;
                num_experts = meta.num_merged;
                expert_offsets_view.Sizes[0] = static_cast<long>(num_experts + 1);

                const size_t elem_sz = get_dtype_size(weights.DType);
                // gate_up weights: [E, 2*intermediate, hidden]
                const long gu_rows = (weights.Rank >= 2) ? weights.Sizes[1] : 0;
                const long gu_cols = (weights.Rank >= 3) ? weights.Sizes[2] : 0;
                const size_t expert_bytes = static_cast<size_t>(gu_rows * gu_cols) * elem_sz;
                // Foreign experts: under cpu_training re-fetch their rows from the
                // GLOBAL pinned host master so spilled tokens get an exact dgrad.
                // Without a host master (resident mode, LLEP state evicted) fall
                // back to zeros — the legacy approximation.
                std::vector<std::pair<int, int>> foreign_slot;  // (global_e, stage slot)
                for (int m = 0; m < meta.num_merged; ++m) {
                    const int e = meta.merged_to_global[m];
                    const int li = e - meta.native_start;
                    if (li < 0 || li >= meta.num_local) {
                        foreign_slot.emplace_back(e, static_cast<int>(foreign_slot.size()));
                    }
                }
                // cpu_training: the replay dispatch for this ep_key fetched this
                // layer's foreign experts into the ring arena moments ago, and
                // the arena keeps them alive until the next dispatch. Reuse
                // those pointers for dgrad instead of re-staging the same rows
                // from the pinned host master on MainStream (native pointers
                // are rebuilt from the op's fresh `weights` tensor either way).
                const std::vector<const void*>* llep_foreign_ptrs = nullptr;
                if (llep_it != mLLEPStates.end() && llep_it->second.active &&
                    static_cast<int>(llep_it->second.gate_up_weight_ptrs.size()) == meta.num_merged &&
                    llep_it->second.merged_to_global == meta.merged_to_global) {
                    llep_foreign_ptrs = &llep_it->second.gate_up_weight_ptrs;
                }
                Tensor foreign_stage;
                bool have_foreign_stage = false;
                if (!foreign_slot.empty() && !llep_foreign_ptrs && mOptions.CpuTraining) {
                    Tensor* master =
                        mWeights.master_tensor("blocks[" + std::to_string(layer_idx) + "].experts_gate_up");
                    if (master && master->Rank >= 3 && master->DType == weights.DType) {
                        foreign_stage = mRunState.temp_alloc(weights.DType,
                                                             {static_cast<long>(foreign_slot.size()), gu_rows, gu_cols},
                                                             "moe_grouped_gemm_gate_up_foreign_stage");
                        mTemps.push_back(foreign_stage);
                        for (const auto& [e, slot] : foreign_slot) {
                            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(foreign_stage.Data) +
                                                           static_cast<size_t>(slot) * expert_bytes,
                                                       static_cast<const std::byte*>(master->Data) +
                                                           static_cast<size_t>(e) * expert_bytes,
                                                       expert_bytes,
                                                       cudaMemcpyHostToDevice,
                                                       mRunState.MainStream));
                        }
                        have_foreign_stage = true;
                    }
                }
                Tensor zero_weight;
                if (!foreign_slot.empty() && !have_foreign_stage) {
                    std::vector<long> zw_shape = {1L, gu_rows, gu_cols};
                    zero_weight =
                        mRunState.temp_alloc(weights.DType, zw_shape, "moe_grouped_gemm_gate_up_zero_weight");
                    fill_zero(zero_weight, mRunState.MainStream);
                    mTemps.push_back(zero_weight);
                }

                reconstructed_weight_ptrs.resize(meta.num_merged);
                for (int m = 0; m < meta.num_merged; ++m) {
                    const int global_e = meta.merged_to_global[m];
                    const int local_idx = global_e - meta.native_start;
                    if (local_idx >= 0 && local_idx < meta.num_local) {
                        reconstructed_weight_ptrs[m] =
                            static_cast<const std::byte*>(weights.Data) + static_cast<size_t>(local_idx) * expert_bytes;
                    } else if (llep_foreign_ptrs) {
                        reconstructed_weight_ptrs[m] = (*llep_foreign_ptrs)[m];
                    } else if (have_foreign_stage) {
                        int slot = 0;
                        for (const auto& [e, s] : foreign_slot) {
                            if (e == global_e) {
                                slot = s;
                                break;
                            }
                        }
                        reconstructed_weight_ptrs[m] = static_cast<const std::byte*>(foreign_stage.Data) +
                                                       static_cast<size_t>(slot) * expert_bytes;
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

    const int weight_experts =
        llep_weight_ptrs ? num_experts : ((weights.Rank > 0) ? static_cast<int>(weights.Sizes[0]) : num_experts);

    // Get host offsets from cache (populates on first backward access for this layer).
    const int* cached_host_offsets = get_or_sync_moe_host_offsets(ep_key, expert_offsets_ptr->get<int>(), num_experts);

    MoeCompactInfo compact = cached_host_offsets ? build_moe_compact_info_from_host(cached_host_offsets,
                                                                                    num_experts,
                                                                                    weight_experts,
                                                                                    layer_idx,
                                                                                    "moe_grouped_gemm_gate_up_backward")
                                                 : build_moe_compact_info(expert_offsets_ptr->get<int>(),
                                                                          num_experts,
                                                                          weight_experts,
                                                                          mRunState.MainStream,
                                                                          layer_idx,
                                                                          "moe_grouped_gemm_gate_up_backward");
    const bool weight_is_compact = compact.weight_is_compact;
    const int* host_offsets_ptr = cached_host_offsets;
    if (!host_offsets_ptr && !compact.host_offsets.empty()) {
        host_offsets_ptr = compact.host_offsets.data();
    }
    std::vector<int> host_offsets_sanitized;
    if (host_offsets_ptr && num_experts > 0) {
        const long total_tokens = d_gate_up.Sizes[0];
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
    } else if (d_gate_up.DType == ETensorDType::BF16) {
        moe_grouped_gemm_gate_up_backward(d_input_ptr->get<nv_bfloat16>(),
                                          d_gate_up.get<nv_bfloat16>(),
                                          weights.get<nv_bfloat16>(),
                                          expert_offsets_ptr->get<int>(),
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
        moe_grouped_gemm_gate_up_backward(d_input_ptr->get<float>(),
                                          d_gate_up.get<float>(),
                                          weights.get<float>(),
                                          expert_offsets_ptr->get<int>(),
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

    // Apply grouped MoE LoRA backward (gate/up) when enabled.
    if (lora_enabled) {
        auto& lora_block = mLoRAWeights->get_block(layer_idx, mRunState.MainStream);
        if (lora_block.moe.use_grouped) {
            // When LLEP is active, use merged LoRA tensors [num_merged, ...]
            const auto* lora_grouped_ptr = &lora_block.moe.grouped;
            bool llep_lora_active = false;
            {
                auto llep_lora_it = mLLEPStates.find(ep_key);
                if (llep_lora_it != mLLEPStates.end() && llep_lora_it->second.has_merged_lora) {
                    lora_grouped_ptr = &llep_lora_it->second.merged_lora;
                    llep_lora_active = true;
                }
            }
            const auto& grouped = *lora_grouped_ptr;
            const int rank = mLoRAConfig->rank;
            const float scaling = mLoRAConfig->scaling();
            const float dropout = mLoRAConfig->dropout;
            const bool training = mLoRARunState->is_training;
            const long total_tokens_l = d_gate_up.Sizes[0];
            const int total_tokens_i = static_cast<int>(total_tokens_l);

            auto get_dropout_seed = [&](int proj_type) -> unsigned int {
                return mLoRARunState->dropout_base_seed + static_cast<unsigned int>(layer_idx) * 1000000u +
                       static_cast<unsigned int>(proj_type) * 100000u +
                       static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;
            };

            auto view_or_temp = [&](Tensor& buf, long rows, long cols) -> Tensor {
                const long need = rows * cols;
                if (!buf.Data || buf.DType != d_gate_up.DType || buf.nelem() < need) {
                    Tensor tmp = mRunState.temp_alloc(d_gate_up.DType, {rows, cols}, "moe_grouped_gemm_gate_up_temp");
                    mTemps.push_back(tmp);
                    return tmp;
                }
                Tensor view = buf;
                view.DType = d_gate_up.DType;
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
                if (in_t.DType == ETensorDType::BF16) {
                    moe_grouped_gemm(out_t.get<nv_bfloat16>(),
                                     in_t.get<nv_bfloat16>(),
                                     weight_view.get<nv_bfloat16>(),
                                     expert_offsets_ptr->get<int>(),
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
                                     expert_offsets_ptr->get<int>(),
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
                    int weight_rows = (d_weight_view.Rank > 0) ? static_cast<int>(d_weight_view.Sizes[0]) : num_experts;
                    if (mOptions.EPSize > 1 && layer_idx >= 0 && d_weight_view.Rank >= 3 && weight_rows > num_experts) {
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
                    if (grad_output.DType == ETensorDType::BF16) {
                        if (d_weight_view.DType != ETensorDType::BF16) {
                            throw std::runtime_error(
                                "MoE LoRA backward: lora_dtype=fp32 with bf16 activations not supported. "
                                "Set lora_dtype='bf16' in your config.");
                        }
                        moe_grouped_gemm_weight_grad(d_weight_view.get<nv_bfloat16>(),
                                                     grad_output.get<nv_bfloat16>(),
                                                     in.get<nv_bfloat16>(),
                                                     expert_offsets_ptr->get<int>(),
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
                                                     expert_offsets_ptr->get<int>(),
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
            // LLEP: wgrads are computed in MERGED expert space (the offsets are
            // merged) — stage them in [num_merged] temps, then scatter-add native
            // rows into the local grad storage and send foreign rows to their
            // owner ranks (see scatter_exchange below).
            const ep::EPLayerMeta* wg_meta = nullptr;
            if (llep_lora_active) {
                auto wg_it = mEPLayerMeta.find(ep_key);
                if (wg_it != mEPLayerMeta.end() && !wg_it->second.expert_to_gpu.empty()) {
                    wg_meta = &wg_it->second;
                }
            }
            const bool llep_wgrad = llep_lora_active && wg_meta != nullptr && mComm != nullptr;
            if (mLoRAGrads && mComm && (!llep_lora_active || llep_wgrad)) {
                lora_grads = &mLoRAGrads->get_block_full(layer_idx, mRunState.MainStream, *mComm, lora_accum);
            }
            const float grad_beta = lora_accum ? 1.0f : 0.0f;

            auto merged_temp = [&](long rows, long cols) -> Tensor {
                Tensor t = mRunState.temp_alloc(
                    d_gate_up.DType, {static_cast<long>(num_experts), rows, cols}, "llep_merged_wgrad");
                fill_zero(t, mRunState.MainStream);
                mTemps.push_back(t);
                return t;
            };
            Tensor mg_gate_up_A, mg_gate_up_B, mg_gate_A, mg_gate_B, mg_up_A, mg_up_B;

            auto row_view = [&](const Tensor& t, int row) -> Tensor {
                const long r1 = t.Sizes[1], r2 = t.Sizes[2];
                const std::size_t off =
                    static_cast<std::size_t>(row) * r1 * r2 * get_dtype_size(t.DType);
                return Tensor::from_pointer(static_cast<std::byte*>(t.Data) + off, t.Device, t.DType,
                                            std::vector<long>{r1, r2});
            };
            auto row_add = [&](Tensor& dst3, int dst_row, const Tensor& src3, int src_row) {
                Tensor d = row_view(dst3, dst_row);
                Tensor sv = row_view(src3, src_row);
                vector_add_sr(d, d, sv, 1.0f, d.nelem(), 0, mRunState.MainStream);
            };
            // Scatter merged wgrads into local rows; exchange foreign rows to owners.
            // Send/recv order: ascending global expert id, A before B — identical on
            // every rank, so the pairwise WT-comm schedule matches.
            auto scatter_exchange = [&](Tensor& mgA, Tensor& mgB,
                                        modules::LoRAGroupedLayerWeights<Tensor>& local) {
                const auto& meta = *wg_meta;
                const int nl = std::max(meta.num_local, 1);
                const int my_ep = meta.native_start / nl;
                if (!lora_accum) {
                    fill_zero(local.A, mRunState.MainStream);
                    fill_zero(local.B, mRunState.MainStream);
                }
                std::vector<int> g2m(meta.expert_to_gpu.size(), -1);
                for (int m = 0; m < meta.num_merged; ++m) g2m[meta.merged_to_global[m]] = m;
                for (int m = 0; m < meta.num_merged; ++m) {
                    const int ln = meta.merged_to_global[m] - meta.native_start;
                    if (ln >= 0 && ln < meta.num_local) {
                        row_add(local.A, ln, mgA, m);
                        row_add(local.B, ln, mgB, m);
                    }
                }
                const std::size_t a_bytes = static_cast<std::size_t>(mgA.Sizes[1]) * mgA.Sizes[2] *
                                            get_dtype_size(mgA.DType);
                const std::size_t b_bytes = static_cast<std::size_t>(mgB.Sizes[1]) * mgB.Sizes[2] *
                                            get_dtype_size(mgB.DType);
                std::vector<std::tuple<int, Tensor, Tensor>> recvs;  // (expert, tmpA, tmpB)
                mComm->weight_transfer_group_start();
                const int E_total = static_cast<int>(meta.expert_to_gpu.size());
                for (int e = 0; e < E_total; ++e) {
                    const int owner = e / nl;
                    const int helper = meta.expert_to_gpu[e];
                    if (helper == owner) continue;
                    if (helper == my_ep && g2m[e] >= 0) {
                        Tensor sa = row_view(mgA, g2m[e]);
                        Tensor sb = row_view(mgB, g2m[e]);
                        mComm->send_wt(sa.Data, a_bytes, owner, mRunState.MainStream);
                        mComm->send_wt(sb.Data, b_bytes, owner, mRunState.MainStream);
                    } else if (owner == my_ep) {
                        Tensor ta = mRunState.temp_alloc(mgA.DType, {mgA.Sizes[1], mgA.Sizes[2]}, "llep_wgrad_recv");
                        Tensor tb = mRunState.temp_alloc(mgB.DType, {mgB.Sizes[1], mgB.Sizes[2]}, "llep_wgrad_recv");
                        mTemps.push_back(ta);
                        mTemps.push_back(tb);
                        mComm->recv_wt(ta.Data, a_bytes, helper, mRunState.MainStream);
                        mComm->recv_wt(tb.Data, b_bytes, helper, mRunState.MainStream);
                        recvs.emplace_back(e, ta, tb);
                    }
                }
                mComm->weight_transfer_group_end();
                for (auto& [e, ta, tb] : recvs) {
                    const int ln = e - meta.native_start;
                    Tensor da = row_view(local.A, ln);
                    Tensor db = row_view(local.B, ln);
                    vector_add_sr(da, da, ta, 1.0f, da.nelem(), 0, mRunState.MainStream);
                    vector_add_sr(db, db, tb, 1.0f, db.nelem(), 0, mRunState.MainStream);
                }
            };

            if (total_tokens_i > 0 && rank > 0) {
                Tensor lora_intermediate = view_or_temp(mLoRARunState->moe_lora_intermediate1, total_tokens_l, rank);
                const int gate_up_dim = 2 * intermediate_size;
                const bool has_gate_up = grouped.gate_up.has_value() && grouped.gate_up->has_value();

                if (has_gate_up) {
                    const unsigned int seed_gate_up = get_dropout_seed(7);
                    dispatch_grouped_gemm(lora_intermediate,
                                          inp,
                                          grouped.gate_up->A,
                                          rank,
                                          hidden_size,
                                          1.0f,
                                          0.0f,
                                          EMMTranspose::TN);
                    scale_and_dropout(lora_intermediate, seed_gate_up);
                    if (lora_grads && lora_grads->moe.grouped.gate_up.has_value()) {
                        dispatch_weight_grad(llep_wgrad ? (mg_gate_up_B.Data ? mg_gate_up_B : (mg_gate_up_B = merged_temp(gate_up_dim, rank))) : lora_grads->moe.grouped.gate_up->B,
                                             d_gate_up,
                                             lora_intermediate,
                                             gate_up_dim,
                                             rank,
                                             grad_beta);
                    }
                    dispatch_grouped_gemm(lora_intermediate,
                                          d_gate_up,
                                          grouped.gate_up->B,
                                          rank,
                                          gate_up_dim,
                                          1.0f,
                                          0.0f,
                                          EMMTranspose::NN);
                    scale_and_dropout(lora_intermediate, seed_gate_up);
                    if (lora_grads && lora_grads->moe.grouped.gate_up.has_value()) {
                        dispatch_weight_grad(llep_wgrad ? (mg_gate_up_A.Data ? mg_gate_up_A : (mg_gate_up_A = merged_temp(rank, hidden_size))) : lora_grads->moe.grouped.gate_up->A,
                                             lora_intermediate,
                                             inp,
                                             rank,
                                             hidden_size,
                                             grad_beta);
                    }
                    dispatch_grouped_gemm(*d_input_ptr,
                                          lora_intermediate,
                                          grouped.gate_up->A,
                                          hidden_size,
                                          rank,
                                          1.0f,
                                          1.0f,
                                          EMMTranspose::NN);
                } else {
                    Tensor d_up = view_or_temp(mLoRARunState->moe_lora_up, total_tokens_l, intermediate_size);
                    Tensor d_gate = view_or_temp(mLoRARunState->moe_lora_gate, total_tokens_l, intermediate_size);

                    if (op.attrs.gate_up_interleaved) {
                        split_gate_up_interleaved(d_gate_up,
                                                  d_up,
                                                  d_gate,
                                                  total_tokens_i,
                                                  intermediate_size,
                                                  mRunState.MainStream);
                    } else {
                        split_gate_up(d_gate_up, d_up, d_gate, total_tokens_i, intermediate_size, mRunState.MainStream);
                    }

                    // Gate projection
                    if (grouped.gate.has_value() && grouped.gate->has_value()) {
                        const unsigned int seed_gate = get_dropout_seed(5);
                        // intermediate = x @ A^T
                        dispatch_grouped_gemm(lora_intermediate,
                                              inp,
                                              grouped.gate->A,
                                              rank,
                                              hidden_size,
                                              1.0f,
                                              0.0f,
                                              EMMTranspose::TN);
                        scale_and_dropout(lora_intermediate, seed_gate);
                        if (lora_grads && lora_grads->moe.grouped.gate.has_value()) {
                            dispatch_weight_grad(llep_wgrad ? (mg_gate_B.Data ? mg_gate_B : (mg_gate_B = merged_temp(intermediate_size, rank))) : lora_grads->moe.grouped.gate->B,
                                                 d_gate,
                                                 lora_intermediate,
                                                 intermediate_size,
                                                 rank,
                                                 grad_beta);
                        }
                        // intermediate = d_gate @ B
                        dispatch_grouped_gemm(lora_intermediate,
                                              d_gate,
                                              grouped.gate->B,
                                              rank,
                                              intermediate_size,
                                              1.0f,
                                              0.0f,
                                              EMMTranspose::NN);
                        scale_and_dropout(lora_intermediate, seed_gate);
                        if (lora_grads && lora_grads->moe.grouped.gate.has_value()) {
                            dispatch_weight_grad(llep_wgrad ? (mg_gate_A.Data ? mg_gate_A : (mg_gate_A = merged_temp(rank, hidden_size))) : lora_grads->moe.grouped.gate->A,
                                                 lora_intermediate,
                                                 inp,
                                                 rank,
                                                 hidden_size,
                                                 grad_beta);
                        }
                        // dx += intermediate @ A
                        dispatch_grouped_gemm(*d_input_ptr,
                                              lora_intermediate,
                                              grouped.gate->A,
                                              hidden_size,
                                              rank,
                                              1.0f,
                                              1.0f,
                                              EMMTranspose::NN);
                    }

                    // Up projection
                    if (grouped.up.has_value() && grouped.up->has_value()) {
                        const unsigned int seed_up = get_dropout_seed(4);
                        dispatch_grouped_gemm(lora_intermediate,
                                              inp,
                                              grouped.up->A,
                                              rank,
                                              hidden_size,
                                              1.0f,
                                              0.0f,
                                              EMMTranspose::TN);
                        scale_and_dropout(lora_intermediate, seed_up);
                        if (lora_grads && lora_grads->moe.grouped.up.has_value()) {
                            dispatch_weight_grad(llep_wgrad ? (mg_up_B.Data ? mg_up_B : (mg_up_B = merged_temp(intermediate_size, rank))) : lora_grads->moe.grouped.up->B,
                                                 d_up,
                                                 lora_intermediate,
                                                 intermediate_size,
                                                 rank,
                                                 grad_beta);
                        }
                        dispatch_grouped_gemm(lora_intermediate,
                                              d_up,
                                              grouped.up->B,
                                              rank,
                                              intermediate_size,
                                              1.0f,
                                              0.0f,
                                              EMMTranspose::NN);
                        scale_and_dropout(lora_intermediate, seed_up);
                        if (lora_grads && lora_grads->moe.grouped.up.has_value()) {
                            dispatch_weight_grad(llep_wgrad ? (mg_up_A.Data ? mg_up_A : (mg_up_A = merged_temp(rank, hidden_size))) : lora_grads->moe.grouped.up->A,
                                                 lora_intermediate,
                                                 inp,
                                                 rank,
                                                 hidden_size,
                                                 grad_beta);
                        }
                        dispatch_grouped_gemm(*d_input_ptr,
                                              lora_intermediate,
                                              grouped.up->A,
                                              hidden_size,
                                              rank,
                                              1.0f,
                                              1.0f,
                                              EMMTranspose::NN);
                    }
                }

                if (llep_wgrad && lora_grads) {
                    if (mg_gate_up_A.Data && mg_gate_up_B.Data && lora_grads->moe.grouped.gate_up.has_value()) {
                        scatter_exchange(mg_gate_up_A, mg_gate_up_B, *lora_grads->moe.grouped.gate_up);
                    }
                    if (mg_gate_A.Data && mg_gate_B.Data && lora_grads->moe.grouped.gate.has_value()) {
                        scatter_exchange(mg_gate_A, mg_gate_B, *lora_grads->moe.grouped.gate);
                    }
                    if (mg_up_A.Data && mg_up_B.Data && lora_grads->moe.grouped.up.has_value()) {
                        scatter_exchange(mg_up_A, mg_up_B, *lora_grads->moe.grouped.up);
                    }
                }
            } else if (llep_wgrad && lora_grads && rank > 0) {
                // Zero tokens received on this rank (routine for all-padding
                // chunks under chunked-sequence training): still walk the
                // plan-driven wgrad exchange — every rank derives the same
                // pairwise WT-comm schedule from the shared plan, and peers
                // block on this rank's sends/recvs regardless of its local
                // row count. Merged contributions are zero; row-0 of the
                // dummies is never read (this rank owns no merged experts).
                // Pair selection mirrors the LoRA config via the LOCAL grads
                // (uniform across ranks), not the merged view (null here).
                auto empty_merged = [&](long rows, long cols) -> Tensor {
                    Tensor t = mRunState.temp_alloc(d_gate_up.DType, {1L, rows, cols}, "llep_wgrad_empty");
                    mTemps.push_back(t);
                    return t;
                };
                const int gate_up_dim = 2 * intermediate_size;
                if (lora_grads->moe.grouped.gate_up.has_value()) {
                    Tensor ea = empty_merged(rank, hidden_size);
                    Tensor eb = empty_merged(gate_up_dim, rank);
                    scatter_exchange(ea, eb, *lora_grads->moe.grouped.gate_up);
                }
                if (lora_grads->moe.grouped.gate.has_value()) {
                    Tensor ea = empty_merged(rank, hidden_size);
                    Tensor eb = empty_merged(intermediate_size, rank);
                    scatter_exchange(ea, eb, *lora_grads->moe.grouped.gate);
                }
                if (lora_grads->moe.grouped.up.has_value()) {
                    Tensor ea = empty_merged(rank, hidden_size);
                    Tensor eb = empty_merged(intermediate_size, rank);
                    scatter_exchange(ea, eb, *lora_grads->moe.grouped.up);
                }
            }
        }
    }

    store_tensor(op.outputs[0], *d_input_ptr);

    // Weight gradient computation would go here if needed (for fine-tuning experts)
}

namespace {

// -----------------------------------------------------------------------------
// MoE Grouped GEMM Gate+Up backward rule
// Forward: out = moe_grouped_gemm_gate_up(inp, weights, scatter_indices)
// Backward: d_inp = moe_grouped_gemm_gate_up_backward(d_out, inp, weights, scatter_indices)
// Note: weights gradient is computed but not propagated (frozen expert weights)
// -----------------------------------------------------------------------------
std::vector<Operation> moe_grouped_gemm_gate_up_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string inp = fwd.inputs[0];
        std::string weights = fwd.inputs[1];
        std::string scatter_indices = fwd.inputs[2];

        std::string inp_ref = ctx.is_param(inp) ? inp : saved_ref(inp);
        std::string weights_ref = ctx.is_param(weights) ? weights : saved_ref(weights);
        std::string scatter_ref = saved_ref(scatter_indices);
        AttrMap attrs = copy_attrs(fwd.attrs, {"gate_up_interleaved"});

        ops.push_back(make_operation("moe_grouped_gemm_gate_up_backward_" + std::to_string(ctx.op_counter++),
                                     "moe_grouped_gemm_gate_up_backward",
                                     "moe_grouped_gemm_gate_up_backward",
                                     {ctx.d_output, inp_ref, weights_ref, scatter_ref},
                                     {ctx.d_inputs[0]},
                                     attrs));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("moe_grouped_gemm_gate_up", ::dsl::moe_grouped_gemm_gate_up_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// MoE Grouped GEMM Gate+Up
// Input[0]: x [total_tokens, hidden_size]
// Input[1]: weights [num_experts, 2*intermediate, hidden_size]
// Input[2]: scatter_indices
// Output: [total_tokens, 2*intermediate]
// ------------------------------------------------------------------------
const int _moe_grouped_gemm_gate_up_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "moe_grouped_gemm_gate_up";
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

const int _moe_grouped_gemm_gate_up_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "moe_grouped_gemm_gate_up_backward";
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
