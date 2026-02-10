#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "runtime/dsl/graph_executor_helpers.h"
#include "runtime/lora/lora_model_utils.h"
#include "runtime/lora/lora_run_state.h"

namespace dsl {

void CompiledExecutor::dispatch_matmul(const CompiledOp& op, const modules::ForwardHook* hook) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    const std::string& weight_name = op.inputs[1].name;

    std::optional<Tensor> bias;
    if (op.type == CompiledOpType::MatmulBias && op.inputs.size() > 2) {
        bias = resolve_tensor(op.inputs[2]);
    }

    int M = 0, N = 0, K = 0;
    matmul_dims(a, b, op.attrs.transpose, M, N, K);

    // Router matmul: match HF by computing logits in FP32 using FP32 inputs/weights.
    const bool is_router = (weight_name.find("router_weight") != std::string::npos);
    if (is_router && out.DType == ETensorDType::FP32 &&
        (a.DType != ETensorDType::FP32 || b.DType != ETensorDType::FP32)) {
        auto shape_vec = [](const Tensor& t) {
            return std::vector<long>(t.Sizes.begin(), t.Sizes.begin() + t.Rank);
        };
        Tensor a_f = a;
        if (a.DType != ETensorDType::FP32) {
            a_f = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(a));
            mTemps.push_back(a_f);
            if (a.DType == ETensorDType::BF16) {
                convert_dtype(a_f.get<float>(), a.get<nv_bfloat16>(), a.nelem(), mRunState.MainStream);
            } else {
                throw std::runtime_error("router matmul: unsupported input dtype");
            }
        }

        Tensor b_f = b;
        if (b.DType != ETensorDType::FP32) {
            b_f = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(b));
            mTemps.push_back(b_f);
            if (b.DType == ETensorDType::BF16) {
                convert_dtype(b_f.get<float>(), b.get<nv_bfloat16>(), b.nelem(), mRunState.MainStream);
            } else {
                throw std::runtime_error("router matmul: unsupported weight dtype");
            }
        }

        std::optional<Tensor> bias_f;
        if (bias.has_value()) {
            if (bias->DType == ETensorDType::FP32) {
                bias_f = bias;
            } else {
                Tensor tmp = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(*bias));
                mTemps.push_back(tmp);
                if (bias->DType == ETensorDType::BF16) {
                    convert_dtype(tmp.get<float>(), bias->get<nv_bfloat16>(), bias->nelem(), mRunState.MainStream);
                } else {
                    throw std::runtime_error("router matmul: unsupported bias dtype");
                }
                bias_f = tmp;
            }
        }

        EMMTranspose mode_col = swap_transpose(op.attrs.transpose);
        matmul(out, b_f, a_f, bias_f, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, false, mRunState.MainStream);
        return;
    }

    bool used_recipe = false;
    modules::MatmulContext ctx{};
    modules::MatmulContext* ctx_ptr = nullptr;
    if (mRecipe && op.attrs.transpose == EMMTranspose::NT && a.Sizes[0] == mB * mT) {
        if (op.attrs.allow_quant && op.attrs.matmul_op.has_value()) {
            ctx.out = &out;
            ctx.inp = &a;
            ctx.weight = &b;
            ctx.bias = bias ? &*bias : nullptr;
            ctx.B = static_cast<int>(mB);
            ctx.T = static_cast<int>(mT);
            ctx.C_in = K;
            ctx.C_out = N;
            ctx.run_state = &mRunState;
            ctx.stream = mRunState.MainStream;
            ctx.layer_idx = op.attrs.layer_idx;
            ctx.op = *op.attrs.matmul_op;
            ctx.allow_fp8 = mRecipe->uses_fp8_forward();
            ctx.allow_fp4 = mRecipe->uses_fp4_forward();

            // Wire FP8/FP4 buffers + static weight caches (GraphExecutor primes caches before CUDA graph capture).
            if (ctx.allow_fp8) {
                ctx.inp_quant = fp8_forward_buffer(mRunState, *op.attrs.matmul_op);
                ctx.delayed_quantizer_idx = fp8_quantizer_index(mRunState, *op.attrs.matmul_op, op.attrs.layer_idx);
                if (b.DType == ETensorDType::FP8_E4M3) {
                    ctx.cached_weight = &b;
                } else if (mFP8Cache) {
                    auto it = mFP8Cache->find(weight_name);
                    if (it != mFP8Cache->end() && it->second.initialized && it->second.weight.Data) {
                        ctx.cached_weight = &it->second.weight;
                    }
                }
            }
            if (ctx.allow_fp4 && mFP4Cache) {
                auto it = mFP4Cache->find(weight_name);
                if (it != mFP4Cache->end() && it->second.initialized &&
                    it->second.data.Data && it->second.scales.Data && it->second.amax.Data) {
                    ctx.cached_fp4_data = &it->second.data;
                    ctx.cached_fp4_scales = &it->second.scales;
                    ctx.cached_fp4_amax = it->second.amax.get<float>();
                }
            }

            mRecipe->forward_matmul(ctx);
            used_recipe = true;
            ctx_ptr = &ctx;
        }
    }

    if (!used_recipe) {
        EMMTranspose mode_col = swap_transpose(op.attrs.transpose);
        matmul(out, b, a, bias, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, false, mRunState.MainStream);
    }

    // Apply shared-expert LoRA contributions (Nemotron/DeepSeek) for shared expert matmuls.
    if (mLoRAConfig && mLoRAWeights && mLoRARunState && mLoRAConfig->enabled()) {
        int shared_layer = -1;
        std::string field;
        if (parse_block_param(weight_name, shared_layer, field)) {
            const bool is_shared_up = (field == "shared_expert_up");
            const bool is_shared_down = (field == "shared_expert_down");
            if ((is_shared_up || is_shared_down) && shared_layer >= 0) {
                auto& lora_block = mLoRAWeights->get_block(shared_layer, mRunState.MainStream);
                if (lora_block.moe.shared.has_value()) {
                    auto& shared = *lora_block.moe.shared;
                    const auto& lora_layer = is_shared_up ? shared.up : shared.down;
                    if (lora_layer.has_value() && lora_layer->has_value()) {
                        Tensor a_flat = a;
                        Tensor out_flat = out;
                        if (a_flat.Rank > 2 && a_flat.Sizes[0] == mB && a_flat.Sizes[1] == mT) {
                            a_flat = view_tensor(a_flat, {mB * mT, a_flat.Sizes[a_flat.Rank - 1]});
                        }
                        if (out_flat.Rank > 2 && out_flat.Sizes[0] == mB && out_flat.Sizes[1] == mT) {
                            out_flat = view_tensor(out_flat, {mB * mT, out_flat.Sizes[out_flat.Rank - 1]});
                        }

                        const int BT = static_cast<int>(a_flat.Sizes[0]);
                        const int in_features = static_cast<int>(a_flat.Sizes[1]);
                        const int out_features = static_cast<int>(out_flat.Sizes[1]);
                        const int rank = mLoRAConfig->rank;
                        const float scaling = mLoRAConfig->scaling();
                        const float dropout = mLoRAConfig->dropout;
                        const bool training = mLoRARunState->is_training;
                        const int proj_type = is_shared_up ? 7 : 8;
                        const unsigned int dropout_seed = mLoRARunState->dropout_base_seed
                            + static_cast<unsigned int>(shared_layer) * 1000000u
                            + static_cast<unsigned int>(proj_type) * 100000u
                            + static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;

                        modules::detail::apply_lora_contribution(
                            out_flat, 0, a_flat, lora_layer.value(),
                            mLoRARunState->intermediate, mLoRARunState->slice,
                            scaling, dropout, dropout_seed, training,
                            BT, in_features, out_features, rank,
                            mRunState.CublasLtHandle, mRunState.CuBlasWorkspace, mRunState.MainStream);
                    }
                }
            }
        }
    }

    if (mForwardPlan && op.attrs.matmul_op.has_value() && op.attrs.layer_idx >= 0 &&
        static_cast<std::size_t>(op.attrs.layer_idx) < mForwardPlan->size() &&
        *op.attrs.matmul_op != modules::MatmulOp::LMHead) {
        MatmulForwardPlan plan{};
        plan.valid = true;
        plan.use_recipe = used_recipe;
        plan.has_bias = bias.has_value();
        if (used_recipe && ctx_ptr) {
            plan.allow_fp8 = ctx_ptr->allow_fp8;
            plan.allow_fp4 = ctx_ptr->allow_fp4;
            plan.delayed_quantizer_idx = ctx_ptr->delayed_quantizer_idx;
            plan.use_fp8_cache = (ctx_ptr->cached_weight && ctx_ptr->cached_weight->Data);
            plan.use_fp4_cache = (ctx_ptr->cached_fp4_data && ctx_ptr->cached_fp4_scales);
        }
        auto& layer_plan = (*mForwardPlan)[static_cast<std::size_t>(op.attrs.layer_idx)];
        switch (*op.attrs.matmul_op) {
            case modules::MatmulOp::QKV:
                layer_plan.qkv = plan;
                break;
            case modules::MatmulOp::AttnOut:
                layer_plan.out_proj = plan;
                break;
            case modules::MatmulOp::MLPUp:
                layer_plan.mlp_up = plan;
                break;
            case modules::MatmulOp::MLPDown:
                layer_plan.mlp_down = plan;
                break;
            default:
                break;
        }
    }

    // Hook invocation
    if (hook && *hook && op.attrs.forward_hook_point.has_value()) {
        (*hook)(op.attrs.layer_idx, mRunState.MainStream, *op.attrs.forward_hook_point, mHookContext);
    }

}

void CompiledExecutor::dispatch_matmul_backward(const CompiledOp& op, const modules::BackwardHook* hook) {
    // inputs: d_out, A, B (weight)
    // outputs: dA, dB
    const std::string& weight_name = (op.inputs.size() > 2) ? op.inputs[2].name : "";
    const bool is_lm_head = (weight_name == "lm_head" || weight_name == "lm_head_weight");
    const bool skip_lm_head = is_lm_head && mOptions.LMHeadChunks > 1;

    EMMTranspose mode = op.attrs.transpose;
    const int layer_idx = op.attrs.layer_idx;
    const bool allow_quant = op.attrs.allow_quant;

    // Check if weight gradient should be skipped BEFORE allocating (frozen weights in LoRA mode)
    bool skip_weight_grad = true;
    const std::string& dB_name = op.outputs.size() > 1 ? op.outputs[1].name : "";
    if (!dB_name.empty()) {
        std::string weight_name;
        if (auto base = base_param_from_grad(dB_name)) {
            weight_name = *base;
        } else {
            weight_name = dB_name;
            if (weight_name.rfind("d_", 0) == 0) {
                weight_name = weight_name.substr(2);
            }
        }
        bool accum = false;
        Tensor* grad = mGrads.get_param_grad(weight_name, accum);
        skip_weight_grad = (grad == nullptr || !grad->Data);
    }

    if (skip_lm_head) {
        if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
            (void)ensure_output_tensor(op.outputs[0]);
        }
        if (!skip_weight_grad && op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
            (void)ensure_output_tensor(op.outputs[1]);
        }
        return;
    }

    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& a = resolve_tensor(op.inputs[1]);
    Tensor& b = resolve_tensor(op.inputs[2]);

    const bool is_qkv_op = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::QKV);

    // Now allocate output tensors - skip dB if weights are frozen.
    // For backward ops, compiled shapes of saved-tensor-derived outputs may be empty
    // (backward compiler can't track saved tensor shapes). Derive from runtime inputs.
    auto ensure_backward_output = [&](const TensorRef& ref, const Tensor& shape_source) -> Tensor& {
        if (ref.shape.empty() && shape_source.Rank > 0) {
            std::vector<long> shape(shape_source.Sizes.begin(), shape_source.Sizes.begin() + shape_source.Rank);
            Tensor t = mRunState.temp_alloc(shape_source.DType, shape);
            fill_zero(t, mRunState.MainStream);
            mTemps.push_back(t);
            auto [it, _] = mTensorMap.insert_or_assign(ref.name, t);
            return it->second;
        }
        return ensure_output_tensor(ref);
    };

    Tensor* dA_ptr = nullptr;
    Tensor* dB_ptr = nullptr;

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        dA_ptr = &ensure_backward_output(op.outputs[0], a);
    }
    if (!skip_weight_grad && op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        dB_ptr = &ensure_backward_output(op.outputs[1], b);
    }

    if (!dA_ptr && !dB_ptr) {
        return;
    }

    bool do_accumulate = mAccumulateTensors.count(dB_name) > 0;
    if (!do_accumulate && !dB_name.empty()) {
        if (auto base = base_param_from_grad(dB_name)) {
            do_accumulate = mAccumulateTensors.count("d_" + *base) > 0;
        }
    }

    bool used_recipe = false;
    bool used_fp8 = false;
    bool has_dout_quant = false;

    const bool disable_qkv_recipe_bwd =
        is_qkv_op && skip_weight_grad && (mConfig.NumExperts > 0);
    if (mRecipe && mode == EMMTranspose::NT && a.Sizes[0] == mB * mT && allow_quant && !disable_qkv_recipe_bwd) {
        Tensor dA_tmp{};
        Tensor dB_tmp{};
        Tensor* dA_use = dA_ptr;
        Tensor* dB_use = dB_ptr;

        if (!dA_use) {
            dA_tmp = mRunState.temp_alloc(a.DType, {a.Sizes[0], a.Sizes[1]});
            mTemps.push_back(dA_tmp);
            dA_use = &dA_tmp;
        }
        if (!dB_use) {
            dB_tmp = mRunState.temp_alloc(b.DType, {b.Sizes[0], b.Sizes[1]});
            mTemps.push_back(dB_tmp);
            dB_use = &dB_tmp;
        }

        modules::MatmulContext ctx;
        ctx.dinp = dA_use;
        ctx.dweight = dB_use;
        ctx.dout = &d_out;
        ctx.inp = &a;
        ctx.weight = &b;
        ctx.B = static_cast<int>(mB);
        ctx.T = static_cast<int>(mT);
        ctx.C_in = static_cast<int>(a.Sizes[1]);
        ctx.C_out = static_cast<int>(b.Sizes[0]);
        ctx.run_state = &mRunState;
        ctx.stream = mRunState.MainStream;
        ctx.layer_idx = layer_idx;
        ctx.op = op.attrs.matmul_op.value_or(modules::MatmulOp::LMHead);
        ctx.accumulate = do_accumulate;
        ctx.skip_weight_grad = skip_weight_grad || !dB_ptr;
        ctx.allow_fp8 = allow_quant && mRecipe->uses_fp8_hybrid_backward();
        ctx.allow_fp4 = allow_quant && mRecipe->uses_fp4_forward();
        ctx.seed = mRngSeedFn ? mRngSeedFn() : 0u;

        if (ctx.allow_fp8 && op.attrs.matmul_op.has_value()) {
            ctx.dout_quant = fp8_grad_buffer(mRunState, *op.attrs.matmul_op);
            if (!ctx.dout_quant || !ctx.dout_quant->Data) {
                ctx.allow_fp8 = false;
            }
        }
        if (ctx.allow_fp8 && mFP8CacheT) {
            auto it = mFP8CacheT->find(weight_name);
            if (it != mFP8CacheT->end() && it->second.initialized && it->second.weight.Data) {
                // For FP8 backward, cache stores W^T in FP8 (K, N) to skip per-op quantize+transpose.
                ctx.cached_weight = &it->second.weight;
            }
        }
        if (ctx.allow_fp4 && mRecipe) {
            // NVFP4QuartetRecipe uses the forward-layout FP4 cache and performs an explicit
            // dequant->transpose->Hadamard->requant pipeline for per-step re-randomization.
            // Standard NVFP4 uses the transposed cache (W^T) directly for dgrad.
            const bool is_quartet = (mRecipe->name() == std::string_view{"nvfp4-quartet"});
            auto* cache = is_quartet ? mFP4Cache : mFP4CacheT;
            if (cache) {
                auto it = cache->find(weight_name);
                if (it != cache->end() && it->second.initialized &&
                    it->second.data.Data && it->second.scales.Data && it->second.amax.Data) {
                    ctx.cached_fp4_data = &it->second.data;
                    ctx.cached_fp4_scales = &it->second.scales;
                    ctx.cached_fp4_amax = it->second.amax.get<float>();
                }
            }
        }
        used_fp8 = ctx.allow_fp8;
        has_dout_quant = (ctx.dout_quant && ctx.dout_quant->Data);

        mRecipe->backward_matmul(ctx);
        used_recipe = true;
    }

    if (!used_recipe) {
        Tensor d_out_mat = d_out;
        Tensor a_mat = a;
        auto maybe_flatten_bt = [&](Tensor& t) {
            if (t.Rank > 2 && t.Sizes[0] == mB && t.Sizes[1] == mT) {
                t = view_tensor(t, {mB * mT, t.Sizes[t.Rank - 1]});
            }
        };
        // Ensure matmul inputs are rank-2 by flattening [B, T, K] -> [B*T, K].
        // This handles cases where *_flat tensors were not materialized as views.
        if (disable_qkv_recipe_bwd && is_qkv_op) {
            maybe_flatten_bt(d_out_mat);
            maybe_flatten_bt(a_mat);
        } else {
            maybe_flatten_bt(d_out_mat);
            maybe_flatten_bt(a_mat);
        }

        // Fallback: explicit matmuls for dA and dB
        EMMTranspose mode_dA = EMMTranspose::NN;
        EMMTranspose mode_dB = EMMTranspose::NN;
        switch (mode) {
            case EMMTranspose::NN:
                mode_dA = EMMTranspose::NT;
                mode_dB = EMMTranspose::TN;
                break;
            case EMMTranspose::NT:
                mode_dA = EMMTranspose::NN;
                mode_dB = EMMTranspose::TN;
                break;
            case EMMTranspose::TN:
                mode_dA = EMMTranspose::NT;
                mode_dB = EMMTranspose::NN;
                break;
            case EMMTranspose::TT:
                mode_dA = EMMTranspose::TT;
                mode_dB = EMMTranspose::TT;
                break;
        }

        if (dA_ptr) {
            int M = 0, N = 0, K = 0;
            matmul_dims(d_out_mat, b, mode_dA, M, N, K);
            EMMTranspose mode_col = swap_transpose(mode_dA);
            matmul(*dA_ptr, b, d_out_mat, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   N, M, K, mode_col, false, mRunState.MainStream);
        }
        if (dB_ptr && !skip_weight_grad) {
            const Tensor* lhs = nullptr;
            const Tensor* rhs = nullptr;
            EMMTranspose mode_rm = EMMTranspose::NN;
            switch (mode) {
                case EMMTranspose::NN:
                    // dB = A^T * d_out
                    lhs = &a_mat;
                    rhs = &d_out_mat;
                    mode_rm = EMMTranspose::TN;
                    break;
                case EMMTranspose::NT:
                    // dB = d_out^T * A
                    lhs = &d_out_mat;
                    rhs = &a_mat;
                    mode_rm = EMMTranspose::TN;
                    break;
                case EMMTranspose::TN:
                    // dB = A * d_out
                    lhs = &a_mat;
                    rhs = &d_out_mat;
                    mode_rm = EMMTranspose::NN;
                    break;
                case EMMTranspose::TT:
                    // dB = d_out^T * A^T
                    lhs = &d_out_mat;
                    rhs = &a_mat;
                    mode_rm = EMMTranspose::TT;
                    break;
            }

            int M = 0, N = 0, K = 0;
            matmul_dims(*lhs, *rhs, mode_rm, M, N, K);
            EMMTranspose mode_col = swap_transpose(mode_rm);
            matmul(*dB_ptr, *rhs, *lhs, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   N, M, K, mode_col, do_accumulate, mRunState.MainStream);
        }
    }

    // Record qkv dA pointer for LN1 wiring verification.
    if (is_qkv_op && dA_ptr && layer_idx >= 0) {
        if (g_qkv_dA_ptr_by_layer.empty() && mConfig.NumLayers > 0) {
            g_qkv_dA_ptr_by_layer.assign(static_cast<std::size_t>(mConfig.NumLayers), nullptr);
            g_qkv_dA_micro_by_layer.assign(static_cast<std::size_t>(mConfig.NumLayers), -1);
        }
        if (layer_idx < static_cast<int>(g_qkv_dA_ptr_by_layer.size())) {
            g_qkv_dA_ptr_by_layer[static_cast<std::size_t>(layer_idx)] =
                reinterpret_cast<std::byte*>(dA_ptr->Data);
            g_qkv_dA_micro_by_layer[static_cast<std::size_t>(layer_idx)] = mMicroStep;
        }
    }

    // Shared-expert LoRA backward (Nemotron/DeepSeek).
    if (mLoRAConfig && mLoRAWeights && mLoRAGrads && mLoRARunState &&
        mLoRAConfig->enabled() && mComm) {
        int shared_layer = -1;
        std::string field;
        if (parse_block_param(weight_name, shared_layer, field)) {
            const bool is_shared_up = (field == "shared_expert_up");
            const bool is_shared_down = (field == "shared_expert_down");
            if ((is_shared_up || is_shared_down) && shared_layer >= 0) {
                auto& lora_block = mLoRAWeights->get_block(shared_layer, mRunState.MainStream);
                if (lora_block.moe.shared.has_value()) {
                    auto& shared = *lora_block.moe.shared;
                    const auto& lora_layer = is_shared_up ? shared.up : shared.down;
                    if (lora_layer.has_value() && lora_layer->has_value()) {
                        bool lora_accum = false;
                        auto& lora_grads = mLoRAGrads->get_block_full(shared_layer, mRunState.MainStream, *mComm, lora_accum);
                        lora_accum = lora_accum || do_accumulate;

                        if (lora_grads.moe.shared.has_value()) {
                            auto* grad_layer = is_shared_up ? &lora_grads.moe.shared->up : &lora_grads.moe.shared->down;
                            if (grad_layer && grad_layer->has_value() && grad_layer->value().has_value()) {
                            Tensor a_flat = a;
                            Tensor d_out_flat = d_out;
                            if (a_flat.Rank > 2 && a_flat.Sizes[0] == mB && a_flat.Sizes[1] == mT) {
                                a_flat = view_tensor(a_flat, {mB * mT, a_flat.Sizes[a_flat.Rank - 1]});
                            }
                            if (d_out_flat.Rank > 2 && d_out_flat.Sizes[0] == mB && d_out_flat.Sizes[1] == mT) {
                                d_out_flat = view_tensor(d_out_flat, {mB * mT, d_out_flat.Sizes[d_out_flat.Rank - 1]});
                            }

                            Tensor dA_tmp{};
                            Tensor* dA_use = dA_ptr;
                            if (!dA_use) {
                                dA_tmp = mRunState.temp_alloc(a_flat.DType, {a_flat.Sizes[0], a_flat.Sizes[1]});
                                fill_zero(dA_tmp, mRunState.MainStream);
                                mTemps.push_back(dA_tmp);
                                dA_use = &dA_tmp;
                            }

                            const int BT = static_cast<int>(a_flat.Sizes[0]);
                            const int in_features = static_cast<int>(a_flat.Sizes[1]);
                            const int out_features = static_cast<int>(d_out_flat.Sizes[1]);
                            const int rank = mLoRAConfig->rank;
                            const float scaling = mLoRAConfig->scaling();
                            const float dropout = mLoRAConfig->dropout;
                            const bool training = mLoRARunState->is_training;
                            const int proj_type = is_shared_up ? 7 : 8;
                            const unsigned int dropout_seed = mLoRARunState->dropout_base_seed
                                + static_cast<unsigned int>(shared_layer) * 1000000u
                                + static_cast<unsigned int>(proj_type) * 100000u
                                + static_cast<unsigned int>(mLoRARunState->micro_step) * 10000u;

                            modules::detail::backward_lora_layer(
                                grad_layer->value().A, grad_layer->value().B,
                                *dA_use,
                                d_out_flat, 0,
                                a_flat,
                                lora_layer->A, lora_layer->B,
                                scaling,
                                dropout, dropout_seed, training,
                                mLoRARunState->intermediate, mLoRARunState->slice,
                                BT, in_features, out_features, rank, lora_accum,
                                mRunState.CublasLtHandle, mRunState.CuBlasWorkspace, mRunState.MainStream);
                            }
                        }
                    }
                }
            }
        }
    }


    // Hook invocation for LoRA backward
    // Skip dense MLP hooks for MoE models - MoE has different backward path (grouped GEMM)
    const bool is_moe = mConfig.NumExperts > 0;
    const bool is_mlp_hook = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::MLPUp ||
         *op.attrs.matmul_op == modules::MatmulOp::MLPDown);
    if (hook && *hook && op.attrs.backward_hook_point.has_value() && !(is_moe && is_mlp_hook)) {
        // Temporarily map grads to current backward tensors for LoRA hooks, then restore.
        struct GradPtrs {
            std::byte* d_swiglu{nullptr};
            std::byte* d_ln2{nullptr};
            std::byte* d_att{nullptr};
            std::byte* d_ln1{nullptr};
            std::byte* d_res_ffn{nullptr};
            std::byte* d_mlp_up{nullptr};
            std::byte* d_res_att{nullptr};
            std::byte* d_att_out{nullptr};
            std::byte* d_qkv{nullptr};
        } prev{};

        if (op.attrs.matmul_op.has_value() && layer_idx >= 0) {
            auto& grads = mRunState.simplified_grads(layer_idx);
            prev.d_swiglu = reinterpret_cast<std::byte*>(grads.d_swiglu.Data);
            prev.d_ln2 = reinterpret_cast<std::byte*>(grads.d_ln2.Data);
            prev.d_att = reinterpret_cast<std::byte*>(grads.d_att.Data);
            prev.d_ln1 = reinterpret_cast<std::byte*>(grads.d_ln1.Data);
            prev.d_res_ffn = reinterpret_cast<std::byte*>(grads.d_res_ffn.Data);
            prev.d_mlp_up = reinterpret_cast<std::byte*>(grads.d_mlp_up.Data);
            prev.d_res_att = reinterpret_cast<std::byte*>(grads.d_res_att.Data);
            prev.d_att_out = reinterpret_cast<std::byte*>(grads.d_att_out.Data);
            prev.d_qkv = reinterpret_cast<std::byte*>(grads.d_qkv.Data);

            if (dA_ptr) {
                switch (*op.attrs.matmul_op) {
                    case modules::MatmulOp::MLPDown:
                        grads.d_swiglu.Data = dA_ptr->Data;
                        break;
                    case modules::MatmulOp::MLPUp:
                        grads.d_ln2.Data = dA_ptr->Data;
                        break;
                    case modules::MatmulOp::AttnOut:
                        grads.d_att.Data = dA_ptr->Data;
                        break;
                    case modules::MatmulOp::QKV:
                        grads.d_ln1.Data = dA_ptr->Data;
                        break;
                    default:
                        break;
                }
            }

            switch (*op.attrs.matmul_op) {
                case modules::MatmulOp::MLPDown:
                    grads.d_res_ffn.Data = d_out.Data;
                    break;
                case modules::MatmulOp::MLPUp:
                    grads.d_mlp_up.Data = d_out.Data;
                    break;
                case modules::MatmulOp::AttnOut:
                    grads.d_att_out.Data = d_out.Data;
                    break;
                case modules::MatmulOp::QKV:
                    grads.d_qkv.Data = d_out.Data;
                    break;
                default:
                    break;
            }
        }

        // Ensure activations needed by LoRA hooks are available.
        if (layer_idx >= 0 && op.attrs.matmul_op.has_value()) {
            auto& acts = mRunState.simplified_acts(layer_idx);
            if (*op.attrs.matmul_op == modules::MatmulOp::MLPDown) {
                // LoRA backward hook needs acts.swiglu (forward activation).
                // With recompute enabled, swiglu may have been stack-allocated and freed.
                if (!acts.swiglu.Data && acts.mlp_up.Data) {
                    mRunState.temp_acquire(acts.swiglu);
                    const int Bv = static_cast<int>(mB);
                    const int Tv = static_cast<int>(mT);
                    const int D = static_cast<int>(mConfig.IntermediateSize);
                    switch (mConfig.activation_type) {
                        case modules::ActivationType::SwiGLU:
                        case modules::ActivationType::GeGLU:
                            swiglu_forward(acts.swiglu, acts.mlp_up, nullptr, Bv, Tv, D, mRunState.MainStream);
                            break;
                        case modules::ActivationType::ReLU2: {
                            const long N = static_cast<long>(Bv) * static_cast<long>(Tv) * static_cast<long>(D);
                            relu2_forward(acts.swiglu, acts.mlp_up, N, mRunState.MainStream);
                        } break;
                        case modules::ActivationType::SiLU: {
                            const long N = static_cast<long>(Bv) * static_cast<long>(Tv) * static_cast<long>(D);
                            silu_forward(acts.swiglu, acts.mlp_up, N, mRunState.MainStream);
                        } break;
                        default:
                            throw std::runtime_error("matmul: unsupported activation type for LoRA swiglu recompute");
                    }
                }
            }
        }
        (*hook)(layer_idx, do_accumulate, mRunState.MainStream, *op.attrs.backward_hook_point, mHookContext);

        if (op.attrs.matmul_op.has_value() && layer_idx >= 0) {
            auto& grads = mRunState.simplified_grads(layer_idx);
            grads.d_swiglu.Data = prev.d_swiglu;
            grads.d_ln2.Data = prev.d_ln2;
            grads.d_att.Data = prev.d_att;
            grads.d_ln1.Data = prev.d_ln1;
            grads.d_res_ffn.Data = prev.d_res_ffn;
            grads.d_mlp_up.Data = prev.d_mlp_up;
            grads.d_res_att.Data = prev.d_res_att;
            grads.d_att_out.Data = prev.d_att_out;
            grads.d_qkv.Data = prev.d_qkv;
        }
    }
}

}  // namespace dsl
