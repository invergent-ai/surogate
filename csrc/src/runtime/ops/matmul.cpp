#include "runtime/executor/compiled_ops.h"
#include "runtime/dsl/tensor_slot_dispatch.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/dsl/tensor_role.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/core/forward_hooks.h"
#include "runtime/lora/lora_model_utils.h"
#include "runtime/lora/lora_run_state.h"
#include "runtime/lora/lora_slice_dispatch.h"

namespace dsl {

namespace {

bool is_shared_expert_weight_name(const std::string& weight_name) {
    return tensor_role_is_shared_expert_name(weight_name);
}

bool is_router_weight_name(const std::string& weight_name) {
    return tensor_role_is_router_name(weight_name);
}

void attach_input_role(modules::MatmulContext& ctx, const CompiledGraph* graph, const TensorRef& ref) {
    if (!graph) {
        return;
    }
    if (const TensorRole* role = graph->role_for_tensor_id(ref.tensor_id)) {
        ctx.input_role = *role;
        ctx.has_input_role = true;
    }
}

struct LoRAForwardApplyContext {
    const std::vector<LoRASlice>* slices = nullptr;
    int layer_idx = -1;
    Tensor input_2d;
    Tensor output_2d;
    int BT = 0;
    modules::ModularLoRAWeightsManager* weights = nullptr;
    const modules::ModularLoRAConfig* config = nullptr;
    modules::LoRARunState* run_state = nullptr;
    cublasLtHandle_t handle = nullptr;
    Tensor* workspace = nullptr;
    cudaStream_t stream = nullptr;
};

void apply_lora_forward_payload(void* opaque) {
    auto* ctx = static_cast<LoRAForwardApplyContext*>(opaque);
    if (!ctx || !ctx->slices || !ctx->workspace) return;
    modules::detail::apply_lora_slices_forward(*ctx->slices,
                                               ctx->layer_idx,
                                               ctx->input_2d,
                                               ctx->output_2d,
                                               ctx->BT,
                                               ctx->weights,
                                               ctx->config,
                                               ctx->run_state,
                                               ctx->handle,
                                               *ctx->workspace,
                                               ctx->stream);
}

}  // namespace

// flatten_bt now lives in runtime/executor/graph_executor_utils.h so it
// can be reused by LoRA slice dispatch (lora_slice_dispatch.h).

void CompiledExecutor::dispatch_matmul(const CompiledOp& op, const modules::ForwardHook* hook) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    const std::string& weight_name = op.inputs[1].name;
    const bool is_gate_projection = is_mlp_gate_weight(weight_name);

    const bool is_shared_weight = is_shared_expert_weight_name(weight_name);

    std::optional<Tensor> bias;
    if (op.type == CompiledOpType::MatmulBias && op.inputs.size() > 2) {
        bias = resolve_tensor(op.inputs[2]);
    }

    int M = 0, N = 0, K = 0;
    matmul_dims(a, b, op.attrs.transpose, M, N, K);

    // Router matmul: match HF by computing logits in FP32 using FP32 inputs/weights.
    // Skip the string search for dense (non-MoE) models — they have no router.
    const bool is_router = (mConfig.NumExperts > 0) && is_router_weight_name(weight_name);
    if (is_router &&
        (a.DType != ETensorDType::FP32 || b.DType != ETensorDType::FP32 || out.DType != ETensorDType::FP32)) {
        auto shape_vec = [](const Tensor& t) {
            return std::vector<long>(t.Sizes.begin(), t.Sizes.begin() + t.Rank);
        };
        Tensor a_f = a;
        if (a.DType != ETensorDType::FP32) {
            a_f = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(a), "matmul_router_a_fp32");
            mTemps.push_back(a_f);
            if (a.DType == ETensorDType::BF16) {
                convert_dtype(a_f.get<float>(), a.get<nv_bfloat16>(), a.nelem(), mRunState.MainStream);
            } else {
                throw std::runtime_error("router matmul: unsupported input dtype");
            }
        }

        Tensor b_f = b;
        if (b.DType != ETensorDType::FP32) {
            b_f = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(b), "matmul_router_b_fp32");
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
                Tensor tmp = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(*bias), "matmul_router_bias_fp32");
                mTemps.push_back(tmp);
                if (bias->DType == ETensorDType::BF16) {
                    convert_dtype(tmp.get<float>(), bias->get<nv_bfloat16>(), bias->nelem(), mRunState.MainStream);
                } else {
                    throw std::runtime_error("router matmul: unsupported bias dtype");
                }
                bias_f = tmp;
            }
        }

        Tensor out_f = out;
        if (out.DType != ETensorDType::FP32) {
            out_f = mRunState.temp_alloc(ETensorDType::FP32, shape_vec(out), "matmul_router_out_fp32");
            mTemps.push_back(out_f);
        }

        EMMTranspose mode_col = swap_transpose(op.attrs.transpose);
        matmul(out_f,
               b_f,
               a_f,
               bias_f,
               nullptr,
               nullptr,
               mRunState.CublasLtHandle,
               mRunState.CuBlasWorkspace,
               N,
               M,
               K,
               mode_col,
               false,
               mRunState.MainStream);
        if (out.DType != ETensorDType::FP32) {
            if (out.DType == ETensorDType::BF16) {
                convert_dtype(out.get<nv_bfloat16>(), out_f.get<float>(), out.nelem(), mRunState.MainStream);
            } else if (out.DType == ETensorDType::FP32) {
                // no-op
            } else {
                throw std::runtime_error("router matmul: unsupported output dtype");
            }
        }
        return;
    }

    bool used_recipe = false;
    modules::MatmulContext ctx{};
    modules::MatmulContext* ctx_ptr = nullptr;
    try {
        if (mRecipe && op.attrs.transpose == EMMTranspose::NT && a.Sizes[0] == mB * mT && !is_shared_weight) {
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
                ctx.op_caps = op.default_caps;
                ctx.matmul_caps = op.matmul_caps;
                ctx.epilogue_support = op.epilogue_support;
                ctx.storage_compat = op.storage_compat;
                attach_input_role(ctx, mCurrentGraph, op.inputs[0]);
                ctx.allow_fp8 = mRecipe->uses_fp8_forward();
                ctx.allow_fp4 = mRecipe->uses_fp4_forward();

                // Wire FP8/FP4 buffers + static weight caches (GraphExecutor primes caches before CUDA graph capture).
                if (ctx.allow_fp8) {
                    ctx.inp_quant = fp8_forward_buffer(mRunState, *op.attrs.matmul_op);
                    ctx.delayed_quantizer_idx = fp8_quantizer_index(mRunState, *op.attrs.matmul_op, op.attrs.layer_idx);

                    // Check if the upstream activation dispatch has already pre-quantized
                    // the input into the FP8 buffer (co-located quantization).
                    DslRunState::FP8BufferReady ready_flag = fp8_ready_flag_for_matmul_op(*op.attrs.matmul_op);
                    if (ready_flag != DslRunState::FP8Ready_None) {
                        if (is_gate_projection && mRunState.is_fp8_buffer_ready(ready_flag)) {
                            ctx.inp_quant_ready = true;
                        } else if (mRunState.consume_fp8_buffer_ready(ready_flag)) {
                            ctx.inp_quant_ready = true;
                        }
                    }

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
                    if (it != mFP4Cache->end() && it->second.initialized && it->second.data.Data &&
                        it->second.scales.Data && it->second.amax.Data) {
                        ctx.cached_fp4_data = &it->second.data;
                        ctx.cached_fp4_scales = &it->second.scales;
                        ctx.cached_fp4_amax = it->second.amax.get<float>();
                    }
                }

                used_recipe = true;
                mRecipe->forward_matmul(ctx);
                ctx_ptr = &ctx;
            }
        }

        if (!used_recipe) {
            EMMTranspose mode_col = swap_transpose(op.attrs.transpose);
            matmul(out,
                   b,
                   a,
                   bias,
                   nullptr,
                   nullptr,
                   mRunState.CublasLtHandle,
                   mRunState.CuBlasWorkspace,
                   N,
                   M,
                   K,
                   mode_col,
                   false,
                   mRunState.MainStream);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("dispatch_matmul failed for op_id='" + op.op_id + "' weight='" + weight_name +
                                 "' transpose=" + std::to_string(static_cast<int>(op.attrs.transpose)) +
                                 " used_recipe=" + std::string(used_recipe ? "1" : "0") + " M=" + std::to_string(M) +
                                 " N=" + std::to_string(N) + " K=" + std::to_string(K) +
                                 " a_rank=" + std::to_string(a.Rank) + " b_rank=" + std::to_string(b.Rank) +
                                 " out_rank=" + std::to_string(out.Rank) +
                                 " a_dtype=" + std::to_string(static_cast<int>(a.DType)) +
                                 " b_dtype=" + std::to_string(static_cast<int>(b.DType)) +
                                 " out_dtype=" + std::to_string(static_cast<int>(out.DType)) + ": " + e.what());
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
            case modules::MatmulOp::QKV: layer_plan.qkv = plan; break;
            case modules::MatmulOp::AttnOut: layer_plan.out_proj = plan; break;
            case modules::MatmulOp::MLPUp:
                if (!is_gate_projection) {
                    layer_plan.mlp_up = plan;
                }
                break;
            case modules::MatmulOp::MLPDown: layer_plan.mlp_down = plan; break;
            default: break;
        }
    }

    (void)hook;

    LoRAForwardApplyContext lora_apply_ctx;
    AfterProduceHookPayload after_produce_payload;
    if (!op.attrs.lora_slices.empty()) {
        Tensor a_flat = flatten_bt(a, mB, mT);
        Tensor out_flat = flatten_bt(out, mB, mT);
        lora_apply_ctx.slices = &op.attrs.lora_slices;
        lora_apply_ctx.layer_idx = op.attrs.layer_idx;
        lora_apply_ctx.input_2d = a_flat;
        lora_apply_ctx.output_2d = out_flat;
        lora_apply_ctx.BT = static_cast<int>(a_flat.Sizes[0]);
        lora_apply_ctx.weights = mLoRAWeights;
        lora_apply_ctx.config = mLoRAConfig;
        lora_apply_ctx.run_state = mLoRARunState;
        lora_apply_ctx.handle = mRunState.CublasLtHandle;
        lora_apply_ctx.workspace = &mRunState.CuBlasWorkspace;
        lora_apply_ctx.stream = mRunState.MainStream;
        after_produce_payload.action_context = &lora_apply_ctx;
        after_produce_payload.apply_lora = apply_lora_forward_payload;
    }

    // Rebind the per-layer activation slot to the just-produced buffer so
    // backward replays read the live tensor rather than a stale one.
    if (op.attrs.forward_hook_point.has_value() && op.attrs.layer_idx >= 0 && op.attrs.layer_idx < mConfig.NumLayers) {
        TensorSlot slot = TensorSlot::Mapped;
        switch (*op.attrs.forward_hook_point) {
            case modules::ForwardHookPoint::AfterQKVProjection: slot = TensorSlot::BlockQKV; break;
            case modules::ForwardHookPoint::AfterAttnOutProjection: slot = TensorSlot::BlockAttOut; break;
            case modules::ForwardHookPoint::AfterMLPUpProjection: slot = TensorSlot::BlockMLPUp; break;
            case modules::ForwardHookPoint::AfterMLPDownProjection: slot = TensorSlot::BlockMLPDown; break;
            case modules::ForwardHookPoint::AfterRouterProjection: slot = TensorSlot::BlockRouterLogits; break;
            default: break;
        }
        if (slot != TensorSlot::Mapped) {
            if (Tensor* t = block_activation_ptr(mRunState, op.attrs.layer_idx, slot)) {
                t->Data = out.Data;
            }
        }
        dispatch_schema_hook(HookEventKind::AfterProduce,
                             op.attrs.layer_idx,
                             op.attrs.hook_schema_id,
                             op.attrs.forward_hook_schema_slot,
                             op.attrs.lora_slices.empty() ? nullptr : &after_produce_payload);
    }

    if (!op.attrs.lora_slices.empty() && !after_produce_payload.lora_applied) {
        apply_lora_forward_payload(&lora_apply_ctx);
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

    // Check if weight gradient should be skipped BEFORE allocating (frozen weights in LoRA mode).
    // Classifier-backed: ParamGrad tids carry the base param name on TensorMeta, the single
    // source of truth. ActivationGrad / AccumTemp / Scratch return nullopt, which correctly
    // skips `mGrads.get_param_grad` for non-parameter outputs.
    bool skip_weight_grad = true;
    const std::string& dB_name = op.outputs.size() > 1 ? op.outputs[1].name : "";
    if (!dB_name.empty() && mCurrentGraph) {
        if (auto weight_name = base_param_from_grad_kind(op.outputs[1].tensor_id, *mCurrentGraph)) {
            bool accum = false;
            Tensor* grad = mGrads.get_param_grad(*weight_name, accum);
            skip_weight_grad = (grad == nullptr || !grad->Data);
        }
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

    const bool is_qkv_op = op.attrs.matmul_op.has_value() && (*op.attrs.matmul_op == modules::MatmulOp::QKV);

    // Now allocate output tensors - skip dB if weights are frozen.
    // For backward ops, compiled shapes of saved-tensor-derived outputs may be empty
    // (backward compiler can't track saved tensor shapes). Derive from runtime inputs.
    auto ensure_backward_output =
        [&](const TensorRef& ref, const Tensor& shape_source, bool allow_shape_fallback) -> Tensor& {
        const bool debug_mlp_bwd = []() {
            const char* env = std::getenv("SUROGATE_DEBUG_MATMUL_BWD");
            return env && std::string(env) != "0";
        }();
        const auto expected_nelem = shape_source.nelem();
        const auto expected_dtype = shape_source.DType;
        const std::vector<long> expected_shape(shape_source.Sizes.begin(),
                                               shape_source.Sizes.begin() + shape_source.Rank);

        auto alloc_temp_fallback = [&]() -> Tensor& {
            Tensor t = mRunState.temp_alloc(expected_dtype, expected_shape, "matmul_bwd_temp");
            fill_zero(t, mRunState.MainStream);
            mTemps.push_back(t);
            store_tensor(ref, t);
            if (debug_mlp_bwd && ref.name.find("mlp_x_flat") != std::string::npos) {
                std::fprintf(stderr,
                             "[MATMUL_BWD_OUT] name=%s path=fallback ptr=%p shape=[%ld,%ld] shape_empty=%d tid=%d\n",
                             ref.name.c_str(),
                             static_cast<void*>(t.Data),
                             expected_shape.size() > 0 ? expected_shape[0] : -1,
                             expected_shape.size() > 1 ? expected_shape[1] : -1,
                             ref.shape.empty() ? 1 : 0,
                             ref.tensor_id);
            }
            return mTensors[ref.tensor_id];
        };

        if (ref.shape.empty() && shape_source.Rank > 0) {
            return alloc_temp_fallback();
        }

        Tensor& out_ref = ensure_output_tensor(ref);
        // Require actual storage, not just matching metadata. `ensure_output_tensor`'s
        // Mapped/fuzzy fallbacks can return a tensor with the right shape/dtype but a
        // null `Data` pointer (e.g., when a slot is configured to be recomputed or
        // shared-but-not-yet-populated). Running a cuBLAS GEMM on such a tensor aborts
        // with CUBLAS_STATUS_INVALID_VALUE. Treat a null-storage result as "not ready"
        // and fall through to the temp allocation path below.
        if (out_ref.Data && out_ref.DType == expected_dtype && out_ref.nelem() == expected_nelem) {
            if (debug_mlp_bwd && ref.name.find("mlp_x_flat") != std::string::npos) {
                std::fprintf(stderr,
                             "[MATMUL_BWD_OUT] name=%s path=ensure ptr=%p nelem=%ld shape_empty=%d tid=%d\n",
                             ref.name.c_str(),
                             static_cast<void*>(out_ref.Data),
                             out_ref.nelem(),
                             ref.shape.empty() ? 1 : 0,
                             ref.tensor_id);
            }
            return out_ref;
        }
        if (!out_ref.Data && allow_shape_fallback) {
            return alloc_temp_fallback();
        }

        if (!allow_shape_fallback) {
            throw std::runtime_error("matmul_backward: weight-grad output tensor shape/dtype mismatch for " + ref.name);
        }
        if (debug_mlp_bwd && ref.name.find("mlp_x_flat") != std::string::npos) {
            std::fprintf(stderr,
                         "[MATMUL_BWD_OUT] name=%s path=shape_mismatch current_ptr=%p current_nelem=%ld expected=%ld "
                         "shape_empty=%d tid=%d\n",
                         ref.name.c_str(),
                         static_cast<void*>(out_ref.Data),
                         out_ref.nelem(),
                         expected_nelem,
                         ref.shape.empty() ? 1 : 0,
                         ref.tensor_id);
        }
        return alloc_temp_fallback();
    };

    Tensor* dA_ptr = nullptr;
    Tensor* dB_ptr = nullptr;

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        dA_ptr = &ensure_backward_output(op.outputs[0], a, /*allow_shape_fallback=*/true);
    }
    if (!skip_weight_grad && op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        dB_ptr = &ensure_backward_output(op.outputs[1], b, /*allow_shape_fallback=*/false);
    }

    if (!dA_ptr && !dB_ptr) {
        return;
    }

    // Defensive invariant: matmul_backward's two outputs (dA = activation grad,
    // dB = parameter grad) are written by a SINGLE cuBLAS GEMM and therefore
    // MUST live in disjoint storage. Aliasing them — which previously occurred
    // when `ensure_output_tensor`'s fuzzy fallback landed both on the same
    // simplified-grads slot — produces a cuBLAS INVALID_VALUE at dispatch time
    // on some shapes and silent gradient corruption on others. If we detect an
    // alias here, reallocate dA into a fresh temp. This is a safety net —
    // the root fix is classifier-driven routing (TensorKind) at resolution time
    // so the aliasing can't be produced in the first place. Phase 1+ migrates
    // callers to that path one at a time; this check catches anything that
    // slipped through.
    if (dA_ptr && dB_ptr && dA_ptr->Data && dA_ptr->Data == dB_ptr->Data) {
        std::vector<long> a_shape(a.Sizes.begin(), a.Sizes.begin() + a.Rank);
        Tensor dA_fresh = mRunState.temp_alloc(a.DType, a_shape, "matmul_bwd_dA_disalias");
        fill_zero(dA_fresh, mRunState.MainStream);
        mTemps.push_back(dA_fresh);
        store_tensor(op.outputs[0], dA_fresh);
        dA_ptr = &mTensors[static_cast<std::size_t>(op.outputs[0].tensor_id)];
        std::fprintf(stderr,
                     "[matmul_backward][disalias] %s and %s collided on %p — dA reallocated to %p\n",
                     op.outputs[0].name.c_str(),
                     op.outputs[1].name.c_str(),
                     static_cast<void*>(dB_ptr->Data),
                     static_cast<void*>(dA_ptr->Data));
    }

    // Bulk debug: log every matmul_backward's output pointers when requested.
    if (const char* env = std::getenv("SUROGATE_DEBUG_MATMUL_BWD_ALL")) {
        (void)env;
        std::fprintf(stderr,
                     "[matmul_backward] op=%s dA_name=%s dB_name=%s dA_ptr=%p dB_ptr=%p a=%p b=%p d_out=%p\n",
                     op.op_id.c_str(),
                     op.outputs.empty() ? "" : op.outputs[0].name.c_str(),
                     op.outputs.size() < 2 ? "" : op.outputs[1].name.c_str(),
                     static_cast<void*>(dA_ptr ? dA_ptr->Data : nullptr),
                     static_cast<void*>(dB_ptr ? dB_ptr->Data : nullptr),
                     static_cast<void*>(a.Data),
                     static_cast<void*>(b.Data),
                     static_cast<void*>(d_out.Data));
    }

    bool do_accumulate = mAccumulateTensors.count(dB_name) > 0;
    if (!do_accumulate && !dB_name.empty() && mCurrentGraph) {
        // Classifier-backed accum lookup: the name of the underlying parameter
        // comes from TensorKind::ParamGrad's base_param_tid — never a string-strip.
        if (auto base = base_param_from_grad_kind(op.outputs[1].tensor_id, *mCurrentGraph)) {
            do_accumulate = mAccumulateTensors.count("d_" + *base) > 0;
        }
    }

    bool used_recipe = false;
    bool used_fp8 = false;
    bool has_dout_quant = false;

    const bool disable_qkv_recipe_bwd = is_qkv_op && skip_weight_grad && (mConfig.NumExperts > 0);
    if (mRecipe && mode == EMMTranspose::NT && a.Sizes[0] == mB * mT && allow_quant && !disable_qkv_recipe_bwd) {
        Tensor dA_tmp{};
        Tensor dB_tmp{};
        Tensor* dA_use = dA_ptr;
        Tensor* dB_use = dB_ptr;

        if (!dA_use) {
            dA_tmp = mRunState.temp_alloc(a.DType, {a.Sizes[0], a.Sizes[1]}, "matmul_dA_tmp");
            mTemps.push_back(dA_tmp);
            dA_use = &dA_tmp;
        }
        if (!dB_use) {
            dB_tmp = mRunState.temp_alloc(b.DType, {b.Sizes[0], b.Sizes[1]}, "matmul_dB_tmp");
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
        ctx.op_caps = op.default_caps;
        ctx.matmul_caps = op.matmul_caps;
        ctx.epilogue_support = op.epilogue_support;
        ctx.storage_compat = op.storage_compat;
        attach_input_role(ctx, mCurrentGraph, op.inputs[0]);
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
                if (it != cache->end() && it->second.initialized && it->second.data.Data && it->second.scales.Data &&
                    it->second.amax.Data) {
                    ctx.cached_fp4_data = &it->second.data;
                    ctx.cached_fp4_scales = &it->second.scales;
                    ctx.cached_fp4_amax = it->second.amax.get<float>();
                }
            }
        }
        used_fp8 = ctx.allow_fp8;
        has_dout_quant = (ctx.dout_quant && ctx.dout_quant->Data);

        used_recipe = true;
        mRecipe->backward_matmul(ctx);
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

        // Validate d_out shape matches expected forward output shape.
        // The gradient buffer may be mapped to a wrong-sized slot for ops
        // whose intermediate shapes don't match any pre-allocated slot
        // (e.g., PLI gate: [B*T, PLI_D] mapped to [B*T, C]).
        {
            const bool transB = (mode == EMMTranspose::NT || mode == EMMTranspose::TT);
            const long expected_cols = transB ? b.Sizes[0] : b.Sizes[1];
            if (d_out_mat.Rank == 2 && d_out_mat.Sizes[1] != expected_cols) {
                d_out_mat = view_tensor(d_out_mat, {d_out_mat.Sizes[0], expected_cols});
            }
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
            matmul(*dA_ptr,
                   b,
                   d_out_mat,
                   std::nullopt,
                   nullptr,
                   nullptr,
                   mRunState.CublasLtHandle,
                   mRunState.CuBlasWorkspace,
                   N,
                   M,
                   K,
                   mode_col,
                   false,
                   mRunState.MainStream);
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
            matmul(*dB_ptr,
                   *rhs,
                   *lhs,
                   std::nullopt,
                   nullptr,
                   nullptr,
                   mRunState.CublasLtHandle,
                   mRunState.CuBlasWorkspace,
                   N,
                   M,
                   K,
                   mode_col,
                   do_accumulate,
                   mRunState.MainStream);
        }
    }

    if (!op.attrs.lora_slices.empty() && mLoRAConfig && mLoRAConfig->enabled() && mLoRAWeights && mLoRAGrads &&
        mLoRARunState && mComm && layer_idx >= 0) {
        Tensor a_flat = flatten_bt(a, mB, mT);
        Tensor d_out_flat = flatten_bt(d_out, mB, mT);
        // If the base matmul produced a d_input buffer, accumulate LoRA's dx
        // contribution into it. Otherwise pass an empty tensor (``skip_dx``).
        Tensor dx_target = (dA_ptr && dA_ptr->Data) ? flatten_bt(*dA_ptr, mB, mT) : Tensor{};
        const int BT = static_cast<int>(a_flat.Sizes[0]);
        modules::detail::apply_lora_slices_backward(op.attrs.lora_slices,
                                                    layer_idx,
                                                    a_flat,
                                                    d_out_flat,
                                                    dx_target,
                                                    BT,
                                                    mLoRAWeights,
                                                    mLoRAGrads,
                                                    mLoRAConfig,
                                                    mLoRARunState,
                                                    *mComm,
                                                    do_accumulate,
                                                    mRunState.CublasLtHandle,
                                                    mRunState.CuBlasWorkspace,
                                                    mRunState.MainStream);
    }
}

namespace {

// -----------------------------------------------------------------------------
// Matmul backward rule
// Forward: C = A @ B (with optional transpose modes)
// Backward: dA = dC @ B.T, dB = A.T @ dC (adjusted for transpose modes)
// -----------------------------------------------------------------------------
std::vector<Operation> matmul_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    const std::string& A = fwd.inputs[0];
    const std::string& B = fwd.inputs[1];
    const std::string& dC = ctx.d_output;

    // Parse transpose mode from forward op
    std::string trans = get_string_attr(fwd.attrs, "transpose", "NN");

    // Determine backward transpose modes based on forward mode
    // Forward: C = op(A) @ op(B) where op depends on transpose flags
    // For NN: C = A @ B       -> dA = dC @ B.T (NT), dB = A.T @ dC (TN)
    // For NT: C = A @ B.T     -> dA = dC @ B (NN),   dB = dC.T @ A (TN) = A.T @ dC.T ...
    // For TN: C = A.T @ B     -> dA = B @ dC.T (NT), dB = A @ dC (NN)
    // For TT: C = A.T @ B.T   -> dA = B.T @ dC.T,    dB = dC.T @ A.T

    // Determine references for A and B in backward pass:
    // - Parameters are available at backward time (gathered from weight manager)
    // - Activations must be saved from forward pass (use saved_ref)
    std::string A_for_dB = ctx.is_param(A) ? A : saved_ref(A);
    std::string B_for_dA = ctx.is_param(B) ? B : saved_ref(B);

    AttrMap attrs;
    attrs["transpose"] = AttrValue(trans);

    std::vector<std::string> inputs = {dC, A_for_dB, B_for_dA};
    std::vector<std::string> outputs = {ctx.d_inputs[0], ctx.d_inputs[1]};

    ops.push_back(make_operation("matmul_backward_" + std::to_string(ctx.op_counter++),
                                 "matmul_backward",
                                 "matmul_backward",
                                 inputs,
                                 outputs,
                                 attrs));

    return ops;
}

}  // namespace

namespace {

// -----------------------------------------------------------------------------
// Matmul + Bias backward rule
// Forward: C = A @ B (+ bias), with optional transpose modes
// Backward: dA = dC @ B.T, dB = A.T @ dC (adjusted for transpose modes), dBias = sum(dC)
// -----------------------------------------------------------------------------
std::vector<Operation> matmul_bias_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    const std::string& A = fwd.inputs[0];
    const std::string& B = fwd.inputs[1];
    const std::string& dC = ctx.d_output;
    const std::string bias = (fwd.inputs.size() > 2) ? fwd.inputs[2] : "";

    std::string trans = get_string_attr(fwd.attrs, "transpose", "NN");

    std::string A_for_dB = ctx.is_param(A) ? A : saved_ref(A);
    std::string B_for_dA = ctx.is_param(B) ? B : saved_ref(B);

    AttrMap attrs;
    attrs["transpose"] = AttrValue(trans);
    std::vector<std::string> inputs = {dC, A_for_dB, B_for_dA};
    std::vector<std::string> outputs = {ctx.d_inputs[0], ctx.d_inputs[1]};
    ops.push_back(make_operation("matmul_bias_backward_" + std::to_string(ctx.op_counter++),
                                 "matmul_backward",
                                 "matmul_backward",
                                 inputs,
                                 outputs,
                                 attrs));

    if (ctx.needs_grad(2) && !bias.empty()) {
        std::vector<std::string> outputs;
        outputs.push_back("");
        outputs.push_back(ctx.d_inputs[2]);
        ops.push_back(make_operation("matmul_bias_dBias_" + std::to_string(ctx.op_counter++),
                                     "bias_add_backward",
                                     "bias_add_backward",
                                     {dC, bias},
                                     outputs));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("matmul_bias", ::dsl::matmul_bias_backward);

REGISTER_AUTODIFF("matmul", ::dsl::matmul_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// Matmul
// ------------------------------------------------------------------------
const int _matmul_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "matmul";
    sig.min_inputs = 2;
    sig.max_inputs = 2;
    sig.min_outputs = 1;
    sig.max_outputs = 3;
    sig.validator = [](const auto& inputs, const auto& outputs, const AttrMap& attrs, const ShapeEnv&) {
        if (inputs.size() < 2 || outputs.empty()) {
            ShapeValidationError err;
            err.message = "matmul requires 2 inputs and 1 output";
            return std::make_optional(err);
        }
        return validators::check_matmul_dims(inputs[0], inputs[1], outputs[0], attrs);
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// Matmul + Bias
// ------------------------------------------------------------------------
const int _matmul_bias_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "matmul_bias";
    sig.min_inputs = 3;
    sig.max_inputs = 3;
    sig.min_outputs = 1;
    sig.max_outputs = 3;
    sig.validator = [](const auto& inputs, const auto& outputs, const AttrMap& attrs, const ShapeEnv&) {
        if (inputs.size() < 3 || outputs.empty()) {
            ShapeValidationError err;
            err.message = "matmul_bias requires 3 inputs and 1 output";
            return std::make_optional(err);
        }

        // Check matmul dims
        auto matmul_err = validators::check_matmul_dims(inputs[0], inputs[1], outputs[0], attrs);
        if (matmul_err) return matmul_err;

        // Check bias shape (should be broadcastable with output)
        const auto& bias_shape = inputs[2];
        const auto& out_shape = outputs[0];
        if (bias_shape.size() > out_shape.size()) {
            ShapeValidationError err;
            err.message = "matmul_bias: bias rank exceeds output rank";
            return std::make_optional(err);
        }

        // Bias last dim should match output last dim
        if (!bias_shape.empty() && !out_shape.empty()) {
            if (bias_shape.back() != out_shape.back() && bias_shape.back() != 1) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "matmul_bias: bias last dim (" << bias_shape.back() << ") doesn't match output last dim ("
                    << out_shape.back() << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// MatmulBackward
// ------------------------------------------------------------------------
const int _matmul_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "matmul_backward";
    sig.min_inputs = 3;
    sig.max_inputs = 3;
    sig.min_outputs = 2;
    sig.max_outputs = 2;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& a = inputs[1];
        const auto& b = inputs[2];
        const auto& d_a = outputs[0];
        const auto& d_b = outputs[1];

        // Check shapes match forward matmul
        if (auto err = validators::check_matmul_dims(a, b, d_out, attrs)) {
            return err;
        }

        // Check gradient shapes match input shapes
        if (auto err = validators::check_same_numel(d_a, a, "d_a", "a", "matmul_backward")) {
            return err;
        }
        if (auto err = validators::check_same_numel(d_b, b, "d_b", "b", "matmul_backward")) {
            return err;
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
