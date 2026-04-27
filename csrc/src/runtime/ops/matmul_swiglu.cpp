#include "runtime/executor/compiled_ops.h"
#include "runtime/dsl/tensor_slot_dispatch.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/lora/lora_slice_dispatch.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_matmul_swiglu(const CompiledOp& op, const modules::ForwardHook* hook) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    Tensor& up_out = ensure_output_tensor(op.outputs[1]);
    const std::string& weight_name = op.inputs[1].name;

    int M = 0, N = 0, K = 0;
    matmul_dims(a, b, op.attrs.transpose, M, N, K);
    const long D = N / 2;

    bool used_recipe = false;
    modules::MatmulContext ctx{};
    modules::MatmulContext* ctx_ptr = nullptr;
    if (mRecipe && op.attrs.transpose == EMMTranspose::NT && a.Sizes[0] == mB * mT && op.attrs.allow_quant &&
        op.attrs.matmul_op.has_value()) {
        ctx.out = &up_out;
        ctx.inp = &a;
        ctx.weight = &b;
        ctx.bias = nullptr;
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
        ctx.allow_fp8 = mRecipe->uses_fp8_forward();
        ctx.allow_fp4 = mRecipe->uses_fp4_forward();

        if (ctx.allow_fp8) {
            ctx.inp_quant = fp8_forward_buffer(mRunState, *op.attrs.matmul_op);
            ctx.delayed_quantizer_idx = fp8_quantizer_index(mRunState, *op.attrs.matmul_op, op.attrs.layer_idx);

            // Check if the upstream rmsnorm dispatch has already pre-quantized
            // the LN2 output into the FP8 buffer.
            DslRunState::FP8BufferReady ready_flag = fp8_ready_flag_for_matmul_op(*op.attrs.matmul_op);
            if (ready_flag != DslRunState::FP8Ready_None && mRunState.consume_fp8_buffer_ready(ready_flag)) {
                ctx.inp_quant_ready = true;
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
            if (it != mFP4Cache->end() && it->second.initialized && it->second.data.Data && it->second.scales.Data &&
                it->second.amax.Data) {
                ctx.cached_fp4_data = &it->second.data;
                ctx.cached_fp4_scales = &it->second.scales;
                ctx.cached_fp4_amax = it->second.amax.get<float>();
            }
        }

        mRecipe->forward_matmul(ctx);
        used_recipe = true;
        ctx_ptr = &ctx;
    }

    if (!used_recipe) {
        matmul(up_out,
               b,
               a,
               std::nullopt,
               nullptr,
               nullptr,
               mRunState.CublasLtHandle,
               mRunState.CuBlasWorkspace,
               N,
               M,
               K,
               swap_transpose(op.attrs.transpose),
               false,
               mRunState.MainStream);
    }

    (void)hook;

    // Rebind the per-layer mlp_up activation slot to the live buffer so
    // backward replays read the fresh tensor.
    if (op.attrs.forward_hook_point.has_value() && op.attrs.layer_idx >= 0 && op.attrs.layer_idx < mConfig.NumLayers) {
        if (Tensor* mlp_up = block_activation_ptr(mRunState, op.attrs.layer_idx, TensorSlot::BlockMLPUp)) {
            mlp_up->Data = up_out.Data;
        }
    }

    // Apply DSL-declared LoRA adapters (up/gate/...) to ``up_out`` before
    // the SwiGLU activation.
    if (!op.attrs.lora_slices.empty()) {
        Tensor a_flat = flatten_bt(a, mB, mT);
        Tensor up_flat = flatten_bt(up_out, mB, mT);
        const int BT = static_cast<int>(a_flat.Sizes[0]);
        modules::detail::apply_lora_slices_forward(op.attrs.lora_slices,
                                                   op.attrs.layer_idx,
                                                   a_flat,
                                                   up_flat,
                                                   BT,
                                                   mLoRAWeights,
                                                   mLoRAConfig,
                                                   mLoRARunState,
                                                   mRunState.CublasLtHandle,
                                                   mRunState.CuBlasWorkspace,
                                                   mRunState.MainStream);
    }

    Tensor up_3d = view_tensor(up_out, {mB, mT, static_cast<long>(N)});
    Tensor out_3d = view_tensor(out, {mB, mT, D});
    swiglu_forward(out_3d,
                   up_3d,
                   nullptr,
                   static_cast<int>(mB),
                   static_cast<int>(mT),
                   static_cast<int>(D),
                   mRunState.MainStream);

    // Pre-quantize swiglu output into FP8 buffer for the downstream MLPDown matmul.
    // This co-locates quantization with the data producer (better L2 locality)
    // and allows the matmul recipe to skip its own quantization pass.
    if (mRecipe && mRecipe->uses_fp8_forward() && mRunState.has_fp8_forward() && !mRunState.has_fp8_delayed_scaling()) {
        auto& fp8_buf = mRunState.fp8_forward_quants().swiglu;
        if (fp8_buf.Data && fp8_buf.abs_max() && fp8_buf.scale()) {
            const long num_elements = mB * mT * D;
            Tensor swiglu_flat = view_tensor(out_3d, {mB * mT, D});
            quantize_with_abs_max(fp8_buf,
                                  fp8_buf.scale(),
                                  swiglu_flat,
                                  fp8_buf.abs_max(),
                                  num_elements,
                                  mRunState.DeviceProp,
                                  mRunState.MainStream);
            mRunState.set_fp8_buffer_ready(DslRunState::FP8Ready_SwiGLU);
        }
    }

    // Record forward plan for recompute (treat matmul_swiglu as the MLPUp projection).
    if (mForwardPlan && op.attrs.matmul_op.has_value() && op.attrs.layer_idx >= 0 &&
        static_cast<std::size_t>(op.attrs.layer_idx) < mForwardPlan->size() &&
        *op.attrs.matmul_op == modules::MatmulOp::MLPUp) {
        MatmulForwardPlan plan{};
        plan.valid = true;
        plan.use_recipe = used_recipe;
        plan.has_bias = false;
        if (used_recipe && ctx_ptr) {
            plan.allow_fp8 = ctx_ptr->allow_fp8;
            plan.allow_fp4 = ctx_ptr->allow_fp4;
            plan.delayed_quantizer_idx = ctx_ptr->delayed_quantizer_idx;
            plan.use_fp8_cache = (ctx_ptr->cached_weight && ctx_ptr->cached_weight->Data);
            plan.use_fp4_cache = (ctx_ptr->cached_fp4_data && ctx_ptr->cached_fp4_scales);
        }
        (*mForwardPlan)[static_cast<std::size_t>(op.attrs.layer_idx)].mlp_up = plan;
    }
}

void CompiledExecutor::dispatch_matmul_swiglu_backward(const CompiledOp& op, const modules::BackwardHook* hook) {
    // Combined backward for matmul + swiglu (fused op in forward)
    // inputs: d_swiglu_out, ln2 (matmul input), mlp_up_weight, mlp_up (pre-swiglu)
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);
    Tensor mlp_up = resolve_tensor(op.inputs[3]);

    const int layer_idx = op.attrs.layer_idx;
    const bool allow_quant = op.attrs.allow_quant;
    const std::string& weight_name = op.inputs.size() > 2 ? op.inputs[2].name : "";

    // Recompute mlp_up if the saved tensor was stack-allocated and freed
    bool recomputed_mlp_up = false;
    if (!mlp_up.Data || (mRunState.Stack.owns(mlp_up.Data) && !mRunState.Stack.is_live(mlp_up.Data))) {
        int M = 0, N = 0, K = 0;
        Tensor inp_flat = (inp.Rank == 2) ? inp : view_tensor(inp, {mB * mT, inp.Sizes[inp.Rank - 1]});
        matmul_dims(inp_flat, weight, op.attrs.transpose, M, N, K);
        const long D2 = N;
        Tensor mlp_up_flat = mRunState.temp_alloc(inp.DType, {mB * mT, D2}, "matmul_swiglu_backward_recompute_mlp_up");
        mTemps.push_back(mlp_up_flat);

        EMMTranspose mode_col = swap_transpose(op.attrs.transpose);
        matmul(mlp_up_flat,
               weight,
               inp_flat,
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

        mlp_up = view_tensor(mlp_up_flat, {mB, mT, D2});
        if (layer_idx >= 0) {
            if (Tensor* mlp_up_slot = block_activation_ptr(mRunState, layer_idx, TensorSlot::BlockMLPUp)) {
                mlp_up_slot->Data = mlp_up.Data;
            }
        }
        recomputed_mlp_up = true;
    }

    // First: swiglu backward
    Tensor* d_mlp_up_ptr = nullptr;
    if (layer_idx >= 0) {
        d_mlp_up_ptr = block_gradient_ptr(mRunState, layer_idx, TensorSlot::BlockDMLPUp);
        if (d_mlp_up_ptr && !d_mlp_up_ptr->Data) {
            mRunState.temp_acquire(*d_mlp_up_ptr);
            mTemps.push_back(*d_mlp_up_ptr);
        }
    }
    Tensor d_mlp_up = d_mlp_up_ptr ? *d_mlp_up_ptr
                                   : mRunState.temp_alloc(mlp_up.DType,
                                                          {mlp_up.Sizes[0], mlp_up.Sizes[1], mlp_up.Sizes[2]},
                                                          "matmul_swiglu_backward_d_mlp_up");
    if (!d_mlp_up_ptr) {
        mTemps.push_back(d_mlp_up);
    }

    // For FP8 hybrid backward, record abs_max of d_mlp_up for subsequent quantization
    float* abs_max_ptr =
        mRunState.has_fp8_hybrid_backward() ? mRunState.simplified_quant_grads().d_mlp_up.abs_max() : nullptr;

    const long D = d_out.Sizes[2];
    swiglu_backward(d_mlp_up,
                    d_out,
                    mlp_up,
                    abs_max_ptr,
                    static_cast<int>(d_out.Sizes[0]),
                    static_cast<int>(d_out.Sizes[1]),
                    static_cast<int>(D),
                    mRunState.MainStream);

    // Then: matmul backward
    Tensor d_mlp_up_flat = view_tensor(d_mlp_up, {mB * mT, 2 * D});

    Tensor* d_inp_ptr = nullptr;
    Tensor* d_weight_ptr = nullptr;

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        d_inp_ptr = &ensure_output_tensor(op.outputs[0]);
    }
    // Skip weight gradient if frozen (LoRA-only mode). Classifier-backed:
    // ParamGrad tids carry the base param name via TensorMeta; other kinds
    // return nullopt so non-parameter outputs correctly skip the grad store.
    bool skip_weight_grad = true;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty() && mCurrentGraph) {
        if (auto base = base_param_from_grad_kind(op.outputs[1].tensor_id, *mCurrentGraph)) {
            bool accum = false;
            Tensor* grad = mGrads.get_param_grad(*base, accum);
            skip_weight_grad = (grad == nullptr || !grad->Data);
        }
    }
    if (!skip_weight_grad && op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        d_weight_ptr = &ensure_output_tensor(op.outputs[1]);
    }

    bool do_accumulate = false;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        do_accumulate = (mAccumulateTensors.count(op.outputs[1].name) > 0);
        if (!do_accumulate && mCurrentGraph) {
            if (auto base = base_param_from_grad_kind(op.outputs[1].tensor_id, *mCurrentGraph)) {
                do_accumulate = mAccumulateTensors.count("d_" + *base) > 0;
            }
        }
    }

    bool used_recipe = false;
    if (mRecipe && op.attrs.transpose == EMMTranspose::NT && d_mlp_up_flat.Sizes[0] == mB * mT && allow_quant &&
        op.attrs.matmul_op.has_value()) {
        Tensor inp_flat = (inp.Rank == 2) ? inp : view_tensor(inp, {mB * mT, inp.Sizes[inp.Rank - 1]});

        Tensor dA_tmp{};
        Tensor dB_tmp{};
        Tensor* dA_use = d_inp_ptr;
        Tensor* dB_use = d_weight_ptr;

        if (!dA_use) {
            dA_tmp = mRunState.temp_alloc(inp.DType,
                                          {inp_flat.Sizes[0], inp_flat.Sizes[1]},
                                          "matmul_swiglu_backward_dA_tmp");
            mTemps.push_back(dA_tmp);
            dA_use = &dA_tmp;
        }
        if (!dB_use && !skip_weight_grad) {
            dB_tmp =
                mRunState.temp_alloc(weight.DType, {weight.Sizes[0], weight.Sizes[1]}, "matmul_swiglu_backward_dB_tmp");
            mTemps.push_back(dB_tmp);
            dB_use = &dB_tmp;
        }

        modules::MatmulContext ctx{};
        ctx.dinp = dA_use;
        ctx.dweight = dB_use;
        ctx.dout = &d_mlp_up_flat;
        ctx.inp = &inp_flat;
        ctx.weight = &weight;
        ctx.B = static_cast<int>(mB);
        ctx.T = static_cast<int>(mT);
        ctx.C_in = static_cast<int>(inp_flat.Sizes[1]);
        ctx.C_out = static_cast<int>(weight.Sizes[0]);
        ctx.run_state = &mRunState;
        ctx.stream = mRunState.MainStream;
        ctx.layer_idx = layer_idx;
        ctx.op = *op.attrs.matmul_op;
        ctx.op_caps = op.default_caps;
        ctx.matmul_caps = op.matmul_caps;
        ctx.epilogue_support = op.epilogue_support;
        ctx.storage_compat = op.storage_compat;
        ctx.accumulate = do_accumulate;
        ctx.skip_weight_grad = skip_weight_grad || !d_weight_ptr;
        ctx.allow_fp8 = allow_quant && mRecipe->uses_fp8_hybrid_backward();
        ctx.allow_fp4 = allow_quant && mRecipe->uses_fp4_forward();
        ctx.seed = mRngSeedFn ? mRngSeedFn() : 0u;

        if (ctx.allow_fp8) {
            ctx.dout_quant = fp8_grad_buffer(mRunState, *op.attrs.matmul_op);
            if (!ctx.dout_quant || !ctx.dout_quant->Data) {
                ctx.allow_fp8 = false;
            }
        }
        if (ctx.allow_fp8 && mFP8CacheT) {
            auto it = mFP8CacheT->find(weight_name);
            if (it != mFP8CacheT->end() && it->second.initialized && it->second.weight.Data) {
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

        mRecipe->backward_matmul(ctx);
        used_recipe = true;
    }

    if (!used_recipe) {
        if (d_inp_ptr) {
            EMMTranspose mode_dA = EMMTranspose::NN;
            switch (op.attrs.transpose) {
                case EMMTranspose::NN: mode_dA = EMMTranspose::NT; break;
                case EMMTranspose::NT: mode_dA = EMMTranspose::NN; break;
                case EMMTranspose::TN: mode_dA = EMMTranspose::NT; break;
                case EMMTranspose::TT: mode_dA = EMMTranspose::TT; break;
            }

            int M = 0, N = 0, K = 0;
            matmul_dims(d_mlp_up_flat, weight, mode_dA, M, N, K);
            EMMTranspose mode_col = swap_transpose(mode_dA);
            matmul(*d_inp_ptr,
                   weight,
                   d_mlp_up_flat,
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
        if (d_weight_ptr && !skip_weight_grad) {
            Tensor inp_flat = (inp.Rank == 2) ? inp : view_tensor(inp, {mB * mT, inp.Sizes[inp.Rank - 1]});

            const Tensor* lhs = nullptr;
            const Tensor* rhs = nullptr;
            EMMTranspose mode_rm = EMMTranspose::NN;
            switch (op.attrs.transpose) {
                case EMMTranspose::NN:
                    lhs = &inp_flat;
                    rhs = &d_mlp_up_flat;
                    mode_rm = EMMTranspose::TN;
                    break;
                case EMMTranspose::NT:
                    lhs = &d_mlp_up_flat;
                    rhs = &inp_flat;
                    mode_rm = EMMTranspose::TN;
                    break;
                case EMMTranspose::TN:
                    lhs = &inp_flat;
                    rhs = &d_mlp_up_flat;
                    mode_rm = EMMTranspose::NN;
                    break;
                case EMMTranspose::TT:
                    lhs = &d_mlp_up_flat;
                    rhs = &inp_flat;
                    mode_rm = EMMTranspose::TT;
                    break;
            }

            int M = 0, N = 0, K = 0;
            matmul_dims(*lhs, *rhs, mode_rm, M, N, K);
            EMMTranspose mode_col = swap_transpose(mode_rm);
            matmul(*d_weight_ptr,
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

    if (layer_idx >= 0 && d_inp_ptr) {
        if (Tensor* d_ln2 = block_gradient_ptr(mRunState, layer_idx, TensorSlot::BlockDLN2)) {
            d_ln2->Data = d_inp_ptr->Data;
        }
    }

    (void)hook;

    if (!op.attrs.lora_slices.empty() && mLoRAConfig && mLoRAConfig->enabled() && mLoRAWeights && mLoRAGrads &&
        mLoRARunState && mComm && layer_idx >= 0) {
        Tensor inp_flat = flatten_bt(inp, mB, mT);
        Tensor d_mlp_flat = d_mlp_up_flat;
        Tensor dx_target = (d_inp_ptr && d_inp_ptr->Data) ? flatten_bt(*d_inp_ptr, mB, mT) : Tensor{};
        const int BT = static_cast<int>(inp_flat.Sizes[0]);
        modules::detail::apply_lora_slices_backward(op.attrs.lora_slices,
                                                    layer_idx,
                                                    inp_flat,
                                                    d_mlp_flat,
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

}  // namespace dsl

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// MatmulSwiGLU
// ------------------------------------------------------------------------
const int _matmul_swiglu_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "matmul_swiglu";
    sig.min_inputs = 2;
    sig.max_inputs = 2;
    sig.min_outputs = 2;
    sig.max_outputs = 2;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& a = inputs[0];
        const auto& b = inputs[1];
        const auto& out = outputs[0];
        const auto& up_out = outputs[1];

        // Check matmul dims
        if (auto err = validators::check_matmul_dims(a, b, up_out, attrs)) {
            return err;
        }

        // up_out should have N=2*D where out has N=D
        if (!up_out.empty() && !out.empty()) {
            long up_N = up_out.back();
            long out_N = out.back();
            if (up_N != 2 * out_N) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "matmul_swiglu: up_out last dim (" << up_N << ") must be 2x out last dim (" << out_N << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
        }

        // Check batch dimensions match
        if (a.size() > 2 && out.size() > 2) {
            for (size_t i = 0; i < a.size() - 2 && i < out.size() - 2; ++i) {
                if (a[i] != out[i]) {
                    ShapeValidationError err;
                    std::ostringstream oss;
                    oss << "matmul_swiglu: batch dim mismatch at [" << i << "]: " << a[i] << " != " << out[i];
                    err.message = oss.str();
                    return std::make_optional(err);
                }
            }
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// MatmulSwiGLUBackward
// ------------------------------------------------------------------------
const int _matmul_swiglu_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "matmul_swiglu_backward";
    sig.min_inputs = 3;
    sig.max_inputs = 4;
    sig.min_outputs = 2;
    sig.max_outputs = 2;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& ln2 = inputs[1];
        const auto& weight = inputs[2];
        const auto& mlp_up = inputs[3];
        const auto& d_inp = outputs[0];
        const auto& d_weight = outputs[1];

        // Check mlp_up last dim is 2x d_out last dim
        if (!mlp_up.empty() && !d_out.empty()) {
            long mlp_up_dim = mlp_up.back();
            long d_out_dim = d_out.back();
            if (mlp_up_dim != 2 * d_out_dim) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "matmul_swiglu_backward: mlp_up last dim (" << mlp_up_dim << ") must be 2x d_out last dim ("
                    << d_out_dim << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
        }

        // d_inp should match ln2 (activation input)
        if (auto err = validators::check_same_numel(d_inp, ln2, "d_inp", "ln2", "matmul_swiglu_backward")) {
            return err;
        }

        // d_weight should match weight shape
        if (auto err = validators::check_same_numel(d_weight, weight, "d_weight", "weight", "matmul_swiglu_backward")) {
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
