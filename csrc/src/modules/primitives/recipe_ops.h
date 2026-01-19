// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Recipe-driven primitive ops used by modular transformer blocks.

#ifndef SUROGATE_SRC_MODULES_PRIMITIVES_RECIPE_OPS_H
#define SUROGATE_SRC_MODULES_PRIMITIVES_RECIPE_OPS_H

#include <optional>

#include "modules/matmul_context.h"
#include "modules/module_concept.h"
#include "modules/run_state_types.h"
#include "modules/fp8_run_state.h"
#include "modules/fp4_run_state.h"
#include "modules/model_config.h"
#include "modules/weights/weight_manager_types.h"
#include "recipes/recipe.h"
#include "kernels/kernels.h"
#include "utilities/utils.h"

namespace modules {

// Forward declaration to avoid including weight_manager.h here.
template<typename Block>
class ModularWeightManager;

namespace recipe_ops {

// ============================================================================
// Quantization views
// ============================================================================

struct ForwardQuantView {
    FP8ForwardQuantActivations* fp8 = nullptr;
    FP4ForwardQuantActivations* fp4 = nullptr;
    SimplifiedLayerQuantActivations* qa = nullptr;

    Tensor* ln1_inp_quant() const { return fp8 ? &fp8->ln1 : nullptr; }
    Tensor* ln2_inp_quant() const { return fp8 ? &fp8->ln2 : nullptr; }
    Tensor* att_inp_quant() const { return fp8 ? &fp8->att : nullptr; }
    Tensor* swiglu_inp_quant() const { return fp8 ? &fp8->swiglu : nullptr; }

    float* ln1_abs_max() const {
        if (fp8) return fp8->ln1.abs_max();
        if (fp4) return fp4->ln1_global_amax;
        if (qa && qa->ln1.Data) return qa->ln1.abs_max();
        return nullptr;
    }

    float* ln2_abs_max() const {
        if (fp8) return fp8->ln2.abs_max();
        if (fp4) return fp4->ln2_global_amax;
        if (qa && qa->ln2.Data) return qa->ln2.abs_max();
        return nullptr;
    }

    float* att_abs_max() const {
        if (fp8) return fp8->att.abs_max();
        if (fp4) return fp4->att_global_amax;
        if (qa && qa->att.Data) return qa->att.abs_max();
        return nullptr;
    }

    float* swiglu_abs_max() const {
        if (fp8) return fp8->swiglu.abs_max();
        if (fp4) return fp4->swiglu_global_amax;
        if (qa && qa->swiglu.Data) return qa->swiglu.abs_max();
        return nullptr;
    }
};

struct BackwardQuantView {
    SimplifiedLayerQuantActivations* qa = nullptr;
    SimplifiedQuantGradients* qg = nullptr;

    Tensor* inp_ln1() const { return (qa && qa->ln1.Data) ? &qa->ln1 : nullptr; }
    Tensor* inp_ln2() const { return (qa && qa->ln2.Data) ? &qa->ln2 : nullptr; }
    Tensor* inp_att() const { return (qa && qa->att.Data) ? &qa->att : nullptr; }
    Tensor* inp_swiglu() const { return (qa && qa->swiglu.Data) ? &qa->swiglu : nullptr; }

    Tensor* dout_d_res_ffn() const { return (qg && qg->d_res_ffn.Data) ? &qg->d_res_ffn : nullptr; }
    Tensor* dout_d_res_att() const { return (qg && qg->d_res_att.Data) ? &qg->d_res_att : nullptr; }
    Tensor* dout_d_mlp_up() const { return (qg && qg->d_mlp_up.Data) ? &qg->d_mlp_up : nullptr; }
    Tensor* dout_d_qkv() const { return (qg && qg->d_qkv.Data) ? &qg->d_qkv : nullptr; }

    float* d_res_ffn_abs_max() const { return qg ? qg->d_res_ffn.abs_max() : nullptr; }
    float* d_res_att_abs_max() const { return qg ? qg->d_res_att.abs_max() : nullptr; }
    float* d_mlp_up_abs_max() const { return qg ? qg->d_mlp_up.abs_max() : nullptr; }
    float* d_qkv_abs_max() const { return qg ? qg->d_qkv.abs_max() : nullptr; }
};

// ============================================================================
// Cached weights (FP8/FP4)
// ============================================================================

struct CachedWeights {
    const Tensor* fp8_weight = nullptr;
    const Tensor* fp4_data = nullptr;
    const Tensor* fp4_scales = nullptr;
    const float* fp4_amax = nullptr;
};

inline int fp4_amax_offset(MatmulOp op) {
    switch (op) {
        case MatmulOp::QKV: return 0;
        case MatmulOp::AttnOut: return 1;
        case MatmulOp::MLPUp: return 2;
        case MatmulOp::MLPDown: return 3;
        default: return -1;
    }
}

template<typename Block>
inline CachedWeights forward_cached_weights(ModularWeightManager<Block>* wm, MatmulOp op, bool allow_quant) {
    CachedWeights cw{};
    if (!wm || !allow_quant) return cw;

    if (wm->has_fp8_forward_cache()) {
        const auto& fp8_cache = wm->fp8_weight_cache();
        switch (op) {
            case MatmulOp::QKV: cw.fp8_weight = &fp8_cache.qkv_weight; break;
            case MatmulOp::AttnOut: cw.fp8_weight = &fp8_cache.o_weight; break;
            case MatmulOp::MLPUp: cw.fp8_weight = &fp8_cache.mlp_up_weight; break;
            case MatmulOp::MLPDown: cw.fp8_weight = &fp8_cache.mlp_down_weight; break;
            default: break;
        }
    }

    if (wm->has_fp4_forward_cache()) {
        const auto& fp4_cache = wm->fp4_weight_cache();
        switch (op) {
            case MatmulOp::QKV:
                cw.fp4_data = &fp4_cache.qkv_weight.data;
                cw.fp4_scales = &fp4_cache.qkv_weight.scales;
                break;
            case MatmulOp::AttnOut:
                cw.fp4_data = &fp4_cache.o_weight.data;
                cw.fp4_scales = &fp4_cache.o_weight.scales;
                break;
            case MatmulOp::MLPUp:
                cw.fp4_data = &fp4_cache.mlp_up_weight.data;
                cw.fp4_scales = &fp4_cache.mlp_up_weight.scales;
                break;
            case MatmulOp::MLPDown:
                cw.fp4_data = &fp4_cache.mlp_down_weight.data;
                cw.fp4_scales = &fp4_cache.mlp_down_weight.scales;
                break;
            default:
                break;
        }
        const int offset = fp4_amax_offset(op);
        if (offset >= 0) {
            cw.fp4_amax = wm->fp4_weight_amax().template get<float>() + offset;
        }
    }

    return cw;
}

template<typename Block>
inline CachedWeights dgrad_cached_weights(ModularWeightManager<Block>* wm, MatmulOp op, bool allow_quant) {
    CachedWeights cw{};
    if (!wm || !allow_quant) return cw;
    if (!wm->has_fp4_dgrad_cache()) return cw;

    const auto& fp4_t = wm->fp4_weight_cache_transposed();
    switch (op) {
        case MatmulOp::QKV:
            cw.fp4_data = &fp4_t.qkv_weight.data;
            cw.fp4_scales = &fp4_t.qkv_weight.scales;
            break;
        case MatmulOp::AttnOut:
            cw.fp4_data = &fp4_t.o_weight.data;
            cw.fp4_scales = &fp4_t.o_weight.scales;
            break;
        case MatmulOp::MLPUp:
            cw.fp4_data = &fp4_t.mlp_up_weight.data;
            cw.fp4_scales = &fp4_t.mlp_up_weight.scales;
            break;
        case MatmulOp::MLPDown:
            cw.fp4_data = &fp4_t.mlp_down_weight.data;
            cw.fp4_scales = &fp4_t.mlp_down_weight.scales;
            break;
        default:
            break;
    }
    const int offset = fp4_amax_offset(op);
    if (offset >= 0) {
        cw.fp4_amax = wm->fp4_weight_amax_transposed().template get<float>() + offset;
    }
    return cw;
}

// ============================================================================
// Matmul + SwiGLU helpers
// ============================================================================

template<typename RunState>
inline void forward_matmul(
    const ::recipes::Recipe& recipe,
    RunState& rs,
    Tensor& out,
    Tensor& inp,
    Tensor& weight,
    Tensor* bias,
    int B, int T, int C_in, int C_out,
    int layer_idx,
    MatmulOp op,
    Tensor* inp_quant,
    int delayed_quantizer_idx,
    const CachedWeights& cache,
    cudaStream_t stream,
    bool allow_quant)
{
    MatmulContext ctx;
    ctx.out = &out;
    ctx.inp = &inp;
    ctx.weight = &weight;
    ctx.bias = bias;
    ctx.B = B;
    ctx.T = T;
    ctx.C_in = C_in;
    ctx.C_out = C_out;
    ctx.run_state = &rs;
    ctx.stream = stream;
    ctx.layer_idx = layer_idx;
    ctx.op = op;
    ctx.inp_quant = inp_quant;
    ctx.cached_weight = allow_quant ? cache.fp8_weight : nullptr;
    ctx.delayed_quantizer_idx = allow_quant ? delayed_quantizer_idx : -1;
    ctx.cached_fp4_data = allow_quant ? cache.fp4_data : nullptr;
    ctx.cached_fp4_scales = allow_quant ? cache.fp4_scales : nullptr;
    ctx.cached_fp4_amax = allow_quant ? cache.fp4_amax : nullptr;
    ctx.allow_fp4 = allow_quant;
    ctx.allow_fp8 = allow_quant;

    recipe.forward_matmul(ctx);
}

template<typename RunState>
inline void backward_matmul(
    const ::recipes::Recipe& recipe,
    RunState& rs,
    Tensor& dinp,
    Tensor& dweight,
    Tensor* dbias,
    Tensor& dout,
    Tensor& inp,
    Tensor& weight,
    int B, int T, int C_in, int C_out,
    int layer_idx,
    MatmulOp op,
    bool accumulate,
    bool skip_weight_grad,
    Tensor* inp_quant,
    Tensor* dout_quant,
    Tensor* bias_buffer,
    const CachedWeights& dgrad_cache,
    cudaStream_t stream,
    bool allow_quant)
{
    MatmulContext ctx;
    ctx.dinp = &dinp;
    ctx.dweight = &dweight;
    ctx.dbias = dbias;
    ctx.dout = &dout;
    ctx.inp = &inp;
    ctx.weight = &weight;
    ctx.B = B;
    ctx.T = T;
    ctx.C_in = C_in;
    ctx.C_out = C_out;
    ctx.run_state = &rs;
    ctx.stream = stream;
    ctx.layer_idx = layer_idx;
    ctx.op = op;
    ctx.accumulate = accumulate;
    ctx.skip_weight_grad = skip_weight_grad;
    ctx.inp_quant = inp_quant;
    ctx.dout_quant = dout_quant;
    ctx.bias_buffer = bias_buffer;
    ctx.allow_fp4 = allow_quant;
    ctx.allow_fp8 = allow_quant;
    ctx.cached_fp4_data = allow_quant ? dgrad_cache.fp4_data : nullptr;
    ctx.cached_fp4_scales = allow_quant ? dgrad_cache.fp4_scales : nullptr;
    ctx.cached_fp4_amax = allow_quant ? dgrad_cache.fp4_amax : nullptr;

    recipe.backward_matmul(ctx);
}

inline void swiglu_forward(
    const ::recipes::Recipe& recipe,
    Tensor& out,
    Tensor* scale_out,
    const Tensor& inp,
    float* abs_max_out,
    int B, int T, int D,
    cudaStream_t stream)
{
    SwiGLUContext ctx;
    ctx.out = &out;
    ctx.scale_out = scale_out;
    ctx.inp = &inp;
    ctx.abs_max_out = abs_max_out;
    ctx.B = B;
    ctx.T = T;
    ctx.D = D;
    ctx.stream = stream;
    recipe.swiglu_forward(ctx);
}

inline void swiglu_backward(
    const ::recipes::Recipe& recipe,
    Tensor& dinp,
    const Tensor& dout,
    const Tensor& inp,
    const Tensor* scale,
    float* abs_max_out,
    int B, int T, int D,
    cudaStream_t stream)
{
    SwiGLUContext ctx;
    ctx.dinp = &dinp;
    ctx.dout = &dout;
    ctx.inp = &inp;
    ctx.scale = scale;
    ctx.abs_max_out = abs_max_out;
    ctx.B = B;
    ctx.T = T;
    ctx.D = D;
    ctx.stream = stream;
    recipe.swiglu_backward(ctx);
}

inline void record_abs_max(float* abs_max_ptr, Tensor& t, const cudaDeviceProp& dp, cudaStream_t stream) {
    if (!abs_max_ptr) return;
    ::abs_max(abs_max_ptr, t, (long)t.nelem(), dp, stream);
}

inline void attention_forward(
    Tensor& att,
    Tensor& lse,
    Tensor& qkv,
    Tensor& workspace,
    cudnnHandle_t handle,
    int B, int T, int Hq, int Hkv, int Hs,
    cudaStream_t stream)
{
    attention_forward_cudnn(att, lse, qkv, workspace, handle, B, T, Hq, Hkv, Hs, stream);
}

inline void attention_backward(
    Tensor& d_qkv,
    Tensor& lse,
    Tensor& att,
    Tensor& d_att,
    Tensor& qkv,
    Tensor& workspace,
    cudnnHandle_t handle,
    int B, int T, int Hq, int Hkv, int Hs,
    cudaStream_t stream)
{
    attention_backward_cudnn(d_qkv, lse, att, d_att, qkv, workspace, handle, B, T, Hq, Hkv, Hs, stream);
}

// ============================================================================
// RoPE helpers
// ============================================================================

inline void rope_forward_dispatch(
    Tensor& out,
    Tensor& inp,
    const Tensor& freq_cis,
    const int* pos_ids,
    const ModelConfig& config,
    const ModelOptions& options,
    int B, int T, int Hq, int Hkv, int Hs,
    cudaStream_t stream)
{
    const int rotary_dim = config.Rope.rotary_dim(Hs);
    if (rotary_dim == 0) {
        if (out.Data != inp.Data) {
            CUDA_CHECK(cudaMemcpyAsync(out.Data, inp.Data, inp.bytes(), cudaMemcpyDeviceToDevice, stream));
        }
        return;
    }

    if (options.use_fused_rope && rotary_dim == Hs) {
        rope_fused_forward(out, inp, pos_ids, nullptr, config.RopeTheta, B, T, Hq, Hkv, Hs, stream);
    } else if (rotary_dim == Hs) {
        rope_forward(out, inp, freq_cis, pos_ids, nullptr, B, T, Hq, Hkv, Hs, stream);
    } else {
        rope_forward(out, inp, freq_cis, pos_ids, nullptr, B, T, Hq, Hkv, Hs, rotary_dim, stream);
    }
}

inline void rope_backward_dispatch(
    Tensor& out,
    Tensor& inp,
    const Tensor& freq_cis,
    const int* pos_ids,
    const ModelConfig& config,
    const ModelOptions& options,
    float* abs_max,
    int B, int T, int Hq, int Hkv, int Hs,
    cudaStream_t stream)
{
    const int rotary_dim = config.Rope.rotary_dim(Hs);
    if (rotary_dim == 0) {
        if (out.Data != inp.Data) {
            CUDA_CHECK(cudaMemcpyAsync(out.Data, inp.Data, inp.bytes(), cudaMemcpyDeviceToDevice, stream));
        }
        return;
    }

    if (options.use_fused_rope && rotary_dim == Hs) {
        rope_fused_backward(out, inp, pos_ids, abs_max, config.RopeTheta, B, T, Hq, Hkv, Hs, stream);
    } else if (rotary_dim == Hs) {
        rope_backward(out, inp, freq_cis, pos_ids, abs_max, B, T, Hq, Hkv, Hs, stream);
    } else {
        rope_backward(out, inp, freq_cis, pos_ids, abs_max, B, T, Hq, Hkv, Hs, rotary_dim, stream);
    }
}

// ============================================================================
// QK norm helpers (optional)
// ============================================================================

template<typename AttentionWeights>
inline bool has_qk_norm(const AttentionWeights& w) {
    if constexpr (has_qk_norm_weights<AttentionWeights>::value) {
        return w.q_norm_weight.has_value() && w.k_norm_weight.has_value();
    }
    return false;
}

template<typename AttentionWeights>
inline void qk_norm_forward(
    Tensor& qkv,
    Tensor& q_rstd,
    Tensor& k_rstd,
    const AttentionWeights& w,
    float eps,
    int B, int T, int qkv_channels,
    int Hq, int Hkv, int Hs,
    cudaStream_t stream)
{
    if constexpr (has_qk_norm_weights<AttentionWeights>::value) {
        if (!has_qk_norm(w)) return;
        const int q_rows = Hq * Hs;
        qkv_head_rmsnorm_forward(qkv, q_rstd, w.q_norm_weight.value(),
                                 eps, B, T, qkv_channels,
                                 Hq, Hs, 0, stream);
        qkv_head_rmsnorm_forward(qkv, k_rstd, w.k_norm_weight.value(),
                                 eps, B, T, qkv_channels,
                                 Hkv, Hs, q_rows, stream);
    }
}

template<typename AttentionWeights, typename AttentionGrads>
inline void qk_norm_backward(
    Tensor& d_qkv,
    Tensor& qkv,
    Tensor& q_rstd,
    Tensor& k_rstd,
    const AttentionWeights& w,
    AttentionGrads& g,
    float eps,
    int B, int T, int qkv_channels,
    int Hq, int Hkv, int Hs,
    bool accumulate,
    bool skip_weight_grad,
    cudaStream_t stream)
{
    if constexpr (has_qk_norm_weights<AttentionWeights>::value) {
        if (!has_qk_norm(w)) return;
        const int q_rows = Hq * Hs;

        // Weight gradients
        if (!skip_weight_grad) {
            if constexpr (requires { g.d_q_norm_weight; g.d_k_norm_weight; }) {
                if (g.d_q_norm_weight.has_value()) {
                    qkv_head_rmsnorm_backward_dweight(
                        g.d_q_norm_weight.value(), d_qkv, qkv, w.q_norm_weight.value(),
                        B, T, qkv_channels, Hq, Hs, 0, accumulate, stream);
                }
                if (g.d_k_norm_weight.has_value()) {
                    qkv_head_rmsnorm_backward_dweight(
                        g.d_k_norm_weight.value(), d_qkv, qkv, w.k_norm_weight.value(),
                        B, T, qkv_channels, Hkv, Hs, q_rows, accumulate, stream);
                }
            }
        }

        // dx (in-place)
        qkv_head_rmsnorm_backward_dx(
            d_qkv, qkv, w.q_norm_weight.value(), q_rstd,
            B, T, qkv_channels, Hq, Hs, 0, stream);
        qkv_head_rmsnorm_backward_dx(
            d_qkv, qkv, w.k_norm_weight.value(), k_rstd,
            B, T, qkv_channels, Hkv, Hs, q_rows, stream);
    }
}

// ============================================================================
// MoE router helpers (forward/backward kernels)
// ============================================================================

inline void moe_router_logits(
    Tensor& logits,
    const Tensor& gate,
    const Tensor& input,
    const std::optional<Tensor>& bias,
    cublasLtHandle_t handle,
    Tensor& workspace,
    int BT, int num_experts, int hidden,
    cudaStream_t stream)
{
    matmul(logits, gate, input, bias,
           nullptr, nullptr,
           handle, workspace,
           num_experts, BT, hidden, EMMTranspose::TN, false, stream);
}

inline void moe_router_softmax(
    Tensor& probs,
    const Tensor& logits,
    int BT, int num_experts,
    cudaStream_t stream)
{
    if (logits.DType == ETensorDType::BF16) {
        moe_softmax_forward(probs.get<nv_bfloat16>(), logits.get<nv_bfloat16>(), BT, num_experts, stream);
    } else {
        moe_softmax_forward(probs.get<float>(), logits.get<float>(), BT, num_experts, stream);
    }
}

inline void moe_router_topk(
    Tensor& expert_indices,
    Tensor& routing_weights,
    const Tensor& probs,
    int BT, int num_experts, int top_k,
    bool normalize_after,
    cudaStream_t stream)
{
    if (probs.DType == ETensorDType::BF16) {
        moe_topk_forward(expert_indices.get<int>(), routing_weights.get<nv_bfloat16>(),
                         probs.get<nv_bfloat16>(), BT, num_experts, top_k, normalize_after, stream);
    } else {
        moe_topk_forward(expert_indices.get<int>(), routing_weights.get<float>(),
                         probs.get<float>(), BT, num_experts, top_k, normalize_after, stream);
    }
}

inline void moe_router_counts(
    Tensor& expert_counts,
    const Tensor& expert_indices,
    int BT, int top_k, int num_experts,
    cudaStream_t stream)
{
    moe_compute_expert_counts(expert_counts.get<int>(), expert_indices.get<int>(),
                              BT, top_k, num_experts, stream);
}

inline void moe_router_offsets(
    Tensor& expert_offsets,
    const Tensor& expert_counts,
    int num_experts,
    cudaStream_t stream)
{
    moe_compute_expert_offsets(expert_offsets.get<int>(), expert_counts.get<int>(), num_experts, stream);
}

inline void moe_router_build_indices(
    Tensor& gather_indices,
    Tensor& scatter_indices,
    const Tensor& expert_indices,
    const Tensor& expert_offsets,
    Tensor& expert_positions,
    int BT, int top_k, int num_experts,
    cudaStream_t stream)
{
    moe_build_indices(gather_indices.get<int>(), scatter_indices.get<int>(),
                      expert_indices.get<int>(), expert_offsets.get<int>(), expert_positions.get<int>(),
                      BT, top_k, num_experts, stream);
}

inline void moe_router_aux_loss(
    float* d_aux_loss,
    const Tensor& router_probs,
    const Tensor& expert_indices,
    int BT, int num_experts, int top_k,
    float aux_loss_coef,
    cudaStream_t stream)
{
    moe_compute_aux_loss(d_aux_loss,
                         router_probs.get<float>(),
                         expert_indices.get<int>(),
                         BT, num_experts, top_k, aux_loss_coef, stream);
}

inline void moe_router_z_loss(
    float* d_z_loss,
    const Tensor& router_logits,
    int BT, int num_experts,
    float z_loss_coef,
    cudaStream_t stream)
{
    moe_router_z_loss_forward(d_z_loss, router_logits.get<float>(), BT, num_experts, z_loss_coef, stream);
}

} // namespace recipe_ops
} // namespace modules

#endif // SUROGATE_SRC_MODULES_PRIMITIVES_RECIPE_OPS_H
