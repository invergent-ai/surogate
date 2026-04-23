// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Each REGISTER_OP line wires a CompiledOpType to its forward and/or
// backward dispatch wrapper. The wrappers are thin trampolines around
// CompiledExecutor::dispatch_* member functions; they give the registry
// a uniform (executor, op, hook) signature regardless of whether the
// underlying dispatch needs a hook or not.
//
// Phase 2b/2c can redistribute these REGISTER_OP calls into the
// per-op files under runtime/ops/ once autodiff/shape rules live there
// too. Keeping them here for 2a minimizes churn per op file.

#include "runtime/executor/compiled_ops.h"
#include "runtime/executor/op_registry.h"
#include "runtime/core/backward_hooks.h"
#include "runtime/core/forward_hooks.h"

namespace dsl {
namespace {

// ============================================================================
// Wrapper macros
// ============================================================================
//
// Wrappers come in four flavors depending on whether the dispatch takes
// a hook and whether the hook is forward or backward. Defining them as
// one-liners via macros keeps this file readable despite having ~90
// wrappers.

// Forward dispatch, no hook: e.g. dispatch_embedding(op).
#define FWD_WRAP(wrapper_name, method)                                                      \
    void wrapper_name(CompiledExecutor& exec, const CompiledOp& op, const void* /*hook*/) { \
        exec.method(op);                                                                    \
    }

// Forward dispatch that takes a ForwardHook*.
#define FWD_WRAP_HOOK(wrapper_name, method)                                             \
    void wrapper_name(CompiledExecutor& exec, const CompiledOp& op, const void* hook) { \
        exec.method(op, static_cast<const modules::ForwardHook*>(hook));                \
    }

// Backward dispatch, no hook: e.g. dispatch_rmsnorm_backward(op).
#define BWD_WRAP(wrapper_name, method)                                                      \
    void wrapper_name(CompiledExecutor& exec, const CompiledOp& op, const void* /*hook*/) { \
        exec.method(op);                                                                    \
    }

// Backward dispatch that takes a BackwardHook*.
#define BWD_WRAP_HOOK(wrapper_name, method)                                             \
    void wrapper_name(CompiledExecutor& exec, const CompiledOp& op, const void* hook) { \
        exec.method(op, static_cast<const modules::BackwardHook*>(hook));               \
    }

// ============================================================================
// Forward wrappers
// ============================================================================

FWD_WRAP(fwd_embedding, dispatch_embedding)
FWD_WRAP(fwd_zeros, dispatch_zeros)
FWD_WRAP(fwd_ones, dispatch_ones)
FWD_WRAP(fwd_fused_residual_rmsnorm, dispatch_fused_residual_rmsnorm)
FWD_WRAP(fwd_rmsnorm, dispatch_rmsnorm)
FWD_WRAP(fwd_layernorm, dispatch_layernorm)
FWD_WRAP(fwd_view, dispatch_view)
FWD_WRAP(fwd_transpose, dispatch_transpose)
FWD_WRAP(fwd_split, dispatch_split)
FWD_WRAP(fwd_narrow, dispatch_narrow)
FWD_WRAP(fwd_concat, dispatch_concat)
FWD_WRAP(fwd_add, dispatch_add)
FWD_WRAP_HOOK(fwd_matmul, dispatch_matmul)
FWD_WRAP(fwd_bias_add, dispatch_bias_add)
FWD_WRAP(fwd_swiglu, dispatch_swiglu)
FWD_WRAP(fwd_gelu_glu, dispatch_gelu_glu)
FWD_WRAP(fwd_gpt_oss_moe_act, dispatch_gpt_oss_moe_act)
FWD_WRAP(fwd_silu, dispatch_silu)
FWD_WRAP(fwd_gelu, dispatch_gelu)
FWD_WRAP(fwd_relu2, dispatch_relu2)
FWD_WRAP(fwd_mul, dispatch_mul)
FWD_WRAP(fwd_scale, dispatch_scale)
FWD_WRAP(fwd_mask_scatter, dispatch_mask_scatter)
FWD_WRAP(fwd_deepstack_inject, dispatch_deepstack_inject)
FWD_WRAP_HOOK(fwd_matmul_swiglu, dispatch_matmul_swiglu)
FWD_WRAP(fwd_qkv_qk_norm, dispatch_qkv_qk_norm)
FWD_WRAP(fwd_qkv_qk_norm_rope, dispatch_qkv_qk_norm_rope)
FWD_WRAP(fwd_mrope, dispatch_mrope)
FWD_WRAP(fwd_rope, dispatch_rope)
FWD_WRAP(fwd_flash_attention, dispatch_flash_attention)
FWD_WRAP(fwd_cross_entropy_loss, dispatch_cross_entropy_loss)
FWD_WRAP(fwd_fused_lm_head_loss, dispatch_fused_lm_head_loss)
FWD_WRAP(fwd_moe_softmax, dispatch_moe_softmax)
FWD_WRAP(fwd_moe_sigmoid, dispatch_moe_sigmoid)
FWD_WRAP(fwd_moe_topk, dispatch_moe_topk)
FWD_WRAP(fwd_moe_permute, dispatch_moe_permute)
FWD_WRAP(fwd_moe_grouped_gemm, dispatch_moe_grouped_gemm)
FWD_WRAP(fwd_moe_grouped_gemm_gate_up, dispatch_moe_grouped_gemm_gate_up)
FWD_WRAP(fwd_moe_grouped_gemm_down, dispatch_moe_grouped_gemm_down)
FWD_WRAP(fwd_moe_unpermute, dispatch_moe_unpermute)
FWD_WRAP(fwd_moe_expert_bias_add, dispatch_moe_expert_bias_add)
FWD_WRAP(fwd_ep_dispatch, dispatch_ep_dispatch)
FWD_WRAP(fwd_ep_combine, dispatch_ep_combine)
FWD_WRAP(fwd_mamba_split_proj, dispatch_mamba_split_proj)
FWD_WRAP(fwd_mamba_conv1d, dispatch_mamba_conv1d)
FWD_WRAP(fwd_mamba_split_conv_out, dispatch_mamba_split_conv_out)
FWD_WRAP(fwd_mamba_ssm_scan, dispatch_mamba_ssm_scan)
FWD_WRAP(fwd_mamba_gated_rmsnorm, dispatch_mamba_gated_rmsnorm)
FWD_WRAP_HOOK(fwd_mamba_out_proj, dispatch_mamba_out_proj)
FWD_WRAP(fwd_chunk_gated_delta_rule, dispatch_chunk_gated_delta_rule)
FWD_WRAP(fwd_qwen3_5_decay, dispatch_qwen3_5_decay)
FWD_WRAP(fwd_repeat_interleave_heads, dispatch_repeat_interleave_heads)

// ============================================================================
// Backward wrappers
// ============================================================================

BWD_WRAP(bwd_view, dispatch_view_backward)
BWD_WRAP(bwd_add, dispatch_add_backward)
BWD_WRAP_HOOK(bwd_matmul, dispatch_matmul_backward)
BWD_WRAP(bwd_bias_add, dispatch_bias_add_backward)
BWD_WRAP(bwd_swiglu, dispatch_swiglu_backward)
BWD_WRAP(bwd_gelu_glu, dispatch_gelu_glu_backward)
BWD_WRAP(bwd_gpt_oss_moe_act, dispatch_gpt_oss_moe_act_backward)
BWD_WRAP(bwd_silu, dispatch_silu_backward)
BWD_WRAP(bwd_gelu, dispatch_gelu_backward)
BWD_WRAP(bwd_relu2, dispatch_relu2_backward)
BWD_WRAP(bwd_mul, dispatch_mul_backward)
BWD_WRAP(bwd_scale, dispatch_scale_backward)
BWD_WRAP(bwd_narrow, dispatch_narrow_backward)
BWD_WRAP(bwd_mask_scatter, dispatch_mask_scatter_backward)
BWD_WRAP(bwd_deepstack_inject, dispatch_deepstack_inject_backward)
BWD_WRAP_HOOK(bwd_matmul_swiglu, dispatch_matmul_swiglu_backward)
BWD_WRAP(bwd_qkv_qk_norm, dispatch_qkv_qk_norm_backward)
BWD_WRAP(bwd_rope, dispatch_rope_backward)
BWD_WRAP(bwd_qkv_qk_norm_rope, dispatch_qkv_qk_norm_rope_backward)
BWD_WRAP(bwd_mrope, dispatch_mrope_backward)
BWD_WRAP(bwd_flash_attention, dispatch_flash_attention_backward)
BWD_WRAP(bwd_zeros, dispatch_zeros_backward)
BWD_WRAP(bwd_fused_residual_rmsnorm, dispatch_fused_residual_rmsnorm_backward)
BWD_WRAP(bwd_rmsnorm, dispatch_rmsnorm_backward)
BWD_WRAP(bwd_layernorm, dispatch_layernorm_backward)
BWD_WRAP(bwd_embedding, dispatch_embedding_backward)
BWD_WRAP(bwd_cross_entropy_loss, dispatch_cross_entropy_loss_backward)
BWD_WRAP(bwd_fused_lm_head_loss, dispatch_fused_lm_head_loss_backward)
BWD_WRAP(bwd_moe_softmax, dispatch_moe_softmax_backward)
BWD_WRAP(bwd_moe_sigmoid, dispatch_moe_sigmoid_backward)
BWD_WRAP(bwd_moe_topk, dispatch_moe_topk_backward)
BWD_WRAP(bwd_moe_permute, dispatch_moe_permute_backward)
BWD_WRAP(bwd_moe_grouped_gemm, dispatch_moe_grouped_gemm_backward)
BWD_WRAP(bwd_moe_grouped_gemm_gate_up, dispatch_moe_grouped_gemm_gate_up_backward)
BWD_WRAP(bwd_moe_grouped_gemm_down, dispatch_moe_grouped_gemm_down_backward)
BWD_WRAP(bwd_moe_unpermute, dispatch_moe_unpermute_backward)
BWD_WRAP(bwd_moe_expert_bias_add, dispatch_moe_expert_bias_add_backward)
BWD_WRAP(bwd_ep_dispatch, dispatch_ep_dispatch_backward)
BWD_WRAP(bwd_ep_combine, dispatch_ep_combine_backward)
BWD_WRAP(bwd_mamba_split_proj, dispatch_mamba_split_proj_backward)
BWD_WRAP(bwd_mamba_conv1d, dispatch_mamba_conv1d_backward)
BWD_WRAP(bwd_mamba_split_conv_out, dispatch_mamba_split_conv_out_backward)
BWD_WRAP(bwd_mamba_ssm_scan, dispatch_mamba_ssm_scan_backward)
BWD_WRAP(bwd_mamba_gated_rmsnorm, dispatch_mamba_gated_rmsnorm_backward)
BWD_WRAP_HOOK(bwd_mamba_out_proj, dispatch_mamba_out_proj_backward)
BWD_WRAP(bwd_chunk_gated_delta_rule, dispatch_chunk_gated_delta_rule_backward)
BWD_WRAP(bwd_qwen3_5_decay, dispatch_qwen3_5_decay_backward)
BWD_WRAP(bwd_repeat_interleave_heads, dispatch_repeat_interleave_heads_backward)

#undef FWD_WRAP
#undef FWD_WRAP_HOOK
#undef BWD_WRAP
#undef BWD_WRAP_HOOK

}  // namespace
}  // namespace dsl

// ============================================================================
// Registrations
// ============================================================================
//
// Each REGISTER_OP line maps a CompiledOpType to its dispatch pair.
// Dual-use ops (those that appear in both forward and backward graphs,
// like View/Zeros/Add) register both a forward and a backward wrapper
// so the graph compiler can pick the right one via the is_backward
// flag. Ops that have no handler in one direction register nullptr;
// the executor will throw if it encounters them in that direction.
//
// The exact forward/backward pairings here mirror the old inline
// switches in execute_forward / execute_backward / replay_layer_forward
// one-for-one — any divergence will change runtime behavior.

// Forward-primary ops with a matching *Backward enum value.

// Self-dual ops that appear in both graphs and keep the same behavior
// (Transpose / Split / Narrow / Concat). For these the backward-graph
// entry just re-runs the forward dispatch.

// Dual-use ops with distinct forward / backward behavior:
//   View     fwd → dispatch_view,  bwd → dispatch_view_backward
//   Zeros    fwd → dispatch_zeros, bwd → dispatch_zeros (per execute_backward
//            comment: "may be a true no-op … or a split backward zero-fill")
//   Ones     fwd → dispatch_ones,  bwd → dispatch_zeros_backward (no-op)
//   Add      fwd → dispatch_add,   bwd → dispatch_add (gradient accumulation)

REGISTER_OP("embedding", Embedding, ::dsl::fwd_embedding, nullptr);
REGISTER_OP("embedding_backward", EmbeddingBackward, nullptr, ::dsl::bwd_embedding);
REGISTER_OP("zeros", Zeros, ::dsl::fwd_zeros, ::dsl::fwd_zeros);
REGISTER_OP("zeros_backward", ZerosBackward, nullptr, ::dsl::bwd_zeros);
REGISTER_OP("ones", Ones, ::dsl::fwd_ones, ::dsl::bwd_zeros);
REGISTER_OP("fused_residual_rmsnorm", FusedResidualRMSNorm, ::dsl::fwd_fused_residual_rmsnorm, nullptr);
REGISTER_OP("fused_residual_rmsnorm_backward",
            FusedResidualRMSNormBackward,
            nullptr,
            ::dsl::bwd_fused_residual_rmsnorm);
REGISTER_OP("rmsnorm", RMSNorm, ::dsl::fwd_rmsnorm, nullptr);
REGISTER_OP("rmsnorm_backward", RMSNormBackward, nullptr, ::dsl::bwd_rmsnorm);
REGISTER_OP("layernorm", LayerNorm, ::dsl::fwd_layernorm, nullptr);
REGISTER_OP("layernorm_backward", LayerNormBackward, nullptr, ::dsl::bwd_layernorm);
REGISTER_OP("view", View, ::dsl::fwd_view, ::dsl::bwd_view);
REGISTER_OP("view_backward", ViewBackward, nullptr, ::dsl::bwd_view);
REGISTER_OP("transpose", Transpose, ::dsl::fwd_transpose, ::dsl::fwd_transpose);
REGISTER_OP("split", Split, ::dsl::fwd_split, ::dsl::fwd_split);
REGISTER_OP("narrow", Narrow, ::dsl::fwd_narrow, ::dsl::fwd_narrow);
REGISTER_OP("narrow_backward", NarrowBackward, nullptr, ::dsl::bwd_narrow);
REGISTER_OP("concat", Concat, ::dsl::fwd_concat, ::dsl::fwd_concat);
REGISTER_OP("add", Add, ::dsl::fwd_add, ::dsl::fwd_add);
REGISTER_OP("add_backward", AddBackward, nullptr, ::dsl::bwd_add);
REGISTER_OP("matmul", Matmul, ::dsl::fwd_matmul, nullptr);
REGISTER_OP("matmul_bias", MatmulBias, ::dsl::fwd_matmul, nullptr);
REGISTER_OP("matmul_backward", MatmulBackward, nullptr, ::dsl::bwd_matmul);
REGISTER_OP("bias_add", BiasAdd, ::dsl::fwd_bias_add, nullptr);
REGISTER_OP("bias_add_backward", BiasAddBackward, nullptr, ::dsl::bwd_bias_add);
REGISTER_OP("swiglu", SwiGLU, ::dsl::fwd_swiglu, nullptr);
REGISTER_OP("swiglu_backward", SwiGLUBackward, nullptr, ::dsl::bwd_swiglu);
REGISTER_OP("gelu_glu", GeluGlu, ::dsl::fwd_gelu_glu, nullptr);
REGISTER_OP("gelu_glu_backward", GeluGluBackward, nullptr, ::dsl::bwd_gelu_glu);
REGISTER_OP("gpt_oss_moe_act", GptOssMoeAct, ::dsl::fwd_gpt_oss_moe_act, nullptr);
REGISTER_OP("gpt_oss_moe_act_backward", GptOssMoeActBackward, nullptr, ::dsl::bwd_gpt_oss_moe_act);
REGISTER_OP("silu", Silu, ::dsl::fwd_silu, nullptr);
REGISTER_OP("silu_backward", SiluBackward, nullptr, ::dsl::bwd_silu);
REGISTER_OP("gelu", Gelu, ::dsl::fwd_gelu, nullptr);
REGISTER_OP("gelu_backward", GeluBackward, nullptr, ::dsl::bwd_gelu);
REGISTER_OP("relu2", Relu2, ::dsl::fwd_relu2, nullptr);
REGISTER_OP("relu2_backward", Relu2Backward, nullptr, ::dsl::bwd_relu2);
REGISTER_OP("mul", Mul, ::dsl::fwd_mul, nullptr);
REGISTER_OP("mul_backward", MulBackward, nullptr, ::dsl::bwd_mul);
REGISTER_OP("scale", Scale, ::dsl::fwd_scale, nullptr);
REGISTER_OP("scale_backward", ScaleBackward, nullptr, ::dsl::bwd_scale);
REGISTER_OP("mask_scatter", MaskScatter, ::dsl::fwd_mask_scatter, nullptr);
REGISTER_OP("mask_scatter_backward", MaskScatterBackward, nullptr, ::dsl::bwd_mask_scatter);
REGISTER_OP("deepstack_inject", DeepstackInject, ::dsl::fwd_deepstack_inject, nullptr);
REGISTER_OP("deepstack_inject_backward", DeepstackInjectBackward, nullptr, ::dsl::bwd_deepstack_inject);
REGISTER_OP("matmul_swiglu", MatmulSwiGLU, ::dsl::fwd_matmul_swiglu, nullptr);
REGISTER_OP("matmul_swiglu_backward", MatmulSwiGLUBackward, nullptr, ::dsl::bwd_matmul_swiglu);
REGISTER_OP("qkv_qk_norm", QKVQKNorm, ::dsl::fwd_qkv_qk_norm, nullptr);
REGISTER_OP("qkv_qk_norm_backward", QKVQKNormBackward, nullptr, ::dsl::bwd_qkv_qk_norm);
REGISTER_OP("qkv_qk_norm_rope", QKVQKNormRoPE, ::dsl::fwd_qkv_qk_norm_rope, nullptr);
REGISTER_OP("qkv_qk_norm_rope_backward", QKVQKNormRoPEBackward, nullptr, ::dsl::bwd_qkv_qk_norm_rope);
REGISTER_OP("mrope", MRoPE, ::dsl::fwd_mrope, nullptr);
REGISTER_OP("mrope_backward", MRoPEBackward, nullptr, ::dsl::bwd_mrope);
REGISTER_OP("rope", RoPE, ::dsl::fwd_rope, nullptr);
REGISTER_OP("rope_backward", RoPEBackward, nullptr, ::dsl::bwd_rope);
REGISTER_OP("flash_attention", FlashAttention, ::dsl::fwd_flash_attention, nullptr);
REGISTER_OP("flash_attention_backward", FlashAttentionBackward, nullptr, ::dsl::bwd_flash_attention);
REGISTER_OP("cross_entropy_loss", CrossEntropyLoss, ::dsl::fwd_cross_entropy_loss, nullptr);
REGISTER_OP("cross_entropy_loss_backward", CrossEntropyLossBackward, nullptr, ::dsl::bwd_cross_entropy_loss);
REGISTER_OP("fused_lm_head_loss", FusedLMHeadLoss, ::dsl::fwd_fused_lm_head_loss, nullptr);
REGISTER_OP("fused_lm_head_loss_backward", FusedLMHeadLossBackward, nullptr, ::dsl::bwd_fused_lm_head_loss);

// MoE forward + backward
REGISTER_OP("moe_softmax", MoESoftmax, ::dsl::fwd_moe_softmax, nullptr);
REGISTER_OP("moe_softmax_backward", MoESoftmaxBackward, nullptr, ::dsl::bwd_moe_softmax);
REGISTER_OP("moe_sigmoid", MoESigmoid, ::dsl::fwd_moe_sigmoid, nullptr);
REGISTER_OP("moe_sigmoid_backward", MoESigmoidBackward, nullptr, ::dsl::bwd_moe_sigmoid);
REGISTER_OP("moe_topk", MoETopK, ::dsl::fwd_moe_topk, nullptr);
REGISTER_OP("moe_topk_backward", MoETopKBackward, nullptr, ::dsl::bwd_moe_topk);
REGISTER_OP("moe_permute", MoEPermute, ::dsl::fwd_moe_permute, nullptr);
REGISTER_OP("moe_permute_backward", MoEPermuteBackward, nullptr, ::dsl::bwd_moe_permute);
REGISTER_OP("moe_grouped_gemm", MoEGroupedGemm, ::dsl::fwd_moe_grouped_gemm, nullptr);
REGISTER_OP("moe_grouped_gemm_backward", MoEGroupedGemmBackward, nullptr, ::dsl::bwd_moe_grouped_gemm);
REGISTER_OP("moe_grouped_gemm_gate_up", MoEGroupedGemmGateUp, ::dsl::fwd_moe_grouped_gemm_gate_up, nullptr);
REGISTER_OP("moe_grouped_gemm_gate_up_backward",
            MoEGroupedGemmGateUpBackward,
            nullptr,
            ::dsl::bwd_moe_grouped_gemm_gate_up);
REGISTER_OP("moe_grouped_gemm_down", MoEGroupedGemmDown, ::dsl::fwd_moe_grouped_gemm_down, nullptr);
REGISTER_OP("moe_grouped_gemm_down_backward", MoEGroupedGemmDownBackward, nullptr, ::dsl::bwd_moe_grouped_gemm_down);
REGISTER_OP("moe_unpermute", MoEUnpermute, ::dsl::fwd_moe_unpermute, nullptr);
REGISTER_OP("moe_unpermute_backward", MoEUnpermuteBackward, nullptr, ::dsl::bwd_moe_unpermute);
REGISTER_OP("moe_expert_bias_add", MoEExpertBiasAdd, ::dsl::fwd_moe_expert_bias_add, nullptr);
REGISTER_OP("moe_expert_bias_add_backward", MoEExpertBiasAddBackward, nullptr, ::dsl::bwd_moe_expert_bias_add);

// Expert parallelism forward + backward
REGISTER_OP("ep_dispatch", EpDispatch, ::dsl::fwd_ep_dispatch, nullptr);
REGISTER_OP("ep_dispatch_backward", EpDispatchBackward, nullptr, ::dsl::bwd_ep_dispatch);
REGISTER_OP("ep_combine", EpCombine, ::dsl::fwd_ep_combine, nullptr);
REGISTER_OP("ep_combine_backward", EpCombineBackward, nullptr, ::dsl::bwd_ep_combine);

// Mamba/SSM forward + backward
REGISTER_OP("mamba_split_proj", MambaSplitProj, ::dsl::fwd_mamba_split_proj, nullptr);
REGISTER_OP("mamba_split_proj_backward", MambaSplitProjBackward, nullptr, ::dsl::bwd_mamba_split_proj);
REGISTER_OP("mamba_conv1d", MambaConv1d, ::dsl::fwd_mamba_conv1d, nullptr);
REGISTER_OP("mamba_conv1d_backward", MambaConv1dBackward, nullptr, ::dsl::bwd_mamba_conv1d);
REGISTER_OP("mamba_split_conv_out", MambaSplitConvOut, ::dsl::fwd_mamba_split_conv_out, nullptr);
REGISTER_OP("mamba_split_conv_out_backward", MambaSplitConvOutBackward, nullptr, ::dsl::bwd_mamba_split_conv_out);
REGISTER_OP("mamba_ssm_scan", MambaSsmScan, ::dsl::fwd_mamba_ssm_scan, nullptr);
REGISTER_OP("mamba_ssm_scan_backward", MambaSsmScanBackward, nullptr, ::dsl::bwd_mamba_ssm_scan);
REGISTER_OP("mamba_gated_rmsnorm", MambaGatedRMSNorm, ::dsl::fwd_mamba_gated_rmsnorm, nullptr);
REGISTER_OP("mamba_gated_rmsnorm_backward", MambaGatedRMSNormBackward, nullptr, ::dsl::bwd_mamba_gated_rmsnorm);
REGISTER_OP("mamba_out_proj", MambaOutProj, ::dsl::fwd_mamba_out_proj, nullptr);
REGISTER_OP("mamba_out_proj_backward", MambaOutProjBackward, nullptr, ::dsl::bwd_mamba_out_proj);

// Qwen3.5 gated delta rule forward + backward
REGISTER_OP("chunk_gated_delta_rule", ChunkGatedDeltaRule, ::dsl::fwd_chunk_gated_delta_rule, nullptr);
REGISTER_OP("chunk_gated_delta_rule_backward", ChunkGatedDeltaRuleBackward, nullptr, ::dsl::bwd_chunk_gated_delta_rule);
REGISTER_OP("qwen3_5_decay", Qwen3_5Decay, ::dsl::fwd_qwen3_5_decay, nullptr);
REGISTER_OP("qwen3_5_decay_backward", Qwen3_5DecayBackward, nullptr, ::dsl::bwd_qwen3_5_decay);
REGISTER_OP("repeat_interleave_heads", RepeatInterleaveHeads, ::dsl::fwd_repeat_interleave_heads, nullptr);
REGISTER_OP("repeat_interleave_heads_backward",
            RepeatInterleaveHeadsBackward,
            nullptr,
            ::dsl::bwd_repeat_interleave_heads);
