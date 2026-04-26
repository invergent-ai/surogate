#include "runtime/executor/compiled_ops.h"

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
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_fused_lm_head_loss(const CompiledOp& op) {
    Tensor& xF_flat = resolve_tensor(op.inputs[0]);
    Tensor& weight = resolve_tensor(op.inputs[1]);
    Tensor& targets = resolve_tensor(op.inputs[2]);
    Tensor& loss = ensure_output_tensor(op.outputs[0]);

    const long BT = xF_flat.Sizes[0];
    const int V = static_cast<int>(weight.Sizes[0]);
    const int C = static_cast<int>(weight.Sizes[1]);
    const int P = V;

    const int lmhead_chunks = std::max(1, mOptions.LMHeadChunks);
    const long nano_batch_size = BT / static_cast<long>(lmhead_chunks);
    const int n_chunks = (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE;

    const bool need_lm_head =
        mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled());
    if (need_lm_head && mComm) {
        mWeightManager->gather_lm_head(*mComm, mRunState.MainStream);
    }

    // -----------------------------------------------------------------------
    // Log-prob mode: compute log P(target | context) for all BT tokens, skip loss.
    // Chunked to fit within the output buffer (sized for nano_batch_size rows).
    // -----------------------------------------------------------------------
    if (mLogprobsGpu) {
        const std::size_t xf_stride_lp = get_dtype_size(xF_flat.DType);
        const std::size_t tgt_stride_lp = get_dtype_size(targets.DType);

        for (int nano_step = 0; nano_step < lmhead_chunks; ++nano_step) {
            const long token_offset = static_cast<long>(nano_step) * nano_batch_size;

            // Slice xF_flat for this chunk.
            Tensor xF_slice = xF_flat;
            xF_slice.Data = static_cast<std::byte*>(xF_slice.Data) +
                            static_cast<std::size_t>(token_offset) * xf_stride_lp * static_cast<std::size_t>(C);
            xF_slice.Sizes[0] = nano_batch_size;
            xF_slice.Sizes[1] = C;
            xF_slice.Rank = 2;

            // Slice targets for this chunk.
            Tensor tgt_slice = targets;
            tgt_slice.Data =
                static_cast<std::byte*>(tgt_slice.Data) + static_cast<std::size_t>(token_offset) * tgt_stride_lp;
            tgt_slice.Sizes[0] = nano_batch_size;
            tgt_slice.Rank = 1;

            // Logits buffer fits nano_batch_size rows.
            Tensor logits = mRunState.non_block_activations().output;
            logits.Sizes[0] = nano_batch_size;
            logits.Sizes[1] = V;
            logits.Rank = 2;

            matmul(logits,
                   weight,
                   xF_slice,
                   std::nullopt,
                   nullptr,
                   nullptr,
                   mRunState.CublasLtHandle,
                   mRunState.CuBlasWorkspace,
                   V,
                   static_cast<int>(nano_batch_size),
                   C,
                   swap_transpose(EMMTranspose::NT),
                   false,
                   mRunState.MainStream);

            if (op.attrs.softcap > 0.0f) {
                softcap_logits(logits, op.attrs.softcap, static_cast<int>(nano_batch_size), V, mRunState.MainStream);
            }

            if (mInvTemperatureGpu) {
                const float* inv_t = mInvTemperatureGpu + token_offset;
                scale_logits_rows(logits, inv_t, static_cast<int>(nano_batch_size), V, P, mRunState.MainStream);
            }

            // Extract log-probs for this chunk into the correct offset of the output buffer.
            extract_logprobs(logits,
                             mLogprobsGpu + token_offset,
                             tgt_slice,
                             static_cast<int>(nano_batch_size),
                             V,
                             P,
                             mRunState.MainStream);
        }

        if (need_lm_head) {
            mWeightManager->release_lm_head(mRunState.MainStream);
        }
        return;
    }

    const std::size_t xf_stride = get_dtype_size(xF_flat.DType);
    const std::size_t tgt_stride = get_dtype_size(targets.DType);
    const std::size_t loss_stride = get_dtype_size(loss.DType);
    const std::size_t lse_stride = get_dtype_size(ETensorDType::FP32);
    const std::size_t chunk_lse_stride = lse_stride * static_cast<std::size_t>(n_chunks);

    for (int nano_step = 0; nano_step < lmhead_chunks; ++nano_step) {
        const long token_offset = static_cast<long>(nano_step) * nano_batch_size;

        Tensor xF_slice = xF_flat;
        xF_slice.Data = static_cast<std::byte*>(xF_slice.Data) +
                        static_cast<std::size_t>(token_offset) * xf_stride * static_cast<std::size_t>(C);
        xF_slice.Sizes[0] = nano_batch_size;
        xF_slice.Sizes[1] = C;
        xF_slice.Rank = 2;

        Tensor tgt_slice = targets;
        tgt_slice.Data = static_cast<std::byte*>(tgt_slice.Data) + static_cast<std::size_t>(token_offset) * tgt_stride;
        tgt_slice.Sizes[0] = nano_batch_size;
        tgt_slice.Rank = 1;

        Tensor loss_slice = loss;
        loss_slice.Data =
            static_cast<std::byte*>(loss_slice.Data) + static_cast<std::size_t>(token_offset) * loss_stride;
        loss_slice.Sizes[0] = nano_batch_size;
        loss_slice.Rank = 1;

        Tensor logits = mRunState.non_block_activations().output;
        logits.Sizes[0] = nano_batch_size;
        logits.Sizes[1] = V;
        logits.Rank = 2;

        matmul(logits,
               weight,
               xF_slice,
               std::nullopt,
               nullptr,
               nullptr,
               mRunState.CublasLtHandle,
               mRunState.CuBlasWorkspace,
               static_cast<int>(V),
               static_cast<int>(nano_batch_size),
               static_cast<int>(C),
               swap_transpose(EMMTranspose::NT),
               false,
               mRunState.MainStream);

        // Logit softcapping: softcap * tanh(logits / softcap)
        if (op.attrs.softcap > 0.0f) {
            softcap_logits(logits, op.attrs.softcap, static_cast<int>(nano_batch_size), V, mRunState.MainStream);
        }

        if (mInvTemperatureGpu) {
            const float* inv_t = mInvTemperatureGpu + token_offset;
            scale_logits_rows(logits, inv_t, static_cast<int>(nano_batch_size), V, P, mRunState.MainStream);
        }

        Tensor logsumexp_view{};
        Tensor* logsumexp = nullptr;
        if (mRunState.scratch().cross_entropy_logsumexp.Data) {
            logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
            logsumexp_view.Data =
                static_cast<std::byte*>(logsumexp_view.Data) + static_cast<std::size_t>(token_offset) * lse_stride;
            logsumexp_view.Sizes[0] = nano_batch_size;
            logsumexp_view.Rank = 1;
            logsumexp = &logsumexp_view;
        }

        if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
            if (!mRunState.scratch().cross_entropy_chunk_logsumexp.Data) {
                throw std::runtime_error("fused_lm_head_loss: chunk logsumexp buffer is not allocated");
            }
            Tensor chunk_lse = mRunState.scratch().cross_entropy_chunk_logsumexp;
            chunk_lse.Data =
                static_cast<std::byte*>(chunk_lse.Data) + static_cast<std::size_t>(token_offset) * chunk_lse_stride;
            chunk_lse.Sizes[0] = nano_batch_size;
            chunk_lse.Sizes[1] = n_chunks;
            chunk_lse.Rank = 2;

            chunked_cross_entropy_forward(logits,
                                          loss_slice,
                                          logsumexp,
                                          chunk_lse,
                                          tgt_slice,
                                          &mRunState.ValidTokenCount,
                                          op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                          static_cast<int>(nano_batch_size),
                                          V,
                                          P,
                                          n_chunks,
                                          mRunState.MainStream);
        } else {
            fused_cross_entropy_forward(logits,
                                        loss_slice,
                                        logsumexp,
                                        tgt_slice,
                                        &mRunState.ValidTokenCount,
                                        op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                        static_cast<int>(nano_batch_size),
                                        V,
                                        P,
                                        mRunState.MainStream);
        }
    }

    if (need_lm_head) {
        mWeightManager->release_lm_head(mRunState.MainStream);
    }
}

void CompiledExecutor::dispatch_fused_lm_head_loss_backward(const CompiledOp& op) {
    Tensor& d_loss = resolve_tensor(op.inputs[0]);
    Tensor& xF_flat = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);
    Tensor& targets = resolve_tensor(op.inputs[3]);

    Tensor* d_xF_ptr = nullptr;
    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        d_xF_ptr = &ensure_output_tensor(op.outputs[0]);
    }

    Tensor* d_weight_ptr = nullptr;
    bool d_weight_accumulate = false;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty() && mCurrentGraph) {
        if (auto weight_name = base_param_from_grad_kind(op.outputs[1].tensor_id, *mCurrentGraph)) {
            bool accum = false;
            Tensor* grad = mGrads.get_param_grad(*weight_name, accum);
            d_weight_accumulate = mAccumulateTensors.count(op.outputs[1].name) > 0;
            if (!d_weight_accumulate) {
                d_weight_accumulate = mAccumulateTensors.count("d_" + *weight_name) > 0;
            }
            if (grad && grad->Data) {
                d_weight_ptr = &ensure_output_tensor(op.outputs[1]);
            }
        }
    }

    // Use reduction="sum" semantics by default.
    // dloss = 1.0 means each valid token contributes equally to the gradient sum.
    // If token scaling is enabled, global_norm_sqrt applies 1/valid_token_count;
    // custom losses may pre-scale dloss instead.
    // Robustly seed d_loss even if the name has SSA suffixes or mapped to loss/losses.
    const std::string d_loss_name = strip_ssa_suffix(op.inputs[0].name);
    bool d_loss_seeded = false;
    const bool d_loss_like = starts_with(d_loss_name, "d_loss") || d_loss_name == "loss" || d_loss_name == "losses";
    if (op.inputs[0].slot == TensorSlot::DLoss || op.inputs[0].slot == TensorSlot::Losses || d_loss_like) {
        if (mCustomDLossGpu) {
            // GRPO mode: seed d_loss from externally-computed per-token gradients.
            // mCustomDLossGpu contains B*T float32 values = dL_GRPO/d(log_prob)[t].
            CUDA_CHECK(cudaMemcpyAsync(d_loss.Data,
                                       mCustomDLossGpu,
                                       static_cast<std::size_t>(d_loss.nelem()) * sizeof(float),
                                       cudaMemcpyDeviceToDevice,
                                       mRunState.MainStream));
        } else {
            fill_constant(d_loss, 1.0f, static_cast<std::size_t>(d_loss.nelem()), mRunState.MainStream);
        }
        d_loss_seeded = true;
    }

    const long BT = xF_flat.Sizes[0];
    const int V = static_cast<int>(weight.Sizes[0]);
    const int C = static_cast<int>(weight.Sizes[1]);
    const int P = V;

    const int lmhead_chunks = std::max(1, mOptions.LMHeadChunks);
    const long nano_batch_size = BT / static_cast<long>(lmhead_chunks);
    const bool need_lm_head = mWeightManager &&
                              (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled()) &&
                              (mOptions.LMHeadChunks > 1);
    if (need_lm_head && mComm) {
        mWeightManager->gather_lm_head(*mComm, mRunState.MainStream);
    }

    const std::size_t xf_stride = get_dtype_size(xF_flat.DType);
    const std::size_t tgt_stride = get_dtype_size(targets.DType);
    const std::size_t lse_stride = get_dtype_size(ETensorDType::FP32);
    const std::size_t dloss_stride = get_dtype_size(d_loss.DType);

    for (int nano_step = 0; nano_step < lmhead_chunks; ++nano_step) {
        const long token_offset = static_cast<long>(nano_step) * nano_batch_size;

        Tensor xF_slice = xF_flat;
        xF_slice.Data = static_cast<std::byte*>(xF_slice.Data) +
                        static_cast<std::size_t>(token_offset) * xf_stride * static_cast<std::size_t>(C);
        xF_slice.Sizes[0] = nano_batch_size;
        xF_slice.Sizes[1] = C;
        xF_slice.Rank = 2;

        Tensor tgt_slice = targets;
        tgt_slice.Data = static_cast<std::byte*>(tgt_slice.Data) + static_cast<std::size_t>(token_offset) * tgt_stride;
        tgt_slice.Sizes[0] = nano_batch_size;
        tgt_slice.Rank = 1;

        Tensor dloss_slice = d_loss;
        dloss_slice.Data =
            static_cast<std::byte*>(dloss_slice.Data) + static_cast<std::size_t>(token_offset) * dloss_stride;
        dloss_slice.Sizes[0] = nano_batch_size;
        dloss_slice.Rank = 1;

        Tensor logits = mRunState.non_block_activations().output;
        logits.Sizes[0] = nano_batch_size;
        logits.Sizes[1] = V;
        logits.Rank = 2;

        matmul(logits,
               weight,
               xF_slice,
               std::nullopt,
               nullptr,
               nullptr,
               mRunState.CublasLtHandle,
               mRunState.CuBlasWorkspace,
               static_cast<int>(V),
               static_cast<int>(nano_batch_size),
               static_cast<int>(C),
               swap_transpose(EMMTranspose::NT),
               false,
               mRunState.MainStream);

        // Mirror forward's logit softcapping so softmax probabilities and the
        // cross-entropy gradient match what the forward saw. Without this the
        // backward computes softmax of *uncapped* raw logits, producing
        // gradients many orders of magnitude too large for models like Gemma4
        // that rely on softcap to keep logits finite.
        if (op.attrs.softcap > 0.0f) {
            softcap_logits(logits, op.attrs.softcap, static_cast<int>(nano_batch_size), V, mRunState.MainStream);
        }

        // Save capped logits for softcap backward (only needed when softcap is
        // enabled and we will write d_logits back into the same buffer).
        Tensor capped_copy{};
        const bool need_softcap_bwd = (op.attrs.softcap > 0.0f);
        if (need_softcap_bwd) {
            capped_copy = mRunState.temp_alloc(logits.DType, {nano_batch_size, V}, "fused_lm_head_softcap_capped");
            CUDA_CHECK(cudaMemcpyAsync(capped_copy.Data,
                                       logits.Data,
                                       static_cast<std::size_t>(nano_batch_size) * static_cast<std::size_t>(V) *
                                           get_dtype_size(logits.DType),
                                       cudaMemcpyDeviceToDevice,
                                       mRunState.MainStream));
        }

        Tensor logsumexp_view{};
        Tensor* logsumexp = nullptr;
        if (mRunState.scratch().cross_entropy_logsumexp.Data) {
            logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
            logsumexp_view.Data =
                static_cast<std::byte*>(logsumexp_view.Data) + static_cast<std::size_t>(token_offset) * lse_stride;
            logsumexp_view.Sizes[0] = nano_batch_size;
            logsumexp_view.Rank = 1;
            logsumexp = &logsumexp_view;
        }

        if (mInvTemperatureGpu) {
            const float* inv_t = mInvTemperatureGpu + token_offset;
            scale_logits_rows(logits, inv_t, static_cast<int>(nano_batch_size), V, P, mRunState.MainStream);
        }

        if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
            chunked_cross_entropy_backward(logits,
                                           logits,
                                           logsumexp,
                                           dloss_slice,
                                           tgt_slice,
                                           static_cast<int>(nano_batch_size),
                                           V,
                                           P,
                                           mRunState.MainStream);
        } else {
            fused_cross_entropy_backward(logits,
                                         logits,
                                         logsumexp,
                                         dloss_slice,
                                         tgt_slice,
                                         static_cast<int>(nano_batch_size),
                                         V,
                                         P,
                                         mRunState.MainStream);
        }

        if (mInvTemperatureGpu) {
            const float* inv_t = mInvTemperatureGpu + token_offset;
            // Chain through temperature scaling: dlogits *= inv_temperature
            scale_logits_rows(logits, inv_t, static_cast<int>(nano_batch_size), V, P, mRunState.MainStream);
        }

        // Chain through softcap derivative to convert d_capped → d_raw before
        // propagating to xF / W.
        if (need_softcap_bwd) {
            softcap_logits_backward(logits,
                                    capped_copy,
                                    op.attrs.softcap,
                                    static_cast<int>(nano_batch_size),
                                    V,
                                    mRunState.MainStream);
            mRunState.temp_free(capped_copy);
        }

        if (d_weight_ptr) {
            const bool accumulate = d_weight_accumulate || (nano_step != 0);
            matmul(*d_weight_ptr,
                   xF_slice,
                   logits,
                   std::nullopt,
                   nullptr,
                   nullptr,
                   mRunState.CublasLtHandle,
                   mRunState.CuBlasWorkspace,
                   static_cast<int>(C),
                   static_cast<int>(V),
                   static_cast<int>(nano_batch_size),
                   swap_transpose(EMMTranspose::TN),
                   accumulate,
                   mRunState.MainStream);
        }

        if (d_xF_ptr) {
            Tensor d_xF_slice = *d_xF_ptr;
            const std::size_t dx_stride = get_dtype_size(d_xF_slice.DType);
            d_xF_slice.Data = static_cast<std::byte*>(d_xF_slice.Data) +
                              static_cast<std::size_t>(token_offset) * dx_stride * static_cast<std::size_t>(C);
            d_xF_slice.Sizes[0] = nano_batch_size;
            d_xF_slice.Sizes[1] = C;
            d_xF_slice.Rank = 2;

            matmul(d_xF_slice,
                   weight,
                   logits,
                   std::nullopt,
                   nullptr,
                   nullptr,
                   mRunState.CublasLtHandle,
                   mRunState.CuBlasWorkspace,
                   static_cast<int>(C),
                   static_cast<int>(nano_batch_size),
                   static_cast<int>(V),
                   swap_transpose(EMMTranspose::NN),
                   false,
                   mRunState.MainStream);
        }
    }

    if (need_lm_head) {
        mWeightManager->release_lm_head(mRunState.MainStream);
    }
}

namespace {

// -----------------------------------------------------------------------------
// Fused LMHead + loss backward
// Forward: loss = fused_lm_head_loss(xF_flat, weight, targets)
// Backward: d_xF_flat, d_weight = fused_lm_head_loss_backward(d_loss, xF_flat, weight, targets)
// -----------------------------------------------------------------------------
std::vector<Operation> fused_lm_head_loss_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (!ctx.needs_grad(0) && !ctx.needs_grad(1)) {
        return ops;
    }

    const auto& fwd = ctx.fwd_op;
    const std::string& xF_flat = fwd.inputs[0];
    const std::string& weight = fwd.inputs[1];
    const std::string& targets = fwd.inputs[2];

    std::string xF_ref = ctx.is_param(xF_flat) ? xF_flat : saved_ref(xF_flat);
    std::string weight_ref = ctx.is_param(weight) ? weight : saved_ref(weight);

    std::vector<std::string> inputs = {ctx.d_output, xF_ref, weight_ref, targets};
    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");

    // Forward attrs (softcap, compute_accuracy, etc.) must propagate so that
    // dispatch_fused_lm_head_loss_backward can mirror the forward's softcap /
    // temperature pipeline. Without the softcap attr, the backward would
    // recompute softmax of raw (uncapped) logits and produce gradients orders
    // of magnitude too large for softcap-using models like Gemma4.
    ops.push_back(make_operation("fused_lm_head_loss_backward_" + std::to_string(ctx.op_counter++),
                                 "fused_lm_head_loss_backward",
                                 "fused_lm_head_loss_backward",
                                 inputs,
                                 outputs,
                                 fwd.attrs));

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("fused_lm_head_loss", ::dsl::fused_lm_head_loss_backward);
REGISTER_AUTODIFF("lm_head_loss", ::dsl::fused_lm_head_loss_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// FusedLMHeadLoss
// ------------------------------------------------------------------------
const int _fused_lm_head_loss_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "fused_lm_head_loss";
    sig.min_inputs = 3;
    sig.max_inputs = 3;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap&,
                       const ShapeEnv&) -> std::optional<ShapeValidationError> {
        const auto& xF_flat = inputs[0];
        const auto& weight = inputs[1];
        const auto& targets = inputs[2];
        const auto& loss = outputs[0];

        if (auto err = validators::check_rank(xF_flat, 2, "xF_flat", "fused_lm_head_loss")) {
            return err;
        }
        if (auto err = validators::check_rank(weight, 2, "weight", "fused_lm_head_loss")) {
            return err;
        }
        if (targets.size() != 1 && targets.size() != 2) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "fused_lm_head_loss: targets has rank " << targets.size() << " but expected 1 or 2";
            err.message = oss.str();
            return std::make_optional(err);
        }
        if (auto err = validators::check_rank(loss, 1, "loss", "fused_lm_head_loss")) {
            return err;
        }

        if (!xF_flat.empty() && !weight.empty() && xF_flat[1] != weight[1]) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "fused_lm_head_loss: xF_flat C (" << xF_flat[1] << ") doesn't match weight C (" << weight[1] << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }
        if (!xF_flat.empty() && !targets.empty()) {
            const long target_bt = (targets.size() == 2) ? targets[0] * targets[1] : targets[0];
            if (xF_flat[0] != target_bt) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "fused_lm_head_loss: xF_flat BT (" << xF_flat[0] << ") doesn't match targets BT (" << target_bt
                    << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
        }
        if (!xF_flat.empty() && !loss.empty() && xF_flat[0] != loss[0]) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "fused_lm_head_loss: xF_flat BT (" << xF_flat[0] << ") doesn't match loss BT (" << loss[0] << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// FusedLMHeadLossBackward
// ------------------------------------------------------------------------
const int _fused_lm_head_loss_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "fused_lm_head_loss_backward";
    sig.min_inputs = 4;
    sig.max_inputs = 4;
    sig.min_outputs = 2;
    sig.max_outputs = 2;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap&,
                       const ShapeEnv&) -> std::optional<ShapeValidationError> {
        const auto& d_loss = inputs[0];
        const auto& xF_flat = inputs[1];
        const auto& weight = inputs[2];
        const auto& targets = inputs[3];
        const auto& d_xF_flat = outputs[0];
        const auto& d_weight = outputs[1];

        if (auto err = validators::check_rank(d_loss, 1, "d_loss", "fused_lm_head_loss_backward")) {
            return err;
        }
        if (auto err = validators::check_rank(xF_flat, 2, "xF_flat", "fused_lm_head_loss_backward")) {
            return err;
        }
        if (auto err = validators::check_rank(weight, 2, "weight", "fused_lm_head_loss_backward")) {
            return err;
        }
        if (targets.size() != 1 && targets.size() != 2) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "fused_lm_head_loss_backward: targets has rank " << targets.size() << " but expected 1 or 2";
            err.message = oss.str();
            return std::make_optional(err);
        }
        if (auto err = validators::check_rank(d_xF_flat, 2, "d_xF_flat", "fused_lm_head_loss_backward")) {
            return err;
        }
        if (auto err = validators::check_rank(d_weight, 2, "d_weight", "fused_lm_head_loss_backward")) {
            return err;
        }

        if (!xF_flat.empty() && !d_xF_flat.empty() && xF_flat[0] != d_xF_flat[0]) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "fused_lm_head_loss_backward: xF_flat BT (" << xF_flat[0] << ") doesn't match d_xF_flat BT ("
                << d_xF_flat[0] << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }
        if (!xF_flat.empty() && !d_xF_flat.empty() && xF_flat[1] != d_xF_flat[1]) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "fused_lm_head_loss_backward: xF_flat C (" << xF_flat[1] << ") doesn't match d_xF_flat C ("
                << d_xF_flat[1] << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }
        if (!weight.empty() && !d_weight.empty() && weight[0] != d_weight[0]) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "fused_lm_head_loss_backward: weight V (" << weight[0] << ") doesn't match d_weight V ("
                << d_weight[0] << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }
        if (!weight.empty() && !d_weight.empty() && weight[1] != d_weight[1]) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "fused_lm_head_loss_backward: weight C (" << weight[1] << ") doesn't match d_weight C ("
                << d_weight[1] << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }
        if (!xF_flat.empty() && !targets.empty()) {
            const long target_bt = (targets.size() == 2) ? targets[0] * targets[1] : targets[0];
            if (xF_flat[0] != target_bt) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "fused_lm_head_loss_backward: xF_flat BT (" << xF_flat[0] << ") doesn't match targets BT ("
                    << target_bt << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
        }
        if (!xF_flat.empty() && !d_loss.empty() && xF_flat[0] != d_loss[0]) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "fused_lm_head_loss_backward: xF_flat BT (" << xF_flat[0] << ") doesn't match d_loss BT ("
                << d_loss[0] << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
