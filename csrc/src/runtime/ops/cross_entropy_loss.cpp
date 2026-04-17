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
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_cross_entropy_loss(const CompiledOp& op) {
    Tensor& logits = resolve_tensor(op.inputs[0]);
    Tensor& targets = resolve_tensor(op.inputs[1]);
    Tensor& loss = ensure_output_tensor(op.outputs[0]);

    const int BT = static_cast<int>(logits.Sizes[0]);
    const int V = static_cast<int>(logits.Sizes[1]);
    const int P = V;

    Tensor logsumexp_view{};
    Tensor* logsumexp = nullptr;
    if (mRunState.scratch().cross_entropy_logsumexp.Data) {
        logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
        logsumexp_view.Sizes[0] = BT;
        logsumexp_view.Rank = 1;
        logsumexp = &logsumexp_view;
    }

    if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
        if (!mRunState.scratch().cross_entropy_chunk_logsumexp.Data) {
            throw std::runtime_error("cross_entropy_loss: chunk logsumexp buffer is not allocated");
        }
        const int n_chunks = (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE;
        Tensor chunk_lse = mRunState.scratch().cross_entropy_chunk_logsumexp;
        chunk_lse.Sizes[0] = BT;
        chunk_lse.Sizes[1] = n_chunks;
        chunk_lse.Rank = 2;

        chunked_cross_entropy_forward(logits,
                                      loss,
                                      logsumexp,
                                      chunk_lse,
                                      targets,
                                      &mRunState.ValidTokenCount,
                                      op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                      BT,
                                      V,
                                      P,
                                      n_chunks,
                                      mRunState.MainStream);
    } else {
        fused_cross_entropy_forward(logits,
                                    loss,
                                    logsumexp,
                                    targets,
                                    &mRunState.ValidTokenCount,
                                    op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                    BT,
                                    V,
                                    P,
                                    mRunState.MainStream);
    }
}

void CompiledExecutor::dispatch_cross_entropy_loss_backward(const CompiledOp& op) {
    Tensor& d_loss = resolve_tensor(op.inputs[0]);
    Tensor& logits = resolve_tensor(op.inputs[1]);
    Tensor& targets = resolve_tensor(op.inputs[2]);
    Tensor& d_logits = ensure_output_tensor(op.outputs[0]);

    const int BT = static_cast<int>(logits.Sizes[0]);
    const int V = static_cast<int>(logits.Sizes[1]);
    const int P = V;

    // HuggingFace-style normalization: use reduction="sum" semantics.
    // dloss = 1.0 means each valid token contributes equally to the gradient sum.
    // The actual normalization by accumulated valid tokens happens in global_norm_sqrt.
    // Robustly seed d_loss even if the name has SSA suffixes or mapped to loss/losses.
    const std::string d_loss_name = strip_ssa_suffix(op.inputs[0].name);
    if (op.inputs[0].slot == TensorSlot::DLoss || op.inputs[0].slot == TensorSlot::Losses || d_loss_name == "d_loss" ||
        d_loss_name == "loss" || d_loss_name == "losses") {
        fill_constant(d_loss, 1.0f, static_cast<std::size_t>(d_loss.nelem()), mRunState.MainStream);
    }

    Tensor logsumexp_view{};
    Tensor* logsumexp = nullptr;
    if (mRunState.scratch().cross_entropy_logsumexp.Data) {
        logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
        logsumexp_view.Sizes[0] = BT;
        logsumexp_view.Rank = 1;
        logsumexp = &logsumexp_view;
    }

    if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
        chunked_cross_entropy_backward(d_logits, logits, logsumexp, d_loss, targets, BT, V, P, mRunState.MainStream);
    } else {
        fused_cross_entropy_backward(d_logits, logits, logsumexp, d_loss, targets, BT, V, P, mRunState.MainStream);
    }
}

namespace {

// -----------------------------------------------------------------------------
// Cross-entropy loss backward (typically fused with softmax)
// Forward: loss = cross_entropy(logits, targets)
// Backward: d_logits = softmax(logits) - one_hot(targets)
// Note: This is usually handled by fused_classifier, not standalone
// -----------------------------------------------------------------------------
std::vector<Operation> cross_entropy_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string logits = fwd.inputs[0];
        std::string targets = fwd.inputs[1];

        ops.push_back(make_operation("cross_entropy_backward_" + std::to_string(ctx.op_counter++),
                                     "cross_entropy_backward",
                                     "cross_entropy_backward",
                                     {ctx.d_output, saved_ref(logits), targets},
                                     {ctx.d_inputs[0]}));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("cross_entropy", ::dsl::cross_entropy_backward);
REGISTER_AUTODIFF("cross_entropy_loss", ::dsl::cross_entropy_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// CrossEntropyLoss
// ------------------------------------------------------------------------
const int _cross_entropy_loss_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "cross_entropy_loss";
    sig.min_inputs = 2;
    sig.max_inputs = 2;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap&,
                       const ShapeEnv&) -> std::optional<ShapeValidationError> {
        const auto& logits = inputs[0];
        const auto& targets = inputs[1];
        const auto& loss = outputs[0];

        // logits should be rank 2: [BT, V]
        if (auto err = validators::check_rank(logits, 2, "logits", "cross_entropy_loss")) {
            return err;
        }
        // targets should be rank 1: [BT] or rank 2: [B, T]
        if (targets.size() != 1 && targets.size() != 2) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "cross_entropy_loss: targets has rank " << targets.size() << " but expected 1 or 2";
            err.message = oss.str();
            return std::make_optional(err);
        }
        // loss should be rank 1: [BT]
        if (auto err = validators::check_rank(loss, 1, "loss", "cross_entropy_loss")) {
            return err;
        }

        if (!logits.empty() && !targets.empty()) {
            const long target_bt = (targets.size() == 2) ? targets[0] * targets[1] : targets[0];
            if (logits[0] != target_bt) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "cross_entropy_loss: logits BT (" << logits[0] << ") doesn't match targets BT (" << target_bt
                    << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
        }
        if (!logits.empty() && !loss.empty() && logits[0] != loss[0]) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "cross_entropy_loss: logits BT (" << logits[0] << ") doesn't match loss BT (" << loss[0] << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// CrossEntropyLossBackward
// ------------------------------------------------------------------------
const int _cross_entropy_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "cross_entropy_backward";
    sig.min_inputs = 3;
    sig.max_inputs = 3;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap&,
                       const ShapeEnv&) -> std::optional<ShapeValidationError> {
        const auto& d_loss = inputs[0];
        const auto& logits = inputs[1];
        const auto& targets = inputs[2];
        const auto& d_logits = outputs[0];

        if (auto err = validators::check_rank(d_loss, 1, "d_loss", "cross_entropy_backward")) {
            return err;
        }
        if (auto err = validators::check_rank(logits, 2, "logits", "cross_entropy_backward")) {
            return err;
        }
        if (targets.size() != 1 && targets.size() != 2) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "cross_entropy_backward: targets has rank " << targets.size() << " but expected 1 or 2";
            err.message = oss.str();
            return std::make_optional(err);
        }
        if (auto err = validators::check_rank(d_logits, 2, "d_logits", "cross_entropy_backward")) {
            return err;
        }

        if (!logits.empty() && !d_logits.empty() && logits[0] != d_logits[0]) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "cross_entropy_backward: logits BT (" << logits[0] << ") doesn't match d_logits BT (" << d_logits[0]
                << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }
        if (!logits.empty() && !d_logits.empty() && logits[1] != d_logits[1]) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "cross_entropy_backward: logits V (" << logits[1] << ") doesn't match d_logits V (" << d_logits[1]
                << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }
        if (!logits.empty() && !targets.empty()) {
            const long target_bt = (targets.size() == 2) ? targets[0] * targets[1] : targets[0];
            if (logits[0] != target_bt) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "cross_entropy_backward: logits BT (" << logits[0] << ") doesn't match targets BT (" << target_bt
                    << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
        }
        if (!logits.empty() && !d_loss.empty() && logits[0] != d_loss[0]) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "cross_entropy_backward: logits BT (" << logits[0] << ") doesn't match d_loss BT (" << d_loss[0]
                << ")";
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
