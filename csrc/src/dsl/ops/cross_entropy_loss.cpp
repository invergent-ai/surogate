#include "dsl/compiled_ops.h"

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

#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_helpers.h"
#include "dsl/graph_executor_utils.h"
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

        chunked_cross_entropy_forward(logits, loss, logsumexp, chunk_lse, targets,
                                      &mRunState.ValidTokenCount,
                                      op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                      BT, V, P, n_chunks, mRunState.MainStream);
    } else {
        fused_cross_entropy_forward(logits, loss, logsumexp, targets,
                                    &mRunState.ValidTokenCount,
                                    op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                    BT, V, P, mRunState.MainStream);
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
    if (op.inputs[0].slot == TensorSlot::DLoss ||
        op.inputs[0].slot == TensorSlot::Losses ||
        d_loss_name == "d_loss" || d_loss_name == "loss" || d_loss_name == "losses") {
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
        chunked_cross_entropy_backward(d_logits, logits, logsumexp, d_loss, targets,
                                       BT, V, P, mRunState.MainStream);
    } else {
        fused_cross_entropy_backward(d_logits, logits, logsumexp, d_loss, targets,
                                     BT, V, P, mRunState.MainStream);
    }
}

}  // namespace dsl
