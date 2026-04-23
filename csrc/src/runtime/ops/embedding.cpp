#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_embedding(const CompiledOp& op) {
    Tensor& token_ids = resolve_tensor(op.inputs[0]);
    Tensor& emb = op.inputs.size() > 1 ? resolve_tensor(op.inputs[1]) : mWeights.get("embedding");
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    // Derive dims from the weight tensor so non-main embeddings (e.g.,
    // Gemma4 pli_embedding with weight [vocab, n_layers * PLI_D]) work
    // correctly instead of being clamped to mConfig.HiddenSize.
    int emb_dim = mConfig.HiddenSize;
    int vocab = mConfig.VocabSize;
    if (emb.Rank >= 2) {
        vocab = static_cast<int>(emb.Sizes[0]);
        emb_dim = static_cast<int>(emb.Sizes[1]);
    }

    // Bounds checks — compile-time invariants that turn future layout bugs
    // into clean compile-path errors instead of async illegal-memory-access
    // reports in the encoder kernel.
    const long expected_tokens = mB * mT;
    if (token_ids.nelem() < static_cast<std::size_t>(expected_tokens)) {
        throw std::runtime_error(
            "dispatch_embedding: token_ids buffer too small — have " + std::to_string(token_ids.nelem()) +
            " elements, need mB*mT=" + std::to_string(expected_tokens) + " (B=" + std::to_string(mB) +
            ", T=" + std::to_string(mT) + ", input='" + op.inputs[0].name + "')");
    }
    const long expected_out_elems = mB * mT * emb_dim;
    if (out.nelem() < static_cast<std::size_t>(expected_out_elems)) {
        throw std::runtime_error("dispatch_embedding: output buffer too small — have " + std::to_string(out.nelem()) +
                                 " elements, need mB*mT*C=" + std::to_string(expected_out_elems) +
                                 " (B=" + std::to_string(mB) + ", T=" + std::to_string(mT) +
                                 ", C=" + std::to_string(emb_dim) + ", output='" + op.outputs[0].name + "')");
    }

    encoder_forward(out,
                    token_ids,
                    emb,
                    std::nullopt,
                    static_cast<int>(mB),
                    static_cast<int>(mT),
                    emb_dim,
                    vocab,
                    mRunState.MainStream);
}

void CompiledExecutor::dispatch_embedding_backward(const CompiledOp& op) {
    // Skip embedding backward entirely in LoRA-only mode
    if (mRunState.is_lora_only_mode()) {
        return;
    }

    // inputs: d_encoded, token_ids
    // outputs: d_embedding (sparse update)
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    if (op.outputs.empty() || op.outputs[0].name.empty()) {
        return;  // Skip if no output expected
    }

    // Get the pre-allocated gradient tensor
    Tensor* d_emb_ptr = nullptr;
    if (op.outputs[0].tensor_id >= 0 && static_cast<std::size_t>(op.outputs[0].tensor_id) < mTensors.size() &&
        mTensors[op.outputs[0].tensor_id].Data) {
        d_emb_ptr = &mTensors[op.outputs[0].tensor_id];
    }
    if (!d_emb_ptr) {
        // Gradient not allocated (embedding frozen in LoRA mode)
        return;
    }
    Tensor& d_emb = *d_emb_ptr;
    // Fast atomic fallback for FP32 embedding grads with BF16 upstream grads.
    if (d_emb.DType == ETensorDType::FP32 && d_out.DType == ETensorDType::BF16) {
        encoder_backward_atomic(d_emb.get<float>(),
                                d_out.get<nv_bfloat16>(),
                                mRunState.Inputs.get<int>(),
                                static_cast<int>(mB),
                                static_cast<int>(mT),
                                mConfig.HiddenSize,
                                mRunState.MainStream);
        return;
    }

    // encoder_backward requires CPU-side inputs for deterministic bucketing
    if (!mLastInputsCpu || !mLastInputsCpu->Data) {
        throw std::runtime_error("CompiledExecutor: embedding_backward requires CPU inputs (set_last_inputs_cpu)");
    }

    const int vocab = mConfig.VocabSize;
    const int total_tokens = static_cast<int>(mB * mT);
    const long hidden = (d_emb.Rank > 1) ? d_emb.Sizes[1] : 0;

    unsigned int seed = mRngSeedFn ? mRngSeedFn() : 0;

    encoder_backward(d_emb,
                     mRunState.scratch().encoder_bwd_scratch,
                     mRunState.scratch().encoder_bwd_indices,
                     mRunState.scratch().encoder_bwd_info,
                     d_out,
                     mRunState.Inputs,
                     *mLastInputsCpu,
                     static_cast<int>(mB),
                     static_cast<int>(mT),
                     mConfig.HiddenSize,
                     seed,
                     mRunState.MainStream,
                     mRunState.side_stream_event(),
                     mRunState.side_stream());
}

namespace {

// -----------------------------------------------------------------------------
// Embedding backward rule
// Forward: out = embedding(token_ids, embed_weight)
// Backward: d_embed = embedding_backward(d_out, token_ids)
// Note: no gradient for token_ids (discrete indices)
// -----------------------------------------------------------------------------
std::vector<Operation> embedding_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    std::string token_ids = fwd.inputs[0];

    // Only gradient wrt embedding weights (input 1)
    if (ctx.needs_grad(1)) {
        ops.push_back(make_operation("embedding_backward_" + std::to_string(ctx.op_counter++),
                                     "embedding_backward",
                                     "embedding_backward",
                                     {ctx.d_output, saved_ref(token_ids)},
                                     {ctx.d_inputs[1]}));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("embedding", ::dsl::embedding_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// Embedding
// ------------------------------------------------------------------------
const int _embedding_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "embedding";
    sig.min_inputs = 2;
    sig.max_inputs = 2;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto& inputs, const auto& outputs, const AttrMap&, const ShapeEnv&) {
        if (inputs.size() < 2 || outputs.empty()) {
            ShapeValidationError err;
            err.message = "embedding requires 2 inputs (indices, weight) and 1 output";
            return std::make_optional(err);
        }

        const auto& indices_shape = inputs[0];
        const auto& weight_shape = inputs[1];
        const auto& out_shape = outputs[0];

        // Weight should be 2D: [vocab_size, embedding_dim]
        if (weight_shape.size() != 2) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "embedding: weight must be 2D, got rank " << weight_shape.size();
            err.message = oss.str();
            return std::make_optional(err);
        }

        // Output should be indices_shape + [embedding_dim]
        if (out_shape.size() != indices_shape.size() + 1) {
            ShapeValidationError err;
            err.message = "embedding: output rank should be indices rank + 1";
            return std::make_optional(err);
        }

        for (size_t i = 0; i < indices_shape.size(); ++i) {
            if (out_shape[i] != indices_shape[i]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "embedding: output dim[" << i << "] (" << out_shape[i] << ") doesn't match indices dim[" << i
                    << "] (" << indices_shape[i] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
        }

        if (out_shape.back() != weight_shape[1]) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "embedding: output last dim (" << out_shape.back() << ") doesn't match weight embedding dim ("
                << weight_shape[1] << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// EmbeddingBackward
// ------------------------------------------------------------------------
const int _embedding_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "embedding_backward";
    sig.min_inputs = 1;
    sig.max_inputs = 1;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& d_embedding = outputs[0];

        // Check d_embedding is rank 2
        if (auto err = validators::check_rank(d_embedding, 2, "d_embedding", "embedding_backward")) {
            return err;
        }

        // Check d_out last dim matches d_embedding embedding dim
        if (!d_out.empty() && d_out.back() != d_embedding[1]) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "embedding_backward: d_out last dim (" << d_out.back()
                << ") doesn't match d_embedding embedding dim (" << d_embedding[1] << ")";
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
