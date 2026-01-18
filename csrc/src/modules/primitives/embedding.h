// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_PRIMITIVES_EMBEDDING_H
#define SUROGATE_SRC_MODULES_PRIMITIVES_EMBEDDING_H

#include "modules/module_base.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief Token embedding module
 *
 * Looks up token embeddings from a vocabulary table.
 * Supports optional positional embeddings (though LLaMA uses RoPE instead).
 *
 * Input: (B, T) int32 token indices
 * Output: (B, T, hidden_size) embeddings
 */
class EmbeddingModule : public ModuleBase<EmbeddingModule> {
public:
    /**
     * @brief Configuration for embedding lookup
     */
    struct Config {
        int vocab_size;         ///< Number of tokens in vocabulary
        int hidden_size;        ///< Embedding dimension
        bool use_positional = false;  ///< Whether to add positional embeddings (not for RoPE models)
    };

    /**
     * @brief Weight tensors for embedding
     */
    struct Weights {
        Tensor embeddings;              ///< (vocab_size, hidden_size) token embeddings
        std::optional<Tensor> pos_emb;  ///< (max_seq_len, hidden_size) positional embeddings (optional)
    };

    /**
     * @brief Saved state for backward pass
     */
    struct Activations {
        Tensor input_tokens;            ///< (B, T) input token indices
        Tensor input_tokens_cpu;        ///< CPU copy for deterministic backward
        Tensor output;                  ///< (B, T, hidden_size) output embeddings
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients {
        Tensor d_embeddings;            ///< (vocab_size, hidden_size)
        Tensor scratch;                 ///< Scratch for deterministic backward
        Tensor workload_indices;        ///< For deterministic backward scheduling
        Tensor bucket_info;             ///< For deterministic backward scheduling
    };

    explicit EmbeddingModule(Config config) : mConfig(config) {}

    /**
     * @brief Forward pass: lookup token embeddings
     *
     * @param ctx Module context with CUDA resources
     * @param w Weight tensors
     * @param input Token indices (B, T) as int32
     * @param acts Activation storage for backward
     * @return Embeddings (B, T, hidden_size)
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    /**
     * @brief Backward pass: compute embedding gradients
     *
     * Uses deterministic reduction for reproducibility.
     *
     * @param ctx Module context with CUDA resources
     * @param w Weight tensors
     * @param acts Saved activations from forward
     * @param grad_output Gradient w.r.t. output (B, T, hidden_size)
     * @param grads Gradient storage
     * @param accumulate If true, accumulate into existing gradients
     * @return Empty tensor (no gradient w.r.t. discrete token indices)
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

    // Accessors
    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int vocab_size() const { return mConfig.vocab_size; }
    [[nodiscard]] int hidden_size() const { return mConfig.hidden_size; }

private:
    Config mConfig;
};

inline Tensor EmbeddingModule::forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
    const int B = ctx.B;
    const int T = ctx.T;
    const int C = mConfig.hidden_size;
    const int V = mConfig.vocab_size;

    // Save input for backward
    acts.input_tokens = input;

    encoder_forward(
        acts.output,
        input,
        w.embeddings,
        w.pos_emb,
        B, T, C, V,
        ctx.stream
    );

    return acts.output;
}

inline Tensor EmbeddingModule::backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                                              Tensor& grad_output, Gradients& grads, bool accumulate) {
    const int B = ctx.B;
    const int T = ctx.T;
    const int C = mConfig.hidden_size;

    // Note: encoder_backward requires CPU-side token indices for deterministic scheduling
    // The caller must ensure acts.input_tokens_cpu is populated

    // TODO: Need to pass proper sync events for the async CPU-GPU copy
    // For now, this is a simplified version
    cudaEvent_t sync_event = nullptr;  // Would be provided by run state
    cudaStream_t copy_stream = nullptr;  // Would be provided by run state

    encoder_backward(
        grads.d_embeddings,
        grads.scratch,
        grads.workload_indices,
        grads.bucket_info,
        grad_output,
        acts.input_tokens,
        acts.input_tokens_cpu,
        B, T, C,
        0,  // seed for deterministic reduction
        ctx.stream,
        sync_event,
        copy_stream
    );

    // No gradient w.r.t. discrete input tokens
    return Tensor{};
}

/**
 * @brief Language model head module (output projection)
 *
 * Projects hidden states to vocabulary logits.
 * Often tied with embedding weights (transposed).
 *
 * For memory efficiency, can compute in chunks and fuse with softmax loss.
 */
class LMHeadModule : public ModuleBase<LMHeadModule> {
public:
    /**
     * @brief Configuration for LM head
     */
    struct Config {
        int hidden_size;        ///< Input dimension
        int vocab_size;         ///< Output dimension (vocabulary size)
        bool tied_weights;      ///< Whether weights are tied to embeddings
        int num_chunks = 1;     ///< Number of chunks for memory efficiency
    };

    /**
     * @brief Weight tensors
     */
    struct Weights {
        Tensor weight;          ///< (vocab_size, hidden_size) - may be shared with embeddings
    };

    /**
     * @brief Saved state for backward pass
     */
    struct Activations {
        QuantizableTensor input;    ///< Hidden states (B*T, hidden_size)
        Tensor logits;              ///< Output logits (B*T, vocab_size) or chunked
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients {
        Tensor d_weight;        ///< (vocab_size, hidden_size)
    };

    explicit LMHeadModule(Config config) : mConfig(config) {}

    /**
     * @brief Forward pass: project to logits
     *
     * Note: For training, the fused_classifier kernel is typically used
     * instead, which combines matmul + softmax + loss in one pass.
     *
     * @param ctx Module context
     * @param w Weights
     * @param input Hidden states (B*T, hidden_size)
     * @param acts Activation storage
     * @return Logits (B*T, vocab_size)
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    /**
     * @brief Backward pass
     *
     * In practice, the training path handles this via the LM head loss.
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

    /**
     * @brief Fused forward + loss computation
     *
     * More memory efficient than separate forward + softmax + loss.
     * Computes cross-entropy loss and optionally returns gradient.
     *
     * @param ctx Module context
     * @param w Weights
     * @param input Hidden states
     * @param targets Target token indices
     * @param losses Output loss buffer
     * @param compute_grad If true, also compute gradient w.r.t. logits
     * @return Loss value
     */
    float forward_with_loss(ModuleContext& ctx, Weights& w, Tensor& input,
                            Tensor& targets, Tensor& losses, Tensor* valid_count,
                            bool compute_grad);

private:
    Config mConfig;
};

inline Tensor LMHeadModule::forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int V = mConfig.vocab_size;

    acts.input.Value = input;

    // Simple matmul projection
    matmul(
        acts.logits, w.weight, input, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        V, BT, C, EMMTranspose::TN, false,
        ctx.stream
    );

    return acts.logits;
}

inline float LMHeadModule::forward_with_loss(ModuleContext& ctx, Weights& w, Tensor& input,
                                              Tensor& targets, Tensor& losses, Tensor* valid_count,
                                              bool compute_grad) {
    // For chunked processing (memory efficiency)
    // Note: This is a simplified version. Full implementation would
    // iterate over chunks with fused_classifier.

    // The fused_classifier kernel handles matmul + softmax + loss + optional backward
    // in a single pass for memory efficiency

    // For now, delegate to the existing fused path
    // Full implementation would be in the model, not the module

    return 0.0f;  // Loss is computed by fused_classifier
}

inline Tensor LMHeadModule::backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                                           Tensor& grad_output, Gradients& grads, bool accumulate) {
    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int V = mConfig.vocab_size;

    Tensor grad_input;
    grad_input.DType = acts.input.Value.DType;

    // d_input = d_logits @ W (NN transpose)
    matmul(
        grad_input, w.weight, grad_output, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, BT, V, EMMTranspose::NN, false,
        ctx.stream
    );

    // d_weight = input^T @ d_logits (NT transpose)
    matmul(
        grads.d_weight, acts.input.Value, grad_output, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, V, BT, EMMTranspose::NT, accumulate,
        ctx.stream
    );

    return grad_input;
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_PRIMITIVES_EMBEDDING_H
