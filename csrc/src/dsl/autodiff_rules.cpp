// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Built-in backward rules for DSL automatic differentiation.

#include "autodiff.h"

#include <stdexcept>

namespace dsl {

namespace {

// Helper to find attribute value
const AttrValue* find_attr(const AttrMap& attrs, const std::string& key) {
    auto it = attrs.find(key);
    return it != attrs.end() ? &it->second : nullptr;
}

// Helper to get string attribute
std::string get_string_attr(const AttrMap& attrs, const std::string& key, const std::string& default_val = "") {
    if (auto* attr = find_attr(attrs, key)) {
        if (auto* s = std::get_if<std::string>(&attr->value)) {
            return *s;
        }
    }
    return default_val;
}

// Helper to copy attributes
AttrMap copy_attrs(const AttrMap& src, const std::vector<std::string>& keys) {
    AttrMap dst;
    for (const auto& key : keys) {
        if (auto* attr = find_attr(src, key)) {
            dst[key] = *attr;
        }
    }
    return dst;
}

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

    std::string trans_dA, trans_dB;

    // Determine references for A and B in backward pass:
    // - Parameters are available at backward time (gathered from weight manager)
    // - Activations must be saved from forward pass (use saved_ref)
    std::string A_for_dB = ctx.is_param(A) ? A : saved_ref(A);
    std::string B_for_dA = ctx.is_param(B) ? B : saved_ref(B);

    if (trans == "NN") {
        // C = A @ B
        // dA = dC @ B.T
        // dB = A.T @ dC
        trans_dA = "NT";
        trans_dB = "TN";
    } else if (trans == "NT") {
        // C = A @ B.T
        // dA = dC @ B
        // dB = dC.T @ A = (A.T @ dC).T -> we compute A.T @ dC then it's already correct shape
        trans_dA = "NN";
        trans_dB = "TN";
    } else if (trans == "TN") {
        // C = A.T @ B
        // dA = B @ dC.T
        // dB = A @ dC
        trans_dA = "NT";  // dC @ B.T with swapped args = B @ dC.T ... need to think carefully
        trans_dB = "NN";
        // Actually for TN: dA = B @ dC.T, dB = A @ dC
        // But our matmul is row-major, so we express as:
        // dA = (dC.T @ B.T).T = B @ dC.T -> matmul(B, dC, NT)? Let's use: matmul(dC, B, TN)
    } else if (trans == "TT") {
        // C = A.T @ B.T
        trans_dA = "TT";
        trans_dB = "TT";
    }

    // Generate dA if needed
    if (ctx.needs_grad(0)) {
        AttrMap attrs;
        attrs["transpose"] = AttrValue(trans_dA);

        ops.push_back(make_operation(
            "matmul_dA_" + std::to_string(ctx.op_counter++),
            "matmul",
            "matmul",
            {dC, B_for_dA},
            {ctx.d_inputs[0]},
            attrs));
    }

    // Generate dB if needed
    if (ctx.needs_grad(1)) {
        AttrMap attrs;
        attrs["transpose"] = AttrValue(trans_dB);

        // For NT mode: dB = dC.T @ A, so inputs are {dC, A} with TN transpose
        // For NN mode: dB = A.T @ dC, so inputs are {A, dC} with TN transpose
        // For TN mode: dB = A @ dC, so inputs are {A, dC} with NN transpose
        // For TT mode: dB = dC.T @ A.T, so inputs are {dC, A} with TT transpose
        std::vector<std::string> dB_inputs;
        if (trans == "NT" || trans == "TT") {
            dB_inputs = {dC, A_for_dB};  // dC first for NT/TT
        } else {
            dB_inputs = {A_for_dB, dC};  // A first for NN/TN
        }

        ops.push_back(make_operation(
            "matmul_dB_" + std::to_string(ctx.op_counter++),
            "matmul",
            "matmul",
            dB_inputs,
            {ctx.d_inputs[1]},
            attrs));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// Add backward rule
// Forward: C = A + B
// Backward: dA = dC, dB = dC (with broadcast reduction if needed)
// -----------------------------------------------------------------------------
std::vector<Operation> add_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    // Gradient passes through unchanged (identity for addition)
    // Note: if shapes differ due to broadcasting, would need reduce_sum
    // For now, assume same shapes

    if (ctx.needs_grad(0)) {
        // dA = dC (just alias/copy)
        ops.push_back(make_operation(
            "add_dA_" + std::to_string(ctx.op_counter++),
            "identity",
            "identity",
            {ctx.d_output},
            {ctx.d_inputs[0]}));
    }

    if (ctx.needs_grad(1)) {
        // dB = dC
        ops.push_back(make_operation(
            "add_dB_" + std::to_string(ctx.op_counter++),
            "identity",
            "identity",
            {ctx.d_output},
            {ctx.d_inputs[1]}));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// Multiply backward rule
// Forward: C = A * B (elementwise)
// Backward: dA = dC * B, dB = dC * A
// -----------------------------------------------------------------------------
std::vector<Operation> multiply_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    const std::string& A = fwd.inputs[0];
    const std::string& B = fwd.inputs[1];

    if (ctx.needs_grad(0)) {
        // dA = dC * B
        ops.push_back(make_operation(
            "mul_dA_" + std::to_string(ctx.op_counter++),
            "multiply",
            "multiply",
            {ctx.d_output, saved_ref(B)},
            {ctx.d_inputs[0]}));
    }

    if (ctx.needs_grad(1)) {
        // dB = dC * A
        ops.push_back(make_operation(
            "mul_dB_" + std::to_string(ctx.op_counter++),
            "multiply",
            "multiply",
            {ctx.d_output, saved_ref(A)},
            {ctx.d_inputs[1]}));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// RMSNorm backward rule
// Forward: y, rstd = rmsnorm(x, weight, eps)
// Backward: dx, dweight = rmsnorm_backward(dy, x, weight, rstd, ...)
// -----------------------------------------------------------------------------
std::vector<Operation> rmsnorm_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;

    // Forward inputs: x, weight
    // Forward outputs: y, rstd
    std::string x = fwd.inputs[0];
    std::string weight = fwd.inputs[1];
    std::string rstd = fwd.outputs.size() > 1 ? fwd.outputs[1] : fwd.outputs[0] + "_rstd";

    // Carry forward eps attribute
    AttrMap attrs = copy_attrs(fwd.attrs, {"eps"});

    // Outputs: dx, dweight
    std::vector<std::string> outputs;
    if (ctx.needs_grad(0)) {
        outputs.push_back(ctx.d_inputs[0]);
    } else {
        outputs.push_back(""); // placeholder
    }
    if (ctx.needs_grad(1)) {
        outputs.push_back(ctx.d_inputs[1]);
    }

    ops.push_back(make_operation(
        "rmsnorm_backward_" + std::to_string(ctx.op_counter++),
        "rmsnorm_backward",
        "rmsnorm_backward",
        {ctx.d_output, saved_ref(x), weight, saved_ref(rstd)},
        outputs,
        attrs));

    return ops;
}

// -----------------------------------------------------------------------------
// Fused residual RMSNorm backward rule
// Forward: residual_out, y, rstd = fused_residual_rmsnorm(residual_in, x, weight, eps)
// Backward: d_residual, d_x, d_weight = fused_residual_rmsnorm_backward(...)
// -----------------------------------------------------------------------------
std::vector<Operation> fused_residual_rmsnorm_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;

    // Forward inputs: residual_in, x, weight
    // Forward outputs: residual_out, y, rstd
    std::string residual_out = fwd.outputs[0];
    std::string rstd = fwd.outputs.size() > 2 ? fwd.outputs[2] : fwd.outputs[0] + "_rstd";
    std::string weight = fwd.inputs[2];

    AttrMap attrs = copy_attrs(fwd.attrs, {"eps"});

    // The backward kernel consumes gradients for both outputs:
    //  - d_y: gradient of normalized output (y)
    //  - d_residual_next: gradient flowing from residual_out
    std::string d_residual_next = ctx.d_outputs.size() > 0 ? ctx.d_outputs[0] : "";
    std::string d_y;
    if (ctx.d_outputs.size() > 1 && !ctx.d_outputs[1].empty()) {
        d_y = ctx.d_outputs[1];
    } else {
        d_y = ctx.d_output;
    }

    std::vector<std::string> inputs = {
        d_y,
        d_residual_next,
        saved_ref(residual_out),
        weight,
        saved_ref(rstd)
    };

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");  // d_residual
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");  // d_x
    if (ctx.needs_grad(2)) {
        outputs.push_back(ctx.d_inputs[2]);  // d_weight
    }

    ops.push_back(make_operation(
        "fused_residual_rmsnorm_backward_" + std::to_string(ctx.op_counter++),
        "fused_residual_rmsnorm_backward",
        "fused_residual_rmsnorm_backward",
        inputs,
        outputs,
        attrs));

    return ops;
}

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
        ops.push_back(make_operation(
            "embedding_backward_" + std::to_string(ctx.op_counter++),
            "embedding_backward",
            "embedding_backward",
            {ctx.d_output, saved_ref(token_ids)},
            {ctx.d_inputs[1]}));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// SiLU backward rule
// Forward: y = silu(x) = x * sigmoid(x)
// Backward: dx = dy * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
//             = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
// -----------------------------------------------------------------------------
std::vector<Operation> silu_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string x = fwd.inputs[0];

        ops.push_back(make_operation(
            "silu_backward_" + std::to_string(ctx.op_counter++),
            "silu_backward",
            "silu_backward",
            {ctx.d_output, saved_ref(x)},
            {ctx.d_inputs[0]}));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// GELU backward rule
// Forward: y = gelu(x)
// Backward: dx = dy * gelu'(x)
// -----------------------------------------------------------------------------
std::vector<Operation> gelu_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string x = fwd.inputs[0];

        ops.push_back(make_operation(
            "gelu_backward_" + std::to_string(ctx.op_counter++),
            "gelu_backward",
            "gelu_backward",
            {ctx.d_output, saved_ref(x)},
            {ctx.d_inputs[0]}));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// SwiGLU backward rule
// Forward: out = swiglu(gate, up) = silu(gate) * up
// Backward: d_gate, d_up = swiglu_backward(d_out, gate, up)
// -----------------------------------------------------------------------------
std::vector<Operation> swiglu_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    // DSL swiglu takes a single gate_up input (packed) -> output
    if (fwd.inputs.size() == 1) {
        std::string gate_up = fwd.inputs[0];
        std::vector<std::string> outputs;
        outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
        ops.push_back(make_operation(
            "swiglu_backward_" + std::to_string(ctx.op_counter++),
            "swiglu_backward",
            "swiglu_backward",
            {ctx.d_output, saved_ref(gate_up)},
            outputs));
        return ops;
    }

    // Legacy form: swiglu(gate, up)
    std::string gate = fwd.inputs[0];
    std::string up = fwd.inputs[1];

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");

    ops.push_back(make_operation(
        "swiglu_backward_" + std::to_string(ctx.op_counter++),
        "swiglu_backward",
        "swiglu_backward",
        {ctx.d_output, saved_ref(gate), saved_ref(up)},
        outputs));

    return ops;
}

// -----------------------------------------------------------------------------
// BiasAdd backward rule
// Forward: y = bias_add(x, bias)
// Backward: dx = dy, d_bias = sum(dy)
// -----------------------------------------------------------------------------
std::vector<Operation> bias_add_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");

    std::vector<std::string> inputs;
    inputs.push_back(ctx.d_output);
    if (fwd.inputs.size() > 1) {
        inputs.push_back(fwd.inputs[1]);
    }

    ops.push_back(make_operation(
        "bias_add_backward_" + std::to_string(ctx.op_counter++),
        "bias_add_backward",
        "bias_add_backward",
        inputs,
        outputs));

    return ops;
}

// -----------------------------------------------------------------------------
// RoPE backward rule
// Forward: q_out, k_out = rope(q, k, cos, sin, position_ids)
// Backward: dq, dk = rope_backward(dq_out, dk_out, cos, sin, position_ids)
// -----------------------------------------------------------------------------
std::vector<Operation> rope_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    // DSL rope: out = rope(qkv, freqs, position_ids)
    if (fwd.inputs.size() >= 3) {
        std::string freqs = fwd.inputs[1];
        std::string pos_ids = fwd.inputs[2];
        std::vector<std::string> outputs;
        outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");

        AttrMap attrs = copy_attrs(fwd.attrs, {"rotary_dim"});

        ops.push_back(make_operation(
            "rope_backward_" + std::to_string(ctx.op_counter++),
            "rope_backward",
            "rope_backward",
            {ctx.d_output, freqs, pos_ids},
            outputs,
            attrs));
        return ops;
    }

    // Legacy form: q, k, cos, sin, position_ids
    std::string cos_cache = fwd.inputs.size() > 2 ? fwd.inputs[2] : "cos_cache";
    std::string sin_cache = fwd.inputs.size() > 3 ? fwd.inputs[3] : "sin_cache";
    std::string pos_ids = fwd.inputs.size() > 4 ? fwd.inputs[4] : "position_ids";

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");  // dq
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");  // dk

    AttrMap attrs = copy_attrs(fwd.attrs, {"head_dim", "rope_theta", "rope_scaling"});

    ops.push_back(make_operation(
        "rope_backward_" + std::to_string(ctx.op_counter++),
        "rope_backward",
        "rope_backward",
        {ctx.d_output, cos_cache, sin_cache, pos_ids},
        outputs,
        attrs));

    return ops;
}

// -----------------------------------------------------------------------------
// FlashAttention backward rule
// Forward: out, lse = flash_attention(qkv)
// Backward: d_qkv = flash_attention_backward(d_out, out, lse, qkv)
// -----------------------------------------------------------------------------
std::vector<Operation> flash_attention_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    std::string out = fwd.outputs.empty() ? "out" : fwd.outputs[0];
    std::string lse = fwd.outputs.size() > 1 ? fwd.outputs[1] : out + "_lse";
    std::string qkv = fwd.inputs.empty() ? "qkv" : fwd.inputs[0];

    AttrMap attrs = copy_attrs(fwd.attrs, {"causal", "softmax_scale", "window_size"});

    ops.push_back(make_operation(
        "flash_attention_backward_" + std::to_string(ctx.op_counter++),
        "flash_attention_backward",
        "flash_attention_backward",
        {ctx.d_output, saved_ref(out), saved_ref(lse), saved_ref(qkv)},
        {ctx.needs_grad(0) ? ctx.d_inputs[0] : ""},
        attrs));

    return ops;
}

// -----------------------------------------------------------------------------
// QK-Norm + RoPE backward rule
// Forward: qkv_out, q_rstd, k_rstd = qkv_qk_norm_rope(qkv, q_norm_w, k_norm_w, freqs, pos_ids)
// Backward: d_qkv, d_q_norm_w, d_k_norm_w = qkv_qk_norm_rope_backward(...)
// -----------------------------------------------------------------------------
std::vector<Operation> qkv_qk_norm_rope_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    if (fwd.inputs.size() < 5 || fwd.outputs.size() < 3) {
        return ops;
    }

    std::string qkv_out = fwd.outputs[0];
    std::string q_rstd = fwd.outputs[1];
    std::string k_rstd = fwd.outputs[2];

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");
    outputs.push_back(ctx.needs_grad(2) ? ctx.d_inputs[2] : "");

    ops.push_back(make_operation(
        "qkv_qk_norm_rope_backward_" + std::to_string(ctx.op_counter++),
        "qkv_qk_norm_rope_backward",
        "qkv_qk_norm_rope_backward",
        {ctx.d_output,
         saved_ref(qkv_out),
         fwd.inputs[1], fwd.inputs[2],
         saved_ref(q_rstd), saved_ref(k_rstd),
         fwd.inputs[3], fwd.inputs[4]},
        outputs));

    return ops;
}
// -----------------------------------------------------------------------------
// Attention backward rule
// Forward: out = attention(q, k, v, mask?)
// Backward: dq, dk, dv = attention_backward(d_out, q, k, v, out, softmax_lse, ...)
// -----------------------------------------------------------------------------
std::vector<Operation> attention_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;

    std::string q = fwd.inputs[0];
    std::string k = fwd.inputs[1];
    std::string v = fwd.inputs[2];
    std::string out = fwd.outputs[0];
    // Attention typically also saves softmax_lse
    std::string lse = fwd.outputs.size() > 1 ? fwd.outputs[1] : out + "_lse";

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");  // dq
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");  // dk
    outputs.push_back(ctx.needs_grad(2) ? ctx.d_inputs[2] : "");  // dv

    AttrMap attrs = copy_attrs(fwd.attrs, {"scale", "causal", "window_size"});

    ops.push_back(make_operation(
        "attention_backward_" + std::to_string(ctx.op_counter++),
        "attention_backward",
        "attention_backward",
        {ctx.d_output, saved_ref(q), saved_ref(k), saved_ref(v), saved_ref(out), saved_ref(lse)},
        outputs,
        attrs));

    return ops;
}

// -----------------------------------------------------------------------------
// Softmax backward rule
// Forward: y = softmax(x)
// Backward: dx = y * (dy - sum(dy * y))
// -----------------------------------------------------------------------------
std::vector<Operation> softmax_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string y = fwd.outputs[0];

        AttrMap attrs = copy_attrs(fwd.attrs, {"dim"});

        ops.push_back(make_operation(
            "softmax_backward_" + std::to_string(ctx.op_counter++),
            "softmax_backward",
            "softmax_backward",
            {ctx.d_output, saved_ref(y)},
            {ctx.d_inputs[0]},
            attrs));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// View/Reshape backward rule (no-op, just reshape gradient)
// Forward: y = view(x, shape)
// Backward: dx = view(dy, original_shape)
// -----------------------------------------------------------------------------
std::vector<Operation> view_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;

        // Need to reshape gradient back to input shape.
        // If the forward input is a parameter or graph input, we can reference it without saving.
        // Otherwise, we need to save it for its shape (or use shape_like).
        AttrMap attrs;
        const std::string& fwd_input = fwd.inputs[0];

        // Check if the forward input is a parameter (available at backward time) or an input
        if (ctx.is_param(fwd_input) || ctx.is_input(fwd_input)) {
            // Use the tensor directly (it's available at backward time)
            attrs["shape_like"] = AttrValue{fwd_input};
        } else {
            // Need to save the tensor for its shape
            attrs["shape_like"] = AttrValue{saved_ref(fwd_input)};
        }

        ops.push_back(make_operation(
            "view_backward_" + std::to_string(ctx.op_counter++),
            "view",
            "view",
            {ctx.d_output},
            {ctx.d_inputs[0]},
            attrs));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// Zeros - no backward (constant has zero gradient)
// -----------------------------------------------------------------------------
std::vector<Operation> zeros_backward(const BackwardRuleContext& ctx) {
    // No operations needed - gradient of a constant is zero
    return {};
}

// -----------------------------------------------------------------------------
// Identity/Copy backward
// Forward: y = x
// Backward: dx = dy
// -----------------------------------------------------------------------------
std::vector<Operation> identity_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        ops.push_back(make_operation(
            "identity_backward_" + std::to_string(ctx.op_counter++),
            "identity",
            "identity",
            {ctx.d_output},
            {ctx.d_inputs[0]}));
    }

    return ops;
}

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

        ops.push_back(make_operation(
            "cross_entropy_backward_" + std::to_string(ctx.op_counter++),
            "cross_entropy_backward",
            "cross_entropy_backward",
            {ctx.d_output, saved_ref(logits), targets},
            {ctx.d_inputs[0]}));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// StackedBlocks - compound operation for transformer blocks
// This is a meta-op that doesn't decompose into individual layer backwards
// -----------------------------------------------------------------------------
std::vector<Operation> stacked_blocks_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    // StackedBlocks is handled as a unit - generates StackedBlocksBackward
    std::vector<std::string> outputs;
    for (size_t i = 0; i < ctx.d_inputs.size(); ++i) {
        outputs.push_back(ctx.needs_grad(i) ? ctx.d_inputs[i] : "");
    }

    AttrMap attrs = ctx.fwd_op.attrs;  // Carry all attributes

    ops.push_back(make_operation(
        "StackedBlocksBackward",
        "StackedBlocksBackward",
        "StackedBlocksBackward",
        {ctx.d_output, ctx.d_output},  // d_output for both mlp_down and residual
        outputs,
        attrs));

    return ops;
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// Register all built-in rules
// -----------------------------------------------------------------------------

void register_builtin_backward_rules() {
    auto& reg = BackwardRuleRegistry::instance();

    // Core ops
    reg.register_rule("matmul", matmul_backward);
    reg.register_rule("add", add_backward);
    reg.register_rule("multiply", multiply_backward);
    reg.register_rule("mul", multiply_backward);

    // Normalization
    reg.register_rule("rmsnorm", rmsnorm_backward);
    reg.register_rule("fused_residual_rmsnorm", fused_residual_rmsnorm_backward);

    // Embeddings
    reg.register_rule("embedding", embedding_backward);

    // Activations
    reg.register_rule("silu", silu_backward);
    reg.register_rule("gelu", gelu_backward);
    reg.register_rule("swiglu", swiglu_backward);
    reg.register_rule("bias_add", bias_add_backward);

    // Attention
    reg.register_rule("rope", rope_backward);
    reg.register_rule("qkv_qk_norm_rope", qkv_qk_norm_rope_backward);
    reg.register_rule("flash_attention", flash_attention_backward);
    reg.register_rule("flash_attention_qkv", flash_attention_backward);
    reg.register_rule("attention", attention_backward);
    reg.register_rule("scaled_dot_product_attention", attention_backward);
    reg.register_rule("softmax", softmax_backward);

    // Tensor ops
    reg.register_rule("view", view_backward);
    reg.register_rule("reshape", view_backward);
    reg.register_rule("zeros", zeros_backward);
    reg.register_rule("identity", identity_backward);
    reg.register_rule("copy", identity_backward);

    // Loss
    reg.register_rule("cross_entropy", cross_entropy_backward);
    reg.register_rule("cross_entropy_loss", cross_entropy_backward);

    // Compound ops (handled as units)
    reg.register_rule("StackedBlocks", stacked_blocks_backward);
}

} // namespace dsl
