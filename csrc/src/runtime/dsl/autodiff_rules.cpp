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

// Helper to copy attributes from forward op to backward op.
// Warns when a requested key is missing — this catches attr name mismatches
// between forward IR and backward rules (e.g., "mamba_num_heads" vs "num_heads").
AttrMap copy_attrs(const AttrMap& src, const std::vector<std::string>& keys,
                   const char* rule_name = nullptr) {
    AttrMap dst;
    for (const auto& key : keys) {
        if (auto* attr = find_attr(src, key)) {
            dst[key] = *attr;
        } else if (rule_name) {
            fprintf(stderr,
                    "WARNING [autodiff]: backward rule '%s' requested attr '%s' "
                    "not found in forward op attrs\n",
                    rule_name, key.c_str());
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

    // Determine references for A and B in backward pass:
    // - Parameters are available at backward time (gathered from weight manager)
    // - Activations must be saved from forward pass (use saved_ref)
    std::string A_for_dB = ctx.is_param(A) ? A : saved_ref(A);
    std::string B_for_dA = ctx.is_param(B) ? B : saved_ref(B);


    AttrMap attrs;
    attrs["transpose"] = AttrValue(trans);

    std::vector<std::string> inputs = {dC, A_for_dB, B_for_dA};
    std::vector<std::string> outputs = {ctx.d_inputs[0], ctx.d_inputs[1]};

    ops.push_back(make_operation(
        "matmul_backward_" + std::to_string(ctx.op_counter++),
        "matmul_backward",
        "matmul_backward",
        inputs,
        outputs,
        attrs));

    return ops;
}

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
    ops.push_back(make_operation(
        "matmul_bias_backward_" + std::to_string(ctx.op_counter++),
        "matmul_backward",
        "matmul_backward",
        inputs,
        outputs,
        attrs));

    if (ctx.needs_grad(2) && !bias.empty()) {
        std::vector<std::string> outputs;
        outputs.push_back("");
        outputs.push_back(ctx.d_inputs[2]);
        ops.push_back(make_operation(
            "matmul_bias_dBias_" + std::to_string(ctx.op_counter++),
            "bias_add_backward",
            "bias_add_backward",
            {dC, bias},
            outputs));
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
    // For now, assume same shapes. Emit a single add_backward op so compiled
    // executor can copy into both base gradients in one place.
    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");
    if (ctx.needs_grad(0) || ctx.needs_grad(1)) {
        ops.push_back(make_operation(
            "add_backward_" + std::to_string(ctx.op_counter++),
            "add_backward",
            "add_backward",
            {ctx.d_output},
            outputs));
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
// Masked scatter backward rule
// Forward: out = mask_scatter(x, mask, src)
// Backward: d_x, d_src (mask is non-differentiable)
// -----------------------------------------------------------------------------
std::vector<Operation> mask_scatter_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;
    const auto& fwd = ctx.fwd_op;
    if (fwd.inputs.size() < 2) {
        return ops;
    }
    const std::string& mask = fwd.inputs[1];

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back("");  // mask has no gradient
    outputs.push_back(ctx.needs_grad(2) ? ctx.d_inputs[2] : "");

    if (outputs[0].empty() && outputs[2].empty()) {
        return ops;
    }

    ops.push_back(make_operation(
        "mask_scatter_backward_" + std::to_string(ctx.op_counter++),
        "mask_scatter_backward",
        "mask_scatter_backward",
        {ctx.d_output, mask},
        outputs));
    return ops;
}

// -----------------------------------------------------------------------------
// Deepstack inject backward rule
// Forward: out = deepstack_inject(x, mask, src)  (adds src at masked positions)
// Backward: d_x, d_src (mask is non-differentiable)
// -----------------------------------------------------------------------------
std::vector<Operation> deepstack_inject_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;
    const auto& fwd = ctx.fwd_op;
    if (fwd.inputs.size() < 2) {
        return ops;
    }
    const std::string& mask = fwd.inputs[1];

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back("");  // mask has no gradient
    outputs.push_back(ctx.needs_grad(2) ? ctx.d_inputs[2] : "");

    if (outputs[0].empty() && outputs[2].empty()) {
        return ops;
    }

    ops.push_back(make_operation(
        "deepstack_inject_backward_" + std::to_string(ctx.op_counter++),
        "deepstack_inject_backward",
        "deepstack_inject_backward",
        {ctx.d_output, mask},
        outputs));
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
// ReLU² backward rule
// Forward: y = relu2(x) = x * relu(x) = x * max(0, x)
// For x <= 0: y = 0
// For x > 0:  y = x²
// Backward: dx = dy * 2 * relu(x)
// -----------------------------------------------------------------------------
std::vector<Operation> relu2_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string x = fwd.inputs[0];

        ops.push_back(make_operation(
            "relu2_backward_" + std::to_string(ctx.op_counter++),
            "relu2_backward",
            "relu2_backward",
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
// MRoPE backward rule
// Forward: out = mrope(qkv, freqs, position_ids)
// Backward: d_qkv = mrope_backward(d_out, freqs, position_ids)
// -----------------------------------------------------------------------------
std::vector<Operation> mrope_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    if (fwd.inputs.size() >= 3) {
        std::string freqs = fwd.inputs[1];
        std::string pos_ids = fwd.inputs[2];

        std::vector<std::string> outputs;
        outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");

        AttrMap attrs = copy_attrs(fwd.attrs, {"rotary_dim", "mrope_section"});

        ops.push_back(make_operation(
            "mrope_backward_" + std::to_string(ctx.op_counter++),
            "mrope_backward",
            "mrope_backward",
            {ctx.d_output, freqs, pos_ids},
            outputs,
            attrs));
    }

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

    std::vector<std::string> inputs = {ctx.d_output, saved_ref(out), saved_ref(lse), saved_ref(qkv)};
    bool has_sinks = (fwd.inputs.size() > 1 && !fwd.inputs[1].empty());
    if (has_sinks) {
        inputs.push_back(saved_ref(fwd.inputs[1]));
    }

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    if (has_sinks) {
        outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");
    }

    ops.push_back(make_operation(
        "flash_attention_backward_" + std::to_string(ctx.op_counter++),
        "flash_attention_backward",
        "flash_attention_backward",
        inputs,
        outputs,
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

    // The backward kernel expects qkv_rope (the OUTPUT after QK-Norm + RoPE), NOT the original input.
    // The kernel internally applies inverse RoPE to recover the pre-RoPE values for gradient computation.
    std::string qkv_out = fwd.outputs[0];  // qkv_rope - output after QK-Norm + RoPE
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
         saved_ref(qkv_out),  // Use OUTPUT qkv_rope - kernel applies inverse RoPE internally
         fwd.inputs[1], fwd.inputs[2],
         saved_ref(q_rstd), saved_ref(k_rstd),
         fwd.inputs[3], fwd.inputs[4]},
        outputs));

    return ops;
}

// -----------------------------------------------------------------------------
// QK-Norm backward rule (no RoPE)
// Forward: qkv_out, q_rstd, k_rstd = qkv_qk_norm(qkv, q_norm_w, k_norm_w)
// Backward: d_qkv, d_q_norm_w, d_k_norm_w = qkv_qk_norm_backward(...)
// -----------------------------------------------------------------------------
std::vector<Operation> qkv_qk_norm_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    if (fwd.inputs.size() < 3 || fwd.outputs.size() < 3) {
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
        "qkv_qk_norm_backward_" + std::to_string(ctx.op_counter++),
        "qkv_qk_norm_backward",
        "qkv_qk_norm_backward",
        {ctx.d_output,
         saved_ref(qkv_out),
         fwd.inputs[1], fwd.inputs[2],
         saved_ref(q_rstd), saved_ref(k_rstd)},
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

    ops.push_back(make_operation(
        "fused_lm_head_loss_backward_" + std::to_string(ctx.op_counter++),
        "fused_lm_head_loss_backward",
        "fused_lm_head_loss_backward",
        inputs,
        outputs));

    return ops;
}

// -----------------------------------------------------------------------------
// MoE Sigmoid backward rule
// Forward: probs = moe_sigmoid(logits)
// Backward: d_logits = d_probs * probs * (1 - probs)
// -----------------------------------------------------------------------------
std::vector<Operation> moe_sigmoid_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string out = fwd.outputs.empty() ? "out" : fwd.outputs[0];

        ops.push_back(make_operation(
            "moe_sigmoid_backward_" + std::to_string(ctx.op_counter++),
            "moe_sigmoid_backward",
            "moe_sigmoid_backward",
            {ctx.d_output, saved_ref(out)},
            {ctx.d_inputs[0]}));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// MoE Softmax backward rule
// Forward: probs = moe_softmax(logits)
// Backward: d_logits = softmax_backward(d_probs, probs)
// -----------------------------------------------------------------------------
std::vector<Operation> moe_softmax_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string out = fwd.outputs.empty() ? "out" : fwd.outputs[0];

        ops.push_back(make_operation(
            "moe_softmax_backward_" + std::to_string(ctx.op_counter++),
            "moe_softmax_backward",
            "moe_softmax_backward",
            {ctx.d_output, saved_ref(out)},
            {ctx.d_inputs[0]}));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// MoE TopK backward rule
// Forward: weights, indices = moe_topk(probs, top_k, normalize)
// Backward: d_probs = scatter d_weights to positions indicated by indices
// Note: indices is not differentiable (discrete selection)
// -----------------------------------------------------------------------------
std::vector<Operation> moe_topk_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string probs = fwd.inputs[0];
        std::string indices = fwd.outputs.size() > 1 ? fwd.outputs[1] : "indices";

        AttrMap attrs = copy_attrs(fwd.attrs, {"top_k", "normalize", "scaling_factor", "softmax"});

        ops.push_back(make_operation(
            "moe_topk_backward_" + std::to_string(ctx.op_counter++),
            "moe_topk_backward",
            "moe_topk_backward",
            {ctx.d_outputs[0], saved_ref(probs), saved_ref(indices)},
            {ctx.d_inputs[0]},
            attrs));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// GPT-OSS MoE activation backward rule
// Forward: out = gpt_oss_moe_act(inp, alpha, limit)
// Backward: d_inp = gpt_oss_moe_act_backward(d_out, inp)
// -----------------------------------------------------------------------------
std::vector<Operation> gpt_oss_moe_act_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;
    if (!ctx.needs_grad(0)) {
        return ops;
    }
    const auto& fwd = ctx.fwd_op;
    if (fwd.inputs.empty()) {
        return ops;
    }
    std::string inp = fwd.inputs[0];
    AttrMap attrs = copy_attrs(fwd.attrs, {"alpha", "limit"});
    ops.push_back(make_operation(
        "gpt_oss_moe_act_backward_" + std::to_string(ctx.op_counter++),
        "gpt_oss_moe_act_backward",
        "gpt_oss_moe_act_backward",
        {ctx.d_output, saved_ref(inp)},
        {ctx.d_inputs[0]},
        attrs));
    return ops;
}

// -----------------------------------------------------------------------------
// MoE expert bias add backward rule
// Forward: out = moe_expert_bias_add(x, bias)
// Backward: d_x, d_bias = moe_expert_bias_add_backward(d_out, bias)
// -----------------------------------------------------------------------------
std::vector<Operation> moe_expert_bias_add_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;
    const auto& fwd = ctx.fwd_op;
    if (fwd.inputs.empty()) {
        return ops;
    }

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    if (fwd.inputs.size() > 1) {
        outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");
    }

    std::vector<std::string> inputs = {ctx.d_output};
    if (fwd.inputs.size() > 1) {
        inputs.push_back(fwd.inputs[1]);
    }

    ops.push_back(make_operation(
        "moe_expert_bias_add_backward_" + std::to_string(ctx.op_counter++),
        "moe_expert_bias_add_backward",
        "moe_expert_bias_add_backward",
        inputs,
        outputs));
    return ops;
}

// -----------------------------------------------------------------------------
// MoE Permute backward rule
// Forward: permuted, scatter_indices = moe_permute(x, routing_indices, top_k)
// Backward: d_x = moe_permute_backward(d_permuted, scatter_indices)
// Note: routing_indices is not differentiable
// -----------------------------------------------------------------------------
std::vector<Operation> moe_permute_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string scatter_indices = fwd.outputs.size() > 1 ? fwd.outputs[1] : "scatter_indices";

        AttrMap attrs = copy_attrs(fwd.attrs, {"top_k"});

        ops.push_back(make_operation(
            "moe_permute_backward_" + std::to_string(ctx.op_counter++),
            "moe_permute_backward",
            "moe_permute_backward",
            {ctx.d_outputs[0], saved_ref(scatter_indices)},
            {ctx.d_inputs[0]},
            attrs));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// MoE Grouped GEMM Gate+Up backward rule
// Forward: out = moe_grouped_gemm_gate_up(inp, weights, scatter_indices)
// Backward: d_inp = moe_grouped_gemm_gate_up_backward(d_out, inp, weights, scatter_indices)
// Note: weights gradient is computed but not propagated (frozen expert weights)
// -----------------------------------------------------------------------------
std::vector<Operation> moe_grouped_gemm_gate_up_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string inp = fwd.inputs[0];
        std::string weights = fwd.inputs[1];
        std::string scatter_indices = fwd.inputs[2];

        std::string inp_ref = ctx.is_param(inp) ? inp : saved_ref(inp);
        std::string weights_ref = ctx.is_param(weights) ? weights : saved_ref(weights);
        std::string scatter_ref = saved_ref(scatter_indices);
        AttrMap attrs = copy_attrs(fwd.attrs, {"gate_up_interleaved"});

        ops.push_back(make_operation(
            "moe_grouped_gemm_gate_up_backward_" + std::to_string(ctx.op_counter++),
            "moe_grouped_gemm_gate_up_backward",
            "moe_grouped_gemm_gate_up_backward",
            {ctx.d_output, inp_ref, weights_ref, scatter_ref},
            {ctx.d_inputs[0]},
            attrs));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// MoE Grouped GEMM Down backward rule
// Forward: out = moe_grouped_gemm_down(inp, weights, scatter_indices)
// Backward: d_inp = moe_grouped_gemm_down_backward(d_out, inp, weights, scatter_indices)
// Note: weights gradient is computed but not propagated (frozen expert weights)
// -----------------------------------------------------------------------------
std::vector<Operation> moe_grouped_gemm_down_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string inp = fwd.inputs[0];
        std::string weights = fwd.inputs[1];
        std::string scatter_indices = fwd.inputs[2];

        std::string inp_ref = ctx.is_param(inp) ? inp : saved_ref(inp);
        std::string weights_ref = ctx.is_param(weights) ? weights : saved_ref(weights);
        std::string scatter_ref = saved_ref(scatter_indices);

        ops.push_back(make_operation(
            "moe_grouped_gemm_down_backward_" + std::to_string(ctx.op_counter++),
            "moe_grouped_gemm_down_backward",
            "moe_grouped_gemm_down_backward",
            {ctx.d_output, inp_ref, weights_ref, scatter_ref},
            {ctx.d_inputs[0]}));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// MoE Grouped GEMM backward rule (generic version without fused activation)
// Forward: out = moe_grouped_gemm(inp, weights, scatter_indices)
// Backward: d_inp = moe_grouped_gemm_backward(d_out, inp, weights, scatter_indices)
// Used for Nemotron-H MoE blocks that use relu2 activation instead of swiglu
// -----------------------------------------------------------------------------
std::vector<Operation> moe_grouped_gemm_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string inp = fwd.inputs[0];
        std::string weights = fwd.inputs[1];
        std::string scatter_indices = fwd.inputs[2];

        std::string inp_ref = ctx.is_param(inp) ? inp : saved_ref(inp);
        std::string weights_ref = ctx.is_param(weights) ? weights : saved_ref(weights);
        std::string scatter_ref = saved_ref(scatter_indices);

        ops.push_back(make_operation(
            "moe_grouped_gemm_backward_" + std::to_string(ctx.op_counter++),
            "moe_grouped_gemm_backward",
            "moe_grouped_gemm_backward",
            {ctx.d_output, inp_ref, weights_ref, scatter_ref},
            {ctx.d_inputs[0]}));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// MoE Unpermute backward rule
// Forward: out = moe_unpermute(expert_out, routing_weights, scatter_indices, top_k)
// Backward: d_expert_out, d_routing_weights = moe_unpermute_backward(...)
// -----------------------------------------------------------------------------
std::vector<Operation> moe_unpermute_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    std::string expert_out = fwd.inputs[0];
    std::string routing_weights = fwd.inputs[1];
    std::string scatter_indices = fwd.inputs[2];

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");  // d_expert_out
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");  // d_routing_weights

    AttrMap attrs = copy_attrs(fwd.attrs, {"top_k"});

    ops.push_back(make_operation(
        "moe_unpermute_backward_" + std::to_string(ctx.op_counter++),
        "moe_unpermute_backward",
        "moe_unpermute_backward",
        {ctx.d_output, saved_ref(expert_out), saved_ref(routing_weights), saved_ref(scatter_indices)},
        outputs,
        attrs));

    return ops;
}

// -----------------------------------------------------------------------------
// Mamba split_proj backward rule
// Forward: gate, conv_in, delta = mamba_split_proj(projected)
// Backward: d_projected = mamba_split_proj_backward(d_gate, d_conv_in, d_delta)
// -----------------------------------------------------------------------------
std::vector<Operation> mamba_split_proj_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;

        AttrMap attrs = copy_attrs(fwd.attrs, {"intermediate_size", "conv_dim", "num_heads", "head_dim"},
                                    "mamba_split_proj");

        // d_outputs[0..2] are the gradients of the 3 forward outputs: gate, conv_in, delta
        // d_inputs[0] is where to write the gradient of the forward input: projected
        ops.push_back(make_operation(
            "mamba_split_proj_backward_" + std::to_string(ctx.op_counter++),
            "mamba_split_proj_backward",
            "mamba_split_proj_backward",
            {ctx.d_outputs[0], ctx.d_outputs[1], ctx.d_outputs[2]},
            {ctx.d_inputs[0]},
            attrs));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// Mamba conv1d backward rule
// Forward: out = mamba_conv1d(x, weight, bias)
// Backward: dx, dweight, dbias = mamba_conv1d_backward(d_out, x, weight)
// -----------------------------------------------------------------------------
std::vector<Operation> mamba_conv1d_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    std::string x = fwd.inputs[0];
    std::string weight = fwd.inputs[1];
    std::string bias = (fwd.inputs.size() > 2) ? fwd.inputs[2] : "";

    std::string x_ref = ctx.is_param(x) ? x : saved_ref(x);
    std::string weight_ref = ctx.is_param(weight) ? weight : saved_ref(weight);

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");
    if (fwd.inputs.size() > 2 && ctx.needs_grad(2)) {
        outputs.push_back(ctx.d_inputs[2]);
    }

    AttrMap attrs = copy_attrs(fwd.attrs, {"activation"}, "mamba_conv1d");

    ops.push_back(make_operation(
        "mamba_conv1d_backward_" + std::to_string(ctx.op_counter++),
        "mamba_conv1d_backward",
        "mamba_conv1d_backward",
        {ctx.d_output, x_ref, weight_ref},
        outputs,
        attrs));

    return ops;
}

// -----------------------------------------------------------------------------
// Mamba split_conv_out backward rule
// Forward: u, B, C = mamba_split_conv_out(conv_out)
// Backward: d_conv_out = mamba_split_conv_out_backward(d_u, d_B, d_C)
// -----------------------------------------------------------------------------
std::vector<Operation> mamba_split_conv_out_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;

        AttrMap attrs = copy_attrs(fwd.attrs, {"intermediate_size", "n_groups", "ssm_state_size"},
                                    "mamba_split_conv_out");

        // d_outputs[0..2] are the gradients of the 3 forward outputs: u, B, C
        // d_inputs[0] is where to write the gradient of the forward input: conv_out
        ops.push_back(make_operation(
            "mamba_split_conv_out_backward_" + std::to_string(ctx.op_counter++),
            "mamba_split_conv_out_backward",
            "mamba_split_conv_out_backward",
            {ctx.d_outputs[0], ctx.d_outputs[1], ctx.d_outputs[2]},
            {ctx.d_inputs[0]},
            attrs));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// Mamba SSM scan backward rule
// Forward: out, ssm_state = mamba_ssm_scan(u, delta, A_log, B, C, D_param, dt_bias)
// Backward: du, ddelta, dA_log, dB, dC, dD, ddelta_bias = mamba_ssm_scan_backward(...)
// -----------------------------------------------------------------------------
std::vector<Operation> mamba_ssm_scan_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    std::string u = fwd.inputs[0];
    std::string delta = fwd.inputs[1];
    std::string A_log = fwd.inputs[2];
    std::string B_ssm = fwd.inputs[3];
    std::string C_ssm = fwd.inputs[4];
    std::string D_param = fwd.inputs[5];
    std::string dt_bias = (fwd.inputs.size() > 6) ? fwd.inputs[6] : "";

    std::string u_ref = ctx.is_param(u) ? u : saved_ref(u);
    std::string delta_ref = ctx.is_param(delta) ? delta : saved_ref(delta);
    std::string A_log_ref = ctx.is_param(A_log) ? A_log : saved_ref(A_log);
    std::string B_ssm_ref = ctx.is_param(B_ssm) ? B_ssm : saved_ref(B_ssm);
    std::string C_ssm_ref = ctx.is_param(C_ssm) ? C_ssm : saved_ref(C_ssm);
    std::string D_param_ref = ctx.is_param(D_param) ? D_param : saved_ref(D_param);
    std::string dt_bias_ref = dt_bias.empty() ? "" : (ctx.is_param(dt_bias) ? dt_bias : saved_ref(dt_bias));

    // ssm_state is the second output from forward, referenced via saved
    std::string ssm_state_ref = saved_ref(fwd.outputs[1]);

    std::vector<std::string> inputs = {
        ctx.d_output,  // d_out
        u_ref, delta_ref, A_log_ref, B_ssm_ref, C_ssm_ref, D_param_ref,
        dt_bias_ref, ssm_state_ref
    };

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");  // du
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");  // ddelta
    outputs.push_back(ctx.needs_grad(2) ? ctx.d_inputs[2] : "");  // dA_log
    outputs.push_back(ctx.needs_grad(3) ? ctx.d_inputs[3] : "");  // dB
    outputs.push_back(ctx.needs_grad(4) ? ctx.d_inputs[4] : "");  // dC
    outputs.push_back(ctx.needs_grad(5) ? ctx.d_inputs[5] : "");  // dD
    if (fwd.inputs.size() > 6 && ctx.needs_grad(6)) {
        outputs.push_back(ctx.d_inputs[6]);  // ddelta_bias
    }

    AttrMap attrs = copy_attrs(fwd.attrs, {"num_heads", "head_dim", "chunk_size",
                                           "ssm_state_size", "n_groups", "intermediate_size"},
                                "mamba_ssm_scan");

    ops.push_back(make_operation(
        "mamba_ssm_scan_backward_" + std::to_string(ctx.op_counter++),
        "mamba_ssm_scan_backward",
        "mamba_ssm_scan_backward",
        inputs,
        outputs,
        attrs));

    return ops;
}

// -----------------------------------------------------------------------------
// Mamba gated RMSNorm backward rule
// Forward: out = mamba_gated_rmsnorm(x, gate, weight)
// Backward: dx, dgate, dweight = mamba_gated_rmsnorm_backward(d_out, x, gate, weight, rstd, normed)
// Note: rstd and normed are saved internally by the forward op with op.op_id suffix
// -----------------------------------------------------------------------------
std::vector<Operation> mamba_gated_rmsnorm_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    std::string x = fwd.inputs[0];
    std::string gate = fwd.inputs[1];
    std::string weight = fwd.inputs[2];

    std::string x_ref = ctx.is_param(x) ? x : saved_ref(x);
    std::string gate_ref = ctx.is_param(gate) ? gate : saved_ref(gate);
    std::string weight_ref = ctx.is_param(weight) ? weight : saved_ref(weight);

    // rstd and normed are saved with op_id prefix
    std::string rstd_ref = "saved." + fwd.id + ".rstd";
    std::string normed_ref = "saved." + fwd.id + ".normed";

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");  // dx
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");  // dgate
    outputs.push_back(ctx.needs_grad(2) ? ctx.d_inputs[2] : "");  // dweight

    AttrMap attrs = copy_attrs(fwd.attrs, {"eps", "n_groups"}, "mamba_gated_rmsnorm");

    ops.push_back(make_operation(
        "mamba_gated_rmsnorm_backward_" + std::to_string(ctx.op_counter++),
        "mamba_gated_rmsnorm_backward",
        "mamba_gated_rmsnorm_backward",
        {ctx.d_output, x_ref, gate_ref, weight_ref, rstd_ref, normed_ref},
        outputs,
        attrs));

    return ops;
}

// -----------------------------------------------------------------------------
// Mamba out_proj backward rule
// Forward: out = mamba_out_proj(inp, weight)
// This is just a matmul, so we delegate to matmul backward
// -----------------------------------------------------------------------------
std::vector<Operation> mamba_out_proj_backward(const BackwardRuleContext& ctx) {
    // Delegate to matmul backward since mamba_out_proj is just a matmul
    return matmul_backward(ctx);
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// Register all built-in rules
// -----------------------------------------------------------------------------

void register_builtin_backward_rules() {
    auto& reg = BackwardRuleRegistry::instance();

    // Core ops
    reg.register_rule("matmul", matmul_backward);
    reg.register_rule("matmul_bias", matmul_bias_backward);
    reg.register_rule("add", add_backward);
    reg.register_rule("multiply", multiply_backward);
    reg.register_rule("mul", multiply_backward);
    reg.register_rule("mask_scatter", mask_scatter_backward);
    reg.register_rule("deepstack_inject", deepstack_inject_backward);

    // Normalization
    reg.register_rule("rmsnorm", rmsnorm_backward);
    reg.register_rule("fused_residual_rmsnorm", fused_residual_rmsnorm_backward);

    // Embeddings
    reg.register_rule("embedding", embedding_backward);

    // Activations
    reg.register_rule("silu", silu_backward);
    reg.register_rule("gelu", gelu_backward);
    reg.register_rule("relu2", relu2_backward);
    reg.register_rule("swiglu", swiglu_backward);
    reg.register_rule("bias_add", bias_add_backward);

    // Attention
    reg.register_rule("rope", rope_backward);
    reg.register_rule("mrope", mrope_backward);
    reg.register_rule("qkv_qk_norm", qkv_qk_norm_backward);
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
    reg.register_rule("fused_lm_head_loss", fused_lm_head_loss_backward);
    reg.register_rule("lm_head_loss", fused_lm_head_loss_backward);

    // MoE ops
    reg.register_rule("moe_sigmoid", moe_sigmoid_backward);
    reg.register_rule("moe_softmax", moe_softmax_backward);
    reg.register_rule("moe_topk", moe_topk_backward);
    reg.register_rule("moe_permute", moe_permute_backward);
    reg.register_rule("moe_grouped_gemm_gate_up", moe_grouped_gemm_gate_up_backward);
    reg.register_rule("moe_grouped_gemm_down", moe_grouped_gemm_down_backward);
    reg.register_rule("moe_grouped_gemm", moe_grouped_gemm_backward);
    reg.register_rule("moe_unpermute", moe_unpermute_backward);
    reg.register_rule("gpt_oss_moe_act", gpt_oss_moe_act_backward);
    reg.register_rule("moe_expert_bias_add", moe_expert_bias_add_backward);

    // Compound ops (handled as units)
    reg.register_rule("StackedBlocks", stacked_blocks_backward);

    // Mamba/SSM ops
    reg.register_rule("mamba_split_proj", mamba_split_proj_backward);
    reg.register_rule("mamba_conv1d", mamba_conv1d_backward);
    reg.register_rule("mamba_split_conv_out", mamba_split_conv_out_backward);
    reg.register_rule("mamba_ssm_scan", mamba_ssm_scan_backward);
    reg.register_rule("mamba_gated_rmsnorm", mamba_gated_rmsnorm_backward);
    reg.register_rule("mamba_out_proj", mamba_out_proj_backward);
}

} // namespace dsl
