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

void CompiledExecutor::dispatch_rope(const CompiledOp& op) {
    Tensor& qkv = resolve_tensor(op.inputs[0]);
    Tensor& freqs = resolve_tensor(op.inputs[1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[2]);
    const std::vector<long> out_shape(qkv.Sizes.begin(), qkv.Sizes.begin() + qkv.Rank);
    Tensor out = ensure_output_tensor_or_persistent(ensure_output_tensor(op.outputs[0]),
                                                    mRunState,
                                                    mMoeSavedBuffers,
                                                    mMoeSavedSizes,
                                                    op.op_id + "." + op.outputs[0].name + ".out",
                                                    qkv.DType,
                                                    out_shape,
                                                    "rope");

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = derive_head_size(qkv, Hq, Hkv, static_cast<int>(mConfig.head_size()));

    // Derive actual Hkv from tensor shape to handle Q-only inputs (shared-KV)
    // and other cases where the tensor has fewer heads than global config.
    if (qkv.Rank == 4) {
        const int actual_heads = static_cast<int>(qkv.Sizes[2]);
        if (actual_heads < Hq + 2 * Hkv) {
            Hkv = std::max(0, (actual_heads - Hq) / 2);
        }
    }

    if (mForwardPlan) {
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(op.inputs[0].name, layer_idx, field) && layer_idx >= 0 &&
            static_cast<std::size_t>(layer_idx) < mForwardPlan->size()) {
            AttnForwardPlan plan{};
            plan.valid = true;
            plan.use_qk_norm = false;
            plan.rope_fused = false;
            plan.use_cudnn = true;
            plan.rotary_dim = op.attrs.rotary_dim;
            (*mForwardPlan)[static_cast<std::size_t>(layer_idx)].attn = plan;
        }
    }

    rope_forward(out,
                 qkv,
                 freqs,
                 reinterpret_cast<int*>(pos_ids.Data),
                 nullptr,
                 static_cast<int>(mB),
                 static_cast<int>(mT),
                 Hq,
                 Hkv,
                 Hs,
                 op.attrs.rotary_dim,
                 mRunState.MainStream);
    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_rope_backward(const CompiledOp& op) {
    // inputs: d_qkv_rope, freq_cis, position_ids
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& freqs = resolve_tensor(op.inputs[1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[2]);
    const std::vector<long> d_qkv_shape(d_out.Sizes.begin(), d_out.Sizes.begin() + d_out.Rank);
    Tensor d_qkv = ensure_output_tensor_or_persistent(ensure_output_tensor(op.outputs[0]),
                                                      mRunState,
                                                      mMoeSavedBuffers,
                                                      mMoeSavedSizes,
                                                      op.op_id + "." + op.outputs[0].name + ".d_qkv",
                                                      d_out.DType,
                                                      d_qkv_shape,
                                                      "rope_backward");

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = derive_head_size(d_out, Hq, Hkv, static_cast<int>(mConfig.head_size()));

    // Derive actual Hkv from tensor shape to handle shared-KV Q-only gradients
    // and other cases where the gradient has fewer heads than the global config.
    if (d_out.Rank == 4) {
        const int actual_heads = static_cast<int>(d_out.Sizes[2]);
        if (actual_heads < Hq + 2 * Hkv) {
            // Fewer heads than expected — adjust Hkv to match.
            // If actual == Hq: Q-only (Hkv=0)
            // Otherwise: compute Hkv from remaining heads
            Hkv = std::max(0, (actual_heads - Hq) / 2);
        }
    }

    // For FP8 hybrid backward, record abs_max of d_qkv for subsequent quantization
    float* abs_max_ptr =
        mRunState.has_fp8_hybrid_backward() ? mRunState.simplified_quant_grads().d_qkv.abs_max() : nullptr;

    rope_backward(d_qkv,
                  d_out,
                  freqs,
                  reinterpret_cast<int*>(pos_ids.Data),
                  abs_max_ptr,
                  static_cast<int>(mB),
                  static_cast<int>(mT),
                  Hq,
                  Hkv,
                  Hs,
                  op.attrs.rotary_dim,
                  mRunState.MainStream);
    store_tensor(op.outputs[0], d_qkv);
}

namespace {

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

        ops.push_back(make_operation("rope_backward_" + std::to_string(ctx.op_counter++),
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

    ops.push_back(make_operation("rope_backward_" + std::to_string(ctx.op_counter++),
                                 "rope_backward",
                                 "rope_backward",
                                 {ctx.d_output, cos_cache, sin_cache, pos_ids},
                                 outputs,
                                 attrs));

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("rope", ::dsl::rope_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// RoPE
// ------------------------------------------------------------------------
const int _rope_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "rope";
    sig.min_inputs = 3;
    sig.max_inputs = 3;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& qkv = inputs[0];
        const auto& freqs = inputs[1];
        const auto& pos_ids = inputs[2];
        const auto& out = outputs[0];

        // Check qkv rank >= 2
        if (qkv.size() < 2) {
            ShapeValidationError err;
            err.message = "rope: qkv must have rank >= 2";
            return std::make_optional(err);
        }

        // Check freqs rank >= 2
        if (freqs.size() < 2) {
            ShapeValidationError err;
            err.message = "rope: freqs must have rank >= 2";
            return std::make_optional(err);
        }

        // Check output matches input
        if (auto err = validators::check_same_numel(out, qkv, "out", "qkv", "rope")) {
            return err;
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// RoPEBackward
// ------------------------------------------------------------------------
const int _rope_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "rope_backward";
    sig.min_inputs = 3;
    sig.max_inputs = 3;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& freqs = inputs[1];
        const auto& pos_ids = inputs[2];
        const auto& d_qkv = outputs[0];

        // d_qkv should match d_out
        if (auto err = validators::check_same_numel(d_qkv, d_out, "d_qkv", "d_out", "rope_backward")) {
            return err;
        }

        // Check freqs rank >= 2
        if (freqs.size() < 2) {
            ShapeValidationError err;
            err.message = "rope_backward: freqs must have rank >= 2";
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
