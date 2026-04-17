#include "runtime/executor/compiled_ops.h"

#include <string>
#include <string_view>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_moe_softmax(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    int layer_idx = op.attrs.layer_idx;
    std::string field;
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        parse_block_param(name, layer_idx, field);
    }

    const int num_tokens = static_cast<int>(inp.Sizes[0]);
    const int num_experts = static_cast<int>(inp.Sizes[1]);

    // Allocate output with same shape as input (softmax doesn't change shape)
    std::vector<long> out_shape = {static_cast<long>(num_tokens), static_cast<long>(num_experts)};
    Tensor out = mRunState.temp_alloc(inp.DType, out_shape, "moe_softmax_out");
    mTemps.push_back(out);

    if (inp.DType == ETensorDType::BF16) {
        moe_softmax_forward(out.get<nv_bfloat16>(),
                            inp.get<nv_bfloat16>(),
                            num_tokens,
                            num_experts,
                            mRunState.MainStream);
    } else {
        moe_softmax_forward(out.get<float>(), inp.get<float>(), num_tokens, num_experts, mRunState.MainStream);
    }

    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_moe_softmax_backward(const CompiledOp& op) {
    Tensor& d_probs = resolve_tensor(op.inputs[0]);
    Tensor& softmax_probs = resolve_tensor(op.inputs[1]);
    Tensor& d_logits = ensure_output_tensor(op.outputs[0]);

    const int num_tokens = static_cast<int>(d_probs.Sizes[0]);
    const int num_experts = static_cast<int>(d_probs.Sizes[1]);
    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("d_", 0) == 0) {
            name.remove_prefix(2);
        }
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        std::string field;
        parse_block_param(name, layer_idx, field);
    }
    if (d_probs.DType == ETensorDType::BF16) {
        moe_softmax_backward(d_logits.get<nv_bfloat16>(),
                             d_probs.get<nv_bfloat16>(),
                             softmax_probs.get<nv_bfloat16>(),
                             num_tokens,
                             num_experts,
                             mRunState.MainStream);
    } else {
        moe_softmax_backward(d_logits.get<float>(),
                             d_probs.get<float>(),
                             softmax_probs.get<float>(),
                             num_tokens,
                             num_experts,
                             mRunState.MainStream);
    }

    store_tensor(op.outputs[0], d_logits);
}

namespace {

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

        ops.push_back(make_operation("moe_softmax_backward_" + std::to_string(ctx.op_counter++),
                                     "moe_softmax_backward",
                                     "moe_softmax_backward",
                                     {ctx.d_output, saved_ref(out)},
                                     {ctx.d_inputs[0]}));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("moe_softmax", ::dsl::moe_softmax_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// MoE Softmax: probs = moe_softmax(logits)
// Input: [num_tokens, num_experts], Output: [num_tokens, num_experts]
// ------------------------------------------------------------------------
const int _moe_softmax_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "moe_softmax";
    sig.min_inputs = 1;
    sig.max_inputs = 1;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto& inputs, const auto& outputs, const AttrMap&, const ShapeEnv&) {
        if (inputs.empty() || outputs.empty()) {
            return std::make_optional(ShapeValidationError{"moe_softmax: missing inputs/outputs"});
        }
        if (inputs[0] != outputs[0]) {
            ShapeValidationError err;
            err.message = "moe_softmax: output shape must match input shape";
            return std::make_optional(err);
        }
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

const int _moe_softmax_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "moe_softmax_backward";
    sig.min_inputs = 2;
    sig.max_inputs = 2;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
