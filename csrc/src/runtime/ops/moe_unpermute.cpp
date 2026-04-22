#include "runtime/executor/compiled_ops.h"
#include "runtime/dsl/tensor_slot_dispatch.h"

#include <sstream>
#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
namespace dsl {

void CompiledExecutor::dispatch_moe_unpermute(const CompiledOp& op) {
    Tensor& expert_out = resolve_tensor(op.inputs[0]);
    Tensor& routing_weights = resolve_tensor(op.inputs[1]);
    Tensor& scatter_indices = resolve_tensor(op.inputs[2]);

    const int top_k = op.attrs.top_k;
    const int num_tokens = static_cast<int>(routing_weights.Sizes[0]);
    const int total_tokens = num_tokens * top_k;
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);

    // MoE output shape is dynamic: [num_tokens, hidden_size]
    // Use the preallocated mlp_down buffer to avoid stack allocation issues.
    // The mlp_down buffer has shape (B, T, C) which equals [num_tokens, hidden_size]
    // when viewed as 2D. This buffer survives layer boundary cleanup.
    int layer_idx = mCurrentLayer >= 0 ? mCurrentLayer : 0;
    Tensor* mlp_down_ptr = block_slot_tensor(layer_idx, TensorSlot::BlockMLPDown);
    if (!mlp_down_ptr) {
        throw std::runtime_error("moe_unpermute: mlp_down slot unavailable for layer " + std::to_string(layer_idx));
    }
    Tensor out = view_tensor(*mlp_down_ptr, {static_cast<long>(num_tokens), static_cast<long>(hidden_size)});

    if (expert_out.DType == ETensorDType::BF16) {
        if (routing_weights.DType == ETensorDType::FP32) {
            moe_unpermute_and_combine(out.get<nv_bfloat16>(),
                                      expert_out.get<nv_bfloat16>(),
                                      routing_weights.get<float>(),
                                      scatter_indices.get<int>(),
                                      num_tokens,
                                      total_tokens,
                                      hidden_size,
                                      top_k,
                                      mRunState.MainStream);
        } else if (routing_weights.DType == ETensorDType::BF16) {
            moe_unpermute_and_combine(out.get<nv_bfloat16>(),
                                      expert_out.get<nv_bfloat16>(),
                                      routing_weights.get<nv_bfloat16>(),
                                      scatter_indices.get<int>(),
                                      num_tokens,
                                      total_tokens,
                                      hidden_size,
                                      top_k,
                                      mRunState.MainStream);
        } else {
            throw std::logic_error("moe_unpermute: unsupported routing_weights dtype");
        }
    } else if (expert_out.DType == ETensorDType::FP32) {
        if (routing_weights.DType != ETensorDType::FP32) {
            throw std::logic_error("moe_unpermute: routing_weights dtype must match FP32 expert_out");
        }
        moe_unpermute_and_combine(out.get<float>(),
                                  expert_out.get<float>(),
                                  routing_weights.get<float>(),
                                  scatter_indices.get<int>(),
                                  num_tokens,
                                  total_tokens,
                                  hidden_size,
                                  top_k,
                                  mRunState.MainStream);
    } else {
        throw std::logic_error("moe_unpermute: unsupported expert_out dtype");
    }

    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_moe_unpermute_backward(const CompiledOp& op) {
    Tensor& d_output = resolve_tensor(op.inputs[0]);
    Tensor& expert_out = resolve_tensor(op.inputs[1]);
    Tensor& routing_weights = resolve_tensor(op.inputs[2]);
    Tensor& scatter_indices = resolve_tensor(op.inputs[3]);

    const int top_k = op.attrs.top_k;
    const int num_tokens = static_cast<int>(routing_weights.Sizes[0]);
    const int total_tokens = num_tokens * top_k;
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);

    // Allocate dynamic outputs explicitly to avoid relying on static shapes.
    // d_expert_out shape: [total_tokens, hidden_size]
    // d_routing_weights shape: [num_tokens, top_k]
    Tensor d_expert_out = mRunState.temp_alloc(d_output.DType,
                                               {static_cast<long>(total_tokens), static_cast<long>(hidden_size)},
                                               "moe_unpermute_backward_d_expert_out");
    Tensor d_routing_weights = mRunState.temp_alloc(routing_weights.DType,
                                                    {static_cast<long>(num_tokens), static_cast<long>(top_k)},
                                                    "moe_unpermute_backward_d_routing_weights");
    mTemps.push_back(d_expert_out);
    mTemps.push_back(d_routing_weights);

    // Validate expert_out shape: must be [total_tokens, hidden_size].
    // A shape mismatch (e.g., [1, hidden_size] from a stale/incorrectly saved tensor)
    // causes the kernel to read out of bounds, producing NaN in routing weight gradients.
    if (expert_out.nelem() != static_cast<long>(total_tokens) * hidden_size) {
        std::ostringstream oss;
        oss << "moe_unpermute_backward: expert_out shape mismatch. " << "Expected [" << total_tokens << ", "
            << hidden_size << "] (" << static_cast<long>(total_tokens) * hidden_size << " elems), got [";
        for (int d = 0; d < expert_out.Rank; ++d) {
            if (d > 0) oss << ", ";
            oss << expert_out.Sizes[d];
        }
        oss << "] (" << expert_out.nelem() << " elems). " << "ref_name='" << op.inputs[1].name
            << "' ref_slot=" << static_cast<int>(op.inputs[1].slot);
        throw std::runtime_error(oss.str());
    }

    if (d_output.DType == ETensorDType::BF16) {
        if (routing_weights.DType == ETensorDType::FP32) {
            moe_combine_backward(d_expert_out.get<nv_bfloat16>(),
                                 d_routing_weights.get<float>(),
                                 d_output.get<nv_bfloat16>(),
                                 expert_out.get<nv_bfloat16>(),
                                 routing_weights.get<float>(),
                                 scatter_indices.get<int>(),
                                 num_tokens,
                                 total_tokens,
                                 hidden_size,
                                 top_k,
                                 mRunState.MainStream);
        } else if (routing_weights.DType == ETensorDType::BF16) {
            moe_combine_backward(d_expert_out.get<nv_bfloat16>(),
                                 d_routing_weights.get<nv_bfloat16>(),
                                 d_output.get<nv_bfloat16>(),
                                 expert_out.get<nv_bfloat16>(),
                                 routing_weights.get<nv_bfloat16>(),
                                 scatter_indices.get<int>(),
                                 num_tokens,
                                 total_tokens,
                                 hidden_size,
                                 top_k,
                                 mRunState.MainStream);
        } else {
            throw std::logic_error("moe_unpermute_backward: unsupported routing_weights dtype");
        }
    } else if (d_output.DType == ETensorDType::FP32) {
        if (routing_weights.DType != ETensorDType::FP32) {
            throw std::logic_error("moe_unpermute_backward: routing_weights dtype must match FP32 d_output");
        }
        moe_combine_backward(d_expert_out.get<float>(),
                             d_routing_weights.get<float>(),
                             d_output.get<float>(),
                             expert_out.get<float>(),
                             routing_weights.get<float>(),
                             scatter_indices.get<int>(),
                             num_tokens,
                             total_tokens,
                             hidden_size,
                             top_k,
                             mRunState.MainStream);
    } else {
        throw std::logic_error("moe_unpermute_backward: unsupported d_output dtype");
    }

    store_tensor(op.outputs[0], d_expert_out);
    store_tensor(op.outputs[1], d_routing_weights);
}

namespace {

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

    ops.push_back(
        make_operation("moe_unpermute_backward_" + std::to_string(ctx.op_counter++),
                       "moe_unpermute_backward",
                       "moe_unpermute_backward",
                       {ctx.d_output, saved_ref(expert_out), saved_ref(routing_weights), saved_ref(scatter_indices)},
                       outputs,
                       attrs));

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("moe_unpermute", ::dsl::moe_unpermute_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// MoE Unpermute: out = moe_unpermute(expert_out, routing_weights, scatter_indices)
// Input[0]: expert_out [total_tokens, hidden_size]
// Input[1]: routing_weights [num_tokens, top_k]
// Input[2]: scatter_indices [total_tokens]
// Output: [num_tokens, hidden_size]
// ------------------------------------------------------------------------
const int _moe_unpermute_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "moe_unpermute";
    sig.min_inputs = 3;
    sig.max_inputs = 3;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
        // Dynamic routing shapes; accept for now
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

const int _moe_unpermute_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "moe_unpermute_backward";
    sig.min_inputs = 4;
    sig.max_inputs = 4;
    sig.min_outputs = 2;
    sig.max_outputs = 2;
    sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
