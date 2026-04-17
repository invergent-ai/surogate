#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
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
#include "utilities/utils.h"

namespace dsl {

void CompiledExecutor::dispatch_mrope(const CompiledOp& op) {
    Tensor& qkv_in = resolve_tensor(op.inputs[0]);
    Tensor& freqs = resolve_tensor(op.inputs[1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[2]);

    const std::vector<long> qkv_shape(qkv_in.Sizes.begin(), qkv_in.Sizes.begin() + qkv_in.Rank);
    Tensor qkv_out = ensure_output_tensor_or_persistent(ensure_output_tensor(op.outputs[0]),
                                                        mRunState,
                                                        mMoeSavedBuffers,
                                                        mMoeSavedSizes,
                                                        op.op_id + "." + op.outputs[0].name + ".qkv_out",
                                                        qkv_in.DType,
                                                        qkv_shape,
                                                        "mrope");

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = derive_head_size(qkv_in, Hq, Hkv, static_cast<int>(mConfig.head_size()));
    const int qkv_channels = Hs * (Hq + 2 * Hkv);

    if (qkv_in.Data != qkv_out.Data) {
        CUDA_CHECK(
            cudaMemcpyAsync(qkv_out.Data, qkv_in.Data, qkv_in.bytes(), cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    Tensor qkv_view = qkv_out;
    const long needed = static_cast<long>(mB) * static_cast<long>(mT) * qkv_channels;
    if ((qkv_out.Rank == 4 || (qkv_out.Rank == 3 && qkv_out.Sizes[2] != qkv_channels)) &&
        static_cast<long>(qkv_out.nelem()) >= needed) {
        qkv_view = view_tensor(qkv_out, {mB, mT, qkv_channels});
    }
    int rotary_dim = op.attrs.rotary_dim;

    const int* pos_ptr = reinterpret_cast<int*>(pos_ids.Data);
    int pos_planes = 1;
    if (pos_ids.Rank == 3) {
        pos_planes = static_cast<int>(pos_ids.Sizes[0]);
        if (pos_planes == 4) {
            pos_ptr += static_cast<int>(mB * mT);
            pos_planes = 3;
        }
    }

    mrope_forward(qkv_view,
                  qkv_view,
                  freqs,
                  pos_ptr,
                  pos_planes,
                  op.attrs.mrope_section[0],
                  op.attrs.mrope_section[1],
                  op.attrs.mrope_section[2],
                  nullptr,
                  static_cast<int>(mB),
                  static_cast<int>(mT),
                  Hq,
                  Hkv,
                  Hs,
                  rotary_dim,
                  mRunState.MainStream);

    store_tensor(op.outputs[0], qkv_out);
}

void CompiledExecutor::dispatch_mrope_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    // Allow inputs: [d_out, freqs, position_ids] or legacy [d_out, qkv, freqs, position_ids]
    const bool has_qkv = op.inputs.size() == 4;
    Tensor& freqs = resolve_tensor(op.inputs[has_qkv ? 2 : 1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[has_qkv ? 3 : 2]);

    const std::vector<long> d_qkv_shape(d_out.Sizes.begin(), d_out.Sizes.begin() + d_out.Rank);
    Tensor d_qkv = ensure_output_tensor_or_persistent(ensure_output_tensor(op.outputs[0]),
                                                      mRunState,
                                                      mMoeSavedBuffers,
                                                      mMoeSavedSizes,
                                                      op.op_id + "." + op.outputs[0].name + ".d_qkv",
                                                      d_out.DType,
                                                      d_qkv_shape,
                                                      "mrope_backward");

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = derive_head_size(d_out, Hq, Hkv, static_cast<int>(mConfig.head_size()));
    const int qkv_channels = Hs * (Hq + 2 * Hkv);
    const int rotary_dim = op.attrs.rotary_dim;

    Tensor d_out_view = (d_out.Rank == 4) ? view_tensor(d_out, {mB, mT, static_cast<long>(qkv_channels)}) : d_out;
    Tensor d_qkv_view = (d_qkv.Rank == 4) ? view_tensor(d_qkv, {mB, mT, static_cast<long>(qkv_channels)}) : d_qkv;

    if (d_qkv_view.Data != d_out_view.Data) {
        const std::size_t bytes = static_cast<std::size_t>(d_out_view.nelem()) * get_dtype_size(d_out_view.DType);
        CUDA_CHECK(
            cudaMemcpyAsync(d_qkv_view.Data, d_out_view.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    const int* pos_ptr = reinterpret_cast<int*>(pos_ids.Data);
    int pos_planes = 1;
    if (pos_ids.Rank == 3) {
        pos_planes = static_cast<int>(pos_ids.Sizes[0]);
        if (pos_planes == 4) {
            pos_ptr += static_cast<int>(mB * mT);
            pos_planes = 3;
        }
    }

    mrope_backward(d_qkv_view,
                   d_qkv_view,
                   freqs,
                   pos_ptr,
                   pos_planes,
                   op.attrs.mrope_section[0],
                   op.attrs.mrope_section[1],
                   op.attrs.mrope_section[2],
                   nullptr,
                   static_cast<int>(mB),
                   static_cast<int>(mT),
                   Hq,
                   Hkv,
                   Hs,
                   rotary_dim,
                   mRunState.MainStream);

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        store_tensor(op.outputs[0], d_qkv);
    }
}

namespace {

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

        ops.push_back(make_operation("mrope_backward_" + std::to_string(ctx.op_counter++),
                                     "mrope_backward",
                                     "mrope_backward",
                                     {ctx.d_output, freqs, pos_ids},
                                     outputs,
                                     attrs));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("mrope", ::dsl::mrope_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// MRoPE (multimodal RoPE)
// ------------------------------------------------------------------------
const int _mrope_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "mrope";
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
            err.message = "mrope: qkv must have rank >= 2";
            return std::make_optional(err);
        }

        // Check freqs rank >= 2
        if (freqs.size() < 2) {
            ShapeValidationError err;
            err.message = "mrope: freqs must have rank >= 2";
            return std::make_optional(err);
        }

        // Check position_ids rank (allow 2 or 3)
        if (!pos_ids.empty() && (pos_ids.size() < 2 || pos_ids.size() > 3)) {
            ShapeValidationError err;
            err.message = "mrope: position_ids must have rank 2 or 3";
            return std::make_optional(err);
        }

        // Check output matches input
        if (auto err = validators::check_same_numel(out, qkv, "out", "qkv", "mrope")) {
            return err;
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// MRoPEBackward
// ------------------------------------------------------------------------
const int _mrope_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "mrope_backward";
    sig.min_inputs = 3;
    sig.max_inputs = 4;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& freqs = inputs.size() > 2 ? inputs[inputs.size() - 2] : inputs[1];
        const auto& pos_ids = inputs.size() > 2 ? inputs[inputs.size() - 1] : inputs[2];
        const auto& qkv = inputs.size() > 3 ? inputs[1] : d_out;
        const auto& d_qkv = outputs[0];

        // d_qkv should match d_out and qkv
        if (auto err = validators::check_same_numel(d_qkv, d_out, "d_qkv", "d_out", "mrope_backward")) {
            return err;
        }
        if (auto err = validators::check_same_numel(d_qkv, qkv, "d_qkv", "qkv", "mrope_backward")) {
            return err;
        }

        // Check freqs rank >= 2
        if (freqs.size() < 2) {
            ShapeValidationError err;
            err.message = "mrope_backward: freqs must have rank >= 2";
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
