#include "dsl/compiled_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_view(const CompiledOp& op) {
    Tensor& src = resolve_tensor(op.inputs[0]);
    Tensor view = view_tensor(src, op.attrs.shape);
    mTensorMap[op.outputs[0].name] = view;
}

void CompiledExecutor::dispatch_view_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    std::vector<long> shape = op.attrs.shape;

    // If shape is empty, try to resolve from shape_like reference
    if (shape.empty() && !op.attrs.shape_like.empty()) {
        std::string ref_name = op.attrs.shape_like;

        // Strip "saved." prefix if present
        const std::string saved_prefix = "saved.";
        if (ref_name.rfind(saved_prefix, 0) == 0) {
            ref_name = ref_name.substr(saved_prefix.length());
        }

        // Try to find the reference tensor
        Tensor* ref = nullptr;

        // Check saved tensors first
        if (mSaved) {
            auto it = mSaved->find(ref_name);
            if (it != mSaved->end()) {
                ref = &it->second;
            }
        }

        // Check tensor map
        if (!ref) {
            auto it = mTensorMap.find(ref_name);
            if (it != mTensorMap.end()) {
                ref = &it->second;
            }
        }

        // If reference found and valid, use its shape
        if (ref && ref->Rank > 0) {
            shape.assign(ref->Sizes.begin(), ref->Sizes.begin() + ref->Rank);
        } else {
            // Fallback: infer shape based on output tensor name and input shape
            // View backward typically does one of:
            // 1. Flatten: [B,T,C] -> [B*T,C] (output name contains "_flat")
            // 2. Unflatten: [B*T,C] -> [B,T,C] (output name does not contain "_flat")
            //
            // Check output name for "_flat" suffix to determine direction
            const std::string& out_name = op.outputs[0].name;
            bool wants_flat = out_name.find("_flat") != std::string::npos;

            if (wants_flat) {
                // Flatten to rank 2: [B,T,C] -> [B*T,C] or [B*T,C] -> [B*T,C]
                if (d_out.Rank >= 3) {
                    long flat_dim = 1;
                    for (int i = 0; i < d_out.Rank - 1; ++i) {
                        flat_dim *= d_out.Sizes[i];
                    }
                    shape = {flat_dim, d_out.Sizes[d_out.Rank - 1]};
                } else if (d_out.Rank == 2) {
                    // Already flat, keep shape
                    shape = {d_out.Sizes[0], d_out.Sizes[1]};
                }
            } else {
                // Unflatten or keep shape
                if (d_out.Rank >= 3) {
                    // Already unflat, keep shape
                    shape.assign(d_out.Sizes.begin(), d_out.Sizes.begin() + d_out.Rank);
                } else if (d_out.Rank == 2 && d_out.Sizes[0] == mB * mT) {
                    // Unflatten: [B*T,C] -> [B,T,C]
                    shape = {mB, mT, d_out.Sizes[1]};
                } else if (d_out.Rank == 2) {
                    // Keep as rank 2
                    shape = {d_out.Sizes[0], d_out.Sizes[1]};
                }
            }
        }
    }

    if (shape.empty()) {
        auto shape_str = [](const Tensor& t) {
            std::string s = "[";
            for (int i = 0; i < t.Rank; ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(t.Sizes[i]);
            }
            s += "]";
            return s;
        };
        throw std::runtime_error("CompiledExecutor view_backward: cannot resolve shape for op " + op.op_id +
                                " input=" + op.inputs[0].name + " shape=" + shape_str(d_out) +
                                " output=" + op.outputs[0].name +
                                " shape_like=" + op.attrs.shape_like);
    }
    Tensor view = view_tensor(d_out, shape);
    mTensorMap[op.outputs[0].name] = view;
}


}  // namespace dsl
