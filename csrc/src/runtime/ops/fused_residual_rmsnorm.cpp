#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <atomic>
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

#include <fmt/format.h>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace dsl {

void CompiledExecutor::dispatch_fused_residual_rmsnorm(const CompiledOp& op) {
    Tensor& residual_in = resolve_tensor(op.inputs[0]);
    Tensor& input = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);

    // Detect norm type from weight name BEFORE resolving outputs.
    // This allows binding output tensors directly to canonical buffers,
    // eliminating post-kernel D2D copies.
    int fwd_layer_idx = -1;
    std::string fwd_field;
    parse_block_param(op.inputs[2].name, fwd_layer_idx, fwd_field);
    // Strip _eff suffix for matching (Qwen3.5 uses weight+1 pattern)
    if (fwd_field.size() > 4 && fwd_field.compare(fwd_field.size() - 4, 4, "_eff") == 0) {
        fwd_field = fwd_field.substr(0, fwd_field.size() - 4);
    }

    const bool is_ln2_fwd = (fwd_layer_idx >= 0 && fwd_field.find("ln2") != std::string::npos);
    const bool is_hybrid_norm = (fwd_layer_idx >= 0 &&
        fwd_layer_idx < static_cast<int>(mConfig.NumLayers) &&
        mConfig.architecture == modules::ArchitectureType::Hybrid &&
        fwd_field == "norm_weight");

    // Bind residual_out directly to canonical buffer when possible, so the kernel
    // writes there without a post-kernel D2D copy.
    Tensor* residual_out_ptr = nullptr;
    if (is_ln2_fwd) {
        auto& buf = mRunState.simplified_acts(fwd_layer_idx).residual_att;
        if (buf.Data) {
            residual_out_ptr = &buf;
            store_tensor(op.outputs[0], buf);
        }
    } else if (is_hybrid_norm) {
        Tensor& buf = mRunState.get_residual(fwd_layer_idx, mRunState.MainStream);
        if (buf.Data) {
            residual_out_ptr = &buf;
            store_tensor(op.outputs[0], buf);
        }
    }
    if (!residual_out_ptr) {
        residual_out_ptr = &ensure_output_tensor(op.outputs[0]);
    }
    Tensor& residual_out = *residual_out_ptr;

    Tensor& y = ensure_output_tensor(op.outputs[1]);

    // Bind rstd to canonical buffer for Hybrid norm.
    Tensor* rstd_ptr = nullptr;
    if (is_hybrid_norm) {
        Tensor& buf = mRunState.simplified_acts(fwd_layer_idx).ln1_rstd;
        if (buf.Data) {
            rstd_ptr = &buf;
            store_tensor(op.outputs[2], buf);
        }
    }
    if (!rstd_ptr) {
        rstd_ptr = &ensure_output_tensor(op.outputs[2]);
    }
    Tensor& rstd = *rstd_ptr;

    // Validate dtypes before calling kernel
    if (rstd.DType != ETensorDType::FP32) {
        std::ostringstream oss;
        oss << "fused_residual_rmsnorm: rstd dtype mismatch. Expected FP32, got "
            << dtype_to_str(rstd.DType) << ". Output tensor: " << op.outputs[2].name
            << " (slot=" << static_cast<int>(op.outputs[2].slot) << ")";
        throw std::runtime_error(oss.str());
    }

    // During replay, the LN1/hybrid fused_residual_rmsnorm cannot correctly reconstruct
    // the residual from its components (res_att[K-1] + mlp_down[K-1]) because those live
    // in shared buffers that contain the last layer's values. Instead, use the correct
    // per-layer residual from ResidualManager as the residual input, with a zero "input"
    // so the kernel computes: residual_out = stored_res + 0 = stored_res, y = rmsnorm(stored_res).
    // This applies to both standard LN1 (dense models) and hybrid norm (Nemotron-H).
    if (mInReplay && !is_ln2_fwd && !is_hybrid_norm && fwd_layer_idx >= 0) {
        Tensor& stored_res_ffn = mRunState.get_residual(fwd_layer_idx, mRunState.MainStream);
        if (stored_res_ffn.Data) {
            // Use stored residual as residual_in, zero out input
            Tensor zero_input = mRunState.temp_alloc(input.DType,
                std::vector<long>(input.Sizes.begin(), input.Sizes.begin() + input.Rank));
            mTemps.push_back(zero_input);
            fill_zero(zero_input, mRunState.MainStream);
            // Debug: validate tensor pointers before kernel call
            static const bool replay_dbg = std::getenv("SUROGATE_REPLAY_SYNC") != nullptr;
            if (replay_dbg) {
                fprintf(stderr, "[REPLAY_BYPASS] layer=%d N=%d C=%d res_out=%p y=%p rstd=%p stored=%p zero=%p weight=%p"
                        " res_out.nelem=%ld stored.nelem=%ld y.nelem=%ld rstd.nelem=%ld weight.nelem=%ld\n",
                        fwd_layer_idx, static_cast<int>(mB * mT), mConfig.HiddenSize,
                        (void*)residual_out.Data, (void*)y.Data, (void*)rstd.Data,
                        (void*)stored_res_ffn.Data, (void*)zero_input.Data, (void*)weight.Data,
                        residual_out.nelem(), stored_res_ffn.nelem(), y.nelem(), rstd.nelem(), weight.nelem());
            }
            fused_residual_rmsnorm_forward(residual_out, y, rstd, stored_res_ffn, zero_input, weight, nullptr,
                                           op.attrs.eps, static_cast<int>(mB * mT),
                                           mConfig.HiddenSize, mRunState.MainStream);
            return;
        }
    }

    fused_residual_rmsnorm_forward(residual_out, y, rstd, residual_in, input, weight, nullptr,
                                   op.attrs.eps, static_cast<int>(mB * mT),
                                   mConfig.HiddenSize, mRunState.MainStream);
}

void CompiledExecutor::dispatch_fused_residual_rmsnorm_backward(const CompiledOp& op) {
    // inputs: d_y, d_residual_next (may be empty), residual_out, weight, rstd
    // outputs: d_residual, d_input, d_weight (optional)

    Tensor& d_y = resolve_tensor(op.inputs[0]);

    const bool is_final_norm =
        (op.inputs[3].name.find("final_norm") != std::string::npos ||
         op.inputs[3].name.find("ln_final") != std::string::npos ||
         op.inputs[3].name.find("ln_f") != std::string::npos);

    Tensor* residual_out_ptr = &resolve_tensor(op.inputs[2]);
    Tensor weight_eff_fallback{};
    Tensor weight_eff_ones{};
    Tensor* weight_ptr = nullptr;
    try {
        weight_ptr = &resolve_tensor(op.inputs[3]);
    } catch (const std::exception&) {
        const std::string& eff_name = op.inputs[3].name;
        const bool has_eff_suffix =
            eff_name.size() > 4 &&
            eff_name.compare(eff_name.size() - 4, 4, "_eff") == 0;
        if (!has_eff_suffix) {
            throw;
        }

        const std::string base_name = eff_name.substr(0, eff_name.size() - 4);
        Tensor* base_ptr = nullptr;
        if (mWeights.has(base_name)) {
            base_ptr = &mWeights.get(base_name);
        } else {
            TensorRef base_ref = op.inputs[3];
            base_ref.name = base_name;
            base_ref.tensor_id = mCurrentGraph ? mCurrentGraph->find_tensor_id(base_name) : -1;
            try {
                base_ptr = &resolve_tensor(base_ref);
            } catch (const std::exception&) {
                base_ptr = nullptr;
            }
        }
        if (!base_ptr || !base_ptr->Data) {
            throw;
        }

        std::vector<long> shape(base_ptr->Sizes.begin(), base_ptr->Sizes.begin() + base_ptr->Rank);
        weight_eff_fallback = mRunState.temp_alloc(base_ptr->DType, shape);
        weight_eff_ones = mRunState.temp_alloc(base_ptr->DType, shape);
        mTemps.push_back(weight_eff_fallback);
        mTemps.push_back(weight_eff_ones);
        fill_constant(weight_eff_ones, 1.0f, static_cast<std::size_t>(weight_eff_ones.nelem()),
                      mRunState.MainStream);
        vector_add_sr(weight_eff_fallback, *base_ptr, weight_eff_ones, 1.0f,
                      static_cast<long>(base_ptr->nelem()), 0, mRunState.MainStream);
        weight_ptr = &weight_eff_fallback;
    }
    Tensor& weight = *weight_ptr;
    Tensor* rstd_ptr = &resolve_tensor(op.inputs[4]);

    int ln_layer_idx = -1;
    std::string ln_field;
    if (!op.inputs[3].name.empty()) {
        parse_block_param(op.inputs[3].name, ln_layer_idx, ln_field);
    }
    // Strip _eff suffix from weight name for matching (Qwen3.5 uses weight+1 pattern
    // producing "ln1_weight_eff", "ln2_weight_eff" etc.)
    if (ln_field.size() > 4 && ln_field.compare(ln_field.size() - 4, 4, "_eff") == 0) {
        ln_field = ln_field.substr(0, ln_field.size() - 4);
    }
    if (ln_layer_idx >= 0 && ln_field == "ln1_weight") {
        if (mRunState.has_residual_offloading()) {
            mRunState.fetch_residual(ln_layer_idx, mRunState.side_stream());
        }
        residual_out_ptr = &mRunState.get_residual(ln_layer_idx, mRunState.MainStream);
        rstd_ptr = &mRunState.simplified_acts(ln_layer_idx).ln1_rstd;
    } else if (ln_layer_idx >= 0 && ln_field == "norm_weight") {
        if (mRunState.has_residual_offloading()) {
            mRunState.fetch_residual(ln_layer_idx, mRunState.side_stream());
        }
        residual_out_ptr = &mRunState.get_residual(ln_layer_idx, mRunState.MainStream);
        rstd_ptr = &mRunState.simplified_acts(ln_layer_idx).ln1_rstd;
    }
    if (ln_layer_idx >= 0 && ln_field == "ln2_weight") {
        auto& acts = mRunState.simplified_acts(ln_layer_idx);
        residual_out_ptr = &acts.residual_att;
        rstd_ptr = &acts.ln2_rstd;
    }
    Tensor& residual_out = *residual_out_ptr;

    // d_residual_next is the incoming gradient from the next layer (may be zero/empty)
    Tensor d_residual_zero{};
    Tensor* d_residual_next = nullptr;
    if (!op.inputs[1].name.empty()) {
        d_residual_next = &resolve_tensor(op.inputs[1]);
    } else {
        d_residual_zero = mRunState.temp_alloc(d_y.DType, {mB, mT, static_cast<long>(mConfig.HiddenSize)});
        fill_zero(d_residual_zero, mRunState.MainStream);
        mTemps.push_back(d_residual_zero);
        d_residual_next = &d_residual_zero;
    }
    Tensor* d_residual_input = d_residual_next;
    Tensor* d_residual_stream = d_residual_next;

    // Resolve d_input to canonical gradient buffer BEFORE the kernel, so the kernel
    // writes directly to the correct simplified_grads buffer. This eliminates
    // post-kernel D2D copies that were mirroring data between graph slots and
    // pre-allocated gradient buffers.
    Tensor* d_input_ptr = nullptr;
    if (ln_layer_idx >= 0 && ln_field == "ln2_weight") {
        // LN2: canonical target is d_mlp_down (MLP backward reads this as d_out)
        auto& grads = mRunState.simplified_grads(ln_layer_idx);
        if (grads.d_mlp_down.Data) {
            d_input_ptr = &grads.d_mlp_down;
        }
    } else if (ln_layer_idx >= 0 && (ln_field == "ln1_weight" || ln_field == "norm_weight")) {
        // LN1/norm: canonical target is prev layer's d_mlp_down
        if (op.outputs.size() > 1 && op.outputs[1].slot == TensorSlot::BlockDMLPDown) {
            const int prev_layer = op.outputs[1].layer_idx;
            if (prev_layer >= 0) {
                auto& prev_grads = mRunState.simplified_grads(prev_layer);
                if (prev_grads.d_mlp_down.Data) {
                    d_input_ptr = &prev_grads.d_mlp_down;
                }
            }
        }
    }
    if (!d_input_ptr) {
        d_input_ptr = &ensure_output_tensor(op.outputs[1]);
    }
    Tensor& d_input = *d_input_ptr;

    // d_weight may be nullptr if weight is frozen
    Tensor dummy_weight{};
    Tensor* d_weight_ptr = nullptr;
    bool skip_weight_grad = true;
    if (op.outputs.size() > 2 && !op.outputs[2].name.empty()) {
        d_weight_ptr = &ensure_output_tensor(op.outputs[2]);
        skip_weight_grad = false;
        if (op.outputs[2].slot == TensorSlot::Mapped || op.outputs[2].slot == TensorSlot::Temporary) {
            fill_zero(*d_weight_ptr, mRunState.MainStream);
        }
    } else {
        dummy_weight = mRunState.temp_alloc(weight.DType, {static_cast<long>(mConfig.HiddenSize)});
        mTemps.push_back(dummy_weight);
        d_weight_ptr = &dummy_weight;
    }

    const int C = mConfig.HiddenSize;

    // Determine abs_max pointer for FP8 gradient quantization.
    float* abs_max_ptr = nullptr;
    if (mRunState.has_grad_quants()) {
        const bool is_ln2 = (ln_field == "ln2_weight");
        abs_max_ptr = is_ln2
            ? mRunState.simplified_quant_grads().d_res_att.abs_max()
            : mRunState.simplified_quant_grads().d_res_ffn.abs_max();
    }

    const bool mixed_weight_grad =
        (!skip_weight_grad && d_weight_ptr &&
         d_weight_ptr->DType == ETensorDType::FP32 &&
         d_input.DType == ETensorDType::BF16);

    if (mixed_weight_grad) {
        // Compute d_input in BF16 and weight grad separately in FP32.
        Tensor tmp_dw = mRunState.temp_alloc(ETensorDType::BF16, {static_cast<long>(C)});
        mTemps.push_back(tmp_dw);
        rmsnorm_backward(d_input, tmp_dw, mRunState.scratch().rmsnorm_scratch,
                         *d_residual_input, d_y, residual_out, weight, *rstd_ptr,
                         abs_max_ptr,
                         static_cast<int>(mB), static_cast<int>(mT), C,
                         mRunState.DeviceProp, mRunState.MainStream,
                         /*skip_weight_grad=*/true);
        rmsnorm_backward_dweight_fp32(*d_weight_ptr, d_y, residual_out, *rstd_ptr,
                                      static_cast<int>(mB), static_cast<int>(mT), C,
                                      mRunState.MainStream);
    } else {
        rmsnorm_backward(d_input, *d_weight_ptr, mRunState.scratch().rmsnorm_scratch,
                         *d_residual_input, d_y, residual_out, weight, *rstd_ptr,
                         abs_max_ptr,
                         static_cast<int>(mB), static_cast<int>(mT), C,
                         mRunState.DeviceProp, mRunState.MainStream, skip_weight_grad);
    }

    // Register d_input in mTensors for both graph outputs. Both outputs carry the same
    // gradient (add backward is identity for both inputs). Using store_tensor ensures
    // downstream ops find the result via resolve_tensor (which checks mTensors first).
    store_tensor(op.outputs[1], d_input);
    if (!op.outputs[0].name.empty() && op.outputs[0].name != op.outputs[1].name) {
        store_tensor(op.outputs[0], d_input);
    }

    // Safety net: update d_residual_stream if it points to a different buffer.
    // With direct binding above, this is typically a no-op (pointers match).
    if (d_residual_stream && d_residual_stream->Data && d_residual_stream->Data != d_input.Data) {
        CUDA_CHECK(cudaMemcpyAsync(d_residual_stream->Data, d_input.Data, d_input.bytes(),
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    // Alias related gradient buffers so code reading simplified_grads directly
    // (not via resolve_tensor/mTensors) sees the correct data.
    if (ln_layer_idx >= 0 && ln_field == "ln2_weight") {
        auto& grads = mRunState.simplified_grads(ln_layer_idx);
        if (grads.d_res_ffn.Data && grads.d_res_ffn.Data != d_input.Data) {
            grads.d_res_ffn.Data = d_input.Data;
        }
    }
    if (ln_layer_idx >= 0 && (ln_field == "ln1_weight" || ln_field == "norm_weight") &&
        op.outputs.size() > 1 && op.outputs[1].slot == TensorSlot::BlockDMLPDown) {
        const int prev_layer = op.outputs[1].layer_idx;
        if (prev_layer >= 0) {
            auto& prev_grads = mRunState.simplified_grads(prev_layer);
            if (prev_grads.d_res_ffn.Data && prev_grads.d_res_ffn.Data != d_input.Data) {
                prev_grads.d_res_ffn.Data = d_input.Data;
            }
        }
    }
}


}  // namespace dsl
