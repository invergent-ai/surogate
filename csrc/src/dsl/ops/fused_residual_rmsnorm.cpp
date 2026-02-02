#include "dsl/compiled_ops.h"

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

#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_fused_residual_rmsnorm(const CompiledOp& op) {
    Tensor& residual_in = resolve_tensor(op.inputs[0]);
    Tensor& input = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);

    Tensor& residual_out = ensure_output_tensor(op.outputs[0]);
    Tensor& y = ensure_output_tensor(op.outputs[1]);
    Tensor& rstd = ensure_output_tensor(op.outputs[2]);

    // Validate dtypes before calling kernel
    if (rstd.DType != ETensorDType::FP32) {
        std::ostringstream oss;
        oss << "fused_residual_rmsnorm: rstd dtype mismatch. Expected FP32, got "
            << dtype_to_str(rstd.DType) << ". Output tensor: " << op.outputs[2].name
            << " (slot=" << static_cast<int>(op.outputs[2].slot) << ")";
        throw std::runtime_error(oss.str());
    }

    // DEBUG: Pre-kernel NaN check for LN1 layer 3
    {
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(op.outputs[1].name, layer_idx, field) && layer_idx == 3 && field == "ln1") {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            log_nan_sample("LN1_INPUT_PRE", layer_idx, op.inputs[1].name, input, 3);
            log_nan_sample("LN1_RES_PRE", layer_idx, op.inputs[0].name, residual_in, 3);
            fprintf(stderr,
                    "[LN1_PTRS_PRE] layer=%d residual_in=%p input=%p residual_out=%p y=%p rstd=%p\n",
                    layer_idx, residual_in.Data, input.Data, residual_out.Data, y.Data, rstd.Data);
        }
    }

    // DEBUG: Print ln values after forward
    fused_residual_rmsnorm_forward(residual_out, y, rstd, residual_in, input, weight, nullptr,
                                   op.attrs.eps, static_cast<int>(mB * mT),
                                   mConfig.HiddenSize, mRunState.MainStream);

    // NaN detection for LN1/LN2 forward (layer 25, token 3)
    {
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(op.outputs[1].name, layer_idx, field)) {
            const long sample_token = 3;
            const std::string tag_prefix = (field == "ln1") ? "FWD_LN1" : "FWD_LN2";
            log_nan_sample((tag_prefix + "_RESIDUAL_IN").c_str(), layer_idx, op.inputs[0].name, residual_in, sample_token);
            log_nan_sample((tag_prefix + "_INPUT").c_str(), layer_idx, op.inputs[1].name, input, sample_token);
            log_nan_sample((tag_prefix + "_RES_OUT").c_str(), layer_idx, op.outputs[0].name, residual_out, sample_token);
            log_nan_sample((tag_prefix + "_OUT").c_str(), layer_idx, op.outputs[1].name, y, sample_token);
            log_nan_sample((tag_prefix + "_RSTD").c_str(), layer_idx, op.outputs[2].name, rstd, sample_token);
            if (layer_idx == 0 && field == "ln1") {
                log_tensor_stats("FWD_LN1", layer_idx, op.outputs[1].name, y, 4096);
            }
            if (layer_idx == 3 && field == "ln1") {
                auto& prev_acts = mRunState.simplified_acts(layer_idx - 1);
                fprintf(stderr,
                        "[LN1_INPUT_PTR] layer=%d input=%p prev_mlp_down=%p\n",
                        layer_idx, input.Data, prev_acts.mlp_down.Data);
            }
            if (field == "ln2" && std::getenv("SUROGATE_MOE_DOT_TRACE")) {
                const int target_layer = env_int("SUROGATE_MOE_DOT_LAYER", -1);
                const int target_token = env_int("SUROGATE_MOE_DOT_TOKEN", 0);
                static int ln2_target_trace = 0;
                if (ln2_target_trace < 1 && target_token >= 0 &&
                    (target_layer < 0 || layer_idx == target_layer)) {
                    CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                    fprintf(stderr,
                            "[MOE_LN2_TRACE] layer=%d token=%d\n",
                            layer_idx, target_token);
                    log_tensor_token_row_stats("MOE_LN2_RESIDUAL_IN", residual_in, target_token);
                    log_tensor_token_row_stats("MOE_LN2_INPUT", input, target_token);
                    log_tensor_token_row_stats("MOE_LN2_OUT", y, target_token);
                    log_tensor_token_row_stats("MOE_LN2_RES_OUT", residual_out, target_token);
                    log_tensor_scalar_at("MOE_LN2_RSTD", rstd, target_token);
                    ln2_target_trace++;
                }
            }
            if (field == "ln2" && std::getenv("SUROGATE_MOE_TRACE_LN2")) {
                const int target_token = env_int("SUROGATE_MOE_DOT_TOKEN", 0);
                const int max_layer = env_int("SUROGATE_MOE_TRACE_LN2_MAXL", 4);
                static std::vector<char> ln2_layer_seen;
                if (layer_idx >= 0 && layer_idx <= max_layer && target_token >= 0) {
                    if (static_cast<int>(ln2_layer_seen.size()) <= layer_idx) {
                        ln2_layer_seen.resize(static_cast<std::size_t>(layer_idx + 1), 0);
                    }
                    if (!ln2_layer_seen[static_cast<std::size_t>(layer_idx)]) {
                        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                        fprintf(stderr,
                                "[MOE_LN2_LAYER_TRACE] layer=%d token=%d\n",
                                layer_idx, target_token);
                        log_tensor_token_row_stats("MOE_LN2_TRACE_RES_IN", residual_in, target_token);
                        log_tensor_token_row_stats("MOE_LN2_TRACE_INPUT", input, target_token);
                        log_tensor_token_row_stats("MOE_LN2_TRACE_OUT", y, target_token);
                        log_tensor_token_row_stats("MOE_LN2_TRACE_RES_OUT", residual_out, target_token);
                        log_tensor_scalar_at("MOE_LN2_TRACE_RSTD", rstd, target_token);
                        ln2_layer_seen[static_cast<std::size_t>(layer_idx)] = 1;
                    }
                }
            }
            if (field == "ln1" && std::getenv("SUROGATE_TRACE_LN1_RES")) {
                const int target_layer = env_int("SUROGATE_TRACE_LN1_RES_LAYER", 27);
                static int ln1_res_trace = 0;
                if (layer_idx == target_layer && ln1_res_trace < 4) {
                    fprintf(stderr, "[FWD_LN1_RES_TRACE] layer=%d residual_out=%s input=%s residual_in=%s\n",
                            layer_idx,
                            op.outputs[0].name.c_str(),
                            op.inputs[1].name.c_str(),
                            op.inputs[0].name.c_str());
                    log_tensor_stats_ex("FWD_LN1_RES", layer_idx, op.outputs[0].name, residual_out, 4096, true);
                    log_tensor_stats_ex("FWD_LN1_IN", layer_idx, op.inputs[1].name, input, 4096, true);
                    log_tensor_stats_ex("FWD_LN1_RES_IN", layer_idx, op.inputs[0].name, residual_in, 4096, true);
                    ln1_res_trace++;
                }
            }
        }
    }

    // FIX: For LN2 output (res_att), copy to simplified_acts.residual_att when the
    // graph compiler assigned the wrong slot. This happens for the last layer where
    // the output is named "StackedBlocks_N" instead of "blocks[N].res_att".
    if (op.outputs[1].name.find("ln2") != std::string::npos) {
        int layer_idx = -1;
        std::string field;
        parse_block_param(op.outputs[1].name, layer_idx, field);
        if (layer_idx >= 0) {
            auto& acts = mRunState.simplified_acts(layer_idx);
            // If the output wasn't written to acts.residual_att, copy it there
            if (residual_out.Data != acts.residual_att.Data && acts.residual_att.Data) {
                CUDA_CHECK(cudaMemcpyAsync(acts.residual_att.Data, residual_out.Data,
                                           residual_out.bytes(), cudaMemcpyDeviceToDevice,
                                           mRunState.MainStream));
            }
        }
    }
    if (op.outputs[1].name == "blocks[0].ln1" || op.outputs[1].name == "blocks[25].ln1") {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> vals(8);
        const long C = mConfig.HiddenSize;
        // Trace at position 3 (where tokens differ) - offset by 3*C
        const std::size_t pos3_offset = 3 * static_cast<std::size_t>(C);
        int layer_idx = (op.outputs[1].name == "blocks[0].ln1") ? 0 : 25;
        // Print residual_in at position 3
        cudaMemcpy(vals.data(), reinterpret_cast<float*>(residual_in.Data) + pos3_offset, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[FWD_LN1] Layer %d residual_in[pos3] name=%s data=%p vals=%.6f,%.6f,%.6f,%.6f\n",
                layer_idx, op.inputs[0].name.c_str(), residual_in.Data, vals[0], vals[1], vals[2], vals[3]);
        // Print input at position 3
        cudaMemcpy(vals.data(), reinterpret_cast<float*>(input.Data) + pos3_offset, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[FWD_LN1] Layer %d input[pos3] name=%s data=%p vals=%.6f,%.6f,%.6f,%.6f\n",
                layer_idx, op.inputs[1].name.c_str(), input.Data, vals[0], vals[1], vals[2], vals[3]);
        // Print y (ln output) at position 3
        cudaMemcpy(vals.data(), reinterpret_cast<float*>(y.Data) + pos3_offset, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[FWD_LN1] Layer %d ln1_out[pos3] data=%p vals=%.6f,%.6f,%.6f,%.6f\n",
                layer_idx, y.Data, vals[0], vals[1], vals[2], vals[3]);
    }
}

void CompiledExecutor::dispatch_fused_residual_rmsnorm_backward(const CompiledOp& op) {
    // inputs: d_y, d_residual_next (may be empty), residual_out, weight, rstd
    // outputs: d_residual, d_input, d_weight (optional)

    // DEBUG: Trace what name is being resolved for d_y in layer 26 LN1
    static int rmsnorm_trace_count = 0;
    if (rmsnorm_trace_count < 5 && op.inputs[3].name.find("blocks[26].ln1_weight") != std::string::npos) {
        fprintf(stderr, "[RMSNORM_BWD] Layer 26 LN1: d_y input name='%s', slot=%d, layer_idx=%d\n",
                op.inputs[0].name.c_str(), static_cast<int>(op.inputs[0].slot), op.inputs[0].layer_idx);
        rmsnorm_trace_count++;
    }

    Tensor& d_y = resolve_tensor(op.inputs[0]);

    const bool is_final_norm =
        (op.inputs[3].name.find("final_norm") != std::string::npos ||
         op.inputs[3].name.find("ln_final") != std::string::npos ||
         op.inputs[3].name.find("ln_f") != std::string::npos);

    // DEBUG: Trace final RMSNorm d_y input name/values (layer_idx == -1 / final_norm).
    static int final_ln_trace_count = 0;
    if (final_ln_trace_count < 5 &&
        is_final_norm) {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> vals(4, 0.0f);
        const bool ok = copy_tensor_sample_as_f32(d_y, vals.size(), vals);
        double l2_slice = 0.0;
        std::size_t slice_offset = 0;
        std::size_t slice_count = 0;
        if (d_y.Data && d_y.nelem() > 0) {
            const std::size_t n = static_cast<std::size_t>(d_y.nelem());
            slice_offset = n / 2;
            slice_count = std::min<std::size_t>(4096, n - slice_offset);
            if (slice_count > 0) {
                Tensor tmp = d_y;
                tmp.Data = static_cast<std::byte*>(tmp.Data) +
                           slice_offset * get_dtype_size(tmp.DType);
                tmp.Sizes[0] = static_cast<long>(slice_count);
                tmp.Rank = 1;
                std::vector<float> buf;
                if (copy_tensor_sample_as_f32(tmp, slice_count, buf)) {
                    for (std::size_t i = 0; i < slice_count; ++i) {
                        const double v = static_cast<double>(buf[i]);
                        l2_slice += v * v;
                    }
                    l2_slice = std::sqrt(l2_slice);
                }
            }
        }
        fprintf(stderr,
                "[RMSNORM_BWD_FINAL] weight=%s d_y_name=%s slot=%d layer_idx=%d ptr=%p dtype=%d ok=%d "
                "mid_offset=%zu mid_count=%zu mid_l2=%.9e vals=%.9f,%.9f,%.9f,%.9f\n",
                op.inputs[3].name.c_str(),
                op.inputs[0].name.c_str(),
                static_cast<int>(op.inputs[0].slot),
                op.inputs[0].layer_idx,
                d_y.Data,
                static_cast<int>(d_y.DType),
                ok ? 1 : 0,
                slice_offset,
                slice_count,
                l2_slice,
                vals[0], vals[1], vals[2], vals[3]);
        final_ln_trace_count++;
    }

    // DEBUG: Print d_y values for layer 26 LN1
    static int d_y_trace_count = 0;
    if (d_y_trace_count < 5 && op.inputs[3].name.find("blocks[26].ln1_weight") != std::string::npos) {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> vals(4);
        cudaMemcpy(vals.data(), d_y.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[RMSNORM_BWD] Layer 26 LN1: d_y ptr=%p, values=%.9f,%.9f,%.9f,%.9f\n",
                d_y.Data, vals[0], vals[1], vals[2], vals[3]);
        d_y_trace_count++;
    }
    Tensor* residual_out_ptr = &resolve_tensor(op.inputs[2]);
    Tensor& weight = resolve_tensor(op.inputs[3]);
    Tensor& rstd = resolve_tensor(op.inputs[4]);

    int ln_layer_idx = -1;
    std::string ln_field;
    if (!op.inputs[3].name.empty()) {
        parse_block_param(op.inputs[3].name, ln_layer_idx, ln_field);
    }
    if (ln_layer_idx >= 0 && ln_field == "ln1_weight") {
        // LN1 backward expects residual_out from the forward fused residual op.
        // In the DSL graph, residual_out is res_ffn for the SAME layer index.
        // Ensure the correct per-layer residual buffer is used (especially with offloading).
        if (mRunState.has_residual_offloading()) {
            mRunState.fetch_residual(ln_layer_idx, mRunState.side_stream());
        }
        residual_out_ptr = &mRunState.get_residual(ln_layer_idx, mRunState.MainStream);
    }
    // FIX: LN2 backward needs the saved/recomputed residual_att from simplified_acts.
    // The backward graph may have wrong tensor names for the last layer (e.g., "StackedBlocks_N"
    // instead of "blocks[N].res_att"), causing it to resolve to stale/wrong data.
    // Always use the simplified_acts residual_att which is either saved (no recompute) or
    // recomputed (with recompute) to ensure correct gradient computation.
    if (ln_layer_idx >= 0 && ln_field == "ln2_weight") {
        auto& acts = mRunState.simplified_acts(ln_layer_idx);
        if (ln_layer_idx == 27) {
            fprintf(stderr, "[LN2_BWD_FIX] Layer %d: original residual_out=%p, simplified_acts.residual_att=%p (Data=%p)\n",
                    ln_layer_idx, residual_out_ptr->Data, &acts.residual_att, acts.residual_att.Data);
        }
        residual_out_ptr = &acts.residual_att;
    }
    Tensor& residual_out = *residual_out_ptr;

    // DEBUG: Print dtypes for final RMSNorm inputs to catch mismatches (e.g., BF16 vs I32).
    static int final_ln_dtype_trace = 0;
    if (is_final_norm && final_ln_dtype_trace < 8) {
        const char* dres_name = op.inputs[1].name.empty() ? "<none>" : op.inputs[1].name.c_str();
        Tensor* dres_ptr = nullptr;
        if (!op.inputs[1].name.empty()) {
            dres_ptr = &resolve_tensor(op.inputs[1]);
        }
        fprintf(stderr,
                "[RMSNORM_BWD_FINAL_DTYPES] d_y=%s dtype=%s | d_res_next=%s dtype=%s | residual_out=%s dtype=%s | weight=%s dtype=%s | rstd=%s dtype=%s\n",
                op.inputs[0].name.c_str(), dtype_to_str(d_y.DType),
                dres_name, dres_ptr ? dtype_to_str(dres_ptr->DType) : "<none>",
                op.inputs[2].name.c_str(), dtype_to_str(residual_out.DType),
                op.inputs[3].name.c_str(), dtype_to_str(weight.DType),
                op.inputs[4].name.c_str(), dtype_to_str(rstd.DType));
        final_ln_dtype_trace++;
    }

    // d_residual_next is the incoming gradient from the next layer (may be zero/empty)
    Tensor d_residual_zero{};
    Tensor* d_residual_next = nullptr;
    if (!op.inputs[1].name.empty()) {
        d_residual_next = &resolve_tensor(op.inputs[1]);
    } else {
        // Allocate and zero a temporary for d_residual if none provided
        d_residual_zero = mRunState.temp_alloc(d_y.DType, {mB, mT, static_cast<long>(mConfig.HiddenSize)});
        fill_zero(d_residual_zero, mRunState.MainStream);
        mTemps.push_back(d_residual_zero);
        d_residual_next = &d_residual_zero;
    }
    Tensor* d_residual_input = d_residual_next;
    Tensor* d_residual_stream = d_residual_next;

    // DEBUG: Trace top-layer LN1 d_y and optional zeroing to isolate stale/NaN gradients.
    const int num_layers = static_cast<int>(mConfig.NumLayers);
    static int ln1_top_trace = 0;
    if (ln_field == "ln1_weight" && ln_layer_idx == num_layers - 1 && ln1_top_trace < 8) {
        fprintf(stderr,
                "[RMS_BWD_LN1_TOP] layer=%d d_y=%s slot=%d ptr=%p d_residual_next=%s ptr=%p\n",
                ln_layer_idx,
                op.inputs[0].name.c_str(),
                static_cast<int>(op.inputs[0].slot),
                d_y.Data,
                op.inputs[1].name.c_str(),
                d_residual_next ? d_residual_next->Data : nullptr);
        log_tensor_stats_ex("RMS_BWD_LN1_TOP_DY", ln_layer_idx, op.inputs[0].name, d_y, 4096, true);
        if (d_residual_next && d_residual_next->Data) {
            log_tensor_stats_ex("RMS_BWD_LN1_TOP_DRES", ln_layer_idx, op.inputs[1].name, *d_residual_next, 4096, true);
        }
        ln1_top_trace++;
    }
    static int ln1_top_nan_logged = 0;
    if (ln_field == "ln1_weight" && ln_layer_idx == num_layers - 1 && ln1_top_nan_logged < 4) {
        if (tensor_sample_has_nan_or_inf(d_y, 3)) {
            fprintf(stderr,
                    "[RMS_BWD_LN1_TOP_NAN] layer=%d d_y=%s ptr=%p\n",
                    ln_layer_idx,
                    op.inputs[0].name.c_str(),
                    d_y.Data);
            ln1_top_nan_logged++;
        }
    }
    static int ln1_top_zero = -1;
    if (ln1_top_zero < 0) {
        ln1_top_zero = (std::getenv("SUROGATE_DEBUG_ZERO_LN1_DY") != nullptr) ? 1 : 0;
    }
    if (ln1_top_zero && ln_field == "ln1_weight" && ln_layer_idx == num_layers - 1) {
        fprintf(stderr, "[RMS_BWD_LN1_TOP_ZERO] layer=%d zeroing d_y ptr=%p\n",
                ln_layer_idx, d_y.Data);
        fill_zero(d_y, mRunState.MainStream);
    }

    // DEBUG: Print d_residual_next and check for aliasing with d_input
    if (ln_layer_idx == 26) {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> dres_vals(8);
        cudaMemcpy(dres_vals.data(), d_residual_next->Data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[RMSNORM_BWD_DRES] Layer %d %s: d_residual_next=%s ptr=%p, d_input_out=%s\n",
                ln_layer_idx, ln_field.c_str(),
                op.inputs[1].name.c_str(), d_residual_next->Data, op.outputs[1].name.c_str());
    }

    Tensor& d_input = ensure_output_tensor(op.outputs[1]);

    const Tensor& d_emb_global = mRunState.non_block_gradients().d_embeddings;
    const bool writes_to_embeddings = (d_emb_global.Data && d_input.Data == d_emb_global.Data);
    auto matches_output = [&](std::string_view target) -> bool {
        for (const auto& out_ref : op.outputs) {
            if (out_ref.name.empty()) continue;
            if (strip_ssa_suffix(out_ref.name) == target) {
                return true;
            }
        }
        return false;
    };
    const bool targets_res_ffn = matches_output("d_blocks[0].res_ffn");

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

    // Debug: track qkv matmul dA pointer to verify LN1 d_y wiring.
    if (g_qkv_dA_ptr_by_layer.empty() && mConfig.NumLayers > 0) {
        g_qkv_dA_ptr_by_layer.assign(static_cast<std::size_t>(mConfig.NumLayers), nullptr);
        g_qkv_dA_micro_by_layer.assign(static_cast<std::size_t>(mConfig.NumLayers), -1);
    }
    if (ln_field == "ln1_weight" && ln_layer_idx >= 0 &&
        ln_layer_idx < static_cast<int>(g_qkv_dA_ptr_by_layer.size())) {
        const std::byte* expected_ptr = g_qkv_dA_ptr_by_layer[static_cast<std::size_t>(ln_layer_idx)];
        const int expected_micro = g_qkv_dA_micro_by_layer[static_cast<std::size_t>(ln_layer_idx)];
        if (expected_ptr && expected_micro == mMicroStep && expected_ptr != d_y.Data) {
            fprintf(stderr,
                    "[LN1_DY_PTR_MISMATCH] layer=%d micro_step=%d d_y=%p expected_qkv_dA=%p\n",
                    ln_layer_idx,
                    mMicroStep,
                    d_y.Data,
                    expected_ptr);
        }
    }

    // Determine abs_max pointer for FP8 gradient quantization.
    // LN1 backward produces d_res_ffn (gradient for previous layer's residual).
    // LN2 backward produces d_res_att (gradient for attention path).
    float* abs_max_ptr = nullptr;
    if (mRunState.has_grad_quants()) {
        const bool is_ln2 = (ln_field == "ln2_weight");
        abs_max_ptr = is_ln2
            ? mRunState.simplified_quant_grads().d_res_att.abs_max()
            : mRunState.simplified_quant_grads().d_res_ffn.abs_max();
    }

    // DEBUG: Print rmsnorm backward inputs for all layers to trace divergence
    static int step_count = 0;
    static int print_count = 0;
    if (ln_field == "ln1_weight" && ln_layer_idx == num_layers - 1) {
        step_count++;
        print_count = 0;
    }
    // DEBUG: For layer 26 LN1, print the d_y tensor name to trace where it comes from
    if (ln_layer_idx == 26 && ln_field == "ln1_weight") {
        fprintf(stderr, "[RMSNORM_BWD_NAMES] Layer %d %s: d_y=%s, d_residual_next=%s\n",
                ln_layer_idx, ln_field.c_str(), op.inputs[0].name.c_str(), op.inputs[1].name.c_str());
    }
    if (step_count == 1 && print_count < 60 && (ln_layer_idx >= num_layers - 5 || ln_layer_idx <= 3)) {
        print_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> res_vals(8), rstd_vals(8), dy_vals(8);
        cudaMemcpy(res_vals.data(), residual_out.Data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(rstd_vals.data(), rstd.Data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dy_vals.data(), d_y.Data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[RMSNORM_BWD] Layer %d %s: residual_out=%.6f,%.6f,%.6f, rstd=%.6f,%.6f,%.6f, d_y=%.6f,%.6f,%.6f\n",
                ln_layer_idx, ln_field.c_str(),
                res_vals[0], res_vals[1], res_vals[2],
                rstd_vals[0], rstd_vals[1], rstd_vals[2],
                dy_vals[0], dy_vals[1], dy_vals[2]);
    }

    static int emb_rms_trace = 0;
    if (writes_to_embeddings && emb_rms_trace < 8) {
        fprintf(stderr,
                "[RMS_BWD_EMB] op_id=%s layer=%d field=%s d_y=%s d_res_next=%s\n",
                op.op_id.c_str(), ln_layer_idx, ln_field.c_str(),
                op.inputs[0].name.c_str(), op.inputs[1].name.c_str());
        log_tensor_stats_ex("RMS_BWD_EMB_DY", ln_layer_idx, op.inputs[0].name, d_y, 4096, true);
        log_tensor_stats_ex("RMS_BWD_EMB_RES", ln_layer_idx, op.inputs[2].name, residual_out, 4096, true);
        log_tensor_stats_ex("RMS_BWD_EMB_RSTD", ln_layer_idx, op.inputs[4].name, rstd, 4096, true);
        log_tensor_stats_ex("RMS_BWD_EMB_W", ln_layer_idx, op.inputs[3].name, weight, 4096, true);
        if (d_residual_next && d_residual_next->Data) {
            log_tensor_stats_ex("RMS_BWD_EMB_DRES", ln_layer_idx, op.inputs[1].name, *d_residual_next, 4096, true);
        }
        emb_rms_trace++;
    }

    static int rms_resffn_trace = 0;
    if (targets_res_ffn && rms_resffn_trace < 8) {
        fprintf(stderr,
                "[RMS_BWD_RESFFN] op_id=%s layer=%d field=%s d_y=%s d_res_next=%s\n",
                op.op_id.c_str(), ln_layer_idx, ln_field.c_str(),
                op.inputs[0].name.c_str(), op.inputs[1].name.c_str());
        log_tensor_stats_ex("RMS_BWD_RESFFN_DY", ln_layer_idx, op.inputs[0].name, d_y, 4096, true);
        log_tensor_stats_ex("RMS_BWD_RESFFN_RES", ln_layer_idx, op.inputs[2].name, residual_out, 4096, true);
        log_tensor_stats_ex("RMS_BWD_RESFFN_RSTD", ln_layer_idx, op.inputs[4].name, rstd, 4096, true);
        log_tensor_stats_ex("RMS_BWD_RESFFN_W", ln_layer_idx, op.inputs[3].name, weight, 4096, true);
        if (d_residual_next && d_residual_next->Data) {
            log_tensor_stats_ex("RMS_BWD_RESFFN_DRES", ln_layer_idx, op.inputs[1].name, *d_residual_next, 4096, true);
        }
        rms_resffn_trace++;
    }

    // Magnitude tracing for LN2 backward (MoE path debugging).
    static int ln2_mag_trace = 0;
    if (ln_field == "ln2_weight" && ln_layer_idx <= 2 && ln2_mag_trace < 8) {
        log_tensor_mag("LN2_BWD_DY", ln_layer_idx, op.inputs[0].name, d_y, 4096);
        log_tensor_mag("LN2_BWD_RES", ln_layer_idx, op.inputs[2].name, residual_out, 4096);
        if (d_residual_next && d_residual_next->Data) {
            log_tensor_mag("LN2_BWD_DRES", ln_layer_idx, op.inputs[1].name, *d_residual_next, 4096);
        }
        ln2_mag_trace++;
    }

    // Targeted LN2 backward logs for layer 8/9 to trace the first residual spike.
    static int ln2_l8_trace = 0;
    if (ln_field == "ln2_weight" && (ln_layer_idx == 8 || ln_layer_idx == 9) && ln2_l8_trace < 12) {
        fprintf(stderr,
                "[LN2_BWD_L8] layer=%d d_y=%s res_out=%s rstd=%s weight=%s d_res_next=%s\n",
                ln_layer_idx,
                op.inputs[0].name.c_str(),
                op.inputs[2].name.c_str(),
                op.inputs[4].name.c_str(),
                op.inputs[3].name.c_str(),
                op.inputs[1].name.empty() ? "<none>" : op.inputs[1].name.c_str());
        log_tensor_mag_unbounded("LN2_L8_DY", ln_layer_idx, op.inputs[0].name, d_y, 4096);
        log_tensor_mag_unbounded("LN2_L8_RES", ln_layer_idx, op.inputs[2].name, residual_out, 4096);
        log_tensor_mag_unbounded("LN2_L8_RSTD", ln_layer_idx, op.inputs[4].name, rstd, 4096);
        log_tensor_mag_unbounded("LN2_L8_W", ln_layer_idx, op.inputs[3].name, weight, 4096);
        if (d_residual_next && d_residual_next->Data) {
            log_tensor_mag_unbounded("LN2_L8_DRES", ln_layer_idx, op.inputs[1].name, *d_residual_next, 4096);
        }
        ln2_l8_trace++;
    }

    static int ln1_mag_trace = 0;
    if (ln_field == "ln1_weight" && ln_layer_idx <= 2 && ln1_mag_trace < 8) {
        log_tensor_mag("LN1_BWD_DY", ln_layer_idx, op.inputs[0].name, d_y, 4096);
        log_tensor_mag("LN1_BWD_RES", ln_layer_idx, op.inputs[2].name, residual_out, 4096);
        if (d_residual_next && d_residual_next->Data) {
            log_tensor_mag("LN1_BWD_DRES", ln_layer_idx, op.inputs[1].name, *d_residual_next, 4096);
        }
        ln1_mag_trace++;
    }

    // Targeted LN1 backward logs for layer 8/9 (paired with LN2 logs).
    static int ln1_l8_trace = 0;
    if (ln_field == "ln1_weight" && (ln_layer_idx == 8 || ln_layer_idx == 9) && ln1_l8_trace < 12) {
        fprintf(stderr,
                "[LN1_BWD_L8] layer=%d d_y=%s res_out=%s rstd=%s weight=%s d_res_next=%s\n",
                ln_layer_idx,
                op.inputs[0].name.c_str(),
                op.inputs[2].name.c_str(),
                op.inputs[4].name.c_str(),
                op.inputs[3].name.c_str(),
                op.inputs[1].name.empty() ? "<none>" : op.inputs[1].name.c_str());
        log_tensor_mag_unbounded("LN1_L8_DY", ln_layer_idx, op.inputs[0].name, d_y, 4096);
        log_tensor_mag_unbounded("LN1_L8_RES", ln_layer_idx, op.inputs[2].name, residual_out, 4096);
        log_tensor_mag_unbounded("LN1_L8_RSTD", ln_layer_idx, op.inputs[4].name, rstd, 4096);
        log_tensor_mag_unbounded("LN1_L8_W", ln_layer_idx, op.inputs[3].name, weight, 4096);
        if (d_residual_next && d_residual_next->Data) {
            log_tensor_mag_unbounded("LN1_L8_DRES", ln_layer_idx, op.inputs[1].name, *d_residual_next, 4096);
        }
        ln1_l8_trace++;
    }

    // Pre-check LN2 backward input for NaNs anywhere in d_y.
    static int rms_ln2_pre_nan_logged = 0;
    if (ln_field == "ln2_weight" && rms_ln2_pre_nan_logged < 4) {
        Tensor d_y_scan = d_y;
        if (d_y_scan.Rank > 2 && d_y_scan.Sizes[0] == mB && d_y_scan.Sizes[1] == mT) {
            d_y_scan = view_tensor(d_y_scan, {mB * mT, d_y_scan.Sizes[d_y_scan.Rank - 1]});
        }
        long dy_row = -1;
        float dy_min = 0.0f;
        float dy_max = 0.0f;
        if (find_first_nan_row(d_y_scan, &dy_row, &dy_min, &dy_max)) {
            const long b = (d_y_scan.Rank >= 2 && d_y_scan.Sizes[0] == mB * mT)
                ? (dy_row / static_cast<long>(mT)) : -1;
            const long t = (d_y_scan.Rank >= 2 && d_y_scan.Sizes[0] == mB * mT)
                ? (dy_row % static_cast<long>(mT)) : -1;
            fprintf(stderr,
                    "[RMS_BWD_LN2_PRE_NAN] layer=%d d_y=%s row=%ld b=%ld t=%ld min=%.6f max=%.6f\n",
                    ln_layer_idx,
                    op.inputs[0].name.c_str(),
                    dy_row, b, t, dy_min, dy_max);
            log_tensor_stats_ex("RMS_BWD_LN2_PRE_DY", ln_layer_idx, op.inputs[0].name, d_y, 4096, true);
            log_tensor_stats_ex("RMS_BWD_LN2_PRE_RES", ln_layer_idx, op.inputs[2].name, residual_out, 4096, true);
            log_tensor_stats_ex("RMS_BWD_LN2_PRE_RSTD", ln_layer_idx, op.inputs[4].name, rstd, 4096, true);
            rms_ln2_pre_nan_logged++;
        }
    }

    // DEBUG: Trace LN1 backward at layer 1 to locate upstream explosion.
    bool trace_ln1_l1 = false;
    static int ln1_l1_trace = 0;
    if (ln_field == "ln1_weight" && ln_layer_idx == 1 && ln1_l1_trace < 8) {
        trace_ln1_l1 = true;
        fprintf(stderr,
                "[RMS_BWD_LN1_L1] op_id=%s d_y=%s d_res_next=%s residual_out=%s\n",
                op.op_id.c_str(),
                op.inputs[0].name.c_str(),
                op.inputs[1].name.empty() ? "<none>" : op.inputs[1].name.c_str(),
                op.inputs[2].name.c_str());
        log_tensor_stats_ex("RMS_BWD_LN1_L1_DY", ln_layer_idx, op.inputs[0].name, d_y, 4096, true);
        log_tensor_stats_ex("RMS_BWD_LN1_L1_RES", ln_layer_idx, op.inputs[2].name, residual_out, 4096, true);
        if (d_residual_next && d_residual_next->Data) {
            log_tensor_stats_ex("RMS_BWD_LN1_L1_DRES", ln_layer_idx, op.inputs[1].name, *d_residual_next, 4096, true);
        }
        ln1_l1_trace++;
    }

    if (is_final_norm) {
        const char* dres_name = op.inputs[1].name.empty() ? "<none>" : op.inputs[1].name.c_str();
        const char* dweight_name = (op.outputs.size() > 2 && !op.outputs[2].name.empty())
                                       ? op.outputs[2].name.c_str()
                                       : "<dummy>";
        fprintf(stderr,
                "[RMSNORM_BWD_FINAL_INPUTS] d_input=%s dtype=%s ptr=%p | d_weight=%s dtype=%s ptr=%p | "
                "d_res_next=%s dtype=%s ptr=%p\n",
                op.outputs.size() > 1 ? op.outputs[1].name.c_str() : "<none>",
                dtype_to_str(d_input.DType), d_input.Data,
                dweight_name, d_weight_ptr ? dtype_to_str(d_weight_ptr->DType) : "<none>",
                d_weight_ptr ? d_weight_ptr->Data : nullptr,
                dres_name,
                (d_residual_input && d_residual_input->Data) ? dtype_to_str(d_residual_input->DType) : "<none>",
                d_residual_input ? d_residual_input->Data : nullptr);
        fflush(stderr);
    }

    rmsnorm_backward(d_input, *d_weight_ptr, mRunState.scratch().rmsnorm_scratch,
                     *d_residual_input, d_y, residual_out, weight, rstd,
                     abs_max_ptr,
                     static_cast<int>(mB), static_cast<int>(mT), C,
                     mRunState.DeviceProp, mRunState.MainStream, skip_weight_grad);

    // One-time per-micro-step scanner for the first residual-gradient spike.
    // Enabled with SUROGATE_SCAN_RESIDUAL_SPIKE=1.
    static int spike_scan_enabled = -1;
    if (spike_scan_enabled < 0) {
        spike_scan_enabled = (std::getenv("SUROGATE_SCAN_RESIDUAL_SPIKE") != nullptr) ? 1 : 0;
    }
    if (spike_scan_enabled && ln_layer_idx >= 0 &&
        (ln_field == "ln1_weight" || ln_field == "ln2_weight")) {
        struct ResidualSpikeState {
            int micro_step = -1;
            bool ln1_logged = false;
            bool ln2_logged = false;
            double ln1_prev = 0.0;
            int ln1_prev_layer = -1;
            double ln2_prev = 0.0;
            int ln2_prev_layer = -1;
        };
        static ResidualSpikeState spike_state;
        if (spike_state.micro_step != mMicroStep) {
            spike_state = {};
            spike_state.micro_step = mMicroStep;
        }

        const float ratio_thresh = env_float("SUROGATE_RESIDUAL_SPIKE_RATIO", 50.0f);
        const float abs_thresh = env_float("SUROGATE_RESIDUAL_SPIKE_ABS", 100.0f);

        auto maybe_log_spike = [&](const char* field,
                                   bool& logged,
                                   double& prev_mean,
                                   int& prev_layer) {
            if (logged) {
                return;
            }
            const double cur_mean = sample_mean_abs(d_input, 4096);
            const double in_mean =
                (d_residual_next && d_residual_next->Data) ? sample_mean_abs(*d_residual_next, 4096) : 0.0;
            if (prev_mean > 0.0) {
                const double ratio = cur_mean / prev_mean;
                if (cur_mean >= abs_thresh && ratio >= ratio_thresh) {
                    fprintf(stderr,
                            "[RESIDUAL_SPIKE] micro_step=%d field=%s layer=%d prev_layer=%d "
                            "prev_mean=%.6e cur_mean=%.6e ratio=%.2f in_mean=%.6e d_input=%s d_res_next=%s\n",
                            mMicroStep,
                            field,
                            ln_layer_idx,
                            prev_layer,
                            prev_mean,
                            cur_mean,
                            ratio,
                            in_mean,
                            op.outputs.size() > 1 ? op.outputs[1].name.c_str() : "<none>",
                            op.inputs[1].name.empty() ? "<none>" : op.inputs[1].name.c_str());
                    logged = true;
                }
            }
            prev_mean = cur_mean;
            prev_layer = ln_layer_idx;
        };

        if (ln_field == "ln1_weight") {
            maybe_log_spike("ln1", spike_state.ln1_logged, spike_state.ln1_prev, spike_state.ln1_prev_layer);
        } else if (ln_field == "ln2_weight") {
            maybe_log_spike("ln2", spike_state.ln2_logged, spike_state.ln2_prev, spike_state.ln2_prev_layer);
        }
    }

    static int ln2_out_mag_trace = 0;
    if (ln_field == "ln2_weight" && ln_layer_idx <= 2 && ln2_out_mag_trace < 8) {
        log_tensor_mag("LN2_BWD_DINPUT", ln_layer_idx,
                       op.outputs.size() > 1 ? op.outputs[1].name : "<none>",
                       d_input, 4096);
        ln2_out_mag_trace++;
    }

    static int ln1_out_mag_trace = 0;
    if (ln_field == "ln1_weight" && ln_layer_idx <= 2 && ln1_out_mag_trace < 8) {
        log_tensor_mag("LN1_BWD_DINPUT", ln_layer_idx,
                       op.outputs.size() > 1 ? op.outputs[1].name : "<none>",
                       d_input, 4096);
        ln1_out_mag_trace++;
    }

    // Trace final norm backward outputs to see if explosion starts at the top.
    static int final_bwd_mag_trace = 0;
    if (is_final_norm && final_bwd_mag_trace < 8) {
        if (op.outputs.size() > 0 && !op.outputs[0].name.empty()) {
            log_tensor_mag("FINAL_BWD_DRES", ln_layer_idx, op.outputs[0].name, *d_residual_input, 4096);
        }
        if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
            log_tensor_mag("FINAL_BWD_DX", ln_layer_idx, op.outputs[1].name, d_input, 4096);
        }
        final_bwd_mag_trace++;
    }

    // Post-check LN2 backward output for NaNs anywhere in d_input.
    static int rms_ln2_post_nan_logged = 0;
    if (ln_field == "ln2_weight" && rms_ln2_post_nan_logged < 4) {
        Tensor d_input_scan = d_input;
        if (d_input_scan.Rank > 2 && d_input_scan.Sizes[0] == mB && d_input_scan.Sizes[1] == mT) {
            d_input_scan = view_tensor(d_input_scan, {mB * mT, d_input_scan.Sizes[d_input_scan.Rank - 1]});
        }
        long di_row = -1;
        float di_min = 0.0f;
        float di_max = 0.0f;
        if (find_first_nan_row(d_input_scan, &di_row, &di_min, &di_max)) {
            const long b = (d_input_scan.Rank >= 2 && d_input_scan.Sizes[0] == mB * mT)
                ? (di_row / static_cast<long>(mT)) : -1;
            const long t = (d_input_scan.Rank >= 2 && d_input_scan.Sizes[0] == mB * mT)
                ? (di_row % static_cast<long>(mT)) : -1;
            fprintf(stderr,
                    "[RMS_BWD_LN2_POST_NAN] layer=%d d_input=%s row=%ld b=%ld t=%ld min=%.6f max=%.6f\n",
                    ln_layer_idx,
                    op.outputs.size() > 1 ? op.outputs[1].name.c_str() : "<none>",
                    di_row, b, t, di_min, di_max);
            log_tensor_stats_ex("RMS_BWD_LN2_POST_DINPUT", ln_layer_idx,
                                op.outputs.size() > 1 ? op.outputs[1].name : "<none>", d_input, 4096, true);
            rms_ln2_post_nan_logged++;
        }
    }

    // One-time NaN watchdog for LN1 outputs (feeds next block / MoE input).
    static bool rms_ln1_nan_logged = false;
    if (!rms_ln1_nan_logged && ln_field == "ln1_weight") {
        if (tensor_sample_has_nan_or_inf(d_input, 3)) {
            auto dump_sample = [](const char* tag, const Tensor& t, const std::string& name) {
                std::vector<float> vals(4, 0.0f);
                const bool ok = copy_tensor_sample_as_f32(t, vals.size(), vals);
                int nan = 0;
                int inf = 0;
                for (float v : vals) {
                    if (std::isnan(v)) {
                        nan++;
                    } else if (std::isinf(v)) {
                        inf++;
                    }
                }
                fprintf(stderr,
                        "[%s] name=%s dtype=%s ok=%d nan=%d inf=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                        tag,
                        name.c_str(),
                        dtype_to_str(t.DType),
                        ok ? 1 : 0,
                        nan,
                        inf,
                        vals[0], vals[1], vals[2], vals[3]);
            };
            fprintf(stderr,
                    "[RMS_BWD_LN1_NAN] op=%s layer=%d d_input=%s d_y=%s\n",
                    op.op_id.c_str(),
                    ln_layer_idx,
                    op.outputs.size() > 1 ? op.outputs[1].name.c_str() : "<none>",
                    op.inputs[0].name.c_str());
            dump_sample("RMS_BWD_LN1_NAN_DY", d_y, op.inputs[0].name);
            dump_sample("RMS_BWD_LN1_NAN_DIN", d_input, op.outputs[1].name);
            dump_sample("RMS_BWD_LN1_NAN_RES", residual_out, op.inputs[2].name);
            dump_sample("RMS_BWD_LN1_NAN_RSTD", rstd, op.inputs[4].name);
            if (d_residual_next && d_residual_next->Data) {
                dump_sample("RMS_BWD_LN1_NAN_DRES", *d_residual_next, op.inputs[1].name);
            }
            fprintf(stderr,
                    "[RMS_BWD_LN1_NAN_PTRS] d_y=%p d_res_next=%p d_input=%p residual_out=%p weight=%p rstd=%p\n",
                    d_y.Data,
                    d_residual_next ? d_residual_next->Data : nullptr,
                    d_input.Data,
                    residual_out.Data,
                    weight.Data,
                    rstd.Data);
            auto scan_nan_rows = [&](const char* tag, const Tensor& t, bool flatten_bt) {
                Tensor scan = t;
                if (flatten_bt && scan.Rank > 2 && scan.Sizes[0] == mB && scan.Sizes[1] == mT) {
                    scan = view_tensor(scan, {mB * mT, scan.Sizes[scan.Rank - 1]});
                }
                long row = -1;
                float row_min = 0.0f;
                float row_max = 0.0f;
                if (find_first_nan_row(scan, &row, &row_min, &row_max)) {
                    const long b = (scan.Rank >= 2 && scan.Sizes[0] == mB * mT)
                        ? (row / static_cast<long>(mT)) : -1;
                    const long t_idx = (scan.Rank >= 2 && scan.Sizes[0] == mB * mT)
                        ? (row % static_cast<long>(mT)) : -1;
                    fprintf(stderr,
                            "[%s] layer=%d row=%ld b=%ld t=%ld min=%.6f max=%.6f\n",
                            tag, ln_layer_idx, row, b, t_idx, row_min, row_max);
                    log_tensor_stats_ex(tag, ln_layer_idx, "<scan>", t, 4096, true);
                }
            };
            scan_nan_rows("RMS_BWD_LN1_NAN_DY_ROW", d_y, true);
            if (d_residual_next && d_residual_next->Data) {
                scan_nan_rows("RMS_BWD_LN1_NAN_DRES_ROW", *d_residual_next, true);
            }
            scan_nan_rows("RMS_BWD_LN1_NAN_RES_ROW", residual_out, true);
            if (tensor_sample_has_nan_or_inf(weight, 3)) {
                dump_sample("RMS_BWD_LN1_NAN_W", weight, op.inputs[3].name);
            }
            rms_ln1_nan_logged = true;
        }
    }

    // One-time NaN watchdog for LN2 outputs (feeds attention output matmul).
    static bool rms_ln2_nan_logged = false;
    if (!rms_ln2_nan_logged && ln_field == "ln2_weight") {
        if (tensor_sample_has_nan_or_inf(d_input, 3)) {
            auto dump_sample = [](const char* tag, const Tensor& t, const std::string& name) {
                std::vector<float> vals(4, 0.0f);
                const bool ok = copy_tensor_sample_as_f32(t, vals.size(), vals);
                int nan = 0;
                int inf = 0;
                for (float v : vals) {
                    if (std::isnan(v)) {
                        nan++;
                    } else if (std::isinf(v)) {
                        inf++;
                    }
                }
                fprintf(stderr,
                        "[%s] name=%s dtype=%s ok=%d nan=%d inf=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                        tag,
                        name.c_str(),
                        dtype_to_str(t.DType),
                        ok ? 1 : 0,
                        nan,
                        inf,
                        vals[0], vals[1], vals[2], vals[3]);
            };
            fprintf(stderr,
                    "[RMS_BWD_LN2_NAN] op=%s layer=%d d_input=%s d_y=%s\n",
                    op.op_id.c_str(),
                    ln_layer_idx,
                    op.outputs.size() > 1 ? op.outputs[1].name.c_str() : "<none>",
                    op.inputs[0].name.c_str());
            dump_sample("RMS_BWD_LN2_NAN_DY", d_y, op.inputs[0].name);
            dump_sample("RMS_BWD_LN2_NAN_DIN", d_input, op.outputs[1].name);
            dump_sample("RMS_BWD_LN2_NAN_RES", residual_out, op.inputs[2].name);
            dump_sample("RMS_BWD_LN2_NAN_RSTD", rstd, op.inputs[4].name);
            if (d_residual_next && d_residual_next->Data) {
                dump_sample("RMS_BWD_LN2_NAN_DRES", *d_residual_next, op.inputs[1].name);
            }
            rms_ln2_nan_logged = true;
        }
    }

    if (trace_ln1_l1) {
        log_tensor_stats_ex("RMS_BWD_LN1_L1_DIN", ln_layer_idx, op.outputs[1].name, d_input, 4096, true);
    }

    // DEBUG: One-time per-layer RMS stats to locate divergence between recompute/no-recompute.
    static int rms_layer_trace_enabled = -1;
    if (rms_layer_trace_enabled < 0) {
        rms_layer_trace_enabled = (std::getenv("SUROGATE_TRACE_RMS_LAYER") != nullptr) ? 1 : 0;
    }
    if (rms_layer_trace_enabled && ln_layer_idx >= 0) {
        static std::vector<int> rms_layer_seen;
        const int num_layers = static_cast<int>(mConfig.NumLayers);
        if (rms_layer_seen.empty() && num_layers > 0) {
            rms_layer_seen.assign(static_cast<std::size_t>(num_layers * 2), 0);
        }
        const int field_idx = (ln_field == "ln2_weight") ? 1 : 0;
        const int slot_idx = ln_layer_idx * 2 + field_idx;
        if (slot_idx >= 0 && slot_idx < static_cast<int>(rms_layer_seen.size()) &&
            rms_layer_seen[static_cast<std::size_t>(slot_idx)] == 0) {
            rms_layer_seen[static_cast<std::size_t>(slot_idx)] = 1;
            cudaStreamSynchronize(mRunState.MainStream);
            const double dy_mean = sample_mean_abs(d_y, 4096);
            const double din_mean = sample_mean_abs(d_input, 4096);
            double dres_mean = 0.0;
            if (d_residual_next && d_residual_next->Data) {
                dres_mean = sample_mean_abs(*d_residual_next, 4096);
            }
            fprintf(stderr,
                    "[RMS_LAYER_STAT] layer=%d field=%s dy_mean=%.6e din_mean=%.6e dres_mean=%.6e\n",
                    ln_layer_idx, ln_field.c_str(), dy_mean, din_mean, dres_mean);
        }
    }

    static int final_ln_out_trace = 0;
    if (is_final_norm && final_ln_out_trace < 5) {
        log_tensor_stats_ex("RMS_BWD_FINAL_DINPUT", ln_layer_idx, op.outputs[1].name, d_input, 4096, true);
        final_ln_out_trace++;
    }

    static bool emb_rms_nan_logged = false;
    if (writes_to_embeddings && !emb_rms_nan_logged && tensor_sample_has_nan_or_inf(d_input, 3)) {
        fprintf(stderr,
                "[RMS_BWD_EMB_NAN] op_id=%s layer=%d field=%s\n",
                op.op_id.c_str(), ln_layer_idx, ln_field.c_str());
        log_tensor_stats_ex("RMS_BWD_EMB_DINPUT", ln_layer_idx, op.outputs[1].name, d_input, 4096, true);
        log_tensor_stats_ex("RMS_BWD_EMB_DY_NAN", ln_layer_idx, op.inputs[0].name, d_y, 4096, true);
        log_tensor_stats_ex("RMS_BWD_EMB_RES_NAN", ln_layer_idx, op.inputs[2].name, residual_out, 4096, true);
        log_tensor_stats_ex("RMS_BWD_EMB_RSTD_NAN", ln_layer_idx, op.inputs[4].name, rstd, 4096, true);
        log_tensor_stats_ex("RMS_BWD_EMB_W_NAN", ln_layer_idx, op.inputs[3].name, weight, 4096, true);
        if (d_residual_next && d_residual_next->Data) {
            log_tensor_stats_ex("RMS_BWD_EMB_DRES_NAN", ln_layer_idx, op.inputs[1].name, *d_residual_next, 4096, true);
        }
        emb_rms_nan_logged = true;
    }

    static bool rms_resffn_nan_logged = false;
    if (targets_res_ffn && !rms_resffn_nan_logged && tensor_sample_has_nan_or_inf(d_input, 3)) {
        fprintf(stderr,
                "[RMS_BWD_RESFFN_NAN] op_id=%s layer=%d field=%s\n",
                op.op_id.c_str(), ln_layer_idx, ln_field.c_str());
        log_tensor_stats_ex("RMS_BWD_RESFFN_DINPUT", ln_layer_idx, op.outputs[1].name, d_input, 4096, true);
        log_tensor_stats_ex("RMS_BWD_RESFFN_DY_NAN", ln_layer_idx, op.inputs[0].name, d_y, 4096, true);
        log_tensor_stats_ex("RMS_BWD_RESFFN_RES_NAN", ln_layer_idx, op.inputs[2].name, residual_out, 4096, true);
        log_tensor_stats_ex("RMS_BWD_RESFFN_RSTD_NAN", ln_layer_idx, op.inputs[4].name, rstd, 4096, true);
        log_tensor_stats_ex("RMS_BWD_RESFFN_W_NAN", ln_layer_idx, op.inputs[3].name, weight, 4096, true);
        if (d_residual_next && d_residual_next->Data) {
            log_tensor_stats_ex("RMS_BWD_RESFFN_DRES_NAN", ln_layer_idx, op.inputs[1].name, *d_residual_next, 4096, true);
        }
        rms_resffn_nan_logged = true;
    }

    // DEBUG: Print d_input OUTPUT for layer 26 to trace gradient flow
    if (ln_layer_idx == 26 && ln_field == "ln2_weight") {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> dinp_vals(8);
        cudaMemcpy(dinp_vals.data(), d_input.Data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[RMSNORM_BWD_OUT] Layer %d %s: d_input OUTPUT ptr=%p, values=%.6f,%.6f,%.6f,%.6f\n",
                ln_layer_idx, ln_field.c_str(), d_input.Data,
                dinp_vals[0], dinp_vals[1], dinp_vals[2], dinp_vals[3]);
    }

    // DEBUG: Trace Layer 24 and 25 rmsnorm_backward for explosion debugging
    static int ln_24_25_trace = 0;
    if ((ln_layer_idx == 24 || ln_layer_idx == 25) && ln_24_25_trace < 20) {
        ln_24_25_trace++;
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> dinp_vals(4), dy_vals(4), dres_vals(4);
        cudaMemcpy(dinp_vals.data(), d_input.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dy_vals.data(), d_y.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dres_vals.data(), d_residual_next->Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[RMSNORM_BWD_L24_25] Layer %d %s: d_y_in=%.6f,%.6f,%.6f,%.6f d_res_next=%.6f,%.6f,%.6f,%.6f d_input_out=%.6f,%.6f,%.6f,%.6f\n",
                ln_layer_idx, ln_field.c_str(),
                dy_vals[0], dy_vals[1], dy_vals[2], dy_vals[3],
                dres_vals[0], dres_vals[1], dres_vals[2], dres_vals[3],
                dinp_vals[0], dinp_vals[1], dinp_vals[2], dinp_vals[3]);
    }

    // Copy d_input to d_residual if they're different outputs
    if (!op.outputs[0].name.empty() && op.outputs[0].name != op.outputs[1].name) {
        Tensor& d_residual = ensure_output_tensor(op.outputs[0]);
        CUDA_CHECK(cudaMemcpyAsync(d_residual.Data, d_input.Data, d_input.bytes(),
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    // Update residual_out gradient buffer to include norm contribution.
    if (d_residual_stream && d_residual_stream->Data && d_residual_stream->Data != d_input.Data) {
        CUDA_CHECK(cudaMemcpyAsync(d_residual_stream->Data, d_input.Data, d_input.bytes(),
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    // LN2 backward produces d_res_ffn (gradient for MLP down output). Mirror it into d_mlp_down
    // so downstream matmul backward sees a valid d_out.
    if (ln_layer_idx >= 0 && ln_field == "ln2_weight") {
        Tensor& d_residual = ensure_output_tensor(op.outputs[0]);
        Tensor& d_mlp_down = mRunState.simplified_grads(ln_layer_idx).d_mlp_down;

        if (d_mlp_down.Data && d_mlp_down.Data != d_residual.Data) {
            CUDA_CHECK(cudaMemcpyAsync(d_mlp_down.Data, d_residual.Data, d_residual.bytes(),
                                       cudaMemcpyDeviceToDevice, mRunState.MainStream));
        }
    }

    // LN1 backward writes grad for previous layer's residual stream into d_mlp_down;
    // mirror it into that layer's d_res_ffn so gradient propagation matches modular.
    if (op.outputs.size() > 1 && op.outputs[1].slot == TensorSlot::BlockDMLPDown) {
        const int prev_layer = op.outputs[1].layer_idx;
        if (prev_layer >= 0) {
            Tensor& d_res_ffn_prev = mRunState.simplified_grads(prev_layer).d_res_ffn;
            if (d_res_ffn_prev.Data && d_res_ffn_prev.Data != d_input.Data) {
                CUDA_CHECK(cudaMemcpyAsync(d_res_ffn_prev.Data, d_input.Data, d_input.bytes(),
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));
            }
        }
    }
}


}  // namespace dsl
