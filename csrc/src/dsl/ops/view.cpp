#include "dsl/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
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

    static int view_mlp_down_trace = 0;
    if (view_mlp_down_trace < 8 &&
        op.outputs[0].name.find("mlp_down") != std::string::npos) {
        int layer_idx = op.outputs[0].layer_idx;
        if (layer_idx >= 0 && layer_idx < 4) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            log_nan_sample("VIEW_MLP_DOWN", layer_idx, op.outputs[0].name, view, 3);
            view_mlp_down_trace++;
        }
    }
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

    // One-time NaN watchdog for MoE view gradients (mlp_down -> moe_out).
    static bool moe_view_nan_logged = false;
    if (!moe_view_nan_logged) {
        const bool nan_in = tensor_sample_has_nan_or_inf(d_out, 3);
        if (nan_in) {
            const std::string in_name = strip_ssa_suffix(op.inputs[0].name);
            const std::string out_name = strip_ssa_suffix(op.outputs[0].name);
            if (in_name.find("mlp_down") != std::string::npos ||
                out_name.find("moe_out") != std::string::npos) {
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
                        "[VIEW_BWD_MOE_NAN] op=%s in=%s out=%s\n",
                        op.op_id.c_str(),
                        op.inputs[0].name.c_str(),
                        op.outputs[0].name.c_str());
                dump_sample("VIEW_BWD_MOE_NAN_IN", d_out, op.inputs[0].name);
                dump_sample("VIEW_BWD_MOE_NAN_OUT", view, op.outputs[0].name);
                moe_view_nan_logged = true;
            }
        }
    }

    // DEBUG: Trace d_mlp_down -> d_moe_out mapping.
    static int moe_view_bwd_trace = 0;
    if (moe_view_bwd_trace < 12) {
        auto extract_field = [](std::string_view name, int& layer_idx) -> std::string {
            if (name.rfind("d_", 0) == 0) {
                name.remove_prefix(2);
            }
            std::string field;
            if (parse_block_param(name, layer_idx, field)) {
                return strip_ssa_suffix(field);
            }
            return "";
        };
        int out_layer = -1;
        int in_layer = -1;
        const std::string out_field = extract_field(op.outputs[0].name, out_layer);
        const std::string in_field = extract_field(op.inputs[0].name, in_layer);
        if ((out_field == "moe_out" || out_field == "moe_out_flat") &&
            (in_field == "mlp_down" || in_field == "mlp_down_flat")) {
            const int layer_idx = (out_layer >= 0) ? out_layer : in_layer;
            log_tensor_stats_ex("VIEW_BWD_MOE_DOUT", layer_idx, op.inputs[0].name, d_out, 4096, true);
            log_tensor_stats_ex("VIEW_BWD_MOE_DIN", layer_idx, op.outputs[0].name, view, 4096, true);
            moe_view_bwd_trace++;
        }
    }

    // DEBUG: Trace when d_blocks[0].ln2 is produced via view backward
    static int view_ln2_trace = 0;
    if (view_ln2_trace < 8 && strip_ssa_suffix(op.outputs[0].name) == "d_blocks[0].ln2") {
        int layer_any = -1;
        if (!op.outputs.empty() && op.outputs[0].layer_idx >= 0) {
            layer_any = op.outputs[0].layer_idx;
        } else if (!op.inputs.empty() && op.inputs[0].layer_idx >= 0) {
            layer_any = op.inputs[0].layer_idx;
        }
        fprintf(stderr,
                "[VIEW_BWD_LN2] id=%s in=%s out=%s\n",
                op.op_id.c_str(), op.inputs[0].name.c_str(), op.outputs[0].name.c_str());
        log_tensor_stats_ex("VIEW_BWD_LN2_IN", layer_any, op.inputs[0].name, d_out, 4096, true);
        log_tensor_stats_ex("VIEW_BWD_LN2_OUT", layer_any, op.outputs[0].name, view, 4096, true);
        view_ln2_trace++;
    }

    // DEBUG: Trace when top-layer d_ln1 is produced via view backward
    static int view_ln1_top_trace = 0;
    if (view_ln1_top_trace < 8) {
        const std::string out_base = strip_ssa_suffix(op.outputs[0].name);
        if (out_base.find(".ln1") != std::string::npos) {
            int layer_idx = -1;
            std::string field;
            parse_block_param(out_base, layer_idx, field);
            if (field == "ln1" && layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
                fprintf(stderr,
                        "[VIEW_BWD_LN1_TOP] id=%s in=%s out=%s ptr=%p\n",
                        op.op_id.c_str(),
                        op.inputs[0].name.c_str(),
                        op.outputs[0].name.c_str(),
                        view.Data);
                log_tensor_stats_ex("VIEW_BWD_LN1_TOP_IN", layer_idx, op.inputs[0].name, d_out, 4096, true);
                log_tensor_stats_ex("VIEW_BWD_LN1_TOP_OUT", layer_idx, op.outputs[0].name, view, 4096, true);
                view_ln1_top_trace++;
            }
        }
    }

    // DEBUG: Trace view backward outputs for qkv_rope gradients
    static int view_qkv_rope_trace = 0;
    if (view_qkv_rope_trace < 12 &&
        op.outputs[0].name.find("qkv_rope") != std::string::npos) {
        fprintf(stderr,
                "[VIEW_BWD_QKV_ROPE] id=%s in=%s out=%s in_shape=%s out_shape=%s out_ptr=%p\n",
                op.op_id.c_str(),
                op.inputs[0].name.c_str(),
                op.outputs[0].name.c_str(),
                tensor_shape_str(d_out).c_str(),
                tensor_shape_str(view).c_str(),
                view.Data);
        log_tensor_stats_ex("VIEW_BWD_QKV_ROPE_IN", -1, op.inputs[0].name, d_out, 4096, true);
        log_tensor_stats_ex("VIEW_BWD_QKV_ROPE_OUT", -1, op.outputs[0].name, view, 4096, true);
        view_qkv_rope_trace++;
    }

    // One-time NaN watchdog for qkv_flat view backward.
    static bool view_qkv_flat_nan_logged = false;
    if (!view_qkv_flat_nan_logged) {
        const std::string out_base = strip_ssa_suffix(op.outputs[0].name);
        if (out_base.find("qkv_flat") != std::string::npos) {
            float row_min = 0.0f;
            float row_max = 0.0f;
            const bool row_nan = tensor_row_has_nan_or_inf(d_out, 3, &row_min, &row_max);
            if (row_nan) {
                fprintf(stderr,
                        "[VIEW_BWD_QKV_FLAT_NAN] op=%s in=%s out=%s row_min=%.6f row_max=%.6f\n",
                        op.op_id.c_str(),
                        op.inputs[0].name.c_str(),
                        op.outputs[0].name.c_str(),
                        row_min,
                        row_max);
                view_qkv_flat_nan_logged = true;
            }
        }
    }

    // DEBUG: Trace view backward outputs for mlp_down flat (layer 24)
    static int view_mlp_down_trace = 0;
    if (view_mlp_down_trace < 20 &&
        op.outputs[0].name.find("d_blocks[24].mlp_down") != std::string::npos) {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> in_vals(4, 0.0f), out_vals(4, 0.0f);
        const bool ok_in = copy_tensor_sample_as_f32(d_out, in_vals.size(), in_vals);
        const bool ok_out = copy_tensor_sample_as_f32(view, out_vals.size(), out_vals);
        fprintf(stderr,
                "[VIEW_BWD_MLP_DOWN] out=%s ptr=%p in=%s slot=%d ptr=%p ok_in=%d ok_out=%d "
                "in_vals=%.6f,%.6f,%.6f,%.6f out_vals=%.6f,%.6f,%.6f,%.6f\n",
                op.outputs[0].name.c_str(),
                view.Data,
                op.inputs[0].name.c_str(),
                static_cast<int>(op.inputs[0].slot),
                d_out.Data,
                ok_in ? 1 : 0,
                ok_out ? 1 : 0,
                in_vals[0], in_vals[1], in_vals[2], in_vals[3],
                out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
        view_mlp_down_trace++;
    }

    // DEBUG: Trace view backward outputs for swiglu gradients (all layers)
    static int view_trace_count = 0;
    if (view_trace_count < 100 && op.outputs[0].name.find(".swiglu") != std::string::npos) {
        // Check the actual values in the buffer to diagnose gradient explosion
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> vals(4);
        if (view.Data) {
            cudaMemcpy(vals.data(), view.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        }
        fprintf(stderr, "[VIEW_BWD_SWIGLU] Stored %s in mTensorMap, ptr=%p, input=%s, vals=%.6f,%.6f,%.6f,%.6f\n",
                op.outputs[0].name.c_str(), view.Data, op.inputs[0].name.c_str(),
                vals[0], vals[1], vals[2], vals[3]);
        view_trace_count++;
    }
}


}  // namespace dsl
