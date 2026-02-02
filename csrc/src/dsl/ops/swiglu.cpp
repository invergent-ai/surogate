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

void CompiledExecutor::dispatch_swiglu(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);

    // Handle both 3D [B, T, 2*D] and 2D [N, 2*D] tensors (MoE produces 2D)
    if (inp.Rank == 2) {
        // 2D input: [N, 2*D] -> [N, D] (used by MoE path)
        const long N = inp.Sizes[0];
        const long D = inp.Sizes[1] / 2;

        // MoE output shape is dynamic, allocate with runtime shape
        std::vector<long> out_shape = {N, D};
        Tensor out = mRunState.temp_alloc(inp.DType, out_shape);
        mTemps.push_back(out);

        swiglu_forward(out, inp, nullptr, 1, static_cast<int>(N), static_cast<int>(D), mRunState.MainStream);

        // Store output in tensor map for subsequent ops
        mTensorMap[op.outputs[0].name] = out;
    } else {
        // 3D input: [B, T, 2*D] -> [B, T, D] (standard path)
        Tensor& out = ensure_output_tensor(op.outputs[0]);

        const long B = inp.Sizes[0];
        const long T = inp.Sizes[1];
        const long D = inp.Sizes[2] / 2;
        swiglu_forward(out, inp, nullptr, static_cast<int>(B),
                       static_cast<int>(T), static_cast<int>(D), mRunState.MainStream);
    }

    int layer_idx = -1;
    std::string field;
    if (parse_block_param(op.outputs[0].name, layer_idx, field)) {
        const long sample_token = 3;
        Tensor& out = resolve_tensor(op.outputs[0]);
        log_nan_sample("FWD_SWIGLU", layer_idx, op.outputs[0].name, out, sample_token);

        static int swiglu_name_trace = 0;
        if (swiglu_name_trace < 8 && layer_idx >= 0 && layer_idx < 4) {
            fprintf(stderr,
                    "[SWIGLU_NAME] layer=%d out=%s field=%s rank=%d shape=%s\n",
                    layer_idx,
                    op.outputs[0].name.c_str(),
                    field.c_str(),
                    out.Rank,
                    tensor_shape_str(out).c_str());
            swiglu_name_trace++;
        }

        // DEBUG: For MoE expert_act (2D), find max magnitude to trace explosions.
        if (field == "expert_act") {
            const bool moe_route_trace = (std::getenv("SUROGATE_MOE_ROUTE_TRACE") != nullptr);
            const bool moe_target_trace = (std::getenv("SUROGATE_MOE_DOT_TRACE") != nullptr);
            static int moe_target_swiglu_trace = 0;
            if (moe_target_trace && moe_target_swiglu_trace < 1) {
                const int target_layer = env_int("SUROGATE_MOE_DOT_LAYER", -1);
                const int target_pos = env_int("SUROGATE_MOE_DOT_POS", -1);
                if (target_pos >= 0 && (target_layer < 0 || layer_idx == target_layer)) {
                    CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                    log_tensor_row_stats("MOE_SWIGLU_IN_ROW", inp, target_pos);
                    log_tensor_row_stats("MOE_SWIGLU_OUT_ROW", out, target_pos);
                    if (inp.Rank == 2 && inp.Sizes[1] % 2 == 0) {
                        const std::size_t D = static_cast<std::size_t>(inp.Sizes[1] / 2);
                        std::vector<float> in_row;
                        std::vector<float> out_row;
                        const bool in_ok = copy_tensor_token_sample_as_f32(inp, target_pos, inp.Sizes[1], in_row);
                        const bool out_ok = copy_tensor_token_sample_as_f32(out, target_pos, D, out_row);
                        if (in_ok && out_ok && in_row.size() >= 2 * D && out_row.size() >= D) {
                            float up_min = 0.0f, up_max = 0.0f, gate_min = 0.0f, gate_max = 0.0f;
                            double up_sum = 0.0, gate_sum = 0.0;
                            for (std::size_t i = 0; i < D; ++i) {
                                const float u = in_row[i];
                                const float g = in_row[i + D];
                                if (i == 0) {
                                    up_min = up_max = u;
                                    gate_min = gate_max = g;
                                } else {
                                    up_min = std::min(up_min, u);
                                    up_max = std::max(up_max, u);
                                    gate_min = std::min(gate_min, g);
                                    gate_max = std::max(gate_max, g);
                                }
                                up_sum += std::fabs(static_cast<double>(u));
                                gate_sum += std::fabs(static_cast<double>(g));
                            }
                            const double up_mean = up_sum / static_cast<double>(D);
                            const double gate_mean = gate_sum / static_cast<double>(D);
                            std::size_t max_idx = 0;
                            float max_abs = 0.0f;
                            for (std::size_t i = 0; i < D; ++i) {
                                const float av = std::fabs(out_row[i]);
                                if (av > max_abs) {
                                    max_abs = av;
                                    max_idx = i;
                                }
                            }
                            const float up = in_row[max_idx];
                            const float gate = in_row[max_idx + D];
                            const float swish = gate / (1.0f + std::exp(-gate));
                            const float host_out = swish * up;
                            fprintf(stderr,
                                    "[MOE_SWIGLU_HALVES] row=%d up_min=%.6f up_max=%.6f up_mean_abs=%.6f "
                                    "gate_min=%.6f gate_max=%.6f gate_mean_abs=%.6f\n",
                                    target_pos,
                                    up_min, up_max, up_mean,
                                    gate_min, gate_max, gate_mean);
                            fprintf(stderr,
                                    "[MOE_SWIGLU_CHECK] row=%d idx=%zu gate=%.6f up=%.6f swish=%.6f host_out=%.6f gpu_out=%.6f\n",
                                    target_pos,
                                    max_idx,
                                    gate,
                                    up,
                                    swish,
                                    host_out,
                                    out_row[max_idx]);
                        } else {
                            fprintf(stderr,
                                    "[MOE_SWIGLU_CHECK] row=%d in_ok=%d out_ok=%d in_size=%zu out_size=%zu\n",
                                    target_pos,
                                    in_ok ? 1 : 0,
                                    out_ok ? 1 : 0,
                                    in_row.size(),
                                    out_row.size());
                        }
                    } else {
                        fprintf(stderr,
                                "[MOE_SWIGLU_CHECK] row=%d rank=%d size1=%ld (expected even)\n",
                                target_pos,
                                inp.Rank,
                                (inp.Rank > 1 ? inp.Sizes[1] : -1));
                    }
                    moe_target_swiglu_trace++;
                }
            }
            static int moe_route_swiglu_trace = 0;
            if (moe_route_trace && moe_route_swiglu_trace < 4) {
                const std::size_t total = static_cast<std::size_t>(out.nelem());
                const std::size_t sample = std::min<std::size_t>(4096, total);
                log_tensor_sample_stats("MOE_SWIGLU_OUT", out, 0, sample);
                moe_route_swiglu_trace++;
            }
            static int moe_swiglu_max_trace = 0;
            if (moe_swiglu_max_trace < 4 &&
                (layer_idx == 0 || layer_idx < 4 || layer_idx == static_cast<int>(mConfig.NumLayers) - 1)) {
                const std::size_t total = static_cast<std::size_t>(out.nelem());
                const std::size_t sample = std::min<std::size_t>(total, 1u << 20);
                std::vector<float> vals;
                if (copy_tensor_sample_as_f32(out, sample, vals)) {
                    float max_abs = 0.0f;
                    std::size_t max_idx = 0;
                    for (std::size_t i = 0; i < vals.size(); ++i) {
                        const float av = std::fabs(vals[i]);
                        if (av > max_abs) {
                            max_abs = av;
                            max_idx = i;
                        }
                    }
                    const long D = out.Rank > 1 ? out.Sizes[1] : 1;
                    const long row = D > 0 ? static_cast<long>(max_idx / static_cast<std::size_t>(D)) : 0;
                    fprintf(stderr,
                            "[MOE_FWD_SWIGLU_MAX] layer=%d sample_max=%.6f idx=%zu row=%ld sample=%zu total=%zu\n",
                            layer_idx, vals[max_idx], max_idx, row, sample, total);
                    if (row >= 0 && row < out.Sizes[0]) {
                        log_tensor_row_stats("MOE_SWIGLU_MAX_OUT_ROW", out, row);
                        log_tensor_row_stats("MOE_SWIGLU_MAX_IN_ROW", inp, row);
                    }
                }
                moe_swiglu_max_trace++;
            }

            // DEBUG: check if swiglu introduces NaNs for token 3 rows.
            static int moe_swiglu_row_trace = 0;
            if (moe_swiglu_row_trace < 8 && layer_idx >= 0 && layer_idx < 4) {
                const std::string prefix = "blocks[" + std::to_string(layer_idx) + "].";
                auto it = mTensorMap.find(prefix + "scatter_indices");
                if (it != mTensorMap.end()) {
                    const Tensor& scatter = it->second;
                    const int num_tokens = static_cast<int>(mB * mT);
                    const int top_k = (num_tokens > 0 && inp.Rank == 2)
                        ? static_cast<int>(inp.Sizes[0] / num_tokens)
                        : 1;
                    const int token_idx = 3;
                    std::vector<int> indices(top_k, -1);
                    if (scatter.Data && scatter.DType == ETensorDType::INT32) {
                        const std::size_t w_offset = static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(top_k);
                        CUDA_CHECK(cudaMemcpy(indices.data(),
                                              static_cast<const std::byte*>(scatter.Data) + w_offset * sizeof(int),
                                              top_k * sizeof(int),
                                              cudaMemcpyDeviceToHost));
                    }
                    for (int k = 0; k < top_k; ++k) {
                        const int idx = indices[k];
                        if (idx < 0) continue;
                        float in_min = 0.0f, in_max = 0.0f;
                        float out_min = 0.0f, out_max = 0.0f;
                        const bool in_nan = tensor_row_has_nan_or_inf(inp, idx, &in_min, &in_max);
                        const bool out_nan = tensor_row_has_nan_or_inf(out, idx, &out_min, &out_max);
                        if (in_nan || out_nan) {
                            fprintf(stderr,
                                    "[MOE_SWIGLU_ROW] layer=%d token=%d k=%d idx=%d in_nan=%d in_min=%.6f in_max=%.6f "
                                    "out_nan=%d out_min=%.6f out_max=%.6f\n",
                                    layer_idx, token_idx, k, idx,
                                    in_nan ? 1 : 0, in_min, in_max,
                                    out_nan ? 1 : 0, out_min, out_max);
                        }
                    }
                    moe_swiglu_row_trace++;
                } else {
                    fprintf(stderr,
                            "[MOE_SWIGLU_NO_SCATTER] layer=%d missing=%s\n",
                            layer_idx, (prefix + "scatter_indices").c_str());
                }
            }
        }
    }
}

void CompiledExecutor::dispatch_swiglu_backward(const CompiledOp& op) {
    // inputs: d_out, input (the mlp_up output before swiglu)
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& d_inp = ensure_output_tensor(op.outputs[0]);

    // DEBUG: Print inputs for layer 0
    int debug_layer_idx = -1;
    std::string field;
    if (!op.inputs[1].name.empty()) {
        parse_block_param(op.inputs[1].name, debug_layer_idx, field);
    }
    if (debug_layer_idx == 0) {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> d_out_vals(8), inp_vals(8);
        cudaMemcpy(d_out_vals.data(), d_out.Data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(inp_vals.data(), inp.Data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[SWIGLU_BWD] Layer 0: d_out_name='%s' d_out=%.6f,%.6f,%.6f (ptr=%p), inp=%.6f,%.6f,%.6f (ptr=%p)\n",
                op.inputs[0].name.c_str(),
                d_out_vals[0], d_out_vals[1], d_out_vals[2], d_out.Data,
                inp_vals[0], inp_vals[1], inp_vals[2], inp.Data);
    }

    // For FP8 hybrid backward, record abs_max of d_mlp_up for subsequent quantization
    float* abs_max_ptr = mRunState.has_fp8_hybrid_backward()
        ? mRunState.simplified_quant_grads().d_mlp_up.abs_max()
        : nullptr;

    // Handle both 3D [B, T, D] and 2D [N, D] tensors (MoE produces 2D)
    if (d_out.Rank == 2) {
        // 2D case for MoE: d_out is [N, D], inp is [N, 2*D]
        const long N = d_out.Sizes[0];
        const long D = d_out.Sizes[1];
        const long expected_inp = N * D * 2;
        const long inp_nelem = static_cast<long>(inp.nelem());
        const long d_inp_nelem = static_cast<long>(d_inp.nelem());
        if (inp_nelem != expected_inp || d_inp_nelem != expected_inp) {
            std::ostringstream oss;
            oss << "swiglu_backward: shape mismatch for 2D tensors: "
                << "d_out=[" << N << "," << D << "]"
                << " inp_nelem=" << inp_nelem
                << " d_inp_nelem=" << d_inp_nelem
                << " expected_inp_nelem=" << expected_inp
                << " inp_shape=" << tensor_shape_str(inp)
                << " d_inp_shape=" << tensor_shape_str(d_inp)
                << " d_out_name=" << op.inputs[0].name
                << " inp_name=" << op.inputs[1].name
                << " out_name=" << op.outputs[0].name;
            throw std::runtime_error(oss.str());
        }
        swiglu_backward(d_inp, d_out, inp, abs_max_ptr,
                        1, static_cast<int>(N), static_cast<int>(D), mRunState.MainStream);
    } else {
        // 3D case: d_out is [B, T, D]
        const long D = d_out.Sizes[2];
        swiglu_backward(d_inp, d_out, inp, abs_max_ptr,
                        static_cast<int>(d_out.Sizes[0]),
                        static_cast<int>(d_out.Sizes[1]),
                        static_cast<int>(D), mRunState.MainStream);
    }
}

}  // namespace dsl
