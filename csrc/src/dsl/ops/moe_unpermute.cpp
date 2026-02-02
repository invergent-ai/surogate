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
#include "modules/lora/lora_config.h"
#include "modules/lora/lora_weights_manager.h"

namespace dsl {

void CompiledExecutor::dispatch_moe_unpermute(const CompiledOp& op) {
    Tensor& expert_out = resolve_tensor(op.inputs[0]);
    Tensor& routing_weights = resolve_tensor(op.inputs[1]);
    Tensor& scatter_indices = resolve_tensor(op.inputs[2]);

    const int top_k = op.attrs.top_k;
    const int num_tokens = static_cast<int>(routing_weights.Sizes[0]);
    const int total_tokens = num_tokens * top_k;
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
    int layer_idx_any = op.attrs.layer_idx;
    std::string field_any;
    if (layer_idx_any < 0 && !op.outputs.empty()) {
        std::string_view name = op.outputs[0].name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        parse_block_param(name, layer_idx_any, field_any);
    }
    log_moe_gate_up_weight_sample("PRE_MOE_UNPERMUTE", layer_idx_any, mMicroStep, mWeights, mConfig);

    // MoE output shape is dynamic: [num_tokens, hidden_size]
    // Use the preallocated mlp_down buffer to avoid stack allocation issues.
    // The mlp_down buffer has shape (B, T, C) which equals [num_tokens, hidden_size]
    // when viewed as 2D. This buffer survives layer boundary cleanup.
    int layer_idx = mCurrentLayer >= 0 ? mCurrentLayer : 0;
    auto& acts = mRunState.simplified_acts(layer_idx);
    Tensor out = view_tensor(acts.mlp_down, {static_cast<long>(num_tokens), static_cast<long>(hidden_size)});
    const float moe_out_scale = env_float("SUROGATE_MOE_OUT_SCALE", 1.0f);

    // DEBUG: log unpermute input/output pointers for early layers.
    static int moe_unpermute_ptr_trace = 0;
    if (moe_unpermute_ptr_trace < 8) {
        int dbg_layer = op.attrs.layer_idx;
        std::string field;
        if (dbg_layer < 0 && !op.outputs.empty()) {
            std::string_view name = op.outputs[0].name;
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, dbg_layer, field);
        }
        if (dbg_layer >= 0 && dbg_layer < 4) {
            fprintf(stderr,
                    "[MOE_UNPERMUTE_IN] layer=%d expert_out=%s ptr=%p dtype=%s shape=%s routing=%s ptr=%p dtype=%s scatter=%s ptr=%p out=%s ptr=%p\n",
                    dbg_layer,
                    op.inputs[0].name.c_str(), static_cast<void*>(expert_out.Data), dtype_to_str(expert_out.DType), tensor_shape_str(expert_out).c_str(),
                    op.inputs[1].name.c_str(), static_cast<void*>(routing_weights.Data), dtype_to_str(routing_weights.DType),
                    op.inputs[2].name.c_str(), static_cast<void*>(scatter_indices.Data),
                    op.outputs[0].name.c_str(), static_cast<void*>(out.Data));
            moe_unpermute_ptr_trace++;
        }
    }

    // DEBUG: snapshot scatter indices for layer 2 token 3 to compare with gate_up.
    {
        static int moe_scatter_unpermute_trace = 0;
        int dbg_layer = op.attrs.layer_idx;
        std::string field;
        if (dbg_layer < 0 && !op.outputs.empty()) {
            std::string_view name = op.outputs[0].name;
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, dbg_layer, field);
        }
        if (dbg_layer == 2 && moe_scatter_unpermute_trace < 1 &&
            scatter_indices.Data && scatter_indices.DType == ETensorDType::INT32) {
            const int token_idx = 3;
            std::vector<int> idxs(top_k, -1);
            const std::size_t w_offset = static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(top_k);
            CUDA_CHECK(cudaMemcpy(idxs.data(),
                                  static_cast<const std::byte*>(scatter_indices.Data) + w_offset * sizeof(int),
                                  top_k * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            fprintf(stderr,
                    "[MOE_SCATTER_UNPERMUTE] layer=%d ptr=%p token=%d idxs=(",
                    dbg_layer, static_cast<void*>(scatter_indices.Data), token_idx);
            for (int k = 0; k < top_k; ++k) {
                fprintf(stderr, "%s%d", (k ? "," : ""), idxs[k]);
            }
            fprintf(stderr, ")\n");
            moe_scatter_unpermute_trace++;
        }
    }

    if (expert_out.DType == ETensorDType::BF16) {
        moe_unpermute_and_combine(out.get<nv_bfloat16>(),
                                  expert_out.get<nv_bfloat16>(),
                                  routing_weights.get<nv_bfloat16>(),
                                  scatter_indices.get<int>(),
                                  num_tokens, total_tokens, hidden_size, top_k,
                                  mRunState.MainStream);
    } else {
        moe_unpermute_and_combine(out.get<float>(),
                                  expert_out.get<float>(),
                                  routing_weights.get<float>(),
                                  scatter_indices.get<int>(),
                                  num_tokens, total_tokens, hidden_size, top_k,
                                  mRunState.MainStream);
    }

    const bool moe_target_trace = (std::getenv("SUROGATE_MOE_DOT_TRACE") != nullptr);
    static int moe_target_unpermute_trace = 0;
    if (moe_target_trace && moe_target_unpermute_trace < 1) {
        const int target_layer = env_int("SUROGATE_MOE_DOT_LAYER", -1);
        const int target_pos = env_int("SUROGATE_MOE_DOT_POS", -1);
        if (target_pos >= 0 && (target_layer < 0 || layer_idx_any == target_layer) &&
            scatter_indices.Data && scatter_indices.DType == ETensorDType::INT32) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            std::vector<int> scatter;
            scatter.resize(static_cast<std::size_t>(num_tokens) * static_cast<std::size_t>(top_k), -1);
            CUDA_CHECK(cudaMemcpy(scatter.data(),
                                  scatter_indices.Data,
                                  scatter.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            int found_token = -1;
            int found_k = -1;
            for (int tok = 0; tok < num_tokens && found_token < 0; ++tok) {
                const std::size_t base = static_cast<std::size_t>(tok) * static_cast<std::size_t>(top_k);
                for (int k = 0; k < top_k; ++k) {
                    if (scatter[base + static_cast<std::size_t>(k)] == target_pos) {
                        found_token = tok;
                        found_k = k;
                        break;
                    }
                }
            }
            if (found_token >= 0) {
                log_tensor_row_stats("MOE_UNPERMUTE_IN_ROW_STATS", expert_out, target_pos);
                log_tensor_row_stats("MOE_UNPERMUTE_OUT_ROW_STATS", out, found_token);
                std::vector<float> w;
                const std::size_t w_offset = static_cast<std::size_t>(found_token) * static_cast<std::size_t>(top_k) +
                                             static_cast<std::size_t>(found_k);
                if (copy_tensor_sample_offset_as_f32(routing_weights, w_offset, 1, w)) {
                    fprintf(stderr,
                            "[MOE_UNPERMUTE_WEIGHT] layer=%d token=%d k=%d weight=%.6f\n",
                            layer_idx_any, found_token, found_k,
                            w.empty() ? 0.0f : w[0]);
                }
            } else {
                fprintf(stderr,
                        "[MOE_UNPERMUTE_TARGET_MISS] layer=%d target_pos=%d num_tokens=%d top_k=%d\n",
                        layer_idx_any, target_pos, num_tokens, top_k);
            }
            moe_target_unpermute_trace++;
        }
    }

    const bool moe_route_trace = (std::getenv("SUROGATE_MOE_ROUTE_TRACE") != nullptr);
    static int moe_route_unpermute_trace = 0;
    if (moe_route_trace && moe_route_unpermute_trace < 4) {
        const std::size_t total = static_cast<std::size_t>(out.nelem());
        const std::size_t sample = std::min<std::size_t>(4096, total);
        const std::size_t expert_total = static_cast<std::size_t>(expert_out.nelem());
        const std::size_t expert_sample = std::min<std::size_t>(4096, expert_total);
        const std::size_t weights_total = static_cast<std::size_t>(routing_weights.nelem());
        const std::size_t weights_sample = std::min<std::size_t>(4096, weights_total);
        const std::size_t out_bytes = total * get_dtype_size(out.DType);
        const std::size_t expert_bytes = expert_total * get_dtype_size(expert_out.DType);
        const std::uintptr_t out_ptr = reinterpret_cast<std::uintptr_t>(out.Data);
        const std::uintptr_t expert_ptr = reinterpret_cast<std::uintptr_t>(expert_out.Data);
        const bool overlap = (expert_ptr < out_ptr + out_bytes) &&
                             (out_ptr < expert_ptr + expert_bytes);

        fprintf(stderr,
                "[MOE_UNPERMUTE_PTR] layer=%d expert_out=%p bytes=%zu out=%p bytes=%zu overlap=%d routing=%p\n",
                layer_idx_any,
                static_cast<void*>(expert_out.Data),
                expert_bytes,
                static_cast<void*>(out.Data),
                out_bytes,
                overlap ? 1 : 0,
                static_cast<void*>(routing_weights.Data));

        log_tensor_sample_stats("MOE_UNPERMUTE_EXPERT_OUT", expert_out, 0, sample);
        if (expert_total > expert_sample) {
            log_tensor_sample_stats("MOE_UNPERMUTE_EXPERT_OUT_MID", expert_out, expert_total / 2, expert_sample);
            if (expert_total > expert_sample) {
                log_tensor_sample_stats("MOE_UNPERMUTE_EXPERT_OUT_END",
                                        expert_out,
                                        expert_total - expert_sample,
                                        expert_sample);
            }
        }
        log_tensor_sample_stats("MOE_UNPERMUTE_WEIGHTS", routing_weights, 0, weights_sample);
        log_tensor_sample_stats("MOE_UNPERMUTE_OUT", out, 0, sample);

        if (scatter_indices.Data && scatter_indices.DType == ETensorDType::INT32 && top_k > 0) {
            const int token_idx = 0;
            std::vector<int> idxs(static_cast<std::size_t>(top_k), -1);
            const std::size_t idx_offset = static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(top_k);
            CUDA_CHECK(cudaMemcpy(idxs.data(),
                                  static_cast<const std::byte*>(scatter_indices.Data) + idx_offset * sizeof(int),
                                  static_cast<std::size_t>(top_k) * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            fprintf(stderr,
                    "[MOE_UNPERMUTE_IDX] layer=%d token=%d idxs=(",
                    layer_idx_any, token_idx);
            for (int k = 0; k < top_k; ++k) {
                fprintf(stderr, "%s%d", (k ? "," : ""), idxs[k]);
            }
            fprintf(stderr, ")\n");

            const int hidden = hidden_size;
            const std::size_t row_sample = std::min<std::size_t>(2048, static_cast<std::size_t>(hidden));
            for (int k = 0; k < top_k; ++k) {
                const int expert_pos = idxs[k];
                if (expert_pos < 0 || expert_pos >= total_tokens) {
                    continue;
                }
                const std::size_t row_offset = static_cast<std::size_t>(expert_pos) *
                                               static_cast<std::size_t>(hidden);
                std::vector<float> row_vals;
                if (copy_tensor_sample_offset_as_f32(expert_out, row_offset, row_sample, row_vals)) {
                    float max_abs = 0.0f;
                    std::size_t max_idx = 0;
                    float max_val = 0.0f;
                    for (std::size_t i = 0; i < row_vals.size(); ++i) {
                        const float v = row_vals[i];
                        const float av = std::fabs(v);
                        if (av > max_abs) {
                            max_abs = av;
                            max_idx = i;
                            max_val = v;
                        }
                    }
                    fprintf(stderr,
                            "[MOE_UNPERMUTE_EXPERT_ROW] layer=%d token=%d k=%d pos=%d max_abs=%.6f max_idx=%zu max_val=%.6f vals=",
                            layer_idx_any, token_idx, k, expert_pos, max_abs, max_idx, max_val);
                    const std::size_t print_n = std::min<std::size_t>(8, row_vals.size());
                    for (std::size_t i = 0; i < print_n; ++i) {
                        fprintf(stderr, "%s%.6f", (i ? "," : ""), row_vals[i]);
                    }
                    fprintf(stderr, "\n");
                }
            }
        }
        moe_route_unpermute_trace++;
    }

    // Optional debug scaling for MoE output to test normalization issues.
    if (moe_out_scale != 1.0f) {
        const float add_scale = 0.5f * moe_out_scale;
        vector_add_sr(out, out, out, add_scale, static_cast<long>(out.nelem()), 0, mRunState.MainStream);
    }

    // DEBUG: Trace forward MoE output magnitude for layer 0/top.
    static int moe_fwd_out_trace = 0;
    if (moe_fwd_out_trace < 12) {
        int layer_idx = op.attrs.layer_idx;
        std::string field;
        if (layer_idx < 0 && !op.inputs.empty()) {
            std::string_view name = op.inputs[0].name;
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx, field);
        }
        if (layer_idx < 0 || layer_idx < 4 || layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
            if (moe_fwd_out_trace < 4) {
                CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                log_nan_sample("MOE_FWD_OUT", layer_idx, op.outputs[0].name, out, 3);
                log_nan_sample("MOE_FWD_ROUTING_W", layer_idx, op.inputs[1].name, routing_weights, 3);
            }
            log_tensor_stats_ex("MOE_FWD_OUT", layer_idx, op.outputs[0].name, out, 4096, true);
            moe_fwd_out_trace++;
        }
    }

    // DEBUG: If MoE output spikes, dump routing weights + expert_out samples.
    {
        static int moe_out_spike_trace = 0;
        if (moe_out_spike_trace < 4) {
            int dbg_layer = op.attrs.layer_idx;
            std::string field;
            if (dbg_layer < 0 && !op.outputs.empty()) {
                std::string_view name = op.outputs[0].name;
                if (name.rfind("saved.", 0) == 0) {
                    name.remove_prefix(6);
                }
                parse_block_param(name, dbg_layer, field);
            }
            if (dbg_layer >= 0 && dbg_layer < 4) {
                std::vector<float> out_vals;
                const std::size_t total = static_cast<std::size_t>(out.nelem());
                const std::size_t n = std::min<std::size_t>(4096, total);
                if (n > 0 && copy_tensor_sample_as_f32(out, n, out_vals)) {
                    float max_abs = 0.0f;
                    float min_v = 0.0f;
                    float max_v = 0.0f;
                    std::size_t max_idx = 0;
                    bool has = false;
                    for (std::size_t i = 0; i < out_vals.size(); ++i) {
                        const float v = out_vals[i];
                        if (std::isnan(v) || std::isinf(v)) {
                            continue;
                        }
                        if (!has) {
                            min_v = v;
                            max_v = v;
                            has = true;
                        } else {
                            if (v < min_v) min_v = v;
                            if (v > max_v) max_v = v;
                        }
                        const float av = std::fabs(v);
                        if (av > max_abs) {
                            max_abs = av;
                            max_idx = i;
                        }
                    }
                    if (has && max_abs > 100.0f) {
                        const std::string prefix = "blocks[" + std::to_string(dbg_layer) + "].";
                        auto find_tensor = [&](const std::string& name) -> Tensor* {
                            auto it = mTensorMap.find(name);
                            if (it != mTensorMap.end()) {
                                return &it->second;
                            }
                            if (mWeights.has(name)) {
                                return &mWeights.get(name);
                            }
                            return nullptr;
                        };
                        Tensor* expert_act_t = find_tensor(prefix + "expert_act");
                        Tensor* expert_gate_up_t = find_tensor(prefix + "expert_gate_up");
                        Tensor* permuted_in_t = find_tensor(prefix + "permuted_input");
                        Tensor* ln2_rstd_t = find_tensor(prefix + "ln2_rstd");
                        Tensor* res_att_t = find_tensor(prefix + "res_att");
                        Tensor* res_ffn_t = find_tensor(prefix + "res_ffn");
                        Tensor* att_out_t = find_tensor(prefix + "att_out");

                        const int max_token = static_cast<int>(max_idx / static_cast<std::size_t>(hidden_size));
                        const int max_dim = static_cast<int>(max_idx % static_cast<std::size_t>(hidden_size));
                        if (max_token >= 0 && max_token < num_tokens) {
                            fprintf(stderr,
                                    "[MOE_OUT_SPIKE_MAX] layer=%d token=%d dim=%d val=%.6f\n",
                                    dbg_layer, max_token, max_dim, out_vals[max_idx]);
                        }
                        const int tokens_to_log[] = {max_token, 3};
                        for (int token_idx : tokens_to_log) {
                            if (token_idx < 0 || token_idx >= num_tokens) {
                                continue;
                            }
                            std::vector<float> weights(top_k, 0.0f);
                            const std::size_t w_offset =
                                static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(top_k);
                            (void)copy_tensor_sample_offset_as_f32(routing_weights, w_offset, weights.size(), weights);
                            float w_sum = 0.0f;
                            float w_min = 0.0f;
                            float w_max = 0.0f;
                            bool w_has = false;
                            for (float w : weights) {
                                w_sum += w;
                                if (!w_has) {
                                    w_min = w;
                                    w_max = w;
                                    w_has = true;
                                } else {
                                    if (w < w_min) w_min = w;
                                    if (w > w_max) w_max = w;
                                }
                            }

                            std::vector<int> indices(top_k, -1);
                            int oob = 0;
                            if (scatter_indices.Data && scatter_indices.DType == ETensorDType::INT32) {
                                CUDA_CHECK(cudaMemcpy(indices.data(),
                                                      static_cast<const std::byte*>(scatter_indices.Data) + w_offset * sizeof(int),
                                                      top_k * sizeof(int),
                                                      cudaMemcpyDeviceToHost));
                                for (int k = 0; k < top_k; ++k) {
                                    if (indices[k] < 0 || indices[k] >= total_tokens) {
                                        oob++;
                                    }
                                }
                            }

                            fprintf(stderr,
                                    "[MOE_OUT_SPIKE] layer=%d token=%d out_max_abs=%.6f out_min=%.6f out_max=%.6f "
                                    "w_sum=%.6f w_min=%.6f w_max=%.6f oob=%d weights=(",
                                    dbg_layer, token_idx, max_abs, min_v, max_v, w_sum, w_min, w_max, oob);
                            for (int k = 0; k < top_k; ++k) {
                                fprintf(stderr, "%s%.6f", (k ? "," : ""), weights[k]);
                            }
                            fprintf(stderr, ") indices=(");
                            for (int k = 0; k < top_k; ++k) {
                                fprintf(stderr, "%s%d", (k ? "," : ""), indices[k]);
                            }
                            fprintf(stderr, ")\n");

                            int k_max = 0;
                            for (int k = 1; k < top_k; ++k) {
                                if (weights[k] > weights[k_max]) {
                                    k_max = k;
                                }
                            }
                            const int idx_max = indices[k_max];
                            int expert_id_max = -1;
                            int expert_row = -1;
                            Tensor* expert_offsets_t = find_tensor("moe_expert_offsets");
                            if (expert_offsets_t && expert_offsets_t->Data &&
                                expert_offsets_t->DType == ETensorDType::INT32 &&
                                idx_max >= 0) {
                                const int num_experts = static_cast<int>(mConfig.NumExperts);
                                std::vector<int> h_offsets(num_experts + 1, 0);
                                CUDA_CHECK(cudaMemcpy(h_offsets.data(),
                                                      expert_offsets_t->Data,
                                                      static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                                                      cudaMemcpyDeviceToHost));
                                for (int e = 0; e < num_experts; ++e) {
                                    if (idx_max >= h_offsets[e] && idx_max < h_offsets[e + 1]) {
                                        expert_id_max = e;
                                        expert_row = idx_max - h_offsets[e];
                                        break;
                                    }
                                }
                                fprintf(stderr,
                                        "[MOE_OUT_SPIKE_EXPERT_ID] layer=%d token=%d idx=%d expert=%d row=%d\n",
                                        dbg_layer, token_idx, idx_max, expert_id_max, expert_row);
                            }
                            Tensor* gate_w_t = find_tensor(prefix + "experts_gate_up");
                            Tensor* down_w_t = find_tensor(prefix + "experts_down");
                            if (expert_id_max >= 0 && gate_w_t && gate_w_t->Data) {
                                const int inter_size = (mConfig.MoeIntermediateSize > 0)
                                    ? static_cast<int>(mConfig.MoeIntermediateSize)
                                    : static_cast<int>(mConfig.IntermediateSize);
                                const int mup_size = inter_size * 2;
                                const std::size_t stride =
                                    static_cast<std::size_t>(mup_size) * static_cast<std::size_t>(hidden_size);
                                const std::size_t offset = stride * static_cast<std::size_t>(expert_id_max);
                                const std::size_t sample = std::min<std::size_t>(1024, stride);
                                std::vector<float> wvals;
                                if (copy_tensor_sample_offset_as_f32(*gate_w_t, offset, sample, wvals)) {
                                    float w_max_abs = 0.0f;
                                    for (float v : wvals) {
                                        w_max_abs = std::max(w_max_abs, std::fabs(v));
                                    }
                                    fprintf(stderr,
                                            "[MOE_OUT_SPIKE_W_GATE_UP] layer=%d expert=%d max_abs=%.6f dtype=%s\n",
                                            dbg_layer, expert_id_max, w_max_abs, dtype_to_str(gate_w_t->DType));
                                }
                            }
                            if (expert_id_max >= 0 && down_w_t && down_w_t->Data) {
                                const int inter_size = (mConfig.MoeIntermediateSize > 0)
                                    ? static_cast<int>(mConfig.MoeIntermediateSize)
                                    : static_cast<int>(mConfig.IntermediateSize);
                                const std::size_t stride =
                                    static_cast<std::size_t>(hidden_size) * static_cast<std::size_t>(inter_size);
                                const std::size_t offset = stride * static_cast<std::size_t>(expert_id_max);
                                const std::size_t sample = std::min<std::size_t>(1024, stride);
                                std::vector<float> wvals;
                                if (copy_tensor_sample_offset_as_f32(*down_w_t, offset, sample, wvals)) {
                                    float w_max_abs = 0.0f;
                                    for (float v : wvals) {
                                        w_max_abs = std::max(w_max_abs, std::fabs(v));
                                    }
                                    fprintf(stderr,
                                            "[MOE_OUT_SPIKE_W_DOWN] layer=%d expert=%d max_abs=%.6f dtype=%s\n",
                                            dbg_layer, expert_id_max, w_max_abs, dtype_to_str(down_w_t->DType));
                                }
                                if (max_dim >= 0 && max_dim < hidden_size) {
                                    const std::size_t row_offset =
                                        offset + static_cast<std::size_t>(max_dim) * static_cast<std::size_t>(inter_size);
                                    std::vector<float> row_vals;
                                    if (copy_tensor_sample_offset_as_f32(*down_w_t, row_offset,
                                                                         static_cast<std::size_t>(inter_size), row_vals)) {
                                        float row_max_abs = 0.0f;
                                        for (float v : row_vals) {
                                            row_max_abs = std::max(row_max_abs, std::fabs(v));
                                        }
                                        fprintf(stderr,
                                                "[MOE_OUT_SPIKE_W_DOWN_ROW] layer=%d expert=%d dim=%d max_abs=%.6f "
                                                "vals=%.6f,%.6f,%.6f,%.6f\n",
                                                dbg_layer, expert_id_max, max_dim, row_max_abs,
                                                row_vals[0], row_vals[1], row_vals[2], row_vals[3]);
                                    }
                                }
                            }
                            if (expert_act_t && idx_max >= 0) {
                                const int inter_size = (mConfig.MoeIntermediateSize > 0)
                                    ? static_cast<int>(mConfig.MoeIntermediateSize)
                                    : static_cast<int>(mConfig.IntermediateSize);
                                const std::size_t act_offset =
                                    static_cast<std::size_t>(idx_max) * static_cast<std::size_t>(inter_size);
                                const std::size_t act_sample = static_cast<std::size_t>(inter_size);
                                std::vector<float> act_vals;
                                if (copy_tensor_sample_offset_as_f32(*expert_act_t, act_offset, act_sample, act_vals)) {
                                    float act_max_abs = 0.0f;
                                    for (float v : act_vals) {
                                        act_max_abs = std::max(act_max_abs, std::fabs(v));
                                    }
                                    fprintf(stderr,
                                            "[MOE_OUT_SPIKE_ACT] layer=%d token=%d idx=%d k=%d act_max_abs=%.6f "
                                            "act_vals=%.6f,%.6f,%.6f,%.6f\n",
                                            dbg_layer, token_idx, idx_max, k_max, act_max_abs,
                                            act_vals[0], act_vals[1], act_vals[2], act_vals[3]);
                                }
                            }
                            if (expert_gate_up_t && idx_max >= 0) {
                                const int inter_size = (mConfig.MoeIntermediateSize > 0)
                                    ? static_cast<int>(mConfig.MoeIntermediateSize)
                                    : static_cast<int>(mConfig.IntermediateSize);
                                const int mup_size = inter_size * 2;
                                const std::size_t gate_offset =
                                    static_cast<std::size_t>(idx_max) * static_cast<std::size_t>(mup_size);
                                const std::size_t gate_sample = static_cast<std::size_t>(mup_size);
                                std::vector<float> gate_vals;
                                if (copy_tensor_sample_offset_as_f32(*expert_gate_up_t, gate_offset, gate_sample, gate_vals)) {
                                    float gate_max_abs = 0.0f;
                                    for (float v : gate_vals) {
                                        gate_max_abs = std::max(gate_max_abs, std::fabs(v));
                                    }
                                    fprintf(stderr,
                                            "[MOE_OUT_SPIKE_GATE_UP] layer=%d token=%d idx=%d k=%d gate_max_abs=%.6f "
                                            "gate_vals=%.6f,%.6f,%.6f,%.6f\n",
                                            dbg_layer, token_idx, idx_max, k_max, gate_max_abs,
                                            gate_vals[0], gate_vals[1], gate_vals[2], gate_vals[3]);
                                }
                            }
                            if (permuted_in_t && idx_max >= 0) {
                                const std::size_t in_offset =
                                    static_cast<std::size_t>(idx_max) * static_cast<std::size_t>(hidden_size);
                                const std::size_t in_sample = static_cast<std::size_t>(hidden_size);
                                std::vector<float> in_vals;
                                if (copy_tensor_sample_offset_as_f32(*permuted_in_t, in_offset, in_sample, in_vals)) {
                                    float in_max_abs = 0.0f;
                                    for (float v : in_vals) {
                                        in_max_abs = std::max(in_max_abs, std::fabs(v));
                                    }
                                    fprintf(stderr,
                                            "[MOE_OUT_SPIKE_IN] layer=%d token=%d idx=%d k=%d in_max_abs=%.6f "
                                            "in_vals=%.6f,%.6f,%.6f,%.6f\n",
                                            dbg_layer, token_idx, idx_max, k_max, in_max_abs,
                                            in_vals[0], in_vals[1], in_vals[2], in_vals[3]);
                                }
                            }
                            if (ln2_rstd_t && max_token == token_idx) {
                                if (ln2_rstd_t->Data && ln2_rstd_t->DType == ETensorDType::FP32) {
                                    float rstd_val = 0.0f;
                                    CUDA_CHECK(cudaMemcpy(&rstd_val,
                                                          static_cast<const std::byte*>(ln2_rstd_t->Data) +
                                                              static_cast<std::size_t>(token_idx) * sizeof(float),
                                                          sizeof(float),
                                                          cudaMemcpyDeviceToHost));
                                    fprintf(stderr,
                                            "[MOE_OUT_SPIKE_RSTD] layer=%d token=%d rstd=%.6f\n",
                                            dbg_layer, token_idx, rstd_val);
                                }
                            }
                            if (res_att_t && max_token == token_idx) {
                                const std::size_t res_offset =
                                    static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(hidden_size);
                                const std::size_t res_sample = static_cast<std::size_t>(hidden_size);
                                std::vector<float> res_vals;
                                if (copy_tensor_sample_offset_as_f32(*res_att_t, res_offset, res_sample, res_vals)) {
                                    float res_max_abs = 0.0f;
                                    for (float v : res_vals) {
                                        res_max_abs = std::max(res_max_abs, std::fabs(v));
                                    }
                                    fprintf(stderr,
                                            "[MOE_OUT_SPIKE_RES] layer=%d token=%d res_max_abs=%.6f "
                                            "res_vals=%.6f,%.6f,%.6f,%.6f\n",
                                            dbg_layer, token_idx, res_max_abs,
                                            res_vals[0], res_vals[1], res_vals[2], res_vals[3]);
                                }
                            }
                            if (res_ffn_t && max_token == token_idx) {
                                const std::size_t res_offset =
                                    static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(hidden_size);
                                const std::size_t res_sample = static_cast<std::size_t>(hidden_size);
                                std::vector<float> res_vals;
                                if (copy_tensor_sample_offset_as_f32(*res_ffn_t, res_offset, res_sample, res_vals)) {
                                    float res_max_abs = 0.0f;
                                    for (float v : res_vals) {
                                        res_max_abs = std::max(res_max_abs, std::fabs(v));
                                    }
                                    fprintf(stderr,
                                            "[MOE_OUT_SPIKE_RES_FFN] layer=%d token=%d res_ffn_max_abs=%.6f "
                                            "res_ffn_vals=%.6f,%.6f,%.6f,%.6f\n",
                                            dbg_layer, token_idx, res_max_abs,
                                            res_vals[0], res_vals[1], res_vals[2], res_vals[3]);
                                }
                            }
                            if (att_out_t && max_token == token_idx) {
                                const std::size_t att_offset =
                                    static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(hidden_size);
                                const std::size_t att_sample = static_cast<std::size_t>(hidden_size);
                                std::vector<float> att_vals;
                                if (copy_tensor_sample_offset_as_f32(*att_out_t, att_offset, att_sample, att_vals)) {
                                    float att_max_abs = 0.0f;
                                    for (float v : att_vals) {
                                        att_max_abs = std::max(att_max_abs, std::fabs(v));
                                    }
                                    fprintf(stderr,
                                            "[MOE_OUT_SPIKE_ATT_OUT] layer=%d token=%d att_max_abs=%.6f "
                                            "att_vals=%.6f,%.6f,%.6f,%.6f\n",
                                            dbg_layer, token_idx, att_max_abs,
                                            att_vals[0], att_vals[1], att_vals[2], att_vals[3]);
                                }
                            }

                            const int sample_k = std::min(top_k, 2);
                            for (int k = 0; k < sample_k; ++k) {
                                if (indices[k] < 0 || indices[k] >= total_tokens) {
                                    continue;
                                }
                                std::vector<float> ex_vals(4, 0.0f);
                                (void)copy_tensor_token_sample_as_f32(expert_out, indices[k], ex_vals.size(), ex_vals);
                                fprintf(stderr,
                                        "[MOE_OUT_SPIKE_EXPERT] layer=%d token=%d k=%d idx=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                                        dbg_layer, token_idx, k, indices[k],
                                        ex_vals[0], ex_vals[1], ex_vals[2], ex_vals[3]);
                            }
                            if (max_token == token_idx && max_dim >= 0 && max_dim < hidden_size) {
                                for (int k = 0; k < top_k; ++k) {
                                    if (indices[k] < 0 || indices[k] >= total_tokens) {
                                        continue;
                                    }
                                    const std::size_t elem_offset =
                                        static_cast<std::size_t>(indices[k]) * static_cast<std::size_t>(hidden_size) +
                                        static_cast<std::size_t>(max_dim);
                                    std::vector<float> ex_val(1, 0.0f);
                                    (void)copy_tensor_sample_offset_as_f32(expert_out, elem_offset, ex_val.size(), ex_val);
                                    fprintf(stderr,
                                            "[MOE_OUT_SPIKE_DIM] layer=%d token=%d dim=%d k=%d idx=%d "
                                            "w=%.6f ex=%.6f contrib=%.6f\n",
                                            dbg_layer, token_idx, max_dim, k, indices[k],
                                            weights[k], ex_val[0], weights[k] * ex_val[0]);
                                }
                            }
                        }
                        moe_out_spike_trace++;
                    }
                }
            }
        }
    }

    // DEBUG: If MoE output goes NaN, dump key inputs (no rate limit on NaN detection).
    {
        int layer_idx = op.attrs.layer_idx;
        std::string field;
        if (layer_idx < 0 && !op.outputs.empty()) {
            std::string_view name = op.outputs[0].name;
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx, field);
        }
        if (layer_idx >= 0 && layer_idx < 4) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            if (tensor_sample_has_nan_or_inf(out, 3)) {
                const std::string prefix = "blocks[" + std::to_string(layer_idx) + "].";
                auto find_tensor = [&](const std::string& name) -> Tensor* {
                    auto it = mTensorMap.find(name);
                    if (it != mTensorMap.end()) {
                        return &it->second;
                    }
                    if (mWeights.has(name)) {
                        return &mWeights.get(name);
                    }
                    return nullptr;
                };
                Tensor* permuted_t = find_tensor(prefix + "permuted_input");
                Tensor* gate_up_t = find_tensor(prefix + "expert_gate_up");
                Tensor* act_t = find_tensor(prefix + "expert_act");
                Tensor* down_t = find_tensor(prefix + "expert_down");
                Tensor* ln2_t = find_tensor(prefix + "ln2");
                Tensor* gate_up_w_t = find_tensor(prefix + "experts_gate_up");
                Tensor* gather_t = find_tensor("moe_gather_indices");

                // Map permuted row index -> expert id using stored offsets.
                Tensor* expert_offsets_t = find_tensor("moe_expert_offsets");
                std::vector<int> h_offsets;
                if (expert_offsets_t && expert_offsets_t->Data && expert_offsets_t->DType == ETensorDType::INT32) {
                    const int n = static_cast<int>(mConfig.NumExperts) + 1;
                    h_offsets.resize(n, 0);
                    CUDA_CHECK(cudaMemcpy(h_offsets.data(),
                                          expert_offsets_t->Data,
                                          static_cast<std::size_t>(n) * sizeof(int),
                                          cudaMemcpyDeviceToHost));
                }
                auto find_expert = [&](int idx) {
                    if (h_offsets.empty()) {
                        return -1;
                    }
                    for (int e = 0; e + 1 < static_cast<int>(h_offsets.size()); ++e) {
                        if (idx >= h_offsets[e] && idx < h_offsets[e + 1]) {
                            return e;
                        }
                    }
                    return -1;
                };

                fprintf(stderr,
                        "[MOE_FWD_NAN_PTRS] layer=%d permuted=%p gate_up=%p act=%p down=%p\n",
                        layer_idx,
                        permuted_t ? permuted_t->Data : nullptr,
                        gate_up_t ? gate_up_t->Data : nullptr,
                        act_t ? act_t->Data : nullptr,
                        down_t ? down_t->Data : nullptr);
                log_nan_sample("MOE_FWD_OUT_NAN", layer_idx, op.outputs[0].name, out, 3);
                log_nan_sample("MOE_FWD_ROUTING_W_NAN", layer_idx, op.inputs[1].name, routing_weights, 3);
                log_nan_sample("MOE_FWD_EXPERT_OUT_NAN", layer_idx, op.inputs[0].name, expert_out, 3);
                log_tensor_stats_ex("MOE_FWD_OUT_NAN", layer_idx, op.outputs[0].name, out, 4096, true);

                const int token_idx = 3;
                std::vector<float> weights(top_k, 0.0f);
                const std::size_t w_offset = static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(top_k);
                (void)copy_tensor_sample_offset_as_f32(routing_weights, w_offset, weights.size(), weights);

                std::vector<int> indices(top_k, -1);
                int oob = 0;
                if (scatter_indices.Data && scatter_indices.DType == ETensorDType::INT32) {
                    CUDA_CHECK(cudaMemcpy(indices.data(),
                                          static_cast<const std::byte*>(scatter_indices.Data) + w_offset * sizeof(int),
                                          top_k * sizeof(int),
                                          cudaMemcpyDeviceToHost));
                    for (int k = 0; k < top_k; ++k) {
                        if (indices[k] < 0 || indices[k] >= total_tokens) {
                            oob++;
                        }
                    }
                }
                fprintf(stderr,
                        "[MOE_FWD_NAN_TOKEN] layer=%d token=%d total_tokens=%d oob=%d weights=(",
                        layer_idx, token_idx, total_tokens, oob);
                for (int k = 0; k < top_k; ++k) {
                    fprintf(stderr, "%s%.6f", (k ? "," : ""), weights[k]);
                }
                fprintf(stderr, ") indices=(");
                for (int k = 0; k < top_k; ++k) {
                    fprintf(stderr, "%s%d", (k ? "," : ""), indices[k]);
                }
                fprintf(stderr, ")\n");

                if (oob == 0) {
                    for (int k = 0; k < top_k; ++k) {
                        std::vector<float> ex_vals(4, 0.0f);
                        (void)copy_tensor_token_sample_as_f32(expert_out, indices[k], ex_vals.size(), ex_vals);
                        const int expert_id = find_expert(indices[k]);
                        int expert_count = -1;
                        int expert_start = -1;
                        int expert_end = -1;
                        if (expert_id >= 0 && !h_offsets.empty()) {
                            expert_start = h_offsets[expert_id];
                            expert_end = h_offsets[expert_id + 1];
                            expert_count = expert_end - expert_start;
                        }
                        int assign_idx = -1;
                        int assign_token = -1;
                        int assign_k = -1;
                        if (gather_t && gather_t->Data && gather_t->DType == ETensorDType::INT32) {
                            CUDA_CHECK(cudaMemcpy(&assign_idx,
                                                  static_cast<const std::byte*>(gather_t->Data) + static_cast<std::size_t>(indices[k]) * sizeof(int),
                                                  sizeof(int),
                                                  cudaMemcpyDeviceToHost));
                            if (assign_idx >= 0) {
                                assign_token = assign_idx / top_k;
                                assign_k = assign_idx % top_k;
                            }
                        }
                        fprintf(stderr,
                                "[MOE_FWD_NAN_EXPERT] layer=%d token=%d k=%d idx=%d expert=%d range=[%d,%d) count=%d assign=%d token_from=%d k_from=%d w=%.6f vals=%.6f,%.6f,%.6f,%.6f\n",
                                layer_idx, token_idx, k, indices[k], expert_id, expert_start, expert_end, expert_count,
                                assign_idx, assign_token, assign_k, weights[k],
                                ex_vals[0], ex_vals[1], ex_vals[2], ex_vals[3]);
                        if (expert_id == 122 && permuted_t && permuted_t->Data && gate_up_w_t && gate_up_w_t->Data) {
                            const std::size_t hidden = tensor_row_width(*permuted_t);
                            const std::size_t out_dim = gate_up_t ? tensor_row_width(*gate_up_t) : 0;
                            const std::size_t stride = out_dim * hidden;
                            std::vector<float> in_row(hidden, 0.0f);
                            std::vector<float> w_row(hidden, 0.0f);
                            const bool in_ok = copy_tensor_token_sample_as_f32(*permuted_t, indices[k], in_row.size(), in_row);
                            const bool w_ok = (stride > 0) && copy_tensor_sample_offset_as_f32(*gate_up_w_t,
                                                                                               stride * static_cast<std::size_t>(expert_id),
                                                                                               w_row.size(),
                                                                                               w_row);
                            double dot = 0.0;
                            bool dot_nan = false;
                            int w_nan = 0;
                            int w_nan_idx = -1;
                            float w_nan_val = 0.0f;
                            if (in_ok && w_ok) {
                                for (std::size_t i = 0; i < in_row.size(); ++i) {
                                    const float a = in_row[i];
                                    const float b = w_row[i];
                                    if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b)) {
                                        dot_nan = true;
                                        if ((std::isnan(b) || std::isinf(b)) && w_nan_idx < 0) {
                                            w_nan_idx = static_cast<int>(i);
                                            w_nan_val = b;
                                        }
                                        break;
                                    }
                                    dot += static_cast<double>(a) * static_cast<double>(b);
                                }
                                for (float v : w_row) {
                                    if (std::isnan(v) || std::isinf(v)) {
                                        w_nan++;
                                    }
                                }
                            }
                            fprintf(stderr,
                                    "[MOE_FWD_NAN_DOT] layer=%d idx=%d expert=%d in_ok=%d w_ok=%d dot_nan=%d dot=%.6e w_nan=%d w_nan_idx=%d w_nan_val=%.6e w_ptr=%p\n",
                                    layer_idx, indices[k], expert_id,
                                    in_ok ? 1 : 0, w_ok ? 1 : 0, dot_nan ? 1 : 0, dot,
                                    w_nan, w_nan_idx, static_cast<double>(w_nan_val),
                                    gate_up_w_t ? static_cast<void*>(gate_up_w_t->Data) : nullptr);
                        } else if (expert_id == 122 && !gate_up_w_t) {
                            fprintf(stderr,
                                    "[MOE_FWD_NAN_DOT] layer=%d idx=%d expert=%d missing_weights\n",
                                    layer_idx, indices[k], expert_id);
                        }

                        if (ln2_t && ln2_t->Data) {
                            std::vector<float> ln2_vals(4, 0.0f);
                            (void)copy_tensor_token_sample_as_f32(*ln2_t, token_idx, ln2_vals.size(), ln2_vals);
                            fprintf(stderr,
                                    "[MOE_FWD_NAN_LN2] layer=%d token=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                                    layer_idx, token_idx,
                                    ln2_vals[0], ln2_vals[1], ln2_vals[2], ln2_vals[3]);
                        }

                        if (act_t && act_t->Data) {
                            std::vector<float> act_vals(4, 0.0f);
                            (void)copy_tensor_token_sample_as_f32(*act_t, indices[k], act_vals.size(), act_vals);
                            fprintf(stderr,
                                    "[MOE_FWD_NAN_ACT] layer=%d token=%d k=%d idx=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                                    layer_idx, token_idx, k, indices[k],
                                    act_vals[0], act_vals[1], act_vals[2], act_vals[3]);
                        }
                        if (gate_up_t && gate_up_t->Data) {
                            std::vector<float> gu_vals(4, 0.0f);
                            (void)copy_tensor_token_sample_as_f32(*gate_up_t, indices[k], gu_vals.size(), gu_vals);
                            fprintf(stderr,
                                    "[MOE_FWD_NAN_GATE_UP] layer=%d token=%d k=%d idx=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                                    layer_idx, token_idx, k, indices[k],
                                    gu_vals[0], gu_vals[1], gu_vals[2], gu_vals[3]);
                        }
                        if (permuted_t && permuted_t->Data) {
                            std::vector<float> pin_vals(4, 0.0f);
                            (void)copy_tensor_token_sample_as_f32(*permuted_t, indices[k], pin_vals.size(), pin_vals);
                            float pin_min = 0.0f, pin_max = 0.0f;
                            const bool pin_nan = tensor_row_has_nan_or_inf(*permuted_t, indices[k], &pin_min, &pin_max);
                            fprintf(stderr,
                                    "[MOE_FWD_NAN_PIN] layer=%d token=%d k=%d idx=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                                    layer_idx, token_idx, k, indices[k],
                                    pin_vals[0], pin_vals[1], pin_vals[2], pin_vals[3]);
                            fprintf(stderr,
                                    "[MOE_FWD_NAN_PIN_ROW] layer=%d token=%d k=%d idx=%d nan=%d min=%.6f max=%.6f\n",
                                    layer_idx, token_idx, k, indices[k],
                                    pin_nan ? 1 : 0, pin_min, pin_max);
                        }
                    }
                }
            }
        }
    }

    // DEBUG: Trace per-token MoE combine inputs for layer 0/top.
    static int moe_unpermute_token_trace = 0;
    if (moe_unpermute_token_trace < 4) {
        int layer_idx = op.attrs.layer_idx;
        std::string field;
        if (layer_idx < 0 && !op.inputs.empty()) {
            std::string_view name = op.inputs[0].name;
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx, field);
        }
        if (layer_idx < 0 || layer_idx < 4 || layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
            const int token_idx = 3;
            if (token_idx >= 0 && token_idx < num_tokens) {
                std::vector<float> weights(top_k, 0.0f);
                const std::size_t w_offset = static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(top_k);
                (void)copy_tensor_sample_offset_as_f32(routing_weights, w_offset, weights.size(), weights);

                std::vector<int> indices(top_k, -1);
                if (scatter_indices.Data && scatter_indices.DType == ETensorDType::INT32) {
                    CUDA_CHECK(cudaMemcpy(indices.data(),
                                          static_cast<const std::byte*>(scatter_indices.Data) + w_offset * sizeof(int),
                                          top_k * sizeof(int),
                                          cudaMemcpyDeviceToHost));
                }

                std::vector<float> out_vals(4, 0.0f);
                (void)copy_tensor_token_sample_as_f32(out, token_idx, out_vals.size(), out_vals);

                fprintf(stderr,
                        "[MOE_FWD_TOKEN] layer=%d token=%d weights=(",
                        layer_idx, token_idx);
                for (int k = 0; k < top_k; ++k) {
                    fprintf(stderr, "%s%.6f", (k ? "," : ""), weights[k]);
                }
                fprintf(stderr, ") indices=(");
                for (int k = 0; k < top_k; ++k) {
                    fprintf(stderr, "%s%d", (k ? "," : ""), indices[k]);
                }
                fprintf(stderr, ") out=%.6f,%.6f,%.6f,%.6f\n",
                        out_vals[0], out_vals[1], out_vals[2], out_vals[3]);

                for (int k = 0; k < top_k; ++k) {
                    if (indices[k] < 0) continue;
                    std::vector<float> ex_vals(4, 0.0f);
                    (void)copy_tensor_token_sample_as_f32(expert_out, indices[k], ex_vals.size(), ex_vals);
                    fprintf(stderr,
                            "[MOE_FWD_TOKEN_EXPERT] layer=%d token=%d k=%d idx=%d w=%.6f vals=%.6f,%.6f,%.6f,%.6f\n",
                            layer_idx, token_idx, k, indices[k], weights[k],
                            ex_vals[0], ex_vals[1], ex_vals[2], ex_vals[3]);
                }
            }
            moe_unpermute_token_trace++;
        }
    }

    // DEBUG: Locate max |out| in first 4096 elements and relate to expert_out.
    static int moe_unpermute_max_trace = 0;
    if (moe_unpermute_max_trace < 4) {
        int layer_idx = op.attrs.layer_idx;
        std::string field;
        if (layer_idx < 0 && !op.inputs.empty()) {
            std::string_view name = op.inputs[0].name;
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx, field);
        }
        if (layer_idx < 0 || layer_idx < 4 || layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
            const std::size_t total = static_cast<std::size_t>(out.nelem());
            const std::size_t sample = std::min<std::size_t>(4096, total);
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
                const int token_idx = static_cast<int>(max_idx / static_cast<std::size_t>(hidden_size));
                const int dim = static_cast<int>(max_idx % static_cast<std::size_t>(hidden_size));
                std::vector<float> weights(top_k, 0.0f);
                const std::size_t w_offset = static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(top_k);
                (void)copy_tensor_sample_offset_as_f32(routing_weights, w_offset, weights.size(), weights);
                std::vector<int> indices(top_k, -1);
                if (scatter_indices.Data && scatter_indices.DType == ETensorDType::INT32) {
                    CUDA_CHECK(cudaMemcpy(indices.data(),
                                          static_cast<const std::byte*>(scatter_indices.Data) + w_offset * sizeof(int),
                                          top_k * sizeof(int),
                                          cudaMemcpyDeviceToHost));
                }
                float recomposed = 0.0f;
                std::vector<float> ex_vals(top_k, 0.0f);
                for (int k = 0; k < top_k; ++k) {
                    if (indices[k] < 0) continue;
                    std::vector<float> ex_val(1, 0.0f);
                    const std::size_t ex_offset =
                        static_cast<std::size_t>(indices[k]) * static_cast<std::size_t>(hidden_size) +
                        static_cast<std::size_t>(dim);
                    (void)copy_tensor_sample_offset_as_f32(expert_out, ex_offset, ex_val.size(), ex_val);
                    ex_vals[k] = ex_val[0];
                    recomposed += weights[k] * ex_val[0];
                }
                const float recomposed_scaled = recomposed * moe_out_scale;
                fprintf(stderr,
                        "[MOE_FWD_MAX] layer=%d sample_max=%.6f token=%d dim=%d recomposed=%.6f scaled=%.6f scale=%.6f weights=(",
                        layer_idx, vals[max_idx], token_idx, dim, recomposed, recomposed_scaled, moe_out_scale);
                for (int k = 0; k < top_k; ++k) {
                    fprintf(stderr, "%s%.6f", (k ? "," : ""), weights[k]);
                }
                fprintf(stderr, ") indices=(");
                for (int k = 0; k < top_k; ++k) {
                    fprintf(stderr, "%s%d", (k ? "," : ""), indices[k]);
                }
                fprintf(stderr, ") vals=(");
                for (int k = 0; k < top_k; ++k) {
                    fprintf(stderr, "%s%.6f", (k ? "," : ""), ex_vals[k]);
                }
                fprintf(stderr, ")\n");

                int max_k = -1;
                float max_k_abs = 0.0f;
                for (int k = 0; k < top_k; ++k) {
                    const float av = std::fabs(ex_vals[k]);
                    if (av > max_k_abs) {
                        max_k_abs = av;
                        max_k = k;
                    }
                }
                if (max_k >= 0 && indices[max_k] >= 0) {
                    int expert_id = -1;
                    int expert_pos = indices[max_k];
                    if (layer_idx >= 0 && expert_pos >= 0) {
                        Tensor expert_offsets_view;
                        Tensor* expert_offsets_ptr = nullptr;
                        const std::string key = "blocks[" + std::to_string(layer_idx) + "].moe_expert_offsets";
                        auto it_saved = mMoESavedBuffers.find(key);
                        if (it_saved != mMoESavedBuffers.end() && it_saved->second != nullptr) {
                            expert_offsets_view.DType = ETensorDType::INT32;
                            expert_offsets_view.Rank = 1;
                            expert_offsets_view.Sizes[0] = static_cast<long>(mConfig.NumExperts + 1);
                            expert_offsets_view.Data = static_cast<std::byte*>(it_saved->second);
                            expert_offsets_ptr = &expert_offsets_view;
                        }
                        if (!expert_offsets_ptr) {
                            auto it = mTensorMap.find("moe_expert_offsets");
                            if (it != mTensorMap.end()) {
                                expert_offsets_ptr = &it->second;
                            }
                        }
                        if (expert_offsets_ptr && expert_offsets_ptr->Data) {
                            std::vector<int> host_offsets(static_cast<std::size_t>(mConfig.NumExperts + 1), 0);
                            CUDA_CHECK(cudaMemcpy(host_offsets.data(),
                                                  expert_offsets_ptr->get<int>(),
                                                  host_offsets.size() * sizeof(int),
                                                  cudaMemcpyDeviceToHost));
                            for (int e = 0; e < static_cast<int>(mConfig.NumExperts); ++e) {
                                if (expert_pos >= host_offsets[e] && expert_pos < host_offsets[e + 1]) {
                                    expert_id = e;
                                    break;
                                }
                            }
                        }
                    }
                    fprintf(stderr,
                            "[MOE_FWD_MAX_EXPERT] layer=%d token=%d dim=%d k=%d idx=%d expert=%d w=%.6f val=%.6f\n",
                            layer_idx, token_idx, dim, max_k, indices[max_k], expert_id, weights[max_k], ex_vals[max_k]);
                    log_tensor_row_stats("MOE_FWD_MAX_EXPERT_ROW", expert_out, indices[max_k]);
                    if (layer_idx >= 0) {
                        const std::string pin_name = "blocks[" + std::to_string(layer_idx) + "].permuted_input";
                        const std::string act_name = "blocks[" + std::to_string(layer_idx) + "].expert_act";
                        const std::string gate_name = "blocks[" + std::to_string(layer_idx) + "].expert_gate_up";
                        auto it_pin = mTensorMap.find(pin_name);
                        if (it_pin != mTensorMap.end()) {
                            log_tensor_row_stats("MOE_FWD_MAX_PERM_IN_ROW", it_pin->second, indices[max_k]);
                        }
                        auto it_act = mTensorMap.find(act_name);
                        if (it_act != mTensorMap.end()) {
                            log_tensor_row_stats("MOE_FWD_MAX_ACT_ROW", it_act->second, indices[max_k]);
                        }
                        auto it_gate = mTensorMap.find(gate_name);
                        if (it_gate != mTensorMap.end()) {
                            log_tensor_row_stats("MOE_FWD_MAX_GATE_UP_ROW", it_gate->second, indices[max_k]);
                        }
                    }
                }
            }
            moe_unpermute_max_trace++;
        }
    }

    mTensorMap[op.outputs[0].name] = out;
}

void CompiledExecutor::dispatch_moe_unpermute_backward(const CompiledOp& op) {
    Tensor& d_output = resolve_tensor(op.inputs[0]);
    Tensor& expert_out = resolve_tensor(op.inputs[1]);
    Tensor& routing_weights = resolve_tensor(op.inputs[2]);
    Tensor& scatter_indices = resolve_tensor(op.inputs[3]);

    Tensor& d_expert_out = ensure_output_tensor(op.outputs[0]);
    Tensor& d_routing_weights = ensure_output_tensor(op.outputs[1]);

    const int top_k = op.attrs.top_k;
    const int num_tokens = static_cast<int>(routing_weights.Sizes[0]);
    const int total_tokens = num_tokens * top_k;
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
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

    static int moe_unpermute_trace_layer = 0;
    if (should_trace_moe_layer(layer_idx, moe_unpermute_trace_layer)) {
        log_tensor_mag_unbounded("MOE_UNPERMUTE_BWD_TRACE_DOUT", layer_idx, op.inputs[0].name, d_output, 4096);
        log_tensor_mag_unbounded("MOE_UNPERMUTE_BWD_TRACE_EXPERT", layer_idx, op.inputs[1].name, expert_out, 4096);
        log_tensor_mag_unbounded("MOE_UNPERMUTE_BWD_TRACE_ROUTING", layer_idx, op.inputs[2].name, routing_weights, 4096);
        if (scatter_indices.DType == ETensorDType::INT32 && scatter_indices.Data) {
            const std::size_t n = std::min<std::size_t>(256, static_cast<std::size_t>(scatter_indices.nelem()));
            if (n > 0) {
                std::vector<int> vals(n);
                CUDA_CHECK(cudaMemcpy(vals.data(), scatter_indices.Data, n * sizeof(int), cudaMemcpyDeviceToHost));
                int minv = vals[0];
                int maxv = vals[0];
                for (std::size_t i = 1; i < n; ++i) {
                    minv = std::min(minv, vals[i]);
                    maxv = std::max(maxv, vals[i]);
                }
                fprintf(stderr,
                        "[MOE_UNPERMUTE_BWD_TRACE_SCATTER] layer=%d name=%s ptr=%p n=%zu min=%d max=%d\n",
                        layer_idx,
                        op.inputs[3].name.c_str(),
                        scatter_indices.Data,
                        n,
                        minv,
                        maxv);
            }
        }
    }

    // One-time NaN watchdog to pinpoint MoE unpermute backward issues.
    static int moe_unpermute_nan_logged = 0;
    if (moe_unpermute_nan_logged < 4) {
        auto find_nan = [&](const Tensor& t, long* out_row, float* out_min, float* out_max) {
            Tensor scan = t;
            if (scan.Rank > 2 && scan.Sizes[0] == mB && scan.Sizes[1] == mT) {
                scan = view_tensor(scan, {mB * mT, scan.Sizes[scan.Rank - 1]});
            }
            return find_first_nan_row(scan, out_row, out_min, out_max);
        };
        long row_dout = -1;
        long row_expert = -1;
        long row_routing = -1;
        float min_dout = 0.0f, max_dout = 0.0f;
        float min_expert = 0.0f, max_expert = 0.0f;
        float min_routing = 0.0f, max_routing = 0.0f;
        const bool nan_dout = find_nan(d_output, &row_dout, &min_dout, &max_dout);
        const bool nan_expert = find_nan(expert_out, &row_expert, &min_expert, &max_expert);
        const bool nan_routing = find_nan(routing_weights, &row_routing, &min_routing, &max_routing);
        if (nan_dout || nan_expert || nan_routing) {
            fprintf(stderr,
                    "[MOE_UNPERMUTE_BWD_NAN] layer=%d nan_dout=%d nan_expert=%d nan_routing=%d d_out=%s expert=%s routing=%s\n",
                    layer_idx,
                    nan_dout ? 1 : 0,
                    nan_expert ? 1 : 0,
                    nan_routing ? 1 : 0,
                    op.inputs[0].name.c_str(),
                    op.inputs[1].name.c_str(),
                    op.inputs[2].name.c_str());
            if (nan_dout) {
                const long b = (d_output.Rank > 2 && d_output.Sizes[0] == mB && d_output.Sizes[1] == mT)
                    ? (row_dout / static_cast<long>(mT)) : -1;
                const long t = (d_output.Rank > 2 && d_output.Sizes[0] == mB && d_output.Sizes[1] == mT)
                    ? (row_dout % static_cast<long>(mT)) : -1;
                fprintf(stderr,
                        "[MOE_UNPERMUTE_BWD_NAN_DOUT_ROW] row=%ld b=%ld t=%ld min=%.6f max=%.6f\n",
                        row_dout, b, t, min_dout, max_dout);
            }
            if (nan_expert) {
                fprintf(stderr,
                        "[MOE_UNPERMUTE_BWD_NAN_EXPERT_ROW] row=%ld min=%.6f max=%.6f\n",
                        row_expert, min_expert, max_expert);
            }
            if (nan_routing) {
                fprintf(stderr,
                        "[MOE_UNPERMUTE_BWD_NAN_ROUTING_ROW] row=%ld min=%.6f max=%.6f\n",
                        row_routing, min_routing, max_routing);
            }
            log_tensor_stats_ex("MOE_UNPERMUTE_BWD_NAN_DOUT", layer_idx, op.inputs[0].name, d_output, 4096, true);
            log_tensor_stats_ex("MOE_UNPERMUTE_BWD_NAN_EXPERT", layer_idx, op.inputs[1].name, expert_out, 4096, true);
            log_tensor_stats_ex("MOE_UNPERMUTE_BWD_NAN_ROUTING", layer_idx, op.inputs[2].name, routing_weights, 4096, true);
            moe_unpermute_nan_logged++;
        }
    }
    static int moe_unpermute_mag = 0;
    if (moe_unpermute_mag < 8 && layer_idx <= 2) {
        log_tensor_mag("MOE_UNPERMUTE_BWD_DOUT", layer_idx, op.inputs[0].name, d_output, 4096);
        log_tensor_mag("MOE_UNPERMUTE_BWD_EXPERT", layer_idx, op.inputs[1].name, expert_out, 4096);
        log_tensor_mag("MOE_UNPERMUTE_BWD_ROUTING", layer_idx, op.inputs[2].name, routing_weights, 4096);
        moe_unpermute_mag++;
    }

    // DEBUG: track weight integrity across unpermute backward for layer 2.
    static int moe_unpermute_bwd_w_trace = 0;
    bool pre_nan = false;
    float pre_min = 0.0f, pre_max = 0.0f;
    int weight_check_layer = op.attrs.layer_idx;
    if (weight_check_layer < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("d_", 0) == 0) {
            name.remove_prefix(2);
        }
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        std::string field;
        parse_block_param(name, weight_check_layer, field);
    }
    log_moe_gate_up_weight_sample("PRE_MOE_UNPERMUTE_BWD", weight_check_layer, mMicroStep, mWeights, mConfig);
    if (weight_check_layer == 2 && moe_unpermute_bwd_w_trace < 1 && mWeights.has("blocks[2].experts_gate_up")) {
        Tensor& w = mWeights.get("blocks[2].experts_gate_up");
        pre_nan = tensor_row_has_nan_or_inf(w, 122, &pre_min, &pre_max);
    }

    // DEBUG: Trace MoE unpermute backward inputs (routing weights + scatter indices).
    static int moe_unpermute_trace = 0;
    if (moe_unpermute_trace < 12) {
        int layer_idx = op.attrs.layer_idx;
        std::string field;
        if (layer_idx < 0 && !op.inputs.empty()) {
            std::string_view name = op.inputs[0].name;
            if (name.rfind("d_", 0) == 0) {
                name.remove_prefix(2);
            }
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx, field);
        }
        if (layer_idx < 0 || layer_idx == 0 || layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
            log_tensor_stats_ex("MOE_UNPERMUTE_BWD_DOUT", layer_idx, op.inputs[0].name, d_output, 4096, true);
            log_tensor_stats_ex("MOE_UNPERMUTE_BWD_ROUTING_W", layer_idx, op.inputs[2].name, routing_weights, 4096, true);
            log_tensor_stats_ex("MOE_UNPERMUTE_BWD_EXPERT_OUT", layer_idx, op.inputs[1].name, expert_out, 4096, true);
            if (scatter_indices.DType == ETensorDType::INT32 && scatter_indices.Data) {
                const std::size_t n = std::min<std::size_t>(256, static_cast<std::size_t>(scatter_indices.nelem()));
                if (n > 0) {
                    std::vector<int> vals(n);
                    CUDA_CHECK(cudaMemcpy(vals.data(), scatter_indices.Data, n * sizeof(int), cudaMemcpyDeviceToHost));
                    int minv = vals[0];
                    int maxv = vals[0];
                    for (std::size_t i = 1; i < n; ++i) {
                        minv = std::min(minv, vals[i]);
                        maxv = std::max(maxv, vals[i]);
                    }
                    fprintf(stderr,
                            "[MOE_UNPERMUTE_BWD_SCATTER] layer=%d name=%s ptr=%p n=%zu min=%d max=%d\n",
                            layer_idx,
                            op.inputs[3].name.c_str(),
                            scatter_indices.Data,
                            n,
                            minv,
                            maxv);
                }
            }
            moe_unpermute_trace++;
        }
    }

    if (d_output.DType == ETensorDType::BF16) {
        moe_combine_backward(d_expert_out.get<nv_bfloat16>(),
                             d_routing_weights.get<nv_bfloat16>(),
                             d_output.get<nv_bfloat16>(),
                             expert_out.get<nv_bfloat16>(),
                             routing_weights.get<nv_bfloat16>(),
                             scatter_indices.get<int>(),
                             num_tokens, total_tokens, hidden_size, top_k,
                             mRunState.MainStream);
    } else {
        moe_combine_backward(d_expert_out.get<float>(),
                             d_routing_weights.get<float>(),
                             d_output.get<float>(),
                             expert_out.get<float>(),
                             routing_weights.get<float>(),
                             scatter_indices.get<int>(),
                             num_tokens, total_tokens, hidden_size, top_k,
                             mRunState.MainStream);
    }

    static int moe_unpermute_trace_layer_out = 0;
    if (should_trace_moe_layer(layer_idx, moe_unpermute_trace_layer_out)) {
        log_tensor_mag_unbounded("MOE_UNPERMUTE_BWD_TRACE_DEXPERT", layer_idx, op.outputs[0].name, d_expert_out, 4096);
        log_tensor_mag_unbounded("MOE_UNPERMUTE_BWD_TRACE_DROUTING", layer_idx, op.outputs[1].name, d_routing_weights, 4096);
    }

    if (weight_check_layer == 2 && moe_unpermute_bwd_w_trace < 1 && mWeights.has("blocks[2].experts_gate_up")) {
        Tensor& w = mWeights.get("blocks[2].experts_gate_up");
        float post_min = 0.0f, post_max = 0.0f;
        const bool post_nan = tensor_row_has_nan_or_inf(w, 122, &post_min, &post_max);
        fprintf(stderr,
                "[MOE_UNPERMUTE_BWD_W] layer=%d pre_nan=%d pre_min=%.6f pre_max=%.6f post_nan=%d post_min=%.6f post_max=%.6f\n",
                weight_check_layer,
                pre_nan ? 1 : 0, pre_min, pre_max,
                post_nan ? 1 : 0, post_min, post_max);
        moe_unpermute_bwd_w_trace++;
    }

    mTensorMap[op.outputs[0].name] = d_expert_out;
    mTensorMap[op.outputs[1].name] = d_routing_weights;
}


}  // namespace dsl
