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

void CompiledExecutor::dispatch_moe_permute(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& routing_indices = resolve_tensor(op.inputs[1]);
    Tensor& permuted = ensure_output_tensor(op.outputs[0]);
    Tensor& scatter_indices = ensure_output_tensor(op.outputs[1]);

    const int num_tokens = static_cast<int>(inp.Sizes[0]);
    const int hidden_size = static_cast<int>(inp.Sizes[1]);
    const int top_k = op.attrs.top_k;
    const int total_tokens = num_tokens * top_k;
    const int num_experts = static_cast<int>(mConfig.NumExperts);
    int layer_idx_any = op.attrs.layer_idx;
    std::string field_any;
    if (layer_idx_any < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        parse_block_param(name, layer_idx_any, field_any);
    }
    log_moe_gate_up_weight_sample("PRE_MOE_PERMUTE", layer_idx_any, mMicroStep, mWeights, mConfig);

    // Allocate temporary buffers for permutation indices
    // Use Stack.allocate for small buffers that can be freed at layer boundaries
    Tensor expert_counts = mRunState.Stack.allocate(ETensorDType::INT32, {num_experts}, "moe_expert_counts");
    Tensor expert_offsets = mRunState.Stack.allocate(ETensorDType::INT32, {num_experts + 1}, "moe_expert_offsets");
    Tensor expert_positions = mRunState.Stack.allocate(ETensorDType::INT32, {num_experts}, "moe_expert_positions");
    Tensor gather_indices = mRunState.Stack.allocate(ETensorDType::INT32, {total_tokens}, "moe_gather_indices");

    // Zero-initialize expert_positions before atomicAdd in build_indices
    // Stack memory is reused across forward passes and contains stale values
    fill_zero(expert_positions, mRunState.MainStream);

    // Compute expert counts
    moe_compute_expert_counts(expert_counts.get<int>(),
                              routing_indices.get<int>(),
                              num_tokens, top_k, num_experts, mRunState.MainStream);

    // Compute expert offsets (prefix sum)
    moe_compute_expert_offsets(expert_offsets.get<int>(),
                               expert_counts.get<int>(),
                               num_experts, mRunState.MainStream);

    // Build gather and scatter indices
    moe_build_indices(gather_indices.get<int>(),
                      scatter_indices.get<int>(),
                      routing_indices.get<int>(),
                      expert_offsets.get<int>(),
                      expert_positions.get<int>(),
                      num_tokens, top_k, num_experts, mRunState.MainStream);

    // Cache expert offsets on host for grouped GEMM fast path.
    if (num_experts > 0) {
        mMoEExpertOffsetsData.resize(static_cast<std::size_t>(num_experts + 1));
        CUDA_CHECK(cudaMemcpyAsync(mMoEExpertOffsetsData.data(),
                                   expert_offsets.get<int>(),
                                   static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
    }

    // DEBUG: validate expert_offsets and index coverage for early layers.
    {
        int layer_idx = op.attrs.layer_idx;
        std::string field;
        if (layer_idx < 0 && !op.inputs.empty()) {
            std::string_view name = op.inputs[0].name;
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx, field);
        }
        if (layer_idx >= 0 && layer_idx < 4) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            std::vector<int> h_offsets(num_experts + 1, 0);
            CUDA_CHECK(cudaMemcpy(h_offsets.data(),
                                  expert_offsets.Data,
                                  (num_experts + 1) * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            int offsets_bad = 0;
            for (int i = 1; i <= num_experts; ++i) {
                if (h_offsets[i] < h_offsets[i - 1]) {
                    offsets_bad++;
                }
            }
            const int offsets_last = h_offsets[num_experts];

            std::vector<int> h_scatter(total_tokens, -1);
            std::vector<int> h_gather(total_tokens, -1);
            CUDA_CHECK(cudaMemcpy(h_scatter.data(),
                                  scatter_indices.Data,
                                  total_tokens * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_gather.data(),
                                  gather_indices.Data,
                                  total_tokens * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            int scatter_min = total_tokens;
            int scatter_max = -1;
            int scatter_oob = 0;
            int gather_min = total_tokens;
            int gather_max = -1;
            int gather_oob = 0;
            std::vector<int> seen(total_tokens, 0);
            for (int i = 0; i < total_tokens; ++i) {
                const int dest = h_scatter[i];
                if (dest < 0 || dest >= total_tokens) {
                    scatter_oob++;
                } else {
                    scatter_min = std::min(scatter_min, dest);
                    scatter_max = std::max(scatter_max, dest);
                    seen[dest]++;
                }
                const int src = h_gather[i];
                if (src < 0 || src >= total_tokens) {
                    gather_oob++;
                } else {
                    gather_min = std::min(gather_min, src);
                    gather_max = std::max(gather_max, src);
                }
            }
            int miss = 0;
            int dup = 0;
            for (int i = 0; i < total_tokens; ++i) {
                if (seen[i] == 0) miss++;
                if (seen[i] > 1) dup++;
            }
            fprintf(stderr,
                    "[MOE_IDX_STATS] layer=%d total=%d offsets_last=%d offsets_bad=%d "
                    "scatter_min=%d scatter_max=%d scatter_oob=%d miss=%d dup=%d "
                    "gather_min=%d gather_max=%d gather_oob=%d routing_dtype=%d\n",
                    layer_idx, total_tokens, offsets_last, offsets_bad,
                    scatter_min, scatter_max, scatter_oob, miss, dup,
                    gather_min, gather_max, gather_oob,
                    static_cast<int>(routing_indices.DType));
        }
    }

    // DEBUG: Validate gather/scatter mapping for token 3 in early layers.
    {
        int layer_idx = op.attrs.layer_idx;
        std::string field;
        if (layer_idx < 0 && !op.inputs.empty()) {
            std::string_view name = op.inputs[0].name;
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx, field);
        }
        if (layer_idx >= 0 && layer_idx < 4) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            const int token_idx = 3;
            std::vector<int> scat(top_k, -1);
            CUDA_CHECK(cudaMemcpy(scat.data(),
                                  static_cast<const std::byte*>(scatter_indices.Data) + token_idx * top_k * sizeof(int),
                                  top_k * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            for (int k = 0; k < top_k; ++k) {
                const int assignment_idx = token_idx * top_k + k;
                const int dest_idx = scat[k];
                int gather_idx = -1;
                if (dest_idx >= 0 && dest_idx < total_tokens) {
                    CUDA_CHECK(cudaMemcpy(&gather_idx,
                                          static_cast<const std::byte*>(gather_indices.Data) + dest_idx * sizeof(int),
                                          sizeof(int),
                                          cudaMemcpyDeviceToHost));
                }
                if (gather_idx != assignment_idx) {
                    fprintf(stderr,
                            "[MOE_IDX_MISMATCH] layer=%d token=%d k=%d assignment=%d dest=%d gather=%d\n",
                            layer_idx, token_idx, k, assignment_idx, dest_idx, gather_idx);
                    std::vector<int> ridx(top_k, -1);
                    if (routing_indices.Data && routing_indices.DType == ETensorDType::INT32) {
                        CUDA_CHECK(cudaMemcpy(ridx.data(),
                                              static_cast<const std::byte*>(routing_indices.Data) + token_idx * top_k * sizeof(int),
                                              top_k * sizeof(int),
                                              cudaMemcpyDeviceToHost));
                    }
                    fprintf(stderr, "[MOE_IDX_MISMATCH_RIDX] layer=%d token=%d ridx=(", layer_idx, token_idx);
                    for (int kk = 0; kk < top_k; ++kk) {
                        fprintf(stderr, "%s%d", (kk ? "," : ""), ridx[kk]);
                    }
                    fprintf(stderr, ")\n");
                    break;
                }
            }
        }
    }

    // Permute tokens
    if (inp.DType == ETensorDType::BF16) {
        moe_permute_tokens(permuted.get<nv_bfloat16>(),
                           inp.get<nv_bfloat16>(),
                           gather_indices.get<int>(),
                           total_tokens, num_tokens, hidden_size, top_k, mRunState.MainStream);
    } else {
        moe_permute_tokens(permuted.get<float>(),
                           inp.get<float>(),
                           gather_indices.get<int>(),
                           total_tokens, num_tokens, hidden_size, top_k, mRunState.MainStream);
    }

    // Targeted trace: map a permuted row back to its source token + log source activations.
    {
        const bool moe_target_trace = (std::getenv("SUROGATE_MOE_DOT_TRACE") != nullptr);
        static int moe_target_permute_trace = 0;
        const int target_layer = env_int("SUROGATE_MOE_DOT_LAYER", -1);
        const int target_pos = env_int("SUROGATE_MOE_DOT_POS", -1);
        if (moe_target_trace && moe_target_permute_trace < 1 &&
            target_pos >= 0 && target_pos < total_tokens &&
            (target_layer < 0 || layer_idx_any == target_layer)) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            int gather_idx = -1;
            CUDA_CHECK(cudaMemcpy(&gather_idx,
                                  static_cast<const std::byte*>(gather_indices.Data) + target_pos * sizeof(int),
                                  sizeof(int),
                                  cudaMemcpyDeviceToHost));
            if (gather_idx >= 0) {
                const int token_idx = (top_k > 0) ? (gather_idx / top_k) : -1;
                const int k_idx = (top_k > 0) ? (gather_idx % top_k) : -1;
                fprintf(stderr,
                        "[MOE_PERMUTE_TRACE] layer=%d target_pos=%d gather_idx=%d token_idx=%d k=%d\n",
                        layer_idx_any, target_pos, gather_idx, token_idx, k_idx);
                if (token_idx >= 0 && token_idx < num_tokens) {
                    log_tensor_row_stats("MOE_PERMUTE_OUT_ROW_STATS", permuted, target_pos);
                    log_tensor_row_stats("MOE_PERMUTE_SRC_ROW_STATS", inp, token_idx);

                    const std::string prefix = "blocks[" + std::to_string(layer_idx_any) + "].";
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
                    auto log_token_row = [&](const char* tag, Tensor* t) {
                        if (!t || !t->Data) {
                            return;
                        }
                        if (t->Rank == 2) {
                            log_tensor_row_stats(tag, *t, token_idx);
                            return;
                        }
                        if (t->Rank >= 3 && t->Sizes[0] * t->Sizes[1] == num_tokens) {
                            Tensor flat = view_tensor(*t, {static_cast<long>(num_tokens), static_cast<long>(hidden_size)});
                            log_tensor_row_stats(tag, flat, token_idx);
                            return;
                        }
                        fprintf(stderr,
                                "[%s] layer=%d token=%d rank=%d shape=%s (skip)\n",
                                tag,
                                layer_idx_any,
                                token_idx,
                                t->Rank,
                                tensor_shape_str(*t).c_str());
                    };

                    log_token_row("MOE_LN2_ROW_STATS", find_tensor(prefix + "ln2"));
                    log_token_row("MOE_RES_ATT_ROW_STATS", find_tensor(prefix + "res_att"));
                    log_token_row("MOE_RES_FFN_ROW_STATS", find_tensor(prefix + "res_ffn"));
                    log_token_row("MOE_ATT_OUT_ROW_STATS", find_tensor(prefix + "att_out"));
                }
            } else {
                fprintf(stderr,
                        "[MOE_PERMUTE_TRACE] layer=%d target_pos=%d gather_idx=%d (invalid)\n",
                        layer_idx_any, target_pos, gather_idx);
            }
            moe_target_permute_trace++;
        }
    }

    // Persist per-layer routing buffers for backward (expert_offsets + gather_indices).
    {
        int layer_idx = op.attrs.layer_idx;
        std::string field;
        if (layer_idx < 0 && !op.inputs.empty()) {
            std::string_view name = op.inputs[0].name;
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx, field);
        }
        if (layer_idx >= 0) {
            auto save_buffer = [&](const std::string& suffix, const Tensor& src) {
                if (!src.Data) {
                    return;
                }
                const std::string key = "blocks[" + std::to_string(layer_idx) + "]." + suffix;
                const size_t bytes = src.bytes();
                if (bytes == 0) {
                    return;
                }
                auto buf_it = mMoESavedBuffers.find(key);
                if (buf_it == mMoESavedBuffers.end() || mMoESavedSizes[key] < bytes) {
                    if (buf_it != mMoESavedBuffers.end() && buf_it->second != nullptr) {
                        CUDA_CHECK(cudaFree(buf_it->second));
                    }
                    void* new_buffer = nullptr;
                    CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
                    mMoESavedBuffers[key] = new_buffer;
                    mMoESavedSizes[key] = bytes;
                }
                void* dst_buffer = mMoESavedBuffers[key];
                CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src.Data, bytes,
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));
            };
            save_buffer("moe_expert_offsets", expert_offsets);
            save_buffer("moe_gather_indices", gather_indices);
        }
    }

    // DEBUG: If permuted_input has NaNs for token 3, dump indices and mapping.
    {
        int layer_idx = op.attrs.layer_idx;
        std::string field;
        if (layer_idx < 0 && !op.inputs.empty()) {
            std::string_view name = op.inputs[0].name;
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx, field);
        }
        if (layer_idx >= 0 && layer_idx < 4) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            const int token_idx = 3;
            const std::size_t w_offset = static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(top_k);
            std::vector<int> indices(top_k, -1);
            if (scatter_indices.Data && scatter_indices.DType == ETensorDType::INT32) {
                CUDA_CHECK(cudaMemcpy(indices.data(),
                                      static_cast<const std::byte*>(scatter_indices.Data) + w_offset * sizeof(int),
                                      top_k * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            }
            for (int k = 0; k < top_k; ++k) {
                if (indices[k] < 0 || indices[k] >= total_tokens) {
                    continue;
                }
                if (tensor_sample_has_nan_or_inf(permuted, indices[k])) {
                    std::vector<float> in_vals(4, 0.0f);
                    (void)copy_tensor_token_sample_as_f32(inp, token_idx, in_vals.size(), in_vals);
                    int gather_idx = -1;
                    CUDA_CHECK(cudaMemcpy(&gather_idx,
                                          static_cast<const std::byte*>(gather_indices.Data) + indices[k] * sizeof(int),
                                          sizeof(int),
                                          cudaMemcpyDeviceToHost));
                    fprintf(stderr,
                            "[MOE_PERMUTE_NAN] layer=%d token=%d k=%d perm_idx=%d gather_idx=%d inp_vals=%.6f,%.6f,%.6f,%.6f\n",
                            layer_idx, token_idx, k, indices[k], gather_idx,
                            in_vals[0], in_vals[1], in_vals[2], in_vals[3]);
                    break;
                }
            }
        }
    }

    // DEBUG: trace input/permuted magnitude for layer 0/top.
    static int moe_permute_trace = 0;
    if (moe_permute_trace < 4) {
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
            fprintf(stderr,
                    "[MOE_PERMUTE_PTR] layer=%d inp=%s ptr=%p shape=%s out=%s ptr=%p shape=%s\n",
                    layer_idx,
                    op.inputs[0].name.c_str(),
                    static_cast<void*>(inp.Data),
                    tensor_shape_str(inp).c_str(),
                    op.outputs[0].name.c_str(),
                    static_cast<void*>(permuted.Data),
                    tensor_shape_str(permuted).c_str());
            fprintf(stderr,
                    "[MOE_PERMUTE_IN] layer=%d name=%s shape=%s\n",
                    layer_idx,
                    op.inputs[0].name.c_str(),
                    tensor_shape_str(inp).c_str());
            const std::size_t in_total = static_cast<std::size_t>(inp.nelem());
            const std::size_t out_total = static_cast<std::size_t>(permuted.nelem());
            const std::size_t sample = std::min<std::size_t>(in_total, 1u << 20);
            std::vector<float> in_vals;
            std::vector<float> out_vals;
            if (copy_tensor_sample_as_f32(inp, sample, in_vals) &&
                copy_tensor_sample_as_f32(permuted, std::min<std::size_t>(out_total, 1u << 20), out_vals)) {
                auto max_abs = [](const std::vector<float>& v) {
                    float m = 0.0f;
                    for (float x : v) m = std::max(m, std::fabs(x));
                    return m;
                };
                fprintf(stderr,
                        "[MOE_PERMUTE_MAX] layer=%d inp_max=%.6f out_max=%.6f in_total=%zu out_total=%zu\n",
                        layer_idx,
                        max_abs(in_vals),
                        max_abs(out_vals),
                        in_total, out_total);
            }
            moe_permute_trace++;
        }
    }

    // Store expert_offsets in scatter_indices output for later use
    // Note: scatter_indices tensor is already populated by moe_build_indices

    mTensorMap[op.outputs[0].name] = permuted;
    mTensorMap[op.outputs[1].name] = scatter_indices;
    // Store expert_offsets for use by grouped GEMM and unpermute
    // Note: expert_offsets lives on the stack; store for this layer in case we need it,
    // but grouped GEMM should prefer host offsets to avoid touching possibly-stale device memory.
    mTensorMap["moe_expert_offsets"] = expert_offsets;
    mTensorMap["moe_gather_indices"] = gather_indices;

    // Keep temps for later use
    mTemps.push_back(expert_counts);
    mTemps.push_back(expert_offsets);
    mTemps.push_back(expert_positions);
    mTemps.push_back(gather_indices);
}

void CompiledExecutor::dispatch_moe_permute_backward(const CompiledOp& op) {
    Tensor& d_permuted = resolve_tensor(op.inputs[0]);
    Tensor& gather_indices_saved = resolve_tensor(op.inputs[1]);  // Saved from forward
    Tensor& d_input = ensure_output_tensor(op.outputs[0]);

    // Prefer per-layer saved gather indices when available.
    Tensor* gather_indices = nullptr;
    Tensor gather_indices_view;
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
    log_moe_gate_up_weight_sample("PRE_MOE_GATE_UP_BWD", layer_idx, mMicroStep, mWeights, mConfig);
    log_moe_gate_up_weight_sample("PRE_MOE_PERMUTE_BWD", layer_idx, mMicroStep, mWeights, mConfig);
    if (layer_idx >= 0) {
        const std::string key = "blocks[" + std::to_string(layer_idx) + "].moe_gather_indices";
        auto it = mMoESavedBuffers.find(key);
        if (it != mMoESavedBuffers.end() && it->second != nullptr) {
            const int top_k = op.attrs.top_k > 0 ? op.attrs.top_k : 1;
            const int num_tokens = static_cast<int>(d_input.Sizes[0]);
            const int total_tokens = num_tokens * top_k;
            gather_indices_view.DType = ETensorDType::INT32;
            gather_indices_view.Rank = 1;
            gather_indices_view.Sizes[0] = total_tokens;
            gather_indices_view.Data = static_cast<std::byte*>(it->second);
            gather_indices = &gather_indices_view;
        }
    }
    if (!gather_indices) {
        auto it = mTensorMap.find("moe_gather_indices");
        gather_indices = (it != mTensorMap.end()) ? &it->second : &gather_indices_saved;
    }

    static int moe_permute_trace_layer = 0;
    if (should_trace_moe_layer(layer_idx, moe_permute_trace_layer)) {
        log_tensor_mag_unbounded("MOE_PERMUTE_BWD_TRACE_IN", layer_idx, op.inputs[0].name, d_permuted, 4096);
        const std::size_t perm_total = static_cast<std::size_t>(d_permuted.nelem());
        if (perm_total > 0) {
            log_tensor_sample_stats("MOE_PERMUTE_BWD_TRACE_IN_MID", d_permuted, perm_total / 2, 4096);
        }
        if (gather_indices && gather_indices->DType == ETensorDType::INT32 && gather_indices->Data) {
            const std::size_t n = std::min<std::size_t>(256, static_cast<std::size_t>(gather_indices->nelem()));
            if (n > 0) {
                std::vector<int> vals(n);
                CUDA_CHECK(cudaMemcpy(vals.data(), gather_indices->Data, n * sizeof(int), cudaMemcpyDeviceToHost));
                int minv = vals[0];
                int maxv = vals[0];
                for (std::size_t i = 1; i < n; ++i) {
                    minv = std::min(minv, vals[i]);
                    maxv = std::max(maxv, vals[i]);
                }
                fprintf(stderr,
                        "[MOE_PERMUTE_BWD_TRACE_GATHER] layer=%d name=%s ptr=%p n=%zu min=%d max=%d\n",
                        layer_idx,
                        op.inputs[1].name.c_str(),
                        gather_indices->Data,
                        n,
                        minv,
                        maxv);
            }
        }
    }

    const int top_k = op.attrs.top_k;
    const int num_tokens = static_cast<int>(d_input.Sizes[0]);
    const int hidden_size = static_cast<int>(d_input.Sizes[1]);
    const int total_tokens = num_tokens * top_k;
    static int moe_permute_trace_layer_meta = 0;
    if (should_trace_moe_layer(layer_idx, moe_permute_trace_layer_meta)) {
        const long gather_n = gather_indices ? static_cast<long>(gather_indices->nelem()) : -1;
        fprintf(stderr,
                "[MOE_PERMUTE_BWD_META] layer=%d top_k=%d num_tokens=%d total_tokens=%d hidden=%d gather_nelem=%ld\n",
                layer_idx, top_k, num_tokens, total_tokens, hidden_size, gather_n);
    }
    static int moe_permute_trace_layer_dup = 0;
    if (should_trace_moe_layer(layer_idx, moe_permute_trace_layer_dup)) {
        if (gather_indices && gather_indices->DType == ETensorDType::INT32 && gather_indices->Data &&
            total_tokens > 0 && total_tokens <= 65536) {
            std::vector<int> host_indices(static_cast<std::size_t>(total_tokens));
            CUDA_CHECK(cudaMemcpy(host_indices.data(), gather_indices->Data,
                                  host_indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
            std::vector<int> seen(static_cast<std::size_t>(total_tokens), 0);
            int min_idx = std::numeric_limits<int>::max();
            int max_idx = std::numeric_limits<int>::min();
            int oob = 0;
            int dup = 0;
            for (int v : host_indices) {
                min_idx = std::min(min_idx, v);
                max_idx = std::max(max_idx, v);
                if (v < 0 || v >= total_tokens) {
                    oob++;
                    continue;
                }
                if (seen[static_cast<std::size_t>(v)]++ > 0) {
                    dup++;
                }
            }
            int missing = 0;
            for (int c : seen) {
                if (c == 0) {
                    missing++;
                }
            }
            fprintf(stderr,
                    "[MOE_PERMUTE_BWD_GATHER_CHECK] layer=%d total=%d min=%d max=%d oob=%d dup=%d missing=%d\n",
                    layer_idx, total_tokens, min_idx, max_idx, oob, dup, missing);
        }
    }

    // NaN watchdog for permute backward inputs/outputs (log a few occurrences).
    static int moe_permute_nan_logged = 0;
    if (moe_permute_nan_logged < 4) {
        long row_perm = -1;
        float perm_min = 0.0f, perm_max = 0.0f;
        if (find_first_nan_row(d_permuted, &row_perm, &perm_min, &perm_max)) {
            fprintf(stderr,
                    "[MOE_PERMUTE_BWD_NAN_IN] layer=%d d_permuted=%s row=%ld min=%.6f max=%.6f\n",
                    layer_idx, op.inputs[0].name.c_str(), row_perm, perm_min, perm_max);
            log_tensor_stats_ex("MOE_PERMUTE_BWD_NAN_IN_STATS", layer_idx, op.inputs[0].name, d_permuted, 4096, true);
            moe_permute_nan_logged++;
        }
    }

    static int moe_permute_mag = 0;
    if (moe_permute_mag < 8 && layer_idx <= 2) {
        log_tensor_mag("MOE_PERMUTE_BWD_IN", layer_idx, op.inputs[0].name, d_permuted, 4096);
        moe_permute_mag++;
    }

    // Log gather indices bounds for visibility.
    static int moe_permute_gather_trace = 0;
    if (moe_permute_gather_trace < 4) {
        std::vector<int> host_indices(std::min(total_tokens, 4096));
        cudaMemcpy(host_indices.data(), gather_indices->Data,
                   host_indices.size() * sizeof(int), cudaMemcpyDeviceToHost);
        int min_idx = std::numeric_limits<int>::max();
        int max_idx = std::numeric_limits<int>::min();
        for (int v : host_indices) {
            min_idx = std::min(min_idx, v);
            max_idx = std::max(max_idx, v);
        }
        fprintf(stderr,
                "[MOE_PERMUTE_BWD_GATHER] layer=%d total=%d sample_min=%d sample_max=%d\n",
                layer_idx, total_tokens, min_idx, max_idx);
        moe_permute_gather_trace++;
    }

    // DEBUG: track weight integrity across permute backward for layer 2.
    static int moe_permute_bwd_w_trace = 0;
    bool pre_nan = false;
    float pre_min = 0.0f, pre_max = 0.0f;
    if (layer_idx == 2 && moe_permute_bwd_w_trace < 1 && mWeights.has("blocks[2].experts_gate_up")) {
        Tensor& w = mWeights.get("blocks[2].experts_gate_up");
        pre_nan = tensor_row_has_nan_or_inf(w, 122, &pre_min, &pre_max);
    }

    if (d_permuted.DType == ETensorDType::BF16) {
        auto shape_vec = [](const Tensor& t) {
            return std::vector<long>(t.Sizes.begin(), t.Sizes.begin() + t.Rank);
        };
        Tensor d_perm_f32 = mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(d_permuted), "moe_perm_d_f32");
        Tensor d_in_f32 = mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(d_input), "moe_perm_out_f32");
        convert_dtype(d_perm_f32.get<float>(), d_permuted.get<nv_bfloat16>(), d_permuted.nelem(), mRunState.MainStream);
        fill_zero(d_in_f32, mRunState.MainStream);
        moe_permute_backward(d_in_f32.get<float>(),
                             d_perm_f32.get<float>(),
                             gather_indices->get<int>(),
                             total_tokens, num_tokens, hidden_size, top_k,
                             mRunState.MainStream);
        convert_dtype(d_input.get<nv_bfloat16>(), d_in_f32.get<float>(), d_input.nelem(), mRunState.MainStream);
    } else {
        fill_zero(d_input, mRunState.MainStream);
        moe_permute_backward(d_input.get<float>(),
                             d_permuted.get<float>(),
                             gather_indices->get<int>(),
                             total_tokens, num_tokens, hidden_size, top_k,
                             mRunState.MainStream);
    }

    if (moe_permute_mag < 12 && layer_idx <= 2) {
        log_tensor_mag("MOE_PERMUTE_BWD_OUT", layer_idx, op.outputs[0].name, d_input, 4096);
        moe_permute_mag++;
    }

    static int moe_permute_trace_layer_out = 0;
    if (should_trace_moe_layer(layer_idx, moe_permute_trace_layer_out)) {
        log_tensor_mag_unbounded("MOE_PERMUTE_BWD_TRACE_OUT", layer_idx, op.outputs[0].name, d_input, 4096);
        const std::size_t out_total = static_cast<std::size_t>(d_input.nelem());
        if (out_total > 0) {
            log_tensor_sample_stats("MOE_PERMUTE_BWD_TRACE_OUT_MID", d_input, out_total / 2, 4096);
        }
    }

    // Post-check for NaNs in permute backward output.
    static int moe_permute_out_nan_logged = 0;
    if (moe_permute_out_nan_logged < 4) {
        long row_out = -1;
        float out_min = 0.0f, out_max = 0.0f;
        if (find_first_nan_row(d_input, &row_out, &out_min, &out_max)) {
            fprintf(stderr,
                    "[MOE_PERMUTE_BWD_NAN_OUT] layer=%d d_input=%s row=%ld min=%.6f max=%.6f\n",
                    layer_idx, op.outputs[0].name.c_str(), row_out, out_min, out_max);
            log_tensor_stats_ex("MOE_PERMUTE_BWD_NAN_OUT_STATS", layer_idx, op.outputs[0].name, d_input, 4096, true);
            log_tensor_stats_ex("MOE_PERMUTE_BWD_NAN_IN_STATS", layer_idx, op.inputs[0].name, d_permuted, 4096, true);
            // Dump gather indices bounds to catch OOB writes.
            std::vector<int> host_indices(total_tokens);
            cudaMemcpy(host_indices.data(), gather_indices->Data,
                       host_indices.size() * sizeof(int), cudaMemcpyDeviceToHost);
            int min_idx = std::numeric_limits<int>::max();
            int max_idx = std::numeric_limits<int>::min();
            int oob = 0;
            for (int v : host_indices) {
                min_idx = std::min(min_idx, v);
                max_idx = std::max(max_idx, v);
                if (v < 0 || v >= total_tokens) {
                    oob++;
                }
            }
            fprintf(stderr,
                    "[MOE_PERMUTE_BWD_NAN_OUT_GATHER] layer=%d total=%d min=%d max=%d oob=%d\n",
                    layer_idx, total_tokens, min_idx, max_idx, oob);
            moe_permute_out_nan_logged++;
        }
    }

    if (layer_idx == 2 && moe_permute_bwd_w_trace < 1 && mWeights.has("blocks[2].experts_gate_up")) {
        Tensor& w = mWeights.get("blocks[2].experts_gate_up");
        float post_min = 0.0f, post_max = 0.0f;
        const bool post_nan = tensor_row_has_nan_or_inf(w, 122, &post_min, &post_max);
        fprintf(stderr,
                "[MOE_PERMUTE_BWD_W] layer=%d pre_nan=%d pre_min=%.6f pre_max=%.6f post_nan=%d post_min=%.6f post_max=%.6f\n",
                layer_idx,
                pre_nan ? 1 : 0, pre_min, pre_max,
                post_nan ? 1 : 0, post_min, post_max);
        moe_permute_bwd_w_trace++;
    }

    mTensorMap[op.outputs[0].name] = d_input;
}


}  // namespace dsl
