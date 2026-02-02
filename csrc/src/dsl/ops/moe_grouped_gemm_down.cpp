#include "dsl/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "modules/lora/lora_config.h"
#include "modules/lora/lora_weights_manager.h"

namespace dsl {

void CompiledExecutor::dispatch_moe_grouped_gemm_down(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& weights = resolve_tensor(op.inputs[1]);  // Parameter name resolved by graph compiler
    Tensor& scatter_indices = resolve_tensor(op.inputs[2]);
    (void)scatter_indices;  // Used by kernel through expert_offsets
    const int num_tokens = static_cast<int>(mB * mT);
    int top_k = op.attrs.top_k;
    if (top_k <= 0 && num_tokens > 0 && inp.Rank == 2) {
        top_k = static_cast<int>(inp.Sizes[0] / num_tokens);
    }
    if (top_k <= 0) {
        top_k = 1;
    }

    const int num_experts = static_cast<int>(mConfig.NumExperts);
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
    // Use MoeIntermediateSize for MoE models (may differ from IntermediateSize)
    const int intermediate_size = (mConfig.MoeIntermediateSize > 0)
        ? static_cast<int>(mConfig.MoeIntermediateSize)
        : static_cast<int>(mConfig.IntermediateSize);
    const int weight_experts = (weights.Rank > 0) ? static_cast<int>(weights.Sizes[0]) : num_experts;
    int layer_idx_any = op.attrs.layer_idx;
    if (layer_idx_any < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        std::string field;
        parse_block_param(name, layer_idx_any, field);
    }
    // Get expert offsets from per-layer saved buffers when available.
    Tensor expert_offsets_view;
    Tensor* expert_offsets_ptr = nullptr;
    if (layer_idx_any >= 0) {
        const std::string key = "blocks[" + std::to_string(layer_idx_any) + "].moe_expert_offsets";
        auto it_saved = mMoESavedBuffers.find(key);
        if (it_saved != mMoESavedBuffers.end() && it_saved->second != nullptr) {
            expert_offsets_view.DType = ETensorDType::INT32;
            expert_offsets_view.Rank = 1;
            expert_offsets_view.Sizes[0] = static_cast<long>(mConfig.NumExperts + 1);
            expert_offsets_view.Data = static_cast<std::byte*>(it_saved->second);
            expert_offsets_ptr = &expert_offsets_view;
        }
    }
    if (!expert_offsets_ptr) {
        auto it = mTensorMap.find("moe_expert_offsets");
        if (it == mTensorMap.end()) {
            throw std::runtime_error("moe_grouped_gemm_down: expert_offsets not found");
        }
        expert_offsets_ptr = &it->second;
    }
    Tensor& expert_offsets = *expert_offsets_ptr;
    std::vector<int> host_offsets_local;
    const int* host_offsets_ptr = nullptr;
    if (num_experts > 0 && expert_offsets.Data) {
        host_offsets_local.resize(static_cast<std::size_t>(num_experts + 1), 0);
        CUDA_CHECK(cudaMemcpyAsync(host_offsets_local.data(),
                                   expert_offsets.get<int>(),
                                   static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        host_offsets_ptr = host_offsets_local.data();
    }

    if (host_offsets_ptr) {
        int bad = 0;
        int last = host_offsets_ptr[num_experts];
        for (int e = 1; e <= num_experts; ++e) {
            if (host_offsets_ptr[e] < host_offsets_ptr[e - 1]) {
                bad++;
            }
        }
        if (bad > 0 || last != static_cast<int>(inp.Sizes[0])) {
            fprintf(stderr,
                    "[MOE_OFFSETS_BAD] layer=%d total=%ld last=%d bad=%d num_experts=%d\n",
                    layer_idx_any,
                    static_cast<long>(inp.Sizes[0]),
                    last, bad, num_experts);
        }
    }

    MoeCompactInfo compact = host_offsets_ptr
        ? build_moe_compact_info_from_host(host_offsets_ptr,
                                           num_experts,
                                           weight_experts,
                                           layer_idx_any,
                                           "moe_grouped_gemm_down")
        : build_moe_compact_info(expert_offsets.get<int>(),
                                 num_experts,
                                 weight_experts,
                                 mRunState.MainStream,
                                 layer_idx_any,
                                 "moe_grouped_gemm_down");
    if (!host_offsets_ptr && !compact.host_offsets.empty()) {
        host_offsets_ptr = compact.host_offsets.data();
    }
    const int* active_ptr = compact.active_experts.empty() ? nullptr : compact.active_experts.data();
    const int num_active = compact.active_experts.empty() ? -1 : compact.num_active;
    const bool weight_is_compact = compact.weight_is_compact;

    // MoE output shape is dynamic: [total_tokens, hidden_size]
    // total_tokens = inp.Sizes[0] (permuted token count)
    const long total_tokens = inp.Sizes[0];
    std::vector<long> out_shape = {total_tokens, static_cast<long>(hidden_size)};
    Tensor out = mRunState.temp_alloc(inp.DType, out_shape);
    mTemps.push_back(out);

    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(out, mRunState.MainStream);
    } else if (inp.DType == ETensorDType::BF16) {
        moe_grouped_gemm_down(out.get<nv_bfloat16>(),
                              inp.get<nv_bfloat16>(),
                              weights.get<nv_bfloat16>(),
                              expert_offsets.get<int>(),
                              num_experts, hidden_size, intermediate_size,
                              mRunState.cublas_handle(), mRunState.MainStream,
                              host_offsets_ptr,
                              active_ptr,
                              weight_is_compact,
                              num_active);
    } else {
        moe_grouped_gemm_down(out.get<float>(),
                              inp.get<float>(),
                              weights.get<float>(),
                              expert_offsets.get<int>(),
                              num_experts, hidden_size, intermediate_size,
                              mRunState.cublas_handle(), mRunState.MainStream,
                              host_offsets_ptr,
                              active_ptr,
                              weight_is_compact,
                              num_active);
    }

    const bool moe_dot_trace = (std::getenv("SUROGATE_MOE_DOT_TRACE") != nullptr);
    static int moe_down_dot_trace = 0;
    if (moe_dot_trace && moe_down_dot_trace < 1) {
        const int target_layer = env_int("SUROGATE_MOE_DOT_LAYER", -1);
        const int target_pos = env_int("SUROGATE_MOE_DOT_POS", -1);
        if (target_pos >= 0 && target_pos < total_tokens &&
            (target_layer < 0 || layer_idx_any == target_layer)) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            std::vector<float> in_row;
            std::vector<float> out_row;
            const bool in_ok = copy_tensor_token_sample_as_f32(inp, target_pos, intermediate_size, in_row);
            const bool out_ok = copy_tensor_token_sample_as_f32(out, target_pos, hidden_size, out_row);
            int expert_id = -1;
            if (host_offsets_ptr) {
                for (int e = 0; e < num_experts; ++e) {
                    if (target_pos < host_offsets_ptr[e + 1]) {
                        expert_id = e;
                        break;
                    }
                }
            }
            int weight_idx = expert_id;
            if (weight_is_compact) {
                weight_idx = -1;
                if (active_ptr && num_active > 0) {
                    for (int i = 0; i < num_active; ++i) {
                        if (active_ptr[i] == expert_id) {
                            weight_idx = i;
                            break;
                        }
                    }
                }
            }
            int out_idx = env_int("SUROGATE_MOE_DOWN_OUT", -1);
            if (out_idx < 0 && out_ok && !out_row.empty()) {
                float max_abs = 0.0f;
                for (std::size_t i = 0; i < out_row.size(); ++i) {
                    const float av = std::fabs(out_row[i]);
                    if (av > max_abs) {
                        max_abs = av;
                        out_idx = static_cast<int>(i);
                    }
                }
            }
            fprintf(stderr,
                    "[MOE_DOWN_DOT_TRACE] layer=%d pos=%d expert=%d weight_idx=%d out_idx=%d in_ok=%d out_ok=%d\n",
                    layer_idx_any, target_pos, expert_id, weight_idx, out_idx,
                    in_ok ? 1 : 0, out_ok ? 1 : 0);
            if (in_ok && out_ok && weight_idx >= 0 && out_idx >= 0 && out_idx < hidden_size) {
                const std::size_t w_stride = static_cast<std::size_t>(hidden_size) *
                                             static_cast<std::size_t>(intermediate_size);
                const std::size_t w_offset = static_cast<std::size_t>(weight_idx) * w_stride +
                                             static_cast<std::size_t>(out_idx) * static_cast<std::size_t>(intermediate_size);
                std::vector<float> w_row;
                const bool w_ok = copy_tensor_sample_offset_as_f32(weights, w_offset,
                                                                  static_cast<std::size_t>(intermediate_size),
                                                                  w_row);
                if (w_ok) {
                    double dot = 0.0;
                    double in_l2 = 0.0;
                    double w_l2 = 0.0;
                    for (std::size_t i = 0; i < w_row.size() && i < in_row.size(); ++i) {
                        const float a = in_row[i];
                        const float b = w_row[i];
                        dot += static_cast<double>(a) * static_cast<double>(b);
                        in_l2 += static_cast<double>(a) * static_cast<double>(a);
                        w_l2 += static_cast<double>(b) * static_cast<double>(b);
                    }
                    float w_min = 0.0f, w_max = 0.0f, w_max_abs = 0.0f;
                    double w_sum_abs = 0.0;
                    bool w_has = false;
                    for (float v : w_row) {
                        if (!w_has) {
                            w_min = w_max = v;
                            w_has = true;
                        } else {
                            if (v < w_min) w_min = v;
                            if (v > w_max) w_max = v;
                        }
                        const float av = std::fabs(v);
                        w_sum_abs += static_cast<double>(av);
                        w_max_abs = std::max(w_max_abs, av);
                    }
                    const double w_mean_abs = w_row.empty() ? 0.0 : (w_sum_abs / static_cast<double>(w_row.size()));
                    const float out_val = out_row[static_cast<std::size_t>(out_idx)];
                    fprintf(stderr,
                            "[MOE_DOWN_DOT_VAL] layer=%d pos=%d out_idx=%d out_val=%.6e dot=%.6e in_l2=%.6e w_l2=%.6e\n",
                            layer_idx_any, target_pos, out_idx, static_cast<double>(out_val),
                            dot, std::sqrt(in_l2), std::sqrt(w_l2));
                    fprintf(stderr,
                            "[MOE_DOWN_W_ROW] layer=%d expert=%d out_idx=%d n=%zu min=%.6f max=%.6f max_abs=%.6f mean_abs=%.6f\n",
                            layer_idx_any, expert_id, out_idx, w_row.size(),
                            w_min, w_max, w_max_abs, w_mean_abs);
                } else {
                    fprintf(stderr,
                            "[MOE_DOWN_DOT_TRACE] layer=%d pos=%d out_idx=%d weight_row_copy_failed\n",
                            layer_idx_any, target_pos, out_idx);
                }
            }
            moe_down_dot_trace++;
        }
    }

    const bool moe_target_trace = (std::getenv("SUROGATE_MOE_DOT_TRACE") != nullptr);
    static int moe_target_down_trace = 0;
    if (moe_target_trace && moe_target_down_trace < 1) {
        const int target_layer = env_int("SUROGATE_MOE_DOT_LAYER", -1);
        const int target_pos = env_int("SUROGATE_MOE_DOT_POS", -1);
        if (target_pos >= 0 && (target_layer < 0 || layer_idx_any == target_layer)) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            log_tensor_row_stats("MOE_DOWN_IN_ROW_STATS", inp, target_pos);
            log_tensor_row_stats("MOE_DOWN_OUT_ROW_STATS", out, target_pos);
            moe_target_down_trace++;
        }
    }
    const bool moe_route_trace = (std::getenv("SUROGATE_MOE_ROUTE_TRACE") != nullptr);
    static int moe_route_down_trace = 0;
    if (moe_route_trace && moe_route_down_trace < 4) {
        const std::size_t total = static_cast<std::size_t>(out.nelem());
        const std::size_t sample = std::min<std::size_t>(4096, total);
        log_tensor_sample_stats("MOE_DOWN_OUT", out, 0, sample);
        moe_route_down_trace++;
    }

    static int moe_route_down_row_trace = 0;
    if (moe_route_trace && moe_route_down_row_trace < 2 &&
        scatter_indices.Data && scatter_indices.DType == ETensorDType::INT32 && top_k > 0) {
        const int token_idx = 0;
        std::vector<int> idxs(static_cast<std::size_t>(top_k), -1);
        const std::size_t idx_offset = static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(top_k);
        CUDA_CHECK(cudaMemcpy(idxs.data(),
                              static_cast<const std::byte*>(scatter_indices.Data) + idx_offset * sizeof(int),
                              static_cast<std::size_t>(top_k) * sizeof(int),
                              cudaMemcpyDeviceToHost));
        fprintf(stderr,
                "[MOE_DOWN_IDX] layer=%d token=%d idxs=(",
                layer_idx_any, token_idx);
        for (int k = 0; k < top_k; ++k) {
            fprintf(stderr, "%s%d", (k ? "," : ""), idxs[k]);
        }
        fprintf(stderr, ")\n");

        const std::size_t row_sample = std::min<std::size_t>(2048, static_cast<std::size_t>(intermediate_size));
        for (int k = 0; k < top_k; ++k) {
            const int expert_pos = idxs[k];
            if (expert_pos < 0 || expert_pos >= total_tokens) {
                continue;
            }
            int expert_id = -1;
            if (host_offsets_ptr) {
                for (int e = 0; e < num_experts; ++e) {
                    if (expert_pos < host_offsets_ptr[e + 1]) {
                        expert_id = e;
                        break;
                    }
                }
            }
            const std::size_t row_offset = static_cast<std::size_t>(expert_pos) *
                                           static_cast<std::size_t>(intermediate_size);
            std::vector<float> row_vals;
            if (copy_tensor_sample_offset_as_f32(inp, row_offset, row_sample, row_vals)) {
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
                        "[MOE_DOWN_IN_ROW] layer=%d token=%d k=%d pos=%d expert=%d max_abs=%.6f max_idx=%zu max_val=%.6f\n",
                        layer_idx_any, token_idx, k, expert_pos, expert_id, max_abs, max_idx, max_val);
                if (k == 0 && expert_id >= 0 && !weight_is_compact) {
                    const std::size_t w_stride = static_cast<std::size_t>(hidden_size) *
                                                 static_cast<std::size_t>(intermediate_size);
                    const std::size_t w_offset = w_stride * static_cast<std::size_t>(expert_id);
                    const std::size_t w_sample = std::min<std::size_t>(1024, w_stride);
                    std::vector<float> wvals;
                    if (copy_tensor_sample_offset_as_f32(weights, w_offset, w_sample, wvals)) {
                        float w_max = 0.0f;
                        for (float v : wvals) {
                            w_max = std::max(w_max, std::fabs(v));
                        }
                        fprintf(stderr,
                                "[MOE_DOWN_W_ROW] layer=%d expert=%d max_abs=%.6f\n",
                                layer_idx_any, expert_id, w_max);
                    }
                }
            }
        }
        moe_route_down_row_trace++;
    }
    const bool moe_lora_split_trace = (std::getenv("SUROGATE_MOE_LORA_SPLIT_TRACE") != nullptr);
    static int moe_lora_split_count = 0;
    if (moe_lora_split_trace && moe_lora_split_count < 4 &&
        mLoRAConfig && mLoRAWeights && layer_idx_any >= 0) {
        auto& lora_block = mLoRAWeights->get_block(layer_idx_any, mRunState.MainStream);
        if (lora_block.moe.use_grouped && lora_block.moe.grouped.has_any() &&
            mLoRAConfig->rank > 0) {
            const int rank = mLoRAConfig->rank;
            const float scaling = mLoRAConfig->scaling();
            const long total_tokens = inp.Sizes[0];
            const std::size_t sample = 4096;

            log_tensor_sample_stats("MOE_LORA_DOWN_BASE", out, 0, sample);

            auto compute_delta = [&](const modules::LoRAGroupedLayerWeights<Tensor>& layer,
                                     const char* tag,
                                     int out_features,
                                     int in_features) {
                if (!layer.has_value()) {
                    return;
                }
                if (layer.A.DType != inp.DType || layer.B.DType != inp.DType) {
                    fprintf(stderr,
                            "[MOE_LORA_SPLIT] layer=%d tag=%s dtype_mismatch inp=%s A=%s B=%s\n",
                            layer_idx_any,
                            tag ? tag : "<none>",
                            dtype_to_str(inp.DType),
                            dtype_to_str(layer.A.DType),
                            dtype_to_str(layer.B.DType));
                    return;
                }

                Tensor intermediate = mRunState.temp_alloc(inp.DType, {total_tokens, rank});
                Tensor delta = mRunState.temp_alloc(inp.DType, {total_tokens, out_features});
                mTemps.push_back(intermediate);
                mTemps.push_back(delta);

                if (inp.DType == ETensorDType::BF16) {
                    moe_grouped_gemm(intermediate.get<nv_bfloat16>(),
                                     inp.get<nv_bfloat16>(),
                                     layer.A.get<nv_bfloat16>(),
                                     expert_offsets.get<int>(),
                                     num_experts, rank, in_features,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr,
                                     1.0f, 0.0f, EMMTranspose::TN,
                                     active_ptr, /*weight_is_compact=*/false, num_active);
                    moe_grouped_gemm(delta.get<nv_bfloat16>(),
                                     intermediate.get<nv_bfloat16>(),
                                     layer.B.get<nv_bfloat16>(),
                                     expert_offsets.get<int>(),
                                     num_experts, out_features, rank,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr,
                                     scaling, 0.0f, EMMTranspose::TN,
                                     active_ptr, /*weight_is_compact=*/false, num_active);
                } else {
                    moe_grouped_gemm(intermediate.get<float>(),
                                     inp.get<float>(),
                                     layer.A.get<float>(),
                                     expert_offsets.get<int>(),
                                     num_experts, rank, in_features,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr,
                                     1.0f, 0.0f, EMMTranspose::TN,
                                     active_ptr, /*weight_is_compact=*/false, num_active);
                    moe_grouped_gemm(delta.get<float>(),
                                     intermediate.get<float>(),
                                     layer.B.get<float>(),
                                     expert_offsets.get<int>(),
                                     num_experts, out_features, rank,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr,
                                     scaling, 0.0f, EMMTranspose::TN,
                                     active_ptr, /*weight_is_compact=*/false, num_active);
                }

                log_tensor_sample_stats(tag, delta, 0, sample);
            };

            if (lora_block.moe.grouped.down.has_value()) {
                compute_delta(*lora_block.moe.grouped.down,
                              "MOE_LORA_DOWN_DELTA",
                              hidden_size,
                              intermediate_size);
            }
            moe_lora_split_count++;
        }
    }
    // DEBUG: check gate_up weights after down GEMM to see if they get corrupted.
    static int moe_w122_after_down_trace = 0;
    if (layer_idx_any == 2 && moe_w122_after_down_trace < 2) {
        const std::string wname = "blocks[" + std::to_string(layer_idx_any) + "].experts_gate_up";
        if (mWeights.has(wname)) {
            Tensor& gw = mWeights.get(wname);
            const int expert_id = 122;
            const std::size_t stride = static_cast<std::size_t>(2 * intermediate_size) *
                                       static_cast<std::size_t>(hidden_size);
            const std::size_t offset = stride * static_cast<std::size_t>(expert_id);
            const std::size_t sample = std::min<std::size_t>(stride, 1024);
            std::vector<float> wvals;
            if (copy_tensor_sample_offset_as_f32(gw, offset, sample, wvals)) {
                int nan = 0;
                float max_abs = 0.0f;
                for (float v : wvals) {
                    if (std::isnan(v) || std::isinf(v)) {
                        nan++;
                    } else {
                        max_abs = std::max(max_abs, std::fabs(v));
                    }
                }
                fprintf(stderr,
                        "[MOE_W122_GATE_UP_AFTER_DOWN] layer=%d expert=%d nan=%d max_abs=%.6f ptr=%p\n",
                        layer_idx_any, expert_id, nan, max_abs, static_cast<void*>(gw.Data));
            }
        }
        moe_w122_after_down_trace++;
    }

    // DEBUG: Compare expert down weights (expert 0 vs 1) for layer 0/top.
    static int moe_down_w_trace = 0;
    if (moe_down_w_trace < 4) {
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
            const std::size_t elem_size = get_dtype_size(weights.DType);
            const long stride = static_cast<long>(hidden_size) * static_cast<long>(intermediate_size);
            const std::size_t expected_elems = static_cast<std::size_t>(num_experts) * static_cast<std::size_t>(stride);
            std::vector<float> w0(4, 0.0f);
            std::vector<float> w1(4, 0.0f);
            (void)copy_tensor_sample_offset_as_f32(weights, 0, w0.size(), w0);
            (void)copy_tensor_sample_offset_as_f32(weights, static_cast<std::size_t>(stride), w1.size(), w1);
            fprintf(stderr,
                    "[MOE_W_DOWN] layer=%d w0=%.6f,%.6f,%.6f,%.6f w1=%.6f,%.6f,%.6f,%.6f\n",
                    layer_idx,
                    w0[0], w0[1], w0[2], w0[3],
                    w1[0], w1[1], w1[2], w1[3]);
            fprintf(stderr,
                    "[MOE_W_META_DOWN] layer=%d dtype=%s rank=%d nelem=%zu expected=%zu elem_size=%zu stride=%ld data=%p\n",
                    layer_idx,
                    dtype_to_str(weights.DType),
                    weights.Rank,
                    weights.nelem(),
                    expected_elems,
                    elem_size,
                    stride,
                    static_cast<void*>(weights.Data));
            for (int e = 0; e < std::min(num_experts, 4); ++e) {
                const std::size_t byte_offset = static_cast<std::size_t>(stride) * static_cast<std::size_t>(e) * elem_size;
                const std::byte* ptr = static_cast<const std::byte*>(weights.Data) + byte_offset;
                std::vector<float> we(4, 0.0f);
                (void)copy_tensor_sample_offset_as_f32(weights, static_cast<std::size_t>(stride) * e, we.size(), we);
                fprintf(stderr,
                        "[MOE_W_PTR_DOWN] layer=%d expert=%d ptr=%p vals=%.6f,%.6f,%.6f,%.6f\n",
                        layer_idx, e, static_cast<const void*>(ptr),
                        we[0], we[1], we[2], we[3]);
            }
            const std::size_t sample_count = 256;
            for (int e = 0; e < std::min(num_experts, 2); ++e) {
                std::vector<float> ws(sample_count, 0.0f);
                double sum = 0.0;
                double sum_abs = 0.0;
                if (copy_tensor_sample_offset_as_f32(weights, static_cast<std::size_t>(stride) * e, ws.size(), ws)) {
                    for (float v : ws) {
                        sum += static_cast<double>(v);
                        sum_abs += std::abs(static_cast<double>(v));
                    }
                    fprintf(stderr,
                            "[MOE_W_SUM_DOWN] layer=%d expert=%d sum=%.6e mean_abs=%.6e\n",
                            layer_idx, e,
                            sum, (sum_abs / static_cast<double>(ws.size())));
                }
            }
            fprintf(stderr,
                    "[MOE_GEMM_DOWN] layer=%d path=%s dtype=%s\n",
                    layer_idx,
                    (weights.DType == ETensorDType::BF16 ? "bf16" : "fp32"),
                    dtype_to_str(weights.DType));
            moe_down_w_trace++;
        }
    }

    // DEBUG: Trace forward expert_down magnitude for layer 0/top.
    static int moe_down_trace = 0;
    if (moe_down_trace < 12) {
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
            log_tensor_stats_ex("MOE_FWD_DOWN", layer_idx, op.outputs[0].name, out, 4096, true);
            moe_down_trace++;
        }
    }

    // DEBUG: Find max |expert_down| over a larger sample to catch outliers.
    static int moe_down_max_trace = 0;
    if (moe_down_max_trace < 4) {
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
            const std::size_t sample = std::min<std::size_t>(total, 1u << 20);  // up to ~1M elems
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
                fprintf(stderr,
                        "[MOE_FWD_DOWN_MAX] layer=%d sample_max=%.6f token=%d dim=%d sample=%zu total=%zu\n",
                        layer_idx, vals[max_idx], token_idx, dim, sample, total);
            }
            moe_down_max_trace++;
        }
    }

    // DEBUG: if any expert_down rows for token 3 have NaNs, dump row stats and weights.
    static int moe_down_row_trace = 0;
    if (moe_down_row_trace < 8) {
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
            auto find_expert = [&](int idx) {
                for (int e = 0; e < num_experts; ++e) {
                    if (idx >= h_offsets[e] && idx < h_offsets[e + 1]) {
                        return e;
                    }
                }
                return -1;
            };

            const int token_idx = 3;
            std::vector<int> indices(top_k, -1);
            const std::string prefix = "blocks[" + std::to_string(layer_idx) + "].";
            Tensor* scatter_t = nullptr;
            auto sit = mTensorMap.find(prefix + "scatter_indices");
            if (sit != mTensorMap.end()) {
                scatter_t = &sit->second;
            } else {
                scatter_t = &scatter_indices;
            }
            if (scatter_t && scatter_t->Data && scatter_t->DType == ETensorDType::INT32) {
                const std::size_t w_offset = static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(top_k);
                CUDA_CHECK(cudaMemcpy(indices.data(),
                                      static_cast<const std::byte*>(scatter_t->Data) + w_offset * sizeof(int),
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
                    const int expert_id = find_expert(idx);
                    fprintf(stderr,
                            "[MOE_DOWN_ROW] layer=%d token=%d k=%d idx=%d expert=%d in_nan=%d in_min=%.6f in_max=%.6f "
                            "out_nan=%d out_min=%.6f out_max=%.6f\n",
                            layer_idx, token_idx, k, idx, expert_id,
                            in_nan ? 1 : 0, in_min, in_max,
                            out_nan ? 1 : 0, out_min, out_max);
                    if (expert_id >= 0) {
                        const long stride = static_cast<long>(hidden_size) * static_cast<long>(intermediate_size);
                        std::vector<float> wvals(4, 0.0f);
                        (void)copy_tensor_sample_offset_as_f32(weights,
                                                               static_cast<std::size_t>(stride) * expert_id,
                                                               wvals.size(),
                                                               wvals);
                        fprintf(stderr,
                                "[MOE_DOWN_W_SAMPLE] layer=%d expert=%d w=%.6f,%.6f,%.6f,%.6f\n",
                                layer_idx, expert_id,
                                wvals[0], wvals[1], wvals[2], wvals[3]);
                    }
                }
            }
            moe_down_row_trace++;
        }
    }

    // DEBUG: show expert_down output pointer for early layers.
    static int moe_down_ptr_trace = 0;
    if (moe_down_ptr_trace < 8) {
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
            fprintf(stderr,
                    "[MOE_DOWN_OUT] layer=%d name=%s ptr=%p dtype=%s shape=%s\n",
                    layer_idx,
                    op.outputs[0].name.c_str(),
                    static_cast<void*>(out.Data),
                    dtype_to_str(out.DType),
                    tensor_shape_str(out).c_str());
            moe_down_ptr_trace++;
        }
    }

    mTensorMap[op.outputs[0].name] = out;
}

void CompiledExecutor::dispatch_moe_grouped_gemm_down_backward(const CompiledOp& op) {
    Tensor& d_output = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& weights = resolve_tensor(op.inputs[2]);
    Tensor& d_input = ensure_output_tensor(op.outputs[0]);
    (void)inp;  // Used by weight gradient computation if enabled
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

    static int moe_down_trace_layer = 0;
    if (should_trace_moe_layer(layer_idx, moe_down_trace_layer)) {
        log_tensor_mag_unbounded("MOE_DOWN_BWD_TRACE_DOUT", layer_idx, op.inputs[0].name, d_output, 4096);
        log_tensor_mag_unbounded("MOE_DOWN_BWD_TRACE_INP", layer_idx, op.inputs[1].name, inp, 4096);
        log_tensor_mag_unbounded("MOE_DOWN_BWD_TRACE_W", layer_idx, op.inputs[2].name, weights, 4096);
    }

    // One-time NaN watchdog to pinpoint MoE down backward issues.
    static bool moe_down_nan_logged = false;
    if (!moe_down_nan_logged) {
        const bool nan_dout = tensor_sample_has_nan_or_inf(d_output, 3);
        const bool nan_inp = tensor_sample_has_nan_or_inf(inp, 3);
        if (nan_dout || nan_inp) {
            fprintf(stderr,
                    "[MOE_DOWN_BWD_NAN] layer=%d nan_dout=%d nan_inp=%d d_out=%s inp=%s\n",
                    layer_idx,
                    nan_dout ? 1 : 0,
                    nan_inp ? 1 : 0,
                    op.inputs[0].name.c_str(),
                    op.inputs[1].name.c_str());
            log_tensor_stats_ex("MOE_DOWN_BWD_NAN_DOUT", layer_idx, op.inputs[0].name, d_output, 4096, true);
            log_tensor_stats_ex("MOE_DOWN_BWD_NAN_INP", layer_idx, op.inputs[1].name, inp, 4096, true);
            moe_down_nan_logged = true;
        }
    }
    static int moe_down_mag = 0;
    if (moe_down_mag < 8 && layer_idx <= 2) {
        log_tensor_mag("MOE_DOWN_BWD_DOUT", layer_idx, op.inputs[0].name, d_output, 4096);
        log_tensor_mag("MOE_DOWN_BWD_INP", layer_idx, op.inputs[1].name, inp, 4096);
        moe_down_mag++;
    }
    static std::unordered_set<int> moe_down_grad_layers;
    if (std::getenv("SUROGATE_MOE_GRAD_TRACE") != nullptr) {
        if (moe_down_grad_layers.insert(layer_idx).second) {
            log_tensor_mag_unbounded("MOE_DOWN_BWD_GRAD", layer_idx, op.inputs[0].name, d_output, 4096);
        }
    }

    // DEBUG: Trace MoE down backward inputs to locate explosion source.
    static int moe_down_bwd_trace = 0;
    if (moe_down_bwd_trace < 12) {
        if (layer_idx < 0 || layer_idx < 4 || layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
            log_tensor_stats_ex("MOE_DOWN_BWD_DOUT", layer_idx, op.inputs[0].name, d_output, 4096, true);
            log_tensor_stats_ex("MOE_DOWN_BWD_W", layer_idx, op.inputs[2].name, weights, 4096, true);
            moe_down_bwd_trace++;
        }
    }
    // Focused trace for layer 0 regardless of the global trace budget.
    static int moe_down_bwd_l0_trace = 0;
    if (moe_down_bwd_l0_trace < 4) {
        int layer_idx_l0 = op.attrs.layer_idx;
        std::string field_l0;
        if (layer_idx_l0 < 0 && !op.inputs.empty()) {
            std::string_view name = op.inputs[0].name;
            if (name.rfind("d_", 0) == 0) {
                name.remove_prefix(2);
            }
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx_l0, field_l0);
        }
        if (layer_idx_l0 == 0) {
            log_tensor_stats_ex("MOE_DOWN_BWD_L0_DOUT", layer_idx_l0, op.inputs[0].name, d_output, 4096, true);
            log_tensor_stats_ex("MOE_DOWN_BWD_L0_W", layer_idx_l0, op.inputs[2].name, weights, 4096, true);
            moe_down_bwd_l0_trace++;
        }
    }

    // Use per-layer expert_offsets when available; fall back to global buffer.
    const int* expert_offsets_ptr = nullptr;
    Tensor expert_offsets_view;
    if (layer_idx >= 0) {
        const std::string key = "blocks[" + std::to_string(layer_idx) + "].moe_expert_offsets";
        auto it = mMoESavedBuffers.find(key);
        if (it != mMoESavedBuffers.end() && it->second != nullptr) {
            expert_offsets_view.DType = ETensorDType::INT32;
            expert_offsets_view.Rank = 1;
            expert_offsets_view.Sizes[0] = static_cast<long>(mConfig.NumExperts + 1);
            expert_offsets_view.Data = static_cast<std::byte*>(it->second);
            expert_offsets_ptr = expert_offsets_view.get<int>();
        }
    }
    if (!expert_offsets_ptr) {
        if (mMoEExpertOffsetsGPU == nullptr) {
            throw std::runtime_error("moe_grouped_gemm_down_backward: mMoEExpertOffsetsGPU not allocated");
        }
        expert_offsets_ptr = static_cast<const int*>(mMoEExpertOffsetsGPU);
    }

    const int num_experts = static_cast<int>(mConfig.NumExperts);
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
    // Use MoeIntermediateSize for MoE models (may differ from IntermediateSize)
    const int intermediate_size = (mConfig.MoeIntermediateSize > 0)
        ? static_cast<int>(mConfig.MoeIntermediateSize)
        : static_cast<int>(mConfig.IntermediateSize);

    const int weight_experts = static_cast<int>(weights.Sizes[0]);
    MoeCompactInfo compact = build_moe_compact_info(expert_offsets_ptr,
                                                    num_experts,
                                                    weight_experts,
                                                    mRunState.MainStream,
                                                    layer_idx,
                                                    "moe_grouped_gemm_down_backward");
    const bool weight_is_compact = compact.weight_is_compact;
    const int* host_offsets_ptr = compact.host_offsets.empty() ? nullptr : compact.host_offsets.data();
    const int* active_ptr = compact.active_experts.empty() ? nullptr : compact.active_experts.data();
    const int num_active = compact.num_active;

    // Refresh MoE experts for this layer (selective dequant) before using weights in backward.
    if (!compact.host_offsets.empty()) {
        (void)refresh_moe_experts_if_needed(layer_idx,
                                            compact.host_offsets.data(),
                                            num_experts,
                                            mWeights,
                                            mRunState.MainStream);
    } else {
        std::vector<int> host_offsets_fallback(static_cast<std::size_t>(num_experts + 1), 0);
        CUDA_CHECK(cudaMemcpyAsync(host_offsets_fallback.data(),
                                   expert_offsets_ptr,
                                   static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        (void)refresh_moe_experts_if_needed(layer_idx,
                                            host_offsets_fallback.data(),
                                            num_experts,
                                            mWeights,
                                            mRunState.MainStream);
    }

    // DEBUG: check if weights change across down backward for layer 2.
    static int moe_down_bwd_w_trace = 0;
    bool pre_nan = false;
    float pre_min = 0.0f, pre_max = 0.0f;
    if (layer_idx == 2 && moe_down_bwd_w_trace < 1) {
        pre_nan = tensor_row_has_nan_or_inf(weights, 122, &pre_min, &pre_max);
    }

    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(d_input, mRunState.MainStream);
    } else if (d_output.DType == ETensorDType::BF16) {
        moe_grouped_gemm_down_backward(d_input.get<nv_bfloat16>(),
                                       d_output.get<nv_bfloat16>(),
                                       weights.get<nv_bfloat16>(),
                                       expert_offsets_ptr,
                                       num_experts, hidden_size, intermediate_size,
                                       mRunState.cublas_handle(), mRunState.MainStream,
                                       host_offsets_ptr,
                                       active_ptr,
                                       weight_is_compact,
                                       num_active);
    } else {
        moe_grouped_gemm_down_backward(d_input.get<float>(),
                                       d_output.get<float>(),
                                       weights.get<float>(),
                                       expert_offsets_ptr,
                                       num_experts, hidden_size, intermediate_size,
                                       mRunState.cublas_handle(), mRunState.MainStream,
                                       host_offsets_ptr,
                                       active_ptr,
                                       weight_is_compact,
                                       num_active);
    }

    static int moe_down_trace_layer_out = 0;
    if (should_trace_moe_layer(layer_idx, moe_down_trace_layer_out)) {
        log_tensor_mag_unbounded("MOE_DOWN_BWD_TRACE_DIN", layer_idx, op.outputs[0].name, d_input, 4096);
    }

    if (layer_idx == 2 && moe_down_bwd_w_trace < 1) {
        float post_min = 0.0f, post_max = 0.0f;
        const bool post_nan = tensor_row_has_nan_or_inf(weights, 122, &post_min, &post_max);
        fprintf(stderr,
                "[MOE_DOWN_BWD_W] layer=%d pre_nan=%d pre_min=%.6f pre_max=%.6f post_nan=%d post_min=%.6f post_max=%.6f\n",
                layer_idx,
                pre_nan ? 1 : 0, pre_min, pre_max,
                post_nan ? 1 : 0, post_min, post_max);
        moe_down_bwd_w_trace++;
    }

    // Check the output of down-backward for layer 0 (feeds SwiGLU backward).
    static int moe_down_bwd_l0_out_trace = 0;
    if (moe_down_bwd_l0_out_trace < 4) {
        int layer_idx_l0 = op.attrs.layer_idx;
        std::string field_l0;
        if (layer_idx_l0 < 0 && !op.inputs.empty()) {
            std::string_view name = op.inputs[0].name;
            if (name.rfind("d_", 0) == 0) {
                name.remove_prefix(2);
            }
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx_l0, field_l0);
        }
        if (layer_idx_l0 == 0) {
            log_tensor_stats_ex("MOE_DOWN_BWD_L0_DIN", layer_idx_l0, op.outputs[0].name, d_input, 4096, true);
            moe_down_bwd_l0_out_trace++;
        }
    }

    mTensorMap[op.outputs[0].name] = d_input;

    // Weight gradient computation would go here if needed (for fine-tuning experts)
}


}  // namespace dsl
