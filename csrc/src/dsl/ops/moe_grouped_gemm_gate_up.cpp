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

void CompiledExecutor::dispatch_moe_grouped_gemm_gate_up(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& weights = resolve_tensor(op.inputs[1]);  // Parameter name resolved by graph compiler
    Tensor& scatter_indices = resolve_tensor(op.inputs[2]);
    const int num_tokens = static_cast<int>(mB * mT);
    int top_k = op.attrs.top_k;
    if (top_k <= 0 && num_tokens > 0 && inp.Rank == 2) {
        top_k = static_cast<int>(inp.Sizes[0] / num_tokens);
    }
    if (top_k <= 0) {
        top_k = 1;
    }

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
            throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets not found");
        }
        expert_offsets_ptr = &it->second;
    }
    Tensor& expert_offsets = *expert_offsets_ptr;
    if (expert_offsets.DType != ETensorDType::INT32) {
        throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets dtype is not INT32");
    }
    if (!expert_offsets.Data) {
        throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets has null data");
    }
    auto ensure_device_ptr = [&](const Tensor& t, const char* name) {
        if (!t.Data) {
            return;
        }
        cudaPointerAttributes attr{};
        cudaError_t err = cudaPointerGetAttributes(&attr, t.Data);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "[MOE_PTR_ATTR_ERR] name=%s ptr=%p err=%s\n",
                    name ? name : "<unnamed>",
                    static_cast<void*>(t.Data),
                    cudaGetErrorString(err));
            cudaGetLastError();
            throw std::runtime_error("moe_grouped_gemm_gate_up: pointer attributes unavailable");
        }
#if CUDART_VERSION >= 10000
        const int mem_type = static_cast<int>(attr.type);
#else
        const int mem_type = static_cast<int>(attr.memoryType);
#endif
        if (mem_type == cudaMemoryTypeHost) {
            fprintf(stderr,
                    "[MOE_PTR_HOST] name=%s ptr=%p\n",
                    name ? name : "<unnamed>",
                    static_cast<void*>(t.Data));
            throw std::runtime_error("moe_grouped_gemm_gate_up: pointer on host memory");
        }
    };

    const int num_experts = static_cast<int>(mConfig.NumExperts);
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
    // Use MoeIntermediateSize for MoE models (may differ from IntermediateSize)
    const int intermediate_size = (mConfig.MoeIntermediateSize > 0)
        ? static_cast<int>(mConfig.MoeIntermediateSize)
        : static_cast<int>(mConfig.IntermediateSize);
    const int weight_experts = (weights.Rank > 0) ? static_cast<int>(weights.Sizes[0]) : num_experts;
    log_moe_gate_up_weight_sample("PRE_MOE_DOWN", layer_idx_any, mMicroStep, mWeights, mConfig);
    log_moe_gate_up_weight_sample("PRE_MOE_GATE_UP", layer_idx_any, mMicroStep, mWeights, mConfig);
    const bool offsets_owned = mRunState.Stack.owns(expert_offsets.Data);
    if (offsets_owned && !mRunState.Stack.is_live(expert_offsets.Data)) {
        throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets pointer is not live");
    }
    static int moe_offsets_ptr_trace = 0;
    if (moe_offsets_ptr_trace < 8) {
        log_cuda_ptr_attr("MOE_OFFSETS_PTR", expert_offsets.Data, layer_idx_any, "moe_expert_offsets");
        log_cuda_ptr_attr("MOE_SCATTER_PTR", scatter_indices.Data, layer_idx_any, "moe_scatter_indices");
        moe_offsets_ptr_trace++;
    }
    // DEBUG: sample expert 122 weights to catch NaNs early.
    static int moe_w122_trace = 0;
    if (moe_w122_trace < 4 && weight_experts > 122) {
        const int expert_id = 122;
        const std::size_t stride = static_cast<std::size_t>(2 * intermediate_size) *
                                   static_cast<std::size_t>(hidden_size);
        const std::size_t offset = stride * static_cast<std::size_t>(expert_id);
        const std::size_t sample = std::min<std::size_t>(stride, 1024);
        std::vector<float> wvals;
        if (copy_tensor_sample_offset_as_f32(weights, offset, sample, wvals)) {
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
                    "[MOE_W122_GATE_UP] layer=%d expert=%d nan=%d max_abs=%.6f dtype=%s\n",
                    layer_idx_any, expert_id, nan, max_abs, dtype_to_str(weights.DType));
            fprintf(stderr,
                    "[MOE_W122_PTR] layer=%d ptr=%p stack_owned=%d\n",
                    layer_idx_any,
                    static_cast<void*>(weights.Data),
                    mRunState.Stack.owns(weights.Data) ? 1 : 0);
        }
        moe_w122_trace++;
    }
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
        int oob = 0;
        const int total_tokens = static_cast<int>(inp.Sizes[0]);
        for (int e = 1; e <= num_experts; ++e) {
            if (host_offsets_ptr[e] < host_offsets_ptr[e - 1]) {
                bad++;
            }
            if (host_offsets_ptr[e] < 0 || host_offsets_ptr[e] > total_tokens) {
                oob++;
            }
        }
        if (bad > 0 || oob > 0 || last != total_tokens) {
            fprintf(stderr,
                    "[MOE_OFFSETS_BAD] layer=%d total=%d last=%d bad=%d oob=%d num_experts=%d\n",
                    layer_idx_any,
                    total_tokens,
                    last, bad, oob, num_experts);
            throw std::runtime_error("moe_grouped_gemm_gate_up: expert_offsets invalid");
        }
    }

    // Refresh MoE expert weights for this layer using the current routing offsets.
    if (host_offsets_ptr && layer_idx_any >= 0) {
        (void)refresh_moe_experts_if_needed(layer_idx_any,
                                            host_offsets_ptr,
                                            num_experts,
                                            mWeights,
                                            mRunState.MainStream);
    }

    MoeCompactInfo compact = host_offsets_ptr
        ? build_moe_compact_info_from_host(host_offsets_ptr,
                                           num_experts,
                                           weight_experts,
                                           layer_idx_any,
                                           "moe_grouped_gemm_gate_up")
        : build_moe_compact_info(expert_offsets.get<int>(),
                                 num_experts,
                                 weight_experts,
                                 mRunState.MainStream,
                                 layer_idx_any,
                                 "moe_grouped_gemm_gate_up");
    if (!host_offsets_ptr && !compact.host_offsets.empty()) {
        host_offsets_ptr = compact.host_offsets.data();
    }
    const int* active_ptr = compact.active_experts.empty() ? nullptr : compact.active_experts.data();
    const int num_active = compact.active_experts.empty() ? -1 : compact.num_active;
    const bool weight_is_compact = compact.weight_is_compact;

    // MoE output shape is dynamic: [total_tokens, 2 * intermediate_size]
    // total_tokens = inp.Sizes[0] (permuted token count)
    // Allocate output with correct runtime shape
    const long total_tokens = inp.Sizes[0];
    const long gate_up_dim = 2 * intermediate_size;
    std::vector<long> out_shape = {total_tokens, gate_up_dim};
    Tensor out = mRunState.temp_alloc(inp.DType, out_shape);
    mTemps.push_back(out);
    ensure_device_ptr(inp, "moe_input");
    ensure_device_ptr(weights, "moe_gate_up_weights");
    ensure_device_ptr(out, "moe_gate_up_out");
    ensure_device_ptr(expert_offsets, "moe_expert_offsets");

    if (host_offsets_ptr) {
        const std::size_t in_elems = static_cast<std::size_t>(inp.nelem());
        const std::size_t out_elems = static_cast<std::size_t>(out.nelem());
        const std::size_t w_elems = static_cast<std::size_t>(weights.nelem());
        const std::size_t in_stride = static_cast<std::size_t>(hidden_size);
        const std::size_t out_stride = static_cast<std::size_t>(gate_up_dim);
        const std::size_t w_stride = static_cast<std::size_t>(gate_up_dim) * static_cast<std::size_t>(hidden_size);
        const int n_active = (num_active > 0) ? num_active : num_experts;
        for (int e = 0; e < n_active; ++e) {
            const int global_idx = active_ptr ? active_ptr[e] : e;
            if (global_idx < 0 || global_idx >= num_experts) {
                fprintf(stderr,
                        "[MOE_OFFSETS_OOR] layer=%d global_idx=%d num_experts=%d\n",
                        layer_idx_any, global_idx, num_experts);
                throw std::runtime_error("moe_grouped_gemm_gate_up: active expert index out of range");
            }
            const int start = host_offsets_ptr[global_idx];
            const int end = host_offsets_ptr[global_idx + 1];
            const int tokens_e = end - start;
            if (tokens_e <= 0) {
                continue;
            }
            const std::size_t in_offset = static_cast<std::size_t>(start) * in_stride;
            const std::size_t out_offset = static_cast<std::size_t>(start) * out_stride;
            if (in_offset + static_cast<std::size_t>(tokens_e) * in_stride > in_elems) {
                fprintf(stderr,
                        "[MOE_IN_OOR] layer=%d expert=%d start=%d end=%d in_elems=%zu\n",
                        layer_idx_any, global_idx, start, end, in_elems);
                throw std::runtime_error("moe_grouped_gemm_gate_up: input pointer out of range");
            }
            if (out_offset + static_cast<std::size_t>(tokens_e) * out_stride > out_elems) {
                fprintf(stderr,
                        "[MOE_OUT_OOR] layer=%d expert=%d start=%d end=%d out_elems=%zu\n",
                        layer_idx_any, global_idx, start, end, out_elems);
                throw std::runtime_error("moe_grouped_gemm_gate_up: output pointer out of range");
            }
            const int weight_idx = weight_is_compact ? e : global_idx;
            if (weight_idx < 0 || weight_idx >= weight_experts) {
                fprintf(stderr,
                        "[MOE_W_OOR] layer=%d expert=%d weight_idx=%d weight_experts=%d compact=%d\n",
                        layer_idx_any, global_idx, weight_idx, weight_experts, weight_is_compact ? 1 : 0);
                throw std::runtime_error("moe_grouped_gemm_gate_up: weight index out of range");
            }
            const std::size_t w_offset = static_cast<std::size_t>(weight_idx) * w_stride;
            if (w_offset + w_stride > w_elems) {
                fprintf(stderr,
                        "[MOE_W_RANGE_OOR] layer=%d expert=%d weight_idx=%d w_elems=%zu\n",
                        layer_idx_any, global_idx, weight_idx, w_elems);
                throw std::runtime_error("moe_grouped_gemm_gate_up: weight pointer out of range");
            }
        }
    }
    // Optional debug: zero output to detect unwritten rows.
    static int moe_zero_gate_up = -1;
    if (moe_zero_gate_up < 0) {
        moe_zero_gate_up = (std::getenv("SUROGATE_MOE_ZERO_OUT") != nullptr) ? 1 : 0;
    }
    if (moe_zero_gate_up) {
        fill_zero(out, mRunState.MainStream);
    }

    // Use weights dtype to determine compute precision (QLoRA may return FP32 dequantized weights)
    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(out, mRunState.MainStream);
    } else if (weights.DType == ETensorDType::BF16) {
        moe_grouped_gemm_gate_up(out.get<nv_bfloat16>(),
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
        moe_grouped_gemm_gate_up(out.get<float>(),
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

    const bool moe_target_trace = (std::getenv("SUROGATE_MOE_DOT_TRACE") != nullptr);
    static int moe_target_gate_up_trace = 0;
    if (moe_target_trace && moe_target_gate_up_trace < 1) {
        const int target_layer = env_int("SUROGATE_MOE_DOT_LAYER", -1);
        const int target_pos = env_int("SUROGATE_MOE_DOT_POS", -1);
        if (target_pos >= 0 && (target_layer < 0 || layer_idx_any == target_layer)) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            log_tensor_row_stats("MOE_GATE_UP_IN_ROW", inp, target_pos);
            log_tensor_row_stats("MOE_GATE_UP_OUT_ROW_STATS", out, target_pos);
            moe_target_gate_up_trace++;
        }
    }

    const bool moe_dot_trace = (std::getenv("SUROGATE_MOE_DOT_TRACE") != nullptr);
    static int moe_dot_trace_count = 0;
    if (moe_dot_trace && moe_dot_trace_count < 1) {
        const int target_layer = env_int("SUROGATE_MOE_DOT_LAYER", 1);
        if (target_layer < 0 || layer_idx_any == target_layer) {
            const int target_pos = env_int("SUROGATE_MOE_DOT_POS", 711);
            const int target_expert_env = env_int("SUROGATE_MOE_DOT_EXPERT", 68);
            int target_expert = (target_expert_env >= 0) ? target_expert_env : -1;
            int target_out_idx = env_int("SUROGATE_MOE_DOT_OUT", -1);
            if (target_pos >= 0 && target_pos < total_tokens) {
                if (!host_offsets_ptr) {
                    host_offsets_local.resize(static_cast<std::size_t>(num_experts + 1), 0);
                    CUDA_CHECK(cudaMemcpy(host_offsets_local.data(),
                                          expert_offsets.get<int>(),
                                          static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                                          cudaMemcpyDeviceToHost));
                    host_offsets_ptr = host_offsets_local.data();
                }
                int actual_expert = -1;
                if (host_offsets_ptr) {
                    for (int e = 0; e < num_experts; ++e) {
                        if (target_pos < host_offsets_ptr[e + 1]) {
                            actual_expert = e;
                            break;
                        }
                    }
                }
                if (target_expert < 0) {
                    target_expert = actual_expert;
                }
                if (actual_expert >= 0 && target_expert >= 0 && actual_expert != target_expert) {
                    fprintf(stderr,
                            "[MOE_GATE_UP_DOT_TRACE] layer=%d pos=%d expert_mismatch actual=%d target=%d\n",
                            layer_idx_any, target_pos, actual_expert, target_expert);
                }
                int weight_idx = target_expert;
                if (weight_is_compact) {
                    weight_idx = -1;
                    if (active_ptr && num_active > 0) {
                        for (int i = 0; i < num_active; ++i) {
                            if (active_ptr[i] == target_expert) {
                                weight_idx = i;
                                break;
                            }
                        }
                    }
                }
                if (target_expert >= 0 && weight_idx >= 0 && weight_idx < weight_experts) {
                    CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                    std::vector<float> in_row;
                    std::vector<float> out_row;
                    const std::size_t in_count = static_cast<std::size_t>(hidden_size);
                    const std::size_t out_count = static_cast<std::size_t>(gate_up_dim);
                    const bool in_ok = copy_tensor_token_sample_as_f32(inp, target_pos, in_count, in_row);
                    const bool out_ok = copy_tensor_token_sample_as_f32(out, target_pos, out_count, out_row);
                    float out_val = 0.0f;
                    std::size_t out_idx = 0;
                    if (out_ok && !out_row.empty()) {
                        if (target_out_idx < 0 || target_out_idx >= static_cast<int>(out_row.size())) {
                            float max_abs = 0.0f;
                            for (std::size_t i = 0; i < out_row.size(); ++i) {
                                const float av = std::fabs(out_row[i]);
                                if (av > max_abs) {
                                    max_abs = av;
                                    out_idx = i;
                                }
                            }
                        } else {
                            out_idx = static_cast<std::size_t>(target_out_idx);
                        }
                        out_val = out_row[out_idx];
                    }

                    const std::size_t w_stride = static_cast<std::size_t>(gate_up_dim) *
                                                 static_cast<std::size_t>(hidden_size);
                    const std::size_t w_offset = static_cast<std::size_t>(weight_idx) * w_stride +
                                                 out_idx * static_cast<std::size_t>(hidden_size);
                    std::vector<float> w_row;
                    const bool w_ok = copy_tensor_sample_offset_as_f32(weights, w_offset, in_count, w_row);
                    if (in_ok && w_ok) {
                        auto log_weight_row = [&](const char* tag, std::size_t row) {
                            std::vector<float> row_vals;
                            const std::size_t row_offset = static_cast<std::size_t>(weight_idx) * w_stride +
                                                           row * static_cast<std::size_t>(hidden_size);
                            if (!copy_tensor_sample_offset_as_f32(weights, row_offset, in_count, row_vals)) {
                                fprintf(stderr,
                                        "[%s] layer=%d expert=%d row=%zu copy_failed\n",
                                        tag, layer_idx_any, target_expert, row);
                                return;
                            }
                            std::size_t nan = 0;
                            std::size_t inf = 0;
                            float min_val = 0.0f;
                            float max_val = 0.0f;
                            float max_abs = 0.0f;
                            double sum_abs = 0.0;
                            bool has_finite = false;
                            for (float v : row_vals) {
                                if (std::isnan(v)) {
                                    nan++;
                                    continue;
                                }
                                if (std::isinf(v)) {
                                    inf++;
                                    continue;
                                }
                                if (!has_finite) {
                                    min_val = v;
                                    max_val = v;
                                    has_finite = true;
                                } else {
                                    if (v < min_val) min_val = v;
                                    if (v > max_val) max_val = v;
                                }
                                const float av = std::fabs(v);
                                sum_abs += static_cast<double>(av);
                                if (av > max_abs) {
                                    max_abs = av;
                                }
                            }
                            const double mean_abs = row_vals.empty() ? 0.0 : (sum_abs / static_cast<double>(row_vals.size()));
                            fprintf(stderr,
                                    "[%s] layer=%d expert=%d row=%zu n=%zu nan=%zu inf=%zu min=%.6f max=%.6f max_abs=%.6f mean_abs=%.6f\n",
                                    tag, layer_idx_any, target_expert, row, row_vals.size(),
                                    nan, inf, min_val, max_val, max_abs, mean_abs);
                        };

                        double dot = 0.0;
                        double in_l2 = 0.0;
                        double w_l2 = 0.0;
                        std::size_t in_nan = 0;
                        std::size_t w_nan = 0;
                        std::size_t in_inf = 0;
                        std::size_t w_inf = 0;
                        const int top_k = 8;
                        std::vector<double> top_abs(top_k, -1.0);
                        std::vector<double> top_val(top_k, 0.0);
                        std::vector<std::size_t> top_idx(top_k, 0);
                        for (std::size_t i = 0; i < in_row.size(); ++i) {
                            const float a = in_row[i];
                            const float b = w_row[i];
                            if (std::isnan(a)) in_nan++;
                            if (std::isnan(b)) w_nan++;
                            if (std::isinf(a)) in_inf++;
                            if (std::isinf(b)) w_inf++;
                            if (std::isfinite(a)) {
                                in_l2 += static_cast<double>(a) * static_cast<double>(a);
                            }
                            if (std::isfinite(b)) {
                                w_l2 += static_cast<double>(b) * static_cast<double>(b);
                            }
                            if (!std::isfinite(a) || !std::isfinite(b)) {
                                continue;
                            }
                            const double contrib = static_cast<double>(a) * static_cast<double>(b);
                            dot += contrib;
                            const double abs_c = std::fabs(contrib);
                            int min_slot = 0;
                            for (int k = 1; k < top_k; ++k) {
                                if (top_abs[k] < top_abs[min_slot]) {
                                    min_slot = k;
                                }
                            }
                            if (abs_c > top_abs[min_slot]) {
                                top_abs[min_slot] = abs_c;
                                top_val[min_slot] = contrib;
                                top_idx[min_slot] = i;
                            }
                        }
                        fprintf(stderr,
                                "[MOE_GATE_UP_DOT_TRACE] layer=%d pos=%d expert=%d weight_idx=%d out_idx=%zu "
                                "out_val=%.6e dot=%.6e in_l2=%.6e w_l2=%.6e in_nan=%zu in_inf=%zu w_nan=%zu w_inf=%zu\n",
                                layer_idx_any, target_pos, target_expert, weight_idx, out_idx,
                                static_cast<double>(out_val), dot,
                                std::sqrt(in_l2), std::sqrt(w_l2),
                                in_nan, in_inf, w_nan, w_inf);
                        std::vector<int> order(top_k, 0);
                        for (int k = 0; k < top_k; ++k) {
                            order[k] = k;
                        }
                        std::sort(order.begin(), order.end(), [&](int a, int b) {
                            return top_abs[a] > top_abs[b];
                        });
                        fprintf(stderr, "[MOE_GATE_UP_DOT_TOP] ");
                        for (int k = 0; k < top_k; ++k) {
                            const int slot = order[k];
                            fprintf(stderr,
                                    "%sidx=%zu contrib=%.6e in=%.6e w=%.6e",
                                    (k ? " | " : ""),
                                    top_idx[slot],
                                    top_val[slot],
                                    static_cast<double>(in_row[top_idx[slot]]),
                                    static_cast<double>(w_row[top_idx[slot]]));
                        }
                        fprintf(stderr, "\n");

                        const std::size_t D = static_cast<std::size_t>(gate_up_dim / 2);
                        const std::size_t up_row = out_idx;
                        const std::size_t gate_row = out_idx + D;
                        log_weight_row("MOE_GATE_UP_W_UP_ROW", up_row);
                        if (gate_row < static_cast<std::size_t>(gate_up_dim)) {
                            log_weight_row("MOE_GATE_UP_W_GATE_ROW", gate_row);
                        }
                    } else {
                        fprintf(stderr,
                                "[MOE_GATE_UP_DOT_TRACE] layer=%d pos=%d expert=%d weight_idx=%d out_idx=%zu in_ok=%d w_ok=%d out_ok=%d\n",
                                layer_idx_any, target_pos, target_expert, weight_idx, out_idx,
                                in_ok ? 1 : 0, w_ok ? 1 : 0, out_ok ? 1 : 0);
                    }
                    moe_dot_trace_count++;
                } else {
                    fprintf(stderr,
                            "[MOE_GATE_UP_DOT_TRACE] layer=%d pos=%d expert=%d weight_idx=%d weight_experts=%d compact=%d\n",
                            layer_idx_any, target_pos, target_expert, weight_idx, weight_experts,
                            weight_is_compact ? 1 : 0);
                }
            } else {
                fprintf(stderr,
                        "[MOE_GATE_UP_DOT_TRACE] layer=%d pos=%d out_of_range total=%ld\n",
                        layer_idx_any, target_pos, total_tokens);
            }
        }
    }

    const bool moe_route_trace = (std::getenv("SUROGATE_MOE_ROUTE_TRACE") != nullptr);
    static int moe_route_gate_up_trace = 0;
    if (moe_route_trace && moe_route_gate_up_trace < 4) {
        const std::size_t out_total = static_cast<std::size_t>(out.nelem());
        const std::size_t out_sample = std::min<std::size_t>(4096, out_total);
        log_tensor_sample_stats("MOE_GATE_UP_OUT", out, 0, out_sample);
        log_tensor_sample_stats("MOE_GATE_UP_OUT_GATE", out,
                                static_cast<std::size_t>(intermediate_size), out_sample);

        const int expert_id = 0;
        const std::size_t stride = static_cast<std::size_t>(2 * intermediate_size) *
                                   static_cast<std::size_t>(hidden_size);
        const std::size_t offset = stride * static_cast<std::size_t>(expert_id);
        const std::size_t wsample = std::min<std::size_t>(stride, 1024);
        std::vector<float> wvals;
        if (copy_tensor_sample_offset_as_f32(weights, offset, wsample, wvals)) {
            int nan = 0;
            float max_abs = 0.0f;
            float mean_abs = 0.0f;
            for (float v : wvals) {
                if (std::isnan(v) || std::isinf(v)) {
                    nan++;
                } else {
                    const float av = std::fabs(v);
                    max_abs = std::max(max_abs, av);
                    mean_abs += av;
                }
            }
            if (!wvals.empty()) {
                mean_abs /= static_cast<float>(wvals.size());
            }
            fprintf(stderr,
                    "[MOE_GATE_UP_WT] layer=%d expert=%d nan=%d max_abs=%.6f mean_abs=%.6f ptr=%p\n",
                    layer_idx_any, expert_id, nan, max_abs, mean_abs, static_cast<void*>(weights.Data));
        }
        moe_route_gate_up_trace++;
    }

    static int moe_route_gate_up_row_trace = 0;
    if (moe_route_trace && moe_route_gate_up_row_trace < 2 &&
        scatter_indices.Data && scatter_indices.DType == ETensorDType::INT32 && top_k > 0) {
        const int token_idx = 0;
        std::vector<int> idxs(static_cast<std::size_t>(top_k), -1);
        const std::size_t idx_offset = static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(top_k);
        CUDA_CHECK(cudaMemcpy(idxs.data(),
                              static_cast<const std::byte*>(scatter_indices.Data) + idx_offset * sizeof(int),
                              static_cast<std::size_t>(top_k) * sizeof(int),
                              cudaMemcpyDeviceToHost));
        fprintf(stderr,
                "[MOE_GATE_UP_IDX] layer=%d token=%d idxs=(",
                layer_idx_any, token_idx);
        for (int k = 0; k < top_k; ++k) {
            fprintf(stderr, "%s%d", (k ? "," : ""), idxs[k]);
        }
        fprintf(stderr, ")\n");

        const std::size_t in_row_sample = std::min<std::size_t>(2048, static_cast<std::size_t>(hidden_size));
        const std::size_t out_row_sample = std::min<std::size_t>(2048, static_cast<std::size_t>(2 * intermediate_size));
        for (int k = 0; k < top_k; ++k) {
            const int expert_pos = idxs[k];
            if (expert_pos < 0 || expert_pos >= static_cast<int>(inp.Sizes[0])) {
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
            const std::size_t in_row_offset = static_cast<std::size_t>(expert_pos) *
                                              static_cast<std::size_t>(hidden_size);
            const std::size_t out_row_offset = static_cast<std::size_t>(expert_pos) *
                                               static_cast<std::size_t>(2 * intermediate_size);
            std::vector<float> in_vals;
            std::vector<float> out_vals;
            if (copy_tensor_sample_offset_as_f32(inp, in_row_offset, in_row_sample, in_vals)) {
                float max_abs = 0.0f;
                std::size_t max_idx = 0;
                float max_val = 0.0f;
                for (std::size_t i = 0; i < in_vals.size(); ++i) {
                    const float v = in_vals[i];
                    const float av = std::fabs(v);
                    if (av > max_abs) {
                        max_abs = av;
                        max_idx = i;
                        max_val = v;
                    }
                }
                fprintf(stderr,
                        "[MOE_GATE_UP_IN_ROW] layer=%d token=%d k=%d pos=%d expert=%d max_abs=%.6f max_idx=%zu max_val=%.6f\n",
                        layer_idx_any, token_idx, k, expert_pos, expert_id, max_abs, max_idx, max_val);
            }
            if (copy_tensor_sample_offset_as_f32(out, out_row_offset, out_row_sample, out_vals)) {
                float max_abs = 0.0f;
                std::size_t max_idx = 0;
                float max_val = 0.0f;
                for (std::size_t i = 0; i < out_vals.size(); ++i) {
                    const float v = out_vals[i];
                    const float av = std::fabs(v);
                    if (av > max_abs) {
                        max_abs = av;
                        max_idx = i;
                        max_val = v;
                    }
                }
                fprintf(stderr,
                        "[MOE_GATE_UP_OUT_ROW] layer=%d token=%d k=%d pos=%d expert=%d max_abs=%.6f max_idx=%zu max_val=%.6f\n",
                        layer_idx_any, token_idx, k, expert_pos, expert_id, max_abs, max_idx, max_val);
                if (k == 0 && expert_id >= 0 && !weight_is_compact) {
                    const std::size_t w_stride = static_cast<std::size_t>(2 * intermediate_size) *
                                                 static_cast<std::size_t>(hidden_size);
                    const std::size_t w_offset = w_stride * static_cast<std::size_t>(expert_id);
                    const std::size_t w_sample = std::min<std::size_t>(1024, w_stride);
                    std::vector<float> wvals;
                    if (copy_tensor_sample_offset_as_f32(weights, w_offset, w_sample, wvals)) {
                        float w_max = 0.0f;
                        for (float v : wvals) {
                            w_max = std::max(w_max, std::fabs(v));
                        }
                        fprintf(stderr,
                                "[MOE_GATE_UP_W_ROW] layer=%d expert=%d max_abs=%.6f\n",
                                layer_idx_any, expert_id, w_max);
                    }
                    const std::size_t row = std::min<std::size_t>(max_idx, static_cast<std::size_t>(2 * intermediate_size - 1));
                    const std::size_t w_row_offset = w_offset + row * static_cast<std::size_t>(hidden_size);
                    const std::size_t w_row_sample = std::min<std::size_t>(hidden_size, static_cast<std::size_t>(2048));
                    std::vector<float> wrow;
                    if (copy_tensor_sample_offset_as_f32(weights, w_row_offset, w_row_sample, wrow)) {
                        float w_row_max = 0.0f;
                        for (float v : wrow) {
                            w_row_max = std::max(w_row_max, std::fabs(v));
                        }
                        fprintf(stderr,
                                "[MOE_GATE_UP_W_OUTROW] layer=%d expert=%d out_row=%zu max_abs=%.6f\n",
                                layer_idx_any, expert_id, row, w_row_max);
                    }

                    static int moe_gate_up_absmax_trace = 0;
                    if (moe_gate_up_absmax_trace < 2) {
                        if (auto* provider = mWeights.qlora_provider()) {
                            if (provider->debug_moe_gate_up_absmax(layer_idx_any,
                                                                   expert_id,
                                                                   static_cast<int>(row),
                                                                   mRunState.MainStream)) {
                                moe_gate_up_absmax_trace++;
                            }
                        }
                    }

                    static int moe_gate_up_row_hist = 0;
                    if (layer_idx_any == 1 && moe_gate_up_row_hist < 1 && weight_experts == num_experts) {
                        std::vector<std::pair<float, int>> max_per_expert;
                        max_per_expert.reserve(static_cast<std::size_t>(num_experts));
                        const std::size_t row_stride = static_cast<std::size_t>(hidden_size);
                        const std::size_t row_sample = std::min<std::size_t>(2048, row_stride);
                        for (int e = 0; e < num_experts; ++e) {
                            const std::size_t base = static_cast<std::size_t>(e) * w_stride + row * row_stride;
                            std::vector<float> row_vals;
                            float row_max = 0.0f;
                            if (copy_tensor_sample_offset_as_f32(weights, base, row_sample, row_vals)) {
                                for (float v : row_vals) {
                                    row_max = std::max(row_max, std::fabs(v));
                                }
                            }
                            max_per_expert.emplace_back(row_max, e);
                        }
                        std::partial_sort(max_per_expert.begin(),
                                          max_per_expert.begin() + std::min<std::size_t>(8, max_per_expert.size()),
                                          max_per_expert.end(),
                                          [](const auto& a, const auto& b) { return a.first > b.first; });
                        fprintf(stderr, "[MOE_GATE_UP_ROW_MAX] layer=%d out_row=%zu top=", layer_idx_any, row);
                        const std::size_t top_n = std::min<std::size_t>(8, max_per_expert.size());
                        for (std::size_t i = 0; i < top_n; ++i) {
                            fprintf(stderr, "%s(e%d=%.6f)", (i ? "," : ""),
                                    max_per_expert[i].second, max_per_expert[i].first);
                        }
                        fprintf(stderr, "\n");
                        moe_gate_up_row_hist++;
                    }
                }
            }
        }
        moe_route_gate_up_row_trace++;
    }

    const bool moe_lora_split_trace = (std::getenv("SUROGATE_MOE_LORA_SPLIT_TRACE") != nullptr);
    static int moe_lora_split_state_log = 0;
    if (moe_lora_split_trace && moe_lora_split_state_log < 2) {
        if (!mLoRAConfig || !mLoRAWeights || layer_idx_any < 0) {
            fprintf(stderr,
                    "[MOE_LORA_SPLIT_STATE] layer=%d has_config=%d has_weights=%d\n",
                    layer_idx_any,
                    mLoRAConfig ? 1 : 0,
                    mLoRAWeights ? 1 : 0);
        } else {
            auto& lora_block = mLoRAWeights->get_block(layer_idx_any, mRunState.MainStream);
            fprintf(stderr,
                    "[MOE_LORA_SPLIT_STATE] layer=%d rank=%d scaling=%.6f use_grouped=%d has_any=%d gate=%d up=%d down=%d\n",
                    layer_idx_any,
                    mLoRAConfig->rank,
                    mLoRAConfig->scaling(),
                    lora_block.moe.use_grouped ? 1 : 0,
                    lora_block.moe.grouped.has_any() ? 1 : 0,
                    lora_block.moe.grouped.gate.has_value() ? 1 : 0,
                    lora_block.moe.grouped.up.has_value() ? 1 : 0,
                    lora_block.moe.grouped.down.has_value() ? 1 : 0);
        }
        moe_lora_split_state_log++;
    }
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

            log_tensor_sample_stats("MOE_LORA_UP_BASE", out, 0, sample);
            log_tensor_sample_stats("MOE_LORA_GATE_BASE", out,
                                    static_cast<std::size_t>(intermediate_size), sample);

            auto compute_delta = [&](const modules::LoRAGroupedLayerWeights<Tensor>& layer,
                                     const char* tag,
                                     int out_features) {
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
                                     num_experts, rank, hidden_size,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr,
                                     1.0f, 0.0f, EMMTranspose::TN,
                                     active_ptr, /*weight_is_compact=*/false, num_active);
                } else {
                    moe_grouped_gemm(intermediate.get<float>(),
                                     inp.get<float>(),
                                     layer.A.get<float>(),
                                     expert_offsets.get<int>(),
                                     num_experts, rank, hidden_size,
                                     mRunState.cublas_handle(), mRunState.MainStream,
                                     host_offsets_ptr,
                                     1.0f, 0.0f, EMMTranspose::TN,
                                     active_ptr, /*weight_is_compact=*/false, num_active);
                }

                if (inp.DType == ETensorDType::BF16) {
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

            if (lora_block.moe.grouped.up.has_value()) {
                compute_delta(*lora_block.moe.grouped.up, "MOE_LORA_UP_DELTA", intermediate_size);
            }
            if (lora_block.moe.grouped.gate.has_value()) {
                compute_delta(*lora_block.moe.grouped.gate, "MOE_LORA_GATE_DELTA", intermediate_size);
            }
            moe_lora_split_count++;
        }
    }
    // DEBUG: re-sample expert 122 weights after GEMM to detect corruption.
    static int moe_w122_post_trace = 0;
    if (moe_w122_post_trace < 4 && weight_experts > 122) {
        const int expert_id = 122;
        const std::size_t stride = static_cast<std::size_t>(2 * intermediate_size) *
                                   static_cast<std::size_t>(hidden_size);
        const std::size_t offset = stride * static_cast<std::size_t>(expert_id);
        const std::size_t sample = std::min<std::size_t>(stride, 1024);
        std::vector<float> wvals;
        if (copy_tensor_sample_offset_as_f32(weights, offset, sample, wvals)) {
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
                    "[MOE_W122_GATE_UP_POST] layer=%d expert=%d nan=%d max_abs=%.6f\n",
                    layer_idx_any, expert_id, nan, max_abs);
        }
        moe_w122_post_trace++;
    }

    // DEBUG: Compare expert weights (expert 0 vs 1) for layer 0/top.
    static int moe_gate_up_w_trace = 0;
    if (moe_gate_up_w_trace < 4) {
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
            const long stride = static_cast<long>(gate_up_dim) * static_cast<long>(hidden_size);
            const std::size_t expected_elems = static_cast<std::size_t>(num_experts) * static_cast<std::size_t>(stride);
            std::vector<float> w0(4, 0.0f);
            std::vector<float> w1(4, 0.0f);
            (void)copy_tensor_sample_offset_as_f32(weights, 0, w0.size(), w0);
            (void)copy_tensor_sample_offset_as_f32(weights, static_cast<std::size_t>(stride), w1.size(), w1);
            fprintf(stderr,
                    "[MOE_W_GATE_UP] layer=%d w0=%.6f,%.6f,%.6f,%.6f w1=%.6f,%.6f,%.6f,%.6f\n",
                    layer_idx,
                    w0[0], w0[1], w0[2], w0[3],
                    w1[0], w1[1], w1[2], w1[3]);
            fprintf(stderr,
                    "[MOE_W_META_GATE_UP] layer=%d dtype=%s rank=%d nelem=%zu expected=%zu elem_size=%zu stride=%ld data=%p\n",
                    layer_idx,
                    dtype_to_str(weights.DType),
                    weights.Rank,
                    weights.nelem(),
                    expected_elems,
                    elem_size,
                    stride,
                    static_cast<void*>(weights.Data));
            // Log expert base pointers to detect aliasing.
            for (int e = 0; e < std::min(num_experts, 4); ++e) {
                const std::size_t byte_offset = static_cast<std::size_t>(stride) * static_cast<std::size_t>(e) * elem_size;
                const std::byte* ptr = static_cast<const std::byte*>(weights.Data) + byte_offset;
                std::vector<float> we(4, 0.0f);
                (void)copy_tensor_sample_offset_as_f32(weights, static_cast<std::size_t>(stride) * e, we.size(), we);
                fprintf(stderr,
                        "[MOE_W_PTR_GATE_UP] layer=%d expert=%d ptr=%p vals=%.6f,%.6f,%.6f,%.6f\n",
                        layer_idx, e, static_cast<const void*>(ptr),
                        we[0], we[1], we[2], we[3]);
            }
            // Sample-based checksum for expert 0/1 to detect identical weights.
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
                            "[MOE_W_SUM_GATE_UP] layer=%d expert=%d sum=%.6e mean_abs=%.6e\n",
                            layer_idx, e,
                            sum, (sum_abs / static_cast<double>(ws.size())));
                }
            }
            // Targeted expert sample to catch NaN weights.
            const int debug_expert = 122;
            if (num_experts > debug_expert) {
                std::vector<float> wdbg(16, 0.0f);
                if (copy_tensor_sample_offset_as_f32(weights,
                                                     static_cast<std::size_t>(stride) * debug_expert,
                                                     wdbg.size(),
                                                     wdbg)) {
                    float max_abs = 0.0f;
                    for (float v : wdbg) {
                        max_abs = std::max(max_abs, std::fabs(v));
                    }
                    fprintf(stderr,
                            "[MOE_W_SAMPLE_GATE_UP] layer=%d expert=%d nan=%d max_abs=%.6f vals=%.6f,%.6f,%.6f,%.6f\n",
                            layer_idx,
                            debug_expert,
                            sample_has_nan_or_inf(wdbg) ? 1 : 0,
                            max_abs,
                            wdbg[0], wdbg[1], wdbg[2], wdbg[3]);
                }
            }
            // One-time full scan of a single expert to detect NaNs in weights.
            static int moe_gate_up_w_full_trace = 0;
            if (layer_idx == 2 && moe_gate_up_w_full_trace < 1 && num_experts > debug_expert) {
                const std::size_t elems = static_cast<std::size_t>(stride);
                std::vector<float> wfull;
                if (copy_tensor_sample_offset_as_f32(weights,
                                                     static_cast<std::size_t>(stride) * debug_expert,
                                                     elems,
                                                     wfull)) {
                    std::size_t nan_count = 0;
                    std::size_t inf_count = 0;
                    double sum_abs = 0.0;
                    float max_abs = 0.0f;
                    for (float v : wfull) {
                        if (std::isnan(v)) {
                            nan_count++;
                            continue;
                        }
                        if (std::isinf(v)) {
                            inf_count++;
                            continue;
                        }
                        const float av = std::fabs(v);
                        sum_abs += static_cast<double>(av);
                        if (av > max_abs) {
                            max_abs = av;
                        }
                    }
                    const double mean_abs = wfull.empty() ? 0.0 : (sum_abs / static_cast<double>(wfull.size()));
                    fprintf(stderr,
                            "[MOE_W_FULL_GATE_UP] layer=%d expert=%d elems=%zu nan=%zu inf=%zu max_abs=%.6f mean_abs=%.6f\n",
                            layer_idx, debug_expert, wfull.size(), nan_count, inf_count, max_abs, mean_abs);
                }
                moe_gate_up_w_full_trace++;
            }
            fprintf(stderr,
                    "[MOE_GEMM_GATE_UP] layer=%d path=%s dtype=%s\n",
                    layer_idx,
                    (weights.DType == ETensorDType::BF16 ? "bf16" : "fp32"),
                    dtype_to_str(weights.DType));
            moe_gate_up_w_trace++;
        }
    }

    // DEBUG: Trace forward expert_gate_up magnitude for layer 0/top.
    static int moe_gate_up_trace = 0;
    if (moe_gate_up_trace < 12) {
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
            log_tensor_stats_ex("MOE_FWD_GATE_UP", layer_idx, op.outputs[0].name, out, 4096, true);
            moe_gate_up_trace++;
        }
    }

    // DEBUG: Find max |expert_gate_up| over a larger sample.
    static int moe_gate_up_max_trace = 0;
    if (moe_gate_up_max_trace < 4) {
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
                fprintf(stderr,
                        "[MOE_FWD_GATE_UP_MAX] layer=%d sample_max=%.6f idx=%zu sample=%zu total=%zu\n",
                        layer_idx, vals[max_idx], max_idx, sample, total);
            }
            moe_gate_up_max_trace++;
        }
    }

    // DEBUG: Detect first NaN in expert_gate_up for layer 2 token 3.
    {
        static int moe_gate_up_nan_found = 0;
        int layer_idx = op.attrs.layer_idx;
        std::string field;
        if (layer_idx < 0 && !op.inputs.empty()) {
            std::string_view name = op.inputs[0].name;
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx, field);
        }
        if (layer_idx == 2 && !moe_gate_up_nan_found &&
            scatter_indices.Data && scatter_indices.DType == ETensorDType::INT32) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            const int token_idx = 3;
            std::vector<int> idxs(top_k, -1);
            const std::size_t w_offset = static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(top_k);
            CUDA_CHECK(cudaMemcpy(idxs.data(),
                                  static_cast<const std::byte*>(scatter_indices.Data) + w_offset * sizeof(int),
                                  top_k * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            // Fetch expert offsets to map idx -> expert id.
            std::vector<int> h_offsets(num_experts + 1, 0);
            CUDA_CHECK(cudaMemcpy(h_offsets.data(),
                                  expert_offsets.Data,
                                  static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            auto find_expert = [&](int idx) {
                for (int e = 0; e < num_experts; ++e) {
                    if (idx >= h_offsets[e] && idx < h_offsets[e + 1]) {
                        return e;
                    }
                }
                return -1;
            };

            for (int k = 0; k < top_k; ++k) {
                const int idx = idxs[k];
                if (idx < 0) continue;
                float out_min = 0.0f, out_max = 0.0f;
                const bool out_nan = tensor_row_has_nan_or_inf(out, idx, &out_min, &out_max);
                if (!out_nan) {
                    continue;
                }
                const int expert_id = find_expert(idx);
                std::vector<float> in_vals(4, 0.0f);
                std::vector<float> out_vals(4, 0.0f);
                (void)copy_tensor_token_sample_as_f32(inp, idx, in_vals.size(), in_vals);
                (void)copy_tensor_token_sample_as_f32(out, idx, out_vals.size(), out_vals);
                float in_min = 0.0f, in_max = 0.0f;
                const bool in_nan = tensor_row_has_nan_or_inf(inp, idx, &in_min, &in_max);
                fprintf(stderr,
                        "[MOE_GATE_UP_NAN] layer=%d token=%d k=%d idx=%d expert=%d in_nan=%d in_min=%.6f in_max=%.6f "
                        "out_min=%.6f out_max=%.6f in=%.6f,%.6f,%.6f,%.6f out=%.6f,%.6f,%.6f,%.6f\n",
                        layer_idx, token_idx, k, idx, expert_id,
                        in_nan ? 1 : 0, in_min, in_max,
                        out_min, out_max,
                        in_vals[0], in_vals[1], in_vals[2], in_vals[3],
                        out_vals[0], out_vals[1], out_vals[2], out_vals[3]);

                if (expert_id >= 0) {
                    const long stride = static_cast<long>(gate_up_dim) * static_cast<long>(hidden_size);
                    std::vector<float> in_row(hidden_size, 0.0f);
                    std::vector<float> w_row(hidden_size, 0.0f);
                    const bool in_ok = copy_tensor_token_sample_as_f32(inp, idx, in_row.size(), in_row);
                    const bool w_ok = copy_tensor_sample_offset_as_f32(weights,
                                                                      static_cast<std::size_t>(stride) * static_cast<std::size_t>(expert_id),
                                                                      w_row.size(),
                                                                      w_row);
                    double dot = 0.0;
                    std::size_t in_nan_count = 0;
                    std::size_t in_inf_count = 0;
                    std::size_t w_nan_count = 0;
                    std::size_t w_inf_count = 0;
                    if (in_ok && w_ok) {
                        for (std::size_t i = 0; i < in_row.size(); ++i) {
                            const float a = in_row[i];
                            const float b = w_row[i];
                            if (std::isnan(a)) in_nan_count++;
                            if (std::isinf(a)) in_inf_count++;
                            if (std::isnan(b)) w_nan_count++;
                            if (std::isinf(b)) w_inf_count++;
                            if (std::isfinite(a) && std::isfinite(b)) {
                                dot += static_cast<double>(a) * static_cast<double>(b);
                            }
                        }
                    }
                    fprintf(stderr,
                            "[MOE_GATE_UP_NAN_DOT] layer=%d idx=%d expert=%d in_ok=%d w_ok=%d "
                            "in_nan=%zu in_inf=%zu w_nan=%zu w_inf=%zu dot=%.6e\n",
                            layer_idx, idx, expert_id,
                            in_ok ? 1 : 0, w_ok ? 1 : 0,
                            in_nan_count, in_inf_count, w_nan_count, w_inf_count, dot);
                }

                // Pointer overlap diagnostics (weights should never overlap activations).
                const std::size_t w_bytes = weights.nelem() * get_dtype_size(weights.DType);
                const auto w_base = reinterpret_cast<std::uintptr_t>(weights.Data);
                const auto w_end = w_base + w_bytes;
                auto overlap = [&](const Tensor& t) {
                    if (!t.Data) return false;
                    const auto t_base = reinterpret_cast<std::uintptr_t>(t.Data);
                    const std::size_t t_bytes = t.nelem() * get_dtype_size(t.DType);
                    const auto t_end = t_base + t_bytes;
                    return (t_base < w_end) && (t_end > w_base);
                };
                fprintf(stderr,
                        "[MOE_GATE_UP_NAN_PTRS] layer=%d weights=%p bytes=%zu device=%d stack=%d "
                        "inp=%p overlap=%d out=%p overlap=%d\n",
                        layer_idx,
                        weights.Data, w_bytes, weights.Device,
                        mRunState.Stack.owns(weights.Data) ? 1 : 0,
                        inp.Data, overlap(inp) ? 1 : 0,
                        out.Data, overlap(out) ? 1 : 0);
                moe_gate_up_nan_found = 1;
                break;
            }
        }
    }

    // DEBUG: if any gate_up rows for token 3 have NaNs, dump row stats and weights.
    static int moe_gate_up_row_trace = 0;
    if (moe_gate_up_row_trace < 8) {
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
            if (!scatter_t || !scatter_t->Data || scatter_t->DType != ETensorDType::INT32) {
                fprintf(stderr,
                        "[MOE_GATE_UP_SCATTER_MISSING] layer=%d name=%s dtype=%s ptr=%p\n",
                        layer_idx,
                        scatter_t ? "scatter_indices" : "null",
                        scatter_t ? dtype_to_str(scatter_t->DType) : "<none>",
                        scatter_t ? static_cast<void*>(scatter_t->Data) : nullptr);
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
                            "[MOE_GATE_UP_ROW] layer=%d token=%d k=%d idx=%d expert=%d in_nan=%d in_min=%.6f in_max=%.6f "
                            "out_nan=%d out_min=%.6f out_max=%.6f\n",
                            layer_idx, token_idx, k, idx, expert_id,
                            in_nan ? 1 : 0, in_min, in_max,
                            out_nan ? 1 : 0, out_min, out_max);
                    if (expert_id >= 0) {
                        const long stride = static_cast<long>(gate_up_dim) * static_cast<long>(hidden_size);
                        std::vector<float> wvals(4, 0.0f);
                        (void)copy_tensor_sample_offset_as_f32(weights,
                                                               static_cast<std::size_t>(stride) * expert_id,
                                                               wvals.size(),
                                                               wvals);
                        fprintf(stderr,
                                "[MOE_GATE_UP_W_SAMPLE] layer=%d expert=%d w=%.6f,%.6f,%.6f,%.6f\n",
                                layer_idx, expert_id,
                                wvals[0], wvals[1], wvals[2], wvals[3]);
                    }
                }
            }
            // Targeted dot check for expert 122 to detect NaNs in weights/input.
            static int moe_gate_up_dot_trace = 0;
            if (layer_idx == 2 && moe_gate_up_dot_trace < 1 && top_k > 0) {
                const int target_k = top_k - 1;
                const int idx = (target_k < static_cast<int>(indices.size())) ? indices[target_k] : -1;
                if (idx >= 0) {
                    const int expert_id = find_expert(idx);
                    if (expert_id == 122) {
                        const long stride = static_cast<long>(gate_up_dim) * static_cast<long>(hidden_size);
                        std::vector<float> in_row(hidden_size, 0.0f);
                        std::vector<float> w_row(hidden_size, 0.0f);
                        const bool in_ok = copy_tensor_token_sample_as_f32(inp, idx, in_row.size(), in_row);
                        const bool w_ok = copy_tensor_sample_offset_as_f32(weights,
                                                                          static_cast<std::size_t>(stride) * expert_id,
                                                                          w_row.size(),
                                                                          w_row);
                        float out_val = 0.0f;
                        std::vector<float> out_row(1, 0.0f);
                        const bool out_ok = copy_tensor_token_sample_as_f32(out, idx, out_row.size(), out_row);
                        if (out_ok) {
                            out_val = out_row[0];
                        }
                        double dot = 0.0;
                        bool dot_nan = false;
                        if (in_ok && w_ok) {
                            for (std::size_t i = 0; i < in_row.size(); ++i) {
                                const float a = in_row[i];
                                const float b = w_row[i];
                                if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b)) {
                                    dot_nan = true;
                                    break;
                                }
                                dot += static_cast<double>(a) * static_cast<double>(b);
                            }
                        }
                        fprintf(stderr,
                                "[MOE_GATE_UP_DOT] layer=%d idx=%d expert=%d in_ok=%d w_ok=%d dot_nan=%d dot=%.6e out0=%.6e\n",
                                layer_idx, idx, expert_id,
                                in_ok ? 1 : 0, w_ok ? 1 : 0, dot_nan ? 1 : 0,
                                dot, static_cast<double>(out_val));
                        moe_gate_up_dot_trace++;
                    }
                }
            }
            moe_gate_up_row_trace++;
        }
    }

    // DEBUG: one-time scatter snapshot and dot check for token 3 k=top_k-1 (expert 122).
    {
        static int moe_gate_up_dot2_trace = 0;
        int layer_idx = op.attrs.layer_idx;
        std::string field;
        if (layer_idx < 0 && !op.inputs.empty()) {
            std::string_view name = op.inputs[0].name;
            if (name.rfind("saved.", 0) == 0) {
                name.remove_prefix(6);
            }
            parse_block_param(name, layer_idx, field);
        }
        if (layer_idx == 2 && moe_gate_up_dot2_trace < 1) {
            if (scatter_indices.Data && scatter_indices.DType == ETensorDType::INT32) {
                const int token_idx = 3;
                const int target_k = std::max(0, top_k - 1);
                const std::size_t w_offset = static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(top_k);
                std::vector<int> idxs(top_k, -1);
                CUDA_CHECK(cudaMemcpy(idxs.data(),
                                      static_cast<const std::byte*>(scatter_indices.Data) + w_offset * sizeof(int),
                                      top_k * sizeof(int),
                                      cudaMemcpyDeviceToHost));
                fprintf(stderr,
                        "[MOE_SCATTER_GATE_UP] layer=%d ptr=%p token=%d idxs=(",
                        layer_idx, static_cast<void*>(scatter_indices.Data), token_idx);
                for (int k = 0; k < top_k; ++k) {
                    fprintf(stderr, "%s%d", (k ? "," : ""), idxs[k]);
                }
                fprintf(stderr, ")\n");

                const int idx = (target_k < static_cast<int>(idxs.size())) ? idxs[target_k] : -1;
                if (idx >= 0) {
                    const long stride = static_cast<long>(gate_up_dim) * static_cast<long>(hidden_size);
                    std::vector<float> in_row(hidden_size, 0.0f);
                    std::vector<float> w_row(hidden_size, 0.0f);
                    const bool in_ok = copy_tensor_token_sample_as_f32(inp, idx, in_row.size(), in_row);
                    const bool w_ok = copy_tensor_sample_offset_as_f32(weights,
                                                                      static_cast<std::size_t>(stride) * 122,
                                                                      w_row.size(),
                                                                      w_row);
                    double dot = 0.0;
                    bool dot_nan = false;
                    if (in_ok && w_ok) {
                        for (std::size_t i = 0; i < in_row.size(); ++i) {
                            const float a = in_row[i];
                            const float b = w_row[i];
                            if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b)) {
                                dot_nan = true;
                                break;
                            }
                            dot += static_cast<double>(a) * static_cast<double>(b);
                        }
                    }
                    std::vector<float> out_row(1, 0.0f);
                    const bool out_ok = copy_tensor_token_sample_as_f32(out, idx, out_row.size(), out_row);
                    fprintf(stderr,
                            "[MOE_GATE_UP_DOT2] layer=%d idx=%d in_ok=%d w_ok=%d out_ok=%d dot_nan=%d dot=%.6e out0=%.6e\n",
                            layer_idx, idx,
                            in_ok ? 1 : 0, w_ok ? 1 : 0, out_ok ? 1 : 0,
                            dot_nan ? 1 : 0, dot,
                            out_ok ? static_cast<double>(out_row[0]) : 0.0);
                }
                moe_gate_up_dot2_trace++;
            } else {
                fprintf(stderr,
                        "[MOE_GATE_UP_DOT2] layer=%d scatter_missing\n",
                        layer_idx);
                moe_gate_up_dot2_trace++;
            }
        }
    }

    mTensorMap[op.outputs[0].name] = out;
}

void CompiledExecutor::dispatch_moe_grouped_gemm_gate_up_backward(const CompiledOp& op) {
    Tensor& d_gate_up = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& weights = resolve_tensor(op.inputs[2]);
    Tensor& d_input = ensure_output_tensor(op.outputs[0]);

    // Get expert offsets from stored state (per-layer when available).
    Tensor* expert_offsets_ptr = nullptr;
    Tensor expert_offsets_view;
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

    static int moe_gate_up_trace_layer = 0;
    if (should_trace_moe_layer(layer_idx, moe_gate_up_trace_layer)) {
        log_tensor_mag_unbounded("MOE_GATE_UP_BWD_TRACE_DOUT", layer_idx, op.inputs[0].name, d_gate_up, 4096);
        log_tensor_mag_unbounded("MOE_GATE_UP_BWD_TRACE_INP", layer_idx, op.inputs[1].name, inp, 4096);
        log_tensor_mag_unbounded("MOE_GATE_UP_BWD_TRACE_W", layer_idx, op.inputs[2].name, weights, 4096);
    }

    // One-time NaN watchdog to pinpoint MoE gate-up backward issues.
    static bool moe_gate_up_nan_logged = false;
    if (!moe_gate_up_nan_logged) {
        const bool nan_dout = tensor_sample_has_nan_or_inf(d_gate_up, 3);
        const bool nan_inp = tensor_sample_has_nan_or_inf(inp, 3);
        if (nan_dout || nan_inp) {
            fprintf(stderr,
                    "[MOE_GATE_UP_BWD_NAN] layer=%d nan_dout=%d nan_inp=%d d_gate_up=%s inp=%s\n",
                    layer_idx,
                    nan_dout ? 1 : 0,
                    nan_inp ? 1 : 0,
                    op.inputs[0].name.c_str(),
                    op.inputs[1].name.c_str());
            log_tensor_stats_ex("MOE_GATE_UP_BWD_NAN_DOUT", layer_idx, op.inputs[0].name, d_gate_up, 4096, true);
            log_tensor_stats_ex("MOE_GATE_UP_BWD_NAN_INP", layer_idx, op.inputs[1].name, inp, 4096, true);
            moe_gate_up_nan_logged = true;
        }
    }
    static int moe_gate_up_mag = 0;
    if (moe_gate_up_mag < 8 && layer_idx <= 2) {
        log_tensor_mag("MOE_GATE_UP_BWD_DOUT", layer_idx, op.inputs[0].name, d_gate_up, 4096);
        log_tensor_mag("MOE_GATE_UP_BWD_INP", layer_idx, op.inputs[1].name, inp, 4096);
        moe_gate_up_mag++;
    }
    log_moe_gate_up_weight_sample("PRE_MOE_DOWN_BWD", layer_idx, mMicroStep, mWeights, mConfig);
    if (layer_idx >= 0) {
        const std::string key = "blocks[" + std::to_string(layer_idx) + "].moe_expert_offsets";
        auto it = mMoESavedBuffers.find(key);
        if (it != mMoESavedBuffers.end() && it->second != nullptr) {
            expert_offsets_view.DType = ETensorDType::INT32;
            expert_offsets_view.Rank = 1;
            expert_offsets_view.Sizes[0] = static_cast<long>(mConfig.NumExperts + 1);
            expert_offsets_view.Data = static_cast<std::byte*>(it->second);
            expert_offsets_ptr = &expert_offsets_view;
        }
    }
    if (!expert_offsets_ptr) {
        auto it = mTensorMap.find("moe_expert_offsets");
        if (it == mTensorMap.end()) {
            throw std::runtime_error("moe_grouped_gemm_gate_up_backward: expert_offsets not found");
        }
        expert_offsets_ptr = &it->second;
    }

    // Use the persistent buffer directly instead of tensorMap
    const int* offsets_ptr = static_cast<const int*>(mMoEExpertOffsetsGPU);
    (void)offsets_ptr;  // Used by kernel through expert_offsets

    // Synchronize to ensure all previous async ops are done
    CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));

    const int num_experts = static_cast<int>(mConfig.NumExperts);
    const int hidden_size = static_cast<int>(mConfig.HiddenSize);
    // Use MoeIntermediateSize for MoE models (may differ from IntermediateSize)
    const int intermediate_size = (mConfig.MoeIntermediateSize > 0)
        ? static_cast<int>(mConfig.MoeIntermediateSize)
        : static_cast<int>(mConfig.IntermediateSize);

    const int weight_experts = static_cast<int>(weights.Sizes[0]);
    MoeCompactInfo compact = build_moe_compact_info(expert_offsets_ptr->get<int>(),
                                                    num_experts,
                                                    weight_experts,
                                                    mRunState.MainStream,
                                                    layer_idx,
                                                    "moe_grouped_gemm_gate_up_backward");
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
        // Fallback: copy offsets to host for selection.
        std::vector<int> host_offsets_fallback(static_cast<std::size_t>(num_experts + 1), 0);
        CUDA_CHECK(cudaMemcpyAsync(host_offsets_fallback.data(),
                                   expert_offsets_ptr->get<int>(),
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

    // DEBUG: check if weights change across gate_up backward for layer 2.
    static int moe_gate_up_bwd_trace = 0;
    bool pre_nan = false;
    float pre_min = 0.0f, pre_max = 0.0f;
    if (layer_idx == 2 && moe_gate_up_bwd_trace < 1) {
        pre_nan = tensor_row_has_nan_or_inf(weights, 122, &pre_min, &pre_max);
    }

    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(d_input, mRunState.MainStream);
    } else if (d_gate_up.DType == ETensorDType::BF16) {
        moe_grouped_gemm_gate_up_backward(d_input.get<nv_bfloat16>(),
                                          d_gate_up.get<nv_bfloat16>(),
                                          weights.get<nv_bfloat16>(),
                                          expert_offsets_ptr->get<int>(),
                                          num_experts, hidden_size, intermediate_size,
                                          mRunState.cublas_handle(), mRunState.MainStream,
                                          host_offsets_ptr,
                                          active_ptr,
                                          weight_is_compact,
                                          num_active);
    } else {
        moe_grouped_gemm_gate_up_backward(d_input.get<float>(),
                                          d_gate_up.get<float>(),
                                          weights.get<float>(),
                                          expert_offsets_ptr->get<int>(),
                                          num_experts, hidden_size, intermediate_size,
                                          mRunState.cublas_handle(), mRunState.MainStream,
                                          host_offsets_ptr,
                                          active_ptr,
                                          weight_is_compact,
                                          num_active);
    }

    static int moe_gate_up_trace_layer_out = 0;
    if (should_trace_moe_layer(layer_idx, moe_gate_up_trace_layer_out)) {
        log_tensor_mag_unbounded("MOE_GATE_UP_BWD_TRACE_DIN", layer_idx, op.outputs[0].name, d_input, 4096);
    }

    if (layer_idx == 2 && moe_gate_up_bwd_trace < 1) {
        float post_min = 0.0f, post_max = 0.0f;
        const bool post_nan = tensor_row_has_nan_or_inf(weights, 122, &post_min, &post_max);
        fprintf(stderr,
                "[MOE_GATE_UP_BWD_W] layer=%d pre_nan=%d pre_min=%.6f pre_max=%.6f post_nan=%d post_min=%.6f post_max=%.6f\n",
                layer_idx,
                pre_nan ? 1 : 0, pre_min, pre_max,
                post_nan ? 1 : 0, post_min, post_max);
        moe_gate_up_bwd_trace++;
    }

    mTensorMap[op.outputs[0].name] = d_input;

    // Weight gradient computation would go here if needed (for fine-tuning experts)
}


}  // namespace dsl
