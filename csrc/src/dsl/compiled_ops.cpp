// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compiled operation dispatch for DSL Graph executor.

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

#include "dsl/dsl_runtime.h"
#include "dsl/dsl_weight_manager.h"
#include "dsl/graph_executor_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "dsl/op_shape_signatures.h"
#include "modules/backward_hooks.h"
#include "modules/forward_hooks.h"
#include "modules/fp8_scaling_config.h"
#include "modules/lora/lora_config.h"
#include "modules/lora/lora_weights_manager.h"
#include "modules/matmul_context.h"
#include "modules/model_config.h"
#include "modules/moe/moe_types.h"
#include "recipes/recipe.h"
#include "training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"

namespace dsl {
namespace {

/// Strip trailing SSA-style numeric suffix (e.g., "qkv_rope_7" -> "qkv_rope")
/// The DSL IR generates unique tensor names with suffixes like _0, _7, _10, etc.
/// This function removes these suffixes for field name matching.
std::string strip_ssa_suffix(const std::string& field) {
    auto pos = field.rfind('_');
    if (pos == std::string::npos || pos == 0) {
        return field;
    }
    // Check if everything after the underscore is digits
    bool all_digits = true;
    for (std::size_t i = pos + 1; i < field.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(field[i]))) {
            all_digits = false;
            break;
        }
    }
    if (all_digits && pos + 1 < field.size()) {
        return field.substr(0, pos);
    }
    return field;
}

int moe_trace_layer() {
    static int layer = -2;
    if (layer == -2) {
        const char* env = std::getenv("SUROGATE_MOE_TRACE_LAYER");
        layer = env ? std::atoi(env) : -1;
    }
    return layer;
}

int moe_trace_limit() {
    static int limit = -2;
    if (limit == -2) {
        const char* env = std::getenv("SUROGATE_MOE_TRACE_LIMIT");
        limit = env ? std::atoi(env) : 8;
    }
    return limit;
}

bool should_trace_moe_layer(int layer_idx, int& counter) {
    const int target = moe_trace_layer();
    if (target < 0 || layer_idx != target) {
        return false;
    }
    const int limit = moe_trace_limit();
    if (limit >= 0 && counter >= limit) {
        return false;
    }
    counter++;
    return true;
}

struct MoeCompactInfo {
    std::vector<int> host_offsets;
    std::vector<int> active_experts;
    int num_active = 0;
    bool weight_is_compact = false;
};

void log_cuda_ptr_attr(const char* tag, const void* ptr, int layer_idx, const char* name) {
    if (!ptr) {
        fprintf(stderr, "[%s] layer=%d name=%s ptr=null\n", tag, layer_idx, name ? name : "<unnamed>");
        return;
    }
    cudaPointerAttributes attr{};
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "[%s] layer=%d name=%s ptr=%p attr_err=%s\n",
                tag, layer_idx, name ? name : "<unnamed>", ptr, cudaGetErrorString(err));
        cudaGetLastError();
        return;
    }
#if CUDART_VERSION >= 10000
    fprintf(stderr,
            "[%s] layer=%d name=%s ptr=%p type=%d device=%d\n",
            tag, layer_idx, name ? name : "<unnamed>", ptr,
            static_cast<int>(attr.type), attr.device);
#else
    fprintf(stderr,
            "[%s] layer=%d name=%s ptr=%p type=%d device=%d\n",
            tag, layer_idx, name ? name : "<unnamed>", ptr,
            static_cast<int>(attr.memoryType), attr.device);
#endif
}

MoeCompactInfo build_moe_compact_info(const int* expert_offsets_dev,
                                      int num_experts,
                                      int weight_experts,
                                      cudaStream_t stream,
                                      int layer_idx,
                                      const char* tag) {
    MoeCompactInfo info;
    if (!expert_offsets_dev || num_experts <= 0 || weight_experts <= 0) {
        return info;
    }
    info.weight_is_compact = (weight_experts != num_experts);
    if (!info.weight_is_compact) {
        return info;
    }

    info.host_offsets.resize(num_experts + 1, 0);
    CUDA_CHECK(cudaMemcpyAsync(info.host_offsets.data(),
                               expert_offsets_dev,
                               static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    info.active_experts.reserve(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        if (info.host_offsets[e + 1] > info.host_offsets[e]) {
            info.active_experts.push_back(e);
        }
    }
    info.num_active = static_cast<int>(info.active_experts.size());

    if (weight_experts > 0 && info.num_active != weight_experts) {
        static int warn_count = 0;
        if (warn_count < 8) {
            fprintf(stderr,
                    "[MOE_COMPACT_WARN] tag=%s layer=%d num_active=%d weight_experts=%d num_experts=%d\n",
                    tag, layer_idx, info.num_active, weight_experts, num_experts);
            warn_count++;
        }
        if (info.num_active > weight_experts) {
            info.active_experts.resize(weight_experts);
            info.num_active = weight_experts;
        }
    }

    return info;
}

MoeCompactInfo build_moe_compact_info_from_host(const int* host_offsets,
                                                int num_experts,
                                                int weight_experts,
                                                int layer_idx,
                                                const char* tag) {
    MoeCompactInfo info;
    if (!host_offsets || num_experts <= 0 || weight_experts <= 0) {
        return info;
    }
    info.weight_is_compact = (weight_experts != num_experts);
    if (!info.weight_is_compact) {
        return info;
    }

    info.active_experts.reserve(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        if (host_offsets[e + 1] > host_offsets[e]) {
            info.active_experts.push_back(e);
        }
    }
    info.num_active = static_cast<int>(info.active_experts.size());

    if (weight_experts > 0 && info.num_active != weight_experts) {
        static int warn_count = 0;
        if (warn_count < 8) {
            fprintf(stderr,
                    "[MOE_COMPACT_WARN] tag=%s layer=%d num_active=%d weight_experts=%d num_experts=%d\n",
                    tag, layer_idx, info.num_active, weight_experts, num_experts);
            warn_count++;
        }
        if (info.num_active > weight_experts) {
            info.active_experts.resize(weight_experts);
            info.num_active = weight_experts;
        }
    }

    return info;
}

bool copy_tensor_sample_as_f32(const Tensor& t, std::size_t count, std::vector<float>& out) {
    out.assign(count, 0.0f);
    if (count == 0 || !t.Data) {
        return false;
    }
    switch (t.DType) {
    case ETensorDType::FP32:
        cudaMemcpy(out.data(), t.Data, count * sizeof(float), cudaMemcpyDeviceToHost);
        return true;
    case ETensorDType::BF16: {
        std::vector<nv_bfloat16> tmp(count);
        cudaMemcpy(tmp.data(), t.Data, count * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(tmp[i]);
        }
        return true;
    }
    case ETensorDType::FP16: {
        std::vector<half> tmp(count);
        cudaMemcpy(tmp.data(), t.Data, count * sizeof(half), cudaMemcpyDeviceToHost);
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(tmp[i]);
        }
        return true;
    }
    default:
        return false;
    }
}

double sample_mean_abs(const Tensor& t, std::size_t count) {
    std::vector<float> vals;
    if (!copy_tensor_sample_as_f32(t, count, vals)) {
        return 0.0;
    }
    double sum = 0.0;
    for (float v : vals) {
        sum += std::abs(static_cast<double>(v));
    }
    return vals.empty() ? 0.0 : (sum / static_cast<double>(vals.size()));
}

bool copy_tensor_sample_offset_as_f32(const Tensor& t, std::size_t elem_offset,
                                      std::size_t count, std::vector<float>& out) {
    out.assign(count, 0.0f);
    if (count == 0 || !t.Data) {
        return false;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    if (elem_offset + count > total) {
        return false;
    }
    const std::size_t byte_offset = elem_offset * get_dtype_size(t.DType);
    const std::byte* base = static_cast<const std::byte*>(t.Data) + byte_offset;
    switch (t.DType) {
    case ETensorDType::FP32:
        cudaMemcpy(out.data(), base, count * sizeof(float), cudaMemcpyDeviceToHost);
        return true;
    case ETensorDType::BF16: {
        std::vector<nv_bfloat16> tmp(count);
        cudaMemcpy(tmp.data(), base, count * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(tmp[i]);
        }
        return true;
    }
    case ETensorDType::FP16: {
        std::vector<half> tmp(count);
        cudaMemcpy(tmp.data(), base, count * sizeof(half), cudaMemcpyDeviceToHost);
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(tmp[i]);
        }
        return true;
    }
    default:
        return false;
    }
}

void log_moe_gate_up_weight_sample(const char* tag,
                                   int layer_idx,
                                   int micro_step,
                                   DslParamStore& weights,
                                   const modules::ModelConfig& cfg) {
    if (layer_idx != 2) {
        return;
    }
    static int trace_count = 0;
    if (trace_count >= 64) {
        return;
    }
    const std::string wname = "blocks[" + std::to_string(layer_idx) + "].experts_gate_up";
    if (!weights.has(wname)) {
        fprintf(stderr,
                "[MOE_W_SAMPLE_MISS] tag=%s micro_step=%d layer=%d name=%s\n",
                tag ? tag : "<null>", micro_step, layer_idx, wname.c_str());
        trace_count++;
        return;
    }
    Tensor& gw = weights.get(wname);
    const int hidden_size = static_cast<int>(cfg.HiddenSize);
    const int intermediate_size = (cfg.MoeIntermediateSize > 0)
        ? static_cast<int>(cfg.MoeIntermediateSize)
        : static_cast<int>(cfg.IntermediateSize);
    const int expert_id = 122;
    const std::size_t stride = static_cast<std::size_t>(2 * intermediate_size) *
                               static_cast<std::size_t>(hidden_size);
    const std::size_t offset = stride * static_cast<std::size_t>(expert_id);
    if (gw.nelem() < static_cast<long>(offset + 1)) {
        fprintf(stderr,
                "[MOE_W_SAMPLE_OOR] tag=%s micro_step=%d layer=%d name=%s nelem=%ld offset=%zu\n",
                tag ? tag : "<null>", micro_step, layer_idx, wname.c_str(),
                static_cast<long>(gw.nelem()), offset);
        trace_count++;
        return;
    }
    const std::size_t sample = std::min<std::size_t>(stride, 512);
    std::vector<float> wvals;
    if (!copy_tensor_sample_offset_as_f32(gw, offset, sample, wvals)) {
        fprintf(stderr,
                "[MOE_W_SAMPLE_FAIL] tag=%s micro_step=%d layer=%d name=%s\n",
                tag ? tag : "<null>", micro_step, layer_idx, wname.c_str());
        trace_count++;
        return;
    }
    int nan = 0;
    float max_abs = 0.0f;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    for (float v : wvals) {
        if (std::isnan(v) || std::isinf(v)) {
            nan++;
            continue;
        }
        max_abs = std::max(max_abs, std::fabs(v));
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }
    fprintf(stderr,
            "[MOE_W_SAMPLE] tag=%s micro_step=%d layer=%d expert=%d nan=%d min=%.6f max=%.6f max_abs=%.6f dtype=%s ptr=%p\n",
            tag ? tag : "<null>", micro_step, layer_idx, expert_id, nan,
            min_v, max_v, max_abs, dtype_to_str(gw.DType), static_cast<void*>(gw.Data));
    trace_count++;
}

bool build_selective_info_from_offsets(const int* host_offsets,
                                       int num_experts,
                                       modules::SelectiveExpertInfo& selection) {
    if (!host_offsets || num_experts <= 0) {
        selection.reset();
        return false;
    }
    selection.reset();
    selection.enabled = true;
    selection.num_total = num_experts;
    selection.expert_to_compact.assign(num_experts, -1);
    selection.active_experts.reserve(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        if (host_offsets[e + 1] > host_offsets[e]) {
            selection.expert_to_compact[e] = static_cast<int>(selection.active_experts.size());
            selection.active_experts.push_back(e);
        }
    }
    selection.num_active = static_cast<int>(selection.active_experts.size());
    if (selection.num_active == 0) {
        selection.enabled = false;
        return false;
    }
    return true;
}

bool refresh_moe_experts_if_needed(int layer_idx,
                                   const int* host_offsets,
                                   int num_experts,
                                   DslParamStore& weights,
                                   cudaStream_t stream) {
    if (layer_idx < 0) {
        return false;
    }
    auto* provider = weights.qlora_provider();
    if (!provider) {
        return false;
    }
    modules::SelectiveExpertInfo selection;
    if (!build_selective_info_from_offsets(host_offsets, num_experts, selection)) {
        return false;
    }
    const bool refreshed = provider->refresh_moe_experts(layer_idx, selection, stream);
    static int moe_refresh_trace = 0;
    if (layer_idx == 2 && moe_refresh_trace < 8) {
        const int first = selection.active_experts.empty() ? -1 : selection.active_experts[0];
        fprintf(stderr,
                "[MOE_REFRESH] layer=%d num_active=%d first=%d refreshed=%d\n",
                layer_idx, selection.num_active, first, refreshed ? 1 : 0);
        moe_refresh_trace++;
    }
    return refreshed;
}

bool copy_tensor_token_sample_as_f32(const Tensor& t, long token_idx,
                                     std::size_t count, std::vector<float>& out) {
    out.assign(count, 0.0f);
    if (!t.Data || token_idx < 0 || t.Rank < 2) {
        return false;
    }
    std::size_t row_width = 0;
    if (t.Rank == 2) {
        row_width = static_cast<std::size_t>(t.Sizes[1]);
    } else if (t.Rank >= 3) {
        row_width = 1;
        for (int i = 2; i < t.Rank; ++i) {
            row_width *= static_cast<std::size_t>(t.Sizes[i]);
        }
    }
    if (row_width == 0) {
        return false;
    }
    const std::size_t elem_offset = static_cast<std::size_t>(token_idx) * row_width;
    return copy_tensor_sample_offset_as_f32(t, elem_offset, count, out);
}

bool sample_has_nan_or_inf(const std::vector<float>& vals) {
    for (float v : vals) {
        if (std::isnan(v) || std::isinf(v)) {
            return true;
        }
    }
    return false;
}

bool tensor_sample_has_nan_or_inf(const Tensor& t, long token_idx) {
    std::vector<float> vals(4);
    const bool ok = copy_tensor_token_sample_as_f32(t, token_idx, vals.size(), vals);
    if (!ok) {
        return false;
    }
    return sample_has_nan_or_inf(vals);
}

bool log_tensor_scalar_at(const char* tag, const Tensor& t, long elem_idx) {
    if (!tag || !t.Data || elem_idx < 0) {
        return false;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    if (static_cast<std::size_t>(elem_idx) >= total) {
        return false;
    }
    std::vector<float> vals;
    if (!copy_tensor_sample_offset_as_f32(t, static_cast<std::size_t>(elem_idx), 1, vals)) {
        return false;
    }
    const float v = vals.empty() ? 0.0f : vals[0];
    fprintf(stderr, "[%s] idx=%ld val=%.6f dtype=%s\n", tag, elem_idx, v, dtype_to_str(t.DType));
    return true;
}

bool log_tensor_token_row_stats(const char* tag, const Tensor& t, long token_idx) {
    if (!tag || !t.Data || token_idx < 0) {
        return false;
    }
    std::size_t row_width = 0;
    if (t.Rank == 1) {
        if (static_cast<std::size_t>(token_idx) >= static_cast<std::size_t>(t.nelem())) {
            return false;
        }
        row_width = 1;
    } else if (t.Rank == 2) {
        row_width = static_cast<std::size_t>(t.Sizes[1]);
    } else if (t.Rank >= 3) {
        row_width = 1;
        for (int i = 2; i < t.Rank; ++i) {
            row_width *= static_cast<std::size_t>(t.Sizes[i]);
        }
    }
    if (row_width == 0) {
        return false;
    }
    std::vector<float> vals;
    if (!copy_tensor_token_sample_as_f32(t, token_idx, row_width, vals)) {
        return false;
    }
    std::size_t nan = 0;
    std::size_t inf = 0;
    float min_val = 0.0f;
    float max_val = 0.0f;
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    bool has_finite = false;
    std::size_t max_idx = 0;
    for (std::size_t i = 0; i < vals.size(); ++i) {
        const float v = vals[i];
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
            max_idx = i;
        }
    }
    const double mean_abs = vals.empty() ? 0.0 : (sum_abs / static_cast<double>(vals.size()));
    fprintf(stderr,
            "[%s] token=%ld n=%zu nan=%zu inf=%zu min=%.6f max=%.6f max_abs=%.6f max_idx=%zu mean_abs=%.6f\n",
            tag, token_idx, vals.size(), nan, inf, min_val, max_val, max_abs, max_idx, mean_abs);
    return true;
}

std::size_t tensor_row_width(const Tensor& t);

bool log_tensor_row_stats(const char* tag, const Tensor& t, long row_idx) {
    if (!tag || !t.Data || row_idx < 0) {
        return false;
    }
    const std::size_t row_width = tensor_row_width(t);
    if (row_width == 0) {
        return false;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    const std::size_t elem_offset = static_cast<std::size_t>(row_idx) * row_width;
    if (elem_offset + row_width > total) {
        return false;
    }
    std::vector<float> vals;
    if (!copy_tensor_sample_offset_as_f32(t, elem_offset, row_width, vals)) {
        return false;
    }
    std::size_t nan = 0;
    std::size_t inf = 0;
    float min_val = 0.0f;
    float max_val = 0.0f;
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    bool has_finite = false;
    std::size_t max_idx = 0;
    for (std::size_t i = 0; i < vals.size(); ++i) {
        const float v = vals[i];
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
            max_idx = i;
        }
    }
    const double mean_abs = vals.empty() ? 0.0 : (sum_abs / static_cast<double>(vals.size()));
    fprintf(stderr,
            "[%s] row=%ld n=%zu nan=%zu inf=%zu min=%.6f max=%.6f max_abs=%.6f max_idx=%zu mean_abs=%.6f\n",
            tag, row_idx, vals.size(), nan, inf, min_val, max_val, max_abs, max_idx, mean_abs);
    return true;
}

std::size_t tensor_row_width(const Tensor& t) {
    if (t.Rank <= 1) {
        return static_cast<std::size_t>(t.nelem());
    }
    std::size_t row_width = 1;
    for (int i = 1; i < t.Rank; ++i) {
        row_width *= static_cast<std::size_t>(t.Sizes[i]);
    }
    return row_width;
}

bool tensor_row_has_nan_or_inf(const Tensor& t, long token_idx, float* out_min, float* out_max) {
    if (!t.Data) {
        return false;
    }
    const std::size_t row_width = tensor_row_width(t);
    if (row_width == 0) {
        return false;
    }
    std::vector<float> vals(row_width);
    if (!copy_tensor_token_sample_as_f32(t, token_idx, row_width, vals)) {
        return false;
    }
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    bool has_nan = false;
    for (float v : vals) {
        if (std::isnan(v) || std::isinf(v)) {
            has_nan = true;
            continue;
        }
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
    }
    if (out_min) {
        *out_min = std::isfinite(min_val) ? min_val : 0.0f;
    }
    if (out_max) {
        *out_max = std::isfinite(max_val) ? max_val : 0.0f;
    }
    return has_nan;
}

bool find_first_nan_row(const Tensor& t, long* out_row, float* out_min, float* out_max) {
    if (t.Rank < 1) {
        return false;
    }
    const long rows = static_cast<long>(t.Sizes[0]);
    for (long r = 0; r < rows; ++r) {
        if (tensor_row_has_nan_or_inf(t, r, out_min, out_max)) {
            if (out_row) {
                *out_row = r;
            }
            return true;
        }
    }
    return false;
}

void log_tensor_mag(const char* tag,
                    int layer_idx,
                    const std::string& name,
                    const Tensor& t,
                    std::size_t max_elems) {
    static int mag_log_count = 0;
    if (!t.Data || mag_log_count >= 64) {
        return;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    const std::size_t n = std::min(max_elems, total);
    if (n == 0) {
        return;
    }
    std::vector<float> vals;
    if (!copy_tensor_sample_as_f32(t, n, vals)) {
        return;
    }
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    for (float v : vals) {
        if (std::isnan(v) || std::isinf(v)) {
            continue;
        }
        const float av = std::fabs(v);
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        sum_abs += static_cast<double>(av);
        if (av > max_abs) {
            max_abs = av;
        }
    }
    const double mean_abs = (n > 0) ? (sum_abs / static_cast<double>(n)) : 0.0;
    fprintf(stderr,
            "[TENSOR_MAG] tag=%s layer=%d name=%s ptr=%p n=%zu total=%zu l2=%.6e max_abs=%.6e mean_abs=%.6e dtype=%s\n",
            tag, layer_idx, name.c_str(), t.Data, n, total,
            std::sqrt(sum_sq), max_abs, mean_abs, dtype_to_str(t.DType));
    mag_log_count++;
}

void log_tensor_mag_unbounded(const char* tag,
                              int layer_idx,
                              const std::string& name,
                              const Tensor& t,
                              std::size_t max_elems) {
    if (!t.Data) {
        return;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    const std::size_t n = std::min(max_elems, total);
    if (n == 0) {
        return;
    }
    std::vector<float> vals;
    if (!copy_tensor_sample_as_f32(t, n, vals)) {
        return;
    }
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    for (float v : vals) {
        if (std::isnan(v) || std::isinf(v)) {
            continue;
        }
        const float av = std::fabs(v);
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        sum_abs += static_cast<double>(av);
        if (av > max_abs) {
            max_abs = av;
        }
    }
    const double mean_abs = (n > 0) ? (sum_abs / static_cast<double>(n)) : 0.0;
    fprintf(stderr,
            "[TENSOR_MAG_EX] tag=%s layer=%d name=%s ptr=%p n=%zu total=%zu l2=%.6e max_abs=%.6e mean_abs=%.6e dtype=%s\n",
            tag, layer_idx, name.c_str(), t.Data, n, total,
            std::sqrt(sum_sq), max_abs, mean_abs, dtype_to_str(t.DType));
}

namespace {
std::vector<std::byte*> g_qkv_dA_ptr_by_layer;
std::vector<int> g_qkv_dA_micro_by_layer;
}  // namespace

void log_nan_sample(const char* tag,
                    int layer_idx,
                    const std::string& name,
                    const Tensor& t,
                    long token_idx) {
    static int nan_log_count = 0;
    static bool first_logged = false;
    std::vector<float> vals(4);
    const bool ok = copy_tensor_token_sample_as_f32(t, token_idx, vals.size(), vals);
    if (!ok || !sample_has_nan_or_inf(vals)) {
        return;
    }
    const char* prefix = first_logged ? "[NAN_DETECT]" : "[FIRST_NAN]";
    if (!first_logged) {
        first_logged = true;
    }
    if (nan_log_count < 100) {
        fprintf(stderr,
                "%s tag=%s layer=%d name=%s token=%ld ptr=%p vals=%.6f,%.6f,%.6f,%.6f\n",
                prefix, tag, layer_idx, name.c_str(), token_idx, t.Data,
                vals[0], vals[1], vals[2], vals[3]);
        nan_log_count++;
    }
}

void log_tensor_stats(const char* tag,
                      int layer_idx,
                      const std::string& name,
                      const Tensor& t,
                      std::size_t max_elems) {
    static int stats_log_count = 0;
    if (stats_log_count >= 32 || !t.Data) {
        return;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    const std::size_t n = std::min(max_elems, total);
    if (n == 0) {
        return;
    }
    std::vector<float> vals;
    if (!copy_tensor_sample_as_f32(t, n, vals)) {
        return;
    }
    std::size_t nan_count = 0;
    std::size_t inf_count = 0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    for (float v : vals) {
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
    const double mean_abs = (n > 0) ? (sum_abs / static_cast<double>(n)) : 0.0;
    fprintf(stderr,
            "[TENSOR_STATS] tag=%s layer=%d name=%s ptr=%p n=%zu total=%zu nan=%zu inf=%zu max_abs=%.6f mean_abs=%.6f\n",
            tag, layer_idx, name.c_str(), t.Data, n, total, nan_count, inf_count, max_abs, mean_abs);
    stats_log_count++;
}

void log_tensor_stats_ex(const char* tag,
                         int layer_idx,
                         const std::string& name,
                         const Tensor& t,
                         std::size_t max_elems,
                         bool force) {
    static int stats_log_count = 0;
    static int forced_log_count = 0;
    if (!t.Data) {
        return;
    }
    if (!force && stats_log_count >= 16) {
        return;
    }
    if (force && forced_log_count >= 16) {
        return;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    const std::size_t n = std::min(max_elems, total);
    if (n == 0) {
        return;
    }
    std::vector<float> vals;
    if (!copy_tensor_sample_as_f32(t, n, vals)) {
        return;
    }
    std::size_t nan_count = 0;
    std::size_t inf_count = 0;
    std::size_t finite_count = 0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    float min_val = 0.0f;
    float max_val = 0.0f;
    bool has_finite = false;
    for (float v : vals) {
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (std::isinf(v)) {
            inf_count++;
            continue;
        }
        if (!has_finite) {
            min_val = v;
            max_val = v;
            has_finite = true;
        } else {
            if (v < min_val) {
                min_val = v;
            }
            if (v > max_val) {
                max_val = v;
            }
        }
        finite_count++;
        const float av = std::fabs(v);
        sum_abs += static_cast<double>(av);
        if (av > max_abs) {
            max_abs = av;
        }
    }
    const double mean_abs = (finite_count > 0) ? (sum_abs / static_cast<double>(finite_count)) : 0.0;
    fprintf(stderr,
            "[TENSOR_STATS_EX] tag=%s layer=%d name=%s ptr=%p n=%zu total=%zu nan=%zu inf=%zu "
            "min=%.6f max=%.6f max_abs=%.6f mean_abs=%.6f\n",
            tag, layer_idx, name.c_str(), t.Data, n, total, nan_count, inf_count,
            min_val, max_val, max_abs, mean_abs);
    if (force) {
        forced_log_count++;
    } else {
        stats_log_count++;
    }
}

void log_tensor_sample_stats(const char* tag,
                             const Tensor& t,
                             std::size_t elem_offset,
                             std::size_t sample_elems) {
    if (!t.Data) {
        return;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    if (total == 0 || sample_elems == 0 || elem_offset >= total) {
        return;
    }
    const std::size_t n = std::min(sample_elems, total - elem_offset);
    std::vector<float> vals;
    if (!copy_tensor_sample_offset_as_f32(t, elem_offset, n, vals)) {
        return;
    }
    std::size_t nan_count = 0;
    std::size_t inf_count = 0;
    std::size_t finite_count = 0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    float min_val = 0.0f;
    float max_val = 0.0f;
    bool has_finite = false;
    for (float v : vals) {
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (std::isinf(v)) {
            inf_count++;
            continue;
        }
        if (!has_finite) {
            min_val = v;
            max_val = v;
            has_finite = true;
        } else {
            if (v < min_val) {
                min_val = v;
            }
            if (v > max_val) {
                max_val = v;
            }
        }
        finite_count++;
        const float av = std::fabs(v);
        sum_abs += static_cast<double>(av);
        if (av > max_abs) {
            max_abs = av;
        }
    }
    const double mean_abs = (finite_count > 0) ? (sum_abs / static_cast<double>(finite_count)) : 0.0;
    fprintf(stderr,
            "[TENSOR_SAMPLE_STATS] tag=%s offset=%zu n=%zu total=%zu nan=%zu inf=%zu "
            "min=%.6f max=%.6f max_abs=%.6f mean_abs=%.6f\n",
            tag, elem_offset, n, total, nan_count, inf_count, min_val, max_val, max_abs, mean_abs);
}

float env_float(const char* name, float fallback) {
    if (!name || !*name) {
        return fallback;
    }
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    float out = std::strtof(value, &end);
    if (end == value) {
        return fallback;
    }
    return out;
}

int env_int(const char* name, int fallback) {
    if (!name || !*name) {
        return fallback;
    }
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    long out = std::strtol(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    return static_cast<int>(out);
}

bool infer_known_tensor_shape(std::string_view name,
                              const modules::ModelConfig& config,
                              long B,
                              long T,
                              std::vector<long>& shape) {
    if (starts_with(name, kSavedPrefix)) {
        name = name.substr(kSavedPrefix.size());
    }

    int layer_idx = -1;
    std::string field;
    if (parse_block_param(name, layer_idx, field)) {
        const long C = config.HiddenSize;
        const long D = config.IntermediateSize;
        const long Hq = config.NumQueryHeads;
        const long Hkv = config.NumKeyValHeads;
        const long Hs = config.head_size();
        const long QKV = config.qkv_channels();

        if (field == "ln1" || field == "ln2" || field == "att_out" || field == "mlp_down" ||
            field == "res_att" || field == "res_ffn") {
            shape = {B, T, C};
            return true;
        }
        if (field == "ln1_flat" || field == "ln2_flat" || field == "att_out_flat" || field == "mlp_down_flat") {
            shape = {B * T, C};
            return true;
        }
        if (field == "ln1_rstd" || field == "ln2_rstd") {
            shape = {B, T};
            return true;
        }
        if (field == "qkv" || field == "qkv_rope") {
            shape = {B, T, QKV};
            return true;
        }
        if (field == "qkv_flat" || field == "qkv_biased") {
            shape = {B * T, QKV};
            return true;
        }
        if (field == "q_rstd") {
            shape = {B, T, Hq};
            return true;
        }
        if (field == "k_rstd") {
            shape = {B, T, Hkv};
            return true;
        }
        if (field == "att") {
            shape = {B, T, Hq * Hs};
            return true;
        }
        if (field == "att_flat") {
            shape = {B * T, Hq * Hs};
            return true;
        }
        if (field == "lse") {
            shape = {B, Hq, T};
            return true;
        }
        if (field == "mlp_up") {
            shape = {B, T, 2 * D};
            return true;
        }
        if (field == "mlp_up_flat") {
            shape = {B * T, 2 * D};
            return true;
        }
        if (field == "swiglu") {
            shape = {B, T, D};
            return true;
        }
        if (field == "swiglu_flat") {
            shape = {B * T, D};
            return true;
        }
    }

    if (name == "x0" || name == "encoded" || name == "ln_final" || name == "xF" ||
        name == "final_residual" || name == "residual_final") {
        shape = {B, T, config.HiddenSize};
        return true;
    }
    if (name == "ln_final_rstd") {
        shape = {B, T};
        return true;
    }
    if (name == "token_ids" || name == "position_ids") {
        shape = {B, T};
        return true;
    }
    if (name == "targets" || name == "labels" || name == "loss" || name == "losses" || name == "d_loss") {
        shape = {B * T};
        return true;
    }

    return false;
}

}  // namespace

// ============================================================================
// Operation type conversion
// ============================================================================

CompiledOpType op_type_from_string(const std::string& op_type) {
    // Use a static lookup table for O(1) average case
    static const std::unordered_map<std::string, CompiledOpType> type_map = {
        {"embedding", CompiledOpType::Embedding},
        {"zeros", CompiledOpType::Zeros},
        {"fused_residual_rmsnorm", CompiledOpType::FusedResidualRMSNorm},
        {"view", CompiledOpType::View},
        {"add", CompiledOpType::Add},
        {"matmul", CompiledOpType::Matmul},
        {"matmul_bias", CompiledOpType::MatmulBias},
        {"bias_add", CompiledOpType::BiasAdd},
        {"swiglu", CompiledOpType::SwiGLU},
        {"silu", CompiledOpType::Silu},
        {"mul", CompiledOpType::Mul},
        {"matmul_swiglu", CompiledOpType::MatmulSwiGLU},
        {"qkv_qk_norm_rope", CompiledOpType::QKVQKNormRoPE},
        {"rope", CompiledOpType::RoPE},
        {"flash_attention", CompiledOpType::FlashAttention},
        {"flash_attention_qkv", CompiledOpType::FlashAttention},
        {"cross_entropy", CompiledOpType::CrossEntropyLoss},
        {"cross_entropy_loss", CompiledOpType::CrossEntropyLoss},
        {"fused_lm_head_loss", CompiledOpType::FusedLMHeadLoss},
        {"lm_head_loss", CompiledOpType::FusedLMHeadLoss},
        // MoE forward operations
        {"moe_softmax", CompiledOpType::MoESoftmax},
        {"moe_sigmoid", CompiledOpType::MoESigmoid},
        {"moe_topk", CompiledOpType::MoETopK},
        {"moe_permute", CompiledOpType::MoEPermute},
        {"moe_grouped_gemm_gate_up", CompiledOpType::MoEGroupedGemmGateUp},
        {"moe_grouped_gemm_down", CompiledOpType::MoEGroupedGemmDown},
        {"moe_unpermute", CompiledOpType::MoEUnpermute},
        // Backward operations
        {"view_backward", CompiledOpType::ViewBackward},
        {"add_backward", CompiledOpType::AddBackward},
        {"matmul_backward", CompiledOpType::MatmulBackward},
        {"bias_add_backward", CompiledOpType::BiasAddBackward},
        {"swiglu_backward", CompiledOpType::SwiGLUBackward},
        {"silu_backward", CompiledOpType::SiluBackward},
        {"mul_backward", CompiledOpType::MulBackward},
        {"matmul_swiglu_backward", CompiledOpType::MatmulSwiGLUBackward},
        {"rope_backward", CompiledOpType::RoPEBackward},
        {"qkv_qk_norm_rope_backward", CompiledOpType::QKVQKNormRoPEBackward},
        {"flash_attention_backward", CompiledOpType::FlashAttentionBackward},
        {"zeros_backward", CompiledOpType::ZerosBackward},
        {"fused_residual_rmsnorm_backward", CompiledOpType::FusedResidualRMSNormBackward},
        {"embedding_backward", CompiledOpType::EmbeddingBackward},
        {"cross_entropy_backward", CompiledOpType::CrossEntropyLossBackward},
        {"fused_lm_head_loss_backward", CompiledOpType::FusedLMHeadLossBackward},
        // MoE backward operations
        {"moe_softmax_backward", CompiledOpType::MoESoftmaxBackward},
        {"moe_sigmoid_backward", CompiledOpType::MoESigmoidBackward},
        {"moe_topk_backward", CompiledOpType::MoETopKBackward},
        {"moe_permute_backward", CompiledOpType::MoEPermuteBackward},
        {"moe_grouped_gemm_gate_up_backward", CompiledOpType::MoEGroupedGemmGateUpBackward},
        {"moe_grouped_gemm_down_backward", CompiledOpType::MoEGroupedGemmDownBackward},
        {"moe_unpermute_backward", CompiledOpType::MoEUnpermuteBackward},
    };

    auto it = type_map.find(op_type);
    return it != type_map.end() ? it->second : CompiledOpType::Unknown;
}

const char* op_type_to_string(CompiledOpType type) {
    switch (type) {
        case CompiledOpType::Embedding: return "embedding";
        case CompiledOpType::Zeros: return "zeros";
        case CompiledOpType::FusedResidualRMSNorm: return "fused_residual_rmsnorm";
        case CompiledOpType::View: return "view";
        case CompiledOpType::Add: return "add";
        case CompiledOpType::Matmul: return "matmul";
        case CompiledOpType::MatmulBias: return "matmul_bias";
        case CompiledOpType::BiasAdd: return "bias_add";
        case CompiledOpType::SwiGLU: return "swiglu";
        case CompiledOpType::Silu: return "silu";
        case CompiledOpType::Mul: return "mul";
        case CompiledOpType::MatmulSwiGLU: return "matmul_swiglu";
        case CompiledOpType::QKVQKNormRoPE: return "qkv_qk_norm_rope";
        case CompiledOpType::RoPE: return "rope";
        case CompiledOpType::FlashAttention: return "flash_attention";
        case CompiledOpType::CrossEntropyLoss: return "cross_entropy_loss";
        case CompiledOpType::FusedLMHeadLoss: return "fused_lm_head_loss";
        // MoE forward
        case CompiledOpType::MoESoftmax: return "moe_softmax";
        case CompiledOpType::MoESigmoid: return "moe_sigmoid";
        case CompiledOpType::MoETopK: return "moe_topk";
        case CompiledOpType::MoEPermute: return "moe_permute";
        case CompiledOpType::MoEGroupedGemmGateUp: return "moe_grouped_gemm_gate_up";
        case CompiledOpType::MoEGroupedGemmDown: return "moe_grouped_gemm_down";
        case CompiledOpType::MoEUnpermute: return "moe_unpermute";
        // Backward
        case CompiledOpType::ViewBackward: return "view_backward";
        case CompiledOpType::AddBackward: return "add_backward";
        case CompiledOpType::MatmulBackward: return "matmul_backward";
        case CompiledOpType::BiasAddBackward: return "bias_add_backward";
        case CompiledOpType::SwiGLUBackward: return "swiglu_backward";
        case CompiledOpType::SiluBackward: return "silu_backward";
        case CompiledOpType::MulBackward: return "mul_backward";
        case CompiledOpType::MatmulSwiGLUBackward: return "matmul_swiglu_backward";
        case CompiledOpType::RoPEBackward: return "rope_backward";
        case CompiledOpType::QKVQKNormRoPEBackward: return "qkv_qk_norm_rope_backward";
        case CompiledOpType::FlashAttentionBackward: return "flash_attention_backward";
        case CompiledOpType::ZerosBackward: return "zeros_backward";
        case CompiledOpType::FusedResidualRMSNormBackward: return "fused_residual_rmsnorm_backward";
        case CompiledOpType::EmbeddingBackward: return "embedding_backward";
        case CompiledOpType::CrossEntropyLossBackward: return "cross_entropy_backward";
        case CompiledOpType::FusedLMHeadLossBackward: return "fused_lm_head_loss_backward";
        // MoE backward
        case CompiledOpType::MoESoftmaxBackward: return "moe_softmax_backward";
        case CompiledOpType::MoESigmoidBackward: return "moe_sigmoid_backward";
        case CompiledOpType::MoETopKBackward: return "moe_topk_backward";
        case CompiledOpType::MoEPermuteBackward: return "moe_permute_backward";
        case CompiledOpType::MoEGroupedGemmGateUpBackward: return "moe_grouped_gemm_gate_up_backward";
        case CompiledOpType::MoEGroupedGemmDownBackward: return "moe_grouped_gemm_down_backward";
        case CompiledOpType::MoEUnpermuteBackward: return "moe_unpermute_backward";
        case CompiledOpType::Unknown: return "unknown";
    }
    return "unknown";
}

// ============================================================================
// GraphCompiler implementation
// ============================================================================

GraphCompiler::GraphCompiler(const Module& module,
                             const modules::ModelConfig& config,
                             const RuntimeOptions& options,
                             DslParamStore& weights,
                             DslGradStore& grads)
    : mModule(module)
    , mConfig(config)
    , mOptions(options)
    , mWeights(weights)
    , mGrads(grads)
{
    // Initialize slot registry from DSL layout (no built-in fallback - all slots must be
    // explicitly declared in Python DSL)
    if (mModule.activation_layout.has_value()) {
        mSlotRegistry.init_from_layout(*mModule.activation_layout);
    }
    // If no layout, registry remains empty - all tensors will use Mapped slot
}

void GraphCompiler::update_dimensions(long B, long T) {
    mB = B;
    mT = T;

    // Use make_shape_env + augment_shape_env to get the same symbols
    // as the non-compiled execution path. This ensures DSL IR symbol names
    // (e.g., d_model, hidden_size, num_query_heads) are available.
    mShapeEnv = make_shape_env(mModule, B, T);
    augment_shape_env(mShapeEnv, mModule.config);

    // Also ensure standard short symbols from ModelConfig are present
    // (in case DSL IR uses the canonical short names)
    mShapeEnv.values["C"] = mConfig.HiddenSize;
    mShapeEnv.values["D"] = mConfig.head_size();
    const long moe_m = (mConfig.MoeIntermediateSize > 0)
        ? mConfig.MoeIntermediateSize
        : mConfig.IntermediateSize;
    mShapeEnv.values["M"] = moe_m;
    mShapeEnv.values["MUp"] = 2 * moe_m;
    mShapeEnv.values["V"] = mConfig.VocabSize;
    mShapeEnv.values["Hq"] = mConfig.NumQueryHeads;
    mShapeEnv.values["Hkv"] = mConfig.NumKeyValHeads;
    mShapeEnv.values["QKV"] = mConfig.qkv_channels();
    mShapeEnv.values["AttnDim"] = mConfig.NumQueryHeads * mConfig.head_size();

    // MoE dimensions
    if (mConfig.NumExperts > 0) {
        mShapeEnv.values["E"] = mConfig.NumExperts;
    }
    if (mConfig.NumExpertsPerTok > 0) {
        mShapeEnv.values["K"] = mConfig.NumExpertsPerTok;
    }
    // Shared expert intermediate size (default to regular intermediate size)
    if (mConfig.moe_config.has_value() && mConfig.moe_config->shared_expert_size > 0) {
        mShapeEnv.values["SharedM"] = mConfig.moe_config->shared_expert_size;
        mShapeEnv.values["SharedMUp"] = 2 * mConfig.moe_config->shared_expert_size;
    } else {
        mShapeEnv.values["SharedM"] = mConfig.IntermediateSize;
        mShapeEnv.values["SharedMUp"] = 2 * mConfig.IntermediateSize;
    }
}

CompiledOpType GraphCompiler::classify_op(const std::string& op_type) const {
    return op_type_from_string(op_type);
}

TensorRef GraphCompiler::resolve_tensor_ref(const std::string& name, bool is_output,
                                            const Operation& op, const ShapeEnv& env) {
    TensorRef ref;
    ref.name = name;

    // Check for saved tensor prefix
    std::string effective_name = name;
    if (starts_with(name, kSavedPrefix)) {
        const std::string stripped = std::string(name.substr(kSavedPrefix.size()));
        ref.slot = TensorSlot::Saved;
        ref.name = stripped;
        // Populate shape/dtype from DSL slot registry when available.
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(stripped, layer_idx, field)) {
            ref.layer_idx = layer_idx;
            const std::string base_field = strip_ssa_suffix(field);
            if (auto slot_entry = mSlotRegistry.lookup(base_field)) {
                if (!slot_entry->shape.empty()) {
                    ref.shape = resolve_shape(slot_entry->shape, mShapeEnv);
                }
                if (slot_entry->dtype.has_value()) {
                    ref.dtype = *slot_entry->dtype;
                }
            }
        } else if (auto slot_entry = mSlotRegistry.lookup(strip_ssa_suffix(stripped))) {
            if (!slot_entry->shape.empty()) {
                ref.shape = resolve_shape(slot_entry->shape, mShapeEnv);
            }
            if (slot_entry->dtype.has_value()) {
                ref.dtype = *slot_entry->dtype;
            }
        }
        if (ref.shape.empty()) {
            auto it = mExtraShapes.find(ref.name);
            if (it != mExtraShapes.end()) {
                ref.shape = it->second;
            }
        }
        return ref;
    }

    // Check for block-indexed tensors
    int layer_idx = -1;
    std::string field;
    if (parse_block_param(effective_name, layer_idx, field)) {
        ref.layer_idx = layer_idx;

        // Strip SSA-style numeric suffix (e.g., "qkv_rope_7" -> "qkv_rope")
        // The DSL IR generates unique tensor names with suffixes like _0, _7, _10, etc.
        const std::string base_field = strip_ssa_suffix(field);

        // Map field to slot using the registry (supports both built-in and DSL-defined slots)
        if (auto slot_entry = mSlotRegistry.lookup(base_field)) {
            ref.slot = slot_entry->slot;

            // Handle global slots that appear with block indices (e.g., rope_freqs)
            if (slot_entry->scope == ActivationScope::Global) {
                ref.layer_idx = -1;  // Global, not layer-indexed
                return ref;
            }

            // Use shape from DSL if available
            if (!slot_entry->shape.empty()) {
                ref.shape = resolve_shape(slot_entry->shape, mShapeEnv);
            }
            // Override with extra shapes when present (e.g., view outputs with explicit shapes).
            if (auto it = mExtraShapes.find(ref.name); it != mExtraShapes.end()) {
                ref.shape = it->second;
            }
        } else if (mWeights.has(effective_name)) {
            // Block-indexed weight (e.g., blocks[0].ln1_weight)
            ref.slot = TensorSlot::Parameter;
            return ref;
        } else {
            ref.slot = TensorSlot::Mapped;
        }

        return ref;
    }

    // Check for gradient tensors
    if (starts_with(name, "d_")) {
        const std::string base = name.substr(2);
        if (parse_block_param(base, layer_idx, field)) {
            ref.layer_idx = layer_idx;

            // Look up gradient slot using "d_<field>" name (e.g., "d_ln1", "d_qkv")
            const std::string grad_name = "d_" + strip_ssa_suffix(field);
            if (auto slot_entry = mSlotRegistry.lookup(grad_name)) {
                ref.slot = slot_entry->slot;
                // Use shape from DSL if available
                if (!slot_entry->shape.empty()) {
                    ref.shape = resolve_shape(slot_entry->shape, mShapeEnv);
                }
            } else {
                // Try looking up the activation slot and use its shape for the gradient
                const std::string act_name = strip_ssa_suffix(field);
                if (auto act_entry = mSlotRegistry.lookup(act_name)) {
                    if (!act_entry->shape.empty()) {
                        ref.shape = resolve_shape(act_entry->shape, mShapeEnv);
                    }
                }
                ref.slot = TensorSlot::Mapped;
            }

            if (ref.shape.empty()) {
                const std::string base = name.substr(2);
                auto it = mExtraShapes.find(base);
                if (it != mExtraShapes.end()) {
                    ref.shape = it->second;
                }
            }
            if (std::getenv("SUROGATE_TRACE_MOE_SHAPES") &&
                (base.find("permuted_input") != std::string::npos ||
                 base.find("expert_gate_up") != std::string::npos ||
                 base.find("expert_act") != std::string::npos ||
                 base.find("expert_down") != std::string::npos ||
                 base.find("moe_out") != std::string::npos)) {
                std::ostringstream oss;
                oss << "[MOE_SHAPE_REF] name=" << name << " shape=[";
                for (std::size_t i = 0; i < ref.shape.size(); ++i) {
                    if (i) oss << ",";
                    oss << ref.shape[i];
                }
                oss << "] slot=" << static_cast<int>(ref.slot) << "\n";
                std::fputs(oss.str().c_str(), stderr);
            }
            return ref;
        }
    }

    // Check for global tensors using registry (supports built-in and DSL-defined slots)
    if (auto slot_entry = mSlotRegistry.lookup(name)) {
        ref.slot = slot_entry->slot;
        // Apply dtype override from registry if specified
        if (slot_entry->dtype.has_value()) {
            ref.dtype = *slot_entry->dtype;
        }
    } else if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
        // Substring match for rope frequencies (handles qualified names)
        ref.slot = TensorSlot::FreqCis;
    } else if (mWeights.has(name)) {
        ref.slot = TensorSlot::Parameter;
    } else {
        ref.slot = TensorSlot::Mapped;
    }

    if (ref.shape.empty()) {
        std::vector<long> resolved;
        if (resolve_tensor_shape(ref.name, resolved)) {
            ref.shape = std::move(resolved);
        } else {
            auto it = mExtraShapes.find(ref.name);
            if (it != mExtraShapes.end()) {
                ref.shape = it->second;
            }
        }
    }
    if (auto it = mTensorDtypes.find(ref.name); it != mTensorDtypes.end()) {
        ref.dtype = it->second;
    }
    return ref;
}

CompiledAttrs GraphCompiler::resolve_attrs(const Operation& op, CompiledOpType type,
                                           const ShapeEnv& env) {
    CompiledAttrs attrs;

    // Epsilon for normalization ops
    if (auto* eps_attr = find_attr(op.attrs, "eps")) {
        if (auto v = attr_double(*eps_attr)) {
            attrs.eps = static_cast<float>(*v);
        }
    } else {
        attrs.eps = static_cast<float>(mConfig.RmsNormEps);
    }

    // Transpose mode for matmul ops
    attrs.transpose = parse_transpose(op.attrs);

    // Rotary dimension for RoPE
    if (auto* rd_attr = find_attr(op.attrs, "rotary_dim")) {
        if (auto v = attr_int(*rd_attr)) {
            attrs.rotary_dim = static_cast<int>(*v);
        } else if (auto s = attr_string(*rd_attr)) {
            attrs.rotary_dim = static_cast<int>(resolve_dim(Dim::symbolic(*s), env));
        }
    } else {
        attrs.rotary_dim = mConfig.head_size();
    }

    // Shape attribute (direct shape or shape_like reference)
    if (auto* shape_attr = find_attr(op.attrs, "shape")) {
        attrs.shape = resolve_attr_shape(*shape_attr, env);
    } else if (auto* shape_like_attr = find_attr(op.attrs, "shape_like")) {
        // Store the reference name for runtime lookup
        if (auto ref_name = attr_string(*shape_like_attr)) {
            attrs.shape_like = *ref_name;
        }
    }

    if (auto* acc_attr = find_attr(op.attrs, "compute_accuracy")) {
        if (auto v = attr_bool(*acc_attr)) {
            attrs.compute_accuracy = *v;
        }
    }

    // Matmul-specific attributes
    if (type == CompiledOpType::Matmul || type == CompiledOpType::MatmulBias) {
        if (op.inputs.size() > 1) {
            int layer_idx = -1;
            auto matmul_op = matmul_op_from_weight(op.inputs[1], layer_idx);
            attrs.matmul_op = matmul_op;
            attrs.layer_idx = layer_idx;
            attrs.allow_quant = matmul_op.has_value() &&
                                allow_quant_layer(mOptions, mConfig, layer_idx);
            if (matmul_op.has_value()) {
                switch (*matmul_op) {
                    case modules::MatmulOp::QKV:
                        attrs.forward_hook_point = modules::ForwardHookPoint::AfterQKVProjection;
                        break;
                    case modules::MatmulOp::AttnOut:
                        attrs.forward_hook_point = modules::ForwardHookPoint::AfterAttnOutProjection;
                        break;
                    case modules::MatmulOp::MLPUp:
                        attrs.forward_hook_point = modules::ForwardHookPoint::AfterMLPUpProjection;
                        break;
                    case modules::MatmulOp::MLPDown:
                        attrs.forward_hook_point = modules::ForwardHookPoint::AfterMLPDownProjection;
                        break;
                    default:
                        break;
                }
            }
        }
    }

    // MatmulBackward: weight is at inputs[2], not inputs[1]
    // Also set backward_hook_point for LoRA hook invocation
    if (type == CompiledOpType::MatmulBackward) {
        if (op.inputs.size() > 2) {
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(op.inputs[2], layer_idx, field)) {
                // Set matmul_op and layer_idx
                if (field == "qkv_weight") {
                    attrs.matmul_op = modules::MatmulOp::QKV;
                    attrs.backward_hook_point = modules::BackwardHookPoint::AfterQKVBackward;
                } else if (field == "out_weight") {
                    attrs.matmul_op = modules::MatmulOp::AttnOut;
                    attrs.backward_hook_point = modules::BackwardHookPoint::AfterAttnOutBackward;
                } else if (field == "mlp_up_weight") {
                    attrs.matmul_op = modules::MatmulOp::MLPUp;
                    attrs.backward_hook_point = modules::BackwardHookPoint::AfterMLPUpBackward;
                } else if (field == "mlp_down_weight") {
                    attrs.matmul_op = modules::MatmulOp::MLPDown;
                    attrs.backward_hook_point = modules::BackwardHookPoint::AfterMLPDownBackward;
                }
                attrs.layer_idx = layer_idx;
                attrs.allow_quant = attrs.matmul_op.has_value() &&
                                    allow_quant_layer(mOptions, mConfig, layer_idx);
            }
        }
    }

    // MatmulSwiGLUBackward: fused MLP up+gate backward uses weight at inputs[2]
    // Set backward_hook_point so LoRA can hook into MLPUp gradients.
    if (type == CompiledOpType::MatmulSwiGLUBackward) {
        if (op.inputs.size() > 2) {
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(op.inputs[2], layer_idx, field)) {
                if (field == "mlp_up_weight") {
                    attrs.matmul_op = modules::MatmulOp::MLPUp;
                    attrs.backward_hook_point = modules::BackwardHookPoint::AfterMLPUpBackward;
                }
                attrs.layer_idx = layer_idx;
            }
        }
    }

    // MoE-specific attributes
    if (type == CompiledOpType::MoETopK || type == CompiledOpType::MoEPermute ||
        type == CompiledOpType::MoEUnpermute || type == CompiledOpType::MoETopKBackward ||
        type == CompiledOpType::MoEPermuteBackward || type == CompiledOpType::MoEUnpermuteBackward) {
        // top_k attribute
        if (auto* top_k_attr = find_attr(op.attrs, "top_k")) {
            if (auto v = attr_int(*top_k_attr)) {
                attrs.top_k = static_cast<int>(*v);
            }
        } else {
            // Default from model config
            attrs.top_k = static_cast<int>(mConfig.NumExpertsPerTok);
        }

        // normalize_weights attribute
        if (auto* norm_attr = find_attr(op.attrs, "normalize")) {
            if (auto v = attr_bool(*norm_attr)) {
                attrs.normalize_weights = *v;
            }
        }
    }

    return attrs;
}

void GraphCompiler::annotate_layer_boundaries(CompiledGraph& graph) {
    graph.layer_start_indices.resize(mConfig.NumLayers, SIZE_MAX);
    graph.layer_end_indices.resize(mConfig.NumLayers, SIZE_MAX);

    int current_layer = -1;
    std::size_t layer_start = 0;

    auto is_grad_ref = [](const TensorRef& ref) -> bool {
        if (!ref.name.empty() && ref.name.size() > 2 && ref.name[0] == 'd' && ref.name[1] == '_') {
            return true;
        }
        switch (ref.slot) {
            case TensorSlot::BlockDLN1:
            case TensorSlot::BlockDQKV:
            case TensorSlot::BlockDAtt:
            case TensorSlot::BlockDSwiGLU:
            case TensorSlot::BlockDMLPUp:
            case TensorSlot::BlockDMLPDown:
            case TensorSlot::BlockDLN2:
            case TensorSlot::BlockDResAtt:
            case TensorSlot::BlockDResFFN:
            case TensorSlot::DLoss:
                return true;
            default:
                return false;
        }
    };

    auto ref_layer_idx = [&](const TensorRef& ref) -> int {
        if (ref.layer_idx >= 0) {
            return ref.layer_idx;
        }
        if (ref.name.empty()) {
            return -1;
        }
        std::string_view name = ref.name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return layer_idx;
        }
        return -1;
    };

    auto ref_layer_idx_any = [&](const TensorRef& ref) -> int {
        if (ref.layer_idx >= 0) {
            return ref.layer_idx;
        }
        if (ref.name.empty()) {
            return -1;
        }
        std::string_view name = ref.name;
        if (name.rfind("d_", 0) == 0) {
            name.remove_prefix(2);
        }
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return layer_idx;
        }
        return -1;
    };

    for (std::size_t i = 0; i < graph.ops.size(); ++i) {
        auto& op = graph.ops[i];

        // Check inputs/outputs for layer index. Use the highest layer index found,
        // since some ops (e.g., LN1 fused residual) consume previous-layer tensors
        // but are parameterized by the current layer's weights.
        int detected_layer = -1;
        for (const auto& ref : op.inputs) {
            if (is_grad_ref(ref)) {
                continue;
            }
            const int layer_idx = ref_layer_idx(ref);
            if (layer_idx >= 0) {
                detected_layer = std::max(detected_layer, layer_idx);
            }
        }
        for (const auto& ref : op.outputs) {
            if (is_grad_ref(ref)) {
                continue;
            }
            const int layer_idx = ref_layer_idx(ref);
            if (layer_idx >= 0) {
                detected_layer = std::max(detected_layer, layer_idx);
            }
        }
        if (detected_layer < 0) {
            for (const auto& ref : op.inputs) {
                const int layer_idx = ref_layer_idx_any(ref);
                if (layer_idx >= 0) {
                    detected_layer = std::max(detected_layer, layer_idx);
                }
            }
            for (const auto& ref : op.outputs) {
                const int layer_idx = ref_layer_idx_any(ref);
                if (layer_idx >= 0) {
                    detected_layer = std::max(detected_layer, layer_idx);
                }
            }
        }
        if (op.attrs.layer_idx >= 0) {
            detected_layer = std::max(detected_layer, op.attrs.layer_idx);
        }

        if (detected_layer >= 0 && detected_layer != current_layer) {
            // End previous layer
            if (current_layer >= 0 && current_layer < static_cast<int>(mConfig.NumLayers)) {
                graph.layer_end_indices[current_layer] = i;
                graph.ops[i - 1].layer_end = current_layer;
            }

            // Start new layer
            current_layer = detected_layer;
            if (current_layer < static_cast<int>(mConfig.NumLayers)) {
                graph.layer_start_indices[current_layer] = i;
                op.layer_start = current_layer;
            }
        }
    }

    // End final layer
    if (current_layer >= 0 && current_layer < static_cast<int>(mConfig.NumLayers)) {
        graph.layer_end_indices[current_layer] = graph.ops.size();
        if (!graph.ops.empty()) {
            graph.ops.back().layer_end = current_layer;
        }
    }
}

// ============================================================================
// Shape Validation Methods
// ============================================================================

bool GraphCompiler::resolve_tensor_shape(const std::string& name, std::vector<long>& shape) {
    // Check shape cache first
    auto it = mTensorShapes.find(name);
    if (it != mTensorShapes.end()) {
        shape = it->second.dims;
        return true;
    }

    // Check IR tensor info
    auto check_tensor_info = [&](const std::unordered_map<std::string, TensorInfo>& tensors) {
        auto it = tensors.find(name);
        if (it != tensors.end() && !it->second.shape.empty()) {
            shape = resolve_shape(it->second.shape, mShapeEnv);
            TensorShape ts;
            ts.dims = shape;
            ts.inferred = false;
            mTensorShapes[name] = ts;
            return true;
        }
        return false;
    };

    // Check in graph tensors
    if (check_tensor_info(mModule.forward->inputs)) return true;
    if (check_tensor_info(mModule.forward->outputs)) return true;
    if (check_tensor_info(mModule.forward->params)) return true;
    if (check_tensor_info(mModule.forward->intermediates)) return true;

    // Try pattern-based inference for known tensor names
    if (infer_known_tensor_shape(name, mConfig, mB, mT, shape)) {
        TensorShape ts;
        ts.dims = shape;
        ts.inferred = true;
        mTensorShapes[name] = ts;
        return true;
    }

    // Check for saved tensors (use base name)
    if (starts_with(name, kSavedPrefix)) {
        std::string base_name = std::string(name.substr(kSavedPrefix.size()));
        return resolve_tensor_shape(base_name, shape);
    }

    return false;
}

void GraphCompiler::infer_output_shapes(
    const Operation& op,
    CompiledOpType type,
    const std::vector<std::vector<long>>& input_shapes,
    std::vector<std::vector<long>>& output_shapes) {

    output_shapes.clear();

    // Infer output shapes based on operation type
    switch (type) {
        case CompiledOpType::Matmul:
        case CompiledOpType::MatmulBias: {
            if (input_shapes.size() >= 2 && !input_shapes[0].empty() && !input_shapes[1].empty()) {
                const auto& a_shape = input_shapes[0];
                const auto& b_shape = input_shapes[1];

                // Parse transpose mode
                EMMTranspose mode = parse_transpose(op.attrs);

                // Compute output shape
                std::vector<long> out_shape;

                // Batch dims (min of both inputs)
                size_t min_rank = std::min(a_shape.size(), b_shape.size());
                for (size_t i = 0; i + 2 < min_rank; ++i) {
                    out_shape.push_back(a_shape[i]);
                }

                // M and N dimensions
                if (mode == EMMTranspose::NN || mode == EMMTranspose::NT) {
                    out_shape.push_back(a_shape[a_shape.size() - 2]);  // M
                } else {
                    out_shape.push_back(a_shape[a_shape.size() - 1]);  // M (transposed)
                }

                if (mode == EMMTranspose::NN || mode == EMMTranspose::TN) {
                    out_shape.push_back(b_shape[b_shape.size() - 1]);  // N
                } else {
                    out_shape.push_back(b_shape[b_shape.size() - 2]);  // N (transposed)
                }

                output_shapes.push_back(out_shape);
            }
            break;
        }

        case CompiledOpType::View: {
            // Output shape from attributes
            if (auto* shape_attr = find_attr(op.attrs, "shape")) {
                auto out_shape = resolve_attr_shape(*shape_attr, mShapeEnv);
                output_shapes.push_back(out_shape);
            }
            break;
        }

        case CompiledOpType::Add: {
            // Output shape = broadcast(input shapes)
            if (!input_shapes.empty()) {
                output_shapes.push_back(input_shapes[0]);  // Simplified: assume same shape
            }
            break;
        }

        case CompiledOpType::SwiGLU: {
            // Output last dim = input last dim / 2
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                auto out_shape = input_shapes[0];
                out_shape.back() = out_shape.back() / 2;
                output_shapes.push_back(out_shape);
            }
            break;
        }

        case CompiledOpType::Embedding: {
            // Output = indices_shape + [embedding_dim]
            if (input_shapes.size() >= 2 && !input_shapes[1].empty()) {
                auto out_shape = input_shapes[0];  // indices shape
                out_shape.push_back(input_shapes[1][1]);  // embedding dim
                output_shapes.push_back(out_shape);
            }
            break;
        }

        case CompiledOpType::CrossEntropyLoss: {
            // Output: per-token loss [B*T]
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                const auto& logits_shape = input_shapes[0];
                if (!logits_shape.empty()) {
                    output_shapes.push_back({logits_shape[0]});
                }
            }
            break;
        }

        case CompiledOpType::CrossEntropyLossBackward: {
            // Output: d_logits shape matches logits input
            if (input_shapes.size() > 1 && !input_shapes[1].empty()) {
                output_shapes.push_back(input_shapes[1]);
            }
            break;
        }

        case CompiledOpType::FusedLMHeadLoss: {
            // Output: per-token loss [B*T]
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back({input_shapes[0][0]});
            }
            break;
        }

        case CompiledOpType::FusedLMHeadLossBackward: {
            // Outputs: d_xF_flat [B*T, C], d_lm_head [V, C]
            if (input_shapes.size() > 1 && !input_shapes[1].empty()) {
                output_shapes.push_back(input_shapes[1]);
            }
            if (input_shapes.size() > 2 && !input_shapes[2].empty()) {
                output_shapes.push_back(input_shapes[2]);
            }
            break;
        }

        case CompiledOpType::Zeros: {
            // Try to infer from 'shape' attribute
            if (auto* shape_attr = find_attr(op.attrs, "shape")) {
                auto out_shape = resolve_attr_shape(*shape_attr, mShapeEnv);
                output_shapes.push_back(out_shape);
            }
            break;
        }

        case CompiledOpType::RoPE: {
            // RoPE output shape matches input qkv shape
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);
            }
            break;
        }

        case CompiledOpType::FlashAttention: {
            // FlashAttention outputs: attn_out [B, T, Hq, D], lse [B, Hq, T]
            // Cannot infer output shape from input qkv [B, T, Hq+2*Hkv, D] without
            // knowing Hq and Hkv separately. Leave shapes uninferred.
            break;
        }

        case CompiledOpType::FusedResidualRMSNorm: {
            // Outputs: residual_out [B,T,C], y [B,T,C], rstd [B,T]
            if (input_shapes.size() >= 2 && !input_shapes[0].empty() && !input_shapes[1].empty()) {
                output_shapes.push_back(input_shapes[0]);  // residual_out same as input[0]
                output_shapes.push_back(input_shapes[1]);  // y same as input[1]
                // rstd drops the last dimension
                auto rstd_shape = input_shapes[0];
                if (!rstd_shape.empty()) {
                    rstd_shape.pop_back();
                }
                output_shapes.push_back(rstd_shape);
            }
            break;
        }

        case CompiledOpType::Silu:
        case CompiledOpType::Mul: {
            // Element-wise ops preserve shape
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);
            }
            break;
        }

        case CompiledOpType::QKVQKNormRoPE: {
            // Output qkv_rope has same shape as input qkv
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);  // qkv_rope
                // q_rstd and k_rstd shapes - hard to infer without config
                output_shapes.push_back({});
                output_shapes.push_back({});
            }
            break;
        }

        case CompiledOpType::MoESigmoid:
        case CompiledOpType::MoESoftmax: {
            // Output same shape as input
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                output_shapes.push_back(input_shapes[0]);
            }
            break;
        }

        case CompiledOpType::MoETopK: {
            // Output: routing_weights [B*T, K], routing_indices [B*T, K]
            if (!input_shapes.empty() && !input_shapes[0].empty()) {
                int top_k = 1;
                if (auto* attr = find_attr(op.attrs, "top_k")) {
                    if (auto v = attr_int(*attr)) {
                        top_k = static_cast<int>(*v);
                    }
                }
                std::vector<long> out_shape = {input_shapes[0][0], static_cast<long>(top_k)};
                output_shapes.push_back(out_shape);  // routing_weights
                output_shapes.push_back(out_shape);  // routing_indices
            }
            break;
        }

        case CompiledOpType::MoEPermute: {
            // permuted_input shape depends on scatter_indices, hard to infer statically
            break;
        }

        case CompiledOpType::MoEGroupedGemmGateUp: {
            // Output shape is [total_tokens, 2*M] but total_tokens is dynamic
            break;
        }

        case CompiledOpType::MoEGroupedGemmDown: {
            // Output shape is [total_tokens, C] but total_tokens is dynamic
            break;
        }

        case CompiledOpType::MoEUnpermute: {
            // Output shape [B*T, C] - based on routing structure
            break;
        }

        default:
            // For other operations, output shape not inferred
            break;
    }
}

void GraphCompiler::validate_operation_shapes(
    const Operation& op,
    CompiledOpType type,
    size_t op_index) {

    using namespace shape_checker;

    // Check if validation is disabled via environment variable
    if (env_enabled("SUROGATE_NO_SHAPE_CHECK")) {
        return;
    }

    // Get operation signature
    const auto* sig = OpShapeRegistry::instance().get_signature(op.name);
    if (!sig) {
        // No signature registered - skip validation (only warn in verbose mode)
        return;
    }

    // Resolve input shapes
    std::vector<std::vector<long>> input_shapes;
    input_shapes.reserve(op.inputs.size());
    std::vector<std::string> unresolved_inputs;

    for (const auto& input_name : op.inputs) {
        std::vector<long> shape;
        if (!resolve_tensor_shape(input_name, shape)) {
            unresolved_inputs.push_back(input_name);
            input_shapes.push_back({});  // Empty shape
        } else {
            input_shapes.push_back(shape);
        }
    }

    // If we couldn't resolve some input shapes, we can't validate
    if (!unresolved_inputs.empty()) {
        // Skip validation when input shapes are unknown
        return;
    }

    // Resolve or infer output shapes
    std::vector<std::vector<long>> output_shapes;
    output_shapes.reserve(op.outputs.size());

    for (size_t i = 0; i < op.outputs.size(); ++i) {
        const auto& output_name = op.outputs[i];
        std::vector<long> shape;

        if (resolve_tensor_shape(output_name, shape)) {
            // Shape already known (from IR or previous inference)
            output_shapes.push_back(shape);
        } else {
            // Try to infer from operation semantics
            std::vector<std::vector<long>> inferred_outputs;
            infer_output_shapes(op, type, input_shapes, inferred_outputs);

            if (i < inferred_outputs.size() && !inferred_outputs[i].empty()) {
                shape = inferred_outputs[i];
                output_shapes.push_back(shape);

                // Store inferred shape for future operations
                TensorShape ts;
                ts.dims = shape;
                ts.inferred = true;
                ts.source_op = op.id;
                mTensorShapes[output_name] = ts;
            } else {
                output_shapes.push_back({});  // Unknown shape
            }
        }
    }

    // Run validator
    if (sig->validator) {
        auto error = sig->validator(input_shapes, output_shapes, op.attrs, mShapeEnv);
        if (error) {
            // Build detailed error message
            std::ostringstream oss;
            oss << "\n╔═══════════════════════════════════════════════════════╗\n"
                <<   "║ Found Shape Validation Error during Graph Compilation ║\n"
                <<   "╚═══════════════════════════════════════════════════════╝\n\n"
                <<   "Operation: #" << op_index << " (id: '" << op.id << "')\n"
                <<   "Type:      " << op.name << "\n\n";

            // Show operation attributes if any
            bool has_attrs = false;
            std::ostringstream attrs_oss;
            if (op.attrs.find("transpose") != op.attrs.end()) {
                if (std::holds_alternative<std::string>(op.attrs.at("transpose").value)) {
                    attrs_oss << "transpose=" << std::get<std::string>(op.attrs.at("transpose").value) << " ";
                    has_attrs = true;
                }
            }
            if (op.attrs.find("eps") != op.attrs.end()) {
                if (std::holds_alternative<double>(op.attrs.at("eps").value)) {
                    attrs_oss << "eps=" << std::get<double>(op.attrs.at("eps").value) << " ";
                    has_attrs = true;
                }
            }
            if (op.attrs.find("rotary_dim") != op.attrs.end()) {
                if (std::holds_alternative<std::int64_t>(op.attrs.at("rotary_dim").value)) {
                    attrs_oss << "rotary_dim=" << std::get<std::int64_t>(op.attrs.at("rotary_dim").value) << " ";
                    has_attrs = true;
                }
            }
            if (op.attrs.find("layer_idx") != op.attrs.end()) {
                if (std::holds_alternative<std::int64_t>(op.attrs.at("layer_idx").value)) {
                    attrs_oss << "layer_idx=" << std::get<std::int64_t>(op.attrs.at("layer_idx").value) << " ";
                    has_attrs = true;
                }
            }
            if (has_attrs) {
                oss << "Attributes: " << attrs_oss.str() << "\n\n";
            }

            oss << "Inputs:\n";
            if (op.inputs.empty()) {
                oss << "  (none)\n";
            } else {
                for (size_t i = 0; i < op.inputs.size(); ++i) {
                    oss << "  [" << i << "] " << op.inputs[i] << ": ";
                    if (i < input_shapes.size() && !input_shapes[i].empty()) {
                        oss << "shape=(";
                        for (size_t j = 0; j < input_shapes[i].size(); ++j) {
                            if (j > 0) oss << ", ";
                            oss << input_shapes[i][j];
                        }
                        oss << ")";
                    } else {
                        oss << "<shape unknown>";
                    }
                    oss << "\n";
                }
            }

            oss << "\nOutputs:\n";
            for (size_t i = 0; i < op.outputs.size(); ++i) {
                oss << "  [" << i << "] " << op.outputs[i] << ": ";
                if (i < output_shapes.size() && !output_shapes[i].empty()) {
                    oss << "shape=(";
                    for (size_t j = 0; j < output_shapes[i].size(); ++j) {
                        if (j > 0) oss << ", ";
                        oss << output_shapes[i][j];
                    }
                    oss << ")";
                } else {
                    oss << "<shape unknown or not inferred>";
                }
                oss << "\n";
            }

            oss << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                << "ERROR: " << error->message << "\n";

            if (!error->hint.empty()) {
                oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    << "HINT:  " << error->hint << "\n";
            }

            oss << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                << "Debug Information:\n"
                << "  Graph: " << mModule.name << "\n"
                << "  Batch size (B): " << mB << "\n"
                << "  Sequence length (T): " << mT << "\n"
                << "  Hidden size: " << mConfig.HiddenSize << "\n\n";

            throw std::runtime_error(oss.str());
        }
    }
}

CompiledGraph GraphCompiler::compile(const Graph& graph, long B, long T) {
    update_dimensions(B, T);

    mExtraShapes.clear();
    mTensorShapes.clear();
    mTensorDtypes.clear();

    // Initialize shape database from graph inputs and params
    for (const auto& [name, info] : graph.inputs) {
        if (!info.shape.empty()) {
            TensorShape ts;
            ts.dims = resolve_shape(info.shape, mShapeEnv);
            ts.inferred = false;
            mTensorShapes[name] = ts;
        }
        if (info.dtype) {
            mTensorDtypes[name] = *info.dtype;
        }
    }
    for (const auto& [name, info] : graph.params) {
        if (!info.shape.empty()) {
            TensorShape ts;
            ts.dims = resolve_shape(info.shape, mShapeEnv);
            ts.inferred = false;
            mTensorShapes[name] = ts;
        }
        if (info.dtype) {
            mTensorDtypes[name] = *info.dtype;
        }
    }
    for (const auto& [name, info] : graph.outputs) {
        if (!info.shape.empty()) {
            TensorShape ts;
            ts.dims = resolve_shape(info.shape, mShapeEnv);
            ts.inferred = false;
            mTensorShapes[name] = ts;
        }
        if (info.dtype) {
            mTensorDtypes[name] = *info.dtype;
        }
    }

    if (mModule.forward.has_value()) {
        const auto& fwd = *mModule.forward;
        for (const auto& op : fwd.operations) {
            const std::string& op_type =
                (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;
            if (op_type != "view" && op_type != "reshape") {
                continue;
            }
            std::vector<long> shape;
            if (auto* shape_attr = find_attr(op.attrs, "shape")) {
                shape = resolve_attr_shape(*shape_attr, mShapeEnv);
            } else if (auto* shape_like_attr = find_attr(op.attrs, "shape_like")) {
                if (auto ref_name = attr_string(*shape_like_attr)) {
                    std::string ref = *ref_name;
                    if (starts_with(ref, kSavedPrefix)) {
                        ref = ref.substr(kSavedPrefix.size());
                    }
                    auto it = mExtraShapes.find(ref);
                    if (it != mExtraShapes.end()) {
                        shape = it->second;
                    } else {
                        infer_known_tensor_shape(ref, mConfig, B, T, shape);
                    }
                }
            }
            if (!shape.empty()) {
                for (const auto& out : op.outputs) {
                    if (!out.empty()) {
                        mExtraShapes[out] = shape;
                    }
                }
            }
        }
    }

    CompiledGraph result;
    result.name = graph.name;
    result.ops.reserve(graph.operations.size());
    result.total_ops = graph.operations.size();

    for (std::size_t idx = 0; idx < graph.operations.size(); ++idx) {
        const auto& op = graph.operations[idx];
        const std::string& op_type =
            (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;

        CompiledOp compiled;
        compiled.original_idx = static_cast<std::uint16_t>(idx);
        compiled.op_id = op.id;
        compiled.type = classify_op(op_type);

        if (compiled.type == CompiledOpType::Unknown) {
            throw std::runtime_error("GraphCompiler: unsupported operation type: " + op_type);
        }

        // Validate operation shapes at compile time
        try {
            validate_operation_shapes(op, compiled.type, idx);
        } catch (const std::exception& e) {
            // Re-throw with additional context if validation fails
            std::cerr << "Shape validation failed during graph compilation.\n"
                      << "Operation: " << op.name << " (id: " << op.id << ")\n"
                      << "Error: " << e.what() << "\n"
                      << "To disable shape checking (not recommended), set SUROGATE_NO_SHAPE_CHECK=1\n";
            throw;
        }

        // Pre-resolve inputs
        compiled.inputs.reserve(op.inputs.size());
        for (const auto& input : op.inputs) {
            compiled.inputs.push_back(resolve_tensor_ref(input, false, op, mShapeEnv));
        }

        // Pre-resolve outputs
        compiled.outputs.reserve(op.outputs.size());
        for (std::size_t i = 0; i < op.outputs.size(); ++i) {
            auto ref = resolve_tensor_ref(op.outputs[i], true, op, mShapeEnv);

            // Fix dtype and shape for outputs based on operation type
            // This is needed for Mapped tensors that don't have predefined slots
            if (ref.slot == TensorSlot::Mapped) {
                const long B = mB;
                const long T = mT;
                const long C = mConfig.HiddenSize;
                const long Hq = mConfig.NumQueryHeads;
                const long Hs = mConfig.head_size();
                const long QKV = mConfig.qkv_channels();

                if (compiled.type == CompiledOpType::FusedResidualRMSNorm) {
                    // output[0] = residual_out [B, T, C] BF16
                    // output[1] = y (normalized) [B, T, C] BF16
                    // output[2] = rstd [B*T] FP32
                    if (i == 0 || i == 1) {
                        ref.dtype = !compiled.inputs.empty() ? compiled.inputs[0].dtype
                                                             : ETensorDType::BF16;
                        ref.shape = {B, T, C};
                    } else if (i == 2) {
                        ref.dtype = ETensorDType::FP32;
                        ref.shape = {B * T};
                    }
                } else if (compiled.type == CompiledOpType::CrossEntropyLoss) {
                    // output[0] = loss [B*T] FP32 (per-token)
                    ref.dtype = ETensorDType::FP32;
                    ref.shape = {B * T};
                } else if (compiled.type == CompiledOpType::CrossEntropyLossBackward) {
                    // output[0] = d_logits [B*T, V] (match logits dtype)
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[1].dtype;
                        ref.shape = compiled.inputs[1].shape;
                    } else {
                        ref.dtype = ETensorDType::BF16;
                        ref.shape = {B * T, static_cast<long>(mConfig.VocabSize)};
                    }
                } else if (compiled.type == CompiledOpType::FusedLMHeadLoss) {
                    // output[0] = loss [B*T] FP32 (per-token)
                    ref.dtype = ETensorDType::FP32;
                    ref.shape = {B * T};
                } else if (compiled.type == CompiledOpType::FusedLMHeadLossBackward) {
                    // output[0] = d_xF_flat [B*T, C], output[1] = d_lm_head [V, C]
                    if (i == 0) {
                        if (compiled.inputs.size() > 1) {
                            ref.dtype = compiled.inputs[1].dtype;
                            ref.shape = compiled.inputs[1].shape;
                        } else {
                            ref.dtype = ETensorDType::BF16;
                            ref.shape = {B * T, C};
                        }
                    } else if (i == 1) {
                        if (compiled.inputs.size() > 2) {
                            ref.dtype = compiled.inputs[2].dtype;
                            ref.shape = compiled.inputs[2].shape;
                        } else {
                            ref.dtype = ETensorDType::BF16;
                            ref.shape = {static_cast<long>(mConfig.VocabSize), C};
                        }
                    }
                } else if (compiled.type == CompiledOpType::FusedResidualRMSNormBackward) {
                    // outputs: d_residual [B, T, C], d_input [B, T, C], d_weight [C]
                    const ETensorDType grad_dtype =
                        !compiled.inputs.empty() ? compiled.inputs[0].dtype : ETensorDType::BF16;
                    if (i == 0 || i == 1) {
                        ref.dtype = grad_dtype;
                        ref.shape = {B, T, C};
                    } else if (i == 2) {
                        if (compiled.inputs.size() > 3) {
                            ref.dtype = compiled.inputs[3].dtype;
                        } else {
                            ref.dtype = grad_dtype;
                        }
                        ref.shape = {C};
                    }
                } else if (compiled.type == CompiledOpType::QKVQKNormRoPE) {
                    // output[0] = qkv_out [B, T, QKV] (match input dtype)
                    // output[1] = q_rstd [B, T, Hq] FP32
                    // output[2] = k_rstd [B, T, Hkv] FP32
                    if (i == 0) {
                        // Match input dtype (first input is qkv tensor)
                        if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                        } else {
                            ref.dtype = ETensorDType::BF16;
                        }
                        if (ref.shape.empty()) {
                            ref.shape = {B, T, QKV};
                        }
                    } else if (i == 1) {
                        ref.dtype = ETensorDType::FP32;
                        if (ref.shape.empty()) {
                            ref.shape = {B, T, Hq};
                        }
                    } else if (i == 2) {
                        ref.dtype = ETensorDType::FP32;
                        if (ref.shape.empty()) {
                            ref.shape = {B, T, static_cast<long>(mConfig.NumKeyValHeads)};
                        }
                    }
                } else if (compiled.type == CompiledOpType::FlashAttention) {
                    // output[0] = out [B, T, Hq*Hs] (match qkv dtype)
                    // output[1] = lse [B, Hq, T] FP32
                    if (i == 0) {
                        if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                        }
                        if (ref.shape.empty()) {
                            ref.shape = {B, T, Hq * Hs};
                        }
                    } else if (i == 1) {
                        ref.dtype = ETensorDType::FP32;
                        if (ref.shape.empty()) {
                            ref.shape = {B, Hq, T};
                        }
                    }
                } else if (compiled.type == CompiledOpType::Add ||
                           compiled.type == CompiledOpType::BiasAdd) {
                    // Match output to first input (broadcasting not supported in compiled add path).
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                        }
                    }
                } else if (compiled.type == CompiledOpType::AddBackward ||
                           compiled.type == CompiledOpType::BiasAddBackward) {
                    // Gradients match upstream shape/dtype.
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                        }
                    }
                } else if (compiled.type == CompiledOpType::Matmul ||
                           compiled.type == CompiledOpType::MatmulBias) {
                    // Infer output shape from matmul dimensions: C = A @ B
                    // NT: A [M, K], B [N, K] -> C [M, N]
                    // NN: A [M, K], B [K, N] -> C [M, N]
                    // TN: A [K, M], B [K, N] -> C [M, N]
                    // TT: A [K, M], B [N, K] -> C [M, N]
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                    if (ref.shape.empty() && compiled.inputs.size() >= 2) {
                        const auto& a_shape = compiled.inputs[0].shape;
                        const auto& b_shape = compiled.inputs[1].shape;
                        if (a_shape.size() >= 2 && b_shape.size() >= 2) {
                            // Parse transpose from op.attrs (compiled.attrs not yet resolved!)
                            EMMTranspose transpose = parse_transpose(op.attrs);
                            long M = 0, N = 0;
                            if (transpose == EMMTranspose::NT || transpose == EMMTranspose::NN) {
                                M = a_shape[0];
                            } else {
                                M = a_shape[1];
                            }
                            if (transpose == EMMTranspose::NT || transpose == EMMTranspose::TT) {
                                N = b_shape[0];
                            } else {
                                N = b_shape[1];
                            }
                            ref.shape = {M, N};
                        }
                    }
                } else if (compiled.type == CompiledOpType::MatmulSwiGLU) {
                    // outputs: out [B, T, D], up_out [M, 2D]
                    ETensorDType base_dtype =
                        !compiled.inputs.empty() ? compiled.inputs[0].dtype : ETensorDType::BF16;
                    long Ndim = 0;
                    if (compiled.inputs.size() > 1 && compiled.inputs[1].shape.size() >= 2) {
                        Ndim = compiled.inputs[1].shape[1];
                    }
                    long Ddim = (Ndim > 0) ? (Ndim / 2) : C;
                    long Mdim = mB * mT;
                    if (!compiled.inputs.empty() && compiled.inputs[0].shape.size() >= 1) {
                        Mdim = compiled.inputs[0].shape[0];
                    }

                    if (i == 0) {
                        ref.dtype = base_dtype;
                        ref.shape = {B, T, Ddim};
                    } else if (i == 1) {
                        ref.dtype = base_dtype;
                        ref.shape = {Mdim, Ndim > 0 ? Ndim : (2 * Ddim)};
                    }
                } else if (compiled.type == CompiledOpType::Zeros) {
                    // Preserve explicit output dtype/shape from graph.
                    // Read dtype from op attributes if specified
                    if (auto* dtype_attr = find_attr(op.attrs, "dtype")) {
                        if (auto dtype_str = attr_string(*dtype_attr)) {
                            ref.dtype = dtype_from_str(*dtype_str);
                        }
                    }
                    if (ref.shape.empty()) {
                        ref.shape = {B, T, C};
                    }
                } else if (compiled.type == CompiledOpType::RoPE ||
                           compiled.type == CompiledOpType::RoPEBackward) {
                    // RoPE outputs match input dtype/shape.
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                        }
                    }
                } else if (compiled.type == CompiledOpType::SwiGLU) {
                    // Output dtype matches input; shape is input with last dim / 2.
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                            if (!ref.shape.empty()) {
                                ref.shape.back() = ref.shape.back() / 2;
                            }
                        }
                    }
                } else if (compiled.type == CompiledOpType::SwiGLUBackward) {
                    // Output (d_inp) matches the pre-SwiGLU input shape.
                    // inputs: d_out [N, D], inp [N, 2D] -> output: d_inp [N, 2D]
                    if (compiled.inputs.size() > 1 && !compiled.inputs[1].shape.empty()) {
                        ref.dtype = compiled.inputs[1].dtype;
                        ref.shape = compiled.inputs[1].shape;
                    } else if (!compiled.inputs.empty() && !compiled.inputs[0].shape.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        ref.shape = compiled.inputs[0].shape;
                        if (!ref.shape.empty()) {
                            ref.shape.back() *= 2;
                        }
                    }
                } else if (compiled.type == CompiledOpType::MatmulBackward) {
                    // Match dA/dB shapes to their corresponding inputs (A/B).
                    // inputs: d_out, A_for_dB, B_for_dA -> outputs: dA, dB
                    if (i == 0 && compiled.inputs.size() > 1) {
                        ref.shape = compiled.inputs[1].shape;
                        ref.dtype = compiled.inputs[1].dtype;
                    } else if (i == 1 && compiled.inputs.size() > 2) {
                        ref.shape = compiled.inputs[2].shape;
                        ref.dtype = compiled.inputs[2].dtype;
                    } else {
                        ref.dtype = ETensorDType::BF16;
                    }
                } else if (compiled.type == CompiledOpType::MatmulSwiGLUBackward) {
                    // outputs: d_inp matches ln2 shape/dtype, d_weight matches weight shape/dtype
                    if (i == 0 && compiled.inputs.size() > 1) {
                        ref.shape = compiled.inputs[1].shape;
                        ref.dtype = compiled.inputs[1].dtype;
                    } else if (i == 1 && compiled.inputs.size() > 2) {
                        ref.shape = compiled.inputs[2].shape;
                        ref.dtype = compiled.inputs[2].dtype;
                    } else {
                        ref.dtype = ETensorDType::BF16;
                    }
                } else if (compiled.type == CompiledOpType::View ||
                           compiled.type == CompiledOpType::ViewBackward) {
                    // View preserves dtype from input; shape comes from attributes
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                    // Shape is set from attrs in resolve_attrs, not here
                } else if (compiled.type == CompiledOpType::MoESigmoid ||
                           compiled.type == CompiledOpType::MoESoftmax) {
                    // Output dtype/shape matches input (router logits)
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                        }
                    }
                } else if (compiled.type == CompiledOpType::MoETopK) {
                    // output[0] = routing_weights [B*T, K] (same dtype as input)
                    // output[1] = routing_indices [B*T, K] INT32
                    int top_k = 1;
                    if (auto* attr = find_attr(op.attrs, "top_k")) {
                        if (auto v = attr_int(*attr)) {
                            top_k = static_cast<int>(*v);
                        }
                    }
                    long BT = B * T;
                    if (!compiled.inputs.empty() && !compiled.inputs[0].shape.empty()) {
                        BT = compiled.inputs[0].shape[0];
                    }
                    if (i == 0) {
                        // routing_weights - same dtype as input probs
                        if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                        }
                        ref.shape = {BT, static_cast<long>(top_k)};
                    } else if (i == 1) {
                        // routing_indices - INT32
                        ref.dtype = ETensorDType::INT32;
                        ref.shape = {BT, static_cast<long>(top_k)};
                    }
                } else if (compiled.type == CompiledOpType::MoEPermute) {
                    // output[0] = permuted_input [total_tokens, C] (same dtype as input)
                    // output[1] = scatter_indices [total_tokens] INT32
                    int top_k = 1;
                    if (auto* attr = find_attr(op.attrs, "top_k")) {
                        if (auto v = attr_int(*attr)) {
                            top_k = static_cast<int>(*v);
                        }
                    }
                    long num_tokens = B * T;
                    long hidden_size = C;
                    if (!compiled.inputs.empty() && !compiled.inputs[0].shape.empty()) {
                        num_tokens = compiled.inputs[0].shape[0];
                        if (compiled.inputs[0].shape.size() > 1) {
                            hidden_size = compiled.inputs[0].shape[1];
                        }
                    }
                    long total_tokens = num_tokens * top_k;
                    if (i == 0) {
                        // permuted_input - same dtype as input
                        if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                        }
                        ref.shape = {total_tokens, hidden_size};
                    } else if (i == 1) {
                        // scatter_indices - INT32
                        ref.dtype = ETensorDType::INT32;
                        ref.shape = {total_tokens};
                    }
                } else if (compiled.type == CompiledOpType::MoEGroupedGemmGateUp) {
                    // output[0] = gate_up_out [total_tokens, 2*intermediate] (same dtype as input)
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                    // Shape is dynamic based on scatter_indices, leave empty for runtime
                } else if (compiled.type == CompiledOpType::MoEGroupedGemmDown) {
                    // output[0] = down_out [total_tokens, C] (same dtype as input)
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                    // Shape is dynamic based on scatter_indices, leave empty for runtime
                } else if (compiled.type == CompiledOpType::MoEUnpermute) {
                    // output[0] = combined_out [B*T, C] (same dtype as input)
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                    long num_tokens = B * T;
                    if (!compiled.inputs.empty() && compiled.inputs.size() > 1 &&
                        !compiled.inputs[1].shape.empty()) {
                        // routing_weights shape is [B*T, K]
                        num_tokens = compiled.inputs[1].shape[0];
                    }
                    ref.shape = {num_tokens, C};
                } else if (compiled.type == CompiledOpType::MoESigmoidBackward ||
                           compiled.type == CompiledOpType::MoESoftmaxBackward) {
                    // inputs: d_out, saved.input
                    // output: d_input (same shape/dtype as d_out, which is input[0])
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        if (ref.shape.empty()) {
                            ref.shape = compiled.inputs[0].shape;
                        }
                    }
                } else if (compiled.type == CompiledOpType::MoETopKBackward) {
                    // inputs: d_routing_weights, saved.probs, saved.indices
                    // output: d_probs (same shape/dtype as saved.probs, which is input[1])
                    if (compiled.inputs.size() > 1 && !compiled.inputs[1].shape.empty()) {
                        ref.dtype = compiled.inputs[1].dtype;
                        ref.shape = compiled.inputs[1].shape;
                    } else if (!compiled.inputs.empty()) {
                        // Fallback: use d_routing_weights dtype, derive probs shape
                        ref.dtype = compiled.inputs[0].dtype;
                        // probs is [num_tokens, num_experts], d_routing_weights is [num_tokens, top_k]
                        // We need num_experts from config
                        long num_tokens = B * T;
                        if (!compiled.inputs[0].shape.empty()) {
                            num_tokens = compiled.inputs[0].shape[0];
                        }
                        // Default from model config, then check for explicit attr override
                        long num_experts = static_cast<long>(mConfig.NumExperts);
                        if (auto* attr = find_attr(op.attrs, "num_experts")) {
                            if (auto v = attr_int(*attr)) {
                                num_experts = *v;
                            }
                        }
                        ref.shape = {num_tokens, num_experts};
                    }
                } else if (compiled.type == CompiledOpType::MoEPermuteBackward) {
                    // inputs: d_permuted, saved.scatter_indices
                    // output: d_x (unpermuted gradient)
                    // d_x shape is [num_tokens, hidden_size] where num_tokens = scatter_indices.size() / top_k
                    if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                    }
                    // Derive shape from scatter_indices and top_k
                    int top_k = 1;
                    if (auto* attr = find_attr(op.attrs, "top_k")) {
                        if (auto v = attr_int(*attr)) {
                            top_k = static_cast<int>(*v);
                        }
                    }
                    long total_tokens = B * T * top_k;  // permuted size
                    long hidden_size = C;
                    if (!compiled.inputs.empty() && !compiled.inputs[0].shape.empty()) {
                        total_tokens = compiled.inputs[0].shape[0];
                        if (compiled.inputs[0].shape.size() > 1) {
                            hidden_size = compiled.inputs[0].shape[1];
                        }
                    }
                    long num_tokens = total_tokens / top_k;
                    ref.shape = {num_tokens, hidden_size};
                } else if (compiled.type == CompiledOpType::MoEUnpermuteBackward) {
                    // inputs: d_out, saved.expert_out, saved.routing_weights, saved.scatter_indices
                    // outputs[0]: d_expert_out (same shape as saved.expert_out, input[1])
                    // outputs[1]: d_routing_weights (same shape as saved.routing_weights, input[2])
                    if (i == 0) {
                        // d_expert_out - same shape/dtype as saved.expert_out (input[1])
                        if (compiled.inputs.size() > 1 && !compiled.inputs[1].shape.empty()) {
                            ref.dtype = compiled.inputs[1].dtype;
                            ref.shape = compiled.inputs[1].shape;
                        } else if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                            // Fallback: expert_out is [total_tokens, C]
                            int top_k = 1;
                            if (auto* attr = find_attr(op.attrs, "top_k")) {
                                if (auto v = attr_int(*attr)) {
                                    top_k = static_cast<int>(*v);
                                }
                            }
                            ref.shape = {B * T * top_k, C};
                        }
                    } else if (i == 1) {
                        // d_routing_weights - same shape/dtype as saved.routing_weights (input[2])
                        if (compiled.inputs.size() > 2 && !compiled.inputs[2].shape.empty()) {
                            ref.dtype = compiled.inputs[2].dtype;
                            ref.shape = compiled.inputs[2].shape;
                        } else if (!compiled.inputs.empty()) {
                            ref.dtype = compiled.inputs[0].dtype;
                            // Fallback: routing_weights is [num_tokens, top_k]
                            int top_k = 1;
                            if (auto* attr = find_attr(op.attrs, "top_k")) {
                                if (auto v = attr_int(*attr)) {
                                    top_k = static_cast<int>(*v);
                                }
                            }
                            ref.shape = {B * T, static_cast<long>(top_k)};
                        }
                    }
                } else if (compiled.type == CompiledOpType::MoEGroupedGemmGateUpBackward ||
                           compiled.type == CompiledOpType::MoEGroupedGemmDownBackward) {
                    // inputs: d_out, saved.inp, weights, saved.scatter_indices
                    // output: d_inp (same shape/dtype as saved.inp, input[1])
                    if (compiled.inputs.size() > 1 && !compiled.inputs[1].shape.empty()) {
                        ref.dtype = compiled.inputs[1].dtype;
                        ref.shape = compiled.inputs[1].shape;
                    } else if (compiled.inputs.size() > 3 && !compiled.inputs[3].shape.empty()) {
                        // Fallback: infer total_tokens from scatter_indices length
                        ref.dtype = compiled.inputs[0].dtype;
                        const long total_tokens = compiled.inputs[3].shape[0];
                        ref.shape = {total_tokens, C};
                    } else if (!compiled.inputs.empty()) {
                        ref.dtype = compiled.inputs[0].dtype;
                        // Fallback: inp is permuted input [total_tokens, C]
                        int top_k = 1;
                        if (auto* attr = find_attr(op.attrs, "top_k")) {
                            if (auto v = attr_int(*attr)) {
                                top_k = static_cast<int>(*v);
                            }
                        }
                        ref.shape = {B * T * top_k, C};
                    }
                } else {
                    // Default for activation tensors
                    ref.dtype = ETensorDType::BF16;
                    ref.shape = {B, T, C};
                }
            }

            // Also fix dtype for pre-allocated RSTD slots (must be FP32)
            if ((compiled.type == CompiledOpType::FusedResidualRMSNorm && i == 2) ||
                (compiled.type == CompiledOpType::QKVQKNormRoPE && (i == 1 || i == 2))) {
                ref.dtype = ETensorDType::FP32;
            }

            // Ensure embedding output writes into the persistent encoded buffer.
            if (compiled.type == CompiledOpType::Embedding && i == 0) {
                const long Bdim = mB;
                const long Tdim = mT;
                const long Cdim = mConfig.HiddenSize;
                ref.slot = TensorSlot::Encoded;
                ref.shape = {Bdim, Tdim, Cdim};
            }

            // Track output dtype for downstream operations to reference.
            // This allows intermediate tensors to have their dtypes properly propagated.
            if (!op.outputs[i].empty()) {
                mTensorDtypes[op.outputs[i]] = ref.dtype;
            }

            compiled.outputs.push_back(std::move(ref));
        }

        // Pre-resolve attributes
        compiled.attrs = resolve_attrs(op, compiled.type, mShapeEnv);

        // Statistics
        if (compiled.type == CompiledOpType::Matmul || compiled.type == CompiledOpType::MatmulBias ||
            compiled.type == CompiledOpType::MatmulBackward) {
            result.matmul_ops++;
        } else if (compiled.type == CompiledOpType::View || compiled.type == CompiledOpType::ViewBackward) {
            result.view_ops++;
        }

        result.ops.push_back(std::move(compiled));
    }

    // Annotate layer boundaries for prefetch
    annotate_layer_boundaries(result);

    return result;
}

// ============================================================================
// CompiledExecutor implementation
// ============================================================================

CompiledExecutor::CompiledExecutor(DslRunState& run_state,
                                   DslParamStore& weights,
                                   DslGradStore& grads,
                                   const modules::ModelConfig& config,
                                   const RuntimeOptions& options)
    : mRunState(run_state)
    , mWeights(weights)
    , mGrads(grads)
    , mConfig(config)
    , mOptions(options)
{}

CompiledExecutor::~CompiledExecutor() {
    // Free persistent GPU buffers
    if (mMoEExpertOffsetsGPU) {
        cudaFree(mMoEExpertOffsetsGPU);
        mMoEExpertOffsetsGPU = nullptr;
        mMoEExpertOffsetsGPUSize = 0;
    }

    // Free persistent MoE saved tensor buffers
    for (auto& [name, buffer] : mMoESavedBuffers) {
        if (buffer) {
            cudaFree(buffer);
        }
    }
    mMoESavedBuffers.clear();
    mMoESavedSizes.clear();
}

void CompiledExecutor::set_lora_state(const modules::ModularLoRAConfig* config,
                                      modules::ModularLoRAWeightsManager* weights,
                                      modules::ModularLoRAGradsManager* grads,
                                      modules::LoRARunState* run_state) {
    mLoRAConfig = config;
    mLoRAWeights = weights;
    mLoRAGrads = grads;
    mLoRARunState = run_state;
}

void CompiledExecutor::set_weight_manager(DslWeightManager* weight_manager) {
    mWeightManager = weight_manager;
}

void CompiledExecutor::set_recipe(const recipes::Recipe* recipe) {
    mRecipe = recipe;
}

void CompiledExecutor::set_hook_context(void* context) {
    mHookContext = context;
}

void CompiledExecutor::set_recompute_fn(std::function<void(int, long, long, bool)> fn) {
    mRecomputeFn = std::move(fn);
}

void CompiledExecutor::set_recompute_enabled(bool enabled) {
    mRecomputeEnabled = enabled;
    mLastRecomputeLayer = -1;
    static int set_count = 0;
    if (set_count < 5) {
        fprintf(stderr, "[SET_RECOMPUTE_ENABLED] enabled=%d, mRecomputeFn=%p\n",
                enabled, mRecomputeFn ? (void*)1 : nullptr);
        set_count++;
    }
}

void CompiledExecutor::set_fp8_cache(std::unordered_map<std::string, FP8WeightCacheEntry>* cache) {
    mFP8Cache = cache;
}

void CompiledExecutor::set_fp4_cache(std::unordered_map<std::string, FP4WeightCacheEntry>* cache,
                                     std::unordered_map<std::string, FP4WeightCacheEntry>* cache_t) {
    mFP4Cache = cache;
    mFP4CacheT = cache_t;
}

void CompiledExecutor::set_saved_tensors(std::unordered_map<std::string, Tensor>* saved) {
    mSaved = saved;
}

void CompiledExecutor::set_save_list(const std::vector<std::string>* save_list) {
    mSaveList = save_list;
    mSaveSet.clear();
    if (save_list) {
        mSaveSet.insert(save_list->begin(), save_list->end());
    }
}

void CompiledExecutor::set_last_inputs_cpu(const Tensor* inputs_cpu) {
    mLastInputsCpu = inputs_cpu;
}

void CompiledExecutor::set_rng_seed_fn(std::function<unsigned int()> fn) {
    mRngSeedFn = std::move(fn);
}

const Tensor* CompiledExecutor::try_get_tensor(const std::string& name) const {
    auto it = mTensorMap.find(name);
    if (it == mTensorMap.end()) {
        return nullptr;
    }
    return &it->second;
}

void CompiledExecutor::save_moe_layer_tensors(int layer_idx) {
    // Copy MoE tensors from this layer to persistent storage before stack restore.
    // This allows stack memory to be reclaimed while preserving tensors for backward.
    if (mConfig.NumExperts == 0) {
        return;
    }

    // Build layer prefix pattern (e.g., "blocks[5].")
    std::string layer_prefix = "blocks[" + std::to_string(layer_idx) + "].";

    // Iterate through tensor map looking for MoE tensors from this layer
    for (auto& [name, tensor] : mTensorMap) {
        // Skip global MoE tensors - these are scratch space reused each layer
        // and are NOT needed for backward (backward uses mMoEExpertOffsetsGPU).
        if (name == "moe_expert_offsets" || name == "moe_gather_indices") {
            continue;
        }

        // Check if tensor belongs to this layer
        if (name.find(layer_prefix) != 0) {
            continue;
        }

        // Check if this is an MoE-related tensor that needs persistent storage
        bool is_moe_tensor = (name.find("moe_") != std::string::npos ||
                              name.find("scatter_indices") != std::string::npos ||
                              name.find("routing_weights") != std::string::npos ||
                              name.find("routing_indices") != std::string::npos ||
                              name.find("router_") != std::string::npos ||
                              name.find("permuted") != std::string::npos ||
                              name.find("expert_") != std::string::npos);

        if (!is_moe_tensor || tensor.Data == nullptr) {
            continue;
        }

        const size_t bytes = tensor.bytes();
        if (bytes == 0) {
            continue;
        }

        // Allocate or resize persistent buffer if needed
        auto buf_it = mMoESavedBuffers.find(name);
        if (buf_it == mMoESavedBuffers.end() || mMoESavedSizes[name] < bytes) {
            // Free old buffer if exists
            if (buf_it != mMoESavedBuffers.end() && buf_it->second != nullptr) {
                CUDA_CHECK(cudaFree(buf_it->second));
            }
            // Allocate new buffer
            void* new_buffer = nullptr;
            CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
            mMoESavedBuffers[name] = new_buffer;
            mMoESavedSizes[name] = bytes;
        }

        // Copy data to persistent buffer
        void* dst_buffer = mMoESavedBuffers[name];
        CUDA_CHECK(cudaMemcpyAsync(dst_buffer, tensor.Data, bytes,
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));

        // Update tensor to point to persistent buffer (so backward finds it)
        tensor.Data = static_cast<std::byte*>(dst_buffer);
    }
}

void CompiledExecutor::save_tensors(const std::vector<std::string>& save_list) {
    if (!mSaved) {
        return;
    }

    // Recompute is only active when enabled in runtime options.
    // Do NOT use mRecomputeFn here: it's always set when the graph is compiled,
    // even for no-recompute runs, and would cause metadata-only saves.
    const bool recompute_enabled = mOptions.recompute_enabled();

    auto prefer_live_tensor = [&](const std::string& tensor_name) -> bool {
        if (!recompute_enabled || !mSlotRegistry) {
            return false;
        }
        // Use will_recompute which checks the recompute_policy
        // In FFT mode (!lora_only), tensors with lora_only policy should NOT prefer live
        // because they won't be recomputed and the live buffer may have stale data
        const bool lora_only_mode = mRunState.is_lora_only_mode();
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(tensor_name, layer_idx, field)) {
            return mSlotRegistry->will_recompute(strip_ssa_suffix(field), lora_only_mode);
        }
        const std::string base_name = strip_ssa_suffix(tensor_name);
        return mSlotRegistry->will_recompute(strip_ssa_suffix(tensor_name), lora_only_mode);
    };

    // Helper to copy tensor to persistent buffer when needed in recompute mode.
    // Returns true if tensor was copied to persistent storage, false if metadata-only save.
    auto save_tensor_with_policy = [&](const std::string& name, const Tensor& src,
                                        bool prefer_live) -> void {
        if (prefer_live) {
            // Save metadata only - will resolve from live buffer or recompute
            Tensor meta = src;
            meta.Data = nullptr;
            (*mSaved)[name] = meta;
        } else if (recompute_enabled && src.Data != nullptr) {
            // In recompute mode but tensor won't be recomputed (e.g., lora_only in FFT mode).
            // Copy data to persistent buffer since live buffers will be reused.
            const size_t bytes = src.bytes();
            auto buf_it = mMoESavedBuffers.find(name);
            if (buf_it == mMoESavedBuffers.end() || mMoESavedSizes[name] < bytes) {
                if (buf_it != mMoESavedBuffers.end() && buf_it->second != nullptr) {
                    CUDA_CHECK(cudaFree(buf_it->second));
                }
                void* new_buffer = nullptr;
                CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
                mMoESavedBuffers[name] = new_buffer;
                mMoESavedSizes[name] = bytes;
            }
            void* dst_buffer = mMoESavedBuffers[name];
            CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src.Data, bytes,
                                       cudaMemcpyDeviceToDevice, mRunState.MainStream));
            Tensor saved_tensor;
            saved_tensor.DType = src.DType;
            saved_tensor.Rank = src.Rank;
            for (int i = 0; i < src.Rank; ++i) {
                saved_tensor.Sizes[i] = src.Sizes[i];
            }
            saved_tensor.Data = static_cast<std::byte*>(dst_buffer);
            (*mSaved)[name] = saved_tensor;
        } else {
            // Non-recompute mode: just store reference
            (*mSaved)[name] = src;
        }
    };

    for (const auto& name : save_list) {
        // First check the tensor map (intermediate tensors)
        auto it = mTensorMap.find(name);
        if (it != mTensorMap.end()) {
            const bool is_moe_tensor = (name.find("moe_") != std::string::npos ||
                                        name.find("scatter_indices") != std::string::npos ||
                                        name.find("routing_weights") != std::string::npos ||
                                        name.find("routing_indices") != std::string::npos ||
                                        name.find("router_probs") != std::string::npos ||
                                        name.find("router_logits") != std::string::npos ||
                                        name.find("permuted_input") != std::string::npos ||
                                        name.find("expert_") != std::string::npos);
            const bool prefer_live = prefer_live_tensor(name);

            // For MoE tensors, copy to persistent buffer to prevent buffer reuse corruption
            if (is_moe_tensor && mConfig.NumExperts > 0 && it->second.Data != nullptr) {
                if (prefer_live) {
                    // Recompute mode: store metadata only and recompute later.
                    save_tensor_with_policy(name, it->second, true);
                    continue;
                }
                const Tensor& src = it->second;
                const size_t bytes = src.bytes();

                // Allocate or resize persistent buffer if needed
                auto buf_it = mMoESavedBuffers.find(name);
                if (buf_it == mMoESavedBuffers.end() || mMoESavedSizes[name] < bytes) {
                    // Free old buffer if exists
                    if (buf_it != mMoESavedBuffers.end() && buf_it->second != nullptr) {
                        CUDA_CHECK(cudaFree(buf_it->second));
                    }
                    // Allocate new buffer
                    void* new_buffer = nullptr;
                    CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
                    mMoESavedBuffers[name] = new_buffer;
                    mMoESavedSizes[name] = bytes;
                }

                // Copy data to persistent buffer
                void* dst_buffer = mMoESavedBuffers[name];
                CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src.Data, bytes,
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));

                // Create tensor struct pointing to persistent buffer
                Tensor saved_tensor;
                saved_tensor.DType = src.DType;
                saved_tensor.Rank = src.Rank;
                for (int i = 0; i < src.Rank; ++i) {
                    saved_tensor.Sizes[i] = src.Sizes[i];
                }
                saved_tensor.Data = static_cast<std::byte*>(dst_buffer);

                (*mSaved)[name] = saved_tensor;
            } else {
                // Non-MoE tensor: use standard policy-based saving
                // DEBUG: Print res_ffn values being saved
                static int map_save_trace = 0;
                if (name.find("blocks[25].res_ffn") != std::string::npos && map_save_trace < 16) {
                    map_save_trace++;
                    cudaStreamSynchronize(mRunState.MainStream);
                    std::vector<float> vals(4);
                    cudaMemcpy(vals.data(), it->second.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
                    bool pl = prefer_live;
                    fprintf(stderr, "[SAVE_MAP_RES_FFN] %s src.Data=%p vals=%.6f,%.6f,%.6f,%.6f prefer_live=%d\n",
                            name.c_str(), it->second.Data, vals[0], vals[1], vals[2], vals[3], pl ? 1 : 0);
                }
                save_tensor_with_policy(name, it->second, prefer_live);
            }
            continue;
        }

        // Check special tensors
        if (name == "token_ids") {
            save_tensor_with_policy(name, mRunState.Inputs, prefer_live_tensor(name));
            continue;
        }
        if (name == "position_ids") {
            save_tensor_with_policy(name, mRunState.PositionIDs, prefer_live_tensor(name));
            continue;
        }

        // Try to look up as a pre-allocated activation by creating a TensorRef
        // This handles tensors like "blocks[0].ln1_rstd" that map to slots
        TensorRef ref;
        ref.name = name;
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            ref.layer_idx = layer_idx;
            // Map common saved fields
            const bool prefer_live = prefer_live_tensor(name);
            if (field == "ln1_rstd") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).ln1_rstd, prefer_live);
            } else if (field == "ln2_rstd") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).ln2_rstd, prefer_live);
            } else if (field == "q_rstd") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).q_rstd, prefer_live);
            } else if (field == "k_rstd") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).k_rstd, prefer_live);
            } else if (field == "lse") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).lse, prefer_live);
            } else if (field == "ln1" || field == "ln1_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).ln1, prefer_live);
            } else if (field == "ln2" || field == "ln2_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).ln2, prefer_live);
            } else if (field == "qkv") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).qkv, prefer_live);
            } else if (field == "qkv_rope") {
                // qkv_rope has RoPE applied - save it if available, otherwise fall back to qkv
                auto& acts = mRunState.simplified_acts(layer_idx);
                Tensor& src = acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
                save_tensor_with_policy(name, src, prefer_live);
            } else if (field == "qkv_flat") {
                // Save the flattened version for matmul backward shape resolution
                Tensor qkv = mRunState.simplified_acts(layer_idx).qkv;
                Tensor flat = view_tensor(qkv, {qkv.Sizes[0] * qkv.Sizes[1], qkv.Sizes[2]});
                save_tensor_with_policy(name, flat, prefer_live);
            } else if (field == "att" || field == "att_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).att, prefer_live);
            } else if (field == "att_out" || field == "att_out_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).att_out, prefer_live);
            } else if (field == "mlp_up" || field == "mlp_up_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).mlp_up, prefer_live);
            } else if (field == "swiglu") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).swiglu, prefer_live);
            } else if (field == "swiglu_flat") {
                Tensor swiglu = mRunState.simplified_acts(layer_idx).swiglu;
                Tensor flat = view_tensor(swiglu, {swiglu.Sizes[0] * swiglu.Sizes[1], swiglu.Sizes[2]});
                save_tensor_with_policy(name, flat, prefer_live);
            } else if (field == "mlp_down" || field == "mlp_down_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).mlp_down, prefer_live);
            } else if (field == "res_att" || field == "residual_att") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).residual_att, prefer_live);
            } else if (field == "res_ffn" || field == "residual_ffn") {
                // res_ffn is computed dynamically (residual_att + mlp_down), check mTensorMap
                auto it = mTensorMap.find(name);
                if (it != mTensorMap.end()) {
                    // DEBUG: Print res_ffn values being saved for layer 25
                    static int save_trace = 0;
                    if (layer_idx == 25 && save_trace < 16) {
                        save_trace++;
                        cudaStreamSynchronize(mRunState.MainStream);
                        std::vector<float> vals(4);
                        cudaMemcpy(vals.data(), it->second.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
                        fprintf(stderr, "[SAVE_RES_FFN] %s src.Data=%p vals=%.6f,%.6f,%.6f,%.6f prefer_live=%d\n",
                                name.c_str(), it->second.Data, vals[0], vals[1], vals[2], vals[3], prefer_live ? 1 : 0);
                    }
                    save_tensor_with_policy(name, it->second, prefer_live);
                } else {
                    throw std::runtime_error("CompiledExecutor: res_ffn tensor not found in map: " + name);
                }
            } else if (mWeights.has(name)) {
                (*mSaved)[name] = mWeights.get(name);
            } else {
                throw std::runtime_error("CompiledExecutor: cannot save tensor " + name);
            }
        } else if (name == "ln_final" || name == "xF") {
            save_tensor_with_policy(name, mRunState.non_block_activations().ln_final, prefer_live_tensor(name));
        } else if (name == "final_residual" || name == "residual_final") {
            save_tensor_with_policy(name, mRunState.get_final_residual(), prefer_live_tensor(name));
        } else if (name == "xF_flat") {
            // Save the flattened version for matmul backward
            Tensor ln_final = mRunState.non_block_activations().ln_final;
            Tensor flat = view_tensor(ln_final, {ln_final.Sizes[0] * ln_final.Sizes[1], ln_final.Sizes[2]});
            save_tensor_with_policy(name, flat, prefer_live_tensor(name));
        } else if (name == "ln_final_rstd") {
            save_tensor_with_policy(name, mRunState.non_block_activations().ln_final_rstd, prefer_live_tensor(name));
        } else if (name == "encoded" || name == "x0") {
            save_tensor_with_policy(name, mRunState.non_block_activations().encoded, prefer_live_tensor(name));
        } else if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
            save_tensor_with_policy(name, mRunState.non_block_activations().freq_cis, prefer_live_tensor(name));
        } else if (mWeights.has(name)) {
            (*mSaved)[name] = mWeights.get(name);
        } else {
            throw std::runtime_error("CompiledExecutor: cannot save tensor " + name);
        }
    }

    // For MoE models, copy expert_offsets data to persistent storage for backward pass
    // The original tensor is stack-allocated and will be freed before backward runs
    if (mConfig.NumExperts > 0) {
        auto it = mTensorMap.find("moe_expert_offsets");
        if (it != mTensorMap.end() && it->second.Data) {
            const Tensor& src = it->second;
            const int num_elements = static_cast<int>(src.nelem());
            mMoEExpertOffsetsData.resize(num_elements);
            CUDA_CHECK(cudaMemcpy(mMoEExpertOffsetsData.data(), src.Data,
                                  num_elements * sizeof(int), cudaMemcpyDeviceToHost));
            // Store metadata for reconstruction in backward
            mMoEExpertOffsets = src;  // Copy the tensor metadata (shape, dtype, etc.)
            mMoEExpertOffsets.Data = nullptr;  // Data will be restored from CPU storage
        }
    }
}

Tensor* CompiledExecutor::try_resolve_saved_live(const std::string& name, const Tensor& saved) {
    std::vector<long> shape;
    shape.reserve(static_cast<std::size_t>(saved.Rank));
    for (int i = 0; i < saved.Rank; ++i) {
        shape.push_back(saved.Sizes[i]);
    }

    auto map_view = [&](Tensor& base) -> Tensor* {
        if (!base.Data) {
            return nullptr;
        }
        if (shape.empty() || tensor_shape_matches(base, shape)) {
            return &base;
        }
        if (shape_nelem(shape) != base.nelem()) {
            // DEBUG: Log when element count mismatch
            int layer_idx = -1;
            std::string field;
            parse_block_param(name, layer_idx, field);
            if (layer_idx == 0) {
                fprintf(stderr, "[MAP_VIEW_FAIL] %s: nelem mismatch base=%ld vs shape=%ld\n",
                        name.c_str(), base.nelem(), shape_nelem(shape));
            }
            return nullptr;
        }
        auto [it, _] = mTensorMap.insert_or_assign(name, view_tensor(base, shape));
        return &it->second;
    };

    if (name == "token_ids") {
        return map_view(mRunState.Inputs);
    }
    if (name == "position_ids") {
        return map_view(mRunState.PositionIDs);
    }
    if (name == "encoded" || name == "x0") {
        return map_view(mRunState.non_block_activations().encoded);
    }
    if (name == "ln_final" || name == "xF" || name == "xF_flat") {
        return map_view(mRunState.non_block_activations().ln_final);
    }
    if (name == "ln_final_rstd") {
        return map_view(mRunState.non_block_activations().ln_final_rstd);
    }
    if (name == "final_residual" || name == "residual_final") {
        return map_view(mRunState.get_final_residual());
    }
    if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
        return map_view(mRunState.non_block_activations().freq_cis);
    }

    int layer_idx = -1;
    std::string field;
    if (parse_block_param(name, layer_idx, field)) {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(mConfig.NumLayers)) {
            return nullptr;
        }
        auto& acts = mRunState.simplified_acts(layer_idx);
        if (field == "ln1" || field == "ln1_flat") return map_view(acts.ln1);
        if (field == "ln1_rstd") return map_view(acts.ln1_rstd);
        if (field == "ln2" || field == "ln2_flat") return map_view(acts.ln2);
        if (field == "ln2_rstd") return map_view(acts.ln2_rstd);
        if (field == "q_rstd") return map_view(acts.q_rstd);
        if (field == "k_rstd") return map_view(acts.k_rstd);
        if (field == "qkv" || field == "qkv_flat" || field == "qkv_biased") return map_view(acts.qkv);
        if (field == "qkv_rope") {
            Tensor* base = acts.qkv_rope.Data ? &acts.qkv_rope : &acts.qkv;
            return map_view(*base);
        }
        if (field == "lse") return map_view(acts.lse);
        if (field == "att" || field == "att_flat") return map_view(acts.att);
        if (field == "att_out" || field == "att_out_flat") return map_view(acts.att_out);
        if (field == "res_att" || field == "residual_att") return map_view(acts.residual_att);
        if (field == "mlp_up" || field == "mlp_up_flat") return map_view(acts.mlp_up);
        if (field == "swiglu" || field == "swiglu_flat") return map_view(acts.swiglu);
        if (field == "mlp_down" || field == "mlp_down_flat") return map_view(acts.mlp_down);
        if (field == "router_logits") return map_view(acts.router_logits);
        if (field == "router_probs") return map_view(acts.router_probs);
        if (field == "routing_weights") return map_view(acts.routing_weights);
        if (field == "routing_indices") return map_view(acts.routing_indices);
        if (field == "permuted_input") return map_view(acts.permuted_input);
        if (field == "scatter_indices") return map_view(acts.scatter_indices);
        if (field == "expert_gate_up") return map_view(acts.expert_gate_up);
        if (field == "expert_act") return map_view(acts.expert_act);
        if (field == "expert_down") return map_view(acts.expert_down);
        if (field == "moe_out" || field == "moe_out_flat") return map_view(acts.moe_out);
        if (field == "res_ffn" || field == "residual_ffn") {
            Tensor& res = mRunState.get_residual(layer_idx, mRunState.MainStream);
            return map_view(res);
        }
        if (field == "rope_freqs" || field == "freq_cis") {
            return map_view(mRunState.non_block_activations().freq_cis);
        }
    }

    return nullptr;
}

Tensor& CompiledExecutor::resolve_tensor(const TensorRef& ref) {
    auto& rs = mRunState;

    if (!ref.name.empty()) {
        if (auto base = base_param_from_grad(ref.name)) {
            bool accum = false;
            if (Tensor* grad = mGrads.get_param_grad(*base, accum)) {
                if (grad->Data) {
                    if (!ref.shape.empty()) {
                        auto [it, _] = mTensorMap.insert_or_assign(ref.name, view_tensor(*grad, ref.shape));
                        return it->second;
                    }
                    auto [it, _] = mTensorMap.insert_or_assign(ref.name, *grad);
                    return it->second;
                }
            }
        }
    }

    // DEBUG: Trace resolution for top-layer d_qkv_rope gradients
    static int d_qkv_rope_resolve_trace = 0;
    if (d_qkv_rope_resolve_trace < 8 &&
        ref.name.find("d_blocks[") != std::string::npos &&
        ref.name.find("qkv_rope") != std::string::npos &&
        ref.layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
        fprintf(stderr,
                "[RESOLVE_D_QKV_ROPE] name=%s slot=%d shape_rank=%zu\n",
                ref.name.c_str(), static_cast<int>(ref.slot), ref.shape.size());
        d_qkv_rope_resolve_trace++;
    }

    // If shape is specified and this is a pre-allocated slot, we may need to create a view
    if (!ref.shape.empty() && ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Saved &&
        ref.slot != TensorSlot::Parameter && ref.slot != TensorSlot::Temporary) {
        // Check if we already have a tensor in the map (e.g., from MoE temp allocation)
        auto it = mTensorMap.find(ref.name);
        // DEBUG: Trace mTensorMap lookup for top-layer d_ln1
        static int shape_trace_count = 0;
        if (shape_trace_count < 8 &&
            ref.name.find(".ln1") != std::string::npos &&
            ref.layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
            fprintf(stderr, "[RESOLVE_SHAPE] ref.name=%s, shape.size=%zu, found_in_map=%s, slot=%d\n",
                    ref.name.c_str(), ref.shape.size(),
                    (it != mTensorMap.end() && it->second.Data) ? "YES" : "NO",
                    static_cast<int>(ref.slot));
            if (it != mTensorMap.end() && it->second.Data) {
                cudaStreamSynchronize(mRunState.MainStream);
                std::vector<float> vals(4);
                cudaMemcpy(vals.data(), it->second.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
                fprintf(stderr, "[RESOLVE_SHAPE] d_ln1 ptr=%p, values=%.9f,%.9f,%.9f,%.9f\n",
                        it->second.Data, vals[0], vals[1], vals[2], vals[3]);
            }
            shape_trace_count++;
        }
        // DEBUG: Trace when top-layer d_qkv_rope is already in the map
        static int qkv_rope_shape_trace = 0;
        if (qkv_rope_shape_trace < 8 &&
            ref.name.find("d_blocks[") != std::string::npos &&
            ref.name.find("qkv_rope") != std::string::npos &&
            ref.layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
            fprintf(stderr, "[RESOLVE_SHAPE_QKV_ROPE] ref.name=%s shape.size=%zu found_in_map=%s slot=%d\n",
                    ref.name.c_str(), ref.shape.size(),
                    (it != mTensorMap.end() && it->second.Data) ? "YES" : "NO",
                    static_cast<int>(ref.slot));
            if (it != mTensorMap.end() && it->second.Data) {
                fprintf(stderr,
                        "[RESOLVE_SHAPE_QKV_ROPE] ptr=%p shape=%s\n",
                        it->second.Data,
                        tensor_shape_str(it->second).c_str());
            }
            qkv_rope_shape_trace++;
        }
        if (it != mTensorMap.end() && it->second.Data) {
            // For MoE operations, the tensor map may contain dynamically-shaped temps
            // that differ from the statically-compiled shapes. Prioritize the map tensor
            // if it has valid data, even if shapes differ.
            // Verify shape matches
            bool shape_matches = (it->second.Rank == static_cast<int>(ref.shape.size()));
            if (shape_matches) {
                for (int i = 0; i < it->second.Rank && shape_matches; ++i) {
                    shape_matches = (it->second.Sizes[i] == ref.shape[i]);
                }
            }
            if (shape_matches) {
                return it->second;
            }
            // Shape doesn't match, but we have valid data - use it for MoE dynamic shapes
            // This handles cases like swiglu output [total_tokens, D] vs expected [B, T, D]
            return it->second;
        }
        // Need to create a view - get the base tensor and create view
        Tensor* base = nullptr;
        switch (ref.slot) {
            case TensorSlot::TokenIDs: base = &rs.Inputs; break;
            case TensorSlot::PositionIDs: base = &rs.PositionIDs; break;
            case TensorSlot::Targets: base = &rs.Targets; break;
            case TensorSlot::Losses: base = &rs.Losses; break;
            case TensorSlot::DLoss: base = &rs.scratch().cross_entropy_dloss; break;
            case TensorSlot::BlockDLN1: base = &rs.simplified_grads(ref.layer_idx).d_ln1; break;
            case TensorSlot::BlockDQKV: base = &rs.simplified_grads(ref.layer_idx).d_qkv; break;
            case TensorSlot::BlockDAtt: base = &rs.simplified_grads(ref.layer_idx).d_att; break;
            case TensorSlot::BlockDSwiGLU: base = &rs.simplified_grads(ref.layer_idx).d_swiglu; break;
            case TensorSlot::BlockDMLPUp: base = &rs.simplified_grads(ref.layer_idx).d_mlp_up; break;
            case TensorSlot::BlockDMLPDown: base = &rs.simplified_grads(ref.layer_idx).d_mlp_down; break;
            case TensorSlot::BlockDLN2: base = &rs.simplified_grads(ref.layer_idx).d_ln2; break;
            case TensorSlot::BlockDResAtt: base = &rs.simplified_grads(ref.layer_idx).d_res_att; break;
            case TensorSlot::BlockDResFFN: base = &rs.simplified_grads(ref.layer_idx).d_res_ffn; break;
            case TensorSlot::BlockLN1: base = &rs.simplified_acts(ref.layer_idx).ln1; break;
            case TensorSlot::BlockLN2: base = &rs.simplified_acts(ref.layer_idx).ln2; break;
            case TensorSlot::BlockQKV: base = &rs.simplified_acts(ref.layer_idx).qkv; break;
            case TensorSlot::BlockAtt: base = &rs.simplified_acts(ref.layer_idx).att; break;
            case TensorSlot::BlockAttOut: base = &rs.simplified_acts(ref.layer_idx).att_out; break;
            case TensorSlot::BlockMLPUp: base = &rs.simplified_acts(ref.layer_idx).mlp_up; break;
            case TensorSlot::BlockSwiGLU: base = &rs.simplified_acts(ref.layer_idx).swiglu; break;
            case TensorSlot::BlockMLPDown: base = &rs.simplified_acts(ref.layer_idx).mlp_down; break;
            default: break;
        }
        if (base && base->Data) {
            auto [ins_it, _] = mTensorMap.insert_or_assign(ref.name, view_tensor(*base, ref.shape));
            return ins_it->second;
        }
    }

    // FIX: Always check mTensorMap first for gradient slots before falling back to simplified_grads.
    // This is critical because view_backward stores aliases in mTensorMap, and subsequent ops
    // (like rmsnorm_backward) must use that aliased tensor, not the pre-allocated simplified_grads buffer.
    // Without this check, the gradient chain can break when view_backward creates an alias that
    // points to a different buffer than the pre-allocated slot.
    if (!ref.name.empty()) {
        auto it = mTensorMap.find(ref.name);
        if (it != mTensorMap.end() && it->second.Data) {
            return it->second;
        }
        // DEBUG: Trace when mTensorMap lookup fails for top-layer d_ln1
        static int miss_trace_count = 0;
        if (miss_trace_count < 8 &&
            ref.name.find(".ln1") != std::string::npos &&
            ref.layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
            fprintf(stderr, "[RESOLVE_FIX] NOT in mTensorMap: %s, slot=%d\n",
                    ref.name.c_str(), static_cast<int>(ref.slot));
            miss_trace_count++;
        }
        // DEBUG: Trace when mTensorMap lookup fails for top-layer d_qkv_rope
        static int qkv_rope_miss_trace = 0;
        if (qkv_rope_miss_trace < 8 &&
            ref.name.find("d_blocks[") != std::string::npos &&
            ref.name.find("qkv_rope") != std::string::npos &&
            ref.layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
            fprintf(stderr, "[RESOLVE_FIX_QKV_ROPE] NOT in mTensorMap: %s, slot=%d\n",
                    ref.name.c_str(), static_cast<int>(ref.slot));
            qkv_rope_miss_trace++;
        }
    }

    switch (ref.slot) {
        case TensorSlot::TokenIDs:
            return rs.Inputs;
        case TensorSlot::PositionIDs:
            return rs.PositionIDs;
        case TensorSlot::Targets:
            return rs.Targets;
        case TensorSlot::Losses:
            return rs.Losses;
        case TensorSlot::DLoss:
            return rs.scratch().cross_entropy_dloss;
        case TensorSlot::Encoded:
            return rs.non_block_activations().encoded;
        case TensorSlot::LNFinal:
            return rs.non_block_activations().ln_final;
        case TensorSlot::LNFinalRSTD:
            return rs.non_block_activations().ln_final_rstd;
        case TensorSlot::FinalResidual:
            return rs.get_final_residual();
        case TensorSlot::FreqCis:
            return rs.non_block_activations().freq_cis;
        case TensorSlot::BlockLN1:
            return rs.simplified_acts(ref.layer_idx).ln1;
        case TensorSlot::BlockLN1RSTD:
            return rs.simplified_acts(ref.layer_idx).ln1_rstd;
        case TensorSlot::BlockLN2:
            return rs.simplified_acts(ref.layer_idx).ln2;
        case TensorSlot::BlockLN2RSTD:
            return rs.simplified_acts(ref.layer_idx).ln2_rstd;
        case TensorSlot::BlockQRSTD:
            return rs.simplified_acts(ref.layer_idx).q_rstd;
        case TensorSlot::BlockKRSTD:
            return rs.simplified_acts(ref.layer_idx).k_rstd;
        case TensorSlot::BlockQKV:
            return rs.simplified_acts(ref.layer_idx).qkv;
        case TensorSlot::BlockQKVRoPE: {
            auto& acts = rs.simplified_acts(ref.layer_idx);
            return acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
        }
        case TensorSlot::BlockLSE:
            return rs.simplified_acts(ref.layer_idx).lse;
        case TensorSlot::BlockAtt:
            return rs.simplified_acts(ref.layer_idx).att;
        case TensorSlot::BlockAttOut:
            return rs.simplified_acts(ref.layer_idx).att_out;
        case TensorSlot::BlockResidualAtt:
            return rs.simplified_acts(ref.layer_idx).residual_att;
        case TensorSlot::BlockMLPUp:
            return rs.simplified_acts(ref.layer_idx).mlp_up;
        case TensorSlot::BlockSwiGLU:
            return rs.simplified_acts(ref.layer_idx).swiglu;
        case TensorSlot::BlockMLPDown:
            return rs.simplified_acts(ref.layer_idx).mlp_down;
        case TensorSlot::BlockResidualFFN:
            return rs.get_residual(ref.layer_idx, rs.MainStream);
        case TensorSlot::BlockDLN1:
            return rs.simplified_grads(ref.layer_idx).d_ln1;
        case TensorSlot::BlockDQKV:
            return rs.simplified_grads(ref.layer_idx).d_qkv;
        case TensorSlot::BlockDAtt:
            return rs.simplified_grads(ref.layer_idx).d_att;
        case TensorSlot::BlockDSwiGLU:
            return rs.simplified_grads(ref.layer_idx).d_swiglu;
        case TensorSlot::BlockDMLPUp:
            return rs.simplified_grads(ref.layer_idx).d_mlp_up;
        case TensorSlot::BlockDMLPDown:
            return rs.simplified_grads(ref.layer_idx).d_mlp_down;
        case TensorSlot::BlockDLN2:
            return rs.simplified_grads(ref.layer_idx).d_ln2;
        case TensorSlot::BlockDResAtt:
            return rs.simplified_grads(ref.layer_idx).d_res_att;
        case TensorSlot::BlockDResFFN:
            return rs.simplified_grads(ref.layer_idx).d_res_ffn;
        case TensorSlot::Parameter:
            return mWeights.get(ref.name);
        case TensorSlot::Saved:
            if (mSaved) {
                auto it = mSaved->find(ref.name);
                if (it != mSaved->end()) {
                    // If the saved tensor has actual data, use it directly.
                    // Only resolve from live buffers when Data == nullptr (metadata-only mode).
                    // This is critical for FFT mode where tensors with lora_only recompute_policy
                    // are saved with actual data and should NOT use live buffers.
                    if (it->second.Data != nullptr) {
                        return it->second;
                    }
                    // Metadata-only: try to resolve from live buffer or recompute
                    if (auto live_it = mTensorMap.find(ref.name); live_it != mTensorMap.end()) {
                        return live_it->second;
                    }
                    if (Tensor* live = try_resolve_saved_live(ref.name, it->second)) {
                        return *live;
                    }
                    return it->second;
                }
            }
            throw std::runtime_error("CompiledExecutor: saved tensor not found: " + ref.name);
        case TensorSlot::Mapped: {
            auto it = mTensorMap.find(ref.name);
            if (it != mTensorMap.end()) {
                return it->second;
            }
            throw std::runtime_error("CompiledExecutor: tensor not found: " + ref.name);
        }
        case TensorSlot::Temporary:
            throw std::runtime_error("CompiledExecutor: temporary slot requires allocation");
    }
    throw std::runtime_error("CompiledExecutor: invalid tensor slot");
}

Tensor& CompiledExecutor::ensure_output_tensor(const TensorRef& ref) {
    if (!ref.name.empty()) {
        if (auto base = base_param_from_grad(ref.name)) {
            bool accum = false;
            if (Tensor* grad = mGrads.get_param_grad(*base, accum)) {
                if (grad->Data) {
                    if (!ref.shape.empty()) {
                        auto [it, _] = mTensorMap.insert_or_assign(ref.name, view_tensor(*grad, ref.shape));
                        return it->second;
                    }
                    auto [it, _] = mTensorMap.insert_or_assign(ref.name, *grad);
                    return it->second;
                }
            }
        }
    }

    // DSL-driven aliasing: allow gradients to reuse existing activation buffers.
    if (mSlotRegistry && mSlotRegistry->has_dsl_layout() && !ref.name.empty()) {
        std::string base_name = strip_ssa_suffix(ref.name);
        const bool is_grad_name = starts_with(base_name, "d_");
        std::string parse_name = is_grad_name ? base_name.substr(2) : base_name;
        int layer_idx = -1;
        std::string field;
        std::string lookup_name = base_name;
        if (parse_block_param(parse_name, layer_idx, field)) {
            const std::string base_field = strip_ssa_suffix(field);
            lookup_name = is_grad_name ? ("d_" + base_field) : base_field;
        }
        if (auto slot_entry = mSlotRegistry->lookup(lookup_name)) {
            if (!slot_entry->alias_of.empty()) {
                const std::string alias_field = slot_entry->alias_of;
                std::string alias_name = alias_field;
                if (layer_idx >= 0) {
                    alias_name = "blocks[" + std::to_string(layer_idx) + "]." + alias_field;
                }
                if (auto alias_entry = mSlotRegistry->lookup(alias_field)) {
                    TensorRef alias_ref;
                    alias_ref.name = alias_name;
                    alias_ref.layer_idx = layer_idx;
                    alias_ref.slot = alias_entry->slot;
                    alias_ref.shape = ref.shape;
                    alias_ref.dtype = ref.dtype;
                    if (mSaveSet.find(alias_name) != mSaveSet.end()) {
                        alias_ref.slot = TensorSlot::Saved;
                    }
                    try {
                        Tensor& base = resolve_tensor(alias_ref);
                        Tensor view = ref.shape.empty() ? base : view_tensor(base, ref.shape);
                        if (std::getenv("SUROGATE_TRACE_MOE_ALLOC") &&
                            ref.name.find("permuted_input") != std::string::npos) {
                            std::ostringstream oss;
                            oss << "[MOE_ALIAS] name=" << ref.name << " alias=" << alias_name << " ref_shape=[";
                            for (std::size_t i = 0; i < ref.shape.size(); ++i) {
                                if (i) oss << ",";
                                oss << ref.shape[i];
                            }
                            oss << "] base_shape=[";
                            for (int i = 0; i < base.Rank; ++i) {
                                if (i) oss << ",";
                                oss << base.Sizes[i];
                            }
                            oss << "] view_shape=[";
                            for (int i = 0; i < view.Rank; ++i) {
                                if (i) oss << ",";
                                oss << view.Sizes[i];
                            }
                            oss << "]\n";
                            std::fputs(oss.str().c_str(), stderr);
                        }
                        auto [it, _] = mTensorMap.insert_or_assign(ref.name, view);
                        return it->second;
                    } catch (const std::exception&) {
                        // Fall through to normal allocation if alias resolution fails.
                    }
                }
            }
        }
    }

    // For pre-allocated slots, just return the tensor
    if (ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Temporary) {
        Tensor& t = resolve_tensor(ref);
        if (!t.Data) {
            mRunState.temp_acquire(t);
            mTemps.push_back(t);
        }
        if (!ref.shape.empty()) {
            // Create a view if needed
            auto [it, inserted] = mTensorMap.emplace(ref.name, view_tensor(t, ref.shape));
            if (!inserted) {
                it->second = view_tensor(t, ref.shape);
            }
            return it->second;
        }
        return t;
    }

    // For mapped/temporary tensors, allocate if needed
    auto it = mTensorMap.find(ref.name);
    if (it != mTensorMap.end()) {
        if (std::getenv("SUROGATE_TRACE_MOE_ALLOC") &&
            ref.name.find("permuted_input") != std::string::npos) {
            std::ostringstream oss;
            oss << "[MOE_ALLOC_HIT] name=" << ref.name << " shape=[";
            for (int i = 0; i < it->second.Rank; ++i) {
                if (i) oss << ",";
                oss << it->second.Sizes[i];
            }
            oss << "]\n";
            std::fputs(oss.str().c_str(), stderr);
        }
        return it->second;
    }

    Tensor t = mRunState.temp_alloc(ref.dtype, ref.shape);
    if (std::getenv("SUROGATE_TRACE_MOE_ALLOC") &&
        ref.name.find("permuted_input") != std::string::npos) {
        std::ostringstream oss;
        oss << "[MOE_ALLOC_NEW] name=" << ref.name << " ref_shape=[";
        for (std::size_t i = 0; i < ref.shape.size(); ++i) {
            if (i) oss << ",";
            oss << ref.shape[i];
        }
        oss << "] alloc_shape=[";
        for (int i = 0; i < t.Rank; ++i) {
            if (i) oss << ",";
            oss << t.Sizes[i];
        }
        oss << "]\n";
        std::fputs(oss.str().c_str(), stderr);
    }

    // Zero gradient tensors to prevent stale values from accumulating.
    // Gradient tensor names start with "d_" (e.g., "d_blocks[0].ln1").
    // Many backward kernels accumulate (+=) to their outputs, so they
    // need zeroed buffers to start with. Stack memory is reused across
    // micro-batches and contains stale values from previous backward passes.
    if (ref.name.size() >= 2 && ref.name[0] == 'd' && ref.name[1] == '_') {
        fill_zero(t, mRunState.MainStream);
        static int zero_trace = 0;
        if (zero_trace < 5) {
            fprintf(stderr, "[ZERO_GRAD] Zeroed %s ptr=%p\n", ref.name.c_str(), t.Data);
            zero_trace++;
        }
    }

    mTemps.push_back(t);
    auto [ins_it, inserted] = mTensorMap.emplace(ref.name, t);
    return ins_it->second;
}

void CompiledExecutor::handle_layer_start(int layer_idx) {
    if (mWeightManager && mWeightManager->is_streaming_enabled() && !mCapturing) {
        // Wait for current layer's weights
        mWeightManager->wait_for_gather(layer_idx, mRunState.MainStream);
    }

    // Prefetch next layer
    const int next_layer = layer_idx + 1;
    if (next_layer < static_cast<int>(mConfig.NumLayers) && !mCapturing) {
        if (mWeightManager && mWeightManager->is_streaming_enabled()) {
            if (mComm) {
                mWeightManager->gather_block(next_layer, *mComm, mRunState.side_stream());
            }
        }
    }

    mCurrentLayer = layer_idx;

    // DEBUG: Check previous layer MLP output for NaNs at layer boundary.
    static int pre_ln1_nan_trace = 0;
    if (pre_ln1_nan_trace < 4 && layer_idx == 3) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        auto& prev_acts = mRunState.simplified_acts(layer_idx - 1);
        log_nan_sample("PRE_LN1_MLP_DOWN", layer_idx - 1, "blocks[2].mlp_down", prev_acts.mlp_down, 3);
        pre_ln1_nan_trace++;
    }
}

void CompiledExecutor::handle_layer_end(int layer_idx) {
    // Release previous layer's weights
    if (mWeightManager && mWeightManager->is_streaming_enabled() && !mCapturing) {
        mWeightManager->release_block(layer_idx, mRunState.MainStream);
    }

    // Offload residual if enabled
    if (mRunState.has_residual_offloading() && !mCapturing) {
        mRunState.mark_residual_ready(layer_idx, mRunState.MainStream);
        mRunState.put_residual(layer_idx, mRunState.side_stream());
    }
}

void CompiledExecutor::dispatch_embedding(const CompiledOp& op) {
    Tensor& token_ids = resolve_tensor(op.inputs[0]);
    Tensor& emb = op.inputs.size() > 1 ? resolve_tensor(op.inputs[1]) : mWeights.get("embedding");
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    // One-time embedding metadata log (shapes/dtypes/pointers).
    static int emb_meta_count = 0;
    if (emb_meta_count < 3) {
        emb_meta_count++;
        fprintf(stderr,
                "[EMBED_META] token_ids: dtype=%s rank=%d sizes={%ld,%ld,%ld,%ld,%ld} ptr=%p bytes=%zu\n",
                dtype_to_str(token_ids.DType), token_ids.Rank,
                token_ids.Sizes[0], token_ids.Sizes[1], token_ids.Sizes[2], token_ids.Sizes[3], token_ids.Sizes[4],
                token_ids.Data, token_ids.bytes());
        fprintf(stderr,
                "[EMBED_META] emb: dtype=%s rank=%d sizes={%ld,%ld,%ld,%ld,%ld} ptr=%p bytes=%zu\n",
                dtype_to_str(emb.DType), emb.Rank,
                emb.Sizes[0], emb.Sizes[1], emb.Sizes[2], emb.Sizes[3], emb.Sizes[4],
                emb.Data, emb.bytes());
        fprintf(stderr,
                "[EMBED_META] out: dtype=%s rank=%d sizes={%ld,%ld,%ld,%ld,%ld} ptr=%p bytes=%zu\n",
                dtype_to_str(out.DType), out.Rank,
                out.Sizes[0], out.Sizes[1], out.Sizes[2], out.Sizes[3], out.Sizes[4],
                out.Data, out.bytes());
    }

    // One-time embedding weight sample scan for NaN/Inf and magnitude stats
    static int emb_weight_scan_count = 0;
    if (emb_weight_scan_count < 1) {
        emb_weight_scan_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        const std::size_t total = static_cast<std::size_t>(emb.nelem());
        const std::size_t sample = std::min<std::size_t>(4096, total);
        log_tensor_sample_stats("EMB_WT_SAMPLE0", emb, 0, sample);
        if (total > sample) {
            log_tensor_sample_stats("EMB_WT_SAMPLE_MID", emb, total / 2, sample);
            log_tensor_sample_stats("EMB_WT_SAMPLE_END", emb, total - sample, sample);
        }
    }

    // Token-ID bounds check and NaN-sample mapping
    static int emb_token_check_count = 0;
    if (emb_token_check_count < 5) {
        emb_token_check_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        const long BT = static_cast<long>(token_ids.nelem());
        if (BT > 0 && token_ids.Data) {
            std::vector<std::int32_t> tokens(static_cast<std::size_t>(BT));
            cudaMemcpy(tokens.data(), token_ids.Data, static_cast<std::size_t>(BT) * sizeof(std::int32_t),
                       cudaMemcpyDeviceToHost);
            std::int32_t min_tok = std::numeric_limits<std::int32_t>::max();
            std::int32_t max_tok = std::numeric_limits<std::int32_t>::min();
            int oob_count = 0;
            int first_oob_idx = -1;
            std::int32_t first_oob_val = 0;
            const int vocab = static_cast<int>(mConfig.VocabSize);
            for (long i = 0; i < BT; ++i) {
                const std::int32_t v = tokens[static_cast<std::size_t>(i)];
                if (v < min_tok) min_tok = v;
                if (v > max_tok) max_tok = v;
                if (v < 0 || v >= vocab) {
                    oob_count++;
                    if (first_oob_idx < 0) {
                        first_oob_idx = static_cast<int>(i);
                        first_oob_val = v;
                    }
                }
            }
            const std::int32_t tok3 = (BT > 3) ? tokens[3] : tokens[0];
            fprintf(stderr,
                    "[EMBED_TOKENS] BT=%ld vocab=%d min=%d max=%d oob=%d token3=%d\n",
                    BT, vocab, min_tok, max_tok, oob_count, tok3);
            if (oob_count > 0) {
                fprintf(stderr,
                        "[EMBED_TOKENS_OOB] first_idx=%d first_val=%d\n",
                        first_oob_idx, first_oob_val);
            }
        }
    }

    // One-time prefill of embedding output with NaN pattern to detect unwritten elements
    static int emb_prefill_count = 0;
    if (emb_prefill_count < 1) {
        emb_prefill_count++;
        cudaMemsetAsync(out.Data, 0xFF, out.bytes(), mRunState.MainStream);
        fprintf(stderr, "[EMBED_PREFILL] out.Data=%p bytes=%zu\n", out.Data, out.bytes());
    }

    // DEBUG: Print actual tokens being used by embedding
    static int emb_debug_count = 0;
    if (emb_debug_count < 20) {
        emb_debug_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<std::int32_t> toks(4);
        cudaMemcpy(toks.data(), token_ids.Data, 4 * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[EMBEDDING] token_ids.Data=%p (rs.Inputs=%p) tokens[0..3]=%d,%d,%d,%d\n",
                token_ids.Data, mRunState.Inputs.Data, toks[0], toks[1], toks[2], toks[3]);
    }

    encoder_forward(out, token_ids, emb, std::nullopt,
                    static_cast<int>(mB), static_cast<int>(mT),
                    mConfig.HiddenSize, mConfig.VocabSize, mRunState.MainStream);

    // DEBUG: Print embedding output after encoder_forward
    // Print at offset 3*C to see embedding of token[3] which varies across micro-steps
    static int emb_out_count = 0;
    if (emb_out_count < 20) {
        emb_out_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> vals(4);
        const std::size_t offset = 3 * static_cast<std::size_t>(mConfig.HiddenSize);
        cudaMemcpy(vals.data(), reinterpret_cast<float*>(out.Data) + offset, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[EMBEDDING_OUT] out.Data=%p token[3] embed vals=%.6f,%.6f,%.6f,%.6f\n",
                out.Data, vals[0], vals[1], vals[2], vals[3]);
        const std::size_t row_elems = static_cast<std::size_t>(mConfig.HiddenSize);
        const std::size_t row_sample = std::min<std::size_t>(row_elems, 256);
        log_tensor_sample_stats("EMBED_OUT_ROW3", out, offset, row_sample);
    }

    // NaN detection for embedding output (token 3)
    log_nan_sample("FWD_EMBED_OUT", -1, op.outputs[0].name, out, 3);
    if (tensor_sample_has_nan_or_inf(out, 3)) {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<std::int32_t> toks(4);
        cudaMemcpy(toks.data(), token_ids.Data, 4 * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
        const std::int32_t tok3 = toks[3];
        fprintf(stderr, "[EMBED_NAN_TOKEN] token3_id=%d\n", tok3);
        // Log full-row stats for the embedding weight and output row at token3.
        if (tok3 >= 0 && tok3 < static_cast<std::int32_t>(mConfig.VocabSize)) {
            const std::size_t row_elems = static_cast<std::size_t>(mConfig.HiddenSize);
            const std::size_t emb_offset = static_cast<std::size_t>(tok3) * row_elems;
            const std::size_t out_offset = 3 * row_elems;
            log_tensor_sample_stats("EMB_WT_ROW_TOK3", emb, emb_offset, row_elems);
            log_tensor_sample_stats("EMB_OUT_ROW3_FULL", out, out_offset, row_elems);
        }
        throw std::runtime_error("Embedding output contains NaNs; aborting.");
    }
}

void CompiledExecutor::dispatch_zeros(const CompiledOp& op) {
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    fill_zero(out, mRunState.MainStream);
}

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

void CompiledExecutor::dispatch_add(const CompiledOp& op) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    vector_add_sr(out, a, b, 1.0f, static_cast<long>(a.nelem()), 0, mRunState.MainStream);
}

void CompiledExecutor::dispatch_matmul(const CompiledOp& op, const modules::ForwardHook* hook) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    std::optional<Tensor> bias;
    if (op.type == CompiledOpType::MatmulBias && op.inputs.size() > 2) {
        bias = resolve_tensor(op.inputs[2]);
    }

    int M = 0, N = 0, K = 0;
    matmul_dims(a, b, op.attrs.transpose, M, N, K);

    bool used_recipe = false;
    modules::MatmulContext ctx{};
    modules::MatmulContext* ctx_ptr = nullptr;
    if (mRecipe && op.attrs.transpose == EMMTranspose::NT && a.Sizes[0] == mB * mT) {
        if (op.attrs.allow_quant && op.attrs.matmul_op.has_value()) {
            ctx.out = &out;
            ctx.inp = &a;
            ctx.weight = &b;
            ctx.bias = bias ? &*bias : nullptr;
            ctx.B = static_cast<int>(mB);
            ctx.T = static_cast<int>(mT);
            ctx.C_in = K;
            ctx.C_out = N;
            ctx.run_state = &mRunState;
            ctx.stream = mRunState.MainStream;
            ctx.layer_idx = op.attrs.layer_idx;
            ctx.op = *op.attrs.matmul_op;
            ctx.allow_fp8 = mRecipe->uses_fp8_forward();
            ctx.allow_fp4 = mRecipe->uses_fp4_forward();

            // FP8/FP4 buffers would be set here via pre-resolved cache
            if (ctx.allow_fp8) {
                ctx.inp_quant = fp8_forward_buffer(mRunState, *op.attrs.matmul_op);
                ctx.delayed_quantizer_idx = fp8_quantizer_index(mRunState, *op.attrs.matmul_op, op.attrs.layer_idx);
            }

            mRecipe->forward_matmul(ctx);
            used_recipe = true;
            ctx_ptr = &ctx;
        }
    }

    if (!used_recipe) {
        EMMTranspose mode_col = swap_transpose(op.attrs.transpose);
        matmul(out, b, a, bias, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, false, mRunState.MainStream);
    }

    if (mForwardPlan && op.attrs.matmul_op.has_value() && op.attrs.layer_idx >= 0 &&
        static_cast<std::size_t>(op.attrs.layer_idx) < mForwardPlan->size() &&
        *op.attrs.matmul_op != modules::MatmulOp::LMHead) {
        MatmulForwardPlan plan{};
        plan.valid = true;
        plan.use_recipe = used_recipe;
        plan.has_bias = bias.has_value();
        if (used_recipe && ctx_ptr) {
            plan.allow_fp8 = ctx_ptr->allow_fp8;
            plan.allow_fp4 = ctx_ptr->allow_fp4;
            plan.delayed_quantizer_idx = ctx_ptr->delayed_quantizer_idx;
            plan.use_fp8_cache = (ctx_ptr->cached_weight && ctx_ptr->cached_weight->Data);
            plan.use_fp4_cache = (ctx_ptr->cached_fp4_data && ctx_ptr->cached_fp4_scales);
        }
        auto& layer_plan = (*mForwardPlan)[static_cast<std::size_t>(op.attrs.layer_idx)];
        switch (*op.attrs.matmul_op) {
            case modules::MatmulOp::QKV:
                layer_plan.qkv = plan;
                break;
            case modules::MatmulOp::AttnOut:
                layer_plan.out_proj = plan;
                break;
            case modules::MatmulOp::MLPUp:
                layer_plan.mlp_up = plan;
                break;
            case modules::MatmulOp::MLPDown:
                layer_plan.mlp_down = plan;
                break;
            default:
                break;
        }
    }

    // Hook invocation
    if (hook && *hook && op.attrs.forward_hook_point.has_value()) {
        (*hook)(op.attrs.layer_idx, mRunState.MainStream, *op.attrs.forward_hook_point, mHookContext);
    }

    // NaN detection for forward matmuls (token 3 sample)
    if (op.attrs.matmul_op.has_value()) {
        const long sample_token = 3;
        switch (*op.attrs.matmul_op) {
            case modules::MatmulOp::QKV:
                log_nan_sample("FWD_QKV", op.attrs.layer_idx, op.outputs[0].name, out, sample_token);
                if (op.attrs.layer_idx == 0) {
                    log_tensor_stats("FWD_QKV", op.attrs.layer_idx, op.outputs[0].name, out, 4096);
                    if (used_recipe) {
                        const auto recipe_name = mRecipe ? mRecipe->name() : std::string_view("none");
                        fprintf(stderr,
                                "[MATMUL_RECIPE] op=QKV layer=%d recipe=%.*s allow_fp8=%d allow_fp4=%d use_fp8_cache=%d use_fp4_cache=%d\n",
                                op.attrs.layer_idx,
                                static_cast<int>(recipe_name.size()),
                                recipe_name.data(),
                                ctx_ptr ? (ctx_ptr->allow_fp8 ? 1 : 0) : 0,
                                ctx_ptr ? (ctx_ptr->allow_fp4 ? 1 : 0) : 0,
                                ctx_ptr ? (ctx_ptr->cached_weight && ctx_ptr->cached_weight->Data ? 1 : 0) : 0,
                                ctx_ptr ? (ctx_ptr->cached_fp4_data && ctx_ptr->cached_fp4_scales ? 1 : 0) : 0);
                    } else {
                        fprintf(stderr, "[MATMUL_RECIPE] op=QKV layer=%d used_recipe=0\n",
                                op.attrs.layer_idx);
                    }
                }
                break;
            case modules::MatmulOp::AttnOut:
                log_nan_sample("FWD_ATTN_OUT", op.attrs.layer_idx, op.outputs[0].name, out, sample_token);
                break;
            case modules::MatmulOp::MLPUp:
                log_nan_sample("FWD_MLP_UP", op.attrs.layer_idx, op.outputs[0].name, out, sample_token);
                break;
            case modules::MatmulOp::MLPDown:
                log_nan_sample("FWD_MLP_DOWN", op.attrs.layer_idx, op.outputs[0].name, out, sample_token);
                break;
            default:
                break;
        }
    }
}

void CompiledExecutor::dispatch_bias_add(const CompiledOp& op) {
    Tensor& x = resolve_tensor(op.inputs[0]);
    Tensor& bias = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    const std::size_t bytes = static_cast<std::size_t>(x.nelem()) * get_dtype_size(x.DType);
    CUDA_CHECK(cudaMemcpyAsync(out.Data, x.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
    add_bias_tensor(out, bias, static_cast<int>(x.Sizes[0]), static_cast<int>(x.Sizes[1]),
                    static_cast<int>(x.Sizes[2]), mRunState.MainStream);
}

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

void CompiledExecutor::dispatch_matmul_swiglu(const CompiledOp& op) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    Tensor& up_out = ensure_output_tensor(op.outputs[1]);

    int M = 0, N = 0, K = 0;
    matmul_dims(a, b, op.attrs.transpose, M, N, K);
    const long D = N / 2;

    matmul(up_out, b, a, std::nullopt, nullptr, nullptr,
           mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
           N, M, K, swap_transpose(op.attrs.transpose), false, mRunState.MainStream);

    Tensor up_3d = view_tensor(up_out, {mB, mT, static_cast<long>(N)});
    Tensor out_3d = view_tensor(out, {mB, mT, D});
    swiglu_forward(out_3d, up_3d, nullptr, static_cast<int>(mB),
                   static_cast<int>(mT), static_cast<int>(D), mRunState.MainStream);
}

void CompiledExecutor::dispatch_qkv_qk_norm_rope(const CompiledOp& op) {
    Tensor& qkv_in = resolve_tensor(op.inputs[0]);
    Tensor& q_norm = resolve_tensor(op.inputs[1]);
    Tensor& k_norm = resolve_tensor(op.inputs[2]);
    Tensor& freqs = resolve_tensor(op.inputs[3]);
    Tensor& pos_ids = resolve_tensor(op.inputs[4]);

    // Get output tensor from pre-allocated slot if available
    Tensor& qkv_out = ensure_output_tensor(op.outputs[0]);
    Tensor& q_rstd = ensure_output_tensor(op.outputs[1]);
    Tensor& k_rstd = ensure_output_tensor(op.outputs[2]);

    // DEBUG: Trace qkv_in and qkv_out for layer 0
    int debug_layer_idx = -1;
    std::string debug_field;
    parse_block_param(op.inputs[0].name, debug_layer_idx, debug_field);
    static int fwd_qknorm_count = 0;
    if (debug_layer_idx == 0 && fwd_qknorm_count < 3) {
        fwd_qknorm_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> in_vals(4), out_vals(4);
        cudaMemcpy(in_vals.data(), qkv_in.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_vals.data(), qkv_out.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[FWD_QK_NORM_ROPE] Layer %d: qkv_in=%s ptr=%p, qkv_out=%s ptr=%p, same=%d\n",
                debug_layer_idx, op.inputs[0].name.c_str(), qkv_in.Data,
                op.outputs[0].name.c_str(), qkv_out.Data, (qkv_in.Data == qkv_out.Data) ? 1 : 0);
        fprintf(stderr, "[FWD_QK_NORM_ROPE] Layer %d: qkv_in values=%.6f,%.6f,%.6f,%.6f\n",
                debug_layer_idx, in_vals[0], in_vals[1], in_vals[2], in_vals[3]);
    }

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const bool cudnn_gqa_ok = (Hq == Hkv);
    const int qkv_channels = Hs * (Hq + 2 * Hkv);

    // If input and output are different buffers, copy input to output first
    // The kernel operates in-place on the output buffer
    if (qkv_in.Data != qkv_out.Data) {
        cudaMemcpyAsync(qkv_out.Data, qkv_in.Data,
                        qkv_in.bytes(),
                        cudaMemcpyDeviceToDevice, mRunState.MainStream);
    }

    Tensor qkv_view = (qkv_out.Rank == 4) ? view_tensor(qkv_out, {mB, mT, qkv_channels}) : qkv_out;
    int rotary_dim = op.attrs.rotary_dim;

    const bool rope_fusable = (rotary_dim > 0)
        && ((Hs % 2) == 0)
        && (((Hs / 2) % 32) == 0)
        && (freqs.Rank >= 2)
        && (freqs.Sizes[1] >= Hs)
        && (qkv_view.Rank == 3);

    if (mForwardPlan) {
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(op.inputs[0].name, layer_idx, field) &&
            layer_idx >= 0 && static_cast<std::size_t>(layer_idx) < mForwardPlan->size()) {
            AttnForwardPlan plan{};
            plan.valid = true;
            plan.use_qk_norm = true;
            plan.rope_fused = rope_fusable;
            plan.use_cudnn = cudnn_gqa_ok;
            plan.rotary_dim = rotary_dim;
            (*mForwardPlan)[static_cast<std::size_t>(layer_idx)].attn = plan;
        }
    }

    if (rope_fusable) {
        qkv_qk_norm_rope_forward(qkv_view, q_rstd, k_rstd, q_norm, k_norm,
                                 freqs, reinterpret_cast<int*>(pos_ids.Data),
                                 op.attrs.eps,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 Hq, Hkv, Hs, mRunState.MainStream);
    } else {
        const int q_rows = Hq * Hs;
        qkv_head_rmsnorm_forward(qkv_view, q_rstd, q_norm,
                                 op.attrs.eps,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 qkv_channels, Hq, Hs, 0, mRunState.MainStream);
        qkv_head_rmsnorm_forward(qkv_view, k_rstd, k_norm,
                                 op.attrs.eps,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 qkv_channels, Hkv, Hs, q_rows, mRunState.MainStream);
        rope_forward(qkv_out, qkv_out, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                     static_cast<int>(mB), static_cast<int>(mT), Hq, Hkv, Hs, rotary_dim, mRunState.MainStream);
    }

    // DEBUG: Print qkv_in and qkv_out after computation for layer 0
    if (debug_layer_idx == 0 && fwd_qknorm_count <= 3) {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> in_vals(4), out_vals(4);
        cudaMemcpy(in_vals.data(), qkv_in.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_vals.data(), qkv_out.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[FWD_QK_NORM_ROPE] Layer %d AFTER: qkv_in values=%.6f,%.6f,%.6f,%.6f\n",
                debug_layer_idx, in_vals[0], in_vals[1], in_vals[2], in_vals[3]);
        fprintf(stderr, "[FWD_QK_NORM_ROPE] Layer %d AFTER: qkv_out values=%.6f,%.6f,%.6f,%.6f\n",
                debug_layer_idx, out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
    }

    mTensorMap[op.outputs[0].name] = qkv_out;

    // NaN detection for QKV after RoPE/QK-norm (layer 25, token 3)
    int layer_idx = -1;
    std::string field;
    if (parse_block_param(op.outputs[0].name, layer_idx, field)) {
        const long sample_token = 3;
        log_nan_sample("FWD_QKV_ROPE", layer_idx, op.outputs[0].name, qkv_out, sample_token);
        if (op.outputs.size() > 1) {
            Tensor& q_rstd = resolve_tensor(op.outputs[1]);
            log_nan_sample("FWD_Q_RSTD", layer_idx, op.outputs[1].name, q_rstd, sample_token);
        }
        if (op.outputs.size() > 2) {
            Tensor& k_rstd = resolve_tensor(op.outputs[2]);
            log_nan_sample("FWD_K_RSTD", layer_idx, op.outputs[2].name, k_rstd, sample_token);
        }
        if (layer_idx == 0) {
            log_tensor_stats("FWD_QKV_IN", layer_idx, op.inputs[0].name, qkv_in, 4096);
            log_tensor_stats("FWD_QKV_ROPE", layer_idx, op.outputs[0].name, qkv_out, 4096);
            log_tensor_stats("FWD_Q_NORM_W", layer_idx, op.inputs[1].name, q_norm, 2048);
            log_tensor_stats("FWD_K_NORM_W", layer_idx, op.inputs[2].name, k_norm, 2048);
            if (op.outputs.size() > 1) {
                Tensor& q_rstd = resolve_tensor(op.outputs[1]);
                log_tensor_stats("FWD_Q_RSTD", layer_idx, op.outputs[1].name, q_rstd, 2048);
            }
            if (op.outputs.size() > 2) {
                Tensor& k_rstd = resolve_tensor(op.outputs[2]);
                log_tensor_stats("FWD_K_RSTD", layer_idx, op.outputs[2].name, k_rstd, 2048);
            }
        }
    }
}

void CompiledExecutor::dispatch_rope(const CompiledOp& op) {
    Tensor& qkv = resolve_tensor(op.inputs[0]);
    Tensor& freqs = resolve_tensor(op.inputs[1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[2]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());

    if (mForwardPlan) {
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(op.inputs[0].name, layer_idx, field) &&
            layer_idx >= 0 && static_cast<std::size_t>(layer_idx) < mForwardPlan->size()) {
            AttnForwardPlan plan{};
            plan.valid = true;
            plan.use_qk_norm = false;
            plan.rope_fused = false;
            plan.use_cudnn = true;
            plan.rotary_dim = op.attrs.rotary_dim;
            (*mForwardPlan)[static_cast<std::size_t>(layer_idx)].attn = plan;
        }
    }

    rope_forward(out, qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                 static_cast<int>(mB), static_cast<int>(mT), Hq, Hkv, Hs,
                 op.attrs.rotary_dim, mRunState.MainStream);
}

void CompiledExecutor::dispatch_flash_attention(const CompiledOp& op) {
    Tensor& qkv = resolve_tensor(op.inputs[0]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    Tensor& lse = ensure_output_tensor(op.outputs[1]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const bool cudnn_gqa_ok = (Hq == Hkv);

    if (!mRunState.scratch().cudnn_workspace.Data) {
        mRunState.temp_acquire(mRunState.scratch().cudnn_workspace);
        mTemps.push_back(mRunState.scratch().cudnn_workspace);
    }

    // Debug: log workspace size vs required cuDNN size (limited)
    {
        static int ws_log_count = 0;
        if (ws_log_count < 4) {
            const std::size_t ws_bytes =
                static_cast<std::size_t>(mRunState.scratch().cudnn_workspace.nelem()) *
                static_cast<std::size_t>(get_dtype_size(mRunState.scratch().cudnn_workspace.DType));
            const std::size_t ws_needed =
                cudnn_get_workspace_size(static_cast<int>(mB), static_cast<int>(mT),
                                         Hq, Hkv, Hs, mRunState.CudnnHandle);
            fprintf(stderr,
                    "[CUDNN_WS] B=%d T=%d Hq=%d Hkv=%d HS=%d ws_ptr=%p ws_bytes=%zu ws_needed=%zu\n",
                    static_cast<int>(mB), static_cast<int>(mT), Hq, Hkv, Hs,
                    mRunState.scratch().cudnn_workspace.Data, ws_bytes, ws_needed);
            ws_log_count++;
        }
    }

    // NaN detection on flash-attention input (qkv) before cuDNN call.
    {
        const long sample_token = 3;
        log_nan_sample("FWD_ATTN_IN", op.inputs[0].layer_idx, op.inputs[0].name, qkv, sample_token);
    }
    // One-time stats for layer 0 qkv input to detect large magnitudes or NaNs
    if (op.inputs[0].layer_idx == 0) {
        log_tensor_stats("FWD_ATTN_IN", op.inputs[0].layer_idx, op.inputs[0].name, qkv, 4096);
    }

    // cuDNN attention uses custom strides that map logical (B, Hq, T, HS) dims
    // to (B, T, Hq, HS) contiguous memory layout:
    //   Output strides: {Hq*HS*T, HS, Hq*HS, 1} for dims {B, Hq, T, HS}
    //   This maps element [b,h,t,s] to offset: b*Hq*HS*T + t*Hq*HS + h*HS + s
    //   Which is exactly (B, T, Hq, HS) contiguous layout.
    // DSL allocates output as (B, T, Hq*HS) = (B, T, Hq, HS) contiguous, so
    // we can pass it directly to cuDNN without any transpose.
    //
    // Similarly for QKV input: cuDNN expects (B, T, H, HS) contiguous where H = Hq + 2*Hkv.

    if (!cudnn_gqa_ok) {
        static int cudnn_skip_count = 0;
        if (cudnn_skip_count < 4) {
            fprintf(stderr,
                    "[CUDNN_ATTN_SKIP] Hq=%d Hkv=%d causal=1 reason=GQA\n",
                    Hq, Hkv);
            cudnn_skip_count++;
        }
        attention_forward_custom(out, lse, qkv,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 Hq, Hkv, Hs, mRunState.MainStream);
    } else {
        attention_forward_cudnn(out, lse, qkv, mRunState.scratch().cudnn_workspace,
                                mRunState.CudnnHandle, static_cast<int>(mB), static_cast<int>(mT),
                                Hq, Hkv, Hs, mRunState.MainStream);
    }

    // DEBUG: Print first att values for layer 0
    if (op.outputs[0].layer_idx == 0) {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> vals(8);
        cudaMemcpy(vals.data(), out.Data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[FWD_ATTN] Layer 0 att output ptr=%p, values=%.6f, %.6f, %.6f, %.6f\n",
                out.Data, vals[0], vals[1], vals[2], vals[3]);
        log_tensor_stats_ex("FWD_ATTN_OUT", op.outputs[0].layer_idx, op.outputs[0].name, out, 4096, false);
        log_tensor_stats_ex("FWD_LSE", op.outputs[0].layer_idx, op.outputs[1].name, lse, 4096, false);
    }

    // NaN detection for attention forward (layer 25, token 3)
    {
        const long sample_token = 3;
        log_nan_sample("FWD_ATTN", op.outputs[0].layer_idx, op.outputs[0].name, out, sample_token);
        log_nan_sample("FWD_LSE", op.outputs[0].layer_idx, op.outputs[1].name, lse, sample_token);
        const bool out_nan = tensor_sample_has_nan_or_inf(out, sample_token);
        const bool lse_nan = tensor_sample_has_nan_or_inf(lse, sample_token);
        if (out_nan || lse_nan) {
            log_tensor_stats_ex("FWD_ATTN_OUT_NAN", op.outputs[0].layer_idx, op.outputs[0].name, out, 4096, true);
            log_tensor_stats_ex("FWD_LSE_NAN", op.outputs[0].layer_idx, op.outputs[1].name, lse, 4096, true);
            log_tensor_stats_ex("FWD_ATTN_IN_NAN", op.inputs[0].layer_idx, op.inputs[0].name, qkv, 4096, true);
            // Fallback: recompute attention using the custom kernel to isolate cuDNN issues.
            fprintf(stderr, "[FWD_ATTN_FALLBACK] layer=%d using custom kernel\n", op.outputs[0].layer_idx);
            attention_forward_custom(out, lse, qkv,
                                     static_cast<int>(mB), static_cast<int>(mT),
                                     Hq, Hkv, Hs, mRunState.MainStream);
            log_tensor_stats_ex("FWD_ATTN_FALLBACK_OUT", op.outputs[0].layer_idx, op.outputs[0].name, out, 4096, true);
            log_tensor_stats_ex("FWD_LSE_FALLBACK", op.outputs[0].layer_idx, op.outputs[1].name, lse, 4096, true);
        }
    }
}

void CompiledExecutor::dispatch_cross_entropy_loss(const CompiledOp& op) {
    Tensor& logits = resolve_tensor(op.inputs[0]);
    Tensor& targets = resolve_tensor(op.inputs[1]);
    Tensor& loss = ensure_output_tensor(op.outputs[0]);

    const int BT = static_cast<int>(logits.Sizes[0]);
    const int V = static_cast<int>(logits.Sizes[1]);
    const int P = V;

    Tensor logsumexp_view{};
    Tensor* logsumexp = nullptr;
    if (mRunState.scratch().cross_entropy_logsumexp.Data) {
        logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
        logsumexp_view.Sizes[0] = BT;
        logsumexp_view.Rank = 1;
        logsumexp = &logsumexp_view;
    }

    if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
        if (!mRunState.scratch().cross_entropy_chunk_logsumexp.Data) {
            throw std::runtime_error("cross_entropy_loss: chunk logsumexp buffer is not allocated");
        }
        const int n_chunks = (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE;
        Tensor chunk_lse = mRunState.scratch().cross_entropy_chunk_logsumexp;
        chunk_lse.Sizes[0] = BT;
        chunk_lse.Sizes[1] = n_chunks;
        chunk_lse.Rank = 2;

        chunked_cross_entropy_forward(logits, loss, logsumexp, chunk_lse, targets,
                                      &mRunState.ValidTokenCount,
                                      op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                      BT, V, P, n_chunks, mRunState.MainStream);
    } else {
        fused_cross_entropy_forward(logits, loss, logsumexp, targets,
                                    &mRunState.ValidTokenCount,
                                    op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                    BT, V, P, mRunState.MainStream);
    }
}

void CompiledExecutor::dispatch_fused_lm_head_loss(const CompiledOp& op) {
    Tensor& xF_flat = resolve_tensor(op.inputs[0]);
    Tensor& weight = resolve_tensor(op.inputs[1]);
    Tensor& targets = resolve_tensor(op.inputs[2]);
    Tensor& loss = ensure_output_tensor(op.outputs[0]);

    const long BT = xF_flat.Sizes[0];
    const int V = static_cast<int>(weight.Sizes[0]);
    const int C = static_cast<int>(weight.Sizes[1]);
    const int P = V;

    const int lmhead_chunks = std::max(1, mOptions.LMHeadChunks);
    const long nano_batch_size = BT / static_cast<long>(lmhead_chunks);
    const int n_chunks = (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE;

    const bool need_lm_head =
        mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled());
    if (need_lm_head && mComm) {
        mWeightManager->gather_lm_head(*mComm, mRunState.MainStream);
    }

    const std::size_t xf_stride = get_dtype_size(xF_flat.DType);
    const std::size_t tgt_stride = get_dtype_size(targets.DType);
    const std::size_t loss_stride = get_dtype_size(loss.DType);
    const std::size_t lse_stride = get_dtype_size(ETensorDType::FP32);
    const std::size_t chunk_lse_stride = lse_stride * static_cast<std::size_t>(n_chunks);

    for (int nano_step = 0; nano_step < lmhead_chunks; ++nano_step) {
        const long token_offset = static_cast<long>(nano_step) * nano_batch_size;

        Tensor xF_slice = xF_flat;
        xF_slice.Data = static_cast<std::byte*>(xF_slice.Data) +
                        static_cast<std::size_t>(token_offset) * xf_stride * static_cast<std::size_t>(C);
        xF_slice.Sizes[0] = nano_batch_size;
        xF_slice.Sizes[1] = C;
        xF_slice.Rank = 2;

        Tensor tgt_slice = targets;
        tgt_slice.Data = static_cast<std::byte*>(tgt_slice.Data) +
                         static_cast<std::size_t>(token_offset) * tgt_stride;
        tgt_slice.Sizes[0] = nano_batch_size;
        tgt_slice.Rank = 1;

        Tensor loss_slice = loss;
        loss_slice.Data = static_cast<std::byte*>(loss_slice.Data) +
                          static_cast<std::size_t>(token_offset) * loss_stride;
        loss_slice.Sizes[0] = nano_batch_size;
        loss_slice.Rank = 1;

        Tensor logits = mRunState.non_block_activations().output;
        logits.Sizes[0] = nano_batch_size;
        logits.Sizes[1] = V;
        logits.Rank = 2;

        matmul(logits, weight, xF_slice,
               std::nullopt, nullptr, nullptr, mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               static_cast<int>(V), static_cast<int>(nano_batch_size), static_cast<int>(C),
               swap_transpose(EMMTranspose::NT), false, mRunState.MainStream);

        Tensor logsumexp_view{};
        Tensor* logsumexp = nullptr;
        if (mRunState.scratch().cross_entropy_logsumexp.Data) {
            logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
            logsumexp_view.Data = static_cast<std::byte*>(logsumexp_view.Data) +
                                  static_cast<std::size_t>(token_offset) * lse_stride;
            logsumexp_view.Sizes[0] = nano_batch_size;
            logsumexp_view.Rank = 1;
            logsumexp = &logsumexp_view;
        }

        if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
            if (!mRunState.scratch().cross_entropy_chunk_logsumexp.Data) {
                throw std::runtime_error("fused_lm_head_loss: chunk logsumexp buffer is not allocated");
            }
            Tensor chunk_lse = mRunState.scratch().cross_entropy_chunk_logsumexp;
            chunk_lse.Data = static_cast<std::byte*>(chunk_lse.Data) +
                             static_cast<std::size_t>(token_offset) * chunk_lse_stride;
            chunk_lse.Sizes[0] = nano_batch_size;
            chunk_lse.Sizes[1] = n_chunks;
            chunk_lse.Rank = 2;

            chunked_cross_entropy_forward(logits, loss_slice, logsumexp, chunk_lse, tgt_slice,
                                          &mRunState.ValidTokenCount,
                                          op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                          static_cast<int>(nano_batch_size), V, P, n_chunks, mRunState.MainStream);
        } else {
            fused_cross_entropy_forward(logits, loss_slice, logsumexp, tgt_slice,
                                        &mRunState.ValidTokenCount,
                                        op.attrs.compute_accuracy ? &mRunState.CorrectCount : nullptr,
                                        static_cast<int>(nano_batch_size), V, P, mRunState.MainStream);
        }
    }

    if (need_lm_head) {
        mWeightManager->release_lm_head(mRunState.MainStream);
    }
}

// MoE forward dispatch implementations

void CompiledExecutor::dispatch_silu(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    const long N = static_cast<long>(inp.nelem());
    silu_forward(out, inp, N, mRunState.MainStream);

    mTensorMap[op.outputs[0].name] = out;
}

void CompiledExecutor::dispatch_mul(const CompiledOp& op) {
    // Element-wise multiplication kernel not yet implemented
    // This is only needed for shared_expert path which is disabled by default
    throw std::runtime_error("CompiledExecutor: element-wise mul operation not yet implemented. "
                             "Set use_shared_expert=False in your model config.");
}

void CompiledExecutor::dispatch_moe_softmax(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    int layer_idx = op.attrs.layer_idx;
    std::string field;
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        parse_block_param(name, layer_idx, field);
    }

    log_moe_gate_up_weight_sample("PRE_MOE_SOFTMAX", layer_idx, mMicroStep, mWeights, mConfig);

    const int num_tokens = static_cast<int>(inp.Sizes[0]);
    const int num_experts = static_cast<int>(inp.Sizes[1]);

    // Allocate output with same shape as input (softmax doesn't change shape)
    std::vector<long> out_shape = {static_cast<long>(num_tokens), static_cast<long>(num_experts)};
    Tensor out = mRunState.temp_alloc(inp.DType, out_shape);
    mTemps.push_back(out);

    if (inp.DType == ETensorDType::BF16) {
        moe_softmax_forward(out.get<nv_bfloat16>(),
                            inp.get<nv_bfloat16>(),
                            num_tokens, num_experts, mRunState.MainStream);
    } else {
        moe_softmax_forward(out.get<float>(),
                            inp.get<float>(),
                            num_tokens, num_experts, mRunState.MainStream);
    }

    const bool moe_route_trace = (std::getenv("SUROGATE_MOE_ROUTE_TRACE") != nullptr);
    static int moe_route_trace_count = 0;
    if (moe_route_trace && moe_route_trace_count < 4) {
        const std::size_t total = static_cast<std::size_t>(out.nelem());
        const std::size_t sample = std::min<std::size_t>(4096, total);
        log_tensor_sample_stats("MOE_ROUTER_PROBS", out, 0, sample);
        moe_route_trace_count++;
    }

    mTensorMap[op.outputs[0].name] = out;
}

void CompiledExecutor::dispatch_moe_sigmoid(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);

    // Determine shape - input might have rank=0 if shape wasn't propagated at compile time
    // In MoE context, the input is router logits with shape [num_tokens, num_experts]
    std::vector<long> shape;
    if (inp.Rank == 2) {
        shape = {inp.Sizes[0], inp.Sizes[1]};
    } else if (inp.Rank == 0 && mConfig.NumExperts > 0) {
        // Infer shape from config and current dimensions
        const long num_tokens = mB * mT;
        const long num_experts = static_cast<long>(mConfig.NumExperts);
        shape = {num_tokens, num_experts};
        // Also fix the input tensor shape
        inp.Rank = 2;
        inp.Sizes[0] = num_tokens;
        inp.Sizes[1] = num_experts;
    } else {
        // Fallback to input shape if available
        for (int i = 0; i < inp.Rank; ++i) {
            shape.push_back(inp.Sizes[i]);
        }
    }

    // Allocate output with same shape as input
    Tensor out = mRunState.temp_alloc(inp.DType, shape);
    mTemps.push_back(out);

    const int num_elements = static_cast<int>(out.nelem());

    if (inp.DType == ETensorDType::BF16) {
        moe_sigmoid_forward(out.get<nv_bfloat16>(),
                            inp.get<nv_bfloat16>(),
                            num_elements, mRunState.MainStream);
    } else {
        moe_sigmoid_forward(out.get<float>(),
                            inp.get<float>(),
                            num_elements, mRunState.MainStream);
    }

    const bool moe_route_trace = (std::getenv("SUROGATE_MOE_ROUTE_TRACE") != nullptr);
    static int moe_route_trace_count = 0;
    if (moe_route_trace && moe_route_trace_count < 4) {
        const std::size_t total = static_cast<std::size_t>(out.nelem());
        const std::size_t sample = std::min<std::size_t>(4096, total);
        log_tensor_sample_stats("MOE_ROUTER_PROBS", out, 0, sample);
        moe_route_trace_count++;
    }

    mTensorMap[op.outputs[0].name] = out;
}

void CompiledExecutor::dispatch_moe_topk(const CompiledOp& op) {
    Tensor& probs = resolve_tensor(op.inputs[0]);
    Tensor& weights = ensure_output_tensor(op.outputs[0]);
    Tensor& indices = ensure_output_tensor(op.outputs[1]);

    const int num_tokens = static_cast<int>(probs.Sizes[0]);
    const int num_experts = static_cast<int>(probs.Sizes[1]);
    const int top_k = op.attrs.top_k;
    const bool normalize = op.attrs.normalize_weights;

    if (probs.DType == ETensorDType::BF16) {
        moe_topk_forward(indices.get<int>(),
                         weights.get<nv_bfloat16>(),
                         probs.get<nv_bfloat16>(),
                         num_tokens, num_experts, top_k, normalize, mRunState.MainStream);
    } else {
        moe_topk_forward(indices.get<int>(),
                         weights.get<float>(),
                         probs.get<float>(),
                         num_tokens, num_experts, top_k, normalize, mRunState.MainStream);
    }

    mTensorMap[op.outputs[0].name] = weights;
    mTensorMap[op.outputs[1].name] = indices;

    const bool moe_route_trace = (std::getenv("SUROGATE_MOE_ROUTE_TRACE") != nullptr);
    static int moe_route_topk_trace = 0;
    if (moe_route_trace && moe_route_topk_trace < 4) {
        const std::size_t total = static_cast<std::size_t>(weights.nelem());
        const std::size_t sample = std::min<std::size_t>(4096, total);
        log_tensor_sample_stats("MOE_TOPK_WEIGHTS", weights, 0, sample);
        moe_route_topk_trace++;
    }

    // Resolve layer index for debugging and selective expert refresh.
    int layer_idx = op.attrs.layer_idx;
    std::string field;
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        parse_block_param(name, layer_idx, field);
    }

    log_moe_gate_up_weight_sample("PRE_MOE_TOPK", layer_idx, mMicroStep, mWeights, mConfig);

    // Force selective expert dequantization for this layer based on top-k indices.
    if (layer_idx >= 0) {
        if (auto* provider = mWeights.qlora_provider()) {
            modules::SelectiveExpertInfo selection;
            selection.build_from_router_output(indices, num_experts, mRunState.MainStream);
            (void)provider->refresh_moe_experts(layer_idx, selection, mRunState.MainStream);
            log_moe_gate_up_weight_sample("POST_MOE_TOPK_REFRESH",
                                          layer_idx, mMicroStep, mWeights, mConfig);
        }
    }

    // DEBUG: Verify routing indices are within [0, num_experts).
    static int moe_topk_trace = 0;
    if (layer_idx < 0 || layer_idx < 4 || layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
        const int total = num_tokens * top_k;
        std::vector<int> host_idx(total, -1);
        CUDA_CHECK(cudaMemcpy(host_idx.data(), indices.get<int>(),
                              static_cast<std::size_t>(total) * sizeof(int),
                              cudaMemcpyDeviceToHost));
        int min_idx = std::numeric_limits<int>::max();
        int max_idx = std::numeric_limits<int>::min();
        int oob = 0;
        for (int v : host_idx) {
            min_idx = std::min(min_idx, v);
            max_idx = std::max(max_idx, v);
            if (v < 0 || v >= num_experts) {
                oob++;
            }
        }
        if (moe_topk_trace < 8 || oob > 0 || min_idx < 0) {
            fprintf(stderr,
                    "[MOE_TOPK_IDX] layer=%d shape=[%d,%d] num_experts=%d min=%d max=%d oob=%d\n",
                    layer_idx, num_tokens, top_k, num_experts, min_idx, max_idx, oob);
        }
        if (oob > 0 || min_idx < 0) {
            log_nan_sample("MOE_PROBS_NAN", layer_idx, op.inputs[0].name, probs, 3);
            log_tensor_stats_ex("MOE_PROBS_NAN", layer_idx, op.inputs[0].name, probs, 4096, true);
        }
        moe_topk_trace++;
    }
}

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

// Backward dispatch implementations
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

void CompiledExecutor::dispatch_add_backward(const CompiledOp& op) {
    // Addition backward: gradients pass through unchanged to both inputs
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    // For pre-allocated gradient slots (like d_res_ffn, d_res_att), we must copy the
    // upstream gradient into the original simplified_grads buffer. Simply aliasing
    // the data pointer causes shared storage between residual and branch gradients,
    // which breaks LoRA (it does in-place dx accumulation).
    // IMPORTANT: We must get the base tensor directly from simplified_grads(), not via
    // resolve_tensor(), because resolve_tensor() may return a view from mTensorMap.
    auto assign_output = [&](const TensorRef& ref) {
        auto trace_residual_l8 = [&](const Tensor& out_tensor, const char* tag) {
            static int add_res_trace = 0;
            if (add_res_trace < 20 &&
                (ref.name.find("d_blocks[7].mlp_down") != std::string::npos ||
                 ref.name.find("d_blocks[8].res_att") != std::string::npos ||
                 ref.name.find("d_blocks[8].res_ffn") != std::string::npos)) {
                fprintf(stderr,
                        "[ADD_BWD_L8] %s out=%s ptr=%p in=%s slot=%d ptr=%p\n",
                        tag,
                        ref.name.c_str(),
                        out_tensor.Data,
                        op.inputs[0].name.c_str(),
                        static_cast<int>(op.inputs[0].slot),
                        d_out.Data);
                log_tensor_mag_unbounded("ADD_BWD_L8_DOUT",
                                         ref.layer_idx,
                                         op.inputs[0].name,
                                         d_out,
                                         4096);
                log_tensor_mag_unbounded("ADD_BWD_L8_OUT",
                                         ref.layer_idx,
                                         ref.name,
                                         out_tensor,
                                         4096);
                add_res_trace++;
            }
        };
        auto trace_mlp_down = [&](const Tensor& out_tensor, const char* tag) {
            static int add_mlp_down_trace = 0;
            if (add_mlp_down_trace < 20 &&
                ref.name.find("d_blocks[24].mlp_down") != std::string::npos) {
                cudaStreamSynchronize(mRunState.MainStream);
                std::vector<float> in_vals(4, 0.0f), out_vals(4, 0.0f);
                const bool ok_in = copy_tensor_sample_as_f32(d_out, in_vals.size(), in_vals);
                const bool ok_out = copy_tensor_sample_as_f32(out_tensor, out_vals.size(), out_vals);
                fprintf(stderr,
                        "[ADD_BWD_MLP_DOWN] %s out=%s ptr=%p in=%s slot=%d ptr=%p ok_in=%d ok_out=%d "
                        "in_vals=%.6f,%.6f,%.6f,%.6f out_vals=%.6f,%.6f,%.6f,%.6f\n",
                        tag,
                        ref.name.c_str(),
                        out_tensor.Data,
                        op.inputs[0].name.c_str(),
                        static_cast<int>(op.inputs[0].slot),
                        d_out.Data,
                        ok_in ? 1 : 0,
                        ok_out ? 1 : 0,
                        in_vals[0], in_vals[1], in_vals[2], in_vals[3],
                        out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
                add_mlp_down_trace++;
            }
        };
        Tensor* base_grad = nullptr;
        if (ref.layer_idx >= 0) {
            auto& grads = mRunState.simplified_grads(ref.layer_idx);
            switch (ref.slot) {
                case TensorSlot::BlockDResFFN: base_grad = &grads.d_res_ffn; break;
                case TensorSlot::BlockDResAtt: base_grad = &grads.d_res_att; break;
                case TensorSlot::BlockDLN1: base_grad = &grads.d_ln1; break;
                case TensorSlot::BlockDLN2: base_grad = &grads.d_ln2; break;
                case TensorSlot::BlockDSwiGLU: base_grad = &grads.d_swiglu; break;
                case TensorSlot::BlockDAtt: base_grad = &grads.d_att; break;
                case TensorSlot::BlockDQKV: base_grad = &grads.d_qkv; break;
                case TensorSlot::BlockDMLPUp: base_grad = &grads.d_mlp_up; break;
                case TensorSlot::BlockDMLPDown: base_grad = &grads.d_mlp_down; break;
                default: break;
            }
        }

        if (base_grad) {
            if (base_grad->Data) {
                if (base_grad->DType != d_out.DType) {
                    throw std::runtime_error("dispatch_add_backward: dtype mismatch for " + ref.name);
                }
                if (base_grad->Data != d_out.Data) {
                    CUDA_CHECK(cudaMemcpyAsync(base_grad->Data, d_out.Data, d_out.bytes(),
                                               cudaMemcpyDeviceToDevice, mRunState.MainStream));
                }
                mTensorMap[ref.name] = view_tensor(*base_grad, ref.shape);
                trace_residual_l8(mTensorMap[ref.name], "copy");
                trace_mlp_down(mTensorMap[ref.name], "copy");
                return;
            }
            // For stack-allocated gradient temps, allocate proper storage instead of aliasing.
            // Aliasing to d_out can cause stale memory access when the stack is restored at
            // layer boundaries because the aliased memory gets recycled.
            const bool is_stack_grad = mRunState.large_bwd_temps_on_stack() &&
                (ref.slot == TensorSlot::BlockDQKV ||
                 ref.slot == TensorSlot::BlockDMLPUp ||
                 ref.slot == TensorSlot::BlockDSwiGLU);
            if (is_stack_grad) {
                // Allocate proper stack storage and copy data
                mRunState.temp_acquire(*base_grad);
                mTemps.push_back(*base_grad);
                CUDA_CHECK(cudaMemcpyAsync(base_grad->Data, d_out.Data, d_out.bytes(),
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));
                mTensorMap[ref.name] = view_tensor(*base_grad, ref.shape);
                trace_residual_l8(mTensorMap[ref.name], "stack_copy");
                trace_mlp_down(mTensorMap[ref.name], "stack_copy");
                return;
            }
            // Fall back to aliasing if the base grad has no storage yet (non-stack temps).
            base_grad->Data = d_out.Data;
            mTensorMap[ref.name] = view_tensor(*base_grad, ref.shape);
            trace_residual_l8(mTensorMap[ref.name], "alias");
            trace_mlp_down(mTensorMap[ref.name], "alias");
            return;
        }
        // Default: just expose d_out as-is.
        mTensorMap[ref.name] = d_out;
        trace_residual_l8(mTensorMap[ref.name], "passthrough");
        trace_mlp_down(mTensorMap[ref.name], "passthrough");
    };

    assign_output(op.outputs[0]);
    if (op.outputs.size() > 1) {
        assign_output(op.outputs[1]);
    }
}

void CompiledExecutor::dispatch_matmul_backward(const CompiledOp& op, const modules::BackwardHook* hook) {
    // inputs: d_out, A, B (weight)
    // outputs: dA, dB
    const std::string& weight_name = (op.inputs.size() > 2) ? op.inputs[2].name : "";
    const bool is_lm_head = (weight_name == "lm_head" || weight_name == "lm_head_weight");
    const bool skip_lm_head = is_lm_head && mOptions.LMHeadChunks > 1;

    EMMTranspose mode = op.attrs.transpose;
    const int layer_idx = op.attrs.layer_idx;
    const bool allow_quant = op.attrs.allow_quant;

    // Check if weight gradient should be skipped BEFORE allocating (frozen weights in LoRA mode)
    bool skip_weight_grad = true;
    const std::string& dB_name = op.outputs.size() > 1 ? op.outputs[1].name : "";
    if (!dB_name.empty()) {
        std::string weight_name;
        if (auto base = base_param_from_grad(dB_name)) {
            weight_name = *base;
        } else {
            weight_name = dB_name;
            if (weight_name.rfind("d_", 0) == 0) {
                weight_name = weight_name.substr(2);
            }
        }
        bool accum = false;
        Tensor* grad = mGrads.get_param_grad(weight_name, accum);
        skip_weight_grad = (grad == nullptr || !grad->Data);
    }

    if (skip_lm_head) {
        if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
            (void)ensure_output_tensor(op.outputs[0]);
        }
        if (!skip_weight_grad && op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
            (void)ensure_output_tensor(op.outputs[1]);
        }
        return;
    }

    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& a = resolve_tensor(op.inputs[1]);
    Tensor& b = resolve_tensor(op.inputs[2]);

    const bool trace_matmul_a = (std::getenv("SUROGATE_TRACE_MATMUL_A") != nullptr);
    const bool assert_recompute_a = (std::getenv("SUROGATE_ASSERT_RECOMPUTE_A") != nullptr);
    const bool is_qkv_op = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::QKV);
    const bool is_mlp_up_op = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::MLPUp);
    const bool is_mlp_down_op = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::MLPDown);
    const bool is_attn_out_op = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::AttnOut);
    static int matmul_a_trace = 0;
    if (trace_matmul_a && matmul_a_trace < 12 && (is_qkv_op || is_mlp_up_op)) {
        const int top_layer = static_cast<int>(mConfig.NumLayers) - 1;
        if (layer_idx == 0 || layer_idx == 26 || layer_idx == top_layer) {
            matmul_a_trace++;
            const long sample_token = 3;
            std::vector<float> a_vals(4);
            const bool a_ok = copy_tensor_token_sample_as_f32(a, sample_token, a_vals.size(), a_vals);
            fprintf(stderr,
                    "[MATMUL_BWD_A] micro_step=%d layer=%d op=%s kind=%s input=%s ptr=%p ok=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                    mMicroStep, layer_idx, op.op_id.c_str(),
                    is_qkv_op ? "qkv" : "mlp_up",
                    op.inputs[1].name.c_str(), a.Data, a_ok ? 1 : 0,
                    a_vals[0], a_vals[1], a_vals[2], a_vals[3]);
            log_tensor_stats_ex(is_qkv_op ? "MATMUL_BWD_A_QKV" : "MATMUL_BWD_A_MLP_UP",
                                layer_idx, op.inputs[1].name, a, 4096, true);
        }
    }

    // Targeted matmul backward trace for layers 8/9 (attn out + mlp down paths).
    static int matmul_l8_in_trace = 0;
    if ((layer_idx == 8 || layer_idx == 9) && (is_attn_out_op || is_mlp_down_op) && matmul_l8_in_trace < 12) {
        const char* kind = is_attn_out_op ? "attn_out" : (is_mlp_down_op ? "mlp_down" : "other");
        fprintf(stderr,
                "[MATMUL_BWD_L8_IN] layer=%d kind=%s d_out=%s A=%s W=%s\n",
                layer_idx,
                kind,
                op.inputs[0].name.c_str(),
                op.inputs[1].name.c_str(),
                op.inputs[2].name.c_str());
        log_tensor_mag_unbounded("MATMUL_BWD_L8_DOUT", layer_idx, op.inputs[0].name, d_out, 4096);
        log_tensor_mag_unbounded("MATMUL_BWD_L8_A", layer_idx, op.inputs[1].name, a, 4096);
        log_tensor_mag_unbounded("MATMUL_BWD_L8_W", layer_idx, op.inputs[2].name, b, 4096);
        matmul_l8_in_trace++;
    }

    if (assert_recompute_a && (is_qkv_op || is_mlp_up_op) &&
        layer_idx >= 0 && layer_idx < static_cast<int>(mRecomputeSamples.size())) {
        const auto& sample = mRecomputeSamples[static_cast<std::size_t>(layer_idx)];
        const bool want_ln1 = is_qkv_op;
        const bool valid = want_ln1 ? sample.ln1_valid : sample.ln2_valid;
        if (valid && sample.micro_step == mMicroStep) {
            const long sample_token = 3;
            std::vector<float> a_vals;
            if (copy_tensor_token_sample_as_f32(a, sample_token, 4, a_vals)) {
                const std::array<float, 4>& ref = want_ln1 ? sample.ln1 : sample.ln2;
                float max_diff = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    const float diff = std::fabs(a_vals[static_cast<std::size_t>(i)] - ref[static_cast<std::size_t>(i)]);
                    if (diff > max_diff) max_diff = diff;
                }
                if (max_diff > 1e-2f) {
                    fprintf(stderr,
                            "[RECOMPUTE_A_MISMATCH] micro_step=%d layer=%d kind=%s input=%s max_diff=%.6f "
                            "a=%.6f,%.6f,%.6f,%.6f ref=%.6f,%.6f,%.6f,%.6f\n",
                            mMicroStep, layer_idx, want_ln1 ? "qkv" : "mlp_up",
                            op.inputs[1].name.c_str(), max_diff,
                            a_vals[0], a_vals[1], a_vals[2], a_vals[3],
                            ref[0], ref[1], ref[2], ref[3]);
                    throw std::runtime_error("Recompute A mismatch: matmul input does not match recomputed LN output");
                }
            }
        }
    }

    // Targeted QKV matmul backward trace for layers 8/9.
    static int qkv_l8_trace = 0;
    if (is_qkv_op && (layer_idx == 8 || layer_idx == 9) && qkv_l8_trace < 12) {
        fprintf(stderr,
                "[MATMUL_BWD_QKV_L8] layer=%d d_out=%s A=%s W=%s\n",
                layer_idx,
                op.inputs[0].name.c_str(),
                op.inputs[1].name.c_str(),
                op.inputs[2].name.c_str());
        log_tensor_mag_unbounded("MATMUL_BWD_QKV_L8_DOUT", layer_idx, op.inputs[0].name, d_out, 4096);
        log_tensor_mag_unbounded("MATMUL_BWD_QKV_L8_A", layer_idx, op.inputs[1].name, a, 4096);
        log_tensor_mag_unbounded("MATMUL_BWD_QKV_L8_W", layer_idx, op.inputs[2].name, b, 4096);
        const std::size_t dout_total = static_cast<std::size_t>(d_out.nelem());
        if (dout_total > 4096) {
            log_tensor_sample_stats("MATMUL_BWD_QKV_L8_DOUT_MID", d_out, dout_total / 2, 4096);
        }
        qkv_l8_trace++;
    }

    static int matmul_ln2_trace = 0;
    const bool outputs_ln2 = (!op.outputs.empty() &&
                              !op.outputs[0].name.empty() &&
                              strip_ssa_suffix(op.outputs[0].name) == "d_blocks[0].ln2");
    if (outputs_ln2 && matmul_ln2_trace < 8) {
        fprintf(stderr,
                "[MATMUL_BWD_LN2] id=%s in=%s out=%s weight=%s\n",
                op.op_id.c_str(),
                op.inputs[0].name.c_str(),
                op.outputs[0].name.c_str(),
                op.inputs[2].name.c_str());
        log_tensor_stats_ex("MATMUL_BWD_LN2_DOUT", layer_idx, op.inputs[0].name, d_out, 4096, true);
        log_tensor_stats_ex("MATMUL_BWD_LN2_A", layer_idx, op.inputs[1].name, a, 4096, true);
        log_tensor_stats_ex("MATMUL_BWD_LN2_W", layer_idx, op.inputs[2].name, b, 4096, true);
        matmul_ln2_trace++;
    }

    // DEBUG: Print matmul backward input/output for layer 26 QKV backward
    static int matmul_print_count = 0;
    // Trace Layer 25 and 26 QKV backward for explosion debugging
    static int qkv_25_trace = 0;
    if ((layer_idx == 25 || layer_idx == 26) && is_qkv_op && qkv_25_trace < 20) {
        qkv_25_trace++;
        cudaStreamSynchronize(mRunState.MainStream);
        const int N = static_cast<int>(std::min(static_cast<long>(d_out.nelem()), 10000L));
        std::vector<float> dout_all(N);
        cudaMemcpy(dout_all.data(), d_out.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
        float dout_sum_sq = 0.0f, dout_max = 0.0f;
        for (int i = 0; i < N; ++i) {
            dout_sum_sq += dout_all[i] * dout_all[i];
            if (std::fabs(dout_all[i]) > dout_max) dout_max = std::fabs(dout_all[i]);
        }
        fprintf(stderr, "[MATMUL_BWD_QKV] Layer %d: d_out name='%s' slot=%d ptr=%p L2=%.6f max=%.6f vals=%.6f,%.6f,%.6f,%.6f\n",
                layer_idx, op.inputs[0].name.c_str(), static_cast<int>(op.inputs[0].slot), d_out.Data,
                std::sqrt(dout_sum_sq), dout_max, dout_all[0], dout_all[1], dout_all[2], dout_all[3]);
    }
    if ((layer_idx == 26 && is_qkv_op) && matmul_print_count < 3) {
        matmul_print_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        // Print the input tensor ref to see where d_out is coming from
        fprintf(stderr, "[MATMUL_BWD] Layer %d QKV: d_out name='%s' slot=%d ptr=%p\n",
                layer_idx, op.inputs[0].name.c_str(), static_cast<int>(op.inputs[0].slot), d_out.Data);
        // Compute L2 norm of dout to see the full tensor magnitude
        const int N = static_cast<int>(std::min(static_cast<long>(d_out.nelem()), 10000L));
        std::vector<float> dout_all(N);
        cudaMemcpy(dout_all.data(), d_out.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
        float dout_sum_sq = 0.0f;
        float dout_max = 0.0f;
        int dout_nonzero = 0;
        for (int i = 0; i < N; ++i) {
            dout_sum_sq += dout_all[i] * dout_all[i];
            if (std::fabs(dout_all[i]) > dout_max) dout_max = std::fabs(dout_all[i]);
            if (std::fabs(dout_all[i]) > 1e-10f) dout_nonzero++;
        }
        float dout_norm = std::sqrt(dout_sum_sq);
        fprintf(stderr, "[MATMUL_BWD] Layer %d QKV: dout L2 norm=%.9f, max=%.9f, nonzero=%d/%d\n",
                layer_idx, dout_norm, dout_max, dout_nonzero, N);
        // Print some middle values to see if pattern is different
        int mid = N / 2;
        fprintf(stderr, "[MATMUL_BWD] Layer %d QKV: dout[%d..%d]=%.9f,%.9f,%.9f,%.9f\n",
                layer_idx, mid, mid+3, dout_all[mid], dout_all[mid+1], dout_all[mid+2], dout_all[mid+3]);
    }

    // Now allocate output tensors - skip dB if weights are frozen
    Tensor* dA_ptr = nullptr;
    Tensor* dB_ptr = nullptr;

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        dA_ptr = &ensure_output_tensor(op.outputs[0]);
    }
    if (!skip_weight_grad && op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        dB_ptr = &ensure_output_tensor(op.outputs[1]);
    }

    // FIX: Zero dA buffer before matmul to ensure consistent results regardless of initial values
    // This is needed because the buffer may contain stale gradients from layer 27 in no-recompute mode.
    // Even though matmul uses accumulate=false (beta=0), we explicitly zero to ensure determinism.
    if (dA_ptr && dA_ptr->Data) {
        fill_zero(*dA_ptr, mRunState.MainStream);
    }

    if (!dA_ptr && !dB_ptr) {
        return;
    }

    bool do_accumulate = mAccumulateTensors.count(dB_name) > 0;
    if (!do_accumulate && !dB_name.empty()) {
        if (auto base = base_param_from_grad(dB_name)) {
            do_accumulate = mAccumulateTensors.count("d_" + *base) > 0;
        }
    }

    bool used_recipe = false;
    bool used_fp8 = false;
    bool has_dout_quant = false;

    const bool disable_qkv_recipe_bwd =
        is_qkv_op && skip_weight_grad && (mConfig.NumExperts > 0);
    if (mRecipe && mode == EMMTranspose::NT && a.Sizes[0] == mB * mT && allow_quant && !disable_qkv_recipe_bwd) {
        Tensor dA_tmp{};
        Tensor dB_tmp{};
        Tensor* dA_use = dA_ptr;
        Tensor* dB_use = dB_ptr;

        if (!dA_use) {
            dA_tmp = mRunState.temp_alloc(a.DType, {a.Sizes[0], a.Sizes[1]});
            mTemps.push_back(dA_tmp);
            dA_use = &dA_tmp;
        }
        if (!dB_use) {
            dB_tmp = mRunState.temp_alloc(b.DType, {b.Sizes[0], b.Sizes[1]});
            mTemps.push_back(dB_tmp);
            dB_use = &dB_tmp;
        }

        modules::MatmulContext ctx;
        ctx.dinp = dA_use;
        ctx.dweight = dB_use;
        ctx.dout = &d_out;
        ctx.inp = &a;
        ctx.weight = &b;
        ctx.B = static_cast<int>(mB);
        ctx.T = static_cast<int>(mT);
        ctx.C_in = static_cast<int>(a.Sizes[1]);
        ctx.C_out = static_cast<int>(b.Sizes[0]);
        ctx.run_state = &mRunState;
        ctx.stream = mRunState.MainStream;
        ctx.layer_idx = layer_idx;
        ctx.op = op.attrs.matmul_op.value_or(modules::MatmulOp::LMHead);
        ctx.accumulate = do_accumulate;
        ctx.skip_weight_grad = skip_weight_grad || !dB_ptr;
        ctx.allow_fp8 = allow_quant && mRecipe->uses_fp8_hybrid_backward();
        ctx.allow_fp4 = allow_quant && mRecipe->uses_fp4_forward();

        if (ctx.allow_fp8 && op.attrs.matmul_op.has_value()) {
            ctx.dout_quant = fp8_grad_buffer(mRunState, *op.attrs.matmul_op);
            if (!ctx.dout_quant || !ctx.dout_quant->Data) {
                ctx.allow_fp8 = false;
            }
        }
        used_fp8 = ctx.allow_fp8;
        has_dout_quant = (ctx.dout_quant && ctx.dout_quant->Data);

        mRecipe->backward_matmul(ctx);
        used_recipe = true;
    }

    if (!used_recipe) {
        Tensor d_out_mat = d_out;
        Tensor a_mat = a;
        if (disable_qkv_recipe_bwd && is_qkv_op) {
            if (d_out_mat.Rank > 2 && d_out_mat.Sizes[0] == mB && d_out_mat.Sizes[1] == mT) {
                d_out_mat = view_tensor(d_out_mat, {mB * mT, d_out_mat.Sizes[d_out_mat.Rank - 1]});
            }
            if (a_mat.Rank > 2 && a_mat.Sizes[0] == mB && a_mat.Sizes[1] == mT) {
                a_mat = view_tensor(a_mat, {mB * mT, a_mat.Sizes[a_mat.Rank - 1]});
            }
        }

        // Fallback: explicit matmuls for dA and dB
        EMMTranspose mode_dA = EMMTranspose::NN;
        EMMTranspose mode_dB = EMMTranspose::NN;
        switch (mode) {
            case EMMTranspose::NN:
                mode_dA = EMMTranspose::NT;
                mode_dB = EMMTranspose::TN;
                break;
            case EMMTranspose::NT:
                mode_dA = EMMTranspose::NN;
                mode_dB = EMMTranspose::TN;
                break;
            case EMMTranspose::TN:
                mode_dA = EMMTranspose::NT;
                mode_dB = EMMTranspose::NN;
                break;
            case EMMTranspose::TT:
                mode_dA = EMMTranspose::TT;
                mode_dB = EMMTranspose::TT;
                break;
        }

        if (dA_ptr) {
            int M = 0, N = 0, K = 0;
            matmul_dims(d_out_mat, b, mode_dA, M, N, K);
            EMMTranspose mode_col = swap_transpose(mode_dA);
            matmul(*dA_ptr, b, d_out_mat, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   N, M, K, mode_col, false, mRunState.MainStream);
        }
        if (dB_ptr && !skip_weight_grad) {
            const Tensor* lhs = nullptr;
            const Tensor* rhs = nullptr;
            EMMTranspose mode_rm = EMMTranspose::NN;
            switch (mode) {
                case EMMTranspose::NN:
                    // dB = A^T * d_out
                    lhs = &a_mat;
                    rhs = &d_out_mat;
                    mode_rm = EMMTranspose::TN;
                    break;
                case EMMTranspose::NT:
                    // dB = d_out^T * A
                    lhs = &d_out_mat;
                    rhs = &a_mat;
                    mode_rm = EMMTranspose::TN;
                    break;
                case EMMTranspose::TN:
                    // dB = A * d_out
                    lhs = &a_mat;
                    rhs = &d_out_mat;
                    mode_rm = EMMTranspose::NN;
                    break;
                case EMMTranspose::TT:
                    // dB = d_out^T * A^T
                    lhs = &d_out_mat;
                    rhs = &a_mat;
                    mode_rm = EMMTranspose::TT;
                    break;
            }

            int M = 0, N = 0, K = 0;
            matmul_dims(*lhs, *rhs, mode_rm, M, N, K);
            EMMTranspose mode_col = swap_transpose(mode_rm);
            matmul(*dB_ptr, *rhs, *lhs, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   N, M, K, mode_col, do_accumulate, mRunState.MainStream);
        }
    }

    // One-time NaN watchdog for matmul backward outputs.
    static bool matmul_qkv_nan_logged = false;
    static int matmul_attn_nan_logged = 0;
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
    if (is_qkv_op && dA_ptr && !matmul_qkv_nan_logged && tensor_sample_has_nan_or_inf(*dA_ptr, 3)) {
        auto shape_str = [](const Tensor& t) {
            std::string s = "[";
            for (int i = 0; i < t.Rank; ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(t.Sizes[i]);
            }
            s += "]";
            return s;
        };
        Tensor d_out_scan = d_out;
        if (d_out_scan.Rank > 2 && d_out_scan.Sizes[0] == mB && d_out_scan.Sizes[1] == mT) {
            d_out_scan = view_tensor(d_out_scan, {mB * mT, d_out_scan.Sizes[d_out_scan.Rank - 1]});
        }
        float scan_min = 0.0f;
        float scan_max = 0.0f;
        const bool scan_nan = tensor_row_has_nan_or_inf(d_out_scan, 3, &scan_min, &scan_max);
        fprintf(stderr,
                "[MATMUL_BWD_QKV_NAN] op=%s layer=%d dA=%s d_out=%s weight=%s allow_quant=%d used_recipe=%d used_fp8=%d has_dout_quant=%d dtype=%s ptr=%p d_out_shape=%s d_out_scan_shape=%s a_shape=%s b_shape=%s d_out_row_nan=%d row_min=%.6f row_max=%.6f\n",
                op.op_id.c_str(),
                layer_idx,
                op.outputs[0].name.c_str(),
                op.inputs[0].name.c_str(),
                op.inputs[2].name.c_str(),
                allow_quant ? 1 : 0,
                used_recipe ? 1 : 0,
                used_fp8 ? 1 : 0,
                has_dout_quant ? 1 : 0,
                dtype_to_str(b.DType),
                b.Data,
                shape_str(d_out).c_str(),
                shape_str(d_out_scan).c_str(),
                shape_str(a).c_str(),
                shape_str(b).c_str(),
                scan_nan ? 1 : 0,
                scan_min,
                scan_max);
        dump_sample("MATMUL_BWD_QKV_NAN_DOUT", d_out, op.inputs[0].name);
        dump_sample("MATMUL_BWD_QKV_NAN_DA", *dA_ptr, op.outputs[0].name);
        dump_sample("MATMUL_BWD_QKV_NAN_W", b, op.inputs[2].name);
        matmul_qkv_nan_logged = true;
    }
    if (is_attn_out_op && dA_ptr && matmul_attn_nan_logged < 4) {
        Tensor d_out_scan = d_out;
        if (d_out_scan.Rank > 2 && d_out_scan.Sizes[0] == mB && d_out_scan.Sizes[1] == mT) {
            d_out_scan = view_tensor(d_out_scan, {mB * mT, d_out_scan.Sizes[d_out_scan.Rank - 1]});
        }
        Tensor dA_scan = *dA_ptr;
        if (dA_scan.Rank > 2 && dA_scan.Sizes[0] == mB && dA_scan.Sizes[1] == mT) {
            dA_scan = view_tensor(dA_scan, {mB * mT, dA_scan.Sizes[dA_scan.Rank - 1]});
        }
        long d_out_row = -1;
        long dA_row = -1;
        float d_out_min = 0.0f;
        float d_out_max = 0.0f;
        float dA_min = 0.0f;
        float dA_max = 0.0f;
        const bool d_out_nan = find_first_nan_row(d_out_scan, &d_out_row, &d_out_min, &d_out_max);
        const bool dA_nan = find_first_nan_row(dA_scan, &dA_row, &dA_min, &dA_max);
        if (d_out_nan || dA_nan) {
            fprintf(stderr,
                    "[MATMUL_BWD_ATTN_NAN] op=%s layer=%d d_out=%s dA=%s d_out_nan=%d dA_nan=%d\n",
                    op.op_id.c_str(),
                    layer_idx,
                    op.inputs[0].name.c_str(),
                    op.outputs[0].name.c_str(),
                    d_out_nan ? 1 : 0,
                    dA_nan ? 1 : 0);
            if (d_out_nan) {
                const long b = (d_out_scan.Rank >= 2 && d_out_scan.Sizes[0] == mB * mT)
                    ? (d_out_row / static_cast<long>(mT)) : -1;
                const long t = (d_out_scan.Rank >= 2 && d_out_scan.Sizes[0] == mB * mT)
                    ? (d_out_row % static_cast<long>(mT)) : -1;
                fprintf(stderr,
                        "[MATMUL_BWD_ATTN_NAN_DOUT_ROW] row=%ld b=%ld t=%ld min=%.6f max=%.6f\n",
                        d_out_row, b, t, d_out_min, d_out_max);
            }
            if (dA_nan) {
                const long b = (dA_scan.Rank >= 2 && dA_scan.Sizes[0] == mB * mT)
                    ? (dA_row / static_cast<long>(mT)) : -1;
                const long t = (dA_scan.Rank >= 2 && dA_scan.Sizes[0] == mB * mT)
                    ? (dA_row % static_cast<long>(mT)) : -1;
                fprintf(stderr,
                        "[MATMUL_BWD_ATTN_NAN_DA_ROW] row=%ld b=%ld t=%ld min=%.6f max=%.6f\n",
                        dA_row, b, t, dA_min, dA_max);
            }
            dump_sample("MATMUL_BWD_ATTN_NAN_DOUT", d_out, op.inputs[0].name);
            dump_sample("MATMUL_BWD_ATTN_NAN_DA", *dA_ptr, op.outputs[0].name);
            dump_sample("MATMUL_BWD_ATTN_NAN_W", b, op.inputs[2].name);
            matmul_attn_nan_logged++;
        }
    }

    static int qkv_l8_out_trace = 0;
    if (is_qkv_op && (layer_idx == 8 || layer_idx == 9) && dA_ptr && qkv_l8_out_trace < 12) {
        log_tensor_mag_unbounded("MATMUL_BWD_QKV_L8_DA", layer_idx,
                                 op.outputs.empty() ? "<none>" : op.outputs[0].name, *dA_ptr, 4096);
        if (dB_ptr) {
            log_tensor_mag_unbounded("MATMUL_BWD_QKV_L8_DB", layer_idx,
                                     op.outputs.size() > 1 ? op.outputs[1].name : "<none>", *dB_ptr, 4096);
        }
        const std::size_t da_total = static_cast<std::size_t>(dA_ptr->nelem());
        if (da_total > 4096) {
            log_tensor_sample_stats("MATMUL_BWD_QKV_L8_DA_MID", *dA_ptr, da_total / 2, 4096);
        }
        qkv_l8_out_trace++;
    }

    // Record qkv dA pointer for LN1 wiring verification.
    if (is_qkv_op && dA_ptr && layer_idx >= 0) {
        if (g_qkv_dA_ptr_by_layer.empty() && mConfig.NumLayers > 0) {
            g_qkv_dA_ptr_by_layer.assign(static_cast<std::size_t>(mConfig.NumLayers), nullptr);
            g_qkv_dA_micro_by_layer.assign(static_cast<std::size_t>(mConfig.NumLayers), -1);
        }
        if (layer_idx < static_cast<int>(g_qkv_dA_ptr_by_layer.size())) {
            g_qkv_dA_ptr_by_layer[static_cast<std::size_t>(layer_idx)] =
                reinterpret_cast<std::byte*>(dA_ptr->Data);
            g_qkv_dA_micro_by_layer[static_cast<std::size_t>(layer_idx)] = mMicroStep;
        }
    }

    // Targeted matmul backward output trace for layers 8/9.
    static int matmul_l8_out_trace = 0;
    if ((layer_idx == 8 || layer_idx == 9) && (is_attn_out_op || is_mlp_down_op) &&
        dA_ptr && matmul_l8_out_trace < 12) {
        const char* kind = is_attn_out_op ? "attn_out" : (is_mlp_down_op ? "mlp_down" : "other");
        fprintf(stderr,
                "[MATMUL_BWD_L8_OUT] layer=%d kind=%s dA=%s dB=%s\n",
                layer_idx,
                kind,
                op.outputs.empty() ? "<none>" : op.outputs[0].name.c_str(),
                (op.outputs.size() > 1) ? op.outputs[1].name.c_str() : "<none>");
        log_tensor_mag_unbounded("MATMUL_BWD_L8_DA", layer_idx,
                                 op.outputs.empty() ? "<none>" : op.outputs[0].name, *dA_ptr, 4096);
        if (dB_ptr) {
            log_tensor_mag_unbounded("MATMUL_BWD_L8_DB", layer_idx,
                                     op.outputs.size() > 1 ? op.outputs[1].name : "<none>", *dB_ptr, 4096);
        }
        matmul_l8_out_trace++;
    }

    // Hook invocation for LoRA backward
    // Skip dense MLP hooks for MoE models - MoE has different backward path (grouped GEMM)
    const bool is_moe = mConfig.NumExperts > 0;
    const bool is_mlp_hook = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::MLPUp ||
         *op.attrs.matmul_op == modules::MatmulOp::MLPDown);
    if (hook && *hook && op.attrs.backward_hook_point.has_value() && !(is_moe && is_mlp_hook)) {
        // Temporarily map grads to current backward tensors for LoRA hooks, then restore.
        struct GradPtrs {
            std::byte* d_swiglu{nullptr};
            std::byte* d_ln2{nullptr};
            std::byte* d_att{nullptr};
            std::byte* d_ln1{nullptr};
            std::byte* d_res_ffn{nullptr};
            std::byte* d_mlp_up{nullptr};
            std::byte* d_res_att{nullptr};
            std::byte* d_qkv{nullptr};
        } prev{};

        if (op.attrs.matmul_op.has_value() && layer_idx >= 0) {
            auto& grads = mRunState.simplified_grads(layer_idx);
            prev.d_swiglu = reinterpret_cast<std::byte*>(grads.d_swiglu.Data);
            prev.d_ln2 = reinterpret_cast<std::byte*>(grads.d_ln2.Data);
            prev.d_att = reinterpret_cast<std::byte*>(grads.d_att.Data);
            prev.d_ln1 = reinterpret_cast<std::byte*>(grads.d_ln1.Data);
            prev.d_res_ffn = reinterpret_cast<std::byte*>(grads.d_res_ffn.Data);
            prev.d_mlp_up = reinterpret_cast<std::byte*>(grads.d_mlp_up.Data);
            prev.d_res_att = reinterpret_cast<std::byte*>(grads.d_res_att.Data);
            prev.d_qkv = reinterpret_cast<std::byte*>(grads.d_qkv.Data);

            if (dA_ptr) {
                switch (*op.attrs.matmul_op) {
                    case modules::MatmulOp::MLPDown:
                        grads.d_swiglu.Data = dA_ptr->Data;
                        break;
                    case modules::MatmulOp::MLPUp:
                        grads.d_ln2.Data = dA_ptr->Data;
                        break;
                    case modules::MatmulOp::AttnOut:
                        grads.d_att.Data = dA_ptr->Data;
                        break;
                    case modules::MatmulOp::QKV:
                        grads.d_ln1.Data = dA_ptr->Data;
                        break;
                    default:
                        break;
                }
            }

            switch (*op.attrs.matmul_op) {
                case modules::MatmulOp::MLPDown:
                    grads.d_res_ffn.Data = d_out.Data;
                    break;
                case modules::MatmulOp::MLPUp:
                    grads.d_mlp_up.Data = d_out.Data;
                    break;
                case modules::MatmulOp::AttnOut:
                    grads.d_res_att.Data = d_out.Data;
                    break;
                case modules::MatmulOp::QKV:
                    grads.d_qkv.Data = d_out.Data;
                    break;
                default:
                    break;
            }
        }

        // Ensure activations needed by LoRA hooks are available.
        if (layer_idx >= 0 && op.attrs.matmul_op.has_value()) {
            auto& acts = mRunState.simplified_acts(layer_idx);
            if (*op.attrs.matmul_op == modules::MatmulOp::MLPDown) {
                // LoRA backward hook needs acts.swiglu (forward activation).
                // With recompute enabled, swiglu may have been stack-allocated and freed.
                if (!acts.swiglu.Data && acts.mlp_up.Data) {
                    mRunState.temp_acquire(acts.swiglu);
                    const int Bv = static_cast<int>(mB);
                    const int Tv = static_cast<int>(mT);
                    const int D = static_cast<int>(mConfig.IntermediateSize);
                    swiglu_forward(acts.swiglu, acts.mlp_up, nullptr, Bv, Tv, D, mRunState.MainStream);
                }
            }
        }
        (*hook)(layer_idx, do_accumulate, mRunState.MainStream, *op.attrs.backward_hook_point, mHookContext);

        if (op.attrs.matmul_op.has_value() && layer_idx >= 0) {
            auto& grads = mRunState.simplified_grads(layer_idx);
            grads.d_swiglu.Data = prev.d_swiglu;
            grads.d_ln2.Data = prev.d_ln2;
            grads.d_att.Data = prev.d_att;
            grads.d_ln1.Data = prev.d_ln1;
            grads.d_res_ffn.Data = prev.d_res_ffn;
            grads.d_mlp_up.Data = prev.d_mlp_up;
            grads.d_res_att.Data = prev.d_res_att;
            grads.d_qkv.Data = prev.d_qkv;
        }
    }
}

void CompiledExecutor::dispatch_bias_add_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    // d_input = d_out (pass through)
    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        mTensorMap[op.outputs[0].name] = d_out;
    }

    // d_bias = sum(d_out, axis=[0,1]) for [B,T,C] or axis=0 for [N,C]
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        int Bv = 1, Tv = 1, OC = 1;
        if (d_out.Rank == 2) {
            Bv = static_cast<int>(d_out.Sizes[0]);
            Tv = 1;
            OC = static_cast<int>(d_out.Sizes[1]);
        } else {
            Bv = static_cast<int>(d_out.Sizes[0]);
            Tv = static_cast<int>(d_out.Sizes[1]);
            OC = static_cast<int>(d_out.Sizes[2]);
        }

        Tensor& d_bias = ensure_output_tensor(op.outputs[1]);
        bool accumulate = mAccumulateTensors.count(op.outputs[1].name) > 0;
        if (!accumulate && !op.outputs[1].name.empty()) {
            if (auto base = base_param_from_grad(op.outputs[1].name)) {
                accumulate = mAccumulateTensors.count("d_" + *base) > 0;
            }
        }

        // Allocate scratch buffer for bias reduction
        const int scratch_bytes = get_bias_backward_scratch_size(d_out.DType, OC, mRunState.DeviceProp);
        Tensor scratch = mRunState.temp_alloc(ETensorDType::FP32, {static_cast<long>(scratch_bytes / sizeof(float))});
        mTemps.push_back(scratch);

        if (accumulate) {
            // Accumulate into existing gradient: compute to tmp, then add
            Tensor tmp = mRunState.temp_alloc(d_out.DType, {static_cast<long>(OC)});
            mTemps.push_back(tmp);
            backward_bias(tmp, d_out, nullptr, nullptr, scratch, Bv, Tv, OC, mRunState.DeviceProp, mRunState.MainStream);
            vector_add_sr(d_bias, d_bias, tmp, 1.0f, static_cast<long>(d_bias.nelem()), 0, mRunState.MainStream);
        } else {
            backward_bias(d_bias, d_out, nullptr, nullptr, scratch, Bv, Tv, OC, mRunState.DeviceProp, mRunState.MainStream);
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

void CompiledExecutor::dispatch_matmul_swiglu_backward(const CompiledOp& op, const modules::BackwardHook* hook) {
    // Combined backward for matmul + swiglu (fused op in forward)
    // inputs: d_swiglu_out, ln2 (matmul input), mlp_up_weight, mlp_up (pre-swiglu)
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);
    Tensor mlp_up = resolve_tensor(op.inputs[3]);

    const int layer_idx = op.attrs.layer_idx;

    // Recompute mlp_up if the saved tensor was stack-allocated and freed
    bool recomputed_mlp_up = false;
    if (!mlp_up.Data || (mRunState.Stack.owns(mlp_up.Data) && !mRunState.Stack.is_live(mlp_up.Data))) {
        int M = 0, N = 0, K = 0;
        Tensor inp_flat = (inp.Rank == 2) ? inp : view_tensor(inp, {mB * mT, inp.Sizes[inp.Rank - 1]});
        matmul_dims(inp_flat, weight, op.attrs.transpose, M, N, K);
        const long D2 = N;
        Tensor mlp_up_flat = mRunState.temp_alloc(inp.DType, {mB * mT, D2});
        mTemps.push_back(mlp_up_flat);

        EMMTranspose mode_col = swap_transpose(op.attrs.transpose);
        matmul(mlp_up_flat, weight, inp_flat, std::nullopt, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, false, mRunState.MainStream);

        mlp_up = view_tensor(mlp_up_flat, {mB, mT, D2});
        if (layer_idx >= 0) {
            auto& acts = mRunState.simplified_acts(layer_idx);
            acts.mlp_up.Data = mlp_up.Data;
        }
        recomputed_mlp_up = true;
    }

    // First: swiglu backward
    Tensor* d_mlp_up_ptr = nullptr;
    if (layer_idx >= 0) {
        auto& grads = mRunState.simplified_grads(layer_idx);
        d_mlp_up_ptr = &grads.d_mlp_up;
        if (!d_mlp_up_ptr->Data) {
            mRunState.temp_acquire(*d_mlp_up_ptr);
            mTemps.push_back(*d_mlp_up_ptr);
        }
    }
    Tensor d_mlp_up = d_mlp_up_ptr ? *d_mlp_up_ptr
                                   : mRunState.temp_alloc(mlp_up.DType, {mlp_up.Sizes[0], mlp_up.Sizes[1], mlp_up.Sizes[2]});
    if (!d_mlp_up_ptr) {
        mTemps.push_back(d_mlp_up);
    }

    // For FP8 hybrid backward, record abs_max of d_mlp_up for subsequent quantization
    float* abs_max_ptr = mRunState.has_fp8_hybrid_backward()
        ? mRunState.simplified_quant_grads().d_mlp_up.abs_max()
        : nullptr;

    const long D = d_out.Sizes[2];
    swiglu_backward(d_mlp_up, d_out, mlp_up, abs_max_ptr,
                    static_cast<int>(d_out.Sizes[0]),
                    static_cast<int>(d_out.Sizes[1]),
                    static_cast<int>(D), mRunState.MainStream);

    // Then: matmul backward
    Tensor d_mlp_up_flat = view_tensor(d_mlp_up, {mB * mT, 2 * D});

    Tensor* d_inp_ptr = nullptr;
    Tensor* d_weight_ptr = nullptr;

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        d_inp_ptr = &ensure_output_tensor(op.outputs[0]);
    }
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        d_weight_ptr = &ensure_output_tensor(op.outputs[1]);
    }

    bool do_accumulate = false;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        do_accumulate = (mAccumulateTensors.count(op.outputs[1].name) > 0);
        if (!do_accumulate) {
            if (auto base = base_param_from_grad(op.outputs[1].name)) {
                do_accumulate = mAccumulateTensors.count("d_" + *base) > 0;
            }
        }
    }

    if (d_inp_ptr) {
        EMMTranspose mode_dA = EMMTranspose::NN;
        switch (op.attrs.transpose) {
            case EMMTranspose::NN:
                mode_dA = EMMTranspose::NT;
                break;
            case EMMTranspose::NT:
                mode_dA = EMMTranspose::NN;
                break;
            case EMMTranspose::TN:
                mode_dA = EMMTranspose::NT;
                break;
            case EMMTranspose::TT:
                mode_dA = EMMTranspose::TT;
                break;
        }

        int M = 0, N = 0, K = 0;
        matmul_dims(d_mlp_up_flat, weight, mode_dA, M, N, K);
        EMMTranspose mode_col = swap_transpose(mode_dA);
        matmul(*d_inp_ptr, weight, d_mlp_up_flat, std::nullopt, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, false, mRunState.MainStream);
    }
    if (d_weight_ptr) {
        Tensor inp_flat = (inp.Rank == 2) ? inp : view_tensor(inp, {mB * mT, inp.Sizes[inp.Rank - 1]});

        const Tensor* lhs = nullptr;
        const Tensor* rhs = nullptr;
        EMMTranspose mode_rm = EMMTranspose::NN;
        switch (op.attrs.transpose) {
            case EMMTranspose::NN:
                lhs = &inp_flat;
                rhs = &d_mlp_up_flat;
                mode_rm = EMMTranspose::TN;
                break;
            case EMMTranspose::NT:
                lhs = &d_mlp_up_flat;
                rhs = &inp_flat;
                mode_rm = EMMTranspose::TN;
                break;
            case EMMTranspose::TN:
                lhs = &inp_flat;
                rhs = &d_mlp_up_flat;
                mode_rm = EMMTranspose::NN;
                break;
            case EMMTranspose::TT:
                lhs = &d_mlp_up_flat;
                rhs = &inp_flat;
                mode_rm = EMMTranspose::TT;
                break;
        }

        int M = 0, N = 0, K = 0;
        matmul_dims(*lhs, *rhs, mode_rm, M, N, K);
        EMMTranspose mode_col = swap_transpose(mode_rm);
        matmul(*d_weight_ptr, *rhs, *lhs, std::nullopt, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, do_accumulate, mRunState.MainStream);
    }

    if (layer_idx >= 0 && d_inp_ptr) {
        auto& grads = mRunState.simplified_grads(layer_idx);
        grads.d_ln2.Data = d_inp_ptr->Data;
    }

    // Hook invocation for LoRA backward (MLP up/gate)
    // Skip dense MLP hooks for MoE models - MoE has different backward path (grouped GEMM)
    const bool is_moe = mConfig.NumExperts > 0;
    if (hook && *hook && op.attrs.backward_hook_point.has_value() && !is_moe) {
        (*hook)(layer_idx, do_accumulate, mRunState.MainStream, *op.attrs.backward_hook_point, mHookContext);
    }
}

void CompiledExecutor::dispatch_rope_backward(const CompiledOp& op) {
    // inputs: d_qkv_rope, freq_cis, position_ids
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& freqs = resolve_tensor(op.inputs[1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[2]);
    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());

    // For FP8 hybrid backward, record abs_max of d_qkv for subsequent quantization
    float* abs_max_ptr = mRunState.has_fp8_hybrid_backward()
        ? mRunState.simplified_quant_grads().d_qkv.abs_max()
        : nullptr;

    rope_backward(d_qkv, d_out, freqs, reinterpret_cast<int*>(pos_ids.Data), abs_max_ptr,
                  static_cast<int>(mB), static_cast<int>(mT),
                  Hq, Hkv, Hs, op.attrs.rotary_dim, mRunState.MainStream);
}

void CompiledExecutor::dispatch_qkv_qk_norm_rope_backward(const CompiledOp& op) {
    // inputs (from autodiff): d_qkv_out, qkv_out (saved), q_norm_weight, k_norm_weight, q_rstd, k_rstd, freqs, pos_ids
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& qkv = resolve_tensor(op.inputs[1]);       // Saved QKV output from forward
    Tensor& q_norm = resolve_tensor(op.inputs[2]);
    Tensor& k_norm = resolve_tensor(op.inputs[3]);
    Tensor& q_rstd = resolve_tensor(op.inputs[4]);    // Saved RSTD (FP32)
    Tensor& k_rstd = resolve_tensor(op.inputs[5]);    // Saved RSTD (FP32)
    Tensor& freqs = resolve_tensor(op.inputs[6]);
    Tensor& pos_ids = resolve_tensor(op.inputs[7]);

    // DEBUG: Trace inputs with L2 norms
    int debug_layer_idx = -1;
    std::string debug_field;
    parse_block_param(op.inputs[1].name, debug_layer_idx, debug_field);
    static int qknorm_trace_count = 0;
    if (debug_layer_idx == 0 && qknorm_trace_count < 10) {
        qknorm_trace_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        // Compute L2 norm of d_out
        const int N = static_cast<int>(std::min(static_cast<long>(d_out.nelem()), 10000L));
        std::vector<float> d_out_all(N);
        cudaMemcpy(d_out_all.data(), d_out.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
        float d_out_sum_sq = 0.0f, d_out_max = 0.0f;
        for (int i = 0; i < N; ++i) {
            d_out_sum_sq += d_out_all[i] * d_out_all[i];
            if (std::fabs(d_out_all[i]) > d_out_max) d_out_max = std::fabs(d_out_all[i]);
        }
        fprintf(stderr, "[QK_NORM_ROPE_BWD] Layer %d: d_out INPUT ptr=%p L2=%.6f, max=%.6f, first4=%.9f,%.9f,%.9f,%.9f\n",
                debug_layer_idx, d_out.Data, std::sqrt(d_out_sum_sq), d_out_max, d_out_all[0], d_out_all[1], d_out_all[2], d_out_all[3]);
    }

    // Targeted trace for layers 8/9.
    static int qk_l8_trace = 0;
    if ((debug_layer_idx == 8 || debug_layer_idx == 9) && qk_l8_trace < 12) {
        fprintf(stderr,
                "[QK_BWD_L8_IN] layer=%d d_out=%s qkv=%s q_rstd=%s k_rstd=%s\n",
                debug_layer_idx,
                op.inputs[0].name.c_str(),
                op.inputs[1].name.c_str(),
                op.inputs[4].name.c_str(),
                op.inputs[5].name.c_str());
        log_tensor_mag_unbounded("QK_BWD_L8_DOUT", debug_layer_idx, op.inputs[0].name, d_out, 4096);
        log_tensor_mag_unbounded("QK_BWD_L8_QKV", debug_layer_idx, op.inputs[1].name, qkv, 4096);
        log_tensor_mag_unbounded("QK_BWD_L8_QRSTD", debug_layer_idx, op.inputs[4].name, q_rstd, 4096);
        log_tensor_mag_unbounded("QK_BWD_L8_KRSTD", debug_layer_idx, op.inputs[5].name, k_rstd, 4096);
        const std::size_t dout_total = static_cast<std::size_t>(d_out.nelem());
        const std::size_t qkv_total = static_cast<std::size_t>(qkv.nelem());
        if (dout_total > 4096) {
            log_tensor_sample_stats("QK_BWD_L8_DOUT_MID", d_out, dout_total / 2, 4096);
        }
        if (qkv_total > 4096) {
            log_tensor_sample_stats("QK_BWD_L8_QKV_MID", qkv, qkv_total / 2, 4096);
        }
        qk_l8_trace++;
    }

    const int top_layer = static_cast<int>(mConfig.NumLayers) - 1;
    // DEBUG: Trace top-layer inputs (qkv/rstd/freqs/pos_ids) to catch NaN sources.
    static int qk_top_trace = 0;
    if (debug_layer_idx == top_layer && qk_top_trace < 6) {
        fprintf(stderr,
                "[QK_NORM_ROPE_BWD_TOP] layer=%d qkv=%p d_out=%p q_rstd=%p k_rstd=%p freqs=%p pos_ids=%p qkv_shape=%s d_out_shape=%s\n",
                debug_layer_idx,
                qkv.Data,
                d_out.Data,
                q_rstd.Data,
                k_rstd.Data,
                freqs.Data,
                pos_ids.Data,
                tensor_shape_str(qkv).c_str(),
                tensor_shape_str(d_out).c_str());
        log_tensor_stats_ex("QK_BWD_TOP_QKV", debug_layer_idx, op.inputs[1].name, qkv, 4096, true);
        log_tensor_stats_ex("QK_BWD_TOP_DOUT", debug_layer_idx, op.inputs[0].name, d_out, 4096, true);
        log_tensor_stats_ex("QK_BWD_TOP_Q_RSTD", debug_layer_idx, op.inputs[4].name, q_rstd, 4096, true);
        log_tensor_stats_ex("QK_BWD_TOP_K_RSTD", debug_layer_idx, op.inputs[5].name, k_rstd, 4096, true);
        log_tensor_stats_ex("QK_BWD_TOP_FREQS", debug_layer_idx, op.inputs[6].name, freqs, 4096, true);
        if (pos_ids.Data && pos_ids.nelem() > 0) {
            const std::size_t n = std::min<std::size_t>(8, static_cast<std::size_t>(pos_ids.nelem()));
            std::vector<int> ids(n, 0);
            cudaMemcpy(ids.data(), pos_ids.Data, n * sizeof(int), cudaMemcpyDeviceToHost);
            fprintf(stderr,
                    "[QK_BWD_TOP_POS_IDS] first=%d,%d,%d,%d,%d,%d,%d,%d\n",
                    ids.size() > 0 ? ids[0] : 0,
                    ids.size() > 1 ? ids[1] : 0,
                    ids.size() > 2 ? ids[2] : 0,
                    ids.size() > 3 ? ids[3] : 0,
                    ids.size() > 4 ? ids[4] : 0,
                    ids.size() > 5 ? ids[5] : 0,
                    ids.size() > 6 ? ids[6] : 0,
                    ids.size() > 7 ? ids[7] : 0);
        }
        qk_top_trace++;
    }

    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const int qkv_channels = Hs * (Hq + 2 * Hkv);
    const int q_rows = Hq * Hs;

    Tensor qkv_view = (qkv.Rank == 4) ? view_tensor(qkv, {mB, mT, static_cast<long>(qkv_channels)}) : qkv;
    Tensor d_out_view = (d_out.Rank == 4) ? view_tensor(d_out, {mB, mT, static_cast<long>(qkv_channels)}) : d_out;
    Tensor d_qkv_view = (d_qkv.Rank == 4) ? view_tensor(d_qkv, {mB, mT, static_cast<long>(qkv_channels)}) : d_qkv;

    // Initialize d_qkv with upstream gradient (d_out) so V gradients pass through unchanged.
    // The fused or fallback kernels update Q/K channels in-place.
    if (d_qkv_view.Data != d_out_view.Data) {
        const std::size_t bytes = static_cast<std::size_t>(d_out_view.nelem()) * get_dtype_size(d_out_view.DType);
        CUDA_CHECK(cudaMemcpyAsync(d_qkv_view.Data, d_out_view.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    const bool disable_fused = env_enabled("SUROGATE_DISABLE_FUSED_QK_ROPE_BWD");
    if (disable_fused) {
        // Fallback: undo RoPE on gradients and activations, then run non-RoPE QK RMSNorm backward.
        const int rotary_dim = op.attrs.rotary_dim;
        rope_backward(d_qkv_view, d_qkv_view, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                      static_cast<int>(mB), static_cast<int>(mT),
                      Hq, Hkv, Hs, rotary_dim, mRunState.MainStream);
        rope_backward(qkv_view, qkv_view, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                      static_cast<int>(mB), static_cast<int>(mT),
                      Hq, Hkv, Hs, rotary_dim, mRunState.MainStream);
        qkv_head_rmsnorm_backward_dx(d_qkv_view, qkv_view, q_norm, q_rstd,
                                     static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                     Hq, Hs, 0, mRunState.MainStream);
        qkv_head_rmsnorm_backward_dx(d_qkv_view, qkv_view, k_norm, k_rstd,
                                     static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                     Hkv, Hs, q_rows, mRunState.MainStream);
    } else {
        // Combined backward for Q and K norms with RoPE
        // Q norm backward (with RoPE): channel_offset=0
        qkv_head_rmsnorm_rope_backward_dx(d_qkv_view, qkv_view, q_norm, q_rstd,
                                           freqs, reinterpret_cast<int*>(pos_ids.Data),
                                           static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                           Hq, Hs, 0, mRunState.MainStream, nullptr);

        // K norm backward (with RoPE): channel_offset=q_rows
        qkv_head_rmsnorm_rope_backward_dx(d_qkv_view, qkv_view, k_norm, k_rstd,
                                           freqs, reinterpret_cast<int*>(pos_ids.Data),
                                           static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                           Hkv, Hs, q_rows, mRunState.MainStream, nullptr);
    }

    // One-time NaN watchdog for QK-norm/RoPE backward output.
    static bool qk_bwd_nan_logged = false;
    Tensor d_out_scan = d_out_view;
    Tensor d_qkv_scan = d_qkv_view;
    if (d_out_scan.Rank > 2 && d_out_scan.Sizes[0] == mB && d_out_scan.Sizes[1] == mT) {
        d_out_scan = view_tensor(d_out_scan, {mB * mT, d_out_scan.Sizes[d_out_scan.Rank - 1]});
    }
    if (d_qkv_scan.Rank > 2 && d_qkv_scan.Sizes[0] == mB && d_qkv_scan.Sizes[1] == mT) {
        d_qkv_scan = view_tensor(d_qkv_scan, {mB * mT, d_qkv_scan.Sizes[d_qkv_scan.Rank - 1]});
    }
    float dq_row_min = 0.0f;
    float dq_row_max = 0.0f;
    const bool dq_row_nan = tensor_row_has_nan_or_inf(d_qkv_scan, 3, &dq_row_min, &dq_row_max);
    if (!qk_bwd_nan_logged && dq_row_nan) {
        auto shape_str = [](const Tensor& t) {
            std::string s = "[";
            for (int i = 0; i < t.Rank; ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(t.Sizes[i]);
            }
            s += "]";
            return s;
        };
        float row_min = 0.0f;
        float row_max = 0.0f;
        const bool row_nan = tensor_row_has_nan_or_inf(d_out_scan, 3, &row_min, &row_max);
        fprintf(stderr,
                "[QK_NORM_ROPE_BWD_NAN] layer=%d d_out=%s d_qkv=%s d_out_shape=%s d_qkv_shape=%s d_out_row_nan=%d row_min=%.6f row_max=%.6f d_qkv_row_min=%.6f d_qkv_row_max=%.6f\n",
                debug_layer_idx,
                op.inputs[0].name.c_str(),
                op.outputs[0].name.c_str(),
                shape_str(d_out_scan).c_str(),
                shape_str(d_qkv_scan).c_str(),
                row_nan ? 1 : 0,
                row_min,
                row_max,
                dq_row_min,
                dq_row_max);
        qk_bwd_nan_logged = true;
    }

    // V doesn't have normalization - its gradients pass through unchanged
    // The d_out already contains the V gradients at the correct offset

    // DEBUG: Print output with L2 norm (include Layer 25 and top layer for debugging)
    static int qk_25_trace = 0;
    if ((debug_layer_idx == 25 || debug_layer_idx == top_layer) && qk_25_trace < 12) {
        qk_25_trace++;
        cudaStreamSynchronize(mRunState.MainStream);
        const int N = static_cast<int>(std::min(static_cast<long>(d_qkv.nelem()), 10000L));
        std::vector<float> d_qkv_all(N);
        cudaMemcpy(d_qkv_all.data(), d_qkv.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
        float sum_sq = 0.0f, max_val = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum_sq += d_qkv_all[i] * d_qkv_all[i];
            if (std::fabs(d_qkv_all[i]) > max_val) max_val = std::fabs(d_qkv_all[i]);
        }
        fprintf(stderr, "[QK_NORM_ROPE_BWD] Layer %d: d_qkv OUTPUT L2=%.6f, max=%.6f, vals=%.6f,%.6f,%.6f,%.6f\n",
                debug_layer_idx, std::sqrt(sum_sq), max_val, d_qkv_all[0], d_qkv_all[1], d_qkv_all[2], d_qkv_all[3]);
    }
    static int qk_l8_out_trace = 0;
    if ((debug_layer_idx == 8 || debug_layer_idx == 9) && qk_l8_out_trace < 12) {
        log_tensor_mag_unbounded("QK_BWD_L8_DQKV", debug_layer_idx, op.outputs[0].name, d_qkv, 4096);
        const std::size_t dqkv_total = static_cast<std::size_t>(d_qkv.nelem());
        if (dqkv_total > 4096) {
            log_tensor_sample_stats("QK_BWD_L8_DQKV_MID", d_qkv, dqkv_total / 2, 4096);
        }
        qk_l8_out_trace++;
    }
    if ((debug_layer_idx == 0 || debug_layer_idx == 26) && qknorm_trace_count <= 10) {
        cudaStreamSynchronize(mRunState.MainStream);
        const int N = static_cast<int>(std::min(static_cast<long>(d_qkv.nelem()), 10000L));
        std::vector<float> d_qkv_all(N);
        cudaMemcpy(d_qkv_all.data(), d_qkv.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
        float sum_sq = 0.0f, max_val = 0.0f;
        int nonzero = 0;
        for (int i = 0; i < N; ++i) {
            sum_sq += d_qkv_all[i] * d_qkv_all[i];
            if (std::fabs(d_qkv_all[i]) > max_val) max_val = std::fabs(d_qkv_all[i]);
            if (std::fabs(d_qkv_all[i]) > 1e-10f) nonzero++;
        }
        fprintf(stderr, "[QK_NORM_ROPE_BWD] Layer %d: d_qkv OUTPUT name=%s ptr=%p, L2=%.6f, max=%.6f, nonzero=%d/%d, vals[0..3]=%.9f,%.9f,%.9f,%.9f\n",
                debug_layer_idx, op.outputs[0].name.c_str(), d_qkv.Data, std::sqrt(sum_sq), max_val, nonzero, N,
                d_qkv_all[0], d_qkv_all[1], d_qkv_all[2], d_qkv_all[3]);
    }

    // For FP8 hybrid backward, record abs_max of the final d_qkv for subsequent quantization
    if (mRunState.has_fp8_hybrid_backward()) {
        float* abs_max_ptr = mRunState.simplified_quant_grads().d_qkv.abs_max();
        abs_max(abs_max_ptr, d_qkv_view, static_cast<long>(d_qkv_view.nelem()),
                mRunState.DeviceProp, mRunState.MainStream);
    }
}

void CompiledExecutor::dispatch_flash_attention_backward(const CompiledOp& op) {
    // inputs (from autodiff): d_out, out (attention output), lse, qkv
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& out = resolve_tensor(op.inputs[1]);
    Tensor& lse = resolve_tensor(op.inputs[2]);
    Tensor& qkv = resolve_tensor(op.inputs[3]);
    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    // DEBUG: Print attention backward inputs for layer 0, 25, and 26
    int debug_layer_idx = -1;
    std::string debug_field;
    parse_block_param(op.inputs[1].name, debug_layer_idx, debug_field);

    // Trace Layer 25 flash attention backward for explosion debugging
    static int attn_25_trace = 0;
    if (debug_layer_idx == 25 && attn_25_trace < 10) {
        attn_25_trace++;
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> d_out_vals(4);
        cudaMemcpy(d_out_vals.data(), d_out.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        // Compute L2 norm of d_out
        const int N = static_cast<int>(std::min(static_cast<long>(d_out.nelem()), 10000L));
        std::vector<float> all(N);
        cudaMemcpy(all.data(), d_out.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
        float sum_sq = 0.0f, max_val = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum_sq += all[i] * all[i];
            if (std::fabs(all[i]) > max_val) max_val = std::fabs(all[i]);
        }
        fprintf(stderr, "[FLASH_ATTN_BWD] Layer %d: d_out INPUT L2=%.6f max=%.6f vals=%.6f,%.6f,%.6f,%.6f\n",
                debug_layer_idx, std::sqrt(sum_sq), max_val, d_out_vals[0], d_out_vals[1], d_out_vals[2], d_out_vals[3]);
        // Also trace saved activations (qkv, out, lse) to check if they're corrupted
        const long sample_token = 3;
        std::vector<float> qkv_vals(4), out_vals(4), lse_vals(4);
        const bool qkv_ok = copy_tensor_token_sample_as_f32(qkv, sample_token, qkv_vals.size(), qkv_vals);
        const bool out_ok = copy_tensor_token_sample_as_f32(out, sample_token, out_vals.size(), out_vals);
        const bool lse_ok = copy_tensor_token_sample_as_f32(lse, sample_token, lse_vals.size(), lse_vals);
        fprintf(stderr,
                "[FLASH_ATTN_BWD] Layer %d: token=%ld qkv ptr=%p ok=%d vals=%.6f,%.6f,%.6f,%.6f  "
                "out ptr=%p ok=%d vals=%.6f,%.6f,%.6f,%.6f  lse ok=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                debug_layer_idx, sample_token, qkv.Data, qkv_ok ? 1 : 0,
                qkv_vals[0], qkv_vals[1], qkv_vals[2], qkv_vals[3],
                out.Data, out_ok ? 1 : 0,
                out_vals[0], out_vals[1], out_vals[2], out_vals[3],
                lse_ok ? 1 : 0,
                lse_vals[0], lse_vals[1], lse_vals[2], lse_vals[3]);
    }

    // Targeted trace for layers 8/9 where residual spikes first appear.
    static int attn_8_9_trace = 0;
    if ((debug_layer_idx == 8 || debug_layer_idx == 9) && attn_8_9_trace < 12) {
        fprintf(stderr,
                "[ATTN_BWD_L8] layer=%d d_out=%s out=%s lse=%s qkv=%s\n",
                debug_layer_idx,
                op.inputs[0].name.c_str(),
                op.inputs[1].name.c_str(),
                op.inputs[2].name.c_str(),
                op.inputs[3].name.c_str());
        log_tensor_mag_unbounded("ATTN_BWD_L8_DOUT", debug_layer_idx, op.inputs[0].name, d_out, 4096);
        log_tensor_mag_unbounded("ATTN_BWD_L8_OUT", debug_layer_idx, op.inputs[1].name, out, 4096);
        log_tensor_mag_unbounded("ATTN_BWD_L8_LSE", debug_layer_idx, op.inputs[2].name, lse, 4096);
        log_tensor_mag_unbounded("ATTN_BWD_L8_QKV", debug_layer_idx, op.inputs[3].name, qkv, 4096);
        attn_8_9_trace++;
    }

    const int top_layer = static_cast<int>(mConfig.NumLayers) - 1;
    if (debug_layer_idx == 0 || debug_layer_idx == 26 || debug_layer_idx == top_layer) {
        cudaStreamSynchronize(mRunState.MainStream);
        const long sample_token = 3;
        std::vector<float> qkv_vals(4), out_vals(4), d_out_vals(4);
        const bool qkv_ok = copy_tensor_token_sample_as_f32(qkv, sample_token, qkv_vals.size(), qkv_vals);
        const bool out_ok = copy_tensor_token_sample_as_f32(out, sample_token, out_vals.size(), out_vals);
        const bool d_out_ok = copy_tensor_token_sample_as_f32(d_out, sample_token, d_out_vals.size(), d_out_vals);
        fprintf(stderr,
                "[ATTN_BWD] Layer %d token=%ld qkv_rope ptr=%p ok=%d values=%.6f, %.6f, %.6f, %.6f\n",
                debug_layer_idx, sample_token, qkv.Data, qkv_ok ? 1 : 0,
                qkv_vals[0], qkv_vals[1], qkv_vals[2], qkv_vals[3]);
        fprintf(stderr,
                "[ATTN_BWD] Layer %d token=%ld att ptr=%p ok=%d values=%.6f, %.6f, %.6f, %.6f\n",
                debug_layer_idx, sample_token, out.Data, out_ok ? 1 : 0,
                out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
        fprintf(stderr,
                "[ATTN_BWD] Layer %d token=%ld d_out(d_att) ok=%d values=%.6f, %.6f, %.6f, %.6f\n",
                debug_layer_idx, sample_token, d_out_ok ? 1 : 0,
                d_out_vals[0], d_out_vals[1], d_out_vals[2], d_out_vals[3]);
        fprintf(stderr,
                "[ATTN_BWD] Layer %d shapes: d_out=%s qkv=%s out=%s lse=%s\n",
                debug_layer_idx,
                tensor_shape_str(d_out).c_str(),
                tensor_shape_str(qkv).c_str(),
                tensor_shape_str(out).c_str(),
                tensor_shape_str(lse).c_str());
    }

    Tensor* out_ptr = &out;
    Tensor* lse_ptr = &lse;
    Tensor* qkv_ptr = &qkv;

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const bool cudnn_gqa_ok = (Hq == Hkv);
    const bool force_custom_bwd = (std::getenv("SUROGATE_ATTN_BWD_CUSTOM") != nullptr);
    const bool force_cudnn_bwd = (std::getenv("SUROGATE_ATTN_BWD_FORCE_CUDNN") != nullptr);
    const bool use_cudnn_bwd = force_cudnn_bwd || (cudnn_gqa_ok && !force_custom_bwd);
    const bool gqa_fallback_full = !cudnn_gqa_ok;
    auto shape_vec = [](const Tensor& t) {
        return std::vector<long>(t.Sizes.begin(), t.Sizes.begin() + t.Rank);
    };

    if (!mRunState.scratch().cudnn_workspace.Data) {
        mRunState.temp_acquire(mRunState.scratch().cudnn_workspace);
        mTemps.push_back(mRunState.scratch().cudnn_workspace);
    }

    // Parse layer_idx for use in cuDNN call
    int layer_idx = -1;
    std::string field;
    parse_block_param(op.inputs[3].name, layer_idx, field);

    // NaN watchdog for attention backward inputs (log a few occurrences).
    static int attn_bwd_nan_in_logged = 0;
    Tensor d_out_scan = d_out;
    if (d_out_scan.Rank > 2 && d_out_scan.Sizes[0] == mB && d_out_scan.Sizes[1] == mT) {
        d_out_scan = view_tensor(d_out_scan, {mB * mT, d_out_scan.Sizes[d_out_scan.Rank - 1]});
    }
    float attn_row_min = 0.0f;
    float attn_row_max = 0.0f;
    const bool attn_row_nan = tensor_row_has_nan_or_inf(d_out_scan, 3, &attn_row_min, &attn_row_max);
    if (attn_bwd_nan_in_logged < 4 && attn_row_nan) {
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
                "[FLASH_ATTN_BWD_NAN_IN] layer=%d d_out=%s out=%s lse=%s qkv=%s row_min=%.6f row_max=%.6f\n",
                layer_idx,
                op.inputs[0].name.c_str(),
                op.inputs[1].name.c_str(),
                op.inputs[2].name.c_str(),
                op.inputs[3].name.c_str(),
                attn_row_min,
                attn_row_max);
        dump_sample("FLASH_ATTN_BWD_NAN_DOUT", d_out, op.inputs[0].name);
        dump_sample("FLASH_ATTN_BWD_NAN_OUT", out, op.inputs[1].name);
        dump_sample("FLASH_ATTN_BWD_NAN_LSE", lse, op.inputs[2].name);
        dump_sample("FLASH_ATTN_BWD_NAN_QKV", qkv, op.inputs[3].name);
        attn_bwd_nan_in_logged++;
    }

    // FIX: Zero-initialize d_qkv before cuDNN attention backward to prevent NaN from uninitialized memory.
    // The d_qkv buffer may contain stale values from previous operations, and cuDNN attention backward
    // may read parts of this buffer even though it's expected to be output-only. Without this zero-init,
    // NaN values can appear in the gradient computation and propagate through the backward pass.
    fill_zero(d_qkv, mRunState.MainStream);

    const int attn_chunks = mOptions.AttBwdChunks;
    if (attn_chunks < 1) {
        throw std::runtime_error("attn_bwd_chunks must be >= 1");
    }
    const int chunk_B = (attn_chunks == 1)
        ? static_cast<int>(mB)
        : static_cast<int>(div_exact(mB, static_cast<long>(attn_chunks)));

    // Signature: attention_backward_cudnn(dqkv, stats, out, dout, qkv, workspace, handle, B, T, Hq, Hkv, HS, stream)
    if (attn_chunks == 1) {
        if (!use_cudnn_bwd) {
            static int bwd_skip_count = 0;
            if (bwd_skip_count < 4) {
                fprintf(stderr,
                        "[CUDNN_ATTN_BWD_SKIP] Hq=%d Hkv=%d reason=%s\n",
                        Hq, Hkv, force_custom_bwd ? "forced_custom" : "gqa");
                bwd_skip_count++;
            }
            if (d_out.DType == ETensorDType::BF16) {
                auto& scratch = mRunState.scratch();
                bool have_fallback_bufs =
                    scratch.attn_qkv_f32.Data && scratch.attn_out_f32.Data &&
                    scratch.attn_d_out_f32.Data && scratch.attn_d_qkv_f32.Data;
                if (have_fallback_bufs) {
                    const std::size_t need_qkv = static_cast<std::size_t>(qkv.nelem());
                    const std::size_t need_out = static_cast<std::size_t>(out.nelem());
                    const std::size_t need_d_out = static_cast<std::size_t>(d_out.nelem());
                    const std::size_t need_d_qkv = static_cast<std::size_t>(d_qkv.nelem());
                    const std::size_t have_qkv = static_cast<std::size_t>(scratch.attn_qkv_f32.nelem());
                    const std::size_t have_out = static_cast<std::size_t>(scratch.attn_out_f32.nelem());
                    const std::size_t have_d_out = static_cast<std::size_t>(scratch.attn_d_out_f32.nelem());
                    const std::size_t have_d_qkv = static_cast<std::size_t>(scratch.attn_d_qkv_f32.nelem());
                    const bool too_small =
                        (have_qkv < need_qkv) || (have_out < need_out) ||
                        (have_d_out < need_d_out) || (have_d_qkv < need_d_qkv);
                    if (too_small) {
                        static int scratch_small_log = 0;
                        if (scratch_small_log < 8) {
                            fprintf(stderr,
                                    "[ATTN_F32_SCRATCH_TOO_SMALL] have_qkv=%zu need_qkv=%zu have_out=%zu need_out=%zu "
                                    "have_d_out=%zu need_d_out=%zu have_d_qkv=%zu need_d_qkv=%zu\\n",
                                    have_qkv, need_qkv, have_out, need_out,
                                    have_d_out, need_d_out, have_d_qkv, need_d_qkv);
                            scratch_small_log++;
                        }
                        if (mRunState.Allocator) {
                            if (have_qkv < need_qkv) {
                                scratch.attn_qkv_f32 = mRunState.Allocator->allocate(
                                    ETensorDType::FP32, "attn_qkv_f32", EAllocationType::ON_DEVICE, shape_vec(qkv));
                            }
                            if (have_out < need_out) {
                                scratch.attn_out_f32 = mRunState.Allocator->allocate(
                                    ETensorDType::FP32, "attn_out_f32", EAllocationType::ON_DEVICE, shape_vec(out));
                            }
                            if (have_d_out < need_d_out) {
                                scratch.attn_d_out_f32 = mRunState.Allocator->allocate(
                                    ETensorDType::FP32, "attn_d_out_f32", EAllocationType::ON_DEVICE, shape_vec(d_out));
                            }
                            if (have_d_qkv < need_d_qkv) {
                                scratch.attn_d_qkv_f32 = mRunState.Allocator->allocate(
                                    ETensorDType::FP32, "attn_d_qkv_f32", EAllocationType::ON_DEVICE, shape_vec(d_qkv));
                            }
                        }
                        have_fallback_bufs =
                            scratch.attn_qkv_f32.nelem() >= static_cast<long>(need_qkv) &&
                            scratch.attn_out_f32.nelem() >= static_cast<long>(need_out) &&
                            scratch.attn_d_out_f32.nelem() >= static_cast<long>(need_d_out) &&
                            scratch.attn_d_qkv_f32.nelem() >= static_cast<long>(need_d_qkv);
                        if (!have_fallback_bufs) {
                            static int scratch_disable_log = 0;
                            if (scratch_disable_log < 4) {
                                fprintf(stderr, "[ATTN_F32_SCRATCH_DISABLE] using stack fallback buffers\n");
                                scratch_disable_log++;
                            }
                        }
                    }
                }
                Tensor qkv_f32 = have_fallback_bufs ? view_tensor(scratch.attn_qkv_f32, shape_vec(qkv))
                                                    : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(qkv), "qkv_f32");
                Tensor out_f32 = have_fallback_bufs ? view_tensor(scratch.attn_out_f32, shape_vec(out))
                                                    : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(out), "attn_out_f32");
                Tensor d_out_f32 = have_fallback_bufs ? view_tensor(scratch.attn_d_out_f32, shape_vec(d_out))
                                                      : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(d_out), "d_attn_out_f32");
                Tensor d_qkv_f32 = have_fallback_bufs ? view_tensor(scratch.attn_d_qkv_f32, shape_vec(d_qkv))
                                                      : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(d_qkv), "d_qkv_f32");
                const bool trace_ranges = (std::getenv("SUROGATE_TRACE_ATTN_F32_RANGES") != nullptr);
                static int attn_f32_range_log = 0;
                if (trace_ranges && attn_f32_range_log < 8) {
                    attn_f32_range_log++;
                    auto log_range = [](const char* tag, const Tensor& t) {
                        const auto start = reinterpret_cast<std::uintptr_t>(t.Data);
                        const auto end = start + t.bytes();
                        fprintf(stderr,
                                "[%s] ptr=%p bytes=%zu range=[0x%zx..0x%zx) shape=%s dtype=%s\n",
                                tag,
                                t.Data,
                                t.bytes(),
                                static_cast<std::size_t>(start),
                                static_cast<std::size_t>(end),
                                tensor_shape_str(t).c_str(),
                                dtype_to_str(t.DType));
                    };
                    log_range("ATTN_F32_QKV", qkv_f32);
                    log_range("ATTN_F32_OUT", out_f32);
                    log_range("ATTN_F32_DOUT", d_out_f32);
                    log_range("ATTN_F32_DQKV", d_qkv_f32);
                    log_range("ATTN_BF16_QKV", qkv);
                    log_range("ATTN_BF16_OUT", out);
                    log_range("ATTN_BF16_DOUT", d_out);
                    log_range("ATTN_BF16_DQKV", d_qkv);
                    fprintf(stderr,
                            "[ATTN_F32_STACK] used=%zu unused=%zu\n",
                            mRunState.Stack.bytes_used(),
                            mRunState.Stack.unused_capacity());
                }
                convert_dtype(qkv_f32.get<float>(), qkv.get<nv_bfloat16>(), qkv.nelem(), mRunState.MainStream);
                convert_dtype(d_out_f32.get<float>(), d_out.get<nv_bfloat16>(), d_out.nelem(), mRunState.MainStream);
                // attention_backward_custom uses atomicAdd into d_qkv_f32; ensure it's zeroed.
                fill_zero(d_qkv_f32, mRunState.MainStream);

                if (gqa_fallback_full) {
                    Tensor lse_f32 = have_fallback_bufs && scratch.attn_lse_f32.Data
                        ? view_tensor(scratch.attn_lse_f32, shape_vec(lse))
                        : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(lse), "attn_lse_f32");
                    attention_forward_custom(out_f32, lse_f32, qkv_f32,
                                             static_cast<int>(mB), static_cast<int>(mT),
                                             Hq, Hkv, Hs, mRunState.MainStream);
                    attention_backward_custom(d_qkv_f32, lse_f32, out_f32, d_out_f32, qkv_f32,
                                              static_cast<int>(mB), static_cast<int>(mT),
                                              Hq, Hkv, Hs, mRunState.MainStream);
                } else {
                    convert_dtype(out_f32.get<float>(), out.get<nv_bfloat16>(), out.nelem(), mRunState.MainStream);
                    if (lse.DType == ETensorDType::BF16) {
                        Tensor lse_f32 = have_fallback_bufs && scratch.attn_lse_f32.Data
                            ? view_tensor(scratch.attn_lse_f32, shape_vec(lse))
                            : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(lse), "attn_lse_f32");
                        convert_dtype(lse_f32.get<float>(), lse.get<nv_bfloat16>(), lse.nelem(), mRunState.MainStream);
                        attention_backward_custom(d_qkv_f32, lse_f32, out_f32, d_out_f32, qkv_f32,
                                                  static_cast<int>(mB), static_cast<int>(mT),
                                                  Hq, Hkv, Hs, mRunState.MainStream);
                    } else {
                        attention_backward_custom(d_qkv_f32, *lse_ptr, out_f32, d_out_f32, qkv_f32,
                                                  static_cast<int>(mB), static_cast<int>(mT),
                                                  Hq, Hkv, Hs, mRunState.MainStream);
                    }
                }

                convert_dtype(d_qkv.get<nv_bfloat16>(), d_qkv_f32.get<float>(), d_qkv.nelem(), mRunState.MainStream);
            } else {
                if (gqa_fallback_full) {
                    auto& scratch = mRunState.scratch();
                    const bool have_fallback_bufs = scratch.attn_out_f32.Data && scratch.attn_lse_f32.Data;
                    Tensor out_f32 = have_fallback_bufs ? view_tensor(scratch.attn_out_f32, shape_vec(out))
                                                        : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(out), "attn_out_f32");
                    Tensor lse_f32 = have_fallback_bufs ? view_tensor(scratch.attn_lse_f32, shape_vec(lse))
                                                        : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(lse), "attn_lse_f32");
                    attention_forward_custom(out_f32, lse_f32, *qkv_ptr,
                                             static_cast<int>(mB), static_cast<int>(mT),
                                             Hq, Hkv, Hs, mRunState.MainStream);
                    attention_backward_custom(d_qkv, lse_f32, out_f32, d_out, *qkv_ptr,
                                              static_cast<int>(mB), static_cast<int>(mT),
                                              Hq, Hkv, Hs, mRunState.MainStream);
                } else {
                    attention_backward_custom(d_qkv, *lse_ptr, *out_ptr, d_out, *qkv_ptr,
                                              static_cast<int>(mB), static_cast<int>(mT),
                                              Hq, Hkv, Hs, mRunState.MainStream);
                }
            }
        } else {
            static int bwd_force_count = 0;
            if (force_cudnn_bwd && bwd_force_count < 4) {
                fprintf(stderr,
                        "[CUDNN_ATTN_BWD_FORCE] Hq=%d Hkv=%d\n",
                        Hq, Hkv);
                bwd_force_count++;
            }
            attention_backward_cudnn(d_qkv, *lse_ptr, *out_ptr, d_out, *qkv_ptr,
                                     mRunState.scratch().cudnn_workspace,
                                     mRunState.CudnnHandle,
                                     static_cast<int>(mB), static_cast<int>(mT),
                                     Hq, Hkv, Hs, mRunState.MainStream);
        }

    // DEBUG: Print d_qkv output for layer 0, 25, and 26
    if (debug_layer_idx == 25 && attn_25_trace <= 10) {
            cudaStreamSynchronize(mRunState.MainStream);
            const int N = static_cast<int>(std::min(static_cast<long>(d_qkv.nelem()), 10000L));
            std::vector<float> all(N);
            cudaMemcpy(all.data(), d_qkv.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
            float sum_sq = 0.0f, max_val = 0.0f;
            for (int i = 0; i < N; ++i) {
                sum_sq += all[i] * all[i];
                if (std::fabs(all[i]) > max_val) max_val = std::fabs(all[i]);
            }
            fprintf(stderr, "[FLASH_ATTN_BWD] Layer %d: d_qkv OUTPUT L2=%.6f max=%.6f vals=%.6f,%.6f,%.6f,%.6f\n",
                    debug_layer_idx, std::sqrt(sum_sq), max_val, all[0], all[1], all[2], all[3]);
        }
        if (debug_layer_idx == 0 || debug_layer_idx == 26 || debug_layer_idx == top_layer) {
            cudaStreamSynchronize(mRunState.MainStream);
            std::vector<float> vals(8);
            cudaMemcpy(vals.data(), d_qkv.Data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[ATTN_BWD] Layer %d d_qkv OUTPUT ptr=%p, values=%.6f, %.6f, %.6f, %.6f\n",
                    debug_layer_idx, d_qkv.Data, vals[0], vals[1], vals[2], vals[3]);
        }

        static int attn_8_9_out_trace = 0;
        if ((debug_layer_idx == 8 || debug_layer_idx == 9) && attn_8_9_out_trace < 12) {
            log_tensor_mag_unbounded("ATTN_BWD_L8_DQKV", debug_layer_idx, op.outputs[0].name, d_qkv, 4096);
            const std::size_t dqkv_total = static_cast<std::size_t>(d_qkv.nelem());
            if (dqkv_total > 4096) {
                log_tensor_sample_stats("ATTN_BWD_L8_DQKV_MID", d_qkv, dqkv_total / 2, 4096);
            }
            attn_8_9_out_trace++;
        }

        // NaN watchdog for attention backward outputs (log a few occurrences).
        static int attn_bwd_nan_out_logged = 0;
        Tensor d_qkv_scan = d_qkv;
        if (d_qkv_scan.Rank > 2 && d_qkv_scan.Sizes[0] == mB && d_qkv_scan.Sizes[1] == mT) {
            d_qkv_scan = view_tensor(d_qkv_scan, {mB * mT, d_qkv_scan.Sizes[d_qkv_scan.Rank - 1]});
        }
        float dq_row_min = 0.0f;
        float dq_row_max = 0.0f;
        const bool dq_row_nan = tensor_row_has_nan_or_inf(d_qkv_scan, 3, &dq_row_min, &dq_row_max);
        if (attn_bwd_nan_out_logged < 4 && dq_row_nan) {
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
                    "[FLASH_ATTN_BWD_NAN_OUT] layer=%d d_qkv=%s row_min=%.6f row_max=%.6f\n",
                    layer_idx,
                    op.outputs[0].name.c_str(),
                    dq_row_min,
                    dq_row_max);
            dump_sample("FLASH_ATTN_BWD_NAN_DQKV", d_qkv, op.outputs[0].name);
            auto log_nan_row = [&](const char* tag, const Tensor& t, const std::string& name) {
                long nan_row = -1;
                float row_min = 0.0f;
                float row_max = 0.0f;
                if (!find_first_nan_row(t, &nan_row, &row_min, &row_max)) {
                    fprintf(stderr, "[%s] name=%s any_nan=0\n", tag, name.c_str());
                    return;
                }
                if (t.Rank >= 2 && t.Sizes[0] == mB * mT) {
                    const long b = nan_row / static_cast<long>(mT);
                    const long tok = nan_row % static_cast<long>(mT);
                    fprintf(stderr,
                            "[%s] name=%s any_nan=1 row=%ld (b=%ld t=%ld) row_min=%.6f row_max=%.6f\n",
                            tag, name.c_str(), nan_row, b, tok, row_min, row_max);
                } else {
                    fprintf(stderr,
                            "[%s] name=%s any_nan=1 row=%ld row_min=%.6f row_max=%.6f\n",
                            tag, name.c_str(), nan_row, row_min, row_max);
                }
            };
            Tensor d_out_scan2 = d_out;
            if (d_out_scan2.Rank > 2 && d_out_scan2.Sizes[0] == mB && d_out_scan2.Sizes[1] == mT) {
                d_out_scan2 = view_tensor(d_out_scan2, {mB * mT, d_out_scan2.Sizes[d_out_scan2.Rank - 1]});
            }
            Tensor out_scan2 = out;
            if (out_scan2.Rank > 2 && out_scan2.Sizes[0] == mB && out_scan2.Sizes[1] == mT) {
                out_scan2 = view_tensor(out_scan2, {mB * mT, out_scan2.Sizes[out_scan2.Rank - 1]});
            }
            Tensor qkv_scan2 = qkv;
            if (qkv_scan2.Rank > 2 && qkv_scan2.Sizes[0] == mB && qkv_scan2.Sizes[1] == mT) {
                qkv_scan2 = view_tensor(qkv_scan2, {mB * mT, qkv_scan2.Sizes[qkv_scan2.Rank - 1]});
            }
            log_nan_row("FLASH_ATTN_BWD_NAN_INPUT_DOUT", d_out_scan2, op.inputs[0].name);
            log_nan_row("FLASH_ATTN_BWD_NAN_INPUT_OUT", out_scan2, op.inputs[1].name);
            log_nan_row("FLASH_ATTN_BWD_NAN_INPUT_QKV", qkv_scan2, op.inputs[3].name);
            log_nan_row("FLASH_ATTN_BWD_NAN_INPUT_LSE", lse, op.inputs[2].name);
            attn_bwd_nan_out_logged++;
        }
        return;
    }

    for (int chunk = 0; chunk < attn_chunks; ++chunk) {
        const long start = static_cast<long>(chunk) * static_cast<long>(chunk_B);
        const long end = start + static_cast<long>(chunk_B);
        Tensor d_out_chunk = slice(d_out, 0, start, end);
        Tensor out_chunk = slice(*out_ptr, 0, start, end);
        Tensor lse_chunk = slice(*lse_ptr, 0, start, end);
        Tensor qkv_chunk = slice(*qkv_ptr, 0, start, end);
        Tensor d_qkv_chunk = slice(d_qkv, 0, start, end);

        if (!use_cudnn_bwd) {
            if (d_out_chunk.DType == ETensorDType::BF16) {
                auto& scratch = mRunState.scratch();
                const bool have_fallback_bufs =
                    scratch.attn_qkv_f32.Data && scratch.attn_out_f32.Data &&
                    scratch.attn_d_out_f32.Data && scratch.attn_d_qkv_f32.Data;
                Tensor qkv_f32 = have_fallback_bufs ? slice(scratch.attn_qkv_f32, 0, start, end)
                                                    : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(qkv_chunk), "qkv_f32");
                Tensor out_f32 = have_fallback_bufs ? slice(scratch.attn_out_f32, 0, start, end)
                                                    : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(out_chunk), "attn_out_f32");
                Tensor d_out_f32 = have_fallback_bufs ? slice(scratch.attn_d_out_f32, 0, start, end)
                                                      : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(d_out_chunk), "d_attn_out_f32");
                Tensor d_qkv_f32 = have_fallback_bufs ? slice(scratch.attn_d_qkv_f32, 0, start, end)
                                                      : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(d_qkv_chunk), "d_qkv_f32");
                convert_dtype(qkv_f32.get<float>(), qkv_chunk.get<nv_bfloat16>(), qkv_chunk.nelem(), mRunState.MainStream);
                convert_dtype(d_out_f32.get<float>(), d_out_chunk.get<nv_bfloat16>(), d_out_chunk.nelem(), mRunState.MainStream);

                if (gqa_fallback_full) {
                    Tensor lse_f32 = have_fallback_bufs && scratch.attn_lse_f32.Data
                        ? slice(scratch.attn_lse_f32, 0, start, end)
                        : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(lse_chunk), "attn_lse_f32");
                    attention_forward_custom(out_f32, lse_f32, qkv_f32,
                                             chunk_B, static_cast<int>(mT),
                                             Hq, Hkv, Hs, mRunState.MainStream);
                    attention_backward_custom(d_qkv_f32, lse_f32, out_f32, d_out_f32, qkv_f32,
                                              chunk_B, static_cast<int>(mT),
                                              Hq, Hkv, Hs, mRunState.MainStream);
                } else {
                    convert_dtype(out_f32.get<float>(), out_chunk.get<nv_bfloat16>(), out_chunk.nelem(), mRunState.MainStream);
                    if (lse_chunk.DType == ETensorDType::BF16) {
                        Tensor lse_f32 = have_fallback_bufs && scratch.attn_lse_f32.Data
                            ? slice(scratch.attn_lse_f32, 0, start, end)
                            : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(lse_chunk), "attn_lse_f32");
                        convert_dtype(lse_f32.get<float>(), lse_chunk.get<nv_bfloat16>(), lse_chunk.nelem(), mRunState.MainStream);
                        attention_backward_custom(d_qkv_f32, lse_f32, out_f32, d_out_f32, qkv_f32,
                                                  chunk_B, static_cast<int>(mT),
                                                  Hq, Hkv, Hs, mRunState.MainStream);
                    } else {
                        attention_backward_custom(d_qkv_f32, lse_chunk, out_f32, d_out_f32, qkv_f32,
                                                  chunk_B, static_cast<int>(mT),
                                                  Hq, Hkv, Hs, mRunState.MainStream);
                    }
                }

                convert_dtype(d_qkv_chunk.get<nv_bfloat16>(), d_qkv_f32.get<float>(), d_qkv_chunk.nelem(), mRunState.MainStream);
            } else {
                if (gqa_fallback_full) {
                    auto& scratch = mRunState.scratch();
                    const bool have_fallback_bufs = scratch.attn_out_f32.Data && scratch.attn_lse_f32.Data;
                    Tensor out_f32 = have_fallback_bufs ? slice(scratch.attn_out_f32, 0, start, end)
                                                        : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(out_chunk), "attn_out_f32");
                    Tensor lse_f32 = have_fallback_bufs ? slice(scratch.attn_lse_f32, 0, start, end)
                                                        : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(lse_chunk), "attn_lse_f32");
                    attention_forward_custom(out_f32, lse_f32, qkv_chunk,
                                             chunk_B, static_cast<int>(mT),
                                             Hq, Hkv, Hs, mRunState.MainStream);
                    attention_backward_custom(d_qkv_chunk, lse_f32, out_f32, d_out_chunk, qkv_chunk,
                                              chunk_B, static_cast<int>(mT),
                                              Hq, Hkv, Hs, mRunState.MainStream);
                } else {
                    attention_backward_custom(d_qkv_chunk, lse_chunk, out_chunk, d_out_chunk, qkv_chunk,
                                              chunk_B, static_cast<int>(mT),
                                              Hq, Hkv, Hs, mRunState.MainStream);
                }
            }
        } else {
            static int bwd_force_count = 0;
            if (force_cudnn_bwd && bwd_force_count < 4) {
                fprintf(stderr,
                        "[CUDNN_ATTN_BWD_FORCE] Hq=%d Hkv=%d\n",
                        Hq, Hkv);
                bwd_force_count++;
            }
            attention_backward_cudnn(d_qkv_chunk, lse_chunk, out_chunk, d_out_chunk, qkv_chunk,
                                     mRunState.scratch().cudnn_workspace,
                                     mRunState.CudnnHandle,
                                     chunk_B, static_cast<int>(mT),
                                     Hq, Hkv, Hs, mRunState.MainStream);
        }
    }
}

void CompiledExecutor::dispatch_zeros_backward(const CompiledOp& op) {
    // Zeros backward is a no-op - gradient doesn't flow through zeros initialization
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

void CompiledExecutor::dispatch_embedding_backward(const CompiledOp& op) {
    // Skip embedding backward entirely in LoRA-only mode
    if (mRunState.is_lora_only_mode()) {
        return;
    }

    // inputs: d_encoded, token_ids
    // outputs: d_embedding (sparse update)
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    if (op.outputs.empty() || op.outputs[0].name.empty()) {
        return;  // Skip if no output expected
    }

    // Get the pre-allocated gradient tensor
    auto it = mTensorMap.find(op.outputs[0].name);
    if (it == mTensorMap.end()) {
        // Gradient not allocated (embedding frozen in LoRA mode)
        return;
    }
    Tensor& d_emb = it->second;

    // encoder_backward requires CPU-side inputs for deterministic bucketing
    if (!mLastInputsCpu || !mLastInputsCpu->Data) {
        throw std::runtime_error("CompiledExecutor: embedding_backward requires CPU inputs (set_last_inputs_cpu)");
    }

    static int emb_bwd_log_count = 0;
    const int vocab = mConfig.VocabSize;
    const int total_tokens = static_cast<int>(mB * mT);
    const long hidden = (d_emb.Rank > 1) ? d_emb.Sizes[1] : 0;

    if (emb_bwd_log_count < 16) {
        int cpu_min = std::numeric_limits<int>::max();
        int cpu_max = std::numeric_limits<int>::min();
        int cpu_oob = 0;
        int cpu_oob_samples = 0;
        int token0 = 0;
        int token3 = 0;
        if (mLastInputsCpu->DType == ETensorDType::INT32) {
            const int* cpu = reinterpret_cast<const int*>(mLastInputsCpu->Data);
            for (int i = 0; i < total_tokens; ++i) {
                const int v = cpu[i];
                if (i == 0) token0 = v;
                if (i == 3) token3 = v;
                cpu_min = std::min(cpu_min, v);
                cpu_max = std::max(cpu_max, v);
                if (v < 0 || v >= vocab) {
                    if (cpu_oob_samples < 8) {
                        fprintf(stderr, "[EMB_BWD_CPU_OOB] idx=%d val=%d vocab=%d\n", i, v, vocab);
                        cpu_oob_samples++;
                    }
                    cpu_oob++;
                }
            }
        } else {
            fprintf(stderr, "[EMB_BWD_CPU_INPUTS] unsupported dtype=%s\n", dtype_to_str(mLastInputsCpu->DType));
        }

        fprintf(stderr,
                "[EMB_BWD_CPU_INPUTS] B=%d T=%d vocab=%d min=%d max=%d oob=%d token0=%d token3=%d\n",
                static_cast<int>(mB), static_cast<int>(mT), vocab,
                cpu_min, cpu_max, cpu_oob, token0, token3);

        if (mRunState.Inputs.Data && mRunState.Inputs.DType == ETensorDType::INT32 && total_tokens > 0) {
            const int sample = std::min(total_tokens, 8);
            std::vector<int> gpu(sample, 0);
            CUDA_CHECK(cudaMemcpyAsync(gpu.data(), mRunState.Inputs.Data, sample * sizeof(int),
                                       cudaMemcpyDeviceToHost, mRunState.MainStream));
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            int mismatch = 0;
            if (mLastInputsCpu->DType == ETensorDType::INT32) {
                const int* cpu = reinterpret_cast<const int*>(mLastInputsCpu->Data);
                for (int i = 0; i < sample; ++i) {
                    if (cpu[i] != gpu[i]) {
                        mismatch++;
                    }
                }
            }
            fprintf(stderr, "[EMB_BWD_INPUTS_CMP] sample=%d mismatch=%d\n", sample, mismatch);
        }

        log_nan_sample("BWD_DOUT", op.inputs[0].layer_idx, op.inputs[0].name, d_out, 3);
        if (tensor_sample_has_nan_or_inf(d_out, 3)) {
            log_tensor_stats_ex("BWD_DOUT_NAN", op.inputs[0].layer_idx, op.inputs[0].name, d_out, 4096, true);
        }

        if (hidden > 0) {
            if (tensor_sample_has_nan_or_inf(d_emb, 3)) {
                const std::size_t off = static_cast<std::size_t>(3) * static_cast<std::size_t>(hidden);
                log_tensor_sample_stats("EMB_DGRAD_PRE_ROW3", d_emb, off, static_cast<std::size_t>(hidden));
            }
            if (token3 >= 0 && token3 < vocab && tensor_sample_has_nan_or_inf(d_emb, token3)) {
                const std::size_t off = static_cast<std::size_t>(token3) * static_cast<std::size_t>(hidden);
                log_tensor_sample_stats("EMB_DGRAD_PRE_ROW_TOK3", d_emb, off, static_cast<std::size_t>(hidden));
            }
        }

        emb_bwd_log_count++;
    }

    unsigned int seed = mRngSeedFn ? mRngSeedFn() : 0;

    encoder_backward(d_emb,
                     mRunState.scratch().encoder_bwd_scratch,
                     mRunState.scratch().encoder_bwd_indices,
                     mRunState.scratch().encoder_bwd_info,
                     d_out,
                     mRunState.Inputs,
                     *mLastInputsCpu,
                     static_cast<int>(mB), static_cast<int>(mT), mConfig.HiddenSize,
                     seed,
                     mRunState.MainStream,
                     mRunState.side_stream_event(),
                     mRunState.side_stream());

    if (emb_bwd_log_count < 32 && hidden > 0) {
        int token3 = 0;
        if (mLastInputsCpu->DType == ETensorDType::INT32) {
            const int* cpu = reinterpret_cast<const int*>(mLastInputsCpu->Data);
            if (total_tokens > 3) {
                token3 = cpu[3];
            }
        }
        if (token3 >= 0 && token3 < vocab) {
            if (tensor_sample_has_nan_or_inf(d_emb, token3)) {
                const std::size_t off = static_cast<std::size_t>(token3) * static_cast<std::size_t>(hidden);
                log_tensor_sample_stats("EMB_DGRAD_ROW_TOK3", d_emb, off, static_cast<std::size_t>(hidden));
            }
        }
        if (tensor_sample_has_nan_or_inf(d_emb, 3)) {
            const std::size_t off = static_cast<std::size_t>(3) * static_cast<std::size_t>(hidden);
            log_tensor_sample_stats("EMB_DGRAD_ROW3", d_emb, off, static_cast<std::size_t>(hidden));
        }
        emb_bwd_log_count++;
    }
}

void CompiledExecutor::dispatch_cross_entropy_loss_backward(const CompiledOp& op) {
    Tensor& d_loss = resolve_tensor(op.inputs[0]);
    Tensor& logits = resolve_tensor(op.inputs[1]);
    Tensor& targets = resolve_tensor(op.inputs[2]);
    Tensor& d_logits = ensure_output_tensor(op.outputs[0]);

    const int BT = static_cast<int>(logits.Sizes[0]);
    const int V = static_cast<int>(logits.Sizes[1]);
    const int P = V;

    // HuggingFace-style normalization: use reduction="sum" semantics.
    // dloss = 1.0 means each valid token contributes equally to the gradient sum.
    // The actual normalization by accumulated valid tokens happens in global_norm_sqrt.
    // Robustly seed d_loss even if the name has SSA suffixes or mapped to loss/losses.
    const std::string d_loss_name = strip_ssa_suffix(op.inputs[0].name);
    if (op.inputs[0].slot == TensorSlot::DLoss ||
        op.inputs[0].slot == TensorSlot::Losses ||
        d_loss_name == "d_loss" || d_loss_name == "loss" || d_loss_name == "losses") {
        fill_constant(d_loss, 1.0f, static_cast<std::size_t>(d_loss.nelem()), mRunState.MainStream);
    }

    Tensor logsumexp_view{};
    Tensor* logsumexp = nullptr;
    if (mRunState.scratch().cross_entropy_logsumexp.Data) {
        logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
        logsumexp_view.Sizes[0] = BT;
        logsumexp_view.Rank = 1;
        logsumexp = &logsumexp_view;
    }

    if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
        chunked_cross_entropy_backward(d_logits, logits, logsumexp, d_loss, targets,
                                       BT, V, P, mRunState.MainStream);
    } else {
        fused_cross_entropy_backward(d_logits, logits, logsumexp, d_loss, targets,
                                     BT, V, P, mRunState.MainStream);
    }
}

void CompiledExecutor::dispatch_fused_lm_head_loss_backward(const CompiledOp& op) {
    Tensor& d_loss = resolve_tensor(op.inputs[0]);
    Tensor& xF_flat = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);
    Tensor& targets = resolve_tensor(op.inputs[3]);

    Tensor* d_xF_ptr = nullptr;
    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        d_xF_ptr = &ensure_output_tensor(op.outputs[0]);
    }

    Tensor* d_weight_ptr = nullptr;
    bool d_weight_accumulate = false;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        std::string weight_name = op.outputs[1].name;
        if (auto base = base_param_from_grad(weight_name)) {
            weight_name = *base;
        } else if (weight_name.rfind("d_", 0) == 0) {
            weight_name = weight_name.substr(2);
        }
        bool accum = false;
        Tensor* grad = mGrads.get_param_grad(weight_name, accum);
        d_weight_accumulate = mAccumulateTensors.count(op.outputs[1].name) > 0;
        if (!d_weight_accumulate) {
            if (auto base = base_param_from_grad(op.outputs[1].name)) {
                d_weight_accumulate = mAccumulateTensors.count("d_" + *base) > 0;
            }
        }
        if (grad && grad->Data) {
            d_weight_ptr = &ensure_output_tensor(op.outputs[1]);
        }
    }

    // HuggingFace-style normalization: use reduction="sum" semantics.
    // dloss = 1.0 means each valid token contributes equally to the gradient sum.
    // The actual normalization by accumulated valid tokens happens in global_norm_sqrt.
    // Robustly seed d_loss even if the name has SSA suffixes or mapped to loss/losses.
    const std::string d_loss_name = strip_ssa_suffix(op.inputs[0].name);
    bool d_loss_seeded = false;
    if (op.inputs[0].slot == TensorSlot::DLoss ||
        op.inputs[0].slot == TensorSlot::Losses ||
        d_loss_name == "d_loss" || d_loss_name == "loss" || d_loss_name == "losses") {
        fill_constant(d_loss, 1.0f, static_cast<std::size_t>(d_loss.nelem()), mRunState.MainStream);
        d_loss_seeded = true;
    }

    // DEBUG: Trace loss backward input slot/name and d_loss values.
    static int loss_bwd_trace_count = 0;
    if (loss_bwd_trace_count < 5) {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> vals(4, 0.0f);
        const std::size_t count = std::min<std::size_t>(4, static_cast<std::size_t>(d_loss.nelem()));
        if (count > 0 && d_loss.Data) {
            cudaMemcpy(vals.data(), d_loss.Data, count * sizeof(float), cudaMemcpyDeviceToHost);
        }
        fprintf(stderr,
                "[LM_HEAD_LOSS_BWD] d_loss input name='%s' base='%s' slot=%d layer_idx=%d seeded=%d ptr=%p "
                "nelem=%ld dtype=%d vals=%.9f,%.9f,%.9f,%.9f\n",
                op.inputs[0].name.c_str(),
                d_loss_name.c_str(),
                static_cast<int>(op.inputs[0].slot),
                op.inputs[0].layer_idx,
                d_loss_seeded ? 1 : 0,
                d_loss.Data,
                static_cast<long>(d_loss.nelem()),
                static_cast<int>(d_loss.DType),
                vals[0], vals[1], vals[2], vals[3]);
        loss_bwd_trace_count++;
    }

    const long BT = xF_flat.Sizes[0];
    const int V = static_cast<int>(weight.Sizes[0]);
    const int C = static_cast<int>(weight.Sizes[1]);
    const int P = V;

    const int lmhead_chunks = std::max(1, mOptions.LMHeadChunks);
    const long nano_batch_size = BT / static_cast<long>(lmhead_chunks);
    const bool need_lm_head =
        mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled()) &&
        (mOptions.LMHeadChunks > 1);
    if (need_lm_head && mComm) {
        mWeightManager->gather_lm_head(*mComm, mRunState.MainStream);
    }

    const std::size_t xf_stride = get_dtype_size(xF_flat.DType);
    const std::size_t tgt_stride = get_dtype_size(targets.DType);
    const std::size_t lse_stride = get_dtype_size(ETensorDType::FP32);
    const std::size_t dloss_stride = get_dtype_size(d_loss.DType);

    // DEBUG: Full targets stats (ignore/oob/valid) and head/tail sample.
    static int targets_full_trace_count = 0;
    if (targets_full_trace_count < 3) {
        cudaStreamSynchronize(mRunState.MainStream);
        if (targets.DType == ETensorDType::INT32) {
            const std::size_t n = static_cast<std::size_t>(targets.nelem());
            std::size_t ignore = 0;
            std::size_t oob = 0;
            std::size_t valid = 0;
            const std::size_t chunk = 1u << 20;  // 1M ints max per copy
            const std::byte* base = static_cast<const std::byte*>(targets.Data);
            std::vector<int> buf;
            for (std::size_t offset = 0; offset < n; offset += chunk) {
                const std::size_t count = std::min(chunk, n - offset);
                buf.resize(count);
                CUDA_CHECK(cudaMemcpy(buf.data(), base + offset * tgt_stride,
                                      count * sizeof(int), cudaMemcpyDeviceToHost));
                for (std::size_t i = 0; i < count; ++i) {
                    const int v = buf[i];
                    if (v == -100) {
                        ignore++;
                    } else if (v < 0 || v >= V) {
                        oob++;
                    } else {
                        valid++;
                    }
                }
            }

            int head[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            int tail[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (n > 0) {
                const std::size_t head_count = std::min<std::size_t>(8, n);
                CUDA_CHECK(cudaMemcpy(head, base, head_count * sizeof(int), cudaMemcpyDeviceToHost));
                const std::size_t tail_offset = (n > 8) ? (n - 8) : 0;
                const std::size_t tail_count = std::min<std::size_t>(8, n);
                CUDA_CHECK(cudaMemcpy(tail, base + tail_offset * tgt_stride,
                                      tail_count * sizeof(int), cudaMemcpyDeviceToHost));
            }

            fprintf(stderr,
                    "[LM_HEAD_TARGETS_FULL] nelem=%zu V=%d ignore=%zu oob=%zu valid=%zu "
                    "head=%d,%d,%d,%d,%d,%d,%d,%d tail=%d,%d,%d,%d,%d,%d,%d,%d\n",
                    n, V, ignore, oob, valid,
                    head[0], head[1], head[2], head[3], head[4], head[5], head[6], head[7],
                    tail[0], tail[1], tail[2], tail[3], tail[4], tail[5], tail[6], tail[7]);
        } else {
            fprintf(stderr, "[LM_HEAD_TARGETS_FULL] dtype=%d nelem=%ld (unsupported)\n",
                    static_cast<int>(targets.DType), static_cast<long>(targets.nelem()));
        }
        targets_full_trace_count++;
    }

    static int d_xf_full_trace_count = 0;
    const bool trace_full_dxf = (d_xF_ptr != nullptr && d_xf_full_trace_count < 3);
    double dxf_full_sum_sq = 0.0;
    long first_nonzero_offset = -1;
    bool dlogits_nonzero_logged = false;

    auto tensor_sum_sq = [&](const Tensor& t) -> double {
        const std::size_t n = static_cast<std::size_t>(t.nelem());
        if (n == 0 || !t.Data) {
            return 0.0;
        }
        const std::size_t chunk = 1u << 20;  // 1M elements per copy
        std::vector<float> buf;
        double sum_sq = 0.0;
        for (std::size_t offset = 0; offset < n; offset += chunk) {
            const std::size_t count = std::min(chunk, n - offset);
            buf.resize(count);
            Tensor tmp = t;
            tmp.Data = static_cast<std::byte*>(tmp.Data) +
                       offset * get_dtype_size(tmp.DType);
            tmp.Sizes[0] = static_cast<long>(count);
            tmp.Rank = 1;
            if (!copy_tensor_sample_as_f32(tmp, count, buf)) {
                break;
            }
            for (std::size_t i = 0; i < count; ++i) {
                const double v = static_cast<double>(buf[i]);
                sum_sq += v * v;
            }
        }
        return sum_sq;
    };

    for (int nano_step = 0; nano_step < lmhead_chunks; ++nano_step) {
        const long token_offset = static_cast<long>(nano_step) * nano_batch_size;

        Tensor xF_slice = xF_flat;
        xF_slice.Data = static_cast<std::byte*>(xF_slice.Data) +
                        static_cast<std::size_t>(token_offset) * xf_stride * static_cast<std::size_t>(C);
        xF_slice.Sizes[0] = nano_batch_size;
        xF_slice.Sizes[1] = C;
        xF_slice.Rank = 2;

        Tensor tgt_slice = targets;
        tgt_slice.Data = static_cast<std::byte*>(tgt_slice.Data) +
                         static_cast<std::size_t>(token_offset) * tgt_stride;
        tgt_slice.Sizes[0] = nano_batch_size;
        tgt_slice.Rank = 1;

        // DEBUG: Trace targets for loss backward (ignore/oob counts + first values).
        static int targets_trace_count = 0;
        if (targets_trace_count < 5 && nano_step == 0) {
            cudaStreamSynchronize(mRunState.MainStream);
            const std::size_t sample = std::min<std::size_t>(
                64, static_cast<std::size_t>(tgt_slice.nelem()));
            if (tgt_slice.DType == ETensorDType::INT32 && sample > 0) {
                std::vector<int> host(sample);
                cudaMemcpy(host.data(), tgt_slice.Data, sample * sizeof(int), cudaMemcpyDeviceToHost);
                int ignore = 0;
                int oob = 0;
                for (std::size_t i = 0; i < sample; ++i) {
                    const int v = host[i];
                    if (v == -100) {
                        ignore++;
                    } else if (v < 0 || v >= V) {
                        oob++;
                    }
                }
                int vals[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                const std::size_t vcount = std::min<std::size_t>(8, sample);
                for (std::size_t i = 0; i < vcount; ++i) {
                    vals[i] = host[i];
                }
                fprintf(stderr,
                        "[LM_HEAD_TARGETS] token_offset=%ld dtype=%d nelem=%ld sample=%zu V=%d ignore=%d oob=%d "
                        "vals=%d,%d,%d,%d,%d,%d,%d,%d\n",
                        token_offset,
                        static_cast<int>(tgt_slice.DType),
                        static_cast<long>(tgt_slice.nelem()),
                        sample,
                        V,
                        ignore,
                        oob,
                        vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]);
            } else {
                fprintf(stderr,
                        "[LM_HEAD_TARGETS] token_offset=%ld dtype=%d nelem=%ld sample=%zu (unsupported dtype)\n",
                        token_offset,
                        static_cast<int>(tgt_slice.DType),
                        static_cast<long>(tgt_slice.nelem()),
                        sample);
            }
            targets_trace_count++;
        }

        Tensor dloss_slice = d_loss;
        dloss_slice.Data = static_cast<std::byte*>(dloss_slice.Data) +
                           static_cast<std::size_t>(token_offset) * dloss_stride;
        dloss_slice.Sizes[0] = nano_batch_size;
        dloss_slice.Rank = 1;

        Tensor logits = mRunState.non_block_activations().output;
        logits.Sizes[0] = nano_batch_size;
        logits.Sizes[1] = V;
        logits.Rank = 2;

        matmul(logits, weight, xF_slice,
               std::nullopt, nullptr, nullptr, mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               static_cast<int>(V), static_cast<int>(nano_batch_size), static_cast<int>(C),
               swap_transpose(EMMTranspose::NT), false, mRunState.MainStream);

        // DEBUG: Trace logits before CE backward.
        static int logits_pre_trace_count = 0;
        if (logits_pre_trace_count < 5 && nano_step == 0) {
            cudaStreamSynchronize(mRunState.MainStream);
            std::vector<float> vals(4, 0.0f);
            const bool ok = copy_tensor_sample_as_f32(logits, vals.size(), vals);
            fprintf(stderr,
                    "[LM_HEAD_LOGITS_PRE] token_offset=%ld dtype=%d ptr=%p ok=%d vals=%.9f,%.9f,%.9f,%.9f\n",
                    token_offset,
                    static_cast<int>(logits.DType),
                    logits.Data,
                    ok ? 1 : 0,
                    vals[0], vals[1], vals[2], vals[3]);
            logits_pre_trace_count++;
        }

        Tensor logsumexp_view{};
        Tensor* logsumexp = nullptr;
        if (mRunState.scratch().cross_entropy_logsumexp.Data) {
            logsumexp_view = mRunState.scratch().cross_entropy_logsumexp;
            logsumexp_view.Data = static_cast<std::byte*>(logsumexp_view.Data) +
                                  static_cast<std::size_t>(token_offset) * lse_stride;
            logsumexp_view.Sizes[0] = nano_batch_size;
            logsumexp_view.Rank = 1;
            logsumexp = &logsumexp_view;
        }

        if (V > CROSS_ENTROPY_MAX_FUSED_SIZE) {
            chunked_cross_entropy_backward(logits, logits, logsumexp, dloss_slice, tgt_slice,
                                           static_cast<int>(nano_batch_size), V, P, mRunState.MainStream);
        } else {
            fused_cross_entropy_backward(logits, logits, logsumexp, dloss_slice, tgt_slice,
                                         static_cast<int>(nano_batch_size), V, P, mRunState.MainStream);
        }

        // DEBUG: Trace d_logits (stored in logits buffer) after CE backward.
        static int dlogits_trace_count = 0;
        if (dlogits_trace_count < 5 && nano_step == 0) {
            cudaStreamSynchronize(mRunState.MainStream);
            std::vector<float> vals(4, 0.0f);
            const bool ok = copy_tensor_sample_as_f32(logits, vals.size(), vals);
            fprintf(stderr,
                    "[LM_HEAD_DLOGITS] token_offset=%ld dtype=%d ptr=%p ok=%d vals=%.9f,%.9f,%.9f,%.9f\n",
                    token_offset,
                    static_cast<int>(logits.DType),
                    logits.Data,
                    ok ? 1 : 0,
                    vals[0], vals[1], vals[2], vals[3]);
            dlogits_trace_count++;
        }

        if (d_weight_ptr) {
            const bool accumulate = d_weight_accumulate || (nano_step != 0);
            matmul(*d_weight_ptr, xF_slice, logits,
                   std::nullopt, nullptr, nullptr, mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   static_cast<int>(C), static_cast<int>(V), static_cast<int>(nano_batch_size),
                   swap_transpose(EMMTranspose::TN), accumulate, mRunState.MainStream);
        }

        if (d_xF_ptr) {
            Tensor d_xF_slice = *d_xF_ptr;
            const std::size_t dx_stride = get_dtype_size(d_xF_slice.DType);
            d_xF_slice.Data = static_cast<std::byte*>(d_xF_slice.Data) +
                              static_cast<std::size_t>(token_offset) * dx_stride * static_cast<std::size_t>(C);
            d_xF_slice.Sizes[0] = nano_batch_size;
            d_xF_slice.Sizes[1] = C;
            d_xF_slice.Rank = 2;

            matmul(d_xF_slice, weight, logits,
                   std::nullopt, nullptr, nullptr, mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   static_cast<int>(C), static_cast<int>(nano_batch_size), static_cast<int>(V),
                   swap_transpose(EMMTranspose::NN), false, mRunState.MainStream);

            if (trace_full_dxf) {
                cudaStreamSynchronize(mRunState.MainStream);
                const double sum_sq = tensor_sum_sq(d_xF_slice);
                dxf_full_sum_sq += sum_sq;
                if (sum_sq > 0.0 && first_nonzero_offset < 0) {
                    first_nonzero_offset = token_offset;
                }
                if (!dlogits_nonzero_logged && sum_sq > 0.0) {
                    std::vector<float> vals(4, 0.0f);
                    const bool ok = copy_tensor_sample_as_f32(logits, vals.size(), vals);
                    fprintf(stderr,
                            "[LM_HEAD_DLOGITS_NONZERO] token_offset=%ld dtype=%d ptr=%p ok=%d sumsq=%.9e "
                            "vals=%.9f,%.9f,%.9f,%.9f\n",
                            token_offset,
                            static_cast<int>(logits.DType),
                            logits.Data,
                            ok ? 1 : 0,
                            sum_sq,
                            vals[0], vals[1], vals[2], vals[3]);
                    dlogits_nonzero_logged = true;
                }
            }

            // DEBUG: Trace d_xF output values + L2 norm for loss backward.
            static int d_xf_trace_count = 0;
            if (d_xf_trace_count < 5 && nano_step == 0) {
                cudaStreamSynchronize(mRunState.MainStream);
                std::vector<float> vals(4, 0.0f);
                const bool ok = copy_tensor_sample_as_f32(d_xF_slice, vals.size(), vals);
                double l2_sum = 0.0;
                if (ok) {
                    const std::size_t n = static_cast<std::size_t>(d_xF_slice.nelem());
                    const std::size_t chunk = 1u << 20;  // 1M elements per chunk
                    std::vector<float> buf;
                    for (std::size_t offset = 0; offset < n; offset += chunk) {
                        const std::size_t count = std::min(chunk, n - offset);
                        buf.resize(count);
                        Tensor tmp = d_xF_slice;
                        tmp.Data = static_cast<std::byte*>(tmp.Data) +
                                   offset * get_dtype_size(tmp.DType);
                        tmp.Sizes[0] = static_cast<long>(count);
                        tmp.Rank = 1;
                        if (!copy_tensor_sample_as_f32(tmp, count, buf)) {
                            break;
                        }
                        for (std::size_t i = 0; i < count; ++i) {
                            const double v = static_cast<double>(buf[i]);
                            l2_sum += v * v;
                        }
                    }
                }
                const double l2 = std::sqrt(l2_sum);
                fprintf(stderr,
                        "[LM_HEAD_LOSS_BWD_OUT] d_xF name='%s' slot=%d ptr=%p nelem=%ld dtype=%d ok=%d l2=%.9e "
                        "vals=%.9f,%.9f,%.9f,%.9f\n",
                        op.outputs[0].name.c_str(),
                        static_cast<int>(op.outputs[0].slot),
                        d_xF_slice.Data,
                        static_cast<long>(d_xF_slice.nelem()),
                        static_cast<int>(d_xF_slice.DType),
                        ok ? 1 : 0,
                        l2,
                        vals[0], vals[1], vals[2], vals[3]);
                d_xf_trace_count++;
            }
        }
    }

    if (trace_full_dxf) {
        const double l2 = std::sqrt(dxf_full_sum_sq);
        fprintf(stderr,
                "[LM_HEAD_DXF_FULL] nelem=%ld l2=%.9e nonzero=%d first_nonzero_offset=%ld\n",
                d_xF_ptr ? static_cast<long>(d_xF_ptr->nelem()) : -1,
                l2,
                (dxf_full_sum_sq > 0.0) ? 1 : 0,
                first_nonzero_offset);
        d_xf_full_trace_count++;
    }


    if (need_lm_head) {
        mWeightManager->release_lm_head(mRunState.MainStream);
    }
}

// MoE backward dispatch implementations

void CompiledExecutor::dispatch_silu_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& d_inp = ensure_output_tensor(op.outputs[0]);

    const long N = static_cast<long>(inp.nelem());
    // Kernel signature: silu_backward(dinp, inp, dout, n, stream)
    silu_backward(d_inp, inp, d_out, N, mRunState.MainStream);

    mTensorMap[op.outputs[0].name] = d_inp;
}

void CompiledExecutor::dispatch_mul_backward(const CompiledOp& op) {
    // Element-wise multiplication backward kernel not yet implemented
    // This is only needed for shared_expert path which is disabled by default
    throw std::runtime_error("CompiledExecutor: element-wise mul_backward operation not yet implemented. "
                             "Set use_shared_expert=False in your model config.");
}

void CompiledExecutor::dispatch_moe_softmax_backward(const CompiledOp& op) {
    Tensor& d_probs = resolve_tensor(op.inputs[0]);
    Tensor& softmax_probs = resolve_tensor(op.inputs[1]);
    Tensor& d_logits = ensure_output_tensor(op.outputs[0]);

    const int num_tokens = static_cast<int>(d_probs.Sizes[0]);
    const int num_experts = static_cast<int>(d_probs.Sizes[1]);
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

    static int moe_softmax_mag = 0;
    if (moe_softmax_mag < 8 && layer_idx <= 2) {
        fprintf(stderr,
                "[MOE_SOFTMAX_BWD] layer=%d tokens=%d experts=%d\n",
                layer_idx, num_tokens, num_experts);
        log_tensor_mag("MOE_SOFTMAX_BWD_DPROBS", layer_idx, op.inputs[0].name, d_probs, 4096);
        log_tensor_mag("MOE_SOFTMAX_BWD_PROBS", layer_idx, op.inputs[1].name, softmax_probs, 4096);
        moe_softmax_mag++;
    }

    if (d_probs.DType == ETensorDType::BF16) {
        moe_softmax_backward(d_logits.get<nv_bfloat16>(),
                             d_probs.get<nv_bfloat16>(),
                             softmax_probs.get<nv_bfloat16>(),
                             num_tokens, num_experts, mRunState.MainStream);
    } else {
        moe_softmax_backward(d_logits.get<float>(),
                             d_probs.get<float>(),
                             softmax_probs.get<float>(),
                             num_tokens, num_experts, mRunState.MainStream);
    }

    mTensorMap[op.outputs[0].name] = d_logits;
}

void CompiledExecutor::dispatch_moe_sigmoid_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& sigmoid_out = resolve_tensor(op.inputs[1]);

    // Allocate output with same shape as d_out (not from compile-time inference)
    std::vector<long> d_inp_shape;
    for (int i = 0; i < d_out.Rank; ++i) {
        d_inp_shape.push_back(d_out.Sizes[i]);
    }
    Tensor d_inp = mRunState.temp_alloc(d_out.DType, d_inp_shape);
    mTemps.push_back(d_inp);

    const int num_elements = static_cast<int>(d_out.nelem());

    if (d_out.DType == ETensorDType::BF16) {
        moe_sigmoid_backward(d_inp.get<nv_bfloat16>(),
                             d_out.get<nv_bfloat16>(),
                             sigmoid_out.get<nv_bfloat16>(),
                             num_elements, mRunState.MainStream);
    } else {
        moe_sigmoid_backward(d_inp.get<float>(),
                             d_out.get<float>(),
                             sigmoid_out.get<float>(),
                             num_elements, mRunState.MainStream);
    }

    mTensorMap[op.outputs[0].name] = d_inp;
}

void CompiledExecutor::dispatch_moe_topk_backward(const CompiledOp& op) {
    Tensor& d_routing_weights = resolve_tensor(op.inputs[0]);
    Tensor& probs = resolve_tensor(op.inputs[1]);
    Tensor& expert_indices = resolve_tensor(op.inputs[2]);

    const int num_tokens = static_cast<int>(probs.Sizes[0]);
    const int num_experts = static_cast<int>(probs.Sizes[1]);
    const int top_k = op.attrs.top_k;
    const bool normalize = op.attrs.normalize_weights;
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

    static int moe_topk_mag = 0;
    if (moe_topk_mag < 8 && layer_idx <= 2) {
        fprintf(stderr,
                "[MOE_TOPK_BWD] layer=%d tokens=%d experts=%d top_k=%d normalize=%d\n",
                layer_idx, num_tokens, num_experts, top_k, normalize ? 1 : 0);
        log_tensor_mag("MOE_TOPK_BWD_DROUTING", layer_idx, op.inputs[0].name, d_routing_weights, 4096);
        log_tensor_mag("MOE_TOPK_BWD_PROBS", layer_idx, op.inputs[1].name, probs, 4096);
        moe_topk_mag++;
    }

    // Allocate output with correct shape derived from probs (not from compile-time inference)
    // d_probs must have shape [num_tokens, num_experts] matching probs
    std::vector<long> d_probs_shape = {static_cast<long>(num_tokens), static_cast<long>(num_experts)};
    Tensor d_probs = mRunState.temp_alloc(d_routing_weights.DType, d_probs_shape);
    mTemps.push_back(d_probs);

    // TopK backward kernel only supports FP32
    // If inputs are BF16, cast to FP32 temporaries and cast output back
    if (probs.DType == ETensorDType::BF16) {
        // Allocate FP32 temporaries
        Tensor d_weights_f32 = mRunState.Stack.allocate(ETensorDType::FP32,
            {static_cast<long>(num_tokens), static_cast<long>(top_k)}, "d_weights_f32");
        Tensor probs_f32 = mRunState.Stack.allocate(ETensorDType::FP32,
            {static_cast<long>(num_tokens), static_cast<long>(num_experts)}, "probs_f32");
        Tensor d_probs_f32 = mRunState.Stack.allocate(ETensorDType::FP32,
            {static_cast<long>(num_tokens), static_cast<long>(num_experts)}, "d_probs_f32");

        // Cast inputs to FP32
        convert_dtype(d_weights_f32.get<float>(), d_routing_weights.get<nv_bfloat16>(),
                      d_routing_weights.nelem(), mRunState.MainStream);
        convert_dtype(probs_f32.get<float>(), probs.get<nv_bfloat16>(),
                      probs.nelem(), mRunState.MainStream);

        // Run backward in FP32
        moe_topk_backward(d_probs_f32.get<float>(),
                          d_weights_f32.get<float>(),
                          probs_f32.get<float>(),
                          expert_indices.get<int>(),
                          num_tokens, num_experts, top_k, normalize, mRunState.MainStream);

        // Cast output back to BF16
        convert_dtype(d_probs.get<nv_bfloat16>(), d_probs_f32.get<float>(),
                      d_probs.nelem(), mRunState.MainStream);
    } else {
        // FP32 path
        moe_topk_backward(d_probs.get<float>(),
                          d_routing_weights.get<float>(),
                          probs.get<float>(),
                          expert_indices.get<int>(),
                          num_tokens, num_experts, top_k, normalize, mRunState.MainStream);
    }

    if (moe_topk_mag < 12 && layer_idx <= 2) {
        log_tensor_mag("MOE_TOPK_BWD_DPROBS", layer_idx, op.outputs[0].name, d_probs, 4096);
        moe_topk_mag++;
    }

    mTensorMap[op.outputs[0].name] = d_probs;
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

void CompiledExecutor::execute_forward(const CompiledGraph& graph,
                                       NCCLCommunicator& comm,
                                       bool full,
                                       const modules::ForwardHook* hook) {
    mComm = &comm;
    mTemps.clear();
    mTensorMap.clear();
    mCurrentLayer = -1;

    // Match GraphExecutor behavior: initialize loss/counter buffers for full forward runs.
    // This avoids stale accumulation when tests call CompiledExecutor directly.
    if (full) {
        bool has_loss_op = false;
        for (const auto& op : graph.ops) {
            if (op.type == CompiledOpType::CrossEntropyLoss ||
                op.type == CompiledOpType::FusedLMHeadLoss) {
                has_loss_op = true;
                break;
            }
        }
        if (has_loss_op) {
            fill_zero(mRunState.Losses, mRunState.MainStream);
            fill_zero(mRunState.ValidTokenCount, mRunState.MainStream);
            fill_zero(mRunState.CorrectCount, mRunState.MainStream);
        }
    }
    const int num_layers = static_cast<int>(mConfig.NumLayers);
    std::vector<DeviceMemoryStack::Checkpoint> layer_checkpoints;
    std::vector<std::size_t> layer_temp_marks;
    std::vector<char> layer_active;
    if (num_layers > 0) {
        layer_checkpoints.resize(static_cast<std::size_t>(num_layers));
        layer_temp_marks.resize(static_cast<std::size_t>(num_layers));
        layer_active.assign(static_cast<std::size_t>(num_layers), 0);
    }
    auto prune_stack_tensors = [&]() {
        for (auto it = mTensorMap.begin(); it != mTensorMap.end(); ) {
            // Skip tensors that are needed for backward (in save list)
            if (mSaveSet.count(it->first) > 0) {
                ++it;
                continue;
            }
            // Skip MoE expert_offsets - needed for backward but not in autodiff save list
            if (mConfig.NumExperts > 0 && (it->first == "moe_expert_offsets" || it->first == "moe_gather_indices")) {
                ++it;
                continue;
            }
            if (it->second.Data && mRunState.Stack.owns(it->second.Data) &&
                !mRunState.Stack.is_live(it->second.Data)) {
                it = mTensorMap.erase(it);
            } else {
                ++it;
            }
        }
    };

    // Bind known inputs
    mTensorMap["token_ids"] = mRunState.Inputs;
    mTensorMap["position_ids"] = mRunState.PositionIDs;
    mTensorMap["x0"] = mRunState.non_block_activations().encoded;

    // Ensure non-block weights are gathered if streaming/offload is enabled
    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->gather_embeddings(comm, mRunState.MainStream);
        mWeightManager->gather_final_norm(comm, mRunState.MainStream);
    }

    // Prefetch layer 0 before loop
    if (mConfig.NumLayers > 0 && !mCapturing) {
        if (mWeightManager && mWeightManager->is_streaming_enabled()) {
            mWeightManager->gather_block(0, comm, mRunState.side_stream());
        }
    }

    // Main dispatch loop - no string comparisons, direct function pointer dispatch
    for (std::size_t idx = 0; idx < graph.ops.size(); ++idx) {
        if (!full && !graph.required_mask.empty() && !graph.required_mask[idx]) {
            continue;
        }

        const auto& op = graph.ops[idx];

        // Handle layer boundaries
        if (op.layer_start >= 0) {
            if (op.layer_start < num_layers &&
                !layer_active[static_cast<std::size_t>(op.layer_start)]) {
                layer_checkpoints[static_cast<std::size_t>(op.layer_start)] = mRunState.Stack.checkpoint();
                layer_temp_marks[static_cast<std::size_t>(op.layer_start)] = mTemps.size();
                layer_active[static_cast<std::size_t>(op.layer_start)] = 1;
            }
            handle_layer_start(op.layer_start);
        }

        try {
            // Direct dispatch via switch (branch predictor friendly, no string compare)
            switch (op.type) {
                case CompiledOpType::Embedding:
                    dispatch_embedding(op);
                    break;
                case CompiledOpType::Zeros:
                    dispatch_zeros(op);
                    break;
                case CompiledOpType::FusedResidualRMSNorm:
                    dispatch_fused_residual_rmsnorm(op);
                    break;
                case CompiledOpType::View:
                    dispatch_view(op);
                    break;
                case CompiledOpType::Add:
                    dispatch_add(op);
                    break;
                case CompiledOpType::Matmul:
                case CompiledOpType::MatmulBias:
                    dispatch_matmul(op, hook);
                    break;
                case CompiledOpType::BiasAdd:
                    dispatch_bias_add(op);
                    break;
                case CompiledOpType::SwiGLU:
                    dispatch_swiglu(op);
                    break;
                case CompiledOpType::Silu:
                    dispatch_silu(op);
                    break;
                case CompiledOpType::Mul:
                    dispatch_mul(op);
                    break;
                case CompiledOpType::MatmulSwiGLU:
                    dispatch_matmul_swiglu(op);
                    break;
                case CompiledOpType::QKVQKNormRoPE:
                    dispatch_qkv_qk_norm_rope(op);
                    break;
                case CompiledOpType::RoPE:
                    dispatch_rope(op);
                    break;
                case CompiledOpType::FlashAttention:
                    dispatch_flash_attention(op);
                    break;
                case CompiledOpType::CrossEntropyLoss:
                    dispatch_cross_entropy_loss(op);
                    break;
                case CompiledOpType::FusedLMHeadLoss:
                    dispatch_fused_lm_head_loss(op);
                    break;
                // MoE operations
                case CompiledOpType::MoESoftmax:
                    dispatch_moe_softmax(op);
                    break;
                case CompiledOpType::MoESigmoid:
                    dispatch_moe_sigmoid(op);
                    break;
                case CompiledOpType::MoETopK:
                    dispatch_moe_topk(op);
                    break;
                case CompiledOpType::MoEPermute:
                    dispatch_moe_permute(op);
                    break;
                case CompiledOpType::MoEGroupedGemmGateUp:
                    dispatch_moe_grouped_gemm_gate_up(op);
                    break;
                case CompiledOpType::MoEGroupedGemmDown:
                    dispatch_moe_grouped_gemm_down(op);
                    break;
                case CompiledOpType::MoEUnpermute:
                    dispatch_moe_unpermute(op);
                    break;
                default:
                    throw std::runtime_error("CompiledExecutor: unsupported forward op type");
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "CompiledExecutor forward op " << idx << " (type=" << op_type_to_string(op.type)
                << ", id=" << op.op_id << "): " << e.what();
            throw std::runtime_error(oss.str());
        }

        // Handle layer end
        if (op.layer_end >= 0) {
            // Note: Forward activation stats are not printed because with recompute_block=true,
            // the activation buffers are shared across layers, so they only contain the last
            // layer's data at this point, not the per-layer values.
            if (op.layer_end < num_layers &&
                layer_active[static_cast<std::size_t>(op.layer_end)]) {
                // For MoE models, skip stack restore because:
                // 1. MoE backward needs forward activations (routing_weights, scatter_indices, etc.)
                // 2. The recompute mechanism (recompute_block) doesn't support MoE ops
                // 3. Without recompute, we must preserve all forward tensors for backward
                // TODO: Implement MoE-specific recompute to enable memory savings
                if (mConfig.NumExperts == 0) {
                    mRunState.Stack.restore(layer_checkpoints[static_cast<std::size_t>(op.layer_end)]);
                    if (mTemps.size() > layer_temp_marks[static_cast<std::size_t>(op.layer_end)]) {
                        mTemps.resize(layer_temp_marks[static_cast<std::size_t>(op.layer_end)]);
                    }
                    prune_stack_tensors();
                    if (mRunState.ffn_temps_on_stack()) {
                        auto& acts = mRunState.simplified_acts(op.layer_end);
                        acts.mlp_up.Data = nullptr;
                        acts.swiglu.Data = nullptr;
                    }
                    // Note: cudnn_workspace is persistently allocated, don't clear
                    layer_active[static_cast<std::size_t>(op.layer_end)] = 0;
                }
            }
            handle_layer_end(op.layer_end);
        }
    }

    // DEBUG: Summarize parameter gradient samples to spot divergence.
    static int param_grad_trace_enabled = -1;
    if (param_grad_trace_enabled < 0) {
        param_grad_trace_enabled = (std::getenv("SUROGATE_TRACE_PARAM_GRADS") != nullptr) ? 1 : 0;
    }
    if (param_grad_trace_enabled) {
        struct ParamSample {
            std::string name;
            double mean_abs{0.0};
            double max_abs{0.0};
            std::size_t count{0};
        };
        std::vector<ParamSample> samples;
        double total_sum_sq = 0.0;
        double total_sum_abs = 0.0;
        std::size_t total_count = 0;
        constexpr std::size_t kSampleN = 1024;
        for (const auto& param_name : mGrads.param_names()) {
            bool accumulate = false;
            Tensor* grad_tensor = mGrads.get_param_grad(param_name, accumulate);
            if (!grad_tensor || !grad_tensor->Data || grad_tensor->nelem() <= 0) {
                continue;
            }
            const std::size_t n = std::min<std::size_t>(kSampleN, static_cast<std::size_t>(grad_tensor->nelem()));
            std::vector<float> vals;
            if (!copy_tensor_sample_as_f32(*grad_tensor, n, vals)) {
                continue;
            }
            double sum_abs = 0.0;
            double sum_sq = 0.0;
            double max_abs = 0.0;
            for (float v : vals) {
                if (std::isnan(v) || std::isinf(v)) {
                    continue;
                }
                const double av = std::abs(static_cast<double>(v));
                sum_abs += av;
                sum_sq += av * av;
                if (av > max_abs) {
                    max_abs = av;
                }
            }
            if (!vals.empty()) {
                ParamSample s;
                s.name = param_name;
                s.count = vals.size();
                s.mean_abs = sum_abs / static_cast<double>(vals.size());
                s.max_abs = max_abs;
                samples.push_back(std::move(s));
                total_sum_abs += sum_abs;
                total_sum_sq += sum_sq;
                total_count += vals.size();
            }
        }
        if (!samples.empty()) {
            std::sort(samples.begin(), samples.end(),
                      [](const ParamSample& a, const ParamSample& b) { return a.mean_abs > b.mean_abs; });
            const std::size_t top_n = std::min<std::size_t>(10, samples.size());
            for (std::size_t i = 0; i < top_n; ++i) {
                const auto& s = samples[i];
                fprintf(stderr,
                        "[PARAM_GRAD_SAMPLE] name=%s mean_abs=%.6e max_abs=%.6e n=%zu\n",
                        s.name.c_str(), s.mean_abs, s.max_abs, s.count);
            }
            if (total_count > 0) {
                const double mean_abs = total_sum_abs / static_cast<double>(total_count);
                const double l2 = std::sqrt(total_sum_sq);
                fprintf(stderr,
                        "[PARAM_GRAD_SAMPLE_NORM] sample_l2=%.6e mean_abs=%.6e count=%zu\n",
                        l2, mean_abs, total_count);
            }
        }
    }

    // Free temporaries
    for (auto it = mTemps.rbegin(); it != mTemps.rend(); ++it) {
        mRunState.temp_free(*it);
    }
    mTemps.clear();

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->release_embeddings(mRunState.MainStream);
        mWeightManager->release_final_norm(mRunState.MainStream);
    }
}

void CompiledExecutor::execute_backward(const CompiledGraph& graph,
                                        NCCLCommunicator& comm,
                                        int grad_accum_steps,
                                        int micro_step,
                                        const modules::BackwardHook* hook) {
    mComm = &comm;
    mRunState.reset_simplified_gradients();
    mTemps.clear();
    mTensorMap.clear();
    mAccumulateTensors.clear();
    mCurrentLayer = -1;
    mLastRecomputeLayer = -1;
    mMicroStep = micro_step;

    // Clear activation/non-block gradients for each micro-step.
    // Compiled executor does not go through GraphExecutor's zeroing path.
    fill_zero(mRunState.non_block_gradients().d_ln_final, mRunState.MainStream);
    if (mRunState.non_block_gradients().d_embeddings.Data && !mRunState.is_lora_only_mode()) {
        fill_zero(mRunState.non_block_gradients().d_embeddings, mRunState.MainStream);
    }
    if (mConfig.NumLayers > 0) {
        fill_zero(mRunState.simplified_grads(static_cast<int>(mConfig.NumLayers) - 1).d_res_ffn,
                  mRunState.MainStream);
    }
    mRunState.zero_activation_gradients(mRunState.MainStream);

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->gather_final_norm(comm, mRunState.MainStream);
        if (mOptions.LMHeadChunks <= 1) {
            mWeightManager->gather_lm_head(comm, mRunState.MainStream);
        }
    }

    // Save stack checkpoint at start of backward - we'll restore per-layer to manage memory
    auto initial_checkpoint = mRunState.Stack.checkpoint();
    int last_layer_restored = -1;
    auto clear_shared_grads = [&](int layer_idx) {
        if (!mRunState.large_bwd_temps_on_stack()) {
            return;
        }
        if (layer_idx < 0 || layer_idx >= static_cast<int>(mConfig.NumLayers)) {
            return;
        }
        auto& grads = mRunState.simplified_grads(layer_idx);
        if (grads.d_ln2.Data) {
            fill_zero(grads.d_ln2, mRunState.MainStream);
        }
        if (grads.d_att.Data) {
            fill_zero(grads.d_att, mRunState.MainStream);
        }
        if (grads.d_ln1.Data) {
            fill_zero(grads.d_ln1, mRunState.MainStream);
        }
    };
    auto prune_stack_tensors = [&](int current_layer) {
        for (auto it = mTensorMap.begin(); it != mTensorMap.end(); ) {
            // Skip MoE expert_offsets - needed throughout backward for grouped GEMM ops
            if (mConfig.NumExperts > 0 && it->first == "moe_expert_offsets") {
                ++it;
                continue;
            }
            // Skip cross-layer gradients - these are needed by the previous layer's backward
            // Cross-layer gradients have names like "d_blocks[N].XXX" where N < current_layer
            // They flow from one layer's backward to the previous layer's backward
            if (current_layer >= 0 && it->first.rfind("d_blocks[", 0) == 0) {
                // Parse the layer index from the gradient name
                auto bracket_pos = it->first.find('[');
                auto close_pos = it->first.find(']');
                if (bracket_pos != std::string::npos && close_pos != std::string::npos) {
                    std::string layer_str = it->first.substr(bracket_pos + 1, close_pos - bracket_pos - 1);
                    try {
                        int grad_layer = std::stoi(layer_str);
                        // Preserve gradients for layers below the current one (they'll be needed)
                        if (grad_layer < current_layer) {
                            ++it;
                            continue;
                        }
                    } catch (...) {
                        // If parsing fails, skip this tensor to be safe
                        ++it;
                        continue;
                    }
                }
            }
            // Skip saved tensors for layers below current (needed for their backward)
            // Saved tensors have names like "blocks[N].XXX" where N < current_layer
            if (current_layer >= 0 && it->first.rfind("blocks[", 0) == 0) {
                auto bracket_pos = it->first.find('[');
                auto close_pos = it->first.find(']');
                if (bracket_pos != std::string::npos && close_pos != std::string::npos) {
                    std::string layer_str = it->first.substr(bracket_pos + 1, close_pos - bracket_pos - 1);
                    try {
                        int saved_layer = std::stoi(layer_str);
                        // Preserve saved tensors for layers below the current one
                        if (saved_layer < current_layer) {
                            ++it;
                            continue;
                        }
                    } catch (...) {
                        // If parsing fails, skip this tensor to be safe
                        ++it;
                        continue;
                    }
                }
            }
            if (it->second.Data && mRunState.Stack.owns(it->second.Data) &&
                !mRunState.Stack.is_live(it->second.Data)) {
                it = mTensorMap.erase(it);
            } else {
                ++it;
            }
        }
    };

    // Bind initial gradient tensors (from loss computation)
    // d_logits is stored in the output buffer after loss backward (only when lmhead_chunks == 1)
    auto& output = mRunState.non_block_activations().output;
    if (!output.Data) {
        throw std::runtime_error("CompiledExecutor: output tensor has no data (B=" +
                                std::to_string(mB) + ", T=" + std::to_string(mT) + ")");
    }

    if (mOptions.LMHeadChunks <= 1) {
        Tensor logits_view = view_tensor(output, {mB, mT, static_cast<long>(mConfig.VocabSize)});
        mTensorMap["d_logits"] = logits_view;
        // Also provide flattened version for matmul backward ops
        Tensor logits_flat = view_tensor(output, {mB * mT, static_cast<long>(mConfig.VocabSize)});
        if (logits_flat.Rank != 2) {
            throw std::runtime_error("CompiledExecutor: d_logits_flat has wrong rank=" +
                                    std::to_string(logits_flat.Rank) + " expected 2");
        }
        mTensorMap["d_logits_flat"] = logits_flat;
        // Verify the map entry
        auto& check = mTensorMap["d_logits_flat"];
        if (check.Rank != 2) {
            throw std::runtime_error("CompiledExecutor: d_logits_flat in map has wrong rank=" +
                                    std::to_string(check.Rank));
        }
    }

    // Bind gradient output buffers for final layer norm backward
    // DSL-driven: use slot registry to derive all mappings from gradient_of relationships
    Tensor& d_ln_final_buf = mRunState.non_block_gradients().d_ln_final;
    Tensor& d_embeddings_buf = mRunState.non_block_gradients().d_embeddings;

    Tensor d_ln_final_flat = view_tensor(d_ln_final_buf,
                                         {mB * mT, static_cast<long>(mConfig.HiddenSize)});

    // Helper to determine target buffer based on gradient_of field
    auto get_target_buffer = [&](const std::string& grad_of) -> Tensor* {
        // Final norm gradients (xF, ln_final, residual_final)
        if (grad_of == "xF" || grad_of == "ln_final" || grad_of == "xF_flat" ||
            grad_of == "residual_final" || grad_of == "final_residual") {
            return &d_ln_final_buf;
        }
        // Embedding output gradients (x0, encoded)
        if (grad_of == "x0" || grad_of == "encoded" || grad_of == "embeddings") {
            if (!mRunState.is_lora_only_mode()) {
                return &d_embeddings_buf;
            }
        }
        // Note: d_xN, d_residualN don't map to persistent buffers - they're computed on-the-fly
        return nullptr;
    };

    // Bind global gradient tensors - these are always needed regardless of DSL layout
    // The DSL gradient slots declare shape/dtype but the actual buffers come from RunState
    mTensorMap["d_xF_flat"] = d_ln_final_flat;
    mTensorMap["d_xF"] = d_ln_final_buf;
    mTensorMap["d_ln_final"] = d_ln_final_buf;
    mTensorMap["d_ln_final_flat"] = d_ln_final_flat;

    if (!mRunState.is_lora_only_mode()) {
        mTensorMap["d_encoded"] = d_embeddings_buf;
        mTensorMap["d_x0"] = d_embeddings_buf;
    }

    static bool emb_grad_initial_logged = false;
    if (!emb_grad_initial_logged && d_embeddings_buf.Data) {
        if (tensor_sample_has_nan_or_inf(d_embeddings_buf, 3)) {
            fprintf(stderr, "[EMB_DOUT_NAN_INIT] micro_step=%d ptr=%p\n", micro_step, d_embeddings_buf.Data);
            log_tensor_stats_ex("EMB_DOUT_INIT_NAN", -1, "d_embeddings", d_embeddings_buf, 4096, true);
        }
        emb_grad_initial_logged = true;
    }

    // DSL-driven binding for any additional gradient slots declared in the Python model
    if (mSlotRegistry && mSlotRegistry->has_dsl_layout()) {
        mSlotRegistry->for_each([&](const std::string& slot_name,
                                    const TensorSlotRegistry::SlotEntry& entry) {
            if (entry.scope != ActivationScope::GlobalGradient) return;
            // Skip if already bound above
            if (mTensorMap.find(slot_name) != mTensorMap.end()) return;

            Tensor* target_buf = get_target_buffer(entry.gradient_of);
            if (target_buf && target_buf->Data) {
                mTensorMap[slot_name] = *target_buf;
            }
        });
    }

    // Ensure global block outputs (xN/residualN) map to the last block's gradients.
    // These gradients must survive layer-boundary stack restores in recompute mode.
    if (mConfig.NumLayers > 0) {
        const int last_layer = static_cast<int>(mConfig.NumLayers) - 1;
        auto& last_grads = mRunState.simplified_grads(last_layer);
        if (last_grads.d_mlp_down.Data) {
            mTensorMap["d_xN"] = last_grads.d_mlp_down;
        }
        if (last_grads.d_res_att.Data) {
            mTensorMap["d_residualN"] = last_grads.d_res_att;
        }

        // Heuristic aliasing for non-inlined StackedBlocks outputs (e.g., "StackedBlocks_4").
        if (mSaved) {
            std::vector<std::pair<int, std::string>> stacked;
            stacked.reserve(2);
            for (const auto& kv : *mSaved) {
                const std::string& name = kv.first;
                if (name.rfind("StackedBlocks_", 0) != 0) {
                    continue;
                }
                int idx = -1;
                const char* s = name.c_str() + std::strlen("StackedBlocks_");
                if (*s) {
                    char* end = nullptr;
                    long parsed = std::strtol(s, &end, 10);
                    if (end != s) {
                        idx = static_cast<int>(parsed);
                    }
                }
                if (idx >= 0) {
                    stacked.emplace_back(idx, name);
                }
            }
            if (!stacked.empty()) {
                std::sort(stacked.begin(), stacked.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
                if (stacked.size() == 1) {
                    if (last_grads.d_res_att.Data) {
                        mTensorMap["d_" + stacked[0].second] = last_grads.d_res_att;
                    }
                } else {
                    if (last_grads.d_mlp_down.Data) {
                        mTensorMap["d_" + stacked[0].second] = last_grads.d_mlp_down;
                    }
                    if (last_grads.d_res_att.Data) {
                        mTensorMap["d_" + stacked[1].second] = last_grads.d_res_att;
                    }
                }
            }
        }
    }

    // Bind autodiff-generated gradient names (d_embed_1, etc.) from forward embedding outputs
    // These are dynamically generated and not in the DSL layout
    if (!mRunState.is_lora_only_mode()) {
        for (const auto& emb_out : mEmbeddingOutputs) {
            std::string grad_name = "d_" + emb_out;
            mTensorMap[grad_name] = d_embeddings_buf;
        }
    }

    // Restore MoE expert_offsets from persistent CPU storage
    // This is needed by grouped GEMM backward ops for proper token routing
    if (mConfig.NumExperts > 0 && !mMoEExpertOffsetsData.empty()) {
        // Allocate PERSISTENT GPU buffer for expert_offsets (not stack-allocated)
        // This ensures the memory won't be invalidated by stack restores or temp_free calls
        const int num_elements = static_cast<int>(mMoEExpertOffsetsData.size());
        const size_t needed_bytes = num_elements * sizeof(int);

        // Allocate or resize GPU buffer if needed
        if (mMoEExpertOffsetsGPU == nullptr || mMoEExpertOffsetsGPUSize < needed_bytes) {
            if (mMoEExpertOffsetsGPU) {
                CUDA_CHECK(cudaFree(mMoEExpertOffsetsGPU));
            }
            CUDA_CHECK(cudaMalloc(&mMoEExpertOffsetsGPU, needed_bytes));
            mMoEExpertOffsetsGPUSize = needed_bytes;
        }

        // Copy data from CPU to GPU
        CUDA_CHECK(cudaMemcpyAsync(mMoEExpertOffsetsGPU, mMoEExpertOffsetsData.data(),
                                   needed_bytes, cudaMemcpyHostToDevice, mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));

        // Create tensor wrapper pointing to persistent buffer
        Tensor expert_offsets;
        expert_offsets.DType = ETensorDType::INT32;
        expert_offsets.Rank = 1;
        expert_offsets.Sizes[0] = num_elements;
        expert_offsets.Data = static_cast<std::byte*>(mMoEExpertOffsetsGPU);

        mTensorMap["moe_expert_offsets"] = expert_offsets;
        // Note: NOT adding to mTemps since this is persistent memory managed separately
    }

    // Also bind standard inputs that backward ops may reference
    mTensorMap["token_ids"] = mRunState.Inputs;
    mTensorMap["position_ids"] = mRunState.PositionIDs;

    // Build the set of gradients that require accumulation (not the first micro-step).
    // Also bind parameter gradient tensors to mTensorMap so they're used instead of temporaries.
    // This mirrors the logic in graph_executor_backward.cpp (bind_param_grad).
    for (const auto& param_name : mGrads.param_names()) {
        if (param_name.find("rope_freqs") != std::string::npos) {
            continue;
        }
        bool accumulate = false;
        Tensor* grad_tensor = mGrads.get_param_grad(param_name, accumulate);
        if (grad_tensor && grad_tensor->Data) {
            std::string grad_name = "d_" + param_name;
            mTensorMap[grad_name] = *grad_tensor;
            if (accumulate) {
                mAccumulateTensors.insert(grad_name);
            }
        }
    }

    auto is_grad_ref = [](const TensorRef& ref) -> bool {
        if (!ref.name.empty() && ref.name.size() > 2 && ref.name[0] == 'd' && ref.name[1] == '_') {
            return true;
        }
        switch (ref.slot) {
            case TensorSlot::BlockDLN1:
            case TensorSlot::BlockDQKV:
            case TensorSlot::BlockDAtt:
            case TensorSlot::BlockDSwiGLU:
            case TensorSlot::BlockDMLPUp:
            case TensorSlot::BlockDMLPDown:
            case TensorSlot::BlockDLN2:
            case TensorSlot::BlockDResAtt:
            case TensorSlot::BlockDResFFN:
            case TensorSlot::DLoss:
                return true;
            default:
                return false;
        }
    };

    auto ref_layer_idx = [&](const TensorRef& ref) -> int {
        if (ref.layer_idx >= 0) {
            return ref.layer_idx;
        }
        if (ref.name.empty()) {
            return -1;
        }
        std::string_view name = ref.name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return layer_idx;
        }
        return -1;
    };

    auto ref_layer_idx_any = [&](const TensorRef& ref) -> int {
        if (ref.layer_idx >= 0) {
            return ref.layer_idx;
        }
        if (ref.name.empty()) {
            return -1;
        }
        std::string_view name = ref.name;
        if (name.rfind("d_", 0) == 0) {
            name.remove_prefix(2);
        }
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return layer_idx;
        }
        return -1;
    };

    auto op_layer_idx = [&](const CompiledOp& op) -> int {
        int detected_non_grad = -1;
        for (const auto& ref : op.inputs) {
            if (!is_grad_ref(ref)) {
                const int layer_idx = ref_layer_idx(ref);
                if (layer_idx >= 0) {
                    detected_non_grad = std::max(detected_non_grad, layer_idx);
                }
            }
        }
        for (const auto& ref : op.outputs) {
            if (!is_grad_ref(ref)) {
                const int layer_idx = ref_layer_idx(ref);
                if (layer_idx >= 0) {
                    detected_non_grad = std::max(detected_non_grad, layer_idx);
                }
            }
        }
        return (detected_non_grad >= 0) ? detected_non_grad : -1;
    };

    auto op_layer_idx_any = [&](const CompiledOp& op) -> int {
        int detected_any = -1;
        for (const auto& ref : op.inputs) {
            const int layer_idx = ref_layer_idx_any(ref);
            if (layer_idx >= 0) {
                detected_any = std::max(detected_any, layer_idx);
            }
        }
        for (const auto& ref : op.outputs) {
            const int layer_idx = ref_layer_idx_any(ref);
            if (layer_idx >= 0) {
                detected_any = std::max(detected_any, layer_idx);
            }
        }
        if (op.attrs.layer_idx >= 0) {
            detected_any = std::max(detected_any, op.attrs.layer_idx);
        }
        return detected_any;
    };

    const bool skip_logits_grad = (mOptions.LMHeadChunks > 1);
    auto is_logits_grad_name = [](const std::string& name) {
        return name == "d_logits" || name == "d_logits_flat";
    };
    auto is_logits_grad_op = [&](const CompiledOp& op) {
        for (const auto& ref : op.inputs) {
            if (is_logits_grad_name(ref.name)) return true;
        }
        for (const auto& ref : op.outputs) {
            if (is_logits_grad_name(ref.name)) return true;
        }
        return false;
    };

    const int num_layers = static_cast<int>(mConfig.NumLayers);
    const bool assert_recompute_a = (std::getenv("SUROGATE_ASSERT_RECOMPUTE_A") != nullptr);
    if (assert_recompute_a) {
        if (mRecomputeSamples.size() != static_cast<std::size_t>(num_layers)) {
            mRecomputeSamples.assign(static_cast<std::size_t>(num_layers), RecomputeSample{});
        }
        for (auto& sample : mRecomputeSamples) {
            sample.micro_step = mMicroStep;
            sample.ln1_valid = false;
            sample.ln2_valid = false;
        }
    }
    auto record_recompute_sample = [&](int layer_idx) {
        if (!assert_recompute_a) return;
        if (layer_idx < 0 || layer_idx >= num_layers) return;
        auto& sample = mRecomputeSamples[static_cast<std::size_t>(layer_idx)];
        sample.micro_step = mMicroStep;
        sample.ln1_valid = false;
        sample.ln2_valid = false;
        const long sample_token = 3;
        std::vector<float> vals;
        auto& acts = mRunState.simplified_acts(layer_idx);
        if (acts.ln1.Data && copy_tensor_token_sample_as_f32(acts.ln1, sample_token, 4, vals)) {
            for (int i = 0; i < 4; ++i) sample.ln1[static_cast<std::size_t>(i)] = vals[i];
            sample.ln1_valid = true;
        }
        if (acts.ln2.Data && copy_tensor_token_sample_as_f32(acts.ln2, sample_token, 4, vals)) {
            for (int i = 0; i < 4; ++i) sample.ln2[static_cast<std::size_t>(i)] = vals[i];
            sample.ln2_valid = true;
        }
    };
    std::vector<std::size_t> layer_start_indices(num_layers, SIZE_MAX);
    std::vector<bool> layer_seen_any(num_layers, false);
    for (const auto& op : graph.ops) {
        if (op.layer_start >= 0 && op.layer_start < num_layers) {
            layer_start_indices[op.layer_start] = &op - graph.ops.data();
        }
    }

    for (std::size_t idx = 0; idx < graph.ops.size(); ++idx) {
        const auto& op = graph.ops[idx];
        const int op_layer_any = op_layer_idx_any(op);
        if (skip_logits_grad && is_logits_grad_op(op)) {
            continue;
        }

        // DEBUG: Trace flash attention backward op order and layer indices.
        static int flash_bwd_trace = 0;
        if (op.type == CompiledOpType::FlashAttentionBackward && flash_bwd_trace < 12) {
            int layer_idx_dbg = -1;
            std::string field_dbg;
            if (!op.inputs.empty()) {
                parse_block_param(op.inputs[1].name, layer_idx_dbg, field_dbg);
            }
            fprintf(stderr,
                    "[EXEC_FLASH_BWD] op=%zu layer_idx=%d in_out=%s in_lse=%s in_qkv=%s out=%s\n",
                    idx,
                    layer_idx_dbg,
                    op.inputs.size() > 1 ? op.inputs[1].name.c_str() : "<none>",
                    op.inputs.size() > 2 ? op.inputs[2].name.c_str() : "<none>",
                    op.inputs.size() > 3 ? op.inputs[3].name.c_str() : "<none>",
                    op.outputs.size() > 0 ? op.outputs[0].name.c_str() : "<none>");
            flash_bwd_trace++;
        }

        if (op.layer_start >= 0) {
            handle_layer_start(op.layer_start);
            // Optional trace: watch a specific layer's res_ffn buffer to catch corruption.
            if (std::getenv("SUROGATE_TRACE_RESFFN_LAYER")) {
                const int target_layer = env_int("SUROGATE_TRACE_RESFFN_LAYER", 27);
                static int resffn_trace_count = 0;
                if (target_layer >= 0 && target_layer < num_layers && resffn_trace_count < 64) {
                    Tensor& res = mRunState.get_residual(target_layer, mRunState.MainStream);
                    fprintf(stderr, "[BWD_RESFFN_TRACE] at_layer=%d target=%d res_ptr=%p\n",
                            op.layer_start, target_layer, res.Data);
                    std::vector<float> vals;
                    const std::size_t total = static_cast<std::size_t>(res.nelem());
                    const std::size_t n = std::min<std::size_t>(4096, total);
                    if (copy_tensor_sample_as_f32(res, n, vals)) {
                        std::size_t nan = 0;
                        std::size_t inf = 0;
                        float min_v = std::numeric_limits<float>::infinity();
                        float max_v = -std::numeric_limits<float>::infinity();
                        double sum_abs = 0.0;
                        for (float v : vals) {
                            if (std::isnan(v)) {
                                nan++;
                                continue;
                            }
                            if (std::isinf(v)) {
                                inf++;
                                continue;
                            }
                            min_v = std::min(min_v, v);
                            max_v = std::max(max_v, v);
                            sum_abs += std::fabs(v);
                        }
                        const double mean_abs = n ? (sum_abs / static_cast<double>(n)) : 0.0;
                        fprintf(stderr,
                                "[BWD_RESFFN_TRACE_STATS] target=%d n=%zu total=%zu nan=%zu inf=%zu min=%.6f max=%.6f mean_abs=%.6f\n",
                                target_layer, n, total, nan, inf,
                                std::isfinite(min_v) ? min_v : 0.0f,
                                std::isfinite(max_v) ? max_v : 0.0f,
                                static_cast<float>(mean_abs));
                    }
                    resffn_trace_count++;
                }
            }
            if (mRecomputeEnabled && mRecomputeFn) {
                const int layer_idx = op.layer_start;
                if (layer_idx >= 0 && layer_idx != mLastRecomputeLayer) {
                if (layer_idx < num_layers && !layer_seen_any[static_cast<std::size_t>(layer_idx)]) {
                    clear_shared_grads(layer_idx);
                    layer_seen_any[static_cast<std::size_t>(layer_idx)] = true;
                }
                // DEBUG: Check MoE expert gate_up weights before recompute at layer start.
                static int moe_w_pre_recompute_start_trace = 0;
                if (layer_idx == 2 && moe_w_pre_recompute_start_trace < 1) {
                    const std::string w_name = "blocks[2].experts_gate_up";
                    if (mWeights.has(w_name)) {
                        Tensor& w = mWeights.get(w_name);
                        float w_min = 0.0f, w_max = 0.0f;
                        const bool w_nan = tensor_row_has_nan_or_inf(w, 122, &w_min, &w_max);
                        fprintf(stderr,
                                "[MOE_W_PRE_RECOMP_START] layer=%d name=%s ptr=%p device=%d nan=%d min=%.6f max=%.6f\n",
                                layer_idx, w_name.c_str(),
                                w.Data, w.Device,
                                w_nan ? 1 : 0, w_min, w_max);
                    } else {
                        fprintf(stderr, "[MOE_W_PRE_RECOMP_START] layer=%d name=%s missing\n",
                                layer_idx, w_name.c_str());
                    }
                    moe_w_pre_recompute_start_trace++;
                }
                mRecomputeFn(layer_idx, mB, mT, mRecomputeUseGraphs);
                record_recompute_sample(layer_idx);
                mLastRecomputeLayer = layer_idx;
            }
        }
        }

        if (mRecomputeEnabled && mRecomputeFn) {
            const int layer_idx = op_layer_idx(op);
            const int layer_idx_any = op_layer_idx_any(op);
            // Always recompute when switching layers. This is critical because:
            // - Shared buffers (ln1, ln2, qkv, mlp_up, swiglu) contain only ONE layer's data
            // - If the backward graph interleaves ops from different layers, we MUST
            //   recompute to ensure the correct layer's data is in the shared buffers
            // - The old check (missing_start || op_before_start) would skip recomputation
            //   for layer N's late ops if we had already visited layer N earlier, causing
            //   those ops to read stale data from whatever layer was recomputed last
            static int debug_op_count = 0;
            if (debug_op_count < 10) {
                fprintf(stderr, "[DEBUG_LAYER] op=%d layer_idx=%d layer_idx_any=%d mLastRecompute=%d op_type=%s\n",
                        static_cast<int>(idx), layer_idx, layer_idx_any, mLastRecomputeLayer, op_type_to_string(op.type));
                debug_op_count++;
            }
            // Use layer_idx_any as fallback when layer_idx is -1
            const int effective_layer_idx = (layer_idx >= 0) ? layer_idx : layer_idx_any;
            if (effective_layer_idx >= 0 && effective_layer_idx != mLastRecomputeLayer) {
                if (effective_layer_idx < num_layers && !layer_seen_any[static_cast<std::size_t>(effective_layer_idx)]) {
                    clear_shared_grads(effective_layer_idx);
                    layer_seen_any[static_cast<std::size_t>(effective_layer_idx)] = true;
                }
                // DEBUG: Check MoE expert gate_up weights before recompute for layer 2.
                static int moe_w_pre_recompute_trace = 0;
                if (effective_layer_idx == 2 && moe_w_pre_recompute_trace < 1) {
                    const std::string w_name = "blocks[2].experts_gate_up";
                    if (mWeights.has(w_name)) {
                        Tensor& w = mWeights.get(w_name);
                        float w_min = 0.0f, w_max = 0.0f;
                        const bool w_nan = tensor_row_has_nan_or_inf(w, 122, &w_min, &w_max);
                        fprintf(stderr,
                                "[MOE_W_PRE_RECOMP] layer=%d name=%s ptr=%p device=%d nan=%d min=%.6f max=%.6f\n",
                                effective_layer_idx, w_name.c_str(),
                                w.Data, w.Device,
                                w_nan ? 1 : 0, w_min, w_max);
                    } else {
                        fprintf(stderr, "[MOE_W_PRE_RECOMP] layer=%d name=%s missing\n",
                                effective_layer_idx, w_name.c_str());
                    }
                    moe_w_pre_recompute_trace++;
                }
                static int recompute_call_count = 0;
                if (recompute_call_count < 5) {
                    fprintf(stderr, "[CALLING_RECOMPUTE] layer_idx=%d (effective=%d) num_layers=%d\n", layer_idx, effective_layer_idx, num_layers);
                    recompute_call_count++;
                }
                mRecomputeFn(effective_layer_idx, mB, mT, mRecomputeUseGraphs);
                record_recompute_sample(effective_layer_idx);
                mLastRecomputeLayer = effective_layer_idx;
            }
        }

        try {
            switch (op.type) {
                // Explicit backward ops
                case CompiledOpType::ViewBackward:
                    dispatch_view_backward(op);
                    break;
                case CompiledOpType::AddBackward:
                    dispatch_add_backward(op);
                    break;
                case CompiledOpType::CrossEntropyLossBackward:
                    dispatch_cross_entropy_loss_backward(op);
                    break;
                case CompiledOpType::FusedLMHeadLossBackward:
                    dispatch_fused_lm_head_loss_backward(op);
                    break;
                case CompiledOpType::MatmulBackward:
                    dispatch_matmul_backward(op, hook);
                    // After the first matmul_backward (LM-head backward), free the output tensor
                    // to reclaim ~1.2GB of stack memory. The d_logits data has been consumed.
                    if (idx == 1) {
                        mRunState.temp_free(mRunState.non_block_activations().output);
                        mTemps.clear();
                        // Update initial_checkpoint to reflect the freed output tensor
                        // This prevents subsequent checkpoint restores from re-allocating it
                        initial_checkpoint = mRunState.Stack.checkpoint();
                    }
                    break;
                case CompiledOpType::BiasAddBackward:
                    dispatch_bias_add_backward(op);
                    break;
                case CompiledOpType::SwiGLUBackward:
                    dispatch_swiglu_backward(op);
                    break;
                case CompiledOpType::SiluBackward:
                    dispatch_silu_backward(op);
                    break;
                case CompiledOpType::MulBackward:
                    dispatch_mul_backward(op);
                    break;
                case CompiledOpType::MatmulSwiGLUBackward:
                    dispatch_matmul_swiglu_backward(op, hook);
                    break;
                case CompiledOpType::RoPEBackward:
                    dispatch_rope_backward(op);
                    break;
                case CompiledOpType::QKVQKNormRoPEBackward:
                    dispatch_qkv_qk_norm_rope_backward(op);
                    break;
                case CompiledOpType::FlashAttentionBackward:
                    dispatch_flash_attention_backward(op);
                    break;
                case CompiledOpType::ZerosBackward:
                    dispatch_zeros_backward(op);
                    break;
                case CompiledOpType::FusedResidualRMSNormBackward:
                    dispatch_fused_residual_rmsnorm_backward(op);
                    break;
                case CompiledOpType::EmbeddingBackward:
                    dispatch_embedding_backward(op);
                    break;

                // Forward ops that appear in backward graph (autodiff generates these)
                // View/reshape is the same operation in forward and backward - just reshapes gradient
                case CompiledOpType::View:
                    dispatch_view_backward(op);
                    break;
                // "add" ops in the backward graph are gradient-accumulation nodes,
                // so we must execute them as forward add (sum inputs), not add-backward.
                case CompiledOpType::Add:
                    dispatch_add(op);
                    break;
                // Zeros in backward is a no-op
                case CompiledOpType::Zeros:
                    dispatch_zeros_backward(op);
                    break;

                // MoE backward operations
                case CompiledOpType::MoESoftmaxBackward:
                    dispatch_moe_softmax_backward(op);
                    break;
                case CompiledOpType::MoESigmoidBackward:
                    dispatch_moe_sigmoid_backward(op);
                    break;
                case CompiledOpType::MoETopKBackward:
                    dispatch_moe_topk_backward(op);
                    break;
                case CompiledOpType::MoEPermuteBackward:
                    dispatch_moe_permute_backward(op);
                    break;
                case CompiledOpType::MoEGroupedGemmGateUpBackward:
                    dispatch_moe_grouped_gemm_gate_up_backward(op);
                    break;
                case CompiledOpType::MoEGroupedGemmDownBackward:
                    dispatch_moe_grouped_gemm_down_backward(op);
                    break;
                case CompiledOpType::MoEUnpermuteBackward:
                    dispatch_moe_unpermute_backward(op);
                    break;

                // MoE forward ops that may appear in backward graph
                case CompiledOpType::MoESoftmax:
                case CompiledOpType::MoESigmoid:
                case CompiledOpType::MoETopK:
                case CompiledOpType::MoEPermute:
                case CompiledOpType::MoEGroupedGemmGateUp:
                case CompiledOpType::MoEGroupedGemmDown:
                case CompiledOpType::MoEUnpermute:
                case CompiledOpType::Silu:
                case CompiledOpType::Mul:
                    // These forward MoE ops may appear in backward graph due to autodiff
                    throw std::runtime_error("CompiledExecutor: MoE forward op in backward graph not yet supported");

                default: {
                    std::ostringstream oss;
                    oss << "CompiledExecutor: unsupported backward op type at idx " << idx
                        << " (type=" << op_type_to_string(op.type) << ", id=" << op.op_id << ")";
                    throw std::runtime_error(oss.str());
                }
            }

            // Optional: detect the first op that corrupts a target res_ffn buffer.
            if (std::getenv("SUROGATE_TRACE_RESFFN_WATCH")) {
                const int target_layer = env_int("SUROGATE_TRACE_RESFFN_LAYER", 27);
                const int watch_layer = env_int("SUROGATE_TRACE_RESFFN_WATCH_LAYER", target_layer);
                static bool res_ok = true;
                static int res_watch_count = 0;
                if (op.layer_start >= 0 && op.layer_start == watch_layer) {
                    res_ok = true;
                    res_watch_count = 0;
                }
                const int op_layer = op_layer_idx_any(op);
                if (res_ok && op_layer == watch_layer && target_layer >= 0 && target_layer < num_layers) {
                    Tensor& res = mRunState.get_residual(target_layer, mRunState.MainStream);
                    std::vector<float> vals;
                    const std::size_t total = static_cast<std::size_t>(res.nelem());
                    const std::size_t n = std::min<std::size_t>(4096, total);
                    bool corrupt = false;
                    float max_abs = 0.0f;
                    std::size_t nan = 0;
                    if (copy_tensor_sample_as_f32(res, n, vals)) {
                        for (float v : vals) {
                            if (std::isnan(v) || std::isinf(v)) {
                                nan++;
                                corrupt = true;
                                continue;
                            }
                            max_abs = std::max(max_abs, std::fabs(v));
                        }
                        const float thresh = env_float("SUROGATE_TRACE_RESFFN_THRESH", 1e6f);
                        if (max_abs > thresh) {
                            corrupt = true;
                        }
                    }
                    if (corrupt && res_watch_count < 4) {
                        fprintf(stderr,
                                "[RESFFN_CORRUPT] watch_layer=%d target=%d op=%s op_id=%s max_abs=%.3e nan=%zu\n",
                                watch_layer, target_layer,
                                op_type_to_string(op.type), op.op_id.c_str(),
                                static_cast<double>(max_abs), nan);
                        res_watch_count++;
                        res_ok = false;
                    }
                }
            }

            // Trace first writers for layer-8 residual gradients to pinpoint the spike source.
            static int res_writer_trace_enabled = -1;
            if (res_writer_trace_enabled < 0) {
                res_writer_trace_enabled = (std::getenv("SUROGATE_TRACE_RES_WRITERS") != nullptr) ? 1 : 0;
            }
            if (res_writer_trace_enabled) {
                struct ResWriterState {
                    int micro_step = -1;
                    int res_att_count = 0;
                    int res_ffn_count = 0;
                };
                static ResWriterState res_state;
                if (res_state.micro_step != mMicroStep) {
                    res_state = {};
                    res_state.micro_step = mMicroStep;
                }
                auto log_res_writer = [&](const TensorRef& ref, int& counter) {
                    if (counter >= 12) {
                        return;
                    }
                    Tensor& t = resolve_tensor(ref);
                    fprintf(stderr,
                            "[RES_WRITER] micro_step=%d op_idx=%zu type=%s id=%s name=%s\n",
                            mMicroStep,
                            idx,
                            op_type_to_string(op.type),
                            op.op_id.c_str(),
                            ref.name.c_str());
                    log_tensor_mag_unbounded("RES_WRITER_TENSOR", op_layer_any, ref.name, t, 4096);
                    counter++;
                };
                for (const auto& out_ref : op.outputs) {
                    const std::string base = strip_ssa_suffix(out_ref.name);
                    if (base == "d_blocks[8].res_att") {
                        log_res_writer(out_ref, res_state.res_att_count);
                    } else if (base == "d_blocks[8].res_ffn") {
                        log_res_writer(out_ref, res_state.res_ffn_count);
                    }
                }
            }

            // Track when key gradients first spike or go NaN.
            static std::unordered_map<std::string, int> watch_counts;
            auto log_watch = [&](const TensorRef& ref) {
                if (ref.name.empty()) return;
                const std::string base = strip_ssa_suffix(ref.name);
                if (base != "d_blocks[0].ln1" &&
                    base != "d_blocks[0].ln2" &&
                    base != "d_blocks[0].res_ffn" &&
                    base != "d_blocks[0].mlp_up" &&
                    base != "d_blocks[0].swiglu") {
                    return;
                }
                int& count = watch_counts[base];
                if (count >= 8) {
                    return;
                }
                Tensor& t = resolve_tensor(ref);
                fprintf(stderr,
                        "[BWD_WATCH] op_idx=%zu type=%s id=%s name=%s\n",
                        idx, op_type_to_string(op.type), op.op_id.c_str(), ref.name.c_str());
                log_tensor_stats_ex("BWD_WATCH_OUT", op_layer_any, ref.name, t, 4096, true);
                count++;
            };
            for (const auto& out_ref : op.outputs) {
                log_watch(out_ref);
            }

            static bool emb_nan_logged = false;
            if (!emb_nan_logged && d_embeddings_buf.Data && tensor_sample_has_nan_or_inf(d_embeddings_buf, 3)) {
                fprintf(stderr,
                        "[EMB_DOUT_NAN_AFTER] micro_step=%d op_idx=%zu type=%s id=%s\n",
                        micro_step, idx, op_type_to_string(op.type), op.op_id.c_str());
                log_tensor_stats_ex("EMB_DOUT_NAN_AFTER", -1, "d_embeddings", d_embeddings_buf, 4096, true);
                const long hidden = d_embeddings_buf.Rank > 2 ? d_embeddings_buf.Sizes[2] : 0;
                if (hidden > 0) {
                    const std::size_t off = static_cast<std::size_t>(3) * static_cast<std::size_t>(hidden);
                    log_tensor_sample_stats("EMB_DOUT_NAN_ROW3", d_embeddings_buf, off,
                                            static_cast<std::size_t>(hidden));
                }
                emb_nan_logged = true;
            }

            // Memory management - restore stack checkpoint periodically to free temporaries
            // This prevents memory accumulation during backward pass
            // Option 1: At layer boundaries if annotated
            // TEMPORARILY DISABLED for MoE models due to tensor corruption issues
            // TODO: Fix proper tensor lifetime tracking for MoE backward
            if (mConfig.NumExperts == 0 && op.layer_end >= 0 && op.layer_end != last_layer_restored) {
                // Restore stack and clear temps
                mRunState.Stack.restore(initial_checkpoint);
                mTemps.clear();
                prune_stack_tensors(op.layer_end);
                // Note: cudnn_workspace is persistently allocated, no need to clear
                // Clear stack-allocated tensor pointers in simplified_acts/grads for this layer.
                // These pointers become stale after checkpoint restore.
                if (mRunState.ffn_temps_on_stack()) {
                    auto& acts = mRunState.simplified_acts(op.layer_end);
                    acts.mlp_up.Data = nullptr;
                    acts.swiglu.Data = nullptr;
                }
                if (mRunState.large_bwd_temps_on_stack()) {
                    auto& grads_to_clear = mRunState.simplified_grads(op.layer_end);
                    grads_to_clear.d_qkv.Data = nullptr;
                    grads_to_clear.d_mlp_up.Data = nullptr;
                    grads_to_clear.d_swiglu.Data = nullptr;
                }
                last_layer_restored = op.layer_end;
            }
            // Option 2: Every N ops as fallback (catches non-annotated layers)
            // NOTE: When recompute is disabled, we cannot aggressively prune tensors because
            // the backward graph may reference intermediate tensors (like d_blocks[N].view_K)
            // that were produced earlier but are still needed. The stack restore + prune
            // would remove these tensors from mTensorMap, causing "tensor not found" errors.
            // For now, skip periodic cleanup when recompute is disabled to preserve correctness.
            // Memory usage will be higher but the backward pass will complete successfully.
            // TODO: Implement proper tensor lifetime tracking to enable safe pruning.
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "CompiledExecutor backward op " << idx << " (type=" << op_type_to_string(op.type)
                << ", id=" << op.op_id << "): " << e.what();
            // Add inputs/outputs for debugging
            oss << "\n  inputs: [";
            for (size_t i = 0; i < op.inputs.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << op.inputs[i].name << "(slot=" << static_cast<int>(op.inputs[i].slot) << ")";
            }
            oss << "]";
            oss << "\n  outputs: [";
            for (size_t i = 0; i < op.outputs.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << op.outputs[i].name << "(slot=" << static_cast<int>(op.outputs[i].slot) << ")";
            }
            oss << "]";
            throw std::runtime_error(oss.str());
        }

    }

    // Final cleanup - pass -1 to allow full pruning (backward complete)
    mRunState.Stack.restore(initial_checkpoint);
    prune_stack_tensors(-1);
    mTemps.clear();

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->release_final_norm(mRunState.MainStream);
        if (mOptions.LMHeadChunks <= 1) {
            mWeightManager->release_lm_head(mRunState.MainStream);
        }
    }
}

}  // namespace dsl
