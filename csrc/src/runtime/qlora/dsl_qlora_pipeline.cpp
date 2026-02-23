// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DslQLoRAPipeline: Unified weight loading + quantization pipeline.

#include "runtime/qlora/dsl_qlora_pipeline.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <cuda_bf16.h>
#include <fmt/format.h>

#include "config/pretrained_config.h"
#include "kernels/kernels.h"
#include "runtime/dsl/dsl_weight_loader.h"
#include "runtime/qlora/adapter_merger.h"
#include "runtime/qlora/fp4_block_quantized_tensor.h"
#include "utilities/allocator.h"
#include "utilities/safetensors.h"
#include "utilities/utils.h"

namespace qlora {

namespace {

/// Compute the total number of elements for a weight spec.
/// Uses shape if available, otherwise M * max(K, 1).
size_t spec_num_elements(const WeightLoadSpec& spec) {
    if (!spec.shape.empty()) {
        size_t elem = 1;
        for (long d : spec.shape) elem *= static_cast<size_t>(d);
        return elem;
    }
    return static_cast<size_t>(spec.M) * std::max(spec.K, 1);
}

/// Check if a weight spec represents a 3D (expert-batched) weight.
bool is_expert_weight(const WeightLoadSpec& spec) {
    return spec.shape.size() >= 3;
}

const SafeTensorEntry* try_find_entry(const SafeTensorsReader& reader, std::string_view name) {
    for (const auto& entry : reader.entries()) {
        if (entry.name() == name) {
            return &entry;
        }
    }
    return nullptr;
}

/// Read a safetensors entry into a (possibly padded) target tensor.
/// FP4 scale storage is row-padded to 128-byte alignment, so the allocated
/// tensor may have more elements than the HF entry. This helper reads the
/// actual HF elements and zero-pads the remainder.
void read_raw_padded(const SafeTensorEntry& entry, Tensor& target,
                     cudaStream_t stream) {
    long hf_nelem = 1;
    for (auto d : entry.shape()) hf_nelem *= d;
    long target_nelem = target.nelem();

    if (hf_nelem < target_nelem) {
        // Zero the full buffer, then read HF data into a view of the first hf_nelem elements
        CUDA_CHECK(cudaMemsetAsync(target.Data, 0, target.bytes(), stream));
        Tensor view = Tensor::from_pointer(
            target.Data, target.Device, target.DType,
            std::vector<long>(entry.shape().begin(), entry.shape().end()));
        entry.read_raw(view, 0, hf_nelem, true);
    } else {
        entry.read_raw(target, 0, target_nelem, true);
    }
}

/// Temporary GPU allocation for loading pre-quantized expert weights
/// that will be relocated to pinned host memory for offloading.
/// Uses raw cudaMalloc (not TensorAllocator) so we can free after relocation.
struct GpuTempAlloc {
    void* data = nullptr;
    void* scales = nullptr;
    void* meta = nullptr;

    void free_all() {
        if (data) { cudaFree(data); data = nullptr; }
        if (scales) { cudaFree(scales); scales = nullptr; }
        if (meta) { cudaFree(meta); meta = nullptr; }
    }
};

/// Check if an expert weight spec should use CPU offloading.
bool should_offload_expert(const WeightLoadSpec& spec,
                           const DslQLoRAPipelineConfig& config) {
    return spec.offload_group >= 0 &&
           config.weight_manager_config.enable_offloading &&
           config.weight_manager_config.offload_config.max_resident_groups > 0;
}

/// Allocate QuantizedTensor storage on GPU using raw cudaMalloc.
/// Used for pre-quantized expert weights that need GPU for loading/swizzling
/// before being relocated to pinned host memory for offloading.
/// The returned GpuTempAlloc must be freed via free_all() after relocation.
GpuTempAlloc allocate_qt_gpu_temp(
    int M, int K, QuantizedTensor& qt, const QuantizerConfig& qconfig) {
    GpuTempAlloc alloc;
    qt.M = M;
    qt.K = K;
    qt.global_scale = 1.0f;
    qt.double_quant = false;

    if (qconfig.format == QuantFormat::FP4_BLOCK_2D) {
        using C = modules::FP4BlockScaleConfig;
        qt.format = QuantFormat::FP4_BLOCK_2D;
        qt.block_size = C::BLOCK_SIZE;

        const size_t data_bytes = C::packed_data_bytes(M, K);
        CUDA_CHECK(cudaMalloc(&alloc.data, data_bytes));
        qt.data = Tensor::from_pointer(
            static_cast<std::byte*>(alloc.data), 0, ETensorDType::BYTE,
            std::vector<long>{static_cast<long>(data_bytes)});

        auto [sr, sc] = C::scale_dims(M, K);
        const size_t scales_bytes =
            static_cast<size_t>(sr) * sc * sizeof(__nv_fp8_e4m3);
        CUDA_CHECK(cudaMalloc(&alloc.scales, scales_bytes));
        qt.scales = Tensor::from_pointer(
            static_cast<std::byte*>(alloc.scales), 0, ETensorDType::FP8_E4M3,
            std::vector<long>{static_cast<long>(sr), static_cast<long>(sc)});

        CUDA_CHECK(cudaMalloc(&alloc.meta, sizeof(float)));
        qt.meta = Tensor::from_pointer(
            static_cast<std::byte*>(alloc.meta), 0, ETensorDType::FP32,
            std::vector<long>{1L});
    } else if (qconfig.format == QuantFormat::HF_MXFP4) {
        qt.format = QuantFormat::HF_MXFP4;
        qt.block_size = 32;

        // Packed FP4 E2M1 data: 2 values per byte
        const size_t data_bytes = static_cast<size_t>(M) * K / 2;
        CUDA_CHECK(cudaMalloc(&alloc.data, data_bytes));
        qt.data = Tensor::from_pointer(
            static_cast<std::byte*>(alloc.data), 0, ETensorDType::BYTE,
            std::vector<long>{static_cast<long>(data_bytes)});

        // E8M0 shared exponents: one uint8 per 32-element block
        const size_t scales_bytes = static_cast<size_t>(M) * K / 32;
        CUDA_CHECK(cudaMalloc(&alloc.scales, scales_bytes));
        qt.scales = Tensor::from_pointer(
            static_cast<std::byte*>(alloc.scales), 0, ETensorDType::BYTE,
            std::vector<long>{static_cast<long>(scales_bytes)});
    } else if (qconfig.format == QuantFormat::FP8_PER_BLOCK) {
        qt.format = QuantFormat::FP8_PER_BLOCK;
        qt.block_size = qconfig.block_size;

        // FP8 E4M3 data: 1 byte per value
        const size_t data_bytes = static_cast<size_t>(M) * K;
        CUDA_CHECK(cudaMalloc(&alloc.data, data_bytes));
        qt.data = Tensor::from_pointer(
            static_cast<std::byte*>(alloc.data), 0, ETensorDType::FP8_E4M3,
            std::vector<long>{static_cast<long>(M), static_cast<long>(K)});

        // Per-block FP32 scales (2D tile layout)
        const long scale_rows = (static_cast<long>(M) + qt.block_size - 1) / qt.block_size;
        const long scale_cols = (static_cast<long>(K) + qt.block_size - 1) / qt.block_size;
        const size_t scales_bytes = static_cast<size_t>(scale_rows) * scale_cols * sizeof(float);
        CUDA_CHECK(cudaMalloc(&alloc.scales, scales_bytes));
        qt.scales = Tensor::from_pointer(
            static_cast<std::byte*>(alloc.scales), 0, ETensorDType::FP32,
            std::vector<long>{scale_rows * scale_cols});
    } else {
        throw std::runtime_error(
            "allocate_qt_gpu_temp: unsupported format "
            + std::to_string(static_cast<int>(qconfig.format)));
    }

    return alloc;
}

/// Relocate a QuantizedTensor from GPU to pinned host memory and free
/// the GPU temp buffers. After this call, qt's data/scales/meta point
/// to pinned host memory suitable for the OffloadManager.
void relocate_qt_to_pinned(QuantizedTensor& qt, GpuTempAlloc& gpu,
                           cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto relocate_tensor = [](Tensor& t, void*& gpu_ptr) {
        if (!gpu_ptr || t.is_null() || t.bytes() == 0) return;
        void* pinned = nullptr;
        CUDA_CHECK(cudaMallocHost(&pinned, t.bytes()));
        CUDA_CHECK(cudaMemcpy(pinned, gpu_ptr, t.bytes(),
                              cudaMemcpyDeviceToHost));
        t.Data = static_cast<std::byte*>(pinned);
        t.Device = -1;
    };

    relocate_tensor(qt.data, gpu.data);
    relocate_tensor(qt.scales, gpu.scales);
    relocate_tensor(qt.meta, gpu.meta);

    gpu.free_all();
}

/// Find the maximum weight size (in elements) across all quantizable specs.
/// Used to allocate reusable BF16 load buffers.
/// Excludes 3D expert weights — they are too large for a shared buffer
/// and are handled with per-weight temporary allocation instead.
size_t max_weight_elements(const std::vector<WeightLoadSpec>& specs) {
    size_t max_elem = 0;
    for (const auto& spec : specs) {
        if (is_expert_weight(spec)) continue;  // handled separately
        size_t elem = spec_num_elements(spec);
        if (elem > max_elem) {
            max_elem = elem;
        }
    }
    return max_elem;
}

/// Count quantizable specs (for progress reporting).
int count_quantizable(const std::vector<WeightLoadSpec>& specs) {
    int count = 0;
    for (const auto& s : specs) {
        if (s.quantize) count++;
    }
    return count;
}

/// Show a simple progress bar on stderr.
void show_progress(int current, int total, const char* label,
                    int rank = 0, int world_size = 1) {
    if (total <= 0) return;

    const float progress = static_cast<float>(current + 1) / static_cast<float>(total);

    if (world_size > 1) {
        // Multi-GPU: print a line every 10% (and at 100%)
        const int pct = static_cast<int>(progress * 100.0f);
        const int prev_pct = static_cast<int>(
            static_cast<float>(current) / static_cast<float>(total) * 100.0f);
        if (pct / 25 != prev_pct / 25 || current + 1 == total) {
            fprintf(stderr, "[Rank %d] [%s] %d%% (%d/%d)\n",
                    rank, label, pct, current + 1, total);
            fflush(stderr);
        }
    } else {
        // Single GPU: interactive progress bar
        const int bar_width = 40;
        const int filled = static_cast<int>(progress * bar_width);

        fprintf(stderr, "\r[%s] [", label);
        for (int i = 0; i < bar_width; ++i) {
            fprintf(stderr, "%c", i < filled ? '#' : '.');
        }
        fprintf(stderr, "] %d/%d", current + 1, total);
        if (current + 1 == total) {
            fprintf(stderr, "\n");
        }
        fflush(stderr);
    }
}

/// Create a shaped view of a flat load buffer.
/// Uses the full shape if provided (for 3D expert weights), otherwise {M, K}.
Tensor make_buffer_view(const Tensor& buffer, const WeightLoadSpec& spec) {
    std::vector<long> shape;
    if (!spec.shape.empty()) {
        shape = spec.shape;
    } else if (spec.K > 0) {
        shape = {static_cast<long>(spec.M), static_cast<long>(spec.K)};
    } else {
        shape = {static_cast<long>(spec.M)};
    }
    return Tensor::from_pointer(
        buffer.Data, buffer.Device,
        ETensorDType::BF16,
        shape);
}

/// RAII guard for a CUDA stream.
struct ScopedStream {
    cudaStream_t stream = nullptr;
    ScopedStream() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~ScopedStream() {
        if (stream) cudaStreamDestroy(stream);
    }
    ScopedStream(const ScopedStream&) = delete;
    ScopedStream& operator=(const ScopedStream&) = delete;
};

/// RAII guard for a CUDA event.
struct ScopedEvent {
    cudaEvent_t event = nullptr;
    ScopedEvent() {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    ~ScopedEvent() {
        if (event) cudaEventDestroy(event);
    }
    ScopedEvent(const ScopedEvent&) = delete;
    ScopedEvent& operator=(const ScopedEvent&) = delete;
};

// ---- Mapping resolution helpers for prequantized import ----
// These replicate DslWeightLoader's private methods so we can resolve HF names
// without going through the BF16 load path.

/// Parse "blocks[N].param_name" or "xxx_blocks[N].param_name" -> (layer_idx, param_name).
bool parse_block_param_local(std::string_view name, int& layer_idx, std::string& param_name) {
    auto dot = name.find('.');
    if (dot == std::string_view::npos) return false;
    auto prefix = name.substr(0, dot);
    auto rest = name.substr(dot + 1);

    auto bracket = prefix.find("blocks[");
    if (bracket != std::string_view::npos) {
        auto close = prefix.find(']', bracket);
        if (close == std::string_view::npos) return false;
        auto idx_start = bracket + 7;
        auto idx_str = prefix.substr(idx_start, close - idx_start);
        try {
            layer_idx = std::stoi(std::string(idx_str));
        } catch (...) {
            return false;
        }
        param_name = std::string(rest);
        return true;
    }

    if (prefix == "blocks") {
        auto idx_str = name.substr(dot + 1);
        auto dot2 = idx_str.find('.');
        if (dot2 == std::string_view::npos) return false;
        try {
            layer_idx = std::stoi(std::string(idx_str.substr(0, dot2)));
        } catch (...) {
            return false;
        }
        param_name = std::string(idx_str.substr(dot2 + 1));
        return true;
    }

    return false;
}

/// Find mapping spec for an internal parameter name from the mapping table.
const dsl::MappingSpec* resolve_mapping_spec(
    const dsl::MappingTable& mapping,
    const std::string& internal_name,
    int& layer_idx) {
    layer_idx = -1;

    auto it = mapping.find(internal_name);
    if (it != mapping.end()) {
        return &it->second;
    }

    std::string base;
    if (parse_block_param_local(internal_name, layer_idx, base)) {
        std::string placeholder = std::string("blocks[{layer}].") + base;
        it = mapping.find(placeholder);
        if (it != mapping.end()) {
            return &it->second;
        }
        it = mapping.find(base);
        if (it != mapping.end()) {
            return &it->second;
        }
    }

    return nullptr;
}

/// Substitute {layer} and {expert} placeholders in HF name template.
std::string resolve_hf_name(std::string templ, int layer_idx, int expert_idx = -1) {
    {
        const std::string placeholder = "{layer}";
        std::size_t pos = templ.find(placeholder);
        while (pos != std::string::npos) {
            if (layer_idx < 0) {
                throw std::runtime_error(
                    "import_prequantized_weights: HF mapping uses {layer} but no layer index for '"
                    + templ + "'");
            }
            templ.replace(pos, placeholder.size(), std::to_string(layer_idx));
            pos = templ.find(placeholder, pos);
        }
    }
    {
        const std::string placeholder = "{expert}";
        std::size_t pos = templ.find(placeholder);
        while (pos != std::string::npos) {
            if (expert_idx < 0) {
                throw std::runtime_error(
                    "import_prequantized_weights: HF mapping uses {expert} but no expert index for '"
                    + templ + "'");
            }
            templ.replace(pos, placeholder.size(), std::to_string(expert_idx));
            pos = templ.find(placeholder, pos);
        }
    }
    return templ;
}

/// Simple glob-like matching: `*` matches any sequence of characters.
/// Used for HF `modules_to_not_convert` patterns like "model.layers.*.self_attn".
bool glob_match(const std::string& text, const std::string& pattern) {
    size_t ti = 0, pi = 0;
    size_t star_pi = std::string::npos, star_ti = 0;

    while (ti < text.size()) {
        if (pi < pattern.size() && pattern[pi] == '*') {
            // Record star position and advance pattern
            star_pi = pi++;
            star_ti = ti;
        } else if (pi < pattern.size() && (pattern[pi] == text[ti] || pattern[pi] == '?')) {
            ++pi;
            ++ti;
        } else if (star_pi != std::string::npos) {
            // Backtrack: let star consume one more character
            pi = star_pi + 1;
            ti = ++star_ti;
        } else {
            return false;
        }
    }
    // Consume trailing stars
    while (pi < pattern.size() && pattern[pi] == '*') ++pi;
    return pi == pattern.size();
}

/// Check whether an HF weight name matches any module in `modules_to_not_convert`.
/// Patterns may use `*` as wildcards (e.g., "model.layers.*.self_attn").
/// A pattern matches if the HF name starts with or equals the pattern (allowing
/// the HF name to have additional suffixes like ".weight").
bool is_module_not_converted(const std::string& hf_name,
                             const std::vector<std::string>& modules_to_not_convert) {
    for (const auto& pattern : modules_to_not_convert) {
        if (pattern.find('*') != std::string::npos) {
            // Glob pattern: check if hf_name starts with the pattern
            // (allow trailing .weight, .bias, etc.)
            if (glob_match(hf_name, pattern) || glob_match(hf_name, pattern + ".*")) {
                return true;
            }
        } else {
            // Literal: check prefix or exact match
            if (hf_name == pattern || hf_name.find(pattern) == 0 ||
                hf_name.find(pattern + ".") != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

/// Load pre-quantized expert weights from StackExperts mapping into a stacked QuantizedTensor.
///
/// Reads per-expert quantized data and scales from safetensors, placing them at the
/// correct byte offset within the full stacked tensor. For fuse_gate_up specs,
/// loads up_proj and gate_proj into the first and second halves of each expert's rows.
///
/// For NVFP4 pre-quantized models, also reads per-component global scale values
/// (weight_scale_2) and normalizes all block scales to a unified global_scale.
/// This is critical because each component (expert × projection) was quantized
/// independently with its own global_amax, and using max() across all components
/// would produce incorrect dequantization for components with smaller global_scale.
///
/// Returns true if scale2 was handled (caller should skip outer-loop scale2 handling).
bool load_prequant_experts_stacked(
    SafeTensorsReader& reader,
    const dsl::MappingSpec& mspec,
    const DslQLoRAPipelineConfig& config,
    QuantizedTensor& qt,
    int layer_idx,
    int E, int /*per_M*/, int /*K*/,
    cudaStream_t stream) {

    const size_t total_data_bytes = qt.data.bytes();
    const size_t total_scale_bytes = qt.scales.bytes();
    const size_t per_expert_data_bytes = total_data_bytes / E;
    const size_t per_expert_scale_bytes = total_scale_bytes / E;
    const size_t data_elem_size = get_dtype_size(qt.data.DType);
    const size_t scale_elem_size = get_dtype_size(qt.scales.DType);

    // Track per-component scale2 values for NVFP4 rescaling.
    // Each entry: {scale2_value, byte_offset_in_scales_buffer, num_fp8_elements}
    struct CompScale2 {
        float scale2;
        size_t scale_byte_offset;
        long scale_count;
    };
    std::vector<CompScale2> comp_scales;
    const bool has_scale2 = !config.scale2_suffix.empty() && qt.meta.Data != nullptr;

    // Helper: read a single-element float scale2 from safetensors via the device meta buffer
    auto read_scale2 = [&](const std::string& hf_name) -> float {
        std::string scale2_hf = hf_name + config.scale2_suffix;
        const auto* entry = try_find_entry(reader, scale2_hf);
        if (!entry) return 0.0f;
        entry->read_raw(qt.meta, 0, 1, true);
        float val = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(&val, qt.meta.Data, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return val;
    };

    // EP: compute global expert start index for HF name resolution.
    // E is already the local expert count (spec.shape adjusted by EP config).
    // Local storage uses indices 0..E-1; HF files use global indices.
    const int ep_expert_start = (config.ep_size > 1) ? config.ep_rank * E : 0;

    if (mspec.fuse_gate_up) {
        // Fused gate_up: source pattern points to gate_proj; derive up_proj pattern.
        std::string gate_pattern = mspec.source;
        std::string up_pattern = mspec.source;
        auto pos = up_pattern.find("gate_proj");
        if (pos != std::string::npos) {
            up_pattern.replace(pos, 9, "up_proj");
        }

        // Each expert has 2*D rows: up_proj in first D, gate_proj in second D.
        const size_t half_data_bytes = per_expert_data_bytes / 2;
        const size_t half_scale_bytes = per_expert_scale_bytes / 2;
        const long half_data_elems = static_cast<long>(half_data_bytes / data_elem_size);
        const long half_scale_elems = static_cast<long>(half_scale_bytes / scale_elem_size);

        for (int e = 0; e < E; ++e) {
            const int global_e = ep_expert_start + e;
            const size_t expert_data_off = e * per_expert_data_bytes;
            const size_t expert_scale_off = e * per_expert_scale_bytes;

            // Load up_proj into first half of expert's data
            std::string up_hf = resolve_hf_name(up_pattern, layer_idx, global_e);
            std::string up_data_hf = config.data_suffix.empty()
                ? up_hf : (up_hf + config.data_suffix);

            Tensor up_data = Tensor::from_pointer(
                static_cast<std::byte*>(qt.data.Data) + expert_data_off,
                qt.data.Device, qt.data.DType,
                std::vector<long>{half_data_elems});
            read_raw_padded(reader.find_entry(up_data_hf), up_data, stream);

            std::string up_scale_hf = up_hf + config.scale_suffix;
            Tensor up_scales = Tensor::from_pointer(
                static_cast<std::byte*>(qt.scales.Data) + expert_scale_off,
                qt.scales.Device, qt.scales.DType,
                std::vector<long>{half_scale_elems});
            read_raw_padded(reader.find_entry(up_scale_hf), up_scales, stream);

            // Read up_proj scale2
            if (has_scale2) {
                comp_scales.push_back({read_scale2(up_hf),
                                       expert_scale_off, half_scale_elems});
            }

            // Load gate_proj into second half of expert's data
            std::string gate_hf = resolve_hf_name(gate_pattern, layer_idx, global_e);
            std::string gate_data_hf = config.data_suffix.empty()
                ? gate_hf : (gate_hf + config.data_suffix);
            Tensor gate_data = Tensor::from_pointer(
                static_cast<std::byte*>(qt.data.Data) + expert_data_off + half_data_bytes,
                qt.data.Device, qt.data.DType,
                std::vector<long>{half_data_elems});
            read_raw_padded(reader.find_entry(gate_data_hf), gate_data, stream);

            std::string gate_scale_hf = gate_hf + config.scale_suffix;
            Tensor gate_scales = Tensor::from_pointer(
                static_cast<std::byte*>(qt.scales.Data) + expert_scale_off + half_scale_bytes,
                qt.scales.Device, qt.scales.DType,
                std::vector<long>{half_scale_elems});
            read_raw_padded(reader.find_entry(gate_scale_hf), gate_scales, stream);

            // Read gate_proj scale2
            if (has_scale2) {
                comp_scales.push_back({read_scale2(gate_hf),
                                       expert_scale_off + half_scale_bytes, half_scale_elems});
            }
        }
    } else {
        const long per_expert_data_elems = static_cast<long>(per_expert_data_bytes / data_elem_size);
        const long per_expert_scale_elems = static_cast<long>(per_expert_scale_bytes / scale_elem_size);

        for (int e = 0; e < E; ++e) {
            const int global_e = ep_expert_start + e;
            std::string hf_name = resolve_hf_name(mspec.source, layer_idx, global_e);
            std::string data_hf = config.data_suffix.empty()
                ? hf_name : (hf_name + config.data_suffix);

            // Read quantized data for this expert
            Tensor data_view = Tensor::from_pointer(
                static_cast<std::byte*>(qt.data.Data) + e * per_expert_data_bytes,
                qt.data.Device, qt.data.DType,
                std::vector<long>{per_expert_data_elems});
            read_raw_padded(reader.find_entry(data_hf), data_view, stream);

            // Read scale for this expert
            std::string scale_hf = hf_name + config.scale_suffix;
            Tensor scale_view = Tensor::from_pointer(
                static_cast<std::byte*>(qt.scales.Data) + e * per_expert_scale_bytes,
                qt.scales.Device, qt.scales.DType,
                std::vector<long>{per_expert_scale_elems});
            read_raw_padded(reader.find_entry(scale_hf), scale_view, stream);

            // Read this expert's scale2
            if (has_scale2) {
                comp_scales.push_back({read_scale2(hf_name),
                                       e * per_expert_scale_bytes, per_expert_scale_elems});
            }
        }
    }

    // Rescale block scales so all components share a unified global_scale.
    // Each component was quantized with its own global_scale (= amax / (FP8_MAX * FP4_MAX)).
    // We find the max across all components, then for each component with a smaller
    // global_scale, multiply its FP8 block scales by (component_scale2 / max_scale2).
    // This compensates for the difference so that dequant with max_scale2 produces
    // correct values for all components.
    bool scale2_handled = false;
    if (has_scale2 && !comp_scales.empty()) {
        float max_scale2 = 0.0f;
        for (const auto& cs : comp_scales) {
            max_scale2 = std::max(max_scale2, cs.scale2);
        }

        if (max_scale2 > 0.0f) {
            for (const auto& cs : comp_scales) {
                float ratio = cs.scale2 / max_scale2;
                if (std::abs(ratio - 1.0f) > 1e-6f) {
                    auto* scale_ptr = reinterpret_cast<__nv_fp8_e4m3*>(
                        static_cast<std::byte*>(qt.scales.Data) + cs.scale_byte_offset);
                    rescale_fp8_scales(scale_ptr, cs.scale_count, ratio, stream);
                }
            }

            qt.global_scale = max_scale2;

            // Write to device meta for consistency
            CUDA_CHECK(cudaMemcpyAsync(qt.meta.Data, &qt.global_scale,
                                       sizeof(float), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            scale2_handled = true;
        }
    }

    // Swizzle the full stacked scale buffer from row-major to F8_128x4 for NVFP4.
    // After per-expert loading, the scale buffer is [E*per_M, K/16] row-major.
    // Swizzle globally because the dequant kernel uses global scale_dims(E*per_M, K)
    // offsets. Per-expert swizzle would be incorrect: when per_M is not 128-aligned,
    // the per-expert padded dims differ from the global dims, and 128-row tiles can
    // span expert boundaries in the global layout.
    if (config.quantizer_config.format == QuantFormat::FP4_BLOCK_2D) {
        auto [sr, sc] = modules::FP4BlockScaleConfig::scale_dims(qt.M, qt.K);
        swizzle_fp8_scales_rowmajor_to_f8_128x4(
            qt.scales.get<__nv_fp8_e4m3>(), sr, sc, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return scale2_handled;
}

}  // anonymous namespace

// =============================================================================
// Main pipeline function
// =============================================================================

std::unique_ptr<GenericWeightManager> import_and_quantize_weights(
    const std::string& file_name,
    const DslQLoRAPipelineConfig& config,
    const PretrainedConfig& pt_config,
    std::shared_ptr<TensorAllocator> allocator,
    cudaStream_t stream) {

    auto t_start = std::chrono::steady_clock::now();

    // ---- Step 1: Create quantizer and weight manager ----

    auto quantizer = create_quantizer(config.quantizer_config);
    if (!quantizer && config.quantizer_config.format != QuantFormat::NONE) {
        throw std::runtime_error("Failed to create quantizer for the specified format");
    }

    auto weight_mgr = std::make_unique<GenericWeightManager>(
        config.weight_manager_config,
        std::move(quantizer),
        allocator);

    // ---- Step 2: Register all quantizable weights ----

    for (const auto& spec : config.weight_specs) {
        if (spec.quantize && weight_mgr->quantizer()) {
            weight_mgr->register_weight(spec.name, spec.M, spec.K, spec.offload_group, spec.shape);
        }
        // Full-precision weights will be registered during load (we need the tensor first)
    }

    // ---- Step 3: Open SafeTensors and create weight loader ----

    SafeTensorsReader reader(file_name);

    dsl::ShardConfig shard_config;
    shard_config.shard_idx = config.shard_idx;
    shard_config.num_shards = config.num_shards;

    dsl::MoEWeightConfig moe_config;
    moe_config.num_experts = config.num_experts;
    moe_config.moe_intermediate_size = config.moe_intermediate_size;

    dsl::DslWeightLoader loader(
        reader,
        config.mapping,
        pt_config,
        *allocator,
        shard_config,
        moe_config);

    // ---- Step 3b: Create adapter merger (stacked LoRA) ----

    std::unique_ptr<AdapterMerger> adapter_merger;
    if (!config.adapter_path.empty()) {
        adapter_merger = std::make_unique<AdapterMerger>(
            config.adapter_path,
            config.mapping,
            reader,
            shard_config);
    }

    // ---- Step 4: Allocate double-buffered load buffers ----
    //
    // Double buffering allows overlapping:
    //   - CPU disk I/O for weight N+1  with  GPU quantization of weight N
    //
    // Buffer A and B alternate: while GPU quantizes from buffer A,
    // CPU reads the next weight into buffer B (and vice versa).
    //
    // IMPORTANT: Load buffers are temporary — allocated with cudaMalloc and
    // freed after import. Using the shared TensorAllocator would permanently
    // leak these buffers since it has no deallocation support.

    const size_t max_elem = max_weight_elements(config.weight_specs);
    const int num_quantizable = count_quantizable(config.weight_specs);
    bool use_double_buffer = (num_quantizable >= 2) && (max_elem > 0);

    const size_t load_buf_bytes = max_elem * sizeof(nv_bfloat16);
    void* load_buf_ptrs[2] = {nullptr, nullptr};
    Tensor load_buffers[2];
    if (max_elem > 0) {
        CUDA_CHECK(cudaMalloc(&load_buf_ptrs[0], load_buf_bytes));
        load_buffers[0] = Tensor::from_pointer(
            static_cast<std::byte*>(load_buf_ptrs[0]), 0,
            ETensorDType::BF16,
            std::vector<long>{static_cast<long>(max_elem)});
        if (use_double_buffer) {
            // Try to allocate second buffer; fall back to single-buffer if OOM.
            auto err = cudaMalloc(&load_buf_ptrs[1], load_buf_bytes);
            if (err == cudaSuccess) {
                load_buffers[1] = Tensor::from_pointer(
                    static_cast<std::byte*>(load_buf_ptrs[1]), 0,
                    ETensorDType::BF16,
                    std::vector<long>{static_cast<long>(max_elem)});
            } else {
                // Not enough GPU memory for double buffering — fall back gracefully.
                cudaGetLastError();  // Clear the error
                use_double_buffer = false;
            }
        }
    }

    // Create quantization stream and per-buffer completion events
    ScopedStream quant_stream_guard;
    ScopedEvent buf_ready[2];
    cudaStream_t quant_stream = use_double_buffer ? quant_stream_guard.stream : stream;

    // ---- Step 5: Load and quantize weights ----
    //
    // Two-pass loading to manage GPU memory:
    //   Pass 1: 2D weights + full-precision — uses shared double-buffered pipeline
    //   Pass 2: 3D expert weights — freed shared buffers first, then per-weight temp alloc
    //
    // This ensures the shared load buffers (which can be ~200 MB) are freed
    // before allocating the large per-expert temp buffers (up to ~1.4 GB).

    const int total = static_cast<int>(config.weight_specs.size());
    int loaded = 0;
    int quant_slot = 0;  // Alternates 0/1 for double buffering

    // --- Pass 1: Non-expert weights (2D quantizable + full-precision) ---

    for (int i = 0; i < total; ++i) {
        const auto& spec = config.weight_specs[i];

        if (is_expert_weight(spec)) {
            continue;  // Deferred to pass 2
        }

        show_progress(loaded, total, "QLoRA Import", config.shard_idx, config.num_shards);

        if (spec.quantize && weight_mgr->quantizer()) {
            // 2D quantizable weight: double-buffered load → quantize pipeline
            const int slot = use_double_buffer ? (quant_slot % 2) : 0;

            // Wait for previous quantization that used this buffer slot
            if (use_double_buffer && quant_slot >= 2) {
                CUDA_CHECK(cudaEventSynchronize(buf_ready[slot].event));
            }

            // Create a view of the load buffer with the correct shape
            Tensor target = make_buffer_view(load_buffers[slot], spec);

            // Load BF16 weight from SafeTensors (CPU disk I/O + GPU copy on main stream)
            bool success = loader.load_param(spec.name, target, true, spec.sharded, nullptr, stream);
            if (success) {
                // Ensure GPU copy is complete before merging/quantizing
                CUDA_CHECK(cudaStreamSynchronize(stream));

                // Merge adapter delta if an adapter is being loaded (stacked LoRA)
                if (adapter_merger) {
                    adapter_merger->apply(spec.name, target, stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }

                CUDA_CHECK(cudaGetLastError());

                // Issue quantization on quant stream (async GPU kernel)
                weight_mgr->quantize_and_store(spec.name, target, quant_stream);
                CUDA_CHECK(cudaStreamSynchronize(quant_stream));

                // Record event: buffer slot is free when this quant completes
                if (use_double_buffer) {
                    CUDA_CHECK(cudaEventRecord(buf_ready[slot].event, quant_stream));
                }

                loaded++;
            }

            quant_slot++;
        } else {
            // Full-precision weight: allocate storage and load directly
            std::vector<long> shape;
            if (!spec.shape.empty()) {
                shape = spec.shape;
            } else if (spec.K > 0) {
                shape = {static_cast<long>(spec.M), static_cast<long>(spec.K)};
            } else {
                shape = {static_cast<long>(spec.M)};
            }

            Tensor tensor = allocator->allocate(
                spec.target_dtype,
                spec.name.c_str(),
                EAllocationType::ON_DEVICE,
                shape);

            bool success = loader.load_param(spec.name, tensor, true, spec.sharded, nullptr, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaGetLastError());
            if (success) {
                weight_mgr->register_full_precision(spec.name, tensor);
                loaded++;
            }
        }
    }

    // --- Free shared load buffers before pass 2 to reclaim GPU memory ---

    if (use_double_buffer) {
        CUDA_CHECK(cudaStreamSynchronize(quant_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < 2; ++i) {
        if (load_buf_ptrs[i]) {
            CUDA_CHECK(cudaFree(load_buf_ptrs[i]));
            load_buf_ptrs[i] = nullptr;
            load_buffers[i] = Tensor{};
        }
    }

    // --- Pass 2: 3D expert weights (per-expert streaming) ---
    //
    // Instead of allocating the full [E, per_M, K] temporary buffer (~1.4 GB for
    // 64 experts), we load and quantize one expert at a time using a small
    // [per_M, K] buffer (~22 MB). This is critical for memory-constrained GPUs.

    for (int i = 0; i < total; ++i) {
        const auto& spec = config.weight_specs[i];

        if (!is_expert_weight(spec)) {
            continue;  // Already handled in pass 1
        }
        if (!spec.quantize || !weight_mgr->quantizer()) {
            // Full-precision expert weight (unlikely but handle gracefully).
            // EP with full-precision 3D experts requires per-expert loading which
            // is only implemented for the quantized path. In practice EP always
            // uses QLoRA, so this path should not be hit.
            if (config.ep_size > 1) {
                throw std::runtime_error(
                    "Full-precision expert weights not supported with EP "
                    "(use QLoRA quantization): " + spec.name);
            }
            Tensor tensor = allocator->allocate(
                spec.target_dtype,
                spec.name.c_str(),
                EAllocationType::ON_DEVICE,
                spec.shape);

            bool success = loader.load_param(spec.name, tensor, true, spec.sharded, nullptr, stream);
            if (success) {
                weight_mgr->register_full_precision(spec.name, tensor);
                loaded++;
            }
            continue;
        }

        show_progress(loaded, total, "QLoRA Import", config.shard_idx, config.num_shards);

        // Per-expert dimensions from the 3D shape [E, per_M, K]
        const int E = static_cast<int>(spec.shape[0]);
        const int per_M = static_cast<int>(spec.shape[1]);
        const int K = static_cast<int>(spec.shape[2]);
        const size_t per_expert_bytes = static_cast<size_t>(per_M) * K * sizeof(nv_bfloat16);

        // Allocate a small per-expert load buffer
        void* expert_buf_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&expert_buf_ptr, per_expert_bytes));

        Tensor expert_buf = Tensor::from_pointer(
            static_cast<std::byte*>(expert_buf_ptr), 0,
            ETensorDType::BF16,
            std::vector<long>{static_cast<long>(per_M), static_cast<long>(K)});

        // Load and quantize each expert individually.
        // With EP, E is already the local expert count (shape adjusted in config).
        // Load from global HF index, quantize to local storage index.
        const int ep_expert_start = (config.ep_size > 1) ? config.ep_rank * E : 0;
        for (int e = 0; e < E; ++e) {
            const int global_e = ep_expert_start + e;
            loader.load_expert(spec.name, global_e, expert_buf, true, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Merge adapter delta for this expert (stacked LoRA)
            if (adapter_merger) {
                adapter_merger->apply_expert(spec.name, global_e, expert_buf, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            try {
                weight_mgr->quantize_expert_slice(spec.name, e, per_M, expert_buf, stream);
            } catch (...) {
                cudaFree(expert_buf_ptr);
                throw;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        CUDA_CHECK(cudaFree(expert_buf_ptr));
        loaded++;
    }

    // ---- Step 6: Resolve tied parameters ----

    loader.resolve_tied_params([&](const std::string& name) -> Tensor& {
        return weight_mgr->get(name, stream);
    });

    // ---- Step 7: Synchronize to ensure all operations complete ----

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // ---- Step 8: Report statistics ----

    const size_t quant_bytes = weight_mgr->quantized_bytes();
    const size_t dequant_bytes = weight_mgr->dequant_buffer_bytes();
    const size_t fp_bytes = weight_mgr->full_precision_bytes();
    const size_t total_gpu_bytes = quant_bytes + dequant_bytes + fp_bytes;

    fprintf(stderr, "[QLoRA Import] Loaded %d/%d weights (%d quantized, %d full-precision) "
                    "in %.1f ms%s\n",
            loaded, total,
            weight_mgr->num_quantized(),
            weight_mgr->num_full_precision(),
            elapsed_ms,
            use_double_buffer ? " [double-buffered]" : "");

    fprintf(stderr, "[QLoRA Import] Memory: quantized=%.1f MB, dequant_buf=%.1f MB, "
                    "full_precision=%.1f MB, total=%.1f MB\n",
            static_cast<double>(quant_bytes) / (1024.0 * 1024.0),
            static_cast<double>(dequant_bytes) / (1024.0 * 1024.0),
            static_cast<double>(fp_bytes) / (1024.0 * 1024.0),
            static_cast<double>(total_gpu_bytes) / (1024.0 * 1024.0));

    if (weight_mgr->is_pooled()) {
        fprintf(stderr, "[QLoRA Import] Dequant buffer pool: max_cache=%d "
                        "(saves %.1f MB vs pre-allocated)\n",
                config.weight_manager_config.max_dequant_cache_size,
                static_cast<double>(
                    static_cast<size_t>(weight_mgr->num_quantized()) *
                    max_elem * 2  // BF16 = 2 bytes per element
                    - dequant_bytes
                ) / (1024.0 * 1024.0));
    }

    if (config.weight_manager_config.enable_offloading) {
        const auto* om = weight_mgr->offload_manager();
        if (om) {
            fprintf(stderr, "[QLoRA Import] Offloading: %d groups, %d resident, "
                            "gpu=%.1f MB, cpu=%.1f MB\n",
                    om->num_groups(), om->num_resident(),
                    static_cast<double>(om->gpu_memory_used()) / (1024.0 * 1024.0),
                    static_cast<double>(om->cpu_memory_used()) / (1024.0 * 1024.0));
        }
    }

    return weight_mgr;
}

// =============================================================================
// Pre-quantized import pipeline
// =============================================================================

std::unique_ptr<GenericWeightManager> import_prequantized_weights(
    const std::string& file_name,
    const DslQLoRAPipelineConfig& config,
    const PretrainedConfig& pt_config,
    std::shared_ptr<TensorAllocator> allocator,
    cudaStream_t stream) {

    auto t_start = std::chrono::steady_clock::now();

    // ---- Step 1: Create quantizer and weight manager ----

    auto quantizer_ptr = create_quantizer(config.quantizer_config);
    if (!quantizer_ptr) {
        throw std::runtime_error(
            "import_prequantized_weights: failed to create quantizer for format "
            + std::to_string(static_cast<int>(config.quantizer_config.format)));
    }
    IQuantizer* quantizer = quantizer_ptr.get();

    auto weight_mgr = std::make_unique<GenericWeightManager>(
        config.weight_manager_config,
        std::move(quantizer_ptr),
        allocator);

    // ---- Step 2: Open SafeTensors and create weight loader ----
    // We use DslWeightLoader for full-precision weights and tied param resolution.
    // Quantized weights are loaded directly from safetensors using resolved HF names.

    SafeTensorsReader reader(file_name);

    dsl::ShardConfig shard_config;
    shard_config.shard_idx = config.shard_idx;
    shard_config.num_shards = config.num_shards;

    dsl::MoEWeightConfig moe_config;
    moe_config.num_experts = config.num_experts;
    moe_config.moe_intermediate_size = config.moe_intermediate_size;

    dsl::DslWeightLoader loader(
        reader,
        config.mapping,
        pt_config,
        *allocator,
        shard_config,
        moe_config);

    auto load_full_precision = [&](const WeightLoadSpec& spec,
                                   const char* reason) -> bool {
        std::vector<long> shape;
        if (!spec.shape.empty()) {
            shape = spec.shape;
        } else if (spec.K > 0) {
            shape = {static_cast<long>(spec.M), static_cast<long>(spec.K)};
        } else {
            shape = {static_cast<long>(spec.M)};
        }

        Tensor tensor = allocator->allocate(
            spec.target_dtype,
            spec.name.c_str(),
            EAllocationType::ON_DEVICE,
            shape);

        bool success = loader.load_param(
            spec.name, tensor, true, spec.sharded, nullptr, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (success) {
            weight_mgr->register_full_precision(spec.name, tensor);
        }
        return success;
    };

    // ---- Step 3: Load weights ----
    //
    // Two-pass loading (same structure as online quantization):
    //   Pass 1: 2D weights + full-precision
    //   Pass 2: 3D expert weights
    //
    // For truly pre-quantized weights, we read directly into quantized storage.
    // If a weight's safetensors entry is still in float format (BF16/FP16/FP32),
    // we fall back to online quantization using a reusable temporary buffer.

    const int total = static_cast<int>(config.weight_specs.size());
    int loaded = 0;

    // Reusable BF16 buffer for on-the-fly quantization of non-packed weights.
    // Allocated lazily on first use, freed after pass 1.
    void* quant_tmp_ptr = nullptr;
    size_t quant_tmp_bytes = 0;

    // --- Pass 1: Non-expert weights (2D quantized + full-precision) ---

    for (int i = 0; i < total; ++i) {
        const auto& spec = config.weight_specs[i];

        if (is_expert_weight(spec)) {
            continue;  // Deferred to pass 2
        }

        show_progress(loaded, total, "Prequant Import", config.shard_idx, config.num_shards);

        if (spec.quantize && quantizer) {
            // Resolve HF name from mapping table
            int layer_idx = -1;
            const auto* mspec = resolve_mapping_spec(config.mapping, spec.name, layer_idx);

            // Fall back to Direct mapping with internal name as HF path
            dsl::MappingSpec direct_fallback;
            if (!mspec) {
                direct_fallback.kind = dsl::MappingSpec::Kind::Direct;
                direct_fallback.source = spec.name;
                mspec = &direct_fallback;
            }

            // For Transform mappings (e.g. transpose): the raw quantized data is read
            // as-is from HF safetensors. The transform (e.g. transpose) is applied
            // after dequantization at runtime by GenericWeightManager.
            //
            // For Fuse mappings in pre-quantized models: each component is stored
            // as packed quantized data (e.g., q_proj, k_proj, v_proj are each U8).
            // We dequantize each component to BF16, fuse them, then re-quantize.
            //
            // For non-prequantized Fuse/Split/etc: fall back to full-precision
            // loading via DslWeightLoader.
            bool skip_quant = false;
            if (!config.modules_to_not_convert.empty()) {
                if (mspec->kind == dsl::MappingSpec::Kind::Fuse) {
                    for (const auto& src : mspec->sources) {
                        const std::string src_hf = resolve_hf_name(src, layer_idx);
                        if (is_module_not_converted(src_hf, config.modules_to_not_convert)) {
                            skip_quant = true;
                            break;
                        }
                    }
                } else if (mspec->kind == dsl::MappingSpec::Kind::Direct ||
                           mspec->kind == dsl::MappingSpec::Kind::Transform) {
                    std::string hf_name = resolve_hf_name(
                        mspec->source.empty() ? spec.name : mspec->source, layer_idx);
                    if (is_module_not_converted(hf_name, config.modules_to_not_convert)) {
                        skip_quant = true;
                    }
                }
            }
            if (skip_quant) {
                if (load_full_precision(spec, "modules_to_not_convert")) {
                    loaded++;
                }
                continue;
            }

            if (mspec->kind == dsl::MappingSpec::Kind::Fuse && config.prequantized) {
                bool missing_entries = false;
                for (const auto& src : mspec->sources) {
                    std::string comp_hf = resolve_hf_name(src, layer_idx);
                    std::string comp_data_hf = config.data_suffix.empty()
                        ? comp_hf : (comp_hf + config.data_suffix);
                    const auto* comp_data_entry = try_find_entry(reader, comp_data_hf);
                    if (!comp_data_entry) {
                        missing_entries = true;
                        break;
                    }
                    const auto comp_dtype = comp_data_entry->dtype();
                    if (comp_dtype != ETensorDType::BF16 &&
                        comp_dtype != ETensorDType::FP16 &&
                        comp_dtype != ETensorDType::FP32) {
                        std::string comp_scale_hf = comp_hf + config.scale_suffix;
                        if (!try_find_entry(reader, comp_scale_hf)) {
                            missing_entries = true;
                            break;
                        }
                        if (!config.scale2_suffix.empty()) {
                            std::string comp_scale2_hf = comp_hf + config.scale2_suffix;
                            if (!try_find_entry(reader, comp_scale2_hf)) {
                                missing_entries = true;
                                break;
                            }
                        }
                    }
                }
                if (missing_entries) {
                    if (load_full_precision(spec, "missing prequant blocks")) {
                        loaded++;
                    }
                    continue;
                }
                // Pre-quantized Fuse: dequant each component → BF16, concatenate,
                // re-quantize. Only dim=0 (row concatenation) is supported.
                if (mspec->dim != 0) {
                    throw std::runtime_error(
                        "Pre-quantized Fuse with dim != 0 is not supported for '"
                        + spec.name + "'");
                }

                // Allocate quantized storage for the fused result
                QuantizedTensor qt;
                quantizer->allocate_storage(
                    spec.M, spec.K, qt, *allocator,
                    EAllocationType::ON_DEVICE, spec.name);

                // Grow the reusable BF16 buffer to hold the full fused tensor
                const size_t fused_bytes =
                    static_cast<size_t>(spec.M) * spec.K * sizeof(nv_bfloat16);
                if (fused_bytes > quant_tmp_bytes) {
                    if (quant_tmp_ptr) CUDA_CHECK(cudaFree(quant_tmp_ptr));
                    CUDA_CHECK(cudaMalloc(&quant_tmp_ptr, fused_bytes));
                    quant_tmp_bytes = fused_bytes;
                }

                // Temp GPU buffers for per-component quantized data (reused across components)
                void* comp_data_gpu = nullptr;
                void* comp_scale_gpu = nullptr;
                void* comp_meta_gpu = nullptr;
                size_t comp_data_cap = 0;
                size_t comp_scale_cap = 0;

                int row_offset = 0;

                // Helper: compute element count from safetensors entry shape
                auto entry_nelem = [](const SafeTensorEntry& e) -> long {
                    long n = 1;
                    for (auto d : e.shape()) n *= d;
                    return n;
                };

                for (const auto& src : mspec->sources) {
                    std::string comp_hf = resolve_hf_name(src, layer_idx);
                    std::string comp_data_hf = config.data_suffix.empty()
                        ? comp_hf : (comp_hf + config.data_suffix);

                    const auto& comp_data_entry = reader.find_entry(comp_data_hf);
                    const auto comp_dtype = comp_data_entry.dtype();
                    const auto& comp_shape = comp_data_entry.shape();

                    // BF16 slice in the fused buffer for this component's output
                    auto* slice_ptr = static_cast<std::byte*>(quant_tmp_ptr)
                        + static_cast<size_t>(row_offset) * spec.K * sizeof(nv_bfloat16);

                    if (comp_dtype == ETensorDType::BF16 ||
                        comp_dtype == ETensorDType::FP16 ||
                        comp_dtype == ETensorDType::FP32) {
                        // Float component: shape is [M, K], read directly into BF16 slice
                        int comp_M = static_cast<int>(comp_shape[0]);
                        Tensor slice = Tensor::from_pointer(
                            slice_ptr, 0, ETensorDType::BF16,
                            std::vector<long>{static_cast<long>(comp_M),
                                              static_cast<long>(spec.K)});
                        comp_data_entry.read_tensor(slice, true);
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        row_offset += comp_M;
                    } else {
                        // Pre-quantized component (U8 packed FP4/FP8)
                        // Derive component M from the data entry shape
                        int comp_M;
                        if (comp_shape.size() >= 2) {
                            comp_M = static_cast<int>(comp_shape[0]);
                        } else {
                            // 1D packed: total_bytes * 2 / K for FP4 (2 values per byte)
                            comp_M = static_cast<int>(
                                (entry_nelem(comp_data_entry) * 2) / spec.K);
                        }

                        // Build temp QuantizedTensor for this component
                        QuantizedTensor comp_qt;
                        comp_qt.M = comp_M;
                        comp_qt.K = spec.K;
                        comp_qt.format = qt.format;
                        comp_qt.block_size = qt.block_size;

                        // Grow temp data buffer
                        size_t need_data = static_cast<size_t>(entry_nelem(comp_data_entry));
                        if (need_data > comp_data_cap) {
                            if (comp_data_gpu) CUDA_CHECK(cudaFree(comp_data_gpu));
                            CUDA_CHECK(cudaMalloc(&comp_data_gpu, need_data));
                            comp_data_cap = need_data;
                        }
                        comp_qt.data = Tensor::from_pointer(
                            static_cast<std::byte*>(comp_data_gpu), 0,
                            ETensorDType::BYTE,
                            std::vector<long>{static_cast<long>(need_data)});

                        // Read packed quantized data
                        comp_data_entry.read_raw(
                            comp_qt.data, 0, comp_qt.data.nelem(), true);

                        // Read per-block scales
                        std::string comp_scale_hf = comp_hf + config.scale_suffix;
                        const auto& comp_scale_entry =
                            reader.find_entry(comp_scale_hf);
                        size_t need_scale =
                            static_cast<size_t>(entry_nelem(comp_scale_entry))
                            * get_dtype_size(comp_scale_entry.dtype());
                        if (need_scale > comp_scale_cap) {
                            if (comp_scale_gpu) CUDA_CHECK(cudaFree(comp_scale_gpu));
                            CUDA_CHECK(cudaMalloc(&comp_scale_gpu, need_scale));
                            comp_scale_cap = need_scale;
                        }
                        comp_qt.scales = Tensor::from_pointer(
                            static_cast<std::byte*>(comp_scale_gpu), 0,
                            comp_scale_entry.dtype(),
                            std::vector<long>(comp_scale_entry.shape().begin(),
                                              comp_scale_entry.shape().end()));
                        comp_scale_entry.read_raw(
                            comp_qt.scales, 0, comp_qt.scales.nelem(), true);

                        // Swizzle component scales (NVFP4: row-major → F8_128x4)
                        if (config.quantizer_config.format == QuantFormat::FP4_BLOCK_2D) {
                            auto [sr, sc] = modules::FP4BlockScaleConfig::scale_dims(
                                comp_qt.M, comp_qt.K);
                            swizzle_fp8_scales_rowmajor_to_f8_128x4(
                                comp_qt.scales.get<__nv_fp8_e4m3>(), sr, sc, stream);
                        }

                        // Read second-level global scale (NVFP4: single float)
                        if (!config.scale2_suffix.empty()) {
                            if (!comp_meta_gpu) {
                                CUDA_CHECK(cudaMalloc(&comp_meta_gpu, sizeof(float)));
                            }
                            comp_qt.meta = Tensor::from_pointer(
                                static_cast<std::byte*>(comp_meta_gpu), 0,
                                ETensorDType::FP32, std::vector<long>{1L});
                            std::string comp_scale2_hf =
                                comp_hf + config.scale2_suffix;
                            const auto& comp_s2 =
                                reader.find_entry(comp_scale2_hf);
                            comp_s2.read_raw(comp_qt.meta, 0, 1, true);
                            CUDA_CHECK(cudaMemcpyAsync(
                                &comp_qt.global_scale, comp_qt.meta.Data,
                                sizeof(float), cudaMemcpyDeviceToHost, stream));
                            CUDA_CHECK(cudaStreamSynchronize(stream));
                        }

                        CUDA_CHECK(cudaStreamSynchronize(stream));

                        // Dequantize into the BF16 slice
                        Tensor slice = Tensor::from_pointer(
                            slice_ptr, 0, ETensorDType::BF16,
                            std::vector<long>{static_cast<long>(comp_M),
                                              static_cast<long>(spec.K)});
                        quantizer->dequantize(comp_qt, slice, stream);
                        CUDA_CHECK(cudaStreamSynchronize(stream));

                        row_offset += comp_M;
                    }
                }

                // Free temp component buffers
                if (comp_data_gpu) CUDA_CHECK(cudaFree(comp_data_gpu));
                if (comp_scale_gpu) CUDA_CHECK(cudaFree(comp_scale_gpu));
                if (comp_meta_gpu) CUDA_CHECK(cudaFree(comp_meta_gpu));

                if (row_offset != spec.M) {
                    throw std::runtime_error(
                        "Pre-quantized Fuse row mismatch for '" + spec.name
                        + "': expected " + std::to_string(spec.M)
                        + ", got " + std::to_string(row_offset));
                }

                // Re-quantize the full fused BF16 buffer
                Tensor fused_bf16 = Tensor::from_pointer(
                    static_cast<std::byte*>(quant_tmp_ptr), 0,
                    ETensorDType::BF16,
                    std::vector<long>{static_cast<long>(spec.M),
                                     static_cast<long>(spec.K)});
                quantizer->quantize(fused_bf16, qt, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));

                weight_mgr->store_prequantized(
                    spec.name, std::move(qt), spec.offload_group, spec.shape);
                loaded++;
                continue;
            }

            if (mspec->kind != dsl::MappingSpec::Kind::Direct &&
                mspec->kind != dsl::MappingSpec::Kind::Transform) {
                // Non-Direct/Transform quantized param (Split, TiedTo, etc.)
                // Fall back to full-precision loading via DslWeightLoader
                std::vector<long> fp_shape;
                if (!spec.shape.empty()) {
                    fp_shape = spec.shape;
                } else if (spec.K > 0) {
                    fp_shape = {static_cast<long>(spec.M), static_cast<long>(spec.K)};
                } else {
                    fp_shape = {static_cast<long>(spec.M)};
                }

                Tensor tensor = allocator->allocate(
                    spec.target_dtype, spec.name.c_str(),
                    EAllocationType::ON_DEVICE, fp_shape);
                bool success = loader.load_param(
                    spec.name, tensor, true, spec.sharded, nullptr, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (success) {
                    weight_mgr->register_full_precision(spec.name, tensor);
                    loaded++;
                }
                continue;
            }

            std::string hf_name = resolve_hf_name(
                mspec->source.empty() ? spec.name : mspec->source, layer_idx);

            // Safety check: if HF module is in modules_to_not_convert, load as full-precision
            if (is_module_not_converted(hf_name, config.modules_to_not_convert)) {
                if (load_full_precision(spec, "modules_to_not_convert")) {
                    loaded++;
                }
                continue;
            }

            // Allocate quantized storage
            QuantizedTensor qt;
            quantizer->allocate_storage(
                spec.M, spec.K, qt, *allocator,
                EAllocationType::ON_DEVICE, spec.name);

            // Resolve HF data tensor name
            std::string data_hf = config.data_suffix.empty()
                ? hf_name : (hf_name + config.data_suffix);
            const auto* data_entry_ptr = try_find_entry(reader, data_hf);
            if (!data_entry_ptr) {
                if (load_full_precision(spec, "missing prequant blocks")) {
                    loaded++;
                }
                continue;
            }
            const auto& data_entry = *data_entry_ptr;

            // Check if the HF entry is actually pre-quantized or still float.
            // Some "pre-quantized" models store BF16 weights with quantization
            // metadata in config.json but don't pack the data — quantize on the fly.
            const auto file_dtype = data_entry.dtype();
            if (file_dtype == ETensorDType::BF16 ||
                file_dtype == ETensorDType::FP16 ||
                file_dtype == ETensorDType::FP32) {
                // Weight is still in float format — load as BF16 and quantize
                const size_t needed = static_cast<size_t>(spec.M) * spec.K * sizeof(nv_bfloat16);
                if (needed > quant_tmp_bytes) {
                    if (quant_tmp_ptr) CUDA_CHECK(cudaFree(quant_tmp_ptr));
                    CUDA_CHECK(cudaMalloc(&quant_tmp_ptr, needed));
                    quant_tmp_bytes = needed;
                }
                Tensor bf16_buf = Tensor::from_pointer(
                    static_cast<std::byte*>(quant_tmp_ptr), 0,
                    ETensorDType::BF16,
                    std::vector<long>{static_cast<long>(spec.M), static_cast<long>(spec.K)});

                bool success = loader.load_param(
                    spec.name, bf16_buf, true, spec.sharded, nullptr, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (success) {
                    quantizer->quantize(bf16_buf, qt, stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    weight_mgr->store_prequantized(
                        spec.name, std::move(qt), spec.offload_group, spec.shape);
                    loaded++;
                }
                continue;
            }

            // Read pre-quantized weight data directly from safetensors
            // For FP8/NVFP4: data_suffix is empty, reads from hf_name directly
            // For MXFP4: data_suffix="_blocks", reads from hf_name + "_blocks"
            read_raw_padded(data_entry, qt.data, stream);

            // Read scale tensor
            std::string scale_hf = hf_name + config.scale_suffix;
            const auto* scale_entry_ptr = try_find_entry(reader, scale_hf);
            if (!scale_entry_ptr) {
                if (load_full_precision(spec, "missing prequant scales")) {
                    loaded++;
                }
                continue;
            }

            const auto& scale_entry = *scale_entry_ptr;
            read_raw_padded(scale_entry, qt.scales, stream);

            // For NVFP4: HF stores scales in row-major order, but our dequant
            // kernel expects the F8_128x4 swizzled layout. Swizzle in-place.
            if (config.quantizer_config.format == QuantFormat::FP4_BLOCK_2D) {
                auto [sr, sc] = modules::FP4BlockScaleConfig::scale_dims(spec.M, spec.K);
                swizzle_fp8_scales_rowmajor_to_f8_128x4(
                    qt.scales.get<__nv_fp8_e4m3>(), sr, sc, stream);
            }

            // For NVFP4: read second-level global scale (single float)
            if (!config.scale2_suffix.empty() && qt.meta.Data != nullptr) {
                std::string scale2_hf = hf_name + config.scale2_suffix;
                const auto* scale2_entry_ptr = try_find_entry(reader, scale2_hf);
                if (!scale2_entry_ptr) {
                    if (load_full_precision(spec, "missing prequant scale2")) {
                        loaded++;
                    }
                    continue;
                }
                const auto& scale2_entry = *scale2_entry_ptr;
                // Read into qt.meta (device-side FP32), then copy to host
                scale2_entry.read_raw(qt.meta, 0, 1, true);
                CUDA_CHECK(cudaMemcpyAsync(&qt.global_scale, qt.meta.Data,
                    sizeof(float), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Store in weight manager. Do NOT pass transform_fn for pre-quantized
            // weights: pre-quantized HF models (MXFP4, FP8, NVFP4) store packed
            // data in the GEMM-ready layout (matching DSL parameter shape), so the
            // Transform("transpose") mapping only applies to BF16 weight import.
            weight_mgr->store_prequantized(
                spec.name, std::move(qt), spec.offload_group, spec.shape);
            loaded++;
        } else {
            // Full-precision weight: use DslWeightLoader (handles all mapping kinds)
            std::vector<long> shape;
            if (!spec.shape.empty()) {
                shape = spec.shape;
            } else if (spec.K > 0) {
                shape = {static_cast<long>(spec.M), static_cast<long>(spec.K)};
            } else {
                shape = {static_cast<long>(spec.M)};
            }

            Tensor tensor = allocator->allocate(
                spec.target_dtype,
                spec.name.c_str(),
                EAllocationType::ON_DEVICE,
                shape);

            bool success = loader.load_param(
                spec.name, tensor, true, spec.sharded, nullptr, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaGetLastError());
            if (success) {
                weight_mgr->register_full_precision(spec.name, tensor);
                loaded++;
            }
        }
    }

    // Free temporary quantization buffer from pass 1
    if (quant_tmp_ptr) {
        CUDA_CHECK(cudaFree(quant_tmp_ptr));
        quant_tmp_ptr = nullptr;
        quant_tmp_bytes = 0;
    }

    // --- Pass 2: 3D expert weights (per-expert streaming from pre-quantized data) ---

    for (int i = 0; i < total; ++i) {
        const auto& spec = config.weight_specs[i];

        if (!is_expert_weight(spec)) {
            continue;  // Already handled in pass 1
        }

        if (!spec.quantize || !quantizer) {
            // Full-precision expert weight
            if (config.ep_size > 1) {
                throw std::runtime_error(
                    "Full-precision expert weights not supported with EP "
                    "(use QLoRA quantization): " + spec.name);
            }
            Tensor tensor = allocator->allocate(
                spec.target_dtype,
                spec.name.c_str(),
                EAllocationType::ON_DEVICE,
                spec.shape);

            bool success = loader.load_param(
                spec.name, tensor, true, spec.sharded, nullptr, stream);
            if (success) {
                weight_mgr->register_full_precision(spec.name, tensor);
                loaded++;
            }
            continue;
        }

        show_progress(loaded, total, "Prequant Import", config.shard_idx, config.num_shards);

        // Resolve mapping spec
        int layer_idx = -1;
        const auto* mspec = resolve_mapping_spec(config.mapping, spec.name, layer_idx);
        if (!mspec) {
            throw std::runtime_error(
                "import_prequantized_weights: no mapping for expert param '"
                + spec.name + "'");
        }

        // 3D shape: [E, per_M, K]
        const int E = static_cast<int>(spec.shape[0]);
        const int per_M = static_cast<int>(spec.shape[1]);
        const int K = static_cast<int>(spec.shape[2]);

        // Allocate full stacked QuantizedTensor for [E*per_M, K].
        // When offloading, use raw cudaMalloc (GPU temp) so we can relocate
        // to pinned host after loading + swizzling without leaking
        // TensorAllocator GPU memory.
        QuantizedTensor qt;
        GpuTempAlloc gpu_temp;
        const bool offload_this = should_offload_expert(spec, config);

        if (offload_this) {
            gpu_temp = allocate_qt_gpu_temp(
                spec.M, spec.K, qt, config.quantizer_config);
        } else {
            quantizer->allocate_storage(
                spec.M, spec.K, qt, *allocator,
                EAllocationType::ON_DEVICE, spec.name);
        }

        bool scale2_handled = false;
        try {
            if (mspec->kind == dsl::MappingSpec::Kind::StackExperts) {
                scale2_handled = load_prequant_experts_stacked(
                    reader, *mspec, config, qt, layer_idx, E, per_M, K, stream);
            } else if (mspec->kind == dsl::MappingSpec::Kind::Direct ||
                       mspec->kind == dsl::MappingSpec::Kind::Transform) {
                // Direct or Transform mapping: single stacked tensor in HF.
                // Raw quantized data is read as-is. For Transform (e.g. transpose),
                // the transform is applied after dequantization by GenericWeightManager.
                std::string hf_name = resolve_hf_name(
                    mspec->source.empty() ? spec.name : mspec->source, layer_idx);
                std::string data_hf = config.data_suffix.empty()
                    ? hf_name : (hf_name + config.data_suffix);
                std::string scale_hf = hf_name + config.scale_suffix;

                if (config.ep_size > 1) {
                    // EP: read only the local expert slice from the stacked HF tensor.
                    // HF has [E_total, ...], we need [E_local, ...] starting at ep_expert_start.
                    const int E_total = E * config.ep_size;
                    const int ep_expert_start = config.ep_rank * E;

                    // Data slice
                    const auto& data_entry = reader.find_entry(data_hf);
                    long hf_data_nelem = 1;
                    for (auto d : data_entry.shape()) hf_data_nelem *= d;
                    const long per_expert_data_elems = hf_data_nelem / E_total;
                    const long data_offset = static_cast<long>(ep_expert_start) * per_expert_data_elems;
                    const long data_count = static_cast<long>(E) * per_expert_data_elems;
                    if (data_count < qt.data.nelem()) {
                        // Zero-pad if quantizer allocated more (alignment)
                        CUDA_CHECK(cudaMemsetAsync(qt.data.Data, 0, qt.data.bytes(), stream));
                    }
                    data_entry.read_raw(qt.data, data_offset,
                                        std::min(data_count, static_cast<long>(qt.data.nelem())), true);

                    // Scale slice
                    const auto& scale_entry = reader.find_entry(scale_hf);
                    long hf_scale_nelem = 1;
                    for (auto d : scale_entry.shape()) hf_scale_nelem *= d;
                    const long per_expert_scale_elems = hf_scale_nelem / E_total;
                    const long scale_offset = static_cast<long>(ep_expert_start) * per_expert_scale_elems;
                    const long scale_count = static_cast<long>(E) * per_expert_scale_elems;
                    if (scale_count < qt.scales.nelem()) {
                        CUDA_CHECK(cudaMemsetAsync(qt.scales.Data, 0, qt.scales.bytes(), stream));
                    }
                    scale_entry.read_raw(qt.scales, scale_offset,
                                         std::min(scale_count, static_cast<long>(qt.scales.nelem())), true);
                } else {
                    read_raw_padded(reader.find_entry(data_hf), qt.data, stream);
                    read_raw_padded(reader.find_entry(scale_hf), qt.scales, stream);
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));
            } else {
                throw std::runtime_error(
                    "import_prequantized_weights: unsupported mapping kind for expert param '"
                    + spec.name + "' (expected Direct, Transform, or StackExperts)");
            }

            CUDA_CHECK(cudaStreamSynchronize(stream));
        } catch (...) {
            gpu_temp.free_all();
            throw;
        }

        // Read weight_scale_2 (NVFP4 global scale) for expert weights.
        // For StackExperts: load_prequant_experts_stacked handles per-component
        // scale2 with FP8 scale rescaling (scale2_handled=true). The old path
        // just used max() across experts which is incorrect when scale2 values
        // differ significantly or when fuse_gate_up combines two components.
        if (!config.scale2_suffix.empty() && qt.meta.Data != nullptr) {
            if (mspec->kind == dsl::MappingSpec::Kind::StackExperts && scale2_handled) {
                // Already handled inside load_prequant_experts_stacked —
                // qt.global_scale and meta are set, FP8 scales rescaled.
            } else if (mspec->kind == dsl::MappingSpec::Kind::StackExperts) {
                // Fallback (shouldn't happen for NVFP4, but keep for safety)
                float max_scale2 = 0.0f;
                for (int e = 0; e < E; ++e) {
                    std::string hf_name = resolve_hf_name(mspec->source, layer_idx, e);
                    std::string scale2_hf = hf_name + config.scale2_suffix;
                    const auto* entry = try_find_entry(reader, scale2_hf);
                    if (!entry) continue;

                    entry->read_raw(qt.meta, 0, 1, true);
                    float val;
                    CUDA_CHECK(cudaMemcpyAsync(&val, qt.meta.Data, sizeof(float),
                                               cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    max_scale2 = std::max(max_scale2, val);
                }
                qt.global_scale = max_scale2;
            } else {
                // Direct/Transform: single scale2 for the stacked tensor
                std::string hf_name = resolve_hf_name(
                    mspec->source.empty() ? spec.name : mspec->source, layer_idx);
                std::string scale2_hf = hf_name + config.scale2_suffix;
                const auto* entry = try_find_entry(reader, scale2_hf);
                if (entry) {
                    entry->read_raw(qt.meta, 0, 1, true);
                    CUDA_CHECK(cudaMemcpyAsync(&qt.global_scale, qt.meta.Data,
                                               sizeof(float),
                                               cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            }

            // Write final global_scale to meta tensor for consistency
            CUDA_CHECK(cudaMemcpyAsync(qt.meta.Data, &qt.global_scale,
                                       sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Relocate from GPU temp to pinned host for offloading.
        // After this, qt's data/scales/meta are on pinned host memory
        // and the GPU temp buffers are freed.
        if (offload_this) {
            relocate_qt_to_pinned(qt, gpu_temp, stream);
        }

        // Do NOT pass transform_fn: pre-quantized data is already in DSL layout
        // (e.g., MXFP4 gate_up_proj_blocks is stored as [E, MUp, C] even though
        // the BF16 gate_up_proj is [E, C, MUp] — the quantizer pre-transposes).
        weight_mgr->store_prequantized(
            spec.name, std::move(qt), spec.offload_group, spec.shape);
        loaded++;
    }

    // ---- Step 4: Resolve tied parameters ----

    loader.resolve_tied_params([&](const std::string& name) -> Tensor& {
        return weight_mgr->get(name, stream);
    });

    // ---- Step 5: Synchronize and report ----

    CUDA_CHECK(cudaStreamSynchronize(stream));

    return weight_mgr;
}

// =============================================================================
// External weight import pipeline (zero-copy from vLLM)
// =============================================================================

namespace {

/// Build a lookup map from HF weight name → ExternalWeight pointer.
std::unordered_map<std::string, const ExternalWeight*>
build_external_lookup(const std::vector<ExternalWeight>& external_weights) {
    std::unordered_map<std::string, const ExternalWeight*> lookup;
    lookup.reserve(external_weights.size());
    for (const auto& ew : external_weights) {
        lookup[ew.name] = &ew;
    }
    return lookup;
}

/// Parse a DSL block parameter name to extract the layer index.
/// e.g. "blocks[3].qkv_weight" → layer_idx=3, base="qkv_weight"
/// Returns true if parsed successfully.
bool parse_dsl_block_param(const std::string& name, int& layer_idx, std::string& base) {
    auto bracket = name.find("blocks[");
    if (bracket == std::string::npos) return false;
    auto close = name.find(']', bracket);
    if (close == std::string::npos) return false;
    auto dot = name.find('.', close);
    if (dot == std::string::npos) return false;

    auto idx_start = bracket + 7;  // length of "blocks["
    try {
        layer_idx = std::stoi(name.substr(idx_start, close - idx_start));
    } catch (...) {
        return false;
    }
    base = name.substr(dot + 1);
    return true;
}

/// Find the MappingSpec for an internal name from the mapping table.
/// Similar to DslWeightLoader::find_mapping_spec.
const dsl::MappingSpec* find_spec_in_table(
    const dsl::MappingTable& mapping,
    const std::string& internal_name,
    int& layer_idx) {

    layer_idx = -1;

    // Try exact match first
    auto it = mapping.find(internal_name);
    if (it != mapping.end()) return &it->second;

    // Try block pattern: "X_blocks[N].param" → "X_blocks[{layer}].param"
    std::string base;
    if (parse_dsl_block_param(internal_name, layer_idx, base)) {
        // Extract block prefix (e.g., "blocks", "moe_blocks", "attn_blocks")
        auto bracket = internal_name.find("blocks[");
        std::string prefix = internal_name.substr(0, bracket);
        std::string placeholder = prefix + "blocks[{layer}]." + base;
        it = mapping.find(placeholder);
        if (it != mapping.end()) return &it->second;

        // Try just the base name
        it = mapping.find(base);
        if (it != mapping.end()) return &it->second;
    }

    return nullptr;
}

}  // anonymous namespace

std::unique_ptr<GenericWeightManager> import_external_weights(
    const std::string& file_name,
    const std::vector<ExternalWeight>& external_weights,
    const DslQLoRAPipelineConfig& config,
    const PretrainedConfig& pt_config,
    std::shared_ptr<TensorAllocator> allocator,
    cudaStream_t stream) {

    auto t_start = std::chrono::steady_clock::now();

    // Build lookup from HF name → ExternalWeight
    auto ext_lookup = build_external_lookup(external_weights);

    // Create quantizer and weight manager (same as import_and_quantize_weights)
    auto quantizer = create_quantizer(config.quantizer_config);
    auto weight_mgr = std::make_unique<GenericWeightManager>(
        config.weight_manager_config,
        std::move(quantizer),
        allocator);

    // Open SafeTensors for non-quantized weights (norms, biases, embeddings)
    SafeTensorsReader reader(file_name);

    dsl::ShardConfig shard_config;
    shard_config.shard_idx = config.shard_idx;
    shard_config.num_shards = config.num_shards;

    dsl::MoEWeightConfig moe_config;
    moe_config.num_experts = config.num_experts;
    moe_config.moe_intermediate_size = config.moe_intermediate_size;

    dsl::DslWeightLoader loader(
        reader, config.mapping, pt_config, *allocator, shard_config, moe_config);

    int loaded_external = 0;
    int loaded_disk = 0;
    int skipped_non_quant = 0;
    const int total = static_cast<int>(config.weight_specs.size());
    std::vector<std::string> disk_loaded_names;

    // Fallback: load a weight from disk as full-precision (BF16).
    // Used for weights not found in the external lookup (e.g., embeddings, LM head
    // that vLLM keeps in full precision even in pre-quantized models).
    auto load_full_precision_from_disk = [&](const WeightLoadSpec& spec) -> bool {
        std::vector<long> shape;
        if (!spec.shape.empty()) {
            shape = spec.shape;
        } else if (spec.K > 0) {
            shape = {static_cast<long>(spec.M), static_cast<long>(spec.K)};
        } else {
            shape = {static_cast<long>(spec.M)};
        }

        Tensor tensor = allocator->allocate(
            spec.target_dtype,
            spec.name.c_str(),
            EAllocationType::ON_DEVICE,
            shape);

        bool success = loader.load_param(spec.name, tensor, true, spec.sharded, nullptr, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (success) {
            weight_mgr->register_full_precision(spec.name, tensor);
            loaded_disk++;
            disk_loaded_names.push_back(spec.name);
        }
        return success;
    };

    for (int i = 0; i < total; ++i) {
        const auto& spec = config.weight_specs[i];

        if (!spec.quantize) {
            // Non-quantized weight: load from disk
            load_full_precision_from_disk(spec);
            continue;
        }

        // Quantized weight: find in external lookup by resolving HF name(s)
        int layer_idx = -1;
        const dsl::MappingSpec* map_spec = find_spec_in_table(
            config.mapping, spec.name, layer_idx);

        if (!map_spec) {
            // No mapping found — fall back to disk loading
            load_full_precision_from_disk(spec);
            continue;
        }

        // For Direct mappings: look up the single HF name
        if (map_spec->kind == dsl::MappingSpec::Kind::Direct ||
            map_spec->kind == dsl::MappingSpec::Kind::Transform) {
            std::string hf_name = resolve_hf_name(map_spec->source, layer_idx);

            auto ext_it = ext_lookup.find(hf_name);
            if (ext_it == ext_lookup.end()) {
                // Not in external weights — fall back to disk loading
                // (common for embeddings, LM head in pre-quantized models)
                load_full_precision_from_disk(spec);
                continue;
            }
            const ExternalWeight& ew = *ext_it->second;

            // Build QuantizedTensor from GPU pointers
            QuantizedTensor qt;
            qt.format = ew.format;
            qt.M = ew.M;
            qt.K = ew.K;
            qt.block_size = ew.block_size;
            qt.double_quant = ew.double_quant;
            qt.double_quant_group_size = ew.double_quant_group_size;
            qt.global_scale = ew.global_scale;

            qt.data = Tensor::from_pointer(ew.data_ptr, ew.device, ew.data_dtype, ew.data_shape);
            qt.scales = Tensor::from_pointer(ew.scales_ptr, ew.device, ew.scales_dtype, ew.scales_shape);

            if (ew.meta_ptr) {
                qt.meta = Tensor::from_pointer(ew.meta_ptr, ew.device, ew.meta_dtype, ew.meta_shape);
            }
            if (ew.meta2_ptr) {
                qt.meta2 = Tensor::from_pointer(ew.meta2_ptr, ew.device, ew.meta2_dtype, ew.meta2_shape);
            }

            std::string transform_fn;
            if (map_spec->kind == dsl::MappingSpec::Kind::Transform) {
                transform_fn = map_spec->fn;
            }

            weight_mgr->store_prequantized(
                spec.name, std::move(qt), spec.offload_group, spec.shape, transform_fn);
            loaded_external++;

        } else if (map_spec->kind == dsl::MappingSpec::Kind::StackExperts) {
            // Expert weights: vLLM may store stacked [E, M, K] under a single HF name
            std::string hf_name = resolve_hf_name(map_spec->source, layer_idx);

            auto ext_it = ext_lookup.find(hf_name);
            if (ext_it != ext_lookup.end()) {
                // Found stacked weight — store directly
                const ExternalWeight& ew = *ext_it->second;

                QuantizedTensor qt;
                qt.format = ew.format;
                qt.M = ew.M;
                qt.K = ew.K;
                qt.block_size = ew.block_size;
                qt.double_quant = ew.double_quant;
                qt.double_quant_group_size = ew.double_quant_group_size;
                qt.global_scale = ew.global_scale;

                qt.data = Tensor::from_pointer(ew.data_ptr, ew.device, ew.data_dtype, ew.data_shape);
                qt.scales = Tensor::from_pointer(ew.scales_ptr, ew.device, ew.scales_dtype, ew.scales_shape);
                if (ew.meta_ptr) {
                    qt.meta = Tensor::from_pointer(ew.meta_ptr, ew.device, ew.meta_dtype, ew.meta_shape);
                }
                if (ew.meta2_ptr) {
                    qt.meta2 = Tensor::from_pointer(ew.meta2_ptr, ew.device, ew.meta2_dtype, ew.meta2_shape);
                }

                weight_mgr->store_prequantized(
                    spec.name, std::move(qt), spec.offload_group, spec.shape);
                loaded_external++;
            } else {
                // Not in external weights — fall back to disk loading
                load_full_precision_from_disk(spec);
            }

        } else if (map_spec->kind == dsl::MappingSpec::Kind::Fuse) {
            // Fused weight (e.g., QKV): vLLM may already have the fused version.
            // Try each source name in the mapping.
            bool found = false;
            for (const auto& src : map_spec->sources) {
                std::string hf_name = resolve_hf_name(src, layer_idx);
                auto ext_it = ext_lookup.find(hf_name);
                if (ext_it != ext_lookup.end()) {
                    // Found — this must be the fused weight from vLLM
                    const ExternalWeight& ew = *ext_it->second;

                    QuantizedTensor qt;
                    qt.format = ew.format;
                    qt.M = ew.M;
                    qt.K = ew.K;
                    qt.block_size = ew.block_size;
                    qt.double_quant = ew.double_quant;
                    qt.double_quant_group_size = ew.double_quant_group_size;
                    qt.global_scale = ew.global_scale;

                    qt.data = Tensor::from_pointer(ew.data_ptr, ew.device, ew.data_dtype, ew.data_shape);
                    qt.scales = Tensor::from_pointer(ew.scales_ptr, ew.device, ew.scales_dtype, ew.scales_shape);
                    if (ew.meta_ptr) {
                        qt.meta = Tensor::from_pointer(ew.meta_ptr, ew.device, ew.meta_dtype, ew.meta_shape);
                    }
                    if (ew.meta2_ptr) {
                        qt.meta2 = Tensor::from_pointer(ew.meta2_ptr, ew.device, ew.meta2_dtype, ew.meta2_shape);
                    }

                    // Compute fuse_swap_at: when the external weight has swapped
                    // partition order, swap the two equal halves after dequant.
                    int fuse_swap_at = 0;
                    if (ew.fuse_swap) {
                        fuse_swap_at = ew.M / 2;
                    }

                    weight_mgr->store_prequantized(
                        spec.name, std::move(qt), spec.offload_group, spec.shape,
                        /*transform_fn=*/"", fuse_swap_at);
                    loaded_external++;
                    found = true;
                    break;
                }
            }
            if (!found) {
                // Not in external weights — fall back to disk loading
                load_full_precision_from_disk(spec);
            }

        } else {
            // Unsupported mapping kind — fall back to disk loading
            load_full_precision_from_disk(spec);
        }
    }

    // Resolve tied parameters
    loader.resolve_tied_params([&](const std::string& name) -> Tensor& {
        return weight_mgr->get(name, stream);
    });

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    fmt::print("[external import] Loaded {} weights from GPU pointers, "
               "{} from disk in {:.1f} ms\n",
               loaded_external, loaded_disk, elapsed_ms);
    if (!disk_loaded_names.empty()) {
        fmt::print("[external import] Disk-loaded: ");
        for (size_t i = 0; i < disk_loaded_names.size() && i < 20; ++i) {
            fmt::print("{}{}", (i > 0 ? ", " : ""), disk_loaded_names[i]);
        }
        if (disk_loaded_names.size() > 20) {
            fmt::print(" ... and {} more", disk_loaded_names.size() - 20);
        }
        fmt::print("\n");
    }

    const size_t quant_bytes = weight_mgr->quantized_bytes();
    const size_t fp_bytes = weight_mgr->full_precision_bytes();
    fmt::print("[external import] Memory: quantized={:.1f} MB (borrowed), "
               "full_precision={:.1f} MB (from disk)\n",
               static_cast<double>(quant_bytes) / (1024.0 * 1024.0),
               static_cast<double>(fp_bytes) / (1024.0 * 1024.0));

    return weight_mgr;
}

// =============================================================================
// Config builder
// =============================================================================

DslQLoRAPipelineConfig build_pipeline_config(
    const dsl::MappingTable& mapping,
    const std::vector<WeightLoadSpec>& weight_specs,
    const QuantizerConfig& quantizer_config,
    int shard_idx,
    int num_shards) {

    DslQLoRAPipelineConfig config;
    config.mapping = mapping;
    config.weight_specs = weight_specs;
    config.quantizer_config = quantizer_config;
    config.shard_idx = shard_idx;
    config.num_shards = num_shards;

    // Default weight manager config
    config.weight_manager_config.device_id = quantizer_config.device_id;

    return config;
}

}  // namespace qlora
