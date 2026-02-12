// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <nanobind/ndarray.h>

#include <filesystem>
#include <cstring>
#include <fmt/format.h>

#include "py_train.h"
#include "runtime/training/dataloader.h"
#include "runtime/training/checkpoint.h"
#include "runtime/training/logging.h"
#include "runtime/training/matmul_backend.h"
#include "utilities/gpu_info.h"
#include "utilities/safetensors.h"
#include "utilities/sol.h"
#include "utilities/comm.h"
#include "utilities/crash_handler.h"
#include "config/lora_adapter_config.h"
#include "recipes/recipe_factory.h"
#include "runtime/qlora/qlora_config.h"

namespace nb = nanobind;

using TokenArray = nb::ndarray<std::int32_t, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;
using TokenArray3 = nb::ndarray<std::int32_t, nb::shape<-1, -1, -1>, nb::c_contig, nb::device::cpu>;
using FloatArray = nb::ndarray<float, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;

static std::optional<ETensorDType> opt_dtype_from_str(const std::string& dtype_str) {
    if (dtype_str.empty()) {
        return std::nullopt;
    }
    return dtype_from_str(dtype_str);
}

static EMatmulBackend matmul_backend_from_str(const std::string& backend_str) {
    if (backend_str.empty() || backend_str == "auto") {
        return EMatmulBackend::AUTO;
    } else if (backend_str == "cudnn") {
        return EMatmulBackend::CUBLASLT;
    } else if (backend_str == "cutlass") {
        return EMatmulBackend::CUTLASS;
    }
    throw std::runtime_error("Unknown matmul backend: " + backend_str + " (valid: auto, cudnn, cutlass)");
}

static std::string matmul_backend_to_str(EMatmulBackend backend) {
    switch (backend) {
        case EMatmulBackend::AUTO: return "auto";
        case EMatmulBackend::CUBLASLT: return "cudnn";
        case EMatmulBackend::CUTLASS: return "cutlass";
    }
    return "auto";
}

static nb::object cast_opt_dtype(std::optional<ETensorDType> dtype) {
    if (dtype.has_value()) {
        return nb::cast(dtype_to_str(dtype.value()));
    }
    return nb::none();
}

template<typename NBArray, std::size_t NDims>
static inline auto check_shape(const NBArray& arr, std::string_view name, std::array<int, NDims> expected) {
    if(arr.ndim() != expected.size()) {
        throw std::runtime_error(fmt::format("Expected {} to have {} dimensions, but got {}", name, expected.size(), arr.ndim()));
    }
    for(int dim = 0; dim < expected.size(); ++dim) {
        if (arr.shape(dim) != expected[dim]) {
            throw std::runtime_error(
                fmt::format("Expected {} to have extent {} at dimension {}, but got {}", name, expected[dim], dim,
                            arr.shape(dim)));
        }
    }
}

nb::dlpack::dtype to_dlpack_dtype(ETensorDType dtype) {
    switch (dtype) {
    case ETensorDType::FP32:
        return {static_cast<std::uint8_t>(nb::dlpack::dtype_code::Float), 32, 1};
    case ETensorDType::BF16:
        return {static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
    case ETensorDType::INT8:
        return {static_cast<std::uint8_t>(nb::dlpack::dtype_code::Int), 8, 1};
    case ETensorDType::BYTE:
        return {static_cast<std::uint8_t>(nb::dlpack::dtype_code::UInt), 8, 1};
    case ETensorDType::FP16:
        return {static_cast<std::uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
    case ETensorDType::INT32:
        return {static_cast<std::uint8_t>(nb::dlpack::dtype_code::Int), 32, 1};
    case ETensorDType::FP8_E4M3:
        return {static_cast<std::uint8_t>(nb::dlpack::dtype_code::UInt), 8, 1};  // ugh
    }
    throw std::runtime_error("Unsupported ETensorDType for DLPack export");
}

#define CHECK_SHAPE(obj, ...) check_shape(obj, #obj, std::array{__VA_ARGS__})


NB_MODULE(_surogate, m) {
    // Install crash handler for better stack traces on segfaults and other crashes
    surogate::install_crash_handler();

    nb::class_<GPUUtilInfo>(m, "GPUUtilInfo",
        "Snapshot of GPU utilization/telemetry.\n\n"
        "All fields are read/write and represent the most recently sampled values.\n"
        "Units are implementation-defined; typically MHz for clocks, W for power, C for temperatures, "
        "and bytes for memory counters.")
        .def_rw("clock", &GPUUtilInfo::clock, "Current GPU clock (typically MHz).")
        .def_rw("max_clock", &GPUUtilInfo::max_clock, "Maximum GPU clock (typically MHz).")
        .def_rw("power", &GPUUtilInfo::power, "Current GPU power draw (typically W).")
        .def_rw("power_limit", &GPUUtilInfo::power_limit, "Configured power limit (typically W).")
        .def_rw("fan", &GPUUtilInfo::fan, "Fan speed (typically percent).")
        .def_rw("temperature", &GPUUtilInfo::temperature, "GPU temperature (typically Celsius).")
        .def_rw("temp_slowdown", &GPUUtilInfo::temp_slowdown, "Thermal slowdown threshold (typically Celsius).")
        .def_rw("mem_free", &GPUUtilInfo::mem_free, "Free device memory (bytes).")
        .def_rw("mem_total", &GPUUtilInfo::mem_total, "Total device memory (bytes).")
        .def_rw("mem_reserved", &GPUUtilInfo::mem_reserved, "Reserved device memory (bytes).")
        .def_rw("gpu_utilization", &GPUUtilInfo::gpu_utilization, "GPU utilization (typically percent).")
        .def_rw("mem_utilization", &GPUUtilInfo::mem_utilization, "Memory utilization (typically percent).")
        .def_rw("throttle_reason", &GPUUtilInfo::throttle_reason, "Vendor-specific throttling reason bitmask/string code.")
        .def_rw("pcie_rx", &GPUUtilInfo::pcie_rx, "PCIe receive throughput (implementation-defined units).")
        .def_rw("pcie_tx", &GPUUtilInfo::pcie_tx, "PCIe transmit throughput (implementation-defined units).")
        .def("__repr__", [](const GPUUtilInfo& gpu_util) {
            return fmt::format(
                R"(GPUUtilInfo(clock={}, max_clock={}, fan={}, power={}, power_limit={}, temperature={}, temp_slowdown={}, gpu_util={}, mem_util={}, throttle={}, dram_free={}, dram_total={}, dram_reserved={}, pcie_rx={}, pcie_tx={}))",
                                 gpu_util.clock, gpu_util.max_clock, gpu_util.fan, gpu_util.power, gpu_util.power_limit, gpu_util.temperature, gpu_util.temp_slowdown,
                                 gpu_util.gpu_utilization, gpu_util.mem_utilization, gpu_util.throttle_reason, gpu_util.mem_free, gpu_util.mem_total, gpu_util.mem_reserved,
                                 gpu_util.pcie_rx, gpu_util.pcie_tx);
        }, "Return a debug string representation.")
        ;

    nb::class_<GPUInfo>(m, "GPUInfo",
        "Information about a single GPU device.")
        .def_rw("device_id", &GPUInfo::device_id, "Device ID (0-indexed).")
        .def_rw("name", &GPUInfo::name, "Device name.")
        .def_rw("total_memory", &GPUInfo::total_memory, "Total device memory (bytes).")
        .def_rw("compute_capability_major", &GPUInfo::compute_capability_major, "Compute capability major version.")
        .def_rw("compute_capability_minor", &GPUInfo::compute_capability_minor, "Compute capability minor version.")
        .def("__repr__", [](const GPUInfo& info) {
            return fmt::format(
                R"(GPUInfo(device_id={}, name='{}', total_memory={}, compute_capability={}.{}))",
                info.device_id, info.name, info.total_memory,
                info.compute_capability_major, info.compute_capability_minor);
        }, "Return a debug string representation.")
        ;

    nb::class_<SystemInfo>(m, "SystemInfo",
        "System information utility class.\n\n"
        "Provides methods to query CUDA, NCCL, cuDNN versions and GPU information.")
        .def_static("get_cuda_driver_version", &SystemInfo::get_cuda_driver_version,
            "Get CUDA driver version.\n\n"
            "Returns: Integer version (e.g., 12010 for CUDA 12.1).")
        .def_static("get_cuda_runtime_version", &SystemInfo::get_cuda_runtime_version,
            "Get CUDA runtime version.\n\n"
            "Returns: Integer version (e.g., 12010 for CUDA 12.1).")
        .def_static("get_nccl_version", &SystemInfo::get_nccl_version,
            "Get NCCL version.\n\n"
            "Returns: Integer version (e.g., 2180 for NCCL 2.18.0).")
        .def_static("get_cudnn_version", &SystemInfo::get_cudnn_version,
            "Get cuDNN version.\n\n"
            "Returns: Integer version (e.g., 8906 for cuDNN 8.9.6).")
        .def_static("get_gpu_info", &SystemInfo::get_gpu_info,
            "Get information about all available GPUs.\n\n"
            "Returns: List of GPUInfo objects.")
        ;

    nb::class_<PretrainedConfig>(m, "PretrainedConfig",
        "Model configuration used to build/initialize a transformer.\n\n"
        "Notes:\n"
        "- Some defaults depend on `architecture`.\n"
        "- `dtype` controls the model's compute/storage type where applicable.\n\n"
        "Backwards-compatibility: `LLamaConfig` is an alias of this class.")
        .def("__init__", [](PretrainedConfig *t,
            const std::string& arch, std::optional<int> bos_token_id, std::optional<int> eos_token_id,
            int hidden_size, int intermediate_size, std::optional<int> vocab_size, int num_attention_heads, int num_key_value_heads,
            int num_hidden_layers, std::optional<int> max_position_embeddings, std::optional<float> rope_theta, float rms_norm_eps, bool tie_word_embeddings, std::optional<bool> use_qkv_bias, std::string dtype) {
            // default values depend on selected architecture
            PretrainedConfig::ArchitectureId architecture;
            if(arch == "qwen2" || arch == "Qwen2" || arch == "Qwen2ForCausalLM") {
                architecture = PretrainedConfig::QWEN2;
                eos_token_id = eos_token_id.value_or(151645);
                bos_token_id = bos_token_id.value_or(151643);
                vocab_size = vocab_size.value_or(151936);
                max_position_embeddings = max_position_embeddings.value_or(32768);
                rope_theta = rope_theta.value_or(1000000.0);
                use_qkv_bias = use_qkv_bias.value_or(true);
            } else {
                throw std::runtime_error("At this point, only qwen2 architecture is supported.");
            }

            new (t) PretrainedConfig();
            t->Architecture = architecture;
            t->BosTokenId = bos_token_id.value();
            t->EosTokenId = eos_token_id.value();
            t->PadTokenId = bos_token_id.value();
            t->HiddenSize = hidden_size;
            t->IntermediateSize = intermediate_size;
            t->VocabSize = vocab_size.value();
            t->NumQueryHeads = num_attention_heads;
            t->NumKeyValHeads = num_key_value_heads;
            t->NumLayers = num_hidden_layers;
            t->MaxPositionEmbeddings = max_position_embeddings.value();
            t->RopeTheta = rope_theta.value();
            t->RmsNormEps = rms_norm_eps;
            t->TiedWordEmbeddings = tie_word_embeddings;
            t->UseQKVBias = use_qkv_bias.value();
            t->DType = dtype_from_str(dtype);
        }, nb::kw_only(),
             nb::arg("architecture"),
             nb::arg("bos_token_id") = nb::none(),
             nb::arg("eos_token_id") = nb::none(),
             nb::arg("hidden_size"),
             nb::arg("intermediate_size"),
             nb::arg("vocab_size") = nb::none(),
             nb::arg("num_attention_heads"),
             nb::arg("num_key_value_heads"),
             nb::arg("num_hidden_layers"),
             nb::arg("max_position_embeddings") = nb::none(),
             nb::arg("rope_theta") = nb::none(),
             nb::arg("rms_norm_eps"),
             nb::arg("tie_word_embeddings"),
             nb::arg("use_qkv_bias") = nb::none(),
             nb::arg("dtype") = "bf16",
             "Create a model configuration.\n\n"
             "Parameters:\n"
             "- architecture: Model family identifier (currently: qwen2).\n"
             "- bos_token_id/eos_token_id: Token IDs; if None, architecture defaults are used.\n"
             "- hidden_size/intermediate_size: Transformer dimensions.\n"
             "- vocab_size: Vocabulary size; if None, architecture default is used.\n"
             "- num_attention_heads/num_key_value_heads: Attention head counts.\n"
             "- num_hidden_layers: Number of transformer blocks.\n"
             "- max_position_embeddings: Max sequence length; if None, architecture default is used.\n"
             "- rope_theta: RoPE base; if None, architecture default is used.\n"
             "- rms_norm_eps: Epsilon for RMSNorm.\n"
             "- tie_word_embeddings: Whether input/output embeddings are tied.\n"
             "- use_qkv_bias: Whether QKV projections use bias; if None, architecture default is used.\n"
             "- dtype: Tensor dtype string (e.g. 'bf16', 'fp16', 'fp32').")
        .def_rw("architecture", &PretrainedConfig::Architecture, "Architecture identifier (enum-backed).")
        .def_rw("bos_token_id", &PretrainedConfig::BosTokenId, "Beginning-of-sequence token id.")
        .def_rw("eos_token_id", &PretrainedConfig::EosTokenId, "End-of-sequence token id.")
        .def_rw("hidden_size", &PretrainedConfig::HiddenSize, "Transformer hidden size.")
        .def_rw("intermediate_size", &PretrainedConfig::IntermediateSize, "FFN intermediate size.")
        .def_rw("vocab_size", &PretrainedConfig::VocabSize, "Vocabulary size.")
        .def_rw("num_attention_heads", &PretrainedConfig::NumQueryHeads, "Number of query attention heads.")
        .def_rw("num_key_value_heads", &PretrainedConfig::NumKeyValHeads, "Number of key/value attention heads (for GQA/MQA).")
        .def_rw("num_hidden_layers", &PretrainedConfig::NumLayers, "Number of transformer layers/blocks.")
        .def_rw("max_position_embeddings", &PretrainedConfig::MaxPositionEmbeddings, "Maximum supported sequence length.")
        .def_rw("rope_theta", &PretrainedConfig::RopeTheta, "RoPE base parameter (theta).")
        .def_rw("rms_norm_eps", &PretrainedConfig::RmsNormEps, "Epsilon used in RMSNorm.")
        .def_rw("tie_word_embeddings", &PretrainedConfig::TiedWordEmbeddings, "Whether input/output embeddings are tied.")
        .def_rw("use_qkv_bias", &PretrainedConfig::UseQKVBias, "Whether QKV projections use bias.")
        .def_prop_rw("dtype",
                     [](const PretrainedConfig* cfg){ return dtype_to_str(cfg->DType); },
                     [](PretrainedConfig* cfg, const std::string& dtype_str){ cfg->DType = dtype_from_str(dtype_str); },
                     "Model dtype as a string (e.g. 'bf16', 'fp16', 'fp32').")
        .def_prop_ro("head_size", &PretrainedConfig::head_size, "Attention head size (= hidden_size / num_attention_heads).")
        .def_prop_ro("qkv_channels", &PretrainedConfig::qkv_channels, "Total QKV channel count used internally.")
        .def_prop_ro("model_name", &PretrainedConfig::model_name, "Canonical model name derived from the configuration.")
        .def_static("from_pretrained", [](const std::string& name, const std::string& dtype_str) -> PretrainedConfig*
        {
            std::string hf_path = get_hf_model_files(name);
            if (hf_path.empty()) {
                throw std::runtime_error("Could not find model files for " + name);
            }
            std::string config_path = hf_path + "/config.json";
            auto cfg = load_pretrained_config(config_path.c_str(), dtype_from_str(dtype_str));
            return cfg.release();  // Transfer ownership of polymorphic object
        }, nb::arg("name"), nb::arg("dtype"),
           "Load a config.json from a HuggingFace model directory.\n\n"
           "Parameters:\n"
           "- name: HuggingFace model name or local cache key.\n"
           "- dtype: Override dtype for the loaded configuration.\n\n"
           "Returns: PretrainedConfig")
        .def_static("from_name", [](const std::string& name, const std::string& dtype_str) -> PretrainedConfig*
        {
            auto cfg = create_pretrained_config_from_name(name, dtype_from_str(dtype_str));
            if (!cfg) {
                throw std::runtime_error("Unknown model name: " + name);
            }
            return cfg.release();  // Transfer ownership of polymorphic object
        }, nb::arg("name"), nb::arg("dtype"),
           "Load a configuration from a config.json path or directory.\n\n"
           "Parameters:\n"
           "- name: Path to config.json or a directory containing it.\n"
           "- dtype: Desired dtype.\n\n"
           "Returns: PretrainedConfig")
        ;

    nb::class_<RuntimeOptions>(m, "RuntimeOptions",
        "Execution/training options controlling recomputation, offloading, sharding, and dtypes.\n\n"
        "Recomputation trades compute for memory. Offloading trades host/device transfers for VRAM.\n\n"
        "Backwards-compatibility: `LLamaOptions` is an alias of this class.")
        .def("__init__", [](RuntimeOptions *t, const std::string& recompute, bool offload_residual,
            bool use_cuda_graphs, bool trigger_timing_events,
            bool offload_master, bool offload_quants, bool offload_optimizer, bool offload_grads, bool use_zero_copy,
            bool use_write_combined, bool shard_weights, bool persistent_quants, bool shard_gradients, bool use_all_to_all_reduce,
            bool init_projections_to_zero, bool debug_memory_breakdown, int lmhead_chunks, int attn_bwd_chunks,
            const std::string matmul_type, const std::string gradient_type, const std::string master_dtype,
            const std::string& recipe, const std::string& matmul_backend, bool use_fused_rope,
            int fp8_amax_history, const std::string& fp4_backend,
            int skip_quant_first_layers, int skip_quant_last_layers,
            bool use_dsl_ir) {

            // Build recipe options
            recipes::RecipeConfig recipe_options;
            recipe_options.fp8_amax_history_len = fp8_amax_history;
            recipe_options.fp4_backend = matmul_backend_from_str(fp4_backend);
            recipe_options.skip_quant_first_layers = skip_quant_first_layers;
            recipe_options.skip_quant_last_layers = skip_quant_last_layers;

            // Create the training recipe
            auto training_recipe = recipes::RecipeFactory::create(recipe.empty() ? "bf16" : recipe, recipe_options);
            EMatmulBackend backend = training_recipe->matmul_backend();
            if (!matmul_backend.empty()) {
                backend = matmul_backend_from_str(matmul_backend);
            }

            // Parse recompute level
            RecomputeLevel recompute_level = RuntimeOptions::parse_recompute_level(recompute);

            new (t) RuntimeOptions{
                .Recompute = recompute_level,
                .OffloadResidual = offload_residual,
                .LMHeadChunks = lmhead_chunks,
                .AttBwdChunks = attn_bwd_chunks,
                .UseCudaGraphs = use_cuda_graphs,
                .TriggerTimingEvents = trigger_timing_events,
                .OffloadMaster = offload_master,
                .OffloadQuants = offload_quants,
                .OffloadOptimizer = offload_optimizer,
                .OffloadGrads = offload_grads,
                .UseZeroCopy = use_zero_copy,
                .UseWriteCombined = use_write_combined,
                .ShardWeights = shard_weights,
                .PersistentQuants = persistent_quants,
                .ShardGradients = shard_gradients,
                .UseAllToAllReduce = use_all_to_all_reduce,
                .InitProjectionsToZero = init_projections_to_zero,
                .DebugMemoryBreakdown = debug_memory_breakdown,
                .TrainingRecipe = std::move(training_recipe),
                .RecipeOptions = recipe_options,
                .UseFusedRope = use_fused_rope,
                .UseDslIr = use_dsl_ir,
                .MatmulBackend = backend,
                .MatmulType = opt_dtype_from_str(matmul_type),
                .GradientType = opt_dtype_from_str(gradient_type),
                .MasterDType = opt_dtype_from_str(master_dtype)
            };
        }, nb::kw_only(),
             nb::arg("recompute") = "true",
             nb::arg("offload_residual") = false,
             nb::arg("use_cuda_graphs") = true,
             nb::arg("trigger_timing_events") = false,
             nb::arg("offload_master") = false,
             nb::arg("offload_quants") = false,
             nb::arg("offload_optimizer") = false,
             nb::arg("offload_grads") = false,
             nb::arg("use_zero_copy") = false,
             nb::arg("use_write_combined") = false,
             nb::arg("shard_weights") = false,
             nb::arg("persistent_quants") = false,
             nb::arg("shard_gradients") = false,
             nb::arg("use_all_to_all_reduce") = false,
             nb::arg("init_projections_to_zero") = false,
             nb::arg("debug_memory_breakdown") = false,
             nb::arg("lmhead_chunks") = 1,
             nb::arg("attn_bwd_chunks") = 1,
             nb::arg("matmul_type") = "",
             nb::arg("gradient_type") = "",
             nb::arg("master_dtype") = "",
             nb::arg("recipe") = "bf16",
             nb::arg("matmul_backend") = "",
             nb::arg("use_fused_rope") = false,
             nb::arg("fp8_amax_history") = 1024,
             nb::arg("fp4_backend") = "cutlass",
             nb::arg("skip_quant_first_layers") = 0,
             nb::arg("skip_quant_last_layers") = 0,
             nb::arg("use_dsl_ir") = true,
             "Create runtime/training options.\n\n"
             "Parameters:\n"
             "- recompute: Enable activation recomputation ('true' or 'false').\n"
             "  - 'false': Save all activations. Maximum memory, fastest training.\n"
             "  - 'true': Recompute intermediates from checkpoints. Saves ~17% VRAM.\n"
             "- offload_*: Offload specific buffers/states; may reduce VRAM at performance cost.\n"
             "- use_cuda_graphs: Enable CUDA graphs where supported.\n"
             "- trigger_timing_events: Log additional timing information.\n"
             "- shard_*: Enable sharding of weights/gradients across GPUs.\n"
             "- use_all_to_all_reduce: Use all-to-all based reduction (if supported by backend).\n"
             "- *_type/master_dtype: Dtype strings (empty means default/auto for optional fields).\n"
             "- recipe: Training recipe (bf16, fp8-hybrid, nvfp4, nvfp4-quartet).\n"
             "- matmul_backend: Matmul backend (auto, cublaslt, cutlass).\n"
             "- use_fused_rope: Use fused RoPE kernel with on-the-fly cos/sin computation.\n"
             "- fp8_amax_history: FP8 delayed scaling amax history length (for fp8-hybrid recipe).\n"
             "- fp4_backend: FP4 matmul backend (cudnn, cutlass).\n"
             "- skip_quant_first_layers: Skip quantization for first N layers.\n"
             "- skip_quant_last_layers: Skip quantization for last N layers.\n"
             "- use_dsl_ir: Deprecated (DSL backend is always enabled).")
        .def_prop_rw("recompute",
            [](const RuntimeOptions& self) { return std::string(self.recompute_level_name()); },
            [](RuntimeOptions& self, const std::string& level) { self.Recompute = RuntimeOptions::parse_recompute_level(level); },
            "Enable activation recomputation: 'true' or 'false'.")
        .def_rw("offload_residual", &RuntimeOptions::OffloadResidual, "Offload residual stream buffers.")
        .def_rw("lmhead_chunks", &RuntimeOptions::LMHeadChunks, "Split LM head computation into this many chunks.")
        .def_rw("attn_bwd_chunks", &RuntimeOptions::AttBwdChunks, "Split attention backward into this many chunks.")
        .def_rw("use_cuda_graphs", &RuntimeOptions::UseCudaGraphs, "Enable CUDA graphs for steady-state execution.")
        .def_rw("trigger_timing_events", &RuntimeOptions::TriggerTimingEvents, "Log additional timing information.")
        .def_rw("offload_master", &RuntimeOptions::OffloadMaster, "Offload FP32 master weights (optimizer state).")
        .def_rw("offload_quants", &RuntimeOptions::OffloadQuants, "Offload quantized weights (if applicable).")
        .def_rw("offload_optimizer", &RuntimeOptions::OffloadOptimizer, "Offload optimizer state (momentum and variance buffers).")
        .def_rw("offload_grads", &RuntimeOptions::OffloadGrads, "Offload gradients.")
        .def_rw("offload_experts", &RuntimeOptions::OffloadExperts, "Offload MoE expert NF4 weights to CPU, stream on-demand.")
        .def_rw("selective_expert_dequant", &RuntimeOptions::SelectiveExpertDequant, "Only dequantize router-selected experts (reduces memory).")
        .def_rw("use_zero_copy", &RuntimeOptions::UseZeroCopy, "Use zero-copy buffers where supported.")
        .def_rw("use_write_combined", &RuntimeOptions::UseWriteCombined, "Use write-combined host memory (pinned).")
        .def_rw("shard_weights", &RuntimeOptions::ShardWeights, "Shard model weights across GPUs.")
        .def_rw("persistent_quants", &RuntimeOptions::PersistentQuants, "Keep quant buffers persistent across steps.")
        .def_rw("shard_gradients", &RuntimeOptions::ShardGradients, "Shard gradients across GPUs.")
        .def_rw("use_all_to_all_reduce", &RuntimeOptions::UseAllToAllReduce, "Use all-to-all reduce strategy when reducing gradients.")
        .def_rw("init_projections_to_zero", &RuntimeOptions::InitProjectionsToZero, "Initialize certain projections to zero (for experiments).")
        .def_rw("debug_memory_breakdown", &RuntimeOptions::DebugMemoryBreakdown, "Print detailed memory breakdown after model allocation.")
        .def_rw("use_fused_rope", &RuntimeOptions::UseFusedRope, "Use fused RoPE kernel with on-the-fly cos/sin computation.")
        .def_rw("use_dsl_ir", &RuntimeOptions::UseDslIr, "Deprecated (DSL backend is always enabled).")
        .def_rw("dsl_ir_json", &RuntimeOptions::DslIrJson, "DSL IR JSON payload (generated at compile time).")
        .def_rw("router_aux_loss_coef", &RuntimeOptions::RouterAuxLossCoef, "MoE aux loss coefficient (-1 = use model config).")
        .def_rw("router_z_loss_coef", &RuntimeOptions::RouterZLossCoef, "MoE z-loss coefficient (-1 = use model config).")
        .def_prop_rw("matmul_type", [](const RuntimeOptions* opt){ return opt->matmul_dtype(); },
                     [](RuntimeOptions* opt, const std::string& dtype_str){ opt->MatmulType = opt_dtype_from_str(dtype_str); },
                     "Optional override dtype for matmul kernels (empty/None means default).")
        .def_prop_rw("gradient_type", [](const RuntimeOptions* opt){ return opt->grad_dtype(); },
                     [](RuntimeOptions* opt, const std::string& dtype_str){ opt->GradientType = opt_dtype_from_str(dtype_str); },
                     "Optional override dtype for gradient computations (empty/None means default).")
        .def_prop_rw("master_dtype", [](const RuntimeOptions* opt){ return cast_opt_dtype(opt->MasterDType); },
                     [](RuntimeOptions* opt, const std::string& dtype_str){ opt->MasterDType = opt_dtype_from_str(dtype_str); },
                     "Optional override dtype for master weights (empty/None means default).")
        .def_prop_rw("matmul_backend",
                     [](const RuntimeOptions* opt){ return matmul_backend_to_str(opt->MatmulBackend); },
                     [](RuntimeOptions* opt, const std::string& backend_str){ opt->MatmulBackend = matmul_backend_from_str(backend_str); },
                     "Matmul backend (auto, cublaslt, cutlass).")
        .def_prop_ro("recipe_name", [](const RuntimeOptions* opt){ return std::string(opt->recipe_name()); },
                     "Current training recipe name.")
        .def_prop_ro("fp8_enabled", [](const RuntimeOptions* opt){ return opt->fp8_forward_enabled(); },
                     "Whether FP8 forward pass is enabled.")
        .def_prop_ro("fp4_enabled", &RuntimeOptions::fp4_enabled,
                     "Whether FP4 training is enabled.")
        .def("set_recipe", [](RuntimeOptions* opt, const std::string& recipe_name) {
            opt->TrainingRecipe = recipes::RecipeFactory::create(recipe_name, opt->RecipeOptions);
            opt->MatmulBackend = opt->TrainingRecipe->matmul_backend();
        }, nb::arg("recipe_name"),
           "Set training recipe by name (bf16, fp8-hybrid, nvfp4).")
        ;

    nb::class_<LoRAAdapterConfig>(m, "LoRAAdapterConfig",
        "LoRA (Low-Rank Adaptation) adapter configuration.\n\n"
        "Controls which modules receive LoRA adapters and with which rank/scaling/dtype.\n\n"
        "Backwards-compatibility: `LoRAConfig` is an alias of this class.")
        .def("__init__", [](LoRAAdapterConfig *t, int rank, float alpha, float dropout,
                           const std::vector<std::string>& target_modules, const std::string& dtype,
                           bool use_rslora, bool train_router) {
            new (t) LoRAAdapterConfig{
                .Rank = rank,
                .Alpha = alpha,
                .Dropout = dropout,
                .TargetModules = std::set<std::string>(target_modules.begin(), target_modules.end()),
                .DType = dtype.empty() ? ETensorDType::BF16 : dtype_from_str(dtype),
                .UseRSLoRA = use_rslora,
                .TrainRouter = train_router
            };
        }, nb::kw_only(),
             nb::arg("rank") = 8,
             nb::arg("alpha") = 16.0f,
             nb::arg("dropout") = 0.0f,
             nb::arg("target_modules") = std::vector<std::string>{"q_proj", "k_proj", "v_proj", "o_proj"},
             nb::arg("dtype") = "bf16",
             nb::arg("use_rslora") = false,
             nb::arg("train_router") = false,
             "Create a LoRA configuration.\n\n"
             "Parameters:\n"
             "- rank: LoRA rank.\n"
             "- alpha: LoRA alpha (scaling numerator).\n"
             "- dropout: LoRA dropout probability.\n"
             "- target_modules: Module name suffixes to apply LoRA to.\n"
             "- dtype: Adapter dtype.\n"
             "- use_rslora: Enable RS-LoRA scaling variant.\n"
             "- train_router: Train MoE router gate during LoRA fine-tuning.")
        .def_rw("rank", &LoRAAdapterConfig::Rank, "LoRA rank.")
        .def_rw("alpha", &LoRAAdapterConfig::Alpha, "LoRA alpha.")
        .def_rw("dropout", &LoRAAdapterConfig::Dropout, "LoRA dropout probability.")
        .def_rw("use_rslora", &LoRAAdapterConfig::UseRSLoRA, "Whether to use RS-LoRA variant.")
        .def_rw("train_router", &LoRAAdapterConfig::TrainRouter, "Train MoE router gate during LoRA fine-tuning.")
        .def_prop_rw("target_modules",
                     [](const LoRAAdapterConfig* cfg) {
                         return std::vector<std::string>(cfg->TargetModules.begin(), cfg->TargetModules.end());
                     },
                     [](LoRAAdapterConfig* cfg, const std::vector<std::string>& modules) {
                         cfg->TargetModules = std::set<std::string>(modules.begin(), modules.end());
                     },
                     "List of module suffixes the adapter should apply to.")
        .def_prop_rw("dtype",
                     [](const LoRAAdapterConfig* cfg) { return dtype_to_str(cfg->DType); },
                     [](LoRAAdapterConfig* cfg, const std::string& dtype_str) { cfg->DType = dtype_from_str(dtype_str); },
                     "Adapter dtype as a string.")
        .def_prop_ro("scaling", &LoRAAdapterConfig::scaling, "Computed scaling factor (= alpha / rank, RS-LoRA aware).")
        .def("applies_to", &LoRAAdapterConfig::applies_to, nb::arg("module_name"),
             "Return True if LoRA should be applied to `module_name`.")
        .def("__repr__", [](const LoRAAdapterConfig& cfg) {
            std::string modules;
            for (const auto& m : cfg.TargetModules) {
                if (!modules.empty()) modules += ", ";
                modules += "'" + m + "'";
            }
            return fmt::format("LoRAAdapterConfig(rank={}, alpha={}, dropout={}, target_modules=[{}], dtype='{}', use_rslora={}, train_router={})",
                              cfg.Rank, cfg.Alpha, cfg.Dropout, modules, dtype_to_str(cfg.DType), cfg.UseRSLoRA, cfg.TrainRouter);
        }, "Return a debug string representation.")
        ;

    // QLoRA quantization strategy enum
    nb::enum_<modules::QLoRAQuantStrategy>(m, "QLoRAQuantStrategy",
        "Quantization strategy for QLoRA base weights.")
        .value("NONE", modules::QLoRAQuantStrategy::None, "No quantization (regular LoRA with BF16 base model)")
        .value("FP8", modules::QLoRAQuantStrategy::FP8, "FP8 E4M3 with per-block scales")
        .value("NVFP4", modules::QLoRAQuantStrategy::NVFP4, "FP4 E2M1 with two-level block scales (Blackwell SM100+)")
        .value("BNB", modules::QLoRAQuantStrategy::BitsAndBytes, "BitsAndBytes NF4 with per-block absmax (any GPU)")
        .value("PREQUANT_FP8", modules::QLoRAQuantStrategy::PrequantFP8, "Pre-quantized HF FP8 (DeepSeek-V3/R1)")
        .value("PREQUANT_NVFP4", modules::QLoRAQuantStrategy::PrequantNVFP4, "Pre-quantized HF NVFP4 (ModelOpt)")
        .value("PREQUANT_MXFP4", modules::QLoRAQuantStrategy::PrequantMXFP4, "Pre-quantized HF MXFP4")
        ;

    nb::class_<modules::QLoRAConfig>(m, "QLoRAConfig",
        "QLoRA (Quantized LoRA) configuration for memory-efficient adapter training.\n\n"
        "Configures quantization of base model weights. The base model is stored in a\n"
        "quantized format (FP8, FP4, or NF4) while LoRA adapters remain in full precision.\n\n"
        "Use QLoRAConfig.fp8(), QLoRAConfig.nvfp4(), or QLoRAConfig.bnb() factory methods.")
        .def("__init__", [](modules::QLoRAConfig *t, bool enabled, const std::string& strategy,
                           int block_size, const std::string& base_dtype, const std::string& adapter_dtype) {
            modules::QLoRAQuantStrategy strat = modules::QLoRAQuantStrategy::None;
            if (strategy == "fp8" || strategy == "FP8") {
                strat = modules::QLoRAQuantStrategy::FP8;
            } else if (strategy == "nvfp4" || strategy == "NVFP4" || strategy == "fp4") {
                strat = modules::QLoRAQuantStrategy::NVFP4;
            } else if (strategy == "bnb" || strategy == "BNB" || strategy == "bitsandbytes" || strategy == "nf4") {
                strat = modules::QLoRAQuantStrategy::BitsAndBytes;
            } else if (strategy == "prequant_fp8") {
                strat = modules::QLoRAQuantStrategy::PrequantFP8;
            } else if (strategy == "prequant_nvfp4") {
                strat = modules::QLoRAQuantStrategy::PrequantNVFP4;
            } else if (strategy == "prequant_mxfp4") {
                strat = modules::QLoRAQuantStrategy::PrequantMXFP4;
            } else if (!strategy.empty() && strategy != "none") {
                throw std::runtime_error("Unknown QLoRA strategy: " + strategy + " (valid: none, fp8, nvfp4, bnb, prequant_fp8, prequant_nvfp4, prequant_mxfp4)");
            }

            new (t) modules::QLoRAConfig{
                .enabled = enabled,
                .strategy = strat,
                .scale_config = {.block_size = block_size},
                .base_dtype = base_dtype.empty() ? ETensorDType::FP8_E4M3 : dtype_from_str(base_dtype),
                .adapter_dtype = adapter_dtype.empty() ? ETensorDType::BF16 : dtype_from_str(adapter_dtype)
            };
        }, nb::kw_only(),
             nb::arg("enabled") = false,
             nb::arg("strategy") = "none",
             nb::arg("block_size") = 128,
             nb::arg("base_dtype") = "",
             nb::arg("adapter_dtype") = "bf16",
             "Create a QLoRA configuration.\n\n"
             "Parameters:\n"
             "- enabled: Whether QLoRA is enabled.\n"
             "- strategy: Quantization strategy ('none', 'fp8', 'nvfp4', 'bnb').\n"
             "- block_size: Block size for per-block quantization (FP8: 64/128/256, FP4: 16, BnB: 64).\n"
             "- base_dtype: Storage dtype for quantized base weights.\n"
             "- adapter_dtype: Dtype for LoRA adapter weights (not quantized).")
        .def_rw("enabled", &modules::QLoRAConfig::enabled, "Whether QLoRA is enabled.")
        .def_prop_rw("strategy",
                     [](const modules::QLoRAConfig* cfg) { return modules::to_string(cfg->strategy); },
                     [](modules::QLoRAConfig* cfg, const std::string& strat) {
                         if (strat == "fp8" || strat == "FP8") {
                             cfg->strategy = modules::QLoRAQuantStrategy::FP8;
                         } else if (strat == "nvfp4" || strat == "NVFP4" || strat == "fp4") {
                             cfg->strategy = modules::QLoRAQuantStrategy::NVFP4;
                         } else if (strat == "bnb" || strat == "BNB" || strat == "bitsandbytes" || strat == "nf4") {
                             cfg->strategy = modules::QLoRAQuantStrategy::BitsAndBytes;
                         } else if (strat == "prequant_fp8") {
                             cfg->strategy = modules::QLoRAQuantStrategy::PrequantFP8;
                         } else if (strat == "prequant_nvfp4") {
                             cfg->strategy = modules::QLoRAQuantStrategy::PrequantNVFP4;
                         } else if (strat == "prequant_mxfp4") {
                             cfg->strategy = modules::QLoRAQuantStrategy::PrequantMXFP4;
                         } else {
                             cfg->strategy = modules::QLoRAQuantStrategy::None;
                         }
                     },
                     "Quantization strategy as a string.")
        .def_prop_rw("block_size",
                     [](const modules::QLoRAConfig* cfg) { return cfg->scale_config.block_size; },
                     [](modules::QLoRAConfig* cfg, int size) { cfg->scale_config.block_size = size; },
                     "Block size for per-block quantization.")
        .def_prop_rw("base_dtype",
                     [](const modules::QLoRAConfig* cfg) { return dtype_to_str(cfg->base_dtype); },
                     [](modules::QLoRAConfig* cfg, const std::string& dtype_str) { cfg->base_dtype = dtype_from_str(dtype_str); },
                     "Storage dtype for quantized base weights.")
        .def_prop_rw("adapter_dtype",
                     [](const modules::QLoRAConfig* cfg) { return dtype_to_str(cfg->adapter_dtype); },
                     [](modules::QLoRAConfig* cfg, const std::string& dtype_str) { cfg->adapter_dtype = dtype_from_str(dtype_str); },
                     "Dtype for LoRA adapter weights.")
        .def_rw("enable_four_over_six", &modules::QLoRAConfig::enable_four_over_six,
                "Enable Four Over Six (4/6) adaptive block scaling for NVFP4 quantization.")
        .def_prop_ro("is_quantized", &modules::QLoRAConfig::is_quantized,
                     "Whether quantization is active (enabled and strategy != None).")
        .def_prop_ro("is_fp4", &modules::QLoRAConfig::is_fp4,
                     "Whether using FP4 quantization.")
        .def_prop_ro("is_fp8", &modules::QLoRAConfig::is_fp8,
                     "Whether using FP8 quantization.")
        .def_prop_ro("is_bnb", &modules::QLoRAConfig::is_bnb,
                     "Whether using BitsAndBytes NF4 quantization.")
        .def_static("fp8", [](int block_size) {
            return modules::QLoRAConfig::fp8(block_size);
        }, nb::arg("block_size") = 128,
           "Create FP8 QLoRA configuration.\n\n"
           "Parameters:\n- block_size: Block size for per-block quantization (64, 128, or 256).\n\n"
           "Returns: QLoRAConfig with FP8 E4M3 base weights.")
        .def_static("nvfp4", []() {
            return modules::QLoRAConfig::nvfp4();
        },
           "Create NVFP4 QLoRA configuration.\n\n"
           "Requires Blackwell GPU (SM100+) for native FP4 instructions.\n"
           "Uses two-level block scaling: FP8 E4M3 per 16 values, FP32 global scale.\n\n"
           "Returns: QLoRAConfig with FP4 E2M1 base weights.")
        .def_static("none", []() {
            return modules::QLoRAConfig::none();
        },
           "Create disabled QLoRA configuration (regular LoRA).\n\n"
           "Returns: QLoRAConfig with quantization disabled.")
        .def_static("bnb", [](int block_size, bool double_quant) {
            return modules::QLoRAConfig::bnb(block_size, double_quant);
        }, nb::arg("block_size") = 64, nb::arg("double_quant") = true,
           "Create BitsAndBytes NF4 QLoRA configuration.\n\n"
           "Works on any CUDA GPU (no SM89+ or SM100+ requirement).\n"
           "Uses NF4 (Normal Float 4-bit) quantization with per-block absmax scaling.\n\n"
           "Parameters:\n"
           "- block_size: Number of consecutive elements per quantization block (64, 128, 256, 512).\n"
           "- double_quant: Enable double quantization (quantize absmax to INT8) for extra memory savings.\n\n"
           "Returns: QLoRAConfig with NF4 base weights.")
        .def_static("prequant_fp8", []() {
            return modules::QLoRAConfig::prequant_fp8();
        },
           "Create pre-quantized FP8 QLoRA configuration.\n\n"
           "For loading HF models with fine-grained FP8 quantization (e.g., DeepSeek-V3/R1).\n"
           "No online quantization — weights are loaded as-is from safetensors.\n\n"
           "Returns: QLoRAConfig for pre-quantized FP8 loading.")
        .def_static("prequant_nvfp4", []() {
            return modules::QLoRAConfig::prequant_nvfp4();
        },
           "Create pre-quantized NVFP4 QLoRA configuration.\n\n"
           "For loading HF models quantized with NVIDIA ModelOpt NVFP4.\n"
           "Uses two-level scaling: FP8 E4M3 block scales + FP32 global scale.\n\n"
           "Returns: QLoRAConfig for pre-quantized NVFP4 loading.")
        .def_static("prequant_mxfp4", []() {
            return modules::QLoRAConfig::prequant_mxfp4();
        },
           "Create pre-quantized MXFP4 QLoRA configuration.\n\n"
           "For loading HF models with microscaling FP4 quantization.\n"
           "Uses E8M0 shared exponents per 32-element block.\n\n"
           "Returns: QLoRAConfig for pre-quantized MXFP4 loading.")
        .def_rw("bnb_double_quant", &modules::QLoRAConfig::bnb_double_quant,
                "Enable double quantization for BnB (quantize absmax values to INT8).")
        .def_rw("bnb_double_quant_group_size", &modules::QLoRAConfig::bnb_double_quant_group_size,
                "Group size for double quantization (number of absmax values per group).")
        .def_rw("num_experts", &modules::QLoRAConfig::num_experts,
                "Number of experts for MoE models (0 = dense model, >0 = MoE model).")
        .def_rw("num_experts_per_tok", &modules::QLoRAConfig::num_experts_per_tok,
                "Number of experts selected per token (top-k routing).")
        .def_rw("moe_intermediate_size", &modules::QLoRAConfig::moe_intermediate_size,
                "Per-expert MLP intermediate size (0 = use regular intermediate_size).")
        .def_prop_ro("is_moe", &modules::QLoRAConfig::is_moe,
                     "Whether this is an MoE model (num_experts > 0).")
        .def_prop_ro("is_prequantized", &modules::QLoRAConfig::is_prequantized,
                     "Whether loading a pre-quantized HF model (no online quantization).")
        .def_prop_rw("modules_to_not_convert",
                     [](const modules::QLoRAConfig* cfg) {
                         nb::list result;
                         for (const auto& m : cfg->modules_to_not_convert) {
                             result.append(nb::cast(m));
                         }
                         return result;
                     },
                     [](modules::QLoRAConfig* cfg, const nb::list& modules) {
                         cfg->modules_to_not_convert.clear();
                         for (auto item : modules) {
                             cfg->modules_to_not_convert.push_back(nb::cast<std::string>(item));
                         }
                     },
                     "HF module paths that should NOT be quantized (loaded as full-precision).")
        .def("__repr__", [](const modules::QLoRAConfig& cfg) {
            if (cfg.is_bnb()) {
                return fmt::format("QLoRAConfig(enabled={}, strategy='{}', block_size={}, bnb_double_quant={}, adapter_dtype='{}')",
                                  cfg.enabled, modules::to_string(cfg.strategy), cfg.scale_config.block_size,
                                  cfg.bnb_double_quant, dtype_to_str(cfg.adapter_dtype));
            }
            return fmt::format("QLoRAConfig(enabled={}, strategy='{}', block_size={}, base_dtype='{}', adapter_dtype='{}')",
                              cfg.enabled, modules::to_string(cfg.strategy), cfg.scale_config.block_size,
                              dtype_to_str(cfg.base_dtype), dtype_to_str(cfg.adapter_dtype));
        }, "Return a debug string representation.")
        ;

    // Optimizer type enum
    nb::enum_<optimizers::OptimizerType>(m, "OptimizerType",
        "Optimizer algorithm types.\n\n"
        "Values:\n"
        "- ADAMW_8BIT: 8-bit AdamW with blockwise quantization.\n"
        "- NORMUON: NorMuon hybrid optimizer (orthogonalized momentum for 2D weights, AdamW for others).")
        .value("ADAMW_8BIT", optimizers::OptimizerType::ADAMW_8BIT, "8-bit AdamW with blockwise quantization")
        .value("NORMUON", optimizers::OptimizerType::NORMUON, "NorMuon hybrid optimizer")
        ;

    nb::class_<optimizers::OptimizerConfig>(m, "OptimizerConfig",
        "Optimizer configuration.\n\n"
        "Contains all hyperparameters for supported optimizers.\n"
        "Parameters for unused optimizers are ignored.")
        .def("__init__", [](optimizers::OptimizerConfig *t,
                           const std::string& optimizer,
                           float learning_rate, float weight_decay, float grad_clip,
                           float adamw_beta1, float adamw_beta2, float adamw_epsilon,
                           float normuon_momentum, float normuon_beta2, float normuon_lr, bool normuon_cautious_wd) {
            auto cfg = optimizers::OptimizerConfig{};
            if (optimizer == "adamw_8bit" || optimizer == "adamw") {
                cfg.type = optimizers::OptimizerType::ADAMW_8BIT;
            } else if (optimizer == "normuon") {
                cfg.type = optimizers::OptimizerType::NORMUON;
            } else {
                throw std::runtime_error("Unknown optimizer type: " + optimizer + " (valid: adamw_8bit, normuon)");
            }
            cfg.learning_rate = learning_rate;
            cfg.weight_decay = weight_decay;
            cfg.grad_clip = grad_clip;
            cfg.adamw_beta1 = adamw_beta1;
            cfg.adamw_beta2 = adamw_beta2;
            cfg.adamw_epsilon = adamw_epsilon;
            cfg.normuon_momentum = normuon_momentum;
            cfg.normuon_beta2 = normuon_beta2;
            cfg.normuon_lr = normuon_lr;
            cfg.normuon_cautious_wd = normuon_cautious_wd;
            new (t) optimizers::OptimizerConfig(cfg);
        }, nb::kw_only(),
             nb::arg("optimizer") = "adamw_8bit",
             nb::arg("learning_rate") = 2e-4f,
             nb::arg("weight_decay") = 0.1f,
             nb::arg("grad_clip") = 0.0f,
             nb::arg("adamw_beta1") = 0.9f,
             nb::arg("adamw_beta2") = 0.999f,
             nb::arg("adamw_epsilon") = 1e-8f,
             nb::arg("normuon_momentum") = 0.95f,
             nb::arg("normuon_beta2") = 0.95f,
             nb::arg("normuon_lr") = 0.02f,
             nb::arg("normuon_cautious_wd") = true,
             "Create an optimizer configuration.\n\n"
             "Parameters:\n"
             "- optimizer: Type of optimizer ('adamw_8bit' or 'normuon').\n"
             "- learning_rate: Base learning rate.\n"
             "- weight_decay: Weight decay coefficient.\n"
             "- grad_clip: Gradient clipping threshold (0 = disabled).\n"
             "- adamw_beta1/beta2/epsilon: AdamW hyperparameters.\n"
             "- normuon_momentum/beta2/lr/cautious_wd: NorMuon hyperparameters.\n")
        .def_rw("learning_rate", &optimizers::OptimizerConfig::learning_rate, "Base learning rate.")
        .def_rw("weight_decay", &optimizers::OptimizerConfig::weight_decay, "Weight decay coefficient.")
        .def_rw("grad_clip", &optimizers::OptimizerConfig::grad_clip, "Gradient clipping threshold.")
        .def_rw("adamw_beta1", &optimizers::OptimizerConfig::adamw_beta1, "AdamW beta1.")
        .def_rw("adamw_beta2", &optimizers::OptimizerConfig::adamw_beta2, "AdamW beta2.")
        .def_rw("adamw_epsilon", &optimizers::OptimizerConfig::adamw_epsilon, "AdamW epsilon.")
        .def_rw("normuon_momentum", &optimizers::OptimizerConfig::normuon_momentum, "NorMuon momentum (beta1).")
        .def_rw("normuon_beta2", &optimizers::OptimizerConfig::normuon_beta2, "NorMuon variance EMA (beta2).")
        .def_rw("normuon_lr", &optimizers::OptimizerConfig::normuon_lr, "NorMuon learning rate.")
        .def_rw("normuon_cautious_wd", &optimizers::OptimizerConfig::normuon_cautious_wd, "Use cautious weight decay.")
        .def_prop_rw("type",
                     [](const optimizers::OptimizerConfig* cfg) { return optimizers::to_string(cfg->type); },
                     [](optimizers::OptimizerConfig* cfg, const std::string& type) {
                         if (type == "adamw_8bit" || type == "adamw") {
                             cfg->type = optimizers::OptimizerType::ADAMW_8BIT;
                         } else if (type == "normuon") {
                             cfg->type = optimizers::OptimizerType::NORMUON;
                         }
                     },
                     "Optimizer type as string.")
        .def_static("adamw_8bit", &optimizers::OptimizerConfig::adamw_8bit,
             nb::arg("lr") = 2e-4f, nb::arg("beta1") = 0.9f, nb::arg("beta2") = 0.999f,
             nb::arg("epsilon") = 1e-8f, nb::arg("weight_decay") = 0.1f, nb::arg("grad_clip") = 0.0f,
             "Create AdamW 8-bit configuration.")
        .def_static("normuon", &optimizers::OptimizerConfig::normuon,
             nb::arg("lr") = 0.02f, nb::arg("momentum") = 0.95f, nb::arg("beta2") = 0.95f,
             nb::arg("weight_decay") = 0.01f, nb::arg("grad_clip") = 0.0f, nb::arg("cautious_wd") = true,
             "Create NorMuon configuration.\n\n"
             "NorMuon uses orthogonalized momentum for 2D weight matrices and AdamW for other parameters.")
        .def("__repr__", [](const optimizers::OptimizerConfig& cfg) {
            return fmt::format("OptimizerConfig(type='{}', lr={}, weight_decay={}, grad_clip={})",
                              optimizers::to_string(cfg.type), cfg.learning_rate, cfg.weight_decay, cfg.grad_clip);
        }, "Return a debug string representation.")
        ;

    nb::class_<MultiGPUPyTrainer>(m, "SurogateTrainer",
        "Multi-GPU trainer wrapper.\n\n"
        "Provides training/evaluation steps and checkpoint/weight import/export.\n"
        "Some operations may run asynchronously (see method docs).")
        .def("__init__", [](MultiGPUPyTrainer *t, int ngpu, const PretrainedConfig& config, RuntimeOptions options, int batch_size, int seq_len, int grad_accum, bool memcpy_all_gather, bool memcpy_send_recv, std::optional<LoRAAdapterConfig> lora_config, std::optional<modules::QLoRAConfig> qlora_config) {
            options.ModelType = config.DType;
            new (t) MultiGPUPyTrainer(ngpu, config, options, batch_size, seq_len, grad_accum, memcpy_all_gather, memcpy_send_recv, lora_config, qlora_config);
        }, nb::arg("ngpu"), nb::arg("config"), nb::arg("options"), nb::arg("batch_size"), nb::arg("seq_len"), nb::arg("grad_accum"),
             nb::arg("memcpy_all_gather") = true, nb::arg("memcpy_send_recv") = true, nb::arg("lora_config") = std::nullopt, nb::arg("qlora_config") = std::nullopt,
             "Create a trainer instance.\n\n"
             "Parameters:\n"
             "- ngpu: Number of GPUs to use.\n"
             "- config: Model configuration.\n"
             "- options: Runtime/training options.\n"
             "- batch_size: Per-GPU batch size (effective batch is batch_size * world_size).\n"
             "- seq_len: Sequence length.\n"
             "- grad_accum: Gradient accumulation steps.\n"
             "- memcpy_all_gather/memcpy_send_recv: Enable memcpy-based collectives where supported.\n"
             "- lora_config: Optional LoRA configuration for adapter training (freezes base model).\n"
             "- qlora_config: Optional QLoRA configuration for quantized base weights (FP8/FP4).")
        .def("import_weights", &MultiGPUPyTrainer::import_weights, nb::arg("path"),
             "Import weights from a HuggingFace model file.\n\n"
             "Parameters:\n- path: Path to model.safetensors or model.safetensors.index.json.")
        .def("export_model", &MultiGPUPyTrainer::export_model, nb::arg("path"),
             "Export model weights and config to a directory.\n\n"
             "Parameters:\n- path: Output directory path.")
        .def("export_adapter", &MultiGPUPyTrainer::export_adapter, nb::arg("path"), nb::arg("base_model_path") = "",
             "Export LoRA adapter weights to a directory (PEFT-compatible format).\n\n"
             "Only works if the model was created with a LoRA configuration.\n"
             "Creates adapter_model.safetensors and adapter_config.json.\n\n"
             "Parameters:\n- path: Output directory path.\n"
             "- base_model_path: Optional path/name of base model for adapter_config.json.")
        .def_static("from_pretrained", [](const std::string& name, int ngpu, std::string dtype, RuntimeOptions options, int batch_size, int seq_len, int grad_accum, bool memcpy_all_gather, bool memcpy_send_recv, std::optional<LoRAAdapterConfig> lora_config, std::optional<modules::QLoRAConfig> qlora_config){
            std::string hf_path = get_hf_model_files(name);
            if (hf_path.empty()) {
                throw std::runtime_error("Could not find model files for " + name);
            }
            std::string config_path = hf_path + "/config.json";
            std::string model_path = hf_path + "/model.safetensors";
            if (!std::filesystem::exists(model_path)) {
                model_path = hf_path + "/model.safetensors.index.json";
            }
            auto config_ptr = load_pretrained_config(config_path.c_str(), dtype_from_str(dtype));
            const PretrainedConfig& config = *config_ptr;
            options.ModelType = config.DType;
            auto trainer = new MultiGPUPyTrainer(ngpu, config, options, batch_size, seq_len, grad_accum, memcpy_all_gather, memcpy_send_recv, lora_config, qlora_config);
            trainer->import_weights(model_path);
            return trainer;
            }, nb::arg("name"), nb::arg("ngpu"), nb::arg("dtype"), nb::arg("options"), nb::arg("batch_size"), nb::arg("seq_len"), nb::arg("grad_accum"),
                    nb::arg("memcpy_all_gather") = true, nb::arg("memcpy_send_recv") = true, nb::arg("lora_config") = std::nullopt, nb::arg("qlora_config") = std::nullopt,
                    "Create a trainer and import weights from a HuggingFace model name.\n\n"
                    "Parameters:\n"
                    "- name: HuggingFace model name.\n"
                    "- ngpu: Number of GPUs.\n"
                    "- dtype: Desired dtype for the loaded config.\n"
                    "- options: Runtime/training options.\n"
                    "- batch_size/seq_len/grad_accum: Training shape parameters.\n"
                    "- memcpy_all_gather/memcpy_send_recv: Enable memcpy-based collectives.\n"
                    "- lora_config: Optional LoRA configuration for adapter training.\n"
                    "- qlora_config: Optional QLoRA configuration for quantized base weights (FP8/FP4).\n\n"
                    "Returns: SurogateTrainer")
        .def("init_weights", &MultiGPUPyTrainer::init_weights,
             "Initialize weights from scratch (random init).")
        .def("load_checkpoint", &MultiGPUPyTrainer::load_checkpoint, nb::arg("path"), nb::arg("step"),
             "Load a checkpoint.\n\n"
             "Parameters:\n- path: Checkpoint directory.\n- step: Step number to load.")
        .def("save_checkpoint", &MultiGPUPyTrainer::save_checkpoint, nb::arg("path"), nb::arg("step"),
             "Save a checkpoint.\n\n"
             "Parameters:\n- path: Checkpoint directory.\n- step: Step number to save.")
        .def("step", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets) {
            // Use local_world_size (GPUs on this node) not global world_size
            CHECK_SHAPE(inputs, trainer->batch_size() * trainer->local_world_size(), trainer->seq_length());
            CHECK_SHAPE(targets, trainer->batch_size() * trainer->local_world_size(), trainer->seq_length());

            trainer->step(inputs.data(), targets.data());
        }, nb::arg("inputs"), nb::arg("targets"),
             "Perform one training step (forward + backward).\n\n"
             "This call is asynchronous; the loss becomes available on the next `update()`.\n\n"
             "Parameters:\n"
             "- inputs: int32 token ids shaped [batch_size * local_gpus, seq_length].\n"
             "- targets: int32 token ids shaped [batch_size * local_gpus, seq_length].")
        .def("step", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets, TokenArray position_ids) {
            // Use local_world_size (GPUs on this node) not global world_size
            CHECK_SHAPE(inputs, trainer->batch_size() * trainer->local_world_size(), trainer->seq_length());
            CHECK_SHAPE(targets, trainer->batch_size() * trainer->local_world_size(), trainer->seq_length());
            CHECK_SHAPE(position_ids, trainer->batch_size() * trainer->local_world_size(), trainer->seq_length());

            trainer->step(inputs.data(), targets.data(), position_ids.data());
        }, nb::arg("inputs"), nb::arg("targets"), nb::arg("position_ids"),
             "Perform one training step (forward + backward) with explicit position ids.\n\n"
             "Parameters:\n"
             "- inputs: int32 token ids shaped [batch_size * local_gpus, seq_length].\n"
             "- targets: int32 token ids shaped [batch_size * local_gpus, seq_length].\n"
             "- position_ids: int32 position ids shaped [batch_size * local_gpus, seq_length].")
        .def("step", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets, TokenArray3 position_ids) {
            const int local_gpus = trainer->local_world_size();
            const int batch_size = trainer->batch_size();
            const int seq_len = trainer->seq_length();
            CHECK_SHAPE(inputs, batch_size * local_gpus, seq_len);
            CHECK_SHAPE(targets, batch_size * local_gpus, seq_len);
            if (position_ids.shape(0) < 1) {
                throw std::runtime_error("position_ids must have at least 1 plane");
            }
            if (position_ids.shape(1) != batch_size * local_gpus || position_ids.shape(2) != seq_len) {
                throw std::runtime_error("position_ids must have shape [planes, batch_size * local_gpus, seq_length]");
            }

            if (local_gpus == 1) {
                trainer->step(inputs.data(), targets.data(), position_ids.data());
                return;
            }

            const int planes = static_cast<int>(position_ids.shape(0));
            const std::size_t per_rank = static_cast<std::size_t>(planes) *
                                         static_cast<std::size_t>(batch_size) *
                                         static_cast<std::size_t>(seq_len);
            std::vector<std::int32_t> reordered(static_cast<std::size_t>(local_gpus) * per_rank);
            const std::int32_t* src = position_ids.data();
            for (int r = 0; r < local_gpus; ++r) {
                std::int32_t* dst_rank = reordered.data() + static_cast<std::size_t>(r) * per_rank;
                for (int p = 0; p < planes; ++p) {
                    for (int b = 0; b < batch_size; ++b) {
                        const std::size_t src_base = (static_cast<std::size_t>(p) * batch_size * local_gpus +
                                                      static_cast<std::size_t>(r) * batch_size +
                                                      static_cast<std::size_t>(b)) * seq_len;
                        const std::size_t dst_base = (static_cast<std::size_t>(p) * batch_size +
                                                      static_cast<std::size_t>(b)) * seq_len;
                        std::memcpy(dst_rank + dst_base, src + src_base, sizeof(std::int32_t) * seq_len);
                    }
                }
            }
            trainer->step(inputs.data(), targets.data(), reordered.data());
        }, nb::arg("inputs"), nb::arg("targets"), nb::arg("position_ids"),
             "Perform one training step (forward + backward) with 3D position ids.\n\n"
             "Parameters:\n"
             "- inputs: int32 token ids shaped [batch_size * local_gpus, seq_length].\n"
             "- targets: int32 token ids shaped [batch_size * local_gpus, seq_length].\n"
             "- position_ids: int32 position ids shaped [planes, batch_size * local_gpus, seq_length].")
        .def("set_visual_inputs", [](MultiGPUPyTrainer* trainer,
                                     TokenArray visual_pos_masks,
                                     FloatArray visual_embeds,
                                     std::vector<FloatArray> deepstack_visual_embeds) {
            const int local_gpus = trainer->local_world_size();
            const int batch_size = trainer->batch_size();
            const int seq_len = trainer->seq_length();
            const int hidden = trainer->config().HiddenSize;
            CHECK_SHAPE(visual_pos_masks, batch_size * local_gpus, seq_len);
            if (visual_embeds.shape(0) != static_cast<std::size_t>(batch_size * local_gpus * seq_len) ||
                visual_embeds.shape(1) != hidden) {
                throw std::runtime_error("visual_embeds must have shape [batch_size * local_gpus * seq_length, hidden]");
            }
            for (const auto& arr : deepstack_visual_embeds) {
                if (arr.shape(0) != visual_embeds.shape(0) || arr.shape(1) != visual_embeds.shape(1)) {
                    throw std::runtime_error("deepstack_visual_embeds entries must match visual_embeds shape");
                }
            }
            std::vector<const float*> deepstack_ptrs;
            deepstack_ptrs.reserve(deepstack_visual_embeds.size());
            for (const auto& arr : deepstack_visual_embeds) {
                deepstack_ptrs.push_back(arr.data());
            }
            trainer->set_visual_inputs(visual_pos_masks.data(), visual_embeds.data(), deepstack_ptrs);
        }, nb::arg("visual_pos_masks"), nb::arg("visual_embeds"), nb::arg("deepstack_visual_embeds") = std::vector<FloatArray>{},
             "Set visual inputs for multimodal models.\n\n"
             "Parameters:\n"
             "- visual_pos_masks: int32 mask shaped [batch_size * local_gpus, seq_length] (non-zero = visual).\n"
             "- visual_embeds: float32 packed embeds shaped [batch_size * local_gpus * seq_length, hidden].\n"
             "- deepstack_visual_embeds: optional list of float32 packed embeds, same shape as visual_embeds.\n"
             "  List length should match config.DeepstackVisualLayers.")
        .def("validate", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets) {
            // Use local_world_size (GPUs on this node) not global world_size
            CHECK_SHAPE(inputs, trainer->batch_size() * trainer->local_world_size(), trainer->seq_length());
            CHECK_SHAPE(targets, trainer->batch_size() * trainer->local_world_size(), trainer->seq_length());

            return trainer->validate(inputs.data(), targets.data());
        }, nb::arg("inputs"), nb::arg("targets"),
             "Compute validation loss for one batch (forward only).\n\n"
             "Parameters:\n"
             "- inputs: int32 token ids shaped [batch_size * local_gpus, seq_length].\n"
             "- targets: int32 token ids shaped [batch_size * local_gpus, seq_length].\n\n"
             "Returns: loss (float).")
        .def("validate", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets, TokenArray position_ids) {
            // Use local_world_size (GPUs on this node) not global world_size
            CHECK_SHAPE(inputs, trainer->batch_size() * trainer->local_world_size(), trainer->seq_length());
            CHECK_SHAPE(targets, trainer->batch_size() * trainer->local_world_size(), trainer->seq_length());
            CHECK_SHAPE(position_ids, trainer->batch_size() * trainer->local_world_size(), trainer->seq_length());

            return trainer->validate(inputs.data(), targets.data(), position_ids.data());
        }, nb::arg("inputs"), nb::arg("targets"), nb::arg("position_ids"),
             "Compute validation loss for one batch (forward only) with explicit position ids.\n\n"
             "Parameters:\n"
             "- inputs: int32 token ids shaped [batch_size * local_gpus, seq_length].\n"
             "- targets: int32 token ids shaped [batch_size * local_gpus, seq_length].\n"
             "- position_ids: int32 position ids shaped [batch_size * local_gpus, seq_length].\n\n"
             "Returns: loss (float).")
        .def("validate", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets, TokenArray3 position_ids) {
            const int local_gpus = trainer->local_world_size();
            const int batch_size = trainer->batch_size();
            const int seq_len = trainer->seq_length();
            CHECK_SHAPE(inputs, batch_size * local_gpus, seq_len);
            CHECK_SHAPE(targets, batch_size * local_gpus, seq_len);
            if (position_ids.shape(0) < 1) {
                throw std::runtime_error("position_ids must have at least 1 plane");
            }
            if (position_ids.shape(1) != batch_size * local_gpus || position_ids.shape(2) != seq_len) {
                throw std::runtime_error("position_ids must have shape [planes, batch_size * local_gpus, seq_length]");
            }

            if (local_gpus == 1) {
                return trainer->validate(inputs.data(), targets.data(), position_ids.data());
            }

            const int planes = static_cast<int>(position_ids.shape(0));
            const std::size_t per_rank = static_cast<std::size_t>(planes) *
                                         static_cast<std::size_t>(batch_size) *
                                         static_cast<std::size_t>(seq_len);
            std::vector<std::int32_t> reordered(static_cast<std::size_t>(local_gpus) * per_rank);
            const std::int32_t* src = position_ids.data();
            for (int r = 0; r < local_gpus; ++r) {
                std::int32_t* dst_rank = reordered.data() + static_cast<std::size_t>(r) * per_rank;
                for (int p = 0; p < planes; ++p) {
                    for (int b = 0; b < batch_size; ++b) {
                        const std::size_t src_base = (static_cast<std::size_t>(p) * batch_size * local_gpus +
                                                      static_cast<std::size_t>(r) * batch_size +
                                                      static_cast<std::size_t>(b)) * seq_len;
                        const std::size_t dst_base = (static_cast<std::size_t>(p) * batch_size +
                                                      static_cast<std::size_t>(b)) * seq_len;
                        std::memcpy(dst_rank + dst_base, src + src_base, sizeof(std::int32_t) * seq_len);
                    }
                }
            }

            return trainer->validate(inputs.data(), targets.data(), reordered.data());
        }, nb::arg("inputs"), nb::arg("targets"), nb::arg("position_ids"),
             "Compute validation loss for one batch (forward only) with 3D position ids.\n\n"
             "Parameters:\n"
             "- inputs: int32 token ids shaped [batch_size * local_gpus, seq_length].\n"
             "- targets: int32 token ids shaped [batch_size * local_gpus, seq_length].\n"
             "- position_ids: int32 position ids shaped [planes, batch_size * local_gpus, seq_length].\n\n"
             "Returns: loss (float).")
        .def("update_with_config", [](MultiGPUPyTrainer* trainer, const optimizers::OptimizerConfig& config, int step){
            auto [loss, norm] = trainer->update_with_config(config, step);
            nb::dict ret;
            ret["loss"] = loss;
            ret["norm"] = norm;
            return ret;
        }, nb::arg("config"), nb::arg("step"),
             "Run the optimizer step with full configuration.\n\n"
             "Supports AdamW 8-bit and NorMuon optimizers based on config.type.\n\n"
             "Parameters:\n"
             "- config: OptimizerConfig with all hyperparameters.\n"
             "- step: Global step index.\n\n"
             "Returns: dict with keys {loss: float, norm: float}.")
        .def("train_step_graphed", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets,
                                      const optimizers::OptimizerConfig& config, int step){
            const int rows = trainer->batch_size() * trainer->local_world_size() * trainer->grad_accumulation();
            CHECK_SHAPE(inputs, rows, trainer->seq_length());
            CHECK_SHAPE(targets, rows, trainer->seq_length());

            auto [loss, norm] = trainer->train_step_graphed(inputs.data(), targets.data(), nullptr, config, step);
            nb::dict ret;
            ret["loss"] = loss;
            ret["norm"] = norm;
            return ret;
        }, nb::arg("inputs"), nb::arg("targets"), nb::arg("config"), nb::arg("step"),
             "Run a full training step with CUDA graph capture (forward+backward+optimizer).\n\n"
             "Parameters:\n"
             "- inputs: int32 token ids shaped [grad_accum * local_gpus * batch_size, seq_length].\n"
             "- targets: int32 token ids shaped [grad_accum * local_gpus * batch_size, seq_length].\n"
             "- config: OptimizerConfig with all hyperparameters.\n"
             "- step: Global step index.\n\n"
             "Returns: dict with keys {loss: float, norm: float}.")
        .def("train_step_graphed", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets,
                                      TokenArray position_ids, const optimizers::OptimizerConfig& config, int step){
            const int rows = trainer->batch_size() * trainer->local_world_size() * trainer->grad_accumulation();
            CHECK_SHAPE(inputs, rows, trainer->seq_length());
            CHECK_SHAPE(targets, rows, trainer->seq_length());
            CHECK_SHAPE(position_ids, rows, trainer->seq_length());

            auto [loss, norm] = trainer->train_step_graphed(inputs.data(), targets.data(), position_ids.data(), config, step);
            nb::dict ret;
            ret["loss"] = loss;
            ret["norm"] = norm;
            return ret;
        }, nb::arg("inputs"), nb::arg("targets"), nb::arg("position_ids"), nb::arg("config"), nb::arg("step"),
             "Run a full training step with CUDA graph capture and explicit position ids.\n\n"
             "Parameters:\n"
             "- inputs: int32 token ids shaped [grad_accum * local_gpus * batch_size, seq_length].\n"
             "- targets: int32 token ids shaped [grad_accum * local_gpus * batch_size, seq_length].\n"
             "- position_ids: int32 position ids shaped [grad_accum * local_gpus * batch_size, seq_length].\n"
             "- config: OptimizerConfig with all hyperparameters.\n"
             "- step: Global step index.\n\n"
             "Returns: dict with keys {loss: float, norm: float}.")
        .def("get_gradients", [](MultiGPUPyTrainer* trainer, int gpu_id)
        {
            auto raw = trainer->get_gradients(gpu_id);
            nb::dict ret;
            for (const auto& [name, value] : raw) {
                std::array<std::size_t, 6> shape;
                std::copy_n(value.Sizes.begin(), value.Rank, shape.begin());
                nb::ndarray<> view{value.Data, (size_t)value.Rank, shape.data(), ret, nullptr, to_dlpack_dtype(value.DType), nb::device::cuda::value, value.Device};
                ret[nb::cast(name)] = view;
            }
            return ret;
        }, nb::arg("gpu_id"),
           "Return gradient shards for debugging.\n\n"
           "Parameters:\n- gpu_id: Which GPU's shard to return.\n\n"
           "Returns: dict[str, ndarray] mapping parameter name -> gradient view.\n"
           "Note: blocking; intended for debugging only.")
        .def("get_lora_gradients", [](MultiGPUPyTrainer* trainer, int gpu_id)
        {
            auto raw = trainer->get_lora_gradients(gpu_id);
            nb::dict ret;
            for (const auto& [name, value] : raw) {
                std::array<std::size_t, 6> shape;
                std::copy_n(value.Sizes.begin(), value.Rank, shape.begin());
                nb::ndarray<> view{value.Data, (size_t)value.Rank, shape.data(), ret, nullptr, to_dlpack_dtype(value.DType), nb::device::cuda::value, value.Device};
                ret[nb::cast(name)] = view;
            }
            return ret;
        }, nb::arg("gpu_id"),
           "Return LoRA adapter gradients for debugging.\n\n"
           "Only works if the trainer was constructed with a LoRA configuration.\n\n"
           "Parameters:\n- gpu_id: Which GPU's gradients to return.\n\n"
           "Returns: dict[str, ndarray] mapping adapter parameter name -> gradient view.\n"
           "Note: blocking; intended for debugging only.")
        .def("get_valid_token_count", &MultiGPUPyTrainer::get_valid_token_count, nb::arg("gpu_id"),
             "Return the accumulated valid-token count for the last training step.\n\n"
             "Parameters:\n- gpu_id: Which GPU's count to return.\n\n"
             "Returns: int (valid tokens accumulated across micro-steps, after all-reduce).")
        .def("get_gpu_info", &MultiGPUPyTrainer::get_gpu_info,
             "Return current GPU utilization info for all GPUs (implementation-defined structure).")
        .def("get_moe_stats", [](MultiGPUPyTrainer* trainer) {
            auto [aux_loss, z_loss, utilization, imbalance, valid] = trainer->get_moe_stats();
            nb::dict ret;
            ret["aux_loss"] = aux_loss;
            ret["z_loss"] = z_loss;
            ret["expert_utilization"] = utilization;
            ret["load_imbalance"] = imbalance;
            ret["valid"] = valid;
            return ret;
        }, "Get MoE training statistics from the last forward pass.\n\n"
           "Returns: dict with keys {aux_loss, z_loss, expert_utilization, load_imbalance, valid}.\n"
           "For non-MoE models, valid=False and other values are zero.")
        .def_prop_ro("world_size", &MultiGPUPyTrainer::world_size, "Number of participating GPUs.")
        .def_prop_ro("batch_size", &MultiGPUPyTrainer::batch_size, "Per-GPU batch size configured for this trainer.")
        .def_prop_ro("seq_length", &MultiGPUPyTrainer::seq_length, "Sequence length configured for this trainer.")
        .def("get_allocator_info", [](MultiGPUPyTrainer* trainer, int gpu_id) {
            auto alloc = trainer->get_allocations(gpu_id);
            nb::dict ret;
            for(const auto& [name, size] : alloc) {
                nb::dict res;
                res["device"] = size.OnDevice;
                res["managed"] = size.Managed;
                res["pinned"] = size.PinnedHost;
                res["pageable"] = size.PageableHost;
                ret[nb::cast(name)] = res;
            }

            auto stack = trainer->get_stack_info(gpu_id);
            for (const auto& [name, size] : stack) {
                nb::dict res;
                res["stack"] = size;
                ret[nb::cast(name)] = res;
            }
            return ret;
            }, nb::arg("gpu_id") = 0,
            "Get current memory allocator statistics.\n\n"
            "Parameters:\n- gpu_id: Which GPU to query.\n\n"
            "Returns: dict[str, dict] with per-segment counters; stack entries include {'stack': bytes}.")
        .def_static("create_multinode", [](int ngpu, int node_rank, int num_nodes,
                nb::bytes nccl_id, nb::bytes node_master_nccl_id,
                PretrainedConfig config, RuntimeOptions options,
                int batch_size, int seq_len, int grad_accum,
                bool memcpy_all_gather, bool memcpy_send_recv,
                std::optional<LoRAAdapterConfig> lora_config,
                std::optional<modules::QLoRAConfig> qlora_config) {
            if (nccl_id.size() != 128) {
                throw std::runtime_error(fmt::format("nccl_id must be exactly 128 bytes, got {}", nccl_id.size()));
            }
            if (node_master_nccl_id.size() != 128) {
                throw std::runtime_error(fmt::format("node_master_nccl_id must be exactly 128 bytes, got {}", node_master_nccl_id.size()));
            }
            options.ModelType = config.DType;
            return new MultiGPUPyTrainer(ngpu, node_rank, num_nodes,
                nccl_id.c_str(), node_master_nccl_id.c_str(),
                config, options, batch_size, seq_len, grad_accum,
                memcpy_all_gather, memcpy_send_recv, lora_config, qlora_config);
        }, nb::arg("ngpu"), nb::arg("node_rank"), nb::arg("num_nodes"),
           nb::arg("nccl_id"), nb::arg("node_master_nccl_id"),
           nb::arg("config"), nb::arg("options"),
           nb::arg("batch_size"), nb::arg("seq_len"), nb::arg("grad_accum"),
           nb::arg("memcpy_all_gather") = true, nb::arg("memcpy_send_recv") = true,
           nb::arg("lora_config") = std::nullopt, nb::arg("qlora_config") = std::nullopt,
           "Create a trainer instance for multi-node distributed training.\n\n"
           "Used with Ray for coordinating training across multiple machines.\n"
           "NCCL IDs must be generated on the master node using generate_nccl_id()\n"
           "and shared with all worker nodes before calling this method.\n\n"
           "Parameters:\n"
           "- ngpu: Number of local GPUs on this node.\n"
           "- node_rank: This node's rank (0 to num_nodes-1).\n"
           "- num_nodes: Total number of nodes in the cluster.\n"
           "- nccl_id: 128-byte NCCL unique ID for global communicator.\n"
           "- node_master_nccl_id: 128-byte NCCL unique ID for node master communicator.\n"
           "- config: Model configuration.\n"
           "- options: Runtime/training options.\n"
           "- batch_size: Per-GPU batch size.\n"
           "- seq_len: Sequence length.\n"
           "- grad_accum: Gradient accumulation steps.\n"
           "- memcpy_all_gather/memcpy_send_recv: Enable memcpy-based collectives.\n"
           "- lora_config: Optional LoRA configuration.\n"
           "- qlora_config: Optional QLoRA configuration.\n\n"
           "Returns: SurogateTrainer")
        ;

    nb::class_<DataLoader>(m, "DataLoader",
        "Streaming token dataset loader.\n\n"
        "Loads fixed-size chunks from a list of token files and fills preallocated arrays.\n"
        "Supports distributed sharding via rank/world_size for multi-node training.")
        .def("__init__", [](DataLoader *d, const std::vector<std::string>& file_list, int chunk_size,
                           int rank = 0, int world_size = 1, unsigned long seed = 42) {
            new (d) DataLoader(file_list, chunk_size, rank, world_size, seed);
        }, nb::arg("file_list"), nb::arg("chunk_size"), nb::arg("rank") = 0, nb::arg("world_size") = 1, nb::arg("seed") = 42,
           "Create a DataLoader.\n\n"
           "Parameters:\n"
           "- file_list: List of dataset file paths.\n"
           "- chunk_size: Chunk size in tokens.\n"
           "- rank: Global rank of this worker (0 to world_size-1). Default: 0.\n"
           "- world_size: Total number of workers. Default: 1.\n"
           "- seed: RNG seed controlling shuffling/order.\n\n"
           "For distributed training, each worker should use different rank with the same world_size.\n"
           "Data is sharded via strided access: rank i reads chunks i, i+world_size, i+2*world_size, etc.")
        .def("load_batch", [](DataLoader* d, TokenArray inputs, TokenArray targets) {
            Tensor inp_t{ETensorDType::INT32, {static_cast<long>(inputs.shape(0)), static_cast<long>(inputs.shape(1))},
                        reinterpret_cast<std::byte*>(inputs.data()), nullptr, 2, inputs.device_id()};
            Tensor tgt_t{ETensorDType::INT32, {static_cast<long>(targets.shape(0)), static_cast<long>(targets.shape(1))},
                        reinterpret_cast<std::byte*>(targets.data()), nullptr, 2, inputs.device_id()};
            d->load_batch(inp_t, tgt_t);
        }, nb::arg("inputs"), nb::arg("targets"),
             "Fill `inputs` and `targets` with the next batch.\n\n"
             "Parameters:\n"
             "- inputs: Preallocated int32 array [batch, seq_len].\n"
             "- targets: Preallocated int32 array [batch, seq_len].")
        .def("load_batch", [](DataLoader* d, TokenArray inputs, TokenArray targets, TokenArray position_ids) {
            CHECK_SHAPE(position_ids, static_cast<int>(inputs.shape(0)), static_cast<int>(inputs.shape(1)));
            Tensor inp_t{ETensorDType::INT32, {static_cast<long>(inputs.shape(0)), static_cast<long>(inputs.shape(1))},
                        reinterpret_cast<std::byte*>(inputs.data()), nullptr, 2, inputs.device_id()};
            Tensor tgt_t{ETensorDType::INT32, {static_cast<long>(targets.shape(0)), static_cast<long>(targets.shape(1))},
                        reinterpret_cast<std::byte*>(targets.data()), nullptr, 2, inputs.device_id()};
            Tensor pos_t{ETensorDType::INT32, {static_cast<long>(position_ids.shape(0)), static_cast<long>(position_ids.shape(1))},
                        reinterpret_cast<std::byte*>(position_ids.data()), nullptr, 2, position_ids.device_id()};
            d->load_batch(inp_t, tgt_t, &pos_t);
        }, nb::arg("inputs"), nb::arg("targets"), nb::arg("position_ids"),
             "Fill `inputs`, `targets`, and `position_ids` with the next batch.\n\n"
             "Parameters:\n"
             "- inputs: Preallocated int32 array [batch, seq_len].\n"
             "- targets: Preallocated int32 array [batch, seq_len].\n"
             "- position_ids: Preallocated int32 array [batch, seq_len].")
        .def("epoch", &DataLoader::epoch, "Return the current epoch number (0-based).")
        .def("progress", &DataLoader::progress, "Return progress within the current epoch (percent).")
        .def("advance_epoch", &DataLoader::advance_epoch, "Advance to the next epoch and reshuffle chunk order.")
        .def("has_next", &DataLoader::has_next, nb::arg("chunks") = 1,
             "Return True if at least `chunks` more chunks are available in the current epoch.")
        .def("set_state", &DataLoader::set_state, nb::arg("seed"), nb::arg("epoch"), nb::arg("file_index"), nb::arg("chunk_index"),
             "Set the internal iteration state.\n\n"
             "Parameters:\n- seed: RNG seed.\n- epoch: Epoch number.\n- file_index: Current file index.\n- chunk_index: Current chunk index within the file.")
        .def_prop_ro("seq_len", &DataLoader::seq_len, "Sequence length produced by this loader.")
        .def_prop_ro("vocab_size", &DataLoader::vocab_size, "Vocabulary size declared by the dataset.")
        .def_prop_ro("num_files", &DataLoader::num_files, "Number of files in `file_list`.")
        .def_prop_ro("num_chunks", &DataLoader::num_chunks, "Total number of chunks across all files.")
        .def_prop_ro("num_tokens", &DataLoader::num_tokens, "Total number of tokens across all files.")
        .def_prop_ro("seed", &DataLoader::seed, "Current RNG seed.")
        ;

    m.def("find_latest_checkpoint", find_latest_checkpoint,
          "Find the latest checkpoint step in a checkpoint directory.\n\n"
          "Returns: implementation-defined (typically an integer step or optional).");
    m.def("get_all_checkpoints", get_all_checkpoints,
          "List all checkpoints in a checkpoint directory.\n\n"
          "Returns: implementation-defined (typically a list of step numbers).");
    m.def("get_checkpoint_path", get_checkpoint_path,
          "Build the filesystem path for a given checkpoint.\n\n"
          "Returns: checkpoint path string.");
    m.def("clean_old_checkpoints", clean_old_checkpoints,
          "Delete old checkpoints according to the library's retention policy.\n\n"
          "Use with care: this removes files from disk.");
    m.def("get_num_gpus", [](){ int count; CUDA_CHECK(cudaGetDeviceCount(&count)); return count; },
          "Return the number of CUDA devices visible to this process.");

    m.def("generate_nccl_id", []() -> nb::bytes {
        auto id = NCCLCommunicator::generate_nccl_id();
        return nb::bytes(reinterpret_cast<const char*>(id.data()), 128);
    }, "Generate a new NCCL unique ID (128 bytes) for multi-node coordination.\n\n"
       "This ID must be shared across all nodes before launching distributed training.\n"
       "Typically, the master node generates the ID and broadcasts it to all workers.");

    nb::enum_<TrainingRunLogger::EVerbosity>(m, "LogVerbosity",
        "Logger verbosity level.")
        .value("SILENT", TrainingRunLogger::EVerbosity::SILENT, "No output.")
        .value("QUIET", TrainingRunLogger::EVerbosity::QUIET, "Minimal output.")
        .value("DEFAULT", TrainingRunLogger::EVerbosity::DEFAULT, "Default output.")
        .value("VERBOSE", TrainingRunLogger::EVerbosity::VERBOSE, "Verbose output.")
        ;

    nb::class_<TrainingRunLogger>(m, "TrainingRunLogger", nb::dynamic_attr(),
        "Training run logger.\n\n"
        "Writes a structured log to `file_name` and optionally forwards messages to a Python callback.")
        .def("__init__", [](TrainingRunLogger *t, const std::string& file_name, nb::object callback_obj, TrainingRunLogger::EVerbosity verbosity) {
            new (t) TrainingRunLogger(file_name, 0, verbosity);
            if(!callback_obj.is_none()) {
                auto cb = nb::cast<nb::callable>(callback_obj);
                // set as an attribute on the python object to keep all ownership with python
                nb::setattr(nb::cast(t), "_callback", cb);
                t->set_callback([t](const std::string_view& msg) {
                    nb::gil_scoped_acquire gil;
                    auto cb = nb::cast<nb::callable>(nb::getattr(nb::cast(t), "_callback"));
                    cb(nb::cast(std::string(msg)));
                });
            }
        }, nb::arg("file_name"), nb::arg("callback") = nb::none(), nb::arg("verbosity") = TrainingRunLogger::EVerbosity::DEFAULT,
           "Create a logger.\n\n"
           "Parameters:\n"
           "- file_name: Output log file path.\n"
           "- callback: Optional Python callable that receives log strings.\n"
           "- verbosity: LogVerbosity level.")
        .def("log_cmd", [](TrainingRunLogger* logger, const std::vector<std::string>& args) {
            std::vector<const char*> argv;
            argv.reserve(args.size());
            for (const auto& arg : args) {
                argv.push_back(arg.c_str());
            }
            logger->log_cmd(args.size(), argv.data());
        }, nb::arg("args"),
           "Log command line arguments.\n\n"
           "Parameters:\n- args: List of argv-style strings.")
        .def("log_options", [](TrainingRunLogger* logger, const nb::dict& options) {
            std::vector<std::pair<std::string_view, std::variant<bool, long, float, std::string>>> cpp_options;
            std::vector<std::string> keys;
            keys.reserve(options.size());
            cpp_options.reserve(options.size());
            for (auto item : options) {
                nb::object value = nb::cast<nb::object>(item.second);
                keys.push_back(nb::cast<std::string>(item.first));
                std::string& key = keys.back(); // ensure key has sufficient lifetime
                if (nb::isinstance<nb::bool_>(value)) {
                    cpp_options.emplace_back(key, nb::cast<bool>(value));
                } else if (nb::isinstance<nb::int_>(value)) {
                    cpp_options.emplace_back(key, nb::cast<long>(value));
                } else if (nb::isinstance<nb::float_>(value)) {
                    cpp_options.emplace_back(key, nb::cast<float>(value));
                } else if (nb::isinstance<nb::str>(value)) {
                    cpp_options.emplace_back(key, nb::cast<std::string>(value));
                } else {
                    throw std::runtime_error("Unsupported option type for key: " + key);
                }
            }
            logger->log_options(cpp_options);
        }, nb::arg("options"),
           "Log training options.\n\n"
           "Parameters:\n- options: dict[str, (bool|int|float|str)]. Unsupported value types raise.")
        .def("log_dataset", &TrainingRunLogger::log_dataset, nb::arg("train_loader"), nb::arg("eval_loader"),
             "Log dataset information.\n\n"
             "Parameters:\n- train_loader: Training DataLoader.\n- eval_loader: Evaluation DataLoader.")
        .def("log_step", nb::overload_cast<int, float, int, int, float, float, float>(&TrainingRunLogger::log_step),
             nb::arg("step"), nb::arg("epoch"), nb::arg("step_tokens"), nb::arg("duration_ms"),
             nb::arg("norm"), nb::arg("loss"), nb::arg("lr"),
             "Log a training step.\n\n"
             "Parameters:\n"
             "- step: Global step index.\n"
             "- epoch: Current epoch.\n"
             "- step_tokens: Tokens processed in this step.\n"
             "- duration_ms: Step wall time.\n"
             "- norm: Gradient norm.\n"
             "- loss: Training loss.\n"
             "- lr: Learning rate.")
        .def("log_step_moe", nb::overload_cast<int, float, int, int, float, float, float, float, float, float, float>(&TrainingRunLogger::log_step),
             nb::arg("step"), nb::arg("epoch"), nb::arg("step_tokens"), nb::arg("duration_ms"),
             nb::arg("norm"), nb::arg("loss"), nb::arg("lr"),
             nb::arg("moe_aux_loss"), nb::arg("moe_z_loss"), nb::arg("moe_load_imbalance"),
             nb::arg("moe_expert_utilization"),
             "Log a training step with MoE metrics inline.\n\n"
             "Parameters:\n"
             "- step: Global step index.\n"
             "- epoch: Current epoch.\n"
             "- step_tokens: Tokens processed in this step.\n"
             "- duration_ms: Step wall time.\n"
             "- norm: Gradient norm.\n"
             "- loss: Training loss.\n"
             "- lr: Learning rate.\n"
             "- moe_aux_loss: MoE auxiliary load balancing loss.\n"
             "- moe_z_loss: MoE router z-loss.\n"
             "- moe_load_imbalance: MoE load imbalance ratio.\n"
             "- moe_expert_utilization: Fraction of experts receiving tokens.")
        .def("set_phase", &TrainingRunLogger::set_phase, nb::arg("phase"),
             "Set the current training phase label shown in step logs.\n\n"
             "Parameters:\n- phase: Phase name (e.g. 'warmup', 'converging', 'plateau').")
        .def("log_eval", &TrainingRunLogger::log_eval,
             nb::arg("step"), nb::arg("epoch"), nb::arg("eval_tokens"), nb::arg("duration_ms"), nb::arg("loss"),
             "Log an evaluation step.\n\n"
             "Parameters:\n"
             "- step: Global step index.\n"
             "- epoch: Current epoch.\n"
             "- eval_tokens: Tokens processed.\n"
             "- duration_ms: Eval wall time.\n"
             "- loss: Eval loss.")
        .def("log_moe_stats", &TrainingRunLogger::log_moe_stats,
             nb::arg("step"), nb::arg("aux_loss"), nb::arg("z_loss"),
             nb::arg("expert_utilization"), nb::arg("load_imbalance"),
             "Log MoE training statistics.\n\n"
             "Parameters:\n"
             "- step: Global step index.\n"
             "- aux_loss: Load balancing auxiliary loss.\n"
             "- z_loss: Router z-loss.\n"
             "- expert_utilization: Fraction of experts receiving tokens (0-1).\n"
             "- load_imbalance: Ratio of max to mean token counts (1.0 = balanced).")
        .def("log_gpu_state", &TrainingRunLogger::log_gpu_state,
             nb::arg("step"), nb::arg("gpu_id"), nb::arg("gpu_util"),
             "Log GPU utilization state.\n\n"
             "Parameters:\n- step: Global step.\n- gpu_id: GPU index.\n- gpu_util: GPUUtilInfo snapshot.")
        .def("log_message", &TrainingRunLogger::log_message,
             nb::arg("step"), nb::arg("msg"),
             "Log a free-form message.\n\n"
             "Parameters:\n- step: Global step.\n- msg: Message string.")
        .def("log_allocator", [](TrainingRunLogger* logger, const nb::dict& stats) {
            std::vector<std::pair<std::string, sSegmentMemory>> cpp_stats;
            std::vector<std::pair<std::string, long>> cpp_stack;
            cpp_stats.reserve(stats.size());
            for (auto item : stats) {
                std::string key = nb::cast<std::string>(item.first);
                nb::dict value = nb::cast<nb::dict>(item.second);
                if (value.contains("stack")) {
                    cpp_stack.emplace_back(key, nb::cast<long>(value["stack"]));
                } else {
                    long device = nb::cast<long>(value["device"]);
                    long managed = nb::cast<long>(value["managed"]);
                    long pinned = nb::cast<long>(value["pinned"]);
                    long pageable = nb::cast<long>(value["pageable"]);
                    cpp_stats.emplace_back(key, sSegmentMemory{device, managed, pinned, pageable});
                }
            }
            logger->log_allocator(cpp_stats, cpp_stack);
        }, nb::arg("stats"),
           "Log memory allocator statistics.\n\n"
           "Parameters:\n"
           "- stats: dict[str, dict]. For segments: keys {device, managed, pinned, pageable}. "
           "For stacks: key {'stack': bytes}.")
         .def("set_expected_time_per_token", [](TrainingRunLogger* logger, const MultiGPUPyTrainer* trainer){
             auto& config = trainer->config();
             auto& options = trainer->options();
             // Use sol_compute_dtype() for accurate SOL estimation:
             // - QLoRA FP4/FP8: dequantizes to BF16, so actual compute is BF16
             // - Non-QLoRA FP8/FP4 recipes: actual compute in FP8/FP4
             const bool is_qlora = trainer->is_qlora();
             auto ops = get_transformer_ops(
                 config.NumLayers * ((long)config.HiddenSize * (config.IntermediateSize * 3 + config.HiddenSize * 1 + config.qkv_channels())),
                 options.sol_compute_dtype(is_qlora), (long)config.VocabSize * config.HiddenSize, config.DType,
                 config.NumQueryHeads * config.head_size(), config.NumLayers, trainer->seq_length());
             logger->log_sol_estimate(ops, trainer->world_size());
         }, nb::arg("trainer"),
             "Log a compute/throughput estimate based on the current model/trainer configuration.\n\n"
             "Parameters:\n- trainer: SurogateTrainer instance to read config/options/shape from.")
        ;

    // Disable leak warnings during interpreter shutdown - these are false positives
    // caused by Python's non-deterministic cleanup order, not actual memory leaks.
    // See: https://nanobind.readthedocs.io/en/latest/refleaks.html
    nb::set_leak_warnings(false);
}
