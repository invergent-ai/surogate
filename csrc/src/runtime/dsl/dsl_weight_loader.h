// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DslWeightLoader: Standalone weight loading engine that interprets
// DSL mapping specifications to load HuggingFace SafeTensors into
// pre-allocated Tensor buffers.
//
// This class encapsulates all weight-loading logic (Direct, Fuse, Split,
// Transform, TiedTo, StackExperts) independent of DslModel. It produces
// BF16 (or original-dtype) tensors with no quantization dependency.
//
// Usage:
//   SafeTensorsReader reader(path);
//   DslWeightLoader loader(reader, mapping, config, allocator, {shard_idx, num_shards});
//   for (const auto& name : param_names) {
//       Tensor& param = store.get(name);
//       loader.load_param(name, param, is_sharded, &global_template, stream);
//   }
//   loader.resolve_tied_params([&](const std::string& n) -> Tensor& { return store.get(n); });

#ifndef SUROGATE_SRC_RUNTIME_DSL_DSL_WEIGHT_LOADER_H
#define SUROGATE_SRC_RUNTIME_DSL_DSL_WEIGHT_LOADER_H

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "runtime/dsl/mapping_spec.h"
#include "utilities/tensor.h"

class SafeTensorsReader;
class SafeTensorEntry;
class TensorAllocator;
class PretrainedConfig;

namespace dsl {

/// Sharding configuration for multi-GPU weight loading.
struct ShardConfig {
    int shard_idx = 0;     ///< This GPU's shard index (0-based).
    int num_shards = 1;    ///< Total number of shards (GPUs).
};

/// MoE configuration for StackExperts loading.
struct MoEWeightConfig {
    int num_experts = 0;               ///< Number of experts (0 = auto from spec).
    int moe_intermediate_size = 0;     ///< Expert intermediate dimension.
};

/// Standalone weight loader that reads HF SafeTensors using MappingSpec rules.
///
/// The loader is constructed once per import operation and processes all
/// parameters sequentially. TiedTo parameters are deferred and resolved
/// after all other parameters have been loaded.
///
/// Thread safety: NOT thread-safe. Use one loader per import operation.
class DslWeightLoader {
public:
    /// Construct a weight loader.
    ///
    /// @param reader       SafeTensors reader (file already opened).
    /// @param mapping      Mapping table from internal names to HF specs.
    /// @param config       Pretrained model configuration (for fuse slice inference).
    /// @param allocator    Tensor allocator (for temporary buffers in Transform).
    /// @param shard        Sharding configuration for multi-GPU loading.
    /// @param moe_config   MoE configuration for StackExperts (optional).
    DslWeightLoader(SafeTensorsReader& reader,
                    const MappingTable& mapping,
                    const PretrainedConfig& config,
                    TensorAllocator& allocator,
                    ShardConfig shard = {},
                    MoEWeightConfig moe_config = {});

    ~DslWeightLoader();

    /// Load a single parameter by internal name.
    ///
    /// Looks up the mapping spec for `name`, resolves {layer} placeholders,
    /// and loads the weight data into `target`. For TiedTo mappings, the
    /// parameter is deferred to resolve_tied_params().
    ///
    /// @param name          Internal parameter name (e.g. "blocks[3].qkv_weight").
    /// @param target        Pre-allocated tensor to fill with loaded data.
    /// @param allow_cast    Allow dtype conversion during load.
    /// @param param_sharded Whether this parameter is row-sharded across GPUs.
    /// @param global_template  For sharded params: the unsharded template tensor
    ///                         (provides global shape). nullptr if not sharded.
    /// @param stream        CUDA stream for async operations (Transform, StackExperts).
    ///                      nullptr uses the default stream.
    ///
    /// @return true if the parameter was loaded (or deferred as TiedTo),
    ///         false if the mapping was optional and the HF tensor was not found.
    ///
    /// @throws std::runtime_error if a required HF tensor is missing or shapes mismatch.
    bool load_param(const std::string& name,
                    Tensor& target,
                    bool allow_cast,
                    bool param_sharded = false,
                    const Tensor* global_template = nullptr,
                    cudaStream_t stream = nullptr);

    /// Try loading a parameter from multiple mapping specs in order.
    ///
    /// Attempts each spec in `options` until one succeeds. This enables
    /// fallback patterns (e.g. try fused QKV first, fall back to separate Q/K/V).
    ///
    /// @param options       Ordered list of (mapping_key, MappingSpec) to try.
    /// @param name          Internal parameter name (for error messages).
    /// @param target        Pre-allocated tensor to fill.
    /// @param allow_cast    Allow dtype conversion during load.
    /// @param layer_idx     Layer index for {layer} substitution (-1 = global).
    /// @param param_sharded Whether this parameter is row-sharded.
    /// @param global_template  Unsharded template tensor (for sharded params).
    /// @param stream        CUDA stream for async operations.
    ///
    /// @return true if any option succeeded, false if all failed (all optional).
    ///
    /// @throws std::runtime_error if no option succeeded and at least one was required.
    bool try_multiple(const std::vector<std::pair<std::string, MappingSpec>>& options,
                      const std::string& name,
                      Tensor& target,
                      bool allow_cast,
                      int layer_idx = -1,
                      bool param_sharded = false,
                      const Tensor* global_template = nullptr,
                      cudaStream_t stream = nullptr);

    /// Load a single expert from a StackExperts parameter into a 2D target buffer.
    ///
    /// Used for per-expert streaming during quantized import: instead of
    /// allocating the full [E, M, K] stacked tensor, the caller loads one
    /// expert at a time into a small [per_M, K] buffer.
    ///
    /// @param name        Internal parameter name (must have StackExperts mapping).
    /// @param expert_idx  Expert index (0-based).
    /// @param target      Pre-allocated 2D tensor [per_M, K] to fill.
    /// @param allow_cast  Allow dtype conversion during load.
    /// @param stream      CUDA stream for async operations.
    /// @return true if loaded successfully.
    bool load_expert(const std::string& name, int expert_idx,
                     Tensor& target, bool allow_cast, cudaStream_t stream = nullptr);

    /// Resolve all deferred TiedTo parameters.
    ///
    /// Must be called after all other parameters have been loaded.
    /// Copies data from source to destination for each TiedTo pair.
    ///
    /// @param get_tensor  Callback to retrieve a tensor by internal name.
    void resolve_tied_params(const std::function<Tensor&(const std::string&)>& get_tensor);

    /// Get the list of deferred TiedTo pairs (dest, source).
    [[nodiscard]] const std::vector<std::pair<std::string, std::string>>& tied_params() const {
        return mTiedParams;
    }

    /// Check whether a HF tensor name exists in the reader (non-throwing).
    [[nodiscard]] bool has_entry(const std::string& hf_name) const;

    /// Get the SafeTensorsReader (for advanced use cases).
    [[nodiscard]] SafeTensorsReader& reader() { return mReader; }

private:
    // ---- Mapping kind handlers ----

    bool load_direct(const MappingSpec& spec, const std::string& name,
                     Tensor& target, int layer_idx, bool allow_cast,
                     bool param_sharded, const Tensor* global_template);

    bool load_fused(const MappingSpec& spec, const std::string& name,
                    Tensor& target, int layer_idx, bool allow_cast,
                    bool param_sharded, const Tensor* global_template);

    bool load_split(const MappingSpec& spec, const std::string& name,
                    Tensor& target, int layer_idx, bool allow_cast,
                    bool param_sharded, const Tensor* global_template);

    bool load_transform(const MappingSpec& spec, const std::string& name,
                        Tensor& target, int layer_idx, bool allow_cast,
                        bool param_sharded, const Tensor* global_template,
                        cudaStream_t stream);

    bool load_stack_experts(const MappingSpec& spec, const std::string& name,
                            Tensor& target, int layer_idx, bool allow_cast,
                            bool param_sharded, cudaStream_t stream);

    // ---- Utilities ----

    /// Compute shard row range for a sharded parameter.
    [[nodiscard]] std::pair<long, long> shard_range(long global_rows, bool param_sharded) const;

    /// Compute row stride (product of all dims except dim0).
    [[nodiscard]] static long row_stride(const std::vector<long>& shape);

    /// Slice a tensor along dim0 (returns view, no copy).
    [[nodiscard]] static Tensor slice_dim0(const Tensor& base, long offset, long length);

    /// Infer fuse slice sizes from config (QKV, gate_up, etc.).
    [[nodiscard]] std::vector<long> infer_fuse_slices(const std::string& name, int num_sources) const;

    /// Format an HF name template by substituting {layer} and {expert}.
    [[nodiscard]] static std::string format_hf_name(std::string templ, int layer_idx, int expert_idx = -1);

    /// Parse "blocks[N].param_name" -> (layer_idx, param_name). Returns false if not a block param.
    [[nodiscard]] static bool parse_block_param(std::string_view name, int& layer_idx, std::string& param_name);

    /// Find the mapping spec for an internal parameter name.
    /// Sets layer_idx if the name is a block parameter. Returns nullptr if no mapping found.
    [[nodiscard]] const MappingSpec* find_mapping_spec(const std::string& internal_name, int& layer_idx) const;

    /// Resolve number of experts for StackExperts (from spec, config, or moe_config).
    [[nodiscard]] int resolve_num_experts(const MappingSpec& spec) const;

    // ---- State ----

    SafeTensorsReader& mReader;
    const MappingTable& mMapping;
    const PretrainedConfig& mConfig;
    TensorAllocator& mAllocator;
    ShardConfig mShard;
    MoEWeightConfig mMoEConfig;

    /// Deferred TiedTo pairs: (destination_name, source_name).
    std::vector<std::pair<std::string, std::string>> mTiedParams;
    Tensor mExpertScratch{};
    std::size_t mExpertScratchBytes = 0;
};

}  // namespace dsl

#endif  // SUROGATE_SRC_RUNTIME_DSL_DSL_WEIGHT_LOADER_H
