// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL parameter store - manages model parameters defined by DSL IR.

#ifndef SUROGATE_SRC_DSL_DSL_PARAM_STORE_H
#define SUROGATE_SRC_DSL_DSL_PARAM_STORE_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <unordered_set>

#include <cuda_runtime.h>

#include "runtime/core/qlora_provider.h"
#include "utilities/allocator.h"
#include "utilities/tensor_container.h"

namespace dsl {

struct Module;
struct Graph;
struct CompiledGraph;
struct PhaseArenas;
class DslWeightManager;
}  // namespace dsl

struct RuntimeOptions;
struct PretrainedConfig;
namespace modules {
struct ModularLoRAConfig;
}  // namespace modules

namespace dsl {

// Stores model parameters defined by the DSL IR.
class DslParamStore final : public ITensorContainer {
public:
    struct Entry {
        Tensor tensor;
        bool trainable = true;
        bool external = false;                   ///< Provided by QLoRA weight provider (no local storage)
        bool managed_by_weight_manager = false;  ///< Provided by DslWeightManager (no local storage)
    };

    DslParamStore(const Module& module,
                  const Graph& graph,
                  const RuntimeOptions& options,
                  const PretrainedConfig& config,
                  const std::shared_ptr<TensorAllocator>& allocator,
                  const modules::ModularLoRAConfig* lora_config = nullptr,
                  const std::unordered_set<std::string>* external_params = nullptr,
                  bool use_weight_manager = false);

    Tensor& get(const std::string& name);
    const Tensor& get(const std::string& name) const;
    bool has(const std::string& name) const;
    bool is_trainable(const std::string& name) const;
    bool is_external(const std::string& name) const;
    /// Return a template tensor (shape + dtype) without forcing provider resolution.
    const Tensor& template_tensor(const std::string& name) const;

    /// Wire an external QLoRA weight provider (optional).
    void set_qlora_provider(QLoRAWeightProvider* provider) {
        mQLoRAProvider = provider;
    }
    /// Access the QLoRA provider (if any).
    [[nodiscard]] QLoRAWeightProvider* qlora_provider() const {
        return mQLoRAProvider;
    }
    /// Wire a DslWeightManager (optional).
    void set_weight_manager(DslWeightManager* manager) {
        mWeightManager = manager;
    }
    /// Set default stream for provider-backed resolution.
    void set_default_stream(cudaStream_t stream) {
        mDefaultStream = stream;
    }

    const std::vector<std::string>& param_names() const {
        return mParamOrder;
    }

    /// Phase 4 M4a: route locally-allocated parameter storage through the
    /// Persistent arena. For each mParams entry whose tid has
    /// `RegionKind::Persistent` with a baked offset, copies the tensor's
    /// bytes to `arenas.persistent_ptr + meta.offset`, frees the original
    /// per-weight allocation, and rebinds `entry.tensor.Data` to the arena
    /// offset. Skips external (QLoRA) and weight-manager-managed entries
    /// (M4b/c territory). No-op unless `SUROGATE_USE_PHASE_PERSISTENT=1`.
    /// Must be called after arenas are allocated and parameter tensors
    /// contain their final values (i.e., after `import_weights`).
    void rebind_to_persistent_arena(const CompiledGraph& graph, const PhaseArenas& arenas, cudaStream_t stream);

    /// Bytes actually needed in the Persistent arena given the set of
    /// locally-allocated (non-external, non-weight-manager-managed) params
    /// in this store. Computed as max(offset + bytes) across those params'
    /// tids in `graph`. Zero if no rebindable param exists. Used by
    /// `GraphExecutor::compile_graphs` to clamp the Persistent arena size
    /// — `compute_arena_sizes` bases its estimate on every ForwardParam
    /// tid in the graph, but QLoRA-external and weight-manager-managed
    /// params should not consume arena bytes (their storage is elsewhere).
    std::size_t rebindable_persistent_bytes(const CompiledGraph& graph) const;

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;

private:
    std::shared_ptr<TensorAllocator> mAllocator;
    std::unordered_map<std::string, Entry> mParams;
    std::vector<std::string> mParamOrder;
    std::unordered_set<std::string> mExternalParams;
    QLoRAWeightProvider* mQLoRAProvider = nullptr;
    DslWeightManager* mWeightManager = nullptr;
    cudaStream_t mDefaultStream = cudaStreamDefault;
    bool mUsesWeightManager = false;
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_DSL_PARAM_STORE_H
