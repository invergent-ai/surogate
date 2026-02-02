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

#include "dsl/qlora_provider.h"
#include "utilities/allocator.h"
#include "utilities/tensor_container.h"

namespace dsl {

struct Module;
struct Graph;
class DslWeightManager;
}

struct RuntimeOptions;
struct PretrainedConfig;
namespace modules { struct ModularLoRAConfig; }

namespace dsl {

// Stores model parameters defined by the DSL IR.
class DslParamStore final : public ITensorContainer {
public:
    struct Entry {
        Tensor tensor;
        bool trainable = true;
        bool external = false;  ///< Provided by QLoRA weight provider (no local storage)
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
    void set_qlora_provider(QLoRAWeightProvider* provider) { mQLoRAProvider = provider; }
    /// Access the QLoRA provider (if any).
    [[nodiscard]] QLoRAWeightProvider* qlora_provider() const { return mQLoRAProvider; }
    /// Wire a DslWeightManager (optional).
    void set_weight_manager(DslWeightManager* manager) { mWeightManager = manager; }
    /// Set default stream for provider-backed resolution.
    void set_default_stream(cudaStream_t stream) { mDefaultStream = stream; }

    const std::vector<std::string>& param_names() const { return mParamOrder; }

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

} // namespace dsl

#endif // SUROGATE_SRC_DSL_DSL_PARAM_STORE_H
