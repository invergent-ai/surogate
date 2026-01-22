// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL runtime components (run state + weights + gradients).

#ifndef SUROGATE_SRC_DSL_DSL_RUNTIME_H
#define SUROGATE_SRC_DSL_DSL_RUNTIME_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dsl/ir.h"
#include "modules/run_state_types.h"
#include "modules/residual_manager.h"
#include "training/model.h"
#include "utilities/allocator.h"
#include "utilities/tensor_container.h"

struct RuntimeOptions;
class NCCLCommunicator;
namespace modules { struct ModularLoRAConfig; }

namespace dsl {

// Stores model parameters defined by the DSL IR.
class DslParamStore final : public ITensorContainer {
public:
    struct Entry {
        Tensor tensor;
        bool trainable = true;
    };

    DslParamStore(const Module& module,
                  const Graph& graph,
                  const RuntimeOptions& options,
                  const PretrainedConfig& config,
                  const std::shared_ptr<TensorAllocator>& allocator,
                  const modules::ModularLoRAConfig* lora_config = nullptr);

    Tensor& get(const std::string& name);
    const Tensor& get(const std::string& name) const;
    bool has(const std::string& name) const;
    bool is_trainable(const std::string& name) const;

    const std::vector<std::string>& param_names() const { return mParamOrder; }

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;

private:
    std::shared_ptr<TensorAllocator> mAllocator;
    std::unordered_map<std::string, Entry> mParams;
    std::vector<std::string> mParamOrder;
};

// Stores parameter gradients for DSL execution.
class DslGradStore {
public:
    DslGradStore(const DslParamStore& params,
                 const std::shared_ptr<TensorAllocator>& allocator);

    void start_micro_step(cudaStream_t stream, int micro_step, int total_steps);
    void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm);

    Tensor* get_param_grad(const std::string& name, bool& accumulate);

    void zero_all(cudaStream_t stream);
    void reduce_all(NCCLCommunicator& comm, cudaStream_t stream);

    const std::vector<std::string>& param_names() const { return mParamOrder; }
    const std::unordered_map<std::string, Tensor>& grads() const { return mGrads; }

private:
    std::shared_ptr<TensorAllocator> mAllocator;
    std::unordered_map<std::string, Tensor> mGrads;
    std::vector<std::string> mParamOrder;
    bool mAccumulate = false;
};

// DSL run state for graph execution (activation buffers, scratch, etc).
class DslRunState final : public IRunState {
public:
    DslRunState(const PretrainedConfig& config,
                const RuntimeOptions& options,
                int B, int T,
                const std::shared_ptr<TensorAllocator>& allocator);
    ~DslRunState();

    modules::SimplifiedLayerActivations& simplified_acts(int layer_idx) { return mSimplifiedActivations[layer_idx]; }
    modules::SimplifiedLayerGradients& simplified_grads(int layer_idx) { return mSimplifiedGradients[layer_idx]; }
    modules::SimplifiedQuantGradients& simplified_quant_grads() { return mSimplifiedQuantGrads; }

    modules::NonBlockActivations& non_block_activations() { return mNonBlockActivations; }
    modules::NonBlockGradientBuffers& non_block_gradients() { return mNonBlockGradients; }
    modules::ScratchBuffers& scratch() { return mScratch; }

    Tensor& get_residual(int layer_idx, cudaStream_t stream);
    Tensor& get_final_residual();

    bool ffn_temps_on_stack() const { return mRecomputeBlock; }
    bool large_bwd_temps_on_stack() const { return mRecomputeBlock; }
    bool is_lora_only_mode() const { return false; }

    cudaStream_t side_stream() const { return mSideStream; }
    cudaEvent_t side_stream_event() const { return mSideStreamEvent; }

    // IRunState overrides (quantization unsupported in DSL runtime for now).
    [[nodiscard]] bool has_activation_quants() const override { return false; }
    [[nodiscard]] bool has_grad_quants() const override { return false; }
    [[nodiscard]] bool has_fp8_forward() const override { return false; }
    [[nodiscard]] bool has_fp8_hybrid_backward() const override { return false; }
    [[nodiscard]] bool has_fp8_delayed_scaling() const override { return false; }
    [[nodiscard]] bool has_fp4_forward() const override { return false; }
    [[nodiscard]] bool has_fp4_backward() const override { return false; }

private:
    void allocate_non_block_state(const PretrainedConfig& cfg);
    void allocate_simplified_activations(const PretrainedConfig& cfg);
    void allocate_simplified_gradients(const PretrainedConfig& cfg);
    void allocate_scratch_buffers(const PretrainedConfig& cfg);
    void allocate_residual_buffers(const PretrainedConfig& cfg);
    void create_cuda_resources();
    void release_cuda_resources() noexcept;

    std::shared_ptr<TensorAllocator> mAllocator;
    Tensor mStackBuffer{};
    bool mRecomputeBlock = false;
    ETensorDType mActivationDtype = ETensorDType::BF16;
    ETensorDType mGradDtype = ETensorDType::BF16;

    modules::NonBlockActivations mNonBlockActivations;
    modules::NonBlockGradientBuffers mNonBlockGradients;
    modules::ScratchBuffers mScratch;

    std::vector<modules::SimplifiedLayerActivations> mSimplifiedActivations;
    std::vector<modules::SimplifiedLayerGradients> mSimplifiedGradients;
    modules::SimplifiedQuantGradients mSimplifiedQuantGrads;

    std::unique_ptr<modules::ResidualManager> mResidualManager;

    // CUDA resources
    cudaStream_t mSideStream = nullptr;
    cudaEvent_t mSideStreamEvent = nullptr;
};

} // namespace dsl

#endif // SUROGATE_SRC_DSL_DSL_RUNTIME_H
