// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model wrapper (IR validation + execution).

#ifndef SUROGATE_SRC_DSL_DSL_MODEL_H
#define SUROGATE_SRC_DSL_DSL_MODEL_H

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "dsl/ir.h"
#include "modules/model_config.h"
#include "modules/lora/lora_config.h"
#include "modules/lora/lora_grads_manager.h"
#include "modules/lora/lora_optimizer_state.h"
#include "modules/lora/lora_run_state.h"
#include "modules/lora/lora_weights_manager.h"
#include "training/model.h"
#include "utilities/allocator.h"
#include "utilities/tensor_container.h"

namespace dsl {

class IGraphExecutor;
class DslParamStore;
class DslGradStore;
class DslRunState;
class DslWeightManager;

class EmptyTensorContainer final : public ITensorContainer {
public:
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>&) override {}
};

namespace detail {

class AdamW8BitMomentumContainer final : public ITensorContainer {
public:
    AdamW8BitMomentumContainer(Tensor* state1 = nullptr, Tensor* absmax1 = nullptr)
        : mState1(state1), mAbsmax1(absmax1) {}

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        if (!mState1 || !mState1->Data) return;
        callback("adamw8bit.state1", TensorShard(*mState1));
        if (mAbsmax1 && mAbsmax1->Data) {
            callback("adamw8bit.absmax1", TensorShard(*mAbsmax1));
        }
    }

    void update_pointers(Tensor* state1, Tensor* absmax1) {
        mState1 = state1;
        mAbsmax1 = absmax1;
    }

private:
    Tensor* mState1 = nullptr;
    Tensor* mAbsmax1 = nullptr;
};

class AdamW8BitVarianceContainer final : public ITensorContainer {
public:
    AdamW8BitVarianceContainer(Tensor* state2 = nullptr, Tensor* absmax2 = nullptr)
        : mState2(state2), mAbsmax2(absmax2) {}

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        if (!mState2 || !mState2->Data) return;
        callback("adamw8bit.state2", TensorShard(*mState2));
        if (mAbsmax2 && mAbsmax2->Data) {
            callback("adamw8bit.absmax2", TensorShard(*mAbsmax2));
        }
    }

    void update_pointers(Tensor* state2, Tensor* absmax2) {
        mState2 = state2;
        mAbsmax2 = absmax2;
    }

private:
    Tensor* mState2 = nullptr;
    Tensor* mAbsmax2 = nullptr;
};

}  // namespace detail

class DslModel final : public IModel {
public:
    DslModel(const PretrainedConfig& config,
             const RuntimeOptions& options,
             const std::string& ir_json,
             const std::shared_ptr<TensorAllocator>& allocator,
             const std::optional<modules::ModularLoRAConfig>& lora_config = std::nullopt);
    ~DslModel() override;

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override;
    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) override;
    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override;
    void update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t, float epsilon, float weight_decay, float grad_clip) override;
    void update_with_config(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) override;
    void update_with_graph_params(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config,
                                  const float* opt_params, const int* opt_step);
    void prepare_optimizer_state_for_graph(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config);
    void zero_grads(cudaStream_t stream);
    void set_internal_graphs_enabled(bool enabled);
    [[nodiscard]] bool internal_graphs_enabled() const;

    ITensorContainer& weights() override;
    ITensorContainer& opt_momentum() override;
    ITensorContainer& opt_momentum_scales() override;
    ITensorContainer& opt_variance() override;
    ITensorContainer& opt_variance_scales() override;

    // LoRA adapter API (no-op for non-LoRA models).
    void export_adapter(const std::string& directory, NCCLCommunicator& comm, const std::string& base_model_path = "");
    void import_adapter(const std::string& file_name, NCCLCommunicator& comm);
    void save_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) override;
    void load_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) override;

    [[nodiscard]] bool lora_enabled() const { return mLoRAConfig.has_value() && mLoRAConfig->enabled(); }
    [[nodiscard]] bool qlora_enabled() const { return false; }
    [[nodiscard]] bool is_moe_model() const { return mIsMoEModel; }
    [[nodiscard]] std::size_t lora_num_parameters() const {
        return mLoRAWeights ? mLoRAWeights->num_parameters() : 0;
    }
    [[nodiscard]] std::size_t qlora_quantized_weights_bytes() const { return 0; }
    [[nodiscard]] float qlora_memory_savings_ratio() const { return 0.f; }
    DslModel& base_model() { return *this; }
    [[nodiscard]] const modules::ModelConfig& config() const { return mModelConfig; }

    // Weight streaming/sharding support
    [[nodiscard]] bool is_weight_streaming_enabled() const;
    [[nodiscard]] DslWeightManager* weight_manager() { return mWeightManager.get(); }

    modules::ModularLoRAWeightsManager& lora_weights();
    modules::ModularLoRAGradsManager& lora_grads();
    modules::LoRARunState& lora_run_state();
    [[nodiscard]] const modules::ModularLoRAConfig& lora_config() const { return *mLoRAConfig; }

    std::vector<std::byte> rng_state() const override;
    void set_rng_state(const std::vector<std::byte>& state) override;

    void init_weights(NCCLCommunicator& comm) override;
    void import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) override;
    void on_restore_checkpoint(NCCLCommunicator& comm) override;
    void export_weights(const std::string& file_name, NCCLCommunicator& comm) override;
    void prepare_optimizer_for_checkpoint_load() override;

    void allocate_run_state(const RuntimeOptions& options, NCCLCommunicator& comm,
                            int B, int T, bool allocate_optimizer) override;

    float get_loss() const override;
    float get_accuracy() const override;

    std::string_view model_type() const override;
    IRunState& get_run_state() const override;

    struct MappingSpec {
        enum class Kind { Direct, Fuse, Split, Transform, TiedTo, Unknown };
        Kind kind = Kind::Unknown;
        std::string source;
        std::vector<std::string> sources;
        std::vector<std::pair<long, long>> ranges;
        std::string fn;
        std::string target;
        int dim = 0;
        bool optional = false;
    };

private:
    void validate_ir();
    const Module& pick_model_module(const IRFile& ir) const;
    void validate_config_mapping(const Module& module) const;
    void validate_param_shapes(const Module& module) const;
    void init_optimizer_state(cudaStream_t stream);
    void calculate_gradient_norm(NCCLCommunicator& comm, float grad_clip, cudaStream_t stream, bool grads_reduced);
    void allocate_lora_run_state(NCCLCommunicator& comm, int B, int T);
    void ensure_lora_run_state(NCCLCommunicator& comm, int B, int T);
    void update_lora_adamw_8bit(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2,
                               int t, float epsilon, float weight_decay, float grad_clip);
    void update_adamw_8bit_graph(NCCLCommunicator& comm, float grad_clip,
                                 const float* opt_params, const int* opt_step);
    void update_lora_adamw_8bit_graph(NCCLCommunicator& comm, float grad_clip,
                                      const float* opt_params, const int* opt_step);
    void update_lora_normuon(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step);
    void calculate_lora_gradient_norm(NCCLCommunicator& comm, float grad_clip);
    void initialize_lora_multi_tensor_state(NCCLCommunicator& comm, cudaStream_t stream);
    void update_lora_grad_pointers(NCCLCommunicator& comm, cudaStream_t stream);

    std::unique_ptr<PretrainedConfig> mConfig;
    modules::ModelConfig mModelConfig;
    RuntimeOptions mOptions{};
    std::shared_ptr<TensorAllocator> mAllocator;
    std::unique_ptr<DslRunState> mRunState;
    IRFile mIr;
    const Module* mModule = nullptr;
    std::unique_ptr<DslParamStore> mParams;
    std::unique_ptr<DslGradStore> mGrads;
    std::unique_ptr<DslWeightManager> mWeightManager;  // Optional - for streaming/sharding
    EmptyTensorContainer mEmpty;
    detail::AdamW8BitMomentumContainer mAdamWMomentumContainer;
    detail::AdamW8BitVarianceContainer mAdamWVarianceContainer;
    std::unique_ptr<IGraphExecutor> mExecutor;
    std::vector<std::byte> mRngState;

    // LoRA state (optional)
    std::optional<modules::ModularLoRAConfig> mLoRAConfig;
    std::unique_ptr<modules::ModularLoRAWeightsManager> mLoRAWeights;
    std::unique_ptr<modules::ModularLoRAGradsManager> mLoRAGrads;
    std::unique_ptr<modules::LoRARunState> mLoRARunState;
    std::unique_ptr<modules::LoRAAdamW8BitState> mLoRAAdamW8BitState;
    std::unique_ptr<modules::LoRANorMuonState> mLoRANorMuonState;
    bool mIsMoEModel = false;

    struct AdamW8BitState {
        bool initialized = false;
        size_t total_params = 0;
        size_t total_state_elems = 0;
        size_t num_blocks = 0;
        Tensor quantiles1;
        Tensor quantiles2;
        Tensor state1;
        Tensor state2;
        Tensor absmax1;
        Tensor absmax2;
    };
    std::unique_ptr<AdamW8BitState> mAdamW8BitState;

    std::unordered_map<std::string, MappingSpec> mHfMapping;
    std::unordered_map<std::string, MappingSpec> mHfExport;
};

} // namespace dsl

#endif // SUROGATE_SRC_DSL_DSL_MODEL_H
