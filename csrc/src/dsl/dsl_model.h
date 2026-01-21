// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model wrapper (IR validation + execution).

#ifndef SUROGATE_SRC_DSL_DSL_MODEL_H
#define SUROGATE_SRC_DSL_DSL_MODEL_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dsl/ir.h"
#include "modules/model_config.h"
#include "training/model.h"
#include "utilities/allocator.h"
#include "utilities/tensor_container.h"

namespace dsl {

class IGraphExecutor;
class DslParamStore;
class DslGradStore;
class DslRunState;

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
             const std::shared_ptr<TensorAllocator>& allocator);
    ~DslModel() override;

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override;
    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) override;
    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override;
    void update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t, float epsilon, float weight_decay, float grad_clip) override;
    void update_with_config(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) override;

    ITensorContainer& weights() override;
    ITensorContainer& opt_momentum() override { return mAdamWMomentumContainer; }
    ITensorContainer& opt_momentum_scales() override { return mEmpty; }
    ITensorContainer& opt_variance() override { return mAdamWVarianceContainer; }
    ITensorContainer& opt_variance_scales() override { return mEmpty; }

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

    std::unique_ptr<PretrainedConfig> mConfig;
    modules::ModelConfig mModelConfig;
    RuntimeOptions mOptions{};
    std::shared_ptr<TensorAllocator> mAllocator;
    std::unique_ptr<DslRunState> mRunState;
    IRFile mIr;
    const Module* mModule = nullptr;
    std::unique_ptr<DslParamStore> mParams;
    std::unique_ptr<DslGradStore> mGrads;
    EmptyTensorContainer mEmpty;
    detail::AdamW8BitMomentumContainer mAdamWMomentumContainer;
    detail::AdamW8BitVarianceContainer mAdamWVarianceContainer;
    std::unique_ptr<IGraphExecutor> mExecutor;
    std::vector<std::byte> mRngState;

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
