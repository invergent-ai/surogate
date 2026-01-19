// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Placeholder DSL model wrapper (validation-only).

#ifndef SUROGATE_SRC_DSL_DSL_MODEL_H
#define SUROGATE_SRC_DSL_DSL_MODEL_H

#include <memory>
#include <string>

#include "dsl/ir.h"
#include "training/model.h"
#include "utilities/allocator.h"
#include "utilities/tensor_container.h"

namespace dsl {

class EmptyTensorContainer final : public ITensorContainer {
public:
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>&) override {}
};

class DslModel final : public IModel {
public:
    DslModel(const PretrainedConfig& config,
             const RuntimeOptions& options,
             const std::string& ir_json,
             const std::shared_ptr<TensorAllocator>& allocator);

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override;
    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) override;
    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override;
    void update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t, float epsilon, float weight_decay, float grad_clip) override;

    ITensorContainer& weights() override { return mEmpty; }
    ITensorContainer& opt_momentum() override { return mEmpty; }
    ITensorContainer& opt_momentum_scales() override { return mEmpty; }
    ITensorContainer& opt_variance() override { return mEmpty; }
    ITensorContainer& opt_variance_scales() override { return mEmpty; }

    std::vector<std::byte> rng_state() const override { return {}; }
    void set_rng_state(const std::vector<std::byte>&) override {}

    void init_weights(NCCLCommunicator& comm) override;
    void import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) override;
    void on_restore_checkpoint(NCCLCommunicator& comm) override;
    void export_weights(const std::string& file_name, NCCLCommunicator& comm) override;

    void allocate_run_state(const RuntimeOptions& options, NCCLCommunicator& comm,
                            int B, int T, bool allocate_optimizer) override;

    std::string_view model_type() const override;
    IRunState& get_run_state() const override;

private:
    void validate_ir();
    const Module& pick_model_module(const IRFile& ir) const;
    void validate_config_mapping(const Module& module) const;
    void validate_param_shapes(const Module& module) const;

    std::unique_ptr<PretrainedConfig> mConfig;
    std::shared_ptr<TensorAllocator> mAllocator;
    std::unique_ptr<IRunState> mRunState;
    IRFile mIr;
    const Module* mModule = nullptr;
    EmptyTensorContainer mEmpty;
};

} // namespace dsl

#endif // SUROGATE_SRC_DSL_DSL_MODEL_H
