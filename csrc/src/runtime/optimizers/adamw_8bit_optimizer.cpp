// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// 8-bit AdamW optimizer (Flash-quantized). Owns its own state, checkpoint
// containers, and CPU-streaming fallback path.

#include "runtime/optimizers/adamw_8bit_optimizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "kernels/kernels.h"
#include "runtime/core/fp8_scaling_state.h"
#include "runtime/dsl/dsl_grad_store.h"
#include "runtime/dsl/dsl_model.h"
#include "runtime/dsl/dsl_model_internal.h"
#include "runtime/dsl/dsl_run_state.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/optimizers/cpu_adamw.h"
#include "runtime/optimizers/flash_adamw_8bit.h"
#include "runtime/optimizers/gradient_mask.h"
#include "utilities/comm.h"
#include "utilities/tensor.h"
#include "utilities/tensor_container.h"

namespace optimizers {

using namespace dsl::internal;

namespace {

class AdamW8BitMomentumContainer final : public ITensorContainer {
public:
    AdamW8BitMomentumContainer() = default;

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        if (!mState1 || !mState1->Data) return;
        callback("adamw8bit.state1", TensorShard(*mState1));
        if (mScales1 && mScales1->Data) {
            callback("adamw8bit.scales1", TensorShard(*mScales1));
        }
    }

    void update_pointers(Tensor* state1, Tensor* scales1) {
        mState1 = state1;
        mScales1 = scales1;
    }

private:
    Tensor* mState1 = nullptr;
    Tensor* mScales1 = nullptr;
};

class AdamW8BitVarianceContainer final : public ITensorContainer {
public:
    AdamW8BitVarianceContainer() = default;

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        if (!mState2 || !mState2->Data) return;
        callback("adamw8bit.state2", TensorShard(*mState2));
        if (mScales2 && mScales2->Data) {
            callback("adamw8bit.scales2", TensorShard(*mScales2));
        }
    }

    void update_pointers(Tensor* state2, Tensor* scales2) {
        mState2 = state2;
        mScales2 = scales2;
    }

private:
    Tensor* mState2 = nullptr;
    Tensor* mScales2 = nullptr;
};

/// A view of `n` consecutive elements of `t`, starting at element `offset`.
Tensor flat_range(const Tensor& t, std::size_t offset, std::size_t n) {
    return Tensor::from_pointer(t.Data + offset * get_dtype_size(t.DType),
                                t.Device,
                                t.DType,
                                std::array<long, 1>{static_cast<long>(n)});
}

/// The slice of a whole (replicated) master that this rank updated.
struct OwnedSlice {
    Tensor master;
    std::size_t offset;
    std::size_t n;
};

/// All-gather every rank's updated slice into the whole masters, in place: this rank's slice
/// already sits in its own slot (rank * n), which is NCCL's in-place convention. Every rank
/// passes the same tensors in the same order.
void all_gather_owned_slices(const std::vector<OwnedSlice>& slices,
                             int rank,
                             int world,
                             NCCLCommunicator& comm,
                             cudaEvent_t done,
                             cudaStream_t stream) {
    // Bounds the collectives in one NCCL group.
    constexpr std::size_t kPerTransaction = 64;
    for (std::size_t begin = 0; begin < slices.size(); begin += kPerTransaction) {
        const std::size_t end = std::min(slices.size(), begin + kPerTransaction);
        comm.begin_transaction(stream);
        for (std::size_t i = begin; i < end; ++i) {
            Tensor master = slices[i].master;
            const Tensor own = flat_range(master, slices[i].offset, slices[i].n);
            comm.schedule_all_gather(
                TensorShard(own, rank, world, std::array<long, 1>{static_cast<long>(master.nelem())}),
                master);
        }
        comm.execute_transaction(done);
        // The all-gather runs on the comm stream; `stream` refreshes the work weights from it next.
        CUDA_CHECK(cudaStreamWaitEvent(stream, done, 0));
    }
}

}  // namespace

struct AdamW8BitOptimizer::Impl {
    struct State {
        bool initialized = false;
        size_t total_params = 0;
        size_t total_state_elems = 0;
        size_t num_groups = 0;
        bool offload_state = false;
        bool use_zero_copy = false;
        Tensor state1;   // int8 softsign-quantized momentum
        Tensor state2;   // uint8 sqrt-quantized variance
        Tensor scales1;  // FP16 per-group scales for momentum
        Tensor scales2;  // FP16 per-group scales for variance
    };

    /// Where one trainable parameter's state lives, and which of its elements this rank updates.
    struct ParamSlot {
        std::string name;
        size_t state_offset = 0;   // GROUP_SIZE-aligned offset into state1/state2
        size_t elem_offset = 0;    // first element of the master this rank updates
        size_t n = 0;              // elements this rank updates and keeps state for
        bool wm_sharded = false;   // the master is this rank's ZeRO-3 weight shard
        bool partitioned = false;  // every rank holds the whole master; this rank owns [elem_offset, +n)
    };

    std::unique_ptr<State> state;
    std::vector<ParamSlot> slots;  // in param_names() order, fixed when the state is allocated
    bool any_partitioned = false;
    cudaEvent_t gather_done = nullptr;
    AdamW8BitMomentumContainer momentum_container;
    AdamW8BitVarianceContainer variance_container;

    // CPU FP32 AdamW state for the CPU-streaming path
    CPUAdamWState cpu_state;

    ~Impl() {
        if (gather_done) {
            (void)cudaEventDestroy(gather_done);
        }
    }

    void ensure_state() {
        if (!state) {
            state = std::make_unique<State>();
        }
    }
};

/// One update pass: the eager step passes its hyperparameters; the graphed step passes zeros and
/// the device-resident opt_params / opt_step that every replay reads.
struct AdamW8BitOptimizer::UpdateArgs {
    float learning_rate;
    float beta1;
    float beta2;
    int step;
    float epsilon;
    float weight_decay;  // eager: the configured decay; graphed: 1, a scale on opt_params[4]
    const float* opt_params;
    const int* opt_step;
    const char* caller;
};

AdamW8BitOptimizer::AdamW8BitOptimizer()
    : mImpl(std::make_unique<Impl>()) {
}

AdamW8BitOptimizer::~AdamW8BitOptimizer() = default;

ITensorContainer* AdamW8BitOptimizer::momentum_container() {
    return &mImpl->momentum_container;
}

ITensorContainer* AdamW8BitOptimizer::variance_container() {
    return &mImpl->variance_container;
}

void AdamW8BitOptimizer::init_state(dsl::DslModel& model, cudaStream_t stream) {
    mImpl->ensure_state();
    auto& state = *mImpl->state;
    if (state.initialized) {
        return;
    }

    constexpr size_t GROUP_SIZE = FLASH_ADAMW8BIT_GROUP_SIZE;
    const bool use_weight_manager = (model.mWeightManager != nullptr);
    const bool sharded_weights = use_weight_manager && model.mOptions.ShardWeights && (model.mNumShards > 1);
    // ZeRO-1: a master every rank holds whole -- every master below ZeRO-3, the replicated
    // embedding / lm_head at ZeRO-3 -- gets state for 1/world of its elements on each rank. Each
    // rank updates its slice and the slices are all-gathered after the step. Not across nodes
    // (only the first node writes checkpoints), not with expert parallelism (expert masters differ
    // between EP ranks), not under dispatch-PP (one GPU runs the whole update).
    const int world = model.mNumShards;
    const bool partition =
        world > 1 && model.mNumNodes <= 1 && model.mOptions.EPSize <= 1 && !model.mDispatchPpLocalGrads;

    auto& slots = mImpl->slots;
    slots.clear();
    mImpl->any_partitioned = false;
    std::unordered_set<void*> seen_grads;
    size_t total_params = 0;
    size_t state_elems = 0;
    for (const auto& name : model.mGrads->param_names()) {
        bool accumulate = false;
        Tensor* grad = model.mGrads->get_param_grad(name, accumulate);
        (void)accumulate;
        if (!grad) {
            continue;
        }
        // A tied lm_head shares the embedding's gradient: the embedding's update is the only one
        // either name gets, so the alias needs no state of its own.
        if (grad->Data && !seen_grads.insert(static_cast<void*>(grad->Data)).second) {
            continue;
        }
        Tensor& master = use_weight_manager ? model.mWeightManager->get_master(name) : model.mParams->get(name);
        const size_t nelem = master.nelem();

        Impl::ParamSlot slot;
        slot.name = name;
        slot.wm_sharded = sharded_weights && model.mWeightManager->is_sharded(name);
        // The in-place all-gather needs equal slices, and NCCL cannot gather into an offloaded
        // (host) master; those keep whole state and are updated whole on every rank.
        slot.partitioned =
            partition && !slot.wm_sharded && master.Device >= 0 && nelem > 0 && nelem % static_cast<size_t>(world) == 0;
        slot.n = slot.partitioned ? nelem / static_cast<size_t>(world) : nelem;
        slot.elem_offset = slot.partitioned ? static_cast<size_t>(model.mShardIdx) * slot.n : 0;

        state_elems = (state_elems + GROUP_SIZE - 1) / GROUP_SIZE * GROUP_SIZE;
        slot.state_offset = state_elems;
        state_elems += slot.n;
        total_params += slot.n;
        mImpl->any_partitioned = mImpl->any_partitioned || slot.partitioned;
        slots.push_back(std::move(slot));
    }

    state.total_params = total_params;
    state.total_state_elems = state_elems;
    state.num_groups = flash_adamw8bit_num_scales(state.total_state_elems);

    state.offload_state = model.mOptions.OffloadOptimizer;
    state.use_zero_copy = model.mOptions.UseZeroCopy;
    EAllocationType alloc_kind = EAllocationType::ON_DEVICE;
    if (state.offload_state) {
        if (state.use_zero_copy) {
            alloc_kind = model.mOptions.offload_alloc();
        } else {
            alloc_kind = EAllocationType::ON_DEVICE;
        }
    }

    state.state1 =
        model.mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state1", alloc_kind, {(long)state.total_state_elems});
    state.state2 =
        model.mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state2", alloc_kind, {(long)state.total_state_elems});
    state.scales1 =
        model.mAllocator->allocate(ETensorDType::FP16, "adamw8bit_scales1", alloc_kind, {(long)state.num_groups});
    state.scales2 =
        model.mAllocator->allocate(ETensorDType::FP16, "adamw8bit_scales2", alloc_kind, {(long)state.num_groups});

    init_flash_adamw8bit_state(reinterpret_cast<signed char*>(state.state1.template get<std::byte>()),
                               reinterpret_cast<unsigned char*>(state.state2.template get<std::byte>()),
                               state.scales1.template get<half>(),
                               state.scales2.template get<half>(),
                               state.total_state_elems,
                               stream);

    if (mImpl->any_partitioned && !mImpl->gather_done) {
        CUDA_CHECK(cudaEventCreateWithFlags(&mImpl->gather_done, cudaEventDisableTiming));
    }

    state.initialized = true;
    mImpl->momentum_container.update_pointers(&state.state1, &state.scales1);
    mImpl->variance_container.update_pointers(&state.state2, &state.scales2);
}

void AdamW8BitOptimizer::on_restore_checkpoint(dsl::DslModel& /*model*/) {
    if (mImpl->state && mImpl->state->state1.Data) {
        mImpl->state->initialized = true;
        mImpl->momentum_container.update_pointers(&mImpl->state->state1, &mImpl->state->scales1);
        mImpl->variance_container.update_pointers(&mImpl->state->state2, &mImpl->state->scales2);
    }
}

void AdamW8BitOptimizer::prepare_for_checkpoint_load(dsl::DslModel& model) {
    mImpl->ensure_state();
    cudaStream_t stream = model.mRunState ? model.mRunState->MainStream : cudaStreamDefault;
    if (!mImpl->state->initialized) {
        init_state(model, stream);
    }
}

void AdamW8BitOptimizer::step(dsl::DslModel& model,
                              NCCLCommunicator& comm,
                              const OptimizerConfig& config,
                              int step_idx) {
    if (!model.mRunState || !model.mParams || !model.mGrads) {
        throw std::logic_error("AdamW8BitOptimizer::step called before allocate_run_state()");
    }
    if (model.lora_enabled()) {
        model.update_lora_adamw_8bit(comm,
                                     config.learning_rate,
                                     config.adamw_beta1,
                                     config.adamw_beta2,
                                     step_idx,
                                     config.adamw_epsilon,
                                     config.weight_decay,
                                     config.grad_clip);
        model.lora_weights().advance_sync_generation();
        return;
    }

    // CPU-RAM centric streaming path
    if (model.mGrads->is_streaming_grads()) {
        if (gradient_mask::enabled()) {
            throw std::runtime_error("AdamW8BitOptimizer::step: gradient masks are not supported with streaming grads");
        }
        step_cpu_streaming(model,
                           comm,
                           config.learning_rate,
                           config.adamw_beta1,
                           config.adamw_beta2,
                           step_idx,
                           config.adamw_epsilon,
                           config.weight_decay,
                           config.grad_clip);
        return;
    }

    update(model,
           comm,
           config,
           UpdateArgs{config.learning_rate,
                      config.adamw_beta1,
                      config.adamw_beta2,
                      step_idx,
                      config.adamw_epsilon,
                      config.weight_decay,
                      nullptr,
                      nullptr,
                      "AdamW8BitOptimizer::step"});
}

void AdamW8BitOptimizer::step_graph(dsl::DslModel& model,
                                    NCCLCommunicator& comm,
                                    const OptimizerConfig& config,
                                    const float* opt_params,
                                    const int* opt_step) {
    if (!model.mRunState || !model.mParams || !model.mGrads) {
        throw std::logic_error("AdamW8BitOptimizer::step_graph called before allocate_run_state()");
    }
    if (model.lora_enabled()) {
        model.update_lora_adamw_8bit_graph(comm, config.grad_clip, opt_params, opt_step);
        model.lora_weights().advance_sync_generation();
        return;
    }
    if (!mImpl->state) {
        throw std::logic_error("AdamW8BitOptimizer::step_graph: optimizer state not allocated");
    }

    update(model,
           comm,
           config,
           UpdateArgs{/*learning_rate=*/0.f,
                      /*beta1=*/0.f,
                      /*beta2=*/0.f,
                      /*step=*/1,
                      /*epsilon=*/0.f,
                      /*weight_decay=*/1.f,
                      opt_params,
                      opt_step,
                      "AdamW8BitOptimizer::step_graph"});
}

void AdamW8BitOptimizer::update(dsl::DslModel& model,
                                NCCLCommunicator& comm,
                                const OptimizerConfig& config,
                                const UpdateArgs& args) {
    const std::string caller = args.caller;
    mImpl->ensure_state();

    auto& rs = *model.mRunState;
    cudaStream_t stream = rs.MainStream;
    wait_event_if_not_capturing(stream, rs.BackwardDone);

    const bool use_weight_manager = (model.mWeightManager != nullptr);
    auto master_of = [&](const std::string& name) -> Tensor& {
        return use_weight_manager ? model.mWeightManager->get_master(name) : model.mParams->get(name);
    };
    const bool sharded_weights = use_weight_manager && model.mOptions.ShardWeights && (model.mNumShards > 1);
    auto param_is_sharded = [&](const std::string& name) -> bool {
        return sharded_weights && model.mWeightManager->is_sharded(name);
    };

    // dispatch-PP collects complete grads onto this GPU by hand (no DDP), so skip
    // the cross-GPU grad/norm reductions that would deadlock against the idle pool.
    const bool dispatch_local = model.mDispatchPpLocalGrads;
    const bool grads_reduced = comm.world_size() > 1;
    if (grads_reduced && !dispatch_local) {
        if (model.mGrads->is_reduce_pending()) {
            wait_event_if_not_capturing(stream, rs.all_reduce_done_event());
            model.mGrads->clear_reduce_pending();
        } else {
            model.mGrads->reduce_all(comm, stream);
        }
    }

    const bool grad_mask_enabled = gradient_mask::enabled();
    std::unordered_map<void*, bool> trainable_grad_ptrs;
    if (grad_mask_enabled) {
        for (const auto& name : model.mGrads->param_names()) {
            bool accumulate = false;
            Tensor* grad = model.mGrads->get_param_grad(name, accumulate);
            (void)accumulate;
            if (!grad || !grad->Data) continue;
            auto [it, inserted] =
                trainable_grad_ptrs.emplace(static_cast<void*>(grad->Data), gradient_mask::whole_param_trainable(name));
            if (!inserted) {
                it->second = it->second || gradient_mask::whole_param_trainable(name);
            }
        }

        std::unordered_set<void*> seen_mask_ptrs;
        for (const auto& name : model.mGrads->param_names()) {
            Tensor& val = master_of(name);
            bool accumulate = false;
            Tensor* grad = model.mGrads->get_param_grad(name, accumulate);
            (void)accumulate;
            if (!grad || !grad->Data) continue;

            const bool param_sharded = param_is_sharded(name);
            Tensor grad_view =
                param_sharded ? static_cast<Tensor>(shard_view(*grad, model.mShardIdx, model.mNumShards)) : *grad;
            if (param_sharded && grad_view.nelem() != val.nelem()) {
                throw std::runtime_error(caller + ": sharded grad size mismatch for " + name);
            }

            auto it = trainable_grad_ptrs.find(static_cast<void*>(grad->Data));
            const bool trainable = (it != trainable_grad_ptrs.end()) && it->second;
            if (!trainable) {
                if (seen_mask_ptrs.insert(static_cast<void*>(grad->Data)).second) {
                    fill_zero(grad_view, stream);
                }
                continue;
            }
            gradient_mask::apply_partial_mask(name, grad_view, stream);
        }
    }

    model.calculate_gradient_norm(comm, config.grad_clip, stream, grads_reduced || dispatch_local);
    const float* grad_scale = rs.scratch().norm_buffer.template get<float>() + 1;

    if (!mImpl->state->initialized) {
        if (stream_is_capturing(stream)) {
            throw std::runtime_error(caller + ": optimizer state must be initialized before capture");
        }
        init_state(model, stream);
    }
    if (mImpl->any_partitioned && (dispatch_local || comm.world_size() != model.mNumShards)) {
        throw std::logic_error(caller + ": the optimizer state is partitioned over " +
                               std::to_string(model.mNumShards) + " GPUs, but this update does not run on all of them");
    }

    auto& state = *mImpl->state;
    constexpr size_t GROUP_SIZE = FLASH_ADAMW8BIT_GROUP_SIZE;
    std::vector<OwnedSlice> owned;

    for (const auto& slot : mImpl->slots) {
        Tensor& master = master_of(slot.name);
        bool accumulate = false;
        Tensor* grad = model.mGrads->get_param_grad(slot.name, accumulate);
        (void)accumulate;
        if (!grad || !master.Data || !grad->Data) {
            continue;
        }
        if (grad_mask_enabled) {
            auto it = trainable_grad_ptrs.find(static_cast<void*>(grad->Data));
            if (it != trainable_grad_ptrs.end() && !it->second) {
                continue;
            }
        }

        Tensor val = master;
        Tensor grad_view = *grad;
        if (slot.wm_sharded) {
            grad_view = static_cast<Tensor>(shard_view(*grad, model.mShardIdx, model.mNumShards));
            if (grad_view.nelem() != master.nelem()) {
                throw std::runtime_error(caller + ": sharded grad size mismatch for " + slot.name);
            }
        } else if (slot.partitioned) {
            // The slice is reduced either way: ZeRO-1 all-reduces the whole gradient, and ZeRO-2's
            // reduce-scatter leaves this rank's reduced slice in exactly this range.
            if (grad->nelem() != master.nelem()) {
                throw std::runtime_error(caller + ": grad size mismatch for " + slot.name);
            }
            val = flat_range(master, slot.elem_offset, slot.n);
            grad_view = flat_range(*grad, slot.elem_offset, slot.n);
        }
        if (val.nelem() != slot.n) {
            throw std::logic_error(caller + ": " + slot.name + " changed size after its optimizer state was laid out");
        }

        const float wd = gradient_mask::weight_decay_for(slot.name, args.weight_decay);
        const size_t n = slot.n;
        const size_t group_offset = slot.state_offset / GROUP_SIZE;
        signed char* s1 = reinterpret_cast<signed char*>(state.state1.template get<std::byte>()) + slot.state_offset;
        unsigned char* s2 =
            reinterpret_cast<unsigned char*>(state.state2.template get<std::byte>()) + slot.state_offset;
        half* sc1 = state.scales1.template get<half>() + group_offset;
        half* sc2 = state.scales2.template get<half>() + group_offset;

        if (val.DType == ETensorDType::FP32) {
            if (grad_view.DType == ETensorDType::FP32) {
                flash_adamw_update_8bit(val.template get<float>(),
                                        grad_view.template get<float>(),
                                        s1,
                                        s2,
                                        sc1,
                                        sc2,
                                        n,
                                        args.learning_rate,
                                        args.beta1,
                                        args.beta2,
                                        args.step,
                                        args.epsilon,
                                        wd,
                                        grad_scale,
                                        args.opt_params,
                                        args.opt_step,
                                        stream);
            } else if (grad_view.DType == ETensorDType::BF16) {
                throw std::runtime_error(caller + ": FP32 param with BF16 grad not supported for flash adamw 8-bit");
            } else {
                throw std::runtime_error(caller + ": unsupported grad dtype for " + slot.name);
            }
        } else if (val.DType == ETensorDType::BF16) {
            if (grad_view.DType != ETensorDType::BF16) {
                throw std::runtime_error(caller + ": unsupported grad dtype for " + slot.name);
            }
            flash_adamw_update_8bit(val.template get<nv_bfloat16>(),
                                    grad_view.template get<nv_bfloat16>(),
                                    s1,
                                    s2,
                                    sc1,
                                    sc2,
                                    n,
                                    args.learning_rate,
                                    args.beta1,
                                    args.beta2,
                                    args.step,
                                    args.epsilon,
                                    wd,
                                    grad_scale,
                                    args.opt_params,
                                    args.opt_step,
                                    stream);
        } else {
            throw std::runtime_error(caller + ": unsupported param dtype for " + slot.name);
        }

        if (slot.partitioned) {
            owned.push_back(OwnedSlice{master, slot.elem_offset, slot.n});
        }
    }

    // Each rank updated only its slice of a partitioned master: gather the slices so every rank
    // holds the whole updated master before the work weights are refreshed from it.
    all_gather_owned_slices(owned, model.mShardIdx, model.mNumShards, comm, mImpl->gather_done, stream);

    if (rs.has_fp8_delayed_scaling()) {
        if (auto* fp8_state = rs.get_fp8_scaling_state()) {
            delayed_scaling_update(*fp8_state, stream);
        }
    }

    if (model.mWeightManager) {
        model.mWeightManager->invalidate();
        model.mWeightManager->sync_work_from_master(stream);
    }

    if (!stream_is_capturing(stream)) {
        CUDA_CHECK(cudaEventSynchronize(rs.NormDone));
        if (!std::isfinite(*rs.GradScaleHost)) {
            throw std::runtime_error(caller + ": grad_scale is NaN/Inf");
        }
    }
    record_event_if_not_capturing(rs.OptimizerDone, stream);
}

void AdamW8BitOptimizer::prepare_for_graph(dsl::DslModel& model,
                                           NCCLCommunicator& comm,
                                           const OptimizerConfig& /*config*/) {
    if (!model.mRunState) {
        throw std::logic_error("AdamW8BitOptimizer::prepare_for_graph called before allocate_run_state()");
    }

    auto& rs = *model.mRunState;
    cudaStream_t stream = rs.MainStream;
    bool did_work = false;

    if (model.lora_enabled()) {
        if (!model.mLoRAAdamW8BitState) {
            throw std::logic_error("AdamW8BitOptimizer::prepare_for_graph: LoRA optimizer state not allocated");
        }
        if (!model.mLoRARunState->norm_ptrs_initialized) {
            model.populate_lora_norm_pointers(comm, stream);
            did_work = true;
        }
        if (!model.mLoRAAdamW8BitState->initialized) {
            model.initialize_lora_multi_tensor_state(comm, stream);
            did_work = true;
        }
        if (!model.mLoRAAdamW8BitState->grad_ptrs_initialized) {
            model.update_lora_grad_pointers(comm, stream);
            model.mLoRAAdamW8BitState->grad_ptrs_initialized = true;
            did_work = true;
        }
    } else {
        // A graphed run captures before any eager optimizer step, so the state the
        // first step() would create lazily has to be created here.
        mImpl->ensure_state();
        if (!mImpl->state->initialized) {
            init_state(model, stream);
            did_work = true;
        }
    }

    if (did_work) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

void AdamW8BitOptimizer::step_cpu_streaming(dsl::DslModel& model,
                                            NCCLCommunicator& /*comm*/,
                                            float learning_rate,
                                            float beta_1,
                                            float beta_2,
                                            int t,
                                            float epsilon,
                                            float weight_decay,
                                            float grad_clip) {
    auto& rs = *model.mRunState;
    cudaStream_t stream = rs.MainStream;

    // 1. Wait for backward done
    wait_event_if_not_capturing(stream, rs.BackwardDone);

    // 2. Wait for all D2H gradient copies to complete
    model.mGrads->wait_all_offloads(stream);

    // 3. Compute gradient norm on CPU from offloaded gradients.
    double norm_sq = cpu_gradient_norm_squared(model.mGrads->get_cpu_grads_map(), model.mGrads->param_names());
    float raw_norm = static_cast<float>(std::sqrt(norm_sq));

    float token_scale = 1.0f;
    if (model.mUseTokenScale && rs.ValidTokenCount.Data) {
        int valid_tokens = 0;
        CUDA_CHECK(cudaMemcpy(&valid_tokens, rs.ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost));
        if (valid_tokens > 0) {
            token_scale = 1.0f / static_cast<float>(valid_tokens);
        }
    } else {
        float total_tokens =
            static_cast<float>(rs.B) * static_cast<float>(rs.T) * static_cast<float>(std::max(1, rs.GradAccumSteps));
        token_scale = 1.0f / total_tokens;
    }

    float scaled_norm = raw_norm * token_scale;

    float clip_scale = 1.0f;
    if (grad_clip > 0.0f && scaled_norm > grad_clip) {
        clip_scale = grad_clip / scaled_norm;
    }

    float grad_scale = token_scale * clip_scale;

    if (rs.NormHost) {
        rs.NormHost[0] = scaled_norm;
    }
    CUDA_CHECK(cudaEventRecord(rs.NormDone, stream));

    // 4. Initialize CPU optimizer state if needed
    auto& cpu_state = mImpl->cpu_state;
    if (!cpu_state.initialized) {
        std::size_t total = 0;
        for (const auto& name : model.mGrads->param_names()) {
            total += model.mWeightManager->get_master(name).nelem();
        }
        cpu_state.m.resize(total, 0.0f);
        cpu_state.v.resize(total, 0.0f);
        cpu_state.total_params = total;
        cpu_state.initialized = true;
    }

    // 5. Per-parameter CPU optimizer step
    std::size_t offset = 0;
    for (const auto& name : model.mGrads->param_names()) {
        Tensor& master = model.mWeightManager->get_master(name);
        const Tensor& grad = model.mGrads->get_cpu_grad(name);
        const auto n = static_cast<std::size_t>(master.nelem());

        if (master.DType == ETensorDType::FP32 && grad.DType == ETensorDType::FP32) {
            cpu_adamw_step(master.get<float>(),
                           grad.get<float>(),
                           cpu_state.m.data() + offset,
                           cpu_state.v.data() + offset,
                           n,
                           learning_rate,
                           beta_1,
                           beta_2,
                           t,
                           epsilon,
                           weight_decay,
                           grad_scale);
        } else if (master.DType == ETensorDType::FP32 && grad.DType == ETensorDType::BF16) {
            cpu_adamw_step_bf16(master.get<float>(),
                                grad.Data,
                                cpu_state.m.data() + offset,
                                cpu_state.v.data() + offset,
                                n,
                                learning_rate,
                                beta_1,
                                beta_2,
                                t,
                                epsilon,
                                weight_decay,
                                grad_scale);
        } else if (master.DType == ETensorDType::BF16) {
            cpu_adamw_step_bf16_param(master.Data,
                                      grad.Data,
                                      cpu_state.m.data() + offset,
                                      cpu_state.v.data() + offset,
                                      n,
                                      learning_rate,
                                      beta_1,
                                      beta_2,
                                      t,
                                      epsilon,
                                      weight_decay,
                                      grad_scale);
        } else {
            throw std::runtime_error("AdamW8BitOptimizer::step_cpu_streaming: unsupported dtype combination for " +
                                     name + " (master=" + std::to_string(static_cast<int>(master.DType)) +
                                     ", grad=" + std::to_string(static_cast<int>(grad.DType)) + ")");
        }
        offset += n;
    }

    // 6. Sync updated master weights to GPU work copies
    model.mWeightManager->invalidate();
    model.mWeightManager->sync_work_from_master(stream);

    record_event_if_not_capturing(rs.OptimizerDone, stream);
}

}  // namespace optimizers
