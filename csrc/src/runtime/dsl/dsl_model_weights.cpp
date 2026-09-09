// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model weight I/O operations (init, import, export, checkpoint).

#include "utilities/comm.h"
#include "runtime/dsl/dsl_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "runtime/dsl/dsl_model_internal.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/shared_master_store.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/dsl/dsl_weight_loader.h"
#include "runtime/executor/graph_executor.h"
#include "runtime/lora/lora_model_utils.h"
#include "runtime/qlora/adapter_merger.h"
#include "runtime/qlora/generic_qlora_provider.h"
#include "utilities/utils.h"
#include "utilities/safetensors.h"
#include "utilities/dtype.h"

#include <cuda_bf16.h>

namespace dsl {

std::vector<std::byte> DslModel::rng_state() const {
    if (mExecutor) {
        return mExecutor->rng_state();
    }
    return mRngState;
}

void DslModel::set_rng_state(const std::vector<std::byte>& state) {
    mRngState = state;
    if (mExecutor) {
        mExecutor->set_rng_state(state);
    }
}

void DslModel::init_weights(NCCLCommunicator& comm) {
    if (!mParams) {
        throw std::logic_error("DslModel::init_weights called before parameters are initialized");
    }

    const float scale = 0.02f;
    const float residual_scale = 1.0f / std::sqrt(2.0f * static_cast<float>(mConfig->NumLayers));
    unsigned long long seed = 42ULL;
    unsigned long long subseq = 0ULL;
    const unsigned long long shard_base = static_cast<unsigned long long>(mShardIdx) * 100000ULL;
    const bool use_weight_manager = (mWeightManager != nullptr);

    for (const auto& name : mParams->param_names()) {
        if (mParams->is_external(name) || mParams->is_storage_alias(name)) {
            continue;
        }
        Tensor& param = use_weight_manager ? mWeightManager->get_master(name) : mParams->get(name);
        if (name.ends_with("layer_scalar") || name.ends_with("per_expert_scale")) {
            fill_constant(param, 1.f, param.nelem(), nullptr);
            continue;
        }
        if (internal::is_bias_param_name(name)) {
            fill_zero(param, nullptr);
            continue;
        }
        if (internal::is_norm_param_name(name)) {
            fill_constant(param, 1.f, param.nelem(), nullptr);
            continue;
        }

        // Check if this is a projection weight that should be zeroed
        const bool is_out_proj = internal::contains_ci(name, "out_weight") || internal::contains_ci(name, "o_proj");
        const bool is_mlp_down =
            internal::contains_ci(name, "mlp_down_weight") || internal::contains_ci(name, "down_proj");
        if (mOptions.InitProjectionsToZero && (is_out_proj || is_mlp_down)) {
            fill_zero(param, nullptr);
            continue;
        }

        float stddev = scale;
        if (is_out_proj || is_mlp_down) {
            stddev *= residual_scale;
        }
        const bool param_sharded =
            use_weight_manager && mOptions.ShardWeights && (mNumShards > 1) && mWeightManager->is_sharded(name);
        const unsigned long long param_subseq = param_sharded ? (shard_base + subseq) : subseq;
        fill_normal(param, param.nelem(), 0.f, stddev, seed, param_subseq, nullptr);
        ++subseq;
    }

    if (lora_enabled()) {
        mLoRAWeights->random_init(42, comm);
    }

    if (mWeightManager) {
        cudaStream_t stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
        mWeightManager->sync_work_from_master(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Quantize frozen matmul block masters to FP8 in place so per-stage streaming ships
        // FP8 bytes (no-op unless dispatch-PP + fp8_hybrid). Must run after the masters are
        // populated and before the first gather.
        if (mRunState) {
            mWeightManager->finalize_fp8_block_masters(mRunState->DeviceProp, stream);
            mWeightManager->finalize_fp4_block_masters(mRunState->DeviceProp, stream);
        }
    }

    comm.barrier();
}

void DslModel::import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) {
    if (!mParams) {
        throw std::logic_error("DslModel::import_weights called before parameters are initialized");
    }

    SafeTensorsReader reader(file_name);

    if (qlora_enabled()) {
        if (!mLoRAConfig) {
            throw std::runtime_error("DSL model: QLoRA enabled without LoRA config");
        }
        if (mModelConfig.moe_config.has_value()) {
            const auto& moe = mModelConfig.moe_config.value();
            if (mQLoRAConfig.num_experts == 0) {
                mQLoRAConfig.num_experts = moe.num_experts;
            }
            if (mQLoRAConfig.num_experts_per_tok == 0) {
                mQLoRAConfig.num_experts_per_tok = moe.top_k;
            }
            if (mQLoRAConfig.moe_intermediate_size == 0) {
                mQLoRAConfig.moe_intermediate_size = moe.moe_intermediate_size;
            }
            if (mQLoRAConfig.num_shared_experts == 0 && moe.use_shared_expert) {
                mQLoRAConfig.num_shared_experts = 1;
            }
            if (mQLoRAConfig.moe_shared_expert_intermediate_size == 0 && moe.use_shared_expert) {
                mQLoRAConfig.moe_shared_expert_intermediate_size = moe.shared_expert_size;
            }
        } else if (mModelConfig.NumExperts > 0) {
            // Fallback: use ModelConfig fields directly when moe_config is not set
            // This can happen if MoE config wasn't provided by the DSL
            if (mQLoRAConfig.num_experts == 0) {
                mQLoRAConfig.num_experts = mModelConfig.NumExperts;
            }
            if (mQLoRAConfig.num_experts_per_tok == 0) {
                mQLoRAConfig.num_experts_per_tok = mModelConfig.NumExpertsPerTok;
            }
            if (mQLoRAConfig.moe_intermediate_size == 0) {
                mQLoRAConfig.moe_intermediate_size = mModelConfig.MoeIntermediateSize;
            }
        }

        mQLoRAProvider = internal::create_dsl_qlora_provider(*mModule,
                                                             mModelConfig,
                                                             *mConfig,
                                                             mOptions,
                                                             *mLoRAConfig,
                                                             mQLoRAConfig,
                                                             mAllocator,
                                                             mHfMapping,
                                                             mShardIdx,
                                                             mNumShards,
                                                             mAdapterPath);
        cudaStream_t quant_stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&quant_stream));
        mQLoRAProvider->import_and_quantize(file_name, comm, quant_stream);
        // import_and_quantize done: provider's device-resident storage
        // is finalized. Consolidate it onto a self-managed arena
        // (skipped automatically when the 2× transient peak wouldn't fit).
        mQLoRAProvider->consume_self_arena(quant_stream);
        CUDA_CHECK(cudaStreamSynchronize(quant_stream));
        CUDA_CHECK(cudaStreamDestroy(quant_stream));

        mParams->set_qlora_provider(mQLoRAProvider.get());
        if (mRunState) {
            mParams->set_default_stream(mRunState->MainStream);
        }
    }

    const bool sharded_weights = mWeightManager && mOptions.ShardWeights && (mNumShards > 1);
    const ShardConfig shard_config{mShardIdx, mNumShards};
    MoEWeightConfig moe_config;
    moe_config.num_experts = mModelConfig.moe_config && mModelConfig.moe_config->num_experts > 0
                                 ? mModelConfig.moe_config->num_experts
                                 : mModelConfig.NumExperts;
    DslWeightLoader loader(reader, mHfMapping, *mConfig, *mAllocator, shard_config, moe_config);

    // Adapter merge (stacked LoRA): merge previously-trained adapter into base weights.
    // This runs ONLY on the BF16 path; the QLoRA path handles merging in the pipeline.
    std::unique_ptr<qlora::AdapterMerger> adapter_merger;
    cudaStream_t adapter_stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
    if (!qlora_enabled() && !mAdapterPath.empty()) {
        adapter_merger = std::make_unique<qlora::AdapterMerger>(mAdapterPath, mHfMapping, reader, shard_config);
    }

    // Progress reporting (rank 0 only): importing a 100B+ model pages hundreds of GB
    // into pinned memory and can take minutes — without output it looks hung. Rank 0
    // sweeps the same ordered param list as every other rank and waits on the shared
    // master store for names other ranks claimed, so its loop position tracks global
    // progress.
    const bool report_progress = comm.rank() == 0;
    const std::size_t progress_total = mParams->param_names().size();
    std::size_t progress_done = 0;
    std::size_t progress_bytes = 0;
    const auto progress_start = std::chrono::steady_clock::now();
    auto progress_last = progress_start;

    for (const auto& name : mParams->param_names()) {
        if (report_progress) {
            ++progress_done;
            const auto now = std::chrono::steady_clock::now();
            if (now - progress_last >= std::chrono::seconds(5)) {
                progress_last = now;
                const double gb = static_cast<double>(progress_bytes) / 1e9;
                const double secs = std::chrono::duration<double>(now - progress_start).count();
                fprintf(stderr,
                        "[import] %zu/%zu tensors | %.1f GB | %.0fs elapsed | %.2f GB/s\n",
                        progress_done,
                        progress_total,
                        gb,
                        secs,
                        gb / std::max(secs, 1e-9));
                ::surogate::tick_watchdog_heartbeat();
                fflush(stderr);
            }
        }
        if (mParams->is_external(name) || mParams->is_storage_alias(name)) {
            continue;
        }
        Tensor& param = mWeightManager ? mWeightManager->get_master(name) : mParams->get(name);
        if (report_progress) {
            progress_bytes += param.bytes();
        }

        // Cross-GPU shared frozen base master: populate (read + page-lock) exactly once.
        // The first GPU manager to reach `name` claims it and reads/registers below; the
        // others wait for it and skip the redundant read (they share the same buffer).
        // The scope guard runs register_and_finish at iteration end for the claimer, no
        // matter which mapping branch does the read.
        const bool shared_master = mWeightManager && mWeightManager->is_shared_master(name);
        if (shared_master && !dsl::shared_master_store().try_claim(name)) {
            // Another rank is reading this shared master. Don't wait here — keep
            // sweeping so the rank threads read + page-lock DIFFERENT tensors in
            // parallel (pinning is the import bottleneck, ~1.4 GB/s per thread).
            // A completion pass after the loop waits for stragglers before the
            // tied-weight copies, which read these buffers.
            continue;
        }
        struct FinishGuard {
            const std::string& n;
            bool active;
            ~FinishGuard() {
                if (active) dsl::shared_master_store().register_and_finish(n);
            }
        } finish_guard{name, shared_master};

        const bool param_sharded = sharded_weights && mWeightManager->is_sharded(name);
        DslWeightLoader::ExpertLoadedCallback on_expert_loaded;
        if (adapter_merger) {
            on_expert_loaded = [&](int expert_idx, Tensor& expert) {
                adapter_merger->apply_expert(name, expert_idx, expert, adapter_stream);
            };
        }
        const Tensor& global = mParams->template_tensor(name);
        if (!loader.load_param(name, param, allow_cast, param_sharded, &global, adapter_stream,
                               on_expert_loaded)) {
            // Every allocated parameter must be populated, even when its mapping
            // is optional for model variants that do not allocate that parameter.
            throw std::runtime_error("DSL model: missing HF tensor for allocated param '" + name + "'");
        }
        if (adapter_merger) {
            // TiedTo is resolved from the merged source below; StackExperts has
            // already been merged by the per-expert callback. apply skips both.
            adapter_merger->apply(name, param, adapter_stream);
            CUDA_CHECK(cudaStreamSynchronize(adapter_stream));
        }
    }

    // Completion pass: shared masters claimed by other ranks may still be mid-read
    // (the main loop skips instead of waiting so ranks read in parallel). Every
    // name was claim-attempted above, so each has exactly one reader to wait for.
    if (mWeightManager) {
        for (const auto& name : mParams->param_names()) {
            if (!mParams->is_external(name) && mWeightManager->is_shared_master(name)) {
                dsl::shared_master_store().wait_populated(name);
            }
        }
    }

    if (report_progress) {
        const double gb = static_cast<double>(progress_bytes) / 1e9;
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - progress_start).count();
        fprintf(stderr,
                "[import] complete: %zu tensors | %.1f GB in %.0fs (%.2f GB/s)\n",
                progress_done,
                gb,
                secs,
                gb / std::max(secs, 1e-9));
        fflush(stderr);
    }

    loader.resolve_tied_params(
        [&](const std::string& name) -> Tensor& {
            return mWeightManager ? mWeightManager->get_master(name) : mParams->get(name);
        },
        [&](const std::string& name) { return mParams->is_external(name); });

    if (lora_enabled()) {
        mLoRAWeights->random_init(42, comm);
    }

    if (mWeightManager) {
        cudaStream_t stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
        mWeightManager->sync_work_from_master(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Quantize frozen matmul block masters to FP8 in place so per-stage streaming ships
        // FP8 bytes (no-op unless dispatch-PP + fp8_hybrid). Must run after the masters are
        // populated and before the first gather.
        if (mRunState) {
            mWeightManager->finalize_fp8_block_masters(mRunState->DeviceProp, stream);
            mWeightManager->finalize_fp4_block_masters(mRunState->DeviceProp, stream);
        }
    }

    comm.barrier();
}

void DslModel::on_restore_checkpoint(NCCLCommunicator& comm) {
    (void)comm;
    if (mOptimizer) {
        mOptimizer->on_restore_checkpoint(*this);
    }
}

void DslModel::prepare_optimizer_for_checkpoint_load() {
    if (lora_enabled()) {
        return;
    }
    if (!mOptimizer) {
        optimizers::OptimizerConfig cfg;
        cfg.type = optimizers::OptimizerType::ADAMW_8BIT;
        mOptimizer = optimizers::create_optimizer(cfg);
    }
    mOptimizer->prepare_for_checkpoint_load(*this);
}

void DslModel::export_weights(const std::string& file_name, NCCLCommunicator& comm) {
    if (!mParams) {
        throw std::logic_error("DslModel::export_weights called before parameters are initialized");
    }
    if (mWeightManager && mOptions.ShardWeights && mNumShards > 1) {
        throw std::runtime_error("DslModel::export_weights: export is not supported for sharded weights; use --gpus 1");
    }

    const auto& mapping = !mHfExport.empty() ? mHfExport : mHfMapping;
    SafeTensorWriter writer(file_name);

    struct ExportEntry {
        std::string name;
        Tensor tensor;
        bool needs_transpose = false;
        Tensor source;
    };

    std::vector<ExportEntry> exports;
    exports.reserve(mParams->param_names().size());

    for (const auto& name : mParams->param_names()) {
        Tensor& param = mParams->get(name);
        int layer_idx = -1;
        const MappingSpec* spec = internal::find_mapping_spec(mapping, name, layer_idx);
        if (!spec) {
            MappingSpec fallback;
            fallback.kind = MappingSpec::Kind::Direct;
            fallback.source = name;
            spec = &fallback;
        }

        if (spec->kind == MappingSpec::Kind::Direct) {
            const std::string hf_name = internal::format_hf_name(spec->source.empty() ? name : spec->source, layer_idx);
            exports.push_back({hf_name, param, false, {}});
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Fuse) {
            if (spec->dim != 0) {
                throw std::runtime_error("DSL model: fuse export only supports dim=0 for " + name);
            }
            std::vector<long> slice_sizes =
                internal::infer_fuse_slices(name, *mConfig, static_cast<int>(spec->sources.size()));
            if (slice_sizes.empty()) {
                if (param.Sizes[0] % static_cast<long>(spec->sources.size()) == 0) {
                    const long chunk = param.Sizes[0] / static_cast<long>(spec->sources.size());
                    slice_sizes.assign(spec->sources.size(), chunk);
                } else {
                    throw std::runtime_error("DSL model: cannot infer fuse slices for " + name);
                }
            } else if (slice_sizes.size() != spec->sources.size()) {
                throw std::runtime_error("DSL model: fuse slice count mismatch for " + name);
            }
            long offset = 0;
            for (std::size_t i = 0; i < spec->sources.size(); ++i) {
                const auto& src = spec->sources[i];
                const std::string hf_name = internal::format_hf_name(src, layer_idx);
                const long slice_len = slice_sizes.at(i);
                if (slice_len <= 0) {
                    throw std::runtime_error("DSL model: invalid fuse slice for " + name);
                }
                Tensor slice = internal::slice_dim0(param, offset, slice_len);
                exports.push_back({hf_name, slice, false, {}});
                offset += slice_len;
            }
            if (offset != param.Sizes[0]) {
                throw std::runtime_error("DSL model: fuse slices do not cover full tensor for " + name);
            }
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Transform) {
            if (spec->fn != "transpose") {
                throw std::runtime_error("DSL model: unsupported export transform '" + spec->fn + "' for " + name);
            }
            if (param.Rank != 2) {
                throw std::runtime_error("DSL model: transpose export expects 2D tensor for " + name);
            }
            const std::string hf_name = internal::format_hf_name(spec->source, layer_idx);
            Tensor tmp = mAllocator->allocate(param.DType,
                                              ("export_" + name).c_str(),
                                              EAllocationType::ON_DEVICE,
                                              {param.Sizes[1], param.Sizes[0]});
            exports.push_back({hf_name, tmp, true, param});
            continue;
        }

        if (spec->kind == MappingSpec::Kind::StackExperts) {
            // Export batched expert tensor [num_experts, ...] as individual HF expert tensors
            if (spec->source.empty()) {
                throw std::runtime_error("DSL model: stack_experts export missing pattern for " + name);
            }

            int num_experts = spec->num_experts;
            if (num_experts <= 0 && mModelConfig.moe_config.has_value()) {
                num_experts = mModelConfig.moe_config->num_experts;
            }
            if (num_experts <= 0) {
                num_experts = mModelConfig.NumExperts;
            }
            if (num_experts <= 0) {
                throw std::runtime_error("DSL model: stack_experts export cannot determine num_experts for " + name);
            }

            if (param.Rank < 1 || param.Sizes[0] != num_experts) {
                throw std::runtime_error("DSL model: stack_experts export param size mismatch for " + name);
            }

            const long expert_size = param.nelem() / param.Sizes[0];
            const std::size_t elem_size = get_dtype_size(param.DType);

            if (spec->fuse_gate_up) {
                // Export fused gate_up tensor as separate gate_proj and up_proj per expert
                // Layout: [E, 2*D, C] where first D rows are up, next D rows are gate
                const std::string gate_pattern = spec->source;
                const std::string up_pattern =
                    MappingSpec::derive_up_pattern(spec->source, spec->up_source);

                const long fused_rows = param.Rank >= 2 ? param.Sizes[1] : 1;
                const long D = fused_rows / 2;
                const long C = param.Rank >= 3 ? param.Sizes[2] : 1;
                const long sub_expert_elems = D * C;

                for (int e = 0; e < num_experts; ++e) {
                    const std::size_t base_offset = static_cast<std::size_t>(e) * fused_rows * C * elem_size;

                    // Export up_proj from first D rows
                    std::string up_hf = internal::format_hf_name(up_pattern, layer_idx, e);
                    Tensor up_slice = param;
                    up_slice.Rank = 2;
                    up_slice.Sizes[0] = D;
                    up_slice.Sizes[1] = C;
                    up_slice.Data = param.Data + base_offset;
                    exports.push_back({up_hf, up_slice, false, {}});

                    // Export gate_proj from second D rows
                    std::string gate_hf = internal::format_hf_name(gate_pattern, layer_idx, e);
                    Tensor gate_slice = param;
                    gate_slice.Rank = 2;
                    gate_slice.Sizes[0] = D;
                    gate_slice.Sizes[1] = C;
                    gate_slice.Data = param.Data + base_offset + sub_expert_elems * elem_size;
                    exports.push_back({gate_hf, gate_slice, false, {}});
                }
            } else {
                // Export single tensor (e.g., down_proj) for each expert
                for (int e = 0; e < num_experts; ++e) {
                    std::string hf_name = internal::format_hf_name(spec->source, layer_idx, e);

                    Tensor slice = param;
                    slice.Rank = param.Rank - 1;
                    for (int d = 0; d < slice.Rank; ++d) {
                        slice.Sizes[d] = param.Sizes[d + 1];
                    }
                    slice.Data = param.Data + static_cast<std::size_t>(e) * expert_size * elem_size;
                    exports.push_back({hf_name, slice, false, {}});
                }
            }
            continue;
        }

        throw std::runtime_error("DSL model: unsupported HF export mapping for " + name);
    }

    for (const auto& entry : exports) {
        writer.register_tensor(entry.name, TensorShard(entry.tensor));
    }
    writer.prepare_metadata(&comm);

    cudaStream_t stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
    for (auto& entry : exports) {
        if (entry.needs_transpose) {
            transpose(entry.tensor,
                      entry.source,
                      static_cast<int>(entry.source.Sizes[0]),
                      static_cast<int>(entry.source.Sizes[1]),
                      stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        writer.write_tensor(entry.name, TensorShard(entry.tensor), &comm);
    }

    writer.finalize(&comm);
}

float DslModel::get_loss() const {
    if (!mRunState) {
        return 0.0f;
    }
    float raw_loss = mRunState->get_loss();
    int valid_tokens = 0;
    CUDA_CHECK(cudaMemcpy(&valid_tokens, mRunState->ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost));
    if (valid_tokens > 0) {
        float avg_valid = static_cast<float>(valid_tokens) / static_cast<float>(std::max(1, mRunState->WorldSize));
        return raw_loss / avg_valid;
    }
    return 0.0f;
}

float DslModel::get_accuracy() const {
    return IModel::get_accuracy();
}

std::string_view DslModel::model_type() const {
    return mConfig ? mConfig->model_name() : "DSL";
}

IRunState& DslModel::get_run_state() const {
    if (!mRunState) {
        throw std::logic_error("DslModel::get_run_state() called before allocate_run_state()");
    }
    return *mRunState;
}

bool DslModel::is_weight_streaming_enabled() const {
    return mWeightManager && mWeightManager->is_streaming_enabled();
}

std::vector<std::pair<std::string, Tensor>> DslModel::shared_base_weights() {
    if (!lora_enabled() || qlora_enabled() || mNumShards != 1 || !mParams || mWeightManager) {
        throw std::runtime_error("shared base weights require single-GPU resident BF16 LoRA");
    }
    std::vector<std::pair<std::string, Tensor>> result;
    for (const auto& name : mParams->param_names()) {
        Tensor& tensor = mParams->get(name);
        if (!tensor.Data || tensor.Device < 0 ||
            (tensor.DType != ETensorDType::BF16 && tensor.DType != ETensorDType::FP32)) {
            throw std::runtime_error("shared base weight is not resident BF16 or FP32: " + name);
        }
        result.emplace_back(name, tensor);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}

void DslModel::import_weights_from_external(const std::string& safetensors_path,
                                            const std::vector<qlora::ExternalWeight>& external_weights,
                                            NCCLCommunicator& comm) {
    if (!mParams) {
        throw std::logic_error("DslModel::import_weights_from_external called before parameters are initialized");
    }
    if (!mLoRAConfig) {
        throw std::runtime_error("import_weights_from_external requires LoRA config (QLoRA mode)");
    }

    // Populate MoE config from model
    if (mModelConfig.moe_config.has_value()) {
        const auto& moe = mModelConfig.moe_config.value();
        if (mQLoRAConfig.num_experts == 0) mQLoRAConfig.num_experts = moe.num_experts;
        if (mQLoRAConfig.num_experts_per_tok == 0) mQLoRAConfig.num_experts_per_tok = moe.top_k;
        if (mQLoRAConfig.moe_intermediate_size == 0) mQLoRAConfig.moe_intermediate_size = moe.moe_intermediate_size;
        if (mQLoRAConfig.num_shared_experts == 0 && moe.use_shared_expert) mQLoRAConfig.num_shared_experts = 1;
        if (mQLoRAConfig.moe_shared_expert_intermediate_size == 0 && moe.use_shared_expert)
            mQLoRAConfig.moe_shared_expert_intermediate_size = moe.shared_expert_size;
    } else if (mModelConfig.NumExperts > 0) {
        if (mQLoRAConfig.num_experts == 0) mQLoRAConfig.num_experts = mModelConfig.NumExperts;
        if (mQLoRAConfig.num_experts_per_tok == 0) mQLoRAConfig.num_experts_per_tok = mModelConfig.NumExpertsPerTok;
        if (mQLoRAConfig.moe_intermediate_size == 0)
            mQLoRAConfig.moe_intermediate_size = mModelConfig.MoeIntermediateSize;
    }

    // Create the QLoRA provider (builds pipeline config from IR)
    mQLoRAProvider = internal::create_dsl_qlora_provider(*mModule,
                                                         mModelConfig,
                                                         *mConfig,
                                                         mOptions,
                                                         *mLoRAConfig,
                                                         mQLoRAConfig,
                                                         mAllocator,
                                                         mHfMapping,
                                                         mShardIdx,
                                                         mNumShards,
                                                         mAdapterPath);

    // Use the provider's import_from_external instead of import_and_quantize
    auto* generic_provider = dynamic_cast<qlora::GenericQLoRAProvider*>(mQLoRAProvider.get());
    if (!generic_provider) {
        throw std::runtime_error("import_weights_from_external: expected GenericQLoRAProvider");
    }

    cudaStream_t quant_stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&quant_stream));
    generic_provider->import_from_external(safetensors_path, external_weights, quant_stream);
    CUDA_CHECK(cudaStreamSynchronize(quant_stream));
    CUDA_CHECK(cudaStreamDestroy(quant_stream));

    mParams->set_qlora_provider(mQLoRAProvider.get());
    if (mRunState) {
        mParams->set_default_stream(mRunState->MainStream);
    }

    // Note: all params are marked external in QLoRA mode (is_qlora_param_name matches
    // embedding, lm_head, norms, and all block params). Non-quantized weights (norms,
    // embedding, lm_head) are loaded from disk by the QLoRA pipeline's import_from_external(),
    // stored in GenericWeightManager, and served via resolve_param().

    if (lora_enabled()) {
        mLoRAWeights->random_init(42, comm);
    }

    comm.barrier();
}

}  // namespace dsl
