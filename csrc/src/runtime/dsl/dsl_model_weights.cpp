// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model weight I/O operations (init, import, export, checkpoint).

#include "runtime/dsl/dsl_model.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "runtime/dsl/dsl_model_internal.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/dsl/graph_executor.h"
#include "runtime/lora/lora_model_utils.h"
#include "runtime/qlora/adapter_merger.h"
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
        if (mParams->is_external(name)) {
            continue;
        }
        Tensor& param = use_weight_manager ? mWeightManager->get_master(name) : mParams->get(name);
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
        const bool is_mlp_down = internal::contains_ci(name, "mlp_down_weight") || internal::contains_ci(name, "down_proj");
        if (mOptions.InitProjectionsToZero && (is_out_proj || is_mlp_down)) {
            fill_zero(param, nullptr);
            continue;
        }

        float stddev = scale;
        if (is_out_proj || is_mlp_down) {
            stddev *= residual_scale;
        }
        const bool param_sharded = use_weight_manager && mOptions.ShardWeights && (mNumShards > 1) &&
                                   mWeightManager->is_sharded(name);
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
    }

    comm.barrier();
}

void DslModel::import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) {
    if (!mParams) {
        throw std::logic_error("DslModel::import_weights called before parameters are initialized");
    }

    SafeTensorsReader reader(file_name);
    std::vector<std::pair<std::string, std::string>> tied_params;

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

        mQLoRAProvider = internal::create_dsl_qlora_provider(
            *mModule, mModelConfig, *mConfig, mOptions,
            *mLoRAConfig, mQLoRAConfig, mAllocator,
            mHfMapping, mShardIdx, mNumShards, mAdapterPath);
        cudaStream_t quant_stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&quant_stream));
        mQLoRAProvider->import_and_quantize(file_name, comm, quant_stream);
        CUDA_CHECK(cudaStreamSynchronize(quant_stream));
        CUDA_CHECK(cudaStreamDestroy(quant_stream));

        mParams->set_qlora_provider(mQLoRAProvider.get());
        if (mRunState) {
            mParams->set_default_stream(mRunState->MainStream);
        }
    }

    const bool sharded_weights = mWeightManager && mOptions.ShardWeights && (mNumShards > 1);
    auto shard_range = [&](long global_rows, bool param_sharded) -> std::pair<long, long> {
        if (!param_sharded) {
            return {0, global_rows};
        }
        if (global_rows % mNumShards != 0) {
            throw std::runtime_error("DSL model: sharded load requires dim0 divisible by num_shards");
        }
        const long shard_rows = global_rows / mNumShards;
        const long start = shard_rows * mShardIdx;
        return {start, start + shard_rows};
    };
    auto row_stride = [](const std::vector<long>& shape) -> long {
        long stride = 1;
        for (std::size_t i = 1; i < shape.size(); ++i) {
            stride *= shape[i];
        }
        return stride;
    };

    // Adapter merge (stacked LoRA): merge previously-trained adapter into base weights.
    // This runs ONLY on the BF16 path; the QLoRA path handles merging in the pipeline.
    std::unique_ptr<qlora::AdapterMerger> adapter_merger;
    cudaStream_t adapter_stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
    if (!qlora_enabled() && !mAdapterPath.empty()) {
        ShardConfig shard_config;
        shard_config.shard_idx = mShardIdx;
        shard_config.num_shards = mNumShards;
        adapter_merger = std::make_unique<qlora::AdapterMerger>(
            mAdapterPath, mHfMapping, reader, shard_config);
    }

    for (const auto& name : mParams->param_names()) {
        if (mParams->is_external(name)) {
            continue;
        }
        Tensor& param = mWeightManager ? mWeightManager->get_master(name) : mParams->get(name);
        const bool param_sharded = sharded_weights && mWeightManager->is_sharded(name);
        int layer_idx = -1;
        const MappingSpec* spec = internal::find_mapping_spec(mHfMapping, name, layer_idx);
        MappingSpec direct_fallback;
        if (!spec) {
            direct_fallback.kind = MappingSpec::Kind::Direct;
            direct_fallback.source = name;
            spec = &direct_fallback;
        }

        if (spec->kind == MappingSpec::Kind::TiedTo) {
            tied_params.emplace_back(name, spec->target);
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Direct) {
            const std::string hf_name = internal::format_hf_name(
                spec->source.empty() ? name : spec->source, layer_idx);

            // Handle tied embeddings: if lm_head.weight is not found and embeddings are tied,
            // fall back to model.embed_tokens.weight
            const SafeTensorEntry* entry_ptr = nullptr;
            try {
                entry_ptr = &reader.find_entry(hf_name);
            } catch (const std::out_of_range&) {
                if (mConfig->TiedWordEmbeddings && hf_name == "lm_head.weight") {
                    std::vector<std::string> candidates;
                    int emb_layer_idx = -1;
                    if (const auto* emb_spec = internal::find_mapping_spec(mHfMapping, "embedding", emb_layer_idx)) {
                        if (emb_spec->kind == MappingSpec::Kind::Direct && !emb_spec->source.empty()) {
                            candidates.push_back(internal::format_hf_name(emb_spec->source, emb_layer_idx));
                        }
                    }
                    candidates.push_back("model.embed_tokens.weight");
                    candidates.push_back("model.language_model.embed_tokens.weight");

                    for (const auto& candidate : candidates) {
                        if (candidate.empty()) {
                            continue;
                        }
                        try {
                            entry_ptr = &reader.find_entry(candidate);
                            break;
                        } catch (const std::out_of_range&) {
                            continue;
                        }
                    }

                    if (!entry_ptr) {
                        throw std::runtime_error("DSL model: missing HF tensor '" + hf_name
                                                 + "' (and tied fallback) for param '" + name + "'");
                    }
                } else {
                    throw std::runtime_error("DSL model: missing HF tensor '" + hf_name + "' for param '" + name + "'");
                }
            }
            const auto& entry = *entry_ptr;
            const std::string entry_name = entry.name();
            if (!param_sharded) {
                const auto& file_shape = entry.shape();
                if (static_cast<int>(file_shape.size()) != param.Rank) {
                    // Allow squeezing size-1 dims (e.g., conv1d weight [D,1,K] -> [D,K]).
                    std::vector<long> squeezed;
                    squeezed.reserve(file_shape.size());
                    for (long d : file_shape) {
                        if (d != 1) squeezed.push_back(d);
                    }
                    bool match = (static_cast<int>(squeezed.size()) == param.Rank);
                    for (int i = 0; match && i < param.Rank; ++i) {
                        if (squeezed[i] != param.Sizes[i]) match = false;
                    }
                    if (!match) {
                        throw std::runtime_error("DSL model: shape mismatch for '" + hf_name
                                                 + "': file shape cannot be squeezed to target shape for param '"
                                                 + name + "'");
                    }
                    entry.read_raw(param, 0, param.nelem(), allow_cast);
                } else {
                    entry.read_tensor(param, allow_cast);
                }
            } else {
                const Tensor& global = mParams->template_tensor(name);
                const long global_rows = global.Sizes[0];
                auto [start, end] = shard_range(global_rows, param_sharded);
                const long stride = row_stride(entry.shape());
                (void)end;
                entry.read_raw(param, static_cast<std::ptrdiff_t>(start) * stride, param.nelem(), allow_cast);
            }
            if (adapter_merger) {
                adapter_merger->apply(name, param, adapter_stream);
                CUDA_CHECK(cudaStreamSynchronize(adapter_stream));
            }
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Fuse) {
            if (spec->dim != 0) {
                throw std::runtime_error("DSL model: fuse mapping only supports dim=0 for " + name);
            }
            std::vector<long> slice_sizes = internal::infer_fuse_slices(name, *mConfig,
                                                                        static_cast<int>(spec->sources.size()));
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

            const Tensor& global = mParams->template_tensor(name);
            const long global_rows = global.Sizes[0];
            auto [shard_start, shard_end] = shard_range(global_rows, param_sharded);

            long offset = 0;
            for (std::size_t i = 0; i < spec->sources.size(); ++i) {
                const auto& src = spec->sources[i];
                const std::string hf_name = internal::format_hf_name(src, layer_idx);
                const auto& entry = reader.find_entry(hf_name);
                if (entry.shape().empty()) {
                    throw std::runtime_error("DSL model: empty shape for " + hf_name);
                }
                if (static_cast<int>(entry.shape().size()) != param.Rank) {
                    throw std::runtime_error("DSL model: rank mismatch for " + hf_name);
                }
                for (int j = 1; j < param.Rank; ++j) {
                    if (entry.shape().at(j) != global.Sizes[j]) {
                        throw std::runtime_error("DSL model: shape mismatch for " + hf_name);
                    }
                }

                const long slice_len = slice_sizes.at(i);
                if (!param_sharded) {
                    Tensor slice = internal::slice_dim0(param, offset, slice_len);
                    entry.read_raw(slice, 0, slice.nelem(), allow_cast);
                    offset += slice_len;
                    continue;
                }

                const long src_begin = offset;
                const long src_end = offset + slice_len;
                const long overlap_begin = std::max(src_begin, shard_start);
                const long overlap_end = std::min(src_end, shard_end);
                if (overlap_begin < overlap_end) {
                    const long rows = overlap_end - overlap_begin;
                    const long dst_row_offset = overlap_begin - shard_start;
                    const long src_row_offset = overlap_begin - src_begin;
                    Tensor slice = internal::slice_dim0(param, dst_row_offset, rows);
                    const long stride = row_stride(entry.shape());
                    entry.read_raw(slice, static_cast<std::ptrdiff_t>(src_row_offset) * stride,
                                   slice.nelem(), allow_cast);
                }
                offset += slice_len;
            }
            if (offset != global_rows) {
                throw std::runtime_error("DSL model: fuse mapping size mismatch for " + name);
            }
            if (adapter_merger) {
                adapter_merger->apply(name, param, adapter_stream);
                CUDA_CHECK(cudaStreamSynchronize(adapter_stream));
            }
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Split) {
            if (spec->dim != 0) {
                throw std::runtime_error("DSL model: split mapping only supports dim=0 for " + name);
            }
            if (spec->ranges.empty()) {
                throw std::runtime_error("DSL model: split mapping missing ranges for " + name);
            }
            auto [start, end] = spec->ranges.front();
            if (start < 0 || end <= start) {
                throw std::runtime_error("DSL model: unsupported split range for " + name);
            }
            const long expected = end - start;
            const std::string hf_name = internal::format_hf_name(spec->source, layer_idx);
            const auto& entry = reader.find_entry(hf_name);
            if (!param_sharded) {
                if (param.Sizes[0] != expected) {
                    throw std::runtime_error("DSL model: split range size mismatch for " + name);
                }
                long stride = 1;
                for (int i = 1; i < param.Rank; ++i) {
                    stride *= param.Sizes[i];
                }
                const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(start) * stride;
                entry.read_raw(param, offset, param.nelem(), allow_cast);
            } else {
                const long shard_rows = expected / mNumShards;
                if (expected % mNumShards != 0 || param.Sizes[0] != shard_rows) {
                    throw std::runtime_error("DSL model: split shard size mismatch for " + name);
                }
                const long local_start = start + shard_rows * mShardIdx;
                const long stride = row_stride(entry.shape());
                entry.read_raw(param, static_cast<std::ptrdiff_t>(local_start) * stride, param.nelem(), allow_cast);
            }
            if (adapter_merger) {
                adapter_merger->apply(name, param, adapter_stream);
                CUDA_CHECK(cudaStreamSynchronize(adapter_stream));
            }
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Transform) {
            if (spec->fn != "transpose") {
                throw std::runtime_error("DSL model: unsupported transform '" + spec->fn + "' for " + name);
            }
            const std::string hf_name = internal::format_hf_name(spec->source, layer_idx);
            const auto& entry = reader.find_entry(hf_name);
            cudaStream_t stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
            if (entry.shape().size() == 2 && param.Rank == 2) {
                if (!param_sharded) {
                    Tensor tmp = mAllocator->allocate(param.DType, ("hf_tmp_" + name).c_str(),
                                                      EAllocationType::ON_DEVICE,
                                                      {entry.shape().at(0), entry.shape().at(1)});
                    entry.read_tensor(tmp, allow_cast);
                    transpose(param, tmp, static_cast<int>(entry.shape().at(0)),
                              static_cast<int>(entry.shape().at(1)), stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                } else {
                    const Tensor& global = mParams->template_tensor(name);
                    const long global_rows = global.Sizes[0];
                    auto [start, end] = shard_range(global_rows, param_sharded);
                    if (param.Sizes[0] != (end - start)) {
                        throw std::runtime_error("DSL model: transpose shard size mismatch for " + name);
                    }
                    Tensor tmp_src = mAllocator->allocate(param.DType, ("hf_tmp_src_" + name).c_str(),
                                                          EAllocationType::ON_DEVICE,
                                                          {entry.shape().at(0), entry.shape().at(1)});
                    entry.read_tensor(tmp_src, allow_cast);
                    Tensor tmp_full = mAllocator->allocate(param.DType, ("hf_tmp_full_" + name).c_str(),
                                                           EAllocationType::ON_DEVICE,
                                                           {global.Sizes[0], global.Sizes[1]});
                    transpose(tmp_full, tmp_src, static_cast<int>(entry.shape().at(0)),
                              static_cast<int>(entry.shape().at(1)), stream);
                    Tensor slice = internal::slice_dim0(tmp_full, start, end - start);
                    CUDA_CHECK(cudaMemcpyAsync(param.Data, slice.Data, param.bytes(),
                                               cudaMemcpyDeviceToDevice, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            } else if (entry.shape().size() == 3 && param.Rank == 3) {
                const long E = static_cast<long>(entry.shape().at(0));
                const long A = static_cast<long>(entry.shape().at(1));
                const long B = static_cast<long>(entry.shape().at(2));
                if (!param_sharded) {
                    if (param.Sizes[0] != E || param.Sizes[1] != B || param.Sizes[2] != A) {
                        throw std::runtime_error("DSL model: transpose (3D) shape mismatch for " + name);
                    }
                    Tensor tmp_src = mAllocator->allocate(param.DType, ("hf_tmp_" + name).c_str(),
                                                          EAllocationType::ON_DEVICE,
                                                          {E, A, B});
                    entry.read_tensor(tmp_src, allow_cast);
                    const std::size_t elem_size = get_dtype_size(param.DType);
                    for (long e = 0; e < E; ++e) {
                        Tensor src_view = tmp_src;
                        src_view.Rank = 2;
                        src_view.Sizes[0] = A;
                        src_view.Sizes[1] = B;
                        src_view.Data = static_cast<std::byte*>(tmp_src.Data)
                                        + static_cast<std::size_t>(e) * A * B * elem_size;

                        Tensor dst_view = param;
                        dst_view.Rank = 2;
                        dst_view.Sizes[0] = B;
                        dst_view.Sizes[1] = A;
                        dst_view.Data = static_cast<std::byte*>(param.Data)
                                        + static_cast<std::size_t>(e) * B * A * elem_size;

                        transpose(dst_view, src_view, static_cast<int>(A), static_cast<int>(B), stream);
                    }
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                } else {
                    const Tensor& global = mParams->template_tensor(name);
                    const long global_rows = global.Sizes[0];
                    auto [start, end] = shard_range(global_rows, param_sharded);
                    if (param.Sizes[0] != (end - start) ||
                        global.Sizes[1] != B || global.Sizes[2] != A) {
                        throw std::runtime_error("DSL model: transpose (3D) shard size mismatch for " + name);
                    }
                    Tensor tmp_src = mAllocator->allocate(param.DType, ("hf_tmp_src_" + name).c_str(),
                                                          EAllocationType::ON_DEVICE,
                                                          {E, A, B});
                    entry.read_tensor(tmp_src, allow_cast);
                    Tensor tmp_full = mAllocator->allocate(param.DType, ("hf_tmp_full_" + name).c_str(),
                                                           EAllocationType::ON_DEVICE,
                                                           {global.Sizes[0], global.Sizes[1], global.Sizes[2]});
                    const std::size_t elem_size = get_dtype_size(param.DType);
                    for (long e = 0; e < E; ++e) {
                        Tensor src_view = tmp_src;
                        src_view.Rank = 2;
                        src_view.Sizes[0] = A;
                        src_view.Sizes[1] = B;
                        src_view.Data = static_cast<std::byte*>(tmp_src.Data)
                                        + static_cast<std::size_t>(e) * A * B * elem_size;

                        Tensor dst_view = tmp_full;
                        dst_view.Rank = 2;
                        dst_view.Sizes[0] = B;
                        dst_view.Sizes[1] = A;
                        dst_view.Data = static_cast<std::byte*>(tmp_full.Data)
                                        + static_cast<std::size_t>(e) * B * A * elem_size;

                        transpose(dst_view, src_view, static_cast<int>(A), static_cast<int>(B), stream);
                    }
                    Tensor slice = internal::slice_dim0(tmp_full, start, end - start);
                    CUDA_CHECK(cudaMemcpyAsync(param.Data, slice.Data, param.bytes(),
                                               cudaMemcpyDeviceToDevice, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            } else {
                throw std::runtime_error("DSL model: transpose expects 2D or 3D tensors for " + name);
            }
            if (adapter_merger) {
                adapter_merger->apply(name, param, adapter_stream);
                CUDA_CHECK(cudaStreamSynchronize(adapter_stream));
            }
            continue;
        }

        if (spec->kind == MappingSpec::Kind::StackExperts) {
            // Stack per-expert HF tensors into batched format [num_experts, ...]
            // The pattern has {expert} placeholder that we replace for each expert.
            if (spec->source.empty()) {
                throw std::runtime_error("DSL model: stack_experts missing pattern for " + name);
            }

            // Get number of experts from spec or config
            int num_experts = spec->num_experts;
            if (num_experts <= 0 && mModelConfig.moe_config.has_value()) {
                num_experts = mModelConfig.moe_config->num_experts;
            }
            if (num_experts <= 0) {
                num_experts = mModelConfig.NumExperts;
            }
            if (num_experts <= 0) {
                throw std::runtime_error("DSL model: stack_experts cannot determine num_experts for " + name);
            }

            // param shape is [num_experts, ...] for batched format
            // Each expert tensor has shape [...]
            if (param.Rank < 1 || param.Sizes[0] < num_experts) {
                throw std::runtime_error("DSL model: stack_experts param rank/size mismatch for " + name);
            }

            const long expert_size = param.nelem() / param.Sizes[0];  // Elements per expert
            const std::size_t elem_size = get_dtype_size(param.DType);
            cudaStream_t stream = mRunState ? mRunState->MainStream : cudaStreamDefault;

            // Sharding: determine which experts this shard owns
            const int experts_per_shard = param_sharded ? (num_experts / mNumShards) : num_experts;
            const int expert_start = param_sharded ? (mShardIdx * experts_per_shard) : 0;
            const int expert_end = param_sharded ? (expert_start + experts_per_shard) : num_experts;

            if (spec->fuse_gate_up) {
                // Load gate_proj and up_proj for each expert and fuse into gate_up format
                // gate_up layout: [up; gate] per expert, so first D rows are up, next D rows are gate
                std::string gate_pattern = spec->source;
                std::string up_pattern = spec->source;
                std::size_t pos = up_pattern.find("gate_proj");
                if (pos != std::string::npos) {
                    up_pattern.replace(pos, 9, "up_proj");
                }

                // Expert tensor shape: [2*D, C] where D = intermediate_size
                // Each sub-tensor (gate/up) has shape [D, C]
                const long fused_rows = param.Rank >= 2 ? param.Sizes[1] : 1;  // Assuming [E, 2*D, C]
                const long D = fused_rows / 2;
                const long C = param.Rank >= 3 ? param.Sizes[2] : 1;
                const long sub_expert_elems = D * C;

                for (int e = expert_start; e < expert_end; ++e) {
                    const int local_e = e - expert_start;
                    const std::size_t base_offset = static_cast<std::size_t>(local_e) * fused_rows * C * elem_size;

                    // Load up_proj into first D rows
                    std::string up_hf = internal::format_hf_name(up_pattern, layer_idx, e);
                    const auto& up_entry = reader.find_entry(up_hf);
                    if (up_entry.shape().empty()) {
                        throw std::runtime_error("DSL model: stack_experts missing up_proj " + up_hf);
                    }
                    Tensor up_slice = param;
                    up_slice.Rank = 2;
                    up_slice.Sizes[0] = D;
                    up_slice.Sizes[1] = C;
                    up_slice.Data = param.Data + base_offset;
                    up_entry.read_tensor(up_slice, allow_cast);

                    // Load gate_proj into second D rows
                    std::string gate_hf = internal::format_hf_name(gate_pattern, layer_idx, e);
                    const auto& gate_entry = reader.find_entry(gate_hf);
                    if (gate_entry.shape().empty()) {
                        throw std::runtime_error("DSL model: stack_experts missing gate_proj " + gate_hf);
                    }
                    Tensor gate_slice = param;
                    gate_slice.Rank = 2;
                    gate_slice.Sizes[0] = D;
                    gate_slice.Sizes[1] = C;
                    gate_slice.Data = param.Data + base_offset + sub_expert_elems * elem_size;
                    gate_entry.read_tensor(gate_slice, allow_cast);

                    // Merge adapter delta for this expert (stacked LoRA)
                    if (adapter_merger) {
                        Tensor expert_view = param;
                        expert_view.Rank = 2;
                        expert_view.Sizes[0] = fused_rows;
                        expert_view.Sizes[1] = C;
                        expert_view.Data = param.Data + base_offset;
                        adapter_merger->apply_expert(name, e, expert_view, adapter_stream);
                    }
                }
            } else {
                // Load single tensor (e.g., down_proj) for each expert
                for (int e = expert_start; e < expert_end; ++e) {
                    const int local_e = e - expert_start;
                    std::string hf_name = internal::format_hf_name(spec->source, layer_idx, e);
                    const auto& entry = reader.find_entry(hf_name);
                    if (entry.shape().empty()) {
                        throw std::runtime_error("DSL model: stack_experts missing " + hf_name);
                    }

                    // Create slice for this expert
                    Tensor slice = param;
                    slice.Rank = param.Rank - 1;
                    for (int d = 0; d < slice.Rank; ++d) {
                        slice.Sizes[d] = param.Sizes[d + 1];
                    }
                    slice.Data = param.Data + static_cast<std::size_t>(local_e) * expert_size * elem_size;
                    entry.read_tensor(slice, allow_cast);

                    // Merge adapter delta for this expert (stacked LoRA)
                    if (adapter_merger) {
                        adapter_merger->apply_expert(name, e, slice, adapter_stream);
                    }
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            continue;
        }

        throw std::runtime_error("DSL model: unsupported HF mapping for " + name);
    }

    for (const auto& tie : tied_params) {
        if (mParams->is_external(tie.first) || mParams->is_external(tie.second)) {
            continue;
        }
        Tensor& dst = mWeightManager ? mWeightManager->get_master(tie.first) : mParams->get(tie.first);
        Tensor& src = mWeightManager ? mWeightManager->get_master(tie.second) : mParams->get(tie.second);
        CUDA_CHECK(cudaMemcpy(dst.Data, src.Data, src.bytes(), cudaMemcpyDeviceToDevice));
    }

    if (lora_enabled()) {
        mLoRAWeights->random_init(42, comm);
    }

    if (mWeightManager) {
        cudaStream_t stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
        mWeightManager->sync_work_from_master(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    comm.barrier();
}

void DslModel::on_restore_checkpoint(NCCLCommunicator& comm) {
    (void)comm;
    if (mAdamW8BitState && mAdamW8BitState->state1.Data) {
        mAdamW8BitState->initialized = true;
        mAdamWMomentumContainer.update_pointers(&mAdamW8BitState->state1, &mAdamW8BitState->absmax1);
        mAdamWVarianceContainer.update_pointers(&mAdamW8BitState->state2, &mAdamW8BitState->absmax2);
    }
}

void DslModel::prepare_optimizer_for_checkpoint_load() {
    if (lora_enabled()) {
        return;
    }
    if (!mAdamW8BitState) {
        mAdamW8BitState = std::make_unique<AdamW8BitState>();
    }
    cudaStream_t stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
    if (!mAdamW8BitState->initialized) {
        init_optimizer_state(stream);
    }
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
            const std::string hf_name = internal::format_hf_name(
                spec->source.empty() ? name : spec->source, layer_idx);
            exports.push_back({hf_name, param, false, {}});
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Fuse) {
            if (spec->dim != 0) {
                throw std::runtime_error("DSL model: fuse export only supports dim=0 for " + name);
            }
            std::vector<long> slice_sizes = internal::infer_fuse_slices(name, *mConfig, static_cast<int>(spec->sources.size()));
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
            Tensor tmp = mAllocator->allocate(param.DType, ("export_" + name).c_str(),
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
                std::string gate_pattern = spec->source;
                std::string up_pattern = spec->source;
                std::size_t pos = up_pattern.find("gate_proj");
                if (pos != std::string::npos) {
                    up_pattern.replace(pos, 9, "up_proj");
                }

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
            transpose(entry.tensor, entry.source,
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

}  // namespace dsl
