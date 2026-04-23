// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL Graph executor (DSL-driven).

#include "runtime/executor/graph_executor.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/buffer_plan.h"
#include "runtime/executor/compiled_ops.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/tensor_slot_dispatch.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/executor/graph_executor_internal.h"
#include "runtime/executor/graph_executor_tensors.h"
#include "runtime/executor/graph_executor_utils.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <sstream>

#include <cuda_fp16.h>
#include "runtime/core/fp8_scaling_state.h"
#include "runtime/core/fp8_scaling_config.h"
#include "runtime/lora/lora_model_utils.h"
#include "runtime/lora/lora_run_state.h"
#include "runtime/core/matmul_context.h"
#include "runtime/core/model_config.h"
#include "runtime/training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/allocator.h"
#include "utilities/dtype.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"
#include "kernels/kernels.h"
#include "recipes/nvfp4/nvfp4_recipe.h"

namespace dsl {
namespace {

/// Copy position IDs to the device-side PositionIDs buffer, replicating a single
/// plane across all 3 mRoPE planes when the model uses multimodal RoPE but the
/// caller provides only one plane (e.g. text-only GRPO training).
inline void copy_position_ids_to_device(const Tensor& src_pos, Tensor& dst_pos, long B, long T, cudaStream_t stream) {
    const std::size_t plane_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * sizeof(std::int32_t);
    const std::size_t src_bytes = src_pos.bytes();
    const auto kind = (src_pos.Device == -1) ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;

    // mRoPE: dst is [3, B, T] but src is single-plane [B, T] — replicate.
    if (dst_pos.Rank == 3 && dst_pos.Sizes[0] == 3 && src_bytes <= plane_bytes) {
        for (int p = 0; p < 3; ++p) {
            auto* dst = static_cast<std::byte*>(dst_pos.Data) + p * plane_bytes;
            CUDA_CHECK(cudaMemcpyAsync(dst, src_pos.Data, src_bytes, kind, stream));
        }
    } else {
        CUDA_CHECK(cudaMemcpyAsync(dst_pos.Data, src_pos.Data, src_bytes, kind, stream));
    }
}

/// Overload for raw CPU pointer + known element count (used by execute_logprobs_forward).
inline void copy_position_ids_to_device(const std::int32_t* src_cpu,
                                        std::size_t src_bytes,
                                        Tensor& dst_pos,
                                        long B,
                                        long T,
                                        cudaStream_t stream) {
    const std::size_t plane_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * sizeof(std::int32_t);

    if (dst_pos.Rank == 3 && dst_pos.Sizes[0] == 3 && src_bytes <= plane_bytes) {
        for (int p = 0; p < 3; ++p) {
            auto* dst = static_cast<std::byte*>(dst_pos.Data) + p * plane_bytes;
            CUDA_CHECK(cudaMemcpyAsync(dst, src_cpu, src_bytes, cudaMemcpyHostToDevice, stream));
        }
    } else {
        CUDA_CHECK(cudaMemcpyAsync(dst_pos.Data, src_cpu, src_bytes, cudaMemcpyHostToDevice, stream));
    }
}

// trace_or_execute_cuda_graph_with_stack is now in graph_executor_utils.h

inline bool stream_is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

inline bool is_capture_unsafe_op_type(CompiledOpType type) {
    switch (type) {
        // Qwen3.5 / Triton JIT kernels
        case CompiledOpType::ChunkGatedDeltaRule:
        case CompiledOpType::ChunkGatedDeltaRuleBackward:
        case CompiledOpType::Qwen3_5Decay:
        case CompiledOpType::Qwen3_5DecayBackward:
        // MoE routing / grouped GEMM rely on per-step host metadata and dynamic routing.
        case CompiledOpType::MoESoftmax:
        case CompiledOpType::MoESigmoid:
        case CompiledOpType::MoETopK:
        case CompiledOpType::MoEPermute:
        case CompiledOpType::MoEGroupedGemm:
        case CompiledOpType::MoEGroupedGemmGateUp:
        case CompiledOpType::MoEGroupedGemmDown:
        case CompiledOpType::MoEUnpermute:
        case CompiledOpType::MoEExpertBiasAdd:
        case CompiledOpType::MoESoftmaxBackward:
        case CompiledOpType::MoESigmoidBackward:
        case CompiledOpType::MoETopKBackward:
        case CompiledOpType::MoEPermuteBackward:
        case CompiledOpType::MoEGroupedGemmBackward:
        case CompiledOpType::MoEGroupedGemmGateUpBackward:
        case CompiledOpType::MoEGroupedGemmDownBackward:
        case CompiledOpType::MoEUnpermuteBackward:
        case CompiledOpType::MoEExpertBiasAddBackward:
        // EP ops perform per-step host-side split/reorder bookkeeping.
        case CompiledOpType::EpDispatch:
        case CompiledOpType::EpCombine:
        case CompiledOpType::EpDispatchBackward:
        case CompiledOpType::EpCombineBackward: return true;
        default: return false;
    }
}

inline bool graph_has_capture_unsafe_ops(const CompiledGraph* g) {
    if (!g) {
        return false;
    }
    for (const auto& op : g->ops) {
        if (is_capture_unsafe_op_type(op.type)) {
            return true;
        }
    }
    return false;
}

inline void sync_event_if_not_capturing(cudaEvent_t event, cudaStream_t stream) {
    if (!stream_is_capturing(stream)) {
        CUDA_CHECK(cudaEventSynchronize(event));
    }
}

inline void record_event_if_not_capturing(cudaEvent_t event, cudaStream_t stream) {
    if (!stream_is_capturing(stream)) {
        CUDA_CHECK(cudaEventRecord(event, stream));
    }
}

void reduce_loss(DslRunState& rs, long B, long T, NCCLCommunicator& comm) {
    deterministic_sum(rs.Losses.template get<float>(), rs.Losses.template get<float>(), B * T, rs.MainStream);
    comm.reduce_loss(rs.Losses.template get<float>(), rs.MainStream);
    CUDA_CHECK(cudaMemcpyAsync(rs.LossHost,
                               rs.Losses.template get<float>(),
                               sizeof(float),
                               cudaMemcpyDeviceToHost,
                               rs.MainStream));
}

void add_bias_tensor(Tensor& out, const Tensor& bias, int B, int T, int OC, cudaStream_t stream) {
    if (out.DType != bias.DType) {
        throw std::runtime_error("DSL graph executor: bias_add dtype mismatch");
    }
    if (out.DType == ETensorDType::BF16) {
        add_bias(out.get<nv_bfloat16>(), bias.get<nv_bfloat16>(), B, T, OC, stream);
        return;
    }
    if (out.DType == ETensorDType::FP32) {
        add_bias(out.get<float>(), bias.get<float>(), B, T, OC, stream);
        return;
    }
    throw std::runtime_error("DSL graph executor: bias_add unsupported dtype");
}

Tensor recompute_lora_rmsnorm(modules::LoRARunState& lora_rs,
                              const Tensor& residual,
                              const Tensor& weight,
                              float eps,
                              int B,
                              int T,
                              int C,
                              cudaStream_t stream) {
    if (!lora_rs.recompute_ln.Data || !lora_rs.recompute_rstd.Data) {
        throw std::runtime_error("DSL graph executor: LoRA recompute buffers not allocated");
    }
    rmsnorm_forward(lora_rs.recompute_ln, lora_rs.recompute_rstd, residual, weight, nullptr, eps, B, T, C, stream);
    return lora_rs.recompute_ln;
}

// ---------------------------------------------------------------------------
// Debug tensor dump helpers (env-driven, for onboarding / forward-match tests)
// ---------------------------------------------------------------------------

std::string debug_dump_sanitize(const std::string& name) {
    std::string result;
    result.reserve(name.size());
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') {
            result += c;
        } else {
            result += '_';
        }
    }
    return result;
}

void debug_dump_tensor(const std::string& name, const Tensor& t, const std::string& dump_dir, cudaStream_t stream) {
    if (!t.Data || t.nelem() <= 0) {
        return;
    }
    const std::string safe = debug_dump_sanitize(name);
    const std::size_t nelem = static_cast<std::size_t>(t.nelem());
    std::vector<float> host_data(nelem);

    if (t.DType == ETensorDType::FP32) {
        CUDA_CHECK(cudaMemcpy(host_data.data(), t.Data, nelem * sizeof(float), cudaMemcpyDeviceToHost));
    } else if (t.DType == ETensorDType::BF16) {
        std::vector<uint16_t> bf16_data(nelem);
        CUDA_CHECK(cudaMemcpy(bf16_data.data(), t.Data, nelem * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < nelem; ++i) {
            uint32_t bits = static_cast<uint32_t>(bf16_data[i]) << 16;
            float val;
            std::memcpy(&val, &bits, sizeof(float));
            host_data[i] = val;
        }
    } else if (t.DType == ETensorDType::FP16) {
        std::vector<uint16_t> fp16_data(nelem);
        CUDA_CHECK(cudaMemcpy(fp16_data.data(), t.Data, nelem * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < nelem; ++i) {
            __half h;
            std::memcpy(&h, &fp16_data[i], sizeof(__half));
            host_data[i] = __half2float(h);
        }
    } else {
        // Unsupported dtype for dump — skip silently
        return;
    }

    // Write binary data
    const std::string bin_path = dump_dir + "/" + safe + ".bin";
    FILE* bin_f = std::fopen(bin_path.c_str(), "wb");
    if (bin_f) {
        std::fwrite(host_data.data(), sizeof(float), nelem, bin_f);
        std::fclose(bin_f);
    }

    // Write JSON metadata
    const std::string json_path = dump_dir + "/" + safe + ".json";
    FILE* json_f = std::fopen(json_path.c_str(), "w");
    if (json_f) {
        std::fprintf(json_f, "{\"name\": \"%s\", \"dtype\": \"float32\", \"shape\": [", name.c_str());
        for (int i = 0; i < t.Rank; ++i) {
            std::fprintf(json_f, "%ld%s", t.Sizes[i], (i + 1 < t.Rank) ? ", " : "");
        }
        std::fprintf(json_f, "]}\n");
        std::fclose(json_f);
    }
}

std::vector<std::string> debug_dump_parse_tensor_list(const char* env_val) {
    std::vector<std::string> names;
    if (!env_val || !*env_val) {
        return names;
    }
    std::string s(env_val);
    std::size_t pos = 0;
    while (pos < s.size()) {
        auto next = s.find(',', pos);
        if (next == std::string::npos) {
            next = s.size();
        }
        std::string tok = s.substr(pos, next - pos);
        // Trim whitespace
        while (!tok.empty() && tok.front() == ' ')
            tok.erase(tok.begin());
        while (!tok.empty() && tok.back() == ' ')
            tok.pop_back();
        if (!tok.empty()) {
            names.push_back(std::move(tok));
        }
        pos = next + 1;
    }
    return names;
}

}  // namespace

GraphExecutor::GraphExecutor(const Module& module,
                             DslRunState& run_state,
                             DslParamStore& weights,
                             DslGradStore& grads,
                             const modules::ModelConfig& config,
                             const RuntimeOptions& options,
                             const GraphExecutorOptions& exec_options)
    : mModule(module),
      mRunState(run_state),
      mWeights(weights),
      mGrads(grads),
      mConfig(config),
      mOptions(options),
      mForward(module.forward ? &module.forward.value() : nullptr),
      mBackward(nullptr) {
    mGraphsEnabled = options.UseCudaGraphs;
    mBackwardGraphsEnabled = mGraphsEnabled;
    // Enable per-layer CUDA graphs (more fine-grained than whole-graph capture)
    mPerLayerGraphsEnabled = options.UseCudaGraphs && run_state.per_layer_graphs_enabled();
    init(exec_options);
}

GraphExecutor::~GraphExecutor() {
    // Stack arena (if rebased) is owned by DslRunState — adopted buffer
    // persists past this destructor and is cudaFree'd in ~DslRunState.
    // release_phase_arenas below skips unified_stack_ptr because ownership
    // was transferred (the pointer is null in mPhaseArenas after adopt).
    dsl::release_phase_arenas(mPhaseArenas);
    // Clean up CUDA graphs
    if (mForwardGraph) {
        cudaGraphExecDestroy(mForwardGraph);
        mForwardGraph = nullptr;
    }
    for (auto& graph : mBackwardGraph) {
        if (graph) {
            cudaGraphExecDestroy(graph);
            graph = nullptr;
        }
    }
    // Clean up prefetch event
    if (mPrefetchEvent) {
        cudaEventDestroy(mPrefetchEvent);
        mPrefetchEvent = nullptr;
    }
    // Clean up document masking GPU buffer
    if (mCuSeqlensGpu) {
        cudaFree(mCuSeqlensGpu);
        mCuSeqlensGpu = nullptr;
    }
}

void GraphExecutor::set_lora_state(const modules::ModularLoRAConfig* config,
                                   modules::ModularLoRAWeightsManager* weights,
                                   modules::ModularLoRAGradsManager* grads,
                                   modules::LoRARunState* run_state) {
    mLoRAConfig = config;
    mLoRAWeights = weights;
    mLoRAGrads = grads;
    mLoRARunState = run_state;
    if (mCompiledExecutor) {
        mCompiledExecutor->set_lora_state(mLoRAConfig, mLoRAWeights, mLoRAGrads, mLoRARunState);
    }
}

void GraphExecutor::set_weight_manager(DslWeightManager* weight_manager) {
    mWeightManager = weight_manager;
    if (mCompiledExecutor) {
        mCompiledExecutor->set_weight_manager(mWeightManager);
    }
}

void GraphExecutor::reset_cuda_graphs() {
    // Reset whole-graph captures
    if (mForwardGraph) {
        (void)cudaGraphExecDestroy(mForwardGraph);
        mForwardGraph = nullptr;
    }
    for (auto& g : mBackwardGraph) {
        if (g) {
            (void)cudaGraphExecDestroy(g);
            g = nullptr;
        }
    }
    // Reset per-layer graphs in run state
    mRunState.reset_cuda_graphs();
    // Reset split-attention segment graphs
    if (mCompiledExecutor) {
        mCompiledExecutor->reset_segment_graphs();
    }
}

void GraphExecutor::init(const GraphExecutorOptions& options) {
    if (!mForward) {
        throw std::runtime_error("DSL graph executor: module missing forward graph");
    }
    mHasLossOp = false;
    for (const auto& op : mForward->operations) {
        const std::string& op_type = (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;
        if (op_type == "fused_lm_head_loss" || op_type == "lm_head_loss" || op_type == "cross_entropy_loss" ||
            op_type == "cross_entropy") {
            mHasLossOp = true;
            break;
        }
    }

    // Check if module has explicit backward graph
    if (mModule.backward.has_value()) {
        mBackward = &mModule.backward.value();
    } else if (options.auto_backward) {
        // Derive backward graph automatically using autodiff
        DeriveBackwardOptions derive_opts;
        derive_opts.loss_name = options.loss_name;
        derive_opts.auto_save = true;
        derive_opts.accumulate_grads = true;

        try {
            mDerivedBackward = derive_backward_graph(*mForward, derive_opts);
            mBackward = &mDerivedBackward.value();

            // Merge auto-computed saves with forward.save, but filter out:
            // 1. Graph outputs (they don't need to be saved for backward)
            // 2. Tensors produced by ops that depend on lm_head (not available in full=false mode)
            std::unordered_set<std::string> save_set(mForward->save.begin(), mForward->save.end());
            for (const auto& s : mDerivedBackward->save) {
                save_set.insert(s);
            }
            // Remove graph outputs - they don't need to be saved for backward
            for (const auto& [name, _] : mForward->outputs) {
                save_set.erase(name);
            }
            // Also remove tensors that are produced by ops we don't want to save (e.g., large lm_head logits)
            // For now, we specifically exclude "logits_flat" as it's produced by the lm_head matmul
            save_set.erase("logits_flat");
            save_set.erase("logits");
            mSaveList.assign(save_set.begin(), save_set.end());
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("DSL graph executor: autodiff failed: ") + e.what());
        }
    }

    mViewSources.clear();
    mViewSourcesReverse.clear();
    mEmbeddingOutputs.clear();
    if (mForward) {
        for (const auto& op : mForward->operations) {
            const std::string& op_type =
                (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;
            if ((op_type == "view" || op_type == "reshape") && !op.outputs.empty() && !op.inputs.empty()) {
                const std::string& out = op.outputs.at(0);
                const std::string& in = op.inputs.at(0);
                mViewSources.emplace(out, in);
                mViewSourcesReverse.emplace(in, out);
            }
            if (op_type == "embedding" && !op.outputs.empty()) {
                mEmbeddingOutputs.push_back(op.outputs.at(0));
            }
        }
    }

    if (!mBackward) {
        throw std::runtime_error(
            "DSL graph executor: module missing backward graph (set auto_backward=true to derive automatically)");
    }

    auto is_noncapturable_op = [&](const Operation& op) {
        const std::string& op_type = (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;
        const bool is_embedding_bwd = (op_type == "embedding_backward" || op_type == "encoder_backward" ||
                                       op.name == "embedding_backward" || op.name == "encoder_backward");
        return is_embedding_bwd;
    };

    if (mBackward) {
        const auto& ops = mBackward->operations;
        const std::size_t op_count = ops.size();
        std::vector<char> noncapturable(op_count, 0);
        bool has_noncapturable = false;

        for (std::size_t idx = 0; idx < op_count; ++idx) {
            if (is_noncapturable_op(ops[idx])) {
                noncapturable[idx] = 1;
                has_noncapturable = true;
            }
        }

        if (has_noncapturable && op_count > 1) {
            std::unordered_map<std::string, std::size_t> producer;
            producer.reserve(op_count * 2);
            for (std::size_t idx = 0; idx < op_count; ++idx) {
                for (const auto& out : ops[idx].outputs) {
                    if (!out.empty()) {
                        producer[out] = idx;
                    }
                }
            }

            auto is_param_grad = [&](const std::string& name) {
                if (auto base = base_param_from_grad_heuristic(name)) {
                    return mWeights.has(*base);
                }
                return false;
            };

            std::vector<char> core(op_count, 0);
            for (std::size_t idx = 0; idx < op_count; ++idx) {
                bool has_output = false;
                bool all_param_grads = true;
                for (const auto& out : ops[idx].outputs) {
                    if (out.empty()) {
                        continue;
                    }
                    has_output = true;
                    if (!is_param_grad(out)) {
                        all_param_grads = false;
                        break;
                    }
                }
                if (!has_output || !all_param_grads) {
                    core[idx] = 1;
                }
            }

            std::vector<std::size_t> stack;
            stack.reserve(op_count);
            for (std::size_t idx = 0; idx < op_count; ++idx) {
                if (core[idx]) {
                    stack.push_back(idx);
                }
            }
            while (!stack.empty()) {
                std::size_t idx = stack.back();
                stack.pop_back();
                for (const auto& inp : ops[idx].inputs) {
                    if (inp.empty()) {
                        continue;
                    }
                    auto it = producer.find(inp);
                    if (it == producer.end()) {
                        continue;
                    }
                    std::size_t prod_idx = it->second;
                    if (!core[prod_idx]) {
                        core[prod_idx] = 1;
                        stack.push_back(prod_idx);
                    }
                }
            }

            std::size_t tail_count = 0;
            for (std::size_t idx = 0; idx < op_count; ++idx) {
                if (!core[idx]) {
                    ++tail_count;
                }
            }

            if (tail_count > 0 && tail_count < op_count) {
                Graph* mutable_backward = mDerivedBackward ? &mDerivedBackward.value() : nullptr;
                if (!mutable_backward) {
                    mReorderedBackward = *mBackward;
                    mutable_backward = &mReorderedBackward.value();
                    mBackward = mutable_backward;
                }
                auto& mutable_ops = mutable_backward->operations;
                std::vector<Operation> reordered;
                reordered.reserve(op_count);
                for (std::size_t idx = 0; idx < op_count; ++idx) {
                    if (core[idx]) {
                        reordered.push_back(mutable_ops[idx]);
                    }
                }
                for (std::size_t idx = 0; idx < op_count; ++idx) {
                    if (!core[idx]) {
                        reordered.push_back(mutable_ops[idx]);
                    }
                }
                mutable_ops.swap(reordered);
            }
        }
    }

    // Backward CUDA graphs are not compatible with ops that sync on other streams.
    // If we encounter such ops, capture only the prefix and run the tail uncaptured.
    mBackwardGraphCapturable = true;
    mBackwardGraphCut = mBackward ? mBackward->operations.size() : 0;
    if (mBackward) {
        for (std::size_t idx = 0; idx < mBackward->operations.size(); ++idx) {
            if (is_noncapturable_op(mBackward->operations[idx])) {
                mBackwardGraphCapturable = false;
                mBackwardGraphCut = idx;
                break;
            }
        }
    }
    mBackwardGraphsEnabled = mGraphsEnabled && mBackwardGraphCut > 0;

    // If we didn't derive backward (using explicit backward from module), use forward.save
    if (mSaveList.empty()) {
        mSaveList = mForward->save;
    }

    // Initialize forward plan storage (one per layer)
    if (mForwardPlan.size() != static_cast<std::size_t>(mConfig.NumLayers)) {
        mForwardPlan.resize(static_cast<std::size_t>(mConfig.NumLayers));
    }

    // Initialize compiled execution
    init_compiled_execution();
}

void GraphExecutor::reset_forward_plan() {
    if (mForwardPlan.size() != static_cast<std::size_t>(mConfig.NumLayers)) {
        mForwardPlan.resize(static_cast<std::size_t>(mConfig.NumLayers));
    }
    for (auto& plan : mForwardPlan) {
        plan.qkv = {};
        plan.out_proj = {};
        plan.mlp_up = {};
        plan.mlp_down = {};
        plan.attn = {};
    }
}

void GraphExecutor::record_matmul_plan(int layer_idx, modules::MatmulOp op, const MatmulForwardPlan& plan) {
    if (layer_idx < 0 || static_cast<std::size_t>(layer_idx) >= mForwardPlan.size()) {
        return;
    }
    auto& layer_plan = mForwardPlan[static_cast<std::size_t>(layer_idx)];
    switch (op) {
        case modules::MatmulOp::QKV: layer_plan.qkv = plan; break;
        case modules::MatmulOp::AttnOut: layer_plan.out_proj = plan; break;
        case modules::MatmulOp::MLPUp: layer_plan.mlp_up = plan; break;
        case modules::MatmulOp::MLPDown: layer_plan.mlp_down = plan; break;
        default: break;
    }
}

void GraphExecutor::record_attn_plan(int layer_idx, const AttnForwardPlan& plan) {
    if (layer_idx < 0 || static_cast<std::size_t>(layer_idx) >= mForwardPlan.size()) {
        return;
    }
    mForwardPlan[static_cast<std::size_t>(layer_idx)].attn = plan;
}

const LayerForwardPlan* GraphExecutor::forward_plan(int layer_idx) const {
    if (layer_idx < 0 || static_cast<std::size_t>(layer_idx) >= mForwardPlan.size()) {
        return nullptr;
    }
    return &mForwardPlan[static_cast<std::size_t>(layer_idx)];
}

void GraphExecutor::init_compiled_execution() {
    mCompiler = std::make_unique<GraphCompiler>(mModule, mConfig, mOptions, mWeights, mGrads);
    mCompiledExecutor = std::make_unique<CompiledExecutor>(mRunState, mWeights, mGrads, mConfig, mOptions);

    // Wire up optional components
    mCompiledExecutor->set_lora_state(mLoRAConfig, mLoRAWeights, mLoRAGrads, mLoRARunState);
    mCompiledExecutor->set_weight_manager(mWeightManager);
    if (mOptions.TrainingRecipe) {
        mCompiledExecutor->set_recipe(mOptions.TrainingRecipe.get());
    }
    mCompiledExecutor->set_hook_context(mHookContext);
    mCompiledExecutor->set_recompute_fn([this](int layer_idx, long B, long T, bool /*use_graph*/) {
        if (mCompiledForward) {
            // LoRA replay is slice-driven (see ``CompiledAttrs::lora_slices``).
            // No forward hook is needed; pass nullptr.
            mCompiledExecutor->replay_layer_forward(layer_idx, B, T, *mCompiledForward, nullptr);
        }
    });
    mCompiledExecutor->set_fp8_cache(&mFP8WeightCache);
    mCompiledExecutor->set_fp8_cache_transposed(&mFP8WeightCacheT);
    mCompiledExecutor->set_fp4_cache(&mFP4WeightCache, &mFP4WeightCacheT);
    mCompiledExecutor->set_saved_tensors(&mSaved);
    mCompiledExecutor->set_save_list(&mSaveList);
    mCompiledExecutor->set_forward_plan(&mForwardPlan);
    mCompiledExecutor->set_last_inputs_cpu(&mLastInputsCpu);
    mCompiledExecutor->set_rng_seed_fn([this]() { return next_rng_seed(); });
    mCompiledExecutor->set_embedding_outputs(mEmbeddingOutputs);
    mCompiledExecutor->set_slot_registry(&mCompiler->slot_registry());
    auto resolve_debug_dump_tensor = [this](const std::string& name) -> std::optional<Tensor> {
        const bool debug_dump_resolve = []() {
            const char* env = std::getenv("SUROGATE_DEBUG_DUMP_RESOLVE");
            return env && std::string(env) != "0";
        }();
        auto log_debug_dump_choice = [&](const char* branch, const Tensor& t) {
            if (!debug_dump_resolve) {
                return;
            }
            if (name != "xF" && name != "ln_final" && name != "xF_flat" && name != "residual_final" &&
                name != "final_residual" && name != "xN" && name != "residualN") {
                return;
            }
            std::fprintf(stderr,
                         "[DUMP_RESOLVE] name=%s branch=%s ptr=%p rank=%d sizes=[%ld,%ld,%ld,%ld]\n",
                         name.c_str(),
                         branch,
                         t.Data,
                         t.Rank,
                         t.Sizes[0],
                         t.Sizes[1],
                         t.Sizes[2],
                         t.Sizes[3]);
        };
        auto make_flat_view = [](Tensor t) -> Tensor {
            if (t.Rank >= 3) {
                return view_tensor(t, {t.Sizes[0] * t.Sizes[1], t.Sizes[2]});
            }
            return t;
        };
        auto resolve_last_block_output = [this]() -> std::optional<Tensor> {
            if (mConfig.NumLayers <= 0) {
                return std::nullopt;
            }
            const int last_layer = mConfig.NumLayers - 1;
            if (Tensor* h_out = block_activation_ptr(mRunState, last_layer, TensorSlot::BlockHOut);
                h_out && h_out->Data) {
                return *h_out;
            }
            if (Tensor* mlp_down = block_activation_ptr(mRunState, last_layer, TensorSlot::BlockMLPDown);
                mlp_down && mlp_down->Data) {
                return *mlp_down;
            }
            if (Tensor* res_att = block_activation_ptr(mRunState, last_layer, TensorSlot::BlockResidualAtt);
                res_att && res_att->Data) {
                return *res_att;
            }
            return std::nullopt;
        };

        const auto maybe_rope = [&](std::string_view probe) -> std::optional<Tensor> {
            if (probe.find("rope_freqs") != std::string_view::npos ||
                probe.find("freq_cis") != std::string_view::npos) {
                Tensor t = mRunState.rope_freqs(name);
                if (t.Data) {
                    log_debug_dump_choice("rope", t);
                    return t;
                }
            }
            return std::nullopt;
        };
        if (const Tensor* t = mCompiledExecutor->try_get_tensor(name); t && t->Data) {
            log_debug_dump_choice("direct", *t);
            return *t;
        }

        std::string lookup_name = name;
        const bool is_grad_block_name = starts_with(name, "d_blocks[");
        if (is_grad_block_name) {
            lookup_name = name.substr(2);
        }

        int layer_idx = -1;
        std::string field;
        if (parse_block_param(lookup_name, layer_idx, field)) {
            if (auto rope = maybe_rope(field)) {
                return rope;
            }

            std::string base_field = strip_ssa_suffix(field);
            bool is_flat = false;
            if (base_field.size() >= 5 && base_field.compare(base_field.size() - 5, 5, "_flat") == 0) {
                base_field = base_field.substr(0, base_field.size() - 5);
                is_flat = true;
            }

            // For backward names like "d_blocks[N].h_out", prepend "d_" to the
            // field so we look up the gradient slot (BlockDHOut) in the registry
            // instead of the activation slot (BlockHOut) — block_gradient_ptr
            // only handles BlockD* slots.
            const std::string slot_lookup = is_grad_block_name ? ("d_" + base_field) : base_field;
            const TensorSlot slot = builtin_slot_from_name(slot_lookup);
            if (slot != TensorSlot::Mapped) {
                Tensor* base = is_grad_block_name ? block_gradient_ptr(mRunState, layer_idx, slot)
                                                  : block_activation_ptr(mRunState, layer_idx, slot);
                if (base && base->Data) {
                    Tensor resolved = *base;
                    if (is_flat) {
                        resolved = make_flat_view(resolved);
                    }
                    log_debug_dump_choice(is_grad_block_name ? "block_grad_slot" : "block_slot", resolved);
                    return resolved;
                }
            }
        } else {
            if (auto rope = maybe_rope(name)) {
                return rope;
            }

            std::string base_name = strip_ssa_suffix(name);
            bool is_flat = false;
            if ((base_name == "xN" || base_name == "residualN" || base_name == "residual0" ||
                 base_name == "residual_final" || base_name == "final_residual") &&
                !is_grad_block_name) {
                if (auto last = resolve_last_block_output()) {
                    Tensor resolved = *last;
                    if (base_name.size() >= 5 && base_name.compare(base_name.size() - 5, 5, "_flat") == 0) {
                        resolved = make_flat_view(resolved);
                    }
                    log_debug_dump_choice("last_block_output", resolved);
                    return resolved;
                }
            }
            TensorSlot slot = resolve_slot_with_flat(base_name, &is_flat);
            if (slot != TensorSlot::Mapped && slot != TensorSlot::FreqCis) {
                if (Tensor* base = global_activation_ptr(mRunState, slot); base && base->Data) {
                    Tensor resolved = *base;
                    if (is_flat) {
                        resolved = make_flat_view(resolved);
                    }
                    log_debug_dump_choice("global_slot", resolved);
                    return resolved;
                }
            }
        }
        if (const Tensor* t = mCompiledExecutor->try_get_tensor_fuzzy(name); t && t->Data) {
            log_debug_dump_choice("fuzzy", *t);
            return *t;
        }
        return std::nullopt;
    };

    // Wire up debug dump callback (used by ops for non-finite diagnostics).
    mCompiledExecutor->set_debug_dump_fn(
        [this, resolve_debug_dump_tensor](const std::vector<std::string>& names, int /*layer_idx*/) {
            static const char* dir_env = std::getenv("SUROGATE_DEBUG_DUMP_DIR");
            if (!dir_env || !*dir_env) {
                return;
            }
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            for (const auto& name : names) {
                if (auto t = resolve_debug_dump_tensor(name); t.has_value() && t->Data) {
                    debug_dump_tensor(name, *t, std::string(dir_env), mRunState.MainStream);
                }
            }
        });
    // Per-layer forward dump callback: dump block-prefixed tensors at layer boundaries
    // (before shared activation buffers are overwritten by the next layer).
    mCompiledExecutor->set_debug_dump_layer_fn([this, resolve_debug_dump_tensor](int layer_idx) {
        static const char* dump_tensors_env = std::getenv("SUROGATE_DEBUG_DUMP_TENSORS");
        // Parse the (static, immutable) tensor list once — the layer hook fires
        // 70–700× per multi-step run and re-splitting the comma-separated env on
        // every call would be ~500k tiny allocations for a typical run.
        static const std::vector<std::string> tensor_names = debug_dump_parse_tensor_list(dump_tensors_env);
        // Non-static so the Python-side gradient subcommand can rotate per-step
        // dump directories between trainer.step() calls.
        const char* dump_dir_env = std::getenv("SUROGATE_DEBUG_DUMP_DIR");
        if (!dump_tensors_env || !dump_dir_env || !*dump_tensors_env || !*dump_dir_env) {
            return;
        }
        const std::string fwd_prefix = "blocks[" + std::to_string(layer_idx) + "].";
        auto matches_this_layer = [&](const std::string& n) {
            return n.rfind(fwd_prefix, 0) == 0;
        };
        bool found = false;
        for (const auto& name : tensor_names) {
            if (matches_this_layer(name)) {
                found = true;
                break;
            }
        }
        if (!found) {
            return;
        }
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        for (const auto& name : tensor_names) {
            if (!matches_this_layer(name)) {
                continue;
            }
            if (auto t = resolve_debug_dump_tensor(name); t.has_value() && t->Data) {
                debug_dump_tensor(name, *t, std::string(dump_dir_env), mRunState.MainStream);
            }
        }
    });
    // Backward layer dumps are separate so forward files don't get overwritten
    // during trainer.step().
    mCompiledExecutor->set_debug_dump_backward_layer_fn([this, resolve_debug_dump_tensor](int layer_idx) {
        static const char* dump_tensors_env = std::getenv("SUROGATE_DEBUG_DUMP_TENSORS");
        static const std::vector<std::string> tensor_names = debug_dump_parse_tensor_list(dump_tensors_env);
        const char* dump_dir_env = std::getenv("SUROGATE_DEBUG_DUMP_DIR");
        if (!dump_tensors_env || !dump_dir_env || !*dump_tensors_env || !*dump_dir_env) {
            return;
        }
        const std::string bwd_prefix = "d_blocks[" + std::to_string(layer_idx) + "].";
        bool found = false;
        for (const auto& name : tensor_names) {
            if (name.rfind(bwd_prefix, 0) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return;
        }
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        for (const auto& name : tensor_names) {
            if (name.rfind(bwd_prefix, 0) != 0) {
                continue;
            }
            if (auto t = resolve_debug_dump_tensor(name); t.has_value() && t->Data) {
                debug_dump_tensor(name, *t, std::string(dump_dir_env), mRunState.MainStream);
            }
        }
    });

    // Graphs will be compiled lazily on first forward when B/T are known
    mCompiledB = 0;
    mCompiledT = 0;
}

void GraphExecutor::compile_graphs(long B, long T) {
    if (!mCompiler || !mCompiledExecutor) {
        return;
    }

    // Recompile if batch/sequence dimensions changed
    if (B != mCompiledB || T != mCompiledT) {
        // Reset the tensor-id namespace so the forward+backward pair shares
        // a single tid space. Each compile() inherits the prior compile's
        // tid map, so a tensor name has the same tid in both graphs.
        // This prevents runtime slot collisions in mTensors when a forward
        // TensorRef leaks into backward execution.
        mCompiler->reset_tid_namespace();
        if (mForward) {
            mCompiledForward =
                std::make_unique<CompiledGraph>(mCompiler->compile(*mForward, B, T, /*is_backward=*/false));
            mCompiledForward->compute_layer_segments();
            mCompiledExecutor->set_forward_graph(mCompiledForward.get());
        }
        if (mBackward) {
            mCompiledBackward =
                std::make_unique<CompiledGraph>(mCompiler->compile(*mBackward, B, T, /*is_backward=*/true));
            mCompiledBackward->compute_layer_segments();
        }

        // Cross-graph region fixup (design/buffer-runtime-v4.md, M5.a):
        // promote forward activations consumed by backward to SaveForBwd in
        // both compiles. Filtered to match the runtime save-snapshot path
        // at compiled_ops_execute.cpp.
        //
        // When recompute is active, the forward-replay branch at
        // compiled_ops_execute.cpp pre-empts arena-backed saves entirely:
        // most block tensors are metadata-only (regenerated by replay in
        // backward), and the model-specific exceptions (e.g. qwen3_5
        // ln1/ln1_rstd) go through mMoeSavedBuffers rather than the
        // arena. An arena allocation would be dead weight — pass an empty
        // save list so finalize_save_for_bwd promotes nothing and the
        // SaveForBwd region sizes to 0.
        //
        // When recompute is inactive, the per-layer persist loop runs and
        // can consume the arena for tids in mSaveList. Filter out the
        // names that will still be skipped at runtime (no-Stack-source /
        // zero-byte / recompute-flagged) so the arena is sized to what
        // actually materializes, not to every fwd∧bwd-crossing tid.
        if (mCompiledForward && mCompiledBackward) {
            const bool recompute_active = mOptions.recompute_enabled();
            // Under recompute: pass an empty set so finalize_save_for_bwd
            // promotes zero tids and sizes SaveForBwd to 0 (replay
            // regenerates activations just-in-time, so no save needed).
            // Under no-recompute: pass nullopt so every fwd∧bwd-crossing
            // block tid is promoted to SaveForBwd and gets
            // retain_through_forward applied — name-match filtering
            // against mSaveList misses SSA-alias tids (`x_flat` aliases
            // `ln1`) whose canonical name differs from any save-list
            // entry, leaving them in FwdStack where within-layer coloring
            // clobbers their bytes before save_tensors captures them.
            std::optional<std::unordered_set<std::string>> save_name_arg;
            if (recompute_active) {
                save_name_arg = std::unordered_set<std::string>{};
            }
            dsl::finalize_save_for_bwd(*mCompiledForward,
                                       *mCompiledBackward,
                                       std::move(save_name_arg),
                                       /*fwd_per_layer_sections=*/!recompute_active);
        }

        // Phase-tree arena allocation (M5.d). Sized from baked offsets;
        // gated on SUROGATE_USE_PHASE_ARENAS=1 because it adds a second copy
        // of persistent+accumulator memory (no consumer yet). Re-allocate on
        // each (B, T) recompile — sizes may change if shapes change.
        //
        // Phase 3 subsystem #3: if SUROGATE_USE_PHASE_STACK=1, additionally
        // rebase mRunState.Stack onto the unified_stack arena (sized to match
        // the existing Stack capacity). This is the first subsystem flip to
        // actually route Qwen3's runtime allocations through an arena buffer.

        // Stack was rebased + adopted by DslRunState in a prior compile; keep
        // it there across (B,T) recompiles. The adopted buffer is owned by
        // DslRunState and survives arena release. We do NOT unbind (that
        // would need the original TensorAllocator-backed buffer, which was
        // freed after the adopt — see below).
        dsl::release_phase_arenas(mPhaseArenas);
        if (mCompiledForward && mCompiledBackward) {
            {
                const std::size_t stack_bytes = mRunState.Stack.capacity();
                // Phase 3 subsystem #4: bwd_cross_layer arena. Fixed 64 MiB;
                // the per-step bump resets at backward start. 0 bytes for dense
                // transformers (typical), < 16 MiB for small-MoE aux-loss.
                // Interpreter falls back to cudaMalloc if arena is insufficient.
                const std::size_t bwd_cross_layer_bytes = 64ULL * 1024 * 1024;
                // Phase 3 subsystem #6: moe_saved arena sized 256 MiB when
                // NumExperts > 0. Cross-step monotonic bump; cudaMalloc
                // fallback on exhaustion.
                const std::size_t moe_saved_bytes = mConfig.NumExperts > 0 ? 256ULL * 1024 * 1024 : 0;
                dsl::compute_arena_sizes(mPhaseArenas,
                                         *mCompiledForward,
                                         *mCompiledBackward,
                                         static_cast<int>(mConfig.NumLayers),
                                         stack_bytes,
                                         bwd_cross_layer_bytes,
                                         moe_saved_bytes);
                // Persistent arena is sized in three slabs:
                //   [0, base)                = base-weight region (layout offsets)
                //   [base, base+wm)          = DslWeightManager master/work/prefetch
                //   [base+wm, base+wm+lora)  = LoRA adapter master/work
                // Base-weight bytes clamp to what DslParamStore can actually
                // rebind — layout estimates over every ForwardParam tid, but
                // QLoRA-external and weight-manager-managed params live in
                // their own storage, so sizing for them would waste tens of
                // GiB. Each slab stays 0 when its owner is absent. QLoRA
                // quantized storage has its own provider-side self-arena
                // (GenericWeightManager::consume_self_arena) — not part of
                // this slab layout because the provider isn't wired until
                // after compile_graphs runs.
                std::size_t lora_slab_bytes = 0;
                std::size_t wm_slab_bytes = 0;
                std::size_t base_persistent_bytes = 0;
                std::size_t extras_slab_bytes = 0;
                if (mPhaseArenas.persistent_bytes > 0) {
                    base_persistent_bytes = mWeights.rebindable_persistent_bytes(*mCompiledForward);
                    mPhaseArenas.persistent_bytes = base_persistent_bytes;
                    if (mWeightManager) {
                        wm_slab_bytes = mWeightManager->total_persistent_bytes();
                        mPhaseArenas.persistent_bytes += wm_slab_bytes;
                    }
                    if (mLoRAWeights) {
                        lora_slab_bytes = mLoRAWeights->total_persistent_bytes();
                        mPhaseArenas.persistent_bytes += lora_slab_bytes;
                    }
                    // Non-graph persistent buffers (today: `output` logits
                    // scratch). These have no tid in the compiled graph, so
                    // the `persistent_tids` budget doesn't cover them; grow
                    // the arena here and route via
                    // rebind_non_graph_persistent_to_arena below.
                    extras_slab_bytes = mRunState.non_graph_persistent_extras_bytes();
                    mPhaseArenas.persistent_bytes += extras_slab_bytes;
                }
                // Accumulator slab clamps to what DslGradStore can actually
                // rebind; ZeRO-2 / offloaded / streaming / cpu-training
                // configs return 0 and keep allocator-owned gradient storage.
                if (mPhaseArenas.accumulator_bytes > 0) {
                    mPhaseArenas.accumulator_bytes = mGrads.rebindable_accumulator_bytes(*mCompiledBackward);
                }
                dsl::allocate_phase_arenas(mPhaseArenas);
                // Shadow coverage report: of the tids the arena plan claims,
                // how many actually fit (offset+bytes <= region capacity).
                dsl::validate_arena_coverage(mPhaseArenas, *mCompiledForward);
                dsl::validate_arena_coverage(mPhaseArenas, *mCompiledBackward);
                // Phase 4 M1: op-operand coverage baseline. Counts TensorRef
                // uses (inputs + outputs across all ops) whose tid resolves to
                // an arena-backed (region, offset) pair. Drives the deletion
                // sequence: M2/M3/M4 widen coverage; once at 100%, legacy
                // TensorSlot / SimplifiedLayerActivations / shared_tag /
                // builtin_slot_from_name / string-match dispatch can be cut.
                dsl::validate_op_operand_coverage(mPhaseArenas, *mCompiledForward);
                dsl::validate_op_operand_coverage(mPhaseArenas, *mCompiledBackward);
                // Bind the arena for in-executor consumption (e.g., SaveForBwd
                // persist writing straight to arena slot instead of cudaMalloc).
                if (mCompiledExecutor) {
                    mCompiledExecutor->set_phase_arenas(&mPhaseArenas);
                }
                // Rebase Stack onto the unified_stack arena on the first
                // (B,T). Stack must be empty here (post-compile, pre-step
                // boundary). Transfer ownership to DslRunState and free the
                // original Stack buffer so total memory doesn't double. On
                // subsequent recompiles the adopted buffer persists — no
                // re-rebase needed.
                if (!mStackRebasedToArena && mPhaseArenas.unified_stack_ptr != nullptr) {
                    mRunState.rebase_stack_to_external(mPhaseArenas.unified_stack_ptr,
                                                       mPhaseArenas.unified_stack_bytes);
                    mRunState.adopt_external_stack(mPhaseArenas.unified_stack_ptr, mPhaseArenas.unified_stack_bytes);
                    mPhaseArenas.unified_stack_ptr = nullptr;
                    mPhaseArenas.unified_stack_bytes = 0;
                    mRunState.free_allocator_stack_buffer();
                    mStackRebasedToArena = true;
                }
                // Phase 4 M3 groundwork: dump per-layer per-slot
                // (offset, bytes, region) for the simplified-activation
                // slots. Lets us sanity-check the coloring before designing
                // the runtime frame discipline that full migration needs.
                if (const char* dbg = std::getenv("SUROGATE_DEBUG_ACT_OFFSETS")) {
                    if (std::string(dbg) == "1" && mCompiledForward) {
                        dump_simplified_activation_offsets();
                    }
                }
                // Rebind each slab into its reserved region of the
                // Persistent arena. Each call is a no-op when its owner
                // is absent or has nothing device-resident to rebind.
                mWeights.rebind_to_persistent_arena(*mCompiledForward, mPhaseArenas, mRunState.MainStream);
                if (mWeightManager && wm_slab_bytes > 0 && mPhaseArenas.persistent_ptr != nullptr) {
                    std::byte* wm_base = mPhaseArenas.persistent_ptr + base_persistent_bytes;
                    mWeightManager->rebind_to_persistent_arena(wm_base, wm_slab_bytes, mRunState.MainStream);
                }
                if (mLoRAWeights && lora_slab_bytes > 0 && mPhaseArenas.persistent_ptr != nullptr) {
                    std::byte* lora_base = mPhaseArenas.persistent_ptr + base_persistent_bytes + wm_slab_bytes;
                    mLoRAWeights->rebind_to_persistent_arena(lora_base, lora_slab_bytes, mRunState.MainStream);
                }
                if (mPhaseArenas.persistent_ptr != nullptr) {
                    mRunState.rebind_non_block_to_persistent_arena(*mCompiledForward,
                                                                   mPhaseArenas,
                                                                   mRunState.MainStream);
                }
                if (extras_slab_bytes > 0 && mPhaseArenas.persistent_ptr != nullptr) {
                    std::byte* extras_base =
                        mPhaseArenas.persistent_ptr + base_persistent_bytes + wm_slab_bytes + lora_slab_bytes;
                    mRunState.rebind_non_graph_persistent_to_arena(extras_base,
                                                                   extras_slab_bytes,
                                                                   mRunState.MainStream);
                }
                // Route device-resident gradients into the Accumulator arena.
                // No-op when rebindable_accumulator_bytes clamped to 0
                // (ZeRO-2 / offload / streaming / cpu-training configs).
                if (mPhaseArenas.accumulator_bytes > 0 && mPhaseArenas.accumulator_ptr != nullptr) {
                    mGrads.rebind_to_accumulator_arena(*mCompiledBackward, mPhaseArenas, mRunState.MainStream);
                }
            }
        } else if (mCompiledExecutor) {
            mCompiledExecutor->set_phase_arenas(nullptr);
        }

        mCompiledB = B;
        mCompiledT = T;

        // Resize split-attention segment graph storage when dimensions change
        if (mCompiledForward && mCompiledBackward) {
            mCompiledExecutor->resize_segment_graphs(*mCompiledForward, *mCompiledBackward);
        }
    }
}

void GraphExecutor::dump_simplified_activation_offsets() {
    if (!mCompiledForward) return;

    static constexpr const char* kSlotNames[] = {
        "ln1_rstd",
        "ln1",
        "ln2_rstd",
        "ln2",
        "q_rstd",
        "k_rstd",
        "qkv",
        "qkv_rope",
        "lse",
        "att",
        "att_out",
        "residual_att",
        "mlp_up",
        "swiglu",
        "mlp_down",
        "h_out",
    };

    const int num_layers = static_cast<int>(mConfig.NumLayers);
    std::cerr << "[act-offsets] fwd_stack_arena_bytes=" << mPhaseArenas.fwd_stack_bytes << "\n";
    for (int L = 0; L < num_layers; ++L) {
        const std::string prefix = "blocks[" + std::to_string(L) + "].";
        for (const char* slot : kSlotNames) {
            const std::string name = prefix + slot;
            const int tid = mCompiledForward->find_tensor_id(name);
            if (tid < 0) continue;
            const auto& meta = mCompiledForward->tensor_meta[static_cast<std::size_t>(tid)];
            std::cerr << "[act-offsets] L=" << L << " slot=" << slot << " tid=" << tid
                      << " region=" << dsl::region_kind_name(meta.region);
            if (meta.offset != SIZE_MAX)
                std::cerr << " offset=" << meta.offset;
            else
                std::cerr << " offset=UNASSIGNED";
            std::cerr << " bytes=" << meta.bytes << "\n";
        }
    }
}

// ---------------------------------------------------------------------------
// estimate_backward_stack_peak — static analysis of backward graph
//
// Walk the compiled backward graph and compute the peak stack-resident memory
// within any single layer's backward pass.  At runtime the stack is restored
// to initial_checkpoint at every layer_end boundary, so the peak within one
// layer is the binding constraint.
//
// Two sources of stack memory are modelled:
//   1. Graph-level Temporary-slot outputs (shape known at compile time).
//   2. Op-internal temporaries allocated by dispatch functions (estimated from
//      op input shapes for known heavy ops like ChunkGatedDeltaRuleBackward).
// ---------------------------------------------------------------------------
long GraphExecutor::estimate_backward_stack_peak(long B, long T) {
    // Thin wrapper: compile the backward graph if needed, then delegate to
    // `dsl::graph_backward_stack_peak`. The plan provides the stack-residency
    // flags (`large_bwd_temps_on_stack`, `ffn_temps_on_stack`) that used to
    // be queried via `mRunState`.
    compile_graphs(B, T);
    return graph_backward_stack_peak(mCompiledBackward.get(), mRunState.buffer_plan());
}

void GraphExecutor::set_internal_graphs_enabled(bool enabled) {
    const bool allow = enabled && mOptions.UseCudaGraphs;
    mGraphsEnabled = allow;
    mBackwardGraphsEnabled = allow;
    mPerLayerGraphsEnabled = allow && mRunState.per_layer_graphs_enabled();
}

bool GraphExecutor::internal_graphs_enabled() const {
    return mGraphsEnabled;
}

bool GraphExecutor::has_capture_unsafe_ops() const {
    return graph_has_capture_unsafe_ops(mCompiledForward.get());
}

size_t GraphExecutor::saved_buffers_total_bytes() const {
    return mCompiledExecutor ? mCompiledExecutor->saved_buffers_total_bytes() : 0;
}

int GraphExecutor::saved_buffers_count() const {
    return mCompiledExecutor ? mCompiledExecutor->saved_buffers_count() : 0;
}

const std::unordered_map<std::string, size_t>& GraphExecutor::saved_buffers_sizes() const {
    if (mCompiledExecutor) {
        return mCompiledExecutor->saved_buffers_sizes();
    }
    static const std::unordered_map<std::string, size_t> empty;
    return empty;
}

void GraphExecutor::execute_forward(long B,
                                    long T,
                                    NCCLCommunicator& comm,
                                    bool full,
                                    const modules::ForwardHook* hook) {
    // LoRA is slice-driven via ``CompiledAttrs::lora_slices``; the forward
    // hook parameter is kept for ABI stability with callers that still
    // build hook values, but nothing is invoked from it.
    (void)hook;

    compile_graphs(B, T);

    if (!mCompiledForward || !mCompiledExecutor) {
        throw std::runtime_error("DSL graph executor: compiled forward graph not available");
    }

    auto& rs = mRunState;
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool in_capture = (cudaStreamIsCapturing(rs.MainStream, &capture_status) == cudaSuccess &&
                             capture_status != cudaStreamCaptureStatusNone);
    // Some ops are capture-unsafe due to JIT kernels and/or per-step host-side
    // bookkeeping (MoE/EP routing metadata). Route through split mode so these
    // ops run eagerly while other segments can still use CUDA graphs.
    const bool has_capture_unsafe_ops = graph_has_capture_unsafe_ops(mCompiledForward.get());
    // When doc masking or capture-unsafe ops are present, use split-attention mode.
    const bool doc_masking_active = (mCuSeqlensGpu != nullptr);
    const bool has_tiled_mlp = mCompiledForward && !mCompiledForward->mlp_tile_groups.empty();
    const bool needs_split = doc_masking_active || has_capture_unsafe_ops || has_tiled_mlp;
    const bool use_split_attention = needs_split && mOptions.UseCudaGraphs && !in_capture;
    const bool use_graphs = mGraphsEnabled && !in_capture && !needs_split;
    if (use_graphs && (mGraphB != B || mGraphT != T)) {
        reset_cuda_graphs();
        mGraphB = B;
        mGraphT = T;
    }
    mCompiledExecutor->set_split_attention_graphs(use_split_attention);
    const bool recompute_active = mOptions.recompute_enabled() && mCompiledForward != nullptr;
    mCompiledExecutor->set_recompute_enabled(recompute_active);
    const bool capturing = use_graphs && mForwardGraph == nullptr;
    if (!use_graphs || capturing) {
        mSaved.clear();
        reset_forward_plan();
    }
    if (capturing) {
        // Preallocate persistent save buffers before CUDA graph capture to avoid cudaMalloc
        // inside save_tensors (which is not allowed during capture).
        mCompiledExecutor->set_dimensions(B, T);
        mCompiledExecutor->prepare_saved_buffers_for_capture(mSaveList, mCompiledForward.get());

        // Prime FP8/FP4 weight caches BEFORE capture so matmul dispatch can consume cached weights
        // without allocating during cudaStreamBeginCapture.
        prime_fp8_weight_cache({});
        prime_fp4_weight_cache({});
    } else if (!use_graphs && !in_capture) {
        // External/full-step CUDA graph capture paths (outside GraphExecutor) can still
        // call save_tensors() while the stream is captured. Preallocate persistent save
        // buffers during regular eager execution so later captured passes don't need
        // cudaMalloc (which is forbidden during capture).
        mCompiledExecutor->set_dimensions(B, T);
        mCompiledExecutor->prepare_saved_buffers_for_capture(mSaveList, mCompiledForward.get());

        // Prime FP4 weight caches on first call. This covers split-attention mode
        // (sample_packing + CUDA graphs) where use_graphs is false but we still need
        // cached weights to avoid re-quantizing on every matmul.
        if (!mWeightCachesPrimed) {
            prime_fp4_weight_cache({});
            mWeightCachesPrimed = true;
        }
    } else if (!mWeightCachesPrimed) {
        // Prime FP4 weight caches on first eager execution (e.g., QLoRA where CUDA
        // graphs are disabled). FP4 on-the-fly quantization (two-level block scaling,
        // F8_128x4 swizzle, FP4 packing) is expensive and dominates step time without
        // caching. FP8 caches are NOT primed here: FP8 on-the-fly quantization is
        // cheap (single abs_max + per-element scale), and priming FP8 caches adds
        // ~2x model size in persistent GPU memory (FP8 + FP8 transposed), causing
        // OOM on memory-constrained GPUs with QLoRA.
        prime_fp4_weight_cache({});
        mWeightCachesPrimed = true;
    }

    auto run_ops = [&]() {
        mCompiledExecutor->set_dimensions(B, T);
        mCompiledExecutor->set_recompute_enabled(recompute_active);
        mCompiledExecutor->set_capturing(capturing);
        mCompiledExecutor->execute_forward(*mCompiledForward, comm, full, hook);
        // Save tensors for backward (same list as non-compiled path).
        mCompiledExecutor->save_tensors(mSaveList);
    };

    trace_or_execute_cuda_graph_with_stack(run_ops,
                                           rs.MainStream,
                                           mForwardGraph,
                                           use_graphs,
                                           rs.Stack,
                                           mForwardCheckpoint);
    // On CUDA graph replay, run_ops isn't executed, so saved tensors are stale.
    // Refresh them here to reflect the current forward activations.
    if (use_graphs && !capturing) {
        mCompiledExecutor->save_tensors(mSaveList);
    }
    mCompiledExecutor->set_capturing(false);
}

void GraphExecutor::execute_backward(long B,
                                     long T,
                                     NCCLCommunicator& comm,
                                     int grad_accum_steps,
                                     int micro_step,
                                     const modules::BackwardHook* hook,
                                     bool skip_zeroing) {
    compile_graphs(B, T);

    if (!mCompiledBackward || !mCompiledExecutor) {
        throw std::runtime_error("DSL graph executor: compiled backward graph not available");
    }

    auto& rs = mRunState;
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool in_capture = (cudaStreamIsCapturing(rs.MainStream, &capture_status) == cudaSuccess &&
                             capture_status != cudaStreamCaptureStatusNone);
    const bool has_capture_unsafe_ops =
        graph_has_capture_unsafe_ops(mCompiledForward ? mCompiledForward.get() : mCompiledBackward.get());
    const bool doc_masking_active_bwd = (mCuSeqlensGpu != nullptr);
    const bool has_capture_unsafe_bwd = has_capture_unsafe_ops;
    const bool has_tiled_mlp_bwd = mCompiledBackward && !mCompiledBackward->mlp_tile_groups.empty();
    const bool needs_split_bwd = doc_masking_active_bwd || has_capture_unsafe_bwd || has_tiled_mlp_bwd;
    const bool use_split_attention_bwd = needs_split_bwd && mOptions.UseCudaGraphs && !in_capture;
    const bool use_graphs = mBackwardGraphsEnabled && mBackwardGraphCapturable && !in_capture && !needs_split_bwd;
    mCompiledExecutor->set_split_attention_graphs(use_split_attention_bwd);
    if (use_graphs && (mGraphB != B || mGraphT != T)) {
        reset_cuda_graphs();
        mGraphB = B;
        mGraphT = T;
    }
    const bool recompute_active = mOptions.recompute_enabled() && mCompiledForward != nullptr;
    const int graph_idx = (micro_step > 0) ? 1 : 0;
    const bool capturing = use_graphs && mBackwardGraph[graph_idx] == nullptr;
    if (capturing) {
        // Same reason as forward: avoid allocating inside capture when a recipe wants cached weights.
        prime_fp8_weight_cache({});
        prime_fp8_weight_cache_transposed({});
        prime_fp4_weight_cache({});
    }

    auto run_ops = [&]() {
        mCompiledExecutor->set_dimensions(B, T);
        mCompiledExecutor->set_recompute_enabled(recompute_active);
        mCompiledExecutor->set_recompute_use_graphs(use_graphs && !capturing);
        mCompiledExecutor->set_capturing(capturing);
        mCompiledExecutor->execute_backward(*mCompiledBackward, comm, grad_accum_steps, micro_step, hook, skip_zeroing);
    };

    trace_or_execute_cuda_graph_with_stack(run_ops,
                                           rs.MainStream,
                                           mBackwardGraph[graph_idx],
                                           use_graphs,
                                           rs.Stack,
                                           mBackwardCheckpoint[graph_idx]);
    mCompiledExecutor->set_capturing(false);
}

unsigned int GraphExecutor::next_rng_seed() {
    return static_cast<unsigned int>(mRng());
}

std::vector<std::byte> GraphExecutor::rng_state() const {
    std::stringstream tmp;
    static_cast<std::ostream&>(tmp) << mRng;
    auto view = tmp.rdbuf()->view();
    std::vector<std::byte> state;
    state.reserve(view.size());
    std::transform(view.begin(), view.end(), std::back_inserter(state), [](char c) {
        return static_cast<std::byte>(c);
    });
    return state;
}

void GraphExecutor::set_rng_state(const std::vector<std::byte>& state) {
    std::stringstream tmp;
    tmp.write(reinterpret_cast<const char*>(state.data()), state.size());
    static_cast<std::istream&>(tmp) >> mRng;
}

void GraphExecutor::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    auto& rs = mRunState;

    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRARunState) {
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = true;
    }

    const bool in_capture = stream_is_capturing(rs.MainStream);
    if (micro_step == 0) {
        if (!in_capture) {
            CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.OptimizerDone, 0));
        }
        if (rs.has_fp8_delayed_scaling()) {
            if (auto* fp8_state = rs.get_fp8_scaling_state()) {
                if (!mFP8ScalingInitialized) {
                    fp8_state->reset(rs.MainStream);
                    mFP8ScalingInitialized = true;
                }
                fp8_state->zero_recorded_amaxes(rs.MainStream);
            }
        }
        rs.reset_moe_stats();
    }

    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];

    if (mHasLossOp && micro_step == 0) {
        fill_zero(rs.Losses, rs.MainStream);
        fill_zero(rs.ValidTokenCount, rs.MainStream);
        fill_zero(rs.CorrectCount, rs.MainStream);
    }

    // Copy inputs and position ids to device.
    {
        const std::size_t input_bytes =
            static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        copy_position_ids_to_device(position_ids, rs.PositionIDs, B, T, rs.MainStream);
        if (mHasLossOp) {
            const std::size_t target_bytes =
                static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(rs.Targets.DType);
            CUDA_CHECK(cudaMemcpyAsync(rs.Targets.Data,
                                       rs.Targets_CPU.Data,
                                       target_bytes,
                                       cudaMemcpyHostToDevice,
                                       rs.MainStream));
        }
        if (rs.VisualPosMasks.Data && rs.VisualPosMasks_CPU.Data) {
            const std::size_t mask_bytes = rs.VisualPosMasks.bytes();
            CUDA_CHECK(cudaMemcpyAsync(rs.VisualPosMasks.Data,
                                       rs.VisualPosMasks_CPU.Data,
                                       mask_bytes,
                                       cudaMemcpyHostToDevice,
                                       rs.MainStream));
        }
        if (rs.VisualEmbeds.Data && rs.VisualEmbeds_CPU.Data) {
            const std::size_t embed_bytes = rs.VisualEmbeds.bytes();
            CUDA_CHECK(cudaMemcpyAsync(rs.VisualEmbeds.Data,
                                       rs.VisualEmbeds_CPU.Data,
                                       embed_bytes,
                                       cudaMemcpyHostToDevice,
                                       rs.MainStream));
        }
        if (!rs.DeepstackVisualEmbeds.empty() &&
            rs.DeepstackVisualEmbeds.size() == rs.DeepstackVisualEmbeds_CPU.size()) {
            for (std::size_t i = 0; i < rs.DeepstackVisualEmbeds.size(); ++i) {
                if (!rs.DeepstackVisualEmbeds[i].Data || !rs.DeepstackVisualEmbeds_CPU[i].Data) {
                    continue;
                }
                const std::size_t bytes = rs.DeepstackVisualEmbeds[i].bytes();
                CUDA_CHECK(cudaMemcpyAsync(rs.DeepstackVisualEmbeds[i].Data,
                                           rs.DeepstackVisualEmbeds_CPU[i].Data,
                                           bytes,
                                           cudaMemcpyHostToDevice,
                                           rs.MainStream));
            }
        }
        record_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    }

    execute_forward(B, T, comm, /*full=*/false, nullptr);

    sync_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    record_event_if_not_capturing(rs.ForwardDone, rs.MainStream);
}

float GraphExecutor::validate(Tensor inputs,
                              Tensor position_ids,
                              Tensor targets,
                              NCCLCommunicator& comm,
                              int micro_step) {
    auto& rs = mRunState;

    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRARunState) {
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = false;
    }

    const bool in_capture = stream_is_capturing(rs.MainStream);
    if (micro_step == 0) {
        if (!in_capture) {
            CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.OptimizerDone, 0));
        }
        if (rs.has_fp8_delayed_scaling()) {
            if (auto* fp8_state = rs.get_fp8_scaling_state()) {
                if (!mFP8ScalingInitialized) {
                    fp8_state->reset(rs.MainStream);
                    mFP8ScalingInitialized = true;
                }
                fp8_state->zero_recorded_amaxes(rs.MainStream);
            }
        }
        rs.reset_moe_stats();
    }
    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];

    if (mHasLossOp) {
        fill_zero(rs.Losses, rs.MainStream);
        fill_zero(rs.ValidTokenCount, rs.MainStream);
        fill_zero(rs.CorrectCount, rs.MainStream);
    }

    // Copy inputs and position ids to device.
    {
        const std::size_t input_bytes =
            static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        copy_position_ids_to_device(position_ids, rs.PositionIDs, B, T, rs.MainStream);
        if (mHasLossOp) {
            const std::size_t target_bytes =
                static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
            if (targets.Device == -1) {
                CUDA_CHECK(cudaMemcpyAsync(rs.Targets.Data,
                                           targets.Data,
                                           target_bytes,
                                           cudaMemcpyHostToDevice,
                                           rs.MainStream));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(rs.Targets.Data,
                                           targets.Data,
                                           target_bytes,
                                           cudaMemcpyDeviceToDevice,
                                           rs.MainStream));
            }
        }
        if (rs.VisualPosMasks.Data && rs.VisualPosMasks_CPU.Data) {
            const std::size_t mask_bytes = rs.VisualPosMasks.bytes();
            CUDA_CHECK(cudaMemcpyAsync(rs.VisualPosMasks.Data,
                                       rs.VisualPosMasks_CPU.Data,
                                       mask_bytes,
                                       cudaMemcpyHostToDevice,
                                       rs.MainStream));
        }
        if (rs.VisualEmbeds.Data && rs.VisualEmbeds_CPU.Data) {
            const std::size_t embed_bytes = rs.VisualEmbeds.bytes();
            CUDA_CHECK(cudaMemcpyAsync(rs.VisualEmbeds.Data,
                                       rs.VisualEmbeds_CPU.Data,
                                       embed_bytes,
                                       cudaMemcpyHostToDevice,
                                       rs.MainStream));
        }
        if (!rs.DeepstackVisualEmbeds.empty() &&
            rs.DeepstackVisualEmbeds.size() == rs.DeepstackVisualEmbeds_CPU.size()) {
            for (std::size_t i = 0; i < rs.DeepstackVisualEmbeds.size(); ++i) {
                if (!rs.DeepstackVisualEmbeds[i].Data || !rs.DeepstackVisualEmbeds_CPU[i].Data) {
                    continue;
                }
                const std::size_t bytes = rs.DeepstackVisualEmbeds[i].bytes();
                CUDA_CHECK(cudaMemcpyAsync(rs.DeepstackVisualEmbeds[i].Data,
                                           rs.DeepstackVisualEmbeds_CPU[i].Data,
                                           bytes,
                                           cudaMemcpyHostToDevice,
                                           rs.MainStream));
            }
        }
        record_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    }

    execute_forward(B, T, comm, /*full=*/false, nullptr);

    sync_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    record_event_if_not_capturing(rs.ForwardDone, rs.MainStream);

    reduce_loss(rs, B, T, comm);
    comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, rs.MainStream);
    comm.all_reduce_sum_int(rs.CorrectCount.template get<int>(), /*n=*/1, rs.MainStream);

    CUDA_CHECK(
        cudaMemcpyAsync(rs.NormHost, rs.ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost, rs.MainStream));
    CUDA_CHECK(
        cudaMemcpyAsync(rs.AccuracyHost, rs.CorrectCount.Data, sizeof(int), cudaMemcpyDeviceToHost, rs.MainStream));
    CUDA_CHECK(cudaDeviceSynchronize());

    int valid_tokens = *reinterpret_cast<int*>(rs.NormHost);
    int correct_tokens = *reinterpret_cast<int*>(rs.AccuracyHost);
    if (valid_tokens > 0) {
        float avg_valid = static_cast<float>(valid_tokens) / static_cast<float>(std::max(1, comm.world_size()));
        *rs.LossHost /= avg_valid;
        *rs.AccuracyHost = (static_cast<float>(correct_tokens) / static_cast<float>(valid_tokens)) * 100.0f;
    } else {
        *rs.LossHost = 0.0f;
        *rs.AccuracyHost = 0.0f;
    }

    rs.temp_free(rs.non_block_activations().output);

    return *rs.LossHost;
}

void GraphExecutor::backward(Tensor inputs,
                             Tensor targets,
                             NCCLCommunicator& comm,
                             int grad_accum_steps,
                             int micro_step) {
    auto& rs = mRunState;
    auto& grads = mGrads;
    const auto& config = mConfig;
    rs.GradAccumSteps = std::max(1, grad_accum_steps);
    rs.WorldSize = std::max(1, comm.world_size());

    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];
    mLastInputsCpu = inputs;

    const bool in_capture = stream_is_capturing(rs.MainStream);
    const cudaStream_t target_stream = in_capture ? rs.MainStream : rs.side_stream();
    // In the DSL training loop, forward() copies targets from rs.Targets_CPU when the forward
    // graph includes a loss op. In graphed full-step execution this would otherwise duplicate
    // a large H2D memcpy node for every micro-step.
    const bool skip_target_copy =
        mHasLossOp && targets.Device == -1 && rs.Targets_CPU.Data != nullptr && rs.Targets_CPU.Data == targets.Data;
    // Copy targets to device (side stream in eager, main stream during capture).
    {
        if (!in_capture) {
            CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.BackwardDone, 0));
        }
        if (!skip_target_copy) {
            const std::size_t target_bytes =
                static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
            CUDA_CHECK(
                cudaMemcpyAsync(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice, target_stream));
            record_event_if_not_capturing(rs.TransferDone, target_stream);
        }
    }

    if (micro_step == 0) {
        const cudaStream_t grad_stream = in_capture ? rs.MainStream : rs.side_stream();
        grads.start_micro_step(grad_stream, micro_step, grad_accum_steps);
        record_event_if_not_capturing(rs.side_stream_event(), grad_stream);
    } else {
        grads.start_micro_step(rs.MainStream, micro_step, grad_accum_steps);
    }

    // Zero non-block gradient buffers
    fill_zero(rs.non_block_gradients().d_ln_final, rs.MainStream);
    if (rs.non_block_gradients().d_embeddings.Data && !rs.is_lora_only_mode()) {
        fill_zero(rs.non_block_gradients().d_embeddings, rs.MainStream);
    }
    if (config.NumLayers > 0) {
        if (Tensor* d_res_ffn = block_gradient_ptr(rs, config.NumLayers - 1, TensorSlot::BlockDResFFN)) {
            fill_zero(*d_res_ffn, rs.MainStream);
        }
    }

    // Zero all activation gradient buffers to prevent stale gradients from accumulating.
    // This is critical for FFT mode where rmsnorm_backward accumulates (+=) to dinp.
    // Without zeroing, stale gradients from previous steps can cause gradient explosion.
    rs.zero_activation_gradients(rs.MainStream);

    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRAGrads && mLoRARunState) {
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = true;
        mLoRAGrads->start_micro_step(rs.MainStream, micro_step, grad_accum_steps);
    }

    const bool last_step = micro_step == grad_accum_steps - 1;
    if (last_step) {
        reduce_loss(rs, B, T, comm);
        comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, rs.MainStream);
    }

    if (!in_capture) {
        if (!skip_target_copy) {
            CUDA_CHECK(cudaStreamWaitEvent(rs.MainStream, rs.TransferDone, 0));
        }
        if (micro_step == 0) {
            CUDA_CHECK(cudaStreamWaitEvent(rs.MainStream, rs.side_stream_event(), 0));
        }
    }

    execute_backward(B, T, comm, grad_accum_steps, micro_step, nullptr, /*skip_zeroing=*/true);

    grads.end_micro_step(rs.MainStream, comm);
    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRAGrads) {
        mLoRAGrads->end_micro_step(rs.MainStream, comm);
    }

    // Start async all-reduce on last micro-step (overlaps with CPU work and optimizer prep)
    // Note: LoRA gradients are already reduced in end_micro_step() above
    if (last_step && comm.world_size() > 1) {
        // Ensure all per-layer gradient reductions on side_stream complete before
        // recording all_reduce_done_event. Layer reductions were started during backward
        // on side_stream to overlap with compute on MainStream.
        if (grads.is_overlapped_enabled()) {
            CUDA_CHECK(cudaEventRecord(rs.side_stream_event(), rs.side_stream()));
            CUDA_CHECK(cudaStreamWaitEvent(rs.MainStream, rs.side_stream_event(), 0));
        }
        grads.reduce_all_async(comm, rs.MainStream, rs.all_reduce_done_event());
    }

    record_event_if_not_capturing(rs.BackwardDone, rs.MainStream);
    if (!skip_target_copy) {
        sync_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    }
}

void GraphExecutor::forward_with_hook(Tensor inputs,
                                      Tensor position_ids,
                                      NCCLCommunicator& comm,
                                      int micro_step,
                                      const modules::ForwardHook& hook) {
    auto& rs = mRunState;

    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRARunState) {
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = true;
    }

    const bool in_capture = stream_is_capturing(rs.MainStream);
    if (micro_step == 0) {
        if (!in_capture) {
            CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.OptimizerDone, 0));
        }
        if (rs.has_fp8_delayed_scaling()) {
            if (auto* fp8_state = rs.get_fp8_scaling_state()) {
                if (!mFP8ScalingInitialized) {
                    fp8_state->reset(rs.MainStream);
                    mFP8ScalingInitialized = true;
                }
                fp8_state->zero_recorded_amaxes(rs.MainStream);
            }
        }
        rs.reset_moe_stats();
    }

    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];

    if (mHasLossOp && micro_step == 0) {
        fill_zero(rs.Losses, rs.MainStream);
        fill_zero(rs.ValidTokenCount, rs.MainStream);
        fill_zero(rs.CorrectCount, rs.MainStream);
    }

    // Copy inputs and position ids to device.
    {
        const std::size_t input_bytes =
            static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        copy_position_ids_to_device(position_ids, rs.PositionIDs, B, T, rs.MainStream);
        if (mHasLossOp) {
            const std::size_t target_bytes =
                static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(rs.Targets.DType);
            CUDA_CHECK(cudaMemcpyAsync(rs.Targets.Data,
                                       rs.Targets_CPU.Data,
                                       target_bytes,
                                       cudaMemcpyHostToDevice,
                                       rs.MainStream));
        }
        if (rs.VisualPosMasks.Data && rs.VisualPosMasks_CPU.Data) {
            const std::size_t mask_bytes = rs.VisualPosMasks.bytes();
            CUDA_CHECK(cudaMemcpyAsync(rs.VisualPosMasks.Data,
                                       rs.VisualPosMasks_CPU.Data,
                                       mask_bytes,
                                       cudaMemcpyHostToDevice,
                                       rs.MainStream));
        }
        if (rs.VisualEmbeds.Data && rs.VisualEmbeds_CPU.Data) {
            const std::size_t embed_bytes = rs.VisualEmbeds.bytes();
            CUDA_CHECK(cudaMemcpyAsync(rs.VisualEmbeds.Data,
                                       rs.VisualEmbeds_CPU.Data,
                                       embed_bytes,
                                       cudaMemcpyHostToDevice,
                                       rs.MainStream));
        }
        if (!rs.DeepstackVisualEmbeds.empty() &&
            rs.DeepstackVisualEmbeds.size() == rs.DeepstackVisualEmbeds_CPU.size()) {
            for (std::size_t i = 0; i < rs.DeepstackVisualEmbeds.size(); ++i) {
                if (!rs.DeepstackVisualEmbeds[i].Data || !rs.DeepstackVisualEmbeds_CPU[i].Data) {
                    continue;
                }
                const std::size_t bytes = rs.DeepstackVisualEmbeds[i].bytes();
                CUDA_CHECK(cudaMemcpyAsync(rs.DeepstackVisualEmbeds[i].Data,
                                           rs.DeepstackVisualEmbeds_CPU[i].Data,
                                           bytes,
                                           cudaMemcpyHostToDevice,
                                           rs.MainStream));
            }
        }
        record_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    }

    // Configure graphs for hooked execution (may differ in topology)
    if (hook) {
        rs.configure_forward_graphs(/*hooked=*/true);
    }

    execute_forward(B, T, comm, /*full=*/false, hook ? &hook : nullptr);

    sync_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    record_event_if_not_capturing(rs.ForwardDone, rs.MainStream);
}

float GraphExecutor::validate_with_hook(Tensor inputs,
                                        Tensor position_ids,
                                        Tensor targets,
                                        NCCLCommunicator& comm,
                                        int micro_step,
                                        const modules::ForwardHook& hook) {
    auto& rs = mRunState;

    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRARunState) {
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = false;
    }

    const bool in_capture = stream_is_capturing(rs.MainStream);
    if (micro_step == 0) {
        if (!in_capture) {
            CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.OptimizerDone, 0));
        }
        if (rs.has_fp8_delayed_scaling()) {
            if (auto* fp8_state = rs.get_fp8_scaling_state()) {
                if (!mFP8ScalingInitialized) {
                    fp8_state->reset(rs.MainStream);
                    mFP8ScalingInitialized = true;
                }
                fp8_state->zero_recorded_amaxes(rs.MainStream);
            }
        }
        rs.reset_moe_stats();
    }
    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];

    if (mHasLossOp) {
        fill_zero(rs.Losses, rs.MainStream);
        fill_zero(rs.ValidTokenCount, rs.MainStream);
        fill_zero(rs.CorrectCount, rs.MainStream);
    }

    // Copy inputs and position ids to device.
    {
        const std::size_t input_bytes =
            static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        copy_position_ids_to_device(position_ids, rs.PositionIDs, B, T, rs.MainStream);
        if (mHasLossOp) {
            const std::size_t target_bytes =
                static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
            if (targets.Device == -1) {
                CUDA_CHECK(cudaMemcpyAsync(rs.Targets.Data,
                                           targets.Data,
                                           target_bytes,
                                           cudaMemcpyHostToDevice,
                                           rs.MainStream));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(rs.Targets.Data,
                                           targets.Data,
                                           target_bytes,
                                           cudaMemcpyDeviceToDevice,
                                           rs.MainStream));
            }
        }
        if (rs.VisualPosMasks.Data && rs.VisualPosMasks_CPU.Data) {
            const std::size_t mask_bytes = rs.VisualPosMasks.bytes();
            CUDA_CHECK(cudaMemcpyAsync(rs.VisualPosMasks.Data,
                                       rs.VisualPosMasks_CPU.Data,
                                       mask_bytes,
                                       cudaMemcpyHostToDevice,
                                       rs.MainStream));
        }
        if (rs.VisualEmbeds.Data && rs.VisualEmbeds_CPU.Data) {
            const std::size_t embed_bytes = rs.VisualEmbeds.bytes();
            CUDA_CHECK(cudaMemcpyAsync(rs.VisualEmbeds.Data,
                                       rs.VisualEmbeds_CPU.Data,
                                       embed_bytes,
                                       cudaMemcpyHostToDevice,
                                       rs.MainStream));
        }
        if (!rs.DeepstackVisualEmbeds.empty() &&
            rs.DeepstackVisualEmbeds.size() == rs.DeepstackVisualEmbeds_CPU.size()) {
            for (std::size_t i = 0; i < rs.DeepstackVisualEmbeds.size(); ++i) {
                if (!rs.DeepstackVisualEmbeds[i].Data || !rs.DeepstackVisualEmbeds_CPU[i].Data) {
                    continue;
                }
                const std::size_t bytes = rs.DeepstackVisualEmbeds[i].bytes();
                CUDA_CHECK(cudaMemcpyAsync(rs.DeepstackVisualEmbeds[i].Data,
                                           rs.DeepstackVisualEmbeds_CPU[i].Data,
                                           bytes,
                                           cudaMemcpyHostToDevice,
                                           rs.MainStream));
            }
        }
        record_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    }

    // Configure graphs for hooked execution
    if (hook) {
        rs.configure_forward_graphs(/*hooked=*/true);
    }

    execute_forward(B, T, comm, /*full=*/false, hook ? &hook : nullptr);

    sync_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    record_event_if_not_capturing(rs.ForwardDone, rs.MainStream);

    reduce_loss(rs, B, T, comm);
    comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, rs.MainStream);
    comm.all_reduce_sum_int(rs.CorrectCount.template get<int>(), /*n=*/1, rs.MainStream);

    CUDA_CHECK(
        cudaMemcpyAsync(rs.NormHost, rs.ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost, rs.MainStream));
    CUDA_CHECK(
        cudaMemcpyAsync(rs.AccuracyHost, rs.CorrectCount.Data, sizeof(int), cudaMemcpyDeviceToHost, rs.MainStream));
    CUDA_CHECK(cudaDeviceSynchronize());

    int valid_tokens = *reinterpret_cast<int*>(rs.NormHost);
    int correct_tokens = *reinterpret_cast<int*>(rs.AccuracyHost);
    if (valid_tokens > 0) {
        float avg_valid = static_cast<float>(valid_tokens) / static_cast<float>(std::max(1, comm.world_size()));
        *rs.LossHost /= avg_valid;
        *rs.AccuracyHost = (static_cast<float>(correct_tokens) / static_cast<float>(valid_tokens)) * 100.0f;
    } else {
        *rs.LossHost = 0.0f;
        *rs.AccuracyHost = 0.0f;
    }

    rs.temp_free(rs.non_block_activations().output);

    return *rs.LossHost;
}

void GraphExecutor::backward_with_hook(Tensor inputs,
                                       Tensor targets,
                                       NCCLCommunicator& comm,
                                       int grad_accum_steps,
                                       int micro_step,
                                       const modules::BackwardHook& hook) {
    auto& rs = mRunState;
    auto& grads = mGrads;
    const auto& config = mConfig;
    rs.GradAccumSteps = std::max(1, grad_accum_steps);
    rs.WorldSize = std::max(1, comm.world_size());

    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];
    mLastInputsCpu = inputs;

    const bool in_capture = stream_is_capturing(rs.MainStream);
    const cudaStream_t target_stream = in_capture ? rs.MainStream : rs.side_stream();
    const bool skip_target_copy =
        mHasLossOp && targets.Device == -1 && rs.Targets_CPU.Data != nullptr && rs.Targets_CPU.Data == targets.Data;
    // Copy targets to device (side stream in eager, main stream during capture).
    {
        if (!in_capture) {
            CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.BackwardDone, 0));
        }
        if (!skip_target_copy) {
            const std::size_t target_bytes =
                static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
            CUDA_CHECK(
                cudaMemcpyAsync(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice, target_stream));
            record_event_if_not_capturing(rs.TransferDone, target_stream);
        }
    }

    if (micro_step == 0) {
        const cudaStream_t grad_stream = in_capture ? rs.MainStream : rs.side_stream();
        grads.start_micro_step(grad_stream, micro_step, grad_accum_steps);
        record_event_if_not_capturing(rs.side_stream_event(), grad_stream);
    } else {
        grads.start_micro_step(rs.MainStream, micro_step, grad_accum_steps);
    }

    // Zero non-block gradient buffers
    fill_zero(rs.non_block_gradients().d_ln_final, rs.MainStream);
    if (rs.non_block_gradients().d_embeddings.Data && !rs.is_lora_only_mode()) {
        fill_zero(rs.non_block_gradients().d_embeddings, rs.MainStream);
    }
    if (config.NumLayers > 0) {
        if (Tensor* d_res_ffn = block_gradient_ptr(rs, config.NumLayers - 1, TensorSlot::BlockDResFFN)) {
            fill_zero(*d_res_ffn, rs.MainStream);
        }
    }

    // Zero all activation gradient buffers to prevent stale gradients from accumulating.
    // This is critical for FFT mode where rmsnorm_backward accumulates (+=) to dinp.
    // Without zeroing, stale gradients from previous steps can cause gradient explosion.
    rs.zero_activation_gradients(rs.MainStream);

    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRAGrads && mLoRARunState) {
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = true;
        mLoRAGrads->start_micro_step(rs.MainStream, micro_step, grad_accum_steps);
    }

    const bool last_step = micro_step == grad_accum_steps - 1;
    if (last_step) {
        reduce_loss(rs, B, T, comm);
        comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, rs.MainStream);
    }

    if (!in_capture) {
        if (!skip_target_copy) {
            CUDA_CHECK(cudaStreamWaitEvent(rs.MainStream, rs.TransferDone, 0));
        }
        if (micro_step == 0) {
            CUDA_CHECK(cudaStreamWaitEvent(rs.MainStream, rs.side_stream_event(), 0));
        }
    }

    // Configure graphs for hooked execution (may differ in topology)
    if (hook) {
        rs.configure_backward_graphs(/*hooked=*/true);
    }

    execute_backward(B, T, comm, grad_accum_steps, micro_step, hook ? &hook : nullptr, /*skip_zeroing=*/true);

    grads.end_micro_step(rs.MainStream, comm);
    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRAGrads) {
        mLoRAGrads->end_micro_step(rs.MainStream, comm);
    }

    // Start async all-reduce on last micro-step (overlaps with CPU work and optimizer prep)
    // Note: LoRA gradients are already reduced in end_micro_step() above
    if (last_step && comm.world_size() > 1) {
        // Ensure all per-layer gradient reductions on side_stream complete before
        // recording all_reduce_done_event. Layer reductions were started during backward
        // on side_stream to overlap with compute on MainStream.
        if (grads.is_overlapped_enabled()) {
            CUDA_CHECK(cudaEventRecord(rs.side_stream_event(), rs.side_stream()));
            CUDA_CHECK(cudaStreamWaitEvent(rs.MainStream, rs.side_stream_event(), 0));
        }
        grads.reduce_all_async(comm, rs.MainStream, rs.all_reduce_done_event());
    }

    record_event_if_not_capturing(rs.BackwardDone, rs.MainStream);
    sync_event_if_not_capturing(rs.TransferDone, rs.MainStream);
}

// ============================================================================
// Log-prob forward (no KV-cache, no gradients, log-probability extraction)
// ============================================================================

void GraphExecutor::execute_logprobs_forward(long B,
                                             long T,
                                             const std::int32_t* input_ids_cpu,
                                             const std::int32_t* targets_cpu,
                                             float* logprobs_cpu,
                                             const modules::ForwardHook* hook,
                                             NCCLCommunicator& comm,
                                             const std::int32_t* position_ids_cpu,
                                             const float* temperatures_cpu) {
    if (!mCompiledExecutor) {
        throw std::runtime_error("GraphExecutor: compiled executor not initialized");
    }

    compile_graphs(B, T);
    if (!mCompiledForward) {
        throw std::runtime_error("GraphExecutor: compiled forward graph not available");
    }

    DslRunState& rs = mRunState;
    const int BT = static_cast<int>(B * T);
    const std::size_t token_bytes = static_cast<std::size_t>(BT) * sizeof(std::int32_t);

    // Copy input IDs and targets to device.
    CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, input_ids_cpu, token_bytes, cudaMemcpyHostToDevice, rs.MainStream));
    CUDA_CHECK(cudaMemcpyAsync(rs.Targets.Data, targets_cpu, token_bytes, cudaMemcpyHostToDevice, rs.MainStream));

    // Copy or build position IDs.
    if (position_ids_cpu) {
        // Use caller-provided position IDs (e.g. packed sequences with resets).
        // Replicate single plane to all 3 mRoPE planes for text-only training.
        copy_position_ids_to_device(position_ids_cpu, token_bytes, rs.PositionIDs, B, T, rs.MainStream);
    } else {
        // Default: [0, 1, ..., T-1] repeated B times.
        std::vector<std::int32_t> pos_cpu(static_cast<std::size_t>(BT));
        for (long b = 0; b < B; ++b) {
            for (long t = 0; t < T; ++t) {
                pos_cpu[static_cast<std::size_t>(b * T + t)] = static_cast<std::int32_t>(t);
            }
        }
        copy_position_ids_to_device(pos_cpu.data(), token_bytes, rs.PositionIDs, B, T, rs.MainStream);
    }

    // Allocate GPU buffer for per-token log-probs.
    float* logprobs_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&logprobs_gpu, static_cast<std::size_t>(BT) * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(logprobs_gpu, 0, static_cast<std::size_t>(BT) * sizeof(float), rs.MainStream));

    // Optional per-token inverse temperature buffer.
    float* inv_temperature_gpu = nullptr;
    if (temperatures_cpu) {
        std::vector<float> inv_temp(static_cast<std::size_t>(BT));
        for (int i = 0; i < BT; ++i) {
            inv_temp[static_cast<std::size_t>(i)] = 1.0f / temperatures_cpu[i];
        }
        CUDA_CHECK(cudaMalloc(&inv_temperature_gpu, static_cast<std::size_t>(BT) * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(inv_temperature_gpu,
                                   inv_temp.data(),
                                   static_cast<std::size_t>(BT) * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   rs.MainStream));
        mCompiledExecutor->set_inv_temperature_context(inv_temperature_gpu);
    }

    // Configure logprobs context (intercepted in dispatch_fused_lm_head_loss).
    mCompiledExecutor->set_logprobs_context(logprobs_gpu);
    mCompiledExecutor->set_dimensions(B, T);

    // No activations need to survive for backward.
    static const std::vector<std::string> empty_save_list;
    mCompiledExecutor->set_save_list(&empty_save_list);

    // Run forward with optional LoRA hook (nullptr = no LoRA = reference model).
    mCompiledExecutor->execute_forward(*mCompiledForward, comm, /*full=*/false, hook);

    // Copy results back to CPU.
    CUDA_CHECK(cudaMemcpyAsync(logprobs_cpu,
                               logprobs_gpu,
                               static_cast<std::size_t>(BT) * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               rs.MainStream));
    CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));

    // Cleanup.
    CUDA_CHECK(cudaFree(logprobs_gpu));
    mCompiledExecutor->set_logprobs_context(nullptr);
    if (inv_temperature_gpu) {
        mCompiledExecutor->set_inv_temperature_context(nullptr);
        CUDA_CHECK(cudaFree(inv_temperature_gpu));
    }
    mCompiledExecutor->set_save_list(&mSaveList);
}

// ============================================================================
// Custom d_loss backward (GRPO: external per-token gradient multipliers)
// ============================================================================

void GraphExecutor::backward_with_custom_dloss(Tensor inputs,
                                               Tensor targets,
                                               const float* per_token_grads_cpu,
                                               NCCLCommunicator& comm,
                                               int grad_accum_steps,
                                               int micro_step,
                                               const modules::BackwardHook* hook,
                                               const float* temperatures_cpu) {
    auto& rs = mRunState;
    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];
    const std::size_t BT = static_cast<std::size_t>(B) * static_cast<std::size_t>(T);

    // Allocate a temporary GPU buffer and upload the custom per-token gradients.
    // These replace the standard d_loss=1.0 seeding in dispatch_fused_lm_head_loss_backward.
    float* custom_dloss_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&custom_dloss_gpu, BT * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(custom_dloss_gpu,
                               per_token_grads_cpu,
                               BT * sizeof(float),
                               cudaMemcpyHostToDevice,
                               rs.MainStream));
    mCompiledExecutor->set_custom_dloss_context(custom_dloss_gpu);

    float* inv_temperature_gpu = nullptr;
    if (temperatures_cpu) {
        std::vector<float> inv_temp(BT);
        for (std::size_t i = 0; i < BT; ++i) {
            inv_temp[i] = 1.0f / temperatures_cpu[i];
        }
        CUDA_CHECK(cudaMalloc(&inv_temperature_gpu, BT * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(inv_temperature_gpu,
                                   inv_temp.data(),
                                   BT * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   rs.MainStream));
        mCompiledExecutor->set_inv_temperature_context(inv_temperature_gpu);
    }

    // Run standard backward (with or without LoRA backward hook).
    if (hook) {
        backward_with_hook(inputs, targets, comm, grad_accum_steps, micro_step, *hook);
    } else {
        backward(inputs, targets, comm, grad_accum_steps, micro_step);
    }

    // Synchronize to ensure the D→D copy of custom_dloss_gpu→d_loss completed
    // before freeing the temporary buffer.  The copy is launched early in the
    // backward graph (dispatch_fused_lm_head_loss_backward), so this sync also
    // waits for the full backward to complete on the GPU.
    CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));
    mCompiledExecutor->set_custom_dloss_context(nullptr);
    CUDA_CHECK(cudaFree(custom_dloss_gpu));
    if (inv_temperature_gpu) {
        mCompiledExecutor->set_inv_temperature_context(nullptr);
        CUDA_CHECK(cudaFree(inv_temperature_gpu));
    }
}

// ============================================================================
// Document masking (Flash Attention varlen)
// ============================================================================

void GraphExecutor::set_doc_masking(const std::int32_t* cu_seqlens_cpu, int num_docs, int max_seqlen, int total_q) {
    const int count = num_docs + 1;
    mCuSeqlensCpu.assign(cu_seqlens_cpu, cu_seqlens_cpu + count);
    // Reallocate GPU buffer if size changed
    if (mCuSeqlensGpu && mCuSeqlensCount != count) {
        CUDA_CHECK(cudaFree(mCuSeqlensGpu));
        mCuSeqlensGpu = nullptr;
    }
    if (!mCuSeqlensGpu) {
        CUDA_CHECK(cudaMalloc(&mCuSeqlensGpu, static_cast<std::size_t>(count) * sizeof(std::int32_t)));
        mCuSeqlensCount = count;
    }
    CUDA_CHECK(cudaMemcpyAsync(mCuSeqlensGpu,
                               cu_seqlens_cpu,
                               static_cast<std::size_t>(count) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice,
                               mRunState.MainStream));
    mDocMaskingNumDocs = num_docs;
    mDocMaskingMaxSeqlen = max_seqlen;
    mDocMaskingTotalQ = total_q;
    if (mCompiledExecutor) {
        mCompiledExecutor->set_doc_masking_context(mCuSeqlensGpu, mCuSeqlensCpu.data(), num_docs, max_seqlen, total_q);
    }
}

void GraphExecutor::clear_doc_masking() {
    if (mCompiledExecutor) {
        mCompiledExecutor->clear_doc_masking_context();
    }
    mDocMaskingNumDocs = 0;
    mDocMaskingMaxSeqlen = 0;
    mDocMaskingTotalQ = 0;
    // Keep GPU buffer allocated for reuse (freed in destructor)
}

void GraphExecutor::set_inv_temperature_context(const float* inv_temperature_gpu) {
    if (mCompiledExecutor) {
        mCompiledExecutor->set_inv_temperature_context(inv_temperature_gpu);
    }
}

}  // namespace dsl
