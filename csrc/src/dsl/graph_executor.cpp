// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL Graph executor (DSL-driven).

#include "dsl/graph_executor.h"
#include "dsl/autodiff.h"
#include "dsl/compiled_ops.h"
#include "dsl/dsl_runtime.h"
#include "dsl/dsl_weight_manager.h"
#include "dsl/graph_executor_internal.h"
#include "dsl/graph_executor_tensors.h"
#include "dsl/graph_executor_utils.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include "modules/fp8_scaling_state.h"
#include "modules/fp8_scaling_config.h"
#include "modules/lora/lora_model_utils.h"
#include "modules/lora/lora_run_state.h"
#include "modules/matmul_context.h"
#include "modules/model_config.h"
#include "training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/allocator.h"
#include "utilities/dtype.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"
#include "kernels/kernels.h"
#include "recipes/nvfp4/nvfp4_recipe.h"

namespace dsl {
namespace {

/**
 * @brief Execute a callable with stack checkpoint/restore for CUDA graph compatibility.
 *
 * When graphs are enabled and use temp_alloc() inside the captured function, we must
 * ensure the stack is in the same state before each graph replay. This function:
 * - On first capture: saves a checkpoint after the function runs
 * - On replay: restores the stack to the checkpoint before launching the graph
 *
 * This ensures temp_alloc returns the same memory addresses that were captured in the graph.
 * (Same pattern as modules::detail::trace_or_execute_cuda_graph_with_stack)
 */
template<typename Function>
inline void trace_or_execute_cuda_graph_with_stack(Function&& function, cudaStream_t stream,
                                                    cudaGraphExec_t& instance, bool enabled,
                                                    DeviceMemoryStack& stack,
                                                    DeviceMemoryStack::Checkpoint& checkpoint) {
    if (!enabled) {
        function();
        return;
    }

    // Fast path: restore stack state and replay existing executable.
    if (instance != nullptr) {
        stack.restore(checkpoint);
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
        return;
    }

    // Capture path: save checkpoint before capture so we know where to restore to.
    checkpoint = stack.checkpoint();

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    function();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphLaunch(instance, stream));
}

inline bool stream_is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
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
    CUDA_CHECK(cudaMemcpyAsync(rs.LossHost, rs.Losses.template get<float>(), sizeof(float), cudaMemcpyDeviceToHost, rs.MainStream));
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

Tensor recompute_lora_rmsnorm(modules::LoRARunState& lora_rs, const Tensor& residual, const Tensor& weight,
                              float eps, int B, int T, int C, cudaStream_t stream) {
    if (!lora_rs.recompute_ln.Data || !lora_rs.recompute_rstd.Data) {
        throw std::runtime_error("DSL graph executor: LoRA recompute buffers not allocated");
    }
    rmsnorm_forward(lora_rs.recompute_ln, lora_rs.recompute_rstd,
                    residual, weight, nullptr, eps, B, T, C, stream);
    return lora_rs.recompute_ln;
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
}

void GraphExecutor::set_lora_state(const modules::ModularLoRAConfig* config,
                                   modules::ModularLoRAWeightsManager* weights,
                                   modules::ModularLoRAGradsManager* grads,
                                   modules::LoRARunState* run_state) {
    mLoRAConfig = config;
    mLoRAWeights = weights;
    mLoRAGrads = grads;
    mLoRARunState = run_state;
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
    // Reset per-layer recompute graphs
    reset_recompute_graphs();
    // Reset per-layer graphs in run state
    mRunState.reset_cuda_graphs();
}

void GraphExecutor::init(const GraphExecutorOptions& options) {
    if (!mForward) {
        throw std::runtime_error("DSL graph executor: module missing forward graph");
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
            // 1. Graph outputs (they're re-computed during backward by classifier)
            // 2. Tensors produced by ops that depend on lm_head (not available in full=false mode)
            std::unordered_set<std::string> save_set(mForward->save.begin(), mForward->save.end());
            for (const auto& s : mDerivedBackward->save) {
                save_set.insert(s);
            }
            // Remove graph outputs - they're handled by the classifier
            for (const auto& [name, _] : mForward->outputs) {
                save_set.erase(name);
            }
            // Also remove tensors that are produced by ops that come after the last save-able tensor
            // (i.e., tensors that depend on lm_head which isn't available during training forward)
            // For now, we specifically exclude "logits_flat" as it's produced by the lm_head matmul
            save_set.erase("logits_flat");
            save_set.erase("logits");
            mSaveList.assign(save_set.begin(), save_set.end());

            if (options.debug_print_backward) {
                std::cerr << "[Autodiff] Derived backward graph with "
                          << mDerivedBackward->operations.size() << " operations\n";
                for (const auto& op : mDerivedBackward->operations) {
                    std::cerr << "  " << op.kernel_type << ": [";
                    for (size_t i = 0; i < op.inputs.size(); ++i) {
                        if (i > 0) std::cerr << ", ";
                        std::cerr << op.inputs[i];
                    }
                    std::cerr << "] -> [";
                    for (size_t i = 0; i < op.outputs.size(); ++i) {
                        if (i > 0) std::cerr << ", ";
                        std::cerr << op.outputs[i];
                    }
                    std::cerr << "]\n";
                }
                std::cerr << "[Autodiff] Auto-computed saves: ";
                for (const auto& s : mSaveList) {
                    std::cerr << s << " ";
                }
                std::cerr << "\n";
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("DSL graph executor: autodiff failed: ") + e.what());
        }
    }

    mViewSources.clear();
    mViewSourcesReverse.clear();
    mEmbeddingOutputs.clear();
    if (mForward) {
        for (const auto& op : mForward->operations) {
            const std::string& op_type = op.kernel_type.empty() ? op.name : op.kernel_type;
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
        const std::string& op_type = op.kernel_type.empty() ? op.name : op.kernel_type;
        const bool is_embedding_bwd = (op_type == "embedding_backward" || op_type == "encoder_backward"
                                       || op.name == "embedding_backward" || op.name == "encoder_backward");
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
                if (auto base = base_param_from_grad(name)) {
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

    // Initialize compiled execution if enabled
    mUseCompiledExecution = options.use_compiled_execution;
    if (mUseCompiledExecution) {
        init_compiled_execution();
    }
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
    mCompiledExecutor->set_recompute_fn(
        [this](int layer_idx, long B, long T, bool use_graph) {
            recompute_layer_optimized(layer_idx, B, T, use_graph);
        });
    mCompiledExecutor->set_fp8_cache(&mFP8WeightCache);
    mCompiledExecutor->set_fp4_cache(&mFP4WeightCache, &mFP4WeightCacheT);
    mCompiledExecutor->set_saved_tensors(&mSaved);
    mCompiledExecutor->set_last_inputs_cpu(&mLastInputsCpu);
    mCompiledExecutor->set_rng_seed_fn([this]() { return next_rng_seed(); });

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
        if (mForward) {
            mCompiledForward = std::make_unique<CompiledGraph>(mCompiler->compile(*mForward, B, T));
        }
        if (mBackward) {
            mCompiledBackward = std::make_unique<CompiledGraph>(mCompiler->compile(*mBackward, B, T));
        }
        mCompiledB = B;
        mCompiledT = T;
    }
}

void GraphExecutor::execute_forward_compiled(long B, long T, NCCLCommunicator& comm, bool full,
                                              const modules::ForwardHook* hook) {
    compile_graphs(B, T);

    if (!mCompiledForward || !mCompiledExecutor) {
        throw std::runtime_error("DSL graph executor: compiled forward graph not available");
    }

    auto& rs = mRunState;
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool in_capture = (cudaStreamIsCapturing(rs.MainStream, &capture_status) == cudaSuccess &&
                             capture_status != cudaStreamCaptureStatusNone);
    const bool use_graphs = mGraphsEnabled && !in_capture;
    if (use_graphs && (mGraphB != B || mGraphT != T)) {
        reset_cuda_graphs();
        mGraphB = B;
        mGraphT = T;
    }
    const bool capturing = use_graphs && mForwardGraph == nullptr;
    if (!use_graphs || capturing) {
        mSaved.clear();
    }

    auto run_ops = [&]() {
        mCompiledExecutor->set_dimensions(B, T);
        mCompiledExecutor->set_capturing(capturing);
        mCompiledExecutor->execute_forward(*mCompiledForward, comm, full, hook);
        // Save tensors for backward (same list as non-compiled path).
        mCompiledExecutor->save_tensors(mSaveList);
    };

    trace_or_execute_cuda_graph_with_stack(run_ops, rs.MainStream, mForwardGraph, use_graphs,
                                           rs.Stack, mForwardCheckpoint);
    mCompiledExecutor->set_capturing(false);
}

void GraphExecutor::execute_backward_compiled(long B, long T, NCCLCommunicator& comm, int grad_accum_steps,
                                               int micro_step, const modules::BackwardHook* hook) {
    compile_graphs(B, T);

    if (!mCompiledBackward || !mCompiledExecutor) {
        throw std::runtime_error("DSL graph executor: compiled backward graph not available");
    }

    auto& rs = mRunState;
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool in_capture = (cudaStreamIsCapturing(rs.MainStream, &capture_status) == cudaSuccess &&
                             capture_status != cudaStreamCaptureStatusNone);
    const bool use_graphs = mBackwardGraphsEnabled && mBackwardGraphCapturable && !in_capture;
    if (use_graphs && (mGraphB != B || mGraphT != T)) {
        reset_cuda_graphs();
        mGraphB = B;
        mGraphT = T;
    }
    const int graph_idx = (micro_step > 0) ? 1 : 0;
    const bool capturing = use_graphs && mBackwardGraph[graph_idx] == nullptr;

    auto run_ops = [&]() {
        mCompiledExecutor->set_dimensions(B, T);
        init_recompute_flags();
        mCompiledExecutor->set_recompute_enabled(mRecomputeFlags.any);
        mCompiledExecutor->set_recompute_use_graphs(use_graphs && !capturing);
        mCompiledExecutor->set_capturing(capturing);
        mCompiledExecutor->execute_backward(*mCompiledBackward, comm, grad_accum_steps, micro_step, hook);
    };

    trace_or_execute_cuda_graph_with_stack(run_ops, rs.MainStream, mBackwardGraph[graph_idx], use_graphs,
                                           rs.Stack, mBackwardCheckpoint[graph_idx]);
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
    std::transform(view.begin(), view.end(), std::back_inserter(state),
                   [](char c) { return static_cast<std::byte>(c); });
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

    // Copy inputs and position ids to device.
    {
        const std::size_t input_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        const std::size_t pos_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(position_ids.DType);
        if (position_ids.Device == -1) {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyDeviceToDevice, rs.MainStream));
        }
        record_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    }

    if (mUseCompiledExecution) {
        execute_forward_compiled(B, T, comm, /*full=*/false, nullptr);
    } else {
        execute_forward_graph(B, T, comm, /*full=*/false);
    }

    sync_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    record_event_if_not_capturing(rs.ForwardDone, rs.MainStream);
}

float GraphExecutor::validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) {
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

    // Copy inputs and position ids to device.
    {
        const std::size_t input_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        const std::size_t pos_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(position_ids.DType);
        if (position_ids.Device == -1) {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyDeviceToDevice, rs.MainStream));
        }
        record_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    }

    if (mUseCompiledExecution) {
        execute_forward_compiled(B, T, comm, /*full=*/false, nullptr);
    } else {
        execute_forward_graph(B, T, comm, /*full=*/false);
    }

    sync_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    record_event_if_not_capturing(rs.ForwardDone, rs.MainStream);

    fill_zero(rs.Losses, rs.MainStream);
    fill_zero(rs.ValidTokenCount, rs.MainStream);
    fill_zero(rs.CorrectCount, rs.MainStream);

    const std::size_t target_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
    if (targets.Device == -1) {
        CUDA_CHECK(cudaMemcpy(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyDeviceToDevice));
    }

    run_classifier(B, T, comm, /*grad_accum_steps=*/1, micro_step, /*compute_accuracy=*/true);

    reduce_loss(rs, B, T, comm);
    comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, rs.MainStream);
    comm.all_reduce_sum_int(rs.CorrectCount.template get<int>(), /*n=*/1, rs.MainStream);

    CUDA_CHECK(cudaMemcpyAsync(rs.NormHost, rs.ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost, rs.MainStream));
    CUDA_CHECK(cudaMemcpyAsync(rs.AccuracyHost, rs.CorrectCount.Data, sizeof(int), cudaMemcpyDeviceToHost, rs.MainStream));
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

void GraphExecutor::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) {
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
    // Copy targets to device (side stream in eager, main stream during capture).
    {
        if (!in_capture) {
            CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.BackwardDone, 0));
        }
        const std::size_t target_bytes =
            static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice, target_stream));
        record_event_if_not_capturing(rs.TransferDone, target_stream);
    }

    if (micro_step == 0) {
        fill_zero(rs.Losses, rs.MainStream);
        fill_zero(rs.ValidTokenCount, rs.MainStream);
        const cudaStream_t grad_stream = in_capture ? rs.MainStream : rs.side_stream();
        grads.start_micro_step(grad_stream, micro_step, grad_accum_steps);
        record_event_if_not_capturing(rs.side_stream_event(), grad_stream);
    } else {
        grads.start_micro_step(rs.MainStream, micro_step, grad_accum_steps);
    }

    // Zero d_ln_final.
    fill_zero(rs.non_block_gradients().d_ln_final, rs.MainStream);
    if (config.NumLayers > 0) {
        fill_zero(rs.simplified_grads(config.NumLayers - 1).d_res_ffn, rs.MainStream);
    }

    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRAGrads && mLoRARunState) {
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = true;
        mLoRAGrads->start_micro_step(rs.MainStream, micro_step, grad_accum_steps);
    }

    run_classifier(B, T, comm, grad_accum_steps, micro_step, /*compute_accuracy=*/false);

    const bool last_step = micro_step == grad_accum_steps - 1;
    if (last_step) {
        reduce_loss(rs, B, T, comm);
        comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, rs.MainStream);
    }

    if (mUseCompiledExecution) {
        execute_backward_compiled(B, T, comm, grad_accum_steps, micro_step, nullptr);
    } else {
        execute_backward_graph(B, T, comm, grad_accum_steps, micro_step);
    }

    grads.end_micro_step(rs.MainStream, comm);
    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRAGrads) {
        mLoRAGrads->end_micro_step(rs.MainStream, comm);
    }

    // Start async all-reduce on last micro-step (overlaps with CPU work and optimizer prep)
    // Note: LoRA gradients are already reduced in end_micro_step() above
    if (last_step && comm.world_size() > 1) {
        grads.reduce_all_async(comm, rs.MainStream, rs.all_reduce_done_event());
    }

    record_event_if_not_capturing(rs.BackwardDone, rs.MainStream);
    sync_event_if_not_capturing(rs.TransferDone, rs.MainStream);
}

void GraphExecutor::forward_with_hook(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step,
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

    // Copy inputs and position ids to device.
    {
        const std::size_t input_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        const std::size_t pos_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(position_ids.DType);
        if (position_ids.Device == -1) {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyDeviceToDevice, rs.MainStream));
        }
        record_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    }

    // Configure graphs for hooked execution (may differ in topology)
    if (hook) {
        rs.configure_forward_graphs(/*hooked=*/true);
    }

    if (mUseCompiledExecution) {
        execute_forward_compiled(B, T, comm, /*full=*/false, hook ? &hook : nullptr);
    } else {
        execute_forward_graph(B, T, comm, /*full=*/false, hook ? &hook : nullptr);
    }

    sync_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    record_event_if_not_capturing(rs.ForwardDone, rs.MainStream);
}

float GraphExecutor::validate_with_hook(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm,
                                        int micro_step, const modules::ForwardHook& hook) {
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

    // Copy inputs and position ids to device.
    {
        const std::size_t input_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        const std::size_t pos_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(position_ids.DType);
        if (position_ids.Device == -1) {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyDeviceToDevice, rs.MainStream));
        }
        record_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    }

    // Configure graphs for hooked execution
    if (hook) {
        rs.configure_forward_graphs(/*hooked=*/true);
    }

    if (mUseCompiledExecution) {
        execute_forward_compiled(B, T, comm, /*full=*/false, hook ? &hook : nullptr);
    } else {
        execute_forward_graph(B, T, comm, /*full=*/false, hook ? &hook : nullptr);
    }

    sync_event_if_not_capturing(rs.TransferDone, rs.MainStream);
    record_event_if_not_capturing(rs.ForwardDone, rs.MainStream);

    fill_zero(rs.Losses, rs.MainStream);
    fill_zero(rs.ValidTokenCount, rs.MainStream);
    fill_zero(rs.CorrectCount, rs.MainStream);

    const std::size_t target_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
    if (targets.Device == -1) {
        CUDA_CHECK(cudaMemcpy(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyDeviceToDevice));
    }

    run_classifier(B, T, comm, /*grad_accum_steps=*/1, micro_step, /*compute_accuracy=*/true);

    reduce_loss(rs, B, T, comm);
    comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, rs.MainStream);
    comm.all_reduce_sum_int(rs.CorrectCount.template get<int>(), /*n=*/1, rs.MainStream);

    CUDA_CHECK(cudaMemcpyAsync(rs.NormHost, rs.ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost, rs.MainStream));
    CUDA_CHECK(cudaMemcpyAsync(rs.AccuracyHost, rs.CorrectCount.Data, sizeof(int), cudaMemcpyDeviceToHost, rs.MainStream));
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

void GraphExecutor::backward_with_hook(Tensor inputs, Tensor targets, NCCLCommunicator& comm,
                                       int grad_accum_steps, int micro_step, const modules::BackwardHook& hook) {
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
    // Copy targets to device (side stream in eager, main stream during capture).
    {
        if (!in_capture) {
            CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.BackwardDone, 0));
        }
        const std::size_t target_bytes =
            static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice, target_stream));
        record_event_if_not_capturing(rs.TransferDone, target_stream);
    }

    if (micro_step == 0) {
        fill_zero(rs.Losses, rs.MainStream);
        fill_zero(rs.ValidTokenCount, rs.MainStream);
        const cudaStream_t grad_stream = in_capture ? rs.MainStream : rs.side_stream();
        grads.start_micro_step(grad_stream, micro_step, grad_accum_steps);
        record_event_if_not_capturing(rs.side_stream_event(), grad_stream);
    } else {
        grads.start_micro_step(rs.MainStream, micro_step, grad_accum_steps);
    }

    // Zero d_ln_final.
    fill_zero(rs.non_block_gradients().d_ln_final, rs.MainStream);
    if (config.NumLayers > 0) {
        fill_zero(rs.simplified_grads(config.NumLayers - 1).d_res_ffn, rs.MainStream);
    }

    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRAGrads && mLoRARunState) {
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = true;
        mLoRAGrads->start_micro_step(rs.MainStream, micro_step, grad_accum_steps);
    }

    run_classifier(B, T, comm, grad_accum_steps, micro_step, /*compute_accuracy=*/false);

    const bool last_step = micro_step == grad_accum_steps - 1;
    if (last_step) {
        reduce_loss(rs, B, T, comm);
        comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, rs.MainStream);
    }

    // Configure graphs for hooked execution (may differ in topology)
    if (hook) {
        rs.configure_backward_graphs(/*hooked=*/true);
    }

    if (mUseCompiledExecution) {
        execute_backward_compiled(B, T, comm, grad_accum_steps, micro_step, hook ? &hook : nullptr);
    } else {
        execute_backward_graph(B, T, comm, grad_accum_steps, micro_step, hook ? &hook : nullptr);
    }

    grads.end_micro_step(rs.MainStream, comm);
    if (mLoRAConfig && mLoRAConfig->enabled() && mLoRAGrads) {
        mLoRAGrads->end_micro_step(rs.MainStream, comm);
    }

    // Start async all-reduce on last micro-step (overlaps with CPU work and optimizer prep)
    // Note: LoRA gradients are already reduced in end_micro_step() above
    if (last_step && comm.world_size() > 1) {
        grads.reduce_all_async(comm, rs.MainStream, rs.all_reduce_done_event());
    }

    record_event_if_not_capturing(rs.BackwardDone, rs.MainStream);
    sync_event_if_not_capturing(rs.TransferDone, rs.MainStream);
}

void GraphExecutor::run_classifier(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step, bool compute_accuracy) {
    auto& rs = mRunState;
    const auto& config = mConfig;
    (void)comm;
    if (mOptions.LMHeadChunks != 1) {
        throw std::runtime_error("DSL graph executor: lmhead_chunks > 1 not supported yet");
    }

    const size_t V = config.VocabSize;
    const size_t Vp = config.VocabSize;
    const float d_loss = compute_accuracy ? 1.0f : (1.0f / static_cast<float>(B * T * grad_accum_steps));

    rs.temp_acquire(rs.non_block_activations().output);

    const bool in_capture = stream_is_capturing(rs.MainStream);
    // Ensure targets and gradient zeroing are visible on main stream.
    if (!in_capture) {
        CUDA_CHECK(cudaStreamWaitEvent(rs.MainStream, rs.TransferDone, 0));
        if (!compute_accuracy && micro_step == 0) {
            CUDA_CHECK(cudaStreamWaitEvent(rs.MainStream, rs.side_stream_event(), 0));
        }
    }

    Tensor lnf_flat = view_tensor(rs.non_block_activations().ln_final, {B * T, config.HiddenSize});
    Tensor logits = rs.non_block_activations().output;

    // Row-major logits = lnf @ lm_head.T -> map to column-major backend by swapping A/B,
    // swapping M/N, and swapping transpose flags.
    Tensor& lm_head = mWeights.has("lm_head")
        ? mWeights.get("lm_head")
        : mWeights.get("lm_head_weight");

    matmul(logits, lm_head, lnf_flat,
           std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace,
           static_cast<int>(V), static_cast<int>(B * T), static_cast<int>(config.HiddenSize),
           swap_transpose(EMMTranspose::NT), false, rs.MainStream);

    Tensor tgt = rs.Targets;
    tgt.Sizes[0] = static_cast<long>(B * T);
    tgt.Rank = 1;

    Tensor losses = rs.Losses;
    losses.Sizes[0] = static_cast<long>(B * T);
    losses.Rank = 1;

    if (compute_accuracy) {
        fused_classifier(logits, losses, d_loss, tgt,
                         &rs.ValidTokenCount, &rs.CorrectCount,
                         static_cast<int>(B * T), static_cast<int>(V), static_cast<int>(Vp), true, rs.MainStream);
    } else {
        fused_classifier(logits, losses, d_loss, tgt,
                         &rs.ValidTokenCount,
                         static_cast<int>(B * T), static_cast<int>(V), static_cast<int>(Vp), true, rs.MainStream);
    }
}


}  // namespace dsl
