// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL run state - activation buffers, scratch, and CUDA graph management.

#ifndef SUROGATE_SRC_DSL_DSL_RUN_STATE_H
#define SUROGATE_SRC_DSL_DSL_RUN_STATE_H

#include <array>
#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "runtime/dsl/buffer_plan.h"
#include "runtime/dsl/dsl_runtime_config.h"
#include "runtime/dsl/tensor_slot_registry.h"
#include "utilities/stack.h"
#include "runtime/core/run_state_types.h"
#include "runtime/core/residual_manager.h"
#include "runtime/training/model.h"
#include "runtime/training/runtime_options.h"
#include "utilities/allocator.h"
namespace modules {
class FP8ScalingState;
}  // namespace modules

namespace dsl {

class CompiledExecutor;  // forward decl — see runtime/executor/compiled_ops.h
struct CompiledGraph;    // forward decl — see runtime/dsl/graph_compiler.h
struct PhaseArenas;      // forward decl — see runtime/dsl/graph_compiler.h

// DSL run state for graph execution (activation buffers, scratch, etc).
class DslRunState final : public IRunState {
public:
    /// Caller must size the stack via `dsl::required_stack_bytes(plan, ...)`;
    /// this fallback only exists as a last-resort default and is never used
    /// by the production allocation path (see `DslModel::allocate_run_state`).
    static constexpr std::size_t kDefaultStackBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // 2 GiB

    DslRunState(const PretrainedConfig& config,
                const DslRuntimeConfig& runtime_config,
                const RuntimeOptions& options,
                int B,
                int T,
                const std::shared_ptr<TensorAllocator>& allocator,
                bool lora_only_mode = false,
                bool prequantized = false,
                std::size_t stack_bytes = kDefaultStackBytes,
                const ActivationLayoutIR* activation_layout = nullptr);
    ~DslRunState();

    /// Swap the backing stack buffer (used to resize the stack after the
    /// backward graph is compiled and peak-modelled more accurately).
    void set_stack_buffer(Tensor buffer, const DeviceMemoryStack::AllocationList& high_mark = {});

    /// Redirect Stack to an externally-owned device buffer. Does not take
    /// ownership — caller must ensure the buffer outlives this DslRunState
    /// or call unbind_external_stack() first. mStackBuffer is unchanged,
    /// so unbinding restores the original backing. Stack must be empty
    /// (no live allocations) at call time.
    void rebase_stack_to_external(std::byte* ptr, std::size_t bytes);

    /// Restore Stack to the previously-owned mStackBuffer. Inverse of
    /// rebase_stack_to_external. Stack must be empty. Fails if the original
    /// was freed via free_allocator_stack_buffer.
    void unbind_external_stack();

    /// Free the TensorAllocator-owned Stack buffer allocated at construction.
    /// After this call, unbind_external_stack is invalid — Stack must stay on
    /// the external buffer for the rest of this DslRunState's lifetime.
    /// Typically called right after rebase_stack_to_external + adopt_external_stack.
    void free_allocator_stack_buffer();

    /// Take ownership of an externally-rebased Stack buffer. The buffer is
    /// cudaFree'd in ~DslRunState, outliving any GraphExecutor that handed it
    /// over. Only call after rebase_stack_to_external.
    void adopt_external_stack(std::byte* ptr, std::size_t bytes);

    /// Reallocate the DSL stack at `new_size_bytes`, freeing the old buffer
    /// first so VRAM is actually reclaimed (otherwise `TensorAllocator`
    /// retains the old allocation until its own destructor runs).
    void resize_stack_to(long new_size_bytes);

    /// Shrink the stack to `Stack.max_utilization() + safety_bytes` if the
    /// savings exceed `min_savings_bytes`. Intended to be called *once* after
    /// the first full forward+backward completes, to recover headroom left by
    /// the upfront heuristic (which must be conservative — see
    /// `dsl::required_stack_bytes`).
    ///
    /// Only shrinks; never grows. The caller must ensure the stack is empty
    /// at the point of this call (post-backward is a natural point — the
    /// DSL frees everything when a step ends).
    ///
    /// @returns the new stack size in bytes if a resize happened, 0 otherwise.
    long shrink_stack_to_high_water_mark(long safety_bytes = 64L * 1024 * 1024,
                                         long min_savings_bytes = 128L * 1024 * 1024);

    /// `block_activation_ptr` / `block_gradient_ptr` consult this back-ref
    /// to route slot lookups through the executor's tid-keyed mTensors
    /// cache. Set by the executor at `execute_forward` /
    /// `execute_backward` / `replay_layer_forward` entry, cleared at exit.
    /// Returns nullptr when no executor is active (e.g., pre-execute
    /// paths, model init).
    void set_active_executor(CompiledExecutor* exec) {
        mActiveExecutor = exec;
    }
    CompiledExecutor* active_executor() const {
        return mActiveExecutor;
    }
    /// Delegate helper: returns `exec->executor_tid_slot(layer_idx, slot)`
    /// when an executor is active, nullptr otherwise. Implemented in
    /// dsl_run_state.cpp to keep this header free of compiled_ops.h.
    Tensor* active_executor_slot(int layer_idx, TensorSlot slot);
    modules::SimplifiedQuantGradients& simplified_quant_grads() {
        return mSimplifiedQuantGrads;
    }
    modules::FP8ForwardQuantActivations& fp8_forward_quants() {
        return mFP8ForwardQuants;
    }

    /// @brief FP8 forward buffer freshness tracking.
    /// When an activation dispatch (swiglu, rmsnorm) pre-quantizes its output
    /// into the FP8 buffer, it sets the corresponding bit. The matmul dispatch
    /// checks and clears the bit to skip redundant quantization.
    enum FP8BufferReady : uint8_t {
        FP8Ready_None = 0,
        FP8Ready_LN1 = 1 << 0,     ///< fp8_forward_quants().ln1 is pre-quantized
        FP8Ready_LN2 = 1 << 1,     ///< fp8_forward_quants().ln2 is pre-quantized
        FP8Ready_Att = 1 << 2,     ///< fp8_forward_quants().att is pre-quantized
        FP8Ready_SwiGLU = 1 << 3,  ///< fp8_forward_quants().swiglu is pre-quantized
    };
    void set_fp8_buffer_ready(FP8BufferReady flag) {
        mFP8BufferReadyFlags |= flag;
    }
    bool consume_fp8_buffer_ready(FP8BufferReady flag) {
        if (mFP8BufferReadyFlags & flag) {
            mFP8BufferReadyFlags &= ~flag;
            return true;
        }
        return false;
    }
    [[nodiscard]] bool is_fp8_buffer_ready(FP8BufferReady flag) const {
        return (mFP8BufferReadyFlags & flag) != 0;
    }
    void reset_fp8_buffer_ready() {
        mFP8BufferReadyFlags = FP8Ready_None;
    }

    /// @brief Zero all activation gradient buffers (d_res_ffn, d_res_att) for all layers.
    /// Call this at the start of each backward pass to prevent stale gradients from accumulating.
    void zero_activation_gradients(cudaStream_t stream);

    /// Set the (ptr, bytes) list of activation-gradient buffers to zero
    /// at bwd entry. Called by CompiledExecutor::populate_bwd_stack_
    /// bindings once per compile; the device-side arrays are re-used
    /// across steps. No-op for count=0.
    void set_activation_grad_zero_list(const std::vector<std::uint64_t>& ptrs, const std::vector<std::uint64_t>& sizes);

    modules::NonBlockActivations& non_block_activations() {
        return mNonBlockActivations;
    }
    modules::NonBlockGradientBuffers& non_block_gradients() {
        return mNonBlockGradients;
    }
    modules::ScratchBuffers& scratch() {
        return mScratch;
    }
    Tensor& rope_freqs(std::string_view name);
    const Tensor& rope_freqs(std::string_view name) const;

    /// Move every non-block activation / gradient / rope buffer whose tid is
    /// Persistent-region into the pre-sized Persistent arena. Mirrors the
    /// weight-rebind pattern in DslParamStore: copy device-resident bytes to
    /// `arenas.persistent_ptr + meta.offset`, free the allocator-owned
    /// buffer, and repoint the Tensor at the arena slot. Called on every
    /// `compile_graphs` recompile (not gated) so rebind re-fires if the
    /// Persistent arena is re-allocated under a new (B,T).
    void
    rebind_non_block_to_persistent_arena(const CompiledGraph& graph, const PhaseArenas& arenas, cudaStream_t stream);

    /// Bytes needed for non-graph persistent buffers that don't have a tid
    /// in the compiled graph (today: the `output` logits scratch used by
    /// `fused_lm_head_loss`). Used by `GraphExecutor` to grow the Persistent
    /// arena beyond the `ForwardParam` / wm / lora slabs.
    std::size_t non_graph_persistent_extras_bytes() const;

    /// Rebind non-graph-tid persistent buffers (`output`) into the arena
    /// slab at `base`. Bump-allocated in a fixed order; caller must have
    /// reserved exactly `non_graph_persistent_extras_bytes()`.
    void rebind_non_graph_persistent_to_arena(std::byte* base, std::size_t bytes, cudaStream_t stream);

    Tensor& get_residual(int layer_idx, cudaStream_t stream);
    Tensor& get_final_residual();

    // Residual offloading support (for large model training)
    void fetch_residual(int layer_idx, cudaStream_t stream);
    void put_residual(int layer_idx, cudaStream_t stream);
    void mark_residual_ready(int layer_idx, cudaStream_t stream);
    void release_residual(int layer_idx, cudaStream_t stream);
    bool has_residual_offloading() const {
        return mOffloadResiduals;
    }

    /// Immutable buffer plan built at construction. Exposed for stack-sizing
    /// helpers (`dsl::required_stack_bytes`, `dsl::graph_backward_stack_peak`)
    /// and for tests that need to inspect the allocation decisions.
    const BufferPlan& buffer_plan() const {
        return mBufferPlan;
    }

    bool ffn_temps_on_stack() const {
        return mBufferPlan.ffn_temps_on_stack;
    }
    bool large_bwd_temps_on_stack() const {
        return mBufferPlan.large_bwd_temps_on_stack;
    }
    bool is_lora_only_mode() const {
        return mLoraOnlyMode;
    }
    bool has_hybrid_blocks() const {
        return mRuntimeConfig.has_per_layer_dims();
    }
    bool is_prequantized() const {
        return mPrequantized;
    }

    /// @brief Get the recompute level
    RecomputeLevel recompute_level() const {
        return mRecomputeLevel;
    }

    /// @brief Check if any recomputation is enabled
    bool recompute_enabled() const {
        return mRecomputeLevel != RecomputeLevel::None;
    }

    cudaStream_t side_stream() const {
        return mSideStream;
    }
    cudaEvent_t side_stream_event() const {
        return mSideStreamEvent;
    }
    cudaEvent_t all_reduce_done_event() const {
        return mAllReduceDone;
    }
    cublasHandle_t cublas_handle() const {
        return mCublasHandle;
    }

    // ========================================================================
    // Per-layer CUDA graphs (for efficient layer-by-layer execution)
    // ========================================================================

    /// @brief Get forward CUDA graph exec for a layer (nullptr if not captured)
    cudaGraphExec_t& forward_block_graph(int layer_idx) {
        return mForwardBlockGraphs.at(static_cast<std::size_t>(layer_idx));
    }

    /// @brief Get backward CUDA graph exec for a layer
    /// @param layer_idx Layer index
    /// @param accumulate If true, returns graph for gradient accumulation mode
    cudaGraphExec_t& backward_block_graph(int layer_idx, bool accumulate) {
        return mBackwardBlockGraphs.at(static_cast<std::size_t>(layer_idx))[accumulate ? 1 : 0];
    }

    /// @brief Get stack checkpoint for forward graph replay (ensures consistent temp addresses)
    DeviceMemoryStack::Checkpoint& forward_block_stack_checkpoint(int layer_idx) {
        return mForwardBlockStackCheckpoints.at(static_cast<std::size_t>(layer_idx));
    }

    /// @brief Get stack checkpoint for backward graph replay
    DeviceMemoryStack::Checkpoint& backward_block_stack_checkpoint(int layer_idx, bool accumulate) {
        return mBackwardBlockStackCheckpoints.at(static_cast<std::size_t>(layer_idx))[accumulate ? 1 : 0];
    }

    /// @brief Reconfigure forward graphs when hook presence changes
    /// Destroys existing graphs to force re-capture with new topology
    void configure_forward_graphs(bool hooked);

    /// @brief Reconfigure backward graphs when hook presence changes
    void configure_backward_graphs(bool hooked);

    /// @brief Reset all CUDA graphs (call when batch/sequence dimensions change)
    void reset_cuda_graphs();

    /// @brief Check if per-layer CUDA graphs are enabled
    bool per_layer_graphs_enabled() const {
        return mPerLayerGraphsEnabled;
    }

    /// @brief Enable/disable per-layer CUDA graphs
    void set_per_layer_graphs_enabled(bool enabled) {
        mPerLayerGraphsEnabled = enabled;
    }

    /// @brief Get number of layers (for graph array sizing)
    int num_layers() const {
        return mNumLayers;
    }

    // IRunState overrides (quantization unsupported in DSL runtime for now).
    [[nodiscard]] bool has_activation_quants() const override {
        return mMatmulDtype != mActivationDtype;
    }
    [[nodiscard]] bool has_grad_quants() const override {
        return mGradQuantDtype != mGradDtype;
    }
    [[nodiscard]] bool has_fp8_forward() const override {
        return mEnableFp8Forward;
    }
    [[nodiscard]] bool has_fp8_hybrid_backward() const override {
        return mEnableFp8Forward && mGradQuantDtype == ETensorDType::FP8_E5M2;
    }
    [[nodiscard]] bool has_fp8_delayed_scaling() const override {
        return mFP8ScalingState != nullptr;
    }
    [[nodiscard]] bool has_fp4_forward() const override {
        return false;
    }
    [[nodiscard]] bool has_fp4_backward() const override {
        return false;
    }
    [[nodiscard]] Tensor* get_fp8_forward_buffer(int op) override;
    [[nodiscard]] Tensor* get_gradient_quant_buffer(int op) override;
    [[nodiscard]] modules::FP8ScalingState* get_fp8_scaling_state() override {
        return mFP8ScalingState.get();
    }

    // MoE stats overrides
    [[nodiscard]] MoEStats get_moe_stats() const override;
    void reset_moe_stats() override;
    [[nodiscard]] bool is_moe_model() const override {
        return mNumMoEExperts > 0;
    }

    /// @brief Get device pointer to MoE stats buffer for kernel accumulation.
    /// Layout: [aux_loss_sum, z_loss_sum, utilization_sum, load_imbalance_sum, layer_count]
    float* moe_stats_device() {
        return mMoEStatsDevice;
    }

    /// @brief Set MoE config and allocate stats buffers (call once after construction)
    void set_moe_config(int num_experts, float aux_loss_coef);
    [[nodiscard]] int moe_num_experts() const {
        return mNumMoEExperts;
    }
    [[nodiscard]] float moe_aux_loss_coef() const {
        return mMoEAuxLossCoef;
    }

private:
    CompiledExecutor* mActiveExecutor = nullptr;  // unowned back-ref

    void allocate_non_block_state(const PretrainedConfig& cfg);
    void allocate_simplified_quant_buffers(const PretrainedConfig& cfg, const RuntimeOptions& options);
    void allocate_scratch_buffers(const PretrainedConfig& cfg);
    void allocate_residual_buffers(const PretrainedConfig& cfg, bool offload_residuals);
    void create_cuda_resources();
    void release_cuda_resources() noexcept;

    std::shared_ptr<TensorAllocator> mAllocator;
    DslRuntimeConfig mRuntimeConfig;
    Tensor mStackBuffer{};

    // If non-null, DslRunState owns this buffer and cudaFree's it in ~dtor.
    // Set by adopt_external_stack when GraphExecutor transfers the Stack arena.
    std::byte* mOwnedExternalStack = nullptr;
    std::size_t mOwnedExternalStackBytes = 0;
    RecomputeLevel mRecomputeLevel = RecomputeLevel::Enabled;
    bool mLoraOnlyMode = false;
    bool mPrequantized = false;
    bool mCpuTraining = false;
    ETensorDType mActivationDtype = ETensorDType::BF16;
    ETensorDType mGradDtype = ETensorDType::BF16;
    ETensorDType mMatmulDtype = ETensorDType::BF16;
    ETensorDType mGradQuantDtype = ETensorDType::BF16;
    bool mEnableFp8Forward = false;
    bool mOffloadResiduals = false;
    int mLMHeadChunks = 1;
    int mAttnBwdChunks = 1;

    modules::NonBlockActivations mNonBlockActivations;
    modules::NonBlockGradientBuffers mNonBlockGradients;
    modules::ScratchBuffers mScratch;
    TensorSlotRegistry mSlotRegistry;
    /// Compile-time buffer-sharing plan. Built once in the constructor (after
    /// the slot registry is initialized from the DSL layout) and consumed by
    /// allocate_simplified_{activations,gradients}. See buffer_plan.h.
    BufferPlan mBufferPlan;
    std::vector<Tensor> mPerLayerRopeFreqs;

    // Precomputed list of activation-gradient buffers to zero at the start of backward.
    // Stored as device arrays of (ptr, bytes) to zero in a single kernel launch.
    Tensor mActGradZeroPtrs{};
    Tensor mActGradZeroSizes{};
    int mActGradZeroCount = 0;

    modules::SimplifiedQuantGradients mSimplifiedQuantGrads;
    modules::FP8ForwardQuantActivations mFP8ForwardQuants;
    uint8_t mFP8BufferReadyFlags = 0;
    Tensor mFP8ForwardStats{};
    Tensor mGradQuantStats{};

    std::unique_ptr<modules::FP8ScalingState> mFP8ScalingState;

    std::unique_ptr<modules::ResidualManager> mResidualManager;

    // CUDA resources
    cudaStream_t mSideStream = nullptr;
    cudaEvent_t mSideStreamEvent = nullptr;
    cudaEvent_t mAllReduceDone = nullptr;    ///< Recorded after async all-reduce completes
    cublasHandle_t mCublasHandle = nullptr;  ///< cuBLAS handle for MoE grouped GEMM

    // Per-layer CUDA graph support
    int mNumLayers = 0;
    bool mPerLayerGraphsEnabled = false;
    bool mForwardGraphsHooked = false;
    bool mBackwardGraphsHooked = false;

    // Per-layer CUDA graph executables
    std::vector<cudaGraphExec_t> mForwardBlockGraphs;
    // Two backward graphs per layer: [0] for accumulate=false (first micro-step),
    // [1] for accumulate=true (subsequent micro-steps with gradient accumulation)
    std::vector<std::array<cudaGraphExec_t, 2>> mBackwardBlockGraphs;

    // Stack checkpoints for CUDA graph compatibility
    // When graphs use temp_alloc, we must restore the stack to the same state
    // before each graph replay so allocations return consistent addresses
    std::vector<DeviceMemoryStack::Checkpoint> mForwardBlockStackCheckpoints;
    std::vector<std::array<DeviceMemoryStack::Checkpoint, 2>> mBackwardBlockStackCheckpoints;

    // Helper to destroy graph arrays
    void destroy_cuda_graphs() noexcept;
    void allocate_graph_arrays(int num_layers);

    // MoE routing stats accumulation buffer (GPU)
    // Layout: [aux_loss_sum, z_loss_sum, utilization_sum, load_imbalance_sum, layer_count]
    static constexpr int kMoEStatsSize = 5;
    float* mMoEStatsDevice = nullptr;  ///< Device buffer for kernel accumulation
    float* mMoEStatsHost = nullptr;    ///< Pinned host buffer for readback
    int mNumMoEExperts = 0;            ///< 0 = not MoE
    float mMoEAuxLossCoef = 0.01f;     ///< Auxiliary loss coefficient
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_DSL_RUN_STATE_H
