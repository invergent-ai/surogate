// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "model.h"

#include <cmath>

#include "utilities/allocator.h"
#include "kernels/kernels.h"

cudnnHandle_t create_cudnn_handle();
cublasLtHandle_t create_cublaslt_handle();

float IModel::get_loss() const {
    return get_run_state().get_loss();
}
float IModel::get_norm() const {
    return get_run_state().get_norm();
}
float IModel::get_accuracy() const {
    return get_run_state().get_accuracy();
}

void IModel::update_with_config(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) {
    // Default implementation: dispatch to AdamW update
    // Derived classes can override to support NorMuon and other optimizers
    update(comm, config.learning_rate, config.adamw_beta1, config.adamw_beta2,
           step, config.adamw_epsilon, config.weight_decay, config.grad_clip);
}

Tensor& IModel::get_input_buffer() {
    return get_run_state().Inputs_CPU;
}

Tensor& IModel::get_target_buffer() {
    return get_run_state().Targets_CPU;
}

Tensor& IModel::get_position_ids_buffer() {
    return get_run_state().PositionIDs_CPU;
}


IRunState::IRunState(std::unique_ptr<PretrainedConfig> config, long batch_size, long seq_len, std::shared_ptr<TensorAllocator> alloc) : Config(std::move(config)), B(batch_size), T(seq_len), Allocator(std::move(alloc)) {
    int did;
    CUDA_CHECK(cudaGetDevice(&did));
    DeviceId = did;
    CUDA_CHECK(cudaGetDeviceProperties(&DeviceProp, did));

    Inputs = Allocator->allocate(ETensorDType::INT32, "inputs", {B, T});
    PositionIDs = Allocator->allocate(ETensorDType::INT32, "pos_ids", {B, T});
    Targets = Allocator->allocate(ETensorDType::INT32, "targets", {B, T});
    Inputs_CPU = Allocator->allocate(ETensorDType::INT32, "inputs_cpu", EAllocationType::PINNED, {B, T});
    PositionIDs_CPU = Allocator->allocate(ETensorDType::INT32, "pos_ids_cpu", EAllocationType::PINNED, {B, T});
    Targets_CPU = Allocator->allocate(ETensorDType::INT32, "targets_cpu", EAllocationType::PINNED, {B, T});
    Losses = Allocator->allocate(ETensorDType::FP32, "losses", {B, T});
    ValidTokenCount = Allocator->allocate(ETensorDType::INT32, "valid_token_count", {1});
    CorrectCount = Allocator->allocate(ETensorDType::INT32, "correct_count", {1});

    CudnnHandle = create_cudnn_handle();
    CublasLtHandle = create_cublaslt_handle();
    // Ensure fallback cuBLAS handle exists before any CUDA graph capture.
    init_cublas_fallback_handle();

    // https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace
    // recommended workspace size 32MB for sm_90+
    // SM120+ (Blackwell) FP8 operations may require larger workspace
    std::size_t cublas_ws_size = 32*1024*1024;  // 32MB default
    if (DeviceProp.major >= 12) {
        cublas_ws_size = 128*1024*1024;  // 128MB for Blackwell FP8
    }
    CuBlasWorkspace = Allocator->allocate(ETensorDType::BYTE, "cublas_ws", {static_cast<long>(cublas_ws_size)});

    // CUTLASS workspace for SM120 MX FP8 operations
    // Size includes: FP8 A buffer, FP8 B buffer, MX scales, GEMM workspace
    // 128MB is sufficient for typical LLM dimensions (B*T=4096, hidden=4096, intermediate=11008)
    if (DeviceProp.major >= 12) {  // SM120+ (Blackwell GeForce)
        CutlassWorkspace = Allocator->allocate(ETensorDType::BYTE, "cutlass_ws", {128*1024*1024});
    }

    MainStream = create_named_stream("main stream");

    ForwardDone = create_named_event("forward_done");
    BackwardDone = create_named_event("backward_done");
    TransferDone = create_named_event("transfer_done");
    NormDone = create_named_event("norm_done");
    OptimizerDone = create_named_event("optimizer_done");

    // Prime events so first-step stream waits (e.g. side-stream prefetch waiting on OptimizerDone)
    // don't deadlock when no prior work has recorded them yet.
    CUDA_CHECK(cudaEventRecord(ForwardDone, MainStream));
    CUDA_CHECK(cudaEventRecord(BackwardDone, MainStream));
    CUDA_CHECK(cudaEventRecord(TransferDone, MainStream));
    CUDA_CHECK(cudaEventRecord(NormDone, MainStream));
    CUDA_CHECK(cudaEventRecord(OptimizerDone, MainStream));
    CUDA_CHECK(cudaStreamSynchronize(MainStream));

    Tensor host_buffer = Allocator->allocate(ETensorDType::FP32, "host_buffer", EAllocationType::PINNED, {3});
    NormHost = host_buffer.get<float>();
    LossHost = host_buffer.get<float>() + 1;
    AccuracyHost = host_buffer.get<float>() + 2;
}

void destroy_cudnn_handle(cudnnHandle_t handle) noexcept;
void destroy_cublaslt_handle(cublasLtHandle_t handle) noexcept;

IRunState::~IRunState() {
    // Skip cleanup if this was moved-from (MainStream will be null)
    if (!MainStream) return;
    
    auto destroy_event = [](cudaEvent_t& ev) noexcept {
        if (ev) {
            (void)cudaEventDestroy(ev);
            ev = nullptr;
        }
    };
    auto destroy_stream = [](cudaStream_t& stream) noexcept {
        if (stream) {
            (void)cudaStreamDestroy(stream);
            stream = nullptr;
        }
    };
    
    if (MainStream) {
        (void)cudaStreamSynchronize(MainStream);
    }
    
    destroy_event(ForwardDone);
    destroy_event(BackwardDone);
    destroy_event(TransferDone);
    destroy_event(NormDone);
    destroy_event(OptimizerDone);
    destroy_event(TimingOptimizerStart);
    destroy_event(TimingOptimizerEnd);
    
    for (auto& ev : TimingForwardStart) destroy_event(ev);
    for (auto& ev : TimingForwardEnd) destroy_event(ev);
    for (auto& ev : TimingHeadStart) destroy_event(ev);
    for (auto& ev : TimingHeadEnd) destroy_event(ev);
    for (auto& ev : TimingBackwardStart) destroy_event(ev);
    for (auto& ev : TimingBackwardEnd) destroy_event(ev);
    
    if (CublasLtHandle) {
        destroy_cublaslt_handle(CublasLtHandle);
        CublasLtHandle = nullptr;
    }
    if (CudnnHandle) {
        destroy_cudnn_handle(CudnnHandle);
        CudnnHandle = nullptr;
    }
    
    destroy_stream(MainStream);
}

IRunState::IRunState(IRunState&& other) noexcept
    : Config(std::move(other.Config)),
      B(other.B), T(other.T),
      GradAccumSteps(other.GradAccumSteps),
      WorldSize(other.WorldSize),
      Allocator(std::move(other.Allocator)),
      Stack(std::move(other.Stack)),
      Inputs(std::move(other.Inputs)),
      PositionIDs(std::move(other.PositionIDs)),
      Targets(std::move(other.Targets)),
      Losses(std::move(other.Losses)),
      ValidTokenCount(std::move(other.ValidTokenCount)),
      CorrectCount(std::move(other.CorrectCount)),
      NormHost(other.NormHost),
      LossHost(other.LossHost),
      AccuracyHost(other.AccuracyHost),
      DeviceProp(other.DeviceProp),
      DeviceId(other.DeviceId),
      MainStream(other.MainStream),
      ForwardDone(other.ForwardDone),
      BackwardDone(other.BackwardDone),
      TransferDone(other.TransferDone),
      NormDone(other.NormDone),
      OptimizerDone(other.OptimizerDone),
      CudnnHandle(other.CudnnHandle),
      CublasLtHandle(other.CublasLtHandle),
      CuBlasWorkspace(std::move(other.CuBlasWorkspace)),
      TimingOptimizerStart(other.TimingOptimizerStart),
      TimingOptimizerEnd(other.TimingOptimizerEnd),
      TimingForwardStart(std::move(other.TimingForwardStart)),
      TimingForwardEnd(std::move(other.TimingForwardEnd)),
      TimingHeadStart(std::move(other.TimingHeadStart)),
      TimingHeadEnd(std::move(other.TimingHeadEnd)),
      TimingBackwardStart(std::move(other.TimingBackwardStart)),
      TimingBackwardEnd(std::move(other.TimingBackwardEnd)),
      Inputs_CPU(std::move(other.Inputs_CPU)),
      PositionIDs_CPU(std::move(other.PositionIDs_CPU)),
      Targets_CPU(std::move(other.Targets_CPU))
{
    // Null out the source's CUDA handles so its destructor doesn't free them
    other.MainStream = nullptr;
    other.ForwardDone = nullptr;
    other.BackwardDone = nullptr;
    other.TransferDone = nullptr;
    other.NormDone = nullptr;
    other.OptimizerDone = nullptr;
    other.CudnnHandle = nullptr;
    other.CublasLtHandle = nullptr;
    other.TimingOptimizerStart = nullptr;
    other.TimingOptimizerEnd = nullptr;
    other.NormHost = nullptr;
    other.LossHost = nullptr;
    other.DeviceId = -1;
    other.GradAccumSteps = 1;
    other.WorldSize = 1;
}

IRunState& IRunState::operator=(IRunState&& other) noexcept {
    if (this != &other) {
        // First destroy our existing resources
        this->~IRunState();
        
        // Then move from other
        new (this) IRunState(std::move(other));
    }
    return *this;
}

void IRunState::setup_timing_events(int micro_steps) {
    TimingOptimizerStart = create_named_event("timing_opt_start", true);
    TimingOptimizerEnd = create_named_event("timing_opt_done", true);
    for(int i = TimingForwardStart.size(); i < micro_steps + 1; ++i) {
        TimingForwardStart.push_back(create_named_event(("timing_fwd_" + std::to_string(i) + "_start").c_str(), true));
        TimingForwardEnd.push_back(create_named_event(("timing_fwd_" + std::to_string(i) + "_end").c_str(), true));
        TimingHeadStart.push_back(create_named_event(("timing_head_" + std::to_string(i) + "_start").c_str(), true));
        TimingHeadEnd.push_back(create_named_event(("timing_head_" + std::to_string(i) + "_end").c_str(), true));
        TimingBackwardStart.push_back(create_named_event(("timing_bwd_" + std::to_string(i) + "_start").c_str(), true));
        TimingBackwardEnd.push_back(create_named_event(("timing_bwd_" + std::to_string(i) + "_end").c_str(), true));
    }
}

float IRunState::get_loss() const {
    CUDA_CHECK(cudaEventSynchronize(BackwardDone));
    return LossHost[0];
}

float IRunState::get_norm() const {
    CUDA_CHECK(cudaEventSynchronize(NormDone));
    return NormHost[0];
}

float IRunState::get_accuracy() const {
    // Accuracy is set during validate(), no need to wait for specific event
    return AccuracyHost[0];
}

Tensor IRunState::temp_alloc(ETensorDType dtype, const std::vector<long>& shape) {
    return Stack.allocate(dtype, shape);
}

void IRunState::temp_acquire(Tensor& target) {
    if(target.Device != Stack.device_id()) {
        throw std::logic_error("device mismatch");
    }

    target.Data = Stack.allocate(target.bytes());
}

void IRunState::temp_free(Tensor& tensor) {
    Stack.free(tensor);
}


std::pair<float, float> IRunState::record_step(float loss, float norm) {
    LossOutliers.record(loss);
    NormOutliers.record(norm);

    return {LossOutliers.eval(loss), NormOutliers.eval(norm)};
}


IRunState::OutlierDetector::OutlierDetector(int window_size) : mWindowSize(window_size){
    mValues.reserve(window_size);
}

void IRunState::OutlierDetector::record(float value) {
    double v_n = value;
    if (mValues.size() < mWindowSize) {
        // simply add the value and accumulate buffers
        mValues.push_back(value);
        mSum += v_n;
        mSumSq += v_n * v_n;
    } else {
        // we need to subtract the old value, and add the new one.
        double v_o = mValues[mIndex];
        mSum += v_n - v_o;
        mSumSq += v_n * v_n - v_o * v_o;

        mValues[mIndex] = value;
        mIndex = (mIndex + 1) % mWindowSize;
    }

    // periodically recompute to prevent accumulation of
    // rounding errors
    if (mIndex == 0 && mValues.size() == mWindowSize) {
        re_evaluate();
    }
}

float IRunState::OutlierDetector::eval(float value) const {
    if (mValues.size() < mWindowSize) {
        return 0.0;
    } else {
        double mean = mSum / mWindowSize;
        double variance = mSumSq / mWindowSize - mean * mean;
        double std_dev = std::sqrt(variance);
        if (std_dev == 0.0) {
            return 0.0;
        }
        return (static_cast<double>(value) - mean) / std_dev;
    }
}

void IRunState::OutlierDetector::re_evaluate() {
    mSum = 0.0;
    mSumSq = 0.0;
    for (float val : mValues) {
        mSum += val;
        mSumSq += val * val;
    }
}

void IRunState::OutlierDetector::reset(int window_size, int index, std::vector<float> values) {
    mWindowSize = window_size;
    mIndex = index;
    mValues = std::move(values);
    re_evaluate();
}
