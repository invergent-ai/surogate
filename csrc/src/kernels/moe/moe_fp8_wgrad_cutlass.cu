// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "kernels/moe/moe_fp8_wgrad_cutlass.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "utilities/utils.h"

namespace {

__global__ void pack_moe_wgrad_grad_e5m2_rowmajor_kernel(__nv_fp8_e5m2* __restrict__ dst,
                                                         const __nv_fp8_e5m2* __restrict__ src,
                                                         const int* __restrict__ src_offsets,
                                                         const int* __restrict__ tokens_per_group,
                                                         const int* __restrict__ dst_offsets,
                                                         int M) {
    const int group = blockIdx.y;
    const int K = tokens_per_group[group];
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * K;
    if (linear >= total) {
        return;
    }
    const int m = linear / K;
    const int k = linear - m * K;
    dst[dst_offsets[group] + linear] = src[(src_offsets[group] + k) * M + m];
}

__global__ void pack_moe_wgrad_input_e4m3_colmajor_kernel(__nv_fp8_e4m3* __restrict__ dst,
                                                          const __nv_fp8_e4m3* __restrict__ src,
                                                          const int* __restrict__ src_offsets,
                                                          const int* __restrict__ tokens_per_group,
                                                          const int* __restrict__ dst_offsets,
                                                          int N) {
    const int group = blockIdx.y;
    const int K = tokens_per_group[group];
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K;
    if (linear >= total) {
        return;
    }
    const int n = linear / K;
    const int k = linear - n * K;
    dst[dst_offsets[group] + linear] = src[(src_offsets[group] + k) * N + n];
}

void check_cutlass_status(cutlass::Status status, const char* where) {
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cutlassGetStatusString(status));
    }
}

#if defined(CUTLASS_ARCH_MMA_F16_SM89_SUPPORTED)

void moe_grouped_gemm_weight_grad_fp8_cutlass_sm89_dense(nv_bfloat16* d_weight,
                                                         const __nv_fp8_e5m2* grad_output,
                                                         const __nv_fp8_e4m3* input,
                                                         int num_experts,
                                                         int M,
                                                         int N,
                                                         cudaStream_t stream,
                                                         const int* host_offsets,
                                                         float alpha,
                                                         float beta,
                                                         const int* active_expert_indices,
                                                         bool weight_is_compact,
                                                         int num_active_experts) {
    using ElementA = cutlass::float_e5m2_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementOutput = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    static constexpr int kStages = 3;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementOutput,
        LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<128, 256, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                     128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                     ElementAccumulator,
                                                     ElementAccumulator>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        kStages>;

    const int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;
    std::vector<int> src_offsets_host;
    std::vector<int> tokens_host;
    std::vector<int> packed_grad_offsets_host;
    std::vector<int> packed_input_offsets_host;
    std::vector<int> weight_indices_host;
    src_offsets_host.reserve(n_active);
    tokens_host.reserve(n_active);
    packed_grad_offsets_host.reserve(n_active);
    packed_input_offsets_host.reserve(n_active);
    weight_indices_host.reserve(n_active);

    int packed_grad_elements = 0;
    int packed_input_elements = 0;
    int max_tokens = 0;
    for (int e = 0; e < n_active; ++e) {
        const int global_idx = active_expert_indices ? active_expert_indices[e] : e;
        if (global_idx < 0 || global_idx >= num_experts) {
            throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8: active expert index out of range");
        }
        const int tokens_e = host_offsets[global_idx + 1] - host_offsets[global_idx];
        if (tokens_e <= 0) {
            continue;
        }
        if ((tokens_e % 16) != 0) {
            throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8 Ada path requires token counts aligned to 16");
        }

        src_offsets_host.push_back(host_offsets[global_idx]);
        tokens_host.push_back(tokens_e);
        packed_grad_offsets_host.push_back(packed_grad_elements);
        packed_input_offsets_host.push_back(packed_input_elements);
        weight_indices_host.push_back(weight_is_compact ? e : global_idx);
        packed_grad_elements += M * tokens_e;
        packed_input_elements += N * tokens_e;
        max_tokens = std::max(max_tokens, tokens_e);
    }

    const int gemm_count = static_cast<int>(tokens_host.size());
    if (gemm_count == 0) {
        return;
    }

    __nv_fp8_e5m2* packed_grad = nullptr;
    __nv_fp8_e4m3* packed_input = nullptr;
    int* src_offsets = nullptr;
    int* tokens_per_group = nullptr;
    int* packed_grad_offsets = nullptr;
    int* packed_input_offsets = nullptr;
    void* workspace = nullptr;

    CUDA_CHECK(
        cudaMallocAsync(reinterpret_cast<void**>(&packed_grad), sizeof(__nv_fp8_e5m2) * packed_grad_elements, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&packed_input),
                               sizeof(__nv_fp8_e4m3) * packed_input_elements,
                               stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&src_offsets), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&tokens_per_group), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&packed_grad_offsets), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&packed_input_offsets), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMemcpyAsync(src_offsets,
                               src_offsets_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(tokens_per_group,
                               tokens_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(packed_grad_offsets,
                               packed_grad_offsets_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(packed_input_offsets,
                               packed_input_offsets_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));

    const int pack_threads = 256;
    const dim3 pack_grad_grid((M * max_tokens + pack_threads - 1) / pack_threads, gemm_count);
    pack_moe_wgrad_grad_e5m2_rowmajor_kernel<<<pack_grad_grid, pack_threads, 0, stream>>>(packed_grad,
                                                                                          grad_output,
                                                                                          src_offsets,
                                                                                          tokens_per_group,
                                                                                          packed_grad_offsets,
                                                                                          M);
    CUDA_CHECK(cudaGetLastError());

    const dim3 pack_input_grid((N * max_tokens + pack_threads - 1) / pack_threads, gemm_count);
    pack_moe_wgrad_input_e4m3_colmajor_kernel<<<pack_input_grid, pack_threads, 0, stream>>>(packed_input,
                                                                                            input,
                                                                                            src_offsets,
                                                                                            tokens_per_group,
                                                                                            packed_input_offsets,
                                                                                            N);
    CUDA_CHECK(cudaGetLastError());

    Gemm gemm;
    for (int i = 0; i < gemm_count; ++i) {
        const int K = tokens_host[i];
        const auto* a = reinterpret_cast<const ElementA*>(packed_grad + packed_grad_offsets_host[i]);
        const auto* b = reinterpret_cast<const ElementB*>(packed_input + packed_input_offsets_host[i]);
        auto* d = reinterpret_cast<ElementOutput*>(d_weight + static_cast<long>(weight_indices_host[i]) * M * N);

        typename Gemm::Arguments args({M, N, K}, {a, K}, {b, K}, {d, M}, {d, M}, {alpha, beta});
        check_cutlass_status(Gemm::can_implement(args), "moe_grouped_gemm_weight_grad_fp8 Ada can_implement");

        const size_t workspace_size = Gemm::get_workspace_size(args);
        if (workspace_size > 0 && !workspace) {
            CUDA_CHECK(cudaMallocAsync(&workspace, workspace_size, stream));
        }

        check_cutlass_status(gemm.initialize(args, workspace, stream),
                             "moe_grouped_gemm_weight_grad_fp8 Ada initialize");
        check_cutlass_status(gemm.run(stream), "moe_grouped_gemm_weight_grad_fp8 Ada run");
    }

    CUDA_CHECK(cudaFreeAsync(packed_grad, stream));
    CUDA_CHECK(cudaFreeAsync(packed_input, stream));
    CUDA_CHECK(cudaFreeAsync(src_offsets, stream));
    CUDA_CHECK(cudaFreeAsync(tokens_per_group, stream));
    CUDA_CHECK(cudaFreeAsync(packed_grad_offsets, stream));
    CUDA_CHECK(cudaFreeAsync(packed_input_offsets, stream));
    if (workspace) {
        CUDA_CHECK(cudaFreeAsync(workspace, stream));
    }
}

#endif

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

void moe_grouped_gemm_weight_grad_fp8_cutlass_sm90(nv_bfloat16* d_weight,
                                                   const __nv_fp8_e5m2* grad_output,
                                                   const __nv_fp8_e4m3* input,
                                                   int num_experts,
                                                   int M,
                                                   int N,
                                                   cudaStream_t stream,
                                                   const int* host_offsets,
                                                   float alpha,
                                                   float beta,
                                                   const int* active_expert_indices,
                                                   bool weight_is_compact,
                                                   int num_active_experts) {
    using namespace cute;
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
    using ElementA = cutlass::float_e5m2_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::bfloat16_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using ElementAccumulator = float;
    using ArchTag = cutlass::arch::Sm90;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_2, _1, _1>;
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementAccumulator,
        ElementC,
        LayoutC*,
        AlignmentC,
        ElementC,
        LayoutC*,
        AlignmentC,
        EpilogueSchedule,
        cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        ElementA,
        LayoutA*,
        AlignmentA,
        ElementB,
        LayoutB*,
        AlignmentB,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;

    const int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;
    std::vector<ProblemShape::UnderlyingProblemShape> problem_shapes_host;
    std::vector<const ElementA*> ptr_A_host;
    std::vector<const ElementB*> ptr_B_host;
    std::vector<const ElementC*> ptr_C_host;
    std::vector<ElementC*> ptr_D_host;
    std::vector<StrideA> stride_A_host;
    std::vector<StrideB> stride_B_host;
    std::vector<StrideC> stride_C_host;
    std::vector<StrideD> stride_D_host;
    std::vector<int> src_offsets_host;
    std::vector<int> tokens_host;
    std::vector<int> packed_offsets_host;

    problem_shapes_host.reserve(n_active);
    ptr_A_host.reserve(n_active);
    ptr_B_host.reserve(n_active);
    ptr_C_host.reserve(n_active);
    ptr_D_host.reserve(n_active);
    stride_A_host.reserve(n_active);
    stride_B_host.reserve(n_active);
    stride_C_host.reserve(n_active);
    stride_D_host.reserve(n_active);
    src_offsets_host.reserve(n_active);
    tokens_host.reserve(n_active);
    packed_offsets_host.reserve(n_active);

    int packed_grad_elements = 0;
    for (int e = 0; e < n_active; ++e) {
        const int global_idx = active_expert_indices ? active_expert_indices[e] : e;
        if (global_idx < 0 || global_idx >= num_experts) {
            throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8: active expert index out of range");
        }
        const int tokens_e = host_offsets[global_idx + 1] - host_offsets[global_idx];
        if (tokens_e <= 0) {
            continue;
        }
        if ((tokens_e % 16) != 0) {
            throw std::runtime_error(
                "moe_grouped_gemm_weight_grad_fp8 CUTLASS path requires token counts aligned to 16");
        }

        const int weight_idx = weight_is_compact ? e : global_idx;
        const auto* input_e =
            reinterpret_cast<const ElementB*>(input + static_cast<long>(host_offsets[global_idx]) * N);
        auto* d_weight_e = reinterpret_cast<ElementC*>(d_weight + static_cast<long>(weight_idx) * M * N);

        problem_shapes_host.push_back({M, N, tokens_e});
        ptr_A_host.push_back(nullptr);
        ptr_B_host.push_back(input_e);
        ptr_C_host.push_back(d_weight_e);
        ptr_D_host.push_back(d_weight_e);
        src_offsets_host.push_back(host_offsets[global_idx]);
        tokens_host.push_back(tokens_e);
        packed_offsets_host.push_back(packed_grad_elements);
        packed_grad_elements += M * tokens_e;

        // A is packed below as row-major grad_output^T: logical (M, K).
        // B is input^T as CUTLASS's logical (N, K), backed by row-major (K, N), so use column-major.
        stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, tokens_e, 1}));
        stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, tokens_e, 1}));
        stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
        stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));
    }

    const int gemm_count = static_cast<int>(problem_shapes_host.size());
    if (gemm_count == 0) {
        return;
    }

    ProblemShape::UnderlyingProblemShape* problem_shapes = nullptr;
    const ElementA** ptr_A = nullptr;
    const ElementB** ptr_B = nullptr;
    const ElementC** ptr_C = nullptr;
    ElementC** ptr_D = nullptr;
    StrideA* stride_A = nullptr;
    StrideB* stride_B = nullptr;
    StrideC* stride_C = nullptr;
    StrideD* stride_D = nullptr;
    __nv_fp8_e5m2* packed_grad = nullptr;
    int* src_offsets = nullptr;
    int* tokens_per_group = nullptr;
    int* packed_offsets = nullptr;
    void* workspace = nullptr;

    CUDA_CHECK(
        cudaMallocAsync(reinterpret_cast<void**>(&packed_grad), sizeof(__nv_fp8_e5m2) * packed_grad_elements, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&src_offsets), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&tokens_per_group), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&packed_offsets), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&problem_shapes),
                               sizeof(ProblemShape::UnderlyingProblemShape) * gemm_count,
                               stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr_A), sizeof(ElementA*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr_B), sizeof(ElementB*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr_C), sizeof(ElementC*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr_D), sizeof(ElementC*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&stride_A), sizeof(StrideA) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&stride_B), sizeof(StrideB) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&stride_C), sizeof(StrideC) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&stride_D), sizeof(StrideD) * gemm_count, stream));

    CUDA_CHECK(cudaMemcpyAsync(src_offsets,
                               src_offsets_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(tokens_per_group,
                               tokens_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(packed_offsets,
                               packed_offsets_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));

    int max_tokens = 0;
    for (int tokens : tokens_host) {
        max_tokens = std::max(max_tokens, tokens);
    }
    const int pack_threads = 256;
    const dim3 pack_grid((M * max_tokens + pack_threads - 1) / pack_threads, gemm_count);
    pack_moe_wgrad_grad_e5m2_rowmajor_kernel<<<pack_grid, pack_threads, 0, stream>>>(packed_grad,
                                                                                     grad_output,
                                                                                     src_offsets,
                                                                                     tokens_per_group,
                                                                                     packed_offsets,
                                                                                     M);
    CUDA_CHECK(cudaGetLastError());

    for (int i = 0; i < gemm_count; ++i) {
        ptr_A_host[i] = reinterpret_cast<const ElementA*>(packed_grad + packed_offsets_host[i]);
    }

    CUDA_CHECK(cudaMemcpyAsync(problem_shapes,
                               problem_shapes_host.data(),
                               sizeof(ProblemShape::UnderlyingProblemShape) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(
        cudaMemcpyAsync(ptr_A, ptr_A_host.data(), sizeof(ElementA*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(ptr_B, ptr_B_host.data(), sizeof(ElementB*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(ptr_C, ptr_C_host.data(), sizeof(ElementC*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(ptr_D, ptr_D_host.data(), sizeof(ElementC*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(stride_A, stride_A_host.data(), sizeof(StrideA) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(stride_B, stride_B_host.data(), sizeof(StrideB) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(stride_C, stride_C_host.data(), sizeof(StrideC) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(stride_D, stride_D_host.data(), sizeof(StrideD) * gemm_count, cudaMemcpyHostToDevice, stream));

    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args{};
    fusion_args.alpha = alpha;
    fusion_args.beta = beta;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cutlass::KernelHardwareInfo hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(device_id);

    arguments = typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                         {gemm_count, problem_shapes, problem_shapes_host.data()},
                                         {ptr_A, stride_A, ptr_B, stride_B},
                                         {fusion_args, ptr_C, stride_C, ptr_D, stride_D},
                                         hw_info};

    Gemm gemm;
    check_cutlass_status(Gemm::can_implement(arguments), "moe_grouped_gemm_weight_grad_fp8 CUTLASS can_implement");

    const size_t workspace_size = Gemm::get_workspace_size(arguments);
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMallocAsync(&workspace, workspace_size, stream));
    }

    check_cutlass_status(gemm.initialize(arguments, workspace, stream),
                         "moe_grouped_gemm_weight_grad_fp8 CUTLASS initialize");
    check_cutlass_status(gemm.run(stream), "moe_grouped_gemm_weight_grad_fp8 CUTLASS run");

    CUDA_CHECK(cudaFreeAsync(problem_shapes, stream));
    CUDA_CHECK(cudaFreeAsync(ptr_A, stream));
    CUDA_CHECK(cudaFreeAsync(ptr_B, stream));
    CUDA_CHECK(cudaFreeAsync(ptr_C, stream));
    CUDA_CHECK(cudaFreeAsync(ptr_D, stream));
    CUDA_CHECK(cudaFreeAsync(stride_A, stream));
    CUDA_CHECK(cudaFreeAsync(stride_B, stream));
    CUDA_CHECK(cudaFreeAsync(stride_C, stream));
    CUDA_CHECK(cudaFreeAsync(stride_D, stream));
    CUDA_CHECK(cudaFreeAsync(packed_grad, stream));
    CUDA_CHECK(cudaFreeAsync(src_offsets, stream));
    CUDA_CHECK(cudaFreeAsync(tokens_per_group, stream));
    CUDA_CHECK(cudaFreeAsync(packed_offsets, stream));
    if (workspace) {
        CUDA_CHECK(cudaFreeAsync(workspace, stream));
    }
}

#endif

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

void moe_grouped_gemm_weight_grad_fp8_cutlass_sm100_dense(nv_bfloat16* d_weight,
                                                          const __nv_fp8_e5m2* grad_output,
                                                          const __nv_fp8_e4m3* input,
                                                          int num_experts,
                                                          int M,
                                                          int N,
                                                          cudaStream_t stream,
                                                          const int* host_offsets,
                                                          float alpha,
                                                          float beta,
                                                          const int* active_expert_indices,
                                                          bool weight_is_compact,
                                                          int num_active_experts) {
    if (beta != 0.0f) {
        throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8 Blackwell path currently requires beta=0");
    }

    using namespace cute;
    using ElementA = cutlass::float_e5m2_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = void;
    using ElementD = cutlass::bfloat16_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::ColumnMajor;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using MmaTileShape = Shape<_128, _16, _128>;
    using ClusterShape = Shape<_4, _1, _1>;

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    constexpr int AlignmentC = 0;
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        LayoutC,
        AlignmentC,
        ElementD,
        LayoutD,
        AlignmentD,
        cutlass::epilogue::TmaWarpSpecialized1Sm>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        AlignmentA,
        ElementB,
        LayoutB,
        AlignmentB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    const int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;
    std::vector<int> src_offsets_host;
    std::vector<int> tokens_host;
    std::vector<int> packed_offsets_host;
    std::vector<int> weight_indices_host;
    src_offsets_host.reserve(n_active);
    tokens_host.reserve(n_active);
    packed_offsets_host.reserve(n_active);
    weight_indices_host.reserve(n_active);

    int packed_grad_elements = 0;
    int max_tokens = 0;
    for (int e = 0; e < n_active; ++e) {
        const int global_idx = active_expert_indices ? active_expert_indices[e] : e;
        if (global_idx < 0 || global_idx >= num_experts) {
            throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8: active expert index out of range");
        }
        const int tokens_e = host_offsets[global_idx + 1] - host_offsets[global_idx];
        if (tokens_e <= 0) {
            continue;
        }
        if ((tokens_e % 16) != 0) {
            throw std::runtime_error(
                "moe_grouped_gemm_weight_grad_fp8 Blackwell path requires token counts aligned to 16");
        }
        src_offsets_host.push_back(host_offsets[global_idx]);
        tokens_host.push_back(tokens_e);
        packed_offsets_host.push_back(packed_grad_elements);
        weight_indices_host.push_back(weight_is_compact ? e : global_idx);
        packed_grad_elements += M * tokens_e;
        max_tokens = std::max(max_tokens, tokens_e);
    }

    const int gemm_count = static_cast<int>(tokens_host.size());
    if (gemm_count == 0) {
        return;
    }

    __nv_fp8_e5m2* packed_grad = nullptr;
    int* src_offsets = nullptr;
    int* tokens_per_group = nullptr;
    int* packed_offsets = nullptr;
    void* workspace = nullptr;
    CUDA_CHECK(
        cudaMallocAsync(reinterpret_cast<void**>(&packed_grad), sizeof(__nv_fp8_e5m2) * packed_grad_elements, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&src_offsets), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&tokens_per_group), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&packed_offsets), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMemcpyAsync(src_offsets,
                               src_offsets_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(tokens_per_group,
                               tokens_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(packed_offsets,
                               packed_offsets_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));

    const int pack_threads = 256;
    const dim3 pack_grid((M * max_tokens + pack_threads - 1) / pack_threads, gemm_count);
    pack_moe_wgrad_grad_e5m2_rowmajor_kernel<<<pack_grid, pack_threads, 0, stream>>>(packed_grad,
                                                                                     grad_output,
                                                                                     src_offsets,
                                                                                     tokens_per_group,
                                                                                     packed_offsets,
                                                                                     M);
    CUDA_CHECK(cudaGetLastError());

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cutlass::KernelHardwareInfo hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(device_id);

    Gemm gemm;
    for (int i = 0; i < gemm_count; ++i) {
        const int K = tokens_host[i];
        const auto* a = reinterpret_cast<const ElementA*>(packed_grad + packed_offsets_host[i]);
        const auto* b = reinterpret_cast<const ElementB*>(input + static_cast<long>(src_offsets_host[i]) * N);
        auto* d = reinterpret_cast<ElementD*>(d_weight + static_cast<long>(weight_indices_host[i]) * M * N);

        StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

        typename Gemm::Arguments arguments;
        decltype(arguments.epilogue.thread) fusion_args{};
        fusion_args.alpha = alpha;
        fusion_args.alpha_ptr = nullptr;

        arguments = typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                             {M, N, K, 1},
                                             {a, stride_a, b, stride_b},
                                             {fusion_args, nullptr, stride_c, d, stride_d},
                                             hw_info};

        check_cutlass_status(Gemm::can_implement(arguments),
                             "moe_grouped_gemm_weight_grad_fp8 Blackwell can_implement");
        const size_t workspace_size = Gemm::get_workspace_size(arguments);
        if (workspace_size > 0 && !workspace) {
            CUDA_CHECK(cudaMallocAsync(&workspace, workspace_size, stream));
        }
        check_cutlass_status(gemm.initialize(arguments, workspace, stream),
                             "moe_grouped_gemm_weight_grad_fp8 Blackwell initialize");
        check_cutlass_status(gemm.run(stream), "moe_grouped_gemm_weight_grad_fp8 Blackwell run");
    }

    CUDA_CHECK(cudaFreeAsync(packed_grad, stream));
    CUDA_CHECK(cudaFreeAsync(src_offsets, stream));
    CUDA_CHECK(cudaFreeAsync(tokens_per_group, stream));
    CUDA_CHECK(cudaFreeAsync(packed_offsets, stream));
    if (workspace) {
        CUDA_CHECK(cudaFreeAsync(workspace, stream));
    }
}

#endif

}  // namespace

void moe_grouped_gemm_weight_grad_fp8_cutlass(nv_bfloat16* d_weight,
                                              const __nv_fp8_e5m2* grad_output,
                                              const __nv_fp8_e4m3* input,
                                              int num_experts,
                                              int M,
                                              int N,
                                              cudaStream_t stream,
                                              const int* host_offsets,
                                              float alpha,
                                              float beta,
                                              const int* active_expert_indices,
                                              bool weight_is_compact,
                                              int num_active_experts) {
    cudaDeviceProp props{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    if (props.major < 9) {
        throw std::runtime_error(
            "moe_grouped_gemm_weight_grad_fp8 CUTLASS path requires Ada/Hopper/Blackwell tensor cores");
    }

    if (props.major >= 10) {
#if defined(CUTLASS_ARCH_MMA_F16_SM89_SUPPORTED)
        // Blackwell GeForce FP8 examples use blockwise/groupwise scale operands.
        // Until the native SM100/SM120 E5M2 grouped path is complete, keep the
        // env-gated FP8 wgrad route functional through the Ada FP8 tensor-op kernel.
        moe_grouped_gemm_weight_grad_fp8_cutlass_sm89_dense(d_weight,
                                                            grad_output,
                                                            input,
                                                            num_experts,
                                                            M,
                                                            N,
                                                            stream,
                                                            host_offsets,
                                                            alpha,
                                                            beta,
                                                            active_expert_indices,
                                                            weight_is_compact,
                                                            num_active_experts);
#else
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || \
    defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
        moe_grouped_gemm_weight_grad_fp8_cutlass_sm100_dense(d_weight,
                                                             grad_output,
                                                             input,
                                                             num_experts,
                                                             M,
                                                             N,
                                                             stream,
                                                             host_offsets,
                                                             alpha,
                                                             beta,
                                                             active_expert_indices,
                                                             weight_is_compact,
                                                             num_active_experts);
#else
        throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8 Blackwell CUTLASS support was not compiled");
#endif
#endif
    } else if (props.major == 9) {
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
        moe_grouped_gemm_weight_grad_fp8_cutlass_sm90(d_weight,
                                                      grad_output,
                                                      input,
                                                      num_experts,
                                                      M,
                                                      N,
                                                      stream,
                                                      host_offsets,
                                                      alpha,
                                                      beta,
                                                      active_expert_indices,
                                                      weight_is_compact,
                                                      num_active_experts);
#else
        throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8 Hopper CUTLASS support was not compiled");
#endif
    } else {
#if defined(CUTLASS_ARCH_MMA_F16_SM89_SUPPORTED)
        moe_grouped_gemm_weight_grad_fp8_cutlass_sm89_dense(d_weight,
                                                            grad_output,
                                                            input,
                                                            num_experts,
                                                            M,
                                                            N,
                                                            stream,
                                                            host_offsets,
                                                            alpha,
                                                            beta,
                                                            active_expert_indices,
                                                            weight_is_compact,
                                                            num_active_experts);
#else
        throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8 Ada CUTLASS support was not compiled");
#endif
    }
}
