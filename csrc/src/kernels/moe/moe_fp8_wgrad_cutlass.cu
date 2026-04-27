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
#include "cutlass/detail/blockwise_scale_layout.hpp"
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

// Pack grad_output (total_tokens, M) row-major into a per-group packed buffer
// laid out as (M, K_pad) row-major. The trailing K_pad - K_actual columns are
// zero-filled so the GEMM contributes nothing for those K positions. This is
// what the SM120 grouped blockwise FP8 kernel needs because its tile/scale K
// must equal ScaleGranularityK (typically 128) and the problem K must be a
// multiple of that block size.
__global__ void pack_moe_wgrad_grad_e5m2_padded_kernel(__nv_fp8_e5m2* __restrict__ dst,
                                                       const __nv_fp8_e5m2* __restrict__ src,
                                                       const int* __restrict__ src_offsets,
                                                       const int* __restrict__ tokens_per_group,
                                                       const int* __restrict__ tokens_padded_per_group,
                                                       const int* __restrict__ dst_offsets,
                                                       int M) {
    const int group = blockIdx.y;
    const int K_act = tokens_per_group[group];
    const int K_pad = tokens_padded_per_group[group];
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * K_pad;
    if (linear >= total) {
        return;
    }
    const int m = linear / K_pad;
    const int k = linear - m * K_pad;
    if (k < K_act) {
        dst[dst_offsets[group] + linear] = src[(src_offsets[group] + k) * M + m];
    } else {
        dst[dst_offsets[group] + linear] = static_cast<__nv_fp8_e5m2>(0.0f);
    }
}

// Pack input (total_tokens, N) row-major into a per-group packed buffer laid out
// as (N, K_pad) row-major. For CUTLASS's grouped GEMM with LayoutB=ColumnMajor
// the expected stride for problem shape (N, K_pad) is (K_pad, _1, _0), which
// reads B[n, k] from offset n*K_pad + k — i.e., (N, K_pad) row-major in memory
// with K contiguous within each "row" (the K-major TN layout the SM120
// blockwise mainloop requires). For k < K_actual we copy the original input;
// for k in [K_actual, K_pad) we write zeros so the K-padded GEMM contributes
// nothing.
__global__ void pack_moe_wgrad_input_e4m3_padded_kernel(__nv_fp8_e4m3* __restrict__ dst,
                                                        const __nv_fp8_e4m3* __restrict__ src,
                                                        const int* __restrict__ src_offsets,
                                                        const int* __restrict__ tokens_per_group,
                                                        const int* __restrict__ tokens_padded_per_group,
                                                        const int* __restrict__ dst_offsets,
                                                        int N) {
    const int group = blockIdx.y;
    const int K_act = tokens_per_group[group];
    const int K_pad = tokens_padded_per_group[group];
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K_pad;
    if (linear >= total) {
        return;
    }
    // dst is (N, K_pad) row-major: dst[n * K_pad + k]
    const int n = linear / K_pad;
    const int k = linear - n * K_pad;
    if (k < K_act) {
        dst[dst_offsets[group] + linear] = src[(src_offsets[group] + k) * N + n];
    } else {
        dst[dst_offsets[group] + linear] = static_cast<__nv_fp8_e4m3>(0.0f);
    }
}

__global__ void fill_float_kernel(float* __restrict__ dst, int n, float value) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    dst[idx] = value;
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

// Native Blackwell FP8 grouped GEMM for weight gradients. Blackwell grouped FP8
// tensor cores require blockwise scale operands, so we use constant 1.0 SF
// tensors and fold the per-tensor scales into `alpha`. The tile/scale K
// granularity is 128; each group's K is padded up to 128 with zeros so partial
// tiles produce exactly the unscaled grad^T @ input result.
template <typename ArchTag,
          typename MmaTileShapeMNK,
          typename ClusterShapeMNK,
          typename ScaleConfig,
          typename EpilogueSchedule,
          typename KernelSchedule>
void moe_grouped_gemm_weight_grad_fp8_cutlass_sm1xx_grouped(const char* arch_name,
                                                            nv_bfloat16* d_weight,
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
    using ElementA = cutlass::float_e5m2_t;  // grad^T (M, K)
    using ElementB = cutlass::float_e4m3_t;  // input^T (N, K)
    using ElementC = cutlass::bfloat16_t;    // d_weight (M, N) col-major
    using ElementD = ElementC;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::ColumnMajor;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementSF = ElementAccumulator;

    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    constexpr int AlignmentD = AlignmentC;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag,
                                                                  cutlass::arch::OpClassTensorOp,
                                                                  MmaTileShapeMNK,
                                                                  ClusterShapeMNK,
                                                                  cutlass::epilogue::collective::EpilogueTileAuto,
                                                                  ElementAccumulator,
                                                                  ElementCompute,
                                                                  ElementC,
                                                                  LayoutC*,
                                                                  AlignmentC,
                                                                  ElementD,
                                                                  LayoutD*,
                                                                  AlignmentD,
                                                                  EpilogueSchedule>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        cute::tuple<LayoutA*, LayoutSFA*>,
        AlignmentA,
        ElementB,
        cute::tuple<LayoutB*, LayoutSFB*>,
        AlignmentB,
        ElementAccumulator,
        MmaTileShapeMNK,
        ClusterShapeMNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;

    constexpr int kKBlock = 128;
    auto round_up = [](int v, int a) {
        return ((v + a - 1) / a) * a;
    };

    const int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;

    std::vector<typename ProblemShape::UnderlyingProblemShape> problem_shapes_host;
    std::vector<const ElementA*> ptr_A_host;
    std::vector<const ElementB*> ptr_B_host;
    std::vector<const ElementC*> ptr_C_host;
    std::vector<ElementC*> ptr_D_host;
    std::vector<const ElementSF*> ptr_SFA_host;
    std::vector<const ElementSF*> ptr_SFB_host;
    std::vector<StrideA> stride_A_host;
    std::vector<StrideB> stride_B_host;
    std::vector<StrideC> stride_C_host;
    std::vector<StrideD> stride_D_host;
    std::vector<LayoutSFA> layout_SFA_host;
    std::vector<LayoutSFB> layout_SFB_host;
    std::vector<int> src_offsets_host;
    std::vector<int> tokens_host;
    std::vector<int> tokens_padded_host;
    std::vector<int> packed_grad_offsets_host;
    std::vector<int> packed_input_offsets_host;
    std::vector<int> sfa_offsets_host;
    std::vector<int> sfb_offsets_host;

    problem_shapes_host.reserve(n_active);
    ptr_A_host.reserve(n_active);
    ptr_B_host.reserve(n_active);
    ptr_C_host.reserve(n_active);
    ptr_D_host.reserve(n_active);
    ptr_SFA_host.reserve(n_active);
    ptr_SFB_host.reserve(n_active);
    stride_A_host.reserve(n_active);
    stride_B_host.reserve(n_active);
    stride_C_host.reserve(n_active);
    stride_D_host.reserve(n_active);
    layout_SFA_host.reserve(n_active);
    layout_SFB_host.reserve(n_active);
    src_offsets_host.reserve(n_active);
    tokens_host.reserve(n_active);
    tokens_padded_host.reserve(n_active);
    packed_grad_offsets_host.reserve(n_active);
    packed_input_offsets_host.reserve(n_active);
    sfa_offsets_host.reserve(n_active);
    sfb_offsets_host.reserve(n_active);

    int total_packed_grad = 0;
    int total_packed_input = 0;
    int total_sfa = 0;
    int total_sfb = 0;
    int max_tokens_padded = 0;
    for (int e = 0; e < n_active; ++e) {
        const int global_idx = active_expert_indices ? active_expert_indices[e] : e;
        if (global_idx < 0 || global_idx >= num_experts) {
            throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8: active expert index out of range");
        }
        const int K_act = host_offsets[global_idx + 1] - host_offsets[global_idx];
        if (K_act <= 0) {
            continue;
        }
        if ((K_act % 16) != 0) {
            throw std::runtime_error(std::string("moe_grouped_gemm_weight_grad_fp8 ") + arch_name +
                                     " path requires token counts aligned to 16");
        }
        const int K_pad = round_up(K_act, kKBlock);
        const int weight_idx = weight_is_compact ? e : global_idx;
        ElementC* d_ptr = reinterpret_cast<ElementC*>(d_weight + static_cast<long>(weight_idx) * M * N);

        problem_shapes_host.push_back({M, N, K_pad});
        ptr_A_host.push_back(nullptr);  // packed grad pointer assigned after pack
        ptr_B_host.push_back(nullptr);  // packed input pointer assigned after pack
        ptr_C_host.push_back(d_ptr);
        ptr_D_host.push_back(d_ptr);
        ptr_SFA_host.push_back(nullptr);  // assigned after SF allocation
        ptr_SFB_host.push_back(nullptr);

        src_offsets_host.push_back(host_offsets[global_idx]);
        tokens_host.push_back(K_act);
        tokens_padded_host.push_back(K_pad);
        packed_grad_offsets_host.push_back(total_packed_grad);
        packed_input_offsets_host.push_back(total_packed_input);
        total_packed_grad += M * K_pad;
        total_packed_input += N * K_pad;
        max_tokens_padded = std::max(max_tokens_padded, K_pad);

        stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K_pad, 1}));
        stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K_pad, 1}));
        stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
        stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));

        auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K_pad, 1));
        auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K_pad, 1));
        layout_SFA_host.push_back(layout_sfa);
        layout_SFB_host.push_back(layout_sfb);
        sfa_offsets_host.push_back(total_sfa);
        sfb_offsets_host.push_back(total_sfb);
        total_sfa += static_cast<int>(size(filter_zeros(layout_sfa)));
        total_sfb += static_cast<int>(size(filter_zeros(layout_sfb)));
    }

    const int gemm_count = static_cast<int>(problem_shapes_host.size());
    if (gemm_count == 0) {
        return;
    }

    // Device allocations.
    __nv_fp8_e5m2* packed_grad = nullptr;
    __nv_fp8_e4m3* packed_input = nullptr;
    ElementSF* d_sfa = nullptr;
    ElementSF* d_sfb = nullptr;
    int* d_src_offsets = nullptr;
    int* d_tokens = nullptr;
    int* d_tokens_padded = nullptr;
    int* d_packed_grad_offsets = nullptr;
    int* d_packed_input_offsets = nullptr;
    typename ProblemShape::UnderlyingProblemShape* d_problem_shapes = nullptr;
    const ElementA** d_ptr_A = nullptr;
    const ElementB** d_ptr_B = nullptr;
    const ElementC** d_ptr_C = nullptr;
    ElementC** d_ptr_D = nullptr;
    const ElementSF** d_ptr_SFA = nullptr;
    const ElementSF** d_ptr_SFB = nullptr;
    StrideA* d_stride_A = nullptr;
    StrideB* d_stride_B = nullptr;
    StrideC* d_stride_C = nullptr;
    StrideD* d_stride_D = nullptr;
    LayoutSFA* d_layout_SFA = nullptr;
    LayoutSFB* d_layout_SFB = nullptr;
    void* d_workspace = nullptr;

    CUDA_CHECK(
        cudaMallocAsync(reinterpret_cast<void**>(&packed_grad), sizeof(__nv_fp8_e5m2) * total_packed_grad, stream));
    CUDA_CHECK(
        cudaMallocAsync(reinterpret_cast<void**>(&packed_input), sizeof(__nv_fp8_e4m3) * total_packed_input, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_sfa), sizeof(ElementSF) * total_sfa, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_sfb), sizeof(ElementSF) * total_sfb, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_src_offsets), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_tokens), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_tokens_padded), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_packed_grad_offsets), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_packed_input_offsets), sizeof(int) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_problem_shapes),
                               sizeof(typename ProblemShape::UnderlyingProblemShape) * gemm_count,
                               stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_ptr_A), sizeof(ElementA*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_ptr_B), sizeof(ElementB*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_ptr_C), sizeof(ElementC*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_ptr_D), sizeof(ElementC*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_ptr_SFA), sizeof(ElementSF*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_ptr_SFB), sizeof(ElementSF*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_stride_A), sizeof(StrideA) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_stride_B), sizeof(StrideB) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_stride_C), sizeof(StrideC) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_stride_D), sizeof(StrideD) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_layout_SFA), sizeof(LayoutSFA) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_layout_SFB), sizeof(LayoutSFB) * gemm_count, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_src_offsets,
                               src_offsets_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_tokens, tokens_host.data(), sizeof(int) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_tokens_padded,
                               tokens_padded_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_packed_grad_offsets,
                               packed_grad_offsets_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_packed_input_offsets,
                               packed_input_offsets_host.data(),
                               sizeof(int) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));

    // Pack grad and input with K-padding (zero-fill for k >= K_actual).
    const int pack_threads = 256;
    {
        const dim3 grid((M * max_tokens_padded + pack_threads - 1) / pack_threads, gemm_count);
        pack_moe_wgrad_grad_e5m2_padded_kernel<<<grid, pack_threads, 0, stream>>>(packed_grad,
                                                                                  grad_output,
                                                                                  d_src_offsets,
                                                                                  d_tokens,
                                                                                  d_tokens_padded,
                                                                                  d_packed_grad_offsets,
                                                                                  M);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const dim3 grid((N * max_tokens_padded + pack_threads - 1) / pack_threads, gemm_count);
        pack_moe_wgrad_input_e4m3_padded_kernel<<<grid, pack_threads, 0, stream>>>(packed_input,
                                                                                   input,
                                                                                   d_src_offsets,
                                                                                   d_tokens,
                                                                                   d_tokens_padded,
                                                                                   d_packed_input_offsets,
                                                                                   N);
        CUDA_CHECK(cudaGetLastError());
    }

    // Fill scale tensors with 1.0 — per-tensor scaling is folded into alpha.
    if (total_sfa > 0) {
        const int blocks = (total_sfa + pack_threads - 1) / pack_threads;
        fill_float_kernel<<<blocks, pack_threads, 0, stream>>>(d_sfa, total_sfa, 1.0f);
        CUDA_CHECK(cudaGetLastError());
    }
    if (total_sfb > 0) {
        const int blocks = (total_sfb + pack_threads - 1) / pack_threads;
        fill_float_kernel<<<blocks, pack_threads, 0, stream>>>(d_sfb, total_sfb, 1.0f);
        CUDA_CHECK(cudaGetLastError());
    }

    // Now that the packed buffers and SF arrays exist, populate per-group pointer lists.
    for (int i = 0; i < gemm_count; ++i) {
        ptr_A_host[i] = reinterpret_cast<const ElementA*>(packed_grad + packed_grad_offsets_host[i]);
        ptr_B_host[i] = reinterpret_cast<const ElementB*>(packed_input + packed_input_offsets_host[i]);
        ptr_SFA_host[i] = d_sfa + sfa_offsets_host[i];
        ptr_SFB_host[i] = d_sfb + sfb_offsets_host[i];
    }

    CUDA_CHECK(cudaMemcpyAsync(d_problem_shapes,
                               problem_shapes_host.data(),
                               sizeof(typename ProblemShape::UnderlyingProblemShape) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_ptr_A, ptr_A_host.data(), sizeof(ElementA*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_ptr_B, ptr_B_host.data(), sizeof(ElementB*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_ptr_C, ptr_C_host.data(), sizeof(ElementC*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_ptr_D, ptr_D_host.data(), sizeof(ElementC*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ptr_SFA,
                               ptr_SFA_host.data(),
                               sizeof(ElementSF*) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ptr_SFB,
                               ptr_SFB_host.data(),
                               sizeof(ElementSF*) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_stride_A,
                               stride_A_host.data(),
                               sizeof(StrideA) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_stride_B,
                               stride_B_host.data(),
                               sizeof(StrideB) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_stride_C,
                               stride_C_host.data(),
                               sizeof(StrideC) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_stride_D,
                               stride_D_host.data(),
                               sizeof(StrideD) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_layout_SFA,
                               layout_SFA_host.data(),
                               sizeof(LayoutSFA) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_layout_SFB,
                               layout_SFB_host.data(),
                               sizeof(LayoutSFB) * gemm_count,
                               cudaMemcpyHostToDevice,
                               stream));

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cutlass::KernelHardwareInfo hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(device_id);

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

    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {gemm_count, d_problem_shapes, problem_shapes_host.data()},
        {d_ptr_A, d_stride_A, d_ptr_B, d_stride_B, d_ptr_SFA, d_layout_SFA, d_ptr_SFB, d_layout_SFB},
        {fusion_args, d_ptr_C, d_stride_C, d_ptr_D, d_stride_D},
        hw_info};

    Gemm gemm;
    check_cutlass_status(Gemm::can_implement(arguments), "moe_grouped_gemm_weight_grad_fp8 SM1xx can_implement");

    const size_t workspace_size = Gemm::get_workspace_size(arguments);
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMallocAsync(&d_workspace, workspace_size, stream));
    }

    check_cutlass_status(gemm.initialize(arguments, d_workspace, stream),
                         "moe_grouped_gemm_weight_grad_fp8 SM1xx initialize");
    check_cutlass_status(gemm.run(stream), "moe_grouped_gemm_weight_grad_fp8 SM1xx run");

    if (std::getenv("SUROGATE_DEBUG_FP8_MOE_WGRAD")) {
        std::fprintf(stderr,
                     "[FP8 MoE wgrad] %s grouped path: groups=%d M=%d N=%d K_pad_max=%d alpha=%g\n",
                     arch_name,
                     gemm_count,
                     M,
                     N,
                     max_tokens_padded,
                     alpha);
    }

    CUDA_CHECK(cudaFreeAsync(packed_grad, stream));
    CUDA_CHECK(cudaFreeAsync(packed_input, stream));
    CUDA_CHECK(cudaFreeAsync(d_sfa, stream));
    CUDA_CHECK(cudaFreeAsync(d_sfb, stream));
    CUDA_CHECK(cudaFreeAsync(d_src_offsets, stream));
    CUDA_CHECK(cudaFreeAsync(d_tokens, stream));
    CUDA_CHECK(cudaFreeAsync(d_tokens_padded, stream));
    CUDA_CHECK(cudaFreeAsync(d_packed_grad_offsets, stream));
    CUDA_CHECK(cudaFreeAsync(d_packed_input_offsets, stream));
    CUDA_CHECK(cudaFreeAsync(d_problem_shapes, stream));
    CUDA_CHECK(cudaFreeAsync(d_ptr_A, stream));
    CUDA_CHECK(cudaFreeAsync(d_ptr_B, stream));
    CUDA_CHECK(cudaFreeAsync(d_ptr_C, stream));
    CUDA_CHECK(cudaFreeAsync(d_ptr_D, stream));
    CUDA_CHECK(cudaFreeAsync(d_ptr_SFA, stream));
    CUDA_CHECK(cudaFreeAsync(d_ptr_SFB, stream));
    CUDA_CHECK(cudaFreeAsync(d_stride_A, stream));
    CUDA_CHECK(cudaFreeAsync(d_stride_B, stream));
    CUDA_CHECK(cudaFreeAsync(d_stride_C, stream));
    CUDA_CHECK(cudaFreeAsync(d_stride_D, stream));
    CUDA_CHECK(cudaFreeAsync(d_layout_SFA, stream));
    CUDA_CHECK(cudaFreeAsync(d_layout_SFB, stream));
    if (d_workspace) {
        CUDA_CHECK(cudaFreeAsync(d_workspace, stream));
    }
}

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

void moe_grouped_gemm_weight_grad_fp8_cutlass_sm100_grouped(nv_bfloat16* d_weight,
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
    using MmaTileShapeMNK = Shape<_128, _32, _128>;
    using ClusterShapeMNK = Shape<_1, _1, _1>;
    using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<1, 32, 128>;
    moe_grouped_gemm_weight_grad_fp8_cutlass_sm1xx_grouped<
        cutlass::arch::Sm100,
        MmaTileShapeMNK,
        ClusterShapeMNK,
        ScaleConfig,
        cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100>("SM100",
                                                                          d_weight,
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
}

#endif

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

void moe_grouped_gemm_weight_grad_fp8_cutlass_sm120_grouped(nv_bfloat16* d_weight,
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
    using MmaTileShapeMNK = Shape<_128, _32, _128>;
    using ClusterShapeMNK = Shape<_1, _1, _1>;
    using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 32, 128>;
    moe_grouped_gemm_weight_grad_fp8_cutlass_sm1xx_grouped<cutlass::arch::Sm120,
                                                           MmaTileShapeMNK,
                                                           ClusterShapeMNK,
                                                           ScaleConfig,
                                                           cutlass::epilogue::collective::EpilogueScheduleAuto,
                                                           cutlass::gemm::KernelScheduleSm120Blockwise>(
        "SM120",
        d_weight,
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
}

#endif

#endif  // SM100 / SM120 / SM121 supported

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

    if (props.major == 12) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
        // GeForce Blackwell (RTX 50, DGX Spark): native SM120 grouped FP8
        // blockwise tensor-core path.
        moe_grouped_gemm_weight_grad_fp8_cutlass_sm120_grouped(d_weight,
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
#elif defined(CUTLASS_ARCH_MMA_F16_SM89_SUPPORTED)
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
        throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8 SM120 CUTLASS support was not compiled");
#endif
    } else if (props.major == 10) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
        // Datacenter Blackwell (B200/B300): native SM100 grouped FP8
        // blockwise tensor-core path.
        moe_grouped_gemm_weight_grad_fp8_cutlass_sm100_grouped(d_weight,
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
#elif defined(CUTLASS_ARCH_MMA_F16_SM89_SUPPORTED)
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
        throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8 SM100 CUTLASS support was not compiled");
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
