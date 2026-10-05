// Hopper FP8 routed experts (see fp8_moe_sm90.h). The grouped GEMM is CUTLASS 3.x's sm90
// ptr-array kernel with blockwise scaling, configured as in CUTLASS's example 68
// (68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling, BSD-3-Clause), with
// vLLM's grouped-MoE conventions (csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x*.cu{h},
// Apache-2.0): problem shapes and operand pointers built per expert on the device from the
// expert offsets, and a swapped-operand variant for narrow rounds, where the weight rows fill
// the 128-row M tile and the few rows an expert receives sit on a narrow N tile.
//
// Both activation planes are quantised to E4M3 per row per 128 values with the dense FP8 path's
// rule (scale = amax / 448), and stored K-major, [rows][k / 128]: an expert's scales are then a
// contiguous run starting at its first row, whatever its row count, so its pointer is an offset
// and its stride is the round's.
//
// Isolated translation unit in the 90a-only archive, like the dense block-FP8 GEMM next door.

#include "ops/sparse_moe/fp8_sm90/fp8_moe_sm90.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"

namespace sinfer::ops::detail::fp8_moe_sm90 {
namespace {

using namespace cute;

constexpr int kBlock         = 128;
constexpr std::size_t kAlign = 256;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using GroupShape   = ProblemShape::UnderlyingProblemShape;

// clang-format off
template <class TileShape, bool SwapAB, bool Cooperative>
struct GroupedGemm {
  static constexpr bool kSwap = SwapAB;
  using ElementAB = cutlass::float_e4m3_t;
  using ElementD  = cutlass::bfloat16_t;

  // Not swapped: A is the expert's input rows [rows, k], B its weight [n, k] read as the
  // column-major (k, n) operand, D [rows, n]. Swapped: A is the weight [n, k], B the input rows,
  // and D the column-major (n, rows) view of the same [rows, n] block.
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = std::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  static constexpr int AlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

  // The weight's scales: one per 128 x 128 block, [n / 128][k / 128]. The input's: one per row
  // per 128, [rows][k / 128]. Both K-major.
  using ScaleConfig = std::conditional_t<SwapAB,
      cutlass::detail::Sm90BlockwiseScaleConfig<kBlock, 1, kBlock, cute::GMMA::Major::K, cute::GMMA::Major::K>,
      cutlass::detail::Sm90BlockwiseScaleConfig<1, kBlock, kBlock, cute::GMMA::Major::K, cute::GMMA::Major::K>>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ArchTag       = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ClusterShape  = Shape<_1, _1, _1>;
  // Ping-pong gives each consumer warpgroup whole tiles in turn; cooperative splits a tile's M
  // between the two, which a 128 x 128 tile needs: its FP32 accumulator and the blockwise
  // promotion's second one would not fit one warpgroup's registers.
  using KernelSchedule = std::conditional_t<Cooperative,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8BlockScaledAccum>;
  using EpilogueSchedule = std::conditional_t<Cooperative,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      void, LayoutD*, AlignmentD,
      ElementD, LayoutD*, AlignmentD,
      EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementD, float, void, float>
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementAB, cute::tuple<LayoutA*, LayoutSFA*>, AlignmentAB,
      ElementAB, cute::tuple<LayoutB*, LayoutSFB*>, AlignmentAB,
      float,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename GemmKernel::InternalStrideA;
  using StrideB = typename GemmKernel::InternalStrideB;
  using StrideD = typename GemmKernel::InternalStrideD;
  static_assert(std::is_same_v<typename GemmKernel::InternalStrideC, StrideD>);
};
// clang-format on

using Swap16  = GroupedGemm<Shape<_128, _16, _128>, true, false>;
using Swap32  = GroupedGemm<Shape<_128, _32, _128>, true, false>;
using Swap64  = GroupedGemm<Shape<_128, _64, _128>, true, false>;
using Wide128 = GroupedGemm<Shape<_128, _128, _128>, false, true>;

struct Hardware {
    int device   = -1;
    int cc       = 0;
    int sm_count = 0;
};

const Hardware& hardware() {
    static thread_local Hardware cached;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return cached; }
    if (cached.device != device) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        cached.device   = device;
        cached.cc       = major * 10 + minor;
        cached.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
    }
    return cached;
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("fp8_moe_sm90: ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

constexpr std::size_t round_up(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

/// A bump allocator over the workspace. With a null base it only measures, so the sizing and
/// the carving are the same code.
struct Bump {
    char* base       = nullptr;
    std::size_t used = 0;

    template <class T>
    T* take(std::size_t count) {
        used = round_up(used);
        T* p = base == nullptr ? nullptr : reinterpret_cast<T*>(base + used);
        used += count * sizeof(T);
        return p;
    }
};

/// One GEMM's per-expert argument arrays, filled on the device.
template <class Def>
struct GroupArgs {
    GroupShape* shapes                         = nullptr;
    const typename Def::ElementAB** a          = nullptr;
    const typename Def::ElementAB** b          = nullptr;
    const float** sfa                          = nullptr;
    const float** sfb                          = nullptr;
    typename Def::ElementD** d                 = nullptr;
    typename Def::StrideA* stride_a            = nullptr;
    typename Def::StrideB* stride_b            = nullptr;
    typename Def::StrideD* stride_d            = nullptr;
    typename Def::LayoutSFA* layout_sfa        = nullptr;
    typename Def::LayoutSFB* layout_sfb        = nullptr;
};

template <class Def>
GroupArgs<Def> carve_args(Bump& bump, int experts) {
    GroupArgs<Def> args;
    args.shapes     = bump.take<GroupShape>(experts);
    args.a          = bump.take<const typename Def::ElementAB*>(experts);
    args.b          = bump.take<const typename Def::ElementAB*>(experts);
    args.sfa        = bump.take<const float*>(experts);
    args.sfb        = bump.take<const float*>(experts);
    args.d          = bump.take<typename Def::ElementD*>(experts);
    args.stride_a   = bump.take<typename Def::StrideA>(experts);
    args.stride_b   = bump.take<typename Def::StrideB>(experts);
    args.stride_d   = bump.take<typename Def::StrideD>(experts);
    args.layout_sfa = bump.take<typename Def::LayoutSFA>(experts);
    args.layout_sfb = bump.take<typename Def::LayoutSFB>(experts);
    return args;
}

template <class Def>
std::size_t args_bytes(int experts) {
    Bump bump;
    carve_args<Def>(bump, experts);
    return round_up(bump.used);
}

std::size_t max_args_bytes(int experts) {
    return std::max({args_bytes<Swap16>(experts), args_bytes<Swap32>(experts),
                     args_bytes<Swap64>(experts), args_bytes<Wide128>(experts)});
}

/// CUTLASS's own scratch: per-SM TMA descriptors the kernel rewrites as it moves between experts.
/// It depends on the SM count only, not on the problem.
template <class Def>
std::size_t cutlass_bytes(int sm_count) {
    typename Def::Gemm::Arguments args{};
    args.mode             = cutlass::gemm::GemmUniversalMode::kGrouped;
    args.hw_info.sm_count = sm_count;
    return round_up(Def::Gemm::get_workspace_size(args));
}

std::size_t max_cutlass_bytes(int sm_count) {
    return std::max({cutlass_bytes<Swap16>(sm_count), cutlass_bytes<Swap32>(sm_count),
                     cutlass_bytes<Swap64>(sm_count), cutlass_bytes<Wide128>(sm_count)});
}

// One thread per expert: its problem (an empty expert is a zero-row problem the tile scheduler
// skips), its operand and scale pointers, and its strides and scale layouts.
template <class Def>
__global__ void fill_group_args_kernel(const int* __restrict__ offsets, int experts, int n, int k,
                                       const std::uint8_t* __restrict__ act,
                                       const float* __restrict__ act_scales,
                                       const std::uint8_t* __restrict__ weight,
                                       const float* __restrict__ weight_scales,
                                       __nv_bfloat16* __restrict__ out, GroupArgs<Def> args) {
    const int expert = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (expert >= experts) { return; }
    const int first   = offsets[expert];
    const int rows    = offsets[expert + 1] - first;
    const int kblocks = k / kBlock;

    using ElementAB          = typename Def::ElementAB;
    using ScaleConfig        = typename Def::ScaleConfig;
    const auto* act_e        = reinterpret_cast<const ElementAB*>(act + static_cast<std::int64_t>(first) * k);
    const float* act_scale_e = act_scales + static_cast<std::int64_t>(first) * kblocks;
    const auto* weight_e =
        reinterpret_cast<const ElementAB*>(weight + static_cast<std::int64_t>(expert) * n * k);
    const float* weight_scale_e =
        weight_scales + static_cast<std::int64_t>(expert) * (n / kBlock) * kblocks;

    args.d[expert] = reinterpret_cast<typename Def::ElementD*>(out + static_cast<std::int64_t>(first) * n);
    if constexpr (Def::kSwap) {
        args.shapes[expert]     = make_shape(n, rows, k);
        args.a[expert]          = weight_e;
        args.sfa[expert]        = weight_scale_e;
        args.b[expert]          = act_e;
        args.sfb[expert]        = act_scale_e;
        args.stride_a[expert]   = cutlass::make_cute_packed_stride(typename Def::StrideA{}, make_shape(n, k, 1));
        args.stride_b[expert]   = cutlass::make_cute_packed_stride(typename Def::StrideB{}, make_shape(rows, k, 1));
        args.stride_d[expert]   = cutlass::make_cute_packed_stride(typename Def::StrideD{}, make_shape(n, rows, 1));
        args.layout_sfa[expert] = ScaleConfig::tile_atom_to_shape_SFA(make_shape(n, rows, k, 1));
        args.layout_sfb[expert] = ScaleConfig::tile_atom_to_shape_SFB(make_shape(n, rows, k, 1));
    } else {
        args.shapes[expert]     = make_shape(rows, n, k);
        args.a[expert]          = act_e;
        args.sfa[expert]        = act_scale_e;
        args.b[expert]          = weight_e;
        args.sfb[expert]        = weight_scale_e;
        args.stride_a[expert]   = cutlass::make_cute_packed_stride(typename Def::StrideA{}, make_shape(rows, k, 1));
        args.stride_b[expert]   = cutlass::make_cute_packed_stride(typename Def::StrideB{}, make_shape(n, k, 1));
        args.stride_d[expert]   = cutlass::make_cute_packed_stride(typename Def::StrideD{}, make_shape(rows, n, 1));
        args.layout_sfa[expert] = ScaleConfig::tile_atom_to_shape_SFA(make_shape(rows, n, k, 1));
        args.layout_sfb[expert] = ScaleConfig::tile_atom_to_shape_SFB(make_shape(rows, n, k, 1));
    }
}

template <class Def>
void grouped_gemm(Bump& args_region, const int* offsets, int experts, int n, int k,
                  const std::uint8_t* act, const float* act_scales, const std::uint8_t* weight,
                  const float* weight_scales, __nv_bfloat16* out, void* cutlass_workspace,
                  std::size_t cutlass_capacity, cudaStream_t stream) {
    const GroupArgs<Def> args = carve_args<Def>(args_region, experts);
    fill_group_args_kernel<Def><<<(experts + 127) / 128, 128, 0, stream>>>(
        offsets, experts, n, k, act, act_scales, weight, weight_scales, out, args);
    check_cuda(cudaGetLastError(), "argument fill");

    using Gemm = typename Def::Gemm;
    const Hardware& hw = hardware();
    typename Gemm::Arguments arguments{};
    arguments.mode          = cutlass::gemm::GemmUniversalMode::kGrouped;
    // Device-side shapes only: the expert sizes are known on the device, never on the host.
    arguments.problem_shape = ProblemShape{experts, args.shapes, nullptr};
    arguments.mainloop      = {args.a,   args.stride_a,   args.b,   args.stride_b,
                               args.sfa, args.layout_sfa, args.sfb, args.layout_sfb};
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta  = 0.0f;
    arguments.epilogue.ptr_C        = nullptr;
    arguments.epilogue.dC           = args.stride_d;
    arguments.epilogue.ptr_D        = args.d;
    arguments.epilogue.dD           = args.stride_d;
    arguments.hw_info.device_id     = hw.device;
    arguments.hw_info.sm_count      = hw.sm_count;

    // One adapter per instantiation and thread, as for the dense kernel. `initialize` writes no
    // device memory here: the descriptors in the workspace are rebuilt by every launch, so the
    // round stays capture-safe with the workspace carved from the caller's arena.
    static thread_local Gemm gemm;
    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_moe_sm90: CUTLASS declined the grouped GEMM");
    }
    if (Gemm::get_workspace_size(arguments) > cutlass_capacity) {
        throw std::runtime_error("fp8_moe_sm90: CUTLASS workspace larger than reserved");
    }
    if (gemm.initialize(arguments, cutlass_workspace, stream) != cutlass::Status::kSuccess ||
        gemm.run(stream) != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_moe_sm90: grouped GEMM launch failed");
    }
}

// ---- activation planes: E4M3 per row per 128, a warp per (row, block) ----
constexpr int kQuantWarps = 8;

__device__ __forceinline__ void quantize_block(const float (&v)[4], std::uint8_t* codes,
                                               float* scale_out, int lane) {
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
    const float scale   = amax > 0.0f ? amax / 448.0f : 1.0f;
    const float inverse = 1.0f / scale;
    const __nv_fp8x2_storage_t lo =
        __nv_cvt_float2_to_fp8x2(make_float2(v[0] * inverse, v[1] * inverse), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t hi =
        __nv_cvt_float2_to_fp8x2(make_float2(v[2] * inverse, v[3] * inverse), __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<std::uint32_t*>(codes) =
        static_cast<std::uint32_t>(lo) | (static_cast<std::uint32_t>(hi) << 16);
    if (lane == 0) { *scale_out = scale; }
}

__device__ __forceinline__ void load_bf16x4(const __nv_bfloat16* p, float (&v)[4]) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const float2 a  = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
    const float2 b  = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
    v[0] = a.x;
    v[1] = a.y;
    v[2] = b.x;
    v[3] = b.y;
}

/// Packed column c reads its token's row of x: the gather and the quantisation in one pass.
__global__ __launch_bounds__(kQuantWarps * 32) void gather_quantize_kernel(
    const __nv_bfloat16* __restrict__ x, int hidden, const int* __restrict__ column_token,
    int columns, std::uint8_t* __restrict__ codes, float* __restrict__ scales) {
    const int kblocks      = hidden / kBlock;
    const std::int64_t item = static_cast<std::int64_t>(blockIdx.x) * kQuantWarps + (threadIdx.x >> 5);
    if (item >= static_cast<std::int64_t>(columns) * kblocks) { return; }
    const int column = static_cast<int>(item / kblocks);
    const int block  = static_cast<int>(item % kblocks);
    const int lane   = static_cast<int>(threadIdx.x & 31);
    const int token  = column_token[column];
    float v[4];
    load_bf16x4(x + static_cast<std::int64_t>(token) * hidden + block * kBlock + lane * 4, v);
    quantize_block(v, codes + static_cast<std::int64_t>(column) * hidden + block * kBlock + lane * 4,
                   scales + item, lane);
}

template <Activation kActivation>
__device__ __forceinline__ float gate(float g) {
    if constexpr (kActivation == Activation::GegluTanh) {
        const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        return 0.5f * g * (1.0f + tanhf(inner));
    } else {
        return g / (1.0f + __expf(-g));
    }
}

/// [rows][2I] gate/up product to the E4M3 [rows][I] down input, gate in [0, I), up in [I, 2I).
template <Activation kActivation>
__global__ __launch_bounds__(kQuantWarps * 32) void gate_quantize_kernel(
    const __nv_bfloat16* __restrict__ gate_up, int intermediate, int rows,
    std::uint8_t* __restrict__ codes, float* __restrict__ scales) {
    const int kblocks       = intermediate / kBlock;
    const std::int64_t item = static_cast<std::int64_t>(blockIdx.x) * kQuantWarps + (threadIdx.x >> 5);
    if (item >= static_cast<std::int64_t>(rows) * kblocks) { return; }
    const int row   = static_cast<int>(item / kblocks);
    const int block = static_cast<int>(item % kblocks);
    const int lane  = static_cast<int>(threadIdx.x & 31);
    const __nv_bfloat16* source =
        gate_up + static_cast<std::int64_t>(row) * 2 * intermediate + block * kBlock + lane * 4;
    float g[4];
    float u[4];
    load_bf16x4(source, g);
    load_bf16x4(source + intermediate, u);
    float h[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { h[i] = gate<kActivation>(g[i]) * u[i]; }
    quantize_block(h, codes + static_cast<std::int64_t>(row) * intermediate + block * kBlock + lane * 4,
                   scales + item, lane);
}

std::int64_t blocks_for(std::int64_t items) { return (items + kQuantWarps - 1) / kQuantWarps; }

// ---- narrow rounds: one GEMV per packed column, the weights dequantised as they stream ----
// A round of a few tokens touches a few experts, and the grouped GEMM's fixed cost (two
// argument fills, the scheduler walking every expert, per-expert TMA descriptors) is most of
// its time. Here each warp owns a few output rows of one column, reads its expert's codes once
// and keeps the activation in BF16, so nothing is quantised on the way. Columns that share an
// expert re-read its codes, mostly from L2.
constexpr int kGemvWarps    = 4;
constexpr int kGemvGateRows = 4; // intermediate rows per warp: 4 gate and 4 up weight rows
constexpr int kGemvDownRows = 8; // hidden rows per warp
constexpr int kGemvChunk    = 16; // E4M3 codes per lane per load
/// Up to this many columns `Auto` takes the GEMV path (op test sweep, H100 SXM, 35B-A3B shape).
constexpr std::int64_t kGemvAutoAssignments = 128;

/// The expert packed column `column` belongs to: the last e with offsets[e] <= column.
__device__ __forceinline__ int expert_of(const int* __restrict__ offsets, int experts, int column) {
    int lo = 0;
    int hi = experts - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (offsets[mid] <= column) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

__device__ __forceinline__ void load_bf16x16(const __nv_bfloat16* p, float (&v)[kGemvChunk]) {
    const uint4 a = __ldg(reinterpret_cast<const uint4*>(p));
    const uint4 b = __ldg(reinterpret_cast<const uint4*>(p) + 1);
    const std::uint32_t words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words[i]));
        v[2 * i]       = f.x;
        v[2 * i + 1]   = f.y;
    }
}

/// Sum of 16 E4M3 codes times 16 activations, unscaled.
__device__ __forceinline__ float dot16(const uint4 codes, const float (&v)[kGemvChunk]) {
    const std::uint32_t words[4] = {codes.x, codes.y, codes.z, codes.w};
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 lo = __half2(__nv_cvt_fp8x2_to_halfraw2(
            static_cast<__nv_fp8x2_storage_t>(words[i] & 0xFFFFu), __NV_E4M3));
        const __half2 hi = __half2(__nv_cvt_fp8x2_to_halfraw2(
            static_cast<__nv_fp8x2_storage_t>(words[i] >> 16), __NV_E4M3));
        const float2 a = __half22float2(lo);
        const float2 b = __half22float2(hi);
        acc = fmaf(a.x, v[4 * i], acc);
        acc = fmaf(a.y, v[4 * i + 1], acc);
        acc = fmaf(b.x, v[4 * i + 2], acc);
        acc = fmaf(b.y, v[4 * i + 3], acc);
    }
    return acc;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, o); }
    return v;
}

/// mid[c][j] = act(gate_j . x) * (up_j . x) for packed column c = blockIdx.y; a warp owns
/// kGemvGateRows consecutive j, all inside one 128-row scale block.
template <Activation kActivation>
__global__ __launch_bounds__(kGemvWarps * 32) void gemv_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, int hidden, int inter, const int* __restrict__ column_token,
    const int* __restrict__ offsets, int experts, const std::uint8_t* __restrict__ codes,
    const float* __restrict__ scales, __nv_bfloat16* __restrict__ mid) {
    const int column = static_cast<int>(blockIdx.y);
    const int lane   = static_cast<int>(threadIdx.x & 31);
    const int j0     = (static_cast<int>(blockIdx.x) * kGemvWarps + static_cast<int>(threadIdx.x >> 5)) *
                   kGemvGateRows;
    if (j0 >= inter) { return; }
    const int expert             = expert_of(offsets, experts, column);
    const __nv_bfloat16* x_row   = x + static_cast<std::int64_t>(column_token[column]) * hidden;
    const int kblocks            = hidden / kBlock;
    const std::uint8_t* gate_row = codes + (static_cast<std::int64_t>(expert) * 2 * inter + j0) * hidden;
    const std::uint8_t* up_row   = gate_row + static_cast<std::int64_t>(inter) * hidden;
    const float* expert_scales   = scales + static_cast<std::int64_t>(expert) * (2 * inter / kBlock) * kblocks;
    const float* gate_scales     = expert_scales + (j0 / kBlock) * kblocks;
    const float* up_scales       = expert_scales + ((inter + j0) / kBlock) * kblocks;

    float g[kGemvGateRows] = {};
    float u[kGemvGateRows] = {};
#pragma unroll 2
    for (int c = lane; c < hidden / kGemvChunk; c += 32) {
        float v[kGemvChunk];
        load_bf16x16(x_row + c * kGemvChunk, v);
        uint4 wg[kGemvGateRows];
        uint4 wu[kGemvGateRows];
#pragma unroll
        for (int r = 0; r < kGemvGateRows; ++r) {
            wg[r] = __ldg(reinterpret_cast<const uint4*>(gate_row + static_cast<std::int64_t>(r) * hidden) + c);
            wu[r] = __ldg(reinterpret_cast<const uint4*>(up_row + static_cast<std::int64_t>(r) * hidden) + c);
        }
        const int kb       = c / (kBlock / kGemvChunk);
        const float gscale = __ldg(gate_scales + kb);
        const float uscale = __ldg(up_scales + kb);
#pragma unroll
        for (int r = 0; r < kGemvGateRows; ++r) {
            g[r] = fmaf(gscale, dot16(wg[r], v), g[r]);
            u[r] = fmaf(uscale, dot16(wu[r], v), u[r]);
        }
    }
    float h = 0.0f;
#pragma unroll
    for (int r = 0; r < kGemvGateRows; ++r) {
        const float gate_sum = warp_sum(g[r]);
        const float up_sum   = warp_sum(u[r]);
        if (lane == r) { h = gate<kActivation>(gate_sum) * up_sum; }
    }
    if (lane < kGemvGateRows) {
        mid[static_cast<std::int64_t>(column) * inter + j0 + lane] = __float2bfloat16(h);
    }
}

/// out[c][n] = down_n . mid[c] for packed column c = blockIdx.y; a warp owns kGemvDownRows
/// consecutive n, all inside one 128-row scale block.
__global__ __launch_bounds__(kGemvWarps * 32) void gemv_down_kernel(
    const __nv_bfloat16* __restrict__ mid, int hidden, int inter, const int* __restrict__ offsets,
    int experts, const std::uint8_t* __restrict__ codes, const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ out) {
    const int column = static_cast<int>(blockIdx.y);
    const int lane   = static_cast<int>(threadIdx.x & 31);
    const int n0     = (static_cast<int>(blockIdx.x) * kGemvWarps + static_cast<int>(threadIdx.x >> 5)) *
                   kGemvDownRows;
    if (n0 >= hidden) { return; }
    const int expert            = expert_of(offsets, experts, column);
    const __nv_bfloat16* m_row  = mid + static_cast<std::int64_t>(column) * inter;
    const int kblocks           = inter / kBlock;
    const std::uint8_t* rows    = codes + (static_cast<std::int64_t>(expert) * hidden + n0) * inter;
    const float* row_scales     = scales + (static_cast<std::int64_t>(expert) * (hidden / kBlock) + n0 / kBlock) * kblocks;

    float acc[kGemvDownRows] = {};
#pragma unroll 2
    for (int c = lane; c < inter / kGemvChunk; c += 32) {
        float v[kGemvChunk];
        load_bf16x16(m_row + c * kGemvChunk, v);
        uint4 w[kGemvDownRows];
#pragma unroll
        for (int r = 0; r < kGemvDownRows; ++r) {
            w[r] = __ldg(reinterpret_cast<const uint4*>(rows + static_cast<std::int64_t>(r) * inter) + c);
        }
        const float scale = __ldg(row_scales + c / (kBlock / kGemvChunk));
#pragma unroll
        for (int r = 0; r < kGemvDownRows; ++r) { acc[r] = fmaf(scale, dot16(w[r], v), acc[r]); }
    }
    float result = 0.0f;
#pragma unroll
    for (int r = 0; r < kGemvDownRows; ++r) {
        const float sum = warp_sum(acc[r]);
        if (lane == r) { result = sum; }
    }
    if (lane < kGemvDownRows) {
        out[static_cast<std::int64_t>(column) * hidden + n0 + lane] = __float2bfloat16(result);
    }
}

void run_gemv(const Geometry& g, const __nv_bfloat16* x, int columns, const int* column_token,
              const int* offsets, const Fp8RoutedExperts& experts, __nv_bfloat16* mid,
              __nv_bfloat16* out, cudaStream_t stream) {
    constexpr int kThreads = kGemvWarps * 32;
    const dim3 gate_grid(static_cast<unsigned>((g.intermediate / kGemvGateRows + kGemvWarps - 1) / kGemvWarps),
                         static_cast<unsigned>(columns));
    if (g.activation == Activation::GegluTanh) {
        gemv_gate_up_kernel<Activation::GegluTanh><<<gate_grid, kThreads, 0, stream>>>(
            x, g.hidden, g.intermediate, column_token, offsets, g.experts, experts.gate_up_codes,
            experts.gate_up_scales, mid);
    } else {
        gemv_gate_up_kernel<Activation::Swiglu><<<gate_grid, kThreads, 0, stream>>>(
            x, g.hidden, g.intermediate, column_token, offsets, g.experts, experts.gate_up_codes,
            experts.gate_up_scales, mid);
    }
    check_cuda(cudaGetLastError(), "gemv gate/up");
    const dim3 down_grid(static_cast<unsigned>((g.hidden / kGemvDownRows + kGemvWarps - 1) / kGemvWarps),
                         static_cast<unsigned>(columns));
    gemv_down_kernel<<<down_grid, kThreads, 0, stream>>>(mid, g.hidden, g.intermediate, offsets,
                                                         g.experts, experts.down_codes,
                                                         experts.down_scales, out);
    check_cuda(cudaGetLastError(), "gemv down");
}

/// The workspace's fixed carve: the planes for `max_assignments` rows, two GEMMs' argument
/// arrays and CUTLASS's scratch.
struct Planes {
    std::uint8_t* act_codes  = nullptr;
    float* act_scales        = nullptr;
    std::uint8_t* mid_codes  = nullptr;
    float* mid_scales        = nullptr;
    __nv_bfloat16* gemv_mid  = nullptr;
    char* args[2]            = {};
    void* cutlass            = nullptr;
    std::size_t args_bytes   = 0;
    std::size_t cutlass_size = 0;
    std::size_t total        = 0;
};

Planes carve_planes(const Geometry& g, std::int64_t assignments, void* base) {
    const int sm_count = hardware().sm_count;
    Bump bump{static_cast<char*>(base)};
    Planes p;
    p.act_codes    = bump.take<std::uint8_t>(static_cast<std::size_t>(assignments) * g.hidden);
    p.act_scales   = bump.take<float>(static_cast<std::size_t>(assignments) * (g.hidden / kBlock));
    p.mid_codes    = bump.take<std::uint8_t>(static_cast<std::size_t>(assignments) * g.intermediate);
    p.mid_scales   = bump.take<float>(static_cast<std::size_t>(assignments) * (g.intermediate / kBlock));
    p.gemv_mid     = bump.take<__nv_bfloat16>(
        static_cast<std::size_t>(std::min<std::int64_t>(assignments, kGemvMaxAssignments)) * g.intermediate);
    p.args_bytes   = max_args_bytes(g.experts);
    p.args[0]      = bump.take<char>(p.args_bytes);
    p.args[1]      = bump.take<char>(p.args_bytes);
    p.cutlass_size = max_cutlass_bytes(sm_count);
    p.cutlass      = bump.take<char>(p.cutlass_size);
    p.total        = round_up(bump.used);
    return p;
}

Tile tile_override() {
    static const Tile forced = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_MOE_FP8_TILE");
        if (raw == nullptr) { return Tile::Auto; }
        const std::string_view v(raw);
        if (v == "16") { return Tile::Swap16; }
        if (v == "32") { return Tile::Swap32; }
        if (v == "64") { return Tile::Swap64; }
        if (v == "128") { return Tile::Wide128; }
        if (v == "gemv") { return Tile::Gemv; }
        return Tile::Auto;
    }();
    return forced;
}

/// By the mean rows per expert, the one width-dependent quantity the host knows. The thresholds
/// come from the op test's sweep on an H100 (test_sparse_moe_fp8_sm90).
Tile auto_tile(const Geometry& g, std::int32_t tokens) {
    const std::int64_t assignments = static_cast<std::int64_t>(tokens) * g.experts_per_token;
    if (assignments <= kGemvAutoAssignments) { return Tile::Gemv; }
    const std::int64_t rows_per_expert =
        static_cast<std::int64_t>(tokens) * g.experts_per_token / std::max(g.experts, 1);
    // Measured on an H100 SXM at the Qwen3.6-35B-A3B geometry (256 experts, top-8): past the
    // GEMV path the 16-row tile wins until about 8 rows per expert, 32 rows to 16, 64 rows
    // below 64, then the cooperative 128x128 tile.
    if (rows_per_expert < 8) { return Tile::Swap16; }
    if (rows_per_expert <= 16) { return Tile::Swap32; }
    if (rows_per_expert < 64) { return Tile::Swap64; }
    return Tile::Wide128;
}

template <class Fn>
void with_tile(Tile tile, Fn&& fn) {
    switch (tile) {
    case Tile::Swap16: fn.template operator()<Swap16>(); break;
    case Tile::Swap32: fn.template operator()<Swap32>(); break;
    case Tile::Swap64: fn.template operator()<Swap64>(); break;
    case Tile::Wide128: fn.template operator()<Wide128>(); break;
    case Tile::Gemv:
    case Tile::Auto: throw std::logic_error("fp8_moe_sm90: not a grouped-GEMM tile");
    }
}

} // namespace

bool available() noexcept { return hardware().cc == 90; }

bool supports(const Geometry& g) noexcept {
    return g.hidden > 0 && g.intermediate > 0 && g.experts > 0 && g.experts_per_token > 0 &&
           g.hidden % kBlock == 0 && g.intermediate % kBlock == 0;
}

Tile resolved_tile(const Geometry& geometry, std::int32_t tokens) {
    const std::int64_t assignments = static_cast<std::int64_t>(tokens) * geometry.experts_per_token;
    const Tile forced              = tile_override();
    if (forced != Tile::Auto && (forced != Tile::Gemv || assignments <= kGemvMaxAssignments)) {
        return forced;
    }
    return auto_tile(geometry, tokens);
}

std::size_t workspace_bytes(const Geometry& geometry, std::int32_t max_tokens) {
    if (!supports(geometry)) {
        throw std::invalid_argument("fp8_moe_sm90: hidden and intermediate must be multiples of 128");
    }
    const std::int64_t assignments =
        static_cast<std::int64_t>(std::max(max_tokens, 1)) * geometry.experts_per_token;
    return carve_planes(geometry, assignments, nullptr).total;
}

void run(const Geometry& geometry, const __nv_bfloat16* x, std::int32_t tokens,
         const std::int32_t* column_token, const std::int32_t* expert_offsets,
         const Fp8RoutedExperts& experts, void* workspace, std::size_t workspace_capacity,
         __nv_bfloat16* gate_up_scratch, __nv_bfloat16* out, cudaStream_t stream, Tile tile) {
    if (!available()) {
        throw std::runtime_error("fp8_moe_sm90: needs an sm_90 device and a build with 90a");
    }
    if (!supports(geometry)) {
        throw std::invalid_argument("fp8_moe_sm90: hidden and intermediate must be multiples of 128");
    }
    if (tokens <= 0) { return; }
    const std::int64_t assignments = static_cast<std::int64_t>(tokens) * geometry.experts_per_token;
    const Planes planes            = carve_planes(geometry, assignments, workspace);
    if (planes.total > workspace_capacity) {
        throw std::invalid_argument("fp8_moe_sm90: workspace smaller than workspace_bytes(tokens)");
    }

    const int hidden = geometry.hidden;
    const int inter  = geometry.intermediate;
    const int rows   = static_cast<int>(assignments);

    if (tile == Tile::Auto) { tile = resolved_tile(geometry, tokens); }
    if (tile == Tile::Gemv) {
        if (assignments > kGemvMaxAssignments) {
            throw std::invalid_argument("fp8_moe_sm90: the GEMV path takes at most kGemvMaxAssignments columns");
        }
        run_gemv(geometry, x, rows, column_token, expert_offsets, experts, planes.gemv_mid, out, stream);
        return;
    }

    gather_quantize_kernel<<<blocks_for(assignments * (hidden / kBlock)), kQuantWarps * 32, 0, stream>>>(
        x, hidden, column_token, rows, planes.act_codes, planes.act_scales);
    check_cuda(cudaGetLastError(), "gather");

    with_tile(tile, [&]<class Def>() {
        Bump region{planes.args[0]};
        grouped_gemm<Def>(region, expert_offsets, geometry.experts, 2 * inter, hidden,
                          planes.act_codes, planes.act_scales, experts.gate_up_codes,
                          experts.gate_up_scales, gate_up_scratch, planes.cutlass,
                          planes.cutlass_size, stream);
    });

    const auto gate_blocks = blocks_for(assignments * (inter / kBlock));
    if (geometry.activation == Activation::GegluTanh) {
        gate_quantize_kernel<Activation::GegluTanh><<<gate_blocks, kQuantWarps * 32, 0, stream>>>(
            gate_up_scratch, inter, rows, planes.mid_codes, planes.mid_scales);
    } else {
        gate_quantize_kernel<Activation::Swiglu><<<gate_blocks, kQuantWarps * 32, 0, stream>>>(
            gate_up_scratch, inter, rows, planes.mid_codes, planes.mid_scales);
    }
    check_cuda(cudaGetLastError(), "gate");

    with_tile(tile, [&]<class Def>() {
        Bump region{planes.args[1]};
        grouped_gemm<Def>(region, expert_offsets, geometry.experts, hidden, inter,
                          planes.mid_codes, planes.mid_scales, experts.down_codes,
                          experts.down_scales, out, planes.cutlass, planes.cutlass_size, stream);
    });
}

} // namespace sinfer::ops::detail::fp8_moe_sm90
