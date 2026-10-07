// sinfer::ops - the batch-consistent W8G32 route on Hopper's wgmma (see w8_rowsplit_wgmma_sm90.h).
//
// The medium-T kernel (w8_rowsplit_gemm_medium_t_splitk.cuh) and the pipelined one give each of
// four K-split warps one 64-wide K slice of every 256-wide group, chain that slice's four k16
// steps of mma.sync m16n8k16 into the warp's accumulators group after group, and combine the
// splits as ((s0 + s1) + (s2 + s3)). Here a warpgroup takes a split warp's place: its four warps
// hold the same weight fragments (each weight scaled in FP32 and rounded once to BF16) for 64 rows,
// and each k16 step is one wgmma m64nNk16 with A from those registers and the split's activations
// from shared memory. On an H100 a chained wgmma k16 step rounds exactly like mma.sync's (a probe
// over 16.8M outputs, PATCHES.md #114), so every output is the medium-T kernel's bits. What
// changes is a K step's cost: the activations are not loaded into registers once per row tile,
// and wgmma issues at Hopper's full tensor rate, where mma.sync m16n8k16 left a 64-column round
// on the vocabulary head compute-bound.
//
// Staging is warp-specialised, as Hopper GEMMs do it: one producer warp issues TMA loads of a
// group's codes, scales and activations into `Stages` buffers and the four warpgroups take them
// off full/empty mbarriers, so no barrier spans the CTA inside the K loop and a warpgroup runs up
// to `Stages` - 1 groups ahead of a slower one. TMA writes each tile with the 128-byte swizzle: the
// activations land as wgmma's K-major SW128 layout (64 K values a column, 16-byte chunks XORed by
// the column's low three bits), which is also how the mma.sync kernels stage them; the codes land
// as two 128-byte halves of each row's 256-byte group, chunks XORed by the row's low three bits.

#include "ops/linear/w8/w8_rowsplit_wgmma_sm90.h"

#include "core/device.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace sinfer::ops::detail {
namespace {

template <int TileCols, int Stages>
struct W8WgmmaLayout {
    static constexpr int kKSplits    = 4;
    static constexpr int kTileK      = 64;
    static constexpr int kGroupK     = kKSplits * kTileK;
    static constexpr int kRows       = 64; // a CTA's rows: one wgmma M
    static constexpr int kConsumers  = 128 * kKSplits; // a warpgroup a K split
    static constexpr int kThreads    = kConsumers + 32; // and the warp that issues the loads
    static constexpr int kScaleBytes = kGroupK / 32 * 2;
    static constexpr int kFragments  = TileCols / 8; // float4 accumulators a thread
    static constexpr std::size_t kSplitActBytes = std::size_t{TileCols} * kTileK * 2;
    static constexpr std::size_t kActBytes      = kKSplits * kSplitActBytes;
    static constexpr std::size_t kCodeHalfBytes = std::size_t{kRows} * (kGroupK / 2);
    static constexpr std::size_t kCodeBytes     = 2 * kCodeHalfBytes;
    static constexpr std::size_t kScaleStage    = std::size_t{kRows} * kScaleBytes;
    static constexpr std::size_t kActAt         = 0;
    static constexpr std::size_t kCodesAt       = kActAt + Stages * kActBytes;
    static constexpr std::size_t kScalesAt      = kCodesAt + Stages * kCodeBytes;
    static constexpr std::size_t kBarriersAt    = kScalesAt + Stages * kScaleStage;
    static constexpr std::size_t kUsed          = kBarriersAt + 2 * Stages * sizeof(std::uint64_t);
    static constexpr std::size_t kBytes         = kUsed + 1024; // the base is aligned up to 1024
    static constexpr std::uint32_t kStageBytes =
        static_cast<std::uint32_t>(kActBytes + kCodeBytes + kScaleStage);
    // The 128-byte swizzle repeats every 1024 bytes, so every swizzled tile starts on one.
    static_assert(kSplitActBytes % 1024 == 0 && kCodeHalfBytes % 1024 == 0 && kScaleBytes == 16);
    // The split partial sums fit in the stages after the K loop.
    static_assert(std::size_t{kKSplits} * 128 * kFragments * 16 <= kBarriersAt);
};

// The kernel's TMA descriptors: activations (BF16, a box of 64 K x the tile's columns), codes
// (a box of 128 K bytes x the CTA's rows) and scales (a row's 16 bytes over one group).
struct alignas(64) W8WgmmaMaps {
    CUtensorMap x, codes, scales;
};

#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define SINFER_W8_WGMMA_DEVICE 1
#endif

#ifdef SINFER_W8_WGMMA_DEVICE
__device__ __forceinline__ void fence_operand(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void fence_operand(unsigned& r) { asm volatile("" : "+r"(r)::"memory"); }

// K-major, 128-byte swizzle, 8-row groups 1024 bytes apart (DeepGEMM's make_smem_desc for K-major).
__device__ __forceinline__ std::uint64_t sw128_desc(const void* smem) {
    const auto addr = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
    return static_cast<std::uint64_t>((addr & 0x3FFFF) >> 4) |
           (static_cast<std::uint64_t>(1024 >> 4) << 32) | (std::uint64_t{1} << 62);
}

__device__ __forceinline__ void mbar_init(std::uint64_t* barrier, std::uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(barrier)), "r"(arrivals)
                 : "memory");
}

__device__ __forceinline__ void mbar_wait(std::uint64_t* barrier, std::uint32_t parity) {
    asm volatile("{\n.reg .pred done;\n"
                 "wait_loop:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                 "@done bra wait_done;\n"
                 "bra wait_loop;\n"
                 "wait_done:\n}\n" ::"r"(smem_addr(barrier)),
                 "r"(parity), "r"(0x989680)
                 : "memory");
}

__device__ __forceinline__ void mbar_arrive(std::uint64_t* barrier) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_addr(barrier)) : "memory");
}

__device__ __forceinline__ void mbar_expect(std::uint64_t* barrier, std::uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(barrier)),
                 "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void tma_load(void* dst, const CUtensorMap* map, int inner, int outer,
                                         std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];" ::"r"(smem_addr(dst)),
                 "l"(map), "r"(inner), "r"(outer), "r"(smem_addr(barrier))
                 : "memory");
}

template <int N>
struct Wgmma;

// D[64 x N] += A[64 x 16] (registers) x B[16 x N] (shared memory), FP32 accumulators.
template <>
struct Wgmma<16> {
    static __device__ __forceinline__ void run(float* d, const unsigned* a, std::uint64_t desc) {
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %12, p, 1, 1, 0;\n}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                       "+f"(d[6]), "+f"(d[7])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
    }
};

template <>
struct Wgmma<32> {
    static __device__ __forceinline__ void run(float* d, const unsigned* a, std::uint64_t desc) {
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, "
                     "{%16,%17,%18,%19}, %20, p, 1, 1, 0;\n}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                       "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
                       "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
    }
};

template <>
struct Wgmma<64> {
    static __device__ __forceinline__ void run(float* d, const unsigned* a, std::uint64_t desc) {
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
                     "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                     "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
                     "{%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                       "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
                       "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
                       "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                       "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
                       "+f"(d[30]), "+f"(d[31])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
    }
};
#endif

template <int TileCols, int Stages>
__global__ __launch_bounds__(W8WgmmaLayout<TileCols, Stages>::kThreads, 1) void
w8_rowsplit_wgmma_kernel(const __grid_constant__ W8WgmmaMaps maps, __nv_bfloat16* __restrict__ out,
                         int n, int active_cols, int hidden) {
#ifdef SINFER_W8_WGMMA_DEVICE
    using Layout             = W8WgmmaLayout<TileCols, Stages>;
    constexpr int kTileK     = Layout::kTileK;
    constexpr int kGroupK    = Layout::kGroupK;
    constexpr int kConsumers = Layout::kConsumers;
    constexpr int kAcc       = TileCols / 2; // accumulators a thread
    constexpr unsigned kMask = 0xffffffffu;
    static_assert(Stages >= 2 && Stages <= 6);

    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem = reinterpret_cast<unsigned char*>(
        (reinterpret_cast<std::uintptr_t>(smem_raw) + 1023) & ~std::uintptr_t{1023});
    auto* full  = reinterpret_cast<std::uint64_t*>(smem + Layout::kBarriersAt);
    auto* empty = full + Stages;

    const int kGroups      = hidden / kGroupK;
    const int column_begin = static_cast<int>(blockIdx.y) * TileCols;
    active_cols = min(TileCols, active_cols - column_begin);
    const int tid      = static_cast<int>(threadIdx.x);
    const int cta_row0 = static_cast<int>(blockIdx.x) * Layout::kRows;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < Stages; ++s) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], kConsumers / 32); // every consumer warp releases a stage
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (tid >= kConsumers) {
        // The producer: one lane refills each stage once all 16 consumer warps released it.
        if (tid == kConsumers) {
            for (int group_index = 0; group_index < kGroups; ++group_index) {
                const int buf = group_index % Stages;
                mbar_wait(&empty[buf], ((group_index / Stages) & 1) ^ 1);
                mbar_expect(&full[buf], Layout::kStageBytes);
                const int k0 = group_index * kGroupK;
                unsigned char* act = smem + Layout::kActAt + buf * Layout::kActBytes;
#pragma unroll
                for (int split = 0; split < Layout::kKSplits; ++split) {
                    tma_load(act + split * Layout::kSplitActBytes, &maps.x, k0 + split * kTileK,
                             column_begin, &full[buf]);
                }
                unsigned char* codes = smem + Layout::kCodesAt + buf * Layout::kCodeBytes;
                tma_load(codes, &maps.codes, k0, cta_row0, &full[buf]);
                tma_load(codes + Layout::kCodeHalfBytes, &maps.codes, k0 + kGroupK / 2, cta_row0,
                         &full[buf]);
                tma_load(smem + Layout::kScalesAt + buf * Layout::kScaleStage, &maps.scales,
                         group_index * Layout::kScaleBytes, cta_row0, &full[buf]);
            }
        }
        return;
    }

    const int k_split   = tid >> 7;
    const int wg_tid    = tid & 127;
    const int warp      = wg_tid >> 5; // within the warpgroup: its 16 of the 64 rows
    const int lane      = tid & 31;
    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int top_row   = warp * 16 + gid;
    const int half_koff = (k_split & 1) * kTileK; // the split's slice within its 128-byte half

    float acc[kAcc];
#pragma unroll
    for (int i = 0; i < kAcc; ++i) { acc[i] = 0.0f; }
    const auto fence_acc = [&] {
#pragma unroll
        for (int i = 0; i < kAcc; ++i) { fence_operand(acc[i]); }
    };

    // Each group's weight fragments (the pipelined kernel's, row for row) go to one of two
    // register sets in turn, so a group is dequantised while the previous group's wgmmas still
    // run. Waiting for those (wait_group 1) is what frees both their registers and their stage.
    unsigned af[2][4][4];
    const auto fence_frags = [&](auto& frags) {
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { fence_operand(frags[ks][i]); }
        }
    };
    const auto run_group = [&](auto& frags, auto& previous, int group_index) {
        const int buf = group_index % Stages;
        mbar_wait(&full[buf], (group_index / Stages) & 1);

        const unsigned char* code_half = smem + Layout::kCodesAt + buf * Layout::kCodeBytes +
                                         (k_split >> 1) * Layout::kCodeHalfBytes;
        const unsigned char* scale_stage = smem + Layout::kScalesAt + buf * Layout::kScaleStage;
        unsigned lane_scale_pair = 0;
        if (lid < 2) {
            lane_scale_pair = *reinterpret_cast<const unsigned*>(
                scale_stage + (top_row + lid * 8) * Layout::kScaleBytes + k_split * 4);
        }
        const unsigned top_scale_pair = __shfl_sync(kMask, lane_scale_pair, lane & ~3);
        const unsigned bot_scale_pair = __shfl_sync(kMask, lane_scale_pair, (lane & ~3) + 1);
        const auto load_code_pair = [&](int code_row, int col) {
            const int byte   = half_koff + col;
            const int offset = code_row * (kGroupK / 2) + (((byte >> 4) ^ (code_row & 7)) << 4) + (byte & 15);
            return static_cast<unsigned>(*reinterpret_cast<const unsigned short*>(code_half + offset));
        };
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            // K steps 0-1 take the first of the slice's two 32-wide scale groups.
            const unsigned top_bits = ks < 2 ? top_scale_pair & 0xffffu : top_scale_pair >> 16;
            const unsigned bot_bits = ks < 2 ? bot_scale_pair & 0xffffu : bot_scale_pair >> 16;
            const float top_scale   = __half2float(__ushort_as_half(top_bits));
            const float bot_scale   = __half2float(__ushort_as_half(bot_bits));
            const int code_col      = ks * 16 + lid * 2;
            auto& frag              = frags[ks];
            frag[0] = w8_small_t_bf16_pair_from_s8(load_code_pair(top_row, code_col), top_scale);
            frag[1] = w8_small_t_bf16_pair_from_s8(load_code_pair(top_row + 8, code_col), bot_scale);
            frag[2] = w8_small_t_bf16_pair_from_s8(load_code_pair(top_row, code_col + 8), top_scale);
            frag[3] = w8_small_t_bf16_pair_from_s8(load_code_pair(top_row + 8, code_col + 8), bot_scale);
        }
        const std::uint64_t desc = sw128_desc(smem + Layout::kActAt + buf * Layout::kActBytes +
                                              k_split * Layout::kSplitActBytes);
        fence_acc();
        fence_frags(frags);
        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            // The step's 16 K values sit 32 bytes into each 128-byte column row.
            Wgmma<TileCols>::run(acc, frags[ks], desc + static_cast<std::uint64_t>(2 * ks));
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        fence_acc();
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fence_acc();
        fence_frags(previous);
        if (group_index > 0) {
            // The warp's reads of the previous group's stage are done: release it.
            __syncwarp();
            if (lane == 0) { mbar_arrive(&empty[(group_index - 1) % Stages]); }
        }
    };
    for (int group_index = 0; group_index < kGroups; group_index += 2) {
        run_group(af[0], af[1], group_index);
        if (group_index + 1 < kGroups) { run_group(af[1], af[0], group_index + 1); }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    fence_acc();
    fence_frags(af[0]);
    fence_frags(af[1]);

    // Every consumer is past the K loop, and every load it waited for has landed: the stages
    // are free for the partial sums. The producer warp has exited, so the consumers sync on
    // their own barrier.
    asm volatile("bar.sync 1, %0;" ::"n"(kConsumers) : "memory");
    // The medium-T kernel's split combine, warpgroup for warp: ((s0 + s1) + (s2 + s3)).
    // Fragment f is accumulators 4 * f .. + 3: rows gid and gid + 8 of the warp's 16, columns
    // 8 * f + 2 * lid and the next.
    auto* partial   = reinterpret_cast<float*>(smem);
    const auto slot = [&](int split, int f) {
        return partial + ((split * Layout::kFragments + f) * 128 + wg_tid) * 4;
    };
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int f = 0; f < Layout::kFragments; ++f) {
            store_vec(slot(k_split, f), make_float4(acc[4 * f], acc[4 * f + 1], acc[4 * f + 2], acc[4 * f + 3]));
        }
    }
    asm volatile("bar.sync 1, %0;" ::"n"(kConsumers) : "memory");
    if ((k_split & 1) == 0) {
#pragma unroll
        for (int f = 0; f < Layout::kFragments; ++f) {
            const float4 partner = load_vec<float4>(slot(k_split + 1, f));
            acc[4 * f] += partner.x;
            acc[4 * f + 1] += partner.y;
            acc[4 * f + 2] += partner.z;
            acc[4 * f + 3] += partner.w;
            if (k_split != 0) {
                store_vec(slot(k_split, f), make_float4(acc[4 * f], acc[4 * f + 1], acc[4 * f + 2], acc[4 * f + 3]));
            }
        }
    }
    asm volatile("bar.sync 1, %0;" ::"n"(kConsumers) : "memory");
    if (k_split == 0) {
        const auto store = [&](int row, int col, float value) {
            out[static_cast<std::int64_t>(column_begin + col) * n + row] = __float2bfloat16_rn(value);
        };
        const int row0 = cta_row0 + top_row;
#pragma unroll
        for (int f = 0; f < Layout::kFragments; ++f) {
            const float4 partner = load_vec<float4>(slot(2, f));
            acc[4 * f] += partner.x;
            acc[4 * f + 1] += partner.y;
            acc[4 * f + 2] += partner.z;
            acc[4 * f + 3] += partner.w;
            const int col0 = f * 8 + 2 * lid;
            if (col0 < active_cols) {
                store(row0, col0, acc[4 * f]);
                store(row0 + 8, col0, acc[4 * f + 2]);
            }
            if (col0 + 1 < active_cols) {
                store(row0, col0 + 1, acc[4 * f + 1]);
                store(row0 + 8, col0 + 1, acc[4 * f + 3]);
            }
        }
    }
#elif defined(__CUDA_ARCH__)
    __trap();
#endif
}

struct Hardware {
    int cc = 0, smem_optin = 0;
};

const Hardware& hardware() {
    static std::mutex mutex;
    static std::map<int, Hardware> cache;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = cache.find(device); found != cache.end()) { return found->second; }
    Hardware h;
    int major = 0, minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&h.smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    h.cc = major * 10 + minor;
    return cache[device] = h;
}

bool encode(CUtensorMap& map, CUtensorMapDataType type, const void* base, std::uint64_t inner,
            std::uint64_t outer, std::uint64_t outer_stride_bytes, std::uint32_t box_inner,
            std::uint32_t box_outer, CUtensorMapSwizzle swizzle) {
    const cuuint64_t dims[2]          = {inner, outer};
    const cuuint64_t strides[1]       = {outer_stride_bytes};
    const cuuint32_t box[2]           = {box_inner, box_outer};
    const cuuint32_t element_steps[2] = {1, 1};
    return cuTensorMapEncodeTiled(&map, type, 2, const_cast<void*>(base), dims, strides, box,
                                  element_steps, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <int TileCols, int Stages>
bool launch(const W8WgmmaProblem& p, cudaStream_t stream) {
    using Layout = W8WgmmaLayout<TileCols, Stages>;
    if (p.n % Layout::kRows != 0 ||
        Layout::kBytes > static_cast<std::size_t>(hardware().smem_optin)) {
        return false;
    }
    // Activations: 64 K values x the tile's columns a box (columns past the round read as zero).
    // Codes: 128 K bytes x the CTA's rows, two boxes a group. Scales: a row's 16 bytes a group.
    W8WgmmaMaps maps{};
    if (!encode(maps.x, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.x, p.k, p.columns,
                std::uint64_t{2} * p.k, Layout::kTileK, TileCols, CU_TENSOR_MAP_SWIZZLE_128B) ||
        !encode(maps.codes, CU_TENSOR_MAP_DATA_TYPE_UINT8, p.codes, p.k, p.n, p.k,
                Layout::kGroupK / 2, Layout::kRows, CU_TENSOR_MAP_SWIZZLE_128B) ||
        !encode(maps.scales, CU_TENSOR_MAP_DATA_TYPE_UINT8, p.scales, p.k / 16, p.n, p.k / 16,
                Layout::kScaleBytes, Layout::kRows, CU_TENSOR_MAP_SWIZZLE_NONE)) {
        return false;
    }
    auto* kernel = w8_rowsplit_wgmma_kernel<TileCols, Stages>;
    // The attribute is per device; setting it again is cheap and keeps a second device right.
    static std::mutex mutex;
    static std::map<int, bool> set;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!set[device]) {
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(Layout::kBytes)));
            set[device] = true;
        }
    }
    const dim3 grid(static_cast<unsigned>(p.n / Layout::kRows),
                    static_cast<unsigned>((p.columns + TileCols - 1) / TileCols));
    kernel<<<grid, Layout::kThreads, Layout::kBytes, stream>>>(maps, p.out, p.n, p.columns, p.k);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool launch_columns(const W8WgmmaProblem& p, cudaStream_t stream) {
    // Stages by what fits the 227 KiB an H100 block may opt into: a group is 25 KiB at 16
    // columns, 33 at 32 and 49 at 64.
    if (p.columns <= 16) { return launch<16, 4>(p, stream); }
    if (p.columns <= 32) { return launch<32, 4>(p, stream); }
    return launch<64, 4>(p, stream);
}

bool enabled_by_env() {
    static const bool enabled = [] {
        const char* value = std::getenv("SUROGATE_SERVE_W8_WGMMA");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; }

} // namespace

bool w8_wgmma_available() noexcept {
    if (!enabled_by_env()) { return false; }
    try {
        return hardware().cc == 90;
    } catch (...) {
        return false;
    }
}

std::int32_t w8_wgmma_min_columns() noexcept {
    static const std::int32_t value = [] {
        const char* text = std::getenv("SUROGATE_SERVE_W8_WGMMA_MIN_COLUMNS");
        if (text == nullptr || *text == '\0') { return 33; }
        char* end         = nullptr;
        const long parsed = std::strtol(text, &end, 10);
        return (end != nullptr && *end == '\0' && parsed >= 1 && parsed <= 65)
                   ? static_cast<std::int32_t>(parsed) : 33;
    }();
    return value;
}

bool w8_wgmma_consistent(const W8WgmmaProblem& p, cudaStream_t stream) {
    if (!w8_wgmma_available() || p.k <= 0 || p.k % 256 != 0 || p.n <= 0 || p.n % 64 != 0 ||
        p.columns < 1 || p.columns > 64 || !aligned16(p.x) || !aligned16(p.codes) ||
        !aligned16(p.scales) || p.out == nullptr) {
        return false;
    }
    return launch_columns(p, stream);
}

} // namespace sinfer::ops::detail
