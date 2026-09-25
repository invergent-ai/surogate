// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

// bf16 grouped GEMM for the MoE experts on CUTLASS's grouped kernel: one launch for all experts, no
// library lock (moe_expert_gemms in moe_common.cuh says why not cuBLAS grouped GEMM), and one tile
// configuration for every expert. The last point keeps a token's result independent of how many
// other tokens its expert received: each output element sums over K in the same order whatever the
// problem's N, which row packing relies on (a packed row must match the row alone bit for bit).
// There is no split-K, so the result is also the same run to run.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Not moe_common.cuh: kernels.h declares a global Tensor, which collides with cute::Tensor inside the
// CUTLASS headers. The declaration this file defines is in moe_common.cuh.
#include "utilities/utils.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

namespace {

using cutlass::layout::ColumnMajor;
using cutlass::layout::RowMajor;

// cuBLAS column-major convention throughout: C (m x n, column-major) = op(A) (m x k) * op(B) (k x n).
// op = N on a column-major operand is a ColumnMajor layout; op = T is the same bytes as RowMajor.
template <typename LayoutA, typename LayoutB>
using GroupedGemm = cutlass::gemm::device::GemmGrouped<typename cutlass::gemm::kernel::DefaultGemmGrouped<
    cutlass::bfloat16_t,
    LayoutA,
    cutlass::ComplexTransform::kNone,
    8,
    cutlass::bfloat16_t,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    8,
    cutlass::bfloat16_t,
    ColumnMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 8, float, float>,
    // Unused by the grouped kernels (CUTLASS example 24); required by the template.
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel>;

// Per-call device arguments, staged through a ring of kSlots pinned host and device slots per thread and
// device: one H2D copy per call. Each slot carries an event recorded behind the GEMM that reads it, and
// is reused kSlots calls later only once that event has completed, whatever stream either call used.
// The host therefore waits only when it runs kSlots grouped GEMMs ahead of its own GPU, and it holds no
// lock while it waits. The wait polls (as stream_wait_spin in ep_strategy.cpp does): a blocking driver
// wait showed rare lost-wakeup stalls under this process's 8-worker-thread load. No stream handle is
// kept, so a stream destroyed since (a trainer rebuilt on the same thread) is never touched.
struct Ring {
    static constexpr int kSlots = 256;
    std::byte* pinned = nullptr;  // kSlots * slot_bytes, one allocation
    std::byte* device = nullptr;  // kSlots * slot_bytes, one allocation
    std::size_t slot_bytes = 0;
    cudaEvent_t read_done[kSlots] = {};  // recorded after the GEMM that read the slot
    bool pending[kSlots] = {};
    int next = 0;
    ~Ring() {
        // Best effort at thread exit: the driver may already be torn down.
        for (auto& e : read_done)
            if (e) (void)cudaEventDestroy(e);
        if (pinned) (void)cudaFreeHost(pinned);
        if (device) (void)cudaFree(device);
    }
};

void event_wait_spin(cudaEvent_t event) {
    while (true) {
        const cudaError_t st = cudaEventQuery(event);
        if (st == cudaSuccess) return;
        if (st != cudaErrorNotReady) CUDA_CHECK(st);
        (void)cudaGetLastError();
        std::this_thread::yield();
    }
}

Ring& ring() {
    constexpr int kMaxDevices = 64;
    static thread_local std::unique_ptr<Ring> rings[kMaxDevices];
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (dev < 0 || dev >= kMaxDevices) throw std::runtime_error("MoE grouped GEMM: device index out of range");
    if (!rings[dev]) rings[dev] = std::make_unique<Ring>();
    return *rings[dev];
}

// Copies `bytes` of host data to the next slot in stream order and returns the slot index. The caller
// launches the GEMM that reads the slot, then calls release() on the same stream.
int stage(Ring& r, const std::byte* host, std::size_t bytes, cudaStream_t stream) {
    if (bytes > r.slot_bytes) {
        // First use, or more experts with tokens than ever before on this thread (32 KB covers 546).
        for (int i = 0; i < Ring::kSlots; ++i) {
            if (r.pending[i]) event_wait_spin(r.read_done[i]);
            r.pending[i] = false;
        }
        if (r.pinned) CUDA_CHECK(cudaFreeHost(r.pinned));
        if (r.device) CUDA_CHECK(cudaFree(r.device));
        r.pinned = r.device = nullptr;
        std::size_t grown = std::max<std::size_t>({32768, 2 * r.slot_bytes, (bytes + 4095) / 4096 * 4096});
        r.slot_bytes = 0;
        CUDA_CHECK(cudaHostAlloc(&r.pinned, Ring::kSlots * grown, cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(&r.device, Ring::kSlots * grown));
        r.slot_bytes = grown;
    }
    const int i = r.next;
    r.next = (r.next + 1) % Ring::kSlots;
    if (r.pending[i]) {
        event_wait_spin(r.read_done[i]);
        r.pending[i] = false;
    }
    if (!r.read_done[i]) CUDA_CHECK(cudaEventCreateWithFlags(&r.read_done[i], cudaEventDisableTiming));
    std::byte* h = r.pinned + static_cast<std::size_t>(i) * r.slot_bytes;
    std::memcpy(h, host, bytes);
    CUDA_CHECK(cudaMemcpyAsync(r.device + static_cast<std::size_t>(i) * r.slot_bytes,
                               h,
                               bytes,
                               cudaMemcpyHostToDevice,
                               stream));
    return i;
}

void release(Ring& r, int i, cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(r.read_done[i], stream));
    r.pending[i] = true;
}

// Resident threadblocks per device for one kernel (CUTLASS computes it from the occupancy).
template <typename Gemm>
int threadblock_count() {
    static std::array<std::atomic<int>, 64> cache{};
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (dev < 0 || dev >= static_cast<int>(cache.size())) return Gemm::sufficient();
    int n = cache[dev].load(std::memory_order_relaxed);
    if (n <= 0) {
        n = Gemm::sufficient();
        cache[dev].store(n, std::memory_order_relaxed);
    }
    return n;
}

inline std::size_t align16(std::size_t x) {
    return (x + 15) / 16 * 16;
}

template <typename LayoutA, typename LayoutB>
void run_grouped(const std::vector<int>& m,
                 const std::vector<int>& n,
                 const std::vector<int>& k,
                 float alpha,
                 const std::vector<const nv_bfloat16*>& A,
                 const std::vector<int>& lda,
                 const std::vector<const nv_bfloat16*>& B,
                 const std::vector<int>& ldb,
                 float beta,
                 const std::vector<nv_bfloat16*>& C,
                 const std::vector<int>& ldc,
                 cudaStream_t stream) {
    using Gemm = GroupedGemm<LayoutA, LayoutB>;
    using Element = cutlass::bfloat16_t;
    const std::size_t count = m.size();

    // One blob: problem sizes, then the A, B, C pointer arrays, then lda, ldb, ldc (int64).
    const std::size_t off_ptr_a = align16(count * sizeof(cutlass::gemm::GemmCoord));
    const std::size_t off_ptr_b = off_ptr_a + align16(count * sizeof(void*));
    const std::size_t off_ptr_c = off_ptr_b + align16(count * sizeof(void*));
    const std::size_t off_lda = off_ptr_c + align16(count * sizeof(void*));
    const std::size_t off_ldb = off_lda + align16(count * sizeof(int64_t));
    const std::size_t off_ldc = off_ldb + align16(count * sizeof(int64_t));
    const std::size_t bytes = off_ldc + align16(count * sizeof(int64_t));
    thread_local std::vector<std::byte> host;
    host.assign(bytes, std::byte{0});
    auto* sizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(host.data());
    auto* pa = reinterpret_cast<const void**>(host.data() + off_ptr_a);
    auto* pb = reinterpret_cast<const void**>(host.data() + off_ptr_b);
    auto* pc = reinterpret_cast<void**>(host.data() + off_ptr_c);
    auto* la = reinterpret_cast<int64_t*>(host.data() + off_lda);
    auto* lb = reinterpret_cast<int64_t*>(host.data() + off_ldb);
    auto* lc = reinterpret_cast<int64_t*>(host.data() + off_ldc);
    for (std::size_t i = 0; i < count; ++i) {
        sizes[i] = cutlass::gemm::GemmCoord(m[i], n[i], k[i]);
        pa[i] = A[i];
        pb[i] = B[i];
        pc[i] = C[i];
        la[i] = lda[i];
        lb[i] = ldb[i];
        lc[i] = ldc[i];
    }
    Ring& r = ring();
    const int slot = stage(r, host.data(), bytes, stream);
    std::byte* dev = r.device + static_cast<std::size_t>(slot) * r.slot_bytes;

    typename Gemm::EpilogueOutputOp::Params epilogue(alpha, beta);
    typename Gemm::Arguments args(reinterpret_cast<cutlass::gemm::GemmCoord*>(dev),
                                  static_cast<int>(count),
                                  threadblock_count<Gemm>(),
                                  epilogue,
                                  reinterpret_cast<Element**>(dev + off_ptr_a),
                                  reinterpret_cast<Element**>(dev + off_ptr_b),
                                  reinterpret_cast<Element**>(dev + off_ptr_c),
                                  reinterpret_cast<Element**>(dev + off_ptr_c),
                                  reinterpret_cast<int64_t*>(dev + off_lda),
                                  reinterpret_cast<int64_t*>(dev + off_ldb),
                                  reinterpret_cast<int64_t*>(dev + off_ldc),
                                  reinterpret_cast<int64_t*>(dev + off_ldc));
    Gemm gemm;
    cutlass::Status status = gemm.initialize(args, nullptr, stream);
    if (status == cutlass::Status::kSuccess) status = gemm.run(stream);
    release(r, slot, stream);  // after the H2D copy in any case, and after the GEMM when it launched
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string("MoE grouped GEMM: CUTLASS failed: ") +
                                 cutlass::cutlassGetStatusString(status));
    }
}

// The kernels load 8 bf16 (16 bytes) at a time along each operand's contiguous dimension.
bool aligned8(int x) {
    return x % 8 == 0;
}
bool aligned16(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

}  // namespace

bool moe_cutlass_grouped_gemm_bf16(cublasOperation_t transa,
                                   cublasOperation_t transb,
                                   const std::vector<int>& m,
                                   const std::vector<int>& n,
                                   const std::vector<int>& k,
                                   float alpha,
                                   const std::vector<const nv_bfloat16*>& A,
                                   const std::vector<int>& lda,
                                   const std::vector<const nv_bfloat16*>& B,
                                   const std::vector<int>& ldb,
                                   float beta,
                                   const std::vector<nv_bfloat16*>& C,
                                   const std::vector<int>& ldc,
                                   cudaStream_t stream) {
    const bool ta = transa != CUBLAS_OP_N, tb = transb != CUBLAS_OP_N;
    if (ta && tb) return false;  // not instantiated (no MoE caller uses TT)
    for (std::size_t i = 0; i < m.size(); ++i) {
        // What cublasGemmEx would reject goes to it, so the caller gets its error, not a bad launch.
        if (m[i] <= 0 || n[i] <= 0 || k[i] <= 0) return false;
        if (lda[i] < (ta ? k[i] : m[i]) || ldb[i] < (tb ? n[i] : k[i]) || ldc[i] < m[i]) return false;
        // Contiguous extent of op(A) (m x k): k when A is transposed (RowMajor), else m.
        // Contiguous extent of op(B) (k x n): n when B is transposed (RowMajor), else k. C: m.
        if (!aligned8(ta ? k[i] : m[i]) || !aligned8(tb ? n[i] : k[i]) || !aligned8(m[i])) return false;
        if (!aligned8(lda[i]) || !aligned8(ldb[i]) || !aligned8(ldc[i])) return false;
        if (!aligned16(A[i]) || !aligned16(B[i]) || !aligned16(C[i])) return false;
    }
    if (ta) {
        run_grouped<RowMajor, ColumnMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
    } else if (tb) {
        run_grouped<ColumnMajor, RowMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
    } else {
        run_grouped<ColumnMajor, ColumnMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
    }
    return true;
}
