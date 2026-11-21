// Inkling-2-Mini attention LoRA delta (lane e1, S10): per-token, fixed-order arithmetic (api/ops/inkling_mix.h
// inkling_lora_delta). Compiled with -fmad=false: every fused multiply-add is an explicit fmaf.

#include "api/ops/inkling_mix.h"
#include "api/ops/lora.h"

#include "core/device.h" // CUDA_CHECK
#include "core/pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

constexpr int kMaxRank       = 64;
constexpr int kShrinkThreads = 128; // 4 warps; warp w owns ranks r = w, w + 4, ...
constexpr int kExpandThreads = 256;

struct LoraDeltaArgs {
    const LoraBank* table          = nullptr;
    InklingLoraSlice slices[kInklingLoraMaxSlices]{};
    std::int32_t slice_count       = 0;
    std::int32_t rank              = 0;
    std::int32_t tokens            = 0;
    const std::int32_t* token_slot = nullptr;
    const std::int32_t* uniform    = nullptr;
    float* low                     = nullptr; // [tokens][slices][rank]
};

__device__ __forceinline__ int slot_of(const LoraDeltaArgs& a, int t) {
    if (a.token_slot != nullptr) { return a.token_slot[t]; }
    return a.uniform != nullptr ? *a.uniform : -1;
}

// grid (tokens, slices): low[t][s][r] = sum_k A[r, k] * x[k, t]; lane l chains k = l, l + 32, ... from 0, then a
// fixed xor tree over the 32 lane sums. Independent of every other token.
__global__ void __launch_bounds__(kShrinkThreads) lora_shrink_kernel(const __grid_constant__ LoraDeltaArgs a) {
    pdl::wait_for_dependencies();
    const int t    = blockIdx.x;
    const int s    = blockIdx.y;
    const int slot = slot_of(a, t);
    if (slot < 0) { return; }
    const InklingLoraSlice& sl = a.slices[s];
    const LoraBank& bank       = a.table[sl.bank];
    const auto* A = static_cast<const __nv_bfloat16*>(bank.a) + static_cast<std::int64_t>(slot) * bank.a_stride;
    const auto* x = static_cast<const __nv_bfloat16*>(sl.x) + static_cast<std::int64_t>(t) * sl.x_stride;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float* low     = a.low + (static_cast<std::int64_t>(t) * a.slice_count + s) * a.rank;
    for (int r = warp; r < a.rank; r += kShrinkThreads / 32) {
        const __nv_bfloat16* row = A + static_cast<std::int64_t>(r) * bank.k;
        float acc = 0.0F;
        for (int k = lane; k < bank.k; k += 32) {
            acc = fmaf(__bfloat162float(row[k]), __bfloat162float(x[k]), acc);
        }
        for (int m = 16; m >= 1; m >>= 1) { acc += __shfl_xor_sync(0xffffffffU, acc, m); }
        if (lane == 0) { low[r] = acc; }
    }
}

// grid (row blocks, tokens, slices): delta[j] = sum_r B[j, r] * low[r] (fp32 chain from 0, r order); the output row
// becomes bf16_rn(out + delta) where delta != 0.
__global__ void __launch_bounds__(kExpandThreads) lora_expand_kernel(const __grid_constant__ LoraDeltaArgs a) {
    pdl::wait_for_dependencies();
    const int t    = blockIdx.y;
    const int s    = blockIdx.z;
    const int slot = slot_of(a, t);
    if (slot < 0) { return; }
    const InklingLoraSlice& sl = a.slices[s];
    const LoraBank& bank       = a.table[sl.bank];
    const int j                = blockIdx.x * kExpandThreads + threadIdx.x;
    if (j >= sl.rows || j >= bank.n) { return; }
    const auto* B = static_cast<const __nv_bfloat16*>(bank.b) + static_cast<std::int64_t>(slot) * bank.b_stride +
                    static_cast<std::int64_t>(j) * a.rank;
    const float* low = a.low + (static_cast<std::int64_t>(t) * a.slice_count + s) * a.rank;
    float delta      = 0.0F;
    for (int r = 0; r < a.rank; ++r) { delta = fmaf(__bfloat162float(B[r]), low[r], delta); }
    if (delta != 0.0F) {
        auto* out = static_cast<__nv_bfloat16*>(sl.out) + static_cast<std::int64_t>(t) * sl.out_stride + j;
        *out      = __float2bfloat16_rn(__bfloat162float(*out) + delta);
    }
}

void require(bool ok, const char* what) {
    if (!ok) { throw std::invalid_argument(std::string("inkling_lora_delta: ") + what); }
}

} // namespace

std::size_t inkling_lora_delta_workspace_bytes(std::int32_t slices, std::int32_t max_rank,
                                               std::int32_t tokens) noexcept {
    const std::size_t bytes = static_cast<std::size_t>(slices) * static_cast<std::size_t>(max_rank) *
                              static_cast<std::size_t>(tokens) * sizeof(float);
    return (bytes + 255U) & ~std::size_t{255};
}

void inkling_lora_delta(const void* table, const InklingLoraSlice* slices, std::int32_t slice_count,
                        std::int32_t max_rank, std::int32_t tokens, const std::int32_t* token_slot,
                        const std::int32_t* uniform_slot, void* workspace, std::size_t workspace_bytes,
                        cudaStream_t stream) {
    require(table != nullptr, "no bank table");
    require(slice_count >= 1 && slice_count <= kInklingLoraMaxSlices, "1..6 slices");
    require(max_rank >= 1 && max_rank <= kMaxRank, "max_rank in [1, 64]");
    require(tokens >= 0, "negative token count");
    require(token_slot != nullptr || uniform_slot != nullptr, "token slots or a uniform slot");
    if (tokens == 0) { return; }
    require(workspace != nullptr &&
                workspace_bytes >= inkling_lora_delta_workspace_bytes(slice_count, max_rank, tokens),
            "workspace too small");
    LoraDeltaArgs a;
    a.table       = static_cast<const LoraBank*>(table);
    a.slice_count = slice_count;
    a.rank        = max_rank;
    a.tokens      = tokens;
    a.token_slot  = token_slot;
    a.uniform     = uniform_slot;
    a.low         = static_cast<float*>(workspace);
    int widest    = 0;
    for (int s = 0; s < slice_count; ++s) {
        const InklingLoraSlice& sl = slices[s];
        require(sl.x != nullptr && sl.out != nullptr && sl.bank >= 0, "slice without input / output / bank");
        require(sl.x_stride > 0 && sl.out_stride >= sl.rows && sl.rows > 0, "slice strides / rows");
        a.slices[s] = sl;
        widest      = std::max(widest, static_cast<int>(sl.rows)); // row blocks cover the widest slice
    }
    CUDA_CHECK(pdl::launch_dependent({dim3(static_cast<unsigned>(tokens), static_cast<unsigned>(slice_count)),
                                      dim3(kShrinkThreads), 0, stream},
                                     lora_shrink_kernel, a));
    CUDA_CHECK(pdl::launch_dependent({dim3(static_cast<unsigned>((widest + kExpandThreads - 1) / kExpandThreads),
                                           static_cast<unsigned>(tokens), static_cast<unsigned>(slice_count)),
                                      dim3(kExpandThreads), 0, stream},
                                     lora_expand_kernel, a));
}

} // namespace sinfer::ops
