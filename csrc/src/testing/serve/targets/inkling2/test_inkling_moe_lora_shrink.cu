// Inkling-2-Mini expert LoRA shrink (detail::inkling::launch_lora_shrink): the rank-group kernel against the original
// one-warp-per-(slot, bank) kernel, kept here verbatim as the reference. PASS = every low-rank fp32 value bit-identical over
// ranks (including rank % 4 != 0), gate_up and down, token counts, per-token / uniform / base slots, padded expert ids and a
// live-token count below the round width (exit 77 without a GPU).

#include "api/ops/lora.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/sparse_moe/inkling/inkling_moe_internal.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace inkling = sinfer::ops::detail::inkling;
using inkling::LiveTokens;
using inkling::LoraRound;
using sinfer::ops::LoraBank;
constexpr int kTopK = inkling::kTopK;

// ------------------------------------------------------------------ reference: the kernel as of v1.5.16, verbatim
namespace ref {
__device__ __forceinline__ int live_count(LiveTokens live) {
    return live.device != nullptr ? min(*live.device, live.tokens) : live.tokens;
}
__device__ __forceinline__ int lora_slot_of(const LoraRound& lora, int token) {
    if (lora.table == nullptr) { return -1; }
    if (lora.token_slot != nullptr) { return lora.token_slot[token]; }
    return lora.uniform_slot != nullptr ? *lora.uniform_slot : -1;
}
__device__ __forceinline__ const LoraBank& lora_bank(const LoraRound& lora, int expert, int port) {
    return lora.table[2 + 3 * expert + port];
}
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_xor_sync(0xFFFFFFFFU, value, offset));
    }
    return value;
}

__global__ __launch_bounds__(256) void lora_shrink_kernel(const __nv_bfloat16* __restrict__ source,
                                                          std::int32_t source_k, bool gate_up,
                                                          const std::int32_t* __restrict__ ids,
                                                          LiveTokens live, LoraRound lora) {
    sinfer::pdl::wait_for_dependencies();
    sinfer::pdl::trigger_dependents();
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int banks = gate_up ? 2 : 1;
    const int task  = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    const int index = task / banks;
    const int bank_index = task - index * banks;
    if (index >= live.tokens * kTopK) { return; }
    const int token  = index / kTopK;
    const int expert = __ldcg(ids + index);
    if (expert < 0 || token >= live_count(live)) { return; }
    const int slot = lora_slot_of(lora, token);
    if (slot < 0) { return; }
    const LoraBank& bank = lora_bank(lora, expert, gate_up ? bank_index : 2);
    const __nv_bfloat16* x =
        source + static_cast<std::int64_t>(gate_up ? token : index) * source_k;
    const auto* a = static_cast<const __nv_bfloat16*>(bank.a) + static_cast<std::int64_t>(slot) * bank.a_stride;
    float* low    = lora.low + static_cast<std::int64_t>(index) * lora.low_stride + bank_index * bank.rank;
    for (int r0 = 0; r0 < bank.rank; r0 += 16) {
        float acc[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) { acc[r] = 0.0F; }
        for (int k = lane * 8; k < source_k; k += 256) {
            const uint4 xv = __ldcg(reinterpret_cast<const uint4*>(x + k)); // x / y rows through L2 (PDL audit)
            const std::uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                if (r0 + r < bank.rank) {
                    const uint4 av = *reinterpret_cast<const uint4*>(
                        a + static_cast<std::int64_t>(r0 + r) * source_k + k);
                    const std::uint32_t aw[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        acc[r] = __fmaf_rn(__uint_as_float(aw[i] << 16), __uint_as_float(xw[i] << 16), acc[r]);
                        acc[r] = __fmaf_rn(__uint_as_float(aw[i] & 0xFFFF0000U),
                                           __uint_as_float(xw[i] & 0xFFFF0000U), acc[r]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const float total = warp_sum(acc[r]);
            if (lane == 0 && r0 + r < bank.rank) { low[r0 + r] = total; }
        }
    }
}
} // namespace ref

struct Dev {
    void* p = nullptr;
    explicit Dev(std::size_t bytes) { CUDA_CHECK(cudaMalloc(&p, bytes == 0 ? 1 : bytes)); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&)            = delete;
    Dev& operator=(const Dev&) = delete;
};

std::uint16_t to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFU + ((u >> 16) & 1U);
    return static_cast<std::uint16_t>(u >> 16);
}

template <class T> void up(const Dev& d, const std::vector<T>& v) {
    CUDA_CHECK(cudaMemcpy(d.p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}

// mode 0: per-token slots (some -1), 1: uniform cell, 2: uniform -1. live_below: a device live count of T - 2.
int check(int rank, int T, bool gate_up, int mode, bool live_below, std::uint32_t seed, bool bench = false) {
    constexpr int kUsedExperts = 6, kSlots = 2;
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    const int k     = gate_up ? inkling::kHidden : inkling::kIntermediate;
    const int banks = gate_up ? 2 : 1;
    const int rows  = gate_up ? T : T * kTopK;
    std::vector<std::uint16_t> src(static_cast<std::size_t>(rows) * k);
    for (auto& e : src) { e = to_bf16(nd(g)); }
    std::vector<std::int32_t> ids(static_cast<std::size_t>(T) * kTopK);
    std::uniform_int_distribution<int> pick(-1, kUsedExperts - 1);
    for (auto& e : ids) { e = pick(g); }
    std::vector<std::int32_t> slots(T);
    std::uniform_int_distribution<int> spick(-1, kSlots - 1);
    for (auto& e : slots) { e = spick(g); }
    std::vector<Dev*> owned;
    std::vector<LoraBank> table(2 + 3 * kUsedExperts);
    for (int e = 0; e < kUsedExperts; ++e) {
        for (int port = 0; port < 3; ++port) {
            const int kk = port == 2 ? inkling::kIntermediate : inkling::kHidden;
            std::vector<std::uint16_t> a(static_cast<std::size_t>(kSlots) * rank * kk);
            for (auto& v : a) { v = to_bf16(0.05F * nd(g)); }
            owned.push_back(new Dev(a.size() * 2));
            up(*owned.back(), a);
            LoraBank b{};
            b.a        = owned.back()->p;
            b.a_stride = static_cast<std::int64_t>(rank) * kk;
            b.rank     = rank;
            b.k        = kk;
            table[2 + 3 * e + port] = b;
        }
    }
    Dev d_table(table.size() * sizeof(LoraBank)), d_src(src.size() * 2), d_ids(ids.size() * 4), d_slots(slots.size() * 4),
        d_cell(4), d_live(4);
    up(d_table, table);
    up(d_src, src);
    up(d_ids, ids);
    up(d_slots, slots);
    up(d_cell, std::vector<std::int32_t>{mode == 2 ? -1 : 1});
    up(d_live, std::vector<std::int32_t>{T - 2});
    const int low_stride = banks * rank;
    const std::size_t n_low = static_cast<std::size_t>(T) * kTopK * low_stride;
    std::vector<float> out[2];
    for (int arm = 0; arm < 2; ++arm) {
        Dev d_low(n_low * 4);
        CUDA_CHECK(cudaMemset(d_low.p, 0x7F, n_low * 4));
        LoraRound lora;
        lora.table        = static_cast<const LoraBank*>(d_table.p);
        lora.token_slot   = mode == 0 ? static_cast<const std::int32_t*>(d_slots.p) : nullptr;
        lora.uniform_slot = mode == 0 ? nullptr : static_cast<const std::int32_t*>(d_cell.p);
        lora.rank         = rank;
        lora.low          = static_cast<float*>(d_low.p);
        lora.low_stride   = low_stride;
        LiveTokens live{live_below ? static_cast<const std::int32_t*>(d_live.p) : nullptr, T};
        const auto* x = static_cast<const __nv_bfloat16*>(d_src.p);
        const auto* id = static_cast<const std::int32_t*>(d_ids.p);
        const auto once = [&] {
            if (arm == 0) {
                const std::int64_t tasks = static_cast<std::int64_t>(T) * kTopK * banks;
                ref::lora_shrink_kernel<<<static_cast<unsigned>((tasks + 7) / 8), 256>>>(x, k, gate_up, id, live, lora);
                CUDA_CHECK(cudaGetLastError());
            } else {
                inkling::launch_lora_shrink(x, k, gate_up, id, live, lora, nullptr);
            }
        };
        once();
        CUDA_CHECK(cudaDeviceSynchronize());
        if (bench) {
            cudaEvent_t e0, e1;
            CUDA_CHECK(cudaEventCreate(&e0));
            CUDA_CHECK(cudaEventCreate(&e1));
            CUDA_CHECK(cudaEventRecord(e0));
            for (int i = 0; i < 200; ++i) { once(); }
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            std::printf("bench %-7s rank %2d T %3d %s: %.1f us a call\n", gate_up ? "gate_up" : "down", rank, T,
                        arm == 0 ? "reference" : "rank groups", 1000.0F * ms / 200);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        }
        out[arm].resize(n_low);
        CUDA_CHECK(cudaMemcpy(out[arm].data(), d_low.p, n_low * 4, cudaMemcpyDeviceToHost));
    }
    for (Dev* d : owned) { delete d; }
    std::size_t bad = 0, written = 0;
    std::uint32_t fill;
    std::memset(&fill, 0x7F, 4);
    for (std::size_t i = 0; i < n_low; ++i) {
        bad += std::memcmp(&out[0][i], &out[1][i], 4) != 0;
        written += std::memcmp(&out[0][i], &fill, 4) != 0;
    }
    // some value must be written exactly when a live token has an adapter slot and a routed expert
    bool expect = false;
    const int live_n = live_below ? T - 2 : T;
    for (int t = 0; t < live_n; ++t) {
        const int sl = mode == 0 ? slots[t] : mode == 1 ? 1 : -1;
        for (int j = 0; j < kTopK; ++j) { expect = expect || (sl >= 0 && ids[static_cast<std::size_t>(t) * kTopK + j] >= 0); }
    }
    const bool ok = bad == 0 && (written > 0) == expect;
    std::printf("%s %-7s rank %2d T %3d slots %-9s live %-9s: %zu / %zu differ (%zu written)\n", ok ? "ok  " : "FAIL",
                gate_up ? "gate_up" : "down", rank, T, mode == 0 ? "per-token" : mode == 1 ? "uniform" : "uniform-1",
                live_below ? "T-2" : "all", bad, n_low, written);
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no GPU: skipped\n");
        return 77;
    }
    int failures = 0;
    std::uint32_t seed = 4400;
    for (const int rank : {1, 3, 7, 16, 64}) {
        for (const int T : {1, 5, 64}) {
            for (const bool gate_up : {true, false}) {
                failures += check(rank, T, gate_up, 0, false, seed++);
                failures += check(rank, T, gate_up, 1, false, seed++);
            }
        }
    }
    failures += check(16, 8, true, 2, false, seed++);
    failures += check(16, 8, true, 0, true, seed++);
    failures += check(5, 8, false, 1, true, seed++);
    std::printf("%s: %d failing cases\n", failures == 0 ? "PASS" : "FAIL", failures);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        for (const int T : {1, 8, 64}) {
            for (const bool gate_up : {true, false}) { check(16, T, gate_up, 1, false, 7, true); }
        }
    }
    return failures == 0 ? 0 : 1;
}
