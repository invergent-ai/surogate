// Inkling-2-Mini attention LoRA delta (ops::inkling_lora_delta): the tiled shrink / expand kernels against the original
// one-block-per-(token, slice) kernels, kept here verbatim as the reference. PASS = every output bf16 and every low-rank
// fp32 value bit-identical over ranks, slice sets, token counts, slot patterns (uniform, per token, mixed, base tokens,
// zero adapter), odd strides / misaligned workspaces (the scalar fall-backs) and non-finite inputs. With --bench it also
// times both at the attention's shapes (exit 77 without a GPU).

#include "api/ops/inkling_mix.h"
#include "api/ops/lora.h"

#include "core/device.h"
#include "core/pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using sinfer::ops::InklingLoraSlice;
using sinfer::ops::LoraBank;

// ------------------------------------------------------------------ reference: the kernels as of v1.5.16, verbatim
namespace ref {
constexpr int kShrinkThreads = 128;
constexpr int kExpandThreads = 256;

struct LoraDeltaArgs {
    const LoraBank* table          = nullptr;
    InklingLoraSlice slices[sinfer::ops::kInklingLoraMaxSlices]{};
    std::int32_t slice_count       = 0;
    std::int32_t rank              = 0;
    std::int32_t tokens            = 0;
    const std::int32_t* token_slot = nullptr;
    const std::int32_t* uniform    = nullptr;
    float* low                     = nullptr;
};

__device__ __forceinline__ int slot_of(const LoraDeltaArgs& a, int t) {
    if (a.token_slot != nullptr) { return a.token_slot[t]; }
    return a.uniform != nullptr ? *a.uniform : -1;
}

__global__ void __launch_bounds__(kShrinkThreads) lora_shrink_kernel(const __grid_constant__ LoraDeltaArgs a) {
    sinfer::pdl::wait_for_dependencies();
    sinfer::pdl::trigger_dependents();
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

__global__ void __launch_bounds__(kExpandThreads) lora_expand_kernel(const __grid_constant__ LoraDeltaArgs a) {
    sinfer::pdl::wait_for_dependencies();
    sinfer::pdl::trigger_dependents();
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

void delta(const void* table, const InklingLoraSlice* slices, int slice_count, int max_rank, int tokens,
           const std::int32_t* token_slot, const std::int32_t* uniform_slot, void* workspace, cudaStream_t stream) {
    if (tokens == 0) { return; }
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
        a.slices[s] = slices[s];
        widest      = std::max(widest, static_cast<int>(slices[s].rows));
    }
    CUDA_CHECK(sinfer::pdl::launch_dependent({dim3(tokens, slice_count), dim3(kShrinkThreads), 0, stream},
                                             lora_shrink_kernel, a));
    CUDA_CHECK(sinfer::pdl::launch_dependent(
        {dim3((widest + kExpandThreads - 1) / kExpandThreads, tokens, slice_count), dim3(kExpandThreads), 0, stream},
        lora_expand_kernel, a));
}
} // namespace ref

// ------------------------------------------------------------------ helpers
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
    if ((u & 0x7F800000U) == 0x7F800000U) { return static_cast<std::uint16_t>(u >> 16 | ((u & 0xFFFF) ? 0x40 : 0)); }
    u += 0x7FFFU + ((u >> 16) & 1U);
    return static_cast<std::uint16_t>(u >> 16);
}

std::vector<std::uint16_t> randn(std::size_t n, std::mt19937& g, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<std::uint16_t> v(n);
    for (auto& e : v) { e = to_bf16(d(g)); }
    return v;
}

template <class T> void up(const Dev& d, const std::vector<T>& v) {
    CUDA_CHECK(cudaMemcpy(d.p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <class T> std::vector<T> down(const void* p, std::size_t n) {
    std::vector<T> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

struct SliceShape {
    int rows;
    int k;
    int input; // which input plane
};

const std::vector<SliceShape> kQkvr = {{3072, 2304, 0}, {384, 2304, 0}, {3072, 2304, 0},
                                       {24, 2304, 0},   {1024, 2304, 1}, {1024, 2304, 2}};
const std::vector<SliceShape> kWo   = {{2304, 3072, 0}};
// odd row lengths: a last K stage shorter than 32 (2312), K < 32 (24), K % 8 != 0 (20: the direct shrink)
const std::vector<SliceShape> kOdd  = {{1000, 2312, 0}, {24, 24, 1}, {64, 20, 2}};

struct Case {
    std::string name;
    std::vector<SliceShape> shape;
    int rank;
    int tokens;
    int slots;            // adapter slots in the banks
    int mode;             // 0 uniform cell, 1 per token all slot 0, 2 per token mixed (-1 .. slots-1), 3 uniform -1,
                          // 4 per token {-1, 1, -1, 1, 1, -1, 1, -1} repeated, 5 per token {0, 0, 1, 1} repeated
    int zero_slot = -1;   // a slot whose B is zero (zero adapter)
    int pad_b     = 0;    // extra elements between B's slots (misaligned rows when odd)
    int pad_x     = 0;    // extra elements per x / out column
    int ws_offset = 0;    // bytes added to the workspace base (4 = misaligned low rows)
    bool nonfinite = false;
    int pad_a      = 0;    // extra elements between A's slots (odd: misaligned A rows of slot 1, the direct shrink)
};

struct Built {
    std::vector<Dev*> owned;
    std::vector<InklingLoraSlice> slices;
    Dev* table = nullptr;
    Dev* slots = nullptr;
    Dev* cell  = nullptr;
    std::vector<Dev*> outs;
    std::vector<std::vector<std::uint16_t>> out_init;
    std::vector<std::size_t> out_elems;
    bool expect_change = false; // some token has a non-zero adapter
    ~Built() {
        for (Dev* d : owned) { delete d; }
    }
    Dev* make(std::size_t bytes) {
        owned.push_back(new Dev(bytes));
        return owned.back();
    }
};

void build(const Case& c, std::uint32_t seed, Built& b) {
    std::mt19937 g(seed);
    const int T = c.tokens, R = c.rank;
    int inputs = 0;
    for (const auto& s : c.shape) { inputs = std::max(inputs, s.input + 1); }
    std::vector<Dev*> xs;
    std::vector<int> xstride;
    for (int i = 0; i < inputs; ++i) {
        int k = 0;
        for (const auto& s : c.shape) { if (s.input == i) { k = s.k; } }
        const int stride = k + c.pad_x;
        auto x           = randn(static_cast<std::size_t>(std::max(T, 1)) * stride, g, 1.0F);
        if (c.nonfinite && T > 2) {
            x[static_cast<std::size_t>(1) * stride + 5] = 0x7F80;  // +inf in token 1
            x[static_cast<std::size_t>(2) * stride + 17] = 0x7FC1; // NaN in token 2
        }
        Dev* d = b.make(x.size() * 2);
        up(*d, x);
        xs.push_back(d);
        xstride.push_back(stride);
    }
    std::vector<LoraBank> table;
    // one output plane per input group (qkvr: one plane holding every slice; here each slice gets its own plane region)
    int rows_total = 0;
    for (const auto& s : c.shape) { rows_total += s.rows; }
    const int out_stride = rows_total + c.pad_x;
    Dev* out             = b.make(static_cast<std::size_t>(std::max(T, 1)) * out_stride * 2);
    auto out_init        = randn(static_cast<std::size_t>(std::max(T, 1)) * out_stride, g, 1.0F);
    up(*out, out_init);
    b.outs.push_back(out);
    b.out_init.push_back(out_init);
    b.out_elems.push_back(out_init.size());
    int row0 = 0;
    for (std::size_t si = 0; si < c.shape.size(); ++si) {
        const auto& s     = c.shape[si];
        const auto a      = randn(static_cast<std::size_t>(c.slots) * (static_cast<std::size_t>(R) * s.k + c.pad_a), g, 0.05F);
        const std::int64_t b_stride = static_cast<std::int64_t>(s.rows) * R + c.pad_b;
        auto bv           = randn(static_cast<std::size_t>(c.slots) * b_stride, g, 0.05F);
        if (c.zero_slot >= 0) {
            for (std::int64_t i = 0; i < b_stride; ++i) { bv[static_cast<std::size_t>(c.zero_slot * b_stride + i)] = 0; }
        }
        Dev* da = b.make(a.size() * 2);
        Dev* db = b.make(bv.size() * 2);
        up(*da, a);
        up(*db, bv);
        LoraBank bank{};
        bank.a        = da->p;
        bank.b        = db->p;
        bank.a_stride = static_cast<std::int64_t>(R) * s.k + c.pad_a;
        bank.b_stride = b_stride;
        bank.rank     = R;
        bank.n        = s.rows;
        bank.k        = s.k;
        table.push_back(bank);
        InklingLoraSlice sl;
        sl.bank       = static_cast<std::int32_t>(si);
        sl.rows       = s.rows;
        sl.x          = xs[s.input]->p;
        sl.x_stride   = xstride[s.input];
        sl.out        = static_cast<std::uint16_t*>(out->p) + row0;
        sl.out_stride = out_stride;
        b.slices.push_back(sl);
        row0 += s.rows;
    }
    b.table = b.make(table.size() * sizeof(LoraBank));
    up(*b.table, table);
    std::vector<std::int32_t> slots(static_cast<std::size_t>(std::max(T, 1)));
    std::uniform_int_distribution<int> pick(-1, c.slots - 1);
    static const int kPattern4[8] = {-1, 1, -1, 1, 1, -1, 1, -1};
    for (int t = 0; t < T; ++t) {
        slots[t] = c.mode == 2 ? pick(g) : c.mode == 4 ? kPattern4[t % 8] : c.mode == 5 ? (t / 2) % 2 : 0;
    }
    if (c.mode == 2 && T >= 6) { slots[0] = -1; slots[1] = 0; slots[2] = 0; slots[3] = c.slots - 1; slots[4] = -1; slots[5] = 0; }
    const bool per_token = c.mode == 1 || c.mode == 2 || c.mode == 4 || c.mode == 5;
    const int cell       = c.mode == 3 ? -1 : (c.zero_slot == 0 ? 0 : std::min(1, c.slots - 1));
    for (int t = 0; t < T; ++t) {
        const int sl = per_token ? slots[t] : cell;
        b.expect_change = b.expect_change || (sl >= 0 && sl != c.zero_slot);
    }
    b.slots = b.make(slots.size() * 4);
    up(*b.slots, slots);
    b.cell = b.make(4);
    up(*b.cell, std::vector<std::int32_t>{cell});
}

struct Result {
    std::vector<std::uint16_t> out;
    std::vector<float> low;
};

Result run(const Case& c, std::uint32_t seed, bool reference) {
    Built b;
    build(c, seed, b);
    const int S         = static_cast<int>(c.shape.size());
    const std::size_t ws = sinfer::ops::inkling_lora_delta_workspace_bytes(S, c.rank, c.tokens);
    Dev* w              = b.make(ws + 64);
    CUDA_CHECK(cudaMemset(w->p, 0x7F, ws + 64)); // a fixed non-zero pattern: rows the kernels skip stay equal
    void* wp                   = static_cast<char*>(w->p) + c.ws_offset;
    const std::int32_t* tslot  = (c.mode == 1 || c.mode == 2 || c.mode == 4 || c.mode == 5) ? static_cast<const std::int32_t*>(b.slots->p) : nullptr;
    const std::int32_t* ucell  = tslot == nullptr ? static_cast<const std::int32_t*>(b.cell->p) : nullptr;
    if (reference) {
        ref::delta(b.table->p, b.slices.data(), S, c.rank, c.tokens, tslot, ucell, wp, nullptr);
    } else {
        sinfer::ops::inkling_lora_delta(b.table->p, b.slices.data(), S, c.rank, c.tokens, tslot, ucell, wp, ws, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    Result r;
    r.out = down<std::uint16_t>(b.outs[0]->p, b.out_elems[0]);
    const std::size_t n_low = static_cast<std::size_t>(S) * c.rank * c.tokens;
    std::vector<float> low(n_low);
    CUDA_CHECK(cudaMemcpy(low.data(), wp, n_low * 4, cudaMemcpyDeviceToHost));
    r.low = low;
    return r;
}

int check(const Case& c, std::uint32_t seed) {
    const Result a = run(c, seed, true);
    const Result b = run(c, seed, false);
    std::size_t bad_out = 0, bad_low = 0;
    for (std::size_t i = 0; i < a.out.size(); ++i) { bad_out += a.out[i] != b.out[i]; }
    for (std::size_t i = 0; i < a.low.size(); ++i) { bad_low += std::memcmp(&a.low[i], &b.low[i], 4) != 0; }
    // every case but "no adapted token" and "zero adapter only" must move output rows (the delta is live)
    bool expect_change = false;
    std::size_t changed = 0;
    {
        Built bb;
        build(c, seed, bb);
        expect_change = bb.expect_change;
        for (std::size_t i = 0; i < a.out.size(); ++i) { changed += a.out[i] != bb.out_init[0][i]; }
    }
    const bool ok = bad_out == 0 && bad_low == 0 && (changed > 0) == expect_change;
    std::printf("%s %-40s rank %2d T %5d: out %zu / %zu differ, low %zu / %zu differ (%zu changed by the delta)\n",
                ok ? "ok  " : "FAIL", c.name.c_str(), c.rank, c.tokens, bad_out, a.out.size(), bad_low, a.low.size(),
                changed);
    return ok ? 0 : 1;
}

double time_us(const Case& c, bool reference, int iters) {
    Built b;
    build(c, 7, b);
    const int S          = static_cast<int>(c.shape.size());
    const std::size_t ws = sinfer::ops::inkling_lora_delta_workspace_bytes(S, c.rank, c.tokens);
    Dev* w               = b.make(ws);
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    const auto* ucell = static_cast<const std::int32_t*>(b.cell->p);
    const auto once   = [&] {
        if (reference) {
            ref::delta(b.table->p, b.slices.data(), S, c.rank, c.tokens, nullptr, ucell, w->p, st);
        } else {
            sinfer::ops::inkling_lora_delta(b.table->p, b.slices.data(), S, c.rank, c.tokens, nullptr, ucell, w->p, ws, st);
        }
    };
    for (int i = 0; i < 5; ++i) { once(); }
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0, st));
    for (int i = 0; i < iters; ++i) { once(); }
    CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaStreamDestroy(st);
    return 1000.0 * ms / iters;
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "no GPU: skipped\n";
        return 77;
    }
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const int only_t = argc > 2 ? std::atoi(argv[2]) : 0; // --bench <T>: that width only, no bitwise cases
    int failures     = 0;
    std::uint32_t seed = 9100;
    for (const int R : (only_t > 0 ? std::vector<int>{} : std::vector<int>{64, 16, 2, 1, 7, 24})) {
        for (const int T : {1, 2, 3, 5, 8, 9, 24, 64, 67, 300}) {
            failures += check({"qkvr uniform", kQkvr, R, T, 2, 0}, seed++);
            failures += check({"wo per-token", kWo, R, T, 2, 1}, seed++);
            failures += check({"qkvr mixed slots + base tokens", kQkvr, R, T, 3, 2}, seed++);
        }
    }
    for (const int T : (only_t > 0 ? std::vector<int>{} : std::vector<int>{1, 8, 64, 1000, 4096})) {
        failures += check({"qkvr uniform", kQkvr, 64, T, 1, 0}, seed++);
        failures += check({"wo uniform", kWo, 64, T, 1, 0}, seed++);
    }
    for (const int T : (only_t > 0 ? std::vector<int>{} : std::vector<int>{1, 8, 64})) {
        failures += check({"qkvr uniform base (-1)", kQkvr, 64, T, 2, 3}, seed++);
        failures += check({"qkvr uniform zero adapter", kQkvr, 64, T, 2, 0, 0}, seed++);
        failures += check({"qkvr mixed with a zero adapter", kQkvr, 64, T, 3, 2, 1}, seed++);
        failures += check({"qkvr odd B stride (scalar B)", kQkvr, 64, T, 2, 2, -1, 3}, seed++);
        failures += check({"qkvr padded columns", kQkvr, 16, T, 2, 2, -1, 0, 40}, seed++);
        failures += check({"qkvr misaligned workspace (scalar low)", kQkvr, 64, T, 2, 2, -1, 0, 0, 4}, seed++);
        failures += check({"wo misaligned workspace rank 7", kWo, 7, T, 2, 2, -1, 1, 8, 4}, seed++);
        failures += check({"qkvr non-finite x", kQkvr, 64, T, 2, 2, -1, 0, 0, 0, true}, seed++);
        failures += check({"qkvr odd x stride (direct shrink)", kQkvr, 64, T, 2, 0, -1, 0, 3}, seed++);
        failures += check({"odd K: 2312 / 24 / 20", kOdd, 16, T, 2, 0}, seed++);
        failures += check({"odd K, mixed", kOdd, 7, T, 3, 2}, seed++);
        failures += check({"slots -1/1 pattern (staged, dead tokens)", kQkvr, 64, T, 2, 4}, seed++);
        failures += check({"slots 0,0,1,1 pattern (mixed tiles)", kQkvr, 64, T, 2, 5}, seed++);
        failures += check({"wo slots 0,0,1,1 pattern", kWo, 16, T, 2, 5}, seed++);
        failures += check({"odd A slot stride (direct shrink)", kQkvr, 64, T, 2, 0, -1, 0, 0, 0, false, 3}, seed++);
        failures += check({"qkvr x stride 7304", kQkvr, 64, T, 2, 0, -1, 0, 5000}, seed++);
        failures += check({"wo x stride 4072", kWo, 64, T, 2, 1, -1, 0, 1000}, seed++);
    }
    std::printf("%s: %d failing cases\n", failures == 0 ? "PASS" : "FAIL", failures);
    if (bench) {
        for (const int T : (only_t > 0 ? std::vector<int>{only_t} : std::vector<int>{1, 2, 8, 16, 24, 64, 128, 512, 4096})) {
            for (int which = 0; which < 2; ++which) {
                const Case c{which == 0 ? "qkvr" : "wo", which == 0 ? kQkvr : kWo, 64, T, 1, 0};
                const int iters = T >= 512 ? 20 : 200;
                const double old_us = time_us(c, true, iters);
                const double new_us = time_us(c, false, iters);
                std::printf("bench %-4s rank 64 T %5d: reference %9.1f us, tiled %8.1f us (x%.1f)\n", c.name.c_str(), T,
                            old_us, new_us, old_us / new_us);
            }
        }
    }
    return failures == 0 ? 0 : 1;
}
