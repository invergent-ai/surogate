// Memory-efficient attention backward with key splits (Gemma4 global layers:
// head dim 512, GQA 16/2, causal). Every split owns a disjoint key range, so
// dK/dV must be bit-identical to the single-split reference; dQ is the sum of
// per-split fp32 tiles in lock-arrival order and may differ by rounding only.
#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

#include "runtime/attention/mem_eff_runtime.h"
#include "utilities/tensor.h"

namespace {

Tensor raw_tensor(void* ptr, ETensorDType dtype, std::initializer_list<long> shape) {
    Tensor t{};
    t.Data = static_cast<std::byte*>(ptr);
    t.DType = dtype;
    t.Rank = static_cast<int>(shape.size());
    t.Device = 0;
    int i = 0;
    for (long s : shape)
        t.Sizes[i++] = s;
    for (; i < MAX_TENSOR_DIM; ++i)
        t.Sizes[i] = 1;
    return t;
}

struct DeviceBuffer {
    void* ptr = nullptr;
    std::size_t bytes = 0;
    explicit DeviceBuffer(std::size_t n)
        : bytes(n) {
        REQUIRE(cudaMalloc(&ptr, n) == cudaSuccess);
        REQUIRE(cudaMemset(ptr, 0, n) == cudaSuccess);
    }
    ~DeviceBuffer() {
        cudaFree(ptr);
    }
    template <class T>
    T* as() const {
        return static_cast<T*>(ptr);
    }
    void upload(const void* host) const {
        REQUIRE(cudaMemcpy(ptr, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess);
    }
    template <class T>
    std::vector<T> download() const {
        std::vector<T> out(bytes / sizeof(T));
        REQUIRE(cudaMemcpy(out.data(), ptr, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
        return out;
    }
};

struct Scratch {
    std::vector<void*> blocks;
    std::byte* allocate(std::size_t n) {
        void* p = nullptr;
        REQUIRE(cudaMalloc(&p, n + 256) == cudaSuccess);
        blocks.push_back(p);
        return static_cast<std::byte*>(p);
    }
    ~Scratch() {
        for (void* p : blocks)
            cudaFree(p);
    }
};

struct Case {
    int B = 2, T = 512, Hq = 16, Hkv = 2, Hs = 512;
};

struct Grad {
    std::vector<nv_bfloat16> d_qkv;
};

// Runs forward once, then backward with the requested key-split count. `packed`
// optionally replaces the dense document boundaries (one document per batch row).
Grad run(const Case& c, int splits, unsigned seed, const std::vector<int32_t>& packed = {}) {
    const int Htot = c.Hq + 2 * c.Hkv;
    const std::size_t qkv_n = static_cast<std::size_t>(c.B) * c.T * Htot * c.Hs;
    const std::size_t out_n = static_cast<std::size_t>(c.B) * c.T * c.Hq * c.Hs;
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<nv_bfloat16> qkv(qkv_n), d_out(out_n);
    for (auto& v : qkv)
        v = __float2bfloat16(0.5f * dist(rng));
    for (auto& v : d_out)
        v = __float2bfloat16(dist(rng));

    DeviceBuffer d_qkv_in(qkv_n * sizeof(nv_bfloat16)), d_out_buf(out_n * sizeof(nv_bfloat16));
    DeviceBuffer out(out_n * sizeof(nv_bfloat16)), lse(static_cast<std::size_t>(c.B) * c.Hq * c.T * sizeof(float));
    DeviceBuffer d_qkv(qkv_n * sizeof(nv_bfloat16));
    d_qkv_in.upload(qkv.data());
    d_out_buf.upload(d_out.data());
    std::vector<int32_t> cu(c.B + 1);
    for (int b = 0; b <= c.B; ++b)
        cu[b] = b * c.T;
    if (!packed.empty()) cu = packed;
    REQUIRE(cu.back() <= c.B * c.T);
    int max_doc = 0;
    for (std::size_t d = 1; d < cu.size(); ++d)
        max_doc = std::max(max_doc, cu[d] - cu[d - 1]);
    DeviceBuffer cu_dev(cu.size() * sizeof(int32_t));
    cu_dev.upload(cu.data());

    dsl::AttentionParams p;
    p.B = c.B;
    p.T = c.T;
    p.Hq = c.Hq;
    p.Hkv = c.Hkv;
    p.Hs = c.Hs;
    p.causal = true;
    p.window_size = 0;
    p.qkv = raw_tensor(d_qkv_in.ptr, ETensorDType::BF16, {c.B, c.T, Htot, c.Hs});
    p.out = raw_tensor(out.ptr, ETensorDType::BF16, {c.B, c.T, c.Hq, c.Hs});
    p.lse = raw_tensor(lse.ptr, ETensorDType::FP32, {c.B, c.Hq, c.T});
    p.d_out = raw_tensor(d_out_buf.ptr, ETensorDType::BF16, {c.B, c.T, c.Hq, c.Hs});
    p.d_qkv = raw_tensor(d_qkv.ptr, ETensorDType::BF16, {c.B, c.T, Htot, c.Hs});
    p.cu_seqlens = cu_dev.as<int32_t>();
    p.cu_seqlens_cpu = cu.data();
    p.num_docs = static_cast<int>(cu.size()) - 1;
    p.max_doc_seqlen = max_doc;
    p.total_doc_tokens = cu.back();
    p.stream = nullptr;
    int device = 0;
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, device) == cudaSuccess);
    p.sm_version = prop.major * 10 + prop.minor;

    Scratch scratch;
    dsl::MemEffScratchAllocator alloc = [&scratch](std::size_t n) {
        return scratch.allocate(n);
    };
    dsl::mem_eff_forward_with_scratch(p, alloc);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const std::string forced = std::to_string(splits);
    REQUIRE(setenv("SUROGATE_MEM_EFF_KEY_SPLITS", forced.c_str(), 1) == 0);
    REQUIRE(dsl::mem_eff_backward_key_splits(p, max_doc, p.num_docs, c.Hq) == std::min(splits, (max_doc + 63) / 64));
    dsl::mem_eff_backward_with_scratch(p, alloc);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    unsetenv("SUROGATE_MEM_EFF_KEY_SPLITS");
    return {d_qkv.download<nv_bfloat16>()};
}

// Compares the Q, K and V sections of the interleaved [B, T, Hq+2Hkv, Hs] gradient.
void compare(const Case& c, const std::vector<nv_bfloat16>& a, const std::vector<nv_bfloat16>& b) {
    const int Htot = c.Hq + 2 * c.Hkv;
    std::size_t kv_mismatch = 0, q_count = 0;
    double q_err2 = 0.0, q_ref2 = 0.0, q_max_rel = 0.0;
    for (int bt = 0; bt < c.B * c.T; ++bt) {
        for (int h = 0; h < Htot; ++h) {
            for (int d = 0; d < c.Hs; ++d) {
                const std::size_t i = (static_cast<std::size_t>(bt) * Htot + h) * c.Hs + d;
                const float x = __bfloat162float(a[i]), y = __bfloat162float(b[i]);
                if (h >= c.Hq) {
                    kv_mismatch += (x != y);
                } else {
                    q_count++;
                    q_err2 += static_cast<double>(x - y) * (x - y);
                    q_ref2 += static_cast<double>(x) * x;
                    const float denom = std::max(std::fabs(x), std::fabs(y));
                    if (denom > 1e-2f) q_max_rel = std::max(q_max_rel, static_cast<double>(std::fabs(x - y)) / denom);
                }
            }
        }
    }
    INFO("dK/dV mismatches " << kv_mismatch << " dQ rel-L2 " << std::sqrt(q_err2 / std::max(q_ref2, 1e-30))
                             << " dQ max rel " << q_max_rel);
    REQUIRE(kv_mismatch == 0);                                    // each split owns its keys: exact
    REQUIRE(std::sqrt(q_err2 / std::max(q_ref2, 1e-30)) < 2e-3);  // dQ: fp32 order + bf16 rounding
    REQUIRE(q_max_rel < 3.2e-2);                                  // at most a few bf16 ulps per element
    REQUIRE(q_count > 0);
}

}  // namespace

TEST_CASE("mem-eff backward key splits keep dK/dV exact and dQ within rounding", "[attention][mem_eff]") {
    Case c;
    c.T = 1024;  // 16 key blocks: 8 splits and the 16-split maximum are distinct runs
    const Grad one = run(c, 1, 7);
    const Grad again = run(c, 1, 7);
    REQUIRE(one.d_qkv.size() == again.d_qkv.size());
    std::size_t diff = 0;
    for (std::size_t i = 0; i < one.d_qkv.size(); ++i)
        diff += (__bfloat162float(one.d_qkv[i]) != __bfloat162float(again.d_qkv[i]));
    REQUIRE(diff == 0);  // single split: bitwise deterministic
    const Grad eight = run(c, 8, 7);
    compare(c, one.d_qkv, eight.d_qkv);
    const Grad max_splits = run(c, c.T / 64, 7);
    compare(c, one.d_qkv, max_splits.d_qkv);
}

TEST_CASE("mem-eff backward key splits with packed documents of unequal length", "[attention][mem_eff]") {
    // Documents {1024, 320, 64, 5, 635} packed into B=2 rows of T=1024: the split
    // count exceeds the key blocks of the short documents, whose surplus CTAs must
    // exit without touching the workspace.
    Case c;
    c.T = 1024;
    const std::vector<int32_t> cu{0, 1024, 1344, 1408, 1413, 2048};
    const Grad one = run(c, 1, 11, cu);
    const Grad max_splits = run(c, c.T / 64, 11, cu);
    compare(c, one.d_qkv, max_splits.d_qkv);
}

TEST_CASE("mem-eff key split heuristic", "[attention][mem_eff]") {
    dsl::AttentionParams p;
    unsetenv("SUROGATE_MEM_EFF_KEY_SPLITS");
    p.deterministic_bwd = true;
    REQUIRE(dsl::mem_eff_backward_key_splits(p, 4352, 2, 16) == 1);
    p.deterministic_bwd = false;
    REQUIRE(dsl::mem_eff_backward_key_splits(p, 4352, 2, 16) == 1);  // default: reproducible single split
    REQUIRE(setenv("SUROGATE_MEM_EFF_KEY_SPLITS", "garbage", 1) == 0);
    REQUIRE(dsl::mem_eff_backward_key_splits(p, 4352, 2, 16) == 1);  // unparseable opt-in: single split
    REQUIRE(setenv("SUROGATE_MEM_EFF_KEY_SPLITS", "0", 1) == 0);
    REQUIRE(dsl::mem_eff_backward_key_splits(p, 4352, 2, 16) == 1);
    REQUIRE(setenv("SUROGATE_MEM_EFF_KEY_SPLITS", "auto", 1) == 0);
    p.deterministic_bwd = true;
    REQUIRE(dsl::mem_eff_backward_key_splits(p, 4352, 2, 16) == 1);  // deterministic_bwd beats the opt-in
    p.deterministic_bwd = false;
    const int auto_splits = dsl::mem_eff_backward_key_splits(p, 4352, 2, 16);
    REQUIRE(auto_splits >= 1);
    REQUIRE(auto_splits <= 68);  // ceil(4352 / 64) key blocks
    int device = 0, sms = 0;
    REQUIRE(cudaGetDevice(&device) == cudaSuccess);
    REQUIRE(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess);
    REQUIRE(auto_splits * 32 >= std::min(2 * sms, 68 * 32));       // about two CTAs per SM
    REQUIRE(dsl::mem_eff_backward_key_splits(p, 64, 2, 16) == 1);  // one key block: nothing to split
    p.causal = false;
    REQUIRE(dsl::mem_eff_backward_key_splits(p, 4352, 2, 16) == 1);  // non-causal keeps one split
    p.causal = true;
    REQUIRE(setenv("SUROGATE_MEM_EFF_KEY_SPLITS", "1000", 1) == 0);
    REQUIRE(dsl::mem_eff_backward_key_splits(p, 4352, 2, 16) == 68);  // forced count is bounded
    unsetenv("SUROGATE_MEM_EFF_KEY_SPLITS");
}
