// A static driver around the production sparse-MoE decode body, for one geometry.
//
// Two reasons it exists beside sinfer_sparse_moe_bench: NCU cannot profile kernels that live
// in libsinfer.so (the process dies on the first profiled launch), and a change to a codec
// or the D3/D4 bodies relinks the whole library. This compiles one geometry in about thirty
// seconds and profiles cleanly. It times the four stages separately, cold by default, and
// `--dump` writes the output and the activation scratch for an A/B against another build.
//
// Not a CMake target on purpose -- it is the kernel author's loop, not the suite's. Build
// from the repo root (R), for the default Qwen3.6 geometry or any registered one:
//
//   nvcc -O3 -DNDEBUG -std=c++20 "--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]" \
//     -lineinfo -I csrc/src/serve -I csrc/src/third_party -I csrc/src/third_party/utf8proc \
//     -I csrc/build-serve/_deps/json-src/single_include -I csrc/src/testing/serve/bench/ops \
//     -I csrc/src/testing/serve/bench [-DHARNESS_GEOMETRY=kSparseMoeGlm53Geometry] \
//     csrc/src/testing/serve/bench/ops/moe_m1_harness.cu csrc/src/serve/core/tensor.cpp \
//     csrc/src/serve/core/device.cu csrc/src/serve/core/dtype.cpp -o /tmp/harness -lcuda
//
//   CUDA_VISIBLE_DEVICES=3 /tmp/harness --gate q4_k --down q6_k [--warm] [--dump out.bin]
//   env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin HOME=$HOME CUDA_VISIBLE_DEVICES=3 \
//     SUROGATE_SERVE_NO_PDL=1 ncu --metrics gpu__time_duration.sum -k "regex:sparse_moe_d[34]" \
//     --launch-skip 6 --launch-count 8 /tmp/harness --gate q4_k --down q4_k --iters 12
//
// An A/B against HEAD: `git show HEAD:<body or codec> > base/ops/.../<same path>` and put
// `-I base` before `-I csrc/src/serve`; compare two `--dump` files position by position. The
// fixture is pseudo-random (fp16 scale words kept finite) so a codec reading the wrong bytes
// of the right row is caught; the bench's constant fill would not catch it.
#include "core/device.h"
#include "core/pdl.cuh"
#include "core/tensor.h"
#include "api/ops/sparse_moe.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"
#include "ops/linear/ggml/ggml_moe_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/sparse_moe/sparse_moe_route.cuh"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// The bench's DeviceBuffer is declared in core/arena.h and defined in arena.cu, which drags
// in the sleep allocator and the ops owner. A static driver wants one cudaMalloc, so the
// members are defined here instead.
namespace sinfer {
DeviceBuffer::DeviceBuffer(std::size_t n) : bytes(n) { CUDA_CHECK(cudaMalloc(&p, n)); }
DeviceBuffer::~DeviceBuffer() { if (p) { cudaFree(p); } }
DeviceBuffer::DeviceBuffer(DeviceBuffer&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; o.bytes = 0; }
DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) { if (p) { cudaFree(p); } p = o.p; bytes = o.bytes; o.p = nullptr; o.bytes = 0; }
    return *this;
}
void DeviceBuffer::fill(int v) { CUDA_CHECK(cudaMemset(p, v, bytes)); }
void DeviceBuffer::copy_from_host(const void* s, std::size_t n, std::size_t off) {
    CUDA_CHECK(cudaMemcpy(static_cast<char*>(p) + off, s, n, cudaMemcpyHostToDevice));
}
void DeviceBuffer::copy_to_host(void* d, std::size_t n, std::size_t off) const {
    CUDA_CHECK(cudaMemcpy(d, static_cast<const char*>(p) + off, n, cudaMemcpyDeviceToHost));
}
void DeviceBuffer::require_range(std::size_t, std::size_t, const char*) const {}
} // namespace sinfer

#ifndef HARNESS_GEOMETRY
#define HARNESS_GEOMETRY kSparseMoeQwen36Geometry
#endif

namespace sinfer::ops::detail {

#define SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(Registered)                                           \
    constexpr SparseMoeGeometry kGeometry = (Registered);                                          \
    constexpr int kHidden                 = kGeometry.hidden;                                      \
    constexpr int kExperts                = kGeometry.experts;                                     \
    constexpr int kRouterRows             = kGeometry.router_rows();                               \
    constexpr int kTopK                   = kGeometry.experts_per_token;                           \
    constexpr int kIntermediate           = kGeometry.intermediate;                                \
    constexpr bool kHasShared             = kGeometry.has_shared();                                \
    constexpr int kPaths                  = kGeometry.paths();                                     \
    constexpr SparseMoeGating kGating     = kGeometry.gating;                                      \
    constexpr float kRoutedScale          = kGeometry.routed_scale;                                \
    constexpr bool kSharedGated           = kGeometry.shared_gated;                                \
    constexpr float kSwigluLimit          = kGeometry.swiglu_limit;                                \
    constexpr GatedActivation kActivation = kGeometry.activation;                                  \
    constexpr bool kPerExpertScaled       = kGeometry.per_expert_scaled;

namespace geometry_harness {
SINFER_SPARSE_MOE_GEOMETRY_CONSTANTS(HARNESS_GEOMETRY)
#include "ops/sparse_moe/decode/sparse_moe_decode_body.inc"

// ------------------------------------------------------------------------------------------
struct MallocArena {
    std::vector<void*> owned;
    Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape, std::size_t) {
        Tensor t(nullptr, dtype, shape);
        void* p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, t.bytes()));
        CUDA_CHECK(cudaMemset(p, 0, t.bytes()));
        owned.push_back(p);
        return Tensor(p, dtype, shape);
    }
};

struct Codec {
    const char* name;
    QType qtype;
    bool ggml;
    double bytes_per_value;
};

Codec codec_of(const std::string& s) {
    static const Codec table[] = {
        {"q4_k", QType::Q4_K, true, 144.0 / 256}, {"q5_k", QType::Q5_K, true, 176.0 / 256},
        {"q6_k", QType::Q6_K, true, 210.0 / 256}, {"q8_0", QType::Q8_0, true, 34.0 / 32},
        {"q4", QType::Q4G64_F16S, false, 0.5 + 2.0 / 64},
        {"q5", QType::Q5G64_F16S, false, 0.625 + 2.0 / 64},
        {"q6", QType::Q6G64_F16S, false, 0.75 + 2.0 / 64},
        {"w8", QType::W8G32_F16S, false, 1.0 + 2.0 / 32},
    };
    for (const auto& c : table) { if (s == c.name) return c; }
    throw std::invalid_argument("unknown codec " + s);
}

// A constant fill cannot tell a codec that reads the right row's wrong bytes from one that
// reads the right ones. So the payload is pseudo-random, with every fp16 scale word patched
// to a finite value in [0.5, 1): the block layouts put those at known offsets.
void randomize(bench::PackedQuantizedWeight& w, const Codec& c, std::uint32_t seed) {
    std::vector<std::uint8_t> host(w.storage.bytes);
    std::uint32_t state = seed * 2654435761u + 12345u;
    for (auto& b : host) { state = state * 1664525u + 1013904223u; b = static_cast<std::uint8_t>(state >> 24); }
    auto finite_f16 = [&](std::size_t off) {
        if (off + 1 >= host.size()) return;
        std::uint16_t v = static_cast<std::uint16_t>(0x3800u | (((host[off] << 8) | host[off + 1]) & 0x03FFu));
        host[off] = static_cast<std::uint8_t>(v & 0xFF); host[off + 1] = static_cast<std::uint8_t>(v >> 8);
    };
    if (c.ggml) {
        const std::size_t bb = static_cast<std::size_t>(c.bytes_per_value * (c.qtype == QType::Q8_0 ? 32 : 256) + 0.5);
        for (std::size_t off = 0; off + bb <= host.size(); off += bb) {
            switch (c.qtype) {
            case QType::Q4_K: case QType::Q5_K: finite_f16(off); finite_f16(off + 2); break;   // dm
            case QType::Q6_K: finite_f16(off + 208); break;                                      // d, last
            case QType::Q8_0: finite_f16(off); break;                                            // d, first
            default: break;
            }
        }
    } else {
        // Row-split: a separate fp16 scale plane at scale_offset.
        for (std::size_t off = w.scale_offset; off + 1 < w.scale_offset + w.scale_bytes; off += 2) finite_f16(off);
    }
    CUDA_CHECK(cudaMemcpy(w.storage.p, host.data(), host.size(), cudaMemcpyHostToDevice));
}

bench::PackedQuantizedWeight make_weight(const Codec& c, int n, int k, std::uint8_t seed) {
    bench::QuantizedWeightFill fill{static_cast<std::uint8_t>(0x31U ^ seed), 0xa5, 0x1401};
    auto w = c.ggml ? bench::make_ggml_blocks_weight(c.qtype, n, k, fill)
                    : bench::make_row_split_weight(c.qtype, n, k, k, fill);
    randomize(w, c, seed);
    return w;
}

std::uint16_t bf16(float f) { return bench::f32_to_bf16(f); }

int run(int argc, char** argv) {
    std::string gate = "q4_k", down = "q4_k";
    int iters = 50, flush_mib = 256;
    bool warm = false;
    std::string dump;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if (a == "--gate") gate = next();
        else if (a == "--down") down = next();
        else if (a == "--iters") iters = std::atoi(next().c_str());
        else if (a == "--flush-mib") flush_mib = std::atoi(next().c_str());
        else if (a == "--warm") warm = true;
        else if (a == "--dump") dump = next();
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    const Codec cg = codec_of(gate), cd = codec_of(down);
    constexpr int kSharedInter = kGeometry.shared_intermediate;

    auto routed_gate = make_weight(cg, kExperts * 2 * kIntermediate, kHidden, 0x00);
    auto routed_down = make_weight(cd, kExperts * kHidden, kIntermediate, 0x59);
    bench::PackedQuantizedWeight shared_gate, shared_down;
    if constexpr (kHasShared) {
        shared_gate = bench::make_row_split_weight(QType::W8G32_F16S, 2 * kSharedInter, kHidden,
                                                   kHidden, {0x27, 0x00, 0x1405});
        shared_down = bench::make_row_split_weight(QType::W8G32_F16S, kHidden, kSharedInter,
                                                   kSharedInter, {0x73, 0x00, 0x1407});
        randomize(shared_gate, codec_of("w8"), 0x27);
        randomize(shared_down, codec_of("w8"), 0x73);
    }

    // Router: an identity-like BF16 [router_rows, hidden] so x decides the winners.
    std::vector<std::uint16_t> router(static_cast<std::size_t>(kRouterRows) * kHidden, bf16(0.0F));
    for (int e = 0; e < kExperts; ++e) router[static_cast<std::size_t>(e) * kHidden + e] = bf16(16.0F);
    if constexpr (kHasShared) {
        if (kRouterRows > kExperts) router[static_cast<std::size_t>(kExperts) * kHidden + kExperts] = bf16(4.0F);
    }
    DeviceBuffer router_dev(router.size() * 2);
    router_dev.copy_from_host(router.data(), router_dev.bytes);

    std::vector<std::uint16_t> x(kHidden);
    for (int i = 0; i < kHidden; ++i) x[i] = bf16(static_cast<float>((i * 17) % 81 - 40) * 0.001F);
    for (int e = 0; e < kExperts; ++e) x[e] = bf16(-0.25F);
    int selected[kTopK];
    for (int r = 0; r < kTopK; ++r) { selected[r] = (37 * r + 5) % kExperts; x[selected[r]] = bf16(0.25F - r / 64.0F); }
    if (kExperts < kHidden) x[kExperts] = bf16(0.0625F);
    DeviceBuffer x_dev(x.size() * 2);
    x_dev.copy_from_host(x.data(), x_dev.bytes);
    DeviceBuffer dst_dev(static_cast<std::size_t>(kHidden) * 2);
    CUDA_CHECK(cudaMemset(dst_dev.p, 0, dst_dev.bytes));
    DeviceBuffer bias_dev(static_cast<std::size_t>(kExperts) * 4);
    CUDA_CHECK(cudaMemset(bias_dev.p, 0, bias_dev.bytes));
    DeviceBuffer flush(static_cast<std::size_t>(flush_mib) << 20);

    SparseMoeWeights w{};
    {
        Weight r{};
        r.payload = router_dev.p; r.payload_bytes = router_dev.bytes; r.qtype = QType::BF16_CTRL;
        r.qdata = router_dev.p; r.n = kRouterRows; r.k = kHidden; r.layout = QuantLayout::Contiguous;
        r.ndim = 2; r.shape[0] = r.padded_shape[0] = kRouterRows; r.shape[1] = r.padded_shape[1] = kHidden;
        w.router_shared_gate = r;
    }
    w.router_bias       = kGating == SparseMoeGating::SigmoidBiasTopK ? static_cast<const float*>(bias_dev.p) : nullptr;
    w.routed_scale      = kRoutedScale;
    w.shared_gated      = kSharedGated;
    w.swiglu_limit      = kSwigluLimit;
    w.activation        = kActivation;
    w.routed_gate_up    = routed_gate.weight;
    w.routed_down       = routed_down.weight;
    if constexpr (kHasShared) { w.shared_gate_up = shared_gate.weight; w.shared_down = shared_down.weight; }
    w.experts_per_token = kTopK;

    MallocArena arena;
    SparseMoeDecodeWorkspace ws = allocate_sparse_moe_decode_workspace(arena, kGeometry);
    Tensor xt(x_dev.p, DType::BF16, {kHidden, 1});
    Tensor dt(dst_dev.p, DType::BF16, {kHidden, 1});

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t ev[5];
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
    auto* scores = static_cast<const float*>(ws.scratch.data);

    std::vector<float> t1, t2, t3, t4;
    for (int it = 0; it < iters + 5; ++it) {
        if (!warm) CUDA_CHECK(cudaMemsetAsync(flush.p, 0xa5, flush.bytes, stream));
        CUDA_CHECK(cudaEventRecord(ev[0], stream));
        launch_d1(xt, w.router_shared_gate, ws, stream);
        CUDA_CHECK(cudaEventRecord(ev[1], stream));
        sparse_moe_d2_warp_kernel<<<1, 32, 0, stream>>>(scores, static_cast<int*>(ws.ids.data),
            static_cast<float*>(ws.alpha.data), static_cast<float*>(ws.shared_scale.data),
            w.router_bias, w.per_expert_scale);
        CUDA_CHECK(cudaEventRecord(ev[2], stream));
        launch_d2_d3(xt, w, dt, ws, stream, nullptr);   // D2 again (idempotent) + D3
        CUDA_CHECK(cudaEventRecord(ev[3], stream));
        launch_d4_dependent(w, dt, ws, stream);
        CUDA_CHECK(cudaEventRecord(ev[4], stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (it < 5) continue;
        float a, b, c, d;
        CUDA_CHECK(cudaEventElapsedTime(&a, ev[0], ev[1]));
        CUDA_CHECK(cudaEventElapsedTime(&b, ev[1], ev[2]));
        CUDA_CHECK(cudaEventElapsedTime(&c, ev[2], ev[3]));
        CUDA_CHECK(cudaEventElapsedTime(&d, ev[3], ev[4]));
        t1.push_back(a * 1000); t2.push_back(b * 1000); t3.push_back((c - b) * 1000); t4.push_back(d * 1000);
    }
    if (!dump.empty()) {
        // One more run from a zeroed destination, so the dump is one application of the op.
        CUDA_CHECK(cudaMemset(dst_dev.p, 0, dst_dev.bytes));
        launch_d1(xt, w.router_shared_gate, ws, stream);
        launch_d2_d3(xt, w, dt, ws, stream, nullptr);
        launch_d4_dependent(w, dt, ws, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::uint16_t> out(kHidden);
        std::vector<float> act(static_cast<std::size_t>(kPaths) * kIntermediate);
        CUDA_CHECK(cudaMemcpy(out.data(), dst_dev.p, out.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(act.data(), ws.scratch.data, act.size() * 4, cudaMemcpyDeviceToHost));
        FILE* f = std::fopen(dump.c_str(), "wb");
        std::fwrite(out.data(), 2, out.size(), f);
        std::fwrite(act.data(), 4, act.size(), f);
        std::fclose(f);
    }
    auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    // Verify the winners are the ones x asked for.
    std::vector<int> ids(kTopK);
    CUDA_CHECK(cudaMemcpy(ids.data(), ws.ids.data, kTopK * 4, cudaMemcpyDeviceToHost));
    std::sort(ids.begin(), ids.end());
    std::vector<int> want(selected, selected + kTopK);
    std::sort(want.begin(), want.end());
    const bool routing_ok = ids == want;

    const double gate_bytes = static_cast<double>(kTopK) * 2 * kIntermediate * kHidden * cg.bytes_per_value
                            + (kHasShared ? 2.0 * kSharedInter * kHidden * (1 + 2.0 / 32) : 0.0);
    const double down_bytes = static_cast<double>(kTopK) * kHidden * kIntermediate * cd.bytes_per_value
                            + (kHasShared ? 1.0 * kHidden * kSharedInter * (1 + 2.0 / 32) : 0.0);
    const double d3 = med(t3), d4 = med(t4);
    std::printf("geometry hidden=%d experts=%d topk=%d inter=%d shared=%d  gate=%s down=%s  %s  routing=%s\n",
                kHidden, kExperts, kTopK, kIntermediate, kSharedInter, gate.c_str(), down.c_str(),
                warm ? "warm" : "cold", routing_ok ? "ok" : "WRONG");
    std::printf("D1 router %7.2f us | D2 select %6.2f us | D3 gate/up %7.2f us = %6.1f GB/s (%4.1f%% of 1792) | D4 down %7.2f us = %6.1f GB/s (%4.1f%%) | D3+D4 %7.2f us\n",
                med(t1), med(t2), d3, gate_bytes / d3 / 1e3, 100 * gate_bytes / d3 / 1e3 / 1792,
                d4, down_bytes / d4 / 1e3, 100 * down_bytes / d4 / 1e3 / 1792, d3 + d4);
    return routing_ok ? 0 : 1;
}

} // namespace geometry_harness
} // namespace sinfer::ops::detail

int main(int argc, char** argv) { return sinfer::ops::detail::geometry_harness::run(argc, argv); }
