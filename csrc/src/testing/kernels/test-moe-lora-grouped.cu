// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

// Fused grouped expert LoRA (moe_lora_grouped.cu): forward and backward against
// an fp64 host oracle over experts with varied token counts (including empty
// and single-token experts), ranks 8/16/32, both Gemma4 expert shapes, several
// tile sizes, both accumulate modes, grids sized with and without the host copy
// of the offsets, and 300 experts (multi-pass tile scan); reruns must be
// bit-identical; unsupported shapes and a short workspace launch nothing.
//
// Compact expert LoRA (lora_expert_mask): the cuBLAS-path LoRA sequence of the
// grouped MoE ops with [K, ...] adapters over an active-expert list equals the
// full [E, ...] layout whose B is zero for the other experts, bit for bit, and
// stale (NaN) temporaries cannot leak into the output once the product
// temporary is cleared before its B-GEMM.
#include <catch2/catch_test_macros.hpp>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels/kernels.h"

namespace {

template <class T>
struct Device {
    T* ptr = nullptr;
    explicit Device(const std::vector<T>& values) {
        REQUIRE(cudaMalloc(&ptr, std::max<std::size_t>(values.size(), 1) * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMemcpy(ptr, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    ~Device() {
        cudaFree(ptr);
    }
    std::vector<T> read(std::size_t n) const {
        std::vector<T> result(n);
        REQUIRE(cudaMemcpy(result.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
        return result;
    }
};

struct Problem {
    int in;
    int out;
    int rank;
    float scaling;
    std::vector<int> counts;  // tokens per expert
};

struct Inputs {
    int E = 0;
    int T = 0;
    std::vector<int> offsets;
    std::vector<nv_bfloat16> x, A, B, out0, d_out, dx0, dA0, dB0;
};

Inputs make_inputs(const Problem& p, unsigned seed) {
    Inputs in;
    in.E = static_cast<int>(p.counts.size());
    in.offsets.assign(in.E + 1, 0);
    for (int e = 0; e < in.E; ++e) {
        in.offsets[e + 1] = in.offsets[e] + p.counts[e];
    }
    in.T = in.offsets[in.E];
    std::mt19937 gen(seed);
    auto fill = [&](std::vector<nv_bfloat16>& v, std::size_t n, float scale) {
        std::uniform_real_distribution<float> dist(-scale, scale);
        v.resize(n);
        for (auto& e : v) {
            e = __float2bfloat16(dist(gen));
        }
    };
    const std::size_t T = in.T, E = in.E, R = p.rank, I = p.in, O = p.out;
    fill(in.x, T * I, 1.0f);
    fill(in.A, E * R * I, 1.0f / std::sqrt(static_cast<float>(I)));
    fill(in.B, E * O * R, 0.5f);
    fill(in.out0, T * O, 1.0f);
    fill(in.d_out, T * O, 1.0f);
    fill(in.dx0, T * I, 1.0f);
    fill(in.dA0, E * R * I, 1.0f);
    fill(in.dB0, E * O * R, 1.0f);
    return in;
}

struct Oracle {
    std::vector<double> out, dx, dA, dB;
};

// fp64 reference of the fused semantics; experts without tokens keep their
// gradient slabs untouched (the cuBLAS grouped path skips them as well).
Oracle reference(const Problem& p, const Inputs& in, bool out_accumulate, bool grad_accumulate, bool dx_accumulate) {
    const int R = p.rank, I = p.in, O = p.out;
    const double s = p.scaling;
    auto f = [](nv_bfloat16 v) {
        return static_cast<double>(__bfloat162float(v));
    };
    Oracle o;
    o.out.assign(in.out0.size(), 0.0);
    o.dx.assign(in.dx0.size(), 0.0);
    o.dA.resize(in.dA0.size());
    o.dB.resize(in.dB0.size());
    for (std::size_t i = 0; i < in.dA0.size(); ++i)
        o.dA[i] = f(in.dA0[i]);
    for (std::size_t i = 0; i < in.dB0.size(); ++i)
        o.dB[i] = f(in.dB0[i]);
    std::vector<double> h(R), g(R);
    for (int e = 0; e < in.E; ++e) {
        const int t0 = in.offsets[e], t1 = in.offsets[e + 1];
        if (t0 == t1) continue;
        const nv_bfloat16* A = in.A.data() + static_cast<std::size_t>(e) * R * I;
        const nv_bfloat16* B = in.B.data() + static_cast<std::size_t>(e) * O * R;
        double* dA = o.dA.data() + static_cast<std::size_t>(e) * R * I;
        double* dB = o.dB.data() + static_cast<std::size_t>(e) * O * R;
        if (!grad_accumulate) {
            std::fill(dA, dA + static_cast<std::size_t>(R) * I, 0.0);
            std::fill(dB, dB + static_cast<std::size_t>(O) * R, 0.0);
        }
        for (int t = t0; t < t1; ++t) {
            const nv_bfloat16* x = in.x.data() + static_cast<std::size_t>(t) * I;
            const nv_bfloat16* dy = in.d_out.data() + static_cast<std::size_t>(t) * O;
            for (int r = 0; r < R; ++r) {
                double acc = 0.0;
                for (int k = 0; k < I; ++k)
                    acc += f(x[k]) * f(A[r * I + k]);
                h[r] = acc;
                acc = 0.0;
                for (int k = 0; k < O; ++k)
                    acc += f(dy[k]) * f(B[k * R + r]);
                g[r] = acc;
            }
            for (int c = 0; c < O; ++c) {
                double acc = 0.0;
                for (int r = 0; r < R; ++r)
                    acc += h[r] * f(B[c * R + r]);
                o.out[static_cast<std::size_t>(t) * O + c] =
                    (out_accumulate ? f(in.out0[static_cast<std::size_t>(t) * O + c]) : 0.0) + s * acc;
                for (int r = 0; r < R; ++r)
                    dB[c * R + r] += s * f(dy[c]) * h[r];
            }
            for (int c = 0; c < I; ++c) {
                double acc = 0.0;
                for (int r = 0; r < R; ++r)
                    acc += g[r] * f(A[r * I + c]);
                o.dx[static_cast<std::size_t>(t) * I + c] =
                    (dx_accumulate ? f(in.dx0[static_cast<std::size_t>(t) * I + c]) : 0.0) + s * acc;
                for (int r = 0; r < R; ++r)
                    dA[r * I + c] += s * g[r] * f(x[c]);
            }
        }
    }
    return o;
}

double bf16_half_ulp(double v) {
    v = std::fabs(v);
    if (v == 0.0) return 0.0;
    int exp = 0;
    std::frexp(v, &exp);  // v = m * 2^exp, m in [0.5, 1)
    return std::ldexp(1.0, exp - 1 - 8);
}

// got is bf16, ref is fp64. Aggregate: the relative L2 distance to the
// bf16-rounded reference is at most 2e-3 -- the output rounding cancels, so this
// isolates fp32 accumulation and reduction-order error (an element that lands on
// the other side of a rounding boundary costs one bf16 ulp; few do). Per element:
// 2e-3 relative plus half a bf16 ulp of the result plus 2e-3 of the tensor's RMS
// for entries that cancel to ~0.
void check_close(const std::vector<nv_bfloat16>& got, const std::vector<double>& ref, const std::string& what) {
    REQUIRE(got.size() == ref.size());
    double sq_err = 0.0, sq_ref = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double ref_bf16 = static_cast<double>(__bfloat162float(__float2bfloat16(static_cast<float>(ref[i]))));
        const double d = static_cast<double>(__bfloat162float(got[i])) - ref_bf16;
        sq_err += d * d;
        sq_ref += ref[i] * ref[i];
    }
    const double rms = std::sqrt(sq_ref / static_cast<double>(std::max<std::size_t>(ref.size(), 1)));
    INFO(what << ": relative L2 error vs bf16(ref) " << std::sqrt(sq_err / std::max(sq_ref, 1e-300)));
    REQUIRE(std::sqrt(sq_err) <= 2e-3 * std::sqrt(sq_ref) + 1e-12);
    std::size_t worst = 0;
    double worst_excess = -1.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs(static_cast<double>(__bfloat162float(got[i])) - ref[i]);
        const double tol = 2e-3 * std::fabs(ref[i]) + bf16_half_ulp(ref[i]) + 2e-3 * rms;
        if (d - tol > worst_excess) {
            worst_excess = d - tol;
            worst = i;
        }
    }
    INFO(what << "[" << worst << "]: got " << __bfloat162float(got[worst]) << " expected " << ref[worst]);
    REQUIRE(worst_excess <= 0.0);
}

void require_same_bits(const std::vector<nv_bfloat16>& a, const std::vector<nv_bfloat16>& b, const std::string& what) {
    REQUIRE(a.size() == b.size());
    INFO(what);
    REQUIRE(std::memcmp(a.data(), b.data(), a.size() * sizeof(nv_bfloat16)) == 0);
}

struct Mode {
    int tile;         // 0 = automatic
    bool accumulate;  // out (beta 1) and dA/dB (grad_accumulate)
    bool dx_accumulate;
    bool use_host_offsets;  // false: upper-bound grid, surplus CTAs exit in the kernel
};

void run_case(const Problem& p, const Inputs& in, const Mode& m, const Oracle& ref) {
    const std::string tag = "in=" + std::to_string(p.in) + " out=" + std::to_string(p.out) +
                            " rank=" + std::to_string(p.rank) + " E=" + std::to_string(in.E) +
                            " tile=" + std::to_string(m.tile) + " accumulate=" + std::to_string(m.accumulate) +
                            " dx_accumulate=" + std::to_string(m.dx_accumulate) +
                            " host_offsets=" + std::to_string(m.use_host_offsets);
    INFO(tag);
    const int tile = m.tile;
    const bool accumulate = m.accumulate;
    const int* host_offsets = m.use_host_offsets ? in.offsets.data() : nullptr;
    Device<int> offsets(in.offsets);
    Device<nv_bfloat16> x(in.x), A(in.A), B(in.B), d_out(in.d_out);

    // Forward, twice: oracle match and bit-identical reruns.
    std::vector<nv_bfloat16> out_first;
    for (int run = 0; run < 2; ++run) {
        Device<nv_bfloat16> out(in.out0);
        REQUIRE(moe_lora_grouped_forward_bf16(out.ptr,
                                              x.ptr,
                                              A.ptr,
                                              B.ptr,
                                              offsets.ptr,
                                              host_offsets,
                                              in.E,
                                              in.T,
                                              p.in,
                                              p.out,
                                              p.rank,
                                              p.scaling,
                                              accumulate,
                                              nullptr,
                                              tile));
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        const auto got = out.read(in.out0.size());
        if (run == 0) {
            check_close(got, ref.out, tag + " out");
            out_first = got;
        } else {
            require_same_bits(got, out_first, tag + " forward determinism");
        }
    }

    // Backward, twice.
    const std::size_t ws_floats =
        moe_lora_grouped_backward_workspace_floats(host_offsets, in.E, in.T, p.in, p.out, p.rank, tile);
    REQUIRE(ws_floats > 0);
    std::vector<nv_bfloat16> dx_first, dA_first, dB_first;
    for (int run = 0; run < 2; ++run) {
        Device<nv_bfloat16> dx(in.dx0), dA(in.dA0), dB(in.dB0);
        Device<float> workspace(std::vector<float>(ws_floats, 12345.0f));  // never read before written
        REQUIRE(moe_lora_grouped_backward_bf16(dx.ptr,
                                               dA.ptr,
                                               dB.ptr,
                                               d_out.ptr,
                                               x.ptr,
                                               A.ptr,
                                               B.ptr,
                                               offsets.ptr,
                                               host_offsets,
                                               in.E,
                                               in.T,
                                               p.in,
                                               p.out,
                                               p.rank,
                                               p.scaling,
                                               m.dx_accumulate,
                                               /*grad_accumulate=*/accumulate,
                                               workspace.ptr,
                                               ws_floats,
                                               nullptr,
                                               tile));
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        const auto got_dx = dx.read(in.dx0.size());
        const auto got_dA = dA.read(in.dA0.size());
        const auto got_dB = dB.read(in.dB0.size());
        if (run == 0) {
            check_close(got_dx, ref.dx, tag + " dx");
            check_close(got_dA, ref.dA, tag + " dA");
            check_close(got_dB, ref.dB, tag + " dB");
            dx_first = got_dx;
            dA_first = got_dA;
            dB_first = got_dB;
        } else {
            require_same_bits(got_dx, dx_first, tag + " dx determinism");
            require_same_bits(got_dA, dA_first, tag + " dA determinism");
            require_same_bits(got_dB, dB_first, tag + " dB determinism");
        }
    }

    // dx only (no weight gradients): one launch, no workspace.
    {
        Device<nv_bfloat16> dx(in.dx0);
        REQUIRE(moe_lora_grouped_backward_bf16(dx.ptr,
                                               nullptr,
                                               nullptr,
                                               d_out.ptr,
                                               x.ptr,
                                               A.ptr,
                                               B.ptr,
                                               offsets.ptr,
                                               host_offsets,
                                               in.E,
                                               in.T,
                                               p.in,
                                               p.out,
                                               p.rank,
                                               p.scaling,
                                               m.dx_accumulate,
                                               accumulate,
                                               nullptr,
                                               0,
                                               nullptr,
                                               tile));
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        require_same_bits(dx.read(in.dx0.size()), dx_first, tag + " dx-only path");
    }
}

}  // namespace

TEST_CASE("Fused grouped expert LoRA matches the fp64 oracle and is deterministic", "[moe][lora][grouped]") {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) SKIP("CUDA device required");

    // Empty experts, a single-token expert, experts spanning several 32- and
    // 256-token tiles, and a count that is not a multiple of any tile.
    const std::vector<int> counts = {37, 0, 1, 300, 0, 64, 129};
    const float scaling = 2.0f;  // alpha 16 / rank 8
    unsigned seed = 7;
    for (const auto& shape : std::vector<std::pair<int, int>>{{2816, 1408}, {704, 2816}}) {
        for (int rank : {8, 16, 32}) {
            const Problem p{shape.first, shape.second, rank, scaling, counts};
            const Inputs in = make_inputs(p, seed++);
            for (bool accumulate : {false, true}) {
                // accumulate == false also covers dx_accumulate == false.
                const Oracle ref = reference(p, in, accumulate, accumulate, accumulate);
                for (int tile : {0, 32, 256}) {
                    run_case(p, in, Mode{tile, accumulate, accumulate, true}, ref);
                }
                run_case(p, in, Mode{0, accumulate, accumulate, false}, ref);  // upper-bound grid, tile 64
            }
        }
    }
}

TEST_CASE("Fused grouped expert LoRA handles more experts than threads in the tile scan", "[moe][lora][grouped]") {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) SKIP("CUDA device required");
    // 300 experts > 256 threads: the expert scan needs two passes. Mostly tiny
    // or empty experts, with multi-tile experts in both passes.
    std::vector<int> counts(300);
    for (int e = 0; e < 300; ++e) {
        counts[e] = (e * 7) % 5;
    }
    counts[0] = 70;
    counts[299] = 33;
    const Problem p{256, 128, 8, 0.75f, counts};
    const Inputs in = make_inputs(p, 11);
    const Oracle ref = reference(p, in, true, true, true);
    run_case(p, in, Mode{32, true, true, true}, ref);
    run_case(p, in, Mode{0, true, true, false}, ref);
}

TEST_CASE("Fused grouped expert LoRA rejects unsupported shapes without launching", "[moe][lora][grouped]") {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) SKIP("CUDA device required");
    const Problem p{64, 64, 8, 1.0f, {3, 5}};
    const Inputs in = make_inputs(p, 3);
    Device<int> offsets(in.offsets);
    Device<nv_bfloat16> x(in.x), A(in.A), B(in.B), out(in.out0);
    auto forward = [&](int in_features, int out_features, int rank) {
        return moe_lora_grouped_forward_bf16(out.ptr,
                                             x.ptr,
                                             A.ptr,
                                             B.ptr,
                                             offsets.ptr,
                                             in.offsets.data(),
                                             in.E,
                                             in.T,
                                             in_features,
                                             out_features,
                                             rank,
                                             1.0f,
                                             true,
                                             nullptr);
    };
    REQUIRE(forward(64, 64, 8));
    REQUIRE_FALSE(forward(64, 64, 12));  // rank
    REQUIRE_FALSE(forward(60, 64, 8));   // width not a multiple of 8
    REQUIRE_FALSE(forward(64, 60, 8));
    REQUIRE(moe_lora_grouped_backward_workspace_floats(in.offsets.data(), in.E, in.T, 60, 64, 8) == 0);

    // A workspace one float short must be refused before anything is launched:
    // dx keeps its initial bits.
    const std::size_t ws_floats = moe_lora_grouped_backward_workspace_floats(in.offsets.data(), in.E, in.T, 64, 64, 8);
    REQUIRE(ws_floats > 0);
    Device<nv_bfloat16> dx(in.dx0), dA(in.dA0), dB(in.dB0), d_out(in.d_out);
    Device<float> workspace(std::vector<float>(ws_floats, 0.0f));
    REQUIRE_FALSE(moe_lora_grouped_backward_bf16(dx.ptr,
                                                 dA.ptr,
                                                 dB.ptr,
                                                 d_out.ptr,
                                                 x.ptr,
                                                 A.ptr,
                                                 B.ptr,
                                                 offsets.ptr,
                                                 in.offsets.data(),
                                                 in.E,
                                                 in.T,
                                                 64,
                                                 64,
                                                 8,
                                                 1.0f,
                                                 true,
                                                 true,
                                                 workspace.ptr,
                                                 ws_floats - 1,
                                                 nullptr));
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    require_same_bits(dx.read(in.dx0.size()), in.dx0, "workspace rejection leaves dx untouched");
    require_same_bits(dA.read(in.dA0.size()), in.dA0, "workspace rejection leaves dA untouched");
}

// ---------------------------------------------------------------------------
// Compact expert LoRA (lora_expert_mask)
// ---------------------------------------------------------------------------

namespace {

nv_bfloat16 bf16_nan() {
    const std::uint16_t bits = 0x7FC0;
    nv_bfloat16 v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

bool all_finite(const std::vector<nv_bfloat16>& v) {
    return std::all_of(v.begin(), v.end(), [](nv_bfloat16 e) { return std::isfinite(__bfloat162float(e)); });
}

struct MaskedProblem {
    int E = 0;
    int in = 0;
    int out = 0;
    int rank = 0;
    float scaling = 1.0f;
    std::vector<int> offsets;  // host copy, E + 1 entries
    int T = 0;                 // routed rows (tokens x top-k)
};

struct MaskedResult {
    std::vector<nv_bfloat16> out, dx, dA, dB;
};

// The grouped expert-LoRA sequence of the MoE ops' cuBLAS path for one projection
// (moe_grouped_gemm_gate_up.cpp), with the expert arguments those ops pass:
//   forward   h = s * (x A^T);  tmp = h B^T (beta 0);  out += tmp     -- the whole temporary is added
//   backward  h = s * (x A^T);  dB = d_out^T h;  g = s * (d_out B);  dA = g^T x;  dx += g A
// `active` = nullptr: the [E, ...] layout. Otherwise the [K, ...] compact layout, row s = expert active[s],
// as the ops run a lora_expert_mask layer. `poison` fills every temporary with NaN first (what an earlier
// layer or step may leave in the run state's reused buffers); `clear_temp` is the ops' fix: tmp is zeroed
// before its B-GEMM, because that GEMM writes the selected experts' rows only.
MaskedResult run_masked_sequence(const MaskedProblem& p,
                                 const std::vector<nv_bfloat16>& x,
                                 const std::vector<nv_bfloat16>& A,
                                 const std::vector<nv_bfloat16>& B,
                                 const std::vector<nv_bfloat16>& out0,
                                 const std::vector<nv_bfloat16>& d_out,
                                 const std::vector<nv_bfloat16>& dx0,
                                 const std::vector<int>* active,
                                 bool clear_temp,
                                 bool poison,
                                 cublasHandle_t handle) {
    const bool compact = active != nullptr;
    const int rows = compact ? static_cast<int>(active->size()) : p.E;
    const int* act = compact ? active->data() : nullptr;
    const int n_act = compact ? static_cast<int>(active->size()) : -1;
    const std::size_t T = p.T, R = p.rank, I = p.in, O = p.out;
    const nv_bfloat16 zero = __float2bfloat16(0.0f);
    const nv_bfloat16 fill = poison ? bf16_nan() : zero;
    const int* host = p.offsets.data();
    cudaStream_t stream = nullptr;

    Device<int> offsets(p.offsets);
    Device<nv_bfloat16> xd(x), Ad(A), Bd(B), outd(out0), doutd(d_out), dxd(dx0);
    Device<nv_bfloat16> h(std::vector<nv_bfloat16>(T * R, fill)), g(std::vector<nv_bfloat16>(T * R, fill));
    Device<nv_bfloat16> tmp(std::vector<nv_bfloat16>(T * O, fill));
    Device<nv_bfloat16> dA(std::vector<nv_bfloat16>(static_cast<std::size_t>(rows) * R * I, zero));
    Device<nv_bfloat16> dB(std::vector<nv_bfloat16>(static_cast<std::size_t>(rows) * O * R, zero));
    const float half_s = 0.5f * p.scaling;  // vector_add(t, t, t, 0.5 s) scales by s, as the ops do

    // Forward.
    moe_grouped_gemm(h.ptr, xd.ptr, Ad.ptr, offsets.ptr, p.E, p.rank, p.in, handle, stream, host, 1.0f, 0.0f,
                     EMMTranspose::TN, act, compact, n_act);
    vector_add(h.ptr, h.ptr, h.ptr, half_s, static_cast<long>(T * R), stream);
    if (clear_temp) REQUIRE(cudaMemsetAsync(tmp.ptr, 0, T * O * sizeof(nv_bfloat16), stream) == cudaSuccess);
    moe_grouped_gemm(tmp.ptr, h.ptr, Bd.ptr, offsets.ptr, p.E, p.out, p.rank, handle, stream, host, 1.0f, 0.0f,
                     EMMTranspose::TN, act, compact, n_act);
    vector_add(outd.ptr, outd.ptr, tmp.ptr, 1.0f, static_cast<long>(T * O), stream);

    // Backward: the ops recompute h into the same (possibly stale) buffer.
    if (poison) {
        Device<nv_bfloat16> nan_rows(std::vector<nv_bfloat16>(T * R, fill));
        REQUIRE(cudaMemcpy(h.ptr, nan_rows.ptr, T * R * sizeof(nv_bfloat16), cudaMemcpyDeviceToDevice) ==
                cudaSuccess);
    }
    moe_grouped_gemm(h.ptr, xd.ptr, Ad.ptr, offsets.ptr, p.E, p.rank, p.in, handle, stream, host, 1.0f, 0.0f,
                     EMMTranspose::TN, act, compact, n_act);
    vector_add(h.ptr, h.ptr, h.ptr, half_s, static_cast<long>(T * R), stream);
    moe_grouped_gemm_weight_grad(dB.ptr, doutd.ptr, h.ptr, offsets.ptr, p.E, p.out, p.rank, handle, stream, host,
                                 1.0f, 0.0f, act, compact, n_act);
    moe_grouped_gemm(g.ptr, doutd.ptr, Bd.ptr, offsets.ptr, p.E, p.rank, p.out, handle, stream, host, 1.0f, 0.0f,
                     EMMTranspose::NN, act, compact, n_act);
    vector_add(g.ptr, g.ptr, g.ptr, half_s, static_cast<long>(T * R), stream);
    moe_grouped_gemm_weight_grad(dA.ptr, g.ptr, xd.ptr, offsets.ptr, p.E, p.rank, p.in, handle, stream, host, 1.0f,
                                 0.0f, act, compact, n_act);
    moe_grouped_gemm(dxd.ptr, g.ptr, Ad.ptr, offsets.ptr, p.E, p.in, p.rank, handle, stream, host, 1.0f, 1.0f,
                     EMMTranspose::NN, act, compact, n_act);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);

    MaskedResult r;
    r.out = outd.read(T * O);
    r.dx = dxd.read(T * I);
    r.dA = dA.read(static_cast<std::size_t>(rows) * R * I);
    r.dB = dB.read(static_cast<std::size_t>(rows) * O * R);
    return r;
}

std::vector<nv_bfloat16> expert_rows(const std::vector<nv_bfloat16>& full, std::size_t row_elems,
                                     const std::vector<int>& ids) {
    std::vector<nv_bfloat16> rows;
    rows.reserve(ids.size() * row_elems);
    for (int e : ids) {
        rows.insert(rows.end(), full.begin() + static_cast<long>(e * row_elems),
                    full.begin() + static_cast<long>((e + 1) * row_elems));
    }
    return rows;
}

void run_masked_case(int rank, const std::vector<int>& counts, const std::vector<int>& active, unsigned seed) {
    MaskedProblem p;
    p.E = static_cast<int>(counts.size());
    p.in = 64;   // C
    p.out = 32;  // M (expert width)
    p.rank = rank;
    p.scaling = 2.0f;
    p.offsets.assign(p.E + 1, 0);
    for (int e = 0; e < p.E; ++e) p.offsets[e + 1] = p.offsets[e] + counts[e];
    p.T = p.offsets[p.E];
    INFO("rank=" << rank << " seed=" << seed << " rows=" << p.T);
    const std::size_t T = p.T, R = rank, I = p.in, O = p.out, E = p.E;

    std::mt19937 gen(seed);
    auto random = [&](std::size_t n, float scale) {
        std::uniform_real_distribution<float> dist(-scale, scale);
        std::vector<nv_bfloat16> v(n);
        for (auto& e : v) e = __float2bfloat16(dist(gen));
        return v;
    };
    const auto x = random(T * I, 1.0f), out0 = random(T * O, 1.0f), d_out = random(T * O, 1.0f),
               dx0 = random(T * I, 1.0f);
    const auto A_full = random(E * R * I, 0.125f);
    auto B_full = random(E * O * R, 0.5f);
    // The dense reference: B of every unselected expert is zero, so only the selected experts adapt.
    for (int e = 0; e < p.E; ++e) {
        if (std::find(active.begin(), active.end(), e) == active.end()) {
            std::fill(B_full.begin() + static_cast<long>(e * O * R), B_full.begin() + static_cast<long>((e + 1) * O * R),
                      __float2bfloat16(0.0f));
        }
    }
    const auto A_compact = expert_rows(A_full, R * I, active);
    const auto B_compact = expert_rows(B_full, O * R, active);

    cublasHandle_t handle{};
    REQUIRE(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);
    const MaskedResult dense =
        run_masked_sequence(p, x, A_full, B_full, out0, d_out, dx0, nullptr, /*clear_temp=*/false, false, handle);
    const MaskedResult compact =
        run_masked_sequence(p, x, A_compact, B_compact, out0, d_out, dx0, &active, /*clear_temp=*/true, false, handle);
    // Forward output and dx: equal bit for bit (same per-expert GEMMs; the other experts add exactly zero).
    require_same_bits(compact.out, dense.out, "compact out == dense out with B = 0 elsewhere");
    require_same_bits(compact.dx, dense.dx, "compact dx == dense dx");
    // Unselected experts leave their rows untouched: out = out0 and dx = dx0 there.
    for (int e = 0; e < p.E; ++e) {
        if (std::find(active.begin(), active.end(), e) != active.end()) continue;
        for (std::size_t t = static_cast<std::size_t>(p.offsets[e]); t < static_cast<std::size_t>(p.offsets[e + 1]);
             ++t) {
            REQUIRE(std::memcmp(&compact.out[t * O], &out0[t * O], O * sizeof(nv_bfloat16)) == 0);
            REQUIRE(std::memcmp(&compact.dx[t * I], &dx0[t * I], I * sizeof(nv_bfloat16)) == 0);
        }
    }
    // The selected experts' gradients: equal bit for bit.
    require_same_bits(compact.dA, expert_rows(dense.dA, R * I, active), "compact dA == dense dA (selected)");
    require_same_bits(compact.dB, expert_rows(dense.dB, O * R, active), "compact dB == dense dB (selected)");
    REQUIRE(all_finite(compact.out));
    REQUIRE(all_finite(compact.dA));
    REQUIRE(all_finite(compact.dB));

    // Poison: NaN in every temporary. With the product temporary cleared before its B-GEMM, as the ops do,
    // nothing stale reaches out, dx, dA or dB: same bits as before.
    const MaskedResult poisoned =
        run_masked_sequence(p, x, A_compact, B_compact, out0, d_out, dx0, &active, /*clear_temp=*/true, true, handle);
    REQUIRE(all_finite(poisoned.out));
    REQUIRE(all_finite(poisoned.dx));
    require_same_bits(poisoned.out, compact.out, "poisoned temporaries: out unchanged");
    require_same_bits(poisoned.dx, compact.dx, "poisoned temporaries: dx unchanged");
    require_same_bits(poisoned.dA, compact.dA, "poisoned temporaries: dA unchanged");
    require_same_bits(poisoned.dB, compact.dB, "poisoned temporaries: dB unchanged");
    // ... and without that clear the stale rows do leak (the test can see the bug it guards against),
    // whenever an unselected expert has tokens.
    bool unselected_tokens = false;
    for (int e = 0; e < p.E; ++e) {
        if (std::find(active.begin(), active.end(), e) == active.end() && counts[e] > 0) unselected_tokens = true;
    }
    if (unselected_tokens) {
        const MaskedResult leaked = run_masked_sequence(p, x, A_compact, B_compact, out0, d_out, dx0, &active,
                                                        /*clear_temp=*/false, true, handle);
        REQUIRE_FALSE(all_finite(leaked.out));
    }
    REQUIRE(cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS);
}

}  // namespace

TEST_CASE("Compact grouped expert LoRA equals the dense LoRA with unselected experts zeroed",
          "[moe][lora][grouped][mask]") {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) SKIP("CUDA device required");
    // E = 16, top-2 over 32 tokens: 64 routed rows, random per-expert counts (seeded), active {1, 5, 9}.
    std::mt19937 gen(20261018);
    std::vector<int> counts(16, 0);
    std::uniform_int_distribution<int> pick(0, 15);
    for (int t = 0; t < 32; ++t) {
        const int a = pick(gen);
        int b = pick(gen);
        while (b == a) b = pick(gen);
        ++counts[a];
        ++counts[b];
    }
    const std::vector<int> active = {1, 5, 9};
    for (int rank : {4, 8}) {  // 4: the per-expert cublasGemmEx loop; 8: the CUTLASS grouped kernel
        run_masked_case(rank, counts, active, 11u + static_cast<unsigned>(rank));
        // A selected expert without tokens, and an expert list of one.
        auto sparse = counts;
        sparse[5] = 0;
        sparse[6] += counts[5];
        run_masked_case(rank, sparse, active, 23u + static_cast<unsigned>(rank));
        run_masked_case(rank, counts, std::vector<int>{15}, 31u + static_cast<unsigned>(rank));
    }
}
