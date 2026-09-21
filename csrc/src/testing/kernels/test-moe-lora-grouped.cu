// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

// Fused grouped expert LoRA (moe_lora_grouped.cu): forward and backward against
// an fp64 host oracle over experts with varied token counts (including empty
// and single-token experts), ranks 8/16/32, both Gemma4 expert shapes, several
// tile sizes, both accumulate modes, grids sized with and without the host copy
// of the offsets, and 300 experts (multi-pass tile scan); reruns must be
// bit-identical; unsupported shapes and a short workspace launch nothing.
#include <catch2/catch_test_macros.hpp>
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
