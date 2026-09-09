// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#include "runtime/jit/kimi_delta_rule_kernels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "kernels/glm5.h"
#include "utilities/utils.h"

namespace {
constexpr std::array Names = {"kda_metadata",
                              "kda_indices",
                              "kda_norm_fwd",
                              "kda_norm_bwd",
                              "kda_cumsum_fwd",
                              "kda_cumsum_rev",
                              "kda_intra_fwd",
                              "kda_solve",
                              "kda_wy_fwd",
                              "kda_state_fwd",
                              "kda_output",
                              "kda_dav",
                              "kda_state_bwd",
                              "kda_wy_bwd",
                              "kda_intra_bwd",
                              "kda_beta_bwd"};
int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

struct Workspace {
    std::size_t bytes = 0;
    int N, NT, tokens;
    void *cu, *offsets, *indices, *qn, *kn, *qr, *kr, *gc, *Aqk, *Akk, *Akkd, *w, *u, *qg, *kg, *h, *vn;
    void *dout = nullptr, *dv = nullptr, *dh = nullptr, *dvs = nullptr, *dq = nullptr, *dk = nullptr, *dg = nullptr,
         *db = nullptr, *dAqk = nullptr, *dAkk = nullptr, *dq2 = nullptr, *dk2 = nullptr, *dg2 = nullptr,
         *dbp = nullptr;

    Workspace(std::byte* data, int B, int T, int H, int D, int num_docs, bool backward) {
        if (B <= 0 || T <= 0 || H <= 0 || D <= 0 || num_docs < 0 ||
            static_cast<long>(B) * T * H > std::numeric_limits<int>::max())
            throw std::runtime_error("Invalid KDA dimensions");
        tokens = B * T;
        N = num_docs > 0 ? num_docs : B;
        // sum ceil(doc_length/64) <= ceil(tokens/64) + N - 1, also <= tokens.
        // This bound is independent of document lengths during graph replay.
        NT = std::min(tokens, ceildiv(tokens, 64) + N - 1);
        auto alloc = [&](std::size_t count, int elem_bytes) -> void* {
            auto offset = bytes;
            bytes += (count * elem_bytes + 255) / 256 * 256;
            return data ? data + offset : nullptr;
        };
        const std::size_t rows = static_cast<std::size_t>(tokens) * H, n = rows * D;
        cu = alloc(N + 1, 4);
        offsets = alloc(N + 1, 4);
        indices = alloc(2L * NT, 4);
        qn = alloc(n, 2);
        kn = alloc(n, 2);
        qr = alloc(rows, 4);
        kr = alloc(rows, 4);
        gc = alloc(n, 4);
        Aqk = alloc(rows * 64, 2);
        Akk = alloc(rows * 64, 2);
        Akkd = alloc(rows * 16, 4);
        w = alloc(n, 2);
        u = alloc(n, 2);
        qg = alloc(n, 2);
        kg = alloc(n, 2);
        h = alloc(static_cast<std::size_t>(NT) * H * D * D, 2);
        vn = alloc(n, 2);
        if (backward) {
            dout = alloc(n, 2);
            dv = alloc(n, 2);
            dvs = alloc(n, 2);
            dh = alloc(static_cast<std::size_t>(NT) * H * D * D, 2);
            dq = alloc(n, 4);
            dk = alloc(n, 4);
            dg = alloc(n, 4);
            db = alloc(rows, 4);
            dAqk = alloc(rows * 64, 4);
            dAkk = alloc(rows * 64, 4);
            dq2 = alloc(n, 4);
            dk2 = alloc(n, 4);
            dg2 = alloc(n, 4);
            dbp = alloc(rows * ceildiv(D, 32), 4);
        }
    }
};
}  // namespace

void KimiDeltaRuleKernels::load(const std::unordered_map<std::string, std::string>& manifests) {
    for (const auto* name : Names)
        if (auto it = manifests.find(name); it != manifests.end())
            mKernels.insert_or_assign(name, JitKernel::load_manifest(it->second));
    if (!mKernels.empty() && !is_ready()) throw std::runtime_error("Incomplete vendored FLA KDA kernel manifests");
}

bool KimiDeltaRuleKernels::is_ready() const {
    return mKernels.size() == Names.size();
}

std::size_t KimiDeltaRuleKernels::workspace_bytes(int B, int T, int H, int D, int num_docs, bool backward) {
    return Workspace(nullptr, B, T, H, D, num_docs, backward).bytes;
}

void KimiDeltaRuleKernels::run(bool backward,
                               const std::vector<Tensor>& inputs,
                               const std::vector<Tensor>& outputs,
                               const std::int32_t* cu_seqlens,
                               int num_docs,
                               const Tensor& workspace,
                               cudaStream_t stream) const {
    if (!is_ready()) throw std::runtime_error("GLM KDA requires compile_jit_kernels() and JitKernelManifests");
    const int off = backward ? 1 : 0;
    if (inputs.size() < 5 + off || outputs.size() != (backward ? 5 : 1))
        throw std::runtime_error("Invalid KDA input/output count");
    const auto& q = inputs[off];
    if (q.Rank != 4) throw std::runtime_error("KDA requires [B,T,H,D] inputs");
    const int B = q.Sizes[0], Trow = q.Sizes[1], H = q.Sizes[2], D = q.Sizes[3];
    if (H != mKernels.at("kda_state_fwd").meta().const_int("H", -1) ||
        D != mKernels.at("kda_state_fwd").meta().const_int("K", -1))
        throw std::runtime_error("KDA manifest geometry does not match the model");
    for (int i = 0; i < 5; ++i) {
        const auto& x = inputs[off + i];
        if (!x.Data || x.DType != (i == 3 ? ETensorDType::FP32 : ETensorDType::BF16) ||
            x.nelem() != (i == 4 ? q.nelem() / D : q.nelem()))
            throw std::runtime_error("FLA KDA requires BF16 q/k/v/beta and FP32 decay with matching shapes");
    }
    for (int i = 0; i < outputs.size(); ++i)
        if (!outputs[i].Data || outputs[i].DType != (backward ? ETensorDType::FP32 : ETensorDType::BF16) ||
            outputs[i].nelem() != (backward && i == 4 ? q.nelem() / D : q.nelem()))
            throw std::runtime_error("Invalid FLA KDA output shape/dtype");
    if ((cu_seqlens != nullptr) != (num_docs > 0)) throw std::runtime_error("Invalid KDA document metadata");
    Workspace s(workspace.Data, B, Trow, H, D, num_docs, backward);
    if (!workspace.Data || workspace.bytes() < s.bytes) throw std::runtime_error("KDA workspace is too small");
    int T = s.tokens, rows = T * H, one = 1, packed = num_docs > 0;
    float scale = 1.f / std::sqrt(static_cast<float>(D)), eps = 1.e-6f, rcp_ln2 = 1.4426950408889634f;
    void* nil = nullptr;
    auto launch = [&]<typename... Args>(const char* name, dim3 grid, Args... args) {
        std::array<void*, sizeof...(Args)> params{static_cast<void*>(&args)...};
        mKernels.at(name).launch_triton(grid, params.data(), params.size(), stream);
    };
    auto tile = [&](const char* name, const char* key) {
        return mKernels.at(name).meta().const_int(key, -1);
    };
    launch("kda_metadata", dim3(1), cu_seqlens, s.cu, s.offsets, s.N, Trow, packed);
    launch("kda_indices", dim3(s.N + 1), s.offsets, s.indices, s.N, s.NT);
    const dim3 norm_grid(ceildiv(rows, tile("kda_norm_fwd", "BT")));
    launch("kda_norm_fwd", norm_grid, q.Data, s.qn, s.qr, eps, rows);
    launch("kda_norm_fwd", norm_grid, inputs[off + 1].Data, s.kn, s.kr, eps, rows);
    launch("kda_cumsum_fwd",
           dim3(ceildiv(D, tile("kda_cumsum_fwd", "BS")), s.NT, H),
           inputs[off + 3].Data,
           s.gc,
           rcp_ln2,
           s.cu,
           s.indices,
           T);
    auto v = inputs[off + 2].Data, beta = inputs[off + 4].Data;
    // The triangular solver writes only the lower triangle.
    CUDA_CHECK(cudaMemsetAsync(s.Akk, 0, static_cast<std::size_t>(rows) * 64 * 2, stream));
    launch("kda_intra_fwd", dim3(s.NT, 4, H), s.qn, s.kn, s.gc, beta, s.Aqk, s.Akkd, scale, s.cu, s.indices, T);
    launch("kda_solve", dim3(s.NT, H), s.qn, s.kn, s.gc, beta, s.Aqk, s.Akkd, s.Akk, scale, s.cu, s.indices, T);
    launch("kda_wy_fwd", dim3(s.NT, H), s.qn, s.kn, s.qg, s.kg, v, beta, s.w, s.u, s.Akk, s.gc, s.cu, s.indices, T);
    launch("kda_state_fwd",
           dim3(ceildiv(D, tile("kda_state_fwd", "BV")), s.N * H),
           s.kg,
           s.u,
           s.w,
           s.vn,
           nil,
           s.gc,
           s.h,
           nil,
           nil,
           s.cu,
           s.offsets,
           T);
    if (!backward) {
        launch("kda_output",
               dim3(ceildiv(D, tile("kda_output", "BV")), s.NT, H),
               s.qn,
               s.vn,
               s.gc,
               s.h,
               outputs[0].Data,
               s.Aqk,
               s.cu,
               s.indices,
               scale,
               T);
        return;
    }
    if (!inputs[0].Data || inputs[0].nelem() != q.nelem() ||
        (inputs[0].DType != ETensorDType::BF16 && inputs[0].DType != ETensorDType::FP32))
        throw std::runtime_error("Invalid KDA upstream gradient");
    void* dout = inputs[0].Data;
    if (inputs[0].DType != ETensorDType::BF16) {
        Tensor dst = q;
        dst.Data = static_cast<std::byte*>(s.dout);
        glm5_copy_gradient(inputs[0], dst, false, stream);
        dout = s.dout;
    }
    launch("kda_dav", dim3(s.NT, H), s.qn, s.kn, s.vn, s.Aqk, dout, s.dv, s.dAqk, s.cu, s.indices, scale, T);
    launch("kda_state_bwd",
           dim3(ceildiv(D, tile("kda_state_bwd", "BV")), s.N * H),
           s.qg,
           s.kg,
           s.w,
           nil,
           s.gc,
           nil,
           nil,
           dout,
           s.dh,
           s.dv,
           s.dvs,
           s.cu,
           s.offsets,
           scale,
           T);
    launch("kda_wy_bwd",
           dim3(s.NT, H),
           s.qn,
           s.kn,
           v,
           s.vn,
           s.gc,
           beta,
           s.Akk,
           s.h,
           dout,
           s.dh,
           s.dq,
           s.dk,
           s.dvs,
           outputs[2].Data,
           s.dg,
           s.db,
           s.dAkk,
           s.cu,
           s.indices,
           scale,
           T);
    const int NK = ceildiv(D, tile("kda_intra_bwd", "BK"));
    launch("kda_intra_bwd",
           dim3(NK * 4, s.NT, H),
           s.qn,
           s.kn,
           s.gc,
           beta,
           s.dAqk,
           s.dAkk,
           s.dq,
           s.dq2,
           s.dk,
           s.dk2,
           s.dg,
           s.dg2,
           s.dbp,
           s.cu,
           s.indices,
           one,
           T);
    launch("kda_beta_bwd", dim3(ceildiv(rows, tile("kda_beta_bwd", "BLOCK"))), s.dbp, s.db, outputs[4].Data, T);
    launch("kda_cumsum_rev",
           dim3(ceildiv(D, tile("kda_cumsum_rev", "BS")), s.NT, H),
           s.dg2,
           outputs[3].Data,
           rcp_ln2,
           s.cu,
           s.indices,
           T);
    const dim3 bwd_norm_grid(ceildiv(rows, tile("kda_norm_bwd", "BT")));
    launch("kda_norm_bwd", bwd_norm_grid, s.qn, s.qr, s.dq2, outputs[0].Data, eps, rows);
    launch("kda_norm_bwd", bwd_norm_grid, s.kn, s.kr, s.dk2, outputs[1].Data, eps, rows);
}
