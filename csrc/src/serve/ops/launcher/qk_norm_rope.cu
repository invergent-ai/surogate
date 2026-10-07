// sinfer::ops - qk_norm_rope launcher: picks the norm form and the rotation's coefficients the
// separate ops would dispatch for the same arguments (rmsnorm.cu, rope.cu), so the fused kernel
// writes their bits.
#include "ops/launcher/qk_norm_rope.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/qk_norm_rope.cuh"

#include <cstdint>

namespace sinfer::ops::detail {
namespace {

bool aligned4(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 3) == 0; }

template <RmsEpilogue Epilogue, QkNormForm Form, QkRopeAngles Angles>
void launch(const QkNormRope& a, int k_heads, cudaStream_t stream) {
    const Tensor& q   = *a.q;
    const int tokens  = q.ne[2];
    const int heads   = q.ne[1] + k_heads;
    const dim3 grid(static_cast<unsigned>(tokens),
                    static_cast<unsigned>((heads + kQkNormRopeWarps - 1) / kQkNormRopeWarps));
    const auto bf2  = [](const Tensor* t) { return t == nullptr ? nullptr : static_cast<const __nv_bfloat162*>(t->data); };
    const auto out2 = [](Tensor* t) { return t == nullptr ? nullptr : static_cast<__nv_bfloat162*>(t->data); };
    const bool interleaved = a.sections[0] != 0;
    qk_norm_rope_kernel<Epilogue, Form, Angles><<<grid, kQkNormRopeWarps * 32, 0, stream>>>(
        bf2(a.q), bf2(a.k), bf2(a.q_norm), bf2(a.k_norm), out2(a.q_out), out2(a.k_out),
        static_cast<const std::int32_t*>(a.positions->data), a.positions->ne[1], q.ne[0],
        a.rotary_dim, a.active_pairs, a.theta, interleaved ? a.sections[1] : -1,
        interleaved ? a.sections[2] : -1, q.ne[1], k_heads, tokens, a.eps);
}

template <RmsEpilogue Epilogue, QkNormForm Form>
void launch_angles(const QkNormRope& a, int k_heads, QkRopeAngles angles, cudaStream_t stream) {
    switch (angles) {
    case QkRopeAngles::Text1D: launch<Epilogue, Form, QkRopeAngles::Text1D>(a, k_heads, stream); break;
    case QkRopeAngles::TextMrope: launch<Epilogue, Form, QkRopeAngles::TextMrope>(a, k_heads, stream); break;
    default: launch<Epilogue, Form, QkRopeAngles::Generic>(a, k_heads, stream); break;
    }
}

} // namespace

bool qk_norm_rope_launch(const QkNormRope& a, cudaStream_t stream) {
    const Tensor& q    = *a.q;
    const int head_dim = q.ne[0];
    const int axes     = a.positions->ne[1];
    const bool keys    = a.k != nullptr;

    // The norm: rmsnorm.cu's aligned warp-per-row kernels (D128 for a plain gain at 128, else the
    // warp kernel for 64..256 in steps of 64). Anything else runs its CTA or generic kernel.
    if (head_dim < 64 || head_dim > 256 || head_dim % 64 != 0) { return false; }
    if (!aligned4(q.data) || !aligned4(a.q_norm->data) || !aligned4(a.q_out->data)) { return false; }
    if (keys && (!aligned4(a.k->data) || !aligned4(a.k_norm->data) || !aligned4(a.k_out->data))) {
        return false;
    }

    // The rotation: a partner in the same lane or in the first register's warp.
    const int quarter = a.rotary_dim / 4;
    if (a.rotary_dim % 4 != 0 || a.rotary_dim > head_dim ||
        !(quarter % kWarpSize == 0 || (quarter < kWarpSize && (quarter & (quarter - 1)) == 0))) {
        return false;
    }
    if (axes != 1 && axes != 3) { return false; }

    // The coefficients rope.cu's dispatch would compute: its fixed Text1D / TextMrope kernels for
    // a whole rotation of the shapes they are compiled for, the generic kernel for everything else
    // (its DFlash shape included, which the generic kernel computes the fixed kernel's way).
    // rope_interleaved always runs the generic kernel.
    QkRopeAngles angles = QkRopeAngles::Generic;
    const bool whole    = a.active_pairs == a.rotary_dim / 2;
    if (a.sections[0] == 0 && whole && a.rotary_dim == 64 && a.theta == 1.0e7F && head_dim == 256 &&
        ((q.ne[1] == 24 && a.k_heads == 4) || (q.ne[1] == 16 && a.k_heads == 2))) {
        angles = axes == 1 ? QkRopeAngles::Text1D : QkRopeAngles::TextMrope;
    }

    const int k_heads = keys ? a.k_heads : 0;
    if (q.ne[2] > 0) {
        if (a.unit_offset) {
            launch_angles<RmsEpilogue::Offset, QkNormForm::Warp>(a, k_heads, angles, stream);
        } else if (head_dim == 128) {
            launch_angles<RmsEpilogue::Plain, QkNormForm::D128>(a, k_heads, angles, stream);
        } else {
            launch_angles<RmsEpilogue::Plain, QkNormForm::Warp>(a, k_heads, angles, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

} // namespace sinfer::ops::detail
