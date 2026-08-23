#include "ops/linear/q4/q4_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("q4 linear: unsupported shape or T"); }

    switch (k) {
    case 5120:
        switch (n) {
        case 1024:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 15) { return launch_q4_simt_r8_c4; }
            if (t == 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 4096:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 4) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 6144:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 7) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 7168:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 7) { return launch_q4_simt_r8_c4; }
            if (t == 8) { return launch_q4_simt_r8_c8; }
            if (t <= 15) { return launch_q4_simt_r8_c4; }
            if (t == 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 34816:
            if (t == 1) { return launch_q4_gemv_r1_w8_direct; }
            if (t <= 4) { return launch_q4_simt_r8_c4; }
            if (t <= 16) { return launch_q4_simt_r8_c8; }
            return launch_q4_mma_r64_c128;
        case 131072:
            if (t == 1) { return launch_q4_gemv_r4_w1_direct; }
            if (t <= 8) { return launch_q4_draft_head_small_t; }
            return launch_q4_mma_r64_c128;
        default:
            break;
        }
        break;
    case 2048:
        if (n == 131072) {
            if (t == 1) { return launch_q4_gemv_r4_w1_direct; }
            if (t <= 20) { return launch_q4_draft_head_small_t; }
            if (t <= 32) { return launch_q4_mma_r64_c32; }
            if (t <= 48) { return launch_q4_mma_r64_c48; }
            if (t <= 56) { return launch_q4_mma_r64_c56; }
            if (t <= 63) { return launch_q4_mma_r64_c72; }
            if (t == 64) { return launch_q4_mma_r64_c64_endpoint; }
            if (t <= 72) { return launch_q4_mma_r64_c72; }
            if (t <= 80) { return launch_q4_mma_r64_c80; }
            if (t <= 96) { return launch_q4_mma_r64_c96; }
            if (t <= 104) { return launch_q4_mma_r64_c104_bounded; }
            if (t <= 111) { return launch_q4_mma_r64_c112_partial; }
            if (t == 112) { return launch_q4_mma_r64_c112; }
            if (t <= 119) { return launch_q4_mma_r64_c120_partial; }
            if (t == 120) { return launch_q4_mma_r64_c120; }
            return launch_q4_mma_r64_c128;
        }
        break;
    case 1152:
        if (t < 4 || t > 131072 || (t % 4) != 0) { break; }
        switch (n) {
        case 3456:
            if (t <= 36) { return launch_q4_simt_r8_c4; }
            if (t <= 320) { return launch_q4_mma_r64_c64; }
            return launch_q4_mma_r64_c128;
        case 4304:
            if (t == 4) { return launch_q4_simt_r8_c4; }
            if (t == 8) { return launch_q4_simt_r8_c8; }
            if (t == 12) { return launch_q4_simt_r8_c4; }
            if (t <= 24) { return launch_q4_simt_r8_c8; }
            if (t <= 320) { return launch_q4_mma_r64_c64; }
            return launch_q4_mma_r64_c128;
        default:
            break;
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("q4 linear: unsupported shape or T");
}

Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
        return select_q4_a16_launch(n, k, t);
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("q4 linear: unsupported policy");
}

void q4_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const Q4Launch launch = select_q4_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
