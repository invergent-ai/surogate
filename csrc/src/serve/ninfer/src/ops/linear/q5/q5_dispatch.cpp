#include "ops/linear/q5/q5_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Q5Launch select_q5_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("q5 linear: unsupported shape or T"); }

    switch (k) {
    case 5120:
        switch (n) {
        case 1024:
            if (t <= 4) { return launch_q5_simt_r8_c4; }
            if (t <= 16) { return launch_q5_simt_r8_c8; }
            return launch_q5_mma_r64_c128;
        case 6144:
            if (t == 1) { return launch_q5_gemv_r16_s2_x; }
            if (t <= 6) { return launch_q5_simt_split4_exact; }
            if (t <= 24) { return launch_q5_simt_r8_c8; }
            if (t <= 64) { return launch_q5_mma_r64_c64; }
            return launch_q5_mma_r64_c128;
        case 7168:
            if (t == 1) { return launch_q5_gemv_r16_s2_x; }
            if (t <= 6) { return launch_q5_simt_split4_exact; }
            if (t <= 16) { return launch_q5_simt_r8_c4; }
            return launch_q5_mma_r64_c128;
        default:
            break;
        }
        break;
    case 6144:
        if (n == 5120) {
            if (t == 1) { return launch_q5_simt_r8_c4; }
            if (t <= 6) { return launch_q5_simt_split2_exact; }
            if (t <= 24) { return launch_q5_simt_r8_c8; }
            return launch_q5_mma_r64_c128;
        }
        break;
    case 17408:
        if (n == 5120) {
            if (t == 1) { return launch_q5_simt_r8_c4; }
            if (t <= 6) { return launch_q5_simt_split2_exact; }
            if (t <= 24) { return launch_q5_simt_r8_c8; }
            return launch_q5_mma_r64_c128;
        }
        break;
    case 1152:
        if (n == 1152 && t >= 4 && t <= 131072 && (t % 4) == 0) {
            if (t <= 76) { return launch_q5_simt_r8_c4; }
            if (t <= 636) { return launch_q5_mma_r64_c64; }
            if (t <= 700) { return launch_q5_mma_r64_c128; }
            if (t == 704) { return launch_q5_mma_r64_c64; }
            if (t <= 828) { return launch_q5_mma_r64_c128; }
            if (t == 832) { return launch_q5_mma_r64_c64; }
            if (t <= 896) { return launch_q5_mma_r64_c128; }
            if (t <= 960) { return launch_q5_mma_r64_c64; }
            if (t <= 1024) { return launch_q5_mma_r64_c128; }
            if (t <= 1088) { return launch_q5_mma_r64_c64; }
            return launch_q5_mma_r64_c128;
        }
        break;
    case 4304:
        if (n == 1152 && t >= 4 && t <= 131072 && (t % 4) == 0) {
            if (t <= 120) { return launch_q5_simt_r8_c4; }
            if (t <= 1148) { return launch_q5_mma_r64_c64; }
            return launch_q5_mma_r64_c128;
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("q5 linear: unsupported shape or T");
}

Q5Launch select_q5_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
        return select_q5_a16_launch(n, k, t);
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("q5 linear: unsupported policy");
}

void q5_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const Q5Launch launch = select_q5_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
