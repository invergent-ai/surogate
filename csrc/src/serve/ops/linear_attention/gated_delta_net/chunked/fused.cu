#include "ops/linear_attention/gated_delta_net/chunked/fused.cuh"
#include "ops/linear_attention/gated_delta_net/chunked/launch.h"

namespace sinfer::ops::detail::gated_delta_net::chunked {

cudaError_t launch_fused(const fused_config& cfg) {
    stage_validator v{"launch_fused", cfg.H_qk, cfg.H_v, cfg.L};
    SINFER_GATED_DELTA_NET_PROPAGATE(v.check_shape());
    SINFER_GATED_DELTA_NET_PROPAGATE(v.check_full_chunks());
    if (cfg.valid_tokens < 0 || cfg.valid_tokens > cfg.L) { return cudaErrorInvalidValue; }
    if (cfg.q == nullptr || cfg.k == nullptr || cfg.v == nullptr || cfg.g == nullptr ||
        cfg.beta == nullptr || cfg.state_in == nullptr || cfg.state_out == nullptr ||
        cfg.out == nullptr) {
        return cudaErrorInvalidValue;
    }
    SINFER_GATED_DELTA_NET_PROPAGATE(v.check_grid(cfg.H_v, 1));

    constexpr int smem_bytes = fused::smem_layout::bytes;
    SINFER_GATED_DELTA_NET_PROPAGATE(cudaFuncSetAttribute(
        fused::fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    fused::fused_kernel<<<static_cast<unsigned>(cfg.H_v), fused::kThreads, smem_bytes,
                          cfg.stream>>>(
        cfg.q, cfg.k, cfg.v, cfg.g, cfg.beta, cfg.state_in, cfg.state_out, cfg.out,
        head_map::of(cfg.H_qk, cfg.H_v), cfg.scale, cfg.L / BT,
        cfg.valid_tokens ? cfg.valid_tokens : cfg.L);
    return cudaGetLastError();
}

} // namespace sinfer::ops::detail::gated_delta_net::chunked
