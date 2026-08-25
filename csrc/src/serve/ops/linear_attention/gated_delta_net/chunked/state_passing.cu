#include "ops/linear_attention/gated_delta_net/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/chunked/state_passing.cuh"

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

namespace kernel = state_passing;

template <int NStrip>
cudaError_t launch_fixed(const state_passing_config& cfg, head_map qk_map, int NT) {
    using D = kernel::kernel_dims<NStrip>;
    constexpr int smem_bytes =
        kernel::smem_layout<NStrip>::SMEM_FLOATS * static_cast<int>(sizeof(float));

    cudaError_t err = cudaFuncSetAttribute(kernel::state_passing_kernel<NStrip>,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (err != cudaSuccess) { return err; }

    const dim3 grid(static_cast<unsigned>(static_cast<std::int64_t>(cfg.H_v) * D::D_STRIPS), 1, 1);
    const dim3 block(D::THREADS, 1, 1);

    kernel::state_passing_kernel<NStrip><<<grid, block, smem_bytes, cfg.stream>>>(
        cfg.W, cfg.U, cfg.k, cfg.g_cumsum, cfg.state_in, cfg.v_new, cfg.h_chunk, cfg.state_out,
        qk_map, NT);
    return cudaGetLastError();
}

} // namespace

cudaError_t launch_state_passing(const state_passing_config& cfg) {
    stage_validator v{"launch_state_passing", cfg.H_qk, cfg.H_v, cfg.L};
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_shape());
    NINFER_GATED_DELTA_NET_PROPAGATE(v.check_full_chunks());
    if (cfg.W == nullptr || cfg.U == nullptr || cfg.k == nullptr || cfg.g_cumsum == nullptr ||
        cfg.state_in == nullptr || cfg.v_new == nullptr || cfg.h_chunk == nullptr ||
        cfg.state_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    const auto qk_map     = head_map::of((int)cfg.H_qk, (int)cfg.H_v);
    const std::int64_t NT = cfg.L / BT;
    if (cfg.H_v >= 48) {
        NINFER_GATED_DELTA_NET_PROPAGATE(v.check_grid(
            static_cast<std::int64_t>(cfg.H_v) * kernel::kernel_dims<16>::D_STRIPS, 1));
        return launch_fixed<16>(cfg, qk_map, static_cast<int>(NT));
    }
    NINFER_GATED_DELTA_NET_PROPAGATE(
        v.check_grid(static_cast<std::int64_t>(cfg.H_v) * kernel::kernel_dims<32>::D_STRIPS, 1));
    return launch_fixed<32>(cfg, qk_map, static_cast<int>(NT));
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
