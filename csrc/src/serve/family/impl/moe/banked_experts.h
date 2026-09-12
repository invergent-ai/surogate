#pragma once

#include "runtime/engine/kv_capacity.h"

#include "family/impl/moe/expert_cache.h"
#include "api/ops/linear.h"

namespace sinfer::family {

/// Residency metadata kept beside a target's routed weights.
struct BankedExperts {
    std::int32_t layer = -1;
    std::int32_t layers = 0;
    BankPlanes gate_up_planes = BankPlanes::Native;
    BankPlanes down_planes = BankPlanes::Native;
    const std::byte* host_gate_up = nullptr;
    const std::byte* host_down = nullptr;

    [[nodiscard]] BankedMixture mixture(const ops::SparseMoeWeights& op) const {
        return {layer, layers, &op, gate_up_planes, down_planes, host_gate_up, host_down};
    }
};

BankedExperts bind_banked_experts(ops::SparseMoeWeights& weights, const HostBank* bank,
                                  artifact::ObjectHandle gate_up, artifact::ObjectHandle down,
                                  std::int32_t layer, std::int32_t layers);

/// Configure host precision and account for the device cache before KV sizing.
void plan_banked_experts(artifact::Binder& binder, HostBankPlan& bank,
                         const artifact::MaterializationPlan& materialization,
                         const EngineOptions& options, const ops::SparseMoeGeometry& geometry,
                         std::int32_t layers, ops::LinearPolicy policy);

void configure_banked_experts(DeviceContext& device, const EngineOptions& options,
                              const ops::SparseMoeGeometry& geometry, std::int32_t layers,
                              const runtime::SequenceCapacityCurve& curve);

/// Returns false for resident experts; otherwise runs and joins the cached/CPU result.
bool run_banked_experts(const BankedExperts& banked, const ops::SparseMoeWeights& weights,
                        const Tensor& hidden, Tensor& destination, WorkspaceArena& workspace,
                        cudaStream_t stream, const Tensor* router_input = nullptr);

template <class Runtime>
void prepare_banked_experts(const Runtime& runtime) {
    const auto prepare = [](const auto& layer) {
        if constexpr (requires { layer.post_mixer.banked; }) {
            const auto& payload = layer.post_mixer;
            const auto& op = [&]() -> const ops::SparseMoeWeights& {
                if constexpr (requires { payload.op; }) { return payload.op; }
                else { return payload.moe; }
            }();
            if (payload.banked.host_gate_up != nullptr) {
                auto& cache = ExpertCache::for_current_device(ops::sparse_moe_geometry(op),
                                                               payload.banked.layers);
                cache.prepare_split(payload.banked.mixture(op));
            }
        }
    };
    for (const auto& layer : runtime.full_layers) { prepare(layer); }
    for (const auto& layer : runtime.gdn_layers) { prepare(layer); }
}

} // namespace sinfer::family
