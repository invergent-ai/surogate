#include "family/impl/moe/banked_experts.h"
#include "targets/registry.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace sinfer::family {

BankedExperts bind_banked_experts(ops::SparseMoeWeights& weights, const HostBank* bank,
                                  artifact::ObjectHandle gate_up, artifact::ObjectHandle down,
                                  std::int32_t layer, std::int32_t layers) {
    if (bank == nullptr || weights.routed_gate_up.n == 0) { return {}; }
    const auto* up = bank->find(gate_up);
    const auto* dn = bank->find(down);
    if (up == nullptr && dn == nullptr) { return {}; }
    if (up == nullptr || dn == nullptr) {
        throw std::logic_error("both routed expert projections must have the same residency");
    }
    const auto apply = [](Weight& weight, const HostObject& object) {
        if (bank_planes_are_affine(object.planes)) {
            weight = host_affine_weight(object.planes, object, weight.n, weight.k);
        } else if (object.planes == BankPlanes::W8) {
            weight = host_w8_weight(object, weight.n, weight.k);
        }
    };
    const bool converted_nvfp4 = weights.routed_gate_up.qtype == QType::NVFP4;
    apply(weights.routed_gate_up, *up);
    apply(weights.routed_down, *dn);
    if (converted_nvfp4) {
        weights.routed_gate_up_scale = nullptr;
        weights.routed_down_scale = nullptr;
        weights.routed_gate_up_act_scale = nullptr;
        weights.routed_gate_up_alpha = nullptr;
        weights.routed_down_act_scale = nullptr;
        weights.routed_down_alpha = nullptr;
    }
    return {layer, layers, up->planes, dn->planes,
            static_cast<const std::byte*>(up->host), static_cast<const std::byte*>(dn->host)};
}

void plan_banked_experts(artifact::Binder& binder, HostBankPlan& bank,
                         const artifact::MaterializationPlan& materialization,
                         const EngineOptions& options, const ops::SparseMoeGeometry& geometry,
                         std::int32_t layers, ops::LinearPolicy policy) {
    bool banked = false;
    for (auto& object : bank.objects) {
        if (!object.name.ends_with("/moe/routed_gate_up") &&
            !object.name.ends_with("/moe/routed_down")) { continue; }
        banked = true;
        const auto& tensor = std::get<artifact::TensorDescriptor>(binder.descriptor(object.handle));
        const auto rows = static_cast<std::int64_t>(tensor.shape.at(0));
        const auto columns = static_cast<std::int32_t>(tensor.shape.at(1));
        const BankPlanes planes = options.host_expert_bank == EngineOptions::HostExpertBank::Q4
            ? BankPlanes::Q4 : options.host_expert_bank == EngineOptions::HostExpertBank::W8
                ? BankPlanes::W8 : BankPlanes::Auto;
        if (tensor.format == artifact::NumericFormat::W8G32_F16S && object.q8_rows == 0 && columns % 128 == 0) {
            if (planes == BankPlanes::Q4) {
                object.q4_rows = rows;
                object.q4_k = columns;
                object.q4_w8_scale_offset = artifact::row_split_geometry(tensor.format,
                                                                         tensor.shape).scale_plane_offset;
            }
        } else if (object.q8_rows != 0 && object.q8_group_map.empty()) {
            // Decode source blocks directly into the requested host planes.
            object.q8_rows = 0;
            bank_as_planes(object, rows, columns, QType::Q8_0, planes);
        } else {
            bank_as_planes(object, rows, columns, artifact::qtype_for(tensor.format), planes);
        }
        if (tensor.format == artifact::NumericFormat::NVFP4) {
            const bool up = object.name.ends_with("routed_gate_up");
            object.swap_half_rows = up ? geometry.intermediate : 0;
            object.scale_rows = up ? geometry.intermediate : geometry.hidden;
            const auto* scale = binder.reader().find(object.name + "_scale");
            if (scale == nullptr) { throw std::invalid_argument("NVFP4 expert scale is missing"); }
            const auto bytes = binder.reader().payload(*scale).data;
            object.row_scales.resize(bytes.size() / sizeof(float));
            std::memcpy(object.row_scales.data(), bytes.data(), bytes.size());
        }
    }
    ExpertCache::configure_pool_floor(banked
        ? ExpertCache::pool_floor_bytes(geometry, layers, options.expert_slots) : 0);
    ExpertCache::configure_derived_reserve(
        targets::projected_derived_residency_bytes(binder, materialization, policy) +
        static_cast<std::size_t>(materialization.device_capacity_bytes));
    ExpertCache::configure_load_staging(targets::projected_load_staging_bytes(binder, materialization));
}

void configure_banked_experts(DeviceContext& device, const EngineOptions& options,
                              const ops::SparseMoeGeometry& geometry, std::int32_t layers,
                              const runtime::SequenceCapacityCurve& curve) {
    if (options.offload_planning_only || geometry.experts <= 0 || ExpertCache::pool_floor() == 0) {
        return;
    }
    int previous = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    CUDA_CHECK(cudaSetDevice(device.device));
    const auto floor = ExpertCache::derived_reserve() +
        std::max(runtime::minimum_kv_reservation_bytes(options.kv_capacity, curve),
                 ExpertCache::load_staging());
    ExpertCache::configure(options, floor, geometry.experts);
    (void)ExpertCache::for_current_device(geometry, layers);
    CUDA_CHECK(cudaSetDevice(previous));
}

bool run_banked_experts(const BankedExperts& banked, const ops::SparseMoeWeights& weights,
                        const Tensor& hidden, Tensor& destination, WorkspaceArena& workspace,
                        cudaStream_t stream, const Tensor* router_input) {
    if (banked.host_gate_up == nullptr) { return false; }
    auto& cache = ExpertCache::for_current_device(ops::sparse_moe_geometry(weights), banked.layers);
    if (!cache.enabled()) {
        throw std::runtime_error("offloaded experts need enough GPU memory for the minimum expert-cache batch");
    }
    cache.run(banked.mixture(weights), hidden, destination, workspace, stream, router_input);
    cache.add_pending_partial(destination, stream);
    cache.tick_combine();
    return true;
}

} // namespace sinfer::family
