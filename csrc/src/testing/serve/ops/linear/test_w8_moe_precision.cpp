#include "api/ops/sparse_moe.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <iostream>
#include <memory>

using namespace sinfer;
using namespace sinfer::test;

namespace {

struct DeviceWeight {
    DeviceBuffer storage;
    Weight weight;
    explicit DeviceWeight(quantized_weight::PackedWeight packed)
        : storage(to_device(packed.payload)), weight(packed.device_weight(storage.p)) {}
};

quantized_weight::PackedWeight gate_up(int rows, int hidden, int intermediate, bool cancel_down) {
    auto packed = quantized_weight::make_patterned_weight(QType::W8G32_F16S, rows, hidden, 211);
    std::fill_n(packed.payload.begin(), packed.code_plane_bytes, 0);
    for (int row = 0; row < rows; ++row) {
        const bool gate = row % (2 * intermediate) < intermediate;
        const auto base = static_cast<std::size_t>(row) * hidden;
        packed.payload[base] = gate ? 1 : 100;
        quantized_weight::detail::store_u16_le(packed.payload,
            packed.scale_plane_offset + base / 32 * 2, gate ? 0x3c00 : 0x2c01);
        if (cancel_down && !gate) {
            const int neuron = row % intermediate;
            packed.payload[base] = neuron == 0 ? 1 : neuron == intermediate - 32 ? 255 : 0;
            quantized_weight::detail::store_u16_le(packed.payload,
                packed.scale_plane_offset + base / 32 * 2, 0x3c00);
        } else if (!gate) {
            packed.payload[base + hidden - 32] = 100;
            quantized_weight::detail::store_u16_le(packed.payload,
                packed.scale_plane_offset + (base + hidden - 32) / 32 * 2, 0x2c02);
        }
    }
    return packed;
}

quantized_weight::PackedWeight down(int rows, int intermediate, bool cancel_down) {
    auto packed = quantized_weight::make_patterned_weight(QType::W8G32_F16S, rows, intermediate, 223);
    std::fill_n(packed.payload.begin(), packed.code_plane_bytes, cancel_down ? 0 : 1);
    for (std::size_t i = 0; i < packed.scale_plane_bytes; i += 2) {
        quantized_weight::detail::store_u16_le(packed.payload, packed.scale_plane_offset + i, 0x3c00);
    }
    if (cancel_down) {
        for (int row = 0; row < rows; ++row) {
            const auto base = static_cast<std::size_t>(row) * intermediate;
            packed.payload[base] = packed.payload[base + intermediate - 32] = 100;
            quantized_weight::detail::store_u16_le(packed.payload,
                packed.scale_plane_offset + base / 32 * 2, 0x2c01);
            quantized_weight::detail::store_u16_le(packed.payload,
                packed.scale_plane_offset + (base + intermediate - 32) / 32 * 2, 0x2c02);
        }
    }
    return packed;
}

int check(const ops::SparseMoeGeometry& geometry, bool cancel_down) {
    DeviceWeight gate(gate_up(geometry.routed_gate_rows(), geometry.hidden, geometry.intermediate, cancel_down));
    DeviceWeight projection(down(geometry.routed_down_rows(), geometry.intermediate, cancel_down));
    std::unique_ptr<DeviceWeight> shared_gate, shared_down;
    if (geometry.has_shared()) {
        shared_gate = std::make_unique<DeviceWeight>(gate_up(geometry.shared_rows(), geometry.hidden,
                                                            geometry.shared_intermediate, cancel_down));
        shared_down = std::make_unique<DeviceWeight>(down(geometry.hidden, geometry.shared_intermediate, cancel_down));
    }
    DeviceBuffer router(static_cast<std::size_t>(geometry.router_rows()) * geometry.hidden * 2);
    router.fill();
    Weight routing{};
    routing.qtype = QType::BF16_CTRL;
    routing.layout = QuantLayout::Contiguous;
    routing.payload = routing.qdata = router.p;
    routing.payload_bytes = router.bytes;
    routing.n = routing.shape[0] = routing.padded_shape[0] = geometry.router_rows();
    routing.k = routing.shape[1] = routing.padded_shape[1] = geometry.hidden;
    routing.ndim = 2;
    ops::SparseMoeWeights weights{
        .router_shared_gate = routing,
        .routed_gate_up = gate.weight,
        .routed_down = projection.weight,
        .shared_gate_up = shared_gate ? shared_gate->weight : Weight{},
        .shared_down = shared_down ? shared_down->weight : Weight{},
        .experts_per_token = geometry.experts_per_token,
    };
    int failures = 0;
    // The two scaled codes round to the same BF16 value, 6.25. They cancel
    // either in the up projection, or in the down projection against opposite
    // expert activations. Unrounded products leave a nonzero contribution.
    // Every route must preserve the residual through decode and both prefill paths.
    for (int tokens : {1, 2, 19, 20, 33, 129, 768}) {
        std::vector<std::uint16_t> input(static_cast<std::size_t>(geometry.hidden) * tokens, 0);
        for (int t = 0; t < tokens; ++t) {
            input[t * geometry.hidden] = f32_to_bf16(1.0F);
            input[t * geometry.hidden + geometry.hidden - 32] = f32_to_bf16(-1.0F);
        }
        DeviceBuffer x = to_device(input);
        std::vector<std::uint16_t> initial(input.size(), f32_to_bf16(0.25F));
        GuardedDeviceBuffer output(initial.size() * 2);
        output.copy_from_host(initial.data(), initial.size() * 2);
        Tensor tx(x.p, DType::BF16, {geometry.hidden, tokens});
        Tensor ty(output.data(), DType::BF16, {geometry.hidden, tokens});
        WorkspaceArena workspace(ops::sparse_moe_workspace_capacity_bytes(
            geometry, QType::W8G32_F16S, QType::W8G32_F16S, tokens, tokens));
        ops::sparse_moe(tx, weights, ops::SparseMoeEpilogue::AddResidual, ty, workspace, nullptr);
        cuda_synchronize();
        failures += output.verify_guards("W8 MoE precision");
        const auto actual = from_device<std::uint16_t>(output.data(), initial.size());
        if (actual != initial) {
            std::cerr << "W8 MoE materialization differs: shared=" << geometry.has_shared()
                      << " down=" << cancel_down << " T=" << tokens << " first=" << bf16_to_f32(actual.front()) << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) { return 77; }
    int failures = 0;
    for (bool cancel_down : {false, true}) {
        failures += check(ops::kSparseMoeQwen3MoeGeometry, cancel_down);
        failures += check(ops::kSparseMoeQwen36Geometry, cancel_down);
    }
    if (!failures) { std::cout << "W8 MoE materialization agrees across decode and prefill\n"; }
    return failures ? 1 : 0;
}
