#include "family/impl/load/routed_nvfp4.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "ops/linear/marlin/marlin_nvfp4.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sinfer::family {

using artifact::NumericFormat;

Weight routed_nvfp4_weight(const artifact::MaterializedArtifact& materialized,
                           artifact::ObjectHandle handle, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle));

    Weight out{};
    out.payload         = bytes;
    out.payload_bytes   = geometry.encoded_bytes;
    out.qtype           = QType::NVFP4;
    out.group_size      = 16;
    out.ndim            = 2;
    out.qdata           = bytes;
    out.scales          = bytes + geometry.scale_plane_offset;
    out.n               = rows;
    out.k               = columns;
    out.group           = 16;
    out.layout          = QuantLayout::BlockScaleK16M128x4;
    out.scale_dtype     = DType::FP8_E4M3FN;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    // Neither divisor applies here, and the kernels do not read them; the converter writes 1.0
    // into the payload's word, which `require_identity_divisor` checks so a checkpoint that
    // means something by it fails loudly instead of being ignored.
    out.weight_scale_divisor = 1.0F;
    out.input_scale_divisor  = 1.0F;
    return out;
}

void require_identity_divisor(const artifact::Binder& binder, artifact::ObjectHandle handle,
                              std::string_view name, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const artifact::PayloadSpan payload = binder.payload(handle);
    if (payload.data.size() < geometry.weight_divisor_offset + sizeof(std::uint32_t)) {
        throw artifact::ArtifactError(std::string(name) + ": NVFP4 payload is short of its divisor");
    }
    std::uint32_t bits = 0;
    std::memcpy(&bits, payload.data.data() + geometry.weight_divisor_offset, sizeof(bits));
    if (std::bit_cast<float>(bits) != 1.0F) {
        throw artifact::ArtifactError(
            std::string(name) +
            ": routed NVFP4 must carry a per-tensor divisor of 1.0; the second level belongs in "
            "the per-expert scale object");
    }
}

void adopt_weight_only_nvfp4(ops::SparseMoeWeights& weights, std::int32_t experts,
                             cudaStream_t stream) {
    if (weights.routed_gate_up.qtype != QType::NVFP4_A16 &&
        weights.shared_gate_up.qtype != QType::NVFP4_A16) {
        return;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (weights.routed_gate_up.qtype == QType::NVFP4_A16) {
        if (weights.routed_down.qtype != QType::NVFP4_A16 || weights.routed_gate_up_scale == nullptr ||
            weights.routed_down_scale == nullptr) {
            throw std::invalid_argument(
                "weight-only NVFP4 experts need both banks and their per-expert scales");
        }
        std::vector<float> halves(static_cast<std::size_t>(2) * experts);
        CUDA_CHECK(cudaMemcpy(halves.data(), weights.routed_gate_up_scale,
                              halves.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (std::int32_t expert = 0; expert < experts; ++expert) {
            if (std::bit_cast<std::uint32_t>(halves[2 * expert]) !=
                std::bit_cast<std::uint32_t>(halves[2 * expert + 1])) {
                throw std::invalid_argument(
                    "weight-only NVFP4 expert " + std::to_string(expert) +
                    ": gate and up second levels differ; one GEMM over its rows carries one");
            }
        }
        ops::detail::nvfp4_marlin_adopt(weights.routed_gate_up, experts,
                                        weights.routed_gate_up_scale, 2, stream);
        ops::detail::nvfp4_marlin_adopt(weights.routed_down, experts, weights.routed_down_scale, 1,
                                        stream);
    }
    if (weights.shared_gate_up.qtype == QType::NVFP4_A16) {
        ops::detail::nvfp4_marlin_adopt(weights.shared_gate_up, 1, nullptr, 1, stream);
        ops::detail::nvfp4_marlin_adopt(weights.shared_down, 1, nullptr, 1, stream);
    }
}

void adopt_weight_only_nvfp4(Weight& weight, cudaStream_t stream) {
    if (weight.qtype != QType::NVFP4_A16) { return; }
    CUDA_CHECK(cudaDeviceSynchronize());
    ops::detail::nvfp4_marlin_adopt(weight, 1, nullptr, 1, stream);
}

} // namespace sinfer::family
