#include "ops/linear/ggml/ggml_swiglu.h"

#include "core/device.h"
#include "ops/linear/ggml/ggml_q8_1.h"
#include "ops/linear/ggml/ggml_swiglu_decode.cuh"
#include "ops/linear/ggml/ggml_swiglu_k.cuh"

#include <stdexcept>
#include <algorithm>

namespace sinfer::ops::detail::ggml {
namespace {
constexpr bool k_quant(GgmlType type) {
    return type == GgmlType::Q4_K || type == GgmlType::Q5_K || type == GgmlType::Q6_K;
}
} // namespace

bool swiglu_prefill_admits(GgmlType gate, GgmlType up, std::int32_t rows, std::int32_t k,
                           std::int32_t tokens) noexcept {
    if (gate != up || !k_quant(gate) || rows <= 0 || k <= 0 || k % QK_K || tokens <= 8) {
        return false;
    }
    // This narrow Q6 band favors the independent projections at large row counts.
    return !(gate == GgmlType::Q6_K && rows >= 8192 && k <= 4096 && tokens <= 32);
}

std::size_t swiglu_workspace_capacity_bytes(GgmlType gate, GgmlType up, std::int32_t rows,
                                            std::int32_t k, std::int32_t first, std::int32_t last) {
    if (rows <= 0 || k <= 0 || k % block_values(gate) || k % block_values(up) || first <= 0 ||
        last < first) {
        throw std::invalid_argument("ggml swiglu workspace: invalid dimensions or token interval");
    }
    std::size_t peak    = linear_workspace_bytes(rows, k, last) + 256;
    const auto capacity = [&](int tokens, bool fused) {
        const std::size_t planes = fused ? 0 : 4 * std::size_t(rows) * tokens + 512;
        return planes + linear_workspace_bytes(rows, k, tokens) + 256;
    };
    for (int tokens = first; tokens <= std::min(last, 8); ++tokens) {
        peak = std::max(peak, capacity(tokens, swiglu_decode_admits(gate, up, rows, k, tokens)));
    }
    for (int tokens : {std::min(last, 32), last}) {
        if (tokens >= std::max(first, 9)) {
            peak =
                std::max(peak, capacity(tokens, swiglu_prefill_admits(gate, up, rows, k, tokens)));
        }
    }
    return peak;
}

void swiglu_prefill_launch(GgmlType type, const void* gate, const void* up, std::int32_t rows,
                           std::int32_t k, const __nv_bfloat16* x, std::int32_t tokens,
                           __nv_bfloat16* out, void* scratch, std::size_t scratch_bytes,
                           cudaStream_t stream) {
    if (!swiglu_prefill_admits(type, type, rows, k, tokens) || !gate || !up || !x || !out ||
        !scratch || (reinterpret_cast<std::uintptr_t>(scratch) & 15u) ||
        scratch_bytes < linear_workspace_bytes(rows, k, tokens)) {
        throw std::invalid_argument("ggml swiglu prefill: invalid inputs or scratch");
    }
    auto* codes = static_cast<std::int8_t*>(scratch);
    auto* ds    = reinterpret_cast<__half2*>(codes + std::size_t(tokens) * k);
    quantize_q8_1_planes_launch(x, k, tokens, codes, ds, stream);
    const auto launch = [&]<class Codec>() {
        const auto tile = [&]<int Rows, int Columns>() {
            swiglu_k_tile_kernel<Codec, Rows, Columns>
                <<<dim3((rows + Rows / 2 - 1) / (Rows / 2), (tokens + Columns - 1) / Columns),
                   (Columns / 8) * 32, 0, stream>>>(static_cast<const std::uint8_t*>(gate),
                                                    static_cast<const std::uint8_t*>(up), codes, ds,
                                                    rows, k, tokens, out);
        };
        if (rows <= 4096 && tokens <= 128) {
            tile.template operator()<32, 32>();
        } else if (tokens <= 64) {
            tile.template operator()<64, 32>();
        } else {
            tile.template operator()<64, 64>();
        }
    };
    switch (type) {
    case GgmlType::Q4_K:
        launch.template operator()<GgmlQ4KPrefill>();
        break;
    case GgmlType::Q5_K:
        launch.template operator()<GgmlQ5KPrefill>();
        break;
    case GgmlType::Q6_K:
        launch.template operator()<GgmlQ6KPrefill>();
        break;
    default:
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

bool swiglu_decode_admits(GgmlType gate, GgmlType up, std::int32_t rows, std::int32_t k,
                          std::int32_t tokens) noexcept {
    const auto supported = [](GgmlType type) {
        return type == GgmlType::Q8_0 || type == GgmlType::IQ4_NL || k_quant(type);
    };
    if (rows <= 0 || k <= 0 || k % 32 != 0 || tokens <= 0 || tokens > 8 || !supported(gate) ||
        !supported(up)) {
        return false;
    }
    if (k % block_values(gate) != 0 || k % block_values(up) != 0) { return false; }
    const std::uint64_t weight_bytes =
        std::uint64_t(rows) * (std::uint64_t(k / block_values(gate)) * block_bytes(gate) +
                               std::uint64_t(k / block_values(up)) * block_bytes(up));
    if (k_quant(gate) || k_quant(up)) {
        // Preserve full-model scalar throughput once the pair is large;
        // hot-weight probes alone overstate fusion's benefit at these sizes.
        if (tokens == 1 && weight_bytes > 64ull * 1024 * 1024) { return false; }
        // Separate mixed-codec tiles remain faster at wider decode widths.
        if (gate != up && tokens > 4) { return false; }
        // Mixed superblock/32-value formats share the scalar route. K-quant
        // pairs additionally have a tensor-core tile for batched decode.
        return (k_quant(gate) && k_quant(up)) || tokens == 1 ||
               (tokens == 2 && weight_bytes <= 64ull * 1024 * 1024);
    }
    // Keep the independent projections for large Q8-containing MMA workloads:
    // their wider row tiles sustain better bandwidth. Scalar decode and IQ4-only
    // pairs benefit from fusion at these sizes as well.
    return tokens == 1 || (gate == GgmlType::IQ4_NL && up == GgmlType::IQ4_NL) ||
           weight_bytes <= 64ull * 1024 * 1024;
}

void swiglu_decode_launch(GgmlType gate_type, const void* gate, GgmlType up_type, const void* up,
                          std::int32_t rows, std::int32_t k, const __nv_bfloat16* x,
                          std::int32_t tokens, __nv_bfloat16* out, void* scratch,
                          std::size_t scratch_bytes, cudaStream_t stream) {
    if (!swiglu_decode_admits(gate_type, up_type, rows, k, tokens) || gate == nullptr ||
        up == nullptr || x == nullptr || out == nullptr) {
        throw std::invalid_argument("ggml swiglu decode: invalid types, dimensions or pointers");
    }
    if (scratch == nullptr || scratch_bytes < linear_workspace_bytes(rows, k, tokens) ||
        (reinterpret_cast<std::uintptr_t>(scratch) & 15u) != 0) {
        throw std::invalid_argument("ggml swiglu decode: scratch too small or misaligned");
    }
    auto* codes = static_cast<std::int8_t*>(scratch);
    auto* ds    = reinterpret_cast<__half2*>(codes + std::size_t(tokens) * k);
    quantize_q8_1_planes_launch(x, k, tokens, codes, ds, stream);
    const auto launch = [&]<GgmlType Gate, GgmlType Up>() {
        const auto* g     = static_cast<const std::uint8_t*>(gate);
        const auto* u     = static_cast<const std::uint8_t*>(up);
        const auto scalar = [&] {
            constexpr int warps = k_quant(Gate) || k_quant(Up) ? 2 : 4;
            swiglu_decode_kernel<Gate, Up, warps>
                <<<dim3((rows + warps / 2 - 1) / (warps / 2), tokens), warps * 32, 0, stream>>>(
                    g, u, codes, ds, rows, k, tokens, out);
        };
        if constexpr (k_quant(Gate) && k_quant(Up)) {
            using GateCodec = typename PrefillCodecFor<Gate>::Codec;
            using UpCodec   = typename PrefillCodecFor<Up>::Codec;
            if (tokens == 2 || (tokens == 4 && rows <= 4096)) {
                swiglu_k_dual_kernel<GateCodec, UpCodec>
                    <<<dim3(rows, tokens / 2), 64, 0, stream>>>(g, u, codes, ds, rows, k, out);
            } else if (tokens == 1 || (tokens == 3 && rows <= 4096)) {
                scalar();
            } else {
                if constexpr (Gate == Up) {
                    swiglu_k_tile_kernel<GateCodec><<<(rows + 15) / 16, 32, 0, stream>>>(
                        g, u, codes, ds, rows, k, tokens, out);
                } else {
                    swiglu_k_mixed_tile_kernel<GateCodec, UpCodec>
                        <<<(rows + 15) / 16, 32, 0, stream>>>(g, u, codes, ds, rows, k, tokens,
                                                              out);
                }
            }
        } else if constexpr (k_quant(Gate) || k_quant(Up)) {
            scalar();
        } else {
            if (tokens == 1) {
                scalar();
            } else {
                swiglu_decode_mma_kernel<Gate, Up>
                    <<<(rows + 7) / 8, 256, 0, stream>>>(g, u, codes, ds, rows, k, tokens, out);
            }
        }
    };
    const auto launch_up = [&]<GgmlType Gate>() {
        switch (up_type) {
#define SINFER_SWIGLU_UP(Type)                                                                     \
    case GgmlType::Type:                                                                           \
        launch.template operator()<Gate, GgmlType::Type>();                                        \
        break;
            SINFER_SWIGLU_UP(Q4_K)
            SINFER_SWIGLU_UP(Q5_K)
            SINFER_SWIGLU_UP(Q6_K)
            SINFER_SWIGLU_UP(Q8_0)
            SINFER_SWIGLU_UP(IQ4_NL)
#undef SINFER_SWIGLU_UP
        default:
            break; // validated above
        }
    };
    switch (gate_type) {
#define SINFER_SWIGLU_GATE(Type)                                                                   \
    case GgmlType::Type:                                                                           \
        launch_up.template operator()<GgmlType::Type>();                                           \
        break;
        SINFER_SWIGLU_GATE(Q4_K)
        SINFER_SWIGLU_GATE(Q5_K)
        SINFER_SWIGLU_GATE(Q6_K)
        SINFER_SWIGLU_GATE(Q8_0)
        SINFER_SWIGLU_GATE(IQ4_NL)
#undef SINFER_SWIGLU_GATE
    default:
        break; // validated above
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail::ggml
