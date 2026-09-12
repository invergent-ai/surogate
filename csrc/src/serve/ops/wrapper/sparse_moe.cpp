#include "api/ops/lora_store.h"
#include "api/ops/lora_router.h"
#include "ops/linear/ggml/ggml_dispatch.h"
#include "api/ops/sparse_moe.h"

#include "core/nvtx.h"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"
#include "ops/sparse_moe/trtllm/trtllm_moe.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sinfer::ops {
namespace {

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end   = 0;
    std::string name;
};

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

AddressRange address_range(const void* pointer, std::size_t bytes, std::string name) {
    if (pointer == nullptr || bytes == 0) {
        throw std::invalid_argument("sparse_moe: " + name + " storage must be non-empty");
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error("sparse_moe: " + name + " address range overflows");
    }
    return {begin, begin + bytes, std::move(name)};
}

void require_disjoint(const std::vector<AddressRange>& ranges) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            if (ranges[i].begin < ranges[j].end && ranges[j].begin < ranges[i].end) {
                throw std::invalid_argument("sparse_moe: " + ranges[i].name + " overlaps " +
                                            ranges[j].name);
            }
        }
    }
}

std::int32_t require_tensor(const Tensor& tensor, std::int32_t hidden, const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != hidden || tensor.ne[1] < 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr || !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("sparse_moe: invalid ") + name);
    }
    return tensor.ne[1];
}

/// How wide a stored row is for `layout`, against how wide the math reads.
///
/// `row-split-k128-v1` pads K up to 128 (`row_split_geometry`), so a weight of width 704 is
/// stored 768 wide and its `padded_shape[1]` says so. Every mixture registered before Gemma 4
/// had expert widths that were multiples of 128, so the two always coincided and this check
/// could compare `padded_shape[1]` to `k` directly. The GGML superblock and NVFP4 layouts store
/// K exactly and keep the two equal.
std::int32_t stored_columns(QuantLayout layout, std::int32_t k) {
    if (layout != QuantLayout::RowSplit) { return k; }
    return static_cast<std::int32_t>(round_up<std::int64_t>(k, 128));
}

void require_matrix_metadata(const Weight& weight, std::int32_t n, std::int32_t k,
                             const char* name) {
    if (weight.ndim != 2 || weight.n != n || weight.k != k || weight.shape[0] != n ||
        weight.shape[1] != k || weight.shape[2] != 1 || weight.shape[3] != 1 ||
        weight.padded_shape[0] != n ||
        weight.padded_shape[1] != stored_columns(weight.layout, k) ||
        weight.padded_shape[2] != 1 || weight.padded_shape[3] != 1) {
        throw std::invalid_argument(std::string("sparse_moe: invalid shape for ") + name);
    }
}

void require_router(const Weight& weight, const SparseMoeGeometry& geometry,
                    std::vector<AddressRange>& ranges) {
    require_matrix_metadata(weight, geometry.router_rows(), geometry.hidden, "router_shared_gate");
    const std::size_t bytes = static_cast<std::size_t>(geometry.router_rows()) * geometry.hidden * 2;
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.qdata == nullptr || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.payload_bytes < bytes || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(
            "sparse_moe: router_shared_gate must be aligned contiguous BF16");
    }
    ranges.push_back(address_range(weight.qdata, bytes, "router_shared_gate"));
}

struct QuantGeometry {
    std::int32_t group_size;
    std::size_t code_bytes_per_group;
    std::size_t high_bytes_per_group;
    std::size_t scale_bytes_per_group = 2;      // fp16 for the row-split codecs
    QuantLayout layout                = QuantLayout::RowSplit;
    DType scale_dtype                 = DType::FP16;
};

QuantGeometry quant_geometry(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return {64, 32, 0};
    case QType::Q5G64_F16S:
        return {64, 32, 8};
    case QType::Q6G64_F16S:
        return {64, 32, 16};
    case QType::W8G32_F16S:
        return {32, 32, 0};
    case QType::NVFP4:
        // Sixteen values per group, two per code byte, one e4m3 scale byte, and the dense
        // block-scale tiling rather than row-split.
        return {16, 8, 0, 1, QuantLayout::BlockScaleK16M128x4, DType::FP8_E4M3FN};
    default:
        break;
    }
    // A GGML block carries its scales inside itself, so there is one plane and its "code bytes"
    // are the whole block. Both numbers come from the block vocabulary rather than a second
    // table here, so a format the kernels read is a format this accepts.
    if (detail::ggml::is_ggml_qtype(qtype)) {
        const auto type = detail::ggml::ggml_type_for(qtype);
        return {detail::ggml::block_values(type), detail::ggml::block_bytes(type), 0, 0,
                QuantLayout::GgmlBlocks, DType::FP16};
    }
    throw std::invalid_argument("sparse_moe: unsupported quantized weight format");
}

void require_quantized(const Weight& weight, std::int32_t n, std::int32_t k, const char* name,
                       std::vector<AddressRange>& ranges) {
    require_matrix_metadata(weight, n, k, name);
    const QuantGeometry geometry = quant_geometry(weight.qtype);
    // The planes are sized and addressed by what the layout *stores*, which is the padded width
    // where the layout pads. Sizing them from `k` would under-count the payload -- so the
    // capacity check would pass on a truncated buffer -- and would hand the overlap check a
    // range shorter than the one the kernels read.
    const std::size_t stored           = static_cast<std::size_t>(stored_columns(weight.layout, k));
    const std::size_t groups           = static_cast<std::size_t>(n) * stored / geometry.group_size;
    const std::size_t code_bytes       = groups * geometry.code_bytes_per_group;
    const std::size_t high_bytes       = groups * geometry.high_bytes_per_group;
    const std::size_t scale_bytes      = groups * geometry.scale_bytes_per_group;
    const std::size_t required_payload = code_bytes + high_bytes + scale_bytes;
    // A superblock format keeps its scales inside the block, so it has no scale plane to check.
    const bool needs_scale_plane = geometry.scale_bytes_per_group != 0;
    if (weight.layout != geometry.layout || weight.scale_dtype != geometry.scale_dtype ||
        weight.group_size != static_cast<std::uint32_t>(geometry.group_size) ||
        weight.group != geometry.group_size || weight.qdata == nullptr ||
                (needs_scale_plane && (weight.scales == nullptr || !aligned_to(weight.scales, 16))) ||
        weight.payload_bytes < required_payload || weight.high_plane_bytes < high_bytes ||
        !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string("sparse_moe: invalid quantized ") + name);
    }
    if ((high_bytes == 0 && weight.qhigh != nullptr) ||
        (high_bytes != 0 && (weight.qhigh == nullptr || !aligned_to(weight.qhigh, 16)))) {
        throw std::invalid_argument(std::string("sparse_moe: invalid high plane for ") + name);
    }
    ranges.push_back(address_range(weight.qdata, code_bytes, std::string(name) + " code"));
    if (high_bytes != 0) {
        ranges.push_back(address_range(weight.qhigh, high_bytes, std::string(name) + " high"));
    }
    if (needs_scale_plane) {
        ranges.push_back(address_range(weight.scales, scale_bytes, std::string(name) + " scales"));
    }
}

void validate_weights(const SparseMoeWeights& weights, const SparseMoeGeometry& geometry,
                      std::vector<AddressRange>& ranges) {
    require_router(weights.router_shared_gate, geometry, ranges);
    const auto is_ggml_k = [](QType qtype) { return detail::ggml::is_ggml_qtype(qtype); };
    if (weights.routed_gate_up.qtype != QType::Q4G64_F16S &&
        weights.routed_gate_up.qtype != QType::W8G32_F16S &&
        weights.routed_gate_up.qtype != QType::NVFP4 && !is_ggml_k(weights.routed_gate_up.qtype)) {
        throw std::invalid_argument(
            "sparse_moe: routed_gate_up must be Q4, W8, NVFP4, or a GGML K-quant");
    }
    if (weights.routed_down.qtype != QType::Q5G64_F16S &&
        weights.routed_down.qtype != QType::Q6G64_F16S &&
        weights.routed_down.qtype != QType::W8G32_F16S &&
        weights.routed_down.qtype != QType::NVFP4 && !is_ggml_k(weights.routed_down.qtype)) {
        throw std::invalid_argument(
            "sparse_moe: routed_down must be Q5, Q6, W8, NVFP4, or a GGML K-quant");
    }
    if (geometry.gating == SparseMoeGating::SigmoidBiasTopK) {
        if (weights.router_bias == nullptr) {
            throw std::invalid_argument(
                "sparse_moe: a sigmoid router selects on the score plus a per-expert bias and "
                "cannot rank without it");
        }
        ranges.push_back(address_range(weights.router_bias,
                                       static_cast<std::size_t>(geometry.experts) * sizeof(float),
                                       "router_bias"));
    } else if (weights.router_bias != nullptr) {
        throw std::invalid_argument(
            "sparse_moe: this mixture's router is a softmax over the logits and has no bias to "
            "rank with; passing one means the caller expects a router this geometry is not");
    }
    if (geometry.per_expert_scaled) {
        if (weights.per_expert_scale == nullptr) {
            throw std::invalid_argument(
                "sparse_moe: this mixture scales its renormalised routing weights by a learned "
                "per-expert vector and cannot weight its winners without it");
        }
        ranges.push_back(address_range(weights.per_expert_scale,
                                       static_cast<std::size_t>(geometry.experts) * sizeof(float),
                                       "per_expert_scale"));
    } else if (weights.per_expert_scale != nullptr) {
        // Not a harmless extra pointer: nothing would read it, so the round would weight its
        // experts by the checkpoint's routing weights alone and say nothing about the tensor it
        // ignored.
        throw std::invalid_argument(
            "sparse_moe: this mixture's router has no per-expert scale; passing one means the "
            "caller expects a router this geometry is not");
    }
    if (geometry.has_shared()) {
        if (weights.shared_gate_up.qtype != QType::W8G32_F16S ||
            weights.shared_down.qtype != QType::W8G32_F16S) {
            throw std::invalid_argument("sparse_moe: shared weights must be W8");
        }
    } else if (weights.shared_gate_up.qdata != nullptr || weights.shared_down.qdata != nullptr) {
        // A routed-only geometry with shared weights attached is a caller that thinks this
        // mixture has an always-on expert. Nothing downstream would read them, so the round
        // would come out missing a path and say nothing.
        throw std::invalid_argument(
            "sparse_moe: this geometry routes every token, so it must carry no shared weights");
    }
    // NVFP4's second level is not optional: the checkpoint's per-expert global scale spans 3-7x
    // across the experts of one layer, so a missing array is not a small error but a per-expert
    // gain of the wrong size — and it would be silent. Fail here instead.
    if ((weights.routed_gate_up.qtype == QType::NVFP4) != (weights.routed_gate_up_scale != nullptr)) {
        throw std::invalid_argument(
            "sparse_moe: routed_gate_up_scale is required for NVFP4 and rejected otherwise");
    }
    if ((weights.routed_down.qtype == QType::NVFP4) != (weights.routed_down_scale != nullptr)) {
        throw std::invalid_argument(
            "sparse_moe: routed_down_scale is required for NVFP4 and rejected otherwise");
    }
    // The NVFP4 routed profile is served by the vendored W4A4 runner, which also quantises the
    // activations, so it needs the checkpoint's per-expert activation scale and the matching
    // epilogue alpha. Both are as load-bearing as the weight scale and equally silent if absent.
    {
        const bool nvfp4_routed = weights.routed_gate_up.qtype == QType::NVFP4 &&
                                  weights.routed_down.qtype == QType::NVFP4;
        const bool present = weights.routed_gate_up_act_scale != nullptr &&
                             weights.routed_gate_up_alpha != nullptr &&
                             weights.routed_down_act_scale != nullptr &&
                             weights.routed_down_alpha != nullptr;
        const bool any = weights.routed_gate_up_act_scale != nullptr ||
                         weights.routed_gate_up_alpha != nullptr ||
                         weights.routed_down_act_scale != nullptr ||
                         weights.routed_down_alpha != nullptr;
        if (nvfp4_routed ? !present : any) {
            throw std::invalid_argument(
                "sparse_moe: the four routed activation-scale and alpha arrays are required for "
                "NVFP4 routed experts and rejected otherwise");
        }
        if (nvfp4_routed && !detail::trtllm_moe::available()) {
            throw std::invalid_argument(
                "sparse_moe: NVFP4 routed experts need the TRT-LLM MoE runner, which this build "
                "does not contain (it needs the sm_120a architecture)");
        }
    }
    if (weights.slot_of_expert == nullptr) {
        require_quantized(weights.routed_gate_up, geometry.routed_gate_rows(), geometry.hidden,
                          "routed_gate_up", ranges);
        require_quantized(weights.routed_down, geometry.routed_down_rows(), geometry.intermediate,
                          "routed_down", ranges);
    } else {
        // An expert slot pool: the routed weights hold `slots` experts in id-independent order
        // and the per-layer table maps expert ids to row blocks.
        const std::int32_t gate_rows = geometry.expert_rows();
        if (weights.routed_gate_up.n <= 0 || weights.routed_gate_up.n % gate_rows != 0 ||
            weights.routed_down.n <= 0 || weights.routed_down.n % geometry.hidden != 0 ||
            weights.routed_gate_up.n / gate_rows != weights.routed_down.n / geometry.hidden) {
            throw std::invalid_argument(
                "sparse_moe: expert slot pool rows are not a whole number of experts");
        }
        require_quantized(weights.routed_gate_up, weights.routed_gate_up.n, geometry.hidden,
                          "routed_gate_up", ranges);
        require_quantized(weights.routed_down, weights.routed_down.n, geometry.intermediate,
                          "routed_down", ranges);
    }
    if (geometry.has_shared()) {
        require_quantized(weights.shared_gate_up, geometry.shared_rows(), geometry.hidden,
                          "shared_gate_up", ranges);
        require_quantized(weights.shared_down, geometry.hidden, geometry.shared_intermediate,
                          "shared_down", ranges);
    }
}

void require_registered(const SparseMoeGeometry& geometry) {
    for (const SparseMoeGeometry& registered : kSparseMoeGeometries) {
        if (geometry == registered) { return; }
    }
    throw std::invalid_argument(
        "sparse_moe: geometry {hidden " + std::to_string(geometry.hidden) + ", experts " +
        std::to_string(geometry.experts) + ", top-k " +
        std::to_string(geometry.experts_per_token) + ", intermediate " +
        std::to_string(geometry.intermediate) + ", shared " +
        std::to_string(geometry.shared_intermediate) + "} is not registered");
}

} // namespace

void sparse_moe_prepare(const SparseMoeWeights& weights, std::int32_t max_tokens,
                        cudaStream_t stream) {
    if (weights.routed_gate_up.qtype != QType::NVFP4 ||
        weights.routed_down.qtype != QType::NVFP4) {
        return;
    }
    const SparseMoeGeometry geometry = sparse_moe_geometry(weights);
    require_registered(geometry);
    std::vector<AddressRange> ranges;
    validate_weights(weights, geometry, ranges);
    detail::trtllm_moe::Nvfp4RoutedExperts experts;
    experts.gate_up_codes        = weights.routed_gate_up.qdata;
    experts.gate_up_block_scales = weights.routed_gate_up.scales;
    experts.gate_up_act_scale    = weights.routed_gate_up_act_scale;
    experts.gate_up_alpha        = weights.routed_gate_up_alpha;
    experts.down_codes           = weights.routed_down.qdata;
    experts.down_block_scales    = weights.routed_down.scales;
    experts.down_act_scale       = weights.routed_down_act_scale;
    experts.down_alpha           = weights.routed_down_alpha;
    detail::trtllm_moe::prepare(detail::trtllm_moe::Geometry{geometry.hidden, geometry.experts,
                                                            geometry.experts_per_token,
                                                            geometry.intermediate},
                                experts, max_tokens, stream);
}

SparseMoeGeometry sparse_moe_geometry(const SparseMoeWeights& weights) {
    // Every dimension comes from the weight that carries it. The routed experts' width used to
    // be read off the *shared* expert's down projection, which is right only where the two are
    // equal and meaningless where there is no shared expert at all; `routed_down` is [E*hidden,
    // intermediate] and states it directly.
    const bool shared = weights.shared_down.qdata != nullptr;
    const SparseMoeGeometry geometry{
        .hidden              = weights.router_shared_gate.k,
        // The router's extra row is the shared expert's gate, so it is there exactly when the
        // shared expert is.
        // The extra router row is the shared expert's gate, so it is there exactly when the
        // shared expert is *and* is gated -- GLM-5.3 has one and does not gate it.
        .experts             = weights.router_shared_gate.n -
                               (shared && weights.shared_gated ? 1 : 0),
        .experts_per_token   = weights.experts_per_token,
        .intermediate        = weights.routed_down.k,
        // The router's own bias says which gating this is: a softmax router has no such tensor
        // and a sigmoid one cannot rank without it.
        .gating              = weights.router_bias != nullptr ? SparseMoeGating::SigmoidBiasTopK
                                                              : SparseMoeGating::SoftmaxTopK,
        .routed_scale        = weights.routed_scale,
        .shared_gated        = weights.shared_gated,
        .shared_intermediate = shared ? weights.shared_down.k : 0,
        .swiglu_limit        = weights.swiglu_limit,
        // The gate is the caller's statement, like the clamp and the routed scale: no shape
        // carries it, and a checkpoint served through the wrong one computes a different
        // function silently. `require_registered` is what makes the statement checkable -- a
        // geometry whose numbers match a registered mixture but whose gate does not is refused.
        .activation          = weights.activation,
        // The per-expert scale, like the router bias, states itself by being there.
        .per_expert_scaled   = weights.per_expert_scale != nullptr,
    };
    require_registered(geometry);
    return geometry;
}

/// The widest small-T slice a round of `tokens` runs. Small-T starts at two, so a slice that
/// would leave a single token behind gives one back: 47 runs 45 + 2, never 46 + 1. The first
/// slice is always the widest, which is what a workspace has to hold.
constexpr std::int32_t small_t_first_slice(std::int32_t tokens) noexcept {
    const std::int32_t slice =
        tokens < detail::kSparseMoeSmallTMax ? tokens : detail::kSparseMoeSmallTMax;
    return tokens - slice == 1 ? slice - 1 : slice;
}

/// The widest slice any round in `[min_tokens, max_tokens]` runs. `small_t_first_slice` rises
/// with `tokens` apart from the one dip at 47, and an interval that reaches past 46 also
/// contains a round of exactly 46 unless it starts above it.
std::int32_t small_t_widest_slice(std::int32_t min_tokens, std::int32_t max_tokens) noexcept {
    std::int32_t widest = small_t_first_slice(max_tokens);
    if (max_tokens > detail::kSparseMoeSmallTMax && min_tokens <= detail::kSparseMoeSmallTMax) {
        widest = std::max(widest, static_cast<std::int32_t>(detail::kSparseMoeSmallTMax));
    }
    return widest;
}

/// The narrowest round the routed-NVFP4 profile hands to the vendored runner. Below it our own
/// decode and small-T kernels are faster: the runner permutes, groups and reduces for a batch,
/// and at a handful of rows that scaffolding costs more than the GEMMs it enables. Measured, and
/// overridable so the crossover can be re-measured rather than assumed.
std::int32_t trtllm_min_tokens() {
    static const std::int32_t value = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_MOE_TRTLLM_MIN");
        if (raw == nullptr || *raw == '\0') { return kSparseMoeTrtllmMinTokens; }
        const long parsed = std::strtol(raw, nullptr, 10);
        return parsed > 0 ? static_cast<std::int32_t>(parsed) : kSparseMoeTrtllmMinTokens;
    }();
    return value;
}

std::size_t sparse_moe_workspace_capacity_bytes(const SparseMoeGeometry& geometry,
                                                QType routed_gate_up, QType routed_down,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    require_registered(geometry);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("sparse_moe workspace: invalid token interval");
    }
    (void)detail::resolve_sparse_moe_decode_plan(geometry, routed_gate_up, routed_down);
    const bool w8_profile = routed_gate_up == QType::W8G32_F16S && routed_down == QType::W8G32_F16S;
    // The NVFP4 routed profile has no kernel of ours at any width: the prefill family carries it
    // from a single token upwards, with the vendored runner computing the routed half.
    const bool nvfp4_profile = routed_gate_up == QType::NVFP4 && routed_down == QType::NVFP4;
    const std::int32_t prefill_first =
        nvfp4_profile ? trtllm_min_tokens()
        : w8_profile  ? detail::kSparseMoePrefillW8W8Min
                      : (routed_down == QType::Q5G64_F16S ? detail::kSparseMoePrefillQ4Q5Min
                                                          : detail::kSparseMoePrefillQ4Q6Min);
    // Adapted rounds execute their columns concurrently in one set of launches.
    std::size_t required = detail::sparse_moe_decode_workspace_bytes(geometry, max_tokens);
    const std::int32_t small_first = std::max(min_tokens, detail::kSparseMoeSmallTMin);
    const std::int32_t small_last =
        std::min({max_tokens, detail::kSparseMoeSmallTMax, prefill_first - 1});
    if (small_first <= small_last) {
        required =
            std::max(required, detail::sparse_moe_small_t_workspace_bytes(geometry, small_last));
    }
    if (nvfp4_profile && small_last >= detail::kSparseMoeSmallTMin) {
        // Between the small-T bound and the runner's first width, an NVFP4 round still walks the
        // small-T path in slices, so the widest slice any round in that band takes sets the
        // requirement.
        const std::int32_t slice =
            small_t_widest_slice(std::max(min_tokens, detail::kSparseMoeSmallTMin),
                                 std::min(max_tokens, prefill_first - 1));
        required = std::max(required, detail::sparse_moe_small_t_workspace_bytes(geometry, slice));
    }
    const std::int32_t prefill_interval_first = std::max(min_tokens, prefill_first);
    if (prefill_interval_first <= max_tokens) {
        required = std::max(required,
                            detail::sparse_moe_prefill_workspace_bytes(
                                geometry, max_tokens, nvfp4_profile,
                                detail::sparse_moe_routed_int8_profile(routed_gate_up, routed_down)));
    }
    return required;
}

void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream) {
    sparse_moe(x, weights, epilogue, destination, workspace, stream, SparseMoeRoundHook{});
}

void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream,
                const SparseMoeRoundHook& hook) {
    // One tensor for both projections, which is what every mixture but Gemma 4 does.
    sparse_moe(x, x, weights, epilogue, destination, workspace, stream, hook);
}

void sparse_moe(const Tensor& x, const Tensor& router_x, const SparseMoeWeights& weights,
                SparseMoeEpilogue epilogue, Tensor& destination, WorkspaceArena& workspace,
                cudaStream_t stream, const SparseMoeRoundHook& hook) {
    const SparseMoeRoundHook* round_hook = hook.resolve != nullptr ? &hook : nullptr;
    if (epilogue != SparseMoeEpilogue::AddResidual) {
        throw std::invalid_argument("sparse_moe: unsupported epilogue");
    }
    const SparseMoeGeometry geometry = sparse_moe_geometry(weights);
    const std::int32_t tokens        = require_tensor(x, geometry.hidden, "x");
    if (require_tensor(destination, geometry.hidden, "destination") != tokens) {
        throw std::invalid_argument("sparse_moe: x and destination token counts must match");
    }
    const bool separate_router_input = router_x.data != x.data;
    if (require_tensor(router_x, geometry.hidden, "router_x") != tokens) {
        throw std::invalid_argument("sparse_moe: x and router_x token counts must match");
    }
    std::vector<AddressRange> ranges;
    ranges.reserve(16);
    ranges.push_back(address_range(x.data, x.bytes(), "x"));
    // The common call passes one tensor twice, and two identical ranges are not an overlap to
    // report -- so it is named only when it is a buffer of its own.
    if (separate_router_input) {
        ranges.push_back(address_range(router_x.data, router_x.bytes(), "router_x"));
    }
    ranges.push_back(address_range(destination.data, destination.bytes(), "destination"));
    validate_weights(weights, geometry, ranges);

    const QType gate_up = weights.routed_gate_up.qtype;
    const QType down    = weights.routed_down.qtype;
    // The NVFP4 routed profile has two arms. Wide rounds go to the vendored TRT-LLM runner
    // through the prefill family, which is where its grouped GEMMs pay; narrow ones stay on our
    // decode and small-T kernels, which read the same artifact because the codec carries its
    // [up; gate] row order (`Nvfp4CodecFor::kGateRowsFirst`). Rounds between the small-T bound
    // and the crossover walk small-T in slices, as they did before the runner existed.
    const bool nvfp4_routed = gate_up == QType::NVFP4 && down == QType::NVFP4;
    // A GGML K-quant has no prefill kernel of this family yet, so without this every round wider
    // than the small-T bound fell to the per-token decode loop -- one launch per prompt token.
    // Walking small-T slices instead is the same arrangement NVFP4 uses for the same reason.
    const auto is_ggml_k_qtype = [](QType qtype) {
        return detail::ggml::is_ggml_qtype(qtype);
    };
    const bool ggml_k_routed = is_ggml_k_qtype(gate_up) && is_ggml_k_qtype(down);
    const auto& adapter_round = lora_current_round();
    const auto* adapters = lora_active() && adapter_round.valid()
        ? lora_store_for_current_device().bank_table(weights.router_shared_gate.qdata) : nullptr;
    const std::int32_t resolve_tokens = hook.max_resolve_tokens > 0
        ? hook.max_resolve_tokens : std::numeric_limits<std::int32_t>::max();
    if (adapters && tokens > resolve_tokens) {
        // Adapter routing is batched too. Keep each hook invocation within the cache's
        // capacity and preserve the adapter column IDs when advancing through the request.
        const LoraRound saved_round = adapter_round;
        for (std::int32_t offset = 0; offset < tokens; offset += resolve_tokens) {
            const auto count = std::min(resolve_tokens, tokens - offset);
            const Tensor input = x.slice(1, offset, count);
            const Tensor router_input = router_x.slice(1, offset, count);
            Tensor output = destination.slice(1, offset, count);
            if (saved_round.slots) {
                ScopedLoraColumns columns(saved_round.slots->slice(0, offset, count));
                sparse_moe(input, router_input, weights, epilogue, output, workspace, stream, hook);
            } else {
                sparse_moe(input, router_input, weights, epilogue, output, workspace, stream, hook);
            }
        }
        return;
    }
    const bool use_prefill  = !adapters && (nvfp4_routed
                                  ? tokens >= trtllm_min_tokens()
                                  : detail::sparse_moe_uses_prefill(tokens, gate_up, down));
    const bool use_small_t =
        !adapters && !use_prefill && resolve_tokens >= 3 && (detail::sparse_moe_uses_small_t(tokens) ||
                         ((nvfp4_routed || ggml_k_routed) && tokens > 1));
    nvtx::ScopedRange moe_range(use_prefill   ? nvtx::Name::SparseMoePrefill
                                : use_small_t ? nvtx::Name::SparseMoeSmallT
                                              : nvtx::Name::SparseMoeDecode,
                                nvtx::Category::Moe, static_cast<std::uint64_t>(tokens));
    std::size_t required = 0;
    if (use_prefill) {
        required = detail::resolve_sparse_moe_prefill_plan(geometry, tokens, gate_up, down)
                       .workspace_bytes;
    } else if (use_small_t) {
        // Wide rounds run as slices, so the workspace only has to hold the widest one.
        required =
            detail::resolve_sparse_moe_small_t_plan(geometry, small_t_first_slice(tokens), gate_up,
                                                    down)
                .workspace_bytes;
    } else {
        required = detail::sparse_moe_decode_workspace_bytes(geometry, adapters ? tokens : 1);
    }
    if (workspace.base() == nullptr || workspace.capacity() < required ||
        workspace.used() > workspace.capacity() - required) {
        throw std::invalid_argument("sparse_moe: insufficient workspace capacity");
    }
    ranges.push_back(address_range(workspace.base(), workspace.capacity(), "workspace"));
    require_disjoint(ranges);

    auto scope = workspace.scope();
    if (use_prefill) {
        const detail::SparseMoePrefillPlan plan =
            detail::resolve_sparse_moe_prefill_plan(geometry, tokens, gate_up, down);
        const detail::SparseMoePrefillWorkspace views =
            detail::allocate_sparse_moe_prefill_workspace(workspace, geometry, plan.slice_tokens,
                                                          plan.routed_trtllm, plan.routed_int8);
        detail::sparse_moe_prefill_launch(geometry, x, router_x, weights, destination, plan,
                                          views, stream, round_hook);
        return;
    }
    if (use_small_t) {
        for (std::int32_t offset = 0; offset < tokens;) {
            std::int32_t slice = std::min(small_t_first_slice(tokens - offset), resolve_tokens);
            if (tokens - offset - slice == 1) { --slice; }
            auto slice_scope = workspace.scope();
            const detail::SparseMoeSmallTPlan plan =
                detail::resolve_sparse_moe_small_t_plan(geometry, slice, gate_up, down);
            const detail::SparseMoeSmallTWorkspace views =
                detail::allocate_sparse_moe_small_t_workspace(workspace, geometry, slice);
            const Tensor x_slice      = x.slice(1, offset, slice);
            const Tensor router_slice = router_x.slice(1, offset, slice);
            Tensor destination_slice  = destination.slice(1, offset, slice);
            detail::sparse_moe_small_t_launch(geometry, x_slice, router_slice, weights,
                                              destination_slice, plan, views, stream, round_hook);
            offset += slice;
        }
        return;
    }
    auto views = detail::allocate_sparse_moe_decode_workspace(workspace, geometry, adapters ? tokens : 1);
    if (adapters) {
        views.slot_stride = adapter_round.slots ? 1 : 0;
        const auto* slots = adapter_round.slots ? static_cast<const std::int32_t*>(adapter_round.slots->data)
                                                : adapter_round.uniform_cell;
        struct Correction {
            const SparseMoeWeights& weights; const SparseMoeGeometry& geometry;
            const Tensor& scores; const LoraBank* banks; const std::int32_t* slots;
            int stride; const SparseMoeRoundHook* next;
        } correction{weights, geometry, views.scratch, adapters, slots, views.slot_stride, round_hook};
        const SparseMoeRoundHook corrected{
            [](void* context, const Tensor& ids, const Tensor& alpha, const Tensor& input,
               Tensor& output, cudaStream_t stream) {
                const auto& c = *static_cast<Correction*>(context);
                lora_router_bias(c.scores, ids, alpha, c.weights.router_bias, c.weights.per_expert_scale,
                                 c.geometry, c.banks, c.slots, c.stride, stream);
                if (c.next) { c.next->resolve(c.next->context, ids, alpha, input, output, stream); }
            }, &correction};
        detail::sparse_moe_decode_launch(geometry, x, router_x, weights, destination, views,
            stream, weights.router_bias ? &corrected : round_hook, adapters, slots);
        return;
    }
    for (std::int32_t token = 0; token < tokens; ++token) {
        const Tensor x_column      = x.slice(1, token, 1);
        const Tensor router_column = router_x.slice(1, token, 1);
        Tensor destination_column  = destination.slice(1, token, 1);
        detail::sparse_moe_decode_launch(geometry, x_column, router_column, weights,
                                         destination_column, views, stream, round_hook, adapters,
                                         !adapters ? nullptr : adapter_round.slots
                                            ? static_cast<const std::int32_t*>(adapter_round.slots->data) + token
                                            : adapter_round.uniform_cell);
    }
}

} // namespace sinfer::ops
