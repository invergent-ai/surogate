#include "targets/qwen4exp/impl/load/bindings.h"

#include "artifact/reader.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sinfer::targets::qwen4exp::detail {
namespace {

using artifact::NumericFormat;
using artifact::TensorPlacement;

constexpr std::uint64_t kHidden   = TextConfig::hidden;
constexpr std::uint64_t kHcWidth  = TextConfig::hc_width;
constexpr std::uint64_t kHcRank   = TextConfig::hc_low_rank;
constexpr std::uint64_t kHcCount  = TextConfig::hc_count;
constexpr std::uint64_t kVocab    = TextConfig::output_rows;
constexpr std::uint64_t kExperts  = TextConfig::experts;
constexpr std::uint64_t kFfn      = TextConfig::intermediate;
constexpr std::uint64_t kValueDim = TextConfig::value_dim;

// Placement of the layer being bound: Device for the layers this program runs, ValidateOnly
// for the layers of other pipeline stages (validated, never uploaded).
thread_local TensorPlacement g_layer_placement = TensorPlacement::Device;

artifact::ObjectHandle device(artifact::Binder& binder, const std::string& name,
                              NumericFormat format, std::initializer_list<std::uint64_t> shape) {
    return artifact::bind_tensor(binder, name, format, shape, g_layer_placement);
}

// As `host`, but marks the object for W8→Q4G32AM requantisation at bank load.
artifact::ObjectHandle host_q4(artifact::Binder& binder, HostBankPlan& bank,
                               const std::string& name, NumericFormat format,
                               std::initializer_list<std::uint64_t> shape) {
    const artifact::ObjectHandle handle =
        artifact::bind_tensor(binder, name, format, shape, TensorPlacement::ValidateOnly);
    const artifact::RowSplitGeometry geometry = artifact::row_split_geometry(
        format, std::span<const std::uint64_t>(shape.begin(), shape.size()));
    HostObjectPlan plan{handle, binder.payload(handle).data, name};
    plan.q4_rows            = static_cast<std::int64_t>(*shape.begin());
    plan.q4_k               = static_cast<std::int32_t>(*(shape.begin() + 1));
    plan.q4_w8_scale_offset = geometry.scale_plane_offset;
    bank.objects.push_back(std::move(plan));
    return handle;
}

// Validates the object and records its mapping for the pinned host bank.
artifact::ObjectHandle host(artifact::Binder& binder, HostBankPlan& bank, const std::string& name,
                            NumericFormat format, std::initializer_list<std::uint64_t> shape) {
    const artifact::ObjectHandle handle =
        artifact::bind_tensor(binder, name, format, shape, TensorPlacement::ValidateOnly);
    bank.objects.push_back({handle, binder.payload(handle).data, name});
    return handle;
}

HyperConnectionPlan bind_hc(artifact::Binder& binder, const std::string& prefix,
                            bool with_inject) {
    HyperConnectionPlan plan;
    plan.norm = device(binder, prefix + "norm", NumericFormat::FP32, {kHcWidth});
    plan.down = device(binder, prefix + "down", NumericFormat::BF16, {kHcRank, kHcWidth});
    plan.up   = device(binder, prefix + "up", NumericFormat::BF16, {kHcWidth, kHcRank});
    if (with_inject) {
        plan.inject = device(binder, prefix + "inject", NumericFormat::BF16, {kHcCount, kHcWidth});
    }
    return plan;
}

MoePlan bind_moe(artifact::Binder& binder, HostBankPlan& bank, const std::string& prefix,
                 bool q4) {
    const auto routed = [&](const std::string& name, std::initializer_list<std::uint64_t> shape) {
        return q4 ? host_q4(binder, bank, name, NumericFormat::W8G32_F16S, shape)
                  : host(binder, bank, name, NumericFormat::W8G32_F16S, shape);
    };
    return MoePlan{
        .router_shared_gate = device(binder, prefix + "router_shared_gate", NumericFormat::BF16,
                                     {kExperts + 1, kHidden}),
        .routed_gate_up = routed(prefix + "routed_gate_up", {kExperts * 2 * kFfn, kHidden}),
        .routed_down    = routed(prefix + "routed_down", {kExperts * kHidden, kFfn}),
        .shared_gate_up = device(binder, prefix + "shared_gate_up", NumericFormat::W8G32_F16S,
                                 {2 * kFfn, kHidden}),
        .shared_down =
            device(binder, prefix + "shared_down", NumericFormat::W8G32_F16S, {kHidden, kFfn}),
    };
}

template <class T, std::size_t N>
void read_i32_array(artifact::Binder& binder, artifact::ObjectHandle handle, std::array<T, N>& out) {
    const auto bytes = binder.payload(handle).data;
    if (bytes.size() < N * sizeof(std::int32_t)) {
        throw artifact::ArtifactError("PLE hash constant object is too short");
    }
    for (std::size_t i = 0; i < N; ++i) {
        std::int32_t value = 0;
        std::memcpy(&value, bytes.data() + i * sizeof(std::int32_t), sizeof(value));
        out[i] = static_cast<T>(value);
    }
}

ops::HyperConnectionWeights load_hc(const artifact::MaterializedArtifact& backing,
                                    const HyperConnectionPlan& plan, bool with_inject) {
    ops::HyperConnectionWeights out;
    out.norm = artifact::materialized_tensor(backing, plan.norm, NumericFormat::FP32,
                                             {static_cast<std::int32_t>(kHcWidth)});
    out.down = artifact::materialized_weight(backing, plan.down, NumericFormat::BF16,
                                             static_cast<std::int32_t>(kHcRank),
                                             static_cast<std::int32_t>(kHcWidth));
    out.up   = artifact::materialized_weight(backing, plan.up, NumericFormat::BF16,
                                             static_cast<std::int32_t>(kHcWidth),
                                             static_cast<std::int32_t>(kHcRank));
    if (with_inject) {
        out.inject = artifact::materialized_weight(backing, plan.inject, NumericFormat::BF16,
                                                   static_cast<std::int32_t>(kHcCount),
                                                   static_cast<std::int32_t>(kHcWidth));
    }
    return out;
}

// A row-split W8 weight whose planes live in the pinned host bank, addressed through the
// mapped device pointer (the same layout the device materializer produces).
Weight host_w8_weight(const HostObject& object, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::RowSplitGeometry geometry =
        artifact::row_split_geometry(NumericFormat::W8G32_F16S, shape);
    if (geometry.encoded_bytes != object.bytes) {
        throw std::logic_error("host bank object " + object.name + " has an unexpected size");
    }
    const auto* bytes = static_cast<const std::byte*>(object.device);
    Weight out{};
    out.payload          = bytes;
    out.payload_bytes    = geometry.encoded_bytes;
    out.high_plane_bytes = geometry.high_plane_bytes;
    out.qtype            = QType::W8G32_F16S;
    out.layout           = QuantLayout::RowSplit;
    out.group_size       = static_cast<std::uint32_t>(geometry.group_size);
    out.qdata            = bytes;
    out.qhigh            = nullptr;
    out.scales           = bytes + geometry.scale_plane_offset;
    out.n                = rows;
    out.k                = columns;
    out.group            = static_cast<std::int32_t>(geometry.group_size);
    out.scale_dtype      = DType::FP16;
    out.ndim             = 2;
    out.shape[0]         = rows;
    out.shape[1]         = columns;
    out.padded_shape[0]  = rows;
    out.padded_shape[1]  = static_cast<std::int32_t>(geometry.padded_columns);
    return out;
}

// The Q4G32AM flavour: base pointer + shape only. The object is not a W8 plane pair, so the
// W8 size validation and the scale-plane split do not apply; readers derive the Q4 planes from
// the geometry (ops::q4_bank_planes).
Weight host_q4_weight(const HostObject& object, std::int32_t rows, std::int32_t columns) {
    Weight out{};
    out.payload         = static_cast<const std::byte*>(object.device);
    out.payload_bytes   = object.bytes;
    out.qtype           = QType::W8G32_F16S; // metadata only; see above
    out.layout          = QuantLayout::RowSplit;
    out.group_size      = 32;
    out.qdata           = static_cast<const std::byte*>(object.device);
    out.qhigh           = nullptr;
    out.scales          = nullptr;
    out.n               = rows;
    out.k               = columns;
    out.group           = 32;
    out.scale_dtype     = DType::FP16;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

SparseMoePayload load_moe(const artifact::MaterializedArtifact& backing, const HostBank& bank,
                          const MoePlan& plan, ops::HyperConnectionWeights mix, bool q4) {
    SparseMoePayload out;
    out.host_bank_q4          = q4;
    out.op.router_shared_gate = artifact::materialized_weight(
        backing, plan.router_shared_gate, NumericFormat::BF16,
        static_cast<std::int32_t>(kExperts + 1), static_cast<std::int32_t>(kHidden));
    if (q4) {
        // The Q4 objects carry their own plane layout; the routed Weights hold only the
        // mapped base pointer and shape metadata (the slot cache is mandatory, so no kernel
        // ever reads them as W8 planes — expert_slot_weights swaps in the pool's).
        out.op.routed_gate_up = host_q4_weight(bank.object(plan.routed_gate_up),
                                               static_cast<std::int32_t>(kExperts * 2 * kFfn),
                                               static_cast<std::int32_t>(kHidden));
        out.op.routed_down    = host_q4_weight(bank.object(plan.routed_down),
                                               static_cast<std::int32_t>(kExperts * kHidden),
                                               static_cast<std::int32_t>(kFfn));
    } else {
        out.op.routed_gate_up = host_w8_weight(bank.object(plan.routed_gate_up),
                                               static_cast<std::int32_t>(kExperts * 2 * kFfn),
                                               static_cast<std::int32_t>(kHidden));
        out.op.routed_down    = host_w8_weight(bank.object(plan.routed_down),
                                               static_cast<std::int32_t>(kExperts * kHidden),
                                               static_cast<std::int32_t>(kFfn));
    }
    out.op.shared_gate_up = artifact::materialized_weight(
        backing, plan.shared_gate_up, NumericFormat::W8G32_F16S, static_cast<std::int32_t>(2 * kFfn),
        static_cast<std::int32_t>(kHidden));
    out.op.shared_down = artifact::materialized_weight(backing, plan.shared_down,
                                                       NumericFormat::W8G32_F16S,
                                                       static_cast<std::int32_t>(kHidden),
                                                       static_cast<std::int32_t>(kFfn));
    out.op.experts_per_token = TextConfig::experts_per_token;
    out.mix                  = std::move(mix);
    out.host_gate_up = static_cast<const std::byte*>(bank.object(plan.routed_gate_up).host);
    out.host_down    = static_cast<const std::byte*>(bank.object(plan.routed_down).host);
    return out;
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, family::StartupFeatures features,
                               int stage_first, int stage_last, bool host_bank_q4) {
    const bool staged = stage_last > 0;
    ArtifactLoadPlan load_plan;
    BindingPlan& out    = load_plan.bindings;
    // The checkpoint's own dimensions, where it states them: absent members keep the
    // target's compiled value, so an artifact written before the member existed binds
    // exactly as it did.
    out.geometry        = family::TextGeometry::declared<TextConfig>(binder.reader().geometry());
    out.frontend        = family::bind_frontend_resources(binder);
    out.features        = features;
    out.host_bank_q4    = host_bank_q4;
    out.token_embedding = device(binder, "text/token_embedding", NumericFormat::W8G32_F16S,
                                 {kVocab, kHidden});

    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.resident = !staged || (static_cast<int>(layer) >= stage_first &&
                                      static_cast<int>(layer) < stage_last);
        g_layer_placement = target.resident ? TensorPlacement::Device : TensorPlacement::ValidateOnly;
        target.hc_attention      = bind_hc(binder, prefix + "hc_attn/", true);
        target.hc_mlp            = bind_hc(binder, prefix + "hc_ffn/", true);
        target.is_full_attention = TextConfig::is_full_attention(static_cast<int>(layer));
        if (target.is_full_attention) {
            target.attention.query_key_gate_value =
                device(binder, prefix + "attention/query_key_gate_value",
                       NumericFormat::W8G32_F16S, {TextConfig::query_projection_rows, kHidden});
            target.attention.query_norm = device(binder, prefix + "attention/query_norm",
                                                 NumericFormat::BF16, {TextConfig::head_dim});
            target.attention.key_norm   = device(binder, prefix + "attention/key_norm",
                                                 NumericFormat::BF16, {TextConfig::head_dim});
            target.attention.output = device(binder, prefix + "attention/output",
                                             NumericFormat::W8G32_F16S,
                                             {kHidden, TextConfig::query_size});
            // The QSA indexer arrives with the artifact; dense attention serves the exact
            // context range for now, so its weights are validated but not resident.
            // QSA indexer (design/INFERENCE.md, phase 4): resident on the layers this
            // program runs; the selection only engages past `dense_exact_context`.
            target.attention.indexer.query =
                device(binder, prefix + "attention/indexer/query", NumericFormat::BF16,
                       {static_cast<std::uint64_t>(TextConfig::indexer_heads) *
                            TextConfig::indexer_head_dim,
                        kHidden});
            target.attention.indexer.key = device(binder, prefix + "attention/indexer/key",
                                        NumericFormat::BF16, {TextConfig::indexer_head_dim, kHidden});
            target.attention.indexer.query_norm =
                device(binder, prefix + "attention/indexer/query_norm", NumericFormat::BF16,
                       {TextConfig::indexer_head_dim});
            target.attention.indexer.key_norm =
                device(binder, prefix + "attention/indexer/key_norm", NumericFormat::BF16,
                       {TextConfig::indexer_head_dim});
        } else {
            target.gdn.a_log   = device(binder, prefix + "gdn/a_log", NumericFormat::FP32,
                                        {TextConfig::gdn_value_heads});
            target.gdn.dt_bias = device(binder, prefix + "gdn/dt_bias", NumericFormat::FP32,
                                        {TextConfig::gdn_value_heads});
            target.gdn.convolution =
                device(binder, prefix + "gdn/convolution", NumericFormat::BF16,
                       {TextConfig::gdn_conv_kernel, TextConfig::convolution_dim});
            target.gdn.a_b_projection =
                device(binder, prefix + "gdn/a_b_projection", NumericFormat::BF16,
                       {2 * TextConfig::gdn_value_heads, kHidden});
            target.gdn.query_key_value_z =
                device(binder, prefix + "gdn/query_key_value_z", NumericFormat::W8G32_F16S,
                       {TextConfig::gdn_projection_rows, kHidden});
            target.gdn.norm   = device(binder, prefix + "gdn/norm", NumericFormat::BF16,
                                       {TextConfig::gdn_value_head_dim});
            target.gdn.output = device(binder, prefix + "gdn/output", NumericFormat::W8G32_F16S,
                                       {kHidden, kValueDim});
        }
        target.has_ple = static_cast<int>(layer) == TextConfig::ple_layer;
        if (target.has_ple) {
            target.ple.key   = device(binder, prefix + "ple/key", NumericFormat::BF16,
                                      {kHcWidth, TextConfig::ple_embed});
            target.ple.value = device(binder, prefix + "ple/value", NumericFormat::BF16,
                                      {kHidden, TextConfig::ple_embed});
            target.ple.norm_key =
                device(binder, prefix + "ple/norm_key", NumericFormat::FP32, {kHcWidth});
            target.ple.norm_query =
                device(binder, prefix + "ple/norm_query", NumericFormat::FP32, {kHcWidth});
            target.ple.norm_conv =
                device(binder, prefix + "ple/norm_conv", NumericFormat::FP32, {kHcWidth});
            target.ple.convolution = device(binder, prefix + "ple/convolution", NumericFormat::BF16,
                                            {TextConfig::ple_conv_kernel, kHcWidth});
        }
        target.moe = bind_moe(binder, out.host_bank, prefix + "mlp/", host_bank_q4);
    }
    g_layer_placement = TensorPlacement::Device;

    out.output_mix  = bind_hc(binder, "text/output_hc/", false);
    out.output_head = device(binder, "text/output_head", NumericFormat::W8G32_F16S, {kVocab, kHidden});

    const artifact::ObjectHandle multipliers =
        artifact::bind_tensor(binder, "text/ple/multipliers", NumericFormat::I32,
                              {2 * TextConfig::ple_ngram}, TensorPlacement::ValidateOnly);
    const artifact::ObjectHandle offsets =
        artifact::bind_tensor(binder, "text/ple/head_offsets", NumericFormat::I32,
                              {TextConfig::ple_heads}, TensorPlacement::ValidateOnly);
    const artifact::ObjectHandle vocab_sizes =
        artifact::bind_tensor(binder, "text/ple/head_vocab_sizes", NumericFormat::I32,
                              {TextConfig::ple_heads}, TensorPlacement::ValidateOnly);
    std::array<std::uint32_t, 2 * TextConfig::ple_ngram> halves{};
    read_i32_array(binder, multipliers, halves);
    for (int i = 0; i < TextConfig::ple_ngram; ++i) {
        out.ple_multipliers[static_cast<std::size_t>(i)] =
            static_cast<std::uint64_t>(halves[static_cast<std::size_t>(2 * i)]) |
            (static_cast<std::uint64_t>(halves[static_cast<std::size_t>(2 * i + 1)]) << 32U);
    }
    read_i32_array(binder, offsets, out.ple_head_offsets);
    read_i32_array(binder, vocab_sizes, out.ple_head_vocab_sizes);

    out.ple_table = binder.require_resource("text/ple/table.iq4nl",
                                            artifact::ResourceEncoding::RawBytesV1);
    binder.validate_only(out.ple_table);
    {
        const auto payload = binder.payload(out.ple_table).data;
        const std::uint64_t expected = static_cast<std::uint64_t>(TextConfig::ple_table_rows) *
                                       TextConfig::ple_table_row_bytes;
        if (payload.size() != expected) {
            throw artifact::ArtifactError("PLE table has " + std::to_string(payload.size()) +
                                          " bytes, expected " + std::to_string(expected));
        }
        out.host_bank.objects.push_back({out.ple_table, payload, "text/ple/table.iq4nl"});
    }

    // Flash-Next's checkpoint carries a vision tower, so the artifact may hold it and
    // this binder must consume it — the loader refuses any object no binder claims.
    // Its geometry is the family default (27 layers of 1152, head_dim 72), which is
    // what the vision kernels implement, so nothing here is target-specific.
    // Whether an artifact carries the tower is a property of its source, not of the
    // model: the community GGUF exports of this family drop vision entirely. Probe
    // once and bind only what is there, so a text-only artifact loads; asking for
    // --vision without one is the error, not the artifact's existence.
    out.has_vision = binder.has("vision/patch_embedding");
    if (!out.has_vision && features.vision) {
        throw std::runtime_error(
            "flash-next: --vision was requested but this artifact carries no vision tower "
            "(it was converted from a source that has none)");
    }
    if (out.has_vision) {
        const artifact::TensorPlacement vision_placement =
            features.vision ? artifact::TensorPlacement::Device
                            : artifact::TensorPlacement::ValidateOnly;
        out.vision_backbone =
            family::bind_vision_backbone<family::VisionBackboneConfig>(binder, vision_placement);
        out.vision_merger_input =
            family::bind_vision_merger_input<family::VisionBackboneConfig>(binder, vision_placement);
        out.vision_merger_fc2 = artifact::bind_tensor(
            binder, "vision/merger/fc2", artifact::NumericFormat::W8G32_F16S,
            {TextConfig::hidden, family::VisionBackboneConfig::merger_hidden}, vision_placement);
        out.vision_merger_fc2_bias =
            artifact::bind_tensor(binder, "vision/merger/fc2_bias", artifact::NumericFormat::BF16,
                                  {TextConfig::hidden}, vision_placement);
        out.vision_merger_norm =
            family::bind_vision_merger_norm<family::VisionBackboneConfig>(binder, vision_placement);
    }

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)), host_bank(HostBank::shared(plan.host_bank)) {
    // The layer storage is sized here, not by the type: the counts come from the
    // geometry these weights were bound against.
    runtime.geometry = plan.geometry;
    runtime.full_layers.resize(kFullAttentionLayers);
    runtime.gdn_layers.resize(kGdnLayers);
    frontend = family::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena   = &backing.device_arena();
    runtime.features        = plan.features;
    runtime.token_embedding = artifact::materialized_weight(
        backing, plan.token_embedding, NumericFormat::W8G32_F16S, static_cast<std::int32_t>(kVocab),
        static_cast<std::int32_t>(kHidden));

    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (!source.resident) { // another stage's layer: keep the index bookkeeping only
            if (source.is_full_attention) { ++full_index; } else { ++gdn_index; }
            continue;
        }
        ops::HyperConnectionWeights mix_attn = load_hc(backing, source.hc_attention, true);
        ops::HyperConnectionWeights mix_mlp  = load_hc(backing, source.hc_mlp, true);
        if (source.is_full_attention) {
            FullAttentionWeights& target = runtime.full_layers.at(full_index++);
            target.projection.query_key_gate_value = artifact::materialized_weight(
                backing, source.attention.query_key_gate_value, NumericFormat::W8G32_F16S,
                TextConfig::query_projection_rows, static_cast<std::int32_t>(kHidden));
            target.projection.mix = std::move(mix_attn);
            target.query_norm     = artifact::materialized_tensor(
                backing, source.attention.query_norm, NumericFormat::BF16, {TextConfig::head_dim});
            target.key_norm = artifact::materialized_tensor(
                backing, source.attention.key_norm, NumericFormat::BF16, {TextConfig::head_dim});
            target.projection.indexer.query = artifact::materialized_weight(
                backing, source.attention.indexer.query, NumericFormat::BF16,
                static_cast<std::int32_t>(TextConfig::indexer_heads) * TextConfig::indexer_head_dim,
                static_cast<std::int32_t>(kHidden));
            target.projection.indexer.key = artifact::materialized_weight(
                backing, source.attention.indexer.key, NumericFormat::BF16,
                TextConfig::indexer_head_dim, static_cast<std::int32_t>(kHidden));
            target.projection.indexer.query_norm =
                artifact::materialized_tensor(backing, source.attention.indexer.query_norm,
                                              NumericFormat::BF16, {TextConfig::indexer_head_dim});
            target.projection.indexer.key_norm =
                artifact::materialized_tensor(backing, source.attention.indexer.key_norm,
                                              NumericFormat::BF16, {TextConfig::indexer_head_dim});
            target.output = artifact::materialized_weight(
                backing, source.attention.output, NumericFormat::W8G32_F16S,
                static_cast<std::int32_t>(kHidden), TextConfig::query_size);
            target.post_mixer = load_moe(backing, *host_bank, source.moe, std::move(mix_mlp),
                                         plan.host_bank_q4);
            target.post_mixer.layer = static_cast<std::int32_t>(layer);
        } else {
            GdnWeights& target = runtime.gdn_layers.at(gdn_index++);
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32,
                                              {TextConfig::gdn_value_heads});
            target.projection.dt_bias =
                artifact::materialized_tensor(backing, source.gdn.dt_bias, NumericFormat::FP32,
                                              {TextConfig::gdn_value_heads});
            target.convolution = artifact::materialized_tensor(
                backing, source.gdn.convolution, NumericFormat::BF16,
                {TextConfig::convolution_dim, TextConfig::gdn_conv_kernel});
            target.projection.a_b_projection = artifact::materialized_weight(
                backing, source.gdn.a_b_projection, NumericFormat::BF16,
                2 * TextConfig::gdn_value_heads, static_cast<std::int32_t>(kHidden));
            target.projection.query_key_value_z = artifact::materialized_weight(
                backing, source.gdn.query_key_value_z, NumericFormat::W8G32_F16S,
                TextConfig::gdn_projection_rows, static_cast<std::int32_t>(kHidden));
            target.projection.mix = std::move(mix_attn);
            target.norm = artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16,
                                                        {TextConfig::gdn_value_head_dim});
            target.output = artifact::materialized_weight(
                backing, source.gdn.output, NumericFormat::W8G32_F16S,
                static_cast<std::int32_t>(kHidden), static_cast<std::int32_t>(kValueDim));
            target.post_mixer = load_moe(backing, *host_bank, source.moe, std::move(mix_mlp),
                                         plan.host_bank_q4);
            target.post_mixer.layer = static_cast<std::int32_t>(layer);
        }
        if (source.has_ple) {
            PleWeights& ple = runtime.ple;
            ple.layer       = static_cast<int>(layer);
            ple.op.key      = artifact::materialized_weight(backing, source.ple.key,
                                                            NumericFormat::BF16,
                                                            static_cast<std::int32_t>(kHcWidth),
                                                            TextConfig::ple_embed);
            ple.op.value    = artifact::materialized_weight(backing, source.ple.value,
                                                            NumericFormat::BF16,
                                                            static_cast<std::int32_t>(kHidden),
                                                            TextConfig::ple_embed);
            ple.op.norm_key = artifact::materialized_tensor(backing, source.ple.norm_key,
                                                            NumericFormat::FP32,
                                                            {static_cast<std::int32_t>(kHcWidth)});
            ple.op.norm_query = artifact::materialized_tensor(
                backing, source.ple.norm_query, NumericFormat::FP32,
                {static_cast<std::int32_t>(kHcWidth)});
            ple.op.norm_conv = artifact::materialized_tensor(
                backing, source.ple.norm_conv, NumericFormat::FP32,
                {static_cast<std::int32_t>(kHcWidth)});
            ple.op.convolution = artifact::materialized_tensor(
                backing, source.ple.convolution, NumericFormat::BF16,
                {static_cast<std::int32_t>(kHcWidth), TextConfig::ple_conv_kernel});
        }
    }
    if (full_index != runtime.full_layers.size() || gdn_index != runtime.gdn_layers.size()) {
        throw std::logic_error("qwen4exp text topology binding is incomplete");
    }

    runtime.output_mix  = load_hc(backing, plan.output_mix, false);
    runtime.output_head = artifact::materialized_weight(backing, plan.output_head,
                                                        NumericFormat::W8G32_F16S,
                                                        static_cast<std::int32_t>(kVocab),
                                                        static_cast<std::int32_t>(kHidden));

    PleWeights& ple = runtime.ple;
    ple.hash.ngram  = TextConfig::ple_ngram;
    ple.hash.heads  = TextConfig::ple_heads;
    ple.hash.eos_token = TextConfig::eos_token;
    for (int i = 0; i < TextConfig::ple_ngram; ++i) {
        ple.hash.multipliers[i] = plan.ple_multipliers[static_cast<std::size_t>(i)];
    }
    for (int h = 0; h < TextConfig::ple_heads; ++h) {
        ple.hash.head_offsets[h]     = plan.ple_head_offsets[static_cast<std::size_t>(h)];
        ple.hash.head_vocab_sizes[h] = plan.ple_head_vocab_sizes[static_cast<std::size_t>(h)];
    }
    const HostObject& table = host_bank->object(plan.ple_table);
    ple.table.rows          = table.device;
    ple.table.row_count     = TextConfig::ple_table_rows;
    ple.table.row_bytes     = TextConfig::ple_table_row_bytes;
    ple.table.head_dim      = TextConfig::ple_head_dim;
}

} // namespace sinfer::targets::qwen4exp::detail
