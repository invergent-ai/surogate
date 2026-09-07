#include "targets/qwen4exp/impl/load/bindings.h"
#include "ops/linear/ggml/ggml_dispatch.h"

#include "artifact/reader.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
    bank.objects.push_back(host_plan(binder, handle, name));
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

/// One full-attention block's weights. The MTP draft head binds the same set: its block is
/// structurally a trunk block, so it reads the same objects under its own prefix.
void bind_full_attention(artifact::Binder& binder, const std::string& prefix,
                         FullAttentionPlan& out) {
    out.query_key_gate_value =
        device(binder, prefix + "query_key_gate_value", NumericFormat::W8G32_F16S,
               {TextConfig::query_projection_rows, kHidden});
    out.query_norm = device(binder, prefix + "query_norm", NumericFormat::BF16,
                            {TextConfig::head_dim});
    out.key_norm   = device(binder, prefix + "key_norm", NumericFormat::BF16,
                            {TextConfig::head_dim});
    out.output     = device(binder, prefix + "output", NumericFormat::W8G32_F16S,
                            {kHidden, TextConfig::query_size});
    // QSA indexer (design/INFERENCE.md, phase 4): resident on the layers this program runs;
    // the selection only engages past `dense_exact_context`.
    out.indexer.query =
        device(binder, prefix + "indexer/query", NumericFormat::BF16,
               {static_cast<std::uint64_t>(TextConfig::indexer_heads) * TextConfig::indexer_head_dim,
                kHidden});
    out.indexer.key = device(binder, prefix + "indexer/key", NumericFormat::BF16,
                             {TextConfig::indexer_head_dim, kHidden});
    out.indexer.query_norm = device(binder, prefix + "indexer/query_norm", NumericFormat::BF16,
                                    {TextConfig::indexer_head_dim});
    out.indexer.key_norm   = device(binder, prefix + "indexer/key_norm", NumericFormat::BF16,
                                    {TextConfig::indexer_head_dim});
}

MoePlan bind_moe(artifact::Binder& binder, HostBankPlan& bank, const std::string& prefix,
                 family::BankPlanes planes, bool ggml_artifact) {
    // What the bank presents for each routed half. A converted artifact already stores W8
    // row-split planes, and only an explicit `Q4` asks the bank to requantise them. A
    // GGUF-native one names the file's blocks, and the bank either keeps them as they lie (the
    // gather decodes each group on its way into the pool) or turns them into planes on the way
    // into pinned memory -- which planes is per object under `Auto`, so a half stored 4-bit
    // affine becomes Q4G32AM by an exact repack and a wider one becomes W8.
    NumericFormat gate_up_format = NumericFormat::W8G32_F16S;
    NumericFormat down_format    = NumericFormat::W8G32_F16S;
    bool gate_up_q4 = false;
    bool down_q4    = false;
    const auto routed = [&](const std::string& name, std::initializer_list<std::uint64_t> shape,
                            NumericFormat& format, bool& q4) {
        const auto rows    = static_cast<std::int32_t>(*shape.begin());
        const auto columns = static_cast<std::int32_t>(*(shape.begin() + 1));
        if (planes == family::BankPlanes::Q4 && !ggml_artifact) {
            // The requantiser reads W8 planes, which is what a converted artifact stores.
            q4 = true;
            return host_q4(binder, bank, name, NumericFormat::W8G32_F16S, shape);
        }
        const artifact::LinearBinding binding = host_linear(binder, bank, name, rows, columns);
        format = binding.format;
        const family::BankPlanes chosen = family::bank_as_planes(
            bank.objects.back(), rows, columns, artifact::qtype_for(binding.format), planes);
        if (chosen != family::BankPlanes::Native) {
            q4     = chosen == family::BankPlanes::Q4;
            format = NumericFormat::W8G32_F16S; // what the bank presents to the runtime
        }
        return binding.object;
    };
    return MoePlan{
        .router_shared_gate = device(binder, prefix + "router_shared_gate", NumericFormat::BF16,
                                     {kExperts + 1, kHidden}),
        .routed_gate_up = routed(prefix + "routed_gate_up", {kExperts * 2 * kFfn, kHidden},
                                 gate_up_format, gate_up_q4),
        .routed_down    = routed(prefix + "routed_down", {kExperts * kHidden, kFfn},
                                 down_format, down_q4),
        .routed_gate_up_format = gate_up_format,
        .routed_down_format    = down_format,
        .routed_gate_up_q4     = gate_up_q4,
        .routed_down_q4        = down_q4,
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


SparseMoePayload load_moe(const artifact::MaterializedArtifact& backing, const HostBank& bank,
                          const MoePlan& plan, ops::HyperConnectionWeights mix) {
    SparseMoePayload out;
    out.host_gate_up_q4       = plan.routed_gate_up_q4;
    out.host_down_q4          = plan.routed_down_q4;
    out.op.router_shared_gate = artifact::materialized_weight(
        backing, plan.router_shared_gate, NumericFormat::BF16,
        static_cast<std::int32_t>(kExperts + 1), static_cast<std::int32_t>(kHidden));
    // Each half as the bank made it. A Q4G32AM object carries its own plane layout, so its
    // Weight holds only the mapped base pointer and shape metadata (the slot cache is then
    // mandatory: no kernel ever reads it as W8 planes -- expert_slot_weights swaps in the
    // pool's). Blocks the bank kept are read by the gather's codec; W8 planes are the pool's
    // own layout, copied straight through.
    const auto routed = [&](artifact::ObjectHandle handle, bool q4, NumericFormat format,
                            std::int32_t rows, std::int32_t columns) {
        const HostObject& object = bank.object(handle);
        if (q4) { return family::host_q4_weight(object, rows, columns); }
        if (format != NumericFormat::W8G32_F16S) {
            return host_ggml_weight(object, format, rows, columns);
        }
        return host_w8_weight(object, rows, columns);
    };
    out.op.routed_gate_up = routed(plan.routed_gate_up, plan.routed_gate_up_q4,
                                   plan.routed_gate_up_format,
                                   static_cast<std::int32_t>(kExperts * 2 * kFfn),
                                   static_cast<std::int32_t>(kHidden));
    out.op.routed_down    = routed(plan.routed_down, plan.routed_down_q4, plan.routed_down_format,
                                   static_cast<std::int32_t>(kExperts * kHidden),
                                   static_cast<std::int32_t>(kFfn));
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
                               int stage_first, int stage_last, family::BankPlanes bank_planes,
                               LoadProgress progress) {
    const bool staged = stage_last > 0;
    ArtifactLoadPlan load_plan;
    BindingPlan& out    = load_plan.bindings;
    out.host_bank.progress = std::move(progress);
    // The checkpoint's own dimensions, where it states them: absent members keep the
    // target's compiled value, so an artifact written before the member existed binds
    // exactly as it did.
    out.geometry        = family::TextGeometry::declared<TextConfig>(binder.reader().geometry());
    out.frontend        = family::bind_frontend_resources(binder);
    out.features        = features;
    // Whether the artifact stores the GGUF's own blocks or W8 row-split planes: the two take
    // different routes into the bank, and only the first can become Q4G32AM without a
    // requantisation.
    const auto* routed0 = binder.reader().find("text/layers/0/mlp/routed_gate_up");
    const auto* routed0_tensor =
        routed0 != nullptr ? std::get_if<artifact::TensorDescriptor>(routed0) : nullptr;
    const bool routed_is_ggml =
        routed0_tensor != nullptr && routed0_tensor->layout == artifact::StorageLayout::GgmlBlocksV1;
    // A GGUF-native artifact's experts leave the file's block format as they enter the bank --
    // the same bank a converted artifact builds, so the gather is a copy and the host expert
    // path reads planes at memory speed. The blocks stay blocks only under
    // SUROGATE_SERVE_HOST_BANK_NATIVE=1, the A/B switch, where every miss decodes on its way
    // to the device.
    const bool keep_native = std::getenv("SUROGATE_SERVE_HOST_BANK_NATIVE") != nullptr;
    const family::BankPlanes planes =
        (routed_is_ggml && keep_native) ? family::BankPlanes::Native : bank_planes;
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
            bind_full_attention(binder, prefix + "attention/", target.attention);
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
        target.moe = bind_moe(binder, out.host_bank, prefix + "mlp/", planes, routed_is_ggml);
    }
    g_layer_placement = TensorPlacement::Device;

    out.output_mix  = bind_hc(binder, "text/output_hc/", false);
    out.output_head = device(binder, "text/output_head", NumericFormat::W8G32_F16S, {kVocab, kHidden});

    // The NextN draft head. Bound whenever the artifact carries one, because every object an
    // artifact holds has to be consumed by the target that reads it -- but resident only when
    // the run asked for a draft head, since it is 2.75 GB of weights that an ordinary run
    // never touches.
    out.mtp.present = binder.has("mtp/input_projection");
    if (out.mtp.present) {
        MtpPlan& mtp      = out.mtp;
        mtp.resident      = features.mtp();
        g_layer_placement = mtp.resident ? TensorPlacement::Device : TensorPlacement::ValidateOnly;
        mtp.embedding_norm = device(binder, "mtp/embedding_norm", NumericFormat::BF16, {kHidden});
        mtp.hidden_norm    = device(binder, "mtp/hidden_norm", NumericFormat::FP32, {kHcWidth});
        // fc_embedding and fc_hidden fused side by side, so one matmul over
        // concat(embedding_norm(e), hidden_norm(h)) is fc_embedding@e + fc_hidden@h.
        mtp.input_projection = device(binder, "mtp/input_projection", NumericFormat::W8G32_F16S,
                                      {kHidden, 2 * kHidden});
        mtp.layer.is_full_attention = true;
        mtp.layer.hc_attention      = bind_hc(binder, "mtp/layer/hc_attn/", true);
        bind_full_attention(binder, "mtp/layer/attention/", mtp.layer.attention);
        mtp.layer.hc_mlp = bind_hc(binder, "mtp/layer/hc_ffn/", true);
        mtp.layer.moe    = bind_moe(binder, out.host_bank, "mtp/layer/mlp/", planes, routed_is_ggml);
        mtp.head_mix     = bind_hc(binder, "mtp/head_hc/", false);
        g_layer_placement = TensorPlacement::Device;
    }

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

    // A tensor rather than a raw resource: as IQ4_NL blocks it is describable, so the artifact
    // points at the GGUF's 28.8 GB instead of copying them.
    out.ple_table = artifact::bind_tensor(
        binder, "text/ple/table.iq4nl", NumericFormat::IQ4_NL,
        {static_cast<std::uint64_t>(TextConfig::ple_table_rows), 160},
        TensorPlacement::ValidateOnly);
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

/// One full-attention block's weights, out of the artifact and into the runtime view. The
/// NextN draft head loads through here too: its block is a trunk block, so it is the same
/// statements over the same objects.
void load_full_attention(artifact::MaterializedArtifact& backing, const FullAttentionPlan& attention,
                         ops::HyperConnectionWeights mix_attn, FullAttentionWeights& target) {
    target.projection.mix                  = std::move(mix_attn);
    target.projection.query_key_gate_value = artifact::materialized_weight(
        backing, attention.query_key_gate_value, NumericFormat::W8G32_F16S,
        TextConfig::query_projection_rows, static_cast<std::int32_t>(kHidden));
    target.query_norm = artifact::materialized_tensor(backing, attention.query_norm,
                                                      NumericFormat::BF16, {TextConfig::head_dim});
    target.key_norm   = artifact::materialized_tensor(backing, attention.key_norm,
                                                      NumericFormat::BF16, {TextConfig::head_dim});
    target.projection.indexer.query = artifact::materialized_weight(
        backing, attention.indexer.query, NumericFormat::BF16,
        static_cast<std::int32_t>(TextConfig::indexer_heads) * TextConfig::indexer_head_dim,
        static_cast<std::int32_t>(kHidden));
    target.projection.indexer.key = artifact::materialized_weight(
        backing, attention.indexer.key, NumericFormat::BF16, TextConfig::indexer_head_dim,
        static_cast<std::int32_t>(kHidden));
    target.projection.indexer.query_norm = artifact::materialized_tensor(
        backing, attention.indexer.query_norm, NumericFormat::BF16, {TextConfig::indexer_head_dim});
    target.projection.indexer.key_norm = artifact::materialized_tensor(
        backing, attention.indexer.key_norm, NumericFormat::BF16, {TextConfig::indexer_head_dim});
    target.output = artifact::materialized_weight(backing, attention.output,
                                                  NumericFormat::W8G32_F16S,
                                                  static_cast<std::int32_t>(kHidden),
                                                  TextConfig::query_size);
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
            load_full_attention(backing, source.attention, std::move(mix_attn), target);
            target.post_mixer = load_moe(backing, *host_bank, source.moe, std::move(mix_mlp));
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
            target.post_mixer = load_moe(backing, *host_bank, source.moe, std::move(mix_mlp));
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

    // The NextN draft head. Its block loads through the trunk's own loader, because it is a
    // trunk block; only the fold on the way in and the mixer on the way out are its own.
    runtime.mtp_head.present = plan.mtp.present && plan.mtp.resident;
    if (runtime.mtp_head.present) {
        MtpHeadWeights& head = runtime.mtp_head;
        head.embedding_norm  = artifact::materialized_tensor(
            backing, plan.mtp.embedding_norm, NumericFormat::BF16,
            {static_cast<std::int32_t>(kHidden)});
        head.hidden_norm = artifact::materialized_tensor(
            backing, plan.mtp.hidden_norm, NumericFormat::FP32,
            {static_cast<std::int32_t>(kHcWidth)});
        head.input_projection = artifact::materialized_weight(
            backing, plan.mtp.input_projection, NumericFormat::W8G32_F16S,
            static_cast<std::int32_t>(kHidden), static_cast<std::int32_t>(2 * kHidden));
        head.head_mix = load_hc(backing, plan.mtp.head_mix, false);
        load_full_attention(backing, plan.mtp.layer.attention,
                            load_hc(backing, plan.mtp.layer.hc_attention, true),
                            runtime.mtp_block);
        runtime.mtp_block.post_mixer =
            load_moe(backing, *host_bank, plan.mtp.layer.moe,
                     load_hc(backing, plan.mtp.layer.hc_mlp, true));
        // The head's own expert bank, past the trunk's layers: the slot cache keys on this.
        runtime.mtp_block.post_mixer.layer = static_cast<std::int32_t>(kTextLayers);
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
