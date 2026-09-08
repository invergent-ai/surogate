#include "targets/glm5_next/impl/load/bindings.h"

#include "targets/glm5_next/impl/config.h"

#include "artifact/reader.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace sinfer::targets::glm5_next::detail {
namespace {

using artifact::NumericFormat;



/// Placement of the layer being bound: Device for the layers this stage runs, ValidateOnly for
/// another stage's -- validated, never uploaded. A 200 GB checkpoint does not fit on one card,
/// so this is what makes eight of them enough.
thread_local artifact::TensorPlacement g_layer_placement = artifact::TensorPlacement::Device;

/// Placement of the routed experts of the layer being bound. `HostBank` keeps their bytes in
/// pinned host memory and hands the kernels a mapped alias; everything downstream is unchanged,
/// because an object's pointer is an object's pointer.
thread_local artifact::TensorPlacement g_expert_placement = artifact::TensorPlacement::Device;

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return NumericFormat::W8G32_F16S;
    }
    throw std::invalid_argument("glm5_next: invalid weights profile");
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat,
                       std::initializer_list<std::uint64_t> shape) {
    if (shape.size() != 2) { throw std::logic_error("bind_weight: rank-two shape"); }
    const auto dims = std::vector<std::uint64_t>(shape);
    const artifact::LinearBinding binding =
        artifact::bind_linear(binder, name, static_cast<std::int32_t>(dims[0]),
                              static_cast<std::int32_t>(dims[1]), g_layer_placement);
    return WeightPlan{.object = binding.object, .format = binding.format};
}

artifact::ObjectHandle bind_layer_tensor(artifact::Binder& binder, const std::string& name,
                                         NumericFormat format,
                                         std::initializer_list<std::uint64_t> shape) {
    return artifact::bind_tensor(binder, name, format, shape, g_layer_placement);
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
}

[[nodiscard]] std::string layer_prefix(std::size_t layer) {
    return "text/layers/" + std::to_string(layer) + "/";
}

HyperConnectionPlan bind_hyper_connection(artifact::Binder& binder, const std::string& prefix,
                                          std::string_view site,
                                          const family::TextGeometry& g) {
    const std::string base = prefix + "hc/" + std::string(site);
    HyperConnectionPlan out;
    // BF16 rather than the endpoint format: these rows produce every mixing weight the residual
    // is recombined with, at both sites of every layer.
    out.mix   = bind_weight(binder, base + "_mix", NumericFormat::BF16,
                            {static_cast<std::uint64_t>(g.hyper_connection_mix_rows()),
                             static_cast<std::uint64_t>(g.residual)});
    out.base  = bind_layer_tensor(
        binder, base + "_base", NumericFormat::FP32,
        {static_cast<std::uint64_t>(g.hyper_connection_mix_rows())});
    out.scale = bind_layer_tensor(binder, base + "_scale", NumericFormat::FP32, {3});
    return out;
}

void bind_feed_forward(artifact::Binder& binder, const std::string& prefix, bool sparse,
                       NumericFormat weights, const family::TextGeometry& g,
                       FeedForwardPlan& out) {
    out.sparse = sparse;
    if (!sparse) {
        out.gate_up = bind_weight(binder, prefix + "mlp/gate_up", weights,
                                  {static_cast<std::uint64_t>(2 * g.dense_intermediate),
                                   static_cast<std::uint64_t>(g.hidden)});
        out.down    = bind_weight(binder, prefix + "mlp/down", weights,
                                  {static_cast<std::uint64_t>(g.hidden),
                                   static_cast<std::uint64_t>(g.dense_intermediate)});
        return;
    }
    const auto experts = static_cast<std::uint64_t>(moe_geometry(g).experts);
    // The router is BF16 and its bias FP32: a router that picks eight of 288 is the one place
    // in this model where a coarse width changes which experts run rather than by how much.
    out.router      = bind_weight(binder, prefix + "moe/router", NumericFormat::BF16,
                                  {static_cast<std::uint64_t>(moe_geometry(g).router_rows()),
                                   static_cast<std::uint64_t>(g.hidden)});
    out.router_bias = bind_layer_tensor(binder, prefix + "moe/router_bias", NumericFormat::FP32,
                                        {experts});
    // The experts are 99.5 % of a mixture layer's bytes, so they are the only thing worth
    // moving off the card -- and the only thing whose absence a round can absorb, because a
    // token touches eight of 288 while the router and the shared expert run on every one.
    {
        const artifact::TensorPlacement outer = g_layer_placement;
        if (outer == artifact::TensorPlacement::Device) { g_layer_placement = g_expert_placement; }
        out.routed_gate_up =
            bind_weight(binder, prefix + "moe/routed_gate_up", weights,
                        {experts * static_cast<std::uint64_t>(moe_geometry(g).expert_rows()),
                         static_cast<std::uint64_t>(g.hidden)});
        out.routed_down = bind_weight(binder, prefix + "moe/routed_down", weights,
                                      {experts * static_cast<std::uint64_t>(g.hidden),
                                       static_cast<std::uint64_t>(moe_geometry(g).intermediate)});
        g_layer_placement = outer;
    }
    if (g.shared_intermediate > 0) {
    out.shared_gate_up =
        bind_weight(binder, prefix + "moe/shared_gate_up", weights,
                    {static_cast<std::uint64_t>(moe_geometry(g).shared_rows()),
                     static_cast<std::uint64_t>(g.hidden)});
    out.shared_down = bind_weight(binder, prefix + "moe/shared_down", weights,
                                  {static_cast<std::uint64_t>(g.hidden),
                                   static_cast<std::uint64_t>(moe_geometry(g).shared_intermediate)});
    }
}

void bind_latent_attention(artifact::Binder& binder, const std::string& prefix,
                           NumericFormat weights, const family::TextGeometry& g,
                           LatentAttentionPlan& mla) {
    mla.query_a      = bind_weight(binder, prefix + "mla/query_a", weights,
                                   {static_cast<std::uint64_t>(g.q_lora_rank),
                                    static_cast<std::uint64_t>(g.hidden)});
    mla.query_a_norm = bind_layer_tensor(binder, prefix + "mla/query_a_norm", NumericFormat::BF16,
                                         {static_cast<std::uint64_t>(g.q_lora_rank)});
    mla.query_b      = bind_weight(binder, prefix + "mla/query_b", weights,
                                   {static_cast<std::uint64_t>(g.query_heads) *
                                        g.qk_head_dim,
                                    static_cast<std::uint64_t>(g.q_lora_rank)});
    mla.kv_a         = bind_weight(binder, prefix + "mla/kv_a", weights,
                                   {static_cast<std::uint64_t>(g.kv_lora_rank),
                                    static_cast<std::uint64_t>(g.hidden)});
    mla.kv_a_norm    = bind_layer_tensor(binder, prefix + "mla/kv_a_norm", NumericFormat::BF16,
                                         {static_cast<std::uint64_t>(g.kv_lora_rank)});
    // Both halves of the expansion are read where they lie. llama.cpp stores the key half in
    // the orientation it applies to the *query* -- [latent, nope] per head -- and the absorbed
    // form applies it to the query too, so what once had to be transposed into BF16 is now the
    // file's own Q8_0.
    mla.k_b    = bind_weight(binder, prefix + "mla/k_b", weights,
                             {static_cast<std::uint64_t>(g.latent_key_rows()),
                              static_cast<std::uint64_t>(g.qk_head_dim)});
    mla.v_b    = bind_weight(binder, prefix + "mla/v_b", weights,
                             {static_cast<std::uint64_t>(g.query_heads) * g.v_head_dim,
                              static_cast<std::uint64_t>(g.kv_lora_rank)});
    mla.output = bind_weight(binder, prefix + "mla/output", weights,
                             {static_cast<std::uint64_t>(g.hidden),
                              static_cast<std::uint64_t>(g.query_heads) * g.v_head_dim});
}

/// The NextN draft head, bound where the trunk's layers are and placed by whether the run asked
/// for it and whether this stage is where the logits are: a run without `--spec mtp`, or a
/// pipeline stage before the last, validates the head's shapes and uploads nothing. The head
/// keeps its experts beside it -- a token drafted every round would cross PCIe for them every
/// round -- so the trunk's expert offload does not reach it; and on a pipeline those experts
/// are a whole mixture layer's worth per card, which is why only the stage that runs the head
/// carries them (the other stages run the verify forward for their own layers and adopt the
/// head stage's decision).
void bind_mtp_head(artifact::Binder& binder, NumericFormat weights,
                   family::StartupFeatures features, bool holds_head, BindingPlan& out) {
    const family::TextGeometry& g = out.geometry;
    MtpPlan& mtp                  = out.mtp;
    mtp.present                   = g.mtp_layers > 0;
    if (!mtp.present) {
        if (features.mtp()) {
            throw std::runtime_error(
                "glm5_next: --spec mtp asks for the NextN draft head, and this artifact carries "
                "none (the checkpoint it was converted from is trunk-only, or the conversion "
                "predates the head); run without --spec");
        }
        return;
    }
    mtp.resident       = features.mtp() && holds_head;
    g_layer_placement  = mtp.resident ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    g_expert_placement = g_layer_placement;
    const std::string prefix = "mtp/";
    const std::string layer  = prefix + "layer/";
    mtp.embedding_norm   = bind_layer_tensor(binder, prefix + "embedding_norm", NumericFormat::BF16,
                                             {static_cast<std::uint64_t>(g.hidden)});
    mtp.hidden_norm      = bind_layer_tensor(binder, prefix + "hidden_norm", NumericFormat::BF16,
                                             {static_cast<std::uint64_t>(g.hidden)});
    mtp.input_projection = bind_weight(binder, prefix + "input_projection", weights,
                                       {static_cast<std::uint64_t>(g.hidden),
                                        static_cast<std::uint64_t>(g.mtp_input_rows())});
    mtp.input_norm       = bind_layer_tensor(binder, layer + "input_norm", NumericFormat::BF16,
                                             {static_cast<std::uint64_t>(g.hidden)});
    bind_latent_attention(binder, layer, weights, g, mtp.attention);
    mtp.post_attention_norm = bind_layer_tensor(binder, layer + "post_attention_norm",
                                                NumericFormat::BF16,
                                                {static_cast<std::uint64_t>(g.hidden)});
    const bool sparse = true;
    bind_feed_forward(binder, layer, sparse, weights, g, mtp.feed_forward);
    mtp.final_norm = bind_layer_tensor(binder, prefix + "final_norm", NumericFormat::BF16,
                                       {static_cast<std::uint64_t>(g.hidden)});
    g_layer_placement  = artifact::TensorPlacement::Device;
    g_expert_placement = artifact::TensorPlacement::Device;
}

void bind_text_layers(artifact::Binder& binder, WeightsProfile weights_profile, int stage_first,
                      int stage_last, std::uint32_t host_moe_layers, std::uint32_t gpu_layers,
                      BindingPlan& out) {
    const NumericFormat weights   = endpoint_format(weights_profile);
    const family::TextGeometry& g = out.geometry;
    out.text_layers.resize(static_cast<std::size_t>(g.layers));
    std::uint32_t stage_mixture_seen = 0;
    for (std::size_t layer = 0; layer < out.text_layers.size(); ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = layer_prefix(layer);
        target.attends           = g.layer_attends(static_cast<std::int32_t>(layer));
        const bool staged        = stage_last > 0;
        target.resident          = !staged || (static_cast<int>(layer) >= stage_first &&
                                      static_cast<int>(layer) < stage_last);
        // Three placements, and the order matters. A layer another stage runs is validated and
        // nothing else. A layer past `gpu_layers` is read from host memory in its entirety.
        // Otherwise it is resident, and `host_moe_layers` may still move its experts.
        //
        // `host_moe_layers` counts the mixture layers *this stage runs*, not the model's. On a
        // pipeline the stages are not equally tight -- the leading dense layers make the first
        // one light -- so a whole-model count offloads from whichever stage happens to hold the
        // low-numbered layers, which is the one that needed it least. Per stage, the pipeline
        // constructor can give each the amount it actually needs.
        const bool on_card = gpu_layers == 0 || layer < gpu_layers;
        g_layer_placement  = !target.resident ? artifact::TensorPlacement::ValidateOnly
                             : on_card       ? artifact::TensorPlacement::Device
                                             : artifact::TensorPlacement::HostBank;
        const bool sparse_layer = layer >= g.leading_dense_layers;
        const bool offload_experts =
            target.resident && sparse_layer && stage_mixture_seen < host_moe_layers;
        if (target.resident && sparse_layer) { ++stage_mixture_seen; }
        g_expert_placement = offload_experts ? artifact::TensorPlacement::HostBank
                                             : artifact::TensorPlacement::Device;

        target.attention_hc = bind_hyper_connection(binder, prefix, "attn", g);
        target.input_norm   = bind_layer_tensor(binder, prefix + "input_norm",
                                                NumericFormat::BF16,
                                                {static_cast<std::uint64_t>(g.hidden)});
        if (target.attends) {
            bind_latent_attention(binder, prefix, weights, g, target.attention);
        } else {
            KdaPlan& kda        = target.kda;
            kda.query_key_value = bind_weight(binder, prefix + "kda/query_key_value", weights,
                                              {static_cast<std::uint64_t>(g.convolution_dim()),
                                               static_cast<std::uint64_t>(g.hidden)});
            // Tap-major, [K, channels] as the artifact writes it, which is [channels, K] in the
            // engine's own layout.
            kda.convolution = bind_layer_tensor(
                binder, prefix + "kda/convolution", NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.gdn_conv_kernel),
                 static_cast<std::uint64_t>(g.convolution_dim())});
            kda.decay_a    = bind_weight(binder, prefix + "kda/decay_a", weights,
                                         {static_cast<std::uint64_t>(g.kda_gate_rank),
                                          static_cast<std::uint64_t>(g.hidden)});
            kda.decay_b    = bind_weight(binder, prefix + "kda/decay_b", weights,
                                         {static_cast<std::uint64_t>(g.value_dim()),
                                          static_cast<std::uint64_t>(g.kda_gate_rank)});
            kda.decay_bias = bind_layer_tensor(
                binder, prefix + "kda/decay_bias", NumericFormat::FP32,
                {static_cast<std::uint64_t>(g.value_dim())});
            kda.a_log      = bind_layer_tensor(
                binder, prefix + "kda/a_log", NumericFormat::FP32,
                {static_cast<std::uint64_t>(g.gdn_value_heads)});
            kda.beta       = bind_weight(binder, prefix + "kda/beta", weights,
                                         {static_cast<std::uint64_t>(g.gdn_value_heads),
                                          static_cast<std::uint64_t>(g.hidden)});
            kda.gate_a     = bind_weight(binder, prefix + "kda/gate_a", weights,
                                         {static_cast<std::uint64_t>(g.kda_gate_rank),
                                          static_cast<std::uint64_t>(g.hidden)});
            kda.gate_b     = bind_weight(binder, prefix + "kda/gate_b", weights,
                                         {static_cast<std::uint64_t>(g.value_dim()),
                                          static_cast<std::uint64_t>(g.kda_gate_rank)});
            kda.norm       = bind_layer_tensor(
                binder, prefix + "kda/norm", NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.gdn_value_head_dim)});
            kda.output     = bind_weight(binder, prefix + "kda/output", weights,
                                         {static_cast<std::uint64_t>(g.hidden),
                                          static_cast<std::uint64_t>(g.value_dim())});
        }
        target.feed_forward_hc     = bind_hyper_connection(binder, prefix, "ffn", g);
        target.post_attention_norm = bind_layer_tensor(binder, prefix + "post_attention_norm",
                                                       NumericFormat::BF16,
                                                       {static_cast<std::uint64_t>(g.hidden)});
        // Which layers are dense is the artifact's to say too, and it says it the same way:
        // a layer holding a router is a mixture layer.
        const bool sparse = layer >= g.leading_dense_layers;
        bind_feed_forward(binder, prefix, sparse, weights, g, target.feed_forward);
    }
    g_layer_placement = artifact::TensorPlacement::Device;
}

HyperConnectionPayload load_hyper_connection(const artifact::MaterializedArtifact& backing,
                                             const HyperConnectionPlan& plan,
                                             const family::TextGeometry& g, std::int32_t layer) {
    HyperConnectionPayload out;
    out.layer         = layer;
    out.probe_layer_count = g.layers;
    out.streams = g.hc_streams;
    out.sinkhorn_iterations = g.hc_sinkhorn_iterations;
    out.epsilon = g.hc_epsilon;
    out.weights.mix   = materialized_weight(backing, plan.mix, g.hyper_connection_mix_rows(),
                                            g.residual);
    out.weights.base  = artifact::materialized_tensor(
        backing, plan.base, NumericFormat::FP32,
        {static_cast<std::uint64_t>(g.hyper_connection_mix_rows())});
    out.weights.scale = artifact::materialized_tensor(backing, plan.scale, NumericFormat::FP32, {3});
    return out;
}

/// The feed-forward's weights, dense or the mixture, into a payload whose site (norm and
/// hyper-connection) the caller has already set. `bank` is the host bank when the stage has
/// one: a mixture whose routed experts it holds gets their host addresses, which is what lets
/// the expert cache compute them on the CPU rather than stream them over PCIe.
void load_feed_forward_weights(const artifact::MaterializedArtifact& backing,
                               const FeedForwardPlan& source, const family::TextGeometry& g,
                               const family::HostBank* bank, const family::HostBankPlan* bank_plan,
                               std::int32_t layer, FeedForwardPayload& out) {
    out.moe.swiglu_limit = g.swiglu_limit;
    out.sparse = source.sparse;
    out.layer  = layer;
    out.layers = g.layers + g.mtp_layers;
    if (!out.sparse) {
        out.gate_up = materialized_weight(backing, source.gate_up, 2 * g.dense_intermediate,
                                          g.hidden);
        out.down    = materialized_weight(backing, source.down, g.hidden, g.dense_intermediate);
        return;
    }
    out.moe.router_shared_gate = materialized_weight(backing, source.router,
                                                     moe_geometry(g).router_rows(), g.hidden);
    out.moe.router_bias        = static_cast<const float*>(
        artifact::materialized_tensor(backing, source.router_bias, NumericFormat::FP32,
                                      {static_cast<std::uint64_t>(moe_geometry(g).experts)})
            .data);
    // Resident, the weight is the artifact's. Banked, it is what the bank made of the object
    // as it filled: the file's blocks as they lie (the bank has given the artifact the mapped
    // pointer, so the same builder serves), W8 planes, or Q4G32AM planes -- the last two
    // presented over the bank's mapped alias, since nothing in the artifact describes them.
    const auto routed = [&](const WeightPlan& plan, std::int32_t rows, std::int32_t columns,
                            const std::byte*& host, family::BankPlanes& planes) {
        const family::HostObject* object = bank != nullptr ? bank->find(plan.object) : nullptr;
        const family::HostObjectPlan* planned =
            bank_plan != nullptr ? family::find_plan(*bank_plan, plan.object) : nullptr;
        if (object == nullptr) { return materialized_weight(backing, plan, rows, columns); }
        host = static_cast<const std::byte*>(object->host);
        if (planned != nullptr && planned->q5_rows > 0) {
            planes = family::BankPlanes::Q5;
            return family::host_q5_weight(*object, rows, columns);
        }
        if (planned != nullptr && planned->q4_rows > 0) {
            planes = family::BankPlanes::Q4;
            return family::host_q4_weight(*object, rows, columns);
        }
        if (planned != nullptr && planned->decode_rows > 0) {
            planes = family::BankPlanes::W8;
            return family::host_w8_weight(*object, rows, columns);
        }
        return materialized_weight(backing, plan, rows, columns);
    };
    const std::byte* host_gate_up = nullptr;
    const std::byte* host_down    = nullptr;
    out.moe.routed_gate_up = routed(source.routed_gate_up, moe_geometry(g).routed_gate_rows(),
                                    g.hidden, host_gate_up, out.gate_up_planes);
    out.moe.routed_down    = routed(source.routed_down, moe_geometry(g).routed_down_rows(),
                                    moe_geometry(g).intermediate, host_down, out.down_planes);
    if (g.shared_intermediate > 0) {
    out.moe.shared_gate_up = materialized_weight(backing, source.shared_gate_up,
                                                 moe_geometry(g).shared_rows(), g.hidden);
    out.moe.shared_down    = materialized_weight(backing, source.shared_down, g.hidden,
                                                 moe_geometry(g).shared_intermediate);
    }
    out.moe.experts_per_token = moe_geometry(g).experts_per_token;
    out.moe.routed_scale      = moe_geometry(g).routed_scale;
    out.moe.shared_gated      = moe_geometry(g).shared_gated;
    out.moe.swiglu_limit      = moe_geometry(g).swiglu_limit;
    // A layer is banked as a pair, so either both pointers are set or neither.
    if (host_gate_up != nullptr && host_down != nullptr) {
        out.host_gate_up = host_gate_up;
        out.host_down    = host_down;
    } else if (family::bank_planes_are_affine(out.gate_up_planes) ||
               family::bank_planes_are_affine(out.down_planes)) {
        throw std::logic_error("glm5_next: a mixture layer is banked as affine planes on one "
                               "side only");
    }
}

FeedForwardPayload load_feed_forward(const artifact::MaterializedArtifact& backing,
                                     const TextLayerPlan& source, const family::TextGeometry& g,
                                     const family::HostBank* bank,
                                     const family::HostBankPlan* bank_plan, const Tensor& post_norm,
                                     std::int32_t layer) {
    FeedForwardPayload out;
    out.hc          = load_hyper_connection(backing, source.feed_forward_hc, g, layer);
    out.norm        = post_norm;
    out.rms_epsilon = g.rms_epsilon;
    load_feed_forward_weights(backing, source.feed_forward, g, bank, bank_plan, layer, out);
    return out;
}

/// The latent attention's projections into either payload that carries them by these names:
/// the trunk's, beside its hyper-connection, or the draft head's.
template <class Payload>
void load_latent_projection(const artifact::MaterializedArtifact& backing,
                            const LatentAttentionPlan& source, const family::TextGeometry& g,
                            Payload& payload) {
    payload.query_a      = materialized_weight(backing, source.query_a, g.q_lora_rank, g.hidden);
    payload.query_a_norm = artifact::materialized_tensor(
        backing, source.query_a_norm, NumericFormat::BF16,
        {static_cast<std::uint64_t>(g.q_lora_rank)});
    payload.query_b      = materialized_weight(backing, source.query_b,
                                               g.query_heads * g.qk_head_dim,
                                               g.q_lora_rank);
    payload.kv_a         = materialized_weight(backing, source.kv_a, g.kv_lora_rank, g.hidden);
    payload.kv_a_norm    = artifact::materialized_tensor(
        backing, source.kv_a_norm, NumericFormat::BF16,
        {static_cast<std::uint64_t>(g.kv_lora_rank)});
    payload.k_b = materialized_weight(backing, source.k_b, g.latent_key_rows(),
                                      g.qk_head_dim);
    payload.v_b = materialized_weight(backing, source.v_b,
                                      g.query_heads * g.v_head_dim, g.kv_lora_rank);
}

} // namespace

family::TextGeometry declared_geometry_with_schedule(const artifact::Reader& reader) {
    auto geometry = family::TextGeometry::resolved_moe(reader.geometry(), reader.layer_types());
    const auto require = [&](const char* name, bool positive = true) {
        const auto found = reader.geometry().find(name);
        if (found == reader.geometry().end() || (positive && found->second <= 0)) {
            throw std::invalid_argument(std::string("missing or invalid geometry.") + name);
        }
    };
    for (const char* name : {"hc_streams", "hc_sinkhorn_iterations", "hc_epsilon", "q_lora_rank",
                             "kv_lora_rank", "qk_head_dim", "v_head_dim", "gdn_conv_kernel",
                             "gdn_key_heads", "gdn_key_head_dim", "gdn_value_heads",
                             "gdn_value_head_dim", "kda_gate_rank", "kda_gate_bound", "gdn_scale",
                             "dense_intermediate", "routed_scale"}) { require(name); }
    for (const char* name : {"leading_dense_layers", "shared_intermediate", "swiglu_limit", "mtp_layers"}) {
        require(name, false);
    }
    const auto& g = geometry;
    if (g.kv_heads != 1 || g.rotary_dim != 0 || g.head_dim != g.kv_lora_rank ||
        static_cast<std::int64_t>(g.hc_streams) * g.hidden != g.residual ||
        g.gdn_key_heads != g.gdn_value_heads || g.gdn_key_head_dim != g.gdn_value_head_dim ||
        g.leading_dense_layers > g.layers || g.mtp_layers > 1) {
        throw std::invalid_argument("inconsistent GLM checkpoint geometry");
    }
    for (const auto width : {g.qk_head_dim, g.v_head_dim, g.kv_lora_rank}) {
        if (static_cast<std::int64_t>(g.query_heads) * width > INT32_MAX) {
            throw std::invalid_argument("GLM latent projection width exceeds int32");
        }
    }
    return geometry;
}

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features, int stage_first,
                               int stage_last, std::uint32_t host_moe_layers,
                               std::uint32_t gpu_layers, family::BankPlanes bank_planes,
                               LoadProgress progress) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.geometry     = declared_geometry_with_schedule(binder.reader());
    out.frontend     = family::bind_text_only_frontend_resources(binder);
    out.features     = features;

    if (features.vision) {
        throw std::runtime_error("glm5_next is a text-only target: --vision is unsupported");
    }
    if (features.dflash()) {
        throw std::runtime_error(
            "glm5_next has no DFlash stack; its speculation is the checkpoint's own NextN draft "
            "head, --spec mtp");
    }

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    const family::TextGeometry& g         = out.geometry;
    // The embedding table is 0.65 GB and only the first stage reads it: every call site goes
    // through `Hooks::embed`, and the runtime asks `stage_embeds()` at each one, so a later
    // stage validates it and uploads nothing. On a 200 GB checkpoint split eight ways that is
    // the difference between the last stage fitting and not.
    //
    // The output head is *not* pruned the same way, though only the last stage should read it.
    // `sample_from_hidden` reaches it without asking `stage_finishes()`, and every other
    // pipeline-capable target uploads the head everywhere, so the gap has never shown. Pruning
    // it here would trade 0.65 GB for a crash on a path this target cannot yet prove is
    // unreachable; the honest thing is to carry it and leave the gap named.
    const bool staged     = stage_last > 0;
    // The head runs where the logits are (the last stage, or a whole model), and it embeds the
    // tokens it verifies and drafts -- so that stage carries the table too.
    const bool holds_head = !staged || stage_last >= g.layers;
    out.embeds            = !staged || stage_first == 0 || (features.mtp() && holds_head);
    out.finishes          = true;
    g_layer_placement   = out.embeds ? artifact::TensorPlacement::Device
                                     : artifact::TensorPlacement::ValidateOnly;
    out.token_embedding = bind_weight(binder, "text/token_embedding", vocabulary_format,
                                      {static_cast<std::uint64_t>(g.output_rows),
                                       static_cast<std::uint64_t>(g.hidden)});
    g_layer_placement   = artifact::TensorPlacement::Device;
    bind_text_layers(binder, weights_profile, stage_first, stage_last, host_moe_layers,
                     gpu_layers, out);
    g_expert_placement  = artifact::TensorPlacement::Device;
    bind_mtp_head(binder, vocabulary_format, features, holds_head, out);
    out.final_norm      = artifact::bind_device_tensor(
        binder, "text/final_norm", NumericFormat::BF16,
        {static_cast<std::uint64_t>(g.hidden)});
    out.output_head     = bind_weight(binder, "text/output_head", vocabulary_format,
                                      {static_cast<std::uint64_t>(g.output_rows),
                                       static_cast<std::uint64_t>(g.hidden)});

    load_plan.materialization = binder.finish();
    out.host_bank = family::collect_host_bank(binder, load_plan.materialization,
                                              std::move(progress));
    // What the bank makes of a GGUF's expert blocks as it fills them: planes the host expert
    // path reads at memory speed -- by default the narrowest lossless ones per object (Q4G32AM
    // where the file is 4-bit affine, W8 where it is wider), or all-W8 / all-Q4 on request --
    // or the blocks as they lie. A row-split W8 artifact is already planes and the marking
    // leaves it alone.
    for (family::HostObjectPlan& object : out.host_bank.objects) {
        const bool gate_up = object.name.ends_with("moe/routed_gate_up");
        const bool down    = object.name.ends_with("moe/routed_down");
        if (!gate_up && !down) { continue; }
        const artifact::ObjectDescriptor* stored = binder.reader().find(object.name);
        const auto* tensor = stored != nullptr ? std::get_if<artifact::TensorDescriptor>(stored)
                                               : nullptr;
        if (tensor == nullptr) { continue; }
        const std::int64_t rows = gate_up ? moe_geometry(g).routed_gate_rows()
                                          : moe_geometry(g).routed_down_rows();
        const std::int32_t columns = gate_up ? g.hidden : moe_geometry(g).intermediate;
        family::bank_as_planes(object, rows, columns, artifact::qtype_for(tensor->format),
                               bank_planes);
    }
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)),
      host_bank(plan.host_bank.objects.empty() ? nullptr
                                               : family::HostBank::shared(plan.host_bank)) {
    if (host_bank) { host_bank->attach(backing); }
    runtime.geometry              = plan.geometry;
    const family::TextGeometry& g = runtime.geometry;

    std::size_t attending = 0;
    for (const TextLayerPlan& source : plan.text_layers) { attending += source.attends ? 1U : 0U; }
    runtime.full_layers.resize(attending);
    runtime.gdn_layers.resize(plan.text_layers.size() - attending);
    frontend = family::take_text_only_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    if (plan.embeds) {
        runtime.token_embedding =
            materialized_weight(backing, plan.token_embedding, g.output_rows, g.hidden);
    }

    std::size_t full_index = 0;
    std::size_t kda_index  = 0;
    std::int32_t layer     = -1;
    for (const TextLayerPlan& source : plan.text_layers) {
        ++layer;
        if (!source.resident) { // another stage's layer: keep the index bookkeeping only
            (source.attends ? full_index : kda_index) += 1;
            continue;
        }
        const Tensor input_norm = artifact::materialized_tensor(
            backing, source.input_norm, NumericFormat::BF16,
            {static_cast<std::uint64_t>(g.hidden)});
        const Tensor post_norm = artifact::materialized_tensor(
            backing, source.post_attention_norm, NumericFormat::BF16,
            {static_cast<std::uint64_t>(g.hidden)});
        if (source.attends) {
            FullAttentionWeights& target = runtime.full_layers.at(full_index++);
            target.input_norm            = input_norm;
            LatentAttentionPayload payload;
            payload.hc          = load_hyper_connection(backing, source.attention_hc, g, layer);
            payload.norm        = input_norm;
            payload.rms_epsilon = g.rms_epsilon;
            load_latent_projection(backing, source.attention, g, payload);
            target.projection = std::move(payload);
            // No per-head query or key norm: this attention normalises its two low ranks
            // instead, and those live in the projection payload.
            target.output = materialized_weight(backing, source.attention.output, g.hidden,
                                                g.query_heads * g.v_head_dim);
            target.post_attention_norm = post_norm;
            target.post_mixer = load_feed_forward(backing, source, g, host_bank.get(),
                                                  &plan.host_bank, post_norm, layer);
        } else {
            KdaWeights& target = runtime.gdn_layers.at(kda_index++);
            target.input_norm  = input_norm;
            KdaProjectionPayload payload;
            payload.hc               = load_hyper_connection(backing, source.attention_hc, g,
                                                                layer);
            payload.rms_epsilon      = g.rms_epsilon;
            payload.gate_lower_bound = -g.kda_gate_bound;
            payload.query_key_value = materialized_weight(backing, source.kda.query_key_value,
                                                          g.convolution_dim(), g.hidden);
            payload.decay_a = materialized_weight(backing, source.kda.decay_a, g.kda_gate_rank,
                                                  g.hidden);
            payload.decay_b = materialized_weight(backing, source.kda.decay_b, g.value_dim(),
                                                  g.kda_gate_rank);
            payload.decay_bias = artifact::materialized_tensor(
                backing, source.kda.decay_bias, NumericFormat::FP32,
                {static_cast<std::uint64_t>(g.value_dim())});
            payload.a_log = artifact::materialized_tensor(
                backing, source.kda.a_log, NumericFormat::FP32,
                {static_cast<std::uint64_t>(g.gdn_value_heads)});
            payload.beta   = materialized_weight(backing, source.kda.beta, g.gdn_value_heads,
                                                 g.hidden);
            payload.gate_a = materialized_weight(backing, source.kda.gate_a, g.kda_gate_rank,
                                                 g.hidden);
            payload.gate_b = materialized_weight(backing, source.kda.gate_b, g.value_dim(),
                                                 g.kda_gate_rank);
            target.projection  = std::move(payload);
            target.convolution = artifact::materialized_tensor(
                backing, source.kda.convolution, NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.convolution_dim()),
                 static_cast<std::uint64_t>(g.gdn_conv_kernel)});
            target.norm   = artifact::materialized_tensor(
                backing, source.kda.norm, NumericFormat::BF16,
                {static_cast<std::uint64_t>(g.gdn_value_head_dim)});
            target.output = materialized_weight(backing, source.kda.output, g.hidden,
                                                g.value_dim());
            target.post_attention_norm = post_norm;
            target.post_mixer = load_feed_forward(backing, source, g, host_bank.get(),
                                                  &plan.host_bank, post_norm, layer);
        }
    }

    if (plan.features.mtp() && plan.mtp.present && plan.mtp.resident) {
        MtpWeights& mtp      = runtime.mtp.emplace();
        mtp.input_projection = materialized_weight(backing, plan.mtp.input_projection, g.hidden,
                                                   g.mtp_input_rows());
        mtp.embedding_norm   = artifact::materialized_tensor(
            backing, plan.mtp.embedding_norm, NumericFormat::BF16,
            {static_cast<std::uint64_t>(g.hidden)});
        mtp.hidden_norm      = artifact::materialized_tensor(
            backing, plan.mtp.hidden_norm, NumericFormat::BF16,
            {static_cast<std::uint64_t>(g.hidden)});
        mtp.input_norm       = artifact::materialized_tensor(
            backing, plan.mtp.input_norm, NumericFormat::BF16,
            {static_cast<std::uint64_t>(g.hidden)});
        mtp.attention.rms_epsilon = g.rms_epsilon;
        load_latent_projection(backing, plan.mtp.attention, g, mtp.attention);
        // No per-head query or key norm and no output gate, as in the trunk: the family's
        // tail skips both for this target and the two norm slots stay empty.
        mtp.output = materialized_weight(backing, plan.mtp.attention.output, g.hidden,
                                         g.query_heads * g.v_head_dim);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16,
            {static_cast<std::uint64_t>(g.hidden)});
        mtp.post_mixer.norm        = mtp.post_attention_norm;
        mtp.post_mixer.rms_epsilon = g.rms_epsilon;
        // The head's experts stay on the card (bind_mtp_head places them there), so it has
        // no bank to ask; its layer index is the one past the trunk.
        load_feed_forward_weights(backing, plan.mtp.feed_forward, g, nullptr, nullptr, g.layers,
                                  mtp.post_mixer);
        mtp.final_norm = artifact::materialized_tensor(
            backing, plan.mtp.final_norm, NumericFormat::BF16,
            {static_cast<std::uint64_t>(g.hidden)});
    }

    if (plan.finishes) {
        // Every layer this stage runs must have arrived, and no layer it does not run may have.
    // The inventories are indexed by position among the layers of that kind over the *whole*
    // model -- which is what the runtime's `full_idx` and `gdn_idx` compute -- so an index that
    // slipped shows up here as a bound layer in the wrong slot rather than as a null weight
    // eight minutes into a forward pass.
    {
        std::size_t full_at = 0;
        std::size_t kda_at  = 0;
        for (const TextLayerPlan& source : plan.text_layers) {
            const bool attends = source.attends;
            const bool bound   = attends
                                     ? runtime.full_layers.at(full_at).projection.hc.weights.mix.n > 0
                                     : runtime.gdn_layers.at(kda_at).projection.hc.weights.mix.n > 0;
            if (bound != source.resident) {
                throw std::logic_error(
                    std::string("glm5_next: the ") + (attends ? "attention" : "linear") +
                    " layer at inventory index " +
                    std::to_string(attends ? full_at : kda_at) + " is " +
                    (bound ? "bound" : "unbound") + " but this stage " +
                    (source.resident ? "runs" : "does not run") + " it");
            }
            (attends ? full_at : kda_at) += 1;
        }
    }

    runtime.final_norm  = artifact::materialized_tensor(
            backing, plan.final_norm, NumericFormat::BF16,
            {static_cast<std::uint64_t>(g.hidden)});
        runtime.output_head =
            materialized_weight(backing, plan.output_head, g.output_rows, g.hidden);
    }
}

} // namespace sinfer::targets::glm5_next::detail
