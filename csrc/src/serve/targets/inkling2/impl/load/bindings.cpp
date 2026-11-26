#include "targets/inkling2/impl/load/bindings.h"

#include "family/impl/load/dflash.h"
#include "family/impl/load/routed_nvfp4.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <string_view>
#include <variant>

namespace sinfer::targets::inkling2::load {
namespace {

using artifact::ArtifactError;
using artifact::NumericFormat;
using artifact::ObjectHandle;
using artifact::TensorPlacement;
using Json = nlohmann::json;

constexpr std::string_view kSchema = "inkling2-artifact/1";

std::uint64_t u64(std::int64_t value) { return static_cast<std::uint64_t>(value); }

template <class T>
T number(const Json& object, const char* key) {
    if (!object.contains(key) || !object.at(key).is_number()) {
        throw ArtifactError(std::string("inkling2/config.json: missing number ") + key);
    }
    return object.at(key).get<T>();
}

std::vector<std::int32_t> int_list(const Json& value, const char* what) {
    if (!value.is_array()) { throw ArtifactError(std::string("inkling2/config.json: ") + what + " is not an array"); }
    std::vector<std::int32_t> out;
    for (const auto& item : value) { out.push_back(item.get<std::int32_t>()); }
    return out;
}

float read_positive_fp32(const artifact::Binder& binder, ObjectHandle handle, const std::string& name) {
    const auto bytes = binder.payload(handle).data;
    if (bytes.size() != 4) { throw ArtifactError(name + ": expected one FP32 word"); }
    std::uint32_t word = 0;
    std::memcpy(&word, bytes.data(), sizeof(word));
    const float value = std::bit_cast<float>(word);
    if (!std::isfinite(value) || value <= 0.0F) { throw ArtifactError(name + ": not finite and positive"); }
    return value;
}

std::int32_t geometry_int(const artifact::Reader& reader, const char* key) {
    const auto& g  = reader.geometry();
    const auto it  = g.find(key);
    if (it == g.end()) { throw ArtifactError(std::string("inkling2: artifact geometry lacks ") + key); }
    return static_cast<std::int32_t>(it->second);
}

void check_against_root(const artifact::Reader& reader, const ArtifactConfig& c) {
    const std::array<std::pair<const char*, std::int32_t>, 9> pairs = {{
        {"hidden", c.hidden}, {"layers", c.layers}, {"output_rows", c.vocab_padded}, {"token_domain", c.vocab},
        {"query_heads", c.query_heads}, {"kv_heads", c.kv_heads}, {"head_dim", c.head_dim},
        {"experts", c.experts}, {"intermediate", c.expert_intermediate},
    }};
    for (const auto& [key, value] : pairs) {
        if (geometry_int(reader, key) != value) {
            throw ArtifactError(std::string("inkling2: geometry.") + key + " disagrees with inkling2/config.json");
        }
    }
    const auto& types = reader.layer_types();
    if (types.size() != static_cast<std::size_t>(c.layers)) {
        throw ArtifactError("inkling2: layer_types must list every physical layer");
    }
    for (std::int32_t layer = 0; layer < c.layers; ++layer) {
        const bool full = types[layer] == "full_attention";
        if (full != c.global[layer] || (!full && types[layer] != "sliding_attention")) {
            throw ArtifactError("inkling2: layer_types disagree with inkling2/config.json at layer " +
                                std::to_string(layer));
        }
    }
}

struct Binding {
    artifact::Binder& binder;

    ObjectHandle tensor(const std::string& name, NumericFormat format, std::initializer_list<std::uint64_t> shape,
                        TensorPlacement placement) const {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    }
    artifact::LinearBinding linear(const std::string& name, std::int32_t rows, std::int32_t columns,
                                   TensorPlacement placement, std::optional<NumericFormat> format = {}) const {
        auto binding = artifact::bind_linear(binder, name, rows, columns, placement);
        if (format.has_value() && binding.format != *format) {
            throw ArtifactError(name + ": stored as " + std::string(artifact::format_name(binding.format)) +
                                ", expected " + std::string(artifact::format_name(*format)));
        }
        return binding;
    }
};

ExpertBankPlan bind_bank(const Binding& b, const std::string& name, ExpertFormat format, std::int32_t rows,
                         std::int32_t columns, std::int32_t experts, TensorPlacement placement) {
    ExpertBankPlan plan;
    if (format == ExpertFormat::Nvfp4) {
        plan.bank = b.tensor(name, NumericFormat::NVFP4, {u64(rows), u64(columns)}, placement);
        family::require_identity_divisor(b.binder, plan.bank, name, rows, columns);
        plan.amax = b.tensor(name + "_amax", NumericFormat::FP32, {u64(experts)}, placement);
        plan.outer_scale_denominator =
            b.tensor(name + "_outer_scale_denominator", NumericFormat::FP32, {1}, placement);
        plan.outer_scale_denominator_value =
            read_positive_fp32(b.binder, *plan.outer_scale_denominator, name + "_outer_scale_denominator");
    } else {
        plan.bank = b.tensor(name, NumericFormat::MXFP8, {u64(rows), u64(columns)}, placement);
    }
    return plan;
}

Tensor tensor_of(const artifact::MaterializedArtifact& m, ObjectHandle handle, NumericFormat format,
                 std::initializer_list<std::int32_t> ne) {
    return artifact::materialized_tensor(m, handle, format, ne);
}

ExpertBank load_bank(const artifact::MaterializedArtifact& m, const ExpertBankPlan& plan, ExpertFormat format,
                     std::int32_t experts, std::int32_t rows_per_expert, std::int32_t columns, bool interleaved) {
    ExpertBank bank;
    bank.format          = format;
    bank.experts         = experts;
    bank.rows_per_expert = rows_per_expert;
    bank.columns         = columns;
    bank.gate_up_interleaved = interleaved;
    const std::array<std::uint64_t, 2> shape = {u64(experts) * u64(rows_per_expert), u64(columns)};
    const auto* bytes = static_cast<const std::uint8_t*>(m.device_data(plan.bank));
    bank.codes        = bytes;
    if (format == ExpertFormat::Nvfp4) {
        const auto geometry = artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
        bank.scales = bytes + geometry.scale_plane_offset;
        bank.weight = family::routed_nvfp4_weight(m, plan.bank, experts * rows_per_expert, columns);
        bank.amax   = static_cast<const float*>(tensor_of(m, *plan.amax, NumericFormat::FP32, {experts}).data);
        bank.outer_scale_denominator =
            static_cast<const float*>(tensor_of(m, *plan.outer_scale_denominator, NumericFormat::FP32, {1}).data);
        bank.outer_scale_denominator_value = plan.outer_scale_denominator_value;
    } else {
        const auto geometry = artifact::mxfp8_geometry(NumericFormat::MXFP8, shape);
        bank.scales = bytes + geometry.scale_plane_offset;
    }
    return bank;
}

} // namespace

ArtifactConfig parse_artifact_config(std::span<const std::byte> json) {
    Json doc;
    try {
        doc = Json::parse(reinterpret_cast<const char*>(json.data()),
                          reinterpret_cast<const char*>(json.data()) + json.size());
    } catch (const Json::exception& error) {
        throw ArtifactError(std::string("inkling2/config.json: invalid JSON: ") + error.what());
    }
    try {
        if (doc.value("schema", std::string()) != kSchema) {
            throw ArtifactError("inkling2/config.json: schema is not " + std::string(kSchema));
        }
        ArtifactConfig c;
        const Json& t       = doc.at("text");
        c.hidden            = number<std::int32_t>(t, "hidden");
        c.layers            = number<std::int32_t>(t, "layers");
        c.vocab_padded      = number<std::int32_t>(t, "vocab_padded");
        c.vocab             = number<std::int32_t>(t, "vocab");
        c.query_heads       = number<std::int32_t>(t, "query_heads");
        c.kv_heads          = number<std::int32_t>(t, "kv_heads");
        c.head_dim          = number<std::int32_t>(t, "head_dim");
        c.d_rel             = number<std::int32_t>(t, "d_rel");
        c.rel_extent        = number<std::int32_t>(t, "rel_extent");
        c.sliding_window    = number<std::int32_t>(t, "sliding_window");
        c.conv_kernel       = number<std::int32_t>(t, "conv_kernel");
        c.experts           = number<std::int32_t>(t, "experts");
        c.experts_per_token = number<std::int32_t>(t, "experts_per_token");
        c.expert_intermediate   = number<std::int32_t>(t, "expert_intermediate");
        c.log_scaling_n_floor   = number<std::int32_t>(t, "log_scaling_n_floor");
        c.attn_res_block_layers = number<std::int32_t>(t, "attn_res_block_layers");
        c.eos_token_id          = number<std::int32_t>(t, "eos_token_id");
        c.max_context           = number<std::int64_t>(t, "max_context");
        c.rms_eps           = number<double>(t, "rms_eps");
        c.qk_l2_clip        = number<double>(t, "qk_l2_clip");
        c.rel_l2_clip       = number<double>(t, "rel_l2_clip");
        c.log_scaling_alpha = number<double>(t, "log_scaling_alpha");
        c.logit_softcap     = number<double>(t, "logit_softcap");
        c.gate_softcap      = number<double>(t, "gate_softcap");
        c.up_softcap        = number<double>(t, "up_softcap");
        c.route_scale       = number<double>(t, "route_scale");
        const Json& loop    = doc.at("loop");
        c.core_start        = number<std::int32_t>(loop, "core_start");
        c.core_end          = number<std::int32_t>(loop, "core_end");
        c.passes            = number<std::int32_t>(loop, "passes");
        c.gate_rank         = number<std::int32_t>(loop, "gate_rank");
        if (loop.value("is_varloop", false)) { throw ArtifactError("inkling2/config.json: is_varloop is not served"); }
        const Json& layers = doc.at("layers");
        if (!layers.is_array() || layers.size() != static_cast<std::size_t>(c.layers)) {
            throw ArtifactError("inkling2/config.json: layers must list every physical layer");
        }
        for (std::int32_t i = 0; i < c.layers; ++i) {
            const Json& layer = layers[i];
            if (layer.at("layer").get<std::int32_t>() != i) { throw ArtifactError("inkling2/config.json: layers out of order"); }
            const auto attention = layer.at("attention").get<std::string>();
            const auto experts   = layer.at("experts").get<std::string>();
            if ((attention != "local" && attention != "global") || (experts != "nvfp4" && experts != "mxfp8")) {
                throw ArtifactError("inkling2/config.json: layer " + std::to_string(i) + " has an unknown kind");
            }
            c.global.push_back(attention == "global");
            c.expert_format.push_back(experts == "nvfp4" ? ExpertFormat::Nvfp4 : ExpertFormat::Mxfp8);
            if (layer.at("core").get<bool>() != c.is_core(i)) {
                throw ArtifactError("inkling2/config.json: layer " + std::to_string(i) + " core flag disagrees with loop");
            }
        }
        const Json& visits = doc.at("visits");
        for (const auto& row : visits.at("table")) {
            c.visits.push_back({row.at(0).get<std::int32_t>(), row.at(1).get<std::int32_t>()});
        }
        const std::size_t expected = static_cast<std::size_t>(c.layers + (c.passes - 1) * (c.core_end - c.core_start));
        if (c.visits.size() != expected || visits.at("count").get<std::size_t>() != expected) {
            throw ArtifactError("inkling2/config.json: visit table does not match the loop");
        }
        const auto rows = doc.at("moe").at("gate_up_rows").get<std::string>();
        if (rows != "interleaved" && rows != "split") { throw ArtifactError("inkling2/config.json: unknown gate_up_rows"); }
        c.gate_up_interleaved = rows == "interleaved";
        const Json& attention = doc.at("attention");
        if (!attention.at("bf16").get<bool>()) { throw ArtifactError("inkling2/config.json: the BF16 attention is required"); }
        for (const auto& [scheme, _] : attention.at("fp8").items()) { c.fp8_schemes.push_back(scheme); }
        if (doc.contains("dspark")) {
            const Json& d = doc.at("dspark");
            DraftConfig draft;
            draft.layers         = number<std::int32_t>(d, "layers");
            draft.hidden         = number<std::int32_t>(d, "hidden");
            draft.query_heads    = number<std::int32_t>(d, "query_heads");
            draft.kv_heads       = number<std::int32_t>(d, "kv_heads");
            draft.head_dim       = number<std::int32_t>(d, "head_dim");
            draft.intermediate   = number<std::int32_t>(d, "intermediate");
            draft.sliding_window = number<std::int32_t>(d, "sliding_window");
            draft.block_size     = number<std::int32_t>(d, "block_size");
            draft.markov_rank    = number<std::int32_t>(d, "markov_rank");
            draft.max_position   = number<std::int64_t>(d, "max_position");
            draft.rope_theta     = number<double>(d, "rope_theta");
            draft.rms_eps        = number<double>(d, "rms_eps");
            draft.local_layers   = int_list(d.at("local_layers"), "dspark.local_layers");
            draft.target_visits  = int_list(d.at("target_visits"), "dspark.target_visits");
            if (d.value("markov_head_type", std::string()) != "rnn_hidden" || draft.hidden != c.hidden ||
                draft.target_visits.empty() || draft.target_visits.back() != static_cast<std::int32_t>(expected)) {
                throw ArtifactError("inkling2/config.json: unsupported dspark block");
            }
            c.dspark = std::move(draft);
        }
        return c;
    } catch (const Json::exception& error) {
        throw ArtifactError(std::string("inkling2/config.json: ") + error.what());
    }
}

BindingPlan bind_artifact(artifact::Binder& binder, const BindOptions& options) {
    BindingPlan out;
    out.options         = options;
    out.config_resource = artifact::bind_raw_resource(binder, "inkling2/config.json");
    out.config          = parse_artifact_config(binder.payload(out.config_resource).data);
    const ArtifactConfig& c = out.config;
    check_against_root(binder.reader(), c);
    if (binder.reader().identity().architecture != "inkling2") {
        throw ArtifactError("inkling2: identity.architecture is not inkling2");
    }
    if (options.attention.fp8 &&
        std::find(c.fp8_schemes.begin(), c.fp8_schemes.end(), options.attention.scheme) == c.fp8_schemes.end()) {
        throw ArtifactError("inkling2: FP8 attention '" + options.attention.scheme +
                            "' was requested but the artifact carries no such variant");
    }
    out.frontend           = family::bind_text_only_frontend_resources(binder);
    out.special_tokens_map = artifact::bind_raw_resource(binder, "frontend/special_tokens_map.json");
    out.source_config      = artifact::bind_raw_resource(binder, "inkling2/source_config.json");
    out.hf_quant_config    = artifact::bind_raw_resource(binder, "inkling2/hf_quant_config.json");

    const Binding b{binder};
    const std::int32_t D = c.hidden;
    const TensorPlacement gp = options.globals_placement.value_or(artifact::ScopedPlacement::current());
    GlobalsPlan& g    = out.globals;
    g.placement       = GlobalsPlacement{
              .embedding     = options.embedding_placement.value_or(gp),
              .head          = options.head_placement.value_or(gp),
              .small_globals = options.small_globals_placement.value_or(gp),
    };
    const TensorPlacement ep = g.placement.embedding, hp = g.placement.head, sp = g.placement.small_globals;
    // The tower feeds the embedding: it lives on the stage that embeds.
    family::bind_inkling_vision(binder, out.vision, D, options.vision, ep);
    g.embedding       = b.linear("text/embedding", c.vocab_padded, D, ep, NumericFormat::BF16);
    g.embedding_norm  = b.tensor("text/embedding_norm", NumericFormat::BF16, {u64(D)}, ep);
    g.final_norm      = b.tensor("text/final_norm", NumericFormat::BF16, {u64(D)}, hp);
    g.output_head     = b.linear("text/output_head", c.vocab_padded, D, hp, NumericFormat::BF16);
    g.attn_query.resize(c.layers);
    g.mlp_query.resize(c.layers);
    for (std::int32_t layer = 1; layer < c.layers; ++layer) {
        g.attn_query[layer] = b.tensor("text/attn_res/attn_query/" + std::to_string(layer), NumericFormat::FP32,
                                       {c.global[layer] ? 3u : 1u, u64(D)}, sp);
    }
    for (std::int32_t layer = 0; layer < c.layers; ++layer) {
        g.mlp_query[layer] = b.tensor("text/attn_res/mlp_query/" + std::to_string(layer), NumericFormat::FP32,
                                      {1, u64(D)}, sp);
    }
    g.output_query = b.tensor("text/attn_res/output_query", NumericFormat::FP32, {1, u64(D)}, sp);
    const std::uint64_t core_rows = 2 * u64(c.core_end - c.core_start);
    g.cross_pass_queries    = b.tensor("text/loop/cross_pass_queries", NumericFormat::FP32, {core_rows, u64(D)}, sp);
    g.cross_pass_kv_queries = b.tensor("text/loop/cross_pass_kv_queries", NumericFormat::FP32, {core_rows, u64(D)}, sp);

    out.layers.resize(c.layers);
    const std::int32_t I = c.expert_intermediate, E = c.experts, K = c.conv_kernel;
    for (std::int32_t layer = 0; layer < c.layers; ++layer) {
        LayerPlan& L          = out.layers[layer];
        const std::string p   = "text/layers/" + std::to_string(layer) + "/";
        const TensorPlacement place =
            options.layer_placement ? options.layer_placement(layer) : artifact::ScopedPlacement::current();
        L.resident      = place != TensorPlacement::ValidateOnly;
        L.global        = c.global[layer];
        L.core          = c.is_core(layer);
        L.expert_format = c.expert_format[layer];
        L.attn_input_norm = b.tensor(p + "attn/input_norm", NumericFormat::BF16, {L.global ? 3u : 1u, u64(D)}, place);
        const TensorPlacement bf16_attention = options.attention.fp8 ? TensorPlacement::ValidateOnly : place;
        L.qkvr   = b.linear(p + "attn/qkvr", c.qkvr_rows(), D, bf16_attention, NumericFormat::BF16);
        L.output = b.linear(p + "attn/output", D, c.q_rows(), bf16_attention, NumericFormat::BF16);
        for (const std::string& scheme : c.fp8_schemes) {
            const bool chosen = options.attention.fp8 && scheme == options.attention.scheme;
            const TensorPlacement fp8_place = chosen ? place : TensorPlacement::ValidateOnly;
            auto qkvr   = b.linear(p + "attn/fp8-" + scheme + "/qkvr", c.qkvr_rows(), D, fp8_place);
            auto output = b.linear(p + "attn/fp8-" + scheme + "/output", D, c.q_rows(), fp8_place);
            for (const auto* binding : {&qkvr, &output}) {
                if (binding->format != NumericFormat::FP8_E4M3FN_ROW_F32S &&
                    binding->format != NumericFormat::FP8_E4M3FN_1X128_F32S) {
                    throw ArtifactError(p + "attn/fp8-" + scheme + ": not an FP8 attention format");
                }
            }
            if (chosen) {
                L.qkvr_fp8   = qkvr;
                L.output_fp8 = output;
            }
        }
        L.q_scale  = b.tensor(p + "attn/q_scale", NumericFormat::BF16, {u64(c.query_heads), u64(c.head_dim)}, place);
        L.r_scale  = b.tensor(p + "attn/r_scale", NumericFormat::BF16, {u64(c.query_heads), u64(c.d_rel)}, place);
        L.rel_proj = b.tensor(p + "attn/rel_proj", NumericFormat::BF16, {u64(c.d_rel), u64(c.rel_extent)}, place);
        if (L.global) {
            L.k_conv_static = b.tensor(p + "attn/k_conv_static", NumericFormat::BF16, {u64(K), u64(c.kv_rows())}, place);
            L.v_conv_static = b.tensor(p + "attn/v_conv_static", NumericFormat::BF16, {u64(K), u64(c.kv_rows())}, place);
            L.k_conv_proj   = b.linear(p + "attn/k_conv_proj", c.kv_heads * K, D, place, NumericFormat::BF16);
            L.v_conv_proj   = b.linear(p + "attn/v_conv_proj", c.kv_heads * K, D, place, NumericFormat::BF16);
        } else {
            L.k_conv = b.tensor(p + "attn/k_conv", NumericFormat::BF16, {u64(c.kv_rows()), u64(K)}, place);
            L.v_conv = b.tensor(p + "attn/v_conv", NumericFormat::BF16, {u64(c.kv_rows()), u64(K)}, place);
        }
        L.attn_out_conv  = b.tensor(p + "attn_out_conv", NumericFormat::BF16, {u64(D), u64(K)}, place);
        L.mlp_out_conv   = b.tensor(p + "mlp_out_conv", NumericFormat::BF16, {u64(D), u64(K)}, place);
        L.moe_input_norm = b.tensor(p + "moe/input_norm", NumericFormat::BF16, {u64(D)}, place);
        L.router         = b.linear(p + "moe/router", E, D, place, NumericFormat::BF16);
        L.router_bias    = b.tensor(p + "moe/router_bias", NumericFormat::FP32, {u64(E)}, place);
        L.w13 = bind_bank(b, p + "moe/w13", L.expert_format, E * 2 * I, D, E, place);
        L.w2  = bind_bank(b, p + "moe/w2", L.expert_format, E * D, I, E, place);
        if (L.core) {
            const auto r = u64(c.gate_rank);
            L.loop_down        = b.tensor(p + "loop/down", NumericFormat::BF16, {4, r, 2 * u64(D)}, place);
            L.loop_gain_forget = b.tensor(p + "loop/gain_forget", NumericFormat::BF16, {4, 2 * u64(D), r}, place);
        }
    }

    out.has_draft = c.dspark.has_value() && binder.has("mtp/fuse_weight");
    if (c.dspark.has_value() != binder.has("mtp/fuse_weight")) {
        throw ArtifactError("inkling2: the dspark config and the mtp/ objects disagree");
    }
    if (options.draft && !out.has_draft) {
        throw ArtifactError("inkling2: the DSpark drafter was requested but this artifact carries none");
    }
    if (out.has_draft) {
        const DraftConfig& d = *c.dspark;
        const TensorPlacement dp = options.draft ? options.draft_placement.value_or(hp) : TensorPlacement::ValidateOnly;
        out.draft.placement      = dp;
        // DSpark: the dense linears may live in a mapped host bank (re-quantised to FP8 on the device at load)
        const TensorPlacement lp = options.draft && options.draft_linears_host ? TensorPlacement::HostBank : dp;
        const std::int32_t Dd = d.hidden, q = d.query_heads * d.head_dim, kv = d.kv_heads * d.head_dim;
        DraftPlan& m = out.draft;
        m.fuse_weight = b.tensor("mtp/fuse_weight", NumericFormat::BF16,
                                 {u64(d.layers + 1), u64(d.target_visits.size())}, dp);
        m.mask_embed  = b.tensor("mtp/mask_embed", NumericFormat::BF16, {u64(Dd)}, dp);
        m.final_norm  = b.tensor("mtp/final_norm", NumericFormat::BF16, {u64(Dd)}, dp);
        m.layers.resize(d.layers);
        for (std::int32_t layer = 0; layer < d.layers; ++layer) {
            DraftLayerPlan& l   = m.layers[layer];
            const std::string p = "mtp/layers/" + std::to_string(layer) + "/";
            l.feature_norm        = b.tensor(p + "feature_norm", NumericFormat::BF16, {u64(Dd)}, dp);
            l.input_norm          = b.tensor(p + "input_norm", NumericFormat::BF16, {u64(Dd)}, dp);
            l.post_attention_norm = b.tensor(p + "post_attention_norm", NumericFormat::BF16, {u64(Dd)}, dp);
            l.q     = b.linear(p + "attn/q", q, Dd, lp, NumericFormat::BF16);
            l.k     = b.linear(p + "attn/k", kv, Dd, lp, NumericFormat::BF16);
            l.v     = b.linear(p + "attn/v", kv, Dd, lp, NumericFormat::BF16);
            l.o     = b.linear(p + "attn/o", Dd, q, lp, NumericFormat::BF16);
            l.k_ctx = b.linear(p + "attn/k_ctx", kv, Dd, lp, NumericFormat::BF16);
            l.v_ctx = b.linear(p + "attn/v_ctx", kv, Dd, lp, NumericFormat::BF16);
            l.q_norm     = b.tensor(p + "attn/q_norm", NumericFormat::BF16, {u64(d.head_dim)}, dp);
            l.k_norm     = b.tensor(p + "attn/k_norm", NumericFormat::BF16, {u64(d.head_dim)}, dp);
            l.k_ctx_norm = b.tensor(p + "attn/k_ctx_norm", NumericFormat::BF16, {u64(d.head_dim)}, dp);
            l.gate = b.linear(p + "mlp/gate", d.intermediate, Dd, lp, NumericFormat::BF16);
            l.up   = b.linear(p + "mlp/up", d.intermediate, Dd, lp, NumericFormat::BF16);
            l.down = b.linear(p + "mlp/down", Dd, d.intermediate, lp, NumericFormat::BF16);
        }
        const std::int32_t r = d.markov_rank;
        m.markov_feature_norm    = b.tensor("mtp/markov/feature_norm", NumericFormat::BF16, {u64(Dd)}, dp);
        m.markov_hidden_out      = b.linear("mtp/markov/hidden_out", Dd, r, dp, NumericFormat::BF16);
        m.markov_hidden_out_bias = b.tensor("mtp/markov/hidden_out_bias", NumericFormat::BF16, {u64(Dd)}, dp);
        m.markov_w1              = b.linear("mtp/markov/w1", c.vocab_padded, r, dp, NumericFormat::BF16);
        m.markov_rnn_proj        = b.linear("mtp/markov/rnn_proj", 3 * r, 2 * r + 2 * Dd, dp, NumericFormat::BF16);
        m.markov_rnn_proj_bias   = b.tensor("mtp/markov/rnn_proj_bias", NumericFormat::BF16, {3 * u64(r)}, dp);
    }

    // A DFlash checkpoint drafter (converter --dflash-model): the family's `dflash/*` objects, bound on the device when
    // it is the served drafter, validated only otherwise. Its feature ids run to the visits + the final-norm output.
    const artifact::Reader& reader = binder.reader();
    if (!reader.dflash_geometry().empty()) {
        family::TextGeometry g;
        g.hidden      = c.hidden;
        g.output_rows = c.vocab_padded;
        g.dflash      = family::DFlashGeometry::resolved(reader.dflash_geometry(), reader.dflash_target_layers(), c.hidden,
                                                         static_cast<int>(c.visits.size()) + 1, c.vocab_padded);
        family::StartupFeatures features;
        features.speculative = options.dflash ? SpeculativeBackend::DFlash : SpeculativeBackend::None;
        out.dflash           = family::bind_dflash(binder, g, features);
    } else if (options.dflash) {
        throw ArtifactError("inkling2: a DFlash checkpoint drafter was requested but this artifact carries none");
    }
    return out;
}

InklingWeights load_weights(const BindingPlan& plan, const artifact::MaterializedArtifact& m) {
    InklingWeights w;
    w.config               = plan.config;
    const ArtifactConfig& c = plan.config;
    const std::int32_t D   = c.hidden;
    const auto linear = [&](const artifact::LinearBinding& binding, std::int32_t rows, std::int32_t columns) {
        return artifact::materialized_linear(m, binding, rows, columns);
    };
    // Each group of globals is read only where this stage placed it (a 1-card engine places all of them).
    InklingGlobals& g    = w.globals;
    const GlobalsPlan& gs = plan.globals;
    g.has_embedding      = gs.placement.embedding != TensorPlacement::ValidateOnly;
    g.has_head           = gs.placement.head != TensorPlacement::ValidateOnly;
    g.has_small_globals  = gs.placement.small_globals != TensorPlacement::ValidateOnly;
    if (g.has_embedding) {
        g.embedding      = linear(gs.embedding, c.vocab_padded, D);
        g.embedding_norm = tensor_of(m, gs.embedding_norm, NumericFormat::BF16, {D});
    }
    if (g.has_head) {
        g.output_head = linear(gs.output_head, c.vocab_padded, D);
        g.final_norm  = tensor_of(m, gs.final_norm, NumericFormat::BF16, {D});
    }
    if (g.has_small_globals) {
        g.attn_query.resize(c.layers);
        g.mlp_query.resize(c.layers);
        for (std::int32_t layer = 1; layer < c.layers; ++layer) {
            g.attn_query[layer] = tensor_of(m, gs.attn_query[layer], NumericFormat::FP32, {D, c.global[layer] ? 3 : 1});
        }
        for (std::int32_t layer = 0; layer < c.layers; ++layer) {
            g.mlp_query[layer] = tensor_of(m, gs.mlp_query[layer], NumericFormat::FP32, {D, 1});
        }
        g.output_query = tensor_of(m, gs.output_query, NumericFormat::FP32, {D, 1});
        const std::int32_t core_rows = 2 * (c.core_end - c.core_start);
        g.cross_pass_queries    = tensor_of(m, gs.cross_pass_queries, NumericFormat::FP32, {D, core_rows});
        g.cross_pass_kv_queries = tensor_of(m, gs.cross_pass_kv_queries, NumericFormat::FP32, {D, core_rows});
    }
    w.layers.resize(c.layers);
    const std::int32_t I = c.expert_intermediate, E = c.experts, K = c.conv_kernel;
    for (std::int32_t layer = 0; layer < c.layers; ++layer) {
        const LayerPlan& s   = plan.layers[layer];
        InklingLayerWeights& t = w.layers[layer];
        t.layer  = layer;
        t.global = s.global;
        t.core   = s.core;
        if (!s.resident) { continue; }
        t.attn_input_norm = tensor_of(m, s.attn_input_norm, NumericFormat::BF16, {D, s.global ? 3 : 1});
        t.fp8_attention   = s.qkvr_fp8.has_value();
        t.qkvr   = linear(t.fp8_attention ? *s.qkvr_fp8 : s.qkvr, c.qkvr_rows(), D);
        t.output = linear(t.fp8_attention ? *s.output_fp8 : s.output, D, c.q_rows());
        t.q_scale  = tensor_of(m, s.q_scale, NumericFormat::BF16, {c.head_dim, c.query_heads});
        t.r_scale  = tensor_of(m, s.r_scale, NumericFormat::BF16, {c.d_rel, c.query_heads});
        t.rel_proj = tensor_of(m, s.rel_proj, NumericFormat::BF16, {c.rel_extent, c.d_rel});
        if (s.global) {
            t.k_conv_static = tensor_of(m, s.k_conv_static, NumericFormat::BF16, {c.kv_rows(), K});
            t.v_conv_static = tensor_of(m, s.v_conv_static, NumericFormat::BF16, {c.kv_rows(), K});
            t.k_conv_proj   = linear(*s.k_conv_proj, c.kv_heads * K, D);
            t.v_conv_proj   = linear(*s.v_conv_proj, c.kv_heads * K, D);
        } else {
            t.k_conv = tensor_of(m, s.k_conv, NumericFormat::BF16, {K, c.kv_rows()});
            t.v_conv = tensor_of(m, s.v_conv, NumericFormat::BF16, {K, c.kv_rows()});
        }
        t.attn_out_conv  = tensor_of(m, s.attn_out_conv, NumericFormat::BF16, {K, D});
        t.mlp_out_conv   = tensor_of(m, s.mlp_out_conv, NumericFormat::BF16, {K, D});
        t.moe_input_norm = tensor_of(m, s.moe_input_norm, NumericFormat::BF16, {D});
        t.router         = linear(s.router, E, D);
        t.router_bias    = static_cast<const float*>(tensor_of(m, s.router_bias, NumericFormat::FP32, {E}).data);
        t.w13 = load_bank(m, s.w13, s.expert_format, E, 2 * I, D, c.gate_up_interleaved);
        t.w2  = load_bank(m, s.w2, s.expert_format, E, D, I, true);
        if (s.core) {
            t.loop_down        = tensor_of(m, s.loop_down, NumericFormat::BF16, {2 * D, c.gate_rank, 4});
            t.loop_gain_forget = tensor_of(m, s.loop_gain_forget, NumericFormat::BF16, {c.gate_rank, 2 * D, 4});
        }
    }
    w.has_draft = plan.has_draft && plan.options.draft && plan.draft.placement != TensorPlacement::ValidateOnly;
    if (w.has_draft) {
        const DraftConfig& d = *c.dspark;
        const DraftPlan& s   = plan.draft;
        InklingDraftWeights& t = w.draft;
        const std::int32_t Dd = d.hidden, q = d.query_heads * d.head_dim, kv = d.kv_heads * d.head_dim;
        t.fuse_weight = tensor_of(m, s.fuse_weight, NumericFormat::BF16,
                                  {static_cast<std::int32_t>(d.target_visits.size()), d.layers + 1});
        t.mask_embed  = tensor_of(m, s.mask_embed, NumericFormat::BF16, {Dd});
        t.final_norm  = tensor_of(m, s.final_norm, NumericFormat::BF16, {Dd});
        t.layers.resize(d.layers);
        for (std::int32_t layer = 0; layer < d.layers; ++layer) {
            const DraftLayerPlan& a = s.layers[layer];
            InklingDraftLayerWeights& l = t.layers[layer];
            l.feature_norm        = tensor_of(m, a.feature_norm, NumericFormat::BF16, {Dd});
            l.input_norm          = tensor_of(m, a.input_norm, NumericFormat::BF16, {Dd});
            l.post_attention_norm = tensor_of(m, a.post_attention_norm, NumericFormat::BF16, {Dd});
            l.q = linear(a.q, q, Dd);
            l.k = linear(a.k, kv, Dd);
            l.v = linear(a.v, kv, Dd);
            l.o = linear(a.o, Dd, q);
            l.k_ctx = linear(a.k_ctx, kv, Dd);
            l.v_ctx = linear(a.v_ctx, kv, Dd);
            l.q_norm     = tensor_of(m, a.q_norm, NumericFormat::BF16, {d.head_dim});
            l.k_norm     = tensor_of(m, a.k_norm, NumericFormat::BF16, {d.head_dim});
            l.k_ctx_norm = tensor_of(m, a.k_ctx_norm, NumericFormat::BF16, {d.head_dim});
            l.gate = linear(a.gate, d.intermediate, Dd);
            l.up   = linear(a.up, d.intermediate, Dd);
            l.down = linear(a.down, Dd, d.intermediate);
        }
        const std::int32_t r = d.markov_rank;
        t.markov_feature_norm    = tensor_of(m, s.markov_feature_norm, NumericFormat::BF16, {Dd});
        t.markov_hidden_out      = linear(s.markov_hidden_out, Dd, r);
        t.markov_hidden_out_bias = tensor_of(m, s.markov_hidden_out_bias, NumericFormat::BF16, {Dd});
        t.markov_w1              = linear(s.markov_w1, c.vocab_padded, r);
        t.markov_rnn_proj        = linear(s.markov_rnn_proj, 3 * r, 2 * r + 2 * Dd);
        t.markov_rnn_proj_bias   = tensor_of(m, s.markov_rnn_proj_bias, NumericFormat::BF16, {3 * r});
    }
    return w;
}

} // namespace sinfer::targets::inkling2::load
