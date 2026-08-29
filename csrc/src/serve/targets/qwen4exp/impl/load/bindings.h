#pragma once

#include <memory>
#include <api/targets/qwen4exp/package.h>
#include <api/targets/qwen3_6/frontend_resources.h>
#include <api/targets/qwen3_6/model_view.h>
#include <api/targets/qwen3_6/startup_features.h>
#include <api/targets/qwen3_6/vision.h>

#include "api/ops/hyper_connection.h"
#include "api/ops/ngram_ple.h"
#include "api/ops/sparse_moe.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include "targets/qwen4exp/impl/config.h"
#include "targets/qwen4exp/impl/load/host_bank.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen4exp::detail {

inline constexpr std::size_t kTextLayers          = TextConfig::layers;
inline constexpr std::size_t kFullAttentionLayers = TextConfig::full_attention_layers();
inline constexpr std::size_t kGdnLayers           = TextConfig::gdn_layers();

struct HyperConnectionPlan {
    artifact::ObjectHandle norm;
    artifact::ObjectHandle down;
    artifact::ObjectHandle up;
    artifact::ObjectHandle inject; ///< unset for the output mixer
};

struct MoePlan {
    artifact::ObjectHandle router_shared_gate;
    artifact::ObjectHandle routed_gate_up; ///< host resident
    artifact::ObjectHandle routed_down;    ///< host resident
    artifact::ObjectHandle shared_gate_up;
    artifact::ObjectHandle shared_down;
};

struct FullAttentionPlan {
    artifact::ObjectHandle query_key_gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
};

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    artifact::ObjectHandle a_b_projection;
    artifact::ObjectHandle query_key_value_z;
    artifact::ObjectHandle norm;
    artifact::ObjectHandle output;
};

struct PlePlan {
    artifact::ObjectHandle key;
    artifact::ObjectHandle value;
    artifact::ObjectHandle norm_key;
    artifact::ObjectHandle norm_query;
    artifact::ObjectHandle norm_conv;
    artifact::ObjectHandle convolution;
};

struct TextLayerPlan {
    HyperConnectionPlan hc_attention;
    HyperConnectionPlan hc_mlp;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    bool has_ple           = false;
    PlePlan ple{};
    MoePlan moe;
    // A pipeline stage materialises only its own layers on the device: the others are
    // validated against the artifact but never uploaded (their routed experts still join the
    // shared host bank, which every stage of a process maps once).
    bool resident = true;
};

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;
    artifact::ObjectHandle token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    HyperConnectionPlan output_mix;
    artifact::ObjectHandle output_head;
    artifact::ObjectHandle ple_table;
    // The hash constants are read out of the artifact at bind time (the reader is gone by
    // the time the model is constructed).
    std::array<std::uint64_t, TextConfig::ple_ngram> ple_multipliers{};
    std::array<std::int32_t, TextConfig::ple_heads> ple_head_offsets{};
    std::array<std::int32_t, TextConfig::ple_heads> ple_head_vocab_sizes{};
    // Host-resident objects are copied out of the artifact mapping while the reader lives.
    HostBankPlan host_bank;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

/// `stage_first/stage_last` (0/0 = every layer) restrict device residency to the layers of a
/// pipeline stage.
ArtifactLoadPlan bind_artifact(artifact::Binder& binder, qwen3_6::StartupFeatures features,
                               int stage_first = 0, int stage_last = 0);

/// Per-block hyper-connection weights ride on the projection payloads the family hands to the
/// Variant at the norm hooks, so the Variant can mix before its projection.
struct AttentionProjectionPayload {
    Weight query_key_gate_value;
    ops::HyperConnectionWeights mix;
};

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    Weight a_b_projection;
    Weight query_key_value_z;
    ops::HyperConnectionWeights mix;
};

struct SparseMoePayload {
    ops::SparseMoeWeights op;
    ops::HyperConnectionWeights mix;
    std::int32_t layer = -1; // text layer index (the expert slot cache keys its tables by it)
    // Host virtual addresses of the routed expert objects (the Weights above hold the
    // device-mapped aliases); the CPU expert compute reads the planes through these.
    const std::byte* host_gate_up = nullptr;
    const std::byte* host_down    = nullptr;
};

struct PleWeights {
    ops::NgramPleWeights op;
    ops::NgramPleHash hash;
    ops::NgramPleTable table;
    int layer = -1;
};

using FamilyModelView =
    qwen3_6::ModelView<AttentionProjectionPayload, GdnProjectionPayload, SparseMoePayload,
                       AttentionProjectionPayload, SparseMoePayload,
                       qwen3_6::DFlashWeights<1>, kFullAttentionLayers, kGdnLayers>;

/// The family view plus what the residual hooks need: the output mixer and the PLE layer.
struct RuntimeModelView : FamilyModelView {
    ops::HyperConnectionWeights output_mix;
    PleWeights ple;
};

using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;
using DFlashWeights        = RuntimeModelView::DFlash;
using DFlashLayerWeights   = qwen3_6::DFlashLayerWeights;

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    std::shared_ptr<HostBank> host_bank; // shared by the pipeline stages of one process
    qwen3_6::FrontendResources frontend;
    RuntimeModelView runtime;
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan plan,
         artifact::MaterializedArtifact materialized)
        : weights_profile(weights_profile_in), data(std::move(plan), std::move(materialized)) {}

    WeightsProfile weights_profile;
    LoadedModelData data;
};

} // namespace ninfer::targets::qwen4exp::detail
