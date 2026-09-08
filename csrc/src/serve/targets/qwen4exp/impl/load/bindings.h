#pragma once

#include <memory>
#include <api/targets/qwen4exp/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/text_geometry.h>
#include <api/family/model_view.h>
#include <api/family/startup_features.h>
#include <api/family/vision.h>

#include "api/ops/hyper_connection.h"
#include "api/ops/ngram_ple.h"
#include "api/ops/sparse_moe.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include "targets/qwen4exp/impl/config.h"
#include "family/impl/load/host_bank.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace sinfer::targets::qwen4exp::detail {

// The pinned host expert bank is the family's (`family/impl/load/host_bank.h`): any target with
// a mixture can put its experts there and let the kernels read them zero-copy over PCIe. These
// names keep this target's older spelling.
using family::HostBank;
using family::HostBankPlan;
using family::HostObject;
using family::HostObjectPlan;
using family::host_ggml_weight;
using family::host_linear;
using family::host_plan;
using family::host_tensor;
using family::host_w8_weight;



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
    /// What the bank will present for those two. A GGUF-native artifact stores the file's own
    /// GGML blocks, and the bank either keeps them or decodes them into planes; a converted one
    /// stores W8 row-split planes already.
    artifact::NumericFormat routed_gate_up_format = artifact::NumericFormat::W8G32_F16S;
    artifact::NumericFormat routed_down_format    = artifact::NumericFormat::W8G32_F16S;
    /// What the bank holds each half as (`family::BankPlanes`). Independent, because the two
    /// halves of a K_XL mixture are routinely stored at different widths: this checkpoint's
    /// Q4_K gate/up reach Q4G32AM, its Q5_1 down reach Q5G32AM, its Q8_0 down stay W8.
    family::BankPlanes routed_gate_up_planes = family::BankPlanes::Native;
    family::BankPlanes routed_down_planes    = family::BankPlanes::Native;
    artifact::ObjectHandle shared_gate_up;
    artifact::ObjectHandle shared_down;
};

struct IndexerPlan {
    artifact::ObjectHandle query;      // [indexer_heads * indexer_head_dim, hidden]
    artifact::ObjectHandle key;        // [indexer_head_dim, hidden]
    artifact::ObjectHandle query_norm; // [indexer_head_dim]
    artifact::ObjectHandle key_norm;
};

struct FullAttentionPlan {
    artifact::ObjectHandle query_key_gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
    IndexerPlan indexer{};
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

/// The NextN/MTP draft head. Structurally a trunk full-attention block, so it carries a
/// `TextLayerPlan` unchanged; only the tensors on either side of it are the head's own.
/// `embedding_norm`/`hidden_norm`/`input_projection` fold the next token's embedding into the
/// wide residual on the way in, and `head_mix` collapses the four streams on the way out,
/// standing in for the output norm this architecture does not have. The embedding table and
/// the LM head are the trunk's -- the export carries neither.
struct MtpPlan {
    bool present = false;  ///< the artifact carries a head
    bool resident = false; ///< ...and this run asked for one, so its weights are on the device
    artifact::ObjectHandle embedding_norm;   // [hidden]
    artifact::ObjectHandle hidden_norm;      // [residual]
    artifact::ObjectHandle input_projection; // [hidden, 2 * hidden]
    TextLayerPlan layer;
    HyperConnectionPlan head_mix;
};

struct BindingPlan {
    /// The validated checkpoint dimensions used to bind and load this artifact.
    family::TextGeometry geometry;
    family::VisionGeometry vision_geometry;
    family::FrontendResourcePlan frontend;
    family::StartupFeatures features;
    artifact::ObjectHandle token_embedding;
    std::vector<TextLayerPlan> text_layers;
    HyperConnectionPlan output_mix;
    artifact::ObjectHandle output_head;
    artifact::ObjectHandle ple_table;
    // The hash constants are read out of the artifact at bind time (the reader is gone by
    // the time the model is constructed).
    std::vector<std::uint64_t> ple_multipliers{};
    std::vector<std::int32_t> ple_head_offsets{};
    std::vector<std::int32_t> ple_head_vocab_sizes{};
    // Optional vision objects are bound with their declared tower dimensions.
    family::VisionBackbonePlan vision_backbone;
    family::VisionMergerInputPlan vision_merger_input;
    artifact::LinearBinding vision_merger_fc2;
    artifact::ObjectHandle vision_merger_fc2_bias;
    family::VisionMergerNormPlan vision_merger_norm;
    //: false when the source carried no tower (GGUF exports drop it).
    bool has_vision = false;
    MtpPlan mtp;
    // Host-resident objects are copied out of the artifact mapping while the reader lives.
    HostBankPlan host_bank;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

/// `stage_first/stage_last` (0/0 = every layer) restrict device residency to the layers of a
/// pipeline stage.
ArtifactLoadPlan bind_artifact(artifact::Binder& binder, family::StartupFeatures features,
                               int stage_first = 0, int stage_last = 0,
                               family::BankPlanes bank_planes = family::BankPlanes::Auto,
                               LoadProgress progress = {});

/// Per-block hyper-connection weights ride on the projection payloads the family hands to the
/// Variant at the norm hooks, so the Variant can mix before its projection.
/// QSA indexer of one full-attention layer (design/INFERENCE.md, phase 4). Empty tensors mean
/// the layer has no indexer (or it is not materialised on this pipeline stage).
struct IndexerWeights {
    Weight query;      // BF16 [indexer_heads * indexer_head_dim, hidden]
    Weight key;        // BF16 [indexer_head_dim, hidden]
    Tensor query_norm; // BF16 [indexer_head_dim]
    Tensor key_norm;
    [[nodiscard]] bool valid() const noexcept { return query.qdata != nullptr; }
};

struct AttentionProjectionPayload {
    family::TextGeometry geometry;
    Weight query_key_gate_value;
    ops::HyperConnectionWeights mix;
    IndexerWeights indexer;
};

struct GdnProjectionPayload {
    family::TextGeometry geometry;
    Tensor a_log;
    Tensor dt_bias;
    Weight a_b_projection;
    Weight query_key_value_z;
    ops::HyperConnectionWeights mix;
};

struct SparseMoePayload {
    family::TextGeometry geometry;
    ops::SparseMoeWeights op;
    ops::HyperConnectionWeights mix;
    /// What the bank holds each routed half as (`family::BankPlanes`): Q4 or Q5 planes are a
    /// base pointer and a shape, readable by the expert cache alone (the slot cache is then
    /// required); anything else is what the Weight says.
    family::BankPlanes gate_up_planes = family::BankPlanes::Native;
    family::BankPlanes down_planes    = family::BankPlanes::Native;
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
    family::ModelView<AttentionProjectionPayload, GdnProjectionPayload, SparseMoePayload,
                       AttentionProjectionPayload, SparseMoePayload,
                       family::DFlashWeights>;

/// The NextN draft head, as the runtime sees it. The block is a trunk full-attention layer --
/// the same type, run by the same code -- and the four tensors around it are the head's own:
/// `embedding_norm`/`hidden_norm`/`input_projection` fold the next token's embedding into the
/// wide residual, and `head_mix` collapses the four streams afterwards, standing in for the
/// output norm this architecture does not have. The embedding table and the LM head are the
/// trunk's.
struct MtpHeadWeights {
    bool present = false;
    Tensor embedding_norm;
    Tensor hidden_norm;
    Weight input_projection;
    ops::HyperConnectionWeights head_mix;
};

/// The family view plus what the residual hooks need: the output mixer, the PLE layer and the
/// draft head.
struct RuntimeModelView : FamilyModelView {
    ops::HyperConnectionWeights output_mix;
    PleWeights ple;
    MtpHeadWeights mtp_head;
    /// The draft head's block. Held beside `full_layers` rather than in it so the trunk's
    /// layer count stays the trunk's.
    typename FamilyModelView::FullLayer mtp_block;
};

using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;
using DFlashWeights        = RuntimeModelView::DFlash;
using DFlashLayerWeights   = family::DFlashLayerWeights;

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    std::shared_ptr<HostBank> host_bank; // shared by the pipeline stages of one process
    family::FrontendResources frontend;
    RuntimeModelView runtime;
};

SINFER_TARGET_LOADED_MODEL_IMPL();

} // namespace sinfer::targets::qwen4exp::detail
