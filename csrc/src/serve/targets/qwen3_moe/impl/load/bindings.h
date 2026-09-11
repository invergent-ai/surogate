#pragma once

#include "family/impl/load/qwen3_vl.h"

#include "family/impl/moe/banked_experts.h"

#include <api/targets/qwen3_moe/package.h>
#include <api/family/frontend_resources.h>
#include <api/family/text_geometry.h>
#include <api/family/model_view.h>
#include <api/family/startup_features.h>
#include <api/family/vision.h>

#include "targets/qwen3_moe/impl/config.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "family/impl/load/host_bank.h"

#include <memory>
#include "api/ops/sparse_moe.h"
#include "core/tensor.h"

#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace sinfer::targets::qwen3_moe::detail {

// The compiled default for a Qwen3-30B-A3B; an artifact that declares `layers` overrides it,
// which is what lets this one target serve every size of the architecture.
// Every layer is full attention. The empty half of the family's split is not a
// placeholder: the shared ModelView is instantiated with it, and the shared
// runtime's GDN arrays and state pool are sized from it.
inline constexpr std::size_t kGdnLayers = 0;

struct WeightPlan {
    artifact::ObjectHandle object;
    artifact::NumericFormat format = artifact::NumericFormat::BF16;
};

/// The mixture. The router is one row per expert -- no shared-expert gate fused onto it -- and
/// the two routed weights are bound by *stored* format rather than by the profile's
/// expectation: a GGUF serves its experts as its own K-quant superblocks, and reading those
/// through the group-wise codec the profile names would decode every expert wrongly.
struct MoePlan {
    artifact::ObjectHandle router;
    artifact::LinearBinding routed_gate_up;
    artifact::LinearBinding routed_down;
};

/// Qwen3-MoE attention is UNGATED: the fused projection carries `[query | key | value]` rows
/// and nothing else, where the hybrid family's carries `[query | key | output gate | value]`.
/// Naming the member for what it holds is the whole guard against reading it at the family's
/// stride.
struct AttentionPlan {
    WeightPlan query_key_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    WeightPlan output;
};

struct TextLayerPlan {
    bool resident = true;
    artifact::ObjectHandle input_norm;
    AttentionPlan attention;
    artifact::ObjectHandle post_attention_norm;
    MoePlan moe;
};

struct BindingPlan : family::Qwen3VlVisionPlan {
    /// The dimensions bound against: the compiled config with the artifact's
    /// `geometry` member laid over it.
    family::TextGeometry geometry = {};
    /// Text resources plus image/video processor settings for Qwen3-VL-MoE.
    family::FrontendResourcePlan frontend;
    family::StartupFeatures features;

    WeightPlan token_embedding;
    /// One per layer, sized when the artifact is bound rather than by the type.
    std::vector<TextLayerPlan> text_layers;
    artifact::ObjectHandle final_norm;
    /// The released Qwen3-MoE checkpoints ship their own head; where a checkpoint ties it, the
    /// converter resolves that and stores the head as its own object either way.
    WeightPlan output_head;

    /// Objects this stage put in pinned host memory instead of on the card.
    family::HostBankPlan host_bank;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features,
                               std::uint32_t host_moe_layers = 0, std::uint32_t gpu_layers = 0,
                               LoadProgress progress = {});

/// The post-mixer is the mixture: one closed op over the registered geometry.
struct SparseMoePayload {
    family::BankedExperts banked;
    ops::SparseMoeWeights op;
};

struct FusedAttentionProjectionPayload {
    Weight query_key_value;
};

/// The linear-mixer payload the shared ModelView still names. Every layer of this architecture
/// attends, so no instance of it is ever constructed; it exists because the family's
/// `GdnWeights<Projection, PostMixer>` is a type, not an option.
struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
};

/// Likewise for the MTP head: declared so the family's `MtpWeights<...>`
/// instantiates, never materialised (`bind_artifact` refuses speculation).
struct MtpAttentionPayload {
    Weight packed;
    Weight query;
    Weight key;
    Weight output_gate;
    Weight value;
};

using RuntimeModelView =
    family::ModelView<FusedAttentionProjectionPayload, GdnProjectionPayload, SparseMoePayload,
                       MtpAttentionPayload, SparseMoePayload, family::DFlashWeights>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    /// Weights this stage kept in pinned host memory rather than on the card.
    std::shared_ptr<family::HostBank> host_bank;
    family::FrontendResources frontend;
    RuntimeModelView runtime;
};

SINFER_TARGET_LOADED_MODEL_IMPL();

} // namespace sinfer::targets::qwen3_moe::detail
