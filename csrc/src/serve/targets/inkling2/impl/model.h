#pragma once

// Inkling-2-Mini: the loaded model as the family runtime sees it. The weight bindings
// (`impl/load/bindings.h`) bind and materialise the 24 physical layers; this file adds what the target
// derives at load (the pre-scaled AttnRes / message queries, ARTIFACT.md 5) and the 48-visit ModelView whose
// FullLayer entries are aliases into the physical layers (DESIGN-target.md 0.2, 4 (b)).

#include "targets/inkling2/impl/config.h"
#include "targets/inkling2/impl/load/bindings.h"
#include "targets/inkling2/impl/dspark.h"
#include "family/impl/load/host_bank.h"

#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/tensor.h"
#include <api/family/frontend_resources.h>
#include <api/family/model_view.h>
#include <api/family/startup_features.h>
#include <api/family/target_package.h>
#include <api/family/text_geometry.h>
#include <api/targets/inkling2/package.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

namespace sinfer::targets::inkling2::detail {

/// AttnRes pseudo-queries as the reference uses them: FP32, multiplied by fp32(2304^-0.5) at load
/// (`Inkling2AttnRes._load_query`, `Inkling2LoopModel.process_weights_after_loading`).
struct DerivedQueries {
    DeviceBuffer storage;
    std::array<Tensor, TextConfig::physical_layers> attn{}; ///< FP32 [D, 1|3]; [0] empty (layer 0 has none)
    std::array<Tensor, TextConfig::physical_layers> mlp{};  ///< FP32 [D, 1]
    Tensor output;                                          ///< FP32 [D, 1]
    /// Per core layer j (layer 6 + j): FP32 [D, 4] = (cross[2j], kv[2j], kv[2j + 1], cross[2j + 1]) x scale:
    /// the message queries of the streams attn, k, v, mlp.
    std::array<Tensor, TextConfig::core_end - TextConfig::core_start> message{};
    /// The attention's relative-bias tables, one per physical layer (inkling_rel_proj_table of `attn/rel_proj` at load):
    /// BF16 [16, kRelTableRows], shared by the passes of a core layer.
    DeviceBuffer rel_storage;
    std::array<Tensor, TextConfig::physical_layers> rel_table{};
};

/// The AttnRes / cross-pass message queries pre-scaled from RAW globals (attn / mlp / output / message; not the rel
/// tables): the base model's at load, and a decision adapter's from its own queries.
void derive_attn_res_queries(const load::InklingGlobals& g, DerivedQueries& out);

/// What every visit payload points at.
struct InklingModel {
    const load::InklingWeights* weights   = nullptr;
    const DerivedQueries* queries         = nullptr;
    const DSparkDerived* dspark           = nullptr; ///< set when the drafter is bound (DSpark)
};

/// The `FullAttentionProjectionWeights` of a visit: its physical layer and where it sits in the schedule.
struct VisitPayload {
    const load::InklingLayerWeights* layer = nullptr;
    const InklingModel* model              = nullptr;
    std::int32_t visit                     = 0;
};

/// No linear-attention layers: the GDN payload exists because the runtime's types name one.
struct NoGdnPayload {};

/// The `PostMixerWeights` of a visit (the MoE op reads the physical layer's banks).
struct MoePayload {
    const load::InklingLayerWeights* layer = nullptr;
    std::int32_t visit                     = 0;
};

using RuntimeModelView = family::ModelView<VisitPayload, NoGdnPayload, MoePayload, VisitPayload, MoePayload,
                                           family::DFlashWeights>;

struct BindingPlan {
    load::BindingPlan plan;
    family::TextGeometry geometry; ///< the served (48-visit) geometry
    family::StartupFeatures features;
    /// Objects bound HostBank (today only the DSpark dense linears under FP8 drafter weights).
    family::HostBankPlan host_bank;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

/// The 48-visit geometry over an artifact's root geometry (DESIGN-target.md 0.2). Its DFlash geometry is the served
/// drafter's: a DFlash checkpoint's when the artifact carries one (dflash_checkpoint_geometry), else the release's DSpark
/// drafter's (dspark_dflash_geometry, `dspark` set); SUROGATE_SERVE_INKLING_DRAFTER=dspark keeps DSpark.
[[nodiscard]] family::TextGeometry visit_geometry(const artifact::Reader& reader);
/// A DFlash checkpoint's geometry over the 48-visit target, or nullopt when the artifact carries none. The checkpoint
/// names DSpark's four features (visits 5 / 23 / 41 and the final-norm output, 48: SGLang's Inkling DFlash capture is
/// DSpark's); they are relabelled to the visit labels the forward captures them under (visits.h kFeatureLabels).
[[nodiscard]] std::optional<family::DFlashGeometry> dflash_checkpoint_geometry(const artifact::Reader& reader);
/// SUROGATE_SERVE_INKLING_DRAFTER (read once): "dspark" serves the release's DSpark drafter even when the artifact
/// carries a DFlash checkpoint; empty or "dflash" serves the checkpoint when there is one.
[[nodiscard]] bool inkling_dspark_forced();

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    std::shared_ptr<family::HostBank> host_bank; ///< pinned host objects, attached to `backing` before weights load
    family::FrontendResources frontend;
    load::InklingWeights weights;
    DerivedQueries queries;
    DSparkDerived dspark; ///< the drafter's load-time derivations (DSpark); empty without `--speculative dflash`
    InklingModel model;
    RuntimeModelView runtime;
};

SINFER_TARGET_LOADED_MODEL_IMPL();

} // namespace sinfer::targets::inkling2::detail
