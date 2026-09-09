#include "family/vision_standalone.h"

#include "artifact/binder.h"
#include "artifact/typed_binding.h"
#include <api/family/text_geometry.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace sinfer::family {
namespace {

using artifact::NumericFormat;
using artifact::TensorPlacement;

/// The tower's objects, bound exactly as a target binds them. The sequence is the one in
/// `targets/*/impl/load/bindings.cpp`; keeping it here rather than reaching into a target
/// is what makes this independent of which family the checkpoint belongs to.
struct TowerPlan {
    VisionBackbonePlan backbone;
    VisionMergerInputPlan merger_input;
    VisionMergerNormPlan merger_norm;
    artifact::LinearBinding merger_output;
    artifact::ObjectHandle merger_output_bias;
    struct Deepstack {
        std::int32_t layer = 0;
        artifact::LinearBinding fc1, fc2;
        artifact::ObjectHandle fc1_bias, fc2_bias, norm_weight, norm_bias;
    };
    std::vector<Deepstack> deepstack;
};

TowerPlan bind_tower(artifact::Binder& binder, const VisionGeometry& v) {
    const auto placement = TensorPlacement::Device;
    const auto tensor    = [&](const std::string& name, int width) {
        return artifact::bind_tensor(binder, name, NumericFormat::BF16, {width}, placement);
    };
    TowerPlan plan;
    plan.backbone           = bind_vision_backbone(binder, placement, v);
    plan.merger_input       = bind_vision_merger_input(binder, placement, v);
    plan.merger_norm        = bind_vision_merger_norm(binder, placement, v);
    plan.merger_output      = artifact::bind_linear(binder, "vision/merger/fc2", v.output_hidden,
                                                    v.merger_hidden(), placement);
    plan.merger_output_bias = tensor("vision/merger/fc2_bias", v.output_hidden);
    for (int layer = 0; layer < v.layers; ++layer) {
        const auto prefix = "vision/layers/" + std::to_string(layer) + "/deepstack/";
        if (!binder.has(prefix + "fc1")) { continue; }
        plan.deepstack.push_back(TowerPlan::Deepstack{
            .layer       = layer,
            .fc1         = artifact::bind_linear(binder, prefix + "fc1", v.merger_hidden(),
                                                 v.merger_hidden(), placement),
            .fc2         = artifact::bind_linear(binder, prefix + "fc2", v.output_hidden,
                                                 v.merger_hidden(), placement),
            .fc1_bias    = tensor(prefix + "fc1_bias", v.merger_hidden()),
            .fc2_bias    = tensor(prefix + "fc2_bias", v.output_hidden),
            .norm_weight = tensor(prefix + "norm/weight", v.merger_hidden()),
            .norm_bias   = tensor(prefix + "norm/bias", v.merger_hidden()),
        });
    }
    if (plan.deepstack.size() != static_cast<std::size_t>(v.deepstack_layers)) {
        throw std::runtime_error("vision deepstack objects disagree with the declared count");
    }
    return plan;
}

VisionWeights take_tower(const artifact::MaterializedArtifact& backing, const TowerPlan& plan,
                         const VisionGeometry& v) {
    const auto tensor = [&](artifact::ObjectHandle handle, int width) {
        return artifact::materialized_tensor(backing, handle, NumericFormat::BF16, {width});
    };
    VisionWeights vision;
    vision.common = materialize_vision_common(backing, plan.backbone, plan.merger_input,
                                              plan.merger_norm, v);
    vision.merger_fc2      = artifact::materialized_linear(backing, plan.merger_output,
                                                           v.output_hidden, v.merger_hidden());
    vision.merger_fc2_bias = tensor(plan.merger_output_bias, v.output_hidden);
    for (const auto& merger : plan.deepstack) {
        vision.deepstack.push_back(VisionWeights::DeepstackMerger{
            .layer       = merger.layer,
            .fc1         = artifact::materialized_linear(backing, merger.fc1, v.merger_hidden(),
                                                         v.merger_hidden()),
            .fc2         = artifact::materialized_linear(backing, merger.fc2, v.output_hidden,
                                                         v.merger_hidden()),
            .fc1_bias    = tensor(merger.fc1_bias, v.merger_hidden()),
            .fc2_bias    = tensor(merger.fc2_bias, v.output_hidden),
            .norm_weight = tensor(merger.norm_weight, v.merger_hidden()),
            .norm_bias   = tensor(merger.norm_bias, v.merger_hidden()),
        });
    }
    return vision;
}

} // namespace

struct StandaloneVisionTower::State {
    DeviceContext device;
    artifact::Reader reader;
    VisionGeometry geometry;
    artifact::MaterializedArtifact backing;
    VisionWeights weights;
    std::optional<VisionContext> tower;
    /// Grown to the largest item seen. `encode` resets it each call, so one allocation
    /// serves every image of that size or smaller.
    std::optional<DeviceArena> workspace;

    State(const std::filesystem::path& path, int device_id)
        : device(device_id), reader(path) {}
};

StandaloneVisionTower::StandaloneVisionTower(const std::filesystem::path& artifact, int device)
    : state_(std::make_unique<State>(artifact, device)) {
    artifact::Binder binder(state_->reader);
    if (!binder.has("vision/patch_embedding")) {
        throw std::runtime_error(artifact.string() + " declares no vision tower");
    }
    state_->geometry = VisionGeometry::resolved(state_->reader.vision_geometry());
    const TowerPlan plan = bind_tower(binder, state_->geometry);
    // The text model, the frontend resources and everything else this checkpoint ships are
    // deliberately not read: the caller asked for a tower.
    binder.discard_unconsumed();
    state_->backing = artifact::materialize(state_->reader, binder.finish(), state_->device);
    state_->weights = take_tower(state_->backing, plan, state_->geometry);

    // The tower's own output width is the contract the merger was trained against, so it
    // stands in for the text model this loader does not have.
    TextGeometry text{};
    text.hidden  = state_->geometry.output_hidden;
    text.layers  = state_->geometry.layers;
    state_->tower.emplace(state_->device, state_->weights, state_->geometry, text);
}

StandaloneVisionTower::~StandaloneVisionTower()                                     = default;
StandaloneVisionTower::StandaloneVisionTower(StandaloneVisionTower&&) noexcept      = default;
StandaloneVisionTower& StandaloneVisionTower::operator=(StandaloneVisionTower&&) noexcept = default;

const VisionGeometry& StandaloneVisionTower::geometry() const noexcept { return state_->geometry; }

std::size_t StandaloneVisionTower::merged_tokens(const VisionGrid& grid) const {
    const auto merge = static_cast<std::size_t>(state_->geometry.merge);
    return static_cast<std::size_t>(grid.temporal) * static_cast<std::size_t>(grid.height) *
           static_cast<std::size_t>(grid.width) / (merge * merge);
}

std::size_t StandaloneVisionTower::output_bytes(const VisionGrid& grid) const {
    return VisionContext::output_transient_bytes(state_->geometry, merged_tokens(grid));
}

void StandaloneVisionTower::encode(std::span<const std::uint16_t> patches, const VisionGrid& grid,
                                   PromptModality modality, Tensor& output) {
    const VisionItemControl control = build_vision_item_control(grid, modality);
    const std::size_t needed = VisionContext::workspace_bytes(state_->geometry, control);
    if (!state_->workspace || state_->workspace->capacity() < needed) {
        state_->workspace.emplace(needed);
    }
    state_->tower->encode(VisionItemView{patches, &control}, output, *state_->workspace);
    // The tower is stream-ordered, and the buffer leaves here for a caller that has no idea
    // which stream that was -- torch reads it on its own. Returning before the work lands
    // hands back whatever the allocation happened to hold, which reads as a plausible
    // tensor: finite, right shape, silently wrong. Wait for it.
    if (const cudaError_t status = cudaStreamSynchronize(state_->device.stream);
        status != cudaSuccess) {
        throw std::runtime_error(std::string("vision encode failed: ") +
                                 cudaGetErrorString(status));
    }
}

} // namespace sinfer::family
