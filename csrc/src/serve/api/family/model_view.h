#pragma once

#include <api/family/startup_features.h>
#include <api/family/text_geometry.h>
#include <api/family/vision.h>
#include <api/family/vision_geometry.h>

#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace sinfer {

class DeviceArena;

namespace family {

template <class ProjectionPayload, class PostMixerPayload>
struct FullAttentionWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class ProjectionPayload, class PostMixerPayload>
struct GdnWeights {
    Tensor input_norm;
    ProjectionPayload projection;
    Tensor convolution;
    Tensor norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
};

template <class AttentionPayload, class PostMixerPayload>
struct MtpWeights {
    Weight input_projection;
    Tensor embedding_norm;
    Tensor hidden_norm;
    Tensor input_norm;
    AttentionPayload attention;
    Tensor query_norm;
    Tensor key_norm;
    Weight output;
    Tensor post_attention_norm;
    PostMixerPayload post_mixer;
    Tensor final_norm;
};

struct OptimizedProposalWeights {
    Weight head;
    Tensor token_ids;
};

struct DFlashLayerWeights {
    Tensor input_norm;
    Weight query_key_value;
    Weight context_key;
    Weight context_value;
    Tensor query_norm;
    Tensor key_norm;
    Weight attention_output;
    Tensor post_attention_norm;
    Weight gate_up;
    Weight down;
};

struct DFlashWeights {
    Weight feature_projection;
    Tensor context_norm;
    std::vector<DFlashLayerWeights> layers;
    Tensor final_norm;
};

template <class FullProjectionPayload, class GdnProjectionPayload, class MainPostMixerPayload,
          class MtpAttentionPayload, class MtpPostMixerPayload, class DFlashPayload,
          // The tower a target ships. Defaulted to the family's so every existing
          // instantiation is unchanged; a target with its own overrides it, and the
          // vision geometry below starts from it rather than from the family's.
          class VisionCfg = VisionBackboneConfig>
struct ModelView {
    using FullLayer = FullAttentionWeights<FullProjectionPayload, MainPostMixerPayload>;
    using GdnLayer  = GdnWeights<GdnProjectionPayload, MainPostMixerPayload>;
    using MtpLayer  = MtpWeights<MtpAttentionPayload, MtpPostMixerPayload>;
    using DFlash    = DFlashPayload;

    DeviceArena* weights_arena = nullptr;
    /// The dimensions these weights were bound against: the target's compiled config with
    /// whatever the artifact declared laid over it. The runtime reads its sizes from here
    /// rather than from the compiled constants, which is what lets one target serve every
    /// size of its family.
    TextGeometry geometry;
    Weight token_embedding;
    /// Sized when the weights are bound, not by the type: two checkpoints of one family
    /// differ in how many layers attend and how many are linear.
    std::vector<FullLayer> full_layers;
    std::vector<GdnLayer> gdn_layers;
    Tensor final_norm;
    Weight output_head;
    StartupFeatures features;
    std::optional<OptimizedProposalWeights> optimized_proposal;
    std::optional<MtpLayer> mtp;
    std::optional<DFlashPayload> dflash;
    /// The tower's dimensions the vision weights were bound against, carried beside them the
    /// way `geometry` is carried beside the text weights.
    VisionGeometry vision_geometry;
    std::optional<VisionWeightsFor<VisionCfg>> vision;
};

} // namespace family
} // namespace sinfer
