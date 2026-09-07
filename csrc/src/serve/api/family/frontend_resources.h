#pragma once

#include "artifact/binder.h"

#include <api/family/frontend.h>

#include <string>

namespace sinfer::artifact {
class MaterializedArtifact;
}

namespace sinfer::family {

struct FrontendResourcePlan {
    artifact::ObjectHandle tokenizer_json;
    artifact::ObjectHandle tokenizer_config_json;
    artifact::ObjectHandle chat_template_jinja;
    /// Whether the artifact carried one. A base model does not, and `chat_template_jinja`
    /// then names nothing -- the handle is an index, so absence needs its own bit.
    bool has_chat_template = true;
    artifact::ObjectHandle generation_config_json;
    artifact::ObjectHandle preprocessor_config_json;
    artifact::ObjectHandle video_preprocessor_config_json;
    /// Whether the artifact carried the pixel processor configs. A text-only release of a
    /// vision family does not; the frontend reads the empty strings as "never asked for a
    /// pixel" and refuses --vision at load.
    bool has_preprocessor_configs = true;
};

struct FrontendResources {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string chat_template_jinja;
    std::string generation_config_json;
    std::string preprocessor_config_json;
    std::string video_preprocessor_config_json;
};

/// Binds the four resources every target has. A vision-capable target uses the
/// six-resource pair below instead; a text-only one that bound all six would be
/// asking the artifact for a preprocessor config it does not carry.
[[nodiscard]] FrontendResourcePlan bind_text_only_frontend_resources(artifact::Binder& binder);
[[nodiscard]] FrontendResources
take_text_only_frontend_resources(artifact::MaterializedArtifact& artifact,
                                  const FrontendResourcePlan& plan);

[[nodiscard]] FrontendResourcePlan bind_frontend_resources(artifact::Binder& binder);
[[nodiscard]] FrontendResources take_frontend_resources(artifact::MaterializedArtifact& artifact,
                                                        const FrontendResourcePlan& plan);

} // namespace sinfer::family
