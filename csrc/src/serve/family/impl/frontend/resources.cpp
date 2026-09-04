#include <api/family/frontend_resources.h>

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <string>

namespace sinfer::family {
namespace {

std::string take_string(artifact::MaterializedArtifact& materialized,
                        artifact::ObjectHandle handle) {
    const auto bytes = materialized.take_resource_bytes(handle);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

} // namespace

/// A base model publishes no chat template, so the object is bound when it is there and the
/// handle is left naming nothing when it is not. Binding it unconditionally refused every base
/// checkpoint at load, for a resource only the chat endpoints read.
artifact::ObjectHandle bind_optional_chat_template(artifact::Binder& binder) {
    if (!binder.has("frontend/chat_template.jinja")) { return artifact::ObjectHandle{}; }
    return artifact::bind_raw_resource(binder, "frontend/chat_template.jinja");
}

std::string take_chat_template(artifact::MaterializedArtifact& materialized,
                               const FrontendResourcePlan& plan) {
    if (!plan.has_chat_template) { return {}; }
    return take_string(materialized, plan.chat_template_jinja);
}

FrontendResourcePlan bind_text_only_frontend_resources(artifact::Binder& binder) {
    FrontendResourcePlan plan;
    plan.tokenizer_json = artifact::bind_raw_resource(binder, "frontend/tokenizer.json");
    plan.tokenizer_config_json =
        artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json");
    plan.chat_template_jinja = bind_optional_chat_template(binder);
    plan.has_chat_template   = binder.has("frontend/chat_template.jinja");
    plan.generation_config_json =
        artifact::bind_raw_resource(binder, "frontend/generation_config.json");
    return plan;
}

FrontendResources take_text_only_frontend_resources(artifact::MaterializedArtifact& materialized,
                                                    const FrontendResourcePlan& plan) {
    FrontendResources out;
    out.tokenizer_json         = take_string(materialized, plan.tokenizer_json);
    out.tokenizer_config_json  = take_string(materialized, plan.tokenizer_config_json);
    out.chat_template_jinja    = take_chat_template(materialized, plan);
    out.generation_config_json = take_string(materialized, plan.generation_config_json);
    return out;
}

FrontendResourcePlan bind_frontend_resources(artifact::Binder& binder) {
    // A text-only release of a vision family carries no pixel processor configs; the
    // handles are then left naming nothing and the frontend sees empty strings.
    const bool pixels = binder.has("frontend/preprocessor_config.json") &&
                        binder.has("frontend/video_preprocessor_config.json");
    return FrontendResourcePlan{
        .tokenizer_json = artifact::bind_raw_resource(binder, "frontend/tokenizer.json"),
        .tokenizer_config_json =
            artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json"),
        .chat_template_jinja = bind_optional_chat_template(binder),
        .has_chat_template   = binder.has("frontend/chat_template.jinja"),
        .generation_config_json =
            artifact::bind_raw_resource(binder, "frontend/generation_config.json"),
        .preprocessor_config_json =
            pixels ? artifact::bind_raw_resource(binder, "frontend/preprocessor_config.json")
                   : artifact::ObjectHandle{},
        .video_preprocessor_config_json =
            pixels ? artifact::bind_raw_resource(binder, "frontend/video_preprocessor_config.json")
                   : artifact::ObjectHandle{},
        .has_preprocessor_configs = pixels,
    };
}

FrontendResources take_frontend_resources(artifact::MaterializedArtifact& materialized,
                                          const FrontendResourcePlan& plan) {
    return FrontendResources{
        .tokenizer_json           = take_string(materialized, plan.tokenizer_json),
        .tokenizer_config_json    = take_string(materialized, plan.tokenizer_config_json),
        .chat_template_jinja      = take_chat_template(materialized, plan),
        .generation_config_json   = take_string(materialized, plan.generation_config_json),
        .preprocessor_config_json = plan.has_preprocessor_configs
                                        ? take_string(materialized, plan.preprocessor_config_json)
                                        : std::string{},
        .video_preprocessor_config_json =
            plan.has_preprocessor_configs
                ? take_string(materialized, plan.video_preprocessor_config_json)
                : std::string{},
    };
}

} // namespace sinfer::family
