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

FrontendResourcePlan bind_text_only_frontend_resources(artifact::Binder& binder) {
    FrontendResourcePlan plan;
    plan.tokenizer_json = artifact::bind_raw_resource(binder, "frontend/tokenizer.json");
    plan.tokenizer_config_json =
        artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json");
    plan.chat_template_jinja = artifact::bind_raw_resource(binder, "frontend/chat_template.jinja");
    plan.generation_config_json =
        artifact::bind_raw_resource(binder, "frontend/generation_config.json");
    return plan;
}

FrontendResources take_text_only_frontend_resources(artifact::MaterializedArtifact& materialized,
                                                    const FrontendResourcePlan& plan) {
    FrontendResources out;
    out.tokenizer_json         = take_string(materialized, plan.tokenizer_json);
    out.tokenizer_config_json  = take_string(materialized, plan.tokenizer_config_json);
    out.chat_template_jinja    = take_string(materialized, plan.chat_template_jinja);
    out.generation_config_json = take_string(materialized, plan.generation_config_json);
    return out;
}

FrontendResourcePlan bind_frontend_resources(artifact::Binder& binder) {
    return FrontendResourcePlan{
        .tokenizer_json = artifact::bind_raw_resource(binder, "frontend/tokenizer.json"),
        .tokenizer_config_json =
            artifact::bind_raw_resource(binder, "frontend/tokenizer_config.json"),
        .chat_template_jinja = artifact::bind_raw_resource(binder, "frontend/chat_template.jinja"),
        .generation_config_json =
            artifact::bind_raw_resource(binder, "frontend/generation_config.json"),
        .preprocessor_config_json =
            artifact::bind_raw_resource(binder, "frontend/preprocessor_config.json"),
        .video_preprocessor_config_json =
            artifact::bind_raw_resource(binder, "frontend/video_preprocessor_config.json"),
    };
}

FrontendResources take_frontend_resources(artifact::MaterializedArtifact& materialized,
                                          const FrontendResourcePlan& plan) {
    return FrontendResources{
        .tokenizer_json           = take_string(materialized, plan.tokenizer_json),
        .tokenizer_config_json    = take_string(materialized, plan.tokenizer_config_json),
        .chat_template_jinja      = take_string(materialized, plan.chat_template_jinja),
        .generation_config_json   = take_string(materialized, plan.generation_config_json),
        .preprocessor_config_json = take_string(materialized, plan.preprocessor_config_json),
        .video_preprocessor_config_json =
            take_string(materialized, plan.video_preprocessor_config_json),
    };
}

} // namespace sinfer::family
