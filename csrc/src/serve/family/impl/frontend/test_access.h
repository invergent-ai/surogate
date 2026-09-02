#pragma once

#include <api/family/frontend.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

namespace sinfer::family {

class FrontendTestAccess {
public:
    [[nodiscard]] static Frontend create_component(const FrontendResources& resources,
                                                   bool vision_enabled = true);
    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
    /// Runs the artifact-resource validation the Frontend constructor runs, so
    /// the tokenizer_config.json contract is testable from three JSON strings
    /// without a tokenizer. Named apart from the function it calls: an
    /// unqualified call from a member body would find this declaration first.
    static void check_tokenizer_config(const FrontendResources& resources);
};

} // namespace sinfer::family
