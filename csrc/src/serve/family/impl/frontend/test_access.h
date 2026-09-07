#pragma once

#include <api/family/frontend.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include <string_view>
#include <vector>

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
    /// Tokenises with a tokenizer built from the given resources. The pre-tokenizer's digit
    /// rule is read from `tokenizer.json`, and it is the one thing that differs between the
    /// checkpoints this family serves, so it is worth a test that needs no checkpoint.
    [[nodiscard]] static std::vector<int> encode_with(const FrontendResources& resources,
                                                     std::string_view text);
};

} // namespace sinfer::family
