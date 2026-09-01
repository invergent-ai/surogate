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
};

} // namespace sinfer::family
