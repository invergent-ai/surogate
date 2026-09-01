#pragma once
#include "family/impl/runtime/instance.h"

#include <api/family/vision_control.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS {

struct VisionUseSpan {
    std::uint32_t begin      = 0;
    std::uint32_t end        = 0;
    std::uint32_t item_index = 0;
};

struct VisionPrefillPlan {
    std::shared_ptr<const family::VisionControl> control;
    std::vector<VisionUseSpan> uses;
};

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS
