#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"

#include <api/targets/qwen3_6/vision_control.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace sinfer::targets::qwen3_6::detail::SINFER_QWEN36_RUNTIME_NS {

struct VisionUseSpan {
    std::uint32_t begin      = 0;
    std::uint32_t end        = 0;
    std::uint32_t item_index = 0;
};

struct VisionPrefillPlan {
    std::shared_ptr<const qwen3_6::VisionControl> control;
    std::vector<VisionUseSpan> uses;
};

} // namespace sinfer::targets::qwen3_6::detail::SINFER_QWEN36_RUNTIME_NS
