#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"

#include <ninfer/targets/qwen3_6/vision_control.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

struct VisionUseSpan {
    std::uint32_t begin      = 0;
    std::uint32_t end        = 0;
    std::uint32_t item_index = 0;
};

struct VisionPrefillPlan {
    std::shared_ptr<const qwen3_6::VisionControl> control;
    std::vector<VisionUseSpan> uses;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
