// Shape-driven attention LoRA allocation; shared with host regression tests.
#pragma once
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include "lora_types.h"

namespace modules {
inline int attention_target_index(std::string_view name, bool grouped) {
    if (grouped || name.size() != 1) return -1;
    const auto pos = std::string_view("qkvo").find(name);
    return pos == std::string_view::npos ? -1 : static_cast<int>(pos);
}
inline void record_attention_geometry(LoRAAttentionShapes& shapes, int index,
                                     long input, long total_output, int offset,
                                     int declared_size, const std::string& label) {
    // A zero declared size means the remaining output slice, exactly as in
    // resolve_and_validate_slice_size at LoRA forward/backward dispatch.
    const long output = declared_size > 0 ? declared_size : total_output - offset;
    const auto max_int = std::numeric_limits<int>::max();
    if (index < 0 || index >= 4 || input <= 0 || input > max_int ||
        total_output <= 0 || total_output > max_int || offset < 0 ||
        declared_size < 0 || output <= 0 || output > total_output - offset)
        throw std::invalid_argument("Invalid attention LoRA geometry: " + label);
    auto& shape = shapes[index];
    if (shape.input != 0 || shape.output != 0)
        throw std::invalid_argument("Duplicate attention LoRA target: " + label);
    shape = {static_cast<int>(input), static_cast<int>(output)};
}
} // namespace modules
