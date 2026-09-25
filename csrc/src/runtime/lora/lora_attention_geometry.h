// Shape-driven attention LoRA allocation; shared with host regression tests.
#pragma once
#include <array>
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
/// Index of a linear-attention (GatedDeltaNet) LoRA target in kLinearAttentionLoRANames
/// order, or -1. DSL names: lin_qkv, lin_z, lin_a, lin_b, lin_out.
inline int linear_target_index(std::string_view name, bool grouped) {
    if (grouped) return -1;
    constexpr std::array<std::string_view, 5> names = {"lin_qkv", "lin_z", "lin_a", "lin_b", "lin_out"};
    for (int i = 0; i < 5; ++i)
        if (names[static_cast<std::size_t>(i)] == name) return i;
    return -1;
}
inline void record_linear_geometry(LoRALinearShapes& shapes, int index,
                                   long input, long total_output, int offset,
                                   int declared_size, const std::string& label) {
    const long output = declared_size > 0 ? declared_size : total_output - offset;
    const auto max_int = std::numeric_limits<int>::max();
    if (index < 0 || index >= 5 || input <= 0 || input > max_int ||
        total_output <= 0 || total_output > max_int || offset < 0 ||
        declared_size < 0 || output <= 0 || output > total_output - offset)
        throw std::invalid_argument("Invalid linear-attention LoRA geometry: " + label);
    auto& shape = shapes[static_cast<std::size_t>(index)];
    if (shape.input != 0 || shape.output != 0)
        throw std::invalid_argument("Duplicate linear-attention LoRA target: " + label);
    shape = {static_cast<int>(input), static_cast<int>(output)};
}
} // namespace modules
