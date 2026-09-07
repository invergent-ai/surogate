#include "api/engine.h"

#include <type_traits>

static_assert(std::is_move_constructible_v<sinfer::PreparedPrompt>);
static_assert(!std::is_copy_constructible_v<sinfer::PreparedPrompt>);
static_assert(std::is_move_constructible_v<sinfer::Engine>);
static_assert(!std::is_copy_constructible_v<sinfer::Engine>);

int main() {
    const sinfer::EngineOptions options;
    return options.enable_vision ? 1 : 0;
}
