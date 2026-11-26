#pragma once

// Registration side of api/targets/inkling2/decision.h (the package installs the runners at load).

#include "targets/inkling2/impl/model.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace sinfer::targets::inkling2::detail::decision {

/// Builds a runner for the adapter at `dir` over the served base, on the current device, under `name`.
void install(const std::string& name, const LoadedModelData& base, const std::filesystem::path& dir,
             std::int32_t max_tokens);

} // namespace sinfer::targets::inkling2::detail::decision
