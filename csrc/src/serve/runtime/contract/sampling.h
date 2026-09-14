#pragma once

#include "api/types.h"

#include <string_view>

namespace sinfer::runtime {

// Explicit non-null generation settings override each mode's family preset. Missing fields
// retain the family recommendation; malformed supported settings fail at model load.
[[nodiscard]] ModelSamplingDefaults load_sampling_defaults(
    ModelSamplingDefaults family_defaults, std::string_view generation_config_json);

// Resolves one request at the Engine boundary. The model preset supplies every omitted
// model-owned field; an omitted seed remains deterministic for direct Engine callers.
[[nodiscard]] ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                                          SamplingMode mode,
                                                          const SamplingOverrides& overrides);

} // namespace sinfer::runtime
