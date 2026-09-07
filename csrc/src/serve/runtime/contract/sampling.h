#pragma once

#include "api/types.h"

namespace sinfer::runtime {

// Resolves one request at the Engine boundary. The registered preset supplies every omitted
// model-owned field; an omitted seed remains deterministic for direct Engine callers.
[[nodiscard]] ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                                          SamplingMode mode,
                                                          const SamplingOverrides& overrides);

} // namespace sinfer::runtime
