#pragma once

#include "runtime/contract/types.h"

#include <cstddef>

namespace sinfer::runtime {

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

} // namespace sinfer::runtime
