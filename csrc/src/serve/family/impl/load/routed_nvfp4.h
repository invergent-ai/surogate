#pragma once

// Routed NVFP4 experts, as every mixture target reads them.
//
// A routed NVFP4 bank carries its second level and its activation divisors per expert, in
// objects of their own, so the payload's per-tensor divisor word must be the identity and the
// Weight the kernels read has both of its own divisors at one. These were three identical
// copies in three targets; a mixture is a mixture.

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"

#include <cstdint>
#include <string_view>

namespace sinfer::family {

/// One routed NVFP4 matrix as the MoE kernels read it.
///
/// `artifact::materialized_weight` refuses NVFP4 on purpose: the dense path pairs every NVFP4
/// weight with an activation divisor, and building one without that pairing would silently drop
/// a scale. The routed experts carry theirs per expert, so this builds the Weight directly
/// rather than borrowing a contract that does not apply.
Weight routed_nvfp4_weight(const artifact::MaterializedArtifact& materialized,
                           artifact::ObjectHandle handle, std::int32_t rows, std::int32_t columns);

/// The payload's per-tensor divisor word must be the identity: a routed artifact carries its
/// second level per expert, and a per-tensor value here would be dropped on the floor.
void require_identity_divisor(const artifact::Binder& binder, artifact::ObjectHandle handle,
                              std::string_view name, std::int32_t rows, std::int32_t columns);

} // namespace sinfer::family
