#pragma once

#include <api/family/runtime.h>

#include "api/ops/sparse_moe.h"

namespace sinfer::family {

/// The routed-expert routing a round of `phase` asks the sparse-MoE op for.
///
/// A prefill round's width is not the prompt's own: the packing window splits a prompt where its
/// round-mates leave off, and a prompt's remainder can run alone in a round of a few tokens. With
/// the op's width-chosen kernels those tokens took the small-T or decode arithmetic instead of the
/// wide route's, and the prompt's answer moved by a few ulps (or, for a decisions request's shared
/// prefix, every question's did). Prefill rounds therefore pin the wide route at every width
/// (ops::SparseMoeRouting::WidthInvariant), the decode rows a mixed round carries included;
/// verify and decode rounds keep the width-chosen kernels. The workspace a phase plans must be
/// sized with the same routing.
[[nodiscard]] constexpr ops::SparseMoeRouting moe_routing(TextPhase phase) noexcept {
    return phase == TextPhase::Prefill ? ops::SparseMoeRouting::WidthInvariant
                                       : ops::SparseMoeRouting::ByWidth;
}

} // namespace sinfer::family
