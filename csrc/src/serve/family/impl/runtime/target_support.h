#pragma once

// Support every target leaf needs and none of them owns.
//
// A target package is meant to hold what is specific to one model. These are
// not: each was copied into every target and the copies are byte-identical, so
// a change to any of them is an edit in nine files that the compiler will not
// remind anyone to make.

#include <api/family/runtime.h>

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sinfer::family {

/// Cuts [0, max_frontier] into graph execution profiles at the given ends.
///
/// Each profile runs from where the last one stopped to its preferred end, or
/// to max_frontier if that comes first; whatever is left over becomes a final
/// profile. A target supplies the ends it wants captured and nothing else.
[[nodiscard]] inline std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

/// The token interval a workspace-capacity leaf is asked to size for.
inline void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

/// Writes one residual tensor to the parity-dump directory, if one is set.
///
/// Off unless SUROGATE_SERVE_DUMP_RESIDUAL names a directory. The dumps exist to
/// bisect a wrong-output bug against an independent reference, so the selection
/// rules matter as much as the write: only prompt-sized rounds are captured,
/// because a long generation would otherwise write a file per decode step per
/// layer; and the round is chosen by width rather than by "first N", because
/// warmup runs forwards of its own before any request and would otherwise spend
/// the budget before the round under study arrives. The occurrence counter is
/// kept per (tag, width) so the captured round's layers number 0..N-1 whatever
/// ran before them.
///
/// `magic` is the only thing that differs between targets: the header is
/// {magic, rows, columns, occurrence} little-endian int32, and the magic names
/// which target wrote the file so a dump directory cannot be misattributed.
void debug_probe_dump(std::int32_t magic, const char* tag, const Tensor& tensor,
                      std::int32_t layer_count, cudaStream_t stream);

} // namespace sinfer::family
