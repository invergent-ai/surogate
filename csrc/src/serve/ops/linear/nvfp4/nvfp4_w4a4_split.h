#pragma once

#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace sinfer::ops::detail {

// The TMA schedule consumes whole kNvfp4TmaBlockM-token blocks. A serving round is almost
// never block-aligned (a prefill chunk shares the GEMM with the decode batch), so the aligned
// prefix runs on the TMA path and the ragged tail on the mma ladder.
constexpr std::int32_t kNvfp4TmaBlockM = 256;

struct Nvfp4W4a4TmaSplit {
    std::int32_t tma_tokens;  // multiple of kNvfp4TmaBlockM, possibly zero
    std::int32_t tail_tokens; // tokens - tma_tokens
};

// Smallest aligned prefix worth handing to the TMA schedule; below it the whole problem
// stays on the mma ladder. SUROGATE_SERVE_NVFP4_TMA_MIN_TOKENS overrides the default.
inline std::int32_t nvfp4_w4a4_tma_floor() {
    static const std::int32_t floor = [] {
        constexpr std::int32_t kDefault = 2 * kNvfp4TmaBlockM;
        const char* raw = std::getenv("SUROGATE_SERVE_NVFP4_TMA_MIN_TOKENS");
        if (raw == nullptr || *raw == '\0') { return kDefault; }
        const long parsed = std::strtol(raw, nullptr, 10);
        return parsed > 0 ? static_cast<std::int32_t>(parsed) : kDefault;
    }();
    return floor;
}

inline Nvfp4W4a4TmaSplit nvfp4_w4a4_tma_split(std::int32_t tokens) {
    const std::int32_t aligned = (tokens / kNvfp4TmaBlockM) * kNvfp4TmaBlockM;
    if (aligned < nvfp4_w4a4_tma_floor()) { return {0, tokens}; }
    return {aligned, tokens - aligned};
}

// Activation workspace and output columns are token-major, so a tail view is a base offset.
inline Nvfp4W4a4Workspace nvfp4_w4a4_workspace_at(Nvfp4W4a4Workspace workspace,
                                                  std::int32_t input_rows, std::int32_t token) {
    return {workspace.codes + static_cast<std::ptrdiff_t>(token) * (input_rows / 2),
            workspace.scales + static_cast<std::ptrdiff_t>(token) * (input_rows / 16)};
}

inline __nv_bfloat16* nvfp4_w4a4_column(void* data, std::int32_t rows, std::int32_t token) {
    return static_cast<__nv_bfloat16*>(data) + static_cast<std::int64_t>(token) * rows;
}

} // namespace sinfer::ops::detail
