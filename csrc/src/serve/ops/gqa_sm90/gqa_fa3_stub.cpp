// The build's architecture set has no 90a, so it carries no FlashAttention-3: attention stays on
// the split-KV tile kernels, and every entry point but the queries says so.

#include "ops/gqa_sm90/gqa_fa3.h"

#include <stdexcept>

namespace sinfer::ops::detail::gqa_fa3 {
namespace {

[[noreturn]] void unavailable() {
    throw std::runtime_error(
        "gqa_fa3: FlashAttention-3 needs the sm_90a architecture in SUROGATE_SERVE_CUDA_ARCHS "
        "and an H100/H200");
}

} // namespace

bool available() noexcept { return false; }

bool supports(std::int32_t, std::int32_t, std::int32_t) noexcept { return false; }

std::int32_t min_columns() noexcept { return 32; }

bool rows_enabled() noexcept { return false; }

void prompt_metadata(const std::int32_t*, std::int32_t, const std::int32_t*, std::int32_t*,
                     cudaStream_t) {
    unavailable();
}

RowSplits row_splits(std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t,
                     std::int32_t, bool, std::int32_t) noexcept {
    return {};
}

void rows_metadata(const std::int32_t*, std::int32_t, std::int32_t, const std::int32_t*,
                   const std::int32_t*, std::int32_t*, cudaStream_t) {
    unavailable();
}

void segment_kv_lengths(const std::int32_t*, const std::int32_t*, const std::int32_t*,
                        std::int32_t, std::int32_t*, cudaStream_t) {
    unavailable();
}

void run(const PagedLaunch&, void*, std::size_t, cudaStream_t) { unavailable(); }

} // namespace sinfer::ops::detail::gqa_fa3
