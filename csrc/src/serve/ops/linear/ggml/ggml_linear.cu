#include "ops/linear/ggml/ggml_linear.h"

#include "ops/linear/bf16/bf16_cublaslt.h"
#include "ops/linear/ggml/ggml_dequant.h"
#include "ops/linear/ggml/ggml_q8_1.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace sinfer::ops::detail::ggml {
namespace {

// The dequantisation tile's budget. A row tile is expanded to BF16 once and consumed by one
// cuBLASLt call, so the weight is read from memory a single time however wide the batch is --
// the chunked GEMV re-reads it every eight columns, which is what made prefill the slow half.
// 32 MiB covers whole weights at every geometry the tree serves (the widest, a 27B gate_up at
// [34816, 5120], tiles in eleven passes).
constexpr std::size_t kDequantTileBytes = 32u << 20;

std::int32_t rows_per_tile(std::int32_t rows, std::int32_t k) noexcept {
    const std::size_t row_bytes = static_cast<std::size_t>(k) * sizeof(__nv_bfloat16);
    std::size_t fit             = kDequantTileBytes / std::max<std::size_t>(row_bytes, 1);
    fit                         = std::max<std::size_t>(fit & ~std::size_t{7}, 8); // whole 8-row groups
    return static_cast<std::int32_t>(std::min<std::size_t>(fit, static_cast<std::size_t>(rows)));
}

// A wide batch runs the tensor-core route only when the shapes are ones cuBLASLt accepts.
bool wide_route_admits(std::int32_t rows, std::int32_t k, std::int32_t tokens) noexcept {
    return tokens >= wide_min_tokens() && (rows % 8) == 0 && (k % 8) == 0;
}

template <typename DstT, bool Accumulate>
void run_gemv(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
              const __nv_bfloat16* x, std::int32_t tokens, DstT* out, void* scratch,
              cudaStream_t stream) {
    auto* y = static_cast<block_q8_1*>(scratch);
    quantize_q8_1_launch(x, k, tokens, y, stream);
    const std::int32_t blocks_per_token = k / QK8_1;
    for (std::int32_t column = 0; column < tokens; column += kMmvqMaxColumns) {
        const std::int32_t width = std::min(kMmvqMaxColumns, tokens - column);
        mmvq_launch<DstT, Accumulate>(type, blocks,rows, k,
                                      y + static_cast<std::size_t>(column) * blocks_per_token, width,
                                      out + static_cast<std::size_t>(column) * rows, stream);
    }
}

void run_wide(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
              const __nv_bfloat16* x, std::int32_t tokens, __nv_bfloat16* out, void* scratch,
              float beta, cudaStream_t stream) {
    const std::int32_t tile   = rows_per_tile(rows, k);
    const std::size_t row_bytes = static_cast<std::size_t>(k / block_values(type)) * block_bytes(type);
    auto* staging             = static_cast<__nv_bfloat16*>(scratch);
    for (std::int32_t first = 0; first < rows; first += tile) {
        const std::int32_t height = std::min(tile, rows - first);
        const auto* tile_blocks =
            static_cast<const std::byte*>(blocks) + static_cast<std::size_t>(first) * row_bytes;
        dequantize_rows_launch(type, tile_blocks, height, k, staging, stream);
        bf16_cublaslt_gemm_raw(staging, height, k, x, tokens, out + first, rows, beta, stream);
    }
}

template <bool Accumulate>
void run_bf16(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
              const __nv_bfloat16* x, std::int32_t tokens, __nv_bfloat16* out, void* scratch,
              std::size_t scratch_bytes, cudaStream_t stream) {
    if (tokens <= 0 || k <= 0 || (k % block_values(type)) != 0 || rows <= 0) {
        throw std::invalid_argument("ggml linear: W[rows, k] with k a multiple of 256");
    }
    if (scratch == nullptr || scratch_bytes < linear_workspace_bytes(rows, k, tokens) ||
        (reinterpret_cast<std::uintptr_t>(scratch) & 15u) != 0) {
        throw std::invalid_argument("ggml linear: scratch too small or misaligned");
    }
    if (wide_route_admits(rows, k, tokens)) {
        run_wide(type, blocks, rows, k, x, tokens, out, scratch, Accumulate ? 1.0F : 0.0F, stream);
        return;
    }
    run_gemv<__nv_bfloat16, Accumulate>(type, blocks, rows, k, x, tokens, out, scratch, stream);
}

} // namespace

std::int32_t wide_min_tokens() noexcept {
    static const std::int32_t value = [] {
        // Bytes per parameter: the GEMV route re-reads the weight every eight columns, so it
        // moves 0.56*ceil(T/8) for a Q4_K; the wide route always moves ~4.6 (read the codes,
        // write the BF16 tile, read it back in the GEMM). They cross near T = 64, and above it
        // the wide route also has the tensor cores. Tunable because the crossover depends on
        // how much of the weight the L2 holds, which is a per-card property.
        const char* env = std::getenv("SUROGATE_GGML_WIDE_MIN_TOKENS");
        if (env == nullptr) { return std::int32_t{65}; }
        const int parsed = std::atoi(env);
        return parsed > 0 ? static_cast<std::int32_t>(parsed) : std::int32_t{65};
    }();
    return value;
}

std::size_t linear_workspace_bytes(std::int32_t rows, std::int32_t k, std::int32_t tokens) noexcept {
    if (rows <= 0 || k <= 0 || tokens <= 0) { return 0; }
    const std::size_t gemv = q8_1_bytes(k, std::min(tokens, kMmvqMaxColumns * 1024));
    if (!wide_route_admits(rows, k, tokens)) { return gemv; }
    return static_cast<std::size_t>(rows_per_tile(rows, k)) * k * sizeof(__nv_bfloat16);
}

void linear_launch(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
                   const __nv_bfloat16* x, std::int32_t tokens, __nv_bfloat16* out, void* scratch,
                   std::size_t scratch_bytes, cudaStream_t stream) {
    run_bf16<false>(type, blocks, rows, k, x, tokens, out, scratch, scratch_bytes, stream);
}

void linear_add_launch(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
                       const __nv_bfloat16* x, std::int32_t tokens, __nv_bfloat16* residual,
                       void* scratch, std::size_t scratch_bytes, cudaStream_t stream) {
    run_bf16<true>(type, blocks, rows, k, x, tokens, residual, scratch, scratch_bytes, stream);
}

void linear_launch_f32(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
                       const __nv_bfloat16* x, std::int32_t tokens, float* out, void* scratch,
                       std::size_t scratch_bytes, cudaStream_t stream) {
    if (tokens <= 0 || k <= 0 || (k % block_values(type)) != 0 || rows <= 0) {
        throw std::invalid_argument("ggml linear: W[rows, k] with k a multiple of 256");
    }
    if (scratch == nullptr || scratch_bytes < q8_1_bytes(k, tokens) ||
        (reinterpret_cast<std::uintptr_t>(scratch) & 15u) != 0) {
        throw std::invalid_argument("ggml linear: scratch too small or misaligned");
    }
    run_gemv<float, false>(type, blocks, rows, k, x, tokens, out, scratch, stream);
}

} // namespace sinfer::ops::detail::ggml
