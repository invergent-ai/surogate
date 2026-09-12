#include "ops/linear/ggml/ggml_linear.h"

#include "core/device.h"

#include "ops/linear/bf16/bf16_cublaslt.h"
#include "ops/linear/ggml/ggml_dequant.h"
#include "ops/linear/ggml/ggml_dense_decode.cuh"
#include "ops/linear/ggml/ggml_i8_tile.cuh"
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

// Keep the activation quantisation used by GEMV when a graph pads a short
// prompt into a wide batch. Expanding to BF16 changed both activation and weight
// rounding at that boundary, producing different prompt and decode scores.
template <class Codec, int TileRows, int TileCols>
__global__ __launch_bounds__((TileCols / 8) * 32) void dense_i8_kernel(
    const std::uint8_t* blocks, const std::int8_t* codes, const __half2* ds,
    int rows, int k, int tokens, __nv_bfloat16* out, bool accumulate) {
    __shared__ GgmlI8Smem<Codec, TileCols, TileRows> sm;
    const int row0 = blockIdx.x * TileRows;
    const int col0 = blockIdx.y * TileCols;
    auto row_base = [&](int row) {
        return blocks + static_cast<std::int64_t>(min(row0 + row, rows - 1)) *
                            (k / QK_K) * Codec::kBlockBytes;
    };
    auto act_row = [&](int col) { return static_cast<std::int64_t>(col0 + col); };
    float acc[TileRows / 16][4];
    ggml_i8_tile<Codec, TileCols / 8, TileCols, true, TileRows>(
        sm, row_base, act_row, codes, ds, min(TileCols, tokens - col0), k, acc);
    const int lane = threadIdx.x & 31;
    const int col  = col0 + (threadIdx.x >> 5) * 8 + (lane & 3) * 2;
#pragma unroll
    for (int mi = 0; mi < TileRows / 16; ++mi) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = row0 + mi * 16 + (lane >> 2) + (e / 2) * 8;
            const int t   = col + (e & 1);
            if (row < rows && t < tokens) {
                const auto i = static_cast<std::int64_t>(t) * rows + row;
                const float base = accumulate ? __bfloat162float(out[i]) : 0.0f;
                out[i] = __float2bfloat16_rn(acc[mi][e] + base);
            }
        }
    }
}

void run_wide(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
              const __nv_bfloat16* x, std::int32_t tokens, __nv_bfloat16* out, void* scratch,
              float beta, cudaStream_t stream) {
    if (type == GgmlType::Q4_K || type == GgmlType::Q5_K || type == GgmlType::Q6_K) {
        auto* codes = static_cast<std::int8_t*>(scratch);
        auto* ds = reinterpret_cast<__half2*>(codes + static_cast<std::size_t>(tokens) * k);
        quantize_q8_1_planes_launch(x, k, tokens, codes, ds, stream);
        const auto launch = [&]<class Codec>() {
            // Re-reading the weight for every column pays off only for narrow
            // batches; larger projections reach that crossover sooner.
            if (tokens <= 2 || (tokens <= 4 && rows <= 4096)) {
                dense_decode_kernel<Codec><<<dim3(rows, tokens), 32, 0, stream>>>(
                    static_cast<const typename Codec::Block*>(blocks), codes, ds, rows, k,
                    out, beta != 0.0f);
                return;
            }
            const auto tile = [&]<int TileRows, int TileCols>() {
                const dim3 grid((rows + TileRows - 1) / TileRows,
                                (tokens + TileCols - 1) / TileCols);
                dense_i8_kernel<Codec, TileRows, TileCols>
                    <<<grid, (TileCols / 8) * 32, 0, stream>>>(
                        static_cast<const std::uint8_t*>(blocks), codes, ds, rows, k,
                        tokens, out, beta != 0.0f);
            };
            if (tokens <= 8) { tile.template operator()<16, 8>(); }
            else if (tokens <= 32) { tile.template operator()<32, 32>(); }
            else { tile.template operator()<64, 32>(); }
        };
        switch (type) {
        case GgmlType::Q4_K: launch.template operator()<GgmlQ4KPrefill>(); break;
        case GgmlType::Q5_K: launch.template operator()<GgmlQ5KPrefill>(); break;
        case GgmlType::Q6_K: launch.template operator()<GgmlQ6KPrefill>(); break;
        default: break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
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
    if (type == GgmlType::Q4_K || type == GgmlType::Q5_K || type == GgmlType::Q6_K ||
        wide_route_admits(rows, k, tokens)) {
        run_wide(type, blocks, rows, k, x, tokens, out, scratch, Accumulate ? 1.0F : 0.0F, stream);
        return;
    }
    run_gemv<__nv_bfloat16, Accumulate>(type, blocks, rows, k, x, tokens, out, scratch, stream);
}

} // namespace

std::int32_t wide_min_tokens() noexcept {
    static const std::int32_t value = [] {
        // Keep the existing crossover while Q4_K/Q5_K/Q6_K switch to the integer
        // tensor-core tile. Other formats still expand a bounded BF16 weight tile.
        const char* env = std::getenv("SUROGATE_GGML_WIDE_MIN_TOKENS");
        if (env == nullptr) { return std::int32_t{65}; }
        const int parsed = std::atoi(env);
        return parsed > 0 ? static_cast<std::int32_t>(parsed) : std::int32_t{65};
    }();
    return value;
}

std::size_t linear_workspace_bytes(std::int32_t rows, std::int32_t k, std::int32_t tokens) noexcept {
    if (rows <= 0 || k <= 0 || tokens <= 0) { return 0; }
    // Both quantisers write every column before the projection consumes it.
    const std::size_t gemv = q8_1_bytes(k, tokens);
    if (!wide_route_admits(rows, k, tokens)) { return gemv; }
    return std::max(gemv, static_cast<std::size_t>(rows_per_tile(rows, k)) * k * sizeof(__nv_bfloat16));
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

namespace {

__global__ void permute_column_groups_kernel(const __nv_bfloat16* __restrict__ x, int k, int tokens,
                                             const std::int32_t* __restrict__ group_map,
                                             __nv_bfloat16* __restrict__ out) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(k) * tokens;
    if (i >= total) { return; }
    const int t     = static_cast<int>(i / k);
    const int row   = static_cast<int>(i - static_cast<std::size_t>(t) * k);
    const int group = row >> 5;
    const int lane  = row & 31;
    out[static_cast<std::size_t>(t) * k + group_map[group] * 32 + lane] = x[i];
}

} // namespace

void permute_column_groups_launch(const __nv_bfloat16* x, std::int32_t k, std::int32_t tokens,
                                  const std::int32_t* group_map, __nv_bfloat16* out,
                                  cudaStream_t stream) {
    if (k <= 0 || tokens <= 0 || (k % 32) != 0 || group_map == nullptr) {
        throw std::invalid_argument("permute_column_groups: k a multiple of 32 and a map");
    }
    const std::size_t total = static_cast<std::size_t>(k) * tokens;
    const unsigned blocks   = static_cast<unsigned>((total + 255) / 256);
    permute_column_groups_kernel<<<blocks, 256, 0, stream>>>(x, k, tokens, group_map, out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail::ggml
