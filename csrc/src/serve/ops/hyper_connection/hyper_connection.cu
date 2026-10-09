#include "api/ops/hyper_connection.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/bf16/bf16_cublaslt.h"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

constexpr int kThreads    = 256;
constexpr int kMaxStreams = 8;
// The stream norm: eight values per 16-byte load, and two loads per thread in flight per pass,
// which covers a stream of up to 4,096 values in one pass.
constexpr int kNormVec    = 8;
constexpr int kNormChunks = 2;
// W8 projections take the mixer's own kernels up to this many tokens (decode, speculative
// verify); wider rounds dequantise the matrix to BF16 and keep cuBLASLt's tensor cores.
constexpr int kW8DirectTokens = 64;
constexpr int kW8TokenTile    = 8;
// The wide-K projection (down) is sliced so its few rows still spread over every SM.
constexpr int kW8SliceK       = 1280;

__device__ __forceinline__ float block_sum(float value, float* scratch) {
    // scratch holds one float per warp; the result is broadcast to every thread.
    value          = warp_reduce_sum(value);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    __syncthreads();
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float total = 0.0F;
    for (int w = 0; w < static_cast<int>(blockDim.x >> 5); ++w) { total += scratch[w]; }
    return total;
}

// One block per (token, stream): n = x * rsqrt(mean(x^2) + eps) * gamma, stored BF16 for the
// projections. A mixer also keeps the stream's inverse RMS for `finish_kernel` (`inv`, FP32
// [T, streams]) and, when it combines, this stream's share of each inject row's dot with n in
// FP32 (`inject_partial`, [T, streams (this one), streams (the row)]).
// A decode round launches only streams x tokens blocks, so the block's latency is the kernel's:
// each thread takes eight values per 16-byte load and issues every load of a pass (stream,
// gamma, inject rows) before using any, and the inject shares are reduced together. kStreams
// fixes the stream count at compile time (0: up to kMaxStreams at run time), which keeps the
// inject rows in flight within two blocks' worth of registers per SM.
template <bool kInject, int kStreams = 0>
__global__ void __launch_bounds__(kThreads, 2)
    stream_norm_kernel(const __nv_bfloat16* __restrict__ residual, const float* __restrict__ gamma,
                       const __nv_bfloat16* __restrict__ inject_weight, int hidden, int streams,
                       float eps, __nv_bfloat16* __restrict__ normalized,
                       float* __restrict__ inv_out, float* __restrict__ inject_partial) {
    __shared__ float scratch[kThreads / 32];
    constexpr int kRows = kStreams > 0 ? kStreams : kMaxStreams;
    __shared__ float warp_shares[kRows][kThreads / 32];
    const int token  = static_cast<int>(blockIdx.x) / streams;
    const int stream = static_cast<int>(blockIdx.x) - token * streams;
    const int width  = hidden * streams;
    const int chunks = hidden / kNormVec; // hidden % kNormVec == 0, checked by the launchers
    const int tid    = static_cast<int>(threadIdx.x);
    const std::int64_t base =
        static_cast<std::int64_t>(token) * width + static_cast<std::int64_t>(stream) * hidden;
    const __nv_bfloat16* x = residual + base;

    float sum_sq = 0.0F;
    for (int first = 0; first < chunks; first += kThreads * kNormChunks) {
        uint4 packed[kNormChunks];
#pragma unroll
        for (int c = 0; c < kNormChunks; ++c) {
            const int chunk = first + c * kThreads + tid;
            packed[c] = chunk < chunks ? *reinterpret_cast<const uint4*>(x + chunk * kNormVec)
                                       : make_uint4(0U, 0U, 0U, 0U);
        }
#pragma unroll
        for (int c = 0; c < kNormChunks; ++c) {
            const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed[c]);
#pragma unroll
            for (int i = 0; i < kNormVec / 2; ++i) {
                const float2 v = __bfloat1622float2(pairs[i]);
                sum_sq         = fmaf(v.x, v.x, sum_sq);
                sum_sq         = fmaf(v.y, v.y, sum_sq);
            }
        }
    }
    const float total = block_sum(sum_sq, scratch);
    const float inv   = rsqrtf(total / static_cast<float>(hidden) + eps);
    if (inv_out != nullptr && tid == 0) { inv_out[blockIdx.x] = inv; }

    float dots[kRows] = {};
    const float* stream_gamma = gamma + static_cast<std::int64_t>(stream) * hidden;
    for (int first = 0; first < chunks; first += kThreads * kNormChunks) {
        uint4 packed[kNormChunks];
        float4 scale[kNormChunks][2];
        uint4 rows[kInject ? kNormChunks : 1][kInject ? kRows : 1];
#pragma unroll
        for (int c = 0; c < kNormChunks; ++c) {
            const int chunk = first + c * kThreads + tid;
            if (chunk >= chunks) { continue; }
            const int d = chunk * kNormVec;
            packed[c]   = *reinterpret_cast<const uint4*>(x + d);
            scale[c][0] = *reinterpret_cast<const float4*>(stream_gamma + d);
            scale[c][1] = *reinterpret_cast<const float4*>(stream_gamma + d + 4);
            if constexpr (kInject) {
                const __nv_bfloat16* column =
                    inject_weight + static_cast<std::int64_t>(stream) * hidden + d;
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    if (kStreams > 0 || r < streams) {
                        rows[c][r] = *reinterpret_cast<const uint4*>(
                            column + static_cast<std::int64_t>(r) * width);
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < kNormChunks; ++c) {
            const int chunk = first + c * kThreads + tid;
            if (chunk >= chunks) { continue; }
            const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed[c]);
            const float g[kNormVec] = {scale[c][0].x, scale[c][0].y, scale[c][0].z, scale[c][0].w,
                                       scale[c][1].x, scale[c][1].y, scale[c][1].z, scale[c][1].w};
            float v[kNormVec];
            uint4 out;
            auto* out_pairs = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
            for (int i = 0; i < kNormVec / 2; ++i) {
                const float2 pair = __bfloat1622float2(pairs[i]);
                v[2 * i]          = pair.x * inv * g[2 * i];
                v[2 * i + 1]      = pair.y * inv * g[2 * i + 1];
                out_pairs[i]      = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
            }
            *reinterpret_cast<uint4*>(normalized + base + chunk * kNormVec) = out;
            if constexpr (kInject) {
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    if (kStreams > 0 || r < streams) {
                        const auto* weight = reinterpret_cast<const __nv_bfloat162*>(&rows[c][r]);
#pragma unroll
                        for (int i = 0; i < kNormVec / 2; ++i) {
                            const float2 w = __bfloat1622float2(weight[i]);
                            dots[r]        = fmaf(w.x, v[2 * i], dots[r]);
                            dots[r]        = fmaf(w.y, v[2 * i + 1], dots[r]);
                        }
                    }
                }
            }
        }
    }
    if constexpr (kInject) {
        const int lane = tid & 31;
        const int warp = tid >> 5;
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            if (kStreams > 0 || r < streams) {
                const float share = warp_reduce_sum(dots[r]);
                if (lane == 0) { warp_shares[r][warp] = share; }
            }
        }
        __syncthreads();
        if (tid < streams) {
            float share = 0.0F;
#pragma unroll
            for (int w = 0; w < kThreads / 32; ++w) { share += warp_shares[tid][w]; }
            inject_partial[static_cast<std::int64_t>(blockIdx.x) * streams + tid] = share;
        }
    }
}

// lo = silu(lo / streams), in place.
__global__ void silu_scale_kernel(__nv_bfloat16* __restrict__ values, std::int64_t count,
                                  float scale) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    values[i] = __float2bfloat16_rn(silu(__bfloat162float(values[i]) * scale));
}

// W8 (row-split, 32-value groups, FP16 scales) projections for rounds of at most
// kW8DirectTokens. One block covers rows_per_block rows over one K slice for up to TT tokens:
// the slice's activations are staged in shared memory so the block's rows read them once, LPR
// lanes share a row, and each lane takes eight codes at a time, so a load never straddles a
// scale group. A sliced projection writes each slice's FP32 dot to `partial` ([slices, T, n]);
// an unsliced one writes BF16 to `out` ([n, T]). Weights are rounded to BF16 before the dot, as
// every A16 W8 GEMM rounds them.
template <int LPR, int TT>
__global__ void __launch_bounds__(kThreads)
w8_projection_kernel(const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
                     int n, int k, int padded_k, int rows_per_block, int k_slice,
                     const __nv_bfloat16* __restrict__ x, int tokens, float* __restrict__ partial,
                     __nv_bfloat16* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char staged_raw[];
    auto* staged = reinterpret_cast<__nv_bfloat16*>(staged_raw); // [TT][k_slice]
    const int k0      = static_cast<int>(blockIdx.y) * k_slice;
    const int k_len   = min(k_slice, k - k0);
    const int t0      = static_cast<int>(blockIdx.z) * TT;
    const int t_count = min(TT, tokens - t0);

    const int chunks = k_len / 8;
    for (int c = static_cast<int>(threadIdx.x); c < TT * chunks; c += kThreads) {
        const int t = c / chunks;
        const int j = c - t * chunks;
        uint4 v     = make_uint4(0u, 0u, 0u, 0u);
        if (t < t_count) {
            v = *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(t0 + t) * k + k0 + j * 8);
        }
        *reinterpret_cast<uint4*>(staged + t * k_slice + j * 8) = v;
    }
    __syncthreads();

    constexpr int kRowsPerWarp = 32 / LPR;
    constexpr int kRowsPerPass = (kThreads / 32) * kRowsPerWarp;
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int warp          = static_cast<int>(threadIdx.x) >> 5;
    const int group         = lane / LPR;
    const int sub           = lane - group * LPR;
    const int row_base      = static_cast<int>(blockIdx.x) * rows_per_block;
    const int scale_stride  = (padded_k / 32) * 2;
    for (int slot = warp * kRowsPerWarp; slot < rows_per_block; slot += kRowsPerPass) {
        const int row    = row_base + slot + group;
        const bool valid = row < n && slot + group < rows_per_block;
        float acc[TT]    = {};
        if (valid) {
            const std::uint8_t* code_row  = codes + static_cast<std::int64_t>(row) * padded_k + k0;
            const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * scale_stride;
#pragma unroll 4
            for (int kk = sub * 8; kk < k_len; kk += LPR * 8) {
                const uint2 packed = *reinterpret_cast<const uint2*>(code_row + kk);
                const float scale  = __half2float(
                    *reinterpret_cast<const __half*>(scale_row + ((k0 + kk) >> 5) * 2));
                float w[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const std::uint32_t word = j < 4 ? packed.x : packed.y;
                    w[j] = detail::w8_a16_weight(static_cast<std::int8_t>(word >> ((j & 3) * 8)), scale);
                }
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(staged + t * k_slice + kk);
                    const float2 a = bf16x2_bits_to_float2(xv.x);
                    const float2 b = bf16x2_bits_to_float2(xv.y);
                    const float2 c = bf16x2_bits_to_float2(xv.z);
                    const float2 d = bf16x2_bits_to_float2(xv.w);
                    float sum      = acc[t];
                    sum            = fmaf(w[0], a.x, sum);
                    sum            = fmaf(w[1], a.y, sum);
                    sum            = fmaf(w[2], b.x, sum);
                    sum            = fmaf(w[3], b.y, sum);
                    sum            = fmaf(w[4], c.x, sum);
                    sum            = fmaf(w[5], c.y, sum);
                    sum            = fmaf(w[6], d.x, sum);
                    sum            = fmaf(w[7], d.y, sum);
                    acc[t]         = sum;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
#pragma unroll
            for (int offset = LPR / 2; offset > 0; offset >>= 1) {
                acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], offset);
            }
        }
        if (!valid || sub != 0) { continue; }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t >= t_count) { break; }
            const std::int64_t at = static_cast<std::int64_t>(t0 + t) * n + row;
            if (partial != nullptr) {
                partial[static_cast<std::int64_t>(blockIdx.y) * tokens * n + at] = acc[t];
            } else {
                out[at] = __float2bfloat16_rn(acc[t]);
            }
        }
    }
}

// lo = silu(bf16(sum of the slices' dots) / streams): the BF16 route's rounding of the
// projection, then its silu_scale_kernel.
__global__ void w8_low_finish_kernel(const float* __restrict__ partial, int slices,
                                     std::int64_t count, float scale,
                                     __nv_bfloat16* __restrict__ low) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    float dot = 0.0F;
    for (int s = 0; s < slices; ++s) { dot += partial[static_cast<std::int64_t>(s) * count + i]; }
    const float rounded = __bfloat162float(__float2bfloat16_rn(dot));
    low[i]              = __float2bfloat16_rn(silu(rounded * scale));
}

// The BF16 matrix a W8 one stands for, eight values a thread, for the wide rounds' GEMM.
__global__ void w8_dequantize_kernel(const std::uint8_t* __restrict__ codes,
                                     const std::uint8_t* __restrict__ scales, int n, int k,
                                     int padded_k, __nv_bfloat16* __restrict__ out) {
    const std::int64_t i      = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int chunks          = k / 8;
    if (i >= static_cast<std::int64_t>(n) * chunks) { return; }
    const int row             = static_cast<int>(i / chunks);
    const int column          = static_cast<int>(i - static_cast<std::int64_t>(row) * chunks) * 8;
    const uint2 packed        = *reinterpret_cast<const uint2*>(codes + static_cast<std::int64_t>(row) * padded_k + column);
    const float scale         = __half2float(*reinterpret_cast<const __half*>(
        scales + static_cast<std::int64_t>(row) * (padded_k / 32) * 2 + (column >> 5) * 2));
    __align__(16) __nv_bfloat16 values[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const std::uint32_t word = j < 4 ? packed.x : packed.y;
        values[j] = __float2bfloat16_rn(
            detail::w8_a16_weight(static_cast<std::int8_t>(word >> ((j & 3) * 8)), scale));
    }
    *reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(row) * k + column) =
        *reinterpret_cast<const uint4*>(values);
}

// Grid (token, hidden slice), one thread per hidden index: the stream mean of gate * n, with n
// recomputed in FP32 from the residual and the norm's inverse RMS so the mixed input does not
// inherit the BF16 rounding of the projection operand. The first slice of each token also sums
// the norm's per-stream shares into the inject gates. Spreading a token over its hidden slices
// matters at decode, where one block per token left all but one SM idle.
__global__ void finish_kernel(const __nv_bfloat16* __restrict__ residual,
                              const float* __restrict__ gamma,
                              const __nv_bfloat16* __restrict__ gate_logits,
                              const float* __restrict__ inv, const float* __restrict__ inject_partial,
                              int hidden, int streams, __nv_bfloat16* __restrict__ mixed,
                              float* __restrict__ inject) {
    const int token         = static_cast<int>(blockIdx.x);
    const int d             = static_cast<int>(blockIdx.y) * kThreads + static_cast<int>(threadIdx.x);
    const int width         = hidden * streams;
    const std::int64_t base = static_cast<std::int64_t>(token) * width;
    const float* token_inv  = inv + static_cast<std::int64_t>(token) * streams;
    const float mean_scale  = 1.0F / static_cast<float>(streams);
    if (d < hidden) {
        float acc = 0.0F;
        for (int s = 0; s < streams; ++s) {
            const std::int64_t i = base + static_cast<std::int64_t>(s) * hidden + d;
            const float n        = __bfloat162float(residual[i]) * token_inv[s] * gamma[s * hidden + d];
            acc                  = fmaf(sigmoid(__bfloat162float(gate_logits[i])), n, acc);
        }
        mixed[static_cast<std::int64_t>(token) * hidden + d] = __float2bfloat16_rn(acc * mean_scale);
    }
    if (inject == nullptr || blockIdx.y != 0 || static_cast<int>(threadIdx.x) >= streams) { return; }
    const int row = static_cast<int>(threadIdx.x);
    float total   = 0.0F;
    for (int s = 0; s < streams; ++s) {
        total += inject_partial[(static_cast<std::int64_t>(token) * streams + s) * streams + row];
    }
    inject[static_cast<std::int64_t>(token) * streams + row] = 2.0F * sigmoid(total * mean_scale);
}

__global__ void combine_kernel(const float* __restrict__ extra,
                               const __nv_bfloat16* __restrict__ block_output,
                               const float* __restrict__ inject, int hidden, int streams,
                               std::int64_t count, __nv_bfloat16* __restrict__ residual) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const int width         = hidden * streams;
    const std::int64_t tok  = i / width;
    const int within        = static_cast<int>(i - tok * width);
    const int stream        = within / hidden;
    const int d             = within - stream * hidden;
    const float weight      = inject[tok * streams + stream];
    float block = __bfloat162float(block_output[tok * hidden + d]);
    if (extra != nullptr) { block += extra[tok * hidden + d]; }
    const float value = __bfloat162float(residual[i]) + weight * block;
    residual[i] = __float2bfloat16_rn(value);
}

__global__ void broadcast_kernel(const __nv_bfloat16* __restrict__ source, int hidden,
                                 int streams, std::int64_t count,
                                 __nv_bfloat16* __restrict__ residual) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const int width        = hidden * streams;
    const std::int64_t tok = i / width;
    const int d            = static_cast<int>(i - tok * width) % hidden;
    residual[i]            = source[tok * hidden + d];
}

void require_contiguous(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t tokens,
                        const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != rows || tensor.ne[1] != tokens ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr || (reinterpret_cast<std::uintptr_t>(tensor.data) & 15u) != 0) {
        throw std::invalid_argument(std::string("hyper_connection: invalid ") + name);
    }
}

void require_bf16_weight(const Weight& weight, std::int32_t n, std::int32_t k, const char* name) {
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.qdata == nullptr || weight.ndim != 2 || weight.n != n || weight.k != k ||
        weight.payload_bytes < static_cast<std::uint64_t>(n) * k * 2) {
        throw std::invalid_argument(std::string("hyper_connection: ") + name +
                                    " must be contiguous BF16 [" + std::to_string(n) + "," +
                                    std::to_string(k) + "]");
    }
}

unsigned grid_for(std::int64_t count) {
    return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

bool is_w8(const Weight& weight) { return weight.qtype == QType::W8G32_F16S; }

// The low-rank projections are BF16 or W8 row-split; the inject rows are read by the norm
// kernel and stay BF16.
void require_projection_weight(const Weight& weight, std::int32_t n, std::int32_t k,
                               const char* name) {
    if (!is_w8(weight)) {
        require_bf16_weight(weight, n, k, name);
        return;
    }
    const std::int32_t padded_k = weight.padded_shape[1];
    if (weight.layout != QuantLayout::RowSplit || weight.qdata == nullptr ||
        weight.scales == nullptr || weight.n != n || weight.k != k || padded_k < k ||
        (padded_k % 32) != 0 || (reinterpret_cast<std::uintptr_t>(weight.qdata) & 7u) != 0 ||
        (reinterpret_cast<std::uintptr_t>(weight.scales) & 1u) != 0) {
        throw std::invalid_argument(std::string("hyper_connection: ") + name +
                                    " must be BF16 or W8 row-split [" + std::to_string(n) + "," +
                                    std::to_string(k) + "]");
    }
}

// Slices of the K extent for a W8 projection: enough to spread a short matrix over the GPU.
std::int32_t w8_slices(std::int32_t k) { return std::max(1, k / kW8SliceK); }

std::int32_t w8_slice_k(std::int32_t k, std::int32_t slices) {
    const std::int32_t per = (k + slices - 1) / slices;
    return (per + 7) / 8 * 8;
}

std::size_t round_256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// Whether the W8 kernels take this projection at this width. Their staged activations, a
// token tile of one K slice, must fit the default 48 KiB of shared memory: a sliced projection's
// slice is always under 2 * kW8SliceK values, and an unsliced one is held to the same bound.
bool w8_direct(const Weight& weight, std::int32_t tokens, std::int32_t slices) {
    return is_w8(weight) && tokens <= kW8DirectTokens &&
           w8_slice_k(weight.k, slices) <= 2 * kW8SliceK;
}

template <int LPR, int TT>
void launch_w8_projection(const Weight& w, const Tensor& x, std::int32_t slices, float* partial,
                          __nv_bfloat16* out, cudaStream_t stream) {
    const std::int32_t n      = w.n;
    const std::int32_t k      = w.k;
    const std::int32_t tokens = x.ne[1];
    const std::int32_t slice  = w8_slice_k(k, slices);
    // Two passes of rows a block: 16 rows for a warp-per-row projection, 64 for LPR = 8.
    const std::int32_t rows_per_block = 2 * (kThreads / 32) * (32 / LPR);
    const dim3 grid(static_cast<unsigned>((n + rows_per_block - 1) / rows_per_block),
                    static_cast<unsigned>((k + slice - 1) / slice),
                    static_cast<unsigned>((tokens + TT - 1) / TT));
    const std::size_t staged = static_cast<std::size_t>(slice) * TT * sizeof(__nv_bfloat16);
    w8_projection_kernel<LPR, TT><<<grid, kThreads, staged, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales), n,
        k, w.padded_shape[1], rows_per_block, slice, static_cast<const __nv_bfloat16*>(x.data),
        tokens, partial, out);
    CUDA_CHECK(cudaGetLastError());
}

template <int LPR>
void launch_w8_projection_tokens(const Weight& w, const Tensor& x, std::int32_t slices,
                                 float* partial, __nv_bfloat16* out, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (tokens == 1) { launch_w8_projection<LPR, 1>(w, x, slices, partial, out, stream); }
    else if (tokens == 2) { launch_w8_projection<LPR, 2>(w, x, slices, partial, out, stream); }
    else if (tokens <= 4) { launch_w8_projection<LPR, 4>(w, x, slices, partial, out, stream); }
    else { launch_w8_projection<LPR, kW8TokenTile>(w, x, slices, partial, out, stream); }
}

// A W8 projection on the mixer's kernels: a warp a row when its K slice is long, eight lanes a
// row when it is short (the up projection's K is the low rank).
void w8_projection(const Weight& w, const Tensor& x, std::int32_t slices, float* partial,
                   __nv_bfloat16* out, cudaStream_t stream) {
    if (w8_slice_k(w.k, slices) >= 1024) {
        launch_w8_projection_tokens<32>(w, x, slices, partial, out, stream);
    } else {
        launch_w8_projection_tokens<8>(w, x, slices, partial, out, stream);
    }
}

// out = w · x through cuBLASLt; a W8 matrix is first dequantised into the workspace.
void cublaslt_projection(const Weight& w, const Tensor& x, Tensor& out, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    if (!is_w8(w)) {
        detail::bf16_cublaslt_gemm(w, x, out, stream);
        return;
    }
    auto scope     = workspace.scope();
    Tensor matrix  = workspace.alloc(DType::BF16, {w.k, w.n});
    const std::int64_t chunks = static_cast<std::int64_t>(w.n) * (w.k / 8);
    w8_dequantize_kernel<<<grid_for(chunks), kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales), w.n,
        w.k, w.padded_shape[1], static_cast<__nv_bfloat16*>(matrix.data));
    CUDA_CHECK(cudaGetLastError());
    Weight dense{};
    dense.qtype           = QType::BF16_CTRL;
    dense.layout          = QuantLayout::Contiguous;
    dense.payload         = matrix.data;
    dense.qdata           = matrix.data;
    dense.payload_bytes   = static_cast<std::uint64_t>(w.n) * w.k * 2;
    dense.ndim            = 2;
    dense.n               = w.n;
    dense.k               = w.k;
    dense.shape[0]        = w.n;
    dense.shape[1]        = w.k;
    dense.padded_shape[0] = w.n;
    dense.padded_shape[1] = w.k;
    detail::bf16_cublaslt_gemm(dense, x, out, stream);
}

void require_geometry(std::int32_t streams, std::int32_t hidden, std::int32_t low_rank) {
    if (streams < 1 || streams > kMaxStreams || hidden <= 0 || (hidden % 8) != 0 || low_rank <= 0 ||
        (low_rank % 8) != 0) {
        throw std::invalid_argument("hyper_connection: unsupported stream/hidden/low-rank geometry");
    }
}

} // namespace

std::size_t hyper_connection_mix_workspace_capacity_bytes(std::int32_t streams,
                                                          std::int32_t hidden,
                                                          std::int32_t low_rank,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) {
    require_geometry(streams, hidden, low_rank);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("hyper_connection: invalid token interval");
    }
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    const std::size_t tokens = static_cast<std::size_t>(max_tokens);
    // normalized [width,T], low-rank [low_rank,T], gate logits [width,T], the inverse RMS
    // [streams,T] and the inject shares [streams,streams,T]; 256-byte rounded. W8 projections
    // add the down projection's sliced dots on the narrow rounds and one dequantised matrix on
    // the wide ones; the capacity covers them whatever the weights' format.
    const std::size_t per_stream = static_cast<std::size_t>(streams) * tokens * sizeof(float);
    std::size_t bytes = round_256(width * tokens * 2) +
                        round_256(static_cast<std::size_t>(low_rank) * tokens * 2) +
                        round_256(width * tokens * 2) + round_256(per_stream) +
                        round_256(per_stream * streams) + 5 * 256;
    if (min_tokens <= kW8DirectTokens) {
        const std::size_t narrow = std::min<std::size_t>(tokens, kW8DirectTokens);
        bytes += round_256(static_cast<std::size_t>(w8_slices(static_cast<std::int32_t>(width))) *
                           narrow * low_rank * sizeof(float)) + 256;
    }
    if (max_tokens > kW8DirectTokens || low_rank > 2 * kW8SliceK) {
        bytes += round_256(width * static_cast<std::size_t>(low_rank) * 2) + 256;
    }
    return bytes;
}

void hyper_connection_norm(const Tensor& residual, const Tensor& norm, std::int32_t streams,
                           float eps, Tensor& normalized, cudaStream_t stream) {
    const std::int32_t width  = residual.ne[0];
    const std::int32_t tokens = residual.ne[1];
    if (streams < 1 || (width % streams) != 0) {
        throw std::invalid_argument("hyper_connection: residual width is not a stream multiple");
    }
    if (!(eps > 0.0F)) { throw std::invalid_argument("hyper_connection: eps must be positive"); }
    require_contiguous(residual, DType::BF16, width, tokens, "residual");
    require_contiguous(normalized, DType::BF16, width, tokens, "normalized");
    if (norm.dtype != DType::FP32 || norm.ne[0] != width || norm.numel() != width ||
        norm.data == nullptr) {
        throw std::invalid_argument("hyper_connection: norm must be FP32 [streams*hidden]");
    }
    if (streams > kMaxStreams) { throw std::invalid_argument("hyper_connection: too many streams"); }
    if (((width / streams) % kNormVec) != 0) {
        throw std::invalid_argument("hyper_connection: stream width must be a multiple of 8");
    }
    stream_norm_kernel<false><<<static_cast<unsigned>(tokens) * streams, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data), static_cast<const float*>(norm.data),
        nullptr, width / streams, streams, eps, static_cast<__nv_bfloat16*>(normalized.data),
        nullptr, nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_mix(const Tensor& residual, const HyperConnectionWeights& weights,
                          std::int32_t streams, float eps, Tensor& mixed, Tensor* inject,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t width    = residual.ne[0];
    const std::int32_t tokens   = residual.ne[1];
    const std::int32_t low_rank = weights.down.n;
    if (streams < 1 || (width % streams) != 0) {
        throw std::invalid_argument("hyper_connection: residual width is not a stream multiple");
    }
    const std::int32_t hidden = width / streams;
    require_geometry(streams, hidden, low_rank);
    if (!(eps > 0.0F)) { throw std::invalid_argument("hyper_connection: eps must be positive"); }
    require_contiguous(residual, DType::BF16, width, tokens, "residual");
    require_contiguous(mixed, DType::BF16, hidden, tokens, "mixed");
    if (weights.norm.dtype != DType::FP32 || weights.norm.ne[0] != width ||
        weights.norm.numel() != width || weights.norm.data == nullptr) {
        throw std::invalid_argument("hyper_connection: norm must be FP32 [streams*hidden]");
    }
    require_projection_weight(weights.down, low_rank, width, "down");
    require_projection_weight(weights.up, width, low_rank, "up");
    const bool has_inject = weights.inject.n != 0;
    if (has_inject) { require_bf16_weight(weights.inject, streams, width, "inject"); }
    if (inject != nullptr) {
        if (!has_inject) {
            throw std::invalid_argument("hyper_connection: inject requested without inject rows");
        }
        require_contiguous(*inject, DType::FP32, streams, tokens, "inject");
    }

    auto scope        = workspace.scope();
    Tensor normalized = workspace.alloc(DType::BF16, {width, tokens});
    Tensor low        = workspace.alloc(DType::BF16, {low_rank, tokens});
    Tensor gate       = workspace.alloc(DType::BF16, {width, tokens});
    Tensor inv        = workspace.alloc(DType::FP32, {streams, tokens});
    Tensor shares     = workspace.alloc(DType::FP32, {streams * streams, tokens});

    const unsigned norm_blocks = static_cast<unsigned>(tokens) * streams;
    const auto* residual_data  = static_cast<const __nv_bfloat16*>(residual.data);
    const auto* gamma          = static_cast<const float*>(weights.norm.data);
    if (inject != nullptr) {
        const auto* inject_rows = static_cast<const __nv_bfloat16*>(weights.inject.qdata);
        auto* norm_out          = static_cast<__nv_bfloat16*>(normalized.data);
        auto* inv_out           = static_cast<float*>(inv.data);
        auto* share_out         = static_cast<float*>(shares.data);
        if (streams == 4) { // Qwen3.8-Flash-Next
            stream_norm_kernel<true, 4><<<norm_blocks, kThreads, 0, stream>>>(
                residual_data, gamma, inject_rows, hidden, streams, eps, norm_out, inv_out,
                share_out);
        } else {
            stream_norm_kernel<true><<<norm_blocks, kThreads, 0, stream>>>(
                residual_data, gamma, inject_rows, hidden, streams, eps, norm_out, inv_out,
                share_out);
        }
    } else {
        stream_norm_kernel<false><<<norm_blocks, kThreads, 0, stream>>>(
            residual_data, gamma, nullptr, hidden, streams, eps,
            static_cast<__nv_bfloat16*>(normalized.data), static_cast<float*>(inv.data), nullptr);
    }
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t low_count = static_cast<std::int64_t>(low_rank) * tokens;
    const float low_scale        = 1.0F / static_cast<float>(streams);
    const std::int32_t slices = w8_slices(width);
    if (w8_direct(weights.down, tokens, slices)) {
        Tensor dots = workspace.alloc(DType::FP32, {low_rank, tokens * slices});
        w8_projection(weights.down, normalized, slices, static_cast<float*>(dots.data), nullptr,
                      stream);
        w8_low_finish_kernel<<<grid_for(low_count), kThreads, 0, stream>>>(
            static_cast<const float*>(dots.data), slices, low_count, low_scale,
            static_cast<__nv_bfloat16*>(low.data));
    } else {
        cublaslt_projection(weights.down, normalized, low, workspace, stream);
        silu_scale_kernel<<<grid_for(low_count), kThreads, 0, stream>>>(
            static_cast<__nv_bfloat16*>(low.data), low_count, low_scale);
    }
    CUDA_CHECK(cudaGetLastError());
    if (w8_direct(weights.up, tokens, 1)) {
        w8_projection(weights.up, low, 1, nullptr, static_cast<__nv_bfloat16*>(gate.data), stream);
    } else {
        cublaslt_projection(weights.up, low, gate, workspace, stream);
    }
    const dim3 finish_grid(static_cast<unsigned>(tokens),
                           static_cast<unsigned>((hidden + kThreads - 1) / kThreads));
    finish_kernel<<<finish_grid, kThreads, 0, stream>>>(
        residual_data, gamma, static_cast<const __nv_bfloat16*>(gate.data),
        static_cast<const float*>(inv.data), static_cast<const float*>(shares.data), hidden,
        streams, static_cast<__nv_bfloat16*>(mixed.data),
        inject != nullptr ? static_cast<float*>(inject->data) : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_combine(const Tensor& block_output, const Tensor& inject, Tensor& residual,
                              cudaStream_t stream) {
    hyper_connection_combine(block_output, nullptr, inject, residual, stream);
}

void hyper_connection_combine(const Tensor& block_output, const float* extra, const Tensor& inject,
                              Tensor& residual, cudaStream_t stream) {
    const std::int32_t width   = residual.ne[0];
    const std::int32_t tokens  = residual.ne[1];
    const std::int32_t streams = inject.ne[0];
    if (streams < 1 || streams > kMaxStreams || (width % streams) != 0) {
        throw std::invalid_argument("hyper_connection: combine stream count is invalid");
    }
    const std::int32_t hidden = width / streams;
    require_contiguous(residual, DType::BF16, width, tokens, "residual");
    require_contiguous(block_output, DType::BF16, hidden, tokens, "block_output");
    require_contiguous(inject, DType::FP32, streams, tokens, "inject");
    const std::int64_t count = static_cast<std::int64_t>(width) * tokens;
    combine_kernel<<<grid_for(count), kThreads, 0, stream>>>(
        extra,
        static_cast<const __nv_bfloat16*>(block_output.data),
        static_cast<const float*>(inject.data), hidden, streams, count,
        static_cast<__nv_bfloat16*>(residual.data));
    CUDA_CHECK(cudaGetLastError());
}

void broadcast_streams(const Tensor& source, std::int32_t streams, Tensor& residual,
                       cudaStream_t stream) {
    const std::int32_t hidden = source.ne[0];
    const std::int32_t tokens = source.ne[1];
    if (streams < 1 || streams > kMaxStreams) {
        throw std::invalid_argument("hyper_connection: broadcast stream count is invalid");
    }
    require_contiguous(source, DType::BF16, hidden, tokens, "source");
    require_contiguous(residual, DType::BF16, hidden * streams, tokens, "residual");
    const std::int64_t count = static_cast<std::int64_t>(hidden) * streams * tokens;
    broadcast_kernel<<<grid_for(count), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(source.data), hidden, streams, count,
        static_cast<__nv_bfloat16*>(residual.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops
