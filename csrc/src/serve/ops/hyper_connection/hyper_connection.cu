#include "api/ops/hyper_connection.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/bf16/bf16_cublaslt.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kThreads    = 256;
constexpr int kMaxStreams = 8;

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
// projections.
__global__ void stream_norm_kernel(const __nv_bfloat16* __restrict__ residual,
                                   const float* __restrict__ gamma, int hidden, int streams,
                                   float eps, __nv_bfloat16* __restrict__ normalized) {
    __shared__ float scratch[kThreads / 32];
    const int token  = static_cast<int>(blockIdx.x) / streams;
    const int stream = static_cast<int>(blockIdx.x) - token * streams;
    const std::int64_t base =
        static_cast<std::int64_t>(token) * hidden * streams + static_cast<std::int64_t>(stream) * hidden;
    const __nv_bfloat16* x = residual + base;
    float sum_sq           = 0.0F;
    for (int d = static_cast<int>(threadIdx.x); d < hidden; d += kThreads) {
        const float v = __bfloat162float(x[d]);
        sum_sq        = fmaf(v, v, sum_sq);
    }
    const float total = block_sum(sum_sq, scratch);
    const float inv   = rsqrtf(total / static_cast<float>(hidden) + eps);
    for (int d = static_cast<int>(threadIdx.x); d < hidden; d += kThreads) {
        const float v = __bfloat162float(x[d]) * inv * gamma[stream * hidden + d];
        normalized[base + d] = __float2bfloat16_rn(v);
    }
}

// lo = silu(lo / streams), in place.
__global__ void silu_scale_kernel(__nv_bfloat16* __restrict__ values, std::int64_t count,
                                  float scale) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    values[i] = __float2bfloat16_rn(silu(__bfloat162float(values[i]) * scale));
}

// One block per token: the stream mean of gate * n (n recomputed in FP32 from the residual so
// the mixed input does not inherit the BF16 rounding of the projection operand) and the
// inject gates.
__global__ void finish_kernel(const __nv_bfloat16* __restrict__ residual,
                              const float* __restrict__ gamma,
                              const __nv_bfloat16* __restrict__ gate_logits,
                              const __nv_bfloat16* __restrict__ inject_weight, int hidden,
                              int streams, float eps, __nv_bfloat16* __restrict__ mixed,
                              float* __restrict__ inject) {
    __shared__ float scratch[kThreads / 32];
    __shared__ float inv[kMaxStreams];
    const int token         = static_cast<int>(blockIdx.x);
    const int width         = hidden * streams;
    const std::int64_t base = static_cast<std::int64_t>(token) * width;
    const __nv_bfloat16* x  = residual + base;

    for (int s = 0; s < streams; ++s) {
        float sum_sq = 0.0F;
        for (int d = static_cast<int>(threadIdx.x); d < hidden; d += kThreads) {
            const float v = __bfloat162float(x[s * hidden + d]);
            sum_sq        = fmaf(v, v, sum_sq);
        }
        const float total = block_sum(sum_sq, scratch);
        if (threadIdx.x == 0) { inv[s] = rsqrtf(total / static_cast<float>(hidden) + eps); }
    }
    __syncthreads();

    const float mean_scale = 1.0F / static_cast<float>(streams);
    for (int d = static_cast<int>(threadIdx.x); d < hidden; d += kThreads) {
        float acc = 0.0F;
        for (int s = 0; s < streams; ++s) {
            const int i   = s * hidden + d;
            const float n = __bfloat162float(x[i]) * inv[s] * gamma[i];
            acc           = fmaf(sigmoid(__bfloat162float(gate_logits[base + i])), n, acc);
        }
        mixed[static_cast<std::int64_t>(token) * hidden + d] = __float2bfloat16_rn(acc * mean_scale);
    }

    if (inject == nullptr) { return; }
    for (int s = 0; s < streams; ++s) {
        const __nv_bfloat16* row = inject_weight + static_cast<std::int64_t>(s) * width;
        float dot                = 0.0F;
        for (int i = static_cast<int>(threadIdx.x); i < width; i += kThreads) {
            const int stream_of_i = i / hidden;
            const float n = __bfloat162float(x[i]) * inv[stream_of_i] * gamma[i];
            dot           = fmaf(__bfloat162float(row[i]), n, dot);
        }
        const float total = block_sum(dot, scratch);
        if (threadIdx.x == 0) {
            inject[static_cast<std::int64_t>(token) * streams + s] =
                2.0F * sigmoid(total * mean_scale);
        }
    }
}

__global__ void combine_kernel(const __nv_bfloat16* __restrict__ block_output,
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
    const float value       = __bfloat162float(residual[i]) +
                        weight * __bfloat162float(block_output[tok * hidden + d]);
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

void require_geometry(std::int32_t streams, std::int32_t hidden, std::int32_t low_rank) {
    if (streams < 1 || streams > kMaxStreams || hidden <= 0 || (hidden % 8) != 0 || low_rank <= 0 ||
        (low_rank % 8) != 0) {
        throw std::invalid_argument("hyper_connection: unsupported stream/hidden/low-rank geometry");
    }
}

unsigned grid_for(std::int64_t count) {
    return static_cast<unsigned>((count + kThreads - 1) / kThreads);
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
    // normalized [width,T], low-rank [low_rank,T], gate logits [width,T]; 256-byte rounded.
    const auto round = [](std::size_t bytes) { return (bytes + 255) / 256 * 256; };
    return round(width * tokens * 2) + round(static_cast<std::size_t>(low_rank) * tokens * 2) +
           round(width * tokens * 2) + 3 * 256;
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
    require_bf16_weight(weights.down, low_rank, width, "down");
    require_bf16_weight(weights.up, width, low_rank, "up");
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

    stream_norm_kernel<<<static_cast<unsigned>(tokens) * streams, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const float*>(weights.norm.data), hidden, streams, eps,
        static_cast<__nv_bfloat16*>(normalized.data));
    CUDA_CHECK(cudaGetLastError());
    detail::bf16_cublaslt_gemm(weights.down, normalized, low, stream);
    const std::int64_t low_count = static_cast<std::int64_t>(low_rank) * tokens;
    silu_scale_kernel<<<grid_for(low_count), kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(low.data), low_count, 1.0F / static_cast<float>(streams));
    CUDA_CHECK(cudaGetLastError());
    detail::bf16_cublaslt_gemm(weights.up, low, gate, stream);
    finish_kernel<<<static_cast<unsigned>(tokens), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const float*>(weights.norm.data),
        static_cast<const __nv_bfloat16*>(gate.data),
        has_inject ? static_cast<const __nv_bfloat16*>(weights.inject.qdata) : nullptr, hidden,
        streams, eps, static_cast<__nv_bfloat16*>(mixed.data),
        inject != nullptr ? static_cast<float*>(inject->data) : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void hyper_connection_combine(const Tensor& block_output, const Tensor& inject, Tensor& residual,
                              cudaStream_t stream) {
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

} // namespace ninfer::ops
