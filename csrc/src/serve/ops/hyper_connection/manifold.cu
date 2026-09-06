#include "api/ops/manifold_hyper_connection.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

constexpr int kThreads    = 256;
constexpr int kWarps      = kThreads / 32;
constexpr int kMaxStreams = 4;
constexpr int kMaxRows    = (2 + kMaxStreams) * kMaxStreams;
/// The residual column is staged in shared memory, so its width is what bounds the op.
constexpr int kMaxWidth = 32768;

/// The projection, split across the width. One block per (token, slice): each accumulates its
/// slice's contribution to the column's sum of squares and to every projection row, into an
/// FP32 workspace of `1 + rows` floats per token.
///
/// A block per *token* was the first shape, staging the whole column in shared memory so it
/// could be read once for the norm, once per projection row and once for the collapse. That is
/// the right shape when there are many tokens and the wrong one when there is a single token:
/// one CTA of 256 threads then reads the projection's 786 KB alone, on one SM of a hundred and
/// seventy, and a decode round spends 136 us in each of ninety calls. Splitting the width is
/// what puts the card to work; the column is re-read from global for the collapse, which costs
/// 32 KB a token against the 12 ms a round was losing.
__global__ __launch_bounds__(kThreads) void manifold_project_kernel(
    const __nv_bfloat16* __restrict__ residual, const __nv_bfloat16* __restrict__ mix, int width,
    int rows, int slices, float* __restrict__ partials) {
    const int token       = static_cast<int>(blockIdx.y);
    const int slice       = static_cast<int>(blockIdx.x);
    const std::int64_t at = static_cast<std::int64_t>(token) * width;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int stride      = kThreads * slices;

    __shared__ float reduce[kWarps * (kMaxRows + 1)];

    float sum_sq = 0.0F;
    float dot[kMaxRows];
    for (int r = 0; r < rows; ++r) { dot[r] = 0.0F; }
    for (int i = slice * kThreads + static_cast<int>(threadIdx.x); i < width; i += stride) {
        const float value = __bfloat162float(residual[at + i]);
        sum_sq            = fmaf(value, value, sum_sq);
        for (int r = 0; r < rows; ++r) {
            dot[r] = fmaf(__bfloat162float(mix[static_cast<std::int64_t>(r) * width + i]), value,
                          dot[r]);
        }
    }
    sum_sq = warp_reduce_sum(sum_sq);
    if (lane == 0) { reduce[warp * (rows + 1)] = sum_sq; }
    for (int r = 0; r < rows; ++r) {
        const float sum = warp_reduce_sum(dot[r]);
        if (lane == 0) { reduce[warp * (rows + 1) + 1 + r] = sum; }
    }
    __syncthreads();
    if (static_cast<int>(threadIdx.x) <= rows) {
        const int slot = static_cast<int>(threadIdx.x);
        float sum      = 0.0F;
        for (int w = 0; w < kWarps; ++w) { sum += reduce[w * (rows + 1) + slot]; }
        atomicAdd(&partials[static_cast<std::int64_t>(token) * (rows + 1) + slot], sum);
    }
}

/// One thread per token: the accumulated dots become `pre`, `post` and the doubly-stochastic
/// `comb`. Sinkhorn is twenty passes over a `kStreams`-square matrix, serial by nature.
///
/// `kStreams` is a template parameter and not an argument because it indexes the two small
/// arrays below. With a runtime bound the compiler cannot keep them in registers and they land
/// in local memory, which is DRAM: the same arithmetic then took 53.8 us a call where it takes
/// about two, and at ninety calls a token that was five milliseconds of a decode round.
template <int kStreams>
__global__ void manifold_reduce_kernel(const float* __restrict__ partials,
                                       const float* __restrict__ base,
                                       const float* __restrict__ scale, int width, int tokens,
                                       int rows, float rms_eps, float hc_eps,
                                       int sinkhorn_iterations, float* __restrict__ pre_out,
                                       float* __restrict__ post, float* __restrict__ comb) {
    constexpr int streams = kStreams;
    const int token       = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                            static_cast<int>(threadIdx.x);
    if (token >= tokens) { return; }
    const float* accumulated = partials + static_cast<std::int64_t>(token) * (rows + 1);
    const float inv          = rsqrtf(accumulated[0] / static_cast<float>(width) + rms_eps);

    constexpr int kRows = (2 + kStreams) * kStreams;
    float logits[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) { logits[r] = accumulated[1 + r] * inv; }
#pragma unroll
    for (int s = 0; s < streams; ++s) {
        pre_out[static_cast<std::int64_t>(token) * streams + s] =
            sigmoid(logits[s] * scale[0] + base[s]) + hc_eps;
        post[static_cast<std::int64_t>(token) * streams + s] =
            2.0F * sigmoid(logits[streams + s] * scale[1] + base[streams + s]);
    }
    // softmax over j, then Sinkhorn-Knopp onto the doubly-stochastic manifold. Small enough
    // (streams <= 4) that one thread is the clearest way to keep the iteration order the
    // reference's.
    float matrix[kStreams * kStreams];
    #pragma unroll
    for (int i = 0; i < streams; ++i) {
        float maximum = -3.402823466e+38F;
        #pragma unroll
        for (int j = 0; j < streams; ++j) {
            const int index         = 2 * streams + i * streams + j;
            matrix[i * streams + j] = logits[index] * scale[2] + base[index];
            maximum                 = fmaxf(maximum, matrix[i * streams + j]);
        }
        float sum = 0.0F;
        #pragma unroll
        for (int j = 0; j < streams; ++j) {
            matrix[i * streams + j] = __expf(matrix[i * streams + j] - maximum);
            sum += matrix[i * streams + j];
        }
        #pragma unroll
        for (int j = 0; j < streams; ++j) {
            matrix[i * streams + j] = matrix[i * streams + j] / sum + hc_eps;
        }
    }
    #pragma unroll
    for (int j = 0; j < streams; ++j) {
        float sum = 0.0F;
        #pragma unroll
        for (int i = 0; i < streams; ++i) { sum += matrix[i * streams + j]; }
        const float denominator = sum + hc_eps;
        #pragma unroll
        for (int i = 0; i < streams; ++i) { matrix[i * streams + j] /= denominator; }
    }
    #pragma unroll
    for (int iteration = 1; iteration < sinkhorn_iterations; ++iteration) {
        #pragma unroll
        for (int i = 0; i < streams; ++i) {
            float sum = 0.0F;
            #pragma unroll
            for (int j = 0; j < streams; ++j) { sum += matrix[i * streams + j]; }
            const float denominator = sum + hc_eps;
            #pragma unroll
            for (int j = 0; j < streams; ++j) { matrix[i * streams + j] /= denominator; }
        }
        #pragma unroll
        for (int j = 0; j < streams; ++j) {
            float sum = 0.0F;
            #pragma unroll
            for (int i = 0; i < streams; ++i) { sum += matrix[i * streams + j]; }
            const float denominator = sum + hc_eps;
            #pragma unroll
            for (int i = 0; i < streams; ++i) { matrix[i * streams + j] /= denominator; }
        }
    }
    float* out = comb + static_cast<std::int64_t>(token) * streams * streams;
#pragma unroll
    for (int i = 0; i < streams * streams; ++i) { out[i] = matrix[i]; }
}

/// collapsed[d,t] = sum_s pre[s,t] * residual[s*hidden+d, t]. One thread per output element.
__global__ void manifold_collapse_kernel(const __nv_bfloat16* __restrict__ residual,
                                         const float* __restrict__ pre, int hidden, int streams,
                                         std::int64_t count,
                                         __nv_bfloat16* __restrict__ collapsed) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const int d               = static_cast<int>(i % hidden);
    const std::int64_t token  = i / hidden;
    const std::int64_t at     = token * static_cast<std::int64_t>(hidden) * streams;
    const float* token_pre    = pre + token * streams;
    float sum                 = 0.0F;
    for (int s = 0; s < streams; ++s) {
        sum = fmaf(token_pre[s], __bfloat162float(residual[at + s * hidden + d]), sum);
    }
    collapsed[i] = __float2bfloat16_rn(sum);
}

/// One block per token. The column is read once into shared memory and answers three questions:
/// its own RMS, every projection row's dot product, and the weighted collapse at the end.
__global__ __launch_bounds__(kThreads) void manifold_mix_kernel(
    const __nv_bfloat16* __restrict__ residual, const __nv_bfloat16* __restrict__ mix,
    const float* __restrict__ base, const float* __restrict__ scale, int hidden, int streams,
    int rows, float rms_eps, float hc_eps, int sinkhorn_iterations,
    __nv_bfloat16* __restrict__ collapsed, float* __restrict__ post, float* __restrict__ comb) {
    extern __shared__ char smem[];
    const int width           = hidden * streams;
    __nv_bfloat16* column     = reinterpret_cast<__nv_bfloat16*>(smem);
    float* reduce             = reinterpret_cast<float*>(smem + static_cast<std::size_t>(width) * 2);
    float* pre                = reduce + static_cast<std::size_t>(kWarps) * rows;
    const int token           = static_cast<int>(blockIdx.x);
    const std::int64_t offset = static_cast<std::int64_t>(token) * width;
    const int lane            = static_cast<int>(threadIdx.x) & 31;
    const int warp            = static_cast<int>(threadIdx.x) >> 5;

    float sum_sq = 0.0F;
    for (int i = static_cast<int>(threadIdx.x); i < width; i += kThreads) {
        const __nv_bfloat16 raw = residual[offset + i];
        column[i]               = raw;
        const float value       = __bfloat162float(raw);
        sum_sq                  = fmaf(value, value, sum_sq);
    }
    sum_sq = warp_reduce_sum(sum_sq);
    if (lane == 0) { reduce[warp] = sum_sq; }
    __syncthreads();
    float total = 0.0F;
    for (int w = 0; w < kWarps; ++w) { total += reduce[w]; }
    const float inv = rsqrtf(total / static_cast<float>(width) + rms_eps);
    __syncthreads();

    // Every projection row at once: the shared column is read `rows` times from shared rather
    // than `rows` times from global, and each row's weights are read exactly once.
    float dot[kMaxRows];
    for (int r = 0; r < rows; ++r) { dot[r] = 0.0F; }
    for (int i = static_cast<int>(threadIdx.x); i < width; i += kThreads) {
        const float n = __bfloat162float(column[i]);
        for (int r = 0; r < rows; ++r) {
            dot[r] = fmaf(__bfloat162float(mix[static_cast<std::int64_t>(r) * width + i]), n,
                          dot[r]);
        }
    }
    for (int r = 0; r < rows; ++r) {
        const float sum = warp_reduce_sum(dot[r]);
        if (lane == 0) { reduce[warp * rows + r] = sum; }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float logits[kMaxRows];
        for (int r = 0; r < rows; ++r) {
            float sum = 0.0F;
            for (int w = 0; w < kWarps; ++w) { sum += reduce[w * rows + r]; }
            logits[r] = sum * inv;
        }
        for (int s = 0; s < streams; ++s) {
            pre[s] = sigmoid(logits[s] * scale[0] + base[s]) + hc_eps;
            post[static_cast<std::int64_t>(token) * streams + s] =
                2.0F * sigmoid(logits[streams + s] * scale[1] + base[streams + s]);
        }
        // softmax over j, then Sinkhorn-Knopp onto the doubly-stochastic manifold. Small enough
        // (streams <= 4) that one thread is the clearest way to keep the iteration order the
        // reference's.
        float matrix[kMaxStreams * kMaxStreams];
        for (int i = 0; i < streams; ++i) {
            float maximum = -3.402823466e+38F;
            for (int j = 0; j < streams; ++j) {
                const int index = 2 * streams + i * streams + j;
                matrix[i * streams + j] = logits[index] * scale[2] + base[index];
                maximum                 = fmaxf(maximum, matrix[i * streams + j]);
            }
            float sum = 0.0F;
            for (int j = 0; j < streams; ++j) {
                matrix[i * streams + j] = __expf(matrix[i * streams + j] - maximum);
                sum += matrix[i * streams + j];
            }
            for (int j = 0; j < streams; ++j) { matrix[i * streams + j] = matrix[i * streams + j] / sum + hc_eps; }
        }
        for (int j = 0; j < streams; ++j) {
            float sum = 0.0F;
            for (int i = 0; i < streams; ++i) { sum += matrix[i * streams + j]; }
            const float denominator = sum + hc_eps;
            for (int i = 0; i < streams; ++i) { matrix[i * streams + j] /= denominator; }
        }
        for (int iteration = 1; iteration < sinkhorn_iterations; ++iteration) {
            for (int i = 0; i < streams; ++i) {
                float sum = 0.0F;
                for (int j = 0; j < streams; ++j) { sum += matrix[i * streams + j]; }
                const float denominator = sum + hc_eps;
                for (int j = 0; j < streams; ++j) { matrix[i * streams + j] /= denominator; }
            }
            for (int j = 0; j < streams; ++j) {
                float sum = 0.0F;
                for (int i = 0; i < streams; ++i) { sum += matrix[i * streams + j]; }
                const float denominator = sum + hc_eps;
                for (int i = 0; i < streams; ++i) { matrix[i * streams + j] /= denominator; }
            }
        }
        float* out = comb + static_cast<std::int64_t>(token) * streams * streams;
        for (int i = 0; i < streams * streams; ++i) { out[i] = matrix[i]; }
    }
    __syncthreads();

    for (int d = static_cast<int>(threadIdx.x); d < hidden; d += kThreads) {
        float sum = 0.0F;
        for (int s = 0; s < streams; ++s) {
            sum = fmaf(pre[s], __bfloat162float(column[s * hidden + d]), sum);
        }
        collapsed[static_cast<std::int64_t>(token) * hidden + d] = __float2bfloat16_rn(sum);
    }
}

/// One thread per (token, hidden column): it owns every stream of that column, which is what
/// lets the mix be in place.
__global__ void manifold_combine_kernel(const __nv_bfloat16* __restrict__ block_output,
                                        const float* __restrict__ post,
                                        const float* __restrict__ comb, int hidden, int streams,
                                        std::int64_t count, __nv_bfloat16* __restrict__ residual) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const std::int64_t token = i / hidden;
    const int d              = static_cast<int>(i - token * hidden);
    const std::int64_t base  = token * hidden * streams + d;

    float incoming[kMaxStreams];
    for (int s = 0; s < streams; ++s) {
        incoming[s] = __bfloat162float(residual[base + static_cast<std::int64_t>(s) * hidden]);
    }
    const float block        = __bfloat162float(block_output[i]);
    const float* mixing      = comb + token * streams * streams;
    const float* placement   = post + token * streams;
    for (int s = 0; s < streams; ++s) {
        float sum = placement[s] * block;
        for (int source = 0; source < streams; ++source) {
            sum = fmaf(mixing[source * streams + s], incoming[source], sum);
        }
        residual[base + static_cast<std::int64_t>(s) * hidden] = __float2bfloat16_rn(sum);
    }
}

__global__ void collapse_mean_kernel(const __nv_bfloat16* __restrict__ residual, int hidden,
                                     int streams, std::int64_t count,
                                     __nv_bfloat16* __restrict__ mean) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const std::int64_t token = i / hidden;
    const int d              = static_cast<int>(i - token * hidden);
    const std::int64_t base  = token * hidden * streams + d;
    float sum                = 0.0F;
    for (int s = 0; s < streams; ++s) {
        sum += __bfloat162float(residual[base + static_cast<std::int64_t>(s) * hidden]);
    }
    mean[i] = __float2bfloat16_rn(sum / static_cast<float>(streams));
}

void require_contiguous(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t tokens,
                        const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != rows || tensor.ne[1] != tokens ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument(std::string("manifold_hyper_connection: invalid ") + name);
    }
}

void require_geometry(std::int32_t streams, std::int32_t hidden) {
    if (streams < 1 || streams > kMaxStreams || hidden <= 0 || (hidden % 8) != 0 ||
        hidden * streams > kMaxWidth) {
        throw std::invalid_argument(
            "manifold_hyper_connection: streams must be 1.." + std::to_string(kMaxStreams) +
            " and streams*hidden a multiple of 8 no wider than " + std::to_string(kMaxWidth) +
            "; got streams=" + std::to_string(streams) + " hidden=" + std::to_string(hidden));
    }
}

std::size_t shared_bytes(std::int32_t streams, std::int32_t hidden) {
    const int rows = (2 + streams) * streams;
    return static_cast<std::size_t>(hidden) * streams * sizeof(__nv_bfloat16) +
           static_cast<std::size_t>(kWarps) * rows * sizeof(float) +
           static_cast<std::size_t>(streams) * sizeof(float);
}

unsigned grid_for(std::int64_t count) {
    return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

} // namespace

std::size_t manifold_hyper_connection_mix_workspace_capacity_bytes(std::int32_t streams,
                                                                   std::int32_t hidden,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens) {
    require_geometry(streams, hidden);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("manifold_hyper_connection: invalid token interval");
    }
    // `partials` (one sum of squares plus one dot per projection row, per token) and `pre`.
    const std::int32_t rows = (2 + streams) * streams;
    const std::size_t per_token =
        (static_cast<std::size_t>(rows) + 1 + static_cast<std::size_t>(streams)) * sizeof(float);
    // One arena allocation, rounded the way the arena rounds it.
    const std::size_t bytes = per_token * static_cast<std::size_t>(max_tokens);
    return (bytes + 255U) / 256U * 256U + 256U;
}

void manifold_hyper_connection_mix(const Tensor& residual,
                                   const ManifoldHyperConnectionWeights& weights,
                                   std::int32_t streams, float rms_eps, float hc_eps,
                                   std::int32_t sinkhorn_iterations, Tensor& collapsed,
                                   Tensor& post, Tensor& comb, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    (void)workspace;
    const std::int32_t tokens = residual.ne[1];
    if (tokens <= 0) { return; }
    if (residual.ne[0] % streams != 0) {
        throw std::invalid_argument("manifold_hyper_connection: residual rows are not a whole "
                                    "number of streams");
    }
    const std::int32_t hidden = residual.ne[0] / streams;
    require_geometry(streams, hidden);
    const std::int32_t width = hidden * streams;
    const std::int32_t rows  = (2 + streams) * streams;
    require_contiguous(residual, DType::BF16, width, tokens, "residual");
    require_contiguous(collapsed, DType::BF16, hidden, tokens, "collapsed");
    require_contiguous(post, DType::FP32, streams, tokens, "post");
    if (comb.dtype != DType::FP32 || comb.ne[0] != streams || comb.ne[1] != streams ||
        comb.ne[2] != tokens || comb.ne[3] != 1 || !comb.is_contiguous() || comb.data == nullptr) {
        throw std::invalid_argument("manifold_hyper_connection: invalid comb");
    }
    if (weights.mix.qtype != QType::BF16_CTRL || weights.mix.layout != QuantLayout::Contiguous ||
        weights.mix.qdata == nullptr || weights.mix.ndim != 2 || weights.mix.n != rows ||
        weights.mix.k != width) {
        // The expected shape comes from the residual's own width, so a mismatch is as often
        // the caller handing over the wrong tensor as the weight being wrong. Both are named.
        throw std::invalid_argument(
            "manifold_hyper_connection: mix must be contiguous BF16 [" + std::to_string(rows) +
            "," + std::to_string(width) + "] for a residual of " + std::to_string(streams) +
            " x " + std::to_string(hidden) + "; got n " + std::to_string(weights.mix.n) + ", k " +
            std::to_string(weights.mix.k) + ", ndim " + std::to_string(weights.mix.ndim));
    }
    if (weights.base.dtype != DType::FP32 || weights.base.ne[0] != rows ||
        weights.base.data == nullptr) {
        throw std::invalid_argument("manifold_hyper_connection: base must be FP32 [" +
                                    std::to_string(rows) + "]");
    }
    if (weights.scale.dtype != DType::FP32 || weights.scale.ne[0] != 3 ||
        weights.scale.data == nullptr) {
        throw std::invalid_argument("manifold_hyper_connection: scale must be FP32 [3]");
    }
    if (sinkhorn_iterations < 1) {
        throw std::invalid_argument("manifold_hyper_connection: at least one Sinkhorn iteration");
    }

    // Enough blocks to fill the card, and never more slices than there is width to divide:
    // a decode round has one token and would otherwise run a single CTA against the
    // projection's whole 786 KB. Two blocks per SM is the point where more stops helping and
    // the atomics start to cost.
    const int slice_cap = (width + kThreads - 1) / kThreads;
    int slices          = 1;
    {
        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        int multiprocessors = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                                          device_id));
        const int wanted = (2 * multiprocessors + tokens - 1) / tokens;
        slices           = std::max(1, std::min(slice_cap, wanted));
    }

    const std::size_t partial_floats =
        static_cast<std::size_t>(rows + 1) * static_cast<std::size_t>(tokens);
    const std::size_t pre_floats =
        static_cast<std::size_t>(streams) * static_cast<std::size_t>(tokens);
    const DeviceArena::Scope reserved = workspace.scope();
    const DeviceSpan span = workspace.alloc_bytes((partial_floats + pre_floats) * sizeof(float));
    auto* partials        = static_cast<float*>(span.data);
    float* pre            = partials + partial_floats;
    CUDA_CHECK(cudaMemsetAsync(partials, 0, partial_floats * sizeof(float), stream));

    const dim3 project_grid(static_cast<unsigned>(slices), static_cast<unsigned>(tokens));
    manifold_project_kernel<<<project_grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const __nv_bfloat16*>(weights.mix.qdata), width, rows, slices, partials);
    CUDA_CHECK(cudaGetLastError());

    // A thread per token, and the stream count compiled in: the four-stream form is the one
    // every released mHC checkpoint has, and a wider one would need its own instantiation here
    // rather than silently taking a slower path.
    {
        constexpr int kReduceThreads = 128;
        const unsigned grid = (static_cast<unsigned>(tokens) + kReduceThreads - 1) / kReduceThreads;
        if (streams != 4) {
            throw std::invalid_argument(
                "manifold_hyper_connection: only a four-stream residual is compiled; got " +
                std::to_string(streams));
        }
        manifold_reduce_kernel<4><<<grid, kReduceThreads, 0, stream>>>(
            partials, static_cast<const float*>(weights.base.data),
            static_cast<const float*>(weights.scale.data), width, tokens, rows, rms_eps, hc_eps,
            sinkhorn_iterations, pre, static_cast<float*>(post.data),
            static_cast<float*>(comb.data));
    }
    CUDA_CHECK(cudaGetLastError());

    const std::int64_t collapse_count =
        static_cast<std::int64_t>(hidden) * static_cast<std::int64_t>(tokens);
    manifold_collapse_kernel<<<grid_for(collapse_count), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data), pre, hidden, streams, collapse_count,
        static_cast<__nv_bfloat16*>(collapsed.data));
    CUDA_CHECK(cudaGetLastError());
}

void manifold_hyper_connection_combine(const Tensor& block_output, const Tensor& post,
                                       const Tensor& comb, std::int32_t streams, Tensor& residual,
                                       cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    if (tokens <= 0) { return; }
    if (residual.ne[0] % streams != 0) {
        throw std::invalid_argument("manifold_hyper_connection: residual rows are not a whole "
                                    "number of streams");
    }
    const std::int32_t hidden = residual.ne[0] / streams;
    require_geometry(streams, hidden);
    require_contiguous(residual, DType::BF16, hidden * streams, tokens, "residual");
    require_contiguous(block_output, DType::BF16, hidden, tokens, "block_output");
    require_contiguous(post, DType::FP32, streams, tokens, "post");
    if (comb.dtype != DType::FP32 || comb.ne[0] != streams || comb.ne[1] != streams ||
        comb.ne[2] != tokens || !comb.is_contiguous() || comb.data == nullptr) {
        throw std::invalid_argument("manifold_hyper_connection: invalid comb");
    }
    const std::int64_t count = static_cast<std::int64_t>(hidden) * tokens;
    manifold_combine_kernel<<<grid_for(count), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(block_output.data),
        static_cast<const float*>(post.data), static_cast<const float*>(comb.data), hidden,
        streams, count, static_cast<__nv_bfloat16*>(residual.data));
    CUDA_CHECK(cudaGetLastError());
}

void collapse_streams_mean(const Tensor& residual, std::int32_t streams, Tensor& mean,
                           cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    if (tokens <= 0) { return; }
    if (residual.ne[0] % streams != 0) {
        throw std::invalid_argument("collapse_streams_mean: residual rows are not a whole number "
                                    "of streams");
    }
    const std::int32_t hidden = residual.ne[0] / streams;
    require_geometry(streams, hidden);
    require_contiguous(residual, DType::BF16, hidden * streams, tokens, "residual");
    require_contiguous(mean, DType::BF16, hidden, tokens, "mean");
    const std::int64_t count = static_cast<std::int64_t>(hidden) * tokens;
    collapse_mean_kernel<<<grid_for(count), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data), hidden, streams, count,
        static_cast<__nv_bfloat16*>(mean.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops
