#include "api/ops/ngram_ple.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/bf16/bf16_cublaslt.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

constexpr int kThreads    = 256;
constexpr int kMaxStreams = 8;

__constant__ std::int8_t kIq4nlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                             1,    13,   25,  38,  53,  69,  89,  113};

__device__ __forceinline__ float block_sum(float value, float* scratch) {
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

// The k-th predecessor of column t: inside the segment it is a column of this call, before
// the segment start it comes from the slot history (oldest first), and -1 marks "none".
__device__ __forceinline__ int predecessor(const int* ids, const int* history, int t, int k,
                                           int begin, int slot, int history_len, int slots) {
    const int source = t - k;
    if (source >= begin) { return ids[source]; }
    const int back = begin - source; // 1 = the column just before the segment
    if (back > history_len) { return -1; }
    return history[slot * history_len + (history_len - back)];
}

__global__ void rows_kernel(const int* __restrict__ ids, const int* __restrict__ segment_begin,
                            const int* __restrict__ slots, const int* __restrict__ history,
                            NgramPleHash hash, int history_len, int slot_count,
                            int* __restrict__ rows) {
    const int t = static_cast<int>(blockIdx.x);
    const int h = static_cast<int>(threadIdx.x);
    if (h >= hash.heads) { return; }
    const int begin = segment_begin[t];
    const int slot  = slots[t];
    std::int64_t ctx[kNgramPleMaxNgram];
    ctx[0]   = ids[t];
    bool cut = false;
    for (int k = 1; k < hash.ngram; ++k) {
        const int token = cut ? -1 : predecessor(ids, history, t, k, begin, slot, history_len, slot_count);
        cut             = cut || token < 0 || token == hash.eos_token;
        ctx[k]          = cut ? hash.eos_token : token;
    }
    const int per_gram = hash.heads / (hash.ngram - 1);
    const int order    = 2 + h / per_gram; // 2 for the bigram heads, 3 for the trigram heads
    std::uint64_t mixed = static_cast<std::uint64_t>(ctx[0]) * hash.multipliers[0];
    for (int j = 1; j < order; ++j) {
        mixed ^= static_cast<std::uint64_t>(ctx[j]) * hash.multipliers[j];
    }
    rows[static_cast<std::int64_t>(t) * hash.heads + h] = static_cast<int>(
        mixed % static_cast<std::uint64_t>(hash.head_vocab_sizes[h]) + hash.head_offsets[h]);
}

// The last column of a segment rewrites its slot's history with the segment's last tokens
// (older entries come from the previous history when the segment is shorter than it).
__global__ void history_update_kernel(const int* __restrict__ ids,
                                      const int* __restrict__ segment_begin,
                                      const int* __restrict__ slots,
                                      const int* __restrict__ segment_last, int history_len,
                                      int slot_count, int* __restrict__ history) {
    const int t = static_cast<int>(blockIdx.x);
    if (segment_last[t] == 0 || threadIdx.x != 0) { return; }
    const int begin = segment_begin[t];
    const int slot  = slots[t];
    int fresh[kNgramPleMaxNgram];
    for (int j = 0; j < history_len; ++j) {
        // entry j (oldest first) is the token history_len-j columns before the next column
        const int back = history_len - j;
        const int source = t + 1 - back;
        if (source >= begin) {
            fresh[j] = ids[source];
        } else {
            const int older = begin - source; // 1 = the column just before the segment
            fresh[j]        = older > history_len ? -1 : history[slot * history_len + (history_len - older)];
        }
    }
    for (int j = 0; j < history_len; ++j) { history[slot * history_len + j] = fresh[j]; }
}

// Decodes head rows into the [heads*head_dim, T] embedding (BF16) from IQ4_NL blocks.
__global__ void gather_kernel(const int* __restrict__ rows, const std::uint8_t* __restrict__ table,
                              std::int64_t row_count, int row_bytes, int head_dim, int heads,
                              __nv_bfloat16* __restrict__ embedding) {
    const int t       = static_cast<int>(blockIdx.x);
    const int values  = heads * head_dim;
    const int blocks  = head_dim / 32;
    for (int v = static_cast<int>(threadIdx.x); v < values; v += kThreads) {
        const int head  = v / head_dim;
        const int within = v - head * head_dim;
        const int block = within / 32;
        const int lane  = within - block * 32;
        const std::int64_t row = rows[static_cast<std::int64_t>(t) * heads + head];
        const std::uint8_t* data =
            table + row * row_bytes + static_cast<std::int64_t>(block) * 18;
        const float scale = __half2float(__ushort_as_half(
            static_cast<unsigned short>(data[0] | (static_cast<unsigned>(data[1]) << 8))));
        const std::uint8_t packed = data[2 + (lane & 15)];
        const int nibble          = lane < 16 ? (packed & 0x0F) : (packed >> 4);
        (void)blocks;
        (void)row_count;
        embedding[static_cast<std::int64_t>(t) * values + v] =
            __float2bfloat16_rn(scale * static_cast<float>(kIq4nlValues[nibble]));
    }
}

// One block per token: per-stream gates from the normalised key and residual streams, then
// gv = sigmoid(gate) * value and its per-stream normalisation for the convolution.
__global__ void gate_kernel(const __nv_bfloat16* __restrict__ residual,
                            const __nv_bfloat16* __restrict__ key,
                            const __nv_bfloat16* __restrict__ value,
                            const float* __restrict__ norm_key,
                            const float* __restrict__ norm_query,
                            const float* __restrict__ norm_conv, int hidden, int streams,
                            float eps, __nv_bfloat16* __restrict__ gated,
                            __nv_bfloat16* __restrict__ gated_normalized) {
    __shared__ float scratch[kThreads / 32];
    __shared__ float inv_key[kMaxStreams];
    __shared__ float inv_query[kMaxStreams];
    __shared__ float gate[kMaxStreams];
    const int t             = static_cast<int>(blockIdx.x);
    const int width         = hidden * streams;
    const std::int64_t base = static_cast<std::int64_t>(t) * width;
    const __nv_bfloat16* x  = residual + base;
    const __nv_bfloat16* k  = key + base;
    const __nv_bfloat16* v  = value + static_cast<std::int64_t>(t) * hidden;

    for (int s = 0; s < streams; ++s) {
        float sum_k = 0.0F;
        float sum_x = 0.0F;
        for (int d = static_cast<int>(threadIdx.x); d < hidden; d += kThreads) {
            const float kv = __bfloat162float(k[s * hidden + d]);
            const float xv = __bfloat162float(x[s * hidden + d]);
            sum_k          = fmaf(kv, kv, sum_k);
            sum_x          = fmaf(xv, xv, sum_x);
        }
        const float total_k = block_sum(sum_k, scratch);
        const float total_x = block_sum(sum_x, scratch);
        if (threadIdx.x == 0) {
            inv_key[s]   = rsqrtf(total_k / static_cast<float>(hidden) + eps);
            inv_query[s] = rsqrtf(total_x / static_cast<float>(hidden) + eps);
        }
    }
    __syncthreads();
    for (int s = 0; s < streams; ++s) {
        float dot = 0.0F;
        for (int d = static_cast<int>(threadIdx.x); d < hidden; d += kThreads) {
            const int i    = s * hidden + d;
            const float kn = __bfloat162float(k[i]) * inv_key[s] * norm_key[i];
            const float qn = __bfloat162float(x[i]) * inv_query[s] * norm_query[i];
            dot            = fmaf(kn, qn, dot);
        }
        const float total = block_sum(dot, scratch);
        if (threadIdx.x == 0) {
            const float scaled    = total * rsqrtf(static_cast<float>(hidden));
            const float magnitude = sqrtf(fmaxf(fabsf(scaled), 1e-6F));
            gate[s]               = sigmoid(copysignf(magnitude, scaled));
        }
    }
    __syncthreads();
    // gv_s = gate_s * value; its RMS is |gate_s| times the RMS of value.
    float sum_v = 0.0F;
    for (int d = static_cast<int>(threadIdx.x); d < hidden; d += kThreads) {
        const float vv = __bfloat162float(v[d]);
        sum_v          = fmaf(vv, vv, sum_v);
    }
    const float mean_v2 = block_sum(sum_v, scratch) / static_cast<float>(hidden);
    for (int s = 0; s < streams; ++s) {
        const float g   = gate[s];
        const float inv = rsqrtf(mean_v2 * g * g + eps);
        for (int d = static_cast<int>(threadIdx.x); d < hidden; d += kThreads) {
            const int i      = s * hidden + d;
            const float gv   = g * __bfloat162float(v[d]);
            gated[base + i]  = __float2bfloat16_rn(gv);
            gated_normalized[base + i] = __float2bfloat16_rn(gv * inv * norm_conv[i]);
        }
    }
}

// Column t of channel c, `back` positions before it: inside the segment a column of this
// call, before it the slot's convolution state (oldest first), zero beyond the state.
__device__ __forceinline__ float conv_input(const __nv_bfloat16* normalized,
                                            const __nv_bfloat16* conv_state, int width, int c,
                                            int t, int back, int begin, int slot, int history,
                                            int slots) {
    const int source = t - back;
    if (source >= begin) {
        return __bfloat162float(normalized[static_cast<std::int64_t>(source) * width + c]);
    }
    const int older = begin - source; // 1 = the column just before the segment
    if (older > history) { return 0.0F; }
    const int column = history - older;
    return __bfloat162float(
        conv_state[(static_cast<std::int64_t>(slot) * width + c) * history + column]);
}

// residual += gv + silu(sum_k w[c,k] * input(t - (kernel-1-k)*dilation)).
__global__ void ple_conv_kernel(const __nv_bfloat16* __restrict__ gated,
                            const __nv_bfloat16* __restrict__ normalized,
                            const __nv_bfloat16* __restrict__ weight,
                            const __nv_bfloat16* __restrict__ conv_state,
                            const int* __restrict__ segment_begin, const int* __restrict__ slots,
                            int width, int kernel, int dilation, int history, int slot_count,
                            __nv_bfloat16* __restrict__ residual) {
    const int t = static_cast<int>(blockIdx.y);
    const int c = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (c >= width) { return; }
    const int begin = segment_begin[t];
    const int slot  = slots[t];
    float acc       = 0.0F;
    for (int k = 0; k < kernel; ++k) {
        const int back = (kernel - 1 - k) * dilation;
        const float w  = __bfloat162float(weight[static_cast<std::int64_t>(k) * width + c]);
        acc = fmaf(w, conv_input(normalized, conv_state, width, c, t, back, begin, slot, history,
                                 slot_count),
                   acc);
    }
    const std::int64_t i = static_cast<std::int64_t>(t) * width + c;
    const float value = __bfloat162float(residual[i]) + __bfloat162float(gated[i]) + silu(acc);
    residual[i]       = __float2bfloat16_rn(value);
}

// The last column of a segment rewrites its slot's convolution state with the last `history`
// normalised columns (older ones from the previous state when the segment is shorter).
__global__ void ple_conv_state_update_kernel(const __nv_bfloat16* __restrict__ normalized,
                                         const int* __restrict__ segment_begin,
                                         const int* __restrict__ slots,
                                         const int* __restrict__ segment_last, int width,
                                         int history, int slot_count,
                                         __nv_bfloat16* __restrict__ conv_state) {
    const int t = static_cast<int>(blockIdx.y);
    if (segment_last[t] == 0) { return; }
    const int c = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    if (c >= width) { return; }
    const int begin = segment_begin[t];
    const int slot  = slots[t];
    float fresh[16];
    for (int j = 0; j < history; ++j) {
        // entry j (oldest first) is history-j columns before the next column
        fresh[j] = conv_input(normalized, conv_state, width, c, t + 1, history - j, begin, slot,
                              history, slot_count);
    }
    for (int j = 0; j < history; ++j) {
        conv_state[(static_cast<std::int64_t>(slot) * width + c) * history + j] =
            __float2bfloat16_rn(fresh[j]);
    }
}

__global__ void mark_last_kernel(int* __restrict__ flags, const int* __restrict__ count, int base,
                                 int columns) {
    int index = base + *count - 1;
    if (index < 0) { index = 0; }
    if (index >= columns) { index = columns - 1; }
    flags[index] = 1;
}

void require_shape(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t tokens,
                   const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != rows || tensor.ne[1] != tokens ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument(std::string("ngram_ple: invalid ") + name);
    }
}

void require_vector(const Tensor& tensor, DType dtype, std::int32_t n, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != n || tensor.numel() != n || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("ngram_ple: invalid ") + name);
    }
}

void require_bf16_weight(const Weight& weight, std::int32_t n, std::int32_t k, const char* name) {
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.qdata == nullptr || weight.ndim != 2 || weight.n != n || weight.k != k) {
        throw std::invalid_argument(std::string("ngram_ple: ") + name + " must be BF16 [" +
                                    std::to_string(n) + "," + std::to_string(k) + "]");
    }
}

unsigned grid_for(std::int64_t count) {
    return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

} // namespace

void ngram_ple_mark_segment_last(Tensor& flags, const Tensor& count_scalar, std::int32_t base,
                                 cudaStream_t stream) {
    if (flags.dtype != DType::I32 || flags.numel() <= 0 || flags.data == nullptr ||
        count_scalar.dtype != DType::I32 || count_scalar.data == nullptr) {
        throw std::invalid_argument("ngram_ple_mark_segment_last: invalid operands");
    }
    mark_last_kernel<<<1, 1, 0, stream>>>(static_cast<int*>(flags.data),
                                          static_cast<const int*>(count_scalar.data), base,
                                          static_cast<int>(flags.numel()));
    CUDA_CHECK(cudaGetLastError());
}

std::size_t ngram_ple_workspace_capacity_bytes(std::int32_t streams, std::int32_t hidden,
                                               std::int32_t embed_dim, std::int32_t heads,
                                               std::int32_t min_tokens, std::int32_t max_tokens) {
    if (streams < 1 || streams > kMaxStreams || hidden <= 0 || embed_dim <= 0 || heads <= 0 ||
        heads > kNgramPleMaxHeads || min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("ngram_ple: invalid workspace query");
    }
    const auto round   = [](std::size_t bytes) { return (bytes + 255) / 256 * 256; };
    const auto tokens  = static_cast<std::size_t>(max_tokens);
    const auto width   = static_cast<std::size_t>(streams) * hidden;
    // rows I32 [heads,T], embedding BF16 [embed,T], key BF16 [width,T], value BF16 [hidden,T],
    // gated BF16 [width,T], normalised BF16 [width,T]
    return round(static_cast<std::size_t>(heads) * tokens * 4) +
           round(static_cast<std::size_t>(embed_dim) * tokens * 2) + round(width * tokens * 2) +
           round(static_cast<std::size_t>(hidden) * tokens * 2) + 2 * round(width * tokens * 2) +
           6 * 256;
}

void ngram_ple_forward(Tensor& residual, const NgramPleColumns& columns, const NgramPleHash& hash,
                       const NgramPleTable& table, const NgramPleWeights& weights,
                       NgramPleState& state, std::int32_t streams, std::int32_t conv_kernel,
                       std::int32_t conv_dilation, float eps, WorkspaceArena& workspace,
                       cudaStream_t stream) {
    const std::int32_t width  = residual.ne[0];
    const std::int32_t tokens = residual.ne[1];
    if (streams < 1 || streams > kMaxStreams || (width % streams) != 0) {
        throw std::invalid_argument("ngram_ple: residual width is not a stream multiple");
    }
    const std::int32_t hidden = width / streams;
    if (hash.ngram < 2 || hash.ngram > kNgramPleMaxNgram || hash.heads <= 0 ||
        hash.heads > kNgramPleMaxHeads || (hash.heads % (hash.ngram - 1)) != 0) {
        throw std::invalid_argument("ngram_ple: invalid hash geometry");
    }
    if (table.rows == nullptr || table.row_count <= 0 || table.head_dim <= 0 ||
        (table.head_dim % 32) != 0 || table.row_bytes != table.head_dim / 32 * 18) {
        throw std::invalid_argument("ngram_ple: invalid table");
    }
    for (int h = 0; h < hash.heads; ++h) {
        if (hash.head_vocab_sizes[h] <= 0 || hash.head_offsets[h] < 0 ||
            static_cast<std::int64_t>(hash.head_offsets[h]) + hash.head_vocab_sizes[h] >
                table.row_count) {
            throw std::invalid_argument("ngram_ple: head range exceeds the table");
        }
    }
    const std::int32_t embed_dim = hash.heads * table.head_dim;
    const std::int32_t history   = (conv_kernel - 1) * conv_dilation;
    if (conv_kernel < 1 || conv_dilation < 1 || history > 16 || !(eps > 0.0F)) {
        throw std::invalid_argument("ngram_ple: invalid convolution geometry or eps");
    }
    require_shape(residual, DType::BF16, width, tokens, "residual");
    require_vector(columns.ids, DType::I32, tokens, "ids");
    require_vector(columns.segment_begin, DType::I32, tokens, "segment_begin");
    require_vector(columns.slots, DType::I32, tokens, "slots");
    require_vector(columns.segment_last, DType::I32, tokens, "segment_last");
    require_bf16_weight(weights.key, width, embed_dim, "key");
    require_bf16_weight(weights.value, hidden, embed_dim, "value");
    require_vector(weights.norm_key, DType::FP32, width, "norm_key");
    require_vector(weights.norm_query, DType::FP32, width, "norm_query");
    require_vector(weights.norm_conv, DType::FP32, width, "norm_conv");
    if (weights.convolution.dtype != DType::BF16 || weights.convolution.ne[0] != width ||
        weights.convolution.ne[1] != conv_kernel || weights.convolution.numel() !=
                                                        static_cast<std::int64_t>(conv_kernel) * width ||
        !weights.convolution.is_contiguous() || weights.convolution.data == nullptr) {
        throw std::invalid_argument("ngram_ple: convolution must be BF16 [streams*hidden, kernel]");
    }
    const std::int32_t slot_count = state.history.ne[1];
    if (state.history.dtype != DType::I32 || state.history.ne[0] != hash.ngram - 1 ||
        slot_count <= 0 || !state.history.is_contiguous() || state.history.data == nullptr) {
        throw std::invalid_argument("ngram_ple: history must be I32 [ngram-1, slots]");
    }
    if (state.conv_state.dtype != DType::BF16 || state.conv_state.ne[0] != history ||
        state.conv_state.ne[1] != width || state.conv_state.ne[2] != slot_count ||
        !state.conv_state.is_contiguous() || state.conv_state.data == nullptr) {
        throw std::invalid_argument(
            "ngram_ple: conv_state must be BF16 [history, streams*hidden, slots]");
    }

    auto scope        = workspace.scope();
    Tensor rows       = workspace.alloc(DType::I32, {hash.heads, tokens});
    Tensor embedding  = workspace.alloc(DType::BF16, {embed_dim, tokens});
    Tensor key        = workspace.alloc(DType::BF16, {width, tokens});
    Tensor value      = workspace.alloc(DType::BF16, {hidden, tokens});
    Tensor gated      = workspace.alloc(DType::BF16, {width, tokens});
    Tensor normalized = workspace.alloc(DType::BF16, {width, tokens});

    const auto* ids   = static_cast<const int*>(columns.ids.data);
    const auto* begin = static_cast<const int*>(columns.segment_begin.data);
    const auto* slots = static_cast<const int*>(columns.slots.data);
    const auto* last  = static_cast<const int*>(columns.segment_last.data);
    auto* history_ptr = static_cast<int*>(state.history.data);
    auto* conv_state  = static_cast<__nv_bfloat16*>(state.conv_state.data);

    rows_kernel<<<static_cast<unsigned>(tokens), 32, 0, stream>>>(
        ids, begin, slots, history_ptr, hash, hash.ngram - 1, slot_count,
        static_cast<int*>(rows.data));
    CUDA_CHECK(cudaGetLastError());
    history_update_kernel<<<static_cast<unsigned>(tokens), 32, 0, stream>>>(
        ids, begin, slots, last, hash.ngram - 1, slot_count, history_ptr);
    CUDA_CHECK(cudaGetLastError());
    gather_kernel<<<static_cast<unsigned>(tokens), kThreads, 0, stream>>>(
        static_cast<const int*>(rows.data), static_cast<const std::uint8_t*>(table.rows),
        table.row_count, table.row_bytes, table.head_dim, hash.heads,
        static_cast<__nv_bfloat16*>(embedding.data));
    CUDA_CHECK(cudaGetLastError());
    detail::bf16_cublaslt_gemm(weights.key, embedding, key, stream);
    detail::bf16_cublaslt_gemm(weights.value, embedding, value, stream);
    gate_kernel<<<static_cast<unsigned>(tokens), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const __nv_bfloat16*>(key.data),
        static_cast<const __nv_bfloat16*>(value.data),
        static_cast<const float*>(weights.norm_key.data),
        static_cast<const float*>(weights.norm_query.data),
        static_cast<const float*>(weights.norm_conv.data), hidden, streams, eps,
        static_cast<__nv_bfloat16*>(gated.data), static_cast<__nv_bfloat16*>(normalized.data));
    CUDA_CHECK(cudaGetLastError());
    const dim3 conv_grid(grid_for(width), static_cast<unsigned>(tokens));
    ple_conv_kernel<<<conv_grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gated.data),
        static_cast<const __nv_bfloat16*>(normalized.data),
        static_cast<const __nv_bfloat16*>(weights.convolution.data), conv_state, begin, slots,
        width, conv_kernel, conv_dilation, history, slot_count,
        static_cast<__nv_bfloat16*>(residual.data));
    CUDA_CHECK(cudaGetLastError());
    ple_conv_state_update_kernel<<<conv_grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(normalized.data), begin, slots, last, width, history,
        slot_count, conv_state);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops
