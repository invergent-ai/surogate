#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops {

// The MTP attention split is a pure data movement, so its row counts are strides
// rather than structure: they arrive per launch and every geometry in the family
// is served by the one kernel. They were compile-time constants of the 27B (q
// 6144, kv 1024, 14336 together), which is what kept the smaller members of the
// family out of speculative decode.

__global__ void mtp_pack_fc_input_kernel(const __nv_bfloat16* embedding_norm,
                                         const __nv_bfloat16* hidden_norm, __nv_bfloat16* out,
                                         std::int32_t rows) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) { return; }

    const int token             = static_cast<int>(blockIdx.y);
    const std::int64_t in_idx   = static_cast<std::int64_t>(token) * rows + row;
    const std::int64_t out_base = static_cast<std::int64_t>(token) * (2 * rows);
    out[out_base + row]         = embedding_norm[in_idx];
    out[out_base + rows + row]  = hidden_norm[in_idx];
}

__global__ void mtp_split_attn_in_kernel(const __nv_bfloat16* attn_in, __nv_bfloat16* q,
                                         __nv_bfloat16* k, __nv_bfloat16* gate, __nv_bfloat16* v,
                                         std::int32_t tokens, std::int32_t q_rows,
                                         std::int32_t kv_rows) {
    const int attn_rows    = 2 * q_rows + 2 * kv_rows;
    const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(attn_rows) * tokens;
    if (idx >= n) { return; }

    const int row             = static_cast<int>(idx % attn_rows);
    const int token           = static_cast<int>(idx / attn_rows);
    const __nv_bfloat16 value = attn_in[idx];

    if (row < q_rows) {
        q[static_cast<std::int64_t>(token) * q_rows + row] = value;
        return;
    }
    if (row < q_rows + kv_rows) {
        const int local                                       = row - q_rows;
        k[static_cast<std::int64_t>(token) * kv_rows + local] = value;
        return;
    }
    if (row < q_rows + kv_rows + q_rows) {
        const int local                                         = row - q_rows - kv_rows;
        gate[static_cast<std::int64_t>(token) * q_rows + local] = value;
        return;
    }

    const int local = row - q_rows - kv_rows - q_rows;
    v[static_cast<std::int64_t>(token) * kv_rows + local] = value;
}

} // namespace sinfer::ops
