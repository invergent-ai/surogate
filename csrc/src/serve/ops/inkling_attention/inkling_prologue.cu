// Inkling-2-Mini attention prologue, log-n tau, relative-projection table and the dense relative
// logits (test route). ENGINE-INKLING lane c; semantics: api/ops/inkling_attention.h and
// lanes/c/DESIGN-attention.md sections 1 (P1-P8), 2 and 3. Reference: vLLM's
// ops/sconv.py `_attn_prologue_kernel` / `_kv_conv_tile` / `_tap_sum`, ops/norm.py `_l2_clip` /
// `qr_clip_rel_tokens`, ops/attention.py `compute_log_scaling_tau`.

#include "ops/inkling_attention/inkling_attention_common.h"
#include "ops/inkling_attention/inkling_attention_device.cuh"
#include "ops/inkling_attention/inkling_prologue_device.cuh"

#include "core/device.h"
#include "core/pdl.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace inkling_detail {
namespace {

template <bool Global>
__global__ __launch_bounds__(kPrologueThreads) void prologue_kernel(PrologueParams p) {
    // The attention launched after this one may start (its own wait guards what it reads from us).
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();
    if (p.zero != nullptr && blockIdx.x == 0) {
        for (int i = static_cast<int>(threadIdx.x); i < p.zero_count; i += kPrologueThreads) { p.zero[i] = 0; }
    }
    std::int64_t unit = static_cast<std::int64_t>(blockIdx.x) * (kPrologueThreads / 32) +
                        (threadIdx.x >> 5);
    if (unit < p.kv_units) {
        kv_unit<Global>(p, unit);
        return;
    }
    unit -= p.kv_units;
    if (unit < p.q_units) {
        q_unit<Global>(p, unit);
        return;
    }
    unit -= p.q_units;
    if (unit < p.r_units) { r_unit<Global>(p, unit); }
}

__global__ void log_tau_kernel(const std::int32_t* positions, float* tau, std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    tau[i] = log_tau(positions[i]);
}

__global__ void rel_table_kernel(const __nv_bfloat16* rel_proj, __nv_bfloat16* table) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (i >= Geo::kRelDim * Geo::kRelTableRows) { return; }
    const int j = i % Geo::kRelDim;
    const int d = i / Geo::kRelDim - Geo::kRelTablePad;
    table[i]    = (d >= 0 && d < Geo::kRelExtent) ? rel_proj[j * Geo::kRelExtent + d]
                                                  : __float2bfloat16_rn(0.0F);
}

// Test route: one warp per 16 consecutive (column, head) rows, every distance, on the k16 MMA the attention
// kernels use for their bias strips.
__global__ void rel_logits_kernel(const __nv_bfloat16* r_out, const __nv_bfloat16* table,
                                  __nv_bfloat16* rel, std::int64_t rows) {
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int gid           = lane >> 2;
    const int lid           = lane & 3;
    const std::int64_t row0 = (static_cast<std::int64_t>(blockIdx.x) * (blockDim.x >> 5) +
                               (threadIdx.x >> 5)) * 16;
    if (row0 >= rows) { return; }
    const std::int64_t ra = row0 + gid, rb = row0 + gid + 8;
    const auto r_word = [&](std::int64_t row, int col) -> unsigned {
        return row < rows ? load_vec<unsigned>(r_out + row * Geo::kRelDim + col) : 0U;
    };
    const unsigned a0 = r_word(ra, 2 * lid), a1 = r_word(rb, 2 * lid);
    const unsigned a2 = r_word(ra, 2 * lid + 8), a3 = r_word(rb, 2 * lid + 8);
    for (int nt = 0; nt < Geo::kRelExtent / 8; ++nt) {
        const __nv_bfloat16* t = table + (Geo::kRelTablePad + nt * 8 + gid) * Geo::kRelDim + 2 * lid;
        float c0 = 0.0F, c1 = 0.0F, c2 = 0.0F, c3 = 0.0F;
        mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, load_vec<unsigned>(t), load_vec<unsigned>(t + 8));
        const int d = nt * 8 + 2 * lid;
        if (ra < rows) { store_vec(rel + ra * Geo::kRelExtent + d, pack_bf16x2(c0, c1)); }
        if (rb < rows) { store_vec(rel + rb * Geo::kRelExtent + d, pack_bf16x2(c2, c3)); }
    }
}

} // namespace
} // namespace inkling_detail

using inkling_detail::Geo;

void inkling_attention_prologue(const InklingProjections& in, const InklingAttentionWeights& wt,
                                InklingLayerKind kind, const InklingAttentionRows& rows,
                                const InklingConvStateArgs& conv, const PagedKVBatchLayerView& cache,
                                InklingPrologueOutputs& out, cudaStream_t stream) {
    inkling_detail::prologue_launch(in, wt, kind, rows, conv, cache, out, stream, nullptr, 0);
}

void inkling_detail::prologue_launch(const InklingProjections& in, const InklingAttentionWeights& wt,
                                     InklingLayerKind kind, const InklingAttentionRows& rows,
                                     const InklingConvStateArgs& conv, const PagedKVBatchLayerView& cache,
                                     InklingPrologueOutputs& out, cudaStream_t stream, std::int32_t* zero,
                                     std::int32_t zero_count) {
    using namespace inkling_detail;
    PrologueParams p = prologue_params(in, wt, kind, rows, conv, cache, &out, "inkling_attention_prologue");
    p.zero       = zero_count > 0 ? zero : nullptr;
    p.zero_count = zero_count;

    const std::int64_t warps = p.kv_units + p.q_units + p.r_units;
    const pdl::LaunchConfig launch{
        dim3(static_cast<unsigned>((warps + kPrologueThreads / 32 - 1) / (kPrologueThreads / 32))),
        dim3(kPrologueThreads), 0, stream};
    if (kind == InklingLayerKind::Global) {
        CUDA_CHECK(pdl::launch_dependent(launch, prologue_kernel<true>, p));
    } else {
        CUDA_CHECK(pdl::launch_dependent(launch, prologue_kernel<false>, p));
    }
}

inkling_detail::PrologueParams inkling_detail::prologue_params(const InklingProjections& in,
                                                               const InklingAttentionWeights& wt,
                                                               InklingLayerKind kind, const InklingAttentionRows& rows,
                                                               const InklingConvStateArgs& conv,
                                                               const PagedKVBatchLayerView& cache,
                                                               const InklingPrologueOutputs* out, const char* op) {
    using namespace inkling_detail;
    const bool global    = kind == InklingLayerKind::Global;
    const RowsShape rs   = check_rows(rows, op);
    const std::int64_t n = rs.columns;
    check_kv_cache(cache, op);

    PrologueParams p{};
    p.q        = static_cast<const __nv_bfloat16*>(in.q.data);
    p.q_stride = column_stride_elements(in.q, DType::BF16, {Geo::kHeadDim, Geo::kQHeads}, n, 2, op, "q");
    p.r        = static_cast<const __nv_bfloat16*>(in.r.data);
    p.r_stride = column_stride_elements(in.r, DType::BF16, {Geo::kRelDim, Geo::kQHeads}, n, 2, op, "r", 2);
    p.k        = static_cast<const __nv_bfloat16*>(in.k.data);
    p.k_stride = column_stride_elements(in.k, DType::BF16, {1024}, n, 1, op, "k");
    p.v        = static_cast<const __nv_bfloat16*>(in.v.data);
    p.v_stride = column_stride_elements(in.v, DType::BF16, {1024}, n, 1, op, "v");
    if (global) {
        p.k_dyn        = static_cast<const __nv_bfloat16*>(in.k_dyn.data);
        p.k_dyn_stride = column_stride_elements(in.k_dyn, DType::BF16, {4, Geo::kKvHeads}, n, 2, op, "k_dyn");
        p.v_dyn        = static_cast<const __nv_bfloat16*>(in.v_dyn.data);
        p.v_dyn_stride = column_stride_elements(in.v_dyn, DType::BF16, {4, Geo::kKvHeads}, n, 2, op, "v_dyn");
    } else if (in.k_dyn.data != nullptr || in.v_dyn.data != nullptr) {
        fail(op, "a local layer takes no dynamic conv taps");
    }

    require_contiguous(wt.q_scale, DType::BF16, Geo::kQHeads * Geo::kHeadDim, op, "q_scale");
    require_contiguous(wt.r_scale, DType::BF16, Geo::kQHeads * Geo::kRelDim, op, "r_scale");
    require_contiguous(wt.k_conv, DType::BF16, 4 * 1024, op, "k_conv");
    require_contiguous(wt.v_conv, DType::BF16, 4 * 1024, op, "v_conv");
    for (const Tensor* w : {&wt.k_conv, &wt.v_conv}) {
        if (global ? (w->ne[0] != 1024 || w->ne[1] != 4) : (w->ne[0] != 4 || w->ne[1] != 1024)) {
            fail(op, "conv weights are [4, 1024] (local attn/k_conv) or [1024, 4] (global attn/k_conv_static)");
        }
    }
    p.q_scale = static_cast<const __nv_bfloat16*>(wt.q_scale.data);
    p.r_scale = static_cast<const __nv_bfloat16*>(wt.r_scale.data);
    p.k_conv  = static_cast<const __nv_bfloat16*>(wt.k_conv.data);
    p.v_conv  = static_cast<const __nv_bfloat16*>(wt.v_conv.data);

    p.positions     = static_cast<const std::int32_t*>(rows.positions.data);
    p.valid_columns = rs.masked ? static_cast<const std::int32_t*>(rows.valid_columns.data) : nullptr;
    p.table_rows    = static_cast<const std::int32_t*>(rows.kv_table_rows.data);
    p.block_tables  = static_cast<const std::int32_t*>(cache.block_tables.data);
    p.table_stride  = cache.block_tables.ne[0];

    // Conv state: the visit's slots [C, 3, S]; K at k_offset, V at v_offset, 4 channels a lane (8-byte loads).
    const std::int32_t channels_needed = std::max(conv.k_offset, conv.v_offset) + 1024;
    if (conv.states.data == nullptr || conv.states.dtype != DType::BF16 || conv.states.ne[0] < channels_needed ||
        conv.states.ne[1] != 3 || conv.states.nb[0] != 2 || conv.states.nb[1] % 16 != 0 ||
        conv.states.nb[2] % 16 != 0 || reinterpret_cast<std::uintptr_t>(conv.states.data) % 16 != 0 ||
        conv.k_offset < 0 || conv.v_offset < 0 || conv.k_offset % 4 != 0 || conv.v_offset % 4 != 0) {
        fail(op, "conv states are BF16 [C, 3, S] (dim 0 contiguous, 16-byte strides) holding K at k_offset and "
                 "V at v_offset (multiples of 4)");
    }
    require_contiguous(conv.slots, DType::I32, rs.batch, op, "conv slots");
    p.states            = static_cast<__nv_bfloat16*>(conv.states.data);
    p.state_tap_stride  = conv.states.nb[1] / 2;
    p.state_slot_stride = conv.states.nb[2] / 2;
    p.k_offset          = conv.k_offset;
    p.v_offset          = conv.v_offset;
    p.slots             = static_cast<const std::int32_t*>(conv.slots.data);
    p.record            = conv.mode == InklingConvMode::Record;
    p.capture_column    = -1;
    if (p.record) {
        const Tensor& rec = conv.records;
        if (rec.data == nullptr || rec.dtype != DType::BF16 || rec.ne[0] < channels_needed ||
            rec.ne[1] < rs.width || rec.ne[2] < rs.batch || rec.nb[0] != 2 || rec.nb[1] % 8 != 0 ||
            rec.nb[2] % 8 != 0 || reinterpret_cast<std::uintptr_t>(rec.data) % 8 != 0) {
            fail(op, "Record mode needs records BF16 [C, W_max >= W, >= B], dim 0 contiguous, 8-byte strides");
        }
        p.records           = static_cast<__nv_bfloat16*>(rec.data);
        p.record_col_stride = rec.nb[1] / 2;
        p.record_row_stride = rec.nb[2] / 2;
        if (conv.capture_column >= 0) { fail(op, "a rewrite checkpoint capture runs in InPlace mode only"); }
    } else if (conv.capture_column >= 0) {
        if (rs.batch != 1 || conv.capture_column > rs.width || conv.capture_slot < 0 ||
            conv.capture_slot >= conv.states.ne[2]) {
            fail(op, "a capture needs B = 1, capture_column in [0, W] and a capture slot of the pool");
        }
        p.capture_column = conv.capture_column;
        p.capture_slot   = conv.capture_slot;
    }

    p.k_pages = static_cast<std::uint8_t*>(cache.k_pages.data);
    p.v_pages = static_cast<std::uint8_t*>(cache.v_pages.data);

    if (out != nullptr) { // null: the decode attention's in-kernel prologue (q / r stay in its shared memory)
        require_contiguous(out->q, DType::BF16, n * Geo::kQHeads * Geo::kHeadDim, op, "q_out");
        require_contiguous(out->r, DType::BF16, n * Geo::kQHeads * Geo::kRelDim, op, "r_out");
        p.q_out = static_cast<__nv_bfloat16*>(out->q.data);
        p.r_out = static_cast<__nv_bfloat16*>(out->r.data);
    }

    p.width = rs.width;
    p.batch = rs.batch;
    // Decode / verify rows run in one K/V unit; a prompt row is cut into 16-column units (>= 4 columns, so
    // only unit 0 reaches back into the slot).
    p.kv_chunk  = rs.width <= Geo::kMaxSmallColumns ? rs.width : 16;
    p.kv_chunks = (rs.width + p.kv_chunk - 1) / p.kv_chunk;
    p.kv_units  = static_cast<std::int64_t>(rs.batch) * Geo::kKvHeads * 2 * p.kv_chunks;
    p.q_units   = n * Geo::kQHeads;
    p.r_units   = n * (Geo::kQHeads / 2);

    return p;
}

void inkling_log_tau(const Tensor& positions, Tensor& tau, cudaStream_t stream) {
    using namespace inkling_detail;
    const std::int64_t n = positions.numel();
    if (positions.dtype != DType::I32 || !positions.is_contiguous() || tau.dtype != DType::FP32 ||
        !tau.is_contiguous() || tau.numel() != n) {
        throw std::invalid_argument("inkling_log_tau: positions contiguous I32, tau contiguous FP32, same count");
    }
    if (n == 0) { return; }
    log_tau_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), static_cast<float*>(tau.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_rel_proj_table(const Tensor& rel_proj, Tensor& table, cudaStream_t stream) {
    using namespace inkling_detail;
    require_contiguous(rel_proj, DType::BF16, Geo::kRelDim * Geo::kRelExtent, "inkling_rel_proj_table", "rel_proj");
    require_contiguous(table, DType::BF16, Geo::kRelDim * Geo::kRelTableRows, "inkling_rel_proj_table", "table");
    constexpr int total = Geo::kRelDim * Geo::kRelTableRows;
    rel_table_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(rel_proj.data), static_cast<__nv_bfloat16*>(table.data));
    CUDA_CHECK(cudaGetLastError());
}

void inkling_rel_logits(const Tensor& r_out, const Tensor& rel_table, Tensor& rel, cudaStream_t stream) {
    using namespace inkling_detail;
    const char* op = "inkling_rel_logits";
    if (r_out.dtype != DType::BF16 || !r_out.is_contiguous() || r_out.numel() % (Geo::kRelDim * Geo::kQHeads) != 0) {
        throw std::invalid_argument(std::string(op) + ": r_out is contiguous BF16 [16, 24, N]");
    }
    const std::int64_t rows = r_out.numel() / Geo::kRelDim;
    require_contiguous(rel_table, DType::BF16, Geo::kRelDim * Geo::kRelTableRows, op, "rel_table");
    require_contiguous(rel, DType::BF16, rows * Geo::kRelExtent, op, "rel");
    if (rows == 0) { return; }
    const std::int64_t warps = (rows + 15) / 16;
    rel_logits_kernel<<<static_cast<unsigned>((warps + 3) / 4), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(r_out.data), static_cast<const __nv_bfloat16*>(rel_table.data),
        static_cast<__nv_bfloat16*>(rel.data), rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops
