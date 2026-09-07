#include "ops/gdn_input_proj/ggml/ggml_gdn_input.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/linear/ggml/ggml_dispatch.h"
#include "ops/linear/ggml/ggml_mmvq.cuh"
#include "ops/linear/ggml/ggml_q8_1.h"

#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

namespace gg = sinfer::ops::detail::ggml;

// The qkv half's rows are convolution channels: a finished row goes straight through the shared
// convolution epilogue into query/key/value, so the projected plane the composed route needs
// between the GEMV and the convolution never exists. That plane was never the cost -- at one
// token it is a few kilobytes -- but the launch that consumed it was.
template <class Publish>
struct GgmlConvStore {
    GdnConvEpilogue<Publish> conv;
    __device__ __forceinline__ void store(int row, const float (&value)[1]) const {
        conv.store(row, value);
    }
};

// The z half writes its own destination unchanged.
struct GgmlZStore {
    __nv_bfloat16* z;
    __device__ __forceinline__ void store(int row, const float (&value)[1]) const {
        z[row] = __float2bfloat16_rn(value[0]);
    }
};

template <gg::GgmlType type, class Epilogue, bool SmallK>
void launch_scheduled(const void* blocks, std::int32_t rows, std::int32_t k,
                      const gg::block_q8_1* y, const Epilogue& epilogue, cudaStream_t stream) {
    constexpr int nwarps    = gg::calc_nwarps(1);
    constexpr int rows_per  = gg::calc_rows_per_block(1, SmallK, nwarps);
    const dim3 block(gg::kWarpSize, nwarps);
    const dim3 grid(static_cast<unsigned>((rows + rows_per - 1) / rows_per));
    gg::mul_mat_vec_q<type, 1, Epilogue, SmallK><<<grid, block, 0, stream>>>(
        blocks, y, k, rows, k / gg::Traits<type>::qk, k / gg::QK8_1, epilogue);
}

template <gg::GgmlType type, class Epilogue>
void launch_rows(const void* blocks, std::int32_t rows, std::int32_t k, const gg::block_q8_1* y,
                 const Epilogue& epilogue, cudaStream_t stream) {
    // the same small-K schedule the plain linear picks; these are exactly the short-K shapes
    if (gg::prefers_small_k<type>(1, k)) {
        launch_scheduled<type, Epilogue, true>(blocks, rows, k, y, epilogue, stream);
    } else {
        launch_scheduled<type, Epilogue, false>(blocks, rows, k, y, epilogue, stream);
    }
}

template <class Epilogue>
void launch_typed(gg::GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
                  const gg::block_q8_1* y, const Epilogue& epilogue, cudaStream_t stream) {
    switch (type) {
#define SINFER_GDN_CASE(NAME)                                                                      \
    case gg::GgmlType::NAME: launch_rows<gg::GgmlType::NAME>(blocks, rows, k, y, epilogue, stream); return;
        SINFER_GGML_FOR_EACH_TYPE(SINFER_GDN_CASE)
#undef SINFER_GDN_CASE
    }
    throw std::invalid_argument("ggml gdn_input: unknown GGML type");
}

template <class Publish>
GdnConvEpilogue<Publish> conv_epilogue(const Tensor& conv_weight, const void* state_read,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       Tensor& query, Tensor& key, Tensor& value,
                                       std::int32_t channels, Publish publish) {
    const std::int32_t query_rows = query.ne[0];
    const std::int32_t key_rows   = key.ne[0];
    return {
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(state_read),
        static_cast<const std::int32_t*>(initial_slot.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        channels,
        query_rows,
        key_rows,
        value.ne[0],
        0,
        1,
        0,
        publish,
    };
}

// Quantise the activation once; both halves read the same int8 planes.
const gg::block_q8_1* quantise_once(const Tensor& x, std::int32_t k, void* scratch,
                                    std::size_t scratch_bytes, cudaStream_t stream) {
    if (scratch == nullptr || scratch_bytes < gg::q8_1_bytes(k, 1) ||
        (reinterpret_cast<std::uintptr_t>(scratch) & 15u) != 0) {
        throw std::invalid_argument("ggml gdn_input: activation scratch too small or misaligned");
    }
    auto* y = static_cast<gg::block_q8_1*>(scratch);
    gg::quantize_q8_1_launch(static_cast<const __nv_bfloat16*>(x.data), k, 1, y, stream);
    return y;
}

} // namespace

bool ggml_gdn_input_decode_admits(const Weight& query_key_value, const Weight& z,
                                  std::int32_t batch, std::int32_t width) noexcept {
    // A parent whose rows mix formats is projected by rows through the generic route; the
    // fused kernel decodes one format.
    return batch == 1 && width == 1 && ggml::is_ggml_qtype(query_key_value.qtype) &&
           ggml::is_ggml_qtype(z.qtype) && query_key_value.k == z.k &&
           query_key_value.layout == QuantLayout::GgmlBlocks && z.layout == QuantLayout::GgmlBlocks &&
           query_key_value.segment_count == 0 && z.segment_count == 0;
}

std::size_t ggml_gdn_input_decode_workspace_bytes(std::int32_t input_rows) noexcept {
    return input_rows > 0 ? gg::q8_1_bytes(input_rows, 1) + 256 : 0;
}

void ggml_gdn_input_conv_snapshot_decode_launch(
    const Tensor& x, const Weight& query_key_value, const Weight& z_weight,
    const Tensor& conv_weight, Tensor& conv_states, const Tensor& valid_columns,
    const Tensor& initial_slot, const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
    Tensor& value, Tensor& z, void* scratch, std::size_t scratch_bytes, cudaStream_t stream) {
    const std::int32_t k        = query_key_value.k;
    const std::int32_t channels = query_key_value.n;
    const auto* y               = quantise_once(x, k, scratch, scratch_bytes, stream);
    const GgmlConvStore<SnapshotHistoryPublish> conv{conv_epilogue(
        conv_weight, conv_states.data, valid_columns, initial_slot, query, key, value, channels,
        SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                               static_cast<const std::int32_t*>(snapshot_base_slot.data),
                               channels})};
    launch_typed(ggml::ggml_type_for(query_key_value.qtype), query_key_value.qdata, channels, k, y,
                 conv, stream);
    launch_typed(ggml::ggml_type_for(z_weight.qtype), z_weight.qdata, z_weight.n, k, y,
                 GgmlZStore{static_cast<__nv_bfloat16*>(z.data)}, stream);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_gdn_input_conv_record_decode_launch(
    const Tensor& x, const Weight& query_key_value, const Weight& z_weight,
    const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns,
    const Tensor& initial_slot, Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
    Tensor& z, void* scratch, std::size_t scratch_bytes, cudaStream_t stream) {
    const std::int32_t k        = query_key_value.k;
    const std::int32_t channels = query_key_value.n;
    const auto* y               = quantise_once(x, k, scratch, scratch_bytes, stream);
    const GgmlConvStore<RecordColumnPublish> conv{conv_epilogue(
        conv_weight, conv_states.data, valid_columns, initial_slot, query, key, value, channels,
        RecordColumnPublish{static_cast<__nv_bfloat16*>(conv_record.data), channels, 1})};
    launch_typed(ggml::ggml_type_for(query_key_value.qtype), query_key_value.qdata, channels, k, y,
                 conv, stream);
    launch_typed(ggml::ggml_type_for(z_weight.qtype), z_weight.qdata, z_weight.n, k, y,
                 GgmlZStore{static_cast<__nv_bfloat16*>(z.data)}, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
