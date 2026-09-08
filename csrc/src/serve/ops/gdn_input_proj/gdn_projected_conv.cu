#include "ops/gdn_input_proj/gdn_projected_conv.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

template <int Channels, int QueryRows, int KeyRows, int ValueRows, int StaticWidth, class Publish>
__global__ void gdn_projected_conv_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ state_read, const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ initial_state_slots, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, std::int32_t width,
    Publish publish, std::int32_t runtime_channels = 0, std::int32_t runtime_query_rows = 0,
    std::int32_t runtime_key_rows = 0, std::int32_t runtime_value_rows = 0,
    std::int32_t weight_channel_stride = 1, std::int32_t weight_tap_stride = 0) {
    static_assert(Channels == QueryRows + KeyRows + ValueRows);
    const std::int32_t channels = Channels ? Channels : runtime_channels;
    const std::int32_t query_rows = Channels ? QueryRows : runtime_query_rows;
    const std::int32_t key_rows = Channels ? KeyRows : runtime_key_rows;
    const std::int32_t value_rows = Channels ? ValueRows : runtime_value_rows;
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= channels) { return; }
    const std::int32_t batch = static_cast<std::int32_t>(blockIdx.y);
    if constexpr (StaticWidth != 0) { width = StaticWidth; }

    std::int32_t valid                 = valid_columns == nullptr ? width : valid_columns[batch];
    valid                              = valid < 0 ? 0 : (valid > width ? width : valid);
    const std::int64_t slot_stride = static_cast<std::int64_t>(channels) * 3;
    const std::int64_t initial_base =
        static_cast<std::int64_t>(initial_state_slots[batch]) * slot_stride;
    float s0       = __bfloat162float(state_read[initial_base + row]);
    float s1       = __bfloat162float(state_read[initial_base + channels + row]);
    float s2       = __bfloat162float(state_read[initial_base + 2LL * channels + row]);
    const auto tap_stride = weight_tap_stride ? weight_tap_stride : channels;
    const auto weight_row = static_cast<std::int64_t>(row) * weight_channel_stride;
    const float w0 = __bfloat162float(conv_weight[weight_row]);
    const float w1 = __bfloat162float(conv_weight[tap_stride + weight_row]);
    const float w2 = __bfloat162float(conv_weight[2LL * tap_stride + weight_row]);
    const float w3 = __bfloat162float(conv_weight[3LL * tap_stride + weight_row]);

    for (std::int32_t token = 0; token < width; ++token) {
        const std::int64_t column = static_cast<std::int64_t>(batch) * width + token;
        if (token >= valid) {
            if (row < query_rows) {
                query[column * query_rows + row] = __float2bfloat16_rn(0.0F);
            } else if (row < query_rows + key_rows) {
                key[column * key_rows + row - query_rows] = __float2bfloat16_rn(0.0F);
            } else {
                value[column * value_rows + row - query_rows - key_rows] = __float2bfloat16_rn(0.0F);
            }
            continue;
        }

        const float p              = __bfloat162float(projected[column * channels + row]);
        float conv                 = fmaf(w0, s0, 0.0F);
        conv                       = fmaf(w1, s1, conv);
        conv                       = fmaf(w2, s2, conv);
        conv                       = fmaf(w3, p, conv);
        const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
        if (row < query_rows) {
            query[column * query_rows + row] = output;
        } else if (row < query_rows + key_rows) {
            key[column * key_rows + row - query_rows] = output;
        } else {
            value[column * value_rows + row - query_rows - key_rows] = output;
        }
        publish.publish(token, batch, row, s1, s2, p);
        s0 = s1;
        s1 = s2;
        s2 = p;
    }
}

template <int Channels, int QueryRows, int KeyRows, int ValueRows, class Publish>
void launch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
            const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query,
            Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    constexpr int kDefaultThreads = 256;
    const std::int32_t width      = projected.ne[1];
    const std::int32_t batch      = projected.ne[2];
    if constexpr (Channels == 10240) {
        if (width == 4 && batch == 1) {
            constexpr int kT4Threads = 64;
            gdn_projected_conv_kernel<Channels, QueryRows, KeyRows, ValueRows, 4>
                <<<(Channels + kT4Threads - 1) / kT4Threads, kT4Threads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(projected.data),
                    static_cast<const __nv_bfloat16*>(conv_weight.data),
                    static_cast<const __nv_bfloat16*>(state_read.data),
                    valid_columns.data == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(initial_state_slots.data),
                    static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
                    static_cast<__nv_bfloat16*>(value.data), width, publish);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
    const dim3 grid((Channels + kDefaultThreads - 1) / kDefaultThreads,
                    static_cast<unsigned>(batch));
    gdn_projected_conv_kernel<Channels, QueryRows, KeyRows, ValueRows, 0>
        <<<grid, kDefaultThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<const __nv_bfloat16*>(state_read.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(initial_state_slots.data),
            static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data), width, publish);
    CUDA_CHECK(cudaGetLastError());
}

template <class Publish>
void dispatch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
              const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query,
              Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    if (conv_weight.is_contiguous() && projected.ne[0] == 10240 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 6144) {
        launch<10240, 2048, 2048, 6144>(projected, conv_weight, state_read, valid_columns,
                                        initial_state_slots, query, key, value, publish, stream);
        return;
    }
    if (conv_weight.is_contiguous() && projected.ne[0] == 8192 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 4096) {
        launch<8192, 2048, 2048, 4096>(projected, conv_weight, state_read, valid_columns,
                                       initial_state_slots, query, key, value, publish, stream);
        return;
    }
    // qwen3_5 0.8B/2B share one GDN geometry (16x128 keys/values): required for
    // any max_concurrency > 1 (batch decode/verify snapshot path).
    if (conv_weight.is_contiguous() && projected.ne[0] == 6144 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 2048) {
        launch<6144, 2048, 2048, 2048>(projected, conv_weight, state_read, valid_columns,
                                       initial_state_slots, query, key, value, publish, stream);
        return;
    }
    // GLM-5.3-Flash's Kimi Delta Attention: 64 heads of 128 for q, k and v alike.
    if (conv_weight.is_contiguous() && projected.ne[0] == 24576 && query.ne[0] == 8192 && key.ne[0] == 8192 &&
        value.ne[0] == 8192) {
        launch<24576, 8192, 8192, 8192>(projected, conv_weight, state_read, valid_columns,
                                        initial_state_slots, query, key, value, publish, stream);
        return;
    }
    const auto channels = projected.ne[0];
    if (channels <= 0 || query.ne[0] <= 0 || key.ne[0] <= 0 || value.ne[0] <= 0 ||
        static_cast<std::int64_t>(query.ne[0]) + key.ne[0] + value.ne[0] != channels) {
        throw std::invalid_argument("GDN projected-conv outputs do not cover the projected channels");
    }
    constexpr int threads = 256;
    const dim3 grid((channels + threads - 1) / threads, static_cast<unsigned>(projected.ne[2]));
    gdn_projected_conv_kernel<0, 0, 0, 0, 0><<<grid, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(state_read.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data), projected.ne[1], publish,
        channels, query.ne[0], key.ne[0], value.ne[0], conv_weight.nb[0] / 2, conv_weight.nb[1] / 2);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gdn_projected_conv_snapshot_launch(const Tensor& projected, const Tensor& conv_weight,
                                        Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        const Tensor& snapshot_base_slots, Tensor& query,
                                        Tensor& key, Tensor& value, cudaStream_t stream) {
    dispatch(projected, conv_weight, conv_states, valid_columns, initial_state_slots, query, key,
             value,
             SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                    static_cast<const std::int32_t*>(snapshot_base_slots.data),
                                    projected.ne[0]},
             stream);
}

void gdn_projected_conv_record_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                      const Tensor& conv_states, const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, Tensor& query, Tensor& key,
                                      Tensor& value, cudaStream_t stream) {
    dispatch(conv_record, conv_weight, conv_states, valid_columns, initial_state_slots, query, key,
             value, NoHistoryPublish{}, stream);
}

} // namespace sinfer::ops::detail
