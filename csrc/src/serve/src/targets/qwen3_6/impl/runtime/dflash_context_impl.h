#include "targets/qwen3_6/impl/runtime/dflash_context.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

DFlashPersistentState::DFlashPersistentState(DeviceSpan backing,
                                             const DFlashPersistentLayout& layout)
    : local(backing, layout.local),
      rewrite_checkpoint_local(backing, layout.rewrite_checkpoint_local),
      full(backing, layout.full), prefill_features(layout.prefill_features.bind(backing)),
      prefill_positions(layout.prefill_positions.bind(backing)),
      pending_features(layout.pending_features.bind(backing)) {
    if (local.layer_count() != DFlashConfig::local_layers ||
        rewrite_checkpoint_local.layer_count() != DFlashConfig::local_layers ||
        local.capacity() != DFlashConfig::local_capacity ||
        rewrite_checkpoint_local.capacity() != DFlashConfig::local_capacity || full.layers() != 1 ||
        full.max_context() != layout.full.max_context || full.pool().plane_count() != 2 ||
        local.num_kv_heads() != DFlashConfig::kv_heads ||
        rewrite_checkpoint_local.num_kv_heads() != DFlashConfig::kv_heads ||
        local.head_dim() != DFlashConfig::head_dim ||
        rewrite_checkpoint_local.head_dim() != DFlashConfig::head_dim ||
        local.lane_capacity() != rewrite_checkpoint_local.lane_capacity() ||
        local.lane_capacity() != full.pool().table_row_count() ||
        full.pool().plane(0).dtype != DType::BF16 ||
        full.pool().plane(0).ne[0] != DFlashConfig::head_dim ||
        full.pool().plane(0).ne[1] != kPagedKVPageSize ||
        full.pool().plane(0).ne[3] != DFlashConfig::kv_heads) {
        throw std::invalid_argument("DFlash persistent cache layout is invalid");
    }
}

CyclicKVCacheLayerView DFlashPersistentState::local_layer(std::uint32_t layer) const {
    return local.layer_view(layer);
}

PagedKVBatchLayerView DFlashPersistentState::full_batch_layer(std::uint32_t layer) const {
    return full.batch_layer_view(layer);
}

void DFlashPersistentState::save_rewrite_checkpoint(std::int32_t lane, cudaStream_t stream) {
    rewrite_checkpoint_local.copy_lane_from(local, lane, stream);
}

void DFlashPersistentState::restore_rewrite_checkpoint(std::int32_t lane, cudaStream_t stream) {
    local.copy_lane_from(rewrite_checkpoint_local, lane, stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
