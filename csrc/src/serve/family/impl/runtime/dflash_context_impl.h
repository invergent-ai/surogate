#include "family/impl/runtime/dflash_context.h"

#include <stdexcept>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS {

DFlashPersistentState::DFlashPersistentState(DeviceSpan backing,
                                             const DFlashPersistentLayout& layout,
                                             family::DecoderState& decoder_state)
    : local(backing, layout.local),
      decoder(decoder_state),
      prefill_features(layout.prefill_features.bind(backing)),
      prefill_positions(layout.prefill_positions.bind(backing)),
      pending_features(layout.pending_features.bind(backing)) {
    if (layout.full) { full.emplace(backing, *layout.full); }
    if (local.layer_count() != layout.geometry.local_layers ||
        local.capacity() != layout.geometry.local_capacity ||
        local.num_kv_heads() != layout.geometry.kv_heads ||
        local.head_dim() != layout.geometry.head_dim ||
        (local.dtype() != DType::BF16 && local.dtype() != DType::FP8_E4M3FN) ||
        bool(full) != (layout.geometry.local_layers < layout.geometry.layers)) {
        throw std::invalid_argument("DFlash persistent local cache layout is invalid");
    }
    if (full && (full->layers() != 1 || full->max_context() != layout.full->max_context ||
        full->pool().plane_count() != 2 || local.lane_capacity() != full->pool().table_row_count() ||
        full->pool().plane(0).dtype != local.dtype() ||
        full->pool().plane(0).ne[0] != layout.geometry.head_dim ||
        full->pool().plane(0).ne[1] != kPagedKVPageSize ||
        full->pool().plane(0).ne[3] != layout.geometry.kv_heads)) {
        throw std::invalid_argument("DFlash persistent full cache layout is invalid");
    }
}

CyclicKVCacheLayerView DFlashPersistentState::local_layer(std::uint32_t layer) const {
    return local.layer_view(layer);
}

PagedKVBatchLayerView DFlashPersistentState::full_batch_layer(std::uint32_t layer) const {
    if (!full) { throw std::logic_error("DFlash has no full attention layer"); }
    return full->batch_layer_view(layer);
}

void DFlashPersistentState::save_rewrite_checkpoint(std::int32_t lane, cudaStream_t stream) {
    decoder.checkpoint_dflash(lane).copy_lane_from(local, lane, 0, stream);
}

void DFlashPersistentState::restore_rewrite_checkpoint(std::int32_t lane, cudaStream_t stream) {
    local.copy_lane_from(decoder.checkpoint_dflash(lane), 0, lane, stream);
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS
