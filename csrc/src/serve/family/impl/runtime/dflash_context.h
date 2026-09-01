#pragma once
#include "family/impl/runtime/instance.h"

#include "core/cyclic_kv_cache.h"
#include "family/impl/runtime/layouts.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS {

struct DFlashPersistentState {
    CyclicKVCache local;
    CyclicKVCache rewrite_checkpoint_local;
    family::PagedKVCache full;
    Tensor prefill_features;
    Tensor prefill_positions;
    Tensor pending_features;

    DFlashPersistentState(DeviceSpan backing, const DFlashPersistentLayout& layout);

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer) const;
    [[nodiscard]] PagedKVBatchLayerView full_batch_layer(std::uint32_t layer) const;
    void save_rewrite_checkpoint(std::int32_t lane, cudaStream_t stream);
    void restore_rewrite_checkpoint(std::int32_t lane, cudaStream_t stream);
};

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS
