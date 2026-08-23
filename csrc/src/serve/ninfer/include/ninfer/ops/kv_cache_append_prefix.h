#pragma once

#include "core/cyclic_kv_cache.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Host execution-resource promise for kv_cache_append_prefix.
 *
 * Every device count must remain in this inclusive interval on every replay. The envelope fixes
 * launch resources; it does not select or publish committed frontiers.
 */
struct KVCacheAppendPrefixExecutionEnvelope {
    std::uint32_t min_count = 0;
    std::uint32_t max_count = 0;
};

/**
 * Op: append device-selected exact K/V prefixes to batched paged growing-cache storage.
 *
 * k/v are contiguous BF16 [128,8,W,B], positions is contiguous device I32 [W,B], counts and
 * table_rows are contiguous device I32 [B]. For every row b and i in [0,counts[b]), the Op copies
 * k/v[:, :, i, b] bit-for-bit into logical cache position positions[i,b] through table row
 * table_rows[b]. No cache byte for any rejected physical tail is written. Inputs are unchanged,
 * and the Op neither decides nor publishes a frontier. The paged K/V planes use the DFlash Full
 * head-major order [128,64,Nphysical,8].
 *
 * The caller guarantees positive W, 0 <= counts[b] <= W within the declared envelope, valid
 * sequential nonnegative positions, pairwise non-aliasing, and materialized block-table entries
 * for every position allowed by the envelope.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& table_rows,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            PagedKVBatchLayerView cache, cudaStream_t stream);

/**
 * Op: append device-selected exact K/V prefixes to lane-owned cyclic cache storage.
 *
 * k/v, positions, and counts have the same batch geometry as the paged overload; lanes[b] selects
 * the destination cache lane. Absolute position p maps to physical slot p mod 4096. The caller
 * guarantees that each row's live interval ends immediately before positions[0,b] and that
 * advancing it by counts[b] makes every overwritten old slot dead. One row may commit at most the
 * ring capacity, so no two live writes race for one physical slot.
 */
void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& lanes,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            CyclicKVCacheLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops
