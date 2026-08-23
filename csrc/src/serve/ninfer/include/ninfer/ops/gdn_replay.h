#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops {

struct GdnReplayFoldRow {
    std::int32_t linear_state_slot;
    std::int32_t commit_columns;
};

/**
 * Op: gdn_replay_fold
 *
 * Replays each row's [0,commit_columns) record prefix across every registered GDN layer and
 * updates the caller-selected absolute linear-attention state slot in place. rows[b] always maps
 * to physical record row b; rows are not filtered, compressed, or reordered. The active row count
 * is rows.size() and must be in [1,records.spec.record_capacity].
 *
 * linear_state_slot is in [0,states.spec.slot_count), is distinct across active rows, and is the
 * same absolute slot used to produce that row's records. commit_columns is in [0,T]. Zero is a
 * strict no-op for the row: no record or state is read and neither recurrent state nor convolution
 * history is written. For a positive extent, the Op consumes raw key/value/{g,beta} records in
 * order, writes the final FP32 recurrent state, and sets convolution history to
 * tail_3(old_history || conv_record[0:commit_columns]).
 *
 * The Op admits the two registered all-layer geometries only, owns no workspace or metadata
 * allocation, and does not read query or generate token output. The four record planes are
 * read-only, disjoint, and do not overlap either state region.
 */
void gdn_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                     std::span<const GdnReplayFoldRow> rows, cudaStream_t stream);

} // namespace ninfer::ops
