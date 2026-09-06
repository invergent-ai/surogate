#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace sinfer::ops::detail::kimi_delta_net {

void launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk,
                      const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                      cudaStream_t stream);

void launch_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                               const Tensor& beta, float scale, bool normalize_qk,
                               Tensor& ssm_states, const Tensor& valid_columns,
                               const Tensor& initial_state_slots,
                               const Tensor& snapshot_base_slots, Tensor& out,
                               cudaStream_t stream);

/// The replay-record form. Records raw k, v and the per-channel gate (with beta beside it) of
/// every valid column, from a state it reads and never writes.
void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& beta_record, Tensor& out, cudaStream_t stream);

/// The fold of a diagonal-gate record prefix into the all-layer state, the delta net's fold with
/// the other transition. Rows and states are the delta net's own types.
void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const gated_delta_net::GdnReplayFoldKernelRows& rows,
                        std::int32_t active_rows, cudaStream_t stream);

} // namespace sinfer::ops::detail::kimi_delta_net
