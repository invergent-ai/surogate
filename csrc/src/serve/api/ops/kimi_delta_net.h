#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops {

/**
 * Op: kimi_delta_net
 *
 * Kimi Delta Attention's recurrence, which is GLM-5.3's linear mixer. It is the gated delta
 * net's recurrence with one difference, and the difference is the whole reason it is a separate
 * op: the forget gate is one value per *key channel* rather than one per head, so the state
 * decays by a diagonal instead of a scalar.
 *
 * For each value head h, with G = value_heads/qk_heads and its Q/K head qh = floor(h/G),
 * starting from S_h and for t in increasing order:
 *
 *   alpha[c]     = exp(g[c,h,t])                       one per key channel
 *   delta        = beta[h,t] * (v[:,h,t] - S_h * (alpha ⊙ k[:,qh,t]))
 *   S_h[:,c]     = alpha[c] * S_h[:,c] + delta * k[c,qh,t]
 *   ideal[:,h,t] = scale * S_h * q[:,qh,t]
 *
 * Read against `gated_delta_net`: there, `alpha` is a scalar and multiplies the whole state, so
 * it can be pulled out of the dot product. Here it belongs inside it -- the prediction the delta
 * corrects is the state *after* this token's decay, and each key channel has decayed by its own
 * amount.
 *
 * Shapes/dtypes are contiguous q/k BF16 [128,Hqk,T], v/out BF16 [128,Hv,T], g FP32 [128,Hv,T]
 * (the per-channel gate, laid out like v), beta FP32 [Hv,T], and state FP32 [128,128,Hv], where
 * Hqk>=1, Hv>=Hqk, and Hv%Hqk==0. `scale` is 1/sqrt(128). When `normalize_qk` is true the
 * implementation consumes raw q/k and applies x / sqrt(sum(x^2) + 1e-6) to every 128-element row
 * before using it; when false they are consumed as supplied.
 *
 * The oracle evaluates the complete recurrence and `ideal` naively in FP64 from the represented
 * inputs and the FP32 initial state. The BF16 out is promoted and compared with that; output
 * storage rounding belongs to the Op's numerical criterion, not the oracle. Inputs and out do
 * not overlap the state or one another. T may be any positive value.
 *
 * This overload reads and writes the same `ssm_state`, publishing it after all T tokens.
 */
void kimi_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                    const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_state,
                    Tensor& out, cudaStream_t stream);

/**
 * Distinct-state form of the same recurrence. `ssm_state_out` receives the final state;
 * `ssm_state_in` and `ssm_state_out` may be disjoint or exactly the same storage. No other
 * argument may overlap either state.
 */
void kimi_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                    const Tensor& beta, float scale, bool normalize_qk,
                    const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                    cudaStream_t stream);

/**
 * Snapshot form for B independent recurrences, the shape a decode round has: q/k are contiguous
 * BF16 [128,Hqk,W,B], v/out BF16 [128,Hv,W,B], `g` FP32 [128,Hv,W,B], `beta` FP32 [Hv,W,B], and
 * `ssm_states` contiguous [128,128,Hv,Slots] in the engine's linear-attention state storage,
 * which is BF16 -- the op checks its extents, not its dtype. `initial_state_slots` and
 * `snapshot_base_slots` are contiguous I32 [B]; `valid_columns` is contiguous I32 [B] with every
 * value in [1,W], or an empty Tensor meaning every row has W valid columns.
 *
 * Row b starts from initial_state_slots[b] and writes the state after valid column j to
 * snapshot_base_slots[b]+j. Invalid-tail output columns are exact BF16 zero and do not mutate
 * state. The caller reserves disjoint complete [base,base+W) intervals and prevents one row from
 * overwriting another row's initial slot; a row may overwrite its own initial slot after loading
 * it. No arena allocation; `ssm_states` is the only persistent state mutated.
 *
 * The delta net's snapshot form with the gate read per key channel instead of per head -- and
 * the same kernel, which is what makes "these two differ in one function" a fact about the code
 * rather than a claim about it.
 */
void kimi_delta_net_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, bool normalize_qk,
                             Tensor& ssm_states, const Tensor& valid_columns,
                             const Tensor& initial_state_slots,
                             const Tensor& snapshot_base_slots, Tensor& out, cudaStream_t stream);

/**
 * Op: kimi_delta_net_replay_record
 *
 * The replay-record form of the same recurrence, which is what lets a speculative round be
 * verified over this mixer: B lanes are evaluated from absolute state slots that are never
 * written, and what a later fold needs to re-derive the state from the accepted prefix is
 * recorded beside the output. Shapes are the snapshot form's -- q/k BF16 [128,Hqk,T,B], v/out
 * BF16 [128,Hv,T,B], g FP32 [128,Hv,T,B], beta FP32 [Hv,T,B], ssm_states [128,128,Hv,S] -- with
 * T in [2,16], B at most the engine's lane cap, and `normalize_qk` always on, as the delta
 * net's record form has it.
 *
 * For each valid transition, key_record BF16 [128,Hqk,T,B], value_record BF16 [128,Hv,T,B],
 * gate_record FP32 [128,Hv,T,B] and beta_record FP32 [Hv,T,B] receive bit-preserving copies of
 * raw k, v, g and beta. The gate is recorded per channel because that is what the transition
 * reads; the delta net records a {g, beta} pair where this records a plane and a scalar. The
 * invalid record suffix is unchanged and the invalid out suffix is exact BF16 zero. Inputs,
 * state, records and out are pairwise non-overlapping.
 */
void kimi_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& g, const Tensor& beta, float scale,
                                  const Tensor& ssm_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots, Tensor& key_record,
                                  Tensor& value_record, Tensor& gate_record, Tensor& beta_record,
                                  Tensor& out, cudaStream_t stream);

} // namespace sinfer::ops
