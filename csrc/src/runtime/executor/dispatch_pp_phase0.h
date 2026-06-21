#pragma once

// =============================================================================
// Dispatch-PP feasibility gate — debug-only contiguous block sub-range execution.
//
// PURPOSE
//   Prove the compiled GraphExecutor can run a *contiguous transformer-block
//   sub-range* [block_lo .. block_hi] on a single GPU with resident weights and
//   a CPU-boundary activation handoff, matching whole-graph forward/backward
//   numerically. This is the gate that everything else (DispatchScheduler, stage
//   streaming, async optimizer) depends on. No scheduler, no streaming overlap,
//   no external stage weights — those are later plans.
//
// EXECUTOR-BOUNDARY FINDINGS (csrc/src/runtime/executor, dsl):
//   * CompiledGraph exposes per-block op ranges:
//       - layer_start_indices[L]  inclusive first op of block L (graph_compiler.h:607)
//       - layer_end_indices[L]    EXCLUSIVE one-past-last op of block L  (:608)
//     i.e. block range [i..j] maps to op span
//       [layer_start_indices[i], layer_end_indices[j])   (start-inclusive, end-exclusive).
//     SIZE_MAX marks a block absent from this graph (graph_compiler.cpp:2176).
//   * Block boundaries are cleanly separable. annotate_layer_boundaries()
//     (graph_compiler.cpp:2175-2282) opens/closes a block only on non-gradient
//     param/activation refs; gradient-only ops are absorbed into the surrounding
//     span so no op straddles a boundary. Every op is internal to one block, opens
//     it (op.layer_start>=0), or closes it (op.layer_end>=0). The only cross-block
//     dependency is the residual hidden state carried block->block, so execution
//     can stop after block j and resume at block j+1 with just that tensor.
//   * The eager flat-ops loop (compiled_ops_execute_forward.cpp) dispatches ops by
//     index and polls op.layer_start / op.layer_end to drive the per-layer
//     handlers. This harness reuses that exact machinery via debug op-range bounds
//     (CompiledExecutor::set_debug_forward_op_range / _backward_op_range): two
//     contiguous segments run on one shared executor state, in eager mode (no
//     CUDA-graph capture, no instruction-stream path).
//   * The inter-block boundary residual is a stable slot buffer
//     (block_activation_ptr(rs, block, TensorSlot::BlockHOut) with MLPDown /
//     ResidualAtt fallbacks, mirroring GraphExecutor::resolve_last_block_output).
//     Round-tripping it through host memory between segments proves the handoff.
//
// SCOPE
//   Single GPU, weights RESIDENT (supplied == the resident params). This isolates
//   "can the executor run a sub-range correctly" from "can we stream weights"
//   (the later stage-streaming plan's first gate). All entry points below are
//   debug-only and exposed through MultiGPUPyTrainer for the parity test; no
//   production scheduler/runtime is built here.
//
// VERDICT (validated by tests/train/dispatch_pp/test_phase0_subrange.py on
// Qwen3-0.6B/4-layer, single RTX 5090):
//   * FORWARD: running blocks [0..k] then [k+1..last] as two segments on one
//     shared executor state, with the boundary residual round-tripped through
//     host memory, matches the whole-graph final hidden state (rtol 1e-2). The
//     stop/resume + CPU activation handoff works end-to-end.
//   * BACKWARD: the bounded (block op-range, forced-eager) backward executor
//     matches the whole-graph per-block grad norms (rtol 2e-2). Running the
//     block ranges as two SEPARATE backward invocations that share accumulated-
//     gradient state across a CPU boundary needs multi-stage grad-accumulation
//     ownership the executor does not expose; that is deferred to the
//     DispatchScheduler / async-optimizer plans.
//   Gate result: PASS — contiguous block sub-range execution is feasible.
// =============================================================================

#include <cstdint>
#include <vector>

class MultiGPUPyTrainer;

namespace dsl::dispatch_pp_phase0 {

// Whole-graph forward; returns the final hidden state flattened as float32.
std::vector<float> forward_hidden_whole(MultiGPUPyTrainer& trainer,
                                        const std::int32_t* inputs);

// Forward run as two contiguous block sub-ranges [0..split_after_block] then
// [split_after_block+1..last], with the boundary hidden state round-tripped
// through host memory. Returns the final hidden state flattened as float32.
std::vector<float> forward_hidden_subranges(MultiGPUPyTrainer& trainer,
                                            const std::int32_t* inputs,
                                            int split_after_block);

// Whole-graph backward; returns deterministic per-block weight-grad L2 norms in
// ascending block order.
std::vector<float> grad_norms_whole(MultiGPUPyTrainer& trainer,
                                    const std::int32_t* inputs,
                                    const std::int32_t* targets);

// Backward run as two contiguous block sub-ranges (high range first, boundary
// grad round-tripped through host, then low range), matching the dependency
// direction a future scheduler enforces. Returns per-block grad norms in the
// same ascending block order as grad_norms_whole.
std::vector<float> grad_norms_subranges(MultiGPUPyTrainer& trainer,
                                        const std::int32_t* inputs,
                                        const std::int32_t* targets,
                                        int split_after_block);

}  // namespace dsl::dispatch_pp_phase0
