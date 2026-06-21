#pragma once

// =============================================================================
// Dispatch-PP Phase-0 feasibility gate — debug-only sub-range execution.
//
// PURPOSE
//   Prove the compiled GraphExecutor can run a *contiguous transformer-block
//   sub-range* [block_lo .. block_hi] on a single GPU with resident weights and
//   a CPU-boundary activation handoff, matching whole-graph forward/backward
//   numerically. This is the gate that everything else (DispatchScheduler, stage
//   streaming, async optimizer) depends on. No scheduler, no streaming overlap,
//   no external stage weights — those are later plans.
//
// STEP-1 INVESTIGATION FINDINGS (csrc/src/runtime/executor, dsl):
//   * CompiledGraph exposes per-block op ranges:
//       - layer_start_indices[L]  inclusive first op of block L (graph_compiler.h:607)
//       - layer_end_indices[L]    EXCLUSIVE one-past-last op of block L  (:608)
//     i.e. block range [i..j] maps to op span
//       [layer_start_indices[i], layer_end_indices[j])   (start-inclusive, end-exclusive).
//     SIZE_MAX marks a block absent from this graph (graph_compiler.cpp:2176).
//   * Block boundaries are CLEAN (decision gate = PASS). annotate_layer_boundaries()
//     (graph_compiler.cpp:2175-2282) opens/closes a block only on non-gradient
//     param/activation refs; gradient-only ops are absorbed into the surrounding
//     span so no op straddles a boundary. Every op is internal to one block, opens
//     it (op.layer_start>=0), or closes it (op.layer_end>=0). The only cross-block
//     dependency is the residual hidden state carried block->block, so execution
//     can stop after block j and resume at block j+1 with just that tensor.
//   * The eager linear op loop (compiled_ops_execute_forward.cpp:998+) dispatches
//     ops by index and polls op.layer_start / op.layer_end to drive
//     on_fwd_layer_start / handle_layer_end + stack-checkpoint persistence and
//     last-use pruning. This Phase-0 harness reuses that exact per-op machinery
//     bounded to an op range (CompiledExecutor::execute_forward_op_range /
//     execute_backward_op_range), keeping eager mode (no CUDA-graph capture, no
//     split-attention, no instruction-stream path).
//   * Hidden states are readable by name via CompiledExecutor::try_get_tensor /
//     try_get_tensor_fuzzy ("residual_final"/"xF"/"ln_final" for the final hidden;
//     per-block residuals for the inter-block boundary). Injection writes the
//     boundary residual back into the resumed range's input tensor.
//
// SCOPE / HONEST CLAIM
//   Single GPU, weights RESIDENT (supplied == the resident params). This isolates
//   "can the executor run a sub-range correctly" from "can we stream weights"
//   (the later stage-streaming plan's first gate). All entry points below are
//   debug-only and exposed through MultiGPUPyTrainer for the parity test; no
//   production scheduler/runtime is built here.
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
