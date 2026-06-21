// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// Debug-only dispatch-PP contiguous block sub-range parity entry points. Thin
// forwarders over MultiGPUPyTrainer's per-GPU debug methods, used by the parity
// test to compare whole-graph vs sub-range execution. See dispatch_pp_phase0.h.

#include "runtime/executor/dispatch_pp_phase0.h"

#include "binding/py_train.h"

namespace dsl::dispatch_pp_phase0 {

std::vector<float> forward_hidden_whole(MultiGPUPyTrainer& trainer, const std::int32_t* inputs) {
    return trainer.dispatch_pp_forward_hidden(inputs);
}

std::vector<float> forward_hidden_subranges(MultiGPUPyTrainer& trainer,
                                            const std::int32_t* inputs,
                                            int split_after_block) {
    return trainer.dispatch_pp_forward_subranges(inputs, split_after_block);
}

std::vector<float> grad_norms_whole(MultiGPUPyTrainer& trainer,
                                    const std::int32_t* inputs,
                                    const std::int32_t* targets) {
    return trainer.dispatch_pp_grad_norms_whole(inputs, targets);
}

std::vector<float> grad_norms_subranges(MultiGPUPyTrainer& trainer,
                                        const std::int32_t* inputs,
                                        const std::int32_t* targets,
                                        int split_after_block) {
    return trainer.dispatch_pp_grad_norms_subranges(inputs, targets, split_after_block);
}

}  // namespace dsl::dispatch_pp_phase0
