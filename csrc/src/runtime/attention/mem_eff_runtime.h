#pragma once
#include <cstddef>
#include <functional>
#include "runtime/attention/attention_backend.h"
namespace dsl {
using MemEffScratchAllocator = std::function<std::byte*(std::size_t)>;
// Inputs include validated shape and device cu_seqlens. The runtime wrappers
// synthesize dense boundaries and supply their fixed-capacity scratch arena.
// Internal interface also permits testing the exact production dispatch path.
void mem_eff_forward_with_scratch(AttentionParams&, const MemEffScratchAllocator&);
void mem_eff_backward_with_scratch(AttentionParams&, const MemEffScratchAllocator&);
// Key-split count the backward launches for `num_keys` keys and num_batches x Hq
// (batch, head) pairs: 1 by default (bitwise reproducible), when `deterministic_bwd`
// or when not causal; SUROGATE_MEM_EFF_KEY_SPLITS=<n> forces a count (bounded by the
// key blocks) and =auto picks about two CTAs per SM — both non-deterministic opt-ins.
int mem_eff_backward_key_splits(const AttentionParams& p, int num_keys, int num_batches, int Hq);
}  // namespace dsl
