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
}
