#pragma once

#include "core/tensor.h"

namespace sinfer::family::detail {

// Per-column facts a layer prologue (an n-gram memory, for instance) needs about the columns
// of one forward: the token ids, the column where each column's segment starts, the
// persistent state slot, and whether the column ends its segment. All I32 [T] on device.
struct PrologueColumns {
    Tensor ids;
    Tensor segment_begin;
    Tensor slots;
    Tensor segment_last;
};

} // namespace sinfer::family::detail
