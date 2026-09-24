#!/usr/bin/env bash
# FlashAttention-2's deterministic backward picks its dQ split count itself, from the SM count and
# the number of sequences: ceil(num_sm / (batch * heads)). That number decides the order in which
# the per-key-block dQ partials are added, and how many full-size fp32 dQ accumulators the caller
# must allocate. With packed documents "batch" is the document count, so the same document is
# summed in a different order (and needs a different amount of scratch) depending on what it was
# packed with. Let the caller choose: a positive `num_splits` (unused by the backward otherwise)
# overrides the heuristic. Re-running is a no-op.
set -eu
header="csrc/flash_attn/src/flash_bwd_launch_template.h"
[ -f "$header" ] || { echo "patch_flash_attn: $header not found in $(pwd)" >&2; exit 1; }
if ! grep -q 'params.num_splits > 0' "$header"; then
  sed -i 's|gridDimx = (num_sm + params.b \* params.h - 1) / (params.b \* params.h);|gridDimx = params.num_splits > 0 ? params.num_splits : (num_sm + params.b * params.h - 1) / (params.b * params.h);|' "$header"
  grep -q 'params.num_splits > 0' "$header" || {
    echo "patch_flash_attn: the deterministic gridDimx anchor was not found" >&2; exit 1; }
fi
