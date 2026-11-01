#!/usr/bin/env bash
# FlashAttention-2's deterministic backward picks its dQ split count itself, from the SM count and
# the number of sequences: ceil(num_sm / (batch * heads)). That number decides the order in which
# the per-key-block dQ partials are added, and how many full-size fp32 dQ accumulators the caller
# must allocate. With packed documents "batch" is the document count, so the same document is
# summed in a different order (and needs a different amount of scratch) depending on what it was
# packed with. Let the caller choose: a positive `num_splits` (unused by the backward otherwise)
# overrides the heuristic. Re-running is a no-op. Like patch_minja.sh this edits the fetched source in
# place (a FETCHCONTENT_SOURCE_DIR_FLASH_ATTN checkout would be edited too).
set -eu
header="csrc/flash_attn/src/flash_bwd_launch_template.h"
[ -f "$header" ] || { echo "patch_flash_attn: $header not found in $(pwd)" >&2; exit 1; }
if ! grep -q 'params.num_splits > 0' "$header"; then
  sed -i 's|gridDimx = (num_sm + params.b \* params.h - 1) / (params.b \* params.h);|gridDimx = params.num_splits > 0 ? params.num_splits : (num_sm + params.b * params.h - 1) / (params.b * params.h);|' "$header"
  grep -q 'params.num_splits > 0' "$header" || {
    echo "patch_flash_attn: the deterministic gridDimx anchor was not found" >&2; exit 1; }
fi

# Ordered dQ accumulation for the seqk-parallel backward (the chunked kv-prefix attention uses it). With
# params.dq_semaphore set, the CTA of key block n adds its dQ partial for query block m only after key block n - 1 did
# (one counter per sequence, head and query block), so every dQ element sums its key blocks in order: one fp32 dQ that
# stays in L2 like the atomic path, and bitwise reproducible like the split path, which keeps one fp32 dQ per split
# (at a 131k prefix those ran from DRAM, ~50 TFLOPS against ~160 for the atomic path). Causal masking keeps the key
# blocks that reach a query block a prefix [0, N(m)], so the chain never waits on a block that does not contribute;
# a CTA only waits on the CTA one index below, issued before it.
python3 - csrc/flash_attn/src/flash.h csrc/flash_attn/src/flash_bwd_kernel.h <<'PY'
import sys
params, kernel = sys.argv[1], sys.argv[2]
s = open(params).read()
if "dq_semaphore" not in s:
    anchor = "    bool deterministic;\n    index_t dq_accum_split_stride;\n};"
    if s.count(anchor) != 1:
        sys.exit("patch_flash_attn: Flash_bwd_params anchor not found")
    s = s.replace(anchor, "    bool deterministic;\n    index_t dq_accum_split_stride;\n"
                          "    // surogate: ordered dQ accumulation (flash_bwd_kernel.h); [b][h][dq_semaphore_m_stride] int32, zeroed.\n"
                          "    int *dq_semaphore;\n    int dq_semaphore_m_stride;\n};")
    open(params, "w").write(s)
k = open(kernel).read()
if "dq_semaphore" not in k:
    anchor = ("                CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));\n"
              "                #pragma unroll\n"
              "                for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }\n")
    if k.count(anchor) != 1:
        sys.exit("patch_flash_attn: seqk-parallel dQ atomicAdd anchor not found")
    k = k.replace(anchor,
        "                CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));\n"
        "                if (params.dq_semaphore != nullptr) {\n"
        "                    // surogate: ordered dQ -- key block n_block adds after n_block - 1 did.\n"
        "                    int *sem = params.dq_semaphore + (int64_t(bidb) * params.h + bidh) * params.dq_semaphore_m_stride + m_block;\n"
        "                    if (threadIdx.x == 0) { while (atomicAdd(sem, 0) != n_block) { } }\n"
        "                    __syncthreads();\n"
        "                    #pragma unroll\n"
        "                    for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }\n"
        "                    __threadfence();\n"
        "                    __syncthreads();\n"
        "                    if (threadIdx.x == 0) { atomicExch(sem, n_block + 1); }\n"
        "                } else {\n"
        "                    #pragma unroll\n"
        "                    for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }\n"
        "                }\n")
    open(kernel, "w").write(k)
PY
