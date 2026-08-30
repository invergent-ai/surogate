# The 27B prefill gap

**The number.** On the prefill-heavy shape (2,048-token prompts, 16 out, 100 users) the 27B
processes 7,339 prompt tok/s against vLLM's 11,818 on the same card — 62 %, and the only shape
on the board where we lose badly. TTFT follows: 25.8 s against 14.9 s. On the balanced 512/128
shape we lead (5,815 vs 5,003 prefill, 1,302 vs 1,120 decode), so this is specific to
prompt-dominated traffic, where the prefill duty cycle starves everything else.

**What the round model already says.** A 27B round costs 28.8 ms fixed + 114 µs per column, and
114 µs per column is 474 TFLOP/s against the 651–827 the GEMMs measure standalone. Two causes
were profiled on 2026-08-27 and neither has been addressed:

1. **~18 % of a captured window is gaps between kernels.** With 48 layers each launching
   attention, GDN and MLP work, the launch and dependency overhead is a fixed tax per layer
   that a prompt-heavy round pays 48 times per chunk.
2. **The GDN chunked scan moves 145 MB per layer at 641 GB/s**, well under what the card
   sustains, because the scan runs in chunks sized for decode rather than for a 2k-token
   prompt.

## Closed 2026-08-30: there is no gap on the current binary

Two measurements, an hour apart, took this item apart.

**The kernels were never the limit.** `ninfer_bench` — same binary, same card, no server, no
scheduler — does 12,707 prompt tok/s at pp2048 (13,108 at pp512), *above* vLLM's served 11,818.

**And neither was serving.** Re-measured with the current binary at 100 users, chunk 4,096 and
`--max-model-len 4096`:

| | prefill tok/s | TTFT p50 |
|---|---:|---:|
| surogate, 2026-08-30 | **11,208** | 15.9 s |
| vLLM, 2026-08-27 | 11,818 | 14.9 s |
| surogate, 2026-08-27 (the "gap") | 7,339 | 25.8 s |

95 % of vLLM at a comparable TTFT. `SUROGATE_SERVE_ROUND_TIMING=1` confirms there is nothing
left to find in the scheduler: over 5-second windows the executor spends 4,850 ms of 5,000 in
mixed rounds and **0 ms in boundary, admit, resolve and decode**, with 1 ms of preview and 2 ms
of append. The engine is compute-bound in the rounds themselves, which is where it should be.

So the 7,339 was the 2026-08-27 configuration, not a defect — that pass predates the graph
switch fix, the host-split fixes, the current artifact and the context handling. **Both levers
below are dropped, not deferred**: fusing the layer loop and widening the GDN chunked scan
would have been days of kernel work against a number that no longer exists.

What remains worth knowing: admission caps concurrency at 23 running of 100 offered on this
shape (KV at a 4,096 context with 2,048-token prompts), and the queue drains cleanly
(`waiting` falls 76 → 2 across the run). If prefill-heavy traffic ever needs more, that is the
knob — more KV, not more kernel.

---

*The original analysis is kept below for the record; its two levers are the ones now dropped.*

## Superseded: the 2026-08-27 profile



`ninfer_bench` runs the same engine with no server, no scheduler and no concurrency:

| shape | 27B kernels | 27B served @100 users | vLLM served @100 users |
|---|---:|---:|---:|
| pp512 | 13,108 t/s | — | — |
| **pp2048** | **12,707 t/s** | **7,339 t/s** | 11,818 t/s |
| 35B pp2048 (for scale) | 18,291 t/s | — | — |

**Our prefill kernels are faster than vLLM's served rate.** The served number is 58 % of what
the same binary does on the same card without a scheduler, so the 42 % that goes missing is in
serving, not in the kernels — and the two levers below, both kernel work, would have been the
wrong thing to build. They stay documented but demoted.

Where to look instead, in order: admission and the prefill/decode interleave at 100 users with
2,048-token prompts (a prompt that long spans several chunks, and each chunk rides a mixed
round that also carries decode rows); KV materialisation per chunk; and the chunk ladder
(`--max-num-batched-tokens 4096` against a 2,048-token prompt means one chunk, so the graph
bucket ladder may be capturing or falling back per prompt). `SUROGATE_SERVE_ROUND_TIMING=1`
splits a served round into boundary / admit / mixed / decode / preview / resolve / append, which
is the next measurement.

## Lever 1 — fuse the layer loop (demoted: the kernels are not the limit)

Collapse the per-layer launch sequence so a prompt chunk walks the layers with fewer, larger
kernels: the prologue (norm + projections) and the mixer already have their own launches, and
the gaps between them dominate the 18 %. Candidates, cheapest first:

- **Capture the prefill chunk as one graph per bucket** — already done for the mixed path
  (`PrefillGraphFamily`, 128-token buckets). Check whether the prefill-heavy path (chunk 4,096)
  takes the graph route at all, or falls back to eager because the bucket ladder tops out at
  the effective chunk. If it is eager, this is a configuration fix, not a kernel project.
- **Fuse norm + projection** where the shapes allow, removing one launch per layer per phase.
- Only then consider a persistent-grid layer loop, which is the large version.

**Measure first**: capture one prefill-heavy round under Nsight and split it into kernel time
versus gaps. The 18 % figure is from 2026-08-27, before the graph-switch fix and the current
artifact; it may have moved.

## Lever 2 — widen the GDN chunked scan (demoted for the same reason)

The scan's chunk width is chosen for decode. At 2,048-token prompts the same kernel moves
145 MB per layer at 641 GB/s; a wider chunk amortises the state loads across more columns.
`sequence_chunks` and the chunked-scan geometry are the knobs; the risk is register pressure
and the recurrent state's working set, which is why it was left alone.

**Measure first**: `ninfer_bench -p 2048 -n 16` on the 27B gives the prompt-processing rate in
seconds, and `SUROGATE_SERVE_ROUND_TIMING=1` splits a served round. Sweep the scan width
offline against the bench before touching the server.

## What not to do again

The board's rejected-lever list for the 27B is long and was measured, not argued: lanes, chunk
width, MTP, prefill batching, bf16 KV, and the mixed NVFP4/FP8 artifact all moved nothing or
lost. The two levers above are the only ones the profile actually points at, and both start
with a measurement rather than a patch.
