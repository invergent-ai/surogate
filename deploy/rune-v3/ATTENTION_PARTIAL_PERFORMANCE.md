# Rune v3 wide-head attention — 2026-09-27

This follows the [prefill reducer optimization](PREFILL_PERFORMANCE.md). The baseline is
PR #247 (`e8cea0c1`), with the same local NVFP4-experts/BF16-rest artifact and 1120-token
image budget.

## Bottleneck and change

The 512-wide BF16 attention kernel held query fragments and a full-width output accumulator
in each warp. A representative 4096-key, seven-lane standalone launch compiled to 255
registers/thread with a 712-byte stack frame and 1392 bytes each of spill stores and loads.
Nsight Compute measured 3,107,328 local-memory spilling requests, only four theoretical
active warps per SM (8.33% occupancy), and low compute/memory utilization. Shared memory
limited each SM to one block. These measurements support a latency and occupancy problem;
they are not a claim that all attention work is bandwidth-bound.

Dense cached prompt tiles with 512-wide heads now use eight warps in two output groups.
Each warp accumulates 256 output dimensions while both groups share the same staged K/V.
QK and softmax work are repeated per group in the same arithmetic order. Only the first
group writes the shared maximum and normalization outputs, so the result has one writer
per element. Workspace layout, absolute key partitions, FP32 partials and BF16 conversion
are unchanged. Narrow decode, cache-append, sparse attention, int8 and other head widths
keep their existing routes. Both BF16 and FP8 caches use the new dense prompt geometry.

This trades duplicated QK work and 4 KiB more shared memory for smaller accumulators and
more active warps sharing one K/V load. The representative candidate compiled to a
328-byte stack frame and 636 bytes each of spill stores and loads. Register spilling is
reduced, not eliminated. The candidate's detailed profile confirms eight theoretical
active warps per SM and 16.65% achieved occupancy. Total spilling requests fell to
2,808,064 (about 9.6% lower): the per-thread reduction is larger than the aggregate
reduction because the block now has twice as many threads.

## Isolated checks

A paired standalone benchmark compared the production candidate kernel with the previous
kernel over 17 cases: 128–16384 keys, 1/7/16 lanes, sliding windows and masked lanes,
including an empty lane. Every FP32 numerator, maximum and normalization result matched
byte for byte. Four alternating baseline/candidate repeats used 30 timed launches each.

| Keys / lanes | Baseline | Candidate |
|---|---:|---:|
| 128 / 1 | 65.96 µs | 43.05 µs |
| 4096 / 7 | 202.39 µs | 141.80 µs |
| 16384 / 16 | 1303.91 µs | 909.29 µs |

All 17 cases improved. A CPU build ran during these isolated measurements; whole-model
capacity measurements are run separately after all builds finish. Compute Sanitizer
racecheck reported zero hazards/errors/warnings, and memcheck reported zero errors on the
masked sliding-window case. Sanitizer timings are not performance measurements.

## Whole-model measurements

Fresh baseline and candidate runs used one RTX PRO 6000, the same local artifact,
1120 image tokens, 64 scheduler lanes, 16384-token context, automatic KV, 8192-token
prefill window and speculation off. Admission limits were disabled to measure capacity.
All builds finished before these runs. Each text level ran for 60 seconds; the image
level ran for 30 seconds. These are single sweeps, not confidence intervals, and the
completed request mixes differ slightly between timed runs.

| Clients | Baseline req/s | Candidate req/s | Change | Baseline median | Candidate median |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.68 | 11.26 | +5.4% | 40.8 ms | 39.6 ms |
| 8 | 25.31 | 26.58 | +5.0% | 210.8 ms | 200.4 ms |
| 32 | 26.74 | 28.05 | +4.9% | 1112.3 ms | 1060.0 ms |

At four clients, image throughput was 6.11 → 6.20 requests/s (+1.5%),
with median 642.4 → 633.7 ms. No requests failed in either sweep.

All **500 text, 48 image and eight thinking** answer pairs returned HTTP 200 and
matched exactly, including probabilities and thinking text, without numerical tolerance.
The small thinking sample had median 4.167 → 4.232 seconds;
this does not establish a thinking speed improvement. The narrow decode route is unchanged.

## Regression checks

- Full GQA attention suite passed in 66.52 seconds, including BF16/FP8 caches, absolute
  partition boundaries, sliding windows, masked lanes and bitwise packing invariance.
  Added seven- and nine-query cases exercise both sides of the eight-query dispatch.
- Bidirectional attention suite passed in 30.08 seconds.
- Engine, native CLI and Python extension built successfully for sm_120a.

## Local records

Under `/home/flavius/work/agent/`:

- `results/partial-baseline.ncu-rep`, `results/partial-candidate.ncu-rep`: detailed kernel profiles.
- `results/partial-output-micro/`: paired kernel timings and validation notes.
- `partial_compare.cu`, `partial_baseline.cuh`: standalone comparison harness and original kernel.
- `logs/partial-checks.log`: exact comparisons and sanitizer results.
- `logs/partial-output-build.log`, `logs/partial-output-full-build.log`: serving builds.
- `logs/partial-output-validation.log`: full GPU suites and whole-model benchmark sessions.
- `results/partial-output-control/`, `results/partial-output-v1/`: baseline/candidate request results.

## Deployment state

Validation completed; frozen binaries are in `agent/bin/partial-output-v1/`.
The installed release remains PR #247. GPU access is reserved by the user, so the
validated candidate has not been deployed.
