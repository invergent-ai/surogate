# Rune v3 prefill reducer — 2026-09-27

The first optimization after the public-serving work preserves the selected local NVFP4
artifact, the 1120-token image budget and the attention arithmetic. It changes how output
dimensions are assigned to GPU blocks during cached prompt attention.

## What changed

The existing reducer launches 256 threads for 64 output values. Each additional 64-value
chunk repeats the same maximum and ordered normalization sum. Cached BF16/FP8 prompt tiles
with width at least eight and at least 64 total query columns now produce up to 256 values
per block. Narrow decode/verification calls and int8 attention keep the existing route.
Key partitions, FP32 partials, summation order, output conversion and workspace size are
unchanged. This avoids a new numerical policy or further quantization.

A fresh eight-client baseline profile, with CUDA graph node tracing and actual inference
readiness, measured 41.84% of GPU kernel time in attention: 26.72% in the calculation kernels
and 15.12% in reduction. An isolated comparison of the same reducer inputs produced exactly
equal BF16 output bits for 64-, 128- and 256-value chunks. Representative timings:

| Head dimension / query columns / splits | 64 values/block | 256 values/block |
|---|---:|---:|
| 256 / 64 / 8 | 13.96 µs | 6.38 µs |
| 256 / 384 / 8 | 68.39 µs | 26.71 µs |
| 512 / 96 / 32 | 72.52 µs | 32.54 µs |

These are kernel timings, not whole-request speedups.

A final candidate profile confirms the wider reducer is selected. For the same
256-dimensional, 384-query packed tile, average reduction time fell from 68.78 µs
(16 × 4 × 384 blocks) to 25.72 µs (16 × 1 × 384 blocks). Reduction accounted for
7.17% of candidate kernel time versus 15.12% in the baseline. Attention calculation
remains 28.45% of candidate kernel time. Profiles have different request mixes and the
baseline overlapped a CPU compilation; throughput claims below use the separate,
unprofiled runs with no concurrent build.

## Whole-model measurements

One RTX PRO 6000, the same artifact, vision enabled at 1120 image tokens, 64 scheduler lanes,
16384-token context, automatic KV, 8192-token prefill window, speculation off. Admission
limits were disabled for capacity measurement. The baseline binary hash matches the
deployed release from PR #246. Baseline and candidate used separate snapshots and the
existing benchmark/request set; no builds ran during these throughput measurements.

| Clients | Baseline req/s | Candidate req/s | Change | Baseline median | Candidate median |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.30 | 10.70 | +3.9% | 43.4 ms | 41.1 ms |
| 8 | 23.32 | 25.32 | +8.6% | 237.4 ms | 213.6 ms |
| 32 | 24.93 | 26.83 | +7.6% | 1218.0 ms | 1102.2 ms |

All requests succeeded. These are single 60-second levels, including completion of requests
still in flight, rather than speed confidence intervals. Completed request mixes differ
slightly between timed runs. On the identical 500-request text comparison, median latency
was 41.65 ms before and 40.40 ms after.

At four clients, image throughput was 6.16 requests/s for both binaries (30-second levels).
The eight-request thinking sample had medians of 4.253 seconds before and 4.326 seconds
after. This change targets prefill; these measurements do not demonstrate a thinking speedup.

## Correctness and build checks

- All **500 text, 48 image and eight thinking** request pairs returned HTTP 200 and had
  exactly matching answer payloads, including probability values and thinking text.
  Comparison used canonical JSON without rounding or numeric tolerance.
- Full GQA attention and bidirectional attention suites passed (67.39 and 30.71 seconds).
  The GQA suite compares queries alone, full prompts and different packing widths bit for
  bit across BF16/FP8 caches, Rune's 256/512-wide head shapes, sliding windows and key
  partition boundaries. Added 63- and 65-token chunks exercise the new dispatch boundary.
- Engine, native CLI and Python extension builds passed for sm_120a.

## Records and next work

Local evidence under `/home/flavius/work/agent/`:

- `results/prefill-attention-baseline/`: baseline profile, kernel categories and isolated
  reducer timings in `reducer-micro.csv`.
- `results/prefill-reducer-control/` and `results/prefill-reducer-v1/`: throughput and
  answer files; the latter contains `comparison.json` and the source patch used for testing.
- `results/prefill-reducer-profile/`: candidate node trace and kernel categories.
- `logs/prefill-reducer-validation.log`: GPU tests and both benchmark sessions.
- `logs/prefill-reducer-full-build.log`: complete serving build.

The attention calculation kernels remain the next prefill target. The baseline reports
255 registers/thread on the busiest BF16 kernels; detailed profiling should establish
whether register pressure, memory traffic or compute limits the useful work before another
kernel change. Vision attention and thinking's dense vector/normalization kernels remain
separate opportunities documented in `RESULTS.md`.
