# DFlash decision readouts

DFlash helps generate multi-token reasoning, but a decision's candidate scores need only
the target model. Engines configured with DFlash now automatically bypass the drafter when
`requested_output_tokens == 1` and `next_token_candidates` is nonempty. This covers ordinary
decisions and the initial/final scoring phases of thinking decisions. The thinking generation
itself still uses DFlash. No API change or per-request switch is required.

The policy applies to text and images. Image requests still reserve their vision buffers and
perform image encoding. Existing saved GPU-prefix requests keep their text-only restrictions
and admission/packing policy. Increasing their batch width changes the arithmetic of their
full-vocabulary readout; the new grouped-admission path applies only to ordinary candidates.
Ordinary one-token chat requests, which do not ask for candidate logits, keep their existing
sampling path. Other speculative backends are unchanged.

Bypassed requests reserve no drafter KV, collect no drafter features and report speculation
disabled in their per-request statistics. They can use ordinary text prefill graphs and the
small candidate projection. Admission can stage several readouts together, and ready readout
prefills can pack without `SUROGATE_SERVE_DFLASH_PACKED_PREFILL=1`. That opt-in still controls
packing generating DFlash prefills. With active reasoning, readout packs alternate with decode
rounds; they never enter a DFlash verification round. The shared drafter remains loaded for
reasoning requests.

## Validation status

The engine, native CLI, Python extension and both candidate-readout regression targets built
successfully. All five selected CPU tests passed: readout policy, adaptive DFlash, prefill
graph reach, decisions protocol and decisions thinking. Builds used two low-priority CPU
jobs during the GPU pause; builds and CPU tests ran with `CUDA_VISIBLE_DEVICES` empty.

After the user released the GPU, validation found and fixed a saved-prefix regression: the
initial bypass grouped more saved-prefix requests together. The existing candidate-readout
test failed on that candidate and passed on the previous binary. Keeping the existing
saved-prefix scheduling restores its scores without tolerance changes.

The final GPU checks pass on the local paired Rune NVFP4 artifact:

| Check | Result |
|---|---|
| Readouts, images and mixed generation, CUDA graphs | Pass, 96.39 s |
| Same regression, eager execution | Pass, 94.52 s |
| Existing saved/nested-prefix and scalar-scoring regression | Pass, 47.16 s; max logit delta 0 |

Five CPU checks pass again after the scheduling fix. Final serving builds are recorded in
`agent/logs/dflash-prefix-fix-build.log`, CPU results in
`agent/logs/dflash-prefix-fix-cpu-tests.log`, and GPU results in
`agent/logs/dflash-readout-gpu-tests-v5.log`. The frozen candidate is
`agent/bin/dflash-readout-v2/`, source `37ecda39`.

Authenticated HTTP comparisons, mixed-traffic checks and cancellation checks also pass.
The comparison log is `agent/logs/dflash-readout-bench-session.log` and ends with
`BENCH_SESSION_EXIT=0`. Results are under `agent/results/dflash-readout-{off,before,after}/`;
the `after` directory contains `comparison.json`, `question-groups.json` and `stress.json`.

## HTTP performance and answer checks

All three runs used the same local paired NVFP4 artifact on one RTX PRO 6000, with 1120
image tokens, 32 lanes, context16384, prefill8192 and DFlash window7 where enabled. Both
DFlash runs enabled `SUROGATE_SERVE_DFLASH_PACKED_PREFILL=1`. Capacity measurements disabled
production admission limits and used 60 seconds per text level and 30 seconds for images.
These are single sweeps, without confidence intervals.

| Workload | Speculation off | Previous DFlash | DFlash with bypass |
|---|---:|---:|---:|
| Text, 1 client, requests/s | 11.37 | 10.74 | 10.24 |
| Text, 8 clients, requests/s | 26.80 | 19.96 | 22.46 |
| Text, 32 clients, requests/s | 28.91 | 20.32 | 25.59 |
| Images, 4 clients, requests/s | 6.24 | 6.12 | 6.19 |
| Thinking, 32 serial requests, median seconds | 4.259 | 1.780 | 1.736 |

The bypass improves DFlash text capacity by 12.5% at eight clients and 25.9% at 32 clients.
It does not remove all of DFlash's decision overhead: the 32-client result remains 11.5%
below speculation off, and the single-client sweep is 4.7% slower than previous DFlash.
Images show little change. The thinking sample still benefits from speculation, with a
2.45x ratio of median latencies versus off. Every mode generated 12,288 reasoning tokens
across 24 of the 32 requests; the cases entering thinking and the generated answers can
differ. This small sample does not establish an accuracy improvement.

Each mode also returned HTTP200 for 500 text and 48 image requests, both serially and at
eight/four clients, plus the 32 thinking cases. Serial answer checks establish the intended
split:

- All 432 single-question text requests match speculation-off answers exactly.
- All 68 multi-question requests match previous DFlash answers exactly. They retain the
  existing saved-prefix path; 32 differ from speculation off.
- All 48 serial images match both controls exactly.
- Thinking matches 27/32 final choices against off and 29/32 against previous DFlash.
  Thinking text, scores and final choices are not promised to match between speculative
  and ordinary generation.

Across all 1,303 serial text questions, choices match off in 1,295 cases and previous
DFlash in 1,290. Against fixture labels, categorical accuracy is 960/1180 off, 955/1180
previous DFlash and 956/1180 with bypass. At eight clients it is 965/1180, 960/1180 and
961/1180 respectively. Serial image accuracy is 36/48 in every mode; at four clients it
is 38/48, 37/48 and 37/48. Under concurrent scheduling, exact payloads and some choices
differ as batch shapes change; these measurements do not establish universal bitwise parity.

Three mixed rounds of two thinking, four image and eight text requests all succeeded
(42 responses). Four generated streams were cancelled after their first token; the
following decision remained identical after every cancellation. Health remained available,
and final scheduler gauges reported no running, prefilling, decoding, waiting or reserving
requests.

A final check ran the new binary with speculation off before choosing that deployment
default. All 500 text, 48 image and 32 thinking answer payloads match the previous off
binary exactly, all HTTP200. Results are in `agent/results/dflash-readout-final-off/`;
the log `agent/logs/dflash-readout-final-off.log` ends with `FINAL_OFF_EXIT=0`.

## Remaining cost and deployment choice

The saved-prefix path accounts for most of the remaining serial difference. For the 17
requests with at least 17 questions, mean latency is 348ms with speculation off, 616ms
with previous DFlash and 617ms with bypass. The 432 single-question requests average
68.6ms off and 69.7ms with bypass. Widening saved-prefix batches caused the regression
described above; further optimization must preserve their scores. Rune's BF16 LM head
does not qualify for the exact small-candidate projection used by unsegmented GGML heads.

The deployment default remains `RUNE_SPEC=none` for decision capacity. An operator can
select `RUNE_SPEC=dflash`, `RUNE_DRAFT_TOKENS=7` and the local paired artifact when the
thinking latency benefit is more useful. The bypass is automatic in that configuration.
Image budget1120, API-key authentication and the configurable proxy default of one
request/second per IP remain unchanged; the public hostname is still a deployment-day input.

`sinfer_readout_policy_test` is CPU-only and checks agreement between admission and resolved
request planning, decision/thinking phase selection, existing GPU-prefix behavior and the
packing policy under concurrent reasoning. The decisions protocol/thinking and prefill-reach
tests also run without a GPU.

`sinfer_dflash_readout_test` is an optional GPU regression.
It loads an ordinary engine and a DFlash engine sequentially from the same paired artifact and
checks:

- Exact serial candidate logits for 2/16/17/256 candidates, around the small-head threshold.
- Packed readouts and readouts concurrent with multi-token generation; readout statistics must
  show no speculation while generation must actually draft tokens.
- Optional image parity and image batches concurrent with generation at 1120 image tokens.
  Packed comparisons use matching ordinary/DFlash readout batches and a 0.125 absolute logit
  bound. Serial comparisons and serial checks after lane reuse require exact equality.

The original test compared serial and packed execution with a 0.125 logit bound. An ordinary
control reproduced larger differences, including on the synthetic repeated-token prompts:
batching changes BF16/NVFP4 GEMM geometry. Comparing matching packed workloads isolates the
bypass from that existing numerical difference. This does not promise bitwise agreement
between arbitrary batching geometries.

The image fixture uses context8192 and a 2048-token prefill window. Gemma's 1120 soft image
tokens expand into more raw patches, and its bidirectional image span must fit in one text
chunk. Without an image fixture, the smaller context2048/prefill256 configuration remains.

Hold `/home/flavius/work/gpu.lock` for all GPU checks.
Run the new regression with `SUROGATE_DFLASH_READOUT_ARTIFACT` pointing to the local paired
artifact and `SUROGATE_DFLASH_READOUT_IMAGE` pointing to an image fixture. Repeat with
`SUROGATE_DFLASH_READOUT_EAGER=1` to cover the eager route. The existing
`sinfer_candidate_readout_test` should also pass with `SUROGATE_PCD_TEST_ARTIFACT` set to that
artifact and `SUROGATE_PCD_TEST_DFLASH=1`, covering saved-prefix and nested-prefix behavior.

When changing this policy again, repeat authenticated text/image/thinking comparisons,
mixed traffic and cancellation checks alongside the GPU regressions. Preserve the local
artifacts and baseline snapshots so the same weights and workload geometry can be compared.
