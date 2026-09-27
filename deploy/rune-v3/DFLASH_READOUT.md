# DFlash decision readouts

DFlash helps generate multi-token reasoning, but a decision's candidate scores need only
the target model. Engines configured with DFlash now automatically bypass the drafter when
`requested_output_tokens == 1` and `next_token_candidates` is nonempty. This covers ordinary
decisions and the initial/final scoring phases of thinking decisions. The thinking generation
itself still uses DFlash. No API change or per-request switch is required.

The policy applies to text and images. Image requests still reserve their vision buffers and
perform image encoding. Existing saved GPU-prefix requests keep their text-only restrictions.
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
jobs; builds and tests ran with `CUDA_VISIBLE_DEVICES` empty.

**GPU validation is pending**: the user has reserved the GPU. No new throughput or answer-parity
result is claimed. The installed service stays paused on its previously validated release,
with speculation off. Local records are `agent/logs/dflash-bypass-build.log` and
`agent/logs/dflash-bypass-cpu-tests.log`.

`sinfer_readout_policy_test` is CPU-only and checks agreement between admission and resolved
request planning, decision/thinking phase selection, existing GPU-prefix behavior and the
packing policy under concurrent reasoning. The decisions protocol/thinking and prefill-reach
tests also run without a GPU.

`sinfer_dflash_readout_test` is an optional GPU regression, built but not run during the pause.
It loads an ordinary engine and a DFlash engine sequentially from the same paired artifact and
checks:

- Exact serial candidate logits for 2/16/17/256 candidates, around the small-head threshold.
- Packed readouts and readouts concurrent with multi-token generation; readout statistics must
  show no speculation while generation must actually draft tokens.
- Optional image parity and image batches concurrent with generation at 1120 image tokens.
  Serial/packed comparisons allow the same 0.125 absolute logit tolerance as the existing
  candidate-readout test, because BF16 GEMM widths differ across those paths.

Only after the user authorizes GPU work, hold `/home/flavius/work/gpu.lock` for all GPU checks.
Run the new regression with `SUROGATE_DFLASH_READOUT_ARTIFACT` pointing to the local paired
artifact and `SUROGATE_DFLASH_READOUT_IMAGE` pointing to an image fixture. Repeat with
`SUROGATE_DFLASH_READOUT_EAGER=1` to cover the eager route. The existing
`sinfer_candidate_readout_test` should also pass with `SUROGATE_PCD_TEST_ARTIFACT` set to that
artifact and `SUROGATE_PCD_TEST_DFLASH=1`, covering saved-prefix and nested-prefix behavior.

Before merging or deploying, also compare authenticated text/image/thinking decision answers
and throughput against the same target model without DFlash. Exercise cancellation and
repeated alternating traffic to detect stale lane state or leaked vision buffers. Use the
existing frozen attention candidate as the baseline; keep all model artifacts local.
