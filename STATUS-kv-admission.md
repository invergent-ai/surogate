# STATUS: --kv-admission ondemand (checkpoint 2026-10-19 11:57 Bucharest)

Verdict at this checkpoint: **CLEAN** against criteria (1)-(5) of the owner's pre-approval (memory
`kv-admission-merge-preapproved`: "if they are clean, merge directly"). The evidence is in the table below; the
details follow it. Nothing of this work is running: gpubox1 GPU 0 is free and nothing listens on 8470-8479.
The previous checkpoint (11:05) is kept as `/mnt/data/work/rl/kv-admission-2026-10-19/STATUS-kv-admission.checkpoint-1105.md`.

## Verdict against the merge criteria

| # | Criterion (owner, 2026-10-19 10:05) | Evidence | Met |
|---|---|---|---|
| 1 | Forced-preemption divergences explained as near-ties, or a resume bug fixed and re-tested | Text, greedy, no MTP, forced 64x3 vs reserve: every first divergence at or after the first preemption, drift before the resume exactly 0, base top1-top2 gap at the divergence 0.0-0.5 nats (bf16 logit steps); forced vs forced 64/64 identical. Batching alone (no preemption) diverges more: 7/64 identical, gaps up to 1.875. Vision forced 64x2: same picture, gaps 0.0-0.625 | yes |
| 2 | Tiny-KV + 100k stress complete, no OOM / deadlock / lost request, FCFS kept | tiny3: 48/48, 49 real preemptions, FCFS ok. stress (40) and stress2 (24, long answers): all OK with 164-254k-token prompts, FCFS ok, but 0 preemptions. stress3: 6/6 with 3 real preemptions of 164k-token requests (~166k KV tokens recomputed each), victim always the newer request, FCFS ok. vstress2: 32/32 image requests with 96 forced preemptions | yes |
| 3 | Throughput gain shown (BFCL, agentic, 32k cap) | BFCL 3.2x (985 vs 300 tok/s), long agentic 2.7x (768 vs 295 tok/s), 32k cap with 16 concurrent: TTFT p50 0.80 vs 5.74 s and +5% tok/s; the 32k loop outlier showed up once in each mode | yes |
| 4 | All serve tests pass | Full ctest on GPU 0: 197 passed, 27 skipped, 6 failed of 230, none because of this change: 4 cannot load libcudart.so.13 (with the CUDA lib path 3 pass and the sm90 one skips), 2 are W8 expectations that origin/main #310 made stale (the same failures with an origin/main-equivalent build). The opt-in tests that can run here: family_frontend and shutdown_http pass; model_wake_http's 300 ms deadline is shorter than this artifact's 0.4 s wake, on origin/main too | yes |
| 5 | Byte-identical outputs without preemption (64/64) | Reserve vs ondemand, 64 real requests, greedy, MTP: 64/64 without and 64/64 with prefix reuse; no MTP with top-2 logprobs: 64/64; vision: 24/24 | yes |

## Where things are

- Worktree `/mnt/data/work/surogate-wt/serve-kv-admission`, branch `serve-kv-admission`, from origin/main 5004a4c5.
  Local commits only: unsigned, not pushed, no PR.
  - 6b9c9b29 engine change (executor, program, plans, sampler key, flags, metrics, docs)
  - c8a44220 tests (sinfer_kv_admission_test, serve options)
  - de12e110 scripts/kv_admission_e2e.py and DESIGN-kv-admission.md
  - f0ec6a45 bounds the prepared media that on-demand image requests keep for a recompute (half of --media-live-mib)
  - 4c119f44 SUROGATE_SERVE_KV_ADMIT_TRACE: one line per admission, used for the FCFS check
  - 250afbd8 e2e script: logprob-gap analysis, concurrent generate, image and long-answer loads
  - bff0ac0e e2e script: synthetic prompts sized by the measured 2.17 tokens a word (it divided by 1.35, so
    `--prompt-tokens 100000:160000` gave 164-254k-token prompts)
  - the commit after it: this STATUS note
- The design is in `DESIGN-kv-admission.md`; the user docs are in `docs/inference/cli.md`.
- gpubox1 (user densemax):
  - build dir `~/work/flavius/3b5/build-kv-admission`. Its `src/csrc` is byte-identical to the branch's `csrc` (all
    2,255 tracked files checked at this checkpoint). `build.sh` uses -j 8; set J=4 while SWE-Pro builds run. It
    exports PKG_CONFIG_PATH to `~/work/flavius/3b5/ffmpeg-dev-staged`, which holds the FFmpeg 6.1.1 dev files copied
    from gpubox2 (the runtime libraries are the system ones).
  - **Caveat on syncing sources**: `rsync -a` keeps the older cpubox3 mtimes, so ninja rebuilds nothing. After a
    sync, `touch` the changed files on gpubox1.
  - `engine-kv-admission`: f0ec6a45, no FFmpeg. The results up to the 11:05 checkpoint's "Results so far" came from
    this binary.
  - `engine-kv-admission-r2`: 4c119f44 plus FFmpeg media decoding (SHA256SUMS, BUILD.txt); identical to the build
    dir's current outputs. The engine code of HEAD is the same (250afbd8 and the last commit change scripts and notes
    only). Every result from tiny3 on came from this binary.
  - e2e dir `~/work/flavius/3b5/kv-admission`: `run_e2e.sh` ... `run_e2e5.sh` (all phases done), results in `out/`,
    one line per phase in `logs/job.log`. `run_e2e5.sh` = `run_e2e4.sh` plus the phases `vstress2` and `stress3`;
    `stress3.jsonl` is the stress3 request (stress2's synthetic-15, written with the 250afbd8 generator).
  - Copies on the shared filesystem: `/mnt/data/work/rl/kv-admission-2026-10-19/` (scripts including
    `ctest_supplement.sh` and `wake_timing.{sh,py}`, `stress3.jsonl`) and `.../results-gpubox1/{out,logs,ctest}`
    (all e2e results, engine logs and ctest logs as of 11:55; no file holds the API key).
  - The ctest logs: `~/work/flavius/3b5/build-kv-admission/ctest-gpu.log` (full run on GPU 0),
    `ctest-supplement.log` (reruns and opt-in tests), `ctest-cpu.log` (the earlier CPU-only run).

## Results (gpubox1 GPU 0, NVFP4 v5 artifact, FP8 KV, 262k context, --kv-capacity auto = 7,549 pages, unless noted)

### No preemption gives identical output (criterion 5)

- Reserve vs ondemand, 64 real requests run sequentially (BFCL live and multi-turn, EOG with prompts up to about
  150k tokens, synthetic), greedy, max_tokens 384, MTP 2: 64/64 byte-identical without prefix reuse, and 64/64 with
  prefix reuse.
- Same comparison without MTP and with top-2 logprobs: 64/64 identical.
- Vision: 24 image requests, greedy, top-2 logprobs, no MTP: reserve vs ondemand 24/24 identical.

### Forced preemption (criterion 1)

`SUROGATE_SERVE_KV_FORCE_PREEMPT=NxK`: every request is preempted and recomputed after N generated tokens, K times.

- 0 errors in every forced run; outputs stay coherent.
- MTP greedy vs reserve: 21/64 identical (20/64 with prefix reuse).
- Seeded T=0.8 at 48x3: 8/32 identical, both with and without MTP.

**Logprob-gap test** (greedy, no MTP, top-2 logprobs, forced 64x3 vs reserve):
- 30/64 identical, 34 differ.
- Every first divergence is at a token at or after the first preemption: earliest 64, median 126.
- Chosen-token logprob drift before the resume is exactly 0. After the resume it averages 0.019 nats (max 0.10).
- The base run's top1-top2 gap at the divergence point (nats): 0.0 x7, 0.125 x16, 0.25 x9, 0.375 x1, 0.5 x1. These
  are bf16 logit steps, so every divergence is a tie or near-tie.
- Forced vs forced: 64/64 identical, so the resume path is deterministic.

**Control: what batching alone does**, without preemption. The same 64 greedy requests, 16 at a time, vs the
sequential run: 7/64 identical, divergence from token 0, gaps up to 1.875 nats (8 above 0.5). Ordinary batch
composition perturbs outputs more than a recompute does. The divergence is the NVFP4 prefill path (W4A4) vs the
decode path, not a resume bug. With MTP there are no gap numbers, because MTP rounds report no logprobs.

**Vision, forced 64x2** (48 preemptions, every image re-encoded on resume): 24/24 OK, 0/24 identical, first
divergence always at token >= 64, drift before the resume exactly 0 and ~0.02 nats after; gaps 0.0 x7, 0.125 x8,
0.25 x7, 0.5 x1, 0.625 x1 (descriptions of coloured blocks are full of near-ties; 0.625 is inside the control's range).

### Real preemptions, stress and FCFS (criterion 2)

FCFS is checked from `SUROGATE_SERVE_KV_ADMIT_TRACE=1`: the first admission of every request (`preemptions 0`) must
come in executor request-id order, and the id is taken under the queue lock at submission. The id in `kv-admit:` and
`kv-admission: preempted` lines is the executor's, not the HTTP `[req N]` (it counts the warm-up request too).

| Run | Setup | Requests OK | Preemptions | Recomputed KV tokens | FCFS |
|---|---|---|---|---|---|
| smoke | forced 32x2 | 16/16 | 29 forced | | |
| tiny | 24k-token pool | 64/64 | 0 (no pressure) | | |
| tiny2 | 8k-token pool, 32 concurrent, max_tokens 2048 | 64/64 | 4 real | | (no trace yet) |
| tiny3 | 8k-token pool, 48 long answers at once, max_tokens 2048, greedy, MTP | 48/48 in 106 s | 49 real | 74,091 | ok: 49 first admissions in order, 98 admissions |
| stress | 40 synthetic prompts of 164-254k tokens (median 202k), one-sentence answers, no max_tokens, 16 concurrent | 40/40 in 654 s | 0 | 0 | (no trace) |
| stress2 | 24 synthetic prompts of 164-254k tokens (median 199k), long answers, no max_tokens, 16 concurrent | 24/24 in 815 s | 0 | 0 | ok: 25 first admissions |
| stress3 | 6 copies of one 164,216-token prompt, long answer, no max_tokens, 6 concurrent, pool 5,192 pages | 6/6 in 171 s | 3 real | 498,234 | ok: 7 first admissions, 10 admissions |
| vstress2 | 32 image requests, 16 concurrent, forced 48x3, MTP, --max-model-len 131072 | 32/32 in 12 s | 96 forced | 26,279 | |

No run had an OOM, deadlock, hang or lost request: every load counts each request it sent (ok + errors = requests),
and all errors are 0.

- **stress / stress2**: the labels "100-160k" in `logs/job.log` are the generator's targets. Only two such prompts
  fit the 7,549-page pool at a time, and the room they left was never used up by their two answers (stress2's were
  2,122-21,803 tokens, median 4,407; 23 stopped, 1 hit the 262,144 context: 247,263 + 14,882). KV in use peaked at
  99.2% without a preemption or a growth refusal.
- **stress3** was added for that reason (`run_e2e5.sh` phase `stress3`, comments in the script). The pool holds two
  of the prompts (2,570 pages each with the 256 headroom) plus the 1% admission watermark: 2 x 2,570 + 52 = 5,192
  pages = 332,288 tokens. Default headroom and watermark; `--no-prefix-reuse`, so identical prompts share nothing
  (`reused 0` in every admission); `--pending-timeout-ms 1800000` because the last pair waits for the first two.
  Each pair prefills one after the other, both decode until the 52 free pages run out, and the newer request is
  preempted with "a request that arrived earlier needs its pages":
  - victims (executor id: generated tokens) 3:1,819, 5:1,848, 7:1,922; each the newer of the two running requests,
    and 166,034-166,137 KV tokens dropped (prompt + generated - 1);
  - each resumed once the older one finished, alone, with prompt 166,035 / 166,064 / 166,138 = 164,216 + generated,
    and finished with stop_token (4,063 / 3,336 / 3,287 tokens in all); the three unpreempted ones gave 2,889 /
    4,546 / 2,379;
  - 7 growth refusals (each refused growth attempt counts; the policy then preempts).

### Throughput, ondemand vs reserve (criterion 3)

Greedy, so both modes produce the same tokens where the batch composition allows:

| Setting | Reserve | Ondemand | Gain |
|---|---|---|---|
| BFCL live+multi-turn, 192 req, 32 concurrent, default 262144 | 131.6 s, 300 tok/s, TTFT p50 20.0 s | 40.8 s, 985 tok/s, TTFT p50 3.5 s | 3.2x |
| Long agentic (EOG/BFCL-MT bodies), 48 req, 16 concurrent, default 262144 | 143.5 s, 295 tok/s, TTFT p50 39.3 s | 52.4 s, 768 tok/s, TTFT p50 0.94 s | 2.7x |
| Rerun setting 32768, 30 req, 3 concurrent | 45.0 s, 475 tok/s | 47.1 s, 505 tok/s | none expected: 3 reservations already fit |
| Rerun setting 32768, 48 req, 16 concurrent, run 1 (10:40) | 57.1 s, 788 tok/s, TTFT p50 5.7 s | 129.8 s, 533 tok/s, TTFT p50 0.85 s | loop outlier in ondemand |
| Same, run 2 (11:24) | 137.1 s, 504 tok/s, TTFT p50 4.4 s | 44.9 s, 829 tok/s, TTFT p50 0.80 s | loop outlier in reserve |

An earlier sampled run (T=1.0, so outputs differ between modes) showed the same picture: BFCL 96 req 58.2 s to
19.9 s; agentic 48 req 175 s to 52 s; rerun at 3 concurrent 54 s vs 69 s, the ondemand run producing 37% more
output tokens.

**The grerun16 outlier is settled**: one greedy EOG request falls into a repetition loop up to the 32,768 cap
(finish=length) and runs its last ~100 s alone. Run 1 had it under ondemand, run 2 under reserve (eog_1385, 32,768
tokens, 126.6 s), so it is a greedy degeneration that a batch composition happens to trigger, not a scheduler
effect. Like for like, ondemand is ahead in both pairings: without the loop 829 vs 788 tok/s and TTFT p50 0.80 vs
5.74 s; with it 533 vs 504 tok/s and 0.85 vs 4.39 s. (Run 1's per-request files were overwritten by run 2; its
summary lines are in `logs/job.log`.)

### Tests (criterion 4)

**Unit tests** (CPU, `CUDA_VISIBLE_DEVICES=`) pass: sinfer_kv_admission_test (policy, requeue, a scheduler
simulation with up to 235 preemptions, PagedKVPool accounting including shared-prefix forks),
sinfer_serve_options_test, request_log, admission_policy, openai_schema, tool_call_parser, output_parsers.

**Full ctest on gpubox1 GPU 0** (11:32-11:47; `CUDA_VISIBLE_DEVICES=0 ... ctest -j 1 --timeout 900
--output-on-failure` in the build dir, 906.9 s, `ctest-gpu.log`): 197 passed, 27 skipped, 6 failed out of 230. The
tests that failed in the CPU-only run for want of a GPU (lora, short_conv, the hyper connections, the fp8-block and
swiglu linears, ...) pass.

None of the 6 failures comes from this change. The only `sinfer_ops` objects that include a file the branch changes
are the six sampling / sampled-logprob / speculative-round launchers and wrappers: no linear, W8, MoE or attention
object, and none of the failing tests' own objects (`affected-objects.txt` in the build dir lists every compiled
object that includes a changed file, from ninja's dependency records).

| Test | Failure | Cause | Check |
|---|---|---|---|
| sinfer_sparse_moe_fp8_sm90_test, sinfer_sparse_moe_route_test, sinfer_w8_tail_test, sinfer_w8_pipelined_test | `libcudart.so.13: cannot open shared object file` | these link the CUDA runtime dynamically and ctest's environment has no CUDA lib dir (same in the CPU-only run) | with `LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64`: route, tail and pipelined pass; fp8_sm90 skips (an sm90 test on sm_120) |
| sinfer_w8_route_selection_test (host only) | expects the SIMT route for n = 2112 (not whole 128-row groups) | origin/main #310 (c3019cf1) made the generic MMA routes the default; the test still expects the earlier ones | passes with `SUROGATE_SERVE_W8_GENERIC_MMA=0` (restores the earlier routes); the same 6 failures with an origin/main-equivalent binary; every source and header of its two translation units is byte-identical to origin/main |
| sinfer_w8_precision_test (skipped without a GPU) | "Generic W8 reduction changed at batch width 1, 4, 8, 16, 17, 32, 33, 64" | the same #310 routes: below 65 columns the generic W8 linear no longer takes the batch-consistent kernel on sm_120 | passes with `SUROGATE_SERVE_W8_GENERIC_MMA=0`; the same failure with an origin/main-equivalent binary |

The origin/main-equivalent W8 binaries are those of `~/work/flavius/3b5/build-maxtokens` (another agent's tree, only
executed from /tmp; nothing written there): all 770 files under `csrc/src/serve/ops` and `csrc/src/testing/serve/ops`
there have origin/main's blob hashes, and none is newer than the binaries.

**Skipped (27)**, all for resources this machine or a one-GPU run lacks: an artifact, adapter or tokenizer named by an
environment variable (executor_failure_lock, completed_kv, burst_prefix, high_concurrency, dflash_admission,
model_wake_http, model_routing_http, shutdown_http, family_frontend, lora_lifecycle, lora_base_rounds,
target_lora_models, the qwen3_5 / qwen3_5_moe real-model, load-plan and DFlash tests, decisions_thinking_gemma4,
candidate_readout, dflash_readout, gemma_vision), or a second GPU (marlin_fused_parent, elastic_kv_device_gate,
cutlass_streamk_devices, multi_device).

**The opt-in tests that can run with what gpubox1 has** (`ctest-supplement.log`, script
`kv-admission/ctest_supplement.sh`). The executor-level ones need a small dense Qwen3.5 / Gemma4 E / Llama artifact
(completed_kv, executor_failure_lock), two engines on one card (burst_prefix), 257 lanes (high_concurrency), a
groupwise-int 35B artifact (qwen3_5_moe_real), DFlash or LoRA adapters; none of those fits here.
- sinfer_family_frontend_test with `SINFER_FAMILY_TOKENIZER_DIR` = the Surogate3.7 HF dir: passed (8.5 s). The branch
  adds `PreparedPrompt::with_continuation` to the frontend.
- sinfer_shutdown_http_test with the NVFP4 v5 artifact (ondemand by default): passed (14.1 s).
- sinfer_model_wake_http_test with the NVFP4 v5 artifact: failed at its recovery step. The request got 503
  "request expired while waiting for model wake" exactly at the test's `pending_timeout_ms = 300`, because waking this
  artifact (restoring 19.87 GiB) takes longer. (The test's `recovery:` line is lost: it is buffered stdout and the
  test aborts.) Through the CLI with the test's settings (`kv-admission/wake_timing.sh`, `logs/wake-timing.log`),
  sleep -> wake_up -> first request takes 0.391-0.419 s / 0.002 s on origin/main's engine (engine-toolarg-parity:
  584d183a, whose csrc is identical to 5004a4c5), 0.393-0.418 s / 0.001-0.002 s on the branch with ondemand and
  0.391-0.418 s / 0.002 s with reserve. The wake and deadline code (`model_scheduler.*`, `http_server.h`) is
  unchanged by the branch. The test was written for a model that wakes within 300 ms.

## What could change outputs or break a customer setup

See DESIGN "Output identity". In short:
- **The default is ondemand**, also in surogate-engine-cli: every deployment that takes this build gets on-demand
  admission unless it passes `--kv-admission reserve`.
- **Strict FCFS replaces backfill in ondemand mode**: a request at the head that does not fit holds everything
  behind it until running requests free pages (vLLM's FCFS does the same). Reserve mode keeps the backfill path.
  The admission timeout still counts from arrival; with the pool sized for two 164k requests (stress3) the last
  pair's TTFT was 130-142 s.
- **A preempted request is recomputed**: its remaining output can differ from an unpreempted run at near-ties, as
  any batch-composition change already does; a recompute of a 166k-token request costs ~12 s of prefill on a 5090.
- **Seeded sampling**: non-speculative seeded requests keep their RNG stream across a resume by design (only argued:
  8/32 identical under forced preemption, not measured token by token); a seeded MTP request's realization after a
  resume can differ (same keys and distribution).
- **Image requests keep their media until they complete**; it counts against `--media-live-mib`, and while
  on-demand image requests hold more than half of it a new image request gets its full reservation.
- **Not preemptable, so admitted with the full reservation as today**: prompt_logprobs, routing capture,
  target-only / GPU-prefix / candidate readouts. **DFlash and multi-GPU pipelines stay on reserve.**
- Not caused by this change: vision + MTP at 262,144 context does not fit a 32 GB card's minimum runtime
  reservation (8.27 GB + 1 GB headroom vs 8.54 GB after weights) in either mode; vstress2 ran at 131,072.

## Still open

Nothing required for the verdict.

Optional, not required for the verdict: a seeded-sampling gap test (the 8/32 seeded identity is argued, not measured
token by token).

## Next steps

- Coordinator: if the verdict is CLEAN, open the PR and squash-merge it like #323/#324 (owner's pre-approval);
  deploying anywhere is not part of that approval.
- To rerun anything: `ssh gpubox1; cd ~/work/flavius/3b5/kv-admission; (setsid nohup ./run_e2e5.sh PHASE... >
  logs/run-e2e5.out 2>&1 < /dev/null &)`, GPU 0 only, one engine at a time on port 8470; results in `out/`, one line
  per phase in `logs/job.log`. Phases overwrite their earlier output files.
