# On-demand KV admission with preemption (`--kv-admission ondemand`)

Branch `serve-kv-admission` (from main 5004a4c5). Status: design, 2026-10-19.

## Problem

`plan_request_base` entitles every request `prompt + effective_output - 1` pages at admission
(`text_kv_page_entitlement`, MTP head `+ draft_window - 1`). With `--default-max-tokens 262144` and
no `max_tokens` from the client, that is the whole 262k context per request, so a 483k-token
RTX 5090 replica runs one request at a time and the queue times out (BFCL v4, 2026-10-19).

The pool already separates *entitlement* (pages reserved for an allocation, the admission
currency) from *mapping* (pages materialized as a sequence grows). Only the entitlement policy
has to change.

## Policy

`--kv-admission reserve|ondemand` (default `ondemand`), `--kv-admission-headroom N` (tokens,
default 256), `--kv-admission-watermark F` (fraction of KV pages, default 0.01).

1. **Admission** (ondemand): a request is entitled `max(prompt + headroom, prefill graph reach)`
   tokens (MTP head: `+ draft_window - 1`), capped at its reserve-mode entitlement, which is kept
   as its *limit*. It is admitted when that fits and, if anything else is running, the pool keeps
   `watermark` pages free afterwards. The upfront refusal of a request whose *full* reservation
   (prompt + max_tokens) exceeds the pool stays exactly as today (`request reservation exceeds
   Engine shared KV capacity`), so any admitted request can always finish alone: an oversize
   request fails cleanly before it starts, never mid-stream.
2. **Strict FCFS**: in ondemand mode a head that does not fit waits; nothing behind it is
   admitted (the backfill/protection path stays for reserve mode). The admission timeout counts
   from arrival as before.
3. **Growth**: at every round boundary, before the decode membership is built, each decode-ready
   lane's entitlement is grown to cover the most one unit can write
   (`frontier + max(burst 8, draft_window + 1)`; MTP head `+ draft_window`), never past its limit.
   Growth is host accounting only (`set_page_entitlement` through `resize_paged_kv_bundle`, both
   pools atomically, elastic gate included). Lanes are grown oldest arrival first.
4. **Pages short**: first give back idle KV (retained prefix-cache lanes oldest-released first,
   shared prefixes, as admission eviction does), then **preempt** the running request that
   arrived last (highest request id; it may be the lane that needs the page), repeat. Arrival
   rank is kept across resumes, so the oldest running request is never preempted while a newer
   one runs (starvation guard). A lane that is alone and still cannot grow can only be blocked by
   pages no lane owns (GPU prefix images) or the overcommit device gate: it waits while the gate
   reports pressure, else fails with the existing 429 `insufficient KV capacity while GPU
   prefixes are retained`.
5. **Preemption = recompute**: the victim's lane is aborted (KV pages, MTP pages, recurrent GDN
   state, deferred folds all dropped: `clear_lane`), its slot freed, and the request is put back
   into the waiting queue in arrival order, i.e. at the front (all waiting requests are newer
   under strict FCFS), with no admission deadline (it was admitted once). Its prompt becomes
   `original prompt + generated tokens` (`PreparedPrompt::with_suffix` positions; original prompt
   identity kept, so prefix caching and next-turn retention still work). On re-admission it goes
   through the normal admission/prefill path (chunked, mixed rounds, MTP prepare, shared-prefix /
   retained-lane reuse): the recurrent state is rebuilt from position 0 (or from a matching
   cached prefix), never reused stale. The prefill's sampled token is the request's next
   generated token. No swap in v1 (the host archive is 512 MB; one 100k-token lane is ~1 GB).
6. **Request state carried across a resume**: executor-side output session, budget, generated
   tokens, logprobs, first-token time, begin summary (kept, not re-planned); program-side via a
   `KvResumeState` in `ResolvedExecutionOptions`: the penalty occurrence counts (device row copied
   out at preemption, restored after `install_sampling`), completion scores (top_logprobs),
   `prompt_tokens` = original prompt (min_tokens barrier, score positions), constraint frontier =
   original prompt (grammar state replays the generated tokens), timings and speculative stats
   (merged at completion).
7. **RNG**: keys are `(seed, position, purpose)`. The resumed prefill's sample is flagged
   (`SamplingConfig::continue_decode_key`) to draw with the key the decode step it replaces
   would have used: `(seed, position - 1, Decode)`. Non-speculative seeded sampling therefore
   keeps its exact stream. With MTP and temperature > 0 the draws already depend on which drafts a
   verify round checked; after a resume they keep their keys and distribution but can differ in
   realization. Greedy is unaffected by keys.
8. **Not preemptable -> admitted with the full reservation (reserve semantics)**: prompt_logprobs,
   routing capture (SUROGATE_SERVE_CAPTURE_ROUTING), target-only / GPU-prefix / candidate
   readouts (one token anyway). Engine-wide reserve: DFlash backend, multi-GPU pipelines.
   Image requests are preemptable: the executor keeps a clone of the prepared prompt, whose media
   payloads then stay alive (and count against the frontend's `--media-live-mib` budget) until
   the request completes. While on-demand image requests keep more than half of that budget, a
   new image request is planned with its full reservation instead (`reserve_kv`).

## Output identity

- No preemption: the admission policy changes no arithmetic; for the same batch composition the
  outputs are bit-identical to reserve mode. (Across modes, concurrency differs, and this engine's
  NVFP4/MTP kernels are not batch-invariant, so concurrent runs may differ at near-ties exactly as
  they do today between any two load patterns.)
- After a preemption the KV is recomputed by prefill instead of decode. Prefill and decode
  kernels are not bitwise equal on NVFP4/FP8-KV/MTP (same as vLLM's recompute), so greedy output
  is identical unless a recomputed logit flips a near-tie. Measured, not assumed: the e2e test
  forces preemptions (`SUROGATE_SERVE_KV_FORCE_PREEMPT`) on a single-request batch and reports
  how many outputs are byte-identical and where the first divergence is.

## Metrics / logs

RuntimeStats + `/metrics` + `/kv_stats`: `kv_preemptions_total`, `kv_recomputed_tokens_total`
(tokens dropped at preemption), `kv_preempted_waiting`, `kv_growth_refusals_total`; throughput
line adds `kv_entitled=..% kv_in_use=..% preempted=N recomputed=M`; one stderr line per
preemption and per resume (`kv-admission: ...`); startup line with the mode.

## Tests

- Unit (CPU only): pool growth / entitlement / borrowed-prefix accounting through preemption;
  growth+preemption loop (order, self-preemption, idle eviction first, alone/oversize);
  requeue position; a scheduler simulation of many 100k+ prompts with no max_tokens (no
  deadlock, FCFS, every request completes, oldest never preempted by a newer one).
- E2E (GPU, NVFP4 artifact, FP8 KV, MTP 2): greedy identity reserve vs ondemand (sequential),
  forced preemption identity, tiny `--kv-capacity` with concurrent requests, stress with many
  100k+ prompts and no max_tokens; throughput reserve vs ondemand (BFCL-like short requests at
  default max tokens 262144, long agentic prompts).
