# RULER end-to-end smoke test

A minimal end-to-end run that exercises every layer of the RULER (LLM-as-judge
group reward) implementation: config validation → orchestrator wiring →
deferred group scoring → judge HTTP round-trip → metric aggregation → GRPO
optimizer step.

The point of this run is to **verify the plumbing**, not to demonstrate that
RULER beats hand-crafted rewards on a real task. 10 orchestrator steps on a
small env is enough to see every code path execute and to catch regressions.

## What's in this directory

| File | Purpose |
|---|---|
| [`infer.yaml`](infer.yaml) | vLLM serving the *student* (Qwen3-0.6B) the trainer is updating. LoRA enabled. |
| [`judge.yaml`](judge.yaml) | vLLM serving the *judge* (Qwen3-1.7B). Frozen weights, no LoRA, must support OpenAI structured outputs. |
| [`train.yaml`](train.yaml) | GRPO trainer config — LoRA + AdamW + bf16. |
| [`orch.yaml`](orch.yaml) | Orchestrator config with `ruler.enabled: true`. |

## Prerequisites

1. **4-GPU node** (RTX 5090 32GB or equivalent). The layout is:
   - GPU 0 — rollout vLLM (student)
   - GPU 1 — trainer
   - GPU 2 — judge vLLM
   - GPU 3 — spare
2. **Environment installed**: `markdown-table-qa` is referenced in `orch.yaml`. If
   it isn't pip-installed yet:
   ```bash
   uv run surogate vf-init markdown-table-qa
   ```
3. **HF model access**: `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B` must be
   downloadable. Run `huggingface-cli login` if either is gated for you.
4. **Disk**: writes to `./outputs/ruler/`. Allow ~2 GB.

## Step 1 — start the judge vLLM (Terminal A)

The judge is independent infrastructure — it does not participate in
`surogate grpo`'s rollout-vLLM-plus-trainer launch sequence, so you must start
it first and leave it running.

```bash
CUDA_VISIBLE_DEVICES=2 RULER_JUDGE_API_KEY=EMPTY \
    surogate grpo-infer examples/ruler/judge.yaml
```

Wait for `Application startup complete.` (≈30 s after the model finishes
downloading on first run). Sanity-check from another shell:

```bash
curl -s http://localhost:8001/v1/models | python -m json.tool
# Expect: data[0].id == "Qwen/Qwen3-1.7B"
```

If you see the model entry, the judge is ready.

## Step 2 — start the GRPO pipeline (Terminal B)

This single command launches the rollout vLLM (GPU 0), the trainer (GPU 1), and
the orchestrator process — all wired up via filesystem weight broadcast.

```bash
RULER_JUDGE_API_KEY=EMPTY surogate grpo \
    --train examples/ruler/train.yaml \
    --infer examples/ruler/infer.yaml \
    --orch examples/ruler/orch.yaml \
    --vllm-gpus 0 \
    --trainer-gpus 1
```

The `RULER_JUDGE_API_KEY` env var name is set in `orch.yaml`
(`ruler.judge.api_key_var`); the value `EMPTY` is fine for an unauthenticated
local vLLM. Use a real API key here if pointing the judge at a hosted endpoint.

## Step 3 — watch for the RULER wiring banner

Within the first few seconds of the orchestrator coming up, you should see:

```
Initializing RULER (judge_model=Qwen/Qwen3-1.7B, mode=replace, weight=1.0, max_concurrent=8, endpoints=['http://localhost:8001/v1'])
Initialized RULER judge pool with 1 endpoint(s) (max_concurrent=8, timeout=90.0s)
  attached RULER to env 'markdown-table-qa' (mode=replace)
Deferred group scoring enabled for training tasks: markdown-table-qa.
```

The "Deferred group scoring enabled" line is the key confirmation: it means
`task_uses_group_scoring` detected RULER's group reward function and the
orchestrator will route rubric scoring through the deferred path that supports
concurrent judging.

## Step 4 — observe per-group judge calls

With `debug: true` in `orch.yaml`, each completed group emits:

```
RULER judged group of 4 (sent=4, calls=1, latency=420ms, in=2150 out=180): scores=['0.850', '0.620', '0.400', '0.150']
  RULER traj 1: 0.850 — Cleanly extracts the answer and matches the expected format.
  RULER traj 2: 0.620 — Correct value but verbose preamble.
  RULER traj 3: 0.400 — Partial answer, missed second column.
  RULER traj 4: 0.150 — Off-topic response.
```

`sent=4` confirms all rollouts were unique (no all-identical fallback).
`calls=1` confirms one judge HTTP call per group.

## Step 5 — verify metrics

Per-step metrics dump to `/tmp/grpo_metrics.jsonl` (`dump_metrics: true`).
After the run finishes, check the last entry:

```bash
tail -1 /tmp/grpo_metrics.jsonl | python -m json.tool | grep -E "ruler|reward/mean"
```

Expected keys (values are illustrative):

| Key | Meaning |
|---|---|
| `metrics/ruler_score` | mean of per-rollout judge scores |
| `metrics/ruler_judge_calls` | per-rollout share of a judge call (0.25 for groups of 4) |
| `metrics/ruler_input_tokens`, `ruler_output_tokens` | per-rollout share of judge tokens |
| `metrics/ruler_judge_latency_ms` | per-rollout share of judge latency |
| `metrics/ruler_judge_cost_usd` | per-rollout cost (zero unless `ruler.cost.*` is set) |
| `metrics/ruler_judge_failed` | 0.0 in a healthy run |
| `ruler/total_judge_calls` | step total — should equal `batch_size / rollouts_per_example` = **4** |
| `ruler/total_input_tokens`, `total_output_tokens`, `total_judge_latency_ms` | step totals (sum of per-rollout shares) |
| `ruler/score_mean`, `score_std` | distribution of RULER scores this step |
| `ruler/judge_failure_rate` | 0.0 in a healthy run |
| `reward/mean` | should track `ruler/score_mean` exactly under `mode: replace` |

The most important check: `ruler/total_judge_calls == 4` per step. If it's
higher, retries are firing (transient errors or judge parse failures); inspect
the orchestrator log for `RULER judge … error`.

## Step 6 — confirm concurrent judging is happening

Compare `ruler/total_judge_latency_ms` (sum across all 4 group calls) to
`time/step` (the orchestrator step duration). If the judge is the bottleneck
and judging is parallel, total latency should be **roughly 4×** step time
(four calls overlapping in real time). If they're roughly equal, judging is
running serially — investigate the scheduler change.

```python
# quick check
import json, pandas as pd
df = pd.read_json("/tmp/grpo_metrics.jsonl", lines=True)
df[["step", "time/step", "ruler/total_judge_latency_ms"]]
```

## Knobs to flip on subsequent runs

| Change | What it tests |
|---|---|
| `ruler.mode: add` in `orch.yaml` | Composing RULER with the env's existing rubric via `vf.RubricGroup`. Both reward sources should appear in `metrics/`. |
| `ruler.mode: metric` | Observability-only mode: `metrics/ruler_*` populated, `reward/mean` driven entirely by the env's rubric. |
| `ruler.max_concurrent_judges: 1` | Verifies the global semaphore serializes calls — `time/step` should rise; per-call latency unchanged. |
| `ruler.judge.base_url: ["http://localhost:9999/v1"]` (unreachable) | Verifies `swallow_exceptions: true` produces `ruler_judge_failed: 1.0` and zero rewards instead of a crash. |
| `ruler.rubric: "<custom text>"` | Custom grading rubric. Score distributions should reflect the new criteria. |
| `rollouts_per_example: 8` (and bump `batch_size` to a multiple) | Higher group size — judge prompts grow; verify the judge model's `max_model_len` (16384 in `judge.yaml`) is sufficient. |

## Troubleshooting

**`ValueError: ruler.enabled requires verification.enabled=True`** —
`verification.enabled: true` is required in `orch.yaml`; RULER plugs into the
deferred-group-scoring path which is gated on it.

**`ValueError: ruler.enabled requires rollouts_per_example >= 2`** —
RULER is fundamentally relative; a single rollout has no peers to compare
against. Default `rollouts_per_example` in `orch.yaml` is 4.

**Judge errors with `"json_schema is not supported"`** — your vLLM is too old
for OpenAI structured outputs. Upgrade to ≥0.5.5 or pass
`--guided-decoding-backend xgrammar` when starting the judge manually.

**Judge OOM** — Qwen3-1.7B at `max_model_len: 16384` and `gpu_memory_utilization:
0.85` fits comfortably on a 32 GB card. On a 24 GB card, drop
`max_model_len` to 8192 and `gpu_memory_utilization` to 0.7.

**`ruler/judge_failure_rate > 0`** — judge is producing malformed JSON or
timing out. Increase `ruler.request_timeout` and `ruler.max_retries_on_parse_error`,
or use a stronger judge model.

**`LengthFinishReasonError: Could not parse response content as the length limit was reached`** —
the judge consumed `max_completion_tokens` before emitting the JSON. The rubric
auto-retries with a doubled budget (capped at 16K), but if you see this in
the warning log it means the first attempt was wasted. Two ways to avoid it:
1. Bump `ruler.sampling.max_completion_tokens` (4096 is a safe starting point).
2. Disable reasoning/thinking on the judge. For Qwen3 specifically, add
   `ruler.extra_body: { chat_template_kwargs: { enable_thinking: false } }` —
   this is set in `orch.yaml` already.

**Connection refused on :8001** — the judge isn't up yet. Wait for the
`Application startup complete.` line in Terminal A before launching Terminal B.

**Trainer hangs at "Waiting for inference pool to be ready"** — the rollout
vLLM (port 8007) didn't come up. Check `outputs/ruler/logs/` for vLLM errors,
typically OOM or HF download failure.

**Trainer keeps logging `Run run_default: No orchestrator config found at
…/outputs/ruler/run_default/control/orch.yaml`** — the orchestrator's
`output_dir` is not under the trainer's `output_dir`. The orchestrator writes
its control file to `<orch.output_dir>/control/orch.yaml`; the trainer scans
`<train.output_dir>/run_*/control/orch.yaml`. They must match: set
`output_dir: ./outputs/ruler/run_default` in `orch.yaml` (already done in this
example).
