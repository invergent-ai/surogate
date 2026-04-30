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

## Step 1 — start the entire pipeline with one command

`surogate grpo` can spawn the rollout vLLM, the judge vLLM, and the trainer
together when you pass `--judge-infer` and `--judge-gpus`. This is the
production path — one process tree, single shutdown signal, watchdog watches
all four components for unexpected death.

```bash
RULER_JUDGE_API_KEY=EMPTY surogate grpo \
    --train       examples/ruler/train.yaml \
    --infer       examples/ruler/infer.yaml \
    --orch        examples/ruler/orch.yaml \
    --judge-infer examples/ruler/judge.yaml \
    --vllm-gpus   0 \
    --trainer-gpus 1 \
    --judge-gpus  2
```

The runner enforces:

- `--vllm-gpus`, `--trainer-gpus`, `--judge-gpus` are all pairwise disjoint.
- Each GPU count matches the corresponding inference config's `dp * tp` (or `train.gpus`).
- The judge port (in `judge.yaml`) differs from the rollout port (in `infer.yaml`).
- `orch.yaml` has `ruler.enabled: true` (otherwise the judge would never be called).

The `RULER_JUDGE_API_KEY` env var name is set in `orch.yaml`
(`ruler.judge.api_key_var`); the value `EMPTY` is fine for an unauthenticated
local vLLM. Use a real API key here if pointing the judge at a hosted endpoint.

### Alternative: judge in a separate terminal

If you'd rather manage the judge process yourself (separate logs, easier to
restart independently, or pointing at a hosted endpoint), omit the `--judge-*`
flags and start the judge as a free-standing `grpo-infer`:

```bash
# Terminal A — judge vLLM
CUDA_VISIBLE_DEVICES=2 RULER_JUDGE_API_KEY=EMPTY \
    surogate grpo-infer examples/ruler/judge.yaml

# Terminal B — rollout vLLM + trainer + orchestrator
RULER_JUDGE_API_KEY=EMPTY surogate grpo \
    --train examples/ruler/train.yaml \
    --infer examples/ruler/infer.yaml \
    --orch  examples/ruler/orch.yaml \
    --vllm-gpus 0 --trainer-gpus 1
```

Sanity-check the standalone judge from another shell:

```bash
curl -s http://localhost:8001/v1/models | python -m json.tool
# Expect: data[0].id == "Qwen/Qwen3-1.7B"
```

## Step 2 — watch for the RULER wiring banner

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

## Step 3 — observe per-group judge calls

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

## Step 4 — verify metrics

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

## Step 5 — confirm concurrent judging is happening

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

## Step 6 — Read the automatic baseline-vs-trained eval

The training run scored rollouts with the **judge** (RULER). To know whether
training actually improved the model, you need numbers on the env's
**native, ground-truth rubric** — `markdown-table-qa`'s answer-match verifier.
If you eval with the same judge you trained against, you measure judge fit,
not task performance.

The orchestrator handles this automatically. The `eval:` block in
`orch.yaml`:

```yaml
eval:
  env:
    - id: markdown-table-qa
  num_examples: 50
  rollouts_per_example: 4
  interval: 100              # don't run mid-training (max_steps=10 < 100)
  eval_base_model: true      # baseline before training
```

triggers two evaluations against the env's native rubric:

1. **Pre-training baseline** (`eval_base_model: true`) — the base model
   evaluated *before* any policy update happens, on the same 50 examples.
2. **End-of-training final eval** — the trained adapter evaluated on the
   same 50 examples after the loop ends.

Both share the same examples, the same sampling args, the same vLLM, and
the same env. RULER does **not** participate in either — `task_uses_group_scoring`
auto-detection only wraps train envs, not eval envs (see [grpo_orch.py:184-201](https://github.com/invergent-ai/surogate/blob/main/surogate/grpo/orchestrator/grpo_orch.py#L184-L201)).

### What to look for in the orchestrator log

Within the first ~10 seconds of `surogate grpo` starting, you'll see the
baseline eval:

```
INFO Running evals for checkpoint step 0
INFO Evaluating markdown-table-qa (num_examples=50, rollouts_per_example=4)
SUCCESS Evaluated markdown-table-qa in 4.12s (Avg@4=0.4400, Pass@1: 0.2200, Pass@2: 0.3600, Pass@4: 0.5400, ...
```

After the training loop completes, the final eval runs against the trained
model:

```
INFO Running final evals
SUCCESS Evaluated markdown-table-qa in 3.97s (Avg@4=0.5800, Pass@1: 0.3400, Pass@2: 0.4900, Pass@4: 0.7200, ...
```

The delta between the two `SUCCESS` lines is the answer to "did RULER
training improve the model?"

### What to look for in the metrics dump

When `dump_metrics: true` is set in `orch.yaml`, each eval pass appends
one row to `/tmp/grpo_metrics.jsonl` carrying `eval/<env_name>/*` keys
(the row is distinguished from training rows by the `eval/` namespace —
training rows don't have those keys).

```bash
python3 -c "
import json
for line in open('/tmp/grpo_metrics.jsonl'):
    d = json.loads(line)
    if any(k.startswith('eval/') for k in d):
        ckpt = d.get('progress/ckpt_step', '?')
        avg = d.get('eval/markdown-table-qa/avg@4')
        p1 = d.get('eval/markdown-table-qa/pass@1')
        p4 = d.get('eval/markdown-table-qa/pass@4')
        # pass@k may be absent if the rubric isn't binary
        bits = [f'Avg@4={avg:.4f}']
        if p1 is not None: bits.append(f'Pass@1={p1:.4f}')
        if p4 is not None: bits.append(f'Pass@4={p4:.4f}')
        print(f'ckpt_step={ckpt}: ' + ', '.join(bits))
"
```

You'll get two rows: one for `ckpt_step=0` (baseline) and one for the final
step (trained). Compute the delta directly.

If your env's reward is **non-binary** (continuous score in [0, 1] or
similar), `vf-eval` skips `pass@k` and you'll only see `avg@N`,
`completion_len/*`, `is_truncated/mean`, and `no_response/*`. Avg@N is
still the headline number to compare.

### Reading the result

A useful improvement on this smoke run looks like **+5 to +20 points on
`Pass@4`** between the baseline and the RULER-trained model. Things to
expect — none of which are bugs:

- **Flat or slightly worse Avg@4** with only 10 training steps. Convergence
  on Qwen3-0.6B + markdown-table-qa typically takes 100s of steps; 10 is a
  smoke test.
- **`Completion Length` drops** even when accuracy is flat — a common
  RULER signal: the judge prefers concise correct answers, so the policy
  learns to stop generating earlier.
- **`Pass@4` improves more than `Pass@1`** — relative ranking pressure
  encourages diversity in the policy's output distribution.
- **Worse numbers** can mean the judge and the env's rubric disagree on
  what "good" looks like (RULER thinks formatting matters more, the
  rubric thinks exact-match matters more). Inspect a few of the judge's
  per-trajectory explanations from the training log (`debug: true` mode)
  to see what it was rewarding.

### Long-form baseline

For a real comparison (not a smoke), bump `train.yaml` `max_steps: 200`
and `orch.yaml` `max_steps: 200`, and increase `eval.num_examples: 500`.
The smoke configs in this directory are tuned for a ~12 s training run
that exercises every code path; they are not tuned for a model that
would actually move metrics.

### Tracking learning curves with interval evals

The smoke config gives you exactly two eval points (baseline + final).
For real training runs, drop `eval.interval` to a fraction of `max_steps`
to also get mid-training data points — useful for watching the policy
improve, catching plateaus early, and spotting RULER-vs-rubric drift
before the run ends.

A reasonable production config (assume `max_steps: 200` and ~17 s per eval pass):

```yaml
eval:
  env:
    - id: markdown-table-qa
  num_examples: 100        # smaller per-pass for faster intervals
  rollouts_per_example: 4
  interval: 25             # ~8 mid-training evals at ~10% wall-clock overhead
  eval_base_model: true
  # cancel_inflight_rollouts_on_eval: true   # only if eval saturates the rollout vLLM
```

Cost guidance:

| Cadence (`interval`)              | Eval passes | Wall-clock added (Qwen3-0.6B, 100 ex × 4 ro) |
| --------------------------------- | ----------- | -------------------------------------------- |
| `100` (only base + final)         | 2           | ~35 s                                        |
| `50` on `max_steps: 200`          | 5           | ~85 s                                        |
| `25` on `max_steps: 200`          | 9           | ~150 s                                       |
| `10` on `max_steps: 200`          | 21          | ~360 s (heavy)                               |

Eval is **blocking** — weight broadcasts pause for the duration ([grpo_orch.py:504](https://github.com/invergent-ai/surogate/blob/main/surogate/grpo/orchestrator/grpo_orch.py#L504)) — so each pass adds directly to total wall-clock. Rule of thumb: pick `interval ≈ max_steps / 10` to land at ~10% overhead.

#### Plotting the curve

With `dump_metrics: true`, every eval pass appends a row to
`/tmp/grpo_metrics.jsonl`. Plot the curve in two lines of pandas:

```python
import json, pandas as pd
rows = [json.loads(line) for line in open("/tmp/grpo_metrics.jsonl")]
eval_rows = [r for r in rows if any(k.startswith("eval/") for k in r)]
df = pd.DataFrame(eval_rows).set_index("progress/ckpt_step")
df[["eval/markdown-table-qa/avg@4",
    "eval/markdown-table-qa/pass@1",
    "eval/markdown-table-qa/pass@4"]].plot(marker="o")
```

To overlay the training-side RULER score on the same axis (training reward
on the same `step` axis as eval), join on `step`:

```python
all_rows = pd.DataFrame(rows)
train = all_rows[all_rows["ruler/score_mean"].notna()].set_index("step")["ruler/score_mean"]
evalc = pd.DataFrame(eval_rows).set_index("step")["eval/markdown-table-qa/avg@4"]
pd.concat({"train: ruler/score_mean": train, "eval: avg@4": evalc}, axis=1).plot(marker="o")
```

If the two curves diverge — RULER score climbing while eval avg@4 flat or
falling — that's the **judge-rubric drift** signal: the judge is rewarding
something the rubric doesn't care about. Either tighten `ruler.rubric` to
align with the env's verifier, or switch to `ruler.mode: add` so RULER
augments rather than replaces the rubric.

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

**No `eval/<env>/avg@4` rows appear in `/tmp/grpo_metrics.jsonl`** — the
`eval:` block is missing from `orch.yaml`, or the eval env didn't load.
Confirm `Running evals for checkpoint step 0` appears in the orchestrator
log within ~10 s of startup. If not, re-check `orch.yaml` has an `eval:`
block with `eval_base_model: true`.

**Baseline and final eval produce identical numbers** — likely the
training adapter wasn't applied to the eval rollouts. Confirm in the
log that you see two distinct `Loaded new LoRA adapter` lines (one
during training, and the eval reads the latest). Also check
`progress/ckpt_step` differs between the two `eval/*` rows in the
metrics dump.
