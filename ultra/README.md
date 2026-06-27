# Ultra — a Fugu-Ultra generative workflow Conductor

A **parallel track** to the Director router. Where Director picks *one worker per step*,
Ultra trains a **Conductor** (Qwen2.5-7B, GRPO) that emits an entire natural-language
**workflow** — subtasks, worker IDs, and access lists — and is rewarded by executed
workflow correctness (0 malformed · 0.5 valid+incorrect · 1.0 valid+correct).

**Standalone:** Ultra vendors the worker-pool + grading stack (`ultra/workers`,
`ultra/grading`) so it never imports the production `director` package.

## What's here (foundation + one vertical slice)

```
ultra/
  schemas.py     TaskSpec v2, Workflow, SourceManifest, RolloutRecord  (the contract)
  policy.py      source-policy classes + per-source policy table        (ultra-data2 §3)
  registry.py    TaskRegistry: manifest + policy↔split + dedup gates     (§1, §11)
  splits.py      contamination-group / prompt-hash dedup keys
  workers/       vendored OpenRouter pool, cache, budget, FakeProvider
  grading/       vendored judge-free verifiers (math_equal, mc_letter, code_exec, …)
  sources/       SourceAdapter contract + ExistingBankAdapter
  harness/       harness router + DirectQAHarness
  rollout.py     direct_rollout: task → worker → grade → RolloutRecord
```

The vertical slice proves the data path end to end: the router's 2,253-task bank →
`TaskSpec v2` → worker call → grade → `RolloutRecord`.

## Next

- Workflow validator + multi-step DAG executor (workspace lineage for repo harnesses)
- Step-zero headroom test (scaffolds A–F) — the go/no-go gate before any GRPO
- More source adapters (SWE-smith, LiveCodeBench, Math, MMLU-Pro, τ-bench, long-context)
- OpenCode server/SDK harness; the Conductor model + GRPO trainer

## Dev

```bash
cd ultra
uv venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest      # offline; -m network/gpu gated
```
