# `ruler-task`

Generic verifiers env for **RULER (LLM-as-judge) training**. Owns dataset
loading and prompt construction; ships an empty rubric so RULER —
configured orchestrator-side — provides the reward signal.

Use this env when:

- You don't have (and don't want to write) a ground-truth rubric for your task.
- Your data is in HF Hub, a JSONL, a CSV, or a Parquet file.
- You want to drive training entirely from configuration, with no Python.

## Install

```bash
uv run surogate vf-init ruler-task    # registers from `environments/`
# or, in another project that depends on this package:
uv add ./environments/ruler-task
```

## Configure in `orch.yaml`

The env's `args` block mirrors the `load_environment` kwargs. Two end-to-end shapes:

### Shape A — HF Hub dataset

```yaml
env:
  - id: ruler-task
    args:
      task_name: gsm8k
      dataset: openai/gsm8k
      hf_config: main
      dataset_split: train
      eval_split: test
      user_template: "{question}"
      system_prompt: |
        Solve the math problem step by step. End with the final numeric
        answer on its own line, prefixed with ####.
      max_examples: 1000
      max_eval_examples: 200

ruler:
  enabled: true
  mode: replace
  judge_model: "openai/gpt-4o-mini"
  judge:
    base_url: ["http://localhost:8001/v1"]
    api_key_var: RULER_JUDGE_API_KEY
```

### Shape B — local JSONL

For data shaped `{"input": "<the user's prompt>"}` per line:

```yaml
env:
  - id: ruler-task
    args:
      task_name: my-custom-task
      dataset: ./data/train.jsonl
      eval_split: train             # local files always have one "train" split
      system_prompt_path: ./data/system_prompt.txt
      # user_template defaults to "{input}", which matches the JSONL schema
```

For multi-column composition (CSV with `instruction` + `context`):

```yaml
env:
  - id: ruler-task
    args:
      task_name: rag-qa
      dataset: ./data/rag_train.csv
      user_template: |
        Context:
        {context}

        Question: {instruction}
      system_prompt: "Answer using only the context provided."
```

## Argument reference

| Arg                   | Type            | Required          | Default       | Notes                                                                  |
| --------------------- | --------------- | ----------------- | ------------- | ---------------------------------------------------------------------- |
| `task_name`           | `str`           | yes               | —             | Used as the `task` column; must match the env name in `orch.yaml`.     |
| `dataset`             | `str`           | yes               | —             | HF Hub id (e.g. `openai/gsm8k`) or local file path.                    |
| `system_prompt`       | `str`           | exactly one       | —             | Inline system prompt text.                                             |
| `system_prompt_path`  | `str`           | exactly one       | —             | Path to a UTF-8 text file containing the system prompt.                |
| `dataset_split`       | `str`           | no                | `"train"`     | Split for training data (HF) or the "train" pseudo-split (local).      |
| `eval_split`          | `str \| None`   | no                | `None`        | If set, also wires `get_eval_dataset()` from the same source.          |
| `hf_config`           | `str \| None`   | no                | `None`        | HF dataset config name (e.g. `"main"` for GSM8K).                      |
| `user_template`       | `str`           | no                | `"{input}"`   | `str.format`-style template against each row's columns.                |
| `max_examples`        | `int \| None`   | no                | `None`        | Subsample to N rows (deterministic given `seed`).                      |
| `max_eval_examples`   | `int \| None`   | no                | `None`        | Same, for the eval split.                                              |
| `seed`                | `int`           | no                | `42`          | Determinism for shuffle/subsample.                                     |
| `shuffle`             | `bool`          | no                | `False`       | Shuffle the dataset before subsampling.                                |

## Detection: HF Hub id vs. local path

Whichever the user passes as `dataset`:

- If `Path(dataset).exists()` → loaded as a local file by extension (`.jsonl`, `.csv`, `.parquet`).
- Otherwise → loaded as an HF Hub id via `datasets.load_dataset`.

Same convention as HF's own loader. Unambiguous in practice; if a user happens to have a directory named `openai/gsm8k` on disk and meant the HF dataset, rename or move the directory.

## What this env does *not* provide

- **A rubric.** RULER replaces it. If you want native ground-truth scoring as well, run RULER under `mode: add` so RULER + your-other-rubric compose. (Note: composing requires a *different* env that has a real rubric — this env is intentionally rubric-less.)
- **Multi-turn or tool-use.** Single-turn only via `vf.SingleTurnEnv`. For multi-turn, write a custom env or extend this one.
- **Per-row tools.** No `tool_defs` plumbing. Add `info["tool_defs"]` per row in your dataset and use a custom env if you need this.

## Programmatic use (testing, sweeps)

```python
import verifiers as vf

env = vf.load_environment(
    "ruler-task",
    task_name="gsm8k",
    dataset="openai/gsm8k",
    hf_config="main",
    dataset_split="train",
    user_template="{question}",
    system_prompt="Solve step by step...",
    max_examples=100,
)

print(env.dataset.column_names)       # ['example_id', 'task', 'prompt']
print(len(env.dataset))               # 100
print(env.dataset[0]["prompt"][1])    # {'role': 'user', 'content': '...'}
```

## Failure modes

- **Missing required arg** → `ValueError: Missing required arg 'X' for ruler-task`
- **Both `system_prompt` and `system_prompt_path` set** → `ValueError: Provide exactly one of system_prompt or system_prompt_path`
- **`user_template` references a column not in the dataset** → `ValueError: user_template references columns missing from the train split: [...]`
- **Empty dataset / split** → `ValueError: ruler-task 'train' split is empty after subsetting`
- **Unsupported local extension** → `ValueError: Unsupported file extension '.xyz'; use one of ['.csv', '.json', '.jsonl', '.parquet']`
- **Unknown args** → `ValueError: Unrecognized args for ruler-task: [...]` (forward-compat: misspellings fail loudly)
