"""Generic verifiers environment for RULER (LLM-as-judge) training.

This env owns *only* dataset + prompt construction. The rubric is intentionally
empty — RULER, attached orchestrator-side via ``orch.yaml`` ``ruler.enabled:
true``, supplies the reward signal at training time. That separation makes
this env reusable for any task with no ground-truth rubric: bring a dataset of
inputs, write a system prompt, configure RULER, and train.

Customization happens entirely through the args dict in
``GRPOEnvConfig.args`` (in ``orch.yaml``) — see ``load_environment`` below
for the full surface. Designed to be driven programmatically by external
tooling (CI, sweep launchers, etc.) without code changes.

Note on the rubric
==================
``vf.SingleTurnEnv`` automatically attaches a built-in ``num_turns`` reward
function with **weight 0.0**. It contributes nothing to the training reward —
RULER under ``mode: replace`` overrides the rubric wholesale, and under
``mode: add``/``metric`` the zero-weighted ``num_turns`` only surfaces as an
observability metric (``metrics/num_turns``). No user-visible reward leaks in
through it.

Required args
=============
* ``task_name`` (str): the task identifier; becomes the ``task`` column on
  every row and must match the env name in ``orch.yaml``.
* Exactly one of ``system_prompt`` (inline str) or ``system_prompt_path``
  (path to a file containing the prompt).
* ``dataset`` (str): the data source. Either an HF Hub id (e.g.
  ``"openai/gsm8k"``) or a path to a local ``.jsonl`` / ``.csv`` /
  ``.parquet`` file. HF Hub is detected by absence of a path that exists
  on disk (same convention as ``datasets.load_dataset``).

Optional args
=============
* ``hf_config`` (str): HF dataset config name (e.g. ``"main"`` for GSM8K).
* ``dataset_split`` (str, default ``"train"``): split for training data.
* ``eval_split`` (str | None): when set, also wires ``get_eval_dataset()``
  using this split from the same source.
* ``user_template`` (str, default ``"{input}"``): str.format-style template
  rendered against each dataset row to build the user message content.
  Use ``"{question}"`` for GSM8K, ``"Context: {context}\\n\\nQ: {q}"``
  for multi-column composition, etc.
* ``max_examples`` (int | None): subsample to at most N rows. Sampling
  is deterministic given ``seed``.
* ``max_eval_examples`` (int | None): same, for the eval split.
* ``seed`` (int, default 42): determinism for shuffle/subsample.
* ``shuffle`` (bool, default False): shuffle the dataset before sub-sampling.

Surogate-injected args (accepted for compatibility, not used here)
==================================================================
* ``score_rollouts`` (bool): set by the orchestrator. Has no effect — this
  env always returns an empty rubric so RULER can replace it.
* ``max_seq_len`` (int): set by the orchestrator. Reserved for future use.
"""

from __future__ import annotations

import logging
import string
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def _normalize_max_examples(value: int | None) -> int | None:
    """Normalize the codebase's `-1`-means-"all" convention.

    Both ``None`` and ``-1`` (and any other non-positive value) mean
    "use the full dataset". Positive values cap the row count.
    """
    if value is None or value <= 0:
        return None
    return int(value)


_LOCAL_LOADERS = {
    ".jsonl": "json",
    ".json": "json",
    ".csv": "csv",
    ".parquet": "parquet",
}


def load_environment(**kwargs: Any) -> vf.Environment:
    """Build a ``vf.SingleTurnEnv`` from a dataset + system prompt + a row→prompt template.

    The supplied rubric is an empty ``vf.Rubric()``. ``vf.SingleTurnEnv`` wraps
    it in a ``vf.RubricGroup`` and attaches a single weight-0.0 ``num_turns``
    metric — the env contributes zero reward signal, which is the whole point.
    Pair with ``orch.yaml`` ``ruler.enabled: true`` (and typically
    ``ruler.mode: replace``) to drive training reward via an LLM judge.
    """
    task_name: str = _require(kwargs, "task_name", str, allow_empty=False)
    dataset_id: str = _require(kwargs, "dataset", str, allow_empty=False)

    system_prompt = _resolve_system_prompt(
        inline=kwargs.pop("system_prompt", None),
        path=kwargs.pop("system_prompt_path", None),
    )

    user_template: str = kwargs.pop("user_template", "{input}")
    if not isinstance(user_template, str) or not user_template.strip():
        raise ValueError(f"user_template must be a non-empty string (got {user_template!r})")

    hf_config: str | None = kwargs.pop("hf_config", None)
    dataset_split: str = kwargs.pop("dataset_split", "train")
    eval_split: str | None = kwargs.pop("eval_split", None)
    # Both `None` and `-1` (the codebase convention; see GRPOEvalConfig.num_examples)
    # mean "use all rows". Any positive int caps the row count.
    max_examples: int | None = _normalize_max_examples(kwargs.pop("max_examples", None))
    max_eval_examples: int | None = _normalize_max_examples(kwargs.pop("max_eval_examples", None))
    seed: int = int(kwargs.pop("seed", 42))
    shuffle: bool = bool(kwargs.pop("shuffle", False))

    # Surogate-injected kwargs: accept but ignore. The orchestrator sets these on
    # `extra_env_kwargs` for every train env; we never use them because our rubric
    # is empty (RULER provides the reward).
    kwargs.pop("score_rollouts", None)
    kwargs.pop("max_seq_len", None)

    if kwargs:
        raise ValueError(
            f"Unrecognized args for ruler-task: {sorted(kwargs)}. "
            "Check spelling against the load_environment docstring."
        )

    train_ds = _build_split(
        dataset_id=dataset_id,
        split=dataset_split,
        hf_config=hf_config,
        user_template=user_template,
        system_prompt=system_prompt,
        task_name=task_name,
        max_examples=max_examples,
        seed=seed,
        shuffle=shuffle,
        which="train",
    )
    eval_ds = (
        _build_split(
            dataset_id=dataset_id,
            split=eval_split,
            hf_config=hf_config,
            user_template=user_template,
            system_prompt=system_prompt,
            task_name=task_name,
            max_examples=max_eval_examples,
            seed=seed,
            shuffle=shuffle,
            which="eval",
        )
        if eval_split
        else None
    )

    logger.info(
        "Loaded ruler-task '%s': train=%d row(s)%s",
        task_name,
        len(train_ds),
        f", eval={len(eval_ds)} row(s)" if eval_ds is not None else "",
    )

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        rubric=vf.Rubric(),  # empty — RULER replaces this orchestrator-side
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _require(kwargs: dict[str, Any], key: str, ty: type, *, allow_empty: bool) -> Any:
    """Pop a required key from kwargs; validate type and (optionally) non-empty."""
    if key not in kwargs:
        raise ValueError(f"Missing required arg '{key}' for ruler-task")
    value = kwargs.pop(key)
    if not isinstance(value, ty):
        raise TypeError(f"Arg '{key}' must be {ty.__name__}, got {type(value).__name__}")
    if not allow_empty and isinstance(value, str) and not value.strip():
        raise ValueError(f"Arg '{key}' must be a non-empty string")
    return value


def _resolve_system_prompt(*, inline: str | None, path: str | None) -> str:
    """Return the system prompt text, sourcing from inline or file. Exactly one required."""
    if (inline is None) == (path is None):
        raise ValueError(
            "Provide exactly one of system_prompt (inline) or system_prompt_path "
            f"(got inline={inline is not None}, path={path is not None})"
        )
    if inline is not None:
        if not isinstance(inline, str) or not inline.strip():
            raise ValueError("system_prompt must be a non-empty string")
        return inline.strip()
    p = Path(path)  # type: ignore[arg-type]
    if not p.is_file():
        raise FileNotFoundError(f"system_prompt_path does not point to a file: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"system_prompt_path file is empty: {p}")
    return text


def _build_split(
    *,
    dataset_id: str,
    split: str,
    hf_config: str | None,
    user_template: str,
    system_prompt: str,
    task_name: str,
    max_examples: int | None,
    seed: int,
    shuffle: bool,
    which: str,
) -> Dataset:
    """Load a single split from HF Hub or a local file, then map to the verifiers schema."""
    raw = _load_raw_split(dataset_id=dataset_id, split=split, hf_config=hf_config, which=which)

    # Shuffle only when the user explicitly asks. With `shuffle=False`, a user who
    # passes `max_examples=N` against a curated dataset gets the first N rows in
    # source order — matching what `Dataset.select(range(N))` does intuitively.
    if shuffle:
        raw = raw.shuffle(seed=seed)

    if max_examples is not None and len(raw) > max_examples:
        raw = raw.select(range(max_examples))

    if len(raw) == 0:
        raise ValueError(
            f"ruler-task '{which}' split is empty after subsetting "
            f"(dataset={dataset_id!r}, split={split!r}, max_examples={max_examples})"
        )

    columns_present = set(raw.column_names)
    template_keys = _extract_format_keys(user_template)
    missing = template_keys - columns_present
    if missing:
        raise ValueError(
            f"user_template references columns missing from the {which} split: "
            f"{sorted(missing)} not in {sorted(columns_present)}. "
            f"Adjust user_template or rename the dataset columns."
        )

    return raw.map(
        _make_mapper(user_template=user_template, system_prompt=system_prompt, task_name=task_name),
        with_indices=True,
        remove_columns=raw.column_names,
    )


def _load_raw_split(*, dataset_id: str, split: str, hf_config: str | None, which: str) -> Dataset:
    """Load a single split from HF Hub or a local file."""
    p = Path(dataset_id)
    if p.exists():
        if p.is_dir():
            raise ValueError(f"dataset path {p} is a directory; pass a specific .jsonl/.csv/.parquet file")
        loader = _LOCAL_LOADERS.get(p.suffix.lower())
        if loader is None:
            raise ValueError(f"Unsupported file extension {p.suffix!r}; use one of {sorted(_LOCAL_LOADERS)}")
        # `datasets.load_dataset` wraps a single local file in a "train" pseudo-split.
        # Callers always pass a non-empty split name (defaults to "train"); the user
        # can override if they built a multi-split file (rare).
        ds = load_dataset(loader, data_files=str(p), split=split)
        if not isinstance(ds, Dataset):
            raise TypeError(f"Local loader returned {type(ds).__name__}, expected Dataset")
        return ds

    # Treat as HF Hub id.
    try:
        ds = load_dataset(dataset_id, hf_config, split=split)
    except Exception as e:
        raise ValueError(
            f"Failed to load HF dataset '{dataset_id}'"
            f"{' (config=' + hf_config + ')' if hf_config else ''} "
            f"split='{split}' for the {which} env: {type(e).__name__}: {e}"
        ) from e
    if not isinstance(ds, Dataset):
        raise TypeError(
            f"HF loader returned {type(ds).__name__} for '{dataset_id}' split '{split}'; "
            "expected a single Dataset (did you forget hf_config or pass a split that returns DatasetDict?)"
        )
    return ds


def _make_mapper(*, user_template: str, system_prompt: str, task_name: str):
    """Build the row mapper closure used by Dataset.map."""

    def _map(row: dict[str, Any], idx: int) -> dict[str, Any]:
        try:
            user_content = user_template.format_map(_FormatProxy(row))
        except (KeyError, ValueError, IndexError) as e:
            # _FormatProxy raises a friendly KeyError for missing keys; surface it.
            raise ValueError(f"Row {idx} could not be rendered by user_template={user_template!r}: {e}") from e
        return {
            "example_id": idx,
            "task": task_name,
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }

    return _map


class _FormatProxy(dict):
    """Dict that raises a clearer KeyError on missing keys during .format_map()."""

    def __missing__(self, key: str) -> str:
        raise KeyError(f"column '{key}' missing on row")


def _extract_format_keys(template: str) -> set[str]:
    """Return the set of identifier-style keys referenced by a str.format template.

    Conservative: any non-identifier or computed access (``{a.b}``, ``{a[0]}``)
    is ignored, since we only validate plain ``{column}`` references against
    dataset columns. Computed accesses still work at runtime via _FormatProxy.
    """
    keys: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(template):
        if not field_name:
            continue
        head = field_name.split(".", 1)[0].split("[", 1)[0]
        if head.isidentifier():
            keys.add(head)
    return keys
