"""Regression tests for the ruler-task verifiers env.

Covers local file loading (jsonl, csv, parquet via the same code path),
HF Hub loading (gated; runs only when the network is allowed), templating,
subsetting, splits, validation matrix, and surogate-injected kwargs.

The env package lives at `environments/ruler-task/` rather than under
`surogate/`, so we add it to sys.path for import. This mirrors what verifiers
does at runtime via `vf.load_environment(env_id, env_path=...)`.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest
import verifiers as vf

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = REPO_ROOT / "environments" / "ruler-task"
sys.path.insert(0, str(ENV_DIR))

import ruler_task as srt  # noqa: E402  (sys.path injection above)

NETWORK_OK = os.environ.get("RUN_NETWORK_TESTS") == "1"


# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def jsonl_path(tmp_path: Path) -> Path:
    p = tmp_path / "inputs.jsonl"
    p.write_text("\n".join(json.dumps({"input": f"q{i}"}) for i in range(20)))
    return p


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    p = tmp_path / "rag.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["context", "instruction"])
        w.writeheader()
        for i in range(5):
            w.writerow({"context": f"context-{i}", "instruction": f"q-{i}"})
    return p


@pytest.fixture
def system_prompt_file(tmp_path: Path) -> Path:
    p = tmp_path / "system.txt"
    p.write_text("You are a deterministic assistant.\n")
    return p


# ── Local-file loading ──────────────────────────────────────────────────────


def test_jsonl_default_template(jsonl_path: Path):
    env = srt.load_environment(
        task_name="local-jsonl",
        dataset=str(jsonl_path),
        system_prompt="Be helpful.",
    )
    assert isinstance(env, vf.Environment)
    assert env.dataset.column_names == ["example_id", "task", "prompt"]
    assert len(env.dataset) == 20
    row = env.dataset[0]
    assert row["example_id"] == 0
    assert row["task"] == "local-jsonl"
    assert row["prompt"][0] == {"role": "system", "content": "Be helpful."}
    assert row["prompt"][1]["role"] == "user"
    assert row["prompt"][1]["content"] == "q0"


def test_csv_multi_column_template(csv_path: Path, system_prompt_file: Path):
    env = srt.load_environment(
        task_name="rag-qa",
        dataset=str(csv_path),
        user_template="Context:\n{context}\n\nQ: {instruction}",
        system_prompt_path=str(system_prompt_file),
    )
    assert len(env.dataset) == 5
    content = env.dataset[0]["prompt"][1]["content"]
    assert "Context:\ncontext-0" in content
    assert "Q: q-0" in content


def test_eval_split_wired(jsonl_path: Path):
    env = srt.load_environment(
        task_name="t",
        dataset=str(jsonl_path),
        system_prompt="x",
        eval_split="train",
        max_examples=10,
        max_eval_examples=3,
    )
    assert env.eval_dataset is not None
    assert len(env.dataset) == 10
    assert len(env.eval_dataset) == 3


# ── Subsetting / shuffle semantics ───────────────────────────────────────────


def test_max_examples_no_shuffle_returns_first_n_in_source_order(jsonl_path: Path):
    """Regression for the review's blocker: shuffle=False must keep source order."""
    env = srt.load_environment(
        task_name="t",
        dataset=str(jsonl_path),
        system_prompt="x",
        max_examples=5,
        shuffle=False,
    )
    user_msgs = [r["prompt"][1]["content"] for r in env.dataset]
    assert user_msgs == ["q0", "q1", "q2", "q3", "q4"], (
        f"shuffle=False with max_examples must take the head of the dataset, got {user_msgs}"
    )


def test_max_examples_with_shuffle_seeded_deterministic(jsonl_path: Path):
    a = srt.load_environment(
        task_name="t", dataset=str(jsonl_path), system_prompt="x", max_examples=5, shuffle=True, seed=42
    )
    b = srt.load_environment(
        task_name="t", dataset=str(jsonl_path), system_prompt="x", max_examples=5, shuffle=True, seed=42
    )
    c = srt.load_environment(
        task_name="t", dataset=str(jsonl_path), system_prompt="x", max_examples=5, shuffle=True, seed=99
    )
    msgs_a = [r["prompt"][1]["content"] for r in a.dataset]
    msgs_b = [r["prompt"][1]["content"] for r in b.dataset]
    msgs_c = [r["prompt"][1]["content"] for r in c.dataset]
    assert msgs_a == msgs_b, "same seed must produce same shuffle"
    assert msgs_a != msgs_c, "different seed must produce different shuffle"


@pytest.mark.parametrize("sentinel", [None, -1, 0, -100])
def test_max_examples_sentinels_mean_all(jsonl_path: Path, sentinel):
    """None and any non-positive int (codebase convention `-1`) load the full dataset."""
    env = srt.load_environment(
        task_name="t",
        dataset=str(jsonl_path),
        system_prompt="x",
        max_examples=sentinel,
    )
    assert len(env.dataset) == 20


# ── Surogate-injected kwargs ────────────────────────────────────────────────


def test_surogate_injected_kwargs_silently_accepted(jsonl_path: Path):
    """Orchestrator injects these via extra_env_kwargs; we must not error."""
    env = srt.load_environment(
        task_name="t",
        dataset=str(jsonl_path),
        system_prompt="x",
        score_rollouts=False,
        max_seq_len=4096,
    )
    assert isinstance(env, vf.Environment)


# ── Validation matrix ───────────────────────────────────────────────────────


def test_missing_task_name_raises(jsonl_path: Path):
    with pytest.raises(ValueError, match="'task_name'"):
        srt.load_environment(dataset=str(jsonl_path), system_prompt="x")


def test_missing_dataset_raises():
    with pytest.raises(ValueError, match="'dataset'"):
        srt.load_environment(task_name="t", system_prompt="x")


def test_no_system_prompt_raises(jsonl_path: Path):
    with pytest.raises(ValueError, match="exactly one"):
        srt.load_environment(task_name="t", dataset=str(jsonl_path))


def test_both_system_prompts_raises(jsonl_path: Path, system_prompt_file: Path):
    with pytest.raises(ValueError, match="exactly one"):
        srt.load_environment(
            task_name="t",
            dataset=str(jsonl_path),
            system_prompt="x",
            system_prompt_path=str(system_prompt_file),
        )


def test_unknown_kwarg_rejected(jsonl_path: Path):
    with pytest.raises(ValueError, match="Unrecognized"):
        srt.load_environment(
            task_name="t",
            dataset=str(jsonl_path),
            system_prompt="x",
            typo_arg=True,
        )


def test_template_missing_column(jsonl_path: Path):
    with pytest.raises(ValueError, match="missing from the train split"):
        srt.load_environment(
            task_name="t",
            dataset=str(jsonl_path),
            system_prompt="x",
            user_template="{nonexistent_column}",
        )


def test_empty_dataset_file_raises(tmp_path: Path):
    """Empty JSONL → datasets raises a generation error during load."""
    from datasets.exceptions import DatasetGenerationError

    p = tmp_path / "empty.jsonl"
    p.write_text("")
    # The downstream `datasets` library raises before our env sees an empty
    # split; we just need a clear failure rather than silent success.
    with pytest.raises((ValueError, RuntimeError, DatasetGenerationError)):
        srt.load_environment(task_name="t", dataset=str(p), system_prompt="x")


def test_unsupported_extension(tmp_path: Path):
    p = tmp_path / "data.xml"
    p.write_text("<root/>")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        srt.load_environment(task_name="t", dataset=str(p), system_prompt="x")


def test_empty_system_prompt_file(tmp_path: Path, jsonl_path: Path):
    p = tmp_path / "blank.txt"
    p.write_text("   \n\t  \n")
    with pytest.raises(ValueError, match="empty"):
        srt.load_environment(
            task_name="t",
            dataset=str(jsonl_path),
            system_prompt_path=str(p),
        )


# ── Full vf.load_environment integration ────────────────────────────────────


def test_vf_load_environment_resolves(jsonl_path: Path):
    """Mimics how the orchestrator loads the env."""
    env = vf.load_environment(
        "ruler-task",
        task_name="smoke",
        dataset=str(jsonl_path),
        system_prompt="You are a test assistant.",
        max_examples=3,
    )
    assert type(env).__name__ == "SingleTurnEnv"
    assert len(env.dataset) == 3
    # SingleTurnEnv adds a weight-0 num_turns reward func — that's expected
    weights = env.rubric._get_reward_weights()
    assert all(w == 0.0 for w in weights), (
        f"every default reward func must be weight-0 so the env contributes no training reward; got {weights}"
    )


# ── HF Hub loading (gated; opt-in via RUN_NETWORK_TESTS=1) ──────────────────


@pytest.mark.skipif(not NETWORK_OK, reason="set RUN_NETWORK_TESTS=1 to run HF Hub tests")
@pytest.mark.slow
def test_hf_hub_dataset_loads():
    """End-to-end load against a small HF dataset.

    Uses ``openai/gsm8k`` as a stable, well-known fixture. Subsamples to 5 rows
    so the test stays fast even on the first download.
    """
    env = srt.load_environment(
        task_name="gsm8k",
        dataset="openai/gsm8k",
        hf_config="main",
        dataset_split="train",
        user_template="{question}",
        system_prompt="Solve the problem.",
        max_examples=5,
    )
    assert len(env.dataset) == 5
    assert env.dataset.column_names == ["example_id", "task", "prompt"]
    user_msg = env.dataset[0]["prompt"][1]["content"]
    # GSM8K questions always end with a question mark or period.
    assert isinstance(user_msg, str) and len(user_msg) > 10
