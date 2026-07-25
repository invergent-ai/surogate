"""Offline GRPO environment for anonymous planner role/topology decisions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import verifiers as vf
from datasets import Dataset


def _ensure_repo_imports() -> Path:
    root = Path(__file__).resolve().parents[2]
    for path in (root, root / "ultra", root / "director"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


ROOT = _ensure_repo_imports()

from director.agentic.fugu_planner_role_grpo import (  # noqa: E402
    normalize_planner_payload,
    workflow_signature,
)
from ultra.workflow import (  # noqa: E402
    WorkflowValidationError,
    parse_workflow,
    validate_workflow,
)


DEFAULT_TRAIN = ROOT / "scratchpad/fugu_planner_role_grpo_v1/train.jsonl"
DEFAULT_VALIDATION = ROOT / "scratchpad/fugu_planner_role_grpo_v1/validation.jsonl"
IMPLEMENT_RE = re.compile(
    r"\b(?:implement|modify|edit|apply|repair|build|write|create|recover|produce|configure|install|complete)\w*\b",
    re.IGNORECASE,
)
VERIFY_RE = re.compile(
    r"\b(?:verif|test|check|validat|compar|confirm|audit|inspect|prove)\w*\b",
    re.IGNORECASE,
)
REPAIR_RE = re.compile(r"\b(?:repair|fix|correct|remediat)\w*\b", re.IGNORECASE)


def _load_prompt_parser() -> Any:
    path = ROOT / "environments/fugu-ultra-pilot/fugu_ultra_pilot.py"
    spec = importlib.util.spec_from_file_location("fugu_planner_role_prompt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load planner prompt parser: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._extract_workflow_payload


EXTRACT_WORKFLOW_PAYLOAD = _load_prompt_parser()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion:
        last = completion[-1]
        content = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "") if isinstance(part, dict) else getattr(part, "text", ""))
                for part in content
            )
        return str(content or "")
    return str(completion or "")


def _answer(answer: Any) -> dict[str, Any]:
    if isinstance(answer, str):
        answer = json.loads(answer)
    if not isinstance(answer, dict):
        raise TypeError("planner answer must be an object")
    return answer


def _parsed_steps(completion: Any, worker_count: int) -> list[dict[str, Any]] | None:
    try:
        normalized = normalize_planner_payload(
            _completion_text(completion), EXTRACT_WORKFLOW_PAYLOAD
        )
        workflow = parse_workflow(normalized)
        validate_workflow(workflow, worker_count)
    except (WorkflowValidationError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return [
        {
            "worker_id": step.worker_id,
            "subtask": step.subtask.strip(),
            "access": list(step.access),
        }
        for step in workflow.steps
    ]


def _lcs_length(left: Sequence[int], right: Sequence[int]) -> int:
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for index, right_value in enumerate(right, start=1):
            if left_value == right_value:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _semantic_score(steps: Sequence[dict[str, Any]]) -> float:
    joined = "\n".join(str(step["subtask"]) for step in steps)
    final = str(steps[-1]["subtask"]) if steps else ""
    implement = bool(IMPLEMENT_RE.search(joined))
    verify = bool(VERIFY_RE.search(final))
    repair = bool(REPAIR_RE.search(final))
    return 0.4 * implement + 0.4 * verify + 0.2 * repair


def planner_reward(completion: Any, answer: Any, **_: Any) -> float:
    """Reward capability-conditioned topology while retaining workflow quality."""
    expected = _answer(answer)
    predicted = _parsed_steps(completion, int(expected["worker_count"]))
    if predicted is None:
        return -1.0
    unavailable = set(expected["unavailable_worker_ids"])
    if unavailable & {step["worker_id"] for step in predicted}:
        return -1.0

    target = expected["expected_steps"]
    predicted_ids = [step["worker_id"] for step in predicted]
    target_ids = [step["worker_id"] for step in target]
    target_count = len(target)
    valid_component = 0.05
    length_component = 0.10 * float(len(predicted) == target_count)
    ordered_component = 0.25 * (
        _lcs_length(predicted_ids, target_ids) / max(1, target_count)
    )
    positional_component = 0.15 * (
        sum(
            predicted[index]["worker_id"] == target[index]["worker_id"]
            for index in range(min(len(predicted), target_count))
        )
        / max(1, target_count)
    )
    root_component = 0.15 * float(predicted_ids[-1] == target_ids[-1])
    access_component = 0.10 * (
        sum(
            predicted[index]["access"] == target[index]["access"]
            for index in range(min(len(predicted), target_count))
        )
        / max(1, target_count)
    )
    semantic_component = 0.20 * _semantic_score(predicted)
    return round(
        valid_component
        + length_component
        + ordered_component
        + positional_component
        + root_component
        + access_component
        + semantic_component,
        6,
    )


def exact_topology(completion: Any, answer: Any, **_: Any) -> float:
    expected = _answer(answer)
    predicted = _parsed_steps(completion, int(expected["worker_count"]))
    if predicted is None:
        return 0.0
    if set(expected["unavailable_worker_ids"]) & {
        step["worker_id"] for step in predicted
    }:
        return 0.0
    return float(workflow_signature(predicted) == expected["expected_steps"])


def contract_valid(completion: Any, answer: Any, **_: Any) -> float:
    expected = _answer(answer)
    return float(
        _parsed_steps(completion, int(expected["worker_count"])) is not None
    )


def unavailable_selected(completion: Any, answer: Any, **_: Any) -> float:
    expected = _answer(answer)
    predicted = _parsed_steps(completion, int(expected["worker_count"]))
    if predicted is None:
        return 0.0
    unavailable = set(expected["unavailable_worker_ids"])
    return float(bool(unavailable & {step["worker_id"] for step in predicted}))


def _load_rows(path: Path, expected_sha256: str, task_name: str) -> list[dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise RuntimeError(f"frozen planner-role dataset changed: {path}")
    rows = []
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        source = json.loads(line)
        messages = source["messages"]
        if [message["role"] for message in messages] != ["system", "user", "assistant"]:
            raise RuntimeError(f"invalid message roles at {path}:{index + 1}")
        expected_steps = _parsed_steps(messages[2]["content"], 4)
        if expected_steps is None:
            raise RuntimeError(f"invalid target workflow at {path}:{index + 1}")
        unavailable = source["unavailable_worker_ids"]
        if set(unavailable) & {step["worker_id"] for step in expected_steps}:
            raise RuntimeError(f"target selects unavailable worker at {path}:{index + 1}")
        rows.append(
            {
                "example_id": index,
                "task": task_name,
                "prompt": messages[:2],
                "answer": {
                    "expected_steps": workflow_signature(expected_steps),
                    "worker_count": 4,
                    "unavailable_worker_ids": unavailable,
                    "record_id": source["record_id"],
                    "cohort": source["cohort"],
                },
                "info": {
                    "record_id": source["record_id"],
                    "task_id": source["task_id"],
                    "cohort": source["cohort"],
                },
            }
        )
    return rows


def _dataset(
    path: Path,
    expected_sha256: str,
    task_name: str,
    *,
    shuffle: bool,
    seed: int,
    max_examples: int | None,
) -> Dataset:
    rows = _load_rows(path, expected_sha256, task_name)
    if shuffle:
        random.Random(seed).shuffle(rows)
    if max_examples is not None and max_examples > 0:
        rows = rows[:max_examples]
    return Dataset.from_list(rows)


def load_environment(
    *,
    train_path: str = str(DEFAULT_TRAIN),
    validation_path: str = str(DEFAULT_VALIDATION),
    train_sha256: str,
    validation_sha256: str,
    task_name: str = "fugu_planner_role",
    shuffle: bool = True,
    seed: int = 42,
    max_examples: int | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Load the local planner reward environment with no provider surface."""
    train = _dataset(
        Path(train_path).expanduser().resolve(),
        train_sha256,
        task_name,
        shuffle=shuffle,
        seed=seed,
        max_examples=max_examples,
    )
    validation = _dataset(
        Path(validation_path).expanduser().resolve(),
        validation_sha256,
        task_name,
        shuffle=False,
        seed=seed,
        max_examples=None,
    )
    rubric = vf.Rubric(
        funcs=[planner_reward, exact_topology, contract_valid, unavailable_selected],
        weights=[1.0, 0.0, 0.0, 0.0],
    )
    return vf.SingleTurnEnv(
        dataset=train,
        eval_dataset=validation,
        rubric=rubric,
        **kwargs,
    )
