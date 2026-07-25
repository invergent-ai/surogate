"""Offline GRPO environment for Fugu live-control decisions.

The environment never executes a worker. It rewards the trainable conductor
directly against frozen, runtime-validated action signatures collected from
successful agentic trajectories and correction states.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

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

from scratchpad.eval_fugu_live_control_v8 import _prompt_state  # noqa: E402
from ultra.live_control import (  # noqa: E402
    ControlAction,
    ControlContractError,
    parse_control_decision,
    validate_control_decision,
)


DEFAULT_TRAIN = ROOT / "scratchpad/fugu_live_control_training_v14_action_only/train.jsonl"
DEFAULT_VALIDATION = (
    ROOT / "scratchpad/fugu_live_control_training_v14_action_only/validation.jsonl"
)
EXPECTED_TRAIN_SHA256 = "126681450851c950cdca33a36fcd86c44e35838c6c60ec6f5f846610da727a80"
EXPECTED_VALIDATION_SHA256 = "12935133990cb58e418dd5c5f4aa6cced152884e7093036c947687ad339f1277"


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
        raise TypeError("live-control answer must be an object")
    return answer


def _parsed(completion: Any) -> ControlAction | None:
    try:
        return parse_control_decision(_completion_text(completion))
    except (ControlContractError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _predicted(completion: Any, answer: Any) -> tuple[ControlAction | None, bool]:
    expected = _answer(answer)
    decision = _parsed(completion)
    if decision is None:
        return None, False
    try:
        state = _prompt_state(expected["state_prompt"])
        validate_control_decision(decision, state)
        return decision, True
    except (ControlContractError, KeyError, TypeError, ValueError, RuntimeError):
        return decision, False


def _signature(action: ControlAction | dict[str, Any]) -> tuple[str, int | None]:
    if isinstance(action, ControlAction):
        return action.action, action.target_position_id
    return str(action["action"]), action.get("target_position_id")


def decision_reward(completion: Any, answer: Any, **_: Any) -> float:
    """Reward exact live decisions and make unsafe completion strictly worse."""
    expected = _answer(answer)
    parsed = _parsed(completion)
    expected_signature = _signature(expected["expected"])
    if parsed is not None and parsed.action == "complete" and expected_signature[0] != "complete":
        return -1.0
    predicted, valid = _predicted(completion, expected)
    if not valid or predicted is None:
        return -0.5
    predicted_signature = _signature(predicted)
    if predicted_signature == expected_signature:
        return 1.0
    if predicted.action == expected_signature[0]:
        return 0.5
    return 0.0


def exact_signature(completion: Any, answer: Any, **_: Any) -> float:
    predicted, valid = _predicted(completion, answer)
    if not valid or predicted is None:
        return 0.0
    return float(_signature(predicted) == _signature(_answer(answer)["expected"]))


def action_match(completion: Any, answer: Any, **_: Any) -> float:
    predicted, valid = _predicted(completion, answer)
    if not valid or predicted is None:
        return 0.0
    return float(predicted.action == _answer(answer)["expected"]["action"])


def contract_valid(completion: Any, answer: Any, **_: Any) -> float:
    _, valid = _predicted(completion, answer)
    return float(valid)


def false_complete(completion: Any, answer: Any, **_: Any) -> float:
    predicted = _parsed(completion)
    if predicted is None:
        return 0.0
    return float(
        predicted.action == "complete"
        and _answer(answer)["expected"]["action"] != "complete"
    )


def _load_rows(path: Path, expected_sha256: str, task_name: str) -> list[dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise RuntimeError(f"frozen live-control dataset changed: {path}")
    rows = []
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        source = json.loads(line)
        messages = source["messages"]
        if [message["role"] for message in messages] != ["system", "user", "assistant"]:
            raise RuntimeError(f"invalid message roles at {path}:{index + 1}")
        expected = json.loads(messages[2]["content"])
        state = _prompt_state(messages[1]["content"])
        validate_control_decision(parse_control_decision(messages[2]["content"]), state)
        record_id = str(source["record_id"])
        rows.append(
            {
                "example_id": index,
                "task": task_name,
                "prompt": messages[:2],
                "answer": {
                    "expected": expected,
                    "state_prompt": messages[1]["content"],
                    "record_id": record_id,
                    "cohort": "correction" if "__interrupt_train_" in record_id else "v10_replay",
                },
                "info": {
                    "record_id": record_id,
                    "task_id": source["task_id"],
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
    train_path: str = str(DEFAULT_TRAIN),
    validation_path: str = str(DEFAULT_VALIDATION),
    train_sha256: str = EXPECTED_TRAIN_SHA256,
    validation_sha256: str = EXPECTED_VALIDATION_SHA256,
    task_name: str = "fugu_live_control",
    shuffle: bool = True,
    seed: int = 42,
    max_examples: int | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Load the local decision-reward environment with no provider surface."""
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
        funcs=[decision_reward, exact_signature, action_match, contract_valid, false_complete],
        weights=[1.0, 0.0, 0.0, 0.0, 0.0],
    )
    return vf.SingleTurnEnv(
        dataset=train,
        eval_dataset=validation,
        rubric=rubric,
        **kwargs,
    )
