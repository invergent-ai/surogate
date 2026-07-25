"""Build masked structural/outcome training data for the unified Fugu conductor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from surogate.train.tokenize import (
    TokenizedDataFileWriter,
    _to_input_mask,
    pack_and_write,
)
from ultra.live_control import (
    ControlAction,
    ControlBudget,
    ControlStep,
    LiveControlState,
    WorkerProfile,
    build_control_action_messages,
    remap_control_action_workers,
    remap_control_state_workers,
    validate_control_action,
)


SCHEMA_VERSION = "fugu_ornith_structural_outcome_v1"
SEQUENCE_LEN = 2816
CURRENT_POOL_REPEAT = 2
MAX_LEGACY_TRAIN_PREFERENCES = 48
OUTCOME_TASK_CHARS = 1_200
TOOL_TAGS = ("terminal", "filesystem", "test_runner")
STRUCTURAL_PATTERNS = (
    re.compile(r'"worker_id"\s*:\s*\d+'),
    re.compile(r'"access"\s*:\s*\[[^\]]*\]'),
)
REPLAY_PATTERNS = (
    re.compile(r'"action"\s*:\s*"(?:continue|handoff|replan|complete)"'),
    re.compile(r'"target_position_id"\s*:\s*\d+'),
    *STRUCTURAL_PATTERNS,
)


class StructuralTrainingError(ValueError):
    """A training row violates the structural-credit contract."""


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StructuralTrainingError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise StructuralTrainingError(f"JSONL contains a non-object row: {path}")
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    count = 0
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = stable_json(row) + "\n"
            handle.write(line)
            digest.update(line.encode("ascii"))
            count += 1
    return count, digest.hexdigest()


def _verify_distillation_corpus(corpus: Path) -> dict[str, Any]:
    report = _read_json(corpus / "report.json")
    if (
        report.get("schema_version") != "fugu_paid_rollout_distillation_v1"
        or report.get("status") != "ready_for_offline_distillation"
        or report.get("paid_calls") != 0
    ):
        raise StructuralTrainingError("distillation corpus is not admitted")
    for name in ("prompts.jsonl", "candidates.jsonl", "preferences.jsonl"):
        expected = ((report.get("artifacts") or {}).get(name) or {}).get("sha256")
        if not isinstance(expected, str) or sha256_file(corpus / name) != expected:
            raise StructuralTrainingError(f"distillation artifact hash drift: {name}")
    return report


def _token_count(
    tokenizer: PreTrainedTokenizerBase, messages: Sequence[dict[str, str]]
) -> int:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
    )
    return len(encoded["input_ids"])


def _initial_state(prompt: Mapping[str, Any]) -> LiveControlState:
    profiles = prompt.get("capability_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise StructuralTrainingError("distillation prompt has no capability profiles")
    workers = []
    for expected_id, profile in enumerate(profiles):
        if not isinstance(profile, Mapping) or profile.get("worker_id") != expected_id:
            raise StructuralTrainingError("capability profiles are not contiguous")
        tags = profile.get("capability_tags")
        if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
            raise StructuralTrainingError("capability profile tags are invalid")
        workers.append(
            WorkerProfile(
                worker_id=expected_id,
                capability_tags=tuple(tags),
                tool_tags=TOOL_TAGS,
            )
        )
    task_text = prompt.get("task_text")
    if not isinstance(task_text, str) or not task_text.strip():
        raise StructuralTrainingError("distillation prompt has no task text")
    task_text = task_text.strip()
    if len(task_text) > OUTCOME_TASK_CHARS:
        half = OUTCOME_TASK_CHARS // 2
        task_text = (
            task_text[:half]
            + f"\n\n[...{len(task_text) - 2 * half} middle characters omitted...]\n\n"
            + task_text[-half:]
        )
    return LiveControlState(
        original_task=task_text,
        workers=tuple(workers),
        workflow_id=None,
        positions=(),
        active_position_id=None,
        terminal_status="ready",
        terminal_observation="Current Terminal Screen:\nroot@container:/workspace#",
        shared_memory=(),
        budget=ControlBudget(
            paid_calls_used=0,
            paid_call_limit=120,
            elapsed_s=0.0,
            wall_time_limit_s=1_500.0,
        ),
    )


def _chosen_action(candidate: Mapping[str, Any]) -> ControlAction:
    workflow = candidate.get("workflow")
    raw_steps = workflow.get("steps") if isinstance(workflow, Mapping) else None
    if not isinstance(raw_steps, list) or not raw_steps:
        raise StructuralTrainingError("chosen candidate has no valid workflow")
    steps = []
    for position, raw in enumerate(raw_steps):
        if not isinstance(raw, Mapping) or raw.get("position_id") != position:
            raise StructuralTrainingError("chosen workflow positions are invalid")
        worker_id = raw.get("worker_id")
        subtask = raw.get("subtask")
        access = raw.get("access")
        if (
            isinstance(worker_id, bool)
            or not isinstance(worker_id, int)
            or not isinstance(subtask, str)
            or not subtask.strip()
            or not isinstance(access, list)
            or not all(isinstance(item, int) and not isinstance(item, bool) for item in access)
        ):
            raise StructuralTrainingError("chosen workflow step is invalid")
        steps.append(
            ControlStep(
                worker_id=worker_id,
                subtask=subtask.strip(),
                access=tuple(access),
            )
        )
    return ControlAction(
        action="replan",
        reason=(
            "Use the outcome-supported capability roles and access topology for this task."
        ),
        steps=tuple(steps),
    )


def _rotation_map(rotation: int, worker_count: int) -> dict[int, int]:
    return {
        worker_id: (worker_id + rotation) % worker_count
        for worker_id in range(worker_count)
    }


def _rotate(
    state: LiveControlState, action: ControlAction, rotation: int
) -> tuple[LiveControlState, ControlAction]:
    mapping = _rotation_map(rotation, len(state.workers))
    worker_order = tuple(sorted(state.worker_ids, key=mapping.__getitem__))
    rotated_state = remap_control_state_workers(
        state,
        mapping,
        worker_order=worker_order,
    )
    rotated_action = remap_control_action_workers(action, mapping)
    validate_control_action(rotated_action, rotated_state)
    return rotated_state, rotated_action


def _action_content(action: ControlAction) -> str:
    payload: dict[str, Any] = {
        "action": action.action,
        "reason": action.reason,
    }
    if action.target_position_id is not None:
        payload["target_position_id"] = action.target_position_id
    if action.steps:
        payload["steps"] = [asdict(step) for step in action.steps]
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _outcome_rows(
    *, corpus: Path, tokenizer: PreTrainedTokenizerBase
) -> list[dict[str, Any]]:
    prompt_rows = {row["prompt_id"]: row for row in _read_jsonl(corpus / "prompts.jsonl")}
    preferences = [
        row for row in _read_jsonl(corpus / "preferences.jsonl") if row.get("split") == "train"
    ]
    current_preferences = [
        row for row in preferences if row.get("pool_epoch") == "current_pool"
    ]
    legacy_preferences = sorted(
        (row for row in preferences if row.get("pool_epoch") == "legacy_pool"),
        key=lambda row: (
            -float(row["chosen_topology_mean_reward"]),
            -float(row["reward_margin"]),
            -min(
                int(row["chosen_topology_count"]),
                int(row["rejected_topology_count"]),
            ),
            str(row["preference_id"]),
        ),
    )[:MAX_LEGACY_TRAIN_PREFERENCES]
    preferences = [*current_preferences, *legacy_preferences]
    chosen_ids = {row["chosen_candidate_id"] for row in preferences}
    candidates = {
        row["candidate_id"]: row
        for row in _read_jsonl(corpus / "candidates.jsonl")
        if row.get("candidate_id") in chosen_ids
    }
    if set(candidates) != chosen_ids:
        raise StructuralTrainingError("one or more chosen candidates are missing")

    rows: list[dict[str, Any]] = []
    for preference in sorted(
        preferences,
        key=lambda row: (str(row["pool_epoch"]), str(row["prompt_id"])),
    ):
        prompt = prompt_rows.get(preference["prompt_id"])
        candidate = candidates.get(preference["chosen_candidate_id"])
        if prompt is None or candidate is None:
            raise StructuralTrainingError("preference references a missing prompt or candidate")
        state = _initial_state(prompt)
        action = _chosen_action(candidate)
        repeats = CURRENT_POOL_REPEAT if preference["pool_epoch"] == "current_pool" else 1
        for repeat in range(repeats):
            for rotation in range(len(state.workers)):
                rotated_state, rotated_action = _rotate(state, action, rotation)
                messages, prompt_tokens, compacted = build_control_action_messages(
                    rotated_state,
                    prompt_token_counter=lambda value: _token_count(tokenizer, value),
                    max_input_tokens=2_400,
                )
                content = _action_content(rotated_action)
                rows.append(
                    {
                        "row_id": hashlib.sha256(
                            f"{preference['preference_id']}\0{repeat}\0{rotation}".encode(
                                "ascii"
                            )
                        ).hexdigest(),
                        "source": "paid_outcome_topology",
                        "mask_mode": "outcome_structure",
                        "pool_epoch": preference["pool_epoch"],
                        "prompt_id": preference["prompt_id"],
                        "preference_id": preference["preference_id"],
                        "rotation": rotation,
                        "repeat": repeat,
                        "reward_margin": preference["reward_margin"],
                        "chosen_topology_mean_reward": preference[
                            "chosen_topology_mean_reward"
                        ],
                        "prompt_tokens": prompt_tokens,
                        "prompt_compacted": compacted,
                        "messages": [
                            *messages,
                            {"role": "assistant", "content": content},
                        ],
                    }
                )
    return rows


def _replay_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    emitted: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        action = row.get("action")
        if (
            not isinstance(messages, list)
            or len(messages) < 3
            or not isinstance(messages[-1], Mapping)
            or messages[-1].get("role") != "assistant"
            or action not in {"continue", "handoff", "replan", "complete"}
        ):
            raise StructuralTrainingError("unified replay row is invalid")
        emitted.append(
            {
                "row_id": str(row.get("record_id")),
                "source": "accepted_unified_replay",
                "mask_mode": "replay_control",
                "action": action,
                "messages": messages,
            }
        )
    return emitted


def _target_spans(content: str, mode: str) -> list[tuple[int, int]]:
    patterns = STRUCTURAL_PATTERNS if mode == "outcome_structure" else REPLAY_PATTERNS
    spans = sorted((match.start(), match.end()) for pattern in patterns for match in pattern.finditer(content))
    if not spans:
        raise StructuralTrainingError(f"{mode} row has no target structural fields")
    return spans


def _encode_row(
    row: Mapping[str, Any], tokenizer: PreTrainedTokenizerBase
) -> dict[str, np.ndarray]:
    messages = row["messages"]
    prompt_messages = messages[:-1]
    content = messages[-1]["content"]
    if not isinstance(content, str) or not content:
        raise StructuralTrainingError(f"row {row['row_id']} has no assistant content")
    prefix = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not rendered.startswith(prefix) or not rendered[len(prefix) :].startswith(content):
        raise StructuralTrainingError(
            f"row {row['row_id']} chat template does not preserve assistant content"
        )
    content_start = len(prefix)
    spans = [
        (content_start + start, content_start + end)
        for start, end in _target_spans(content, str(row["mask_mode"]))
    ]
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = np.asarray(encoded["input_ids"], dtype=np.int32)
    offsets = encoded["offset_mapping"]
    if len(token_ids) > SEQUENCE_LEN:
        raise StructuralTrainingError(
            f"row {row['row_id']} has {len(token_ids)} tokens, above {SEQUENCE_LEN}"
        )
    target_mask = np.asarray(
        [
            any(start < span_end and end > span_start for span_start, span_end in spans)
            if end > start
            else False
            for start, end in offsets
        ],
        dtype=np.int32,
    )
    if not target_mask.any():
        raise StructuralTrainingError(f"row {row['row_id']} has an empty token mask")
    return {
        "tokens": token_ids,
        "mask": _to_input_mask(target_mask),
        "target_tokens": np.asarray([int(target_mask.sum())], dtype=np.int32),
    }


def build_training_data(
    *,
    corpus: Path,
    replay: Path,
    model: str | Path,
    output_dir: Path,
) -> dict[str, Any]:
    corpus = corpus.resolve()
    replay = replay.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise StructuralTrainingError(f"output directory already exists: {output_dir}")
    corpus_report = _verify_distillation_corpus(corpus)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model),
        local_files_only=True,
        trust_remote_code=False,
    )
    outcome_rows = _outcome_rows(corpus=corpus, tokenizer=tokenizer)
    replay_rows = _replay_rows(replay)
    rows = [*outcome_rows, *replay_rows]
    if len(replay_rows) <= len(outcome_rows):
        raise StructuralTrainingError("accepted replay must remain the row majority")

    docs: list[dict[str, np.ndarray]] = []
    lengths: list[int] = []
    target_counts: list[int] = []
    source_target_counts: Counter[str] = Counter()
    for row in rows:
        encoded = _encode_row(row, tokenizer)
        docs.append(encoded)
        lengths.append(len(encoded["tokens"]))
        target_count = int(encoded["target_tokens"][0])
        target_counts.append(target_count)
        source_target_counts[str(row["source"])] += target_count

    output_dir.mkdir(parents=True)
    try:
        bin_path = output_dir / "train-000.bin"
        with TokenizedDataFileWriter(
            str(bin_path),
            int(tokenizer.vocab_size),
            masking=True,
            non_overlapping=False,
        ) as writer:
            pack_and_write(
                writer,
                docs,
                SEQUENCE_LEN,
                int(tokenizer.pad_token_id),
            )
        row_count, rows_hash = _write_jsonl(output_dir / "rows.jsonl", rows)
        source_counts = Counter(str(row["source"]) for row in rows)
        action_counts = Counter(
            str(row["action"])
            for row in replay_rows
            if isinstance(row.get("action"), str)
        )
        epoch_counts = Counter(
            str(row["pool_epoch"])
            for row in outcome_rows
            if isinstance(row.get("pool_epoch"), str)
        )
        total_batch_size = SEQUENCE_LEN * 6 * 2
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "ready_for_local_structural_training",
            "external_calls": 0,
            "paid_calls": 0,
            "inputs": {
                "distillation_report_sha256": sha256_file(corpus / "report.json"),
                "distillation_preferences_sha256": (
                    corpus_report["artifacts"]["preferences.jsonl"]["sha256"]
                ),
                "replay_path": replay.as_posix(),
                "replay_sha256": sha256_file(replay),
                "model": str(model),
            },
            "counts": {
                "rows": row_count,
                "source_rows": dict(sorted(source_counts.items())),
                "outcome_pool_epoch_rows": dict(sorted(epoch_counts.items())),
                "replay_actions": dict(sorted(action_counts.items())),
                "tokens_unpacked": sum(lengths),
                "tokens_packed": writer.n_tokens,
                "target_tokens": sum(target_counts),
                "source_target_tokens": dict(sorted(source_target_counts.items())),
                "min_row_tokens": min(lengths),
                "max_row_tokens": max(lengths),
                "mean_row_tokens": sum(lengths) / len(lengths),
                "min_target_tokens": min(target_counts),
                "max_target_tokens": max(target_counts),
                "mean_target_tokens": sum(target_counts) / len(target_counts),
                "steps_per_epoch_at_effective_33792_tokens": writer.n_tokens
                / total_batch_size,
            },
            "mask_contract": {
                "paid_outcome_topology": ["worker_id", "access"],
                "accepted_unified_replay": [
                    "action",
                    "target_position_id",
                    "worker_id",
                    "access",
                ],
                "subtask_tokens_directly_optimized": False,
                "workflow_length_reward": False,
            },
            "artifacts": {
                "train-000.bin": {
                    "sha256": sha256_file(bin_path),
                    "tokens": writer.n_tokens,
                },
                "rows.jsonl": {"sha256": rows_hash, "rows": row_count},
            },
            "training_contract": {
                "parent": "output/fugu_ultra_unified_ornith_stage1_v2",
                "adapter_init_mode": "trainable",
                "current_pool_preference_repeat": CURRENT_POOL_REPEAT,
                "maximum_legacy_train_preferences": MAX_LEGACY_TRAIN_PREFERENCES,
                "legacy_selection": (
                    "highest chosen mean reward, reward margin, and repeated evidence"
                ),
                "replay_row_majority": True,
                "fresh_paid_collection_authorized": False,
                "promotion_requires_one_frozen_local_gate": True,
            },
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_training_data(
        corpus=args.corpus,
        replay=args.replay,
        model=args.model,
        output_dir=args.output_dir,
    )
    print(json.dumps({"output_dir": str(args.output_dir), **report["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
