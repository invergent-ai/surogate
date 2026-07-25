"""Convert outcome-labeled public router traces into anonymous live control data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from ultra.live_control import (
    ControlBudget,
    ControlPosition,
    LiveControlState,
    WorkerProfile,
    build_control_decision_messages,
    parse_control_decision,
    validate_control_decision,
)


CORPUS_VERSION = "fugu_public_router_corpus_v1"
DATASET_ID = "JetBrains-Research/agent-trajectories-swesmith-random-subset"
EXPECTED_SOURCE_SHA256 = (
    "67f8150d0483d9550102d97e687cce70991ac2bf877e186ebc4009c749da9d23"
)
EXPECTED_REPLAY_TRAIN_SHA256 = (
    "e3257e57ecf9eb9f13031dfe83d8e80d9ab294c7d7a2acd0027a8ef4e2f7526f"
)
EXPECTED_REPLAY_VALIDATION_SHA256 = (
    "78662174f1956f83fb50942582094b0389a66822812c4e617bd60ea26d932ed4"
)
EXPECTED_SOURCE_COUNTS = {
    "rows": 1_465,
    "resolved": 675,
    "failed": 744,
    "unknown": 46,
    "assistant_turns": 17_901,
    "transitions": 16_436,
    "resolved_transitions": 6_652,
}
SOURCE_MODEL_TO_PROFILE = {
    "gpt-5-mini": "fast_generalist",
    "gpt-5.2": "deep_generalist",
}
PROFILE_TAGS = {
    "fast_generalist": (
        "fast_generalist",
        "repository_navigation",
        "incremental_implementation",
    ),
    "deep_generalist": (
        "deep_generalist",
        "complex_debugging",
        "verification",
    ),
}
PROFILE_CONSTRAINTS = {
    "fast_generalist": (
        "Prefer when rapid repository exploration or incremental implementation "
        "fits the current evidence.",
    ),
    "deep_generalist": (
        "Prefer when the current evidence calls for deeper debugging or verification.",
    ),
}
SPLIT_SEED = "fugu-public-router-v1-20260720"
HOLDOUT_PERCENT = 20
PUBLIC_TRAIN_CAPS = {"continue": 1_000, "handoff": 400}
PUBLIC_VALIDATION_CAPS = {"continue": 200, "handoff": 100}
REPLAY_TRAIN_REPEATS = 4
MAX_TASK_CHARS = 10_000
MAX_TERMINAL_CHARS = 8_000
MAX_ACTIVITY_ANALYSIS_CHARS = 1_600
MAX_ACTIVITY_OBSERVATION_CHARS = 1_600
MAX_COMMAND_CHARS = 500
RECENT_ACTIVITY_LIMIT = 3
SYSTEM_PROMPT = (
    "You are the live conductor for a multi-step, tool-using agentic task. "
    "Return one valid compact control-decision JSON object."
)


class PublicRouterCorpusError(ValueError):
    """A source trace or derived conductor row violates the corpus contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_int(*parts: str) -> int:
    payload = "\x1f".join(parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def _bounded(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return f"[...{len(value) - limit} earlier characters omitted...]\n{value[-limit:]}"


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PublicRouterCorpusError(
                f"invalid replay JSON at {path}:{number}"
            ) from exc
        if not isinstance(row, dict):
            raise PublicRouterCorpusError(f"replay row {number} is not an object")
        result.append(row)
    return result


def _repository(instance_id: str) -> str:
    repository = instance_id.split(".", 1)[0].strip()
    if not repository or "__" not in repository:
        raise PublicRouterCorpusError(f"invalid SWE-Smith instance_id: {instance_id}")
    return repository


def _split(repository: str) -> str:
    value = _stable_int(SPLIT_SEED, repository) % 100
    return "holdout" if value < HOLDOUT_PERCENT else "train"


def _profile_assignment(instance_id: str) -> tuple[dict[str, int], tuple[WorkerProfile, ...]]:
    names = ("fast_generalist", "deep_generalist")
    if _stable_int(SPLIT_SEED, "profile-permutation", instance_id) % 2:
        names = tuple(reversed(names))
    profile_to_worker = {name: index for index, name in enumerate(names)}
    workers = tuple(
        WorkerProfile(
            worker_id=index,
            capability_tags=PROFILE_TAGS[name],
            tool_tags=("terminal", "filesystem", "test_runner"),
            constraints=PROFILE_CONSTRAINTS[name],
        )
        for index, name in enumerate(names)
    )
    return profile_to_worker, workers


def _message_content(message: Any, *, label: str) -> str:
    if not isinstance(message, dict) or set(message) != {"content", "role"}:
        raise PublicRouterCorpusError(f"{label} has an invalid message schema")
    content = message.get("content")
    role = message.get("role")
    if not isinstance(content, str) or role not in {"system", "user", "assistant"}:
        raise PublicRouterCorpusError(f"{label} has invalid role or content")
    return content


_CODE_BLOCK = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _command_summary(content: str) -> str:
    blocks = _CODE_BLOCK.findall(content)
    source = blocks[-1] if blocks else content
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    return _bounded(lines[0] if lines else "No command text retained.", MAX_COMMAND_CHARS)


def _analysis_summary(content: str) -> str:
    without_commands = _CODE_BLOCK.sub("", content)
    return _bounded(without_commands or content, MAX_ACTIVITY_ANALYSIS_CHARS)


def _phase(content: str) -> str:
    lower = content.lower()
    if any(token in lower for token in ("pytest", "test ", "verify", "check")):
        return "verify"
    if any(token in lower for token in ("sed -i", "apply_patch", "cat >", "python - <<")):
        return "implement"
    return "inspect"


def _recent_activity(
    messages: list[dict[str, str]],
    assistant_indexes: list[int],
    transition_index: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    start = max(0, transition_index - RECENT_ACTIVITY_LIMIT)
    for source_turn in range(start, transition_index):
        assistant_index = assistant_indexes[source_turn]
        next_index = (
            assistant_indexes[source_turn + 1]
            if source_turn + 1 < len(assistant_indexes)
            else len(messages)
        )
        assistant = _message_content(
            messages[assistant_index], label=f"assistant turn {source_turn}"
        )
        observations = [
            _message_content(messages[index], label=f"message {index}")
            for index in range(assistant_index + 1, next_index)
            if messages[index].get("role") == "user"
        ]
        result.append(
            {
                "turn": source_turn + 1,
                "task_complete": False,
                "command_count": 1,
                "command_summaries": [_command_summary(assistant)],
                "analysis": _analysis_summary(assistant),
                "plan": "Continue from the resulting repository and terminal state.",
                "observation_excerpt": _bounded(
                    observations[-1] if observations else "No observation retained.",
                    MAX_ACTIVITY_OBSERVATION_CHARS,
                ),
            }
        )
    return result


def _original_task(messages: list[dict[str, str]], first_assistant: int) -> str:
    for index in range(first_assistant):
        if messages[index].get("role") == "user":
            return _bounded(
                _message_content(messages[index], label=f"message {index}"),
                MAX_TASK_CHARS,
            )
    raise PublicRouterCorpusError("source trajectory has no user task")


def _latest_observation(
    messages: list[dict[str, str]], current_assistant_index: int
) -> str:
    for index in range(current_assistant_index - 1, -1, -1):
        if messages[index].get("role") == "user":
            return _bounded(
                _message_content(messages[index], label=f"message {index}"),
                MAX_TERMINAL_CHARS,
            )
    raise PublicRouterCorpusError("transition has no preceding observation")


def _decision_content(action: str) -> str:
    payload: dict[str, Any] = {"action": action}
    if action == "handoff":
        payload["target_position_id"] = 1
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _transition_row(
    *,
    source_index: int,
    instance_id: str,
    repository: str,
    messages: list[dict[str, str]],
    assistant_indexes: list[int],
    selections: list[str],
    transition_index: int,
    source_sha256: str,
) -> dict[str, Any]:
    previous_model = selections[transition_index - 1]
    selected_model = selections[transition_index]
    previous_profile = SOURCE_MODEL_TO_PROFILE[previous_model]
    selected_profile = SOURCE_MODEL_TO_PROFILE[selected_model]
    action = "continue" if previous_profile == selected_profile else "handoff"
    profile_to_worker, workers = _profile_assignment(instance_id)
    active_worker = profile_to_worker[previous_profile]
    alternate_profile = next(name for name in PROFILE_TAGS if name != previous_profile)
    alternate_worker = profile_to_worker[alternate_profile]
    current_assistant_index = assistant_indexes[transition_index]
    history_payload = json.dumps(
        messages[:current_assistant_index],
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    previous_assistant = _message_content(
        messages[assistant_indexes[transition_index - 1]],
        label=f"assistant turn {transition_index - 1}",
    )
    progress = {
        "worker_report": {
            "phase": _phase(previous_assistant),
            "evidence": _bounded(previous_assistant, MAX_ACTIVITY_ANALYSIS_CHARS),
        },
        "completion_requested": False,
        "turns": transition_index,
        "checkpoint": {
            "source": "outcome_verified_public_router",
            "assistant_turn": transition_index,
            "history_sha256": hashlib.sha256(history_payload).hexdigest(),
        },
        "recent_activity": _recent_activity(
            messages, assistant_indexes, transition_index
        ),
    }
    state = LiveControlState(
        original_task=_original_task(messages, assistant_indexes[0]),
        workers=workers,
        workflow_id=_stable_int(CORPUS_VERSION, instance_id) % 2_000_000_000,
        positions=(
            ControlPosition(
                position_id=0,
                worker_id=active_worker,
                subtask=(
                    "Continue solving the repository task from the preserved shell and "
                    "code state. Inspect, implement, test, and report concrete evidence."
                ),
                access=(),
                status="active",
                progress=progress,
            ),
            ControlPosition(
                position_id=1,
                worker_id=alternate_worker,
                subtask=(
                    "Take over the same repository task from the preserved checkpoint. "
                    "Reassess the evidence, debug the current approach, and verify the result."
                ),
                access=(0,),
                status="pending",
            ),
        ),
        active_position_id=0,
        terminal_status="ready",
        terminal_observation=_latest_observation(messages, current_assistant_index),
        shared_memory=(),
        budget=ControlBudget(
            paid_calls_used=transition_index,
            paid_call_limit=120,
            elapsed_s=float(transition_index * 60),
            wall_time_limit_s=7_200.0,
        ),
    )
    decision_content = _decision_content(action)
    decision = parse_control_decision(decision_content)
    validate_control_decision(decision, state)
    prompt_messages, _, compacted = build_control_decision_messages(state)
    if compacted:
        raise PublicRouterCorpusError("unexpected prompt compaction without a counter")
    output_messages = [*prompt_messages, {"role": "assistant", "content": decision_content}]
    learned_surface = json.dumps(output_messages, sort_keys=True).lower()
    leaked = [name for name in SOURCE_MODEL_TO_PROFILE if name.lower() in learned_surface]
    if leaked:
        raise PublicRouterCorpusError(f"source model identity leaked into prompt: {leaked}")
    profile_permutation = {
        profile: worker_id for profile, worker_id in sorted(profile_to_worker.items())
    }
    record_suffix = hashlib.sha256(
        f"{instance_id}:{transition_index}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "record_id": f"hf_router__{record_suffix}",
        "task_id": f"hf_swesmith__{instance_id}",
        "repository": repository,
        "split": _split(repository),
        "variant": "outcome_verified_public_router_transition",
        "action": action,
        "profile_permutation": profile_permutation,
        "messages": output_messages,
        "provenance": {
            "dataset_id": DATASET_ID,
            "source_sha256": source_sha256,
            "source_row_index": source_index,
            "source_transition_index": transition_index,
            "whole_trajectory_outcome": "resolved",
            "counterfactual": False,
        },
    }


def _critic_row(
    *,
    source_index: int,
    instance_id: str,
    repository: str,
    messages: list[dict[str, str]],
    selections: list[str],
    resolved: bool,
    exit_status: str | None,
    source_sha256: str,
) -> dict[str, Any]:
    profile_to_worker, _ = _profile_assignment(instance_id)
    profile_sequence = [
        profile_to_worker[SOURCE_MODEL_TO_PROFILE[model]] for model in selections
    ]
    learned_surface = json.dumps(messages, sort_keys=True).lower()
    leaked = [name for name in SOURCE_MODEL_TO_PROFILE if name.lower() in learned_surface]
    if leaked:
        raise PublicRouterCorpusError(
            f"source model identity leaked into critic messages: {leaked}"
        )
    return {
        "record_id": f"hf_router_critic__{source_index:04d}",
        "task_id": f"hf_swesmith__{instance_id}",
        "repository": repository,
        "split": _split(repository),
        "resolved": resolved,
        "exit_status": exit_status or "unknown",
        "profile_sequence": profile_sequence,
        "messages": messages,
        "provenance": {
            "dataset_id": DATASET_ID,
            "source_sha256": source_sha256,
            "source_row_index": source_index,
            "source_model_to_profile": SOURCE_MODEL_TO_PROFILE,
        },
    }


def _round_robin_select(
    rows: list[dict[str, Any]], caps: dict[str, int]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for action, cap in caps.items():
        by_task: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["action"] == action:
                by_task[row["task_id"]].append(row)
        for task_rows in by_task.values():
            task_rows.sort(
                key=lambda row: _stable_int(
                    SPLIT_SEED, "row-order", action, row["record_id"]
                )
            )
        tasks = sorted(
            by_task,
            key=lambda task: _stable_int(SPLIT_SEED, "task-order", action, task),
        )
        depth = 0
        action_rows: list[dict[str, Any]] = []
        while len(action_rows) < cap:
            added = False
            for task in tasks:
                if depth < len(by_task[task]):
                    action_rows.append(by_task[task][depth])
                    added = True
                    if len(action_rows) == cap:
                        break
            if not added:
                break
            depth += 1
        if len(action_rows) != cap:
            raise PublicRouterCorpusError(
                f"insufficient {action} rows: wanted {cap}, found {len(action_rows)}"
            )
        selected.extend(action_rows)
    return sorted(
        selected,
        key=lambda row: _stable_int(SPLIT_SEED, "selected-order", row["record_id"]),
    )


def _action(row: dict[str, Any]) -> str:
    try:
        return json.loads(row["messages"][-1]["content"])["action"]
    except Exception as exc:
        raise PublicRouterCorpusError("training row has no compact action target") from exc


def _replay_rows(rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for row in rows:
            copied = copy.deepcopy(row)
            copied["record_id"] = f"{row['record_id']}__public_mix_replay_{repeat:02d}"
            copied["mixture_source"] = "accepted_v16_replay"
            copied["mixture_repeat"] = repeat
            output.append(copied)
    return output


def _public_mixture_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        copied = copy.deepcopy(row)
        copied["mixture_source"] = "public_router_success"
        output.append(copied)
    return output


def _validate_prompt_contract(rows: list[dict[str, Any]]) -> dict[str, bool]:
    prompts: dict[str, str] = {}
    ids: set[str] = set()
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id in ids:
            raise PublicRouterCorpusError("derived record IDs are missing or duplicated")
        ids.add(record_id)
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 3
            or messages[0].get("content") != SYSTEM_PROMPT
        ):
            raise PublicRouterCorpusError("derived compact message schema drift")
        content = messages[-1].get("content")
        decision = parse_control_decision(content)
        prompt = messages[1].get("content")
        if not isinstance(prompt, str):
            raise PublicRouterCorpusError("derived prompt is not text")
        previous = prompts.setdefault(prompt, decision.action)
        if previous != decision.action:
            raise PublicRouterCorpusError("identical prompt has conflicting actions")
    return {
        "record_ids_unique": True,
        "prompts_conflict_free": True,
        "compact_decisions_parse": True,
    }


def build_public_router_corpus(
    *,
    source_parquet: Path,
    replay_train_jsonl: Path,
    replay_validation_jsonl: Path,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise PublicRouterCorpusError("pyarrow is required to read the pinned source") from exc

    source_sha256 = _sha256(source_parquet)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise PublicRouterCorpusError("public router source hash drift")
    if _sha256(replay_train_jsonl) != EXPECTED_REPLAY_TRAIN_SHA256:
        raise PublicRouterCorpusError("accepted v16 train replay hash drift")
    if _sha256(replay_validation_jsonl) != EXPECTED_REPLAY_VALIDATION_SHA256:
        raise PublicRouterCorpusError("accepted v16 validation replay hash drift")

    table = pq.read_table(source_parquet)
    expected_columns = {
        "messages",
        "instance_id",
        "n_turns",
        "n_messages",
        "selected_models",
        "resolved",
        "exit_status",
    }
    if set(table.column_names) != expected_columns:
        raise PublicRouterCorpusError("public router source columns drift")
    source = table.to_pydict()
    public_rows: list[dict[str, Any]] = []
    critic_rows: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter(rows=table.num_rows)
    repositories: dict[str, str] = {}
    for source_index in range(table.num_rows):
        instance_id = source["instance_id"][source_index]
        messages = source["messages"][source_index]
        selections = source["selected_models"][source_index]
        resolved = source["resolved"][source_index]
        exit_status = source["exit_status"][source_index]
        if (
            not isinstance(instance_id, str)
            or not isinstance(messages, list)
            or not isinstance(selections, list)
            or (exit_status is not None and not isinstance(exit_status, str))
        ):
            raise PublicRouterCorpusError(f"source row {source_index} schema drift")
        if any(model not in SOURCE_MODEL_TO_PROFILE for model in selections):
            raise PublicRouterCorpusError(f"source row {source_index} has unknown worker")
        for index, message in enumerate(messages):
            _message_content(message, label=f"source row {source_index} message {index}")
        assistant_indexes = [
            index for index, message in enumerate(messages) if message["role"] == "assistant"
        ]
        if (
            len(selections) != len(assistant_indexes)
            or source["n_turns"][source_index] != len(selections)
            or source["n_messages"][source_index] != len(messages)
        ):
            raise PublicRouterCorpusError(f"source row {source_index} turn alignment drift")
        repository = _repository(instance_id)
        repositories[repository] = _split(repository)
        source_counts["assistant_turns"] += len(selections)
        source_counts["transitions"] += max(0, len(selections) - 1)
        if resolved is True:
            outcome_counts["resolved"] += 1
            source_counts["resolved_transitions"] += max(0, len(selections) - 1)
            for transition_index in range(1, len(selections)):
                public_rows.append(
                    _transition_row(
                        source_index=source_index,
                        instance_id=instance_id,
                        repository=repository,
                        messages=messages,
                        assistant_indexes=assistant_indexes,
                        selections=selections,
                        transition_index=transition_index,
                        source_sha256=source_sha256,
                    )
                )
        elif resolved is False:
            outcome_counts["failed"] += 1
        elif resolved is None:
            outcome_counts["unknown"] += 1
        else:
            raise PublicRouterCorpusError(f"source row {source_index} outcome drift")
        if resolved in {True, False}:
            critic_rows.append(
                _critic_row(
                    source_index=source_index,
                    instance_id=instance_id,
                    repository=repository,
                    messages=messages,
                    selections=selections,
                    resolved=resolved,
                    exit_status=exit_status,
                    source_sha256=source_sha256,
                )
            )

    actual_source_counts = {
        "rows": source_counts["rows"],
        "resolved": outcome_counts["resolved"],
        "failed": outcome_counts["failed"],
        "unknown": outcome_counts["unknown"],
        "assistant_turns": source_counts["assistant_turns"],
        "transitions": source_counts["transitions"],
        "resolved_transitions": source_counts["resolved_transitions"],
    }
    if actual_source_counts != EXPECTED_SOURCE_COUNTS:
        raise PublicRouterCorpusError(
            f"public router source count drift: {actual_source_counts}"
        )

    train_public = [row for row in public_rows if row["split"] == "train"]
    holdout_public = [row for row in public_rows if row["split"] == "holdout"]
    train_critic = [row for row in critic_rows if row["split"] == "train"]
    holdout_critic = [row for row in critic_rows if row["split"] == "holdout"]
    selected_train = _round_robin_select(train_public, PUBLIC_TRAIN_CAPS)
    selected_validation = _round_robin_select(
        holdout_public, PUBLIC_VALIDATION_CAPS
    )
    replay_train = _read_jsonl(replay_train_jsonl)
    replay_validation = _read_jsonl(replay_validation_jsonl)
    mixture_train = [
        *_replay_rows(replay_train, REPLAY_TRAIN_REPEATS),
        *_public_mixture_rows(selected_train),
    ]
    mixture_train.sort(
        key=lambda row: _stable_int(SPLIT_SEED, "mixture-train", row["record_id"])
    )
    mixture_validation = [
        *_replay_rows(replay_validation, 1),
        *_public_mixture_rows(selected_validation),
    ]
    mixture_validation.sort(
        key=lambda row: _stable_int(SPLIT_SEED, "mixture-validation", row["record_id"])
    )

    train_repositories = {
        row["repository"] for row in train_public
    }
    holdout_repositories = {
        row["repository"] for row in holdout_public
    }
    source_identity_leaks = 0
    for row in public_rows:
        surface = json.dumps(row["messages"], sort_keys=True).lower()
        source_identity_leaks += sum(
            model.lower() in surface for model in SOURCE_MODEL_TO_PROFILE
        )
    prompt_gates = _validate_prompt_contract(public_rows)
    mixture_prompt_gates = _validate_prompt_contract(mixture_train + mixture_validation)
    mixture_sources = Counter(row["mixture_source"] for row in mixture_train)
    mixture_actions = Counter(_action(row) for row in mixture_train)
    public_actions = {
        split: dict(
            sorted(
                Counter(
                    row["action"]
                    for row in public_rows
                    if row["split"] == split
                ).items()
            )
        )
        for split in ("train", "holdout")
    }
    selected_actions = {
        "train": dict(sorted(Counter(_action(row) for row in selected_train).items())),
        "validation": dict(
            sorted(Counter(_action(row) for row in selected_validation).items())
        ),
    }
    critic_outcomes = {
        "train": dict(
            sorted(Counter(str(row["resolved"]).lower() for row in train_critic).items())
        ),
        "holdout": dict(
            sorted(
                Counter(str(row["resolved"]).lower() for row in holdout_critic).items()
            )
        ),
    }
    gates = {
        **prompt_gates,
        "mixture_contract_valid": all(mixture_prompt_gates.values()),
        "source_snapshot_exact": True,
        "resolved_only_action_imitation": all(
            row["provenance"]["whole_trajectory_outcome"] == "resolved"
            for row in public_rows
        ),
        "repository_disjoint": not (train_repositories & holdout_repositories),
        "source_identity_absent_from_learned_surface": source_identity_leaks == 0,
        "public_action_coverage": all(
            set(public_actions[split]) == {"continue", "handoff"}
            for split in ("train", "holdout")
        ),
        "critic_outcome_coverage": all(
            set(critic_outcomes[split]) == {"false", "true"}
            for split in ("train", "holdout")
        ),
        "accepted_replay_majority": (
            mixture_sources["accepted_v16_replay"]
            > mixture_sources["public_router_success"]
        ),
        "accepted_four_action_replay_preserved": set(mixture_actions)
        == {"continue", "handoff", "replan", "complete"},
    }
    ready = all(gates.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "public_train": output_dir / "public_train.jsonl",
        "public_holdout": output_dir / "public_holdout.jsonl",
        "critic_train": output_dir / "critic_train.jsonl",
        "critic_holdout": output_dir / "critic_holdout.jsonl",
        "selected_train": output_dir / "selected_train.jsonl",
        "selected_validation": output_dir / "selected_validation.jsonl",
        "mixture_train": output_dir / "mixture_train.jsonl",
        "mixture_validation": output_dir / "mixture_validation.jsonl",
    }
    payloads = {
        "public_train": train_public,
        "public_holdout": holdout_public,
        "critic_train": train_critic,
        "critic_holdout": holdout_critic,
        "selected_train": selected_train,
        "selected_validation": selected_validation,
        "mixture_train": mixture_train,
        "mixture_validation": mixture_validation,
    }
    for name, path in paths.items():
        _write_jsonl(path, payloads[name])

    manifest = {
        "version": CORPUS_VERSION,
        "source": {
            "dataset_id": DATASET_ID,
            "path": str(source_parquet),
            "sha256": source_sha256,
            "counts": actual_source_counts,
            "source_model_to_anonymous_profile_provenance": SOURCE_MODEL_TO_PROFILE,
        },
        "split": {
            "unit": "repository",
            "seed": SPLIT_SEED,
            "holdout_percent": HOLDOUT_PERCENT,
            "train_repositories": len(train_repositories),
            "holdout_repositories": len(holdout_repositories),
            "repository_overlap": len(train_repositories & holdout_repositories),
            "train_tasks": len({row["task_id"] for row in train_public}),
            "holdout_tasks": len({row["task_id"] for row in holdout_public}),
        },
        "public_actions": public_actions,
        "selected_actions": selected_actions,
        "critic_outcomes": critic_outcomes,
        "mixture": {
            "replay_train_repeats": REPLAY_TRAIN_REPEATS,
            "train_rows": len(mixture_train),
            "validation_rows": len(mixture_validation),
            "train_sources": dict(sorted(mixture_sources.items())),
            "train_actions": dict(sorted(mixture_actions.items())),
            "replay_fraction": round(
                mixture_sources["accepted_v16_replay"] / len(mixture_train), 6
            ),
        },
        "policy": {
            "action_imitation_source": "resolved_trajectories_only",
            "failed_trajectories": "trajectory_critic_candidates_only",
            "public_action_scope": ["continue", "handoff"],
            "replacement_topology_claim": False,
            "counterfactual_claim": False,
            "profile_ids_permuted_by_task": True,
            "concrete_source_identities_in_learned_surface": False,
            "accepted_replay_is_training_majority": True,
            "external_model_calls": 0,
            "paid_calls": 0,
        },
        "gates": gates,
        "ready_for_bounded_local_training": ready,
        "artifacts": {
            name: {
                "path": str(path),
                "rows": len(payloads[name]),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--replay-train", type=Path, required=True)
    parser.add_argument("--replay-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_public_router_corpus(
        source_parquet=args.source_parquet,
        replay_train_jsonl=args.replay_train,
        replay_validation_jsonl=args.replay_validation,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["ready_for_bounded_local_training"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
