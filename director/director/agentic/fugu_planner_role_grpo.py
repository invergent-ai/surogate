"""Pool-neutral transformations for planner-role policy training."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import replace
from typing import Any, Iterable, Sequence

from ultra.pool_binding import PoolBinding, PoolSlot, pool_fingerprint


MODEL_ID_LIST_RE = re.compile(r"(model[_ ]?id\s*=\s*)(\[[^\n]*\])")
AVAILABLE_MODELS_MARKER = "AVAILABLE LANGUAGE MODELS:\n"


def dihedral_worker_permutations(worker_count: int) -> tuple[tuple[int, ...], ...]:
    """Return deterministic old-id -> new-id rotations and reflections."""
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    candidates = [
        tuple((old_id + shift) % worker_count for old_id in range(worker_count))
        for shift in range(worker_count)
    ]
    candidates.extend(
        tuple((shift - old_id) % worker_count for old_id in range(worker_count))
        for shift in range(worker_count)
    )
    return tuple(dict.fromkeys(candidates))


def validate_permutation(permutation: Sequence[int], worker_count: int) -> None:
    if len(permutation) != worker_count or set(permutation) != set(range(worker_count)):
        raise ValueError(
            f"invalid worker permutation for {worker_count} slots: {tuple(permutation)!r}"
        )


def remap_worker_ids(ids: Iterable[int], permutation: Sequence[int]) -> tuple[int, ...]:
    validate_permutation(permutation, len(permutation))
    remapped = []
    for worker_id in ids:
        if isinstance(worker_id, bool) or not isinstance(worker_id, int):
            raise TypeError(f"worker id must be an integer: {worker_id!r}")
        if not 0 <= worker_id < len(permutation):
            raise ValueError(f"worker id is outside permutation: {worker_id}")
        remapped.append(permutation[worker_id])
    return tuple(remapped)


def remap_binding(binding: PoolBinding, permutation: Sequence[int]) -> PoolBinding:
    """Move complete capability profiles together while changing anonymous slots."""
    worker_count = len(binding.slots)
    validate_permutation(permutation, worker_count)
    remapped: list[PoolSlot | None] = [None] * worker_count
    for old_slot in binding.slots:
        new_id = permutation[old_slot.worker_id]
        remapped[new_id] = replace(old_slot, worker_id=new_id)
    slots = tuple(slot for slot in remapped if slot is not None)
    if len(slots) != worker_count:
        raise RuntimeError("worker permutation dropped a capability profile")
    fingerprint = pool_fingerprint(
        pool_id=binding.pool_id,
        provider_base=binding.provider_base,
        slots=slots,
    )
    return replace(binding, slots=slots, pool_fingerprint=fingerprint)


def remap_steps(
    steps: Sequence[dict[str, Any]], permutation: Sequence[int]
) -> list[dict[str, Any]]:
    """Remap worker IDs without changing position-index access edges."""
    out = []
    for step in steps:
        out.append(
            {
                "worker_id": remap_worker_ids((step["worker_id"],), permutation)[0],
                "subtask": str(step["subtask"]),
                "access": list(step.get("access", [])),
            }
        )
    return out


def assistant_workflow(steps: Sequence[dict[str, Any]]) -> str:
    """Render the deployed three-list conductor wire contract."""
    model_ids = [int(step["worker_id"]) for step in steps]
    subtasks = [str(step["subtask"]) for step in steps]
    access = [list(step.get("access", [])) for step in steps]
    return "\n".join(
        (
            f"model_id = {model_ids!r}",
            f"subtasks = {subtasks!r}",
            f"access_list = {access!r}",
        )
    )


def remap_demonstration_ids(text: str, permutation: Sequence[int]) -> str:
    """Keep format demonstrations consistent with permuted capability slots."""
    validate_permutation(permutation, len(permutation))

    def replace_ids(match: re.Match[str]) -> str:
        try:
            values = ast.literal_eval(match.group(2))
        except (SyntaxError, ValueError) as exc:
            raise ValueError("planner demonstration model_id list did not parse") from exc
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError("planner demonstration model_id must be an integer list")
        remapped = list(remap_worker_ids(values, permutation))
        return f"{match.group(1)}{remapped!r}"

    return MODEL_ID_LIST_RE.sub(replace_ids, text)


def replace_available_models(user_prompt: str, binding: PoolBinding) -> str:
    """Replace the terminal capability table using structured binding data."""
    if user_prompt.count(AVAILABLE_MODELS_MARKER) != 1:
        raise ValueError("planner prompt must contain one available-models section")
    prefix, _ = user_prompt.split(AVAILABLE_MODELS_MARKER, 1)
    lines = [
        f"Model {slot.worker_id}: roles={', '.join(slot.role_prior)}"
        for slot in binding.slots
    ]
    return prefix + AVAILABLE_MODELS_MARKER + "\n".join(lines)


def remap_messages(
    messages: Sequence[dict[str, str]],
    *,
    binding: PoolBinding,
    permutation: Sequence[int],
) -> list[dict[str, str]]:
    """Remap a canonical system/user/assistant planner conversation."""
    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"unexpected planner message roles: {roles!r}")
    remapped_binding = remap_binding(binding, permutation)
    return [
        {
            "role": "system",
            "content": remap_demonstration_ids(messages[0]["content"], permutation),
        },
        {
            "role": "user",
            "content": replace_available_models(
                messages[1]["content"], remapped_binding
            ),
        },
        {"role": "assistant", "content": messages[2]["content"]},
    ]


def normalize_planner_payload(raw: str, paper_payload_parser: Any) -> str:
    """Normalize JSON and paper-style spellings into internal workflow JSON."""
    payload = paper_payload_parser(raw)
    json_payload = payload.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        json_payload,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        json_payload = fenced.group(1)
    try:
        data = json.loads(json_payload)
    except (json.JSONDecodeError, TypeError):
        return payload
    if not isinstance(data, dict) or "steps" in data:
        return payload

    ids = data.get("model_id")
    subtasks = data.get("subtasks")
    access_list = data.get("access_list")
    if not all(isinstance(value, list) for value in (ids, subtasks, access_list)):
        return payload
    if len({len(ids), len(subtasks), len(access_list)}) != 1:
        return payload

    steps: list[dict[str, Any]] = []
    for index, (worker_id, subtask, access_entry) in enumerate(
        zip(ids, subtasks, access_list, strict=True)
    ):
        if isinstance(worker_id, bool) or not isinstance(worker_id, int):
            return payload
        if not isinstance(subtask, str):
            return payload
        if isinstance(access_entry, str):
            access_entry = [access_entry]
        if not isinstance(access_entry, list):
            return payload
        if len(access_entry) == 1 and (
            isinstance(access_entry[0], str)
            and access_entry[0].strip().lower() == "all"
        ):
            access = list(range(index))
        elif all(
            isinstance(entry, int) and not isinstance(entry, bool)
            for entry in access_entry
        ):
            access = access_entry
        else:
            return payload
        steps.append({"worker_id": worker_id, "subtask": subtask, "access": access})
    return json.dumps({"steps": steps})


def workflow_signature(steps: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "worker_id": int(step["worker_id"]),
            "access": [int(index) for index in step.get("access", [])],
        }
        for step in steps
    ]


def permutation_name(permutation: Sequence[int]) -> str:
    return "old_to_new__" + "_".join(str(value) for value in permutation)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
