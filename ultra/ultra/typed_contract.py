"""Typed capability-control contract helpers for training environments.

The 27B r4 conductor's frozen serving contract is the typed control-action
JSON built by `build_control_action_messages(state, capability_refs=True,
guidelines=None)` — NOT the paper's three-list format the pilot environment
speaks natively. Training must prompt and parse the SAME contract the policy
was SFT'd and served with (exact-token rule), so the environment gets these
helpers, which delegate to the very same code paths `TrainedConductor` uses
in production — byte-identical prompts by construction, not by copy.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from ultra.trained_conductor import TrainedConductor


class TypedContractParseError(ValueError):
    """Raised when a completion is not a valid typed replan action."""


def _first_json_object(raw: str) -> str:
    """The leading JSON value of ``raw``, ignoring anything after it.

    Exact-token pipelines deliver the completion WITH its end-of-turn
    framing token (e.g. ``{...}<|im_end|>``), which strict ``json.loads``
    rejects as "Extra data" — that single difference scored every campaign
    rollout 0.0 at launch. Structural fix: decode the first complete JSON
    value and drop the tail; no token-string matching involved.
    """
    stripped = raw.lstrip()
    try:
        _, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return raw  # not JSON at all — let the contract parser report it
    return stripped[:end]


def _conductor_for(role_priors: Sequence[Sequence[str]]) -> TrainedConductor:
    return TrainedConductor(roles=[tuple(r) for r in role_priors])


def typed_control_messages(
    question: str, role_priors: Sequence[Sequence[str]]
) -> list[dict[str, str]]:
    """The conductor's serving prompt for `question` over these profiles.

    Delegates to TrainedConductor._state/_messages so the rendering is the
    SAME code the served policy sees (guidelines=None per the frozen arm-A
    verdict; capability refs; identical budgets/state defaults).
    """
    conductor = _conductor_for(role_priors)
    return conductor._messages(conductor._state(question))


def parse_typed_workflow(
    raw: str, role_priors: Sequence[Sequence[str]]
) -> list[dict[str, Any]]:
    """Parse a typed replan completion into workflow-step dicts.

    Returns [{"worker_id": int, "subtask": str, "access": [int, ...]}, ...]
    with worker ids indexing the SAME order as `role_priors` (which must be
    the lane's worker order). Raises TypedContractParseError on anything
    that is not a valid non-empty replan — callers route that to their
    invalid-workflow (reward 0, trainable) branch.
    """
    from ultra.live_control import (
        WorkerProfile,
        capability_reference_map,
        parse_capability_control_action,
    )

    workers = tuple(
        WorkerProfile(worker_id=index, capability_tags=tuple(tags))
        for index, tags in enumerate(role_priors)
    )
    try:
        action = parse_capability_control_action(
            _first_json_object(raw), capability_reference_map(workers))
    except Exception as exc:  # noqa: BLE001 — contract errors are one class
        raise TypedContractParseError(str(exc)) from exc
    if action.action != "replan" or not action.steps:
        raise TypedContractParseError(
            f"not a non-empty replan (action={action.action!r})")
    return [
        {"worker_id": int(step.worker_id) % len(workers),
         "subtask": step.subtask,
         "access": [int(a) for a in step.access]}
        for step in action.steps
    ]
