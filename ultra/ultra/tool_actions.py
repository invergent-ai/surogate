"""Model-agnostic environment actions emitted by a Fugu worker position."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal


ComputerActionName = Literal[
    "screenshot",
    "screen_size",
    "cursor_position",
    "move",
    "click",
    "drag",
    "mouse_down",
    "mouse_up",
    "scroll",
    "type",
    "key",
    "key_down",
    "key_up",
    "wait",
]


class ToolActionContractError(ValueError):
    """A worker emitted an invalid environment action."""


@dataclass(frozen=True)
class ComputerAction:
    name: ComputerActionName
    arguments: dict[str, Any]


COMPUTER_ACTION_WORKER_CONTRACT = """
Desktop tool contract: when the assigned work requires the graphical desktop,
emit exactly one `computer_action` object in the same JSON response:
`{"name": <action>, "arguments": {...}}`. Supported actions are screenshot,
screen_size, cursor_position, move, click, drag, mouse_down, mouse_up, scroll,
type, key, key_down, key_up, and wait. Coordinates are `[x, y]` normalized to
0..1000. A response may contain terminal `commands` or one `computer_action`,
never both. Do not set task_complete in a response containing either action.
After a computer action, its result and any screenshot return to this private
position on the next turn. Omit `computer_action` when no desktop action is
needed.
""".strip()


_NO_ARGUMENT_ACTIONS = frozenset({"screenshot", "screen_size", "cursor_position"})
_BUTTONS = frozenset({"left", "right", "middle"})
_DIRECTIONS = frozenset({"up", "down", "left", "right"})


def _require_fields(arguments: dict[str, Any], allowed: set[str], required: set[str]) -> None:
    unexpected = set(arguments) - allowed
    if unexpected:
        raise ToolActionContractError(
            f"computer action has unexpected arguments: {sorted(unexpected)}"
        )
    missing = required - set(arguments)
    if missing:
        raise ToolActionContractError(
            f"computer action is missing arguments: {sorted(missing)}"
        )


def _coordinate(value: Any, name: str) -> list[float | int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ToolActionContractError(f"{name} must be a two-item coordinate")
    result: list[float | int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ToolActionContractError(f"{name} coordinates must be numeric")
        if not math.isfinite(float(item)) or not 0 <= float(item) <= 1000:
            raise ToolActionContractError(f"{name} coordinates must be within 0..1000")
        result.append(item)
    return result


def _button(arguments: dict[str, Any]) -> str:
    value = arguments.get("button", "left")
    if value not in _BUTTONS:
        raise ToolActionContractError("button must be left, right, or middle")
    return value


def _keys(value: Any) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ToolActionContractError("keys must contain between one and eight keys")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise ToolActionContractError("every key must be a non-empty string")
    return [key.strip() for key in value]


def parse_computer_action(payload: Any) -> ComputerAction | None:
    """Parse the optional strict `computer_action` member of a worker response."""
    if not isinstance(payload, dict) or "computer_action" not in payload:
        return None
    raw = payload["computer_action"]
    if not isinstance(raw, dict):
        raise ToolActionContractError("computer_action must be an object")
    unexpected = set(raw) - {"name", "arguments"}
    if unexpected:
        raise ToolActionContractError(
            f"computer_action has unexpected fields: {sorted(unexpected)}"
        )
    name = raw.get("name")
    if name not in ComputerActionName.__args__:
        raise ToolActionContractError(f"unsupported computer action: {name!r}")
    arguments = raw.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ToolActionContractError("computer_action.arguments must be an object")

    normalized: dict[str, Any]
    if name in _NO_ARGUMENT_ACTIONS:
        _require_fields(arguments, set(), set())
        normalized = {}
    elif name == "move":
        _require_fields(arguments, {"coordinate"}, {"coordinate"})
        normalized = {"coordinate": _coordinate(arguments["coordinate"], "coordinate")}
    elif name == "click":
        _require_fields(arguments, {"coordinate", "button", "clicks"}, {"coordinate"})
        clicks = arguments.get("clicks", 1)
        if isinstance(clicks, bool) or clicks not in {1, 2, 3}:
            raise ToolActionContractError("clicks must be 1, 2, or 3")
        normalized = {
            "coordinate": _coordinate(arguments["coordinate"], "coordinate"),
            "button": _button(arguments),
            "clicks": clicks,
        }
    elif name == "drag":
        _require_fields(
            arguments,
            {"coordinate", "start_coordinate", "button"},
            {"coordinate"},
        )
        normalized = {
            "coordinate": _coordinate(arguments["coordinate"], "coordinate"),
            "button": _button(arguments),
        }
        if "start_coordinate" in arguments:
            normalized["start_coordinate"] = _coordinate(
                arguments["start_coordinate"], "start_coordinate"
            )
    elif name in {"mouse_down", "mouse_up"}:
        _require_fields(arguments, {"button"}, set())
        normalized = {"button": _button(arguments)}
    elif name == "scroll":
        _require_fields(
            arguments,
            {"direction", "amount", "coordinate"},
            {"direction", "amount"},
        )
        direction = arguments["direction"]
        if direction not in _DIRECTIONS:
            raise ToolActionContractError("scroll direction must be up, down, left, or right")
        amount = arguments["amount"]
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(float(amount))
            or not 0 < float(amount) <= 100
        ):
            raise ToolActionContractError("scroll amount must be within (0, 100]")
        normalized = {"direction": direction, "amount": amount}
        if "coordinate" in arguments:
            normalized["coordinate"] = _coordinate(
                arguments["coordinate"], "coordinate"
            )
    elif name == "type":
        _require_fields(arguments, {"text"}, {"text"})
        value = arguments["text"]
        if not isinstance(value, str) or not value:
            raise ToolActionContractError("type text must be a non-empty string")
        if len(value) > 100_000:
            raise ToolActionContractError("type text exceeds 100000 characters")
        normalized = {"text": value}
    elif name in {"key", "key_down", "key_up"}:
        _require_fields(arguments, {"keys"}, {"keys"})
        normalized = {"keys": _keys(arguments["keys"])}
    else:
        _require_fields(arguments, {"duration"}, {"duration"})
        duration = arguments["duration"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or not 0 <= float(duration) <= 60
        ):
            raise ToolActionContractError("wait duration must be within [0, 60]")
        normalized = {"duration": duration}

    return ComputerAction(name=name, arguments=normalized)
