"""Direct, retry-free client for ALE's CUA computer-server action surface."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

from ultra.tool_actions import ComputerAction


class AleCuaError(RuntimeError):
    """ALE's computer-server rejected or failed a desktop action."""


@dataclass(frozen=True)
class AleCuaObservation:
    action: str
    text: str
    is_error: bool = False
    image_base64: str | None = None
    media_type: str | None = None


Transport = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


_KEY_ALIASES = {
    "arrowup": "up",
    "arrowdown": "down",
    "arrowleft": "left",
    "arrowright": "right",
    "control": "ctrl",
    "option": "alt",
    "meta": "cmd",
    "command": "cmd",
    "win": "cmd",
    "super": "cmd",
    "return": "enter",
    "escape": "esc",
    "pageup": "page_up",
    "pagedown": "page_down",
    "capslock": "caps_lock",
    "printscreen": "print_screen",
}


class AleCuaClient:
    """Execute normalized desktop actions against one ALE sandbox."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s
        self._transport = transport
        self._screen_size: tuple[int, int] | None = None

    async def _send(self, command: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            result = await self._transport(command, params)
        else:
            result = await asyncio.to_thread(self._send_sync, command, params)
        if result.get("success") is False:
            raise AleCuaError(
                f"CUA command {command!r} failed: {result.get('error', 'unknown error')}"
            )
        return result

    def _send_sync(self, command: str, params: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps({"command": command, "params": params}).encode("utf-8")
        request = Request(
            f"{self.endpoint}/cmd",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
            payload = response.read().decode("utf-8", errors="replace")
        result: dict[str, Any] | None = None
        for line in payload.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                candidate = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                result = candidate
        if result is None:
            raise AleCuaError(f"CUA command {command!r} returned no valid SSE result")
        return result

    async def _dimensions(self) -> tuple[int, int]:
        if self._screen_size is None:
            result = await self._send("get_screen_size", {})
            size = result.get("size")
            if isinstance(size, dict):
                width, height = size.get("width"), size.get("height")
            else:
                width, height = result.get("width"), result.get("height")
            if not isinstance(width, int) or not isinstance(height, int):
                raise AleCuaError("CUA get_screen_size returned invalid dimensions")
            if width <= 0 or height <= 0:
                raise AleCuaError("CUA get_screen_size returned non-positive dimensions")
            self._screen_size = (width, height)
        return self._screen_size

    async def _absolute(self, coordinate: list[float | int]) -> dict[str, int]:
        width, height = await self._dimensions()
        return {
            "x": round(float(coordinate[0]) / 1000 * width),
            "y": round(float(coordinate[1]) / 1000 * height),
        }

    @staticmethod
    def _key(value: str) -> str:
        lowered = value.lower()
        return _KEY_ALIASES.get(lowered, lowered)

    async def execute(self, action: ComputerAction) -> AleCuaObservation:
        """Execute one already-validated action without transport retries."""
        name = action.name
        arguments = action.arguments
        if name == "screenshot":
            result = await self._send("screenshot", {})
            image = result.get("image_data")
            if not isinstance(image, str) or not image:
                raise AleCuaError("CUA screenshot returned no image data")
            return AleCuaObservation(
                action=name,
                text="Screenshot captured.",
                image_base64=image,
                media_type="image/png",
            )
        if name == "screen_size":
            width, height = await self._dimensions()
            return AleCuaObservation(name, f"Screen size: {width}x{height}.")
        if name == "cursor_position":
            result = await self._send("get_cursor_position", {})
            position = result.get("position") or {}
            width, height = await self._dimensions()
            x = round(float(position.get("x", 0)) / width * 1000)
            y = round(float(position.get("y", 0)) / height * 1000)
            return AleCuaObservation(name, f"Cursor position: [{x}, {y}].")
        if name == "wait":
            duration = float(arguments["duration"])
            await asyncio.sleep(duration)
            return AleCuaObservation(name, f"Waited {duration:g} seconds.")
        if name == "type":
            await self._send("type_text", {"text": arguments["text"]})
            return AleCuaObservation(name, f"Typed {len(arguments['text'])} characters.")
        if name in {"key", "key_down", "key_up"}:
            keys = [self._key(key) for key in arguments["keys"]]
            if name == "key":
                if len(keys) == 1:
                    await self._send("press_key", {"key": keys[0]})
                else:
                    await self._send("hotkey", {"keys": keys})
            else:
                command = name
                for key in keys:
                    await self._send(command, {"key": key})
            return AleCuaObservation(name, f"{name}: {'+'.join(keys)}.")
        if name == "move":
            point = await self._absolute(arguments["coordinate"])
            await self._send("move_cursor", point)
            return AleCuaObservation(name, f"Moved cursor to {arguments['coordinate']}.")
        if name == "click":
            point = await self._absolute(arguments["coordinate"])
            button = arguments["button"]
            clicks = arguments["clicks"]
            if button == "left" and clicks == 2:
                await self._send("double_click", point)
            elif button == "middle":
                await self._send("move_cursor", point)
                for _ in range(clicks):
                    await self._send("mouse_down", {"button": "middle"})
                    await self._send("mouse_up", {"button": "middle"})
            else:
                command = "right_click" if button == "right" else "left_click"
                for _ in range(clicks):
                    await self._send(command, point)
            return AleCuaObservation(
                name,
                f"Clicked {button} {clicks} time(s) at {arguments['coordinate']}.",
            )
        if name == "drag":
            if "start_coordinate" in arguments:
                start = await self._absolute(arguments["start_coordinate"])
                await self._send("move_cursor", start)
            button = arguments["button"]
            await self._send("mouse_down", {"button": button})
            destination = await self._absolute(arguments["coordinate"])
            await self._send("move_cursor", destination)
            await self._send("mouse_up", {"button": button})
            return AleCuaObservation(name, f"Dragged to {arguments['coordinate']}.")
        if name in {"mouse_down", "mouse_up"}:
            await self._send(name, {"button": arguments["button"]})
            return AleCuaObservation(name, f"{name}: {arguments['button']}.")
        if name == "scroll":
            coordinate = arguments.get("coordinate")
            if coordinate is not None:
                await self._send("move_cursor", await self._absolute(coordinate))
            await self._send(
                "scroll_direction",
                {"direction": arguments["direction"], "clicks": arguments["amount"]},
            )
            return AleCuaObservation(
                name,
                f"Scrolled {arguments['direction']} {arguments['amount']} unit(s).",
            )
        raise AleCuaError(f"unsupported validated CUA action: {name}")
