"""Core value types shared across the worker pool."""

from __future__ import annotations

from dataclasses import dataclass

Message = dict[str, str]  # {"role": ..., "content": ...}


@dataclass(frozen=True)
class Sampling:
    """Sampling parameters. Frozen so it can be part of a cache key.

    ``reasoning_effort`` ("minimal"|"low"|"medium"|"high") maps to OpenRouter's unified
    reasoning control. Per the Fugu report (§4.1.1), workers run at MAXIMUM reasoning
    effort, which needs a high token cap to avoid truncation. Fugu's low latency comes from
    free routing + a single worker call, not from lowering worker reasoning.
    """

    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1024
    seed: int | None = None
    reasoning_effort: str | None = None

    def as_dict(self) -> dict:
        d = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
        # Only key on reasoning_effort when set, so existing (effort-free) cache entries
        # stay valid; None == "don't send the param" == the prior behavior.
        if self.reasoning_effort is not None:
            d["reasoning_effort"] = self.reasoning_effort
        return d


@dataclass
class Completion:
    """A worker response plus accounting metadata."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    error: str | None = None
    finish_reason: str | None = None  # "stop" ok; "length" => truncated (cap too low)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolCompletion:
    """A function-calling worker response: free-text content and/or tool calls."""

    content: str | None
    tool_calls: list[ToolCall]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False

    def as_text(self) -> str:
        """Render for the routing transcript (raw role:content view)."""
        if self.tool_calls:
            calls = "; ".join(f"{t.name}({t.arguments})" for t in self.tool_calls)
            return f"{self.content or ''}\n[tool_calls] {calls}".strip()
        return self.content or ""
