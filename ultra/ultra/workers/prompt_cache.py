"""Provider prompt-caching support.

Discounts the repeated *prefix* of prompts within a multi-turn rollout (system + tools +
transcript history re-sent every turn). OpenAI/Gemini/most providers cache automatically;
Anthropic requires explicit ``cache_control`` breakpoints, which we inject here for
Anthropic-family workers (passed through by OpenRouter).

We mark up to the 4 allowed breakpoints on the stable prefix: the system message, the last
tool definition (caches all tools), and the last message before the current turn (caches
the conversation prefix). The completion cache key is computed on the *original* messages,
so injecting cache_control never changes our disk-cache keys.
"""

from __future__ import annotations

import copy

_EPHEMERAL = {"type": "ephemeral"}


def is_anthropic(model: str) -> bool:
    return model.split("/", 1)[0].lower() == "anthropic" or "claude" in model.lower()


def _mark(content) -> list:
    """Return content as a list of blocks with cache_control on the last block."""
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        blocks = copy.deepcopy(content)
    else:
        return content
    if blocks:
        blocks[-1] = {**blocks[-1], "cache_control": _EPHEMERAL}
    return blocks


def with_cache_control(messages: list, tools: list | None = None, model: str = "") -> tuple[list, list | None]:
    """Inject Anthropic cache_control on the stable prefix. No-op for non-Anthropic models."""
    if not is_anthropic(model):
        return messages, tools
    msgs = copy.deepcopy(messages)
    # breakpoint 1: system message (caches system + tools that follow it)
    for m in msgs:
        if m.get("role") == "system" and m.get("content"):
            m["content"] = _mark(m["content"])
            break
    # breakpoint 2: last message before the current turn (caches the conversation prefix)
    if len(msgs) >= 2 and msgs[-1].get("content") is not None:
        msgs[-1]["content"] = _mark(msgs[-1]["content"])
    # breakpoint 3: last tool definition (caches the whole tools block)
    new_tools = tools
    if tools:
        new_tools = copy.deepcopy(tools)
        new_tools[-1] = {**new_tools[-1], "cache_control": _EPHEMERAL}
    return msgs, new_tools
