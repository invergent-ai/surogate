"""Tests for Anthropic prompt-cache breakpoint injection."""

from __future__ import annotations

from director.shared.prompt_cache import is_anthropic, with_cache_control

MSGS = [
    {"role": "system", "content": "you are an agent"},
    {"role": "user", "content": "first turn"},
    {"role": "assistant", "content": "ok"},
    {"role": "user", "content": "second turn"},
]
TOOLS = [{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "b"}}]


def _has_cc(content) -> bool:
    return isinstance(content, list) and any("cache_control" in b for b in content)


def test_anthropic_detection():
    assert is_anthropic("anthropic/claude-opus-4.8")
    assert is_anthropic("openrouter/anthropic/claude-3")  # claude in slug
    assert not is_anthropic("openai/gpt-5.5")
    assert not is_anthropic("google/gemini-3.5-flash")


def test_injects_for_anthropic():
    msgs, tools = with_cache_control(MSGS, TOOLS, model="anthropic/claude-opus-4.8")
    assert _has_cc(msgs[0]["content"])   # system marked
    assert _has_cc(msgs[-1]["content"])  # last message (rolling prefix) marked
    assert "cache_control" in tools[-1]  # last tool marked
    # original is untouched (deepcopy)
    assert isinstance(MSGS[0]["content"], str)
    assert "cache_control" not in TOOLS[-1]


def test_noop_for_non_anthropic():
    msgs, tools = with_cache_control(MSGS, TOOLS, model="openai/gpt-5.5")
    assert msgs == MSGS and tools == TOOLS  # unchanged (OpenAI auto-caches)
