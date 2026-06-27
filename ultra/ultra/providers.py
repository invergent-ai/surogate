"""Inference-provider registry — swap OpenRouter <-> yunwu without touching call sites.

Select the active provider with ``ULTRA_PROVIDER`` (default ``yunwu``). Each model has a
logical name; the registry maps it to the active provider's slug. Direct chat calls use
``base_url`` + the key in ``key_env``; the OpenCode agentic harness uses ``oc_provider``
(the ``-m <provider>/<slug>`` prefix) plus ``oc_config`` — a provider config file whose
``apiKey`` reads ``{env:<key_env>}`` so the secret stays in the environment, not on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

# logical worker name -> per-provider model slug
MODELS: dict[str, dict[str, str]] = {
    "flash":        {"openrouter": "deepseek/deepseek-v4-flash",    "yunwu": "deepseek-v4-flash"},
    "deepseek-pro": {"openrouter": "deepseek/deepseek-v4-pro",      "yunwu": "deepseek-v4-pro"},
    "glm":          {"openrouter": "z-ai/glm-5.2",                  "yunwu": "glm-5.2"},
    "kimi":         {"openrouter": "moonshotai/kimi-k2.7-code",     "yunwu": "kimi-k2.7-code"},
    "mimo":         {"openrouter": "xiaomi/mimo-v2.5-pro",          "yunwu": "mimo-v2.5-pro"},
    "minimax":      {"openrouter": "minimax/minimax-m3",            "yunwu": "MiniMax-M3"},
    "opus":         {"openrouter": "anthropic/claude-opus-4.8",     "yunwu": "claude-opus-4-8"},
    "gemini":       {"openrouter": "google/gemini-3.1-pro-preview", "yunwu": "gemini-3.1-pro-preview"},
    "gpt":          {"openrouter": "openai/gpt-5.5",                "yunwu": "gpt-5.5"},
}

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

PROVIDERS: dict[str, dict] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "oc_provider": "openrouter",      # OpenCode ships this provider built-in
        "oc_config": None,                # no extra config file needed
    },
    "yunwu": {
        "base_url": "https://yunwu.ai/v1",
        "key_env": "YUNWU_API_KEY",
        "oc_provider": "yunwu",
        "oc_config": str(_CONFIG_DIR / "opencode_yunwu.json"),
    },
}

DEFAULT_PROVIDER = "yunwu"


def active() -> str:
    return os.environ.get("ULTRA_PROVIDER", DEFAULT_PROVIDER)


def provider(name: str | None = None) -> dict:
    return PROVIDERS[name or active()]


def slug(logical: str, name: str | None = None) -> str:
    """Map a logical worker name to the active (or named) provider's model slug."""
    return MODELS[logical][name or active()]


def logical_names() -> list[str]:
    return list(MODELS)
