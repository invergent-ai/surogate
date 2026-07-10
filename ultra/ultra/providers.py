"""Inference-provider registry for split Ultra worker routing.

By default, commercial frontier workers (``opus``, ``gemini``, ``gpt``) use Yunwu
and open/specialist workers use OpenRouter. ``ULTRA_PROVIDER`` remains as a legacy
fallback for unknown models and explicit one-provider canaries; normal production
calls should route by model through ``routed_provider_name`` / ``routed_slug``.

Each model has a logical name; the registry maps it to the selected provider's
slug. Direct chat calls use ``base_url`` + the key in ``key_env``; the OpenCode
agentic harness uses ``oc_provider`` (the ``-m <provider>/<slug>`` prefix) plus
``oc_config`` for custom providers.
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
    "gemini":       {"openrouter": "google/gemini-3.5-flash",       "yunwu": "gemini-3.5-flash"},
    # pool-candidate (released ~2026-07-09, claims > gpt-5.5/glm-5.2; $2/$6 per M)
    "grok":         {"openrouter": "x-ai/grok-4.5", "yunwu": "grok-4.5"},
    "gpt":          {"yunwu": "gpt-5.5"},
    # gpt-5.6 family (landed 2026-07-10, yunwu; playbook [[gpt56-release-playbook]]):
    # terra = 5.5-parity at half price (training-slot economy candidate);
    # sol = opus-price, claims > opus (premium candidate, dominant-worker risk — measure first).
    "gpt-terra":    {"yunwu": "gpt-5.6-terra"},
    "gpt-sol":      {"yunwu": "gpt-5.6-sol"},
}

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_REPO_ROOT = Path(__file__).resolve().parents[2]

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

COMMERCIAL_MODELS = frozenset({"opus", "gemini", "gpt", "gpt-terra", "gpt-sol"})
DISALLOWED_MODEL_PROVIDERS = {
    "gpt": frozenset({"openrouter"}),
    "gpt-terra": frozenset({"openrouter"}),
    "gpt-sol": frozenset({"openrouter"}),
}
DEFAULT_COMMERCIAL_PROVIDER = "yunwu"
DEFAULT_SPECIALIST_PROVIDER = "openrouter"
# gemini via yunwu is 90% cheaper (user 2026-07-10). Yunwu still ignores max_tokens for it
# (re-probed same day: 3.1k tokens on a 400 cap, much milder than the old 13-19k mode), so the
# streaming wall-clock guard covers bare "gemini-*" slugs the same as gpt-*/grok-*.
FORCE_PROVIDER = {"gemini": "yunwu",
                  # grok-4.5: OpenRouter region-blocked (xAI 403) as of 2026-07-09; Yunwu carries it
                  # (unlisted but callable). Flip back to openrouter when the region opens.
                  "grok": "yunwu"}
DEFAULT_PROVIDER = "yunwu"
YUNWU_LIVE_ALLOW_ENV = "ULTRA_ALLOW_YUNWU"

ALIASES = {
    "kimi-code": "kimi",
}


def active() -> str:
    return os.environ.get("ULTRA_PROVIDER", DEFAULT_PROVIDER)


def provider(name: str | None = None) -> dict:
    selected = name or active()
    if selected not in PROVIDERS:
        raise KeyError(f"unknown provider {selected!r}; expected one of {sorted(PROVIDERS)}")
    return PROVIDERS[selected]


def provider_live_calls_allowed(name: str) -> bool:
    """Whether a provider may be used for a real outbound model call.

    Yunwu is intentionally opt-in because it does not report cost back through
    the OpenAI-compatible usage payload in our current setup; external spend
    monitoring is authoritative, so accidental large GRPO/eval runs must fail
    closed locally.
    """

    if name != "yunwu":
        return True
    value = os.environ.get(YUNWU_LIVE_ALLOW_ENV, "")
    return value.lower() in {"1", "true", "yes", "on"}


def assert_live_provider_allowed(name: str, *, model: str | None = None, context: str = "live call") -> None:
    if provider_live_calls_allowed(name):
        return
    model_text = f" for model {model!r}" if model else ""
    raise RuntimeError(
        f"Yunwu {context}{model_text} is disabled locally. "
        f"Set {YUNWU_LIVE_ALLOW_ENV}=1 only for an intentional commercial-provider run."
    )


def logical_name(model: str) -> str | None:
    """Return the canonical logical model name for a logical name or known slug."""

    if model in ALIASES:
        model = ALIASES[model]
    if model in MODELS:
        return model
    for logical, mapping in MODELS.items():
        if model in mapping.values():
            return logical
    return None


def _validate_model_provider(logical: str | None, selected: str) -> None:
    if logical is None:
        return
    disallowed = DISALLOWED_MODEL_PROVIDERS.get(logical, frozenset())
    if selected in disallowed:
        raise ValueError(f"{logical} must not be routed through {selected}")


def routed_provider_name(model: str, override: str | None = None) -> str:
    """Provider for a logical model or known provider slug.

    ``override`` is used by canaries and one-provider debugging. Without an
    override, the default split is Yunwu for commercial frontier workers and
    OpenRouter for all known open/specialist workers.
    """

    logical = logical_name(model)
    if override:
        provider(override)
        _validate_model_provider(logical, override)
        return override
    if logical in FORCE_PROVIDER:
        selected = FORCE_PROVIDER[logical]
        provider(selected)
        _validate_model_provider(logical, selected)
        return selected
    if logical in COMMERCIAL_MODELS:
        selected = os.environ.get("ULTRA_COMMERCIAL_PROVIDER", DEFAULT_COMMERCIAL_PROVIDER)
    elif logical in MODELS:
        selected = os.environ.get("ULTRA_SPECIALIST_PROVIDER", DEFAULT_SPECIALIST_PROVIDER)
    else:
        selected = active()
    provider(selected)
    _validate_model_provider(logical, selected)
    return selected


def slug(logical: str, name: str | None = None) -> str:
    """Map a logical worker name or known slug to its routed provider slug."""

    logical_model = logical_name(logical)
    if logical_model is None:
        return logical
    return MODELS[logical_model][routed_provider_name(logical_model, name)]


def routed_slug(model: str, override: str | None = None) -> str:
    """Return the model slug for the routed provider, preserving unknown slugs."""

    logical_model = logical_name(model)
    if logical_model is None:
        return model
    return MODELS[logical_model][routed_provider_name(logical_model, override)]


def routed_provider(model: str, override: str | None = None) -> dict:
    return provider(routed_provider_name(model, override))


def required_key_envs(models: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    """Provider key env vars required by a routed model set."""

    envs: list[str] = []
    seen: set[str] = set()
    for model in models:
        cfg = routed_provider(model)
        key_env = str(cfg.get("key_env") or "")
        if key_env and key_env not in seen:
            seen.add(key_env)
            envs.append(key_env)
    return envs


def load_dotenv(path: Path | None = None) -> list[str]:
    """Load missing env vars from a dotenv file; returns only loaded key names."""

    dotenv = path or (_REPO_ROOT / ".env")
    if not dotenv.exists():
        return []
    loaded: list[str] = []
    for raw in dotenv.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("'\"")
        loaded.append(key)
    return loaded


def logical_names() -> list[str]:
    return list(MODELS)
