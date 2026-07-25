from pathlib import Path

import pytest

from ultra.providers import (
    YUNWU_LIVE_ALLOW_ENV,
    assert_live_provider_allowed,
    load_dotenv,
    required_key_envs,
    routed_provider_name,
    routed_slug,
    slug,
)


def test_commercial_models_route_to_yunwu_by_default(monkeypatch):
    monkeypatch.delenv("ULTRA_PROVIDER", raising=False)

    assert routed_provider_name("opus") == "yunwu"
    assert routed_slug("opus") == "claude-opus-4-8"
    assert routed_provider_name("gemini") == "yunwu"
    assert routed_slug("gemini") == "gemini-3.5-flash"
    assert routed_provider_name("gpt") == "yunwu"
    assert slug("gpt") == "gpt-5.5"


def test_open_specialist_models_route_to_openrouter_by_default(monkeypatch):
    monkeypatch.setenv("ULTRA_PROVIDER", "yunwu")

    assert routed_provider_name("glm") == "openrouter"
    assert routed_slug("glm") == "z-ai/glm-5.2"
    assert routed_provider_name("kimi-code") == "openrouter"
    assert routed_slug("kimi-code") == "moonshotai/kimi-k2.7-code"
    assert routed_provider_name("mimo") == "openrouter"
    assert routed_slug("mimo") == "xiaomi/mimo-v2.5-pro"
    assert routed_provider_name("flash") == "openrouter"
    assert routed_provider_name("minimax") == "openrouter"
    assert routed_provider_name("deepseek-pro") == "openrouter"


def test_explicit_provider_override_for_canaries():
    assert routed_provider_name("glm", "yunwu") == "yunwu"
    assert routed_slug("glm", "yunwu") == "glm-5.2"


def test_gpt_never_routes_to_openrouter(monkeypatch):
    with pytest.raises(ValueError, match="gpt must not be routed through openrouter"):
        routed_provider_name("gpt", "openrouter")

    monkeypatch.setenv("ULTRA_COMMERCIAL_PROVIDER", "openrouter")
    with pytest.raises(ValueError, match="gpt must not be routed through openrouter"):
        routed_provider_name("gpt")


def test_yunwu_live_calls_require_explicit_opt_in(monkeypatch):
    monkeypatch.delenv(YUNWU_LIVE_ALLOW_ENV, raising=False)

    with pytest.raises(RuntimeError, match="Yunwu direct worker call"):
        assert_live_provider_allowed("yunwu", model="gpt", context="direct worker call")

    monkeypatch.setenv(YUNWU_LIVE_ALLOW_ENV, "1")
    assert_live_provider_allowed("yunwu", model="gpt", context="direct worker call")
    assert_live_provider_allowed("openrouter", model="glm", context="direct worker call")


def test_required_key_envs_follow_routed_model_mix():
    assert required_key_envs(["gpt", "opus", "glm", "mimo"]) == [
        "YUNWU_API_KEY",
        "OPENROUTER_API_KEY",
    ]


def test_load_dotenv_reads_missing_keys_without_overriding(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENROUTER_API_KEY=from-file\nYUNWU_API_KEY=from-file\n")
    monkeypatch.setenv("YUNWU_API_KEY", "already-set")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    loaded = load_dotenv(Path(dotenv))

    assert loaded == ["OPENROUTER_API_KEY"]
    assert "OPENROUTER_API_KEY" in loaded
    assert "YUNWU_API_KEY" not in loaded
