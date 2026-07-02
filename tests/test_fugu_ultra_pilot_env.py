from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "environments" / "fugu-ultra-pilot"
ULTRA_PATH = ROOT / "ultra"
for path in (str(ENV_PATH), str(ULTRA_PATH)):
    if path not in sys.path:
        sys.path.insert(0, path)

import fugu_ultra_pilot as env_mod  # noqa: E402


def test_loads_frozen_pilot_manifest_as_verifiers_env():
    env = env_mod.load_environment(provider_mode="fake", max_examples=3, artifact_dir="")

    assert len(env.dataset) == 3
    assert env.dataset.column_names == ["example_id", "task", "prompt", "answer", "info"]
    assert env.dataset[0]["task"] == "fugu_ultra_pilot"
    assert "Allowed workers:" in env.dataset[0]["prompt"][1]["content"]
    assert "expected_answer" not in env.dataset[0]["prompt"][1]["content"]


def test_lane_filter_uses_lane_local_worker_ids():
    env = env_mod.load_environment(
        provider_mode="fake",
        lane="trace_state_branches",
        max_examples=1,
        artifact_dir="",
    )

    prompt = env.dataset[0]["prompt"][1]["content"]
    assert "Lane: trace_state_branches" in prompt
    assert "0: codex_gpt_coding_agent" in prompt
    assert "1: claude_code_opus_debugger" in prompt
    assert "2: opencode_kimi_builder" in prompt


def test_prompt_reflects_one_step_validation_config(tmp_path):
    config = env_mod._read_json(env_mod.DEFAULT_PILOT_CONFIG)
    config["workflow_policy"]["max_workflow_steps"] = 1
    out = tmp_path / "pilot_config_one_step.json"
    out.write_text(json.dumps(config), encoding="utf-8")

    env = env_mod.load_environment(
        provider_mode="fake",
        pilot_config_path=str(out),
        lane="repo_open_repo_terminal",
        max_examples=1,
        artifact_dir="",
    )

    assert "Use exactly 1 step." in env.dataset[0]["prompt"][0]["content"]
    assert "Use 1 to 3 steps." not in env.dataset[0]["prompt"][0]["content"]


def test_tool_dialog_workers_do_not_override_tau_bench_harness():
    config = env_mod._read_json(env_mod.DEFAULT_PILOT_CONFIG)
    overrides = env_mod._worker_harness_overrides(config)

    assert "tool_dialog_mimo_agent" not in overrides
    assert "tool_dialog_glm_agent" not in overrides
    assert overrides["terminal_kimi_agent"] == "terminal_sandbox"


def test_allow_yunwu_live_sets_explicit_opt_in(monkeypatch):
    monkeypatch.delenv(env_mod.YUNWU_LIVE_ALLOW_ENV, raising=False)

    env = env_mod.load_environment(
        provider_mode="fake",
        max_examples=1,
        artifact_dir="",
        allow_yunwu_live=True,
    )

    assert len(env.dataset) == 1
    assert env_mod.os.environ[env_mod.YUNWU_LIVE_ALLOW_ENV] == "1"


def test_harbor_timeout_setter_updates_process_env(monkeypatch):
    monkeypatch.delenv("ULTRA_HARBOR_TIMEOUT_SECONDS", raising=False)
    env = env_mod.load_environment(provider_mode="fake", max_examples=1, artifact_dir="")

    env.set_kwargs(harbor_timeout_s=17)

    assert env.harbor_timeout_s == 17
    assert env_mod.os.environ["ULTRA_HARBOR_TIMEOUT_SECONDS"] == "17"

    env.set_kwargs(harbor_timeout_s=None)

    assert "ULTRA_HARBOR_TIMEOUT_SECONDS" not in env_mod.os.environ


@pytest.mark.asyncio
async def test_invalid_workflow_scores_zero_without_worker_execution():
    env = env_mod.load_environment(provider_mode="fake", max_examples=1, artifact_dir="")
    runtime = env.ultra_runtime
    state = {}
    info = env_mod._info_dict(env.dataset[0]["info"])

    reward = await runtime.score("not json", info, state)

    assert reward == 0.0
    assert state["_ultra_outcome_class"] == "invalid_workflow_trainable"
    assert state["_ultra_valid_for_training"] is True
    assert state["_ultra_workflow_parse_valid"] is False


@pytest.mark.asyncio
async def test_valid_workflow_scores_through_executor_in_fake_mode():
    env = env_mod.load_environment(
        provider_mode="fake",
        lane="math_science_knowledge",
        max_examples=1,
        artifact_dir="",
    )
    runtime = env.ultra_runtime
    state = {}
    info = env_mod._info_dict(env.dataset[0]["info"])

    reward = await runtime.score(
        '{"steps":[{"worker_id":0,"subtask":"solve directly","access":[],"budget":"short"}]}',
        info,
        state,
    )

    assert reward in {0.5, 1.0}
    assert state["_ultra_workflow_parse_valid"] is True
    assert "_ultra_record" in state
    assert state["_ultra_record"]["workflow"]["steps"][0]["worker_id"] == 0


@pytest.mark.asyncio
async def test_force_step_budget_overrides_workflow_budget_for_validation():
    env = env_mod.load_environment(
        provider_mode="fake",
        lane="math_science_knowledge",
        max_examples=1,
        artifact_dir="",
        force_step_budget="short",
    )
    runtime = env.ultra_runtime
    state = {}
    info = env_mod._info_dict(env.dataset[0]["info"])

    reward = await runtime.score(
        '{"steps":[{"worker_id":0,"subtask":"solve directly","access":[],"budget":"max"}]}',
        info,
        state,
    )

    assert reward in {0.5, 1.0}
    assert state["_ultra_record"]["workflow"]["steps"][0]["budget"] == "short"
