from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from ultra.live_worker_safety import LiveWorkerSafetyError, validate_live_worker_safety


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "environments" / "fugu-ultra-pilot"
if str(ENV_PATH) not in sys.path:
    sys.path.insert(0, str(ENV_PATH))

import fugu_ultra_pilot as env_mod  # noqa: E402


PILOT_CONFIG = (
    ROOT
    / "director/manifests/fugu_clean_v1/grpo_pilot_train/pilot_config_tight_repo_tool_after_parent_repair_sft.json"
)
SAFETY = (
    ROOT
    / "director/manifests/fugu_clean_v1/grpo_pilot_train/live_safety_parent_repair_tight_repo_tool.json"
)


def _pilot_config():
    return env_mod._read_json(PILOT_CONFIG)


def test_parent_repair_live_safety_allows_prepared_repo_lane():
    report = validate_live_worker_safety(
        pilot_config=_pilot_config(),
        lane="repo_open_repo_terminal",
        provider_mode="live",
        allow_yunwu_live=True,
        live_safety_path=SAFETY,
        max_examples=12,
        force_step_budget="long",
    )

    assert report["enforced"] is True
    assert report["lane"] == "repo_open_repo_terminal"
    assert {row["provider"] for row in report["routes"]} == {"openrouter"}


def test_parent_repair_live_safety_allows_prepared_tool_lane_with_opus():
    report = validate_live_worker_safety(
        pilot_config=_pilot_config(),
        lane="tool_dialogue",
        provider_mode="live",
        allow_yunwu_live=True,
        live_safety_path=SAFETY,
        max_examples=16,
        force_step_budget="long",
    )

    assert report["enforced"] is True
    assert {row["worker"] for row in report["routes"] if row["provider"] == "yunwu"} == {
        "direct_opus_reviewer"
    }


def test_live_safety_rejects_live_mode_without_manifest():
    with pytest.raises(LiveWorkerSafetyError, match="require live_safety_path"):
        env_mod.load_environment(
            provider_mode="live",
            lane="repo_open_repo_terminal",
            max_examples=1,
            artifact_dir="",
        )


def test_live_safety_rejects_unapproved_lane():
    with pytest.raises(LiveWorkerSafetyError, match="not allowed"):
        validate_live_worker_safety(
            pilot_config=_pilot_config(),
            lane="math_science_knowledge",
            provider_mode="live",
            allow_yunwu_live=True,
            live_safety_path=SAFETY,
            max_examples=1,
            force_step_budget="long",
        )


def test_live_safety_rejects_missing_short_budget():
    with pytest.raises(LiveWorkerSafetyError, match="force_step_budget"):
        validate_live_worker_safety(
            pilot_config=_pilot_config(),
            lane="repo_open_repo_terminal",
            provider_mode="live",
            allow_yunwu_live=True,
            live_safety_path=SAFETY,
            max_examples=12,
            force_step_budget=None,
        )


def test_live_safety_rejects_forbidden_worker_drift():
    config = copy.deepcopy(_pilot_config())
    config["lane_worker_masks"]["repo_open_repo_terminal"].append("terminal_gpt_agent")

    with pytest.raises(LiveWorkerSafetyError, match="not allowed|forbidden"):
        validate_live_worker_safety(
            pilot_config=config,
            lane="repo_open_repo_terminal",
            provider_mode="live",
            allow_yunwu_live=True,
            live_safety_path=SAFETY,
            max_examples=12,
            force_step_budget="long",
        )


def test_live_safety_rejects_yunwu_worker_without_yunwu_opt_in():
    with pytest.raises(LiveWorkerSafetyError, match="allow_yunwu_live is false"):
        validate_live_worker_safety(
            pilot_config=_pilot_config(),
            lane="tool_dialogue",
            provider_mode="live",
            allow_yunwu_live=False,
            live_safety_path=SAFETY,
            max_examples=16,
            force_step_budget="long",
        )
