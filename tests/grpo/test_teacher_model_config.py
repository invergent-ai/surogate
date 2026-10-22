from __future__ import annotations

import dataclasses
import json

import pytest

from surogate.core.config.grpo_orch_config import GRPOOrchestratorConfig
from surogate.utils.dict import DictDefault


def _config(**cfg) -> GRPOOrchestratorConfig:
    return GRPOOrchestratorConfig(DictDefault({"model": {"name": "policy"}, "env": [{"id": "test"}], **cfg}))


def _teacher(name: str, port: int, **extra) -> dict:
    return {"model": {"name": name}, "client": {"base_url": [f"http://localhost:{port}/v1"]}, **extra}


def test_teacher_model_config_serializes_only_its_own_fields() -> None:
    config = _config(
        teacher_model={
            "model": {"name": "teacher"},
            "client": {
                "base_url": ["http://localhost:8007/v1"],
                "timeout": 1200,
            },
        },
    )

    serialized = dataclasses.asdict(config)["teacher_model"]

    assert set(serialized) == {"model", "client", "rule"}
    assert serialized["model"]["name"] == "teacher"
    assert serialized["client"]["base_url"] == ["http://localhost:8007/v1"]
    assert serialized["rule"] is None


def test_an_env_teacher_takes_the_place_of_the_top_level_one() -> None:
    config = _config(
        teacher_model=_teacher("teacher", 8007),
        env=[
            {"id": "support", "teacher_model": _teacher("teacher", 8007, rule="Check the account first.")},
            {"id": "chat"},
        ],
    )
    support, chat = config.env

    assert config.teacher_for(support) is support.teacher_model
    assert config.teacher_for(support).rule == "Check the account first."
    assert config.teacher_for(chat) is config.teacher_model
    assert config.teacher_for(chat).rule is None


def test_env_teachers_survive_the_startup_dump_and_load_back_the_same() -> None:
    config = _config(
        env=[
            {"id": "support", "teacher_model": _teacher("teacher-a", 8007, rule="Check the account first.")},
            {"id": "code", "name": "code-small", "teacher_model": _teacher("teacher-b", 8008)},
        ],
    )

    # What orchestrate() writes to control/orch.yaml at startup.
    dumped = json.loads(json.dumps(dataclasses.asdict(config), default=str))

    assert dumped["teacher_model"] is None
    assert dumped["env"][0]["teacher_model"] == {
        "model": {"name": "teacher-a", "lora_adapter": None},
        "client": {
            "timeout": 1200,
            "base_url": ["http://localhost:8007/v1"],
            "api_key_var": "SUROGATE_API_KEY",
            "headers": {},
            "skip_model_check": False,
        },
        "rule": "Check the account first.",
    }
    reloaded = GRPOOrchestratorConfig(DictDefault(dumped))
    assert [reloaded.teacher_for(env) for env in reloaded.env] == [config.teacher_for(env) for env in config.env]


def test_every_env_with_its_own_teacher_needs_no_top_level_one() -> None:
    config = _config(
        env=[
            {"id": "support", "teacher_model": _teacher("teacher-a", 8007)},
            {"id": "code", "teacher_model": _teacher("teacher-b", 8008)},
        ],
    )

    assert [config.teacher_for(env).model.name for env in config.env] == ["teacher-a", "teacher-b"]


def test_no_teacher_anywhere_is_fine() -> None:
    config = _config(env=[{"id": "support"}, {"id": "code"}])

    assert [config.teacher_for(env) for env in config.env] == [None, None]


def test_envs_with_and_without_a_teacher_cannot_share_a_run() -> None:
    with pytest.raises(ValueError, match=r"\['chat'\] have no teacher"):
        _config(env=[{"id": "support", "teacher_model": _teacher("teacher", 8007)}, {"id": "chat"}])


@pytest.mark.parametrize("rule", ["", "  \n", 7])
def test_a_rule_must_be_text(rule) -> None:
    with pytest.raises(ValueError, match="teacher_model.rule"):
        _config(teacher_model=_teacher("teacher", 8007, rule=rule))
