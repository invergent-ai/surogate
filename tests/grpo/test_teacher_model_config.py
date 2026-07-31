from __future__ import annotations

import dataclasses

from surogate.core.config.grpo_orch_config import GRPOOrchestratorConfig
from surogate.utils.dict import DictDefault


def test_teacher_model_config_serializes_only_nested_model_and_client() -> None:
    config = GRPOOrchestratorConfig(
        DictDefault(
            {
                "model": {"name": "policy"},
                "teacher_model": {
                    "model": {"name": "teacher"},
                    "client": {
                        "base_url": ["http://localhost:8007/v1"],
                        "timeout": 1200,
                    },
                },
                "env": [{"id": "test"}],
            }
        )
    )

    serialized = dataclasses.asdict(config)["teacher_model"]

    assert set(serialized) == {"model", "client"}
    assert serialized["model"]["name"] == "teacher"
    assert serialized["client"]["base_url"] == ["http://localhost:8007/v1"]
