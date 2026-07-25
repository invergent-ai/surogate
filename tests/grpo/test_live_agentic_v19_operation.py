from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "environments/fugu-live-agentic-grpo"
for path in (ROOT, ROOT / "director", ROOT / "ultra", ENV_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fugu_live_agentic_grpo as env_mod  # noqa: E402
from scratchpad import audit_fugu_live_agentic_grpo_v19 as audit  # noqa: E402
from scratchpad import preflight_fugu_live_agentic_grpo_v19 as preflight  # noqa: E402
from scratchpad import run_fugu_live_agentic_grpo_v19 as runner  # noqa: E402


def test_v19_is_one_single_use_two_sample_group_seeded_from_v10() -> None:
    orch = yaml.safe_load(runner.ORCH_CONFIG.read_text(encoding="utf-8"))
    train = yaml.safe_load(runner.TRAIN_CONFIG.read_text(encoding="utf-8"))
    env = orch["env"][0]
    args = env["args"]

    assert orch["batch_size"] == orch["rollouts_per_example"] == 2
    assert orch["max_steps"] == train["max_steps"] == 1
    assert orch["max_concurrent"] == orch["max_inflight_rollouts"] == 2
    assert orch["max_async_level"] == orch["max_off_policy_steps"] == 0
    assert env["max_retries"] == 0
    assert args["provider_mode"] == "live"
    assert args["allow_yunwu_live"] is True
    assert args["max_parallel_sessions"] == 1
    assert args["worker_timeout_s"] == 600.0
    assert args["max_control_decisions"] == env_mod.MAX_CONTROL_DECISIONS
    assert Path(train["adapter_path"]).resolve() == runner.SEED.resolve()
    assert train["resume_from_checkpoint"] is False
    assert train["loss"]["teacher_tau"] > 0
    assert train["loss"]["kl_tau"] > 0


def test_v19_manifest_is_fresh_oracle_backed_and_exact() -> None:
    rows = env_mod._read_manifest(preflight.TASK_MANIFEST)

    assert len(rows) == 1
    assert rows[0]["task_id"] == preflight.EXPECTED_TASK_ID
    assert rows[0]["conductor_attempted_before"] is False
    assert rows[0]["oracle_reward"] == 1.0


def test_v19_preflight_artifacts_attest_zero_provider_preparation() -> None:
    process = json.loads(preflight.PROCESS_PREFLIGHT.read_text(encoding="utf-8"))
    container = json.loads(preflight.CONTAINER_PREFLIGHT.read_text(encoding="utf-8"))
    metadata = container["agent_result"]["metadata"]

    assert process["passed"] is True
    assert process["external_calls"] == process["paid_calls"] == 0
    assert metadata["workspace_snapshot_preflight_provider_calls"] == 0
    assert metadata["workspace_snapshot_preflight_paid_calls"] == 0
    assert metadata["paid_worker_call_attempts"] == 0
    assert metadata["fugu_routes"] == []
    assert metadata["workspace_root"] == "/testbed"
    assert metadata["protected_test_snapshot_entries"] > 0
    # V19 is closed and single-use; its frozen artifacts attest the exact
    # revision that operation ran, independent of the current runtime.
    assert metadata["runtime_revision"] == "20260717-r48-dynamic-workspace-root"


def test_v19_paths_are_disjoint_from_consumed_v18_operation() -> None:
    assert "v19" in str(runner.STATE)
    assert "v19" in str(runner.LOCK)
    assert "v19" in str(audit.ARTIFACT_ROOT)
    assert "v19" in str(audit.BATCH_PATH)
    assert runner.STATE != ROOT / "scratchpad/fugu_live_agentic_grpo_v18_state.json"
