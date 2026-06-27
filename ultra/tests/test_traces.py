from ultra.traces import TRACE_ADAPTERS, ClaudeCodeTraceAdapter, CodexTraceAdapter, OpenCodeTraceAdapter


def _raw(worker_model="m", task_id="t"):
    return {
        "worker_model": worker_model,
        "task_id": task_id,
        "repo": {"url": "https://example/repo", "base_commit": "abc"},
        "prompt": {"user_task": "Fix the bug"},
        "events": [{"type": "file_edit", "agent_turn": 1, "content_ref": "artifact://edit"}],
        "artifacts": {"final_patch_ref": "artifact://patch"},
        "grade": {"score": 1.0, "success": True},
        "usage": {"cost_usd": None, "wall_time_seconds": 3},
        "privacy": {"redacted": True, "license_status": "ok_for_internal_training"},
    }


def test_trace_adapters_emit_canonical_agent_trace():
    for key, cls in TRACE_ADAPTERS.items():
        trace = cls().normalize(_raw(worker_model=f"{key}-model"))
        assert trace.origin_harness == key
        assert trace.worker_model == f"{key}-model"
        assert trace.events[0].type == "file_edit"
        assert trace.artifacts.final_patch_ref == "artifact://patch"
        assert trace.grade.success is True
        assert trace.usage.cost_usd is None
        assert trace.worker_config_hash.startswith("sha256:")


def test_named_trace_adapters_have_expected_origins():
    assert OpenCodeTraceAdapter().origin_harness == "opencode"
    assert ClaudeCodeTraceAdapter().origin_harness == "claude_code"
    assert CodexTraceAdapter().origin_harness == "codex"
