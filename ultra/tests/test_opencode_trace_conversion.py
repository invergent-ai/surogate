import json

from ultra.schemas import AgentTrace
from ultra.traces.opencode_rollouts import convert_rollouts, validate_traces


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_convert_opencode_rollouts_to_agent_traces(tmp_path):
    src = tmp_path / "rollouts.jsonl"
    _write_jsonl(
        src,
        [
            {
                "task_id": "repo__bug",
                "arm": "direct__kimi-code",
                "stage": "direct",
                "workers": ["kimi-code"],
                "reward": 1.0,
                "cost": 0.25,
                "valid": True,
                "error": None,
                "steps": [
                    {
                        "worker_id": 0,
                        "slug": "moonshotai/kimi-k2.7-code",
                        "status": "ok",
                        "cost": 0.25,
                        "diff_len": 123,
                    }
                ],
                "elapsed_s": 12.0,
            }
        ],
    )
    out_dir = tmp_path / "traces"
    report = convert_rollouts(src, out_dir)
    assert report["n_traces"] == 1
    assert report["valid"] is True

    trace_rows = [json.loads(line) for line in (out_dir / "traces.jsonl").read_text().splitlines()]
    trace = AgentTrace(**trace_rows[0])
    assert trace.origin_harness == "opencode"
    assert trace.task_id == "repo__bug"
    assert trace.worker_model == "moonshotai/kimi-k2.7-code"
    assert trace.grade.success is True
    assert trace.usage.cost_usd == 0.25
    assert trace.events[-1].type == "test_result"

    artifact_rel = trace.events[0].content_ref.removeprefix("artifact://")
    assert (out_dir / "artifacts" / artifact_rel).exists()

    validation = validate_traces(out_dir / "traces.jsonl", artifact_root=out_dir / "artifacts")
    assert validation["missing_event_artifacts"] == 0
