import json

from ultra.workflow_sft_warmstart import build_workflow_sft_warmstart


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _task(task_id, lane, harness="direct_qa", capability="math"):
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": capability,
        "source": {"name": "unit_test", "version": "v1", "policy": "train_allowed"},
        "input": {"messages": [{"role": "user", "content": f"Solve {task_id}."}]},
        "environment": {"harness": harness},
        "grader": {"type": "exact_match", "expected_answer": "ok"},
        "splitting": {"group_id": lane, "split": "grpo_train"},
    }


def _commercial_row(task_id="task_commercial"):
    workflow = {
        "steps": [
            {
                "worker_id": 0,
                "subtask": "Solve the problem.",
                "access": [],
                "budget": "medium",
            }
        ]
    }
    return {
        "record_id": "commercial::1",
        "task_id": task_id,
        "lane": "math_science_knowledge",
        "source": "unit_test",
        "arm": "solo__direct_gpt_reasoner",
        "allowed_workers": [
            {
                "worker_id": 0,
                "name": "direct_gpt_reasoner",
                "backend": "direct_qa",
                "model": "gpt-5.5",
                "role_prior": ["planner"],
            }
        ],
        "workflow": workflow,
        "messages": [
            {"role": "system", "content": "You are the Fugu-Ultra Conductor."},
            {"role": "user", "content": "Allowed workers:\n0: direct_gpt_reasoner\nTask prompt:\nSolve."},
            {"role": "assistant", "content": json.dumps(workflow, sort_keys=True)},
        ],
    }


def test_workflow_sft_warmstart_merges_commercial_and_topology_rows(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest"
    commercial = manifest / "commercial_workflow_sft.jsonl"
    tasks = manifest / "taskspecs.jsonl"
    pilot_config = manifest / "pilot_config.json"
    out_dir = manifest / "workflow_sft_warmstart"

    _write_jsonl(commercial, [_commercial_row()])
    task_rows = [
        _task("task_math", "math_science_knowledge", "direct_qa", "math"),
        _task("task_unit", "unit_and_scientific_code", "code_exec", "unit_code"),
        _task("task_tool", "tool_dialogue", "tau_bench", "tool_dialogue"),
        _task("task_long", "long_context_memory_planning", "long_context", "long_context"),
        _task("task_repo", "trace_state_branches", "opencode", "agentic_coding"),
        _task("task_terminal", "repo_open_repo_terminal", "terminal_sandbox", "agentic_coding"),
    ]
    _write_jsonl(tasks, task_rows)
    pilot_config.write_text(
        json.dumps(
            {
                "task_ids_by_lane": {
                    "math_science_knowledge": ["task_math"],
                    "unit_and_scientific_code": ["task_unit"],
                    "tool_dialogue": ["task_tool"],
                    "long_context_memory_planning": ["task_long"],
                    "trace_state_branches": ["task_repo"],
                    "repo_open_repo_terminal": ["task_terminal"],
                }
            }
        ),
        encoding="utf-8",
    )

    import ultra.workflow_sft_warmstart as module

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    report = build_workflow_sft_warmstart(
        commercial_sft_jsonl=commercial,
        tasks_jsonl=tasks,
        pilot_config_json=pilot_config,
        out_dir=out_dir,
        examples_per_arm=1,
    )

    assert report["counts"]["commercial_input_rows"] == 1
    assert report["counts"]["topology_input_rows"] == 58
    assert report["counts"]["total_rows"] == 59
    assert report["source_kind_counts"] == {"commercial_success": 1, "topology_prior": 58}

    rows = [json.loads(line) for line in (out_dir / "workflow_sft_warmstart.jsonl").read_text().splitlines()]
    assert rows[0]["source_kind"] == "commercial_success"
    assert any(row["source_kind"] == "topology_prior" for row in rows)
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)
    assert all(json.loads(row["messages"][-1]["content"]) == row["workflow"] for row in rows)

    config = (out_dir / "train_workflow_sft_qwen3_8b.yaml").read_text()
    assert "type: conversation" in config
    assert "messages_field: messages" in config
    assert str((out_dir / "workflow_sft_warmstart.jsonl").resolve()) in config
    assert "template: qwen3_nothinking" in config
    assert "cpu_training: true" in config
    assert "merge_adapter: true" in config


def test_workflow_sft_warmstart_dedupes_identical_messages(tmp_path):
    manifest = tmp_path / "manifest"
    commercial = manifest / "commercial_workflow_sft.jsonl"
    tasks = manifest / "taskspecs.jsonl"
    pilot_config = manifest / "pilot_config.json"

    row = _commercial_row()
    _write_jsonl(commercial, [row, row])
    _write_jsonl(tasks, [])
    pilot_config.write_text(json.dumps({"task_ids_by_lane": {}}), encoding="utf-8")

    report = build_workflow_sft_warmstart(
        commercial_sft_jsonl=commercial,
        tasks_jsonl=tasks,
        pilot_config_json=pilot_config,
        out_dir=manifest / "workflow_sft_warmstart",
    )

    assert report["counts"]["commercial_input_rows"] == 2
    assert report["counts"]["dedupe_removed"] == 1
    assert report["counts"]["total_rows"] == 1
