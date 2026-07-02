import json

from ultra.pool_selection import (
    ALL_MODELS,
    PROPOSED_POOL,
    best_subsets,
    load_agentic_tasks,
    load_coding_tasks,
    load_joined_tasks,
    load_live_tau_tasks,
    render_report,
)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _mini_manifest(tmp_path):
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "worker_ids": [
                    "deepseek",
                    "kimi",
                    "glm",
                    "mimo",
                    "minimax",
                    "deepseek_flash",
                ]
            }
        )
    )
    _write_jsonl(
        tmp_path / "probe.jsonl",
        [
            {
                "task_id": "t0",
                "domain": "math",
                "rewards": [0, 0, 1, 1, 0, 1],
            },
            {
                "task_id": "t1",
                "domain": "code",
                "rewards": [0, 1, 0, 0, 0, 0],
            },
            {
                "task_id": "t2",
                "domain": "science",
                "rewards": [1, 0, 0, 0, 0, 0],
            },
        ],
    )
    _write_jsonl(
        tmp_path / "pool_matrix_frontier.jsonl",
        [
            {
                "task_id": "t0",
                "domain": "math",
                "worker_ids": ["opus", "gemini", "gpt", "glm"],
                "rewards": [[1, 1], [1, 0], [0, 0], [1, 0]],
                "r_bar": [1, 0.5, 0, 0.5],
            },
            {
                "task_id": "t1",
                "domain": "code",
                "worker_ids": ["opus", "gemini", "gpt", "glm"],
                "rewards": [[0, 0], [0, 0], [1, 1], [0, 0]],
                "r_bar": [0, 0, 1, 0],
            },
            {
                "task_id": "t2",
                "domain": "science",
                "worker_ids": ["opus", "gemini", "gpt", "glm"],
                "rewards": [[0, 0], [1, 1], [0, 0], [0, 0]],
                "r_bar": [0, 1, 0, 0],
            },
        ],
    )
    rows = []
    for item_id, glm, flash, mimo in [
        ("a", 1, 0, 1),
        ("b", 0, 1, 1),
        ("c", 0, 0, 0),
    ]:
        rewards = {
            "deepseek": 0,
            "kimi": 0,
            "glm": glm,
            "mimo": mimo,
            "minimax": 0,
            "deepseek_flash": flash,
        }
        for worker, reward in rewards.items():
            rows.append(
                {
                    "domain": "tau_retail",
                    "item_id": item_id,
                    "worker": worker,
                    "reward": reward,
                    "cost": 0.01,
                }
            )
    _write_jsonl(tmp_path / "agentic_bank.jsonl", rows)
    _write_jsonl(
        tmp_path / "agentic_frontier_tau4.jsonl",
        [
            {
                "domain": "tau_airline",
                "item_id": "tau-airline-0",
                "worker": "opus",
                "reward": 1,
                "cost": 0.2,
                "valid": True,
            },
            {
                "domain": "tau_airline",
                "item_id": "tau-airline-0",
                "worker": "kimi-code",
                "reward": 0,
                "cost": 0.02,
                "valid": True,
            },
            {
                "domain": "tau_retail",
                "item_id": "tau-retail-0",
                "worker": "opus",
                "reward": 0,
                "cost": 0.2,
                "valid": True,
            },
            {
                "domain": "tau_retail",
                "item_id": "tau-retail-0",
                "worker": "kimi-code",
                "reward": 1,
                "cost": 0.02,
                "valid": True,
            },
        ],
    )
    _write_jsonl(
        tmp_path / "agentic_coding_frontier_direct3.jsonl",
        [
            {
                "task_id": "repo__bug0",
                "stage": "direct",
                "workers": ["kimi-code"],
                "reward": 1,
                "cost": 0.3,
                "valid": True,
            },
            {
                "task_id": "repo__bug0",
                "stage": "direct",
                "workers": ["gpt"],
                "reward": 0,
                "cost": 0.4,
                "valid": True,
            },
            {
                "task_id": "repo__bug1",
                "stage": "direct",
                "workers": ["mimo"],
                "reward": 1,
                "cost": 0.03,
                "valid": True,
            },
            {
                "task_id": "repo__bug1",
                "stage": "direct",
                "workers": ["kimi-code"],
                "reward": 1,
                "cost": 0.3,
                "valid": True,
            },
        ],
    )
    return tmp_path


def test_joined_tasks_match_frontier_threshold_and_aliases(tmp_path):
    manifest = _mini_manifest(tmp_path)
    rows = load_joined_tasks(manifest)
    assert len(rows) == 3
    assert rows[0].scores["flash"] == 1
    assert rows[0].scores["deepseek-pro"] == 0
    assert rows[0].scores["opus"] == 1
    assert rows[0].scores["gemini"] == 1  # r_bar == 0.5 counts as success
    assert rows[1].scores["glm"] == 0  # open-bank GLM is not overwritten by frontier control GLM
    assert rows[1].scores["gpt"] == 1


def test_best_subsets_and_agentic_loading(tmp_path):
    manifest = _mini_manifest(tmp_path)
    rows = load_joined_tasks(manifest)
    best = best_subsets(rows, ALL_MODELS, max_size=2)
    assert best[1][0] == 2 / 3
    assert best[2][0] == 1.0

    agentic = load_agentic_tasks(manifest)
    assert len(agentic) == 3
    assert max(row.scores["mimo"] for row in agentic) == 1


def test_live_tau_and_coding_loading(tmp_path):
    manifest = _mini_manifest(tmp_path)
    tau = load_live_tau_tasks(manifest)
    assert len(tau) == 2
    assert tau[0].scores["opus"] == 1
    assert tau[1].scores["kimi-code"] == 1

    coding = load_coding_tasks(manifest)
    assert len(coding) == 2
    assert coding[0].scores["kimi-code"] == 1
    assert coding[1].scores["mimo"] == 1


def test_report_renders_scientific_sections(tmp_path):
    manifest = _mini_manifest(tmp_path)
    report = render_report(manifest, budget_usd=123.0)
    assert "Ultra Pool Selection Report" in report
    assert "Live Tau Frontier Shard" in report
    assert "Live Coding-Agent Shard" in report
    assert "Saved-rollout audit for commercial failures versus Kimi-Code" in report
    assert "Quality-First Ultra Decision" in report
    assert "Scaffold-Aware Coding Layer" in report
    assert "codex:gpt-5.5" in report
    assert "claude-code:opus-4.8" in report
    assert "Diagnostic Role-Weighted Table" in report
    assert "Diagnostic Equal-Stratum Table" in report
    assert "Coding-Focused Ablation" in report
    assert "Current Scientific Conclusion" in report
    assert "Final proposed quality-first core" in report
    assert "opus+gemini+gpt+kimi-code+mimo+glm+flash" in report
    assert "not empty-diff or tool-calling failures" in report
    assert "coding-ablation/open-six" in report
    assert "quality-first/core-seven" in report
    assert "Preregistered Low-Spend Paid Test" in report
    for model in PROPOSED_POOL:
        assert model in report
