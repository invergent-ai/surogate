import argparse
import json

import pytest

from ultra.pool_tournament import (
    DEFAULT_WORKERS,
    MODEL_SLUGS,
    RolloutSummary,
    estimate_rollout_count,
    preregistered_arms,
    sample_balanced_tasks,
    run_tournament,
    select_subsets_from_records,
)
from ultra.workflow import validate_workflow


def test_model_slugs_are_catalog_backed_candidates():
    assert MODEL_SLUGS["opus"] == "anthropic/claude-opus-4.8"
    assert MODEL_SLUGS["gemini"] == "google/gemini-3.1-pro-preview"
    assert MODEL_SLUGS["gpt"] == "openai/gpt-5.5"
    assert MODEL_SLUGS["glm"] == "z-ai/glm-5.2"
    assert MODEL_SLUGS["flash"] == "deepseek/deepseek-v4-flash"
    assert MODEL_SLUGS["deepseek-pro"] == "deepseek/deepseek-v4-pro"
    assert MODEL_SLUGS["kimi-code"] == "moonshotai/kimi-k2.7-code"
    assert MODEL_SLUGS["mimo"] == "xiaomi/mimo-v2.5-pro"
    assert MODEL_SLUGS["minimax"] == "minimax/minimax-m3"


def test_preregistered_arms_are_valid_workflows():
    index = {name: i for i, name in enumerate(DEFAULT_WORKERS)}
    arms = preregistered_arms()
    assert {a.stage for a in arms} == {"single", "same_worker", "mixed", "challenger"}
    assert any(a.name == "solve__glm__critic__mimo__revise__glm" for a in arms)
    for arm in arms:
        wf = arm.build(index)
        validate_workflow(wf, worker_count=len(DEFAULT_WORKERS))


def test_balanced_sampling_and_rollout_count(tmp_path):
    rows = []
    for domain in ["math", "code", "science", "general"]:
        for i in range(3):
            rows.append(
                {
                    "task_id": f"{domain}-{i}",
                    "domain": domain,
                    "source": domain,
                    "prompt": "Q",
                    "solution": "A",
                    "grader": "mc_letter",
                    "system": "",
                    "split": "test",
                    "verdict": "discriminative",
                }
            )
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    tasks = sample_balanced_tasks(
        path, split="online_validation", tasks_per_domain=2, seed=0, verdict="discriminative"
    )
    assert len(tasks) == 8
    assert {t.metadata.domain for t in tasks} == {"math", "code", "science", "general"}
    rollouts, calls = estimate_rollout_count(tasks, preregistered_arms())
    assert rollouts > len(tasks)
    assert calls > rollouts


def test_balanced_sampling_can_filter_by_open_success_count(tmp_path):
    rows = []
    for domain in ["math", "code", "science", "general"]:
        for i, rewards in enumerate(([0, 0, 0], [1, 0, 0], [1, 1, 0])):
            rows.append(
                {
                    "task_id": f"{domain}-{i}",
                    "domain": domain,
                    "source": domain,
                    "prompt": "Q",
                    "solution": "A",
                    "grader": "mc_letter",
                    "system": "",
                    "split": "test",
                    "verdict": "discriminative",
                    "rewards": rewards,
                }
            )
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    tasks = sample_balanced_tasks(
        path,
        split="online_validation",
        tasks_per_domain=2,
        seed=0,
        verdict="discriminative",
        open_success_min=1,
        open_success_max=1,
    )
    assert len(tasks) == 4
    assert {t.metadata.domain for t in tasks} == {"math", "code", "science", "general"}
    assert {t.task_id.removeprefix("existing_bank__").split("-")[-1] for t in tasks} == {"1"}


def test_subset_selection_scores_worker_mixes():
    records = [
        RolloutSummary("direct__opus", "t0", "code", True, 1.0, 0.01, True, None),
        RolloutSummary("direct__gemini", "t0", "code", False, 0.5, 0.01, True, None),
        RolloutSummary("direct__glm", "t0", "code", False, 0.5, 0.01, True, None),
        RolloutSummary("direct__opus", "t1", "math", False, 0.5, 0.01, True, None),
        RolloutSummary("direct__gemini", "t1", "math", True, 1.0, 0.01, True, None),
        RolloutSummary("direct__glm", "t1", "math", False, 0.5, 0.01, True, None),
        RolloutSummary("debate__flash__glm__synth__opus", "t2", "science", True, 1.0, 0.03, True, None),
        RolloutSummary("direct__opus", "t2", "science", False, 0.5, 0.01, True, None),
        RolloutSummary("direct__gemini", "t2", "science", False, 0.5, 0.01, True, None),
        RolloutSummary("direct__glm", "t2", "science", False, 0.5, 0.01, True, None),
        RolloutSummary("direct__gpt", "t0", "code", False, 0.5, 0.01, True, None),
        RolloutSummary("direct__gpt", "t1", "math", False, 0.5, 0.01, True, None),
        RolloutSummary("direct__gpt", "t2", "science", False, 0.5, 0.01, True, None),
    ]
    selection = select_subsets_from_records(records, max_size=4)
    assert selection["best_by_size"]["1"]["score"] == 1 / 3
    assert selection["best_by_size"]["2"]["score"] == 2 / 3
    assert selection["best_by_size"]["3"]["score"] == 2 / 3
    assert selection["best_by_size"]["4"]["score"] == 1.0
    assert selection["proposed_leave_one_out"]["gpt"]["delta_kept"] == 0.0
    assert selection["proposed_leave_one_out"]["opus"]["delta_kept"] > 0


@pytest.mark.asyncio
async def test_dry_run_does_not_require_openrouter_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    rows = []
    for domain in ["math", "code", "science", "general"]:
        rows.append(
            {
                "task_id": f"{domain}-0",
                "domain": domain,
                "source": domain,
                "prompt": "Q",
                "solution": "A",
                "grader": "mc_letter",
                "system": "",
                "split": "test",
                "verdict": "discriminative",
            }
        )
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    args = argparse.Namespace(
        manifest_path=str(path),
        split="online_validation",
        tasks_per_domain=1,
        all_difficulties=False,
        stages="single",
        arms=None,
        budget=200.0,
        stop_ratio=0.8,
        concurrency=1,
        max_tokens=128,
        temperature=0.2,
        reasoning="high",
        timeout=10.0,
        max_retries=0,
        seed=0,
        cache_dir=str(tmp_path / "cache"),
        out_dir=str(tmp_path / "out"),
        resume=False,
        dry_run=True,
    )
    result = await run_tournament(args)
    assert result["summary"] is None
    assert result["plan"]["tasks"] == 4
    assert result["plan"]["spend_stop_usd"] == 160.0
