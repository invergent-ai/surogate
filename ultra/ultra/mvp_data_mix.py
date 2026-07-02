"""Build the first MVP candidate train distribution from validated local TaskSpecs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .schemas import TaskSpec

MVP_GRPO_VERSION = "fugu_ultra_mvp_grpo_mix_v1"

MVP_SOURCE_QUOTAS: tuple[dict[str, Any], ...] = (
    {
        "lane": "repo_repair_open_repo_terminal",
        "target": 250,
        "sources": (
            {
                "source_name": "generated_repo_tasks",
                "path": "generated_repo_tasks/taskspecs.jsonl",
                "count": 16,
            },
            {
                "source_name": "tasktrove_inferredbugs",
                "path": "tasktrove_harbor/inferredbugs_train_taskspecs.jsonl",
                "count": 100,
            },
            {
                "source_name": "tasktrove_pymethods2test",
                "path": "tasktrove_harbor/pymethods2test_train_taskspecs.jsonl",
                "count": 134,
            },
        ),
        "notes": [
            "Includes generated repo repair plus verifier-backed Harbor terminal tasks.",
            "pymethods2test is prioritized because the OT-Agent ablation found it the strongest RL source.",
        ],
    },
    {
        "lane": "unit_and_scientific_code",
        "target": 225,
        "sources": (
            {
                "source_name": "existing_bank",
                "path": "data_mix/existing_bank_taskspecs.jsonl",
                "count": 225,
                "filters": {"harness": "code_exec", "domain": "code"},
            },
        ),
        "notes": ["Existing bank code-exec tasks supply fast verifier-backed code RL signal."],
    },
    {
        "lane": "math_science_knowledge",
        "target": 250,
        "sources": (
            {
                "source_name": "existing_bank",
                "path": "data_mix/existing_bank_taskspecs.jsonl",
                "count": 84,
                "filters": {"harness": "direct_qa", "domain": "math"},
            },
            {
                "source_name": "existing_bank",
                "path": "data_mix/existing_bank_taskspecs.jsonl",
                "count": 83,
                "filters": {"harness": "direct_qa", "domain": "science"},
            },
            {
                "source_name": "existing_bank",
                "path": "data_mix/existing_bank_taskspecs.jsonl",
                "count": 83,
                "filters": {"harness": "direct_qa", "domain": "general"},
            },
        ),
        "notes": ["Domain-balanced direct reasoning lane; useful curriculum signal, not Ultra proof by itself."],
    },
    {
        "lane": "tool_dialogue",
        "target": 150,
        "sources": (
            {
                "source_name": "tau_custom",
                "path": "tool_dialog_tasks/taskspecs.jsonl",
                "count": 150,
            },
        ),
        "notes": ["Custom tau-style retail, airline, and banking tasks with deterministic simulators."],
    },
    {
        "lane": "long_context_memory_planning",
        "target": 125,
        "sources": (
            {
                "source_name": "longctx_generated",
                "path": "long_context_tasks/taskspecs.jsonl",
                "count": 125,
            },
        ),
        "notes": ["Text-only document-pack memory and synthesis tasks."],
    },
)


def _read_jsonl(path: Path) -> list[TaskSpec]:
    if not path.exists():
        raise FileNotFoundError(f"missing TaskSpec shard: {path}")
    specs: list[TaskSpec] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                specs.append(TaskSpec.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid TaskSpec in {path}:{line_no}: {exc}") from exc
    return specs


def _stable_key(seed: int, lane: str, task: TaskSpec) -> str:
    payload = f"{MVP_GRPO_VERSION}:{seed}:{lane}:{task.task_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _matches_filters(task: TaskSpec, filters: dict[str, str]) -> bool:
    harness = filters.get("harness")
    if harness is not None and task.environment.harness != harness:
        return False
    domain = filters.get("domain")
    if domain is not None and task.metadata.domain != domain:
        return False
    capability = filters.get("capability")
    if capability is not None and task.capability != capability:
        return False
    return True


def _eligible(specs: list[TaskSpec], source_name: str, filters: dict[str, str]) -> list[TaskSpec]:
    return [
        spec
        for spec in specs
        if spec.source.name == source_name
        and spec.source.policy == "train_allowed"
        and spec.splitting.split == "grpo_train"
        and _matches_filters(spec, filters)
    ]


def _sample(
    specs: list[TaskSpec],
    *,
    count: int,
    seed: int,
    lane: str,
    source_name: str,
    path: Path,
) -> list[TaskSpec]:
    if len(specs) < count:
        raise ValueError(
            f"MVP lane {lane!r} needs {count} tasks from {source_name!r} at {path}, "
            f"but only {len(specs)} are eligible"
        )
    return sorted(specs, key=lambda task: _stable_key(seed, lane, task))[:count]


def _counter_json(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _summarize(tasks: list[TaskSpec]) -> dict[str, dict[str, int]]:
    return {
        "sources": _counter_json(Counter(task.source.name for task in tasks)),
        "harnesses": _counter_json(Counter(task.environment.harness for task in tasks)),
        "capabilities": _counter_json(Counter(task.capability for task in tasks)),
        "domains": _counter_json(Counter(str(task.metadata.domain) for task in tasks)),
        "splits": _counter_json(Counter(task.splitting.split for task in tasks)),
        "policies": _counter_json(Counter(task.source.policy for task in tasks)),
    }


def build_mvp_grpo_mix(
    *,
    manifest_dir: Path,
    out_jsonl: Path,
    report_out: Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Write the 1,000-row MVP candidate manifest and return its report."""

    manifest_dir = manifest_dir.resolve()
    shard_cache: dict[Path, list[TaskSpec]] = {}
    selected: list[TaskSpec] = []
    selected_ids: set[str] = set()
    lane_reports: list[dict[str, Any]] = []

    for lane_quota in MVP_SOURCE_QUOTAS:
        lane_name = str(lane_quota["lane"])
        lane_selected: list[TaskSpec] = []
        lane_source_reports: list[dict[str, Any]] = []
        for source_quota in lane_quota["sources"]:
            rel_path = Path(str(source_quota["path"]))
            path = manifest_dir / rel_path
            if path not in shard_cache:
                shard_cache[path] = _read_jsonl(path)
            filters = dict(source_quota.get("filters", {}))
            eligible = _eligible(shard_cache[path], str(source_quota["source_name"]), filters)
            sample = _sample(
                eligible,
                count=int(source_quota["count"]),
                seed=seed,
                lane=lane_name,
                source_name=str(source_quota["source_name"]),
                path=path,
            )
            lane_selected.extend(sample)
            lane_source_reports.append(
                {
                    "source": source_quota["source_name"],
                    "path": str(path),
                    "filters": filters,
                    "eligible": len(eligible),
                    "selected": len(sample),
                }
            )

        if len(lane_selected) != int(lane_quota["target"]):
            raise ValueError(
                f"MVP lane {lane_name!r} selected {len(lane_selected)} tasks, "
                f"expected {lane_quota['target']}"
            )
        for task in lane_selected:
            if task.task_id in selected_ids:
                raise ValueError(f"duplicate task selected for MVP mix: {task.task_id}")
            selected_ids.add(task.task_id)
        selected.extend(lane_selected)
        lane_reports.append(
            {
                "lane": lane_name,
                "target": lane_quota["target"],
                "selected": len(lane_selected),
                "sources": lane_source_reports,
                "counts": _summarize(lane_selected),
                "notes": lane_quota["notes"],
            }
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for task in selected:
            f.write(json.dumps(task.model_dump(mode="json"), sort_keys=True) + "\n")

    report = {
        "version": MVP_GRPO_VERSION,
        "status": "candidate_pending_fixed_workflow_discovery",
        "purpose": "fixed_workflow_discovery_and_grpo_pilot_sampling_candidate",
        "manifest_dir": str(manifest_dir),
        "out_jsonl": str(out_jsonl.resolve()),
        "seed": seed,
        "selected_total": len(selected),
        "target_total": sum(int(lane["target"]) for lane in MVP_SOURCE_QUOTAS),
        "lane_counts": {lane["lane"]: lane["selected"] for lane in lane_reports},
        "counts": _summarize(selected),
        "lanes": lane_reports,
        "notes": [
            "This is a candidate train distribution pending live fixed-workflow discovery, not a final hard-coding Ultra mix.",
            "Build the first GRPO pilot from tasks with observed workflow disagreement or headroom.",
            "Deep SWE and other final-eval-only sources are excluded.",
            "AgentTrove traces are excluded from GRPO because they are not verifier-backed TaskSpecs.",
            "The mix applies the OT-Agent source-ablation prior by including pymethods2test as a fixed RL anchor.",
        ],
        "live_calls": False,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
