"""Preregistered scaffold-aware tournament plan for Fugu-Ultra.

This module does not call providers. It defines the worker configurations and fixed
workflow arms needed before running a quality-first, scaffold-aware role tournament.
"""

from __future__ import annotations

import argparse
import json
import random
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import ToolPermissions, WorkerIdentity, Workflow, WorkflowStep
from .workflow import validate_workflow

DEFAULT_TASK_MIX = {
    "coding_repo": 15,
    "tool_dialog": 10,
    "direct_reasoning": 12,
}

BACKEND_TO_HARNESS = {
    "opencode": "opencode",
    "claude_code": "claude_code",
    "codex": "codex",
    "direct_qa": "direct_qa",
    "tool_dialog": "tool_dialog",
    "terminal": "terminal_sandbox",
}


@dataclass(frozen=True)
class ScaffoldArm:
    name: str
    domain: str
    stage: str
    worker_names: tuple[str, ...]
    workflow: Workflow
    rationale: str

    @property
    def worker_calls(self) -> int:
        return len(self.workflow.steps)

    def model_dump(self, worker_name_by_id: dict[int, str]) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "stage": self.stage,
            "worker_names": list(self.worker_names),
            "worker_calls": self.worker_calls,
            "rationale": self.rationale,
            "workflow": {
                "steps": [
                    {
                        "worker_id": step.worker_id,
                        "worker_name": worker_name_by_id[step.worker_id],
                        "subtask": step.subtask,
                        "access": step.access,
                    }
                    for step in self.workflow.steps
                ]
            },
        }


def canonical_workers() -> list[WorkerIdentity]:
    """Quality-first model pool expanded into scaffold-aware worker identities."""

    return [
        WorkerIdentity(
            worker_id=0,
            name="codex_gpt_coding_agent",
            backend="codex",
            model="gpt-5-codex",
            role_prior=["planner", "builder", "repair"],
            max_turns=100,
            max_reported_cost_usd=2.0,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=1,
            name="claude_code_opus_debugger",
            backend="claude_code",
            model="claude-opus-4.8",
            role_prior=["debugger", "verifier", "security_review", "repair"],
            max_turns=100,
            max_reported_cost_usd=2.0,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=2,
            name="opencode_kimi_builder",
            backend="opencode",
            model="moonshotai/kimi-k2.7-code",
            role_prior=["builder", "implementation", "repair"],
            max_turns=100,
            max_reported_cost_usd=1.0,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=3,
            name="opencode_mimo_repair",
            backend="opencode",
            model="xiaomi/mimo-v2.5-pro",
            role_prior=["agentic_executor", "independent_attempt", "repair"],
            max_turns=100,
            max_reported_cost_usd=1.0,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=4,
            name="opencode_glm_builder",
            backend="opencode",
            model="z-ai/glm-5.2",
            role_prior=["open_generalist", "secondary_builder", "debugger"],
            max_turns=100,
            max_reported_cost_usd=1.0,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=5,
            name="tool_dialog_mimo_agent",
            backend="tool_dialog",
            model="xiaomi/mimo-v2.5-pro",
            role_prior=["tool_dialog", "procedural_task_solver", "repair"],
            max_turns=80,
            max_reported_cost_usd=0.75,
        ),
        WorkerIdentity(
            worker_id=6,
            name="tool_dialog_glm_agent",
            backend="tool_dialog",
            model="z-ai/glm-5.2",
            role_prior=["tool_dialog", "structured_reasoning"],
            max_turns=80,
            max_reported_cost_usd=0.75,
        ),
        WorkerIdentity(
            worker_id=7,
            name="tool_dialog_flash_agent",
            backend="tool_dialog",
            model="deepseek/deepseek-v4-flash",
            role_prior=["tool_dialog", "fast_first_pass"],
            max_turns=80,
            max_reported_cost_usd=0.50,
        ),
        WorkerIdentity(
            worker_id=8,
            name="direct_gemini_synth",
            backend="direct_qa",
            model="google/gemini-3.1-pro-preview",
            role_prior=["science", "factual_synthesis", "long_context"],
            max_reported_cost_usd=0.50,
        ),
        WorkerIdentity(
            worker_id=9,
            name="direct_gpt_reasoner",
            backend="direct_qa",
            model="openai/gpt-5.5",
            role_prior=["planner", "math", "algorithm_design", "alternate_perspective"],
            max_reported_cost_usd=0.75,
        ),
        WorkerIdentity(
            worker_id=10,
            name="direct_opus_reviewer",
            backend="direct_qa",
            model="anthropic/claude-opus-4.8",
            role_prior=["reviewer", "debugger", "verifier", "final_quality_control"],
            max_reported_cost_usd=0.75,
        ),
        WorkerIdentity(
            worker_id=11,
            name="direct_flash_fast",
            backend="direct_qa",
            model="deepseek/deepseek-v4-flash",
            role_prior=["fast_first_pass", "easy_direct_subtasks"],
            max_reported_cost_usd=0.25,
        ),
    ]


def worker_harness_map(workers: list[WorkerIdentity] | None = None) -> dict[str, str]:
    workers = workers or canonical_workers()
    return {worker.name: BACKEND_TO_HARNESS[worker.backend] for worker in workers}


def _index(workers: list[WorkerIdentity]) -> dict[str, int]:
    return {worker.name: worker.worker_id for worker in workers}


def _step(index: dict[str, int], worker: str, subtask: str, access: list[int] | None = None) -> WorkflowStep:
    return WorkflowStep(worker_id=index[worker], subtask=subtask, access=access or [])


def _workflow(*steps: WorkflowStep) -> Workflow:
    return Workflow(steps=list(steps))


def canonical_arms(workers: list[WorkerIdentity] | None = None) -> list[ScaffoldArm]:
    workers = workers or canonical_workers()
    ix = _index(workers)

    coding = "coding_repo"
    tool = "tool_dialog"
    direct = "direct_reasoning"

    arms = [
        ScaffoldArm(
            name="solo__codex_gpt_coding_agent",
            domain=coding,
            stage="single_scaffold",
            worker_names=("codex_gpt_coding_agent",),
            workflow=_workflow(
                _step(ix, "codex_gpt_coding_agent", "Implement the repository fix and run the relevant tests.")
            ),
            rationale="Fair solo baseline for the Codex/GPT coding scaffold.",
        ),
        ScaffoldArm(
            name="solo__claude_code_opus_debugger",
            domain=coding,
            stage="single_scaffold",
            worker_names=("claude_code_opus_debugger",),
            workflow=_workflow(
                _step(ix, "claude_code_opus_debugger", "Implement the repository fix and run the relevant tests.")
            ),
            rationale="Fair solo baseline for the Claude Code/Opus scaffold.",
        ),
        ScaffoldArm(
            name="solo__opencode_kimi_builder",
            domain=coding,
            stage="single_scaffold",
            worker_names=("opencode_kimi_builder",),
            workflow=_workflow(
                _step(ix, "opencode_kimi_builder", "Implement the repository fix and run the relevant tests.")
            ),
            rationale="Fair solo baseline for the strongest observed OpenCode builder.",
        ),
        ScaffoldArm(
            name="solo__opencode_mimo_repair",
            domain=coding,
            stage="single_scaffold",
            worker_names=("opencode_mimo_repair",),
            workflow=_workflow(
                _step(ix, "opencode_mimo_repair", "Implement the repository fix and run the relevant tests.")
            ),
            rationale="Fair solo baseline for MiMo as a cheap independent coding worker.",
        ),
        ScaffoldArm(
            name="solo__opencode_glm_builder",
            domain=coding,
            stage="single_scaffold",
            worker_names=("opencode_glm_builder",),
            workflow=_workflow(
                _step(ix, "opencode_glm_builder", "Implement the repository fix and run the relevant tests.")
            ),
            rationale="Fair solo baseline for GLM as the strongest open generalist.",
        ),
        ScaffoldArm(
            name="codex_build__claude_debug__codex_repair",
            domain=coding,
            stage="role_workflow",
            worker_names=("codex_gpt_coding_agent", "claude_code_opus_debugger"),
            workflow=_workflow(
                _step(ix, "codex_gpt_coding_agent", "Implement the repository fix and run the relevant tests."),
                _step(
                    ix,
                    "claude_code_opus_debugger",
                    "Audit the previous patch for hidden failures, incorrect assumptions, and missing tests.",
                    [0],
                ),
                _step(
                    ix,
                    "codex_gpt_coding_agent",
                    "Repair the implementation using the audit findings and rerun the relevant tests.",
                    [0, 1],
                ),
            ),
            rationale="Tests the GPT builder plus Opus debugger pattern.",
        ),
        ScaffoldArm(
            name="kimi_build__claude_debug__kimi_repair",
            domain=coding,
            stage="role_workflow",
            worker_names=("opencode_kimi_builder", "claude_code_opus_debugger"),
            workflow=_workflow(
                _step(ix, "opencode_kimi_builder", "Implement the repository fix and run the relevant tests."),
                _step(
                    ix,
                    "claude_code_opus_debugger",
                    "Audit the previous patch for hidden failures, incorrect assumptions, and missing tests.",
                    [0],
                ),
                _step(
                    ix,
                    "opencode_kimi_builder",
                    "Repair the implementation using the audit findings and rerun the relevant tests.",
                    [0, 1],
                ),
            ),
            rationale="Tests whether Opus improves the strongest observed OpenCode builder.",
        ),
        ScaffoldArm(
            name="codex_plan__kimi_build__claude_verify",
            domain=coding,
            stage="role_workflow",
            worker_names=("codex_gpt_coding_agent", "opencode_kimi_builder", "claude_code_opus_debugger"),
            workflow=_workflow(
                _step(ix, "codex_gpt_coding_agent", "Analyze the issue and produce an implementation plan."),
                _step(
                    ix,
                    "opencode_kimi_builder",
                    "Implement the fix using the plan, then run the relevant tests.",
                    [0],
                ),
                _step(
                    ix,
                    "claude_code_opus_debugger",
                    "Verify the patch, repair if needed, and leave the repository in a passing state.",
                    [0, 1],
                ),
            ),
            rationale="Separates planning, implementation, and verification across the strongest coding roles.",
        ),
        ScaffoldArm(
            name="claude_diagnose__gpt_alt__kimi_impl",
            domain=coding,
            stage="role_workflow",
            worker_names=("claude_code_opus_debugger", "direct_gpt_reasoner", "opencode_kimi_builder"),
            workflow=_workflow(
                _step(ix, "claude_code_opus_debugger", "Diagnose the likely root cause from repository context."),
                _step(
                    ix,
                    "direct_gpt_reasoner",
                    "Independently re-examine the problem and propose an alternate diagnosis.",
                    [0],
                ),
                _step(
                    ix,
                    "opencode_kimi_builder",
                    "Synthesize the diagnoses, implement the fix, and run the relevant tests.",
                    [0, 1],
                ),
            ),
            rationale="Tests the clean-slate second-specialist pattern before Kimi implementation.",
        ),
        ScaffoldArm(
            name="kimi_mimo_build__claude_synth",
            domain=coding,
            stage="role_workflow",
            worker_names=("opencode_kimi_builder", "opencode_mimo_repair", "claude_code_opus_debugger"),
            workflow=_workflow(
                _step(ix, "opencode_kimi_builder", "Make an independent implementation attempt and run tests."),
                _step(ix, "opencode_mimo_repair", "Make an independent implementation attempt and run tests."),
                _step(
                    ix,
                    "claude_code_opus_debugger",
                    "Compare both attempts, merge or repair the better approach, and verify the result.",
                    [0, 1],
                ),
            ),
            rationale="Tests open-worker diversity with Opus as the final coding verifier/repairer.",
        ),
        ScaffoldArm(
            name="gpt_plan__codex_impl__opus_review",
            domain=coding,
            stage="role_workflow",
            worker_names=("direct_gpt_reasoner", "codex_gpt_coding_agent", "direct_opus_reviewer"),
            workflow=_workflow(
                _step(ix, "direct_gpt_reasoner", "Design the implementation strategy and identify likely edge cases."),
                _step(ix, "codex_gpt_coding_agent", "Implement the fix from the plan and run tests.", [0]),
                _step(ix, "direct_opus_reviewer", "Review the final patch for hidden failures and quality risks.", [0, 1]),
            ),
            rationale="Tests direct GPT planning and direct Opus review around an editable Codex implementation.",
        ),
        ScaffoldArm(
            name="solo__tool_dialog_mimo_agent",
            domain=tool,
            stage="single_scaffold",
            worker_names=("tool_dialog_mimo_agent",),
            workflow=_workflow(_step(ix, "tool_dialog_mimo_agent", "Solve the tool-dialog task to completion.")),
            rationale="Fair solo baseline for the best historical tau/tool-dialog worker.",
        ),
        ScaffoldArm(
            name="solo__tool_dialog_glm_agent",
            domain=tool,
            stage="single_scaffold",
            worker_names=("tool_dialog_glm_agent",),
            workflow=_workflow(_step(ix, "tool_dialog_glm_agent", "Solve the tool-dialog task to completion.")),
            rationale="Fair solo baseline for GLM in tool-dialog mode.",
        ),
        ScaffoldArm(
            name="solo__tool_dialog_flash_agent",
            domain=tool,
            stage="single_scaffold",
            worker_names=("tool_dialog_flash_agent",),
            workflow=_workflow(_step(ix, "tool_dialog_flash_agent", "Solve the tool-dialog task to completion.")),
            rationale="Fair solo baseline for Flash in tool-dialog mode.",
        ),
        ScaffoldArm(
            name="flash_tool_attempt__mimo_repair",
            domain=tool,
            stage="role_workflow",
            worker_names=("tool_dialog_flash_agent", "tool_dialog_mimo_agent"),
            workflow=_workflow(
                _step(ix, "tool_dialog_flash_agent", "Attempt the tool-dialog task quickly and record the state."),
                _step(ix, "tool_dialog_mimo_agent", "Continue from the prior state and repair any mistakes.", [0]),
            ),
            rationale="Tests Flash as a cheap first pass with MiMo as the procedural repair worker.",
        ),
        ScaffoldArm(
            name="glm_tool_attempt__mimo_repair",
            domain=tool,
            stage="role_workflow",
            worker_names=("tool_dialog_glm_agent", "tool_dialog_mimo_agent"),
            workflow=_workflow(
                _step(ix, "tool_dialog_glm_agent", "Attempt the tool-dialog task and record the state."),
                _step(ix, "tool_dialog_mimo_agent", "Continue from the prior state and repair any mistakes.", [0]),
            ),
            rationale="Tests GLM's structured first pass with MiMo repair.",
        ),
        ScaffoldArm(
            name="mimo_tool__opus_review",
            domain=tool,
            stage="role_workflow",
            worker_names=("tool_dialog_mimo_agent", "direct_opus_reviewer"),
            workflow=_workflow(
                _step(ix, "tool_dialog_mimo_agent", "Solve the tool-dialog task to completion."),
                _step(ix, "direct_opus_reviewer", "Review the trajectory and final answer for missed constraints.", [0]),
            ),
            rationale="Tests whether a frontier reviewer improves MiMo's tool-dialog output.",
        ),
        ScaffoldArm(
            name="solo__direct_gemini_synth",
            domain=direct,
            stage="single_scaffold",
            worker_names=("direct_gemini_synth",),
            workflow=_workflow(_step(ix, "direct_gemini_synth", "Solve the task directly.")),
            rationale="Fair solo baseline for Gemini on math/science/general tasks.",
        ),
        ScaffoldArm(
            name="solo__direct_gpt_reasoner",
            domain=direct,
            stage="single_scaffold",
            worker_names=("direct_gpt_reasoner",),
            workflow=_workflow(_step(ix, "direct_gpt_reasoner", "Solve the task directly.")),
            rationale="Fair solo baseline for GPT on math/science/general tasks.",
        ),
        ScaffoldArm(
            name="solo__direct_opus_reviewer",
            domain=direct,
            stage="single_scaffold",
            worker_names=("direct_opus_reviewer",),
            workflow=_workflow(_step(ix, "direct_opus_reviewer", "Solve the task directly.")),
            rationale="Fair solo baseline for Opus on math/science/general tasks.",
        ),
        ScaffoldArm(
            name="solo__direct_flash_fast",
            domain=direct,
            stage="single_scaffold",
            worker_names=("direct_flash_fast",),
            workflow=_workflow(_step(ix, "direct_flash_fast", "Solve the task directly.")),
            rationale="Fair solo baseline for Flash as a strong fast direct worker.",
        ),
        ScaffoldArm(
            name="gpt_math__gemini_verify__opus_final",
            domain=direct,
            stage="role_workflow",
            worker_names=("direct_gpt_reasoner", "direct_gemini_synth", "direct_opus_reviewer"),
            workflow=_workflow(
                _step(ix, "direct_gpt_reasoner", "Derive the mathematical or algorithmic core of the answer."),
                _step(ix, "direct_gemini_synth", "Check factual and scientific constraints independently.", [0]),
                _step(ix, "direct_opus_reviewer", "Produce the final answer after reconciling both analyses.", [0, 1]),
            ),
            rationale="Tests the frontier triad on hard direct reasoning.",
        ),
        ScaffoldArm(
            name="flash_answer__gemini_audit",
            domain=direct,
            stage="role_workflow",
            worker_names=("direct_flash_fast", "direct_gemini_synth"),
            workflow=_workflow(
                _step(ix, "direct_flash_fast", "Produce a fast first-pass answer."),
                _step(ix, "direct_gemini_synth", "Audit and correct the first-pass answer.", [0]),
            ),
            rationale="Tests Flash as a cheap first pass with Gemini factual synthesis.",
        ),
        ScaffoldArm(
            name="opus_answer__gpt_critic__opus_revise",
            domain=direct,
            stage="role_workflow",
            worker_names=("direct_opus_reviewer", "direct_gpt_reasoner"),
            workflow=_workflow(
                _step(ix, "direct_opus_reviewer", "Solve the task directly."),
                _step(ix, "direct_gpt_reasoner", "Critique the answer and identify any missing derivation or edge case.", [0]),
                _step(ix, "direct_opus_reviewer", "Revise the answer using the critique.", [0, 1]),
            ),
            rationale="Tests same-frontier revision with GPT as the alternate perspective.",
        ),
        ScaffoldArm(
            name="gemini_answer__opus_review",
            domain=direct,
            stage="role_workflow",
            worker_names=("direct_gemini_synth", "direct_opus_reviewer"),
            workflow=_workflow(
                _step(ix, "direct_gemini_synth", "Solve the task directly."),
                _step(ix, "direct_opus_reviewer", "Review, correct, and finalize the answer.", [0]),
            ),
            rationale="Tests Gemini factual strength with Opus final quality control.",
        ),
    ]

    for arm in arms:
        validate_workflow(arm.workflow, worker_count=len(workers))
    return arms


def build_plan(task_mix: dict[str, int] | None = None) -> dict[str, Any]:
    mix = dict(DEFAULT_TASK_MIX if task_mix is None else task_mix)
    workers = canonical_workers()
    worker_name_by_id = {worker.worker_id: worker.name for worker in workers}
    arms = canonical_arms(workers)

    arm_rows = [arm.model_dump(worker_name_by_id) for arm in arms]
    rollouts_by_domain = {}
    calls_by_domain = {}
    for domain, n_tasks in mix.items():
        domain_arms = [arm for arm in arms if arm.domain == domain]
        rollouts_by_domain[domain] = n_tasks * len(domain_arms)
        calls_by_domain[domain] = n_tasks * sum(arm.worker_calls for arm in domain_arms)

    return {
        "version": "scaffold_tournament_v1",
        "purpose": "Preregister the quality-first, scaffold-aware Ultra role tournament before live calls.",
        "task_mix": mix,
        "workers": [worker.model_dump() for worker in workers],
        "worker_harnesses": worker_harness_map(workers),
        "arms": arm_rows,
        "domains": sorted(mix),
        "rollouts_by_domain": rollouts_by_domain,
        "worker_calls_by_domain": calls_by_domain,
        "total_rollouts": sum(rollouts_by_domain.values()),
        "total_worker_calls": sum(calls_by_domain.values()),
        "fair_baselines": [
            "best individual model+scaffold worker on the same task set",
            "best direct frontier model",
            "best OpenCode worker",
            "best Claude Code solo worker",
            "best Codex solo worker",
            "best fixed workflow arm",
            "best-of-N single scaffold with comparable call budget",
            "trained Fugu-Ultra Conductor",
        ],
        "decision_rule": [
            "Use paired held-out task outcomes, not direct accuracy alone.",
            "Keep a worker configuration if leave-one-out workflow success drops on held-out tasks.",
            "Treat commercial frontier workers as required baselines even if a tiny coding shard is negative.",
            "Select final training pool over worker configurations, then collapse duplicate model roles only if evidence says they are redundant.",
        ],
        "cost_note": "Yunwu reported cost may be absent; external spend monitoring is authoritative.",
        "live_calls": False,
    }


def default_manifest_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "director" / "manifests" / "fugu_clean_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _balanced_counts(total: int, labels: list[str]) -> dict[str, int]:
    base = total // len(labels)
    remainder = total % len(labels)
    return {label: base + (1 if i < remainder else 0) for i, label in enumerate(labels)}


def _success_count(rewards: list[Any] | None) -> int | None:
    if rewards is None:
        return None
    return sum(1 for reward in rewards if float(reward) > 0.0)


def _rank_by_mid_difficulty(row: dict[str, Any], n_workers: int, rnd: random.Random) -> tuple[float, str, float]:
    success_count = row.get("prior_success_count")
    if success_count is None:
        success_count = n_workers / 2
    all_or_none_penalty = 1.0 if success_count in {0, n_workers} else 0.0
    return (all_or_none_penalty, abs(float(success_count) - (n_workers / 2)), rnd.random())


def _coding_candidates(manifest_dir: Path) -> list[dict[str, Any]]:
    plan_path = manifest_dir / "agentic_coding_frontier_direct3.plan.json"
    rollouts_path = manifest_dir / "agentic_coding_frontier_direct3.jsonl"
    tasks: list[str] = []
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        tasks = [str(task_id) for task_id in plan.get("tasks", [])]

    outcomes: dict[str, dict[str, float]] = defaultdict(dict)
    for row in _read_jsonl(rollouts_path):
        task_id = str(row["task_id"])
        worker = str(row.get("workers", [row.get("arm", "unknown")])[0])
        reward = float(row.get("reward", 0.0)) if row.get("valid", True) else 0.0
        outcomes[task_id][worker] = reward
        if task_id not in tasks:
            tasks.append(task_id)

    candidates = []
    for i, task_id in enumerate(tasks):
        task_outcomes = outcomes.get(task_id, {})
        candidates.append(
            {
                "tournament_task_id": f"coding_repo::{i:04d}::{task_id}",
                "domain": "coding_repo",
                "source": "agentic_coding_frontier_direct3",
                "source_task_id": task_id,
                "harness": "opencode_repo",
                "split": "pool_validation",
                "selection_tags": ["live_coding_shard", "repo_task"],
                "diagnostic": {
                    "prior_success_count": sum(1 for score in task_outcomes.values() if score > 0.0),
                    "prior_worker_count": len(task_outcomes),
                    "prior_outcomes": task_outcomes,
                },
            }
        )
    return candidates


def _deep_swe_root(manifest_dir: Path) -> Path:
    try:
        director_root = manifest_dir.parents[1]
    except IndexError:
        director_root = manifest_dir.parent
    return director_root / "vendor" / "deep_swe" / "tasks"


def _deep_swe_candidates(manifest_dir: Path, seed: int) -> list[dict[str, Any]]:
    root = _deep_swe_root(manifest_dir)
    if not root.exists():
        return []

    candidates = []
    rnd = random.Random(seed)
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        task_toml = task_dir / "task.toml"
        instruction = task_dir / "instruction.md"
        if not task_toml.exists() or not instruction.exists():
            continue
        try:
            meta = tomllib.loads(task_toml.read_text())
        except tomllib.TOMLDecodeError:
            continue
        metadata = meta.get("metadata", {})
        task_id = str(metadata.get("task_id") or task_dir.name)
        language = str(metadata.get("language") or "unknown")
        category = str(metadata.get("category") or "unknown")
        candidates.append(
            {
                "tournament_task_id": f"coding_repo::deep_swe::{task_id}",
                "domain": "coding_repo",
                "source": "deep_swe_local",
                "source_task_id": task_id,
                "harness": "opencode_repo",
                "split": "pool_validation",
                "repo": {
                    "url": metadata.get("repository_url"),
                    "base_commit": metadata.get("base_commit_hash"),
                },
                "task_dir": str(task_dir),
                "instruction_ref": str(instruction),
                "selection_tags": ["deep_swe_local", "repo_task", "coding_deficit_fill", language, category],
                "diagnostic": {
                    "prior_success_count": None,
                    "prior_worker_count": 0,
                    "language": language,
                    "category": category,
                    "display_title": metadata.get("display_title"),
                },
                "_language": language,
                "_rank": rnd.random(),
            }
        )
    return _diverse_by_language(candidates)


def _diverse_by_language(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_language[str(row.get("_language", "unknown"))].append(row)
    for rows in by_language.values():
        rows.sort(key=lambda row: (row["_rank"], row["source_task_id"]))

    out = []
    languages = sorted(by_language)
    while True:
        progressed = False
        for language in languages:
            rows = by_language[language]
            if rows:
                out.append(rows.pop(0))
                progressed = True
        if not progressed:
            return out


def _select_coding_tasks(manifest_dir: Path, requested: int, seed: int) -> tuple[list[dict[str, Any]], int]:
    live = _coding_candidates(manifest_dir)
    if len(live) >= requested:
        return live[:requested], 0

    seen = {row["source_task_id"] for row in live}
    fillers = [row for row in _deep_swe_candidates(manifest_dir, seed) if row["source_task_id"] not in seen]
    selected = [*live, *fillers[: max(0, requested - len(live))]]
    return selected, max(0, requested - len(selected))


def _tau_candidates(manifest_dir: Path, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in _read_jsonl(manifest_dir / "agentic_bank.jsonl"):
        grouped[(str(row["domain"]), str(row["item_id"]))][str(row["worker"])] = float(row.get("reward", 0.0))

    rnd = random.Random(seed)
    candidates = []
    for i, ((domain, item_id), outcomes) in enumerate(sorted(grouped.items())):
        success_count = sum(1 for score in outcomes.values() if score > 0.0)
        candidates.append(
            {
                "tournament_task_id": f"tool_dialog::{i:04d}::{item_id}",
                "domain": "tool_dialog",
                "source_domain": domain,
                "source": "agentic_bank",
                "source_task_id": item_id,
                "harness": "tool_dialog",
                "split": "pool_validation",
                "selection_tags": ["tau", domain, "medium_difficulty_preferred"],
                "prior_success_count": success_count,
                "diagnostic": {
                    "prior_success_count": success_count,
                    "prior_worker_count": len(outcomes),
                    "prior_outcomes": outcomes,
                },
                "_rank": _rank_by_mid_difficulty({"prior_success_count": success_count}, len(outcomes), rnd),
            }
        )
    return candidates


def _direct_candidates(manifest_dir: Path, seed: int) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    rows = []
    for row in _read_jsonl(manifest_dir / "manifest.jsonl"):
        if row.get("split") != "test":
            continue
        if row.get("verdict") != "discriminative":
            continue
        if row.get("domain") not in {"math", "science", "general"}:
            continue
        success_count = _success_count(row.get("rewards"))
        rows.append(
            {
                "tournament_task_id": f"direct_reasoning::{len(rows):04d}::{row['task_id']}",
                "domain": "direct_reasoning",
                "source_domain": row.get("domain"),
                "source": "existing_bank",
                "source_family": row.get("source"),
                "source_task_id": str(row["task_id"]),
                "harness": "direct_qa",
                "split": "online_validation",
                "grader": row.get("grader"),
                "selection_tags": ["existing_bank", "discriminative", str(row.get("domain"))],
                "prior_success_count": success_count,
                "diagnostic": {
                    "prior_success_count": success_count,
                    "prior_worker_count": len(row.get("rewards") or []),
                },
                "_rank": _rank_by_mid_difficulty({"prior_success_count": success_count}, len(row.get("rewards") or []), rnd),
            }
        )
    return rows


def _take_balanced(
    candidates: list[dict[str, Any]],
    *,
    requested: int,
    balance_key: str,
    labels: list[str],
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    counts = _balanced_counts(requested, labels)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        label = str(row.get(balance_key))
        if label in counts:
            by_label[label].append(row)

    selected = []
    deficit = 0
    rnd = random.Random(seed)
    for label in labels:
        rows = by_label.get(label, [])
        for row in rows:
            row.setdefault("_rank", (0.0, 0.0, rnd.random()))
        rows = sorted(rows, key=lambda row: row["_rank"])
        take = min(counts[label], len(rows))
        selected.extend(rows[:take])
        deficit += counts[label] - take
    return selected, deficit


def _strip_internal(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def build_concrete_manifest(
    manifest_dir: Path | None = None,
    *,
    task_mix: dict[str, int] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    manifest_dir = manifest_dir or default_manifest_dir()
    mix = dict(DEFAULT_TASK_MIX if task_mix is None else task_mix)
    plan = build_plan(mix)
    arms = canonical_arms(canonical_workers())
    arms_by_domain: dict[str, list[ScaffoldArm]] = defaultdict(list)
    for arm in arms:
        arms_by_domain[arm.domain].append(arm)

    coding_selected, coding_deficit = _select_coding_tasks(
        manifest_dir,
        requested=mix.get("coding_repo", 0),
        seed=seed + 11,
    )

    tau_selected, tau_deficit = _take_balanced(
        _tau_candidates(manifest_dir, seed + 17),
        requested=mix.get("tool_dialog", 0),
        balance_key="source_domain",
        labels=["tau_retail", "tau_airline"],
        seed=seed + 23,
    )
    direct_selected, direct_deficit = _take_balanced(
        _direct_candidates(manifest_dir, seed + 31),
        requested=mix.get("direct_reasoning", 0),
        balance_key="source_domain",
        labels=["math", "science", "general"],
        seed=seed + 37,
    )

    tasks = [
        *(_strip_internal(row) for row in coding_selected),
        *(_strip_internal(row) for row in tau_selected),
        *(_strip_internal(row) for row in direct_selected),
    ]
    harness_map = worker_harness_map(canonical_workers())
    jobs = []
    for task in tasks:
        for arm in arms_by_domain[task["domain"]]:
            jobs.append(
                {
                    "job_id": f"job_{len(jobs):05d}",
                    "tournament_task_id": task["tournament_task_id"],
                    "domain": task["domain"],
                    "source": task["source"],
                    "source_task_id": task["source_task_id"],
                    "arm": arm.name,
                    "worker_names": list(arm.worker_names),
                    "worker_harnesses": [harness_map[name] for name in arm.worker_names],
                    "worker_calls": arm.worker_calls,
                }
            )

    selected_counts = dict(Counter(task["domain"] for task in tasks))
    deficits = {
        "coding_repo": coding_deficit,
        "tool_dialog": tau_deficit,
        "direct_reasoning": direct_deficit,
    }
    blocked = []
    if coding_deficit:
        blocked.append(
            f"Need {coding_deficit} more repo-coding tasks or trace-derived TaskSpecs "
            "to reach the preregistered coding task count."
        )
    if tau_deficit:
        blocked.append(f"Need {tau_deficit} more tau/tool-dialog tasks to reach the requested count.")
    if direct_deficit:
        blocked.append(f"Need {direct_deficit} more direct reasoning tasks to reach the requested count.")

    return {
        "version": "scaffold_tournament_manifest_v1",
        "seed": seed,
        "manifest_dir": str(manifest_dir),
        "plan_version": plan["version"],
        "requested_task_mix": mix,
        "selected_task_counts": selected_counts,
        "deficits": deficits,
        "blocked_reasons": blocked,
        "tasks": tasks,
        "jobs": jobs,
        "job_count": len(jobs),
        "worker_call_count": sum(job["worker_calls"] for job in jobs),
        "worker_harnesses": harness_map,
        "arms_by_domain": {domain: [arm.name for arm in rows] for domain, rows in sorted(arms_by_domain.items())},
        "fair_baselines": plan["fair_baselines"],
        "decision_rule": plan["decision_rule"],
        "cost_note": plan["cost_note"],
        "live_calls": False,
    }


def write_concrete_manifest(
    manifest_dir: Path,
    out: Path,
    jobs_out: Path | None = None,
    *,
    task_mix: dict[str, int] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    manifest = build_concrete_manifest(manifest_dir, task_mix=task_mix, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if jobs_out is not None:
        jobs_out.parent.mkdir(parents=True, exist_ok=True)
        with jobs_out.open("w") as f:
            for job in manifest["jobs"]:
                f.write(json.dumps(job, sort_keys=True) + "\n")
    return manifest


def analyze_readiness(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    workers = {worker.name: worker for worker in canonical_workers()}
    backend_status = {
        "direct_qa": "ready",
        "opencode": "ready_payload_required",
        "claude_code": "adapter_pending",
        "codex": "adapter_pending",
        "tool_dialog": "harness_pending",
    }

    jobs_by_status: Counter[str] = Counter()
    jobs_by_domain_status: Counter[tuple[str, str]] = Counter()
    jobs_by_backend: Counter[str] = Counter()
    jobs_by_harness: Counter[str] = Counter()
    ready_jobs = []
    blocked_examples = []

    for job in manifest.get("jobs", []):
        backends = [workers[name].backend for name in job["worker_names"] if name in workers]
        for backend in backends:
            jobs_by_backend[backend] += 1
        for harness in job.get("worker_harnesses", []):
            jobs_by_harness[harness] += 1
        statuses = {backend_status.get(backend, "unknown_backend") for backend in backends}
        if statuses == {"ready"}:
            status = "ready"
            ready_jobs.append(job["job_id"])
        elif "unknown_backend" in statuses:
            status = "unknown_backend"
        elif "harness_pending" in statuses:
            status = "harness_pending"
        elif "adapter_pending" in statuses:
            status = "adapter_pending"
        elif statuses == {"ready_payload_required"}:
            status = (
                "opencode_ready_not_live_smoked"
                if job.get("source") == "deep_swe_local"
                else "payload_pending"
            )
        else:
            status = "mixed_pending"
        jobs_by_status[status] += 1
        jobs_by_domain_status[(job["domain"], status)] += 1
        if status != "ready" and len(blocked_examples) < 10:
            blocked_examples.append(
                {
                    "job_id": job["job_id"],
                    "domain": job["domain"],
                    "arm": job["arm"],
                    "worker_names": job["worker_names"],
                    "worker_harnesses": job.get("worker_harnesses", []),
                    "backends": backends,
                    "status": status,
                }
            )

    return {
        "version": "scaffold_tournament_readiness_v1",
        "manifest_path": str(manifest_path),
        "job_count": len(manifest.get("jobs", [])),
        "worker_call_count": manifest.get("worker_call_count", 0),
        "jobs_by_status": dict(jobs_by_status),
        "jobs_by_domain_status": {
            f"{domain}:{status}": count
            for (domain, status), count in sorted(jobs_by_domain_status.items())
        },
        "worker_calls_by_backend": dict(jobs_by_backend),
        "worker_calls_by_harness": dict(jobs_by_harness),
        "ready_job_ids_sample": ready_jobs[:20],
        "blocked_examples": blocked_examples,
        "backend_status": backend_status,
        "next_blockers": [
            "Run a no/low-spend OpenCode + Deep SWE Docker canary on one materialized local repo task.",
            "Recover original payloads for the three saved live SWE-smith coding tasks, or replace them with local repo tasks.",
            "Implement ClaudeCodeHarness for claude-code:opus coding/debugging workers.",
            "Implement CodexHarness for codex:gpt coding workers.",
            "Implement or connect ToolDialogHarness for tau-style tasks.",
        ],
        "live_calls": False,
    }


def write_readiness(manifest_path: Path, out: Path | None = None) -> dict[str, Any]:
    report = analyze_readiness(manifest_path)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def write_plan(path: Path, task_mix: dict[str, int] | None = None) -> dict[str, Any]:
    plan = build_plan(task_mix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render the scaffold-aware Ultra tournament plan")
    parser.add_argument("--coding-tasks", type=int, default=DEFAULT_TASK_MIX["coding_repo"])
    parser.add_argument("--tool-dialog-tasks", type=int, default=DEFAULT_TASK_MIX["tool_dialog"])
    parser.add_argument("--direct-tasks", type=int, default=DEFAULT_TASK_MIX["direct_reasoning"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    mix = {
        "coding_repo": args.coding_tasks,
        "tool_dialog": args.tool_dialog_tasks,
        "direct_reasoning": args.direct_tasks,
    }
    plan = build_plan(mix)
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
