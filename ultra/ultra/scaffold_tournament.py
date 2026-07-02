"""Preregistered scaffold-aware tournament plan for Fugu-Ultra.

This module does not call providers. It defines the worker configurations and fixed
workflow arms needed before running a quality-first, scaffold-aware role tournament.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .providers import slug as _provider_slug
from .schemas import TaskSpec, ToolPermissions, WorkerIdentity, Workflow, WorkflowStep
from .workflow import validate_workflow

REPO_LANE = "repo_open_repo_terminal"
UNIT_CODE_LANE = "unit_and_scientific_code"
DIRECT_LANE = "math_science_knowledge"
TOOL_LANE = "tool_dialogue"
LONG_CONTEXT_LANE = "long_context_memory_planning"
TRACE_BRANCH_LANE = "trace_state_branches"

REPO_ARM_DOMAIN = "repo_coding"
TERMINAL_ARM_DOMAIN = "terminal_sandbox"

TASKTROVE_UNIT_CODE_SOURCES = {
    "tasktrove_code_contests",
    "tasktrove_pymethods2test",
}

TASKTROVE_REPO_TERMINAL_SOURCES = {
    "tasktrove_inferredbugs",
    "tasktrove_multifile_composition",
    "tasktrove_nl2bash",
    "tasktrove_r2egym",
    "tasktrove_repo_scaffold",
    "tasktrove_stack_bash_v3",
    "tasktrove_swegym",
    "tasktrove_swesmith",
}

DEFAULT_TASK_MIX = {
    REPO_LANE: 50,
    UNIT_CODE_LANE: 45,
    DIRECT_LANE: 45,
    TOOL_LANE: 35,
    LONG_CONTEXT_LANE: 25,
}

LEGACY_TASK_MIX_ALIASES = {
    "coding_repo": REPO_LANE,
    "tool_dialog": TOOL_LANE,
    "direct_reasoning": DIRECT_LANE,
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
                        "budget": step.budget,
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
            model=_provider_slug("gpt"),
            role_prior=["planner", "builder", "repair"],
            max_turns=100,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=1,
            name="claude_code_opus_debugger",
            backend="claude_code",
            model=_provider_slug("opus"),
            role_prior=["debugger", "verifier", "security_review", "repair"],
            max_turns=100,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=2,
            name="opencode_kimi_builder",
            backend="opencode",
            model=_provider_slug("kimi"),
            role_prior=["builder", "implementation", "repair"],
            max_turns=100,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=3,
            name="opencode_mimo_repair",
            backend="opencode",
            model=_provider_slug("mimo"),
            role_prior=["agentic_executor", "independent_attempt", "repair"],
            max_turns=100,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=4,
            name="opencode_glm_builder",
            backend="opencode",
            model=_provider_slug("glm"),
            role_prior=["open_generalist", "secondary_builder", "debugger"],
            max_turns=100,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=5,
            name="tool_dialog_mimo_agent",
            backend="tool_dialog",
            model=_provider_slug("mimo"),
            role_prior=["tool_dialog", "procedural_task_solver", "repair"],
            max_turns=80,
        ),
        WorkerIdentity(
            worker_id=6,
            name="tool_dialog_glm_agent",
            backend="tool_dialog",
            model=_provider_slug("glm"),
            role_prior=["tool_dialog", "structured_reasoning"],
            max_turns=80,
        ),
        WorkerIdentity(
            worker_id=7,
            name="tool_dialog_flash_agent",
            backend="tool_dialog",
            model=_provider_slug("flash"),
            role_prior=["tool_dialog", "fast_first_pass"],
            max_turns=80,
        ),
        WorkerIdentity(
            worker_id=8,
            name="direct_gemini_synth",
            backend="direct_qa",
            model=_provider_slug("gemini"),
            role_prior=["science", "factual_synthesis", "long_context"],
        ),
        WorkerIdentity(
            worker_id=9,
            name="direct_gpt_reasoner",
            backend="direct_qa",
            model=_provider_slug("gpt"),
            role_prior=["planner", "math", "algorithm_design", "alternate_perspective"],
        ),
        WorkerIdentity(
            worker_id=10,
            name="direct_opus_reviewer",
            backend="direct_qa",
            model=_provider_slug("opus"),
            role_prior=["reviewer", "debugger", "verifier", "final_quality_control"],
        ),
        WorkerIdentity(
            worker_id=11,
            name="direct_flash_fast",
            backend="direct_qa",
            model=_provider_slug("flash"),
            role_prior=["fast_first_pass", "easy_direct_subtasks"],
        ),
        WorkerIdentity(
            worker_id=12,
            name="terminal_gpt_agent",
            backend="terminal",
            model=_provider_slug("gpt"),
            role_prior=["terminal_solver", "planner", "repair"],
            max_turns=80,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=13,
            name="terminal_kimi_agent",
            backend="terminal",
            model=_provider_slug("kimi"),
            role_prior=["terminal_solver", "implementation", "repair"],
            max_turns=80,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=14,
            name="terminal_mimo_agent",
            backend="terminal",
            model=_provider_slug("mimo"),
            role_prior=["terminal_solver", "cheap_independent_attempt", "repair"],
            max_turns=80,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=15,
            name="terminal_glm_agent",
            backend="terminal",
            model=_provider_slug("glm"),
            role_prior=["terminal_solver", "structured_reasoning"],
            max_turns=80,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=16,
            name="direct_glm_reasoner",
            backend="direct_qa",
            model=_provider_slug("glm"),
            role_prior=["open_generalist", "reasoning", "long_context", "structured_answering"],
        ),
        WorkerIdentity(
            worker_id=17,
            name="direct_mimo_reasoner",
            backend="direct_qa",
            model=_provider_slug("mimo"),
            role_prior=["open_specialist", "agentic_reasoning", "long_context_challenger"],
        ),
        WorkerIdentity(
            worker_id=18,
            name="direct_minimax_reasoner",
            backend="direct_qa",
            model=_provider_slug("minimax"),
            role_prior=["open_specialist", "reasoning", "long_context_challenger"],
        ),
        WorkerIdentity(
            worker_id=19,
            name="opencode_flash_challenger",
            backend="opencode",
            model=_provider_slug("flash"),
            role_prior=["fast_first_pass", "cheap_independent_attempt", "repo_repair_challenger"],
            max_turns=80,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=20,
            name="opencode_minimax_challenger",
            backend="opencode",
            model=_provider_slug("minimax"),
            role_prior=["open_specialist", "repo_repair_challenger", "alternate_builder"],
            max_turns=80,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
        WorkerIdentity(
            worker_id=21,
            name="opencode_deepseek_pro_challenger",
            backend="opencode",
            model=_provider_slug("deepseek-pro"),
            role_prior=["open_specialist", "repo_repair_challenger", "debugger"],
            max_turns=80,
            tool_permissions=ToolPermissions(read_files=True, edit_files=True, run_tests=True),
        ),
    ]


def worker_harness_map(
    workers: list[WorkerIdentity] | None = None,
    *,
    task_harness: str | None = None,
) -> dict[str, str]:
    workers = workers or canonical_workers()
    if task_harness in {"direct_qa", "code_exec", "tool_dialog", "tau_bench", "terminal_sandbox", "long_context"}:
        return {worker.name: task_harness for worker in workers}
    if task_harness in {"opencode", "opencode_repo", "codex", "claude_code"}:
        routed = {}
        for worker in workers:
            if worker.backend in {"opencode", "claude_code", "codex"}:
                routed[worker.name] = BACKEND_TO_HARNESS[worker.backend]
            else:
                routed[worker.name] = "direct_qa"
        return routed
    return {worker.name: BACKEND_TO_HARNESS[worker.backend] for worker in workers}


def _index(workers: list[WorkerIdentity]) -> dict[str, int]:
    return {worker.name: worker.worker_id for worker in workers}


def _step(index: dict[str, int], worker: str, subtask: str, access: list[int] | None = None) -> WorkflowStep:
    return WorkflowStep(worker_id=index[worker], subtask=subtask, access=access or [])


def _workflow(*steps: WorkflowStep) -> Workflow:
    return Workflow(steps=list(steps))


def _normalize_task_mix(task_mix: dict[str, int] | None) -> dict[str, int]:
    source = DEFAULT_TASK_MIX if task_mix is None else task_mix
    normalized = {lane: 0 for lane in DEFAULT_TASK_MIX}
    for key, value in source.items():
        lane = LEGACY_TASK_MIX_ALIASES.get(key, key)
        normalized[lane] = normalized.get(lane, 0) + int(value)
    return normalized


def canonical_arms(workers: list[WorkerIdentity] | None = None) -> list[ScaffoldArm]:
    workers = workers or canonical_workers()
    ix = _index(workers)

    coding = REPO_ARM_DOMAIN
    terminal = TERMINAL_ARM_DOMAIN
    unit_code = UNIT_CODE_LANE
    tool = TOOL_LANE
    direct = DIRECT_LANE
    long_context = LONG_CONTEXT_LANE

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
            name="solo__opencode_flash_challenger",
            domain=coding,
            stage="single_scaffold",
            worker_names=("opencode_flash_challenger",),
            workflow=_workflow(
                _step(ix, "opencode_flash_challenger", "Implement the repository fix and run the relevant tests.")
            ),
            rationale="OpenCode challenger for fast DeepSeek Flash repo-repair behavior.",
        ),
        ScaffoldArm(
            name="solo__opencode_minimax_challenger",
            domain=coding,
            stage="single_scaffold",
            worker_names=("opencode_minimax_challenger",),
            workflow=_workflow(
                _step(ix, "opencode_minimax_challenger", "Implement the repository fix and run the relevant tests.")
            ),
            rationale="OpenCode challenger for MiniMax repo-repair behavior.",
        ),
        ScaffoldArm(
            name="solo__opencode_deepseek_pro_challenger",
            domain=coding,
            stage="single_scaffold",
            worker_names=("opencode_deepseek_pro_challenger",),
            workflow=_workflow(
                _step(ix, "opencode_deepseek_pro_challenger", "Implement the repository fix and run the relevant tests.")
            ),
            rationale="OpenCode challenger for DeepSeek Pro repo-repair behavior.",
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
            name="gpt_plan__codex_impl__opus_review__codex_repair",
            domain=coding,
            stage="role_workflow",
            worker_names=("direct_gpt_reasoner", "codex_gpt_coding_agent", "direct_opus_reviewer"),
            workflow=_workflow(
                _step(ix, "direct_gpt_reasoner", "Design the implementation strategy and identify likely edge cases."),
                _step(ix, "codex_gpt_coding_agent", "Implement the fix from the plan and run tests.", [0]),
                _step(ix, "direct_opus_reviewer", "Review the patch for hidden failures and quality risks.", [0, 1]),
                _step(ix, "codex_gpt_coding_agent", "Repair the patch using the review and rerun tests.", [0, 1, 2]),
            ),
            rationale="Tests direct GPT planning and direct Opus review, with Codex responsible for the final editable state.",
        ),
        ScaffoldArm(
            name="solo__terminal_gpt_agent",
            domain=terminal,
            stage="single_scaffold",
            worker_names=("terminal_gpt_agent",),
            workflow=_workflow(_step(ix, "terminal_gpt_agent", "Solve the terminal task and submit through the verifier.")),
            rationale="Fair solo baseline for GPT inside the Harbor terminal scaffold.",
        ),
        ScaffoldArm(
            name="solo__terminal_kimi_agent",
            domain=terminal,
            stage="single_scaffold",
            worker_names=("terminal_kimi_agent",),
            workflow=_workflow(_step(ix, "terminal_kimi_agent", "Solve the terminal task and submit through the verifier.")),
            rationale="Fair solo baseline for Kimi inside the Harbor terminal scaffold.",
        ),
        ScaffoldArm(
            name="solo__terminal_mimo_agent",
            domain=terminal,
            stage="single_scaffold",
            worker_names=("terminal_mimo_agent",),
            workflow=_workflow(_step(ix, "terminal_mimo_agent", "Solve the terminal task and submit through the verifier.")),
            rationale="Fair solo baseline for MiMo inside the Harbor terminal scaffold.",
        ),
        ScaffoldArm(
            name="solo__terminal_glm_agent",
            domain=terminal,
            stage="single_scaffold",
            worker_names=("terminal_glm_agent",),
            workflow=_workflow(_step(ix, "terminal_glm_agent", "Solve the terminal task and submit through the verifier.")),
            rationale="Fair solo baseline for GLM inside the Harbor terminal scaffold.",
        ),
        ScaffoldArm(
            name="terminal_gpt_plan__kimi_solve",
            domain=terminal,
            stage="role_workflow",
            worker_names=("terminal_gpt_agent", "terminal_kimi_agent"),
            workflow=_workflow(
                _step(ix, "terminal_gpt_agent", "Inspect the task and produce a concise implementation plan."),
                _step(ix, "terminal_kimi_agent", "Implement the plan, run the bundled tests, and submit.", [0]),
            ),
            rationale="Tests planner/solver separation inside Harbor terminal tasks.",
        ),
        ScaffoldArm(
            name="terminal_kimi_attempt__mimo_repair",
            domain=terminal,
            stage="role_workflow",
            worker_names=("terminal_kimi_agent", "terminal_mimo_agent"),
            workflow=_workflow(
                _step(ix, "terminal_kimi_agent", "Make a complete attempt and record the test feedback."),
                _step(ix, "terminal_mimo_agent", "Repair the attempt using the prior feedback and submit.", [0]),
            ),
            rationale="Tests Kimi as the first terminal builder with MiMo as repair worker.",
        ),
        ScaffoldArm(
            name="terminal_glm_diagnose__kimi_solve",
            domain=terminal,
            stage="role_workflow",
            worker_names=("terminal_glm_agent", "terminal_kimi_agent"),
            workflow=_workflow(
                _step(ix, "terminal_glm_agent", "Inspect the task, identify the likely solution strategy, and note edge cases."),
                _step(ix, "terminal_kimi_agent", "Implement the diagnosis, run the bundled tests, and submit.", [0]),
            ),
            rationale="Tests GLM as the structured terminal diagnostician with Kimi as implementation worker.",
        ),
        ScaffoldArm(
            name="terminal_kimi_attempt__glm_repair",
            domain=terminal,
            stage="role_workflow",
            worker_names=("terminal_kimi_agent", "terminal_glm_agent"),
            workflow=_workflow(
                _step(ix, "terminal_kimi_agent", "Make a complete attempt and record the test feedback."),
                _step(ix, "terminal_glm_agent", "Repair the attempt using the prior feedback and submit.", [0]),
            ),
            rationale="Tests GLM as the repair worker after a Kimi implementation attempt.",
        ),
        ScaffoldArm(
            name="terminal_gpt_plan__glm_solve",
            domain=terminal,
            stage="role_workflow",
            worker_names=("terminal_gpt_agent", "terminal_glm_agent"),
            workflow=_workflow(
                _step(ix, "terminal_gpt_agent", "Inspect the task and produce a concise implementation plan."),
                _step(ix, "terminal_glm_agent", "Implement the plan, run the bundled tests, and submit.", [0]),
            ),
            rationale="Tests GPT planning with GLM as the terminal solver.",
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
            name="solo__code_gpt_reasoner",
            domain=unit_code,
            stage="single_scaffold",
            worker_names=("direct_gpt_reasoner",),
            workflow=_workflow(_step(ix, "direct_gpt_reasoner", "Solve the coding task and return the final code.")),
            rationale="Fair solo baseline for GPT on fast verifier-backed code tasks.",
        ),
        ScaffoldArm(
            name="solo__code_gemini_synth",
            domain=unit_code,
            stage="single_scaffold",
            worker_names=("direct_gemini_synth",),
            workflow=_workflow(_step(ix, "direct_gemini_synth", "Solve the coding task and return the final code.")),
            rationale="Fair solo baseline for Gemini on fast verifier-backed code tasks.",
        ),
        ScaffoldArm(
            name="solo__code_opus_reviewer",
            domain=unit_code,
            stage="single_scaffold",
            worker_names=("direct_opus_reviewer",),
            workflow=_workflow(_step(ix, "direct_opus_reviewer", "Solve the coding task and return the final code.")),
            rationale="Fair solo baseline for Opus on fast verifier-backed code tasks.",
        ),
        ScaffoldArm(
            name="solo__code_flash_fast",
            domain=unit_code,
            stage="single_scaffold",
            worker_names=("direct_flash_fast",),
            workflow=_workflow(_step(ix, "direct_flash_fast", "Solve the coding task and return the final code.")),
            rationale="Fair solo baseline for Flash as a fast worker on verifier-backed code tasks.",
        ),
        ScaffoldArm(
            name="solo__code_glm_reasoner",
            domain=unit_code,
            stage="single_scaffold",
            worker_names=("direct_glm_reasoner",),
            workflow=_workflow(_step(ix, "direct_glm_reasoner", "Solve the coding task and return the final code.")),
            rationale="Fair solo baseline for GLM as an open direct worker on verifier-backed code tasks.",
        ),
        ScaffoldArm(
            name="solo__code_mimo_reasoner",
            domain=unit_code,
            stage="single_scaffold",
            worker_names=("direct_mimo_reasoner",),
            workflow=_workflow(_step(ix, "direct_mimo_reasoner", "Solve the coding task and return the final code.")),
            rationale="Fair solo baseline for MiMo as an open direct worker on verifier-backed code tasks.",
        ),
        ScaffoldArm(
            name="solo__code_minimax_reasoner",
            domain=unit_code,
            stage="single_scaffold",
            worker_names=("direct_minimax_reasoner",),
            workflow=_workflow(_step(ix, "direct_minimax_reasoner", "Solve the coding task and return the final code.")),
            rationale="Fair solo baseline for MiniMax as an open direct worker on verifier-backed code tasks.",
        ),
        ScaffoldArm(
            name="flash_code_attempt__gpt_repair",
            domain=unit_code,
            stage="role_workflow",
            worker_names=("direct_flash_fast", "direct_gpt_reasoner"),
            workflow=_workflow(
                _step(ix, "direct_flash_fast", "Produce a fast first-pass implementation."),
                _step(ix, "direct_gpt_reasoner", "Repair and finalize the implementation using the first pass.", [0]),
            ),
            rationale="Tests fast first-pass code generation with GPT repair.",
        ),
        ScaffoldArm(
            name="gpt_code__opus_critic__gpt_revise",
            domain=unit_code,
            stage="role_workflow",
            worker_names=("direct_gpt_reasoner", "direct_opus_reviewer"),
            workflow=_workflow(
                _step(ix, "direct_gpt_reasoner", "Solve the coding task and identify edge cases."),
                _step(ix, "direct_opus_reviewer", "Critique the implementation for hidden failures.", [0]),
                _step(ix, "direct_gpt_reasoner", "Revise and return the final code.", [0, 1]),
            ),
            rationale="Tests solver/critic/revise on verifier-backed code tasks.",
        ),
        ScaffoldArm(
            name="gpt_algorithm__gemini_check__gpt_final",
            domain=unit_code,
            stage="role_workflow",
            worker_names=("direct_gpt_reasoner", "direct_gemini_synth"),
            workflow=_workflow(
                _step(ix, "direct_gpt_reasoner", "Derive the algorithm and draft the implementation."),
                _step(ix, "direct_gemini_synth", "Check the algorithm against edge cases and constraints.", [0]),
                _step(ix, "direct_gpt_reasoner", "Return the final implementation after incorporating the check.", [0, 1]),
            ),
            rationale="Tests algorithm derivation plus independent constraint checking.",
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
            name="solo__direct_glm_reasoner",
            domain=direct,
            stage="single_scaffold",
            worker_names=("direct_glm_reasoner",),
            workflow=_workflow(_step(ix, "direct_glm_reasoner", "Solve the task directly.")),
            rationale="Fair solo baseline for GLM as the strongest open direct worker.",
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
        ScaffoldArm(
            name="solo__long_gemini_synth",
            domain=long_context,
            stage="single_scaffold",
            worker_names=("direct_gemini_synth",),
            workflow=_workflow(_step(ix, "direct_gemini_synth", "Answer from the provided documents only.")),
            rationale="Fair solo baseline for Gemini on text-only long-context tasks.",
        ),
        ScaffoldArm(
            name="solo__long_gpt_reasoner",
            domain=long_context,
            stage="single_scaffold",
            worker_names=("direct_gpt_reasoner",),
            workflow=_workflow(_step(ix, "direct_gpt_reasoner", "Answer from the provided documents only.")),
            rationale="Fair solo baseline for GPT on text-only long-context tasks.",
        ),
        ScaffoldArm(
            name="solo__long_opus_reviewer",
            domain=long_context,
            stage="single_scaffold",
            worker_names=("direct_opus_reviewer",),
            workflow=_workflow(_step(ix, "direct_opus_reviewer", "Answer from the provided documents only.")),
            rationale="Fair solo baseline for Opus on text-only long-context tasks.",
        ),
        ScaffoldArm(
            name="solo__long_flash_fast",
            domain=long_context,
            stage="single_scaffold",
            worker_names=("direct_flash_fast",),
            workflow=_workflow(_step(ix, "direct_flash_fast", "Answer from the provided documents only.")),
            rationale="Fair solo baseline for Flash as a fast direct worker on text-only long-context tasks.",
        ),
        ScaffoldArm(
            name="solo__long_glm_reasoner",
            domain=long_context,
            stage="single_scaffold",
            worker_names=("direct_glm_reasoner",),
            workflow=_workflow(_step(ix, "direct_glm_reasoner", "Answer from the provided documents only.")),
            rationale="Fair solo baseline for GLM as an open direct worker on text-only long-context tasks.",
        ),
        ScaffoldArm(
            name="solo__long_mimo_reasoner",
            domain=long_context,
            stage="single_scaffold",
            worker_names=("direct_mimo_reasoner",),
            workflow=_workflow(_step(ix, "direct_mimo_reasoner", "Answer from the provided documents only.")),
            rationale="Fair solo baseline for MiMo as an open direct worker on text-only long-context tasks.",
        ),
        ScaffoldArm(
            name="solo__long_minimax_reasoner",
            domain=long_context,
            stage="single_scaffold",
            worker_names=("direct_minimax_reasoner",),
            workflow=_workflow(_step(ix, "direct_minimax_reasoner", "Answer from the provided documents only.")),
            rationale="Fair solo baseline for MiniMax as an open direct worker on text-only long-context tasks.",
        ),
        ScaffoldArm(
            name="long_flash_scan__gemini_synth",
            domain=long_context,
            stage="role_workflow",
            worker_names=("direct_flash_fast", "direct_gemini_synth"),
            workflow=_workflow(
                _step(ix, "direct_flash_fast", "Extract the likely relevant evidence from the documents."),
                _step(ix, "direct_gemini_synth", "Use the extracted evidence to produce the final answer.", [0]),
            ),
            rationale="Tests fast evidence extraction with Gemini synthesis.",
        ),
        ScaffoldArm(
            name="long_gpt_extract__gemini_verify__opus_final",
            domain=long_context,
            stage="role_workflow",
            worker_names=("direct_gpt_reasoner", "direct_gemini_synth", "direct_opus_reviewer"),
            workflow=_workflow(
                _step(ix, "direct_gpt_reasoner", "Extract and organize the relevant document evidence."),
                _step(ix, "direct_gemini_synth", "Verify the evidence and flag contradictions.", [0]),
                _step(ix, "direct_opus_reviewer", "Produce the final answer using only verified evidence.", [0, 1]),
            ),
            rationale="Tests extraction, verification, and final synthesis on long-context tasks.",
        ),
        ScaffoldArm(
            name="long_gemini_answer__opus_review",
            domain=long_context,
            stage="role_workflow",
            worker_names=("direct_gemini_synth", "direct_opus_reviewer"),
            workflow=_workflow(
                _step(ix, "direct_gemini_synth", "Answer from the provided documents only."),
                _step(ix, "direct_opus_reviewer", "Review the answer against the documents and finalize it.", [0]),
            ),
            rationale="Tests Gemini long-context synthesis with Opus final quality control.",
        ),
    ]

    for arm in arms:
        validate_workflow(arm.workflow, worker_count=len(workers))
    return arms


def build_plan(task_mix: dict[str, int] | None = None) -> dict[str, Any]:
    mix = _normalize_task_mix(task_mix)
    workers = canonical_workers()
    worker_name_by_id = {worker.worker_id: worker.name for worker in workers}
    arms = canonical_arms(workers)

    arm_rows = [arm.model_dump(worker_name_by_id) for arm in arms]
    arms_by_domain: dict[str, list[ScaffoldArm]] = defaultdict(list)
    for arm in arms:
        arms_by_domain[arm.domain].append(arm)
    arm_domains_by_lane = {
        REPO_LANE: [REPO_ARM_DOMAIN, TERMINAL_ARM_DOMAIN],
        UNIT_CODE_LANE: [UNIT_CODE_LANE],
        DIRECT_LANE: [DIRECT_LANE],
        TOOL_LANE: [TOOL_LANE],
        LONG_CONTEXT_LANE: [LONG_CONTEXT_LANE],
    }
    rollouts_by_domain = {}
    calls_by_domain = {}
    for lane, n_tasks in mix.items():
        lane_arms = [arm for domain in arm_domains_by_lane.get(lane, [lane]) for arm in arms_by_domain.get(domain, [])]
        rollouts_by_domain[lane] = n_tasks * len(lane_arms)
        calls_by_domain[lane] = n_tasks * sum(arm.worker_calls for arm in lane_arms)

    return {
        "version": "scaffold_tournament_v1",
        "purpose": "Preregister the quality-first, scaffold-aware Ultra role tournament before live calls.",
        "task_mix": mix,
        "workers": [worker.model_dump() for worker in workers],
        "worker_harnesses": worker_harness_map(workers),
        "arms": arm_rows,
        "lanes": sorted(mix),
        "arm_domains": sorted(arms_by_domain),
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


def _repo_taskspec_candidates(manifest_dir: Path, seed: int) -> list[dict[str, Any]]:
    paths = [
        manifest_dir / "trace_capture" / "branch_taskspecs.jsonl",
        manifest_dir / "generated_repo_tasks" / "taskspecs.jsonl",
    ]
    rnd = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_jsonl(path):
            try:
                task = TaskSpec.model_validate(row)
            except Exception:
                continue
            if task.source.policy != "train_allowed" or task.splitting.split != "grpo_train":
                continue
            if task.capability != "agentic_coding":
                continue
            tags = list(task.metadata.tags)
            diagnostic: dict[str, Any] = {
                "source_policy": task.source.policy,
                "split": task.splitting.split,
            }
            for asset in task.input.assets:
                if isinstance(asset, dict) and isinstance(asset.get("trace_branch"), dict):
                    trace_branch = asset["trace_branch"]
                    diagnostic.update(
                        {
                            "origin_harness": trace_branch.get("origin_harness"),
                            "previous_success": trace_branch.get("previous_success"),
                            "worker_model": trace_branch.get("worker_model"),
                        }
                    )
                    tags.append("trace_branch")
            candidates.append(
                {
                    "tournament_task_id": f"coding_repo::{task.source.name}::{task.task_id}",
                    "domain": "coding_repo",
                    "source": task.source.name,
                    "source_task_id": task.task_id,
                    "task_jsonl": str(path),
                    "harness": task.environment.harness,
                    "split": task.splitting.split,
                    "selection_tags": sorted(set([*tags, "train_allowed", "repo_task"])),
                    "diagnostic": diagnostic,
                    "_rank": (0.0 if task.source.name == "trace_state_branches" else 1.0, rnd.random(), task.task_id),
                }
            )
    return candidates


def _task_lane(task: TaskSpec) -> str:
    source = task.source.name
    harness = task.environment.harness
    domain = task.metadata.domain
    if source in TASKTROVE_UNIT_CODE_SOURCES:
        return UNIT_CODE_LANE
    if source in TASKTROVE_REPO_TERMINAL_SOURCES:
        return REPO_LANE
    if source in {"generated_repo_tasks"}:
        return REPO_LANE
    if task.grader.type == "swebench_verified_hidden_tests":
        return REPO_LANE
    if source == "trace_state_branches":
        return TRACE_BRANCH_LANE
    if harness == "code_exec" or domain == "code":
        return UNIT_CODE_LANE
    if harness == "direct_qa":
        return DIRECT_LANE
    if harness in {"tool_dialog", "tau_bench"}:
        return TOOL_LANE
    if harness == "long_context":
        return LONG_CONTEXT_LANE
    if harness == "terminal_sandbox":
        return REPO_LANE
    return "other"


def _arm_domain_for_task(task: TaskSpec, lane: str) -> str:
    if lane == TRACE_BRANCH_LANE:
        return REPO_ARM_DOMAIN
    if task.environment.harness == "terminal_sandbox":
        return TERMINAL_ARM_DOMAIN
    if lane == REPO_LANE:
        return REPO_ARM_DOMAIN
    return lane


def _stable_rank(seed: int, lane: str, task: TaskSpec) -> str:
    payload = f"scaffold-discovery:{seed}:{lane}:{task.source.name}:{task.task_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _candidate_from_taskspec(task: TaskSpec, *, lane: str, task_jsonl: Path, seed: int) -> dict[str, Any]:
    tags = sorted(
        set(
            [
                *task.metadata.tags,
                task.source.policy,
                task.splitting.split,
                "fixed_workflow_discovery",
            ]
        )
    )
    diagnostic: dict[str, Any] = {
        "source_policy": task.source.policy,
        "split": task.splitting.split,
        "contamination_group": task.splitting.contamination_group,
    }
    for asset in task.input.assets:
        if isinstance(asset, dict) and isinstance(asset.get("trace_branch"), dict):
            trace_branch = asset["trace_branch"]
            diagnostic.update(
                {
                    "origin_harness": trace_branch.get("origin_harness"),
                    "previous_success": trace_branch.get("previous_success"),
                    "worker_model": trace_branch.get("worker_model"),
                }
            )
            tags.append("trace_branch")
    return {
        "tournament_task_id": f"{lane}::{task.source.name}::{task.task_id}",
        "lane": lane,
        "arm_domain": _arm_domain_for_task(task, lane),
        "source": task.source.name,
        "source_task_id": task.task_id,
        "task_jsonl": str(task_jsonl),
        "harness": task.environment.harness,
        "split": task.splitting.split,
        "capability": task.capability,
        "metadata_domain": task.metadata.domain,
        "metadata_subdomain": task.metadata.subdomain,
        "selection_tags": sorted(set(tags)),
        "diagnostic": diagnostic,
        "_source": task.source.name,
        "_domain": str(task.metadata.domain or "unknown"),
        "_subdomain": str(task.metadata.subdomain or "unknown"),
        "_rank": _stable_rank(seed, lane, task),
    }


def _read_taskspecs(path: Path) -> list[TaskSpec]:
    if not path.exists():
        return []
    rows: list[TaskSpec] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(TaskSpec.model_validate(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid TaskSpec in {path}:{line_no}: {exc}") from exc
    return rows


def _select_balanced_candidates(
    rows: list[dict[str, Any]],
    *,
    requested: int,
    balance_key: str,
) -> tuple[list[dict[str, Any]], int]:
    if requested <= 0:
        return [], 0
    if len(rows) <= requested:
        return sorted(rows, key=lambda row: row["_rank"]), max(0, requested - len(rows))

    labels = sorted({str(row.get(balance_key) or "unknown") for row in rows})
    target_counts = _balanced_counts(requested, labels)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get(balance_key) or "unknown")].append(row)
    for label_rows in by_label.values():
        label_rows.sort(key=lambda row: row["_rank"])

    selected: list[dict[str, Any]] = []
    for label in labels:
        selected.extend(by_label[label][: target_counts[label]])
    selected_ids = {row["tournament_task_id"] for row in selected}
    remainder = sorted(
        [row for row in rows if row["tournament_task_id"] not in selected_ids],
        key=lambda row: row["_rank"],
    )
    selected.extend(remainder[: max(0, requested - len(selected))])
    return selected[:requested], 0


def _select_mvp_tasks(
    manifest_dir: Path,
    *,
    tasks_jsonl: Path | None,
    task_mix: dict[str, int],
    seed: int,
    include_pool_validation: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    path = tasks_jsonl or manifest_dir / "data_mix" / "mvp_grpo_train_taskspecs.jsonl"
    tasks = []
    for task in _read_taskspecs(path):
        if task.source.policy == "train_allowed" and task.splitting.split == "grpo_train":
            tasks.append(task)
        elif (
            include_pool_validation
            and tasks_jsonl is not None
            and task.source.policy == "pool_only"
            and task.splitting.split in {"pool_discovery", "pool_validation", "online_validation"}
        ):
            tasks.append(task)
    by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        lane = _task_lane(task)
        if lane in task_mix:
            by_lane[lane].append(_candidate_from_taskspec(task, lane=lane, task_jsonl=path, seed=seed))

    balance_keys = {
        REPO_LANE: "_source",
        UNIT_CODE_LANE: "_source",
        DIRECT_LANE: "_domain",
        TOOL_LANE: "_subdomain",
        LONG_CONTEXT_LANE: "_source",
    }
    selected: list[dict[str, Any]] = []
    deficits: dict[str, int] = {}
    for lane, requested in task_mix.items():
        lane_selected, deficit = _select_balanced_candidates(
            by_lane.get(lane, []),
            requested=requested,
            balance_key=balance_keys.get(lane, "_source"),
        )
        selected.extend(lane_selected)
        deficits[lane] = deficit
    return selected, deficits


def _select_trace_branch_shard(manifest_dir: Path, *, branch_tasks_jsonl: Path | None, seed: int) -> list[dict[str, Any]]:
    path = branch_tasks_jsonl or manifest_dir / "trace_capture" / "branch_taskspecs.jsonl"
    selected = []
    for task in _read_taskspecs(path):
        if task.source.policy != "train_allowed" or task.splitting.split != "grpo_train":
            continue
        lane = TRACE_BRANCH_LANE
        selected.append(_candidate_from_taskspec(task, lane=lane, task_jsonl=path, seed=seed))
    return sorted(selected, key=lambda row: row["_rank"])


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
    train_allowed = sorted(_repo_taskspec_candidates(manifest_dir, seed), key=lambda row: row["_rank"])
    if len(train_allowed) >= requested:
        trace = [row for row in train_allowed if row["source"] == "trace_state_branches"]
        generated = [row for row in train_allowed if row["source"] == "generated_repo_tasks"]
        if trace and generated and requested > 1:
            trace_take = min(len(trace), max(1, round(requested * 0.7)))
            generated_take = min(len(generated), max(1, requested - trace_take))
            selected = [*trace[:trace_take], *generated[:generated_take]]
            selected_ids = {row["tournament_task_id"] for row in selected}
            remainder = [row for row in train_allowed if row["tournament_task_id"] not in selected_ids]
            selected.extend(remainder[: max(0, requested - len(selected))])
            return selected[:requested], 0
        return train_allowed[:requested], 0

    live = [
        row
        for row in _coding_candidates(manifest_dir)
        if row["source_task_id"] not in {task["source_task_id"] for task in train_allowed}
    ]
    selected = [*train_allowed, *live[: max(0, requested - len(train_allowed))]]
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
    tasks_jsonl: Path | None = None,
    branch_tasks_jsonl: Path | None = None,
    include_pool_validation: bool = False,
) -> dict[str, Any]:
    manifest_dir = manifest_dir or default_manifest_dir()
    mix = _normalize_task_mix(task_mix)
    plan = build_plan(mix)
    arms = canonical_arms(canonical_workers())
    arms_by_domain: dict[str, list[ScaffoldArm]] = defaultdict(list)
    for arm in arms:
        arms_by_domain[arm.domain].append(arm)

    selected, deficits = _select_mvp_tasks(
        manifest_dir,
        tasks_jsonl=tasks_jsonl,
        task_mix=mix,
        seed=seed + 11,
        include_pool_validation=include_pool_validation,
    )
    branch_selected = _select_trace_branch_shard(
        manifest_dir,
        branch_tasks_jsonl=branch_tasks_jsonl,
        seed=seed + 17,
    )
    tasks = [
        *(_strip_internal(row) for row in selected),
        *(_strip_internal(row) for row in branch_selected),
    ]
    workers = canonical_workers()
    jobs = []
    for task in tasks:
        task_harness = str(task["harness"])
        harness_map = worker_harness_map(workers, task_harness=task_harness)
        for arm in arms_by_domain[task["arm_domain"]]:
            jobs.append(
                {
                    "job_id": f"job_{len(jobs):05d}",
                    "tournament_task_id": task["tournament_task_id"],
                    "lane": task["lane"],
                    "arm_domain": task["arm_domain"],
                    "source": task["source"],
                    "source_task_id": task["source_task_id"],
                    "source_policy": task.get("diagnostic", {}).get("source_policy"),
                    "task_jsonl": task.get("task_jsonl"),
                    "task_harness": task_harness,
                    "task_split": task.get("split"),
                    "arm": arm.name,
                    "stage": arm.stage,
                    "worker_names": list(arm.worker_names),
                    "worker_harnesses": [harness_map[name] for name in arm.worker_names],
                    "worker_harness_map": {name: harness_map[name] for name in arm.worker_names},
                    "worker_calls": arm.worker_calls,
                }
            )

    selected_counts = dict(Counter(task["lane"] for task in tasks))
    selected_arm_domain_counts = dict(Counter(task["arm_domain"] for task in tasks))
    blocked = []
    coding_deficit = deficits.get(REPO_LANE, 0)
    if coding_deficit:
        blocked.append(
            f"Need {coding_deficit} more train-allowed repo/open-repo/terminal TaskSpecs "
            "to reach the requested discovery count; Deep SWE remains final-eval-only."
        )
    for lane, deficit in sorted(deficits.items()):
        if lane == REPO_LANE or not deficit:
            continue
        blocked.append(f"Need {deficit} more {lane} TaskSpecs to reach the requested discovery count.")

    return {
        "version": "scaffold_tournament_manifest_v1",
        "seed": seed,
        "manifest_dir": str(manifest_dir),
        "tasks_jsonl": str((tasks_jsonl or manifest_dir / "data_mix" / "mvp_grpo_train_taskspecs.jsonl")),
        "branch_tasks_jsonl": str((branch_tasks_jsonl or manifest_dir / "trace_capture" / "branch_taskspecs.jsonl")),
        "include_pool_validation": include_pool_validation,
        "plan_version": plan["version"],
        "requested_task_mix": mix,
        "selected_task_counts": selected_counts,
        "selected_arm_domain_counts": selected_arm_domain_counts,
        "branch_shard_count": len(branch_selected),
        "deficits": deficits,
        "blocked_reasons": blocked,
        "tasks": tasks,
        "jobs": jobs,
        "job_count": len(jobs),
        "worker_call_count": sum(job["worker_calls"] for job in jobs),
        "default_worker_harnesses": worker_harness_map(workers),
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
    tasks_jsonl: Path | None = None,
    branch_tasks_jsonl: Path | None = None,
    include_pool_validation: bool = False,
) -> dict[str, Any]:
    manifest = build_concrete_manifest(
        manifest_dir,
        task_mix=task_mix,
        seed=seed,
        tasks_jsonl=tasks_jsonl,
        branch_tasks_jsonl=branch_tasks_jsonl,
        include_pool_validation=include_pool_validation,
    )
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
        "claude_code": "ready_cli_required",
        "codex": "ready_cli_required",
        "tool_dialog": "ready",
        "tau_bench": "ready_optional_package",
        "terminal": "ready",
    }

    jobs_by_status: Counter[str] = Counter()
    jobs_by_lane_status: Counter[tuple[str, str]] = Counter()
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
        if statuses <= {"ready", "ready_payload_required", "ready_cli_required"}:
            status = "ready"
            ready_jobs.append(job["job_id"])
        elif "unknown_backend" in statuses:
            status = "unknown_backend"
        elif "harness_pending" in statuses:
            status = "harness_pending"
        elif "ready_cli_required" in statuses:
            status = "cli_canary_required"
        else:
            status = "mixed_pending"
        jobs_by_status[status] += 1
        jobs_by_lane_status[(str(job.get("lane") or job.get("domain")), status)] += 1
        if status != "ready" and len(blocked_examples) < 10:
            blocked_examples.append(
                {
                    "job_id": job["job_id"],
                    "lane": job.get("lane") or job.get("domain"),
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
        "jobs_by_lane_status": {
            f"{domain}:{status}": count
            for (domain, status), count in sorted(jobs_by_lane_status.items())
        },
        "jobs_by_domain_status": {
            f"{domain}:{status}": count
            for (domain, status), count in sorted(jobs_by_lane_status.items())
        },
        "worker_calls_by_backend": dict(jobs_by_backend),
        "worker_calls_by_harness": dict(jobs_by_harness),
        "ready_job_ids_sample": ready_jobs[:20],
        "blocked_examples": blocked_examples,
        "backend_status": backend_status,
        "next_blockers": [
            "Run fixed-workflow discovery on selected train-allowed repo, tool-dialogue, and direct tasks.",
            "Use trace-state branch TaskSpecs and generated repo tasks for coding discovery; Deep SWE remains final-eval-only.",
            "Use the per-job worker_harness_map when executing live discovery jobs.",
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
