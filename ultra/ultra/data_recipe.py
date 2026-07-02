"""Concrete data-recipe artifacts for the Fugu-Ultra task registry.

``MISSION.md`` is the human-readable source of truth. This module turns that recipe
into machine-readable source manifests plus the first promoted TaskSpec shard from the
existing direct bank.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .policy import allowed_splits
from .policy import policy_allows_split
from .registry import RegistryError, TaskRegistry
from .schemas import SourceManifest, SourcePolicy, SplitPolicy, TaskSpec
from .sources.existing_bank import ExistingBankAdapter

DATA_RECIPE_VERSION = "fugu_ultra_data_recipe_v1"


def _counter_json(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _forbidden_uses(policy: SourcePolicy) -> list[str]:
    if policy == "train_allowed":
        return ["final_eval_claim_without_holdout"]
    if policy == "pool_only":
        return ["grpo_train", "online_validation", "final_eval_claim"]
    if policy == "online_validation":
        return ["pool_discovery", "pool_validation", "grpo_train", "final_eval_claim"]
    if policy == "final_eval_only":
        return ["pool_discovery", "pool_validation", "grpo_train", "online_validation", "prompt_tuning"]
    return ["pool_discovery", "pool_validation", "grpo_train", "online_validation", "final_eval_claim"]


SOURCE_LANES: tuple[dict[str, Any], ...] = (
    {
        "source_name": "existing_bank",
        "source_type": "router_bank",
        "version": "fugu_clean_v1",
        "policy": "train_allowed",
        "license": "mixed-public",
        "family": "direct_math_code_science_general",
        "harnesses": ["direct_qa", "code_exec"],
        "artifact": "manifest.jsonl",
        "status": "materialized",
        "split_policy": "source_family",
        "notes": "Router-curated bank; direct-only tasks are one curriculum lane, not Ultra proof.",
    },
    {
        "source_name": "training_repo_canary",
        "source_type": "generated_repo",
        "version": "v1",
        "policy": "train_allowed",
        "family": "repo_repair",
        "harnesses": ["opencode", "codex", "claude_code"],
        "artifact": "training_repo_canaries/taskspecs.jsonl",
        "status": "diagnostic_materialized",
        "split_policy": "template",
        "notes": "Generated train-distribution canary; useful for harness smoke tests, not training volume.",
    },
    {
        "source_name": "generated_repo_tasks",
        "source_type": "generated_repo",
        "version": "v1",
        "policy": "train_allowed",
        "family": "repo_repair",
        "harnesses": ["opencode", "codex", "claude_code"],
        "artifact": "generated_repo_tasks/taskspecs.jsonl",
        "status": "materialized",
        "split_policy": "template",
        "notes": "Small train-allowed repository-repair tasks for fast/medium workflow training.",
    },
    {
        "source_name": "trace_state_branches",
        "source_type": "derived_trace_branch",
        "version": "v1",
        "policy": "train_allowed",
        "family": "repo_repair",
        "harnesses": ["opencode", "codex", "claude_code"],
        "artifact": "trace_capture/branch_taskspecs.jsonl",
        "status": "partial_materialized",
        "split_policy": "base_task_trace",
        "notes": "State-branch repair tasks derived from train-allowed OpenCode/Codex/Claude AgentTraces with patch, workspace, and verifier artifacts.",
    },
    {
        "source_name": "swe_smith",
        "source_type": "public_generated_repo",
        "version": "train",
        "policy": "train_allowed",
        "license": "see-source",
        "family": "repo_repair",
        "harnesses": ["opencode", "codex", "claude_code"],
        "status": "payload_pending_optional",
        "split_policy": "repo_family",
        "notes": "Future train-safe repo-repair source; current MVP excludes saved rows without reusable payloads.",
    },
    {
        "source_name": "github_issue",
        "source_type": "mined_repo",
        "version": "v1",
        "policy": "train_allowed",
        "family": "repo_repair",
        "harnesses": ["opencode", "codex", "claude_code"],
        "status": "pending",
        "split_policy": "repo_family",
        "notes": "Custom mined issues require license, determinism, leakage, and hidden-test gates.",
    },
    {
        "source_name": "deep_swe_local",
        "source_type": "local_hard_repo_eval",
        "version": "local",
        "policy": "final_eval_only",
        "family": "repo_repair",
        "harnesses": ["opencode", "codex", "claude_code"],
        "artifact": "scaffold_repo_taskspecs.jsonl",
        "status": "eval_materialized",
        "split_policy": "source_family",
        "notes": "Hard target distribution; never use for routine canaries, pool selection, or GRPO.",
    },
    {
        "source_name": "swe_bench_verified",
        "source_type": "public_repo_eval",
        "version": "test",
        "policy": "final_eval_only",
        "license": "see-source",
        "family": "repo_repair",
        "harnesses": ["opencode", "codex", "claude_code"],
        "status": "held_out",
        "split_policy": "source_family",
        "notes": "Final held-out repo benchmark only.",
    },
    {
        "source_name": "terminal_custom",
        "source_type": "generated_terminal",
        "version": "v1",
        "policy": "train_allowed",
        "family": "terminal_system",
        "harnesses": ["terminal_sandbox"],
        "status": "pending",
        "split_policy": "template",
        "notes": "Train-safe Terminal-Bench-style Docker tasks with deterministic tests.",
    },
    {
        "source_name": "tasktrove_harbor",
        "source_type": "harbor_task_bundle",
        "version": "v3",
        "policy": "pool_only",
        "license": "see-source",
        "family": "terminal_system",
        "harnesses": ["terminal_sandbox"],
        "status": "umbrella_pending",
        "split_policy": "source_family",
        "notes": "Umbrella lane for future Harbor subsets; each concrete subset must pass validation gates separately.",
    },
    {
        "source_name": "tasktrove_inferredbugs",
        "source_type": "harbor_task_bundle",
        "version": "v3",
        "policy": "train_allowed",
        "license": "apache-2.0",
        "family": "terminal_system",
        "harnesses": ["terminal_sandbox"],
        "artifact": "tasktrove_harbor/inferredbugs_train_taskspecs.jsonl",
        "status": "materialized",
        "split_policy": "source_family",
        "notes": "Verifier-backed TaskTrove subset; Apache-2.0 dataset; no-model and model-backed Harbor canaries passed.",
    },
    {
        "source_name": "tasktrove_pymethods2test",
        "source_type": "harbor_task_bundle",
        "version": "v3",
        "policy": "train_allowed",
        "license": "apache-2.0",
        "family": "terminal_system",
        "harnesses": ["terminal_sandbox"],
        "artifact": "tasktrove_harbor/pymethods2test_train_taskspecs.jsonl",
        "status": "materialized",
        "split_policy": "source_family",
        "notes": "Verifier-backed TaskTrove pymethods2test subset; prioritized because the OT-Agent paper found this source strongest for RL.",
    },
    {
        "source_name": "agenttrove_traces",
        "source_type": "agent_trace_corpus",
        "version": "v1",
        "policy": "diagnostic_only",
        "license": "apache-2.0",
        "family": "trace_sft",
        "harnesses": ["terminal_sandbox"],
        "artifact": "agenttrove_hf/inspection_report.json",
        "status": "inspected_not_task_registry",
        "split_policy": "trace_source_family",
        "notes": "AgentTrove is useful for trace SFT, workflow priors, and role mining; do not use for GRPO until tasks are joined to verifiers.",
    },
    {
        "source_name": "livecodebench_old",
        "source_type": "public_code",
        "version": "old_windows",
        "policy": "train_allowed",
        "license": "see-source",
        "family": "unit_code",
        "harnesses": ["code_exec"],
        "status": "adapter_ready",
        "split_policy": "temporal",
        "notes": "Use older windows only; latest/future windows stay held out.",
    },
    {
        "source_name": "livecodebench_latest",
        "source_type": "public_code_eval",
        "version": "latest_or_future",
        "policy": "final_eval_only",
        "license": "see-source",
        "family": "unit_code",
        "harnesses": ["code_exec"],
        "status": "held_out",
        "split_policy": "temporal",
        "notes": "Held-out code generation evaluation.",
    },
    {
        "source_name": "bigcodebench",
        "source_type": "public_code",
        "version": "train_dev",
        "policy": "train_allowed",
        "license": "see-source",
        "family": "unit_code",
        "harnesses": ["code_exec"],
        "status": "adapter_ready",
        "split_policy": "source_family",
        "notes": "Train/dev only if not reporting on the exact split.",
    },
    {
        "source_name": "scicode_dev",
        "source_type": "public_scientific_code",
        "version": "dev_or_generated",
        "policy": "train_allowed",
        "license": "see-source",
        "family": "scientific_code",
        "harnesses": ["code_exec"],
        "status": "adapter_ready",
        "split_policy": "source_family",
        "notes": "Use dev/generated analogs; canonical test remains held out.",
    },
    {
        "source_name": "math_train",
        "source_type": "public_math",
        "version": "train",
        "policy": "train_allowed",
        "license": "see-source",
        "family": "math_science_knowledge",
        "harnesses": ["direct_qa"],
        "status": "adapter_pending",
        "split_policy": "source_family",
        "notes": "Represents MATH train and old AIME-style lanes.",
    },
    {
        "source_name": "numina_math",
        "source_type": "public_math",
        "version": "train",
        "policy": "train_allowed",
        "license": "see-source",
        "family": "math_science_knowledge",
        "harnesses": ["direct_qa"],
        "status": "adapter_ready",
        "split_policy": "source_family",
        "notes": "Existing direct-bank source plus standalone adapter lane.",
    },
    {
        "source_name": "omni_math",
        "source_type": "public_math",
        "version": "train",
        "policy": "train_allowed",
        "license": "see-source",
        "family": "math_science_knowledge",
        "harnesses": ["direct_qa"],
        "status": "adapter_ready",
        "split_policy": "source_family",
        "notes": "Math train lane.",
    },
    {
        "source_name": "mmlu_pro",
        "source_type": "public_knowledge",
        "version": "train_dev",
        "policy": "train_allowed",
        "license": "see-source",
        "family": "math_science_knowledge",
        "harnesses": ["direct_qa"],
        "status": "adapter_ready",
        "split_policy": "source_family",
        "notes": "Use train/dev subsets only.",
    },
    {
        "source_name": "gpqa_diamond",
        "source_type": "public_science_eval",
        "version": "official",
        "policy": "final_eval_only",
        "license": "see-source",
        "family": "math_science_knowledge",
        "harnesses": ["direct_qa"],
        "status": "held_out",
        "split_policy": "source_family",
        "notes": "Prefer held out for final reporting.",
    },
    {
        "source_name": "hle",
        "source_type": "public_hard_eval",
        "version": "official",
        "policy": "final_eval_only",
        "license": "see-source",
        "family": "math_science_knowledge",
        "harnesses": ["direct_qa"],
        "status": "held_out",
        "split_policy": "source_family",
        "notes": "Prefer held out for final reporting.",
    },
    {
        "source_name": "tau_custom",
        "source_type": "generated_tool_dialogue",
        "version": "v1",
        "policy": "train_allowed",
        "family": "tool_dialogue",
        "harnesses": ["tool_dialog"],
        "artifact": "tool_dialog_tasks/taskspecs.jsonl",
        "status": "materialized",
        "split_policy": "domain_seed",
        "notes": "Custom tau-style domains/seeds are train-allowed; reported official eval seeds are held out.",
    },
    {
        "source_name": "longctx_generated",
        "source_type": "generated_long_context",
        "version": "v1",
        "policy": "train_allowed",
        "family": "long_context_memory_planning",
        "harnesses": ["long_context"],
        "artifact": "long_context_tasks/taskspecs.jsonl",
        "status": "materialized",
        "split_policy": "document_template",
        "notes": "Generated document packs and MRCR/Michelangelo-style tasks.",
    },
    {
        "source_name": "sequential_sim",
        "source_type": "generated_simulator",
        "version": "v1",
        "policy": "train_allowed",
        "family": "sequential_research_loop",
        "harnesses": ["sequential_sim"],
        "status": "late_stage",
        "split_policy": "template",
        "notes": "Late-stage planning simulators, used after cheaper sources are stable.",
    },
    {
        "source_name": "role_probe",
        "source_type": "derived_diagnostic",
        "version": "v1",
        "policy": "diagnostic_only",
        "family": "role_probe",
        "harnesses": ["direct_qa", "code_exec", "opencode", "terminal_sandbox", "tool_dialog"],
        "status": "diagnostic",
        "split_policy": "base_task",
        "notes": "Pool-selection and curriculum diagnostics; not direct SFT labels for the Conductor.",
    },
)

TARGET_REGISTRY_SIZES = {
    "syntax_topology_sft": "500-1500 examples, no worker execution",
    "pool_discovery": "400-600 tasks",
    "pool_validation": "250-400 tasks",
    "grpo_train": "800-1200 candidate tasks; first pilot sampled after fixed-workflow discovery",
    "online_validation": "150-250 tasks",
    "final_eval": "300-500 untouched tasks",
}

GRPO_TRAIN_MVP_MIX = {
    "repo_repair_open_repo_terminal": 250,
    "unit_and_scientific_code": 225,
    "math_science_knowledge": 250,
    "tool_dialogue": 150,
    "long_context_memory_planning": 125,
}

GRPO_TRAIN_MVP_NOTES = [
    "The first MVP is text/tool/repo only.",
    "The 1,000-row mix is a candidate train distribution pending live fixed-workflow discovery, not the final hard-coding Ultra mix.",
    "The task mix totals 1,000 examples across repo, code, reasoning, dialogue, and long-context lanes.",
    "Use a small validated task-source mix before broad expansion; TaskTrove contributes only verifier-backed TaskSpecs.",
    "Prioritize pymethods2test as a fixed RL anchor because the OT-Agent source ablation found it strongest.",
    "Treat pymethods2test-style Python contracts as a high-signal fast-verifiable backbone, not as a replacement for repo/tool tasks.",
    "Keep heterogeneous tool/terminal/dialogue sources for OOD generalization even when their ID scores are weaker.",
    "AgentTrove traces are SFT/role-mining material, not reward-bearing TaskSpecs without verifier reconstruction.",
    "Build the first GRPO pilot from tasks with observed workflow disagreement or headroom, not blindly from every candidate row.",
    "Increase true repo/harness task volume toward 50-100 validated repo or branch TaskSpecs before making coding-agentic progress claims.",
]

TASKTROVE_VALIDATION_GATES = [
    "reward is emitted for success, failure, invalid output, and timeout",
    "failure produces reward=0 or the documented failure score",
    "timeout is classified explicitly",
    "grader is deterministic over repeated local runs",
    "task setup is reproducible from the bundled environment",
    "invalid output is not silently dropped",
]

FIXED_WORKFLOW_DISCOVERY_GATE = {
    "status": "required_before_grpo",
    "subset_size": "150-250 stratified tasks plus trace-branch shard",
    "templates": [
        "single worker",
        "planner -> solver",
        "solver -> critic -> revise",
        "two independent solvers -> synthesizer",
        "builder -> debugger -> repair",
        "tool/dialogue primary -> verifier/repairer",
        "long-context retriever -> synthesizer",
    ],
    "proceed_if": [
        "best fixed workflow beats best single worker on at least one product-critical lane, or workflow-oracle headroom is positive",
        "at least 35-50% of sampled task groups show reward variation among workflows",
    ],
}

IMMEDIATE_BUILD_ORDER = [
    "Emit source manifests from the data recipe.",
    "Promote existing direct-bank rows into canonical TaskSpec v2.",
    "Add split/contamination registry artifacts.",
    "Add validation report artifacts.",
    "Regenerate the 1,000-row balanced MVP candidate train distribution after source updates.",
    "Run source validation and difficulty calibration before live discovery.",
    "Use the MVP candidate distribution for fixed-workflow discovery before adding more source volume.",
    "Build the first GRPO pilot only from validated tasks with observed workflow disagreement or headroom.",
    "Import additional verified train-allowed repo/code/science/terminal sources after the MVP path is validated.",
    "Collect richer train-allowed OpenCode/Codex/Claude Code traces for state-level branches.",
]

TASK_SHARDS = (
    {
        "name": "existing_bank_promoted",
        "path": "data_mix/existing_bank_taskspecs.jsonl",
        "expected_sources": ["existing_bank"],
        "role": "grpo_train_and_online_validation",
    },
    {
        "name": "training_repo_canary",
        "path": "training_repo_canaries/taskspecs.jsonl",
        "expected_sources": ["training_repo_canary"],
        "role": "diagnostic_canary",
    },
    {
        "name": "generated_repo_tasks",
        "path": "generated_repo_tasks/taskspecs.jsonl",
        "expected_sources": ["generated_repo_tasks"],
        "role": "grpo_train_repo_repair",
    },
    {
        "name": "trace_state_branch_tasks",
        "path": "trace_capture/branch_taskspecs.jsonl",
        "expected_sources": ["trace_state_branches"],
        "role": "grpo_train_repo_repair_branching",
    },
    {
        "name": "deep_swe_eval",
        "path": "scaffold_repo_taskspecs.jsonl",
        "expected_sources": ["deep_swe_local"],
        "role": "final_eval_only",
    },
    {
        "name": "tasktrove_inferredbugs_train",
        "path": "tasktrove_harbor/inferredbugs_train_taskspecs.jsonl",
        "expected_sources": ["tasktrove_inferredbugs"],
        "role": "grpo_train_terminal",
    },
    {
        "name": "tasktrove_pymethods2test_train",
        "path": "tasktrove_harbor/pymethods2test_train_taskspecs.jsonl",
        "expected_sources": ["tasktrove_pymethods2test"],
        "role": "grpo_train_terminal",
    },
    {
        "name": "tool_dialog_tasks",
        "path": "tool_dialog_tasks/taskspecs.jsonl",
        "expected_sources": ["tau_custom"],
        "role": "grpo_train_tool_dialogue",
    },
    {
        "name": "long_context_tasks",
        "path": "long_context_tasks/taskspecs.jsonl",
        "expected_sources": ["longctx_generated"],
        "role": "grpo_train_long_context",
    },
)


def build_source_manifests(manifest_dir: Path) -> list[SourceManifest]:
    manifest_dir = manifest_dir.resolve()
    out: list[SourceManifest] = []
    for lane in SOURCE_LANES:
        policy = lane["policy"]
        artifact = lane.get("artifact")
        artifact_path = manifest_dir / artifact if artifact else None
        out.append(
            SourceManifest(
                source_name=lane["source_name"],
                source_type=lane["source_type"],
                version=lane["version"],
                license=lane.get("license"),
                allowed_uses=allowed_splits(policy),
                forbidden_uses=_forbidden_uses(policy),
                raw_artifact_hash=_sha256_file(artifact_path) if artifact_path else None,
                split_policy=SplitPolicy(
                    type=lane["split_policy"],
                    notes=lane["notes"],
                ),
                known_issues=[lane["notes"]],
            )
        )
    return out


def build_source_registry(manifest_dir: Path) -> dict[str, Any]:
    manifest_dir = manifest_dir.resolve()
    policies = Counter(str(lane["policy"]) for lane in SOURCE_LANES)
    families = Counter(str(lane["family"]) for lane in SOURCE_LANES)
    statuses = Counter(str(lane["status"]) for lane in SOURCE_LANES)
    lanes: list[dict[str, Any]] = []
    for lane in SOURCE_LANES:
        artifact = lane.get("artifact")
        artifact_path = manifest_dir / artifact if artifact else None
        lanes.append(
            {
                **lane,
                "artifact_path": str(artifact_path) if artifact_path else None,
                "artifact_exists": bool(artifact_path and artifact_path.exists()),
                "artifact_hash": _sha256_file(artifact_path) if artifact_path else None,
                "allowed_splits": allowed_splits(lane["policy"]),
                "forbidden_uses": _forbidden_uses(lane["policy"]),
            }
        )
    return {
        "version": DATA_RECIPE_VERSION,
        "manifest_dir": str(manifest_dir),
        "objective": "Build a validated multi-source registry for Fugu-Ultra training and evaluation.",
        "source_count": len(SOURCE_LANES),
        "policy_counts": _counter_json(policies),
        "family_counts": _counter_json(families),
        "status_counts": _counter_json(statuses),
        "target_registry_sizes": TARGET_REGISTRY_SIZES,
        "grpo_train_mvp_mix": GRPO_TRAIN_MVP_MIX,
        "grpo_train_mvp_notes": GRPO_TRAIN_MVP_NOTES,
        "tasktrove_validation_gates": TASKTROVE_VALIDATION_GATES,
        "fixed_workflow_discovery_gate": FIXED_WORKFLOW_DISCOVERY_GATE,
        "immediate_build_order": IMMEDIATE_BUILD_ORDER,
        "lanes": lanes,
    }


def write_existing_bank_taskspecs(
    manifest_dir: Path,
    out_jsonl: Path,
    report_out: Path | None = None,
) -> dict[str, Any]:
    bank_path = manifest_dir / "manifest.jsonl"
    adapter = ExistingBankAdapter(bank_path=bank_path)
    registry = TaskRegistry()
    registry.register_manifest(adapter.manifest())

    accepted = []
    rejected: list[dict[str, str]] = []
    counts = {
        "split": Counter(),
        "harness": Counter(),
        "domain": Counter(),
        "capability": Counter(),
        "group_id": Counter(),
    }
    for spec in adapter.materialize_all():
        try:
            registry.add(spec)
        except RegistryError as exc:
            rejected.append({"task_id": spec.task_id, "reason": str(exc)})
            continue
        accepted.append(spec)
        counts["split"][spec.splitting.split] += 1
        counts["harness"][spec.environment.harness] += 1
        counts["domain"][str(spec.metadata.domain)] += 1
        counts["capability"][spec.capability] += 1
        counts["group_id"][spec.splitting.group_id] += 1

    _write_jsonl(out_jsonl, [spec.model_dump(mode="json") for spec in accepted])
    report = {
        "version": "existing_bank_taskspec_promotion_v1",
        "input_jsonl": str(bank_path),
        "out_jsonl": str(out_jsonl),
        "source_manifest": adapter.manifest().model_dump(mode="json"),
        "materialized": len(accepted),
        "rejected": len(rejected),
        "rejection_examples": rejected[:20],
        "counts": {name: _counter_json(counter) for name, counter in counts.items()},
        "live_calls": False,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_registry_validation_report(manifest_dir: Path, out_dir: Path) -> dict[str, Any]:
    manifest_dir = manifest_dir.resolve()
    out_dir = out_dir.resolve()
    manifests = {m.source_name: m for m in build_source_manifests(manifest_dir)}
    registry = TaskRegistry()
    for manifest in manifests.values():
        registry.register_manifest(manifest)

    totals = {
        "tasks": 0,
        "accepted": 0,
        "missing_files": 0,
        "errors": 0,
    }
    counts = {
        "source": Counter(),
        "policy": Counter(),
        "split": Counter(),
        "harness": Counter(),
        "capability": Counter(),
        "contamination_group": Counter(),
    }
    shard_reports: list[dict[str, Any]] = []

    for shard in TASK_SHARDS:
        rel_path = Path(shard["path"])
        path = out_dir / rel_path.relative_to("data_mix") if rel_path.parts and rel_path.parts[0] == "data_mix" else manifest_dir / rel_path
        shard_report: dict[str, Any] = {
            "name": shard["name"],
            "path": str(path),
            "role": shard["role"],
            "exists": path.exists(),
            "tasks": 0,
            "accepted": 0,
            "errors": [],
        }
        if not path.exists():
            totals["missing_files"] += 1
            shard_reports.append(shard_report)
            continue
        with path.open() as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                totals["tasks"] += 1
                shard_report["tasks"] += 1
                try:
                    task = TaskSpec.model_validate(json.loads(line))
                    manifest = manifests.get(task.source.name)
                    if manifest is None:
                        raise ValueError(f"missing SourceManifest for {task.source.name}")
                    if not policy_allows_split(task.source.policy, task.splitting.split):
                        raise ValueError(
                            f"policy {task.source.policy} forbids split {task.splitting.split}"
                        )
                    registry.add(task)
                except (json.JSONDecodeError, RegistryError, ValueError) as exc:
                    totals["errors"] += 1
                    if len(shard_report["errors"]) < 20:
                        shard_report["errors"].append({"line": line_no, "reason": str(exc)})
                    continue
                totals["accepted"] += 1
                shard_report["accepted"] += 1
                counts["source"][task.source.name] += 1
                counts["policy"][task.source.policy] += 1
                counts["split"][task.splitting.split] += 1
                counts["harness"][task.environment.harness] += 1
                counts["capability"][task.capability] += 1
                counts["contamination_group"][task.splitting.contamination_group or task.splitting.group_id] += 1
        shard_reports.append(shard_report)

    return {
        "version": "fugu_ultra_registry_validation_v1",
        "ready": totals["errors"] == 0 and totals["missing_files"] == 0,
        "manifest_dir": str(manifest_dir),
        "out_dir": str(out_dir),
        "totals": totals,
        "counts": {name: _counter_json(counter) for name, counter in counts.items()},
        "shards": shard_reports,
        "live_calls": False,
    }


def write_data_recipe_artifacts(
    manifest_dir: Path,
    out_dir: Path,
    *,
    include_existing_bank: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifests = build_source_manifests(manifest_dir)
    registry = build_source_registry(manifest_dir)

    source_manifests_path = out_dir / "source_manifests.jsonl"
    source_registry_path = out_dir / "source_registry.json"
    _write_jsonl(source_manifests_path, [m.model_dump(mode="json") for m in manifests])
    source_registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")

    existing_bank_report = None
    if include_existing_bank:
        existing_bank_report = write_existing_bank_taskspecs(
            manifest_dir=manifest_dir,
            out_jsonl=out_dir / "existing_bank_taskspecs.jsonl",
            report_out=out_dir / "existing_bank_report.json",
        )

    validation_report = build_registry_validation_report(manifest_dir, out_dir)
    validation_path = out_dir / "registry_validation_report.json"
    validation_path.write_text(json.dumps(validation_report, indent=2, sort_keys=True) + "\n")

    report = {
        "version": DATA_RECIPE_VERSION,
        "manifest_dir": str(manifest_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "source_manifests": str(source_manifests_path),
        "source_registry": str(source_registry_path),
        "source_count": len(manifests),
        "existing_bank_report": existing_bank_report,
        "registry_validation_report": str(validation_path),
        "registry_ready": validation_report["ready"],
        "live_calls": False,
    }
    (out_dir / "data_recipe_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
