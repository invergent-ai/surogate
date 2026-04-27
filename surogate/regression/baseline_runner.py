"""Refactor regression baseline runner.

This module intentionally keeps execution policy simple: it records structured
JSON artifacts for known model/recipe/distribution cases, compares runs against
locked baselines, and emits the FP8/FP4 north-star coverage report. Heavy GPU
training is launched only when the caller passes ``--run``; otherwise missing
model/config prerequisites are represented as explicit ``skipped`` rows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "regression_baselines"
ARTIFACT_SCHEMA_VERSION = 1

DTYPE_TOLERANCE = {
    "bf16": 1e-2,
    "fp8": 5e-2,
    "fp4": 1e-1,
}

CONVERGENCE_REL_TOL = 0.05
STEP_TIME_REGRESSION_TOL = 0.05
CUDA_MEMORY_REGRESSION_TOL = 0.10
NCCL_BYTES_REL_TOL = 0.01
DEFAULT_RUN_TIMEOUT_S = int(os.environ.get("SUROGATE_REGRESSION_TIMEOUT_S", "1800"))


@dataclass(frozen=True)
class RegressionCase:
    model: str
    recipe: str
    distribution: str
    storage: str = "gpu"
    op_kind: str = "dense"
    config: str | None = None
    # Optional local/offline override for the config's Hugging Face model id.
    env_model_path: str | None = None
    supported: bool = True

    @property
    def case_id(self) -> str:
        return "__".join([self.model, self.recipe, self.distribution, self.storage, self.op_kind])


@dataclass
class RegressionResult:
    case: dict[str, Any]
    status: str
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    started_at: float = 0.0
    duration_s: float = 0.0


@dataclass
class RegressionArtifacts:
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    activation_snapshots: dict[str, dict[str, float]] = field(default_factory=dict)
    gradient_snapshots: dict[str, dict[str, float]] = field(default_factory=dict)
    convergence_curve: list[dict[str, float]] = field(default_factory=list)
    step_time_ms: dict[str, float] = field(default_factory=dict)
    cuda_peak_memory_bytes: int | None = None
    nccl: dict[str, dict[str, int]] = field(default_factory=dict)
    descriptor_summary: dict[str, int] = field(default_factory=dict)
    block_schema_summary: dict[str, int] = field(default_factory=dict)
    buffer_plan_summary: dict[str, int] = field(default_factory=dict)
    arena_summary: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path) -> "RegressionArtifacts":
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegressionArtifacts":
        known = {
            "schema_version": data.get("schema_version", ARTIFACT_SCHEMA_VERSION),
            "activation_snapshots": data.get("activation_snapshots", {}),
            "gradient_snapshots": data.get("gradient_snapshots", {}),
            "convergence_curve": data.get("convergence_curve", []),
            "step_time_ms": data.get("step_time_ms", {}),
            "cuda_peak_memory_bytes": data.get("cuda_peak_memory_bytes"),
            "nccl": data.get("nccl", {}),
            "descriptor_summary": data.get("descriptor_summary", {}),
            "block_schema_summary": data.get("block_schema_summary", {}),
            "buffer_plan_summary": data.get("buffer_plan_summary", {}),
            "arena_summary": data.get("arena_summary", {}),
        }
        return cls(**known)


REGRESSION_MATRIX: tuple[RegressionCase, ...] = (
    RegressionCase(
        "qwen3",
        "bf16",
        "single_gpu",
        config="examples/sft/qwen3/qwen3-lora-bf16-1step.yaml",
        env_model_path="QWEN3_MODEL_PATH",
    ),
    RegressionCase(
        "qwen3", "fp8", "single_gpu", config="examples/sft/qwen3/qwen3-lora-fp8.yaml", env_model_path="QWEN3_MODEL_PATH"
    ),
    RegressionCase(
        "gemma4",
        "bf16",
        "single_gpu",
        config="examples/sft/gemma4/gemma4-e2b-lora-bf16.yaml",
        env_model_path="GEMMA4_MODEL_PATH",
    ),
    RegressionCase(
        "gemma4",
        "fp8",
        "single_gpu",
        config="examples/sft/gemma4/gemma4-e2b-lora-fp8.yaml",
        env_model_path="GEMMA4_MODEL_PATH",
    ),
    RegressionCase(
        "nemotron_h",
        "bf16",
        "single_gpu",
        config="examples/sft/nemotron3/nemotron-nano3-qlora-bnb.yaml",
        env_model_path="NEMOTRON_H_MODEL_PATH",
    ),
    RegressionCase(
        "nemotron_h",
        "fp8",
        "single_gpu",
        config="examples/sft/nemotron3/nemotron-nano3-nvfp4.yaml",
        env_model_path="NEMOTRON_H_MODEL_PATH",
    ),
    RegressionCase(
        "qwen3_5",
        "bf16",
        "single_gpu",
        config="examples/sft/qwen35/qwen35-text-lora-bf16-1step.yaml",
        env_model_path="QWEN3_5_MODEL_PATH",
    ),
    RegressionCase(
        "qwen3_5",
        "fp8",
        "single_gpu",
        config="examples/sft/qwen35/qwen35-text-lora-fp8.yaml",
        env_model_path="QWEN3_5_MODEL_PATH",
    ),
    RegressionCase(
        "qwen3_5_moe",
        "fp8",
        "single_gpu",
        op_kind="moe_grouped",
        config="examples/sft/qwen35moe/qwen35moe-lora-fp8.yaml",
        env_model_path="QWEN3_5_MOE_MODEL_PATH",
    ),
    RegressionCase(
        "gpt_oss",
        "fp8",
        "single_gpu",
        op_kind="moe_grouped",
        config="examples/sft/gpt-oss/gptoss-lora-mxfp4.yaml",
        env_model_path="GPT_OSS_MODEL_PATH",
    ),
    RegressionCase(
        "qwen3",
        "bf16",
        "2gpu_dp",
        config="examples/sft/qwen3/qwen3-lora-bf16-bench.yaml",
        env_model_path="QWEN3_MODEL_PATH",
    ),
    RegressionCase(
        "qwen3_5_moe",
        "fp8",
        "2gpu_dp",
        op_kind="moe_grouped",
        config="examples/sft/qwen35moe/qwen35moe-lora-fp8.yaml",
        env_model_path="QWEN3_5_MOE_MODEL_PATH",
    ),
    RegressionCase(
        "qwen3_6_moe",
        "fp8",
        "2gpu_dp_ep",
        op_kind="moe_grouped",
        config="examples/sft/qwen36moe/qwen36moe-lora-fp8.yaml",
        env_model_path="QWEN3_6_MOE_MODEL_PATH",
    ),
    RegressionCase(
        "qwen3",
        "bf16",
        "single_gpu",
        storage="cpu_stream",
        config="examples/sft/qwen3/qwen3-lora-bf16-1step.yaml",
        env_model_path="QWEN3_MODEL_PATH",
    ),
    RegressionCase(
        "qwen3",
        "fp8",
        "2gpu_dp",
        storage="cpu_stream",
        config="examples/sft/qwen3/qwen3-lora-fp8.yaml",
        env_model_path="QWEN3_MODEL_PATH",
    ),
    RegressionCase("qwen3", "fp4", "single_gpu", supported=False),
)


def _case_path(case: RegressionCase, directory: Path) -> Path:
    return directory / f"{case.case_id}.json"


def _missing_reason(case: RegressionCase) -> str | None:
    if not case.supported:
        return "unsupported in regression matrix"
    if case.config and not (REPO_ROOT / case.config).exists():
        return f"config not found: {case.config}"
    return None


def _materialize_case_config(case: RegressionCase, *, steps: int, directory: Path) -> Path | None:
    if not case.config:
        return None
    src = REPO_ROOT / case.config
    data = yaml.safe_load(src.read_text()) or {}
    data["max_steps"] = steps
    data["eval_steps"] = 0
    data["output_dir"] = str(directory / "runs" / case.case_id)
    if case.env_model_path and os.environ.get(case.env_model_path):
        data["model"] = os.environ[case.env_model_path]
    if case.distribution == "2gpu_dp":
        data["gpus"] = 2
        data["ep_size"] = 1
    if case.storage == "cpu_stream":
        data["cpu_training"] = True

    out = directory / "configs" / f"{case.case_id}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False) + "\n")
    return out


def _command_for_case(case: RegressionCase, config_path: Path | None = None) -> list[str]:
    path: str | None = str(config_path) if config_path else case.config
    if not path:
        return []
    return ["surogate", "sft", path]


def _artifact_path(case: RegressionCase, directory: Path) -> Path:
    return directory / "artifacts" / f"{case.case_id}.json"


def _load_case_artifacts(path: Path) -> dict[str, Any]:
    artifacts = RegressionArtifacts.from_json(path)
    return asdict(artifacts)


def _subprocess_text_tail(value: str | bytes | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def run_case(
    case: RegressionCase,
    *,
    run: bool,
    steps: int,
    artifact_dir: Path | None = None,
    timeout_s: int | float | None = DEFAULT_RUN_TIMEOUT_S,
) -> RegressionResult:
    started = time.time()
    result = RegressionResult(case=asdict(case), status="skipped", started_at=started)
    missing = _missing_reason(case)
    if missing:
        result.reason = missing
        return result

    cmd_config = _materialize_case_config(case, steps=steps, directory=artifact_dir) if run and artifact_dir else None
    cmd = _command_for_case(case, cmd_config)
    result.command = cmd
    if not run:
        result.reason = "not executed; pass --run to launch GPU workload"
        return result

    if not cmd:
        result.status = "skipped"
        result.reason = "no command configured"
        return result

    env = os.environ.copy()
    artifact_path = _artifact_path(case, artifact_dir) if artifact_dir else None
    if artifact_path:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        env["SUROGATE_REGRESSION_ARTIFACT"] = str(artifact_path)

    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=env,
            timeout=timeout_s if timeout_s and timeout_s > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        result.duration_s = time.time() - started
        result.status = "failed"
        result.reason = f"command timed out after {timeout_s:g}s"
        result.artifacts = {
            "stdout_tail": _subprocess_text_tail(exc.stdout),
            "stderr_tail": _subprocess_text_tail(exc.stderr),
            "returncode": "timeout",
        }
        return result
    result.duration_s = time.time() - started
    result.artifacts = {
        "stdout_tail": _subprocess_text_tail(proc.stdout),
        "stderr_tail": _subprocess_text_tail(proc.stderr),
        "returncode": proc.returncode,
    }
    if artifact_path and artifact_path.exists():
        result.artifacts["regression_artifact"] = str(artifact_path)
        parsed = _load_case_artifacts(artifact_path)
        result.artifacts["schema_version"] = parsed["schema_version"]
        result.metrics = parsed
    result.status = "passed" if proc.returncode == 0 else "failed"
    result.reason = "" if proc.returncode == 0 else f"command exited {proc.returncode}"
    return result


def _stable_result_dict(result: RegressionResult) -> dict[str, Any]:
    data = asdict(result)
    data["started_at"] = 0.0
    data["duration_s"] = 0.0
    return data


def write_results(results: list[RegressionResult], directory: Path, *, stable: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for result in results:
        case = RegressionCase(**result.case)
        data = _stable_result_dict(result) if stable else asdict(result)
        _case_path(case, directory).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_results(directory: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not directory.exists():
        return results
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        if "case" not in data:
            continue
        case = RegressionCase(**data["case"])
        results[case.case_id] = data
    return results


def _numeric_delta(cur: Any, base: Any) -> float | None:
    if isinstance(cur, int | float) and isinstance(base, int | float):
        return abs(float(cur) - float(base))
    return None


def _relative_increase(cur: float, base: float) -> float:
    if base <= 0:
        return 0.0 if cur <= base else float("inf")
    return (cur - base) / base


def _compare_numeric_maps(
    failures: list[str],
    case_id: str,
    label: str,
    cur: dict[str, Any],
    base: dict[str, Any],
    tolerance: float,
) -> None:
    for name, base_stats in base.items():
        cur_stats = cur.get(name)
        if cur_stats is None:
            failures.append(f"{case_id}: missing {label} snapshot {name}")
            continue
        for key, base_value in base_stats.items():
            delta = _numeric_delta(cur_stats.get(key), base_value)
            if delta is not None and delta > tolerance:
                failures.append(f"{case_id}: {label}.{name}.{key} delta {delta:.6g} > {tolerance:.6g}")


def _compare_convergence(
    failures: list[str], case_id: str, cur: list[dict[str, Any]], base: list[dict[str, Any]]
) -> None:
    if not base:
        return
    if len(cur) < len(base):
        failures.append(f"{case_id}: convergence curve length {len(cur)} < baseline {len(base)}")
        return
    cur_by_step = {int(row["step"]): float(row["loss"]) for row in cur if "step" in row and "loss" in row}
    for base_row in base:
        if "step" not in base_row or "loss" not in base_row:
            continue
        step = int(base_row["step"])
        base_loss = float(base_row["loss"])
        if step not in cur_by_step:
            failures.append(f"{case_id}: missing convergence step {step}")
            continue
        rel = abs(cur_by_step[step] - base_loss) / max(abs(base_loss), 1e-8)
        if rel > CONVERGENCE_REL_TOL:
            failures.append(f"{case_id}: convergence step {step} rel delta {rel:.3%} > {CONVERGENCE_REL_TOL:.3%}")


def _compare_perf_and_memory(failures: list[str], case_id: str, cur: dict[str, Any], base: dict[str, Any]) -> None:
    base_step = base.get("step_time_ms", {})
    cur_step = cur.get("step_time_ms", {})
    for key, base_value in base_step.items():
        cur_value = cur_step.get(key)
        if isinstance(cur_value, int | float) and isinstance(base_value, int | float):
            rel = _relative_increase(float(cur_value), float(base_value))
            if rel > STEP_TIME_REGRESSION_TOL:
                failures.append(f"{case_id}: step_time_ms.{key} regression {rel:.3%} > {STEP_TIME_REGRESSION_TOL:.3%}")

    base_mem = base.get("cuda_peak_memory_bytes")
    cur_mem = cur.get("cuda_peak_memory_bytes")
    if isinstance(cur_mem, int | float) and isinstance(base_mem, int | float):
        rel = _relative_increase(float(cur_mem), float(base_mem))
        if rel > CUDA_MEMORY_REGRESSION_TOL:
            failures.append(
                f"{case_id}: cuda_peak_memory_bytes regression {rel:.3%} > {CUDA_MEMORY_REGRESSION_TOL:.3%}"
            )


def _compare_nccl(failures: list[str], case_id: str, cur: dict[str, Any], base: dict[str, Any]) -> None:
    for op_name, base_stats in base.get("nccl", {}).items():
        cur_stats = cur.get("nccl", {}).get(op_name)
        if cur_stats is None:
            failures.append(f"{case_id}: missing NCCL op {op_name}")
            continue
        if cur_stats.get("count") != base_stats.get("count"):
            failures.append(f"{case_id}: nccl.{op_name}.count {cur_stats.get('count')} != {base_stats.get('count')}")
        base_bytes = base_stats.get("bytes")
        cur_bytes = cur_stats.get("bytes")
        if isinstance(cur_bytes, int | float) and isinstance(base_bytes, int | float):
            rel = abs(float(cur_bytes) - float(base_bytes)) / max(abs(float(base_bytes)), 1.0)
            if rel > NCCL_BYTES_REL_TOL:
                failures.append(f"{case_id}: nccl.{op_name}.bytes rel delta {rel:.3%} > {NCCL_BYTES_REL_TOL:.3%}")


def _compare_descriptor_summary(failures: list[str], case_id: str, cur: dict[str, Any], base: dict[str, Any]) -> None:
    base_summary = base.get("descriptor_summary", {})
    cur_summary = cur.get("descriptor_summary", {})
    for key, base_value in base_summary.items():
        if not isinstance(base_value, int):
            continue
        cur_value = cur_summary.get(key)
        if cur_value is None:
            failures.append(f"{case_id}: missing descriptor_summary.{key}")
        elif cur_value != base_value:
            failures.append(f"{case_id}: descriptor_summary.{key} {cur_value} != {base_value}")


def _compare_block_schema_summary(failures: list[str], case_id: str, cur: dict[str, Any], base: dict[str, Any]) -> None:
    base_summary = base.get("block_schema_summary", {})
    cur_summary = cur.get("block_schema_summary", {})
    for key, base_value in base_summary.items():
        if not isinstance(base_value, int):
            continue
        cur_value = cur_summary.get(key)
        if cur_value is None:
            failures.append(f"{case_id}: missing block_schema_summary.{key}")
        elif cur_value != base_value:
            failures.append(f"{case_id}: block_schema_summary.{key} {cur_value} != {base_value}")


def _compare_buffer_plan_summary(failures: list[str], case_id: str, cur: dict[str, Any], base: dict[str, Any]) -> None:
    base_summary = base.get("buffer_plan_summary", {})
    cur_summary = cur.get("buffer_plan_summary", {})
    for key, base_value in base_summary.items():
        if not isinstance(base_value, int):
            continue
        cur_value = cur_summary.get(key)
        if cur_value is None:
            failures.append(f"{case_id}: missing buffer_plan_summary.{key}")
        elif cur_value != base_value:
            failures.append(f"{case_id}: buffer_plan_summary.{key} {cur_value} != {base_value}")


def _compare_arena_summary(failures: list[str], case_id: str, cur: dict[str, Any], base: dict[str, Any]) -> None:
    base_summary = base.get("arena_summary", {})
    cur_summary = cur.get("arena_summary", {})
    for key, base_value in base_summary.items():
        if not isinstance(base_value, int):
            continue
        cur_value = cur_summary.get(key)
        if cur_value is None:
            failures.append(f"{case_id}: missing arena_summary.{key}")
        elif cur_value != base_value:
            failures.append(f"{case_id}: arena_summary.{key} {cur_value} != {base_value}")


def _compare_artifacts(case_id: str, cur: dict[str, Any], base: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    case = cur.get("case", {})
    tolerance = DTYPE_TOLERANCE.get(case.get("recipe", ""), DTYPE_TOLERANCE["bf16"])
    cur_metrics = cur.get("metrics") or {}
    base_metrics = base.get("metrics") or {}
    if not base_metrics:
        return failures
    if not cur_metrics:
        failures.append(f"{case_id}: missing regression metrics")
        return failures
    _compare_numeric_maps(
        failures,
        case_id,
        "activation",
        cur_metrics.get("activation_snapshots", {}),
        base_metrics.get("activation_snapshots", {}),
        tolerance,
    )
    _compare_numeric_maps(
        failures,
        case_id,
        "gradient",
        cur_metrics.get("gradient_snapshots", {}),
        base_metrics.get("gradient_snapshots", {}),
        tolerance,
    )
    _compare_convergence(
        failures,
        case_id,
        cur_metrics.get("convergence_curve", []),
        base_metrics.get("convergence_curve", []),
    )
    _compare_perf_and_memory(failures, case_id, cur_metrics, base_metrics)
    _compare_nccl(failures, case_id, cur_metrics, base_metrics)
    _compare_descriptor_summary(failures, case_id, cur_metrics, base_metrics)
    _compare_block_schema_summary(failures, case_id, cur_metrics, base_metrics)
    _compare_buffer_plan_summary(failures, case_id, cur_metrics, base_metrics)
    _compare_arena_summary(failures, case_id, cur_metrics, base_metrics)
    return failures


def compare_results(current: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for case_id, cur in current.items():
        base = baseline.get(case_id)
        if not base:
            failures.append(f"{case_id}: missing baseline")
            continue
        if cur["status"] != base["status"]:
            failures.append(f"{case_id}: status {cur['status']} != {base['status']}")
        if cur["status"] == "passed" and base["status"] == "passed":
            failures.extend(_compare_artifacts(case_id, cur, base))
    return failures


def required_capabilities_for_case(case: dict[str, Any]) -> list[str]:
    recipe = case["recipe"]
    op_kind = case["op_kind"]
    if recipe not in {"fp8", "fp4"}:
        return []

    caps: list[str] = []
    if op_kind == "dense":
        caps.append("DenseMatmul")
    elif op_kind == "moe_grouped":
        caps.extend(["GroupedMatmul", "MoERouted"])

    if recipe == "fp8":
        caps.append("FP8Eligible")
    elif recipe == "fp4":
        caps.append("FP4Eligible")
    return caps


def required_moe_capabilities_for_case(case: dict[str, Any]) -> list[str]:
    recipe = case["recipe"]
    op_kind = case["op_kind"]
    if op_kind != "moe_grouped" or recipe not in {"fp8", "fp4"}:
        return []

    caps = ["GroupedGemmEligible"]
    if recipe == "fp8":
        caps.append("FP8GroupedEligible")
    elif recipe == "fp4":
        caps.append("FP4GroupedEligible")
    return caps


def required_matmul_capabilities_for_case(case: dict[str, Any]) -> list[str]:
    recipe = case["recipe"]
    op_kind = case["op_kind"]
    if op_kind != "dense" or recipe not in {"fp8", "fp4"}:
        return []

    if recipe == "fp8":
        return ["FP8ForwardEligible", "FP8BackwardEligible"]
    return ["FP4ForwardEligible", "FP4BackwardEligible"]


def matmul_capability_counts(descriptor_summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "matmul_fp8_forward_eligible_ops",
        "matmul_fp8_backward_eligible_ops",
        "matmul_fp4_forward_eligible_ops",
        "matmul_fp4_backward_eligible_ops",
        "forward_matmul_fp8_forward_eligible_ops",
        "backward_matmul_fp8_backward_eligible_ops",
        "forward_matmul_fp4_forward_eligible_ops",
        "backward_matmul_fp4_backward_eligible_ops",
    )
    return {key: descriptor_summary.get(key) for key in keys}


def hook_target_counts(metrics: dict[str, Any]) -> dict[str, Any]:
    descriptor_summary = metrics.get("descriptor_summary") or {}
    block_schema_summary = metrics.get("block_schema_summary") or {}
    buffer_plan_summary = metrics.get("buffer_plan_summary") or {}
    return {
        "hook_after_produce_targets": block_schema_summary.get("hook_after_produce_targets"),
        "hook_before_consume_targets": block_schema_summary.get("hook_before_consume_targets"),
        "hook_after_all_to_all_targets": block_schema_summary.get("hook_after_all_to_all_targets"),
        "hook_after_reduce_scatter_targets": block_schema_summary.get("hook_after_reduce_scatter_targets"),
        "runtime_hook_after_produce_targets": buffer_plan_summary.get("hook_after_produce_targets"),
        "runtime_hook_before_consume_targets": buffer_plan_summary.get("hook_before_consume_targets"),
        "runtime_hook_after_all_to_all_targets": buffer_plan_summary.get("hook_after_all_to_all_targets"),
        "runtime_hook_after_reduce_scatter_targets": buffer_plan_summary.get("hook_after_reduce_scatter_targets"),
        "hook_registry_registrations": buffer_plan_summary.get("hook_registry_registrations"),
        "hook_registry_distribution_aware_registrations": buffer_plan_summary.get(
            "hook_registry_distribution_aware_registrations"
        ),
        "lora_slices": descriptor_summary.get("lora_slices"),
        "lora_schema_slot_slices": descriptor_summary.get("lora_schema_slot_slices"),
        "grouped_lora_schema_slot_slices": descriptor_summary.get("grouped_lora_schema_slot_slices"),
    }


def descriptor_count_requirements_for_case(case: dict[str, Any]) -> list[str]:
    if not case.get("supported", True):
        return []

    recipe = case["recipe"]
    op_kind = case["op_kind"]
    if recipe not in {"fp8", "fp4"}:
        return []

    if op_kind == "dense":
        if recipe == "fp8":
            return ["matmul_fp8_forward_eligible_ops", "matmul_fp8_backward_eligible_ops"]
        return ["matmul_fp4_forward_eligible_ops", "matmul_fp4_backward_eligible_ops"]

    if op_kind == "moe_grouped":
        if recipe == "fp8":
            return ["moe_fp8_grouped_eligible_ops"]
        return ["moe_fp4_grouped_eligible_ops"]

    return []


def descriptor_requirement_status(case: dict[str, Any], descriptor_summary: dict[str, Any]) -> tuple[str, list[str]]:
    required_counts = descriptor_count_requirements_for_case(case)
    if not required_counts:
        return "not_applicable", []
    if not descriptor_summary:
        return "unknown", required_counts
    missing = [key for key in required_counts if int(descriptor_summary.get(key) or 0) <= 0]
    return ("present" if not missing else "missing"), missing


def hook_readiness_status(case: dict[str, Any], metrics: dict[str, Any]) -> tuple[str, list[str]]:
    counts = hook_target_counts(metrics)
    required: list[str] = []
    if case.get("storage") == "cpu_stream":
        required.append("hook_before_consume_targets")
    if "ep" in str(case.get("distribution") or ""):
        required.append("hook_after_all_to_all_targets")
    if int(counts.get("lora_slices") or 0) > 0:
        required.append("hook_after_produce_targets")
        required.append("lora_schema_slot_slices")
    if not required:
        return "not_applicable", []
    if not metrics:
        return "unknown", required
    missing: list[str] = []
    for key in required:
        value = int(counts.get(key) or 0)
        if key == "lora_schema_slot_slices":
            if value < int(counts.get("lora_slices") or 0):
                missing.append(key)
        elif value <= 0:
            missing.append(key)
    return ("present" if not missing else "missing"), missing


def block_schema_status(block_schema_summary: dict[str, Any]) -> str:
    if not block_schema_summary:
        return "unknown"
    expected = int(block_schema_summary.get("block_schema_expected_layers") or 0)
    records = int(block_schema_summary.get("block_schema_records") or 0)
    missing = int(block_schema_summary.get("block_schema_missing_layers") or 0)
    if expected <= 0:
        return "not_applicable"
    if records >= expected and missing == 0:
        return "present"
    return "missing"


def storage_declaration_status(case: dict[str, Any], block_schema_summary: dict[str, Any]) -> str:
    if case.get("storage") != "cpu_stream":
        return "not_applicable"
    if not block_schema_summary:
        return "unknown"
    streamable = int(block_schema_summary.get("block_schema_auto_resident_slots") or 0) + int(
        block_schema_summary.get("block_schema_cpu_stream_slots") or 0
    )
    return "present" if streamable > 0 else "missing"


def ep_topology_status(case: dict[str, Any], block_schema_summary: dict[str, Any]) -> str:
    if "ep" not in str(case.get("distribution") or ""):
        return "not_applicable"
    if not block_schema_summary:
        return "unknown"
    return "present" if int(block_schema_summary.get("block_schema_ep_layers") or 0) > 0 else "missing"


def coverage_report(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    eligible = 0
    passed = 0
    for result in results.values():
        case = result["case"]
        recipe = case["recipe"]
        metrics = result.get("metrics") or {}
        descriptor_summary = metrics.get("descriptor_summary") or {}
        block_schema_summary = metrics.get("block_schema_summary") or {}
        descriptor_status, missing_descriptor_counts = descriptor_requirement_status(case, descriptor_summary)
        hook_status, missing_hook_counts = hook_readiness_status(case, metrics)
        is_quant = recipe in {"fp8", "fp4"}
        if is_quant and case.get("supported", True):
            eligible += 1
            if result["status"] == "passed":
                passed += 1
        rows.append(
            {
                "model": case["model"],
                "op_kind": case["op_kind"],
                "recipe": recipe,
                "distribution": case["distribution"],
                "storage": case["storage"],
                "status": result["status"],
                "reason": result.get("reason", ""),
                "required_capabilities": required_capabilities_for_case(case),
                "required_moe_capabilities": required_moe_capabilities_for_case(case),
                "required_matmul_capabilities": required_matmul_capabilities_for_case(case),
                "matmul_capability_counts": matmul_capability_counts(descriptor_summary),
                "descriptor_requirement_status": descriptor_status,
                "missing_descriptor_counts": missing_descriptor_counts,
                "fusion_candidate_starts": descriptor_summary.get("fusion_candidate_starts"),
                "block_schema_status": block_schema_status(block_schema_summary),
                "storage_declaration_status": storage_declaration_status(case, block_schema_summary),
                "ep_topology_status": ep_topology_status(case, block_schema_summary),
                "hook_readiness_status": hook_status,
                "missing_hook_counts": missing_hook_counts,
                "hook_target_counts": hook_target_counts(metrics),
                "block_schema_summary": block_schema_summary,
                "buffer_plan_summary": metrics.get("buffer_plan_summary") or {},
            }
        )
    return {
        "metric": "fp8_fp4_coverage",
        "passed": passed,
        "eligible": eligible,
        "coverage": (passed / eligible) if eligible else 0.0,
        "rows": rows,
    }


def select_cases(case_filters: list[str] | None = None) -> tuple[RegressionCase, ...]:
    if not case_filters:
        return REGRESSION_MATRIX
    selected: list[RegressionCase] = []
    requested = set(case_filters)
    matched: set[str] = set()
    for case in REGRESSION_MATRIX:
        if case.case_id in requested or case.model in requested:
            selected.append(case)
            if case.case_id in requested:
                matched.add(case.case_id)
            if case.model in requested:
                matched.add(case.model)
    missing = requested - matched
    if missing:
        raise SystemExit(f"unknown regression case filter(s): {', '.join(sorted(missing))}")
    return tuple(selected)


def case_listing(cases: tuple[RegressionCase, ...] = REGRESSION_MATRIX) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.case_id,
            "model": case.model,
            "recipe": case.recipe,
            "distribution": case.distribution,
            "storage": case.storage,
            "op_kind": case.op_kind,
            "supported": case.supported,
            "env_model_path": case.env_model_path or "",
            "config": case.config or "",
        }
        for case in cases
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASELINE_DIR / "current")
    parser.add_argument("--baseline", type=Path, default=BASELINE_DIR / "locked")
    parser.add_argument(
        "--case", action="append", dest="case_filters", help="Run one case_id or all rows for one model"
    )
    parser.add_argument("--run", action="store_true", help="Launch configured GPU workloads")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=DEFAULT_RUN_TIMEOUT_S,
        help="Per-case wall-clock timeout for launched workloads; 0 disables the timeout",
    )
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--list-cases", action="store_true", help="Print the regression matrix and exit")
    args = parser.parse_args(argv)

    if args.list_cases:
        print(json.dumps(case_listing(select_cases(args.case_filters)), indent=2, sort_keys=True))
        return 0

    results = [
        run_case(case, run=args.run, steps=args.steps, artifact_dir=args.out, timeout_s=args.timeout_s)
        for case in select_cases(args.case_filters)
    ]
    write_results(results, args.out)
    loaded = load_results(args.out)

    if args.update_baseline:
        write_results(results, args.baseline, stable=True)

    if args.report:
        report = coverage_report(loaded)
        report_path = args.out / "north_star_coverage.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))

    if args.compare:
        failures = compare_results(loaded, load_results(args.baseline))
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
