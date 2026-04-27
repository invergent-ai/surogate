"""First-month refactor regression baseline runner.

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
        }
        return cls(**known)


FIRST_MONTH_MATRIX: tuple[RegressionCase, ...] = (
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
        config="examples/sft/gemma4/gemma4-e2b-lora-bf16.yaml",
        env_model_path="GEMMA4_MODEL_PATH",
    ),
    RegressionCase("nemotron_h", "bf16", "single_gpu", env_model_path="NEMOTRON_H_MODEL_PATH"),
    RegressionCase("nemotron_h", "fp8", "single_gpu", env_model_path="NEMOTRON_H_MODEL_PATH"),
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
        "qwen3",
        "bf16",
        "single_gpu",
        storage="cpu_stream",
        config="examples/sft/qwen3/qwen3-lora-bf16-1step.yaml",
        env_model_path="QWEN3_MODEL_PATH",
    ),
    RegressionCase("qwen3", "fp4", "single_gpu", supported=False),
)


def _case_path(case: RegressionCase, directory: Path) -> Path:
    return directory / f"{case.case_id}.json"


def _missing_reason(case: RegressionCase) -> str | None:
    if not case.supported:
        return "unsupported in first-month matrix"
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
    path = config_path or ((REPO_ROOT / case.config) if case.config else None)
    if not path:
        return []
    return ["surogate", "sft", str(path)]


def _artifact_path(case: RegressionCase, directory: Path) -> Path:
    return directory / "artifacts" / f"{case.case_id}.json"


def _load_case_artifacts(path: Path) -> dict[str, Any]:
    artifacts = RegressionArtifacts.from_json(path)
    return asdict(artifacts)


def run_case(case: RegressionCase, *, run: bool, steps: int, artifact_dir: Path | None = None) -> RegressionResult:
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

    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False, env=env)
    result.duration_s = time.time() - started
    result.artifacts = {
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
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


def coverage_report(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    eligible = 0
    passed = 0
    for result in results.values():
        case = result["case"]
        recipe = case["recipe"]
        metrics = result.get("metrics") or {}
        descriptor_summary = metrics.get("descriptor_summary") or {}
        descriptor_status, missing_descriptor_counts = descriptor_requirement_status(case, descriptor_summary)
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
        return FIRST_MONTH_MATRIX
    selected: list[RegressionCase] = []
    requested = set(case_filters)
    matched: set[str] = set()
    for case in FIRST_MONTH_MATRIX:
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


def case_listing(cases: tuple[RegressionCase, ...] = FIRST_MONTH_MATRIX) -> list[dict[str, Any]]:
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
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--list-cases", action="store_true", help="Print the first-month matrix and exit")
    args = parser.parse_args(argv)

    if args.list_cases:
        print(json.dumps(case_listing(select_cases(args.case_filters)), indent=2, sort_keys=True))
        return 0

    results = [
        run_case(case, run=args.run, steps=args.steps, artifact_dir=args.out)
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
