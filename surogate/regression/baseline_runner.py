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


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "regression_baselines"


@dataclass(frozen=True)
class RegressionCase:
    model: str
    recipe: str
    distribution: str
    storage: str = "gpu"
    op_kind: str = "dense"
    config: str | None = None
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
    if case.env_model_path and not os.environ.get(case.env_model_path):
        return f"missing {case.env_model_path}"
    return None


def _command_for_case(case: RegressionCase, steps: int) -> list[str]:
    if not case.config:
        return []
    cmd = ["surogate", "sft", str(REPO_ROOT / case.config)]
    if case.distribution == "2gpu_dp":
        cmd = ["python", "-m", "surogate.train.distributed", str(REPO_ROOT / case.config)]
    return cmd + ["--max-steps", str(steps)]


def run_case(case: RegressionCase, *, run: bool, steps: int) -> RegressionResult:
    started = time.time()
    result = RegressionResult(case=asdict(case), status="skipped", started_at=started)
    missing = _missing_reason(case)
    if missing:
        result.reason = missing
        return result

    cmd = _command_for_case(case, steps)
    result.command = cmd
    if not run:
        result.reason = "not executed; pass --run to launch GPU workload"
        return result

    if not cmd:
        result.status = "skipped"
        result.reason = "no command configured"
        return result

    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    result.duration_s = time.time() - started
    result.artifacts = {
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "returncode": proc.returncode,
    }
    result.status = "passed" if proc.returncode == 0 else "failed"
    result.reason = "" if proc.returncode == 0 else f"command exited {proc.returncode}"
    return result


def write_results(results: list[RegressionResult], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for result in results:
        case = RegressionCase(**result.case)
        _case_path(case, directory).write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")


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


def compare_results(current: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for case_id, cur in current.items():
        base = baseline.get(case_id)
        if not base:
            failures.append(f"{case_id}: missing baseline")
            continue
        if cur["status"] != base["status"]:
            failures.append(f"{case_id}: status {cur['status']} != {base['status']}")
    return failures


def coverage_report(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    eligible = 0
    passed = 0
    for result in results.values():
        case = result["case"]
        recipe = case["recipe"]
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
            }
        )
    return {
        "metric": "fp8_fp4_coverage",
        "passed": passed,
        "eligible": eligible,
        "coverage": (passed / eligible) if eligible else 0.0,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASELINE_DIR / "current")
    parser.add_argument("--baseline", type=Path, default=BASELINE_DIR / "locked")
    parser.add_argument("--run", action="store_true", help="Launch configured GPU workloads")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)

    results = [run_case(case, run=args.run, steps=args.steps) for case in FIRST_MONTH_MATRIX]
    write_results(results, args.out)
    loaded = load_results(args.out)

    if args.update_baseline:
        write_results(results, args.baseline)

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
