"""Fail-closed admission for train-only branch-point analog tasks."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ultra.pool_binding import load_pool_binding


ADMISSION_VERSION = "fugu_branchpoint_analog_admission_v1"


class AnalogAdmissionError(ValueError):
    """The analog band cannot support trusted topology experiments."""


@dataclass(frozen=True)
class AdmittedAnalog:
    task_id: str
    mechanism_id: str
    task_dir: Path
    task_checksum: str
    docker_image: str
    docker_image_id: str


@dataclass(frozen=True)
class AnalogBandAdmission:
    band_id: str
    pool_id: str
    pool_fingerprint: str
    tasks: tuple[AdmittedAnalog, ...]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _object(value: Any, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise AnalogAdmissionError(f"{label} must contain exactly {sorted(fields)}")
    return value


def _project_path(root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise AnalogAdmissionError(f"{label} must be a non-empty path")
    path = Path(raw)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AnalogAdmissionError(f"{label} escapes the project root") from exc
    return path


def _verified_file(root: Path, raw: Any, *, label: str) -> Path:
    record = _object(raw, fields={"path", "sha256"}, label=label)
    path = _project_path(root, record["path"], label=label)
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise AnalogAdmissionError(f"{label} is missing or has hash drift")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalogAdmissionError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise AnalogAdmissionError(f"{label} must be an object")
    return value


def _reward(result: dict[str, Any]) -> float | None:
    value = ((result.get("verifier_result") or {}).get("rewards") or {}).get(
        "reward"
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _validate_job_result(
    result: dict[str, Any], *, agent: str, reward: float, label: str
) -> None:
    stats = result.get("stats") or {}
    evals = stats.get("evals") or {}
    expected_eval = f"{agent}__adhoc"
    if (
        result.get("n_total_trials") != 1
        or stats.get("n_completed_trials") != 1
        or stats.get("n_errored_trials") != 0
        or stats.get("n_cancelled_trials") != 0
        or stats.get("n_retries") != 0
        or stats.get("cost_usd") is not None
        or stats.get("n_input_tokens") is not None
        or stats.get("n_output_tokens") is not None
        or set(evals) != {expected_eval}
    ):
        raise AnalogAdmissionError(f"{label} is not an exact zero-call single trial")
    evaluation = evals[expected_eval]
    metrics = evaluation.get("metrics") or []
    if (
        evaluation.get("n_trials") != 1
        or evaluation.get("n_errors") != 0
        or len(metrics) != 1
        or metrics[0].get("mean") != reward
        or evaluation.get("exception_stats") != {}
    ):
        raise AnalogAdmissionError(f"{label} outcome drift")


def _validate_trial_result(
    result: dict[str, Any],
    *,
    task_id: str,
    task_dir_relative: str,
    task_checksum: str,
    agent: str,
    reward: float,
    label: str,
) -> None:
    config = result.get("config") or {}
    agent_config = config.get("agent") or {}
    agent_info = result.get("agent_info") or {}
    agent_result = result.get("agent_result") or {}
    if (
        result.get("task_name") != task_id
        or (result.get("task_id") or {}).get("path") != task_dir_relative
        or result.get("task_checksum") != task_checksum
        or result.get("exception_info") is not None
        or agent_config.get("name") != agent
        or agent_config.get("model_name") is not None
        or agent_info.get("name") != agent
        or agent_info.get("model_info") is not None
        or agent_result.get("n_input_tokens") is not None
        or agent_result.get("n_output_tokens") is not None
        or agent_result.get("cost_usd") is not None
        or _reward(result) != reward
    ):
        raise AnalogAdmissionError(f"{label} trial evidence drift")


def _validate_outcome(
    root: Path,
    raw: Any,
    *,
    task_id: str,
    task_dir_relative: str,
    task_checksum: str,
    expected_agent: str,
    expected_reward: float,
    label: str,
) -> None:
    outcome = _object(
        raw,
        fields={"agent", "reward", "job_result", "trial_result"},
        label=label,
    )
    if outcome["agent"] != expected_agent or outcome["reward"] != expected_reward:
        raise AnalogAdmissionError(f"{label} registration drift")
    job_path = _verified_file(root, outcome["job_result"], label=f"{label} job")
    trial_path = _verified_file(
        root, outcome["trial_result"], label=f"{label} trial"
    )
    _validate_job_result(
        _read_json(job_path, label=f"{label} job"),
        agent=expected_agent,
        reward=expected_reward,
        label=f"{label} job",
    )
    _validate_trial_result(
        _read_json(trial_path, label=f"{label} trial"),
        task_id=task_id,
        task_dir_relative=task_dir_relative,
        task_checksum=task_checksum,
        agent=expected_agent,
        reward=expected_reward,
        label=f"{label} trial",
    )


def admit_analog_band(
    manifest_path: Path,
    *,
    root: Path,
    observed_image_ids: Mapping[str, str] | None = None,
) -> AnalogBandAdmission:
    """Admit only exact train-only tasks with passing oracles and failing baselines."""
    root = root.resolve()
    manifest = _object(
        _read_json(manifest_path, label="analog admission manifest"),
        fields={
            "version",
            "band_id",
            "validator",
            "pool_binding",
            "policy",
            "tasks",
        },
        label="analog admission manifest",
    )
    if manifest["version"] != ADMISSION_VERSION:
        raise AnalogAdmissionError("unsupported analog admission version")
    if not isinstance(manifest["band_id"], str) or not manifest["band_id"].strip():
        raise AnalogAdmissionError("band_id must be non-empty")

    validator = _verified_file(root, manifest["validator"], label="validator")
    if validator.resolve() != Path(__file__).resolve():
        raise AnalogAdmissionError("validator is not this implementation")

    binding_record = _object(
        manifest["pool_binding"],
        fields={"path", "sha256", "pool_id", "pool_fingerprint"},
        label="pool binding",
    )
    binding_path = _project_path(root, binding_record["path"], label="pool binding")
    if not binding_path.is_file() or sha256(binding_path) != binding_record["sha256"]:
        raise AnalogAdmissionError("pool binding hash drift")
    binding = load_pool_binding(binding_path)
    if (
        binding.pool_id != binding_record["pool_id"]
        or binding.pool_fingerprint != binding_record["pool_fingerprint"]
        or binding.provider_base != "https://yunwu.ai/v1"
    ):
        raise AnalogAdmissionError("pool binding identity or provider drift")

    policy = _object(
        manifest["policy"],
        fields={
            "training_only",
            "terminalbench_derived",
            "admission_model_calls",
            "required_oracle_reward",
            "required_unchanged_reward",
            "learning_target",
        },
        label="analog policy",
    )
    if policy != {
        "training_only": True,
        "terminalbench_derived": False,
        "admission_model_calls": 0,
        "required_oracle_reward": 1.0,
        "required_unchanged_reward": 0.0,
        "learning_target": "initial_role_topology_selection",
    }:
        raise AnalogAdmissionError("analog policy drift")

    raw_tasks = manifest["tasks"]
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 2:
        raise AnalogAdmissionError("analog band must contain exactly two tasks")
    admitted: list[AdmittedAnalog] = []
    seen_tasks: set[str] = set()
    seen_mechanisms: set[str] = set()
    forbidden = {
        binding.provider_base.lower(),
        *[slot.runtime_model.lower() for slot in binding.slots],
        *[slot.model_alias.lower() for slot in binding.slots],
        *[slot.training_name.lower() for slot in binding.slots],
    }

    for index, raw_task in enumerate(raw_tasks):
        task = _object(
            raw_task,
            fields={
                "task_id",
                "mechanism_id",
                "task_dir",
                "task_tree_sha256",
                "instruction_sha256",
                "docker_image",
                "docker_image_id",
                "task_checksum",
                "oracle",
                "unchanged_baseline",
            },
            label=f"tasks[{index}]",
        )
        task_id = task["task_id"]
        mechanism_id = task["mechanism_id"]
        if (
            not isinstance(task_id, str)
            or not task_id.startswith("fugu-train/")
            or task_id in seen_tasks
            or not isinstance(mechanism_id, str)
            or not mechanism_id.strip()
            or mechanism_id in seen_mechanisms
        ):
            raise AnalogAdmissionError("analog task or mechanism identity drift")
        seen_tasks.add(task_id)
        seen_mechanisms.add(mechanism_id)

        task_dir = _project_path(root, task["task_dir"], label=f"{task_id} task_dir")
        if not task_dir.is_dir() or tree_sha256(task_dir) != task["task_tree_sha256"]:
            raise AnalogAdmissionError(f"{task_id} task tree drift")
        instruction = task_dir / "instruction.md"
        task_toml = task_dir / "task.toml"
        if (
            not instruction.is_file()
            or sha256(instruction) != task["instruction_sha256"]
            or not task_toml.is_file()
        ):
            raise AnalogAdmissionError(f"{task_id} instruction drift")
        parsed = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        metadata = parsed.get("metadata") or {}
        task_config = parsed.get("task") or {}
        image = (parsed.get("environment") or {}).get("docker_image")
        if (
            task_config.get("name") != task_id
            or "train-only" not in (metadata.get("tags") or [])
            or "agentic" not in (metadata.get("tags") or [])
            or image != task["docker_image"]
            or not isinstance(task["docker_image_id"], str)
            or not task["docker_image_id"].startswith("sha256:")
        ):
            raise AnalogAdmissionError(f"{task_id} train-only task metadata drift")
        if observed_image_ids is not None and (
            observed_image_ids.get(image) != task["docker_image_id"]
        ):
            raise AnalogAdmissionError(f"{task_id} Docker image drift")

        learned_bytes = b"\n".join(
            path.read_bytes().lower()
            for path in sorted(item for item in task_dir.rglob("*") if item.is_file())
        )
        if any(token.encode("utf-8") in learned_bytes for token in forbidden):
            raise AnalogAdmissionError(f"{task_id} embeds current pool identity")

        relative = task_dir.relative_to(root).as_posix()
        checksum = task["task_checksum"]
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise AnalogAdmissionError(f"{task_id} task checksum is invalid")
        _validate_outcome(
            root,
            task["oracle"],
            task_id=task_id,
            task_dir_relative=relative,
            task_checksum=checksum,
            expected_agent="oracle",
            expected_reward=1.0,
            label=f"{task_id} oracle",
        )
        _validate_outcome(
            root,
            task["unchanged_baseline"],
            task_id=task_id,
            task_dir_relative=relative,
            task_checksum=checksum,
            expected_agent="nop",
            expected_reward=0.0,
            label=f"{task_id} unchanged baseline",
        )
        admitted.append(
            AdmittedAnalog(
                task_id=task_id,
                mechanism_id=mechanism_id,
                task_dir=task_dir,
                task_checksum=checksum,
                docker_image=image,
                docker_image_id=task["docker_image_id"],
            )
        )

    return AnalogBandAdmission(
        band_id=manifest["band_id"],
        pool_id=binding.pool_id,
        pool_fingerprint=binding.pool_fingerprint,
        tasks=tuple(admitted),
    )
