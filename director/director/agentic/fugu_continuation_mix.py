"""Build a replay-heavy continuation mix with outcome-backed current-pool labels."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from director.agentic.fugu_initial_topology_admission import (
    admit_initial_topology_labels,
)
from director.agentic.fugu_recovery_admission import convert_recovery_admission
from director.agentic.fugu_replay_migration import migrate_role_neutral_replay
from ultra.pool_binding import load_pool_binding


MIX_VERSION = "fugu_continuation_mix_v1"


class ContinuationMixError(ValueError):
    """The continuation mix changed provenance, balance, or pool semantics."""


@dataclass(frozen=True)
class ContinuationMix:
    rows: tuple[dict[str, Any], ...]
    action_counts: dict[str, int]
    source_counts: dict[str, int]
    task_count: int


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(value: Any, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ContinuationMixError(f"{label} must contain exactly {sorted(fields)}")
    return value


def _project_file(root: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ContinuationMixError(f"{label} must be a non-empty path")
    path = Path(raw)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ContinuationMixError(f"{label} escapes project root") from exc
    if not path.is_file():
        raise ContinuationMixError(f"{label} is missing")
    return path


def _verified_file(root: Path, raw: Any, *, label: str) -> Path:
    record = _object(raw, fields={"path", "sha256"}, label=label)
    path = _project_file(root, record["path"], label=label)
    if sha256(path) != record["sha256"]:
        raise ContinuationMixError(f"{label} hash drift")
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContinuationMixError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContinuationMixError(f"{label} must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContinuationMixError(f"{label} line {number} is invalid") from exc
        if not isinstance(row, dict):
            raise ContinuationMixError(f"{label} line {number} is not an object")
        rows.append(row)
    return rows


def _repeat_rows(
    rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    source: str,
    weight: int,
    source_sha256: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for repeat in range(weight):
        for raw in rows:
            row = copy.deepcopy(raw)
            row["record_id"] = f"{row['record_id']}__mix_{source}_{repeat:02d}"
            provenance = row.get("provenance")
            if not isinstance(provenance, dict):
                raise ContinuationMixError(f"{source} row lacks provenance")
            provenance.update(
                {
                    "continuation_mix_source": source,
                    "continuation_mix_repeat": repeat,
                    "continuation_mix_source_sha256": source_sha256,
                }
            )
            result.append(row)
    return result


def build_continuation_mix(manifest_path: Path, *, root: Path) -> ContinuationMix:
    root = root.resolve()
    manifest = _object(
        _read_json(manifest_path, label="continuation mix manifest"),
        fields={
            "version",
            "mix_id",
            "mixer",
            "pool_binding",
            "replay_migration",
            "replay_rows",
            "recovery_admission",
            "recovery_rows",
            "initial_topology_admission",
            "initial_topology_rows",
            "weights",
            "policy",
        },
        label="continuation mix manifest",
    )
    if manifest["version"] != MIX_VERSION:
        raise ContinuationMixError("unsupported continuation mix version")
    mixer = _verified_file(root, manifest["mixer"], label="mixer")
    if mixer.resolve() != Path(__file__).resolve():
        raise ContinuationMixError("mixer is not this implementation")
    binding_record = _object(
        manifest["pool_binding"],
        fields={"path", "sha256", "pool_id", "pool_fingerprint"},
        label="pool binding",
    )
    binding_path = _project_file(root, binding_record["path"], label="pool binding")
    if sha256(binding_path) != binding_record["sha256"]:
        raise ContinuationMixError("pool binding hash drift")
    binding = load_pool_binding(binding_path)
    if (
        binding.pool_id != binding_record["pool_id"]
        or binding.pool_fingerprint != binding_record["pool_fingerprint"]
    ):
        raise ContinuationMixError("pool binding identity drift")
    weights = _object(
        manifest["weights"],
        fields={"anti_forgetting_replay", "recovery_boundaries", "initial_topology"},
        label="weights",
    )
    if weights != {
        "anti_forgetting_replay": 1,
        "recovery_boundaries": 8,
        "initial_topology": 32,
    }:
        raise ContinuationMixError("continuation mix weights drift")
    policy = _object(
        manifest["policy"],
        fields={
            "replay_usage",
            "current_pool_fraction_max",
            "causal_superiority_claims",
            "terminalbench_source_tasks_evaluation_excluded",
            "guardian_labels_in_training",
        },
        label="policy",
    )
    if policy != {
        "replay_usage": "anti_forgetting_replay_only",
        "current_pool_fraction_max": 0.15,
        "causal_superiority_claims": False,
        "terminalbench_source_tasks_evaluation_excluded": True,
        "guardian_labels_in_training": False,
    }:
        raise ContinuationMixError("continuation mix policy drift")

    replay_manifest = _verified_file(
        root, manifest["replay_migration"], label="replay migration"
    )
    replay_path = _verified_file(root, manifest["replay_rows"], label="replay rows")
    recovery_manifest = _verified_file(
        root, manifest["recovery_admission"], label="recovery admission"
    )
    recovery_path = _verified_file(
        root, manifest["recovery_rows"], label="recovery rows"
    )
    topology_manifest = _verified_file(
        root,
        manifest["initial_topology_admission"],
        label="initial topology admission",
    )
    topology_path = _verified_file(
        root, manifest["initial_topology_rows"], label="initial topology rows"
    )

    replay = migrate_role_neutral_replay(replay_manifest, root=root)
    recovery = convert_recovery_admission(recovery_manifest, root=root)
    topology = admit_initial_topology_labels(topology_manifest, root=root)
    materialized_replay = _read_jsonl(replay_path, label="replay rows")
    materialized_recovery = _read_jsonl(recovery_path, label="recovery rows")
    materialized_topology = _read_jsonl(topology_path, label="initial topology rows")
    if materialized_replay != list(replay.rows):
        raise ContinuationMixError("materialized replay differs from migration")
    if materialized_recovery != list(recovery.rows):
        raise ContinuationMixError("materialized recovery differs from admission")
    if materialized_topology != list(topology.rows):
        raise ContinuationMixError("materialized topology differs from admission")

    mixed = [
        *_repeat_rows(
            replay.rows,
            source="anti_forgetting_replay",
            weight=weights["anti_forgetting_replay"],
            source_sha256=sha256(replay_path),
        ),
        *_repeat_rows(
            recovery.rows,
            source="recovery_boundaries",
            weight=weights["recovery_boundaries"],
            source_sha256=sha256(recovery_path),
        ),
        *_repeat_rows(
            topology.rows,
            source="initial_topology",
            weight=weights["initial_topology"],
            source_sha256=sha256(topology_path),
        ),
    ]
    ids = [row.get("record_id") for row in mixed]
    if any(not isinstance(value, str) for value in ids) or len(ids) != len(set(ids)):
        raise ContinuationMixError("mixed record IDs are not unique")
    current_pool_rows = (
        len(recovery.rows) * weights["recovery_boundaries"]
        + len(topology.rows) * weights["initial_topology"]
    )
    if current_pool_rows / len(mixed) > policy["current_pool_fraction_max"]:
        raise ContinuationMixError("current-pool labels exceed anti-forgetting cap")
    if any(
        row.get("pool_fingerprint") != binding.pool_fingerprint for row in mixed
    ):
        raise ContinuationMixError("mixed row pool fingerprint drift")
    terminalbench_rows = [row for row in mixed if row.get("terminalbench") is True]
    if any(row.get("evaluation_excluded") is not True for row in terminalbench_rows):
        raise ContinuationMixError("TerminalBench source escaped eval exclusion")
    counts = Counter(row["action"]["action"] for row in mixed)
    sources = Counter(
        row["provenance"]["continuation_mix_source"] for row in mixed
    )
    return ContinuationMix(
        rows=tuple(mixed),
        action_counts=dict(sorted(counts.items())),
        source_counts=dict(sorted(sources.items())),
        task_count=len({row["task_id"] for row in mixed}),
    )
