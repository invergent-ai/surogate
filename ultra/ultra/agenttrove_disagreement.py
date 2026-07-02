"""Offline disagreement priors from AgentTrove metadata.

AgentTrove is trace data, not a local verifier. This scanner only uses metadata
columns to find tasks with mixed historical outcomes so TaskTrove prefilters can
prioritize likely-discriminative tasks before spending live worker calls.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any

VERSION = "fugu_ultra_agenttrove_disagreement_v1"

SUCCESS_STRINGS = {
    "1",
    "1.0",
    "true",
    "success",
    "succeeded",
    "pass",
    "passed",
    "correct",
    "valid_correct_trainable",
}
FAILURE_STRINGS = {
    "0",
    "0.0",
    "false",
    "failure",
    "failed",
    "fail",
    "incorrect",
    "valid_incorrect_trainable",
}

SOURCE_ALIASES = {
    "inferred_bugs": "inferredbugs",
    "inferredbugs": "inferredbugs",
    "py_methods2test": "pymethods2test",
    "pymethods2test": "pymethods2test",
    "methods2test": "pymethods2test",
    "nl2bash": "nl2bash",
    "r2egym": "r2egym",
    "swesmith": "swesmith",
    "swe_smith": "swesmith",
    "code_contests": "code_contests",
    "codeforces": "code_contests",
    "freelancer": "freelancer",
    "stack_bash": "stack_bash",
    "stack_exchange": "stack_exchange",
}

TASK_COLUMNS = ("task_id", "task", "path")
SOURCE_COLUMNS = ("original_source", "trace_source")
TEACHER_COLUMNS = ("original_teacher", "model", "agent")
OUTCOME_COLUMNS = ("reward", "result", "judgment", "verifier_output")
CONVERSATION_COLUMNS = ("conversations",)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _normalize_source(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return SOURCE_ALIASES.get(normalized, normalized)


def suggested_tasktrove_source(value: str) -> str | None:
    normalized = _normalize_source(value)
    if not normalized:
        return None
    return f"tasktrove_{normalized}"


def _first_present(row: dict[str, Any], columns: tuple[str, ...]) -> Any:
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return value
    return None


def _parse_outcome(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value) > 0.0
    if isinstance(value, dict):
        for key in ("success", "reward", "score", "result", "passed"):
            if key in value:
                return _parse_outcome(value[key])
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower in SUCCESS_STRINGS:
        return True
    if lower in FAILURE_STRINGS:
        return False
    try:
        return float(lower) > 0.0
    except ValueError:
        pass
    if lower.startswith("{") and lower.endswith("}"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return _parse_outcome(parsed)
    return None


def _parse_self_reported_completion(value: Any) -> bool | None:
    if not isinstance(value, list):
        return None
    pattern = re.compile(r"""["']?task_complete["']?\s*:\s*(true|false)""", re.IGNORECASE)
    for message in reversed(value):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").lower() != "assistant":
            continue
        content = str(message.get("content") or "")
        matches = pattern.findall(content)
        if matches:
            return matches[-1].lower() == "true"
    return None


_TASK_COMPLETE_PATTERN = re.compile(r"""["']?task_complete["']?\s*:\s*(true|false)""", re.IGNORECASE)


def _parse_task_complete_text(content: str) -> bool | None:
    matches = _TASK_COMPLETE_PATTERN.findall(content)
    if matches:
        return matches[-1].lower() == "true"
    return None


def _parse_self_reported_completion_arrow(list_array: Any, index: int) -> bool | None:
    scalar = list_array[index]
    if not scalar.is_valid:
        return None
    try:
        start = int(list_array.offsets[index].as_py())
        end = int(list_array.offsets[index + 1].as_py())
        values = list_array.values
        roles = values.field("role")
        contents = values.field("content")
    except Exception:
        return _parse_self_reported_completion(scalar.as_py())
    for item_index in range(end - 1, start - 1, -1):
        role = roles[item_index].as_py()
        if str(role or "").lower() != "assistant":
            continue
        content = contents[item_index].as_py()
        if content is None:
            continue
        parsed = _parse_task_complete_text(str(content))
        if parsed is not None:
            return parsed
    return None


def _local_tasktrove_index(manifest_dir: Path | None) -> dict[str, set[str]]:
    if manifest_dir is None:
        return {}
    try:
        from .tasktrove_prefilter import discover_tasktrove_task_files
    except Exception:
        return {}

    index: dict[str, set[str]] = defaultdict(set)
    for source, paths in discover_tasktrove_task_files(manifest_dir).items():
        for path in paths:
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    task_id = str(row.get("task_id") or row.get("source_task_id") or "")
                    if task_id:
                        index[source].add(task_id)
                        if task_id.startswith(f"{source}__"):
                            index[source].add(task_id.split("__", 1)[1])
    return dict(index)


def _available_columns(parquet_path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read AgentTrove parquet shards") from exc

    parquet = pq.ParquetFile(parquet_path)
    try:
        return set(parquet.schema_arrow.names)
    except Exception:  # pragma: no cover - fallback for older pyarrow
        return set(parquet.schema.names)


def _iter_metadata_rows(
    parquet_path: Path,
    *,
    batch_size: int,
    limit_rows: int | None,
    allow_self_reported_completion: bool,
) -> tuple[dict[str, Any], Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read AgentTrove parquet shards") from exc

    available = _available_columns(parquet_path)
    columns = [
        column
        for column in (*TASK_COLUMNS, *SOURCE_COLUMNS, *TEACHER_COLUMNS, *OUTCOME_COLUMNS)
        if column in available
    ]
    if allow_self_reported_completion:
        columns.extend(column for column in CONVERSATION_COLUMNS if column in available and column not in columns)
    metadata = {
        "path": str(parquet_path),
        "available_columns": sorted(available),
        "scanned_columns": columns,
        "has_task_column": any(column in available for column in TASK_COLUMNS),
        "has_outcome_column": any(column in available for column in OUTCOME_COLUMNS),
        "has_conversation_column": any(column in available for column in CONVERSATION_COLUMNS),
    }
    if not columns:
        return metadata, iter(())

    def gen():
        seen = 0
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
            if allow_self_reported_completion and "conversations" in columns:
                scalar_columns = [column for column in columns if column != "conversations"]
                scalar_values = {
                    column: batch.column(batch.schema.get_field_index(column)).to_pylist()
                    for column in scalar_columns
                }
                conversation_array = batch.column(batch.schema.get_field_index("conversations"))
                for row_index in range(batch.num_rows):
                    if limit_rows is not None and seen >= limit_rows:
                        return
                    seen += 1
                    row = {column: values[row_index] for column, values in scalar_values.items()}
                    row["self_reported_task_complete"] = _parse_self_reported_completion_arrow(
                        conversation_array,
                        row_index,
                    )
                    yield row
                continue
            for row in batch.to_pylist():
                if limit_rows is not None and seen >= limit_rows:
                    return
                seen += 1
                yield row

    return metadata, gen()


def _candidate_row(
    key: str,
    stats: dict[str, Any],
    *,
    local_index: dict[str, set[str]],
) -> dict[str, Any]:
    success_count = int(stats["success_count"])
    failure_count = int(stats["failure_count"])
    attempts = success_count + failure_count
    balance = min(success_count, failure_count) / max(success_count, failure_count)
    teachers = dict(sorted(stats["teachers"].items()))
    suggested_source = suggested_tasktrove_source(stats["original_source"])
    task_id = str(stats["task_id"])
    local_match = bool(suggested_source and task_id in local_index.get(suggested_source, set()))
    score = (
        balance * 10.0
        + min(attempts, 32) / 4.0
        + min(len(teachers), 8) / 2.0
        + (2.0 if local_match else 0.0)
    )
    return {
        "candidate_id": f"agenttrove::{key}",
        "original_source": stats["original_source"],
        "suggested_tasktrove_source": suggested_source,
        "task_id": task_id,
        "attempts": attempts,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_count / attempts if attempts else None,
        "teacher_count": len(teachers),
        "teachers": teachers,
        "models": dict(sorted(stats["models"].items())),
        "outcome_sources": dict(sorted(stats["outcome_sources"].items())),
        "local_tasktrove_exact_match": local_match,
        "disagreement_score": round(score, 4),
        "use_policy": "prioritize_for_local_tasktrove_prefilter_not_grpo_label",
    }


def scan_agenttrove_disagreement(
    *,
    parquet_paths: list[Path],
    candidates_out: Path,
    report_out: Path | None = None,
    manifest_dir: Path | None = None,
    source_filter: set[str] | None = None,
    min_attempts: int = 2,
    top_k: int = 1000,
    batch_size: int = 8192,
    limit_rows_per_file: int | None = None,
    allow_self_reported_completion: bool = False,
) -> dict[str, Any]:
    if min_attempts < 1:
        raise ValueError("min_attempts must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    normalized_filter = {_normalize_source(source) for source in source_filter or set()}
    local_index = _local_tasktrove_index(manifest_dir)
    file_reports: list[dict[str, Any]] = []
    task_stats: dict[str, dict[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    outcome_source_counts: Counter[str] = Counter()
    rows_scanned = 0
    rows_with_source = 0
    rows_with_task_id = 0
    rows_with_outcome = 0
    rows_used = 0

    for parquet_path in parquet_paths:
        file_meta, rows = _iter_metadata_rows(
            parquet_path,
            batch_size=batch_size,
            limit_rows=limit_rows_per_file,
            allow_self_reported_completion=allow_self_reported_completion,
        )
        file_rows = 0
        file_used = 0
        for row in rows:
            rows_scanned += 1
            file_rows += 1
            source_raw = _first_present(row, SOURCE_COLUMNS)
            task_raw = _first_present(row, TASK_COLUMNS)
            teacher_raw = _first_present(row, TEACHER_COLUMNS)
            outcome_raw = _first_present(row, OUTCOME_COLUMNS)
            if source_raw not in (None, ""):
                rows_with_source += 1
            if task_raw not in (None, ""):
                rows_with_task_id += 1
            outcome = _parse_outcome(outcome_raw)
            outcome_source = "metadata"
            if outcome is None and allow_self_reported_completion:
                outcome = row.get("self_reported_task_complete")
                if outcome is None:
                    outcome = _parse_self_reported_completion(row.get("conversations"))
                outcome_source = "self_reported_task_complete"
            if outcome is not None:
                rows_with_outcome += 1
                outcome_counts["success" if outcome else "failure"] += 1
                outcome_source_counts[outcome_source] += 1
            else:
                outcome_counts["missing_or_unparseable"] += 1
            if source_raw in (None, "") or task_raw in (None, "") or outcome is None:
                continue
            source = str(source_raw)
            normalized_source = _normalize_source(source)
            if normalized_filter and normalized_source not in normalized_filter:
                continue
            task_id = str(task_raw)
            teacher = str(teacher_raw or "unknown")
            model = str(row.get("model") or teacher)
            key = f"{normalized_source}::{task_id}"
            stats = task_stats.setdefault(
                key,
                {
                    "original_source": source,
                    "normalized_source": normalized_source,
                    "task_id": task_id,
                    "success_count": 0,
                    "failure_count": 0,
                    "teachers": Counter(),
                    "models": Counter(),
                    "outcome_sources": Counter(),
                },
            )
            if outcome:
                stats["success_count"] += 1
            else:
                stats["failure_count"] += 1
            stats["teachers"][teacher] += 1
            stats["models"][model] += 1
            stats["outcome_sources"][outcome_source] += 1
            source_counts[normalized_source] += 1
            teacher_counts[teacher] += 1
            rows_used += 1
            file_used += 1
        file_reports.append({**file_meta, "rows_scanned": file_rows, "rows_used": file_used})

    candidates = [
        _candidate_row(key, stats, local_index=local_index)
        for key, stats in task_stats.items()
        if int(stats["success_count"]) > 0
        and int(stats["failure_count"]) > 0
        and int(stats["success_count"]) + int(stats["failure_count"]) >= min_attempts
    ]
    candidates.sort(
        key=lambda row: (
            float(row["disagreement_score"]),
            int(row["attempts"]),
            int(row["teacher_count"]),
            str(row["original_source"]),
            str(row["task_id"]),
        ),
        reverse=True,
    )
    selected = candidates[:top_k]
    _write_jsonl(candidates_out, selected)

    report = {
        "version": VERSION,
        "purpose": "Rank AgentTrove tasks with historical mixed outcomes as an offline prior for TaskTrove prefiltering.",
        "parquet_paths": [str(path) for path in parquet_paths],
        "candidates_out": str(candidates_out),
        "manifest_dir": str(manifest_dir.resolve()) if manifest_dir else None,
        "source_filter": sorted(normalized_filter),
        "rows_scanned": rows_scanned,
        "rows_with_source": rows_with_source,
        "rows_with_task_id": rows_with_task_id,
        "rows_with_outcome": rows_with_outcome,
        "rows_used": rows_used,
        "task_keys_with_outcomes": len(task_stats),
        "disagreement_candidates": len(candidates),
        "written_candidates": len(selected),
        "source_counts": dict(sorted(source_counts.items())),
        "teacher_counts": dict(teacher_counts.most_common(25)),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "outcome_source_counts": dict(sorted(outcome_source_counts.items())),
        "files": file_reports,
        "top_candidates": selected[:25],
        "notes": [
            "AgentTrove rows are priors only; local Harbor validation and live worker disagreement remain required.",
            "Rows without task identity or parseable outcome are ignored for candidate ranking.",
            "Self-reported task_complete outcomes are weak priors when enabled, not verifier rewards.",
        ],
        "live_calls": False,
    }
    if report_out is not None:
        _write_json(report_out, report)
    return report


def download_agenttrove_parquet(
    *,
    hf_file: str,
    cache_dir: Path | None = None,
    repo_id: str = "open-thoughts/AgentTrove",
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required for --hf-file downloads") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=hf_file,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
