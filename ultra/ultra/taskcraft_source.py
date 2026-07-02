"""TaskCraft source probe for controlled Fugu-Ultra data expansion.

The public TaskCraft data is useful for long-context/tool-discovery task mining,
but it is not directly GRPO-ready: rows are answer-keyed and many depend on live
web/PDF/image tools. This module filters the text-only PDF/HTML subset and emits
candidate rows plus a readiness report. Promotion to TaskSpec happens only after
source documents are frozen and deterministic grading is calibrated.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path
from typing import Any

VERSION = "fugu_ultra_taskcraft_source_probe_v1"
AUDIT_VERSION = "fugu_ultra_taskcraft_readiness_audit_v1"

DATA_FILES = ("pure_qa.jsonl", "multihop_subtask_trace.jsonl", "atomic_trace.jsonl")
TEXT_DOMAIN_MARKERS = ("html_webtool", "pdftool")
IMAGE_MARKERS = ("imagetool", "image_tool", "inspect_file_as_image")
URL_RE = re.compile(r"https?://[^\s'\"<>)}\]]+")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_line_no", line_no)
            rows.append(row)
    return rows


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _tools(row: dict[str, Any]) -> list[str]:
    raw = row.get("tool")
    if raw is None:
        raw = row.get("tool_call_names")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if raw:
        return [str(raw)]
    return []


def _normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _answer_shape(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "empty"
    if len(text) > 120:
        return "long_text"
    if URL_RE.search(text):
        return "url"
    if re.fullmatch(r"[$€£¥]?\s?[\d,]+(?:\.\d+)?(?:\s?[%$€£¥])?(?:\s+[A-Za-z][A-Za-z -]{0,40})?", text):
        return "numeric_or_quantity"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:\s+to\s+\d{4}-\d{2}-\d{2})?", text):
        return "date_or_range"
    if any(sep in text for sep in [";", " and ", " to ", " / "]):
        return "multi_part_short_text"
    return "short_text"


def _parse_agent_payload(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        payload = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_trace_evidence(row: dict[str, Any]) -> dict[str, Any]:
    payload = _parse_agent_payload(row.get("ans_from_agent")) or {}
    trace = payload.get("trace") if isinstance(payload, dict) else {}
    actions = trace.get("actions") if isinstance(trace, dict) else []
    if not isinstance(actions, list):
        actions = []
    observations = [
        str(action.get("observation", ""))
        for action in actions
        if isinstance(action, dict) and action.get("observation") is not None
    ]
    joined_observations = "\n".join(observations)
    urls = sorted(set(URL_RE.findall(joined_observations + "\n" + str(row.get("ans_from_agent", "")))))
    answer = row.get("golden_answer")
    normalized_answer = _normalize_answer(answer)
    observation_contains_answer = bool(normalized_answer and normalized_answer in _normalize_answer(joined_observations))
    agent_answer = payload.get("answer") if isinstance(payload, dict) else None
    agent_answer_contains_answer = bool(normalized_answer and normalized_answer in _normalize_answer(agent_answer))
    return {
        "query": row.get("query"),
        "golden_answer": answer,
        "answer_shape": _answer_shape(answer),
        "content_identifier": row.get("content_identifier"),
        "tool_call_names": row.get("tool_call_names") or [],
        "tool_calls_count": row.get("tool_calls_count"),
        "score": row.get("score"),
        "action_count": len(actions),
        "observation_count": len(observations),
        "url_count": len(urls),
        "urls_sample": urls[:5],
        "has_crawl_observation": any(
            isinstance(action, dict) and str(action.get("tool_name", "")).lower() == "crawl_pages"
            for action in actions
        ),
        "observation_contains_answer": observation_contains_answer,
        "agent_answer_contains_answer": agent_answer_contains_answer,
    }


def _is_image_row(domain: str, tools: list[str]) -> bool:
    haystack = " ".join([domain, *tools]).lower()
    return any(marker in haystack for marker in IMAGE_MARKERS)


def _is_text_pdf_or_html(domain: str) -> bool:
    domain_lower = domain.lower()
    return any(marker in domain_lower for marker in TEXT_DOMAIN_MARKERS)


def _summary_counts(rows: list[dict[str, Any]], *, multihop: bool = False) -> dict[str, Any]:
    domains = Counter(str(row.get("domain", "")) for row in rows)
    total = len(rows)
    image_rows = 0
    text_pdf_html = 0
    hop_ge_3 = 0
    for row in rows:
        domain = str(row.get("domain", ""))
        tools = _tools(row)
        is_image = _is_image_row(domain, tools)
        image_rows += int(is_image)
        text_pdf_html += int((not is_image) and _is_text_pdf_or_html(domain))
        if multihop:
            trace = row.get("trace") or []
            max_hop = max((int(item.get("valid_hop") or 0) for item in trace), default=0)
        else:
            max_hop = int(row.get("valid_hop") or 0)
        hop_ge_3 += int(max_hop >= 3)
    return {
        "rows": total,
        "text_only_domain_rows": total - image_rows,
        "image_domain_rows": image_rows,
        "text_pdf_or_html_rows": text_pdf_html,
        "hop_ge_3_rows": hop_ge_3,
        "top_domains": dict(domains.most_common(12)),
    }


def _pure_candidate(row: dict[str, Any], *, origin_file: str) -> dict[str, Any] | None:
    domain = str(row.get("domain", ""))
    tools = _tools(row)
    if _is_image_row(domain, tools) or not _is_text_pdf_or_html(domain):
        return None
    valid_hop = int(row.get("valid_hop") or 0)
    if valid_hop < 3:
        return None
    line_no = int(row["_line_no"])
    return {
        "candidate_id": f"taskcraft_pure_qa_{line_no:06d}",
        "origin_file": origin_file,
        "line_no": line_no,
        "domain": domain,
        "tools": tools,
        "valid_hop": valid_hop,
        "query": str(row.get("query", "")),
        "golden_answer": row.get("golden_answer"),
        "suggested_lanes": ["long_context_memory_planning", "tool_dialogue"],
        "suggested_harness": "long_context",
        "source_freeze_required": True,
        "deterministic_grader_required": True,
        "grpo_ready": False,
        "readiness_blockers": [
            "source_documents_not_frozen",
            "deterministic_answer_grader_not_calibrated",
            "web_or_pdf_staleness_risk",
            "source_content_alignment_not_audited",
        ],
    }


def _multihop_candidate(row: dict[str, Any], *, origin_file: str) -> dict[str, Any] | None:
    domain = str(row.get("domain", ""))
    if _is_image_row(domain, []) or not _is_text_pdf_or_html(domain):
        return None
    trace = [item for item in row.get("trace") or [] if isinstance(item, dict)]
    if not trace:
        return None
    best = max(trace, key=lambda item: int(item.get("valid_hop") or 0))
    valid_hop = int(best.get("valid_hop") or 0)
    if valid_hop < 3:
        return None
    line_no = int(row["_line_no"])
    return {
        "candidate_id": f"taskcraft_multihop_{line_no:06d}",
        "origin_file": origin_file,
        "line_no": line_no,
        "domain": domain,
        "tools": ["web_tool" if "html" in domain else "pdf_tool"],
        "valid_hop": valid_hop,
        "query": str(best.get("query", "")),
        "golden_answer": best.get("golden_answer"),
        "subtasks": [
            {
                "query": str(item.get("query", "")),
                "golden_answer": item.get("golden_answer"),
                "valid_hop": int(item.get("valid_hop") or 0),
            }
            for item in trace
        ],
        "suggested_lanes": ["long_context_memory_planning", "tool_dialogue"],
        "suggested_harness": "long_context",
        "source_freeze_required": True,
        "deterministic_grader_required": True,
        "grpo_ready": False,
        "readiness_blockers": [
            "source_documents_not_frozen",
            "deterministic_answer_grader_not_calibrated",
            "web_or_pdf_staleness_risk",
            "source_content_alignment_not_audited",
            "multihop_chain_consistency_not_audited",
        ],
    }


def _select_stratified(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(candidates) <= limit:
        return candidates
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        buckets[str(row["domain"])].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda item: (-int(item.get("valid_hop") or 0), str(item["candidate_id"])))
    domains = sorted(buckets, key=lambda domain: (-len(buckets[domain]), domain))
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and domains:
        next_domains = []
        for domain in domains:
            if buckets[domain] and len(selected) < limit:
                selected.append(buckets[domain].pop(0))
            if buckets[domain]:
                next_domains.append(domain)
        domains = next_domains
    return selected


def build_taskcraft_source_probe(
    *,
    dataset_dir: Path,
    candidates_out: Path,
    report_out: Path,
    limit: int = 200,
) -> dict[str, Any]:
    paths = {name: dataset_dir / name for name in DATA_FILES}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing TaskCraft dataset files: {missing}")

    pure_rows = _read_jsonl(paths["pure_qa.jsonl"])
    multihop_rows = _read_jsonl(paths["multihop_subtask_trace.jsonl"])
    atomic_rows = _read_jsonl(paths["atomic_trace.jsonl"])

    candidates: list[dict[str, Any]] = []
    candidates.extend(
        candidate
        for row in pure_rows
        if (candidate := _pure_candidate(row, origin_file="pure_qa.jsonl")) is not None
    )
    candidates.extend(
        candidate
        for row in multihop_rows
        if (candidate := _multihop_candidate(row, origin_file="multihop_subtask_trace.jsonl")) is not None
    )
    selected = _select_stratified(candidates, limit)
    _write_jsonl(candidates_out, selected)

    candidate_domains = Counter(str(row["domain"]) for row in selected)
    candidate_files = Counter(str(row["origin_file"]) for row in selected)
    report = {
        "version": VERSION,
        "status": "candidate_source_not_grpo_ready",
        "dataset_dir": str(dataset_dir.resolve()),
        "candidates_out": str(candidates_out.resolve()),
        "files": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "summaries": {
            "pure_qa": _summary_counts(pure_rows),
            "multihop_subtask_trace": _summary_counts(multihop_rows, multihop=True),
            "atomic_trace": _summary_counts(atomic_rows),
        },
        "candidate_filter": {
            "include_domains": list(TEXT_DOMAIN_MARKERS),
            "exclude_markers": list(IMAGE_MARKERS),
            "min_valid_hop": 3,
            "limit": limit,
            "selection": "domain-stratified, highest-hop first",
        },
        "candidate_count_before_limit": len(candidates),
        "candidate_count": len(selected),
        "candidate_domain_counts": dict(candidate_domains.most_common()),
        "candidate_origin_counts": dict(sorted(candidate_files.items())),
        "raw_dataset_grpo_ready": False,
        "readiness_blockers": [
            "source_documents_not_frozen",
            "deterministic_answer_grader_not_calibrated",
            "web_or_pdf_staleness_risk",
            "answer_key_matching_may_be_semantic_not_exact",
            "source_content_alignment_not_audited",
            "multihop_chain_consistency_not_audited",
        ],
        "recommended_next_steps": [
            "Audit generated multihop chains for source/entity drift before source freeze.",
            "Freeze source documents/pages for a small text-only PDF/HTML slice.",
            "Build deterministic contains/exact/semantic-normalized graders for the frozen slice.",
            "Run high-reasoning single-worker prefilter and role followup before admitting rows to GRPO.",
            "Use generated TaskCraft depth/width tasks only after the same source-freeze and grading gates.",
        ],
    }
    _write_json(report_out, report)
    return report


def build_taskcraft_readiness_audit(
    *,
    dataset_dir: Path,
    candidates_jsonl: Path,
    report_out: Path,
    evidence_out: Path | None = None,
) -> dict[str, Any]:
    """Audit whether TaskCraft candidates can be promoted to GRPO TaskSpecs.

    The audit is intentionally conservative. It can identify candidates that are
    good source-freeze priorities, but it never marks rows GRPO-ready unless the
    frozen-document and grader gates are actually satisfied.
    """

    atomic_path = dataset_dir / "atomic_trace.jsonl"
    if not atomic_path.exists():
        raise FileNotFoundError(f"missing TaskCraft atomic trace file: {atomic_path}")
    candidates = _load_jsonl(candidates_jsonl)

    atomic_by_query: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(atomic_path):
        query = str(row.get("query", ""))
        if query and query not in atomic_by_query:
            atomic_by_query[query] = row

    evidence_rows: list[dict[str, Any]] = []
    blocker_counts: Counter[str] = Counter()
    answer_shapes: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    linkage_counts: Counter[str] = Counter()

    for candidate in candidates:
        subtasks = [item for item in candidate.get("subtasks") or [] if isinstance(item, dict)]
        required_queries = [str(candidate.get("query", "")), *[str(item.get("query", "")) for item in subtasks]]
        required_queries = [query for query in required_queries if query]
        matched = [atomic_by_query[query] for query in required_queries if query in atomic_by_query]
        final_match = atomic_by_query.get(str(candidate.get("query", "")))
        atomic_evidence = [_atomic_trace_evidence(row) for row in matched]
        content_identifiers = sorted(
            {
                str(item.get("content_identifier"))
                for item in atomic_evidence
                if item.get("content_identifier")
            }
        )
        answer_shape = _answer_shape(candidate.get("golden_answer"))
        answer_shapes[answer_shape] += 1
        domain_counts[str(candidate.get("domain"))] += 1
        origin_counts[str(candidate.get("origin_file"))] += 1

        blockers = {
            "source_documents_not_frozen",
            "deterministic_answer_grader_not_calibrated",
            "web_or_pdf_staleness_risk",
            "source_content_alignment_not_audited",
        }
        if subtasks:
            blockers.add("multihop_chain_consistency_not_audited")
        if not matched:
            blockers.add("missing_atomic_trace_source_anchor")
        if len(matched) < len(required_queries):
            blockers.add("incomplete_atomic_trace_linkage")
        if final_match is None:
            blockers.add("final_query_not_atomic_trace_anchored")
        if not any(item.get("observation_contains_answer") for item in atomic_evidence):
            blockers.add("answer_not_verified_in_frozen_observation")
        if answer_shape in {"empty", "long_text", "url"}:
            blockers.add("answer_shape_requires_manual_grader_policy")

        for blocker in blockers:
            blocker_counts[blocker] += 1
        if matched:
            linkage_counts["any_atomic_match"] += 1
        if final_match is not None:
            linkage_counts["final_atomic_match"] += 1
        if required_queries and len(matched) == len(required_queries):
            linkage_counts["all_required_queries_matched"] += 1
        if content_identifiers:
            linkage_counts["any_content_identifier"] += 1
        if any(item.get("has_crawl_observation") for item in atomic_evidence):
            linkage_counts["any_crawl_observation"] += 1
        if any(item.get("observation_contains_answer") for item in atomic_evidence):
            linkage_counts["any_observation_contains_answer"] += 1

        evidence_rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "domain": candidate.get("domain"),
                "origin_file": candidate.get("origin_file"),
                "valid_hop": candidate.get("valid_hop"),
                "answer_shape": answer_shape,
                "required_query_count": len(required_queries),
                "atomic_match_count": len(matched),
                "final_query_atomic_match": final_match is not None,
                "content_identifiers": content_identifiers[:8],
                "atomic_urls_sample": sorted({url for item in atomic_evidence for url in item.get("urls_sample", [])})[:8],
                "has_crawl_observation": any(item.get("has_crawl_observation") for item in atomic_evidence),
                "observation_contains_answer": any(item.get("observation_contains_answer") for item in atomic_evidence),
                "agent_answer_contains_answer": any(item.get("agent_answer_contains_answer") for item in atomic_evidence),
                "grpo_ready": False,
                "readiness_blockers": sorted(blockers),
            }
        )

    freeze_priority = [
        row
        for row in evidence_rows
        if row["content_identifiers"] and row["has_crawl_observation"] and row["observation_contains_answer"]
    ]
    report = {
        "version": AUDIT_VERSION,
        "status": "audited_not_grpo_ready",
        "raw_dataset_grpo_ready": False,
        "dataset_dir": str(dataset_dir.resolve()),
        "candidates_jsonl": str(candidates_jsonl.resolve()),
        "evidence_out": str(evidence_out.resolve()) if evidence_out else None,
        "candidate_count": len(candidates),
        "atomic_trace_rows": len(atomic_by_query),
        "domain_counts": dict(domain_counts.most_common()),
        "origin_counts": dict(sorted(origin_counts.items())),
        "answer_shape_counts": dict(answer_shapes.most_common()),
        "linkage_counts": dict(linkage_counts),
        "readiness_blocker_counts": dict(blocker_counts.most_common()),
        "freeze_priority_count": len(freeze_priority),
        "freeze_priority_examples": [
            {
                "candidate_id": row["candidate_id"],
                "domain": row["domain"],
                "content_identifiers": row["content_identifiers"][:3],
                "atomic_urls_sample": row["atomic_urls_sample"][:3],
            }
            for row in freeze_priority[:12]
        ],
        "decision": {
            "promote_to_grpo": False,
            "reason": (
                "TaskCraft candidates lack complete frozen source documents and complete atomic-trace linkage. "
                "Use the audit to choose a small manual source-freeze slice, not as GRPO data."
            ),
            "next_steps": [
                "Select a small freeze-priority slice with crawl observations and answer-in-observation evidence.",
                "Fetch or otherwise freeze the actual PDF/HTML source documents for that slice.",
                "Build normalized deterministic graders from frozen source spans.",
                "Rerun single-worker and role-workflow discovery before adding any TaskCraft row to GRPO.",
            ],
        },
    }
    if evidence_out is not None:
        _write_jsonl(evidence_out, evidence_rows)
    _write_json(report_out, report)
    return report
