"""Financial-QA training manifests (office lane 3): FinQA + TAT-QA.

FinanceBench — the dataset the user named — publishes only 150 open rows, too
few to train on; the category is served instead by the two big open financial
QA sets with real TRAIN splits: FinQA (6.2k questions over report tables) and
TAT-QA (13.2k over tables+text). Both are reportable benchmarks, so ONLY the
train splits are exported and dev/test stay sealed.

Verifiability filter: rows whose gold is a single number. That is all FinQA
rows with numeric ``exe_ans``, and TAT-QA ``arithmetic``/``count`` questions
(gold parsed as one number; TAT-QA's explicit ``scale`` metadata travels in
the grader payload so percent-form normalization never depends on question
wording). Grading is ``finance_numeric``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CAPABILITY = "office_finance_qa"
GRADER = "finance_numeric"
MAX_PROMPT_CHARS = 12000
PROBE_SAMPLE_NAME = "finance_probe_candidates_taskspecs.jsonl"
DEFAULT_PROBE_SAMPLE = 400

FINQA_PROMPT = """\
Answer the question from this excerpt of a financial report.

{pre_text}

Table:
{table}

{post_text}

Question: {question}

Work it out, then give ONLY the final numeric answer on the last line. If the
quantity is a ratio or rate computed from the table, give the decimal value."""

TATQA_PROMPT = """\
Answer the question from this financial table and the accompanying text.

Table:
{table}

{paragraphs}

Question: {question}

Work it out, then give ONLY the final numeric answer on the last line{scale_note}."""


def _linearize_table(rows: list[list[Any]]) -> str:
    return "\n".join(" | ".join(str(c).strip() for c in row) for row in rows)


def _spec(task_id: str, source_name: str, url: str, prompt: str,
          answer: float, scale: str, group: str) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": CAPABILITY,
        "source": {
            "name": source_name, "url_or_ref": url, "version": f"{source_name}-train",
            "license": "MIT", "policy": "train_allowed", "source_commit": None,
        },
        "input": {"messages": [{"role": "user", "content": prompt}],
                  "context_documents": [], "assets": [], "repo": None, "tools": []},
        "environment": {"harness": "direct_qa", "image": None, "cpu_limit": None,
                        "memory_mb": None, "disk_mb": None,
                        "network_policy": "model-relay-only", "wall_time_seconds": 600},
        "grader": {"type": GRADER, "command": None,
                   "expected_answer": {"answer": answer, "scale": scale},
                   "score_range": [0.0, 1.0], "success_threshold": 1.0,
                   "deterministic": True},
        "splitting": {"split": "grpo_train", "group_id": f"{source_name}_train",
                      "contamination_group": group},
        "metadata": {"domain": CAPABILITY, "subdomain": source_name,
                     "difficulty_estimate": None, "estimated_worker_calls": 1,
                     "requires_tools": False, "requires_long_context": False,
                     "tags": [source_name, "finance", "train", "numeric_answer"]},
    }


def _as_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().rstrip("%").replace(",", "").replace("$", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    if isinstance(value, list) and len(value) == 1:
        return _as_number(value[0])
    return None


def export_finqa(train_json: Path, out_dir: Path) -> dict[str, Any]:
    rows = json.loads(train_json.read_text())
    specs: list[str] = []
    skipped_non_numeric = skipped_oversized = 0
    for row in rows:
        qa = row.get("qa") or {}
        answer = _as_number(qa.get("exe_ans"))
        if answer is None:
            skipped_non_numeric += 1
            continue
        prompt = FINQA_PROMPT.format(
            pre_text=" ".join(row.get("pre_text") or []),
            table=_linearize_table(row.get("table") or []),
            post_text=" ".join(row.get("post_text") or []),
            question=str(qa.get("question") or "").strip(),
        )
        if len(prompt) > MAX_PROMPT_CHARS:
            skipped_oversized += 1
            continue
        rid = hashlib.sha1(str(row["id"]).encode()).hexdigest()[:12]
        specs.append(json.dumps(_spec(
            f"finqa__train__{rid}", "finqa",
            "https://github.com/czyssrs/FinQA", prompt, answer, "",
            f"finqa/{row.get('filename') or row['id']}",
        ), sort_keys=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "finqa_train_taskspecs.jsonl"
    path.write_text("".join(s + "\n" for s in specs))
    return {"source_rows": len(rows), "tasks": len(specs),
            "skipped_non_numeric": skipped_non_numeric,
            "skipped_oversized": skipped_oversized, "manifest": str(path)}


def export_tatqa(train_json: Path, out_dir: Path) -> dict[str, Any]:
    docs = json.loads(train_json.read_text())
    specs: list[str] = []
    skipped_non_numeric = skipped_oversized = 0
    total = 0
    for doc in docs:
        table = _linearize_table((doc.get("table") or {}).get("table") or [])
        paragraphs = "\n".join(
            str(p.get("text") or "").strip() for p in doc.get("paragraphs") or [])
        for q in doc.get("questions") or []:
            total += 1
            if q.get("answer_type") not in ("arithmetic", "count"):
                skipped_non_numeric += 1
                continue
            answer = _as_number(q.get("answer"))
            if answer is None:
                skipped_non_numeric += 1
                continue
            scale = str(q.get("scale") or "")
            prompt = TATQA_PROMPT.format(
                table=table, paragraphs=paragraphs,
                question=str(q.get("question") or "").strip(),
                scale_note=f", expressed in {scale}" if scale else "",
            )
            if len(prompt) > MAX_PROMPT_CHARS:
                skipped_oversized += 1
                continue
            specs.append(json.dumps(_spec(
                f"tatqa__train__{q['uid'][:12]}", "tatqa",
                "https://github.com/NExTplusplus/TAT-QA", prompt, answer, scale,
                f"tatqa/{(doc.get('table') or {}).get('uid') or q['uid']}",
            ), sort_keys=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tatqa_train_taskspecs.jsonl"
    path.write_text("".join(s + "\n" for s in specs))
    return {"source_rows": total, "tasks": len(specs),
            "skipped_non_numeric": skipped_non_numeric,
            "skipped_oversized": skipped_oversized, "manifest": str(path)}


def write_probe_sample(manifests: list[Path], out_dir: Path,
                       *, size: int = DEFAULT_PROBE_SAMPLE) -> dict[str, Any]:
    """Deterministic sha1-ordered sample across both sources combined."""
    rows: list[str] = []
    for manifest in manifests:
        rows.extend(l for l in manifest.read_text().splitlines() if l.strip())
    if not rows:
        raise ValueError("no finance tasks to sample")
    ordered = sorted(rows, key=lambda l: hashlib.sha1(
        json.loads(l)["task_id"].encode()).hexdigest())
    sample = ordered[:size]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PROBE_SAMPLE_NAME
    path.write_text("".join(l + "\n" for l in sample))
    from collections import Counter
    sources = Counter(json.loads(l)["metadata"]["subdomain"] for l in sample)
    return {"pool": len(rows), "probe_candidates": len(sample),
            "by_source": dict(sources), "path": str(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finqa-train", type=Path, required=True)
    parser.add_argument("--tatqa-train", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--probe-sample-size", type=int, default=DEFAULT_PROBE_SAMPLE)
    args = parser.parse_args(argv)
    finqa = export_finqa(args.finqa_train, args.out_dir)
    tatqa = export_tatqa(args.tatqa_train, args.out_dir)
    sample = write_probe_sample(
        [Path(finqa["manifest"]), Path(tatqa["manifest"])],
        args.out_dir, size=args.probe_sample_size)
    print(json.dumps({"finqa": finqa, "tatqa": tatqa, "probe_sample": sample}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
