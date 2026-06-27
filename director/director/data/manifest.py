"""Probe records and curated dataset manifests.

The curation pipeline writes two artifacts under a manifest directory:

  probe.jsonl     one record per probed candidate (reward vector + verdict).
                  Appended incrementally so a large probe is fully resumable.
  manifest.jsonl  the kept + balanced + split items (the training set).
  meta.json       worker pool ordering, sources, seeds, version.

Both record types serialize to JSON and rebuild into ``Task`` objects for labeling.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..shared.tasks import Task

SCHEMA_VERSION = 1

Verdict = Literal["discriminative", "saturated", "dead"]


@dataclass
class ProbeRecord:
    task_id: str
    domain: str
    source: str
    prompt: str
    solution: Any
    grader: str
    system: str | None
    rewards: list[float]  # per worker, ordered by worker_ids
    winners: list[str]  # worker_ids achieving the max reward (when > 0)
    verdict: Verdict
    subdomain: str | None = None
    dataset: str | None = None  # underlying HuggingFace dataset id (provenance)

    def to_task(self) -> Task:
        return Task(
            task_id=self.task_id,
            prompt=self.prompt,
            solution=self.solution,
            grader=self.grader,
            system=self.system,
            metadata={
                "domain": self.domain,
                "source": self.source,
                "subdomain": self.subdomain,
                "dataset": self.dataset,
            },
        )


@dataclass
class CuratedItem(ProbeRecord):
    split: str = "train"


@dataclass
class ManifestMeta:
    worker_ids: list[str]
    sources: list[str]
    seed: int
    schema_version: int = SCHEMA_VERSION
    note: str = ""
    extra: dict = field(default_factory=dict)


def classify(rewards: list[float], worker_ids: list[str], threshold: float = 1.0) -> tuple[Verdict, list[str]]:
    """Verdict + winners from a probe reward vector.

    discriminative => at least one worker correct AND at least one wrong (the signal).
    saturated => everyone correct. dead => no one correct.
    """
    correct = [w for w, r in zip(worker_ids, rewards) if r >= threshold]
    if not correct:
        return "dead", []
    if len(correct) == len(worker_ids):
        return "saturated", correct
    return "discriminative", correct


# ---------------------------------------------------------------------------
# JSONL IO
# ---------------------------------------------------------------------------


def _append_jsonl(path: str, rec) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def _read_jsonl(path: str, cls) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(cls(**json.loads(line)))
    return out


def probe_path(manifest_dir: str) -> str:
    return os.path.join(manifest_dir, "probe.jsonl")


def manifest_path(manifest_dir: str) -> str:
    return os.path.join(manifest_dir, "manifest.jsonl")


def meta_path(manifest_dir: str) -> str:
    return os.path.join(manifest_dir, "meta.json")


def append_probe(manifest_dir: str, rec: ProbeRecord) -> None:
    _append_jsonl(probe_path(manifest_dir), rec)


def read_probes(manifest_dir: str) -> list[ProbeRecord]:
    return _read_jsonl(probe_path(manifest_dir), ProbeRecord)


def probed_ids(manifest_dir: str) -> set[str]:
    """Task ids already probed (for resume)."""
    return {p.task_id for p in read_probes(manifest_dir)}


def write_manifest(manifest_dir: str, items: list[CuratedItem]) -> None:
    path = manifest_path(manifest_dir)
    if os.path.exists(path):
        os.remove(path)
    for it in items:
        _append_jsonl(path, it)


def read_manifest(manifest_dir: str) -> list[CuratedItem]:
    return _read_jsonl(manifest_path(manifest_dir), CuratedItem)


def write_meta(manifest_dir: str, meta: ManifestMeta) -> None:
    os.makedirs(manifest_dir, exist_ok=True)
    with open(meta_path(manifest_dir), "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)


def read_meta(manifest_dir: str) -> ManifestMeta:
    with open(meta_path(manifest_dir), encoding="utf-8") as f:
        return ManifestMeta(**json.load(f))


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def probe_stats(probes: list[ProbeRecord]) -> str:
    by_verdict = Counter(p.verdict for p in probes)
    by_domain = Counter(p.domain for p in probes)
    disc = [p for p in probes if p.verdict == "discriminative"]
    sole = Counter(p.winners[0] for p in disc if len(p.winners) == 1)
    lines = [
        f"probed={len(probes)}  " + "  ".join(f"{k}={v}" for k, v in by_verdict.items()),
        "by domain: " + "  ".join(f"{k}={v}" for k, v in sorted(by_domain.items())),
        f"discriminative={len(disc)} (keep-rate {len(disc)/max(len(probes),1):.1%})",
        "sole-winner counts: " + ("  ".join(f"{k}={v}" for k, v in sole.most_common()) or "(none)"),
    ]
    return "\n".join(lines)


def manifest_stats(items: list[CuratedItem]) -> str:
    by_split = Counter(it.split for it in items)
    by_domain = Counter(it.domain for it in items)
    sole = Counter(it.winners[0] for it in items if len(it.winners) == 1)
    lines = [
        f"curated={len(items)}  " + "  ".join(f"{k}={v}" for k, v in by_split.items()),
        "by domain: " + "  ".join(f"{k}={v}" for k, v in sorted(by_domain.items())),
        "sole-winner balance: " + ("  ".join(f"{k}={v}" for k, v in sole.most_common()) or "(none)"),
    ]
    return "\n".join(lines)
