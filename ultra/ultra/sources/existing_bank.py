"""ExistingBankAdapter — the router's curated single-step bank as Ultra TaskSpecs.

The bank (``director/manifests/fugu_clean_v1/manifest.jsonl``, 2,650 rows) already
carries prompt, solution, grader, split, domain/subdomain and a system prompt, with
eval families excluded upstream by a normalized prompt-hash denylist. Each row becomes
a ``direct_qa`` (or ``code_exec``) TaskSpec the Conductor can route.

The precomputed per-worker ``rewards``/``winners`` are router-era (a different,
6-open-weight pool) and are deliberately NOT carried into the TaskSpec — they are not
Ultra targets. Only the task itself (prompt + solution + grader) is reused.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..policy import SOURCE_POLICY
from ..schemas import (
    EnvironmentSpec,
    GraderSpec,
    SourceManifest,
    SourceRef,
    SplitPolicy,
    SplittingSpec,
    TaskInput,
    TaskMetadata,
    TaskSpec,
)
from .base import RawTaskRef, ValidationReport

# Bank lives in the sibling router project; data, not a code dependency.
_DEFAULT_BANK = (
    Path(__file__).resolve().parents[3] / "director" / "manifests" / "fugu_clean_v1" / "manifest.jsonl"
)

# Graders that require code execution → code_exec harness; everything else is direct QA.
_CODE_GRADERS = {"code_exec", "code_exec_stdio"}

# Bank domain → Ultra capability.
_CAPABILITY = {
    "math": "math",
    "code": "unit_code",
    "science": "science_knowledge",
    "general": "factual_qa",
}

# Bank split → Ultra split. "test" was the router's held-out internal validation.
_SPLIT_MAP = {"train": "grpo_train", "test": "online_validation"}


class ExistingBankAdapter:
    source_name = "existing_bank"
    capability = "direct_qa"  # family default; per-row capability is set from the domain

    def __init__(self, bank_path: str | Path = _DEFAULT_BANK, version: str = "fugu_clean_v1"):
        self.bank_path = Path(bank_path)
        self.version = version

    def manifest(self) -> SourceManifest:
        return SourceManifest(
            source_name=self.source_name,
            source_type="router_bank",
            version=self.version,
            license="mixed-public",
            allowed_uses=[
                "pool_discovery",
                "pool_validation",
                "grpo_train",
                "online_validation",
                "diagnostic",
            ],
            forbidden_uses=["final_eval_claim"],
            split_policy=SplitPolicy(
                type="source_family",
                notes="router-curated bank; eval families excluded upstream by prompt-hash denylist",
            ),
            known_issues=[
                "per-worker rewards are router-era (6 open-weight pool) — diagnostic only, not Ultra targets"
            ],
        )

    def _rows(self) -> Iterator[dict]:
        with self.bank_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def discover(self) -> Iterable[RawTaskRef]:
        for row in self._rows():
            yield RawTaskRef(
                source_name=self.source_name,
                source_version=self.version,
                raw_id=str(row["task_id"]),
            )

    def materialize_all(
        self,
        split: str | None = None,
        harness: str | None = None,
        verdict: str | None = None,
        limit: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[TaskSpec]:
        """Emit TaskSpecs, optionally filtered. ``verdict`` (e.g. "discriminative") filters on
        the bank's router-era difficulty label — a usable medium-difficulty proxy for step-zero."""
        rows = list(self._rows())
        if shuffle:
            import random

            random.Random(seed).shuffle(rows)
        n = 0
        for row in rows:
            if verdict is not None and row.get("verdict") != verdict:
                continue
            spec = self._to_spec(row)
            if split is not None and spec.splitting.split != split:
                continue
            if harness is not None and spec.environment.harness != harness:
                continue
            yield spec
            n += 1
            if limit is not None and n >= limit:
                break

    def materialize(self, ref: RawTaskRef) -> TaskSpec:
        for row in self._rows():
            if str(row["task_id"]) == ref.raw_id:
                return self._to_spec(row)
        raise KeyError(f"task {ref.raw_id!r} not in bank")

    def _to_spec(self, row: dict) -> TaskSpec:
        grader_name = row["grader"]
        harness = "code_exec" if grader_name in _CODE_GRADERS else "direct_qa"
        domain = row.get("domain")
        messages: list[dict] = []
        if row.get("system"):
            messages.append({"role": "system", "content": row["system"]})
        messages.append({"role": "user", "content": row["prompt"]})
        source = row.get("source") or domain or "bank"
        return TaskSpec(
            task_id=f"existing_bank__{row['task_id']}",
            capability=_CAPABILITY.get(domain, "factual_qa"),
            source=SourceRef(
                name=self.source_name,
                version=self.version,
                policy=SOURCE_POLICY["existing_bank"],
                url_or_ref=str(self.bank_path),
            ),
            input=TaskInput(messages=messages),
            environment=EnvironmentSpec(harness=harness),
            grader=GraderSpec(type=grader_name, expected_answer=row["solution"]),
            splitting=SplittingSpec(
                group_id=str(source),
                split=_SPLIT_MAP.get(row.get("split", "train"), "grpo_train"),
                contamination_group=f"{source}::{row['task_id']}",
            ),
            metadata=TaskMetadata(
                domain=domain,
                subdomain=row.get("subdomain"),
                tags=["existing_bank"],
            ),
        )

    def validate(self, task: TaskSpec) -> ValidationReport:
        # Bank tasks were validated upstream (probe → curate → denylist); shape-check only.
        ok = bool(task.input.messages) and task.grader.expected_answer is not None
        return ValidationReport(ready=ok, deterministic=True, leakage_found=False, setup_ok=ok)

    def assign_split(self, task: TaskSpec) -> str:
        return task.splitting.split
