"""Base for custom / generated sources that convert injected raw records into TaskSpecs.

Unlike the HuggingFace adapters, these sources have no single public dataset — GitHub
issues, custom terminal tasks, tau-style dialogues, generated long-context packs and
simulator scenarios are produced upstream and handed to the adapter as raw records.

These adapters are MATERIALIZE-COMPLETE but EXECUTION-PENDING: they emit valid TaskSpecs,
but the harnesses they target (opencode_repo, terminal_sandbox, tool_dialog, long_context,
sequential_sim, research_loop) are not built yet, so their tasks cannot be run or graded
until those land.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..policy import allowed_splits
from ..schemas import HarnessName, SourceManifest, SourcePolicy, SplitPolicy, TaskSpec
from .base import RawTaskRef, ValidationReport


class RawRecordAdapter:
    source_name: str = ""
    capability: str = ""
    policy: SourcePolicy = "train_allowed"
    harness: HarnessName = "direct_qa"
    version: str = "v1"
    source_type: str = "generated"

    def __init__(self, records: list[dict] | None = None):
        self.records = list(records or [])

    def _to_spec(self, raw: dict, i: int) -> TaskSpec | None:
        raise NotImplementedError

    def materialize_all(self) -> Iterator[TaskSpec]:
        for i, raw in enumerate(self.records):
            spec = self._to_spec(raw, i)
            if spec is not None:
                yield spec

    def discover(self) -> Iterable[RawTaskRef]:
        for i in range(len(self.records)):
            yield RawTaskRef(source_name=self.source_name, source_version=self.version, raw_id=str(i))

    def materialize(self, ref: RawTaskRef) -> TaskSpec:
        i = int(ref.raw_id)
        spec = self._to_spec(self.records[i], i)
        if spec is None:
            raise ValueError(f"record {i} did not materialize")
        return spec

    def manifest(self) -> SourceManifest:
        return SourceManifest(
            source_name=self.source_name,
            source_type=self.source_type,
            version=self.version,
            allowed_uses=allowed_splits(self.policy),
            forbidden_uses=[] if self.policy == "train_allowed" else ["grpo_train"],
            split_policy=SplitPolicy(type="template"),
            known_issues=[f"execution-pending: {self.harness} harness not built yet"],
        )

    def validate(self, task: TaskSpec) -> ValidationReport:
        ready = bool(task.input.messages)
        return ValidationReport(ready=ready, setup_ok=ready)

    def assign_split(self, task: TaskSpec) -> str:
        return task.splitting.split
