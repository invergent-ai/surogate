"""Source adapter contract (ultra-data2 §5).

Every source — public benchmark, generated environment, or the existing router bank —
implements this one interface so the registry can ingest them uniformly.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..schemas import TaskSpec


class RawTaskRef(BaseModel):
    source_name: str
    source_version: str
    raw_id: str
    uri: str | None = None
    metadata: dict = Field(default_factory=dict)


class ValidationReport(BaseModel):
    ready: bool
    deterministic: bool = True
    leakage_found: bool = False
    setup_ok: bool = True
    base_fails: bool | None = None
    gold_passes: bool | None = None
    grader_repeats: int = 0
    notes: list[str] = Field(default_factory=list)


@runtime_checkable
class SourceAdapter(Protocol):
    source_name: str
    capability: str

    def discover(self) -> Iterable[RawTaskRef]:
        """Enumerate raw task references available from this source."""
        ...

    def materialize(self, ref: RawTaskRef) -> TaskSpec:
        """Turn one raw reference into a canonical TaskSpec."""
        ...

    def validate(self, task: TaskSpec) -> ValidationReport:
        """Run the per-source validation gates (ultra-data2 §10)."""
        ...

    def assign_split(self, task: TaskSpec) -> str:
        """Assign a split by family/group, never by random row."""
        ...
