"""HuggingFace-backed source adapters: shared loading + TaskSpec construction.

A concrete adapter sets a few class attributes and implements ``_row_to_spec(row, i)`` —
the pure, network-free, unit-testable mapping from a raw dataset row to a canonical
``TaskSpec``. ``materialize_all`` streams rows from HF and applies it; tests exercise
``_row_to_spec`` directly with fixture rows (no network).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..policy import allowed_splits
from ..schemas import (
    EnvironmentSpec,
    GraderSpec,
    HarnessName,
    RepoRef,
    SourceManifest,
    SourcePolicy,
    SourceRef,
    Split,
    SplitPolicy,
    SplittingSpec,
    TaskInput,
    TaskMetadata,
    TaskSpec,
)
from .base import RawTaskRef, ValidationReport

_POLICY_DEFAULT_SPLIT: dict[str, Split] = {
    "train_allowed": "grpo_train",
    "pool_only": "pool_discovery",
    "online_validation": "online_validation",
    "final_eval_only": "final_eval",
    "diagnostic_only": "diagnostic",
}


def default_split_for_policy(policy: SourcePolicy) -> Split:
    return _POLICY_DEFAULT_SPLIT[policy]


def hf_rows(
    dataset_id: str,
    split: str,
    name: str | None = None,
    streaming: bool = False,
    shuffle: bool = False,
    seed: int = 0,
    limit: int | None = None,
    buffer: int = 10000,
) -> Iterator[dict]:
    """Lazily yield rows from a HuggingFace dataset (streamed for the large train sources)."""
    from datasets import load_dataset

    ds = (
        load_dataset(dataset_id, name, split=split, streaming=streaming)
        if name
        else load_dataset(dataset_id, split=split, streaming=streaming)
    )
    if streaming and shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=buffer)
    n = 0
    for r in ds:
        if limit is not None and n >= limit:
            break
        yield r
        n += 1


def make_taskspec(
    *,
    task_id: str,
    capability: str,
    source_name: str,
    source_version: str,
    policy: SourcePolicy,
    harness: HarnessName,
    grader_type: str,
    group_id: str,
    expected_answer=None,
    grader_command: list[str] | None = None,
    prompt: str | None = None,
    system: str | None = None,
    messages: list[dict] | None = None,
    repo: RepoRef | None = None,
    context_documents: list | None = None,
    tools: list | None = None,
    split: Split | None = None,
    contamination_group: str | None = None,
    domain: str | None = None,
    subdomain: str | None = None,
    tags: list[str] | None = None,
    requires_tools: bool = False,
    requires_long_context: bool = False,
    estimated_worker_calls: int | None = None,
    url_or_ref: str | None = None,
    license: str | None = None,
) -> TaskSpec:
    """Construct a canonical TaskSpec, defaulting the split from the source policy."""
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt or ""})
    return TaskSpec(
        task_id=task_id,
        capability=capability,
        source=SourceRef(
            name=source_name,
            version=source_version,
            policy=policy,
            url_or_ref=url_or_ref,
            license=license,
        ),
        input=TaskInput(
            messages=messages,
            repo=repo,
            context_documents=context_documents or [],
            tools=tools or [],
        ),
        environment=EnvironmentSpec(harness=harness),
        grader=GraderSpec(type=grader_type, expected_answer=expected_answer, command=grader_command),
        splitting=SplittingSpec(
            group_id=group_id,
            split=split or default_split_for_policy(policy),
            contamination_group=contamination_group,
        ),
        metadata=TaskMetadata(
            domain=domain,
            subdomain=subdomain,
            tags=tags or [],
            requires_tools=requires_tools,
            requires_long_context=requires_long_context,
            estimated_worker_calls=estimated_worker_calls,
        ),
    )


class HFTaskAdapter:
    """Base for HuggingFace-backed sources. Subclasses set the class attrs below and
    implement ``_row_to_spec``."""

    source_name: str = ""
    capability: str = ""
    dataset_id: str = ""
    hf_split: str = "train"
    hf_name: str | None = None
    streaming: bool = False
    policy: SourcePolicy = "train_allowed"
    harness: HarnessName = "direct_qa"
    version: str = "v1"
    license: str | None = "see-source"

    def _row_to_spec(self, row: dict, i: int) -> TaskSpec | None:
        raise NotImplementedError

    def materialize_all(
        self, limit: int | None = None, shuffle: bool = False, seed: int = 0
    ) -> Iterator[TaskSpec]:
        rows = hf_rows(
            self.dataset_id, self.hf_split, self.hf_name, self.streaming, shuffle, seed, limit
        )
        for i, row in enumerate(rows):
            spec = self._row_to_spec(row, i)
            if spec is not None:
                yield spec

    def discover(self) -> Iterable[RawTaskRef]:
        for i, _row in enumerate(
            hf_rows(self.dataset_id, self.hf_split, self.hf_name, self.streaming)
        ):
            yield RawTaskRef(source_name=self.source_name, source_version=self.version, raw_id=str(i))

    def materialize(self, ref: RawTaskRef) -> TaskSpec:
        raise NotImplementedError("HF adapters emit via materialize_all(); per-id fetch unsupported")

    def manifest(self) -> SourceManifest:
        return SourceManifest(
            source_name=self.source_name,
            source_type="public_benchmark",
            version=self.version,
            license=self.license,
            allowed_uses=allowed_splits(self.policy),
            forbidden_uses=[] if self.policy == "train_allowed" else ["grpo_train"],
            split_policy=SplitPolicy(type="source_family"),
        )

    def validate(self, task: TaskSpec) -> ValidationReport:
        ok = bool(task.input.messages) and (
            task.grader.expected_answer is not None or task.grader.command is not None
        )
        return ValidationReport(ready=ok, setup_ok=ok)

    def assign_split(self, task: TaskSpec) -> str:
        return task.splitting.split
