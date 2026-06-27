import pytest

from ultra.registry import RegistryError, TaskRegistry
from ultra.schemas import (
    EnvironmentSpec,
    GraderSpec,
    SourceManifest,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskSpec,
)


def _task(tid="t1", policy="train_allowed", split="grpo_train", prompt="p"):
    return TaskSpec(
        task_id=tid,
        capability="math",
        source=SourceRef(name="s", version="v", policy=policy),
        input=TaskInput(messages=[{"role": "user", "content": prompt}]),
        environment=EnvironmentSpec(harness="direct_qa"),
        grader=GraderSpec(type="math_equal", expected_answer="x"),
        splitting=SplittingSpec(group_id="g", split=split),
    )


def _registry_with_manifest() -> TaskRegistry:
    r = TaskRegistry()
    r.register_manifest(SourceManifest(source_name="s", source_type="t", version="v"))
    return r


def test_rejects_missing_manifest():
    r = TaskRegistry()
    with pytest.raises(RegistryError, match="no SourceManifest"):
        r.add(_task())


def test_rejects_policy_split_violation():
    r = _registry_with_manifest()
    with pytest.raises(RegistryError, match="forbids split"):
        r.add(_task(policy="final_eval_only", split="grpo_train"))


def test_rejects_dedup_collision():
    r = _registry_with_manifest()
    r.add(_task(tid="a", prompt="identical prompt"))
    with pytest.raises(RegistryError, match="duplicates"):
        r.add(_task(tid="b", prompt="identical prompt"))


def test_accepts_valid_and_indexes_split():
    r = _registry_with_manifest()
    r.add(_task(tid="a", prompt="x"))
    r.add(_task(tid="b", prompt="y"))
    assert len(r) == 2
    assert len(r.by_split("grpo_train")) == 2
    assert r["a"].task_id == "a"
