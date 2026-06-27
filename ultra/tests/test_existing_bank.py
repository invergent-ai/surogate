"""ExistingBankAdapter against the real router bank (skips if the bank isn't present)."""

import pytest

from ultra.registry import TaskRegistry
from ultra.sources.existing_bank import ExistingBankAdapter


def _adapter() -> ExistingBankAdapter:
    a = ExistingBankAdapter()
    if not a.bank_path.exists():
        pytest.skip(f"router bank not found at {a.bank_path}")
    return a


def test_bank_materializes_valid_taskspecs():
    a = _adapter()
    specs = []
    for i, spec in enumerate(a.materialize_all()):
        specs.append(spec)
        if i >= 99:
            break
    assert specs, "bank produced no tasks"

    s = specs[0]
    assert s.task_id.startswith("existing_bank__")
    assert s.grader.expected_answer is not None
    assert s.environment.harness in ("direct_qa", "code_exec")
    assert s.capability in ("math", "unit_code", "science_knowledge", "factual_qa")
    assert s.input.messages[-1]["role"] == "user"

    rep = a.validate(s)
    assert rep.ready is True


def test_bank_tasks_ingest_into_registry():
    a = _adapter()
    r = TaskRegistry()
    r.register_manifest(a.manifest())
    added = 0
    for i, spec in enumerate(a.materialize_all()):
        if i >= 100:
            break
        try:
            r.add(spec)
            added += 1
        except Exception:
            pass  # dedup/policy collisions are acceptable; we only assert ingestion works
    assert added > 0
    assert len(r) == added
