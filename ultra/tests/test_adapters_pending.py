"""Emit-only adapter families: valid TaskSpecs that target not-yet-built harnesses."""

from ultra.registry import TaskRegistry
from ultra.sources import SOURCE_ADAPTERS
from ultra.sources.direct import NuminaMathAdapter
from ultra.sources.longcontext import LongContextDocPackAdapter
from ultra.sources.repo import SWEsmithAdapter
from ultra.sources.roleprobe import RoleProbeAdapter
from ultra.sources.tool import TauBenchAdapter


def test_swesmith_emits_repo_task():
    a = SWEsmithAdapter()
    spec = a._row_to_spec(
        {"instance_id": "django__django-1", "repo": "django/django", "base_commit": "abc123", "problem_statement": "Fix the bug"},
        0,
    )
    assert spec.capability == "agentic_coding"
    assert spec.environment.harness == "opencode_repo"
    assert spec.grader.type == "hidden_tests"
    assert spec.input.repo.base_commit == "abc123"
    assert spec.input.repo.url == "https://github.com/django/django"
    assert spec.splitting.contamination_group == "django/django"  # repo-family split
    assert spec.source.policy == "train_allowed"
    assert spec.metadata.requires_tools is True


def test_tau_emits_tool_task():
    spec = TauBenchAdapter()._to_spec(
        {"domain": "retail", "instruction": "Cancel order 5", "tools": [{"name": "cancel"}], "id": "7"}, 0
    )
    assert spec.capability == "tool_dialogue"
    assert spec.environment.harness == "tool_dialog"
    assert spec.input.tools == [{"name": "cancel"}]
    assert spec.grader.type == "db_state"
    assert spec.metadata.requires_tools is True


def test_longcontext_emits_doc_task_and_skips_unsolvable():
    a = LongContextDocPackAdapter()
    spec = a._to_spec({"question": "Who?", "answer": "Alice", "documents": ["d1", "d2"], "corpus": "c", "id": "1"}, 0)
    assert spec.environment.harness == "long_context"
    assert len(spec.input.context_documents) == 2
    assert spec.metadata.requires_long_context is True
    assert spec.grader.expected_answer == "Alice"
    assert a._to_spec({"question": "Q", "answer": "A"}, 0) is None  # no documents


def test_roleprobe_reuses_base_grader_and_harness():
    base = NuminaMathAdapter()._row_to_spec({"problem": "2+2?", "answer": "4", "problem_is_valid": "Yes"}, 0)
    probe = RoleProbeAdapter().critic_probe(base, "the answer is 5", draft_correct=False)
    assert probe.capability == "role_probe"
    assert probe.environment.harness == base.environment.harness == "direct_qa"
    assert probe.grader.type == base.grader.type == "math_equal"
    assert probe.grader.expected_answer == "4"
    assert probe.splitting.split == "diagnostic"
    assert probe.source.policy == "diagnostic_only"


def test_emit_only_specs_ingest_into_registry():
    a = SWEsmithAdapter()
    spec = a._row_to_spec(
        {"instance_id": "x__y-1", "repo": "x/y", "base_commit": "c", "problem_statement": "p"}, 0
    )
    r = TaskRegistry()
    r.register_manifest(a.manifest())
    r.add(spec)
    assert len(r) == 1


def test_full_adapter_registry_builds():
    assert len(SOURCE_ADAPTERS) >= 27
    for _name, cls in SOURCE_ADAPTERS.items():
        adapter = cls()
        assert adapter.manifest().source_name
