"""Adapter row->TaskSpec conversion (offline, fixture rows — no HuggingFace network)."""

import pytest

from ultra.registry import TaskRegistry
from ultra.sources import SOURCE_ADAPTERS
from ultra.sources.code import CodeContestsAdapter, HumanEvalAdapter
from ultra.sources.direct import MMLUProAdapter, NuminaMathAdapter


def test_numina_math_row():
    a = NuminaMathAdapter()
    spec = a._row_to_spec(
        {"problem": "What is 2+2?", "answer": "4", "problem_is_valid": "Yes"}, 0
    )
    assert spec.capability == "math"
    assert spec.environment.harness == "direct_qa"
    assert spec.grader.type == "math_equal"
    assert spec.grader.expected_answer == "4"
    assert spec.source.policy == "train_allowed"
    assert spec.splitting.split == "grpo_train"
    assert spec.task_id.startswith("numina_math__")


def test_numina_skips_proof_rows():
    a = NuminaMathAdapter()
    assert a._row_to_spec({"problem": "Prove X.", "answer": "proof"}, 0) is None
    assert a._row_to_spec({"problem": "Q", "answer": ""}, 0) is None


def test_mmlu_pro_domain_routing():
    a = MMLUProAdapter()
    hist = a._row_to_spec(
        {"question_id": "q1", "question": "Q?", "options": ["a", "b", "c", "d"], "answer": "B", "category": "history"},
        0,
    )
    assert hist.capability == "factual_qa"
    assert hist.metadata.domain == "general"
    assert hist.grader.type == "mc_letter"
    assert hist.grader.expected_answer == "B"

    phys = a._row_to_spec(
        {"question_id": "q2", "question": "Q?", "options": ["a", "b"], "answer": "A", "category": "physics"},
        1,
    )
    assert phys.capability == "science_knowledge"
    assert phys.metadata.domain == "science"


def test_code_contests_band_and_shape():
    a = CodeContestsAdapter()
    in_band = a._row_to_spec(
        {"name": "p1", "description": "Read n, print 2n", "public_tests": {"input": ["2\n"], "output": ["4\n"]}, "difficulty": 7},
        0,
    )
    assert in_band.environment.harness == "code_exec"
    assert in_band.grader.type == "code_exec_stdio"
    assert in_band.grader.expected_answer["tests"][0] == {"input": "2\n", "output": "4\n"}

    out_of_band = a._row_to_spec(
        {"name": "p2", "description": "easy", "public_tests": {"input": ["1\n"], "output": ["1\n"]}, "difficulty": 2},
        1,
    )
    assert out_of_band is None


def test_humaneval_is_eval_only():
    a = HumanEvalAdapter()
    spec = a._row_to_spec(
        {"task_id": "HumanEval/0", "prompt": "def f():\n", "test": "assert f()==1", "entry_point": "f"}, 0
    )
    assert spec.source.policy == "final_eval_only"
    assert spec.splitting.split == "final_eval"
    assert spec.grader.type == "code_exec"


def test_registry_lists_all_adapters_and_they_build():
    assert "numina_math" in SOURCE_ADAPTERS
    assert len(SOURCE_ADAPTERS) >= 16
    for name, cls in SOURCE_ADAPTERS.items():
        adapter = cls()
        manifest = adapter.manifest()
        assert manifest.source_name


def test_adapter_specs_ingest_into_registry():
    a = NuminaMathAdapter()
    spec = a._row_to_spec({"problem": "2+2?", "answer": "4", "problem_is_valid": "Yes"}, 0)
    r = TaskRegistry()
    r.register_manifest(a.manifest())
    r.add(spec)
    assert len(r) == 1
    assert r.by_split("grpo_train")[0].grader.expected_answer == "4"


def test_harness_registry_has_both_single_call_harnesses():
    import ultra.harness  # noqa: F401  (import populates the registry)
    from ultra.harness import HARNESS_REGISTRY

    assert {"direct_qa", "code_exec"} <= set(HARNESS_REGISTRY)
    assert {"opencode", "claude_code", "codex"} <= set(HARNESS_REGISTRY)
    assert "opencode_repo" in HARNESS_REGISTRY
