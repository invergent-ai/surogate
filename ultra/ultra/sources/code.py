"""Code source adapters: unit-level / competitive / scientific coding.

HumanEval, MBPP, CodeContests, TACO are ported from the router's verified loaders.
LiveCodeBench, BigCodeBench, SciCode are NEW (the router had no loader) and their row
mappings are marked SCHEMA UNVERIFIED — confirm field names against the live dataset
before materialize_all. All route to the ``code_exec`` harness.
"""

from __future__ import annotations

import json

from ..policy import SOURCE_POLICY
from ..schemas import TaskSpec
from .hf import HFTaskAdapter, make_taskspec

_CODE_SYS = "Complete the function in Python. Return only the full function in a code block."
_STDIO_SYS = (
    "Write a complete Python program that reads from stdin and writes to stdout. "
    "Return only the program in a code block."
)


class HumanEvalAdapter(HFTaskAdapter):
    """HumanEval — function-completion benchmark, held out for final evaluation."""

    source_name = "humaneval"
    capability = "unit_code"
    dataset_id = "openai/openai_humaneval"
    hf_split = "test"
    policy = SOURCE_POLICY["humaneval"]

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        return make_taskspec(
            task_id=f"humaneval__{r['task_id'].replace('/', '_')}",
            capability="unit_code",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="code_exec",
            grader_type="code_exec",
            expected_answer={"test": r["test"], "entry_point": r["entry_point"]},
            prompt=r["prompt"],
            system=_CODE_SYS,
            group_id="humaneval",
            domain="code",
            tags=["code", "eval"],
            url_or_ref=self.dataset_id,
        )


class MBPPAdapter(HFTaskAdapter):
    """MBPP — basic Python programming, held out for final evaluation."""

    source_name = "mbpp"
    capability = "unit_code"
    dataset_id = "google-research-datasets/mbpp"
    hf_split = "test"
    hf_name = "full"
    policy = SOURCE_POLICY["mbpp"]

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        tests = "\n".join([r.get("test_setup_code", ""), *r["test_list"]]).strip()
        prompt = f"{r['text']}\n\nYour function must pass these tests:\n" + "\n".join(r["test_list"])
        return make_taskspec(
            task_id=f"mbpp__{r['task_id']}",
            capability="unit_code",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="code_exec",
            grader_type="code_exec",
            expected_answer={"test": tests, "entry_point": ""},
            prompt=prompt,
            system=_CODE_SYS,
            group_id="mbpp",
            domain="code",
            tags=["code", "eval"],
            url_or_ref=self.dataset_id,
        )


class CodeContestsAdapter(HFTaskAdapter):
    """CodeContests (stdin->stdout), banded to the rated mid-difficulty range."""

    source_name = "code_contests"
    capability = "unit_code"
    dataset_id = "deepmind/code_contests"
    hf_split = "train"
    streaming = True
    policy = SOURCE_POLICY["code_contests"]

    def __init__(self, min_difficulty: int = 6, max_difficulty: int = 10, max_tests: int = 8):
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.max_tests = max_tests

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        d = r.get("difficulty") or 0
        if d < self.min_difficulty or d > self.max_difficulty:
            return None
        pt = r["public_tests"]
        tests = [
            {"input": inp, "output": out}
            for inp, out in zip(pt["input"][: self.max_tests], pt["output"][: self.max_tests])
        ]
        if not tests:
            return None
        return make_taskspec(
            task_id=f"code_contests__{r['name'][:40]}",
            capability="unit_code",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="code_exec",
            grader_type="code_exec_stdio",
            expected_answer={"tests": tests, "timeout": 10},
            prompt=r["description"],
            system=_STDIO_SYS,
            group_id="code_contests",
            domain="code",
            subdomain=str(r.get("difficulty")),
            tags=["code", "competitive"],
            url_or_ref=self.dataset_id,
        )


class TACOAdapter(HFTaskAdapter):
    """TACO-verified competitive programming (stdin->stdout), mid-difficulty band."""

    source_name = "taco"
    capability = "unit_code"
    dataset_id = "likaixin/TACO-verified"
    hf_split = "train"
    streaming = True
    policy = SOURCE_POLICY["taco"]

    def __init__(self, difficulties=("EASY", "MEDIUM", "MEDIUM_HARD"), max_tests: int = 8):
        self.difficulties = set(difficulties)
        self.max_tests = max_tests

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        if r.get("difficulty") not in self.difficulties:
            return None
        try:
            io = r["input_output"]
            io = json.loads(io) if isinstance(io, str) else io
        except (json.JSONDecodeError, TypeError):
            return None
        if not io or "fn_name" in io or not io.get("inputs"):
            return None
        tests = [
            {"input": str(inp), "output": str(out)}
            for inp, out in zip(io["inputs"][: self.max_tests], io["outputs"][: self.max_tests])
        ]
        if not tests:
            return None
        return make_taskspec(
            task_id=f"taco__{r['id']}",
            capability="unit_code",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="code_exec",
            grader_type="code_exec_stdio",
            expected_answer={"tests": tests, "timeout": 10},
            prompt=r["question"],
            system=_STDIO_SYS,
            group_id="taco",
            domain="code",
            subdomain=r.get("difficulty"),
            tags=["code", "competitive"],
            url_or_ref=self.dataset_id,
        )


class LiveCodeBenchAdapter(HFTaskAdapter):
    """LiveCodeBench (older contest window) — broad code abilities, time-split trainable.

    SCHEMA UNVERIFIED: confirm field names against the live dataset before
    materialize_all. Assumes ``question_content`` + a JSON ``public_test_cases`` list of
    ``{input, output}``.
    """

    source_name = "livecodebench_old"
    capability = "unit_code"
    dataset_id = "livecodebench/code_generation_lite"
    hf_split = "test"
    policy = SOURCE_POLICY["livecodebench_old"]

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        raw = r.get("public_test_cases")
        try:
            cases = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except (json.JSONDecodeError, TypeError):
            return None
        tests = [{"input": c.get("input", ""), "output": c.get("output", "")} for c in cases]
        if not tests or not r.get("question_content"):
            return None
        return make_taskspec(
            task_id=f"livecodebench__{r.get('question_id', i)}",
            capability="unit_code",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="code_exec",
            grader_type="code_exec_stdio",
            expected_answer={"tests": tests, "timeout": 10},
            prompt=r["question_content"],
            system=_STDIO_SYS,
            group_id="livecodebench",
            domain="code",
            tags=["code", "schema-unverified"],
            url_or_ref=self.dataset_id,
        )


class LiveCodeBenchProAdapter(LiveCodeBenchAdapter):
    """LiveCodeBench Pro (latest window) — held out for final evaluation. SCHEMA UNVERIFIED."""

    source_name = "livecodebench_latest"
    policy = SOURCE_POLICY["livecodebench_latest"]


class BigCodeBenchAdapter(HFTaskAdapter):
    """BigCodeBench — practical function-level code with diverse library calls.

    SCHEMA UNVERIFIED: assumes ``complete_prompt`` + ``test`` (unittest source) +
    ``entry_point``. Confirm against the live dataset before materialize_all.
    """

    source_name = "bigcodebench"
    capability = "unit_code"
    dataset_id = "bigcode/bigcodebench"
    hf_split = "v0.1.0_hf"
    policy = SOURCE_POLICY["bigcodebench"]

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        prompt = r.get("complete_prompt") or r.get("instruct_prompt")
        if not prompt or not r.get("test"):
            return None
        return make_taskspec(
            task_id=f"bigcodebench__{r.get('task_id', i)}",
            capability="unit_code",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="code_exec",
            grader_type="code_exec",
            expected_answer={"test": r["test"], "entry_point": r.get("entry_point", "")},
            prompt=prompt,
            system=_CODE_SYS,
            group_id="bigcodebench",
            domain="code",
            tags=["code", "schema-unverified"],
            url_or_ref=self.dataset_id,
        )


class SciCodeAdapter(HFTaskAdapter):
    """SciCode — realistic scientific research coding.

    SCHEMA UNVERIFIED: SciCode problems are multi-subproblem; this emits the full
    problem as one code task. Confirm field names + grading granularity before use.
    """

    source_name = "scicode_dev"
    capability = "scientific_code"
    dataset_id = "SciCode1/SciCode"
    hf_split = "validation"
    policy = SOURCE_POLICY["scicode_dev"]

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        prompt = r.get("problem_description_main") or r.get("problem")
        test = r.get("test") or r.get("general_tests")
        if not prompt or not test:
            return None
        return make_taskspec(
            task_id=f"scicode__{r.get('problem_id', i)}",
            capability="scientific_code",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="code_exec",
            grader_type="code_exec",
            expected_answer={"test": test, "entry_point": ""},
            prompt=str(prompt),
            system=_CODE_SYS,
            group_id="scicode",
            domain="science",
            tags=["code", "science", "schema-unverified"],
            url_or_ref=self.dataset_id,
        )
