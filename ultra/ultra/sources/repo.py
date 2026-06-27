"""Repository / terminal source adapters (EXECUTION-PENDING: opencode_repo + terminal_sandbox).

These emit the task shell — repo reference, problem statement, and a ``hidden_tests``
grader. The hidden test artifacts (test_patch / FAIL_TO_PASS) deliberately stay OUT of the
TaskSpec (they must never reach the agent workspace); they are re-joined from the source
dataset when the opencode_repo harness + container validation pipeline are built.
"""

from __future__ import annotations

from ..policy import SOURCE_POLICY
from ..schemas import RepoRef, TaskSpec
from .hf import HFTaskAdapter, make_taskspec
from .raw import RawRecordAdapter


class SWEsmithAdapter(HFTaskAdapter):
    """SWE-smith — generated repository-repair tasks."""

    source_name = "swe_smith"
    capability = "agentic_coding"
    dataset_id = "SWE-bench/SWE-smith"
    hf_split = "train"
    policy = SOURCE_POLICY["swe_smith"]
    harness = "opencode_repo"

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        repo = r.get("repo")
        commit = r.get("base_commit")
        if not repo or not commit:
            return None
        return make_taskspec(
            task_id=f"{self.source_name}__{r['instance_id']}",
            capability="agentic_coding",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="opencode_repo",
            grader_type="hidden_tests",
            grader_command=["python", "/grader/grade.py"],
            messages=[{"role": "user", "content": r.get("problem_statement", "")}],
            repo=RepoRef(url=f"https://github.com/{repo}", base_commit=commit),
            group_id=repo,
            contamination_group=repo,  # repo-family split (ultra-data2 §9)
            domain="software_engineering",
            requires_tools=True,
            estimated_worker_calls=3,
            tags=["repo", "tests"],
            url_or_ref=self.dataset_id,
        )


class SWEbenchAdapter(SWEsmithAdapter):
    """SWE-bench Verified — held out for final evaluation."""

    source_name = "swe_bench_verified"
    dataset_id = "princeton-nlp/SWE-bench_Verified"
    hf_split = "test"
    policy = SOURCE_POLICY["swe_bench_verified"]


class GitHubIssueAdapter(RawRecordAdapter):
    """Custom GitHub issue/PR repair tasks (records mined upstream)."""

    source_name = "github_issue"
    capability = "agentic_coding"
    policy = SOURCE_POLICY["github_issue"]
    harness = "opencode_repo"
    source_type = "mined"

    def _to_spec(self, raw: dict, i: int) -> TaskSpec | None:
        repo = raw.get("repo")
        commit = raw.get("base_commit")
        if not repo or not commit:
            return None
        return make_taskspec(
            task_id=f"github_issue__{raw.get('id', i)}",
            capability="agentic_coding",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="opencode_repo",
            grader_type="hidden_tests",
            grader_command=["python", "/grader/grade.py"],
            messages=[{"role": "user", "content": raw.get("problem_statement", "")}],
            repo=RepoRef(url=raw.get("url", f"https://github.com/{repo}"), base_commit=commit),
            group_id=repo,
            contamination_group=repo,
            domain="software_engineering",
            requires_tools=True,
            estimated_worker_calls=3,
            tags=["repo", "mined"],
        )


class TerminalBenchAdapter(RawRecordAdapter):
    """Custom Terminal-Bench-style Docker tasks (native tests)."""

    source_name = "terminal_custom"
    capability = "terminal"
    policy = SOURCE_POLICY["terminal_custom"]
    harness = "terminal_sandbox"

    def _to_spec(self, raw: dict, i: int) -> TaskSpec | None:
        if not raw.get("instruction"):
            return None
        return make_taskspec(
            task_id=f"terminal_custom__{raw.get('id', i)}",
            capability="terminal",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="terminal_sandbox",
            grader_type="hidden_tests",
            grader_command=raw.get("test_command", ["bash", "/tests/run.sh"]),
            messages=[{"role": "user", "content": raw["instruction"]}],
            group_id=raw.get("group", "terminal"),
            contamination_group=raw.get("group"),
            domain="terminal",
            requires_tools=True,
            estimated_worker_calls=4,
            tags=["terminal", "docker"],
        )
