"""Canonical Fugu-Ultra schemas (ultra-data2 §4, §11, §12; ultra-intro §1).

One ``TaskSpec`` for every source/harness, one ``RolloutRecord`` for every executed
workflow, one ``SourceManifest`` per imported source, and the ``Workflow`` the
Conductor emits. These are the contract the multi-source task factory and the GRPO
trainer agree on; a source adapter's only job is to emit a valid ``TaskSpec``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "2.0"

# Harness names — drive the harness router (ultra-data2 §6). Keep in sync with
# ultra.harness.base.HARNESS_REGISTRY.
HarnessName = Literal[
    "direct_qa",
    "code_exec",
    "opencode",
    "opencode_repo",
    "claude_code",
    "codex",
    "terminal_sandbox",
    "tool_dialog",
    "long_context",
    "vision_qa",
    "sequential_sim",
    "research_loop",
]

# Dataset splits (ultra-data2 §3 / §8).
Split = Literal[
    "pool_discovery",
    "pool_validation",
    "grpo_train",
    "online_validation",
    "final_eval",
    "diagnostic",
]

# Source-policy classes (ultra-data2 §3).
SourcePolicy = Literal[
    "train_allowed",
    "pool_only",
    "online_validation",
    "final_eval_only",
    "diagnostic_only",
]


# --------------------------------------------------------------------------- #
# TaskSpec v2
# --------------------------------------------------------------------------- #
class SourceRef(BaseModel):
    name: str
    version: str
    policy: SourcePolicy
    url_or_ref: str | None = None
    license: str | None = None
    source_commit: str | None = None


class RepoRef(BaseModel):
    url: str
    base_commit: str
    subdirectory: str | None = None


class TaskInput(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    assets: list[Any] = Field(default_factory=list)
    repo: RepoRef | None = None
    context_documents: list[Any] = Field(default_factory=list)
    tools: list[Any] = Field(default_factory=list)
    multimodal_assets: list[Any] = Field(default_factory=list)


class EnvironmentSpec(BaseModel):
    harness: HarnessName
    image: str | None = None
    cpu_limit: int | None = None
    memory_mb: int | None = None
    disk_mb: int | None = None
    network_policy: str = "model-relay-only"
    wall_time_seconds: int | None = None


class GraderSpec(BaseModel):
    # ``type`` is a grading-registry name (e.g. "math_equal") for direct graders, or
    # "hidden_tests" / "container_command" for executed graders.
    type: str
    command: list[str] | None = None
    expected_answer: Any | None = None  # gold solution for direct graders
    score_range: tuple[float, float] = (0.0, 1.0)
    success_threshold: float = 1.0
    deterministic: bool = True


class SplittingSpec(BaseModel):
    group_id: str
    split: Split
    contamination_group: str | None = None


class TaskMetadata(BaseModel):
    domain: str | None = None
    subdomain: str | None = None
    difficulty_estimate: float | None = None
    tags: list[str] = Field(default_factory=list)
    requires_vision: bool = False
    requires_tools: bool = False
    requires_long_context: bool = False
    estimated_worker_calls: int | None = None


class TaskSpec(BaseModel):
    schema_version: str = SCHEMA_VERSION
    task_id: str
    capability: str
    source: SourceRef
    input: TaskInput
    environment: EnvironmentSpec
    grader: GraderSpec
    splitting: SplittingSpec
    metadata: TaskMetadata = Field(default_factory=TaskMetadata)


# --------------------------------------------------------------------------- #
# Workflow (Conductor output; ultra-intro §1)
# --------------------------------------------------------------------------- #
class WorkflowStep(BaseModel):
    worker_id: int
    subtask: str
    access: list[int] = Field(default_factory=list)


class Workflow(BaseModel):
    steps: list[WorkflowStep]


# --------------------------------------------------------------------------- #
# SourceManifest (ultra-data2 §11)
# --------------------------------------------------------------------------- #
class SplitPolicy(BaseModel):
    type: str  # "temporal" | "repo_family" | "source_family" | ...
    train_before: str | None = None
    eval_after: str | None = None
    notes: str | None = None


class SourceManifest(BaseModel):
    source_name: str
    source_type: str
    version: str
    license: str | None = None
    allowed_uses: list[str] = Field(default_factory=list)
    forbidden_uses: list[str] = Field(default_factory=list)
    downloaded_at: str | None = None
    source_commit: str | None = None
    raw_artifact_hash: str | None = None
    adapter_version: str = "0.1.0"
    split_policy: SplitPolicy | None = None
    known_issues: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# RolloutRecord (ultra-data2 §12)
# --------------------------------------------------------------------------- #
class ConductorRecord(BaseModel):
    checkpoint: str | None = None
    raw_output: str | None = None
    workflow_parse_valid: bool = True
    old_logprobs_ref: str | None = None


class ExecStep(BaseModel):
    worker_id: int
    harness: HarnessName
    session_ref: str | None = None
    patch_ref: str | None = None
    messages_ref: str | None = None
    tool_events_ref: str | None = None
    text: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    termination: str = "completed"


class Execution(BaseModel):
    steps: list[ExecStep] = Field(default_factory=list)


class Grade(BaseModel):
    score: float
    success: bool
    grader_ref: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RolloutRecord(BaseModel):
    rollout_id: str
    task_id: str
    source_name: str
    capability: str
    harness: HarnessName
    conductor: ConductorRecord = Field(default_factory=ConductorRecord)
    workflow: Workflow
    execution: Execution = Field(default_factory=Execution)
    grade: Grade | None = None
    # Faithful reward (ultra-intro §6): 0 malformed workflow · 0.5 valid+incorrect · 1.0 valid+correct.
    reward: float | None = None
    valid_for_training: bool = True
    failure_class: str | None = None


# --------------------------------------------------------------------------- #
# AgentTrace (end-to-end coding-assistant trajectory normalization)
# --------------------------------------------------------------------------- #
TraceOrigin = Literal["opencode", "claude_code", "codex"]
TraceEventType = Literal["message", "tool_call", "file_edit", "command", "test_result", "error"]


class RepoStateRef(BaseModel):
    url: str | None = None
    base_commit: str | None = None
    initial_tree_hash: str | None = None


class TracePromptRef(BaseModel):
    user_task: str
    system_prompt_ref: str | None = None
    developer_prompt_ref: str | None = None


class TraceEvent(BaseModel):
    type: TraceEventType
    timestamp: str | None = None
    agent_turn: int = 0
    content_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceArtifacts(BaseModel):
    final_patch_ref: str | None = None
    workspace_snapshot_ref: str | None = None
    public_test_log_ref: str | None = None
    hidden_grade_ref: str | None = None


class TraceUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    # Optional provider telemetry. Some providers do not report this; external
    # cost ledgers are authoritative when it is absent or zero.
    cost_usd: float | None = None
    wall_time_seconds: float = 0.0


class TracePrivacy(BaseModel):
    redacted: bool = True
    contains_user_secret: bool = False
    license_status: str = "unknown"


class AgentTrace(BaseModel):
    trace_id: str
    origin_harness: TraceOrigin
    harness_version: str | None = None
    worker_model: str
    worker_config_hash: str | None = None
    task_id: str
    repo: RepoStateRef = Field(default_factory=RepoStateRef)
    prompt: TracePromptRef
    events: list[TraceEvent] = Field(default_factory=list)
    artifacts: TraceArtifacts = Field(default_factory=TraceArtifacts)
    grade: Grade | None = None
    usage: TraceUsage = Field(default_factory=TraceUsage)
    privacy: TracePrivacy = Field(default_factory=TracePrivacy)


class ToolPermissions(BaseModel):
    read_files: bool = True
    edit_files: bool = False
    run_tests: bool = False
    network: bool = False


class WorkerIdentity(BaseModel):
    worker_id: int
    name: str
    backend: Literal["opencode", "claude_code", "codex", "direct_qa", "tool_dialog", "terminal"]
    model: str
    role_prior: list[str] = Field(default_factory=list)
    max_turns: int | None = None
    max_reported_cost_usd: float | None = None
    tool_permissions: ToolPermissions = Field(default_factory=ToolPermissions)
