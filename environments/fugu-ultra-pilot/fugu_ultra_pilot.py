"""Fugu-Ultra Conductor GRPO pilot environment.

This is a normal Verifiers environment loaded by Surogate via:

    env:
      - id: fugu-ultra-pilot
        path: ./environments/fugu-ultra-pilot

Each dataset row prompts the trainable Conductor to emit workflow JSON. The
rubric executes that workflow through Ultra's existing harness executor and
returns the faithful Ultra reward.
"""

from __future__ import annotations

import asyncio
import ast
import json
import os
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset


def _ensure_repo_imports() -> Path:
    root = Path(__file__).resolve().parents[2]
    for path in (root, root / "ultra"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


_REPO_ROOT = _ensure_repo_imports()

from ultra.config import PoolConfig, WorkerSpec  # noqa: E402
from ultra.executor import execute_workflow  # noqa: E402
from ultra.live_worker_safety import validate_live_worker_safety  # noqa: E402
from ultra.providers import YUNWU_LIVE_ALLOW_ENV, load_dotenv, routed_provider_name, routed_slug  # noqa: E402
from ultra.schemas import StepBudget, TaskSpec, Workflow  # noqa: E402
from ultra.workflow import WorkflowValidationError, parse_workflow  # noqa: E402
from ultra.workflow_record_cache import WorkflowRecordCache, workflow_record_cache_key  # noqa: E402
from ultra.workers import FakeProvider, Sampling, WorkerPool  # noqa: E402
from ultra.workers.factory import build_pool  # noqa: E402


DEFAULT_PILOT_CONFIG = (
    _REPO_ROOT / "director" / "manifests" / "fugu_clean_v1" / "grpo_pilot_train" / "pilot_config.json"
)
DEFAULT_TASK_MANIFEST = (
    _REPO_ROOT / "director" / "manifests" / "fugu_clean_v1" / "grpo_pilot_train" / "taskspecs.jsonl"
)
DEFAULT_ARTIFACT_DIR = (
    _REPO_ROOT / "director" / "manifests" / "fugu_clean_v1" / "grpo_pilot_train" / "rollout_artifacts"
)

# Few-shot examples in the paper's exact output shape (prose plan -> three Python lists).
# Taken from the Conductor OOD few-shots (ICLR Fig 17; B.2/Table 4 finds OOD examples train
# best), with model ids remapped into our 4-worker pool [0..3].
_FEWSHOT_EXAMPLES = '''EXAMPLE 1:
Question: Does brain-derived neurotrophic factor enhance the contraction of intestinal muscle strips induced by SP and CGRP in mice? Answer Choices: A. Yes B. No
Assistant Response: This is a factual-recall question rather than one needing algorithmic collaboration, so the best approach is to ask two models independently and have a third reconcile them and format the answer.
model_id = [1, 0, 2]
subtasks = ["Does brain-derived neurotrophic factor enhance the contraction of intestinal muscle strips induced by SP and CGRP in mice? Answer with A for Yes or B for No.", "Does brain-derived neurotrophic factor enhance the contraction of intestinal muscle strips induced by SP and CGRP in mice? Answer with A for Yes or B for No.", "Check the two previous answers and provide the correct answer according to the question's formatting instructions if necessary."]
access_list = [[], [], ["all"]]

EXAMPLE 2:
Question: Evaluate the limit of ( 1/ln(1+t) + 1/ln(1-t) ) as t tends to 0. Provide the final answer in <answer> </answer> tags and use LaTeX notation.
Assistant Response: Given the difficulty of the question, let us try four models. The first two work independently to approximate the limit, potentially via a Taylor expansion; the third verifies their work, optionally using L'Hopital's rule; the final model checks everything and returns the correctly formatted answer.
model_id = [1, 0, 3, 2]
subtasks = ["Understand the question and provide an initial solution to approximate the limit as t tends to 0, potentially by using a Taylor expansion. Show your work in <idea> </idea> tags.", "Understand the question and provide an initial solution to approximate the limit as t tends to 0, potentially by using a Taylor expansion. Show your work in <idea> </idea> tags.", "Verify the work done by the first two models and optionally use L'Hopital's rule or numerical methods to confirm the result. Show your work in <idea> </idea> tags.", "Check the work of the previous models, refine where necessary and obtain the correct final answer. Provide the final answer according to the question's formatting instructions."]
access_list = [[], [], ["all"], ["all"]]

EXAMPLE 3:
Question: Using the numbers [3, 7, 25, 50] and the operations +, -, *, / with each number used at most once, write an arithmetic expression that evaluates exactly to 475. Provide the final expression in <answer> </answer> tags.
Assistant Response: This is a search problem over a small space, so a long pipeline adds little. One strong model can search for a valid expression, and a second model can independently verify the arithmetic and the number-usage constraint before formatting the answer.
model_id = [0, 3]
subtasks = ["Find an arithmetic expression that uses each of the numbers 3, 7, 25 and 50 at most once and evaluates exactly to 475. Search systematically, for example by building large products first and adjusting with the remaining numbers. Show your search in <idea> </idea> tags.", "Verify that the proposed expression evaluates exactly to 475 and uses each allowed number at most once. Fix it if it is wrong, then provide the final expression in <answer> </answer> tags."]
access_list = [[], ["all"]]

EXAMPLE 4:
Question: Using the numbers [2, 5, 8, 9, 75] and the operations +, -, *, / with each number used at most once, write an arithmetic expression that evaluates exactly to 632. Provide the final expression in <answer> </answer> tags.
Assistant Response: The target is far from any single product, so I will run two searchers independently so they cannot anchor on each other's partial attempts, have a third model compare their candidates, a fourth model re-check only the chosen expression's arithmetic in isolation, and a final model format the result.
model_id = [2, 0, 1, 3, 2]
subtasks = ["Search for an arithmetic expression using each of the numbers 2, 5, 8, 9, 75 at most once that evaluates exactly to 632. Consider anchoring on multiples of 75 or 8 and adjusting with the smaller numbers. Show your search in <idea> </idea> tags.", "Search for an arithmetic expression using each of the numbers 2, 5, 8, 9, 75 at most once that evaluates exactly to 632. Consider anchoring on multiples of 75 or 8 and adjusting with the smaller numbers. Show your search in <idea> </idea> tags.", "Compare the two candidate expressions, check which ones satisfy the target and the usage constraint, and choose the best verified candidate.", "Re-compute the chosen expression step by step in isolation and confirm it equals exactly 632 and respects the usage constraint.", "Provide the final expression in <answer> </answer> tags according to the question's formatting instructions."]
access_list = [[], [], ["all"], [3], ["all"]]'''


def _system_prompt(max_workflow_steps: int) -> str:
    return f"""Your role as an assistant involves obtaining answers to questions by an iterative process of querying powerful language models, each with a different skillset.
You are given a user-provided question and a list of available numbered language models with their metadata. Your objective is to output a sequence of up to {max_workflow_steps} workflow steps.
Each routing is made of three elements: A language model, its assigned subtask to accomplish, and an "access list" of past workflow steps it will see in its context when trying to accomplish the subtask.
A subtask could directly ask the language model to solve the given question from scratch, refine the solution of the previous subtask in the sequence, or perform any other completely different task that would facilitate later language models in the sequence to answer the original question with their expertise.
Based on your answer, the first model selected will be prompted with the user question and the first subtask you define. Each following model in the sequence will be prompted with the history of the previous subtask and response messages specified in its access list, and will be asked to accomplish its relative subtask. The answer of the final model and subtask will be provided back as the final solution to the user.
Your response should be provided as three Python lists.
The first list should be called model_id, and contain the integers corresponding to the numbered language models in the sequence you want to prompt.
The second list should be called subtasks, and contain the strings that will be used to prompt the corresponding language model specified in model_id.
The third list should be called access_list, and contain the lists of past routing messages (subtasks and assistant responses) from the previous routing steps to include in the context in the current routing step.
You can pass the string all for any of the routing steps in access_list to provide all the previous routing messages in the language model's context. Alternatively, if you want an agent to attempt its subtask without any access to previous routing steps, you can pass an empty list.
For instance:
{_FEWSHOT_EXAMPLES}"""


BACKEND_HARNESS_OVERRIDES = {
    "opencode": "opencode",
    "claude_code": "claude_code",
    "codex": "codex",
    "terminal": "terminal_sandbox",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def _normalize_max_examples(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return int(value)


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned[:160] or "task"


def _info_dict(info: Any) -> dict[str, Any]:
    if isinstance(info, str):
        return json.loads(info)
    if isinstance(info, dict):
        return dict(info)
    raise TypeError(f"unsupported info type: {type(info).__name__}")


def _completion_text(completion: Any) -> str:
    # Verifiers may deliver messages as dicts OR as pydantic/OpenAI message objects
    # (attribute access). Falling through to str(...) would hand the parser the object
    # REPR, whose escape sequences (\n, \') corrupt the three-list payload.
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            content = last.get("content")
        else:
            content = getattr(last, "content", None)
        if isinstance(content, list):  # content-parts form
            content = "".join(
                str(p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")) for p in content
            )
        if content is not None:
            return str(content)
        return ""
    return str(completion or "")


def _balanced_list(s: str, i: int) -> str | None:
    """Return the balanced ``[...]`` span of s starting at s[i]=='[', respecting string
    literals so brackets inside subtask strings don't throw off the bracket depth."""
    depth = 0
    quote: str | None = None
    esc = False
    for j in range(i, len(s)):
        c = s[j]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if quote is not None:
            if c == quote:
                quote = None
            continue
        if c in "\"'":
            quote = c
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return s[i : j + 1]
    return None


def _extract_workflow_payload(raw: str) -> str:
    """Translate the Conductor's paper-format response into internal Workflow JSON.

    The Conductor deliberates (a prose plan, or a <think> block) and then emits three
    Python lists -- model_id, subtasks, access_list (Conductor prompt, ICLR Fig 13). We
    discard the preamble, read the three lists (taking the last occurrence of each, after
    any reasoning), and map them to {"steps":[{worker_id,subtask,access}]}. An access entry
    of "all" expands to every earlier step; [] means no prior context. Anything that
    doesn't yield three parseable lists is returned as-is so parse_workflow fails it (r=0)."""
    text = raw.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]

    def _grab(name: str) -> str | None:
        last = None
        for m in re.finditer(name + r"\s*=\s*(?=\[)", text):
            cand = _balanced_list(text, m.end())
            if cand is not None:
                last = cand
        return last

    raw_ids = _grab(r"model[_ ]?id")
    raw_subs = _grab(r"subtasks")
    raw_acc = _grab(r"access[_ ]?list")
    if not (raw_ids and raw_subs and raw_acc):
        return text
    try:
        ids = ast.literal_eval(raw_ids)
        subs = ast.literal_eval(raw_subs)
        acc = ast.literal_eval(raw_acc)
    except (ValueError, SyntaxError):
        return text
    if not (isinstance(ids, list) and isinstance(subs, list) and isinstance(acc, list)):
        return text
    steps = []
    for i in range(min(len(ids), len(subs), len(acc))):
        entry = acc[i]
        if isinstance(entry, str):
            entry = [entry]
        entry = entry or []
        if any(isinstance(e, str) and e.strip().lower() == "all" for e in entry):
            access = list(range(i))  # "all" == every earlier step
        else:
            access = [int(e) for e in entry if not isinstance(e, str)]
        steps.append({"worker_id": int(ids[i]), "subtask": str(subs[i]), "access": access})
    return json.dumps({"steps": steps})


def _task_lane_map(pilot_config: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for lane, task_ids in pilot_config.get("task_ids_by_lane", {}).items():
        for task_id in task_ids:
            out[str(task_id)] = str(lane)
    return out


def _worker_rows_for_lane(pilot_config: dict[str, Any], lane: str) -> list[dict[str, Any]]:
    names = list(pilot_config["lane_worker_masks"][lane])
    pool = pilot_config["worker_pool"]
    rows = []
    for idx, name in enumerate(names):
        ident = pool[name]
        roles = ", ".join(str(role) for role in ident.get("role_prior", []))
        backend = str(ident.get("backend") or "")
        model = str(ident.get("model") or "")
        rows.append(
            {
                "worker_id": idx,
                "name": name,
                "backend": backend,
                "model": model,
                "role_prior": roles,
            }
        )
    return rows


def _messages_text(task: TaskSpec, *, max_chars: int) -> str:
    parts = []
    for msg in task.input.messages:
        role = msg.get("role", "message")
        content = str(msg.get("content") or "")
        parts.append(f"[{role}]\n{content}")
    if task.input.repo is not None:
        repo = task.input.repo
        parts.append(f"[repo]\nurl={repo.url}\nbase_commit={repo.base_commit}\nsubdirectory={repo.subdirectory}")
    if task.input.context_documents:
        docs = []
        for i, doc in enumerate(task.input.context_documents, start=1):
            if isinstance(doc, dict):
                title = doc.get("title") or doc.get("id") or f"document-{i}"
                text = doc.get("text") or doc.get("content") or json.dumps(doc, sort_keys=True)
            else:
                title = f"document-{i}"
                text = str(doc)
            docs.append(f"- {title}: {str(text)[:500]}")
        parts.append("[context documents]\n" + "\n".join(docs))
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[truncated for conductor prompt]"
    return text


def _prompt_for_task(task: TaskSpec, pilot_config: dict[str, Any], lane: str, *, max_task_chars: int) -> list[dict[str, str]]:
    workers = _worker_rows_for_lane(pilot_config, lane)
    max_workflow_steps = int(pilot_config["workflow_policy"]["max_workflow_steps"])
    # Models are passed as ordinal numbers with capability metadata only -- no brand names,
    # so the Conductor learns worker strengths from reward, not priors (ICLR App. E).
    worker_lines = [f"Model {row['worker_id']}: roles={row['role_prior']}" for row in workers]
    user = "\n\n".join(
        [
            "USER QUESTION:\n" + _messages_text(task, max_chars=max_task_chars),
            "AVAILABLE LANGUAGE MODELS:\n" + "\n".join(worker_lines),
        ]
    )
    return [{"role": "system", "content": _system_prompt(max_workflow_steps)}, {"role": "user", "content": user}]


def _worker_specs(pilot_config: dict[str, Any]) -> list[WorkerSpec]:
    specs = []
    for name in pilot_config["worker_pool_names"]:
        ident = pilot_config["worker_pool"][name]
        specs.append(WorkerSpec(worker_id=name, model=str(ident["model"])))
    return specs


def _worker_harness_overrides(pilot_config: dict[str, Any]) -> dict[str, str]:
    overrides = {}
    for name, ident in pilot_config["worker_pool"].items():
        backend = str(ident.get("backend") or "")
        harness = BACKEND_HARNESS_OVERRIDES.get(backend)
        if harness is not None:
            overrides[name] = harness
    return overrides


def _execution_fingerprint(worker_models: dict[str, str]) -> dict[str, Any]:
    env_keys = [
        "ULTRA_COMMERCIAL_PROVIDER",
        "ULTRA_SPECIALIST_PROVIDER",
        "ULTRA_PROVIDER",
        "ULTRA_HARBOR_PROVIDER",
        "ULTRA_HARBOR_MAX_TOKENS",
        "ULTRA_CODEX_MODEL",
        "ULTRA_CODEX_PROVIDER",
        "ULTRA_CODEX_WIRE_API",
        "ULTRA_CLAUDE_MODEL",
        "ULTRA_CLAUDE_PROVIDER",
        "ULTRA_CLAUDE_CLI_MODEL",
        "ULTRA_CLAUDE_PROVIDER_SORT",
    ]
    routes = {}
    for worker_id, model in worker_models.items():
        provider = routed_provider_name(model)
        routes[worker_id] = {
            "model": model,
            "provider": provider,
            "provider_model": routed_slug(model, provider),
        }
    return {
        "routes": routes,
        "env": {key: os.environ.get(key) for key in env_keys if os.environ.get(key) is not None},
    }


class UltraPilotRuntime:
    def __init__(
        self,
        *,
        pilot_config: dict[str, Any],
        tasks: list[TaskSpec],
        pool: WorkerPool,
        sampling: Sampling,
        artifact_dir: Path | None,
        workflow_record_cache_dir: Path | None,
        force_step_budget: StepBudget | None = None,
        replay_outcomes: dict[str, set[str]] | None = None,
    ):
        self.pilot_config = pilot_config
        self.tasks_by_id = {task.task_id: task for task in tasks}
        self.pool = pool
        self.sampling = sampling
        self.artifact_dir = artifact_dir
        self.workflow_record_cache = WorkflowRecordCache(workflow_record_cache_dir)
        self.force_step_budget = force_step_budget
        self.replay_outcomes = replay_outcomes
        self.worker_harnesses = _worker_harness_overrides(pilot_config)
        self.lane_masks = {
            str(lane): [str(worker) for worker in workers]
            for lane, workers in pilot_config["lane_worker_masks"].items()
        }
        self.max_workflow_steps = int(pilot_config["workflow_policy"]["max_workflow_steps"])

    async def score(self, raw_completion: str, info: dict[str, Any], state: dict[str, Any]) -> float:
        task_id = str(info["task_id"])
        lane = str(info["lane"])
        task = self.tasks_by_id[task_id]
        worker_ids = self.lane_masks[lane]
        rollout_id = f"vf-{_safe_slug(task_id)}-{uuid.uuid4().hex[:12]}"

        try:
            workflow = parse_workflow(_extract_workflow_payload(raw_completion))
        except WorkflowValidationError as exc:
            state["_ultra_outcome_class"] = "invalid_workflow_trainable"
            state["_ultra_valid_for_training"] = True
            state["_ultra_workflow_parse_valid"] = False
            state["_ultra_failure_class"] = f"invalid_workflow_trainable: {exc}"
            state["_ultra_workflow_cache_hit"] = False
            return 0.0
        if self.force_step_budget is not None:
            workflow = workflow.model_copy(
                update={
                    "steps": [
                        step.model_copy(update={"budget": self.force_step_budget})
                        for step in workflow.steps
                    ]
                }
            )

        if self.replay_outcomes is not None:
            # Offline reward: no live worker. Canonicalize the emitted workflow into a single
            # arm token (solo OR coordination template) and score it against the seed
            # manifest's recorded successful_arms for this task. This lets the conductor learn
            # to ORCHESTRATE (emit a coordination workflow), not just route a solo worker.
            wins = self.replay_outcomes.get(task_id, set())
            models = [
                worker_ids[step.worker_id]
                for step in workflow.steps
                if isinstance(step.worker_id, int) and 0 <= step.worker_id < len(worker_ids)
            ]
            arm = _canonical_arm(models)
            success = arm is not None and arm in wins
            state["_ultra_outcome_class"] = "valid_correct" if success else "valid_incorrect"
            state["_ultra_valid_for_training"] = True
            state["_ultra_workflow_parse_valid"] = True
            state["_ultra_grade_success"] = success
            state["_ultra_failure_class"] = ""
            state["_ultra_workflow_cache_hit"] = False
            state["_ultra_record"] = {}
            return 1.0 if success else 0.5

        artifact_root = None
        if self.artifact_dir is not None:
            artifact_root = self.artifact_dir / _safe_slug(task_id) / rollout_id

        worker_models = {worker_id: self.pool.model_for(worker_id) for worker_id in worker_ids}
        cache_key = workflow_record_cache_key(
            task=task,
            workflow=workflow,
            worker_ids=worker_ids,
            worker_models=worker_models,
            worker_harnesses=self.worker_harnesses,
            sampling=self.sampling,
            max_steps=self.max_workflow_steps,
            execution_fingerprint=_execution_fingerprint(worker_models),
        )
        cached = self.workflow_record_cache.get(cache_key)
        if cached is not None:
            record = cached.model_copy(
                update={
                    "rollout_id": rollout_id,
                    "conductor": cached.conductor.model_copy(update={"raw_output": raw_completion}),
                },
                deep=True,
            )
            state["_ultra_workflow_cache_hit"] = True
        else:
            record = await execute_workflow(
                task,
                workflow,
                self.pool,
                self.sampling,
                rollout_id,
                worker_ids=worker_ids,
                worker_harnesses=self.worker_harnesses,
                raw_output=raw_completion,
                max_steps=self.max_workflow_steps,
                artifact_dir=artifact_root,
            )
            state["_ultra_workflow_cache_hit"] = False
            if record.valid_for_training:
                self.workflow_record_cache.set(cache_key, record)
        state["_ultra_record"] = record.model_dump(mode="json")
        state["_ultra_outcome_class"] = record.outcome_class or ""
        state["_ultra_valid_for_training"] = bool(record.valid_for_training)
        state["_ultra_workflow_parse_valid"] = bool(record.conductor.workflow_parse_valid)
        state["_ultra_failure_class"] = record.failure_class or ""
        state["_ultra_grade_success"] = bool(record.grade.success) if record.grade is not None else False
        reward = float(record.reward) if record.reward is not None and record.valid_for_training else 0.0
        penalty = _redundancy_penalty(workflow)
        state["_ultra_redundancy_penalty"] = penalty
        return max(0.0, reward - penalty)


class FuguUltraPilotEnv(vf.SingleTurnEnv):
    """Verifiers env wrapper that keeps post-load kwargs in sync with runtime."""

    def set_force_step_budget(self, value: StepBudget | None) -> None:
        if value is not None and value not in {"short", "medium", "long", "max"}:
            raise ValueError(f"invalid force_step_budget: {value!r}")
        self.force_step_budget = value
        self.ultra_runtime.force_step_budget = value

    def set_worker_max_tokens(self, value: int) -> None:
        self.worker_max_tokens = int(value)
        sampling = self.ultra_runtime.sampling
        self.ultra_runtime.sampling = Sampling(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=int(value),
            seed=sampling.seed,
            reasoning_effort=sampling.reasoning_effort,
        )

    def set_worker_reasoning_effort(self, value: str | None) -> None:
        self.worker_reasoning_effort = value
        sampling = self.ultra_runtime.sampling
        self.ultra_runtime.sampling = Sampling(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            seed=sampling.seed,
            reasoning_effort=value,
        )

    def set_artifact_dir(self, value: str | None) -> None:
        self.artifact_dir = value
        self.ultra_runtime.artifact_dir = Path(value).expanduser().resolve() if value else None

    def set_workflow_record_cache_dir(self, value: str | None) -> None:
        self.workflow_record_cache_dir = value
        self.ultra_runtime.workflow_record_cache = WorkflowRecordCache(
            Path(value).expanduser().resolve() if value else None
        )

    def set_harbor_timeout_s(self, value: int | float | None) -> None:
        self.harbor_timeout_s = value
        if value is None:
            os.environ.pop("ULTRA_HARBOR_TIMEOUT_SECONDS", None)
        else:
            os.environ["ULTRA_HARBOR_TIMEOUT_SECONDS"] = str(int(float(value)))


_SUBTASK_STOPWORDS = frozenset(
    "a an the and or to of in on for with from using it its this that".split()
)


def _redundancy_penalty(workflow: Any) -> float:
    """Penalize near-duplicate subtasks (the N-x-identical-attempt ensemble wastes worker calls).

    Token-Jaccard over stopword-stripped subtasks so trivial rewordings count as duplicates,
    while genuinely diverse steps (draft/critique/verify/aggregate) share few content tokens.
    0.1 per redundant step, capped at 0.3 to preserve reward ordering:
    correct-redundant (>=0.7) > valid-wrong-clean (0.5) > valid-wrong-redundant (>=0.2) > unparseable (0.0).
    """
    token_sets = []
    for step in workflow.steps:
        tokens = frozenset(re.findall(r"[a-z]+", str(step.subtask).lower())) - _SUBTASK_STOPWORDS
        if tokens:
            token_sets.append(tokens)
    redundant = 0
    for i in range(1, len(token_sets)):
        for j in range(i):
            union = token_sets[i] | token_sets[j]
            if union and len(token_sets[i] & token_sets[j]) / len(union) >= 0.75:
                redundant += 1
                break
    return min(0.3, 0.1 * redundant)


def _ensure_scored(state: dict[str, Any]) -> dict[str, Any]:
    return state.get("_ultra_record") or {}


async def ultra_workflow_reward(completion: Any, info: Any, state: dict[str, Any], ultra_runtime: UltraPilotRuntime) -> float:
    return await ultra_runtime.score(_completion_text(completion), _info_dict(info), state)


def ultra_valid_for_training(state: dict[str, Any]) -> float:
    _ensure_scored(state)
    return 1.0 if state.get("_ultra_valid_for_training", False) else 0.0


def ultra_workflow_parse_valid(state: dict[str, Any]) -> float:
    _ensure_scored(state)
    return 1.0 if state.get("_ultra_workflow_parse_valid", False) else 0.0


def ultra_grade_success(state: dict[str, Any]) -> float:
    _ensure_scored(state)
    return 1.0 if state.get("_ultra_grade_success", False) else 0.0


def ultra_workflow_cache_hit(state: dict[str, Any]) -> float:
    _ensure_scored(state)
    return 1.0 if state.get("_ultra_workflow_cache_hit", False) else 0.0


def ultra_redundancy_penalty(state: dict[str, Any]) -> float:
    _ensure_scored(state)
    return float(state.get("_ultra_redundancy_penalty", 0.0))


# --- Stage-2 multi-turn (recursion-lite, report 3.2.2): plan -> execute+feedback -> repair-plan ---
# Validated at inference 2026-07-08: the UNTRAINED turn-2 loop converted fu1 failures on both
# fshard runs (fu2 0.923 vs best premium solo 0.846). This env trains exactly that behavior.

_MT_REVISE_INSTRUCTION = (
    "The workflow above was executed and its final answer was evaluated as INCORRECT.\n"
    "Outcome of the previous attempt:\n{outcome}\n\n"
    "Using what the previous attempt revealed, design a NEW workflow that diagnoses the error and "
    "produces a correct solution to the original question. Output the three lists as before."
)


async def _mt_score_last_turn(runtime: UltraPilotRuntime, state: dict[str, Any]) -> dict[str, Any]:
    """Score the most recent conductor turn via the SAME runtime.score path as single-turn
    training (identical reward semantics: execution, cache, penalties, validity flags).
    Appends a turn record + inter-workflow shared memory (report 3.2.2)."""
    turn_state: dict[str, Any] = {}
    raw = _completion_text(state["trajectory"][-1]["completion"])
    reward = await runtime.score(raw, _info_dict(state.get("info")), turn_state)
    record = turn_state.get("_ultra_record") or {}
    steps = (record.get("execution") or {}).get("steps") or []
    if steps:
        outcome = str(steps[-1].get("text", ""))[:1500]
    else:
        outcome = f"(no executed steps: {turn_state.get('_ultra_failure_class') or 'unparseable workflow'})"
    rec = {
        "reward": float(reward),
        "success": bool(turn_state.get("_ultra_grade_success", False)),
        "turn_state": turn_state,
        "outcome_text": outcome,
    }
    state.setdefault("turn_records", []).append(rec)
    state.setdefault("shared_memory", []).append(outcome)
    return rec


async def ultra_mt_reward(
    completion: Any, info: Any, state: dict[str, Any], ultra_runtime: UltraPilotRuntime
) -> float:
    """Terminal reward = the LAST executed turn's reward. env_response scores every turn
    except the final generated one (unless the rollout early-exited on success) — score it
    here. Also surfaces the terminal turn's flags at rollout level so the 0-weight metric
    funcs (ultra_valid_for_training etc.) read the same keys as in the single-turn env."""
    del completion, info
    records = state.get("turn_records") or []
    if len(records) < len(state.get("trajectory") or []):
        await _mt_score_last_turn(ultra_runtime, state)
        records = state["turn_records"]
    if not records:
        return 0.0
    last = records[-1]["turn_state"]
    for key in (
        "_ultra_record", "_ultra_outcome_class", "_ultra_valid_for_training",
        "_ultra_workflow_parse_valid", "_ultra_grade_success", "_ultra_failure_class",
        "_ultra_workflow_cache_hit", "_ultra_redundancy_penalty",
    ):
        if key in last:
            state[key] = last[key]
    return float(records[-1]["reward"])


def ultra_mt_turns(state: dict[str, Any]) -> float:
    return float(len(state.get("turn_records") or []))


def ultra_mt_converted(state: dict[str, Any]) -> float:
    """1.0 when turn-1 failed and a later repair turn succeeded — the trained behavior."""
    records = state.get("turn_records") or []
    return 1.0 if len(records) >= 2 and not records[0]["success"] and records[-1]["success"] else 0.0


class FuguUltraMultiTurnEnv(vf.MultiTurnEnv):
    """Recursion-lite conductor env: turn 0 plans; env_response executes + grades it; on failure
    the outcome comes back as a revise turn (intra-workflow isolation stays inside
    execute_workflow via access lists; inter-workflow memory = state["shared_memory"]).
    Early-terminates on success via final_env_response; max_turns bounds the loop."""

    def __init__(self, *, ultra_runtime: UltraPilotRuntime, **kwargs):
        super().__init__(**kwargs)
        self.ultra_runtime = ultra_runtime

    async def setup_state(self, state: dict[str, Any]) -> dict[str, Any]:
        state.setdefault("turn_records", [])
        state.setdefault("shared_memory", [])
        return state

    async def env_response(self, messages: Any, state: dict[str, Any], **kwargs) -> Any:
        del messages, kwargs
        rec = await _mt_score_last_turn(self.ultra_runtime, state)
        if rec["success"]:
            state["final_env_response"] = [{"role": "user", "content": "Solved."}]
            return [{"role": "user", "content": "Solved."}]
        return [{"role": "user", "content": _MT_REVISE_INSTRUCTION.format(outcome=rec["outcome_text"])}]


def _build_pool(
    *,
    pilot_config: dict[str, Any],
    provider_mode: str,
    cache_dir: str | None,
    max_concurrency: int,
    requests_per_minute: float | None,
    timeout_s: float,
    max_retries: int,
) -> WorkerPool:
    workers = _worker_specs(pilot_config)
    if provider_mode == "fake":
        return WorkerPool(workers, FakeProvider(), max_retries=0)
    if provider_mode != "live":
        raise ValueError(f"provider_mode must be 'live' or 'fake', got {provider_mode!r}")
    load_dotenv()
    return build_pool(
        workers,
        PoolConfig(
            split_provider_routing=True,
            cache_dir=cache_dir,
            max_concurrency=max_concurrency,
            requests_per_minute=requests_per_minute,
            timeout_s=timeout_s,
            max_retries=max_retries,
            budget_usd=None,
        ),
    )


def _build_dataset(
    *,
    tasks: list[TaskSpec],
    pilot_config: dict[str, Any],
    task_name: str,
    lane: str | None,
    max_examples: int | None,
    seed: int,
    shuffle: bool,
    max_task_chars: int,
) -> Dataset:
    lane_by_task = _task_lane_map(pilot_config)
    selected = []
    for task in tasks:
        task_lane = lane_by_task.get(task.task_id)
        if task_lane is None:
            continue
        if lane is not None and task_lane != lane:
            continue
        selected.append((task, task_lane))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    if max_examples is not None:
        selected = selected[:max_examples]
    if not selected:
        raise ValueError("fugu-ultra-pilot selected zero tasks")

    rows = []
    for idx, (task, task_lane) in enumerate(selected):
        rows.append(
            {
                "example_id": idx,
                "task": task_name,
                "prompt": _prompt_for_task(task, pilot_config, task_lane, max_task_chars=max_task_chars),
                "answer": task.task_id,
                "info": json.dumps(
                    {
                        "task_id": task.task_id,
                        "lane": task_lane,
                        "source": task.source.name,
                        "harness": task.environment.harness,
                    },
                    sort_keys=True,
                ),
            }
        )
    return Dataset.from_list(rows)


def _canonical_arm(models: list[str]) -> str | None:
    """Canonicalize an emitted workflow's model sequence into a replay arm token, matching the
    precomputed tournament arms (solo + coordination templates). 1 model -> solo; 2 ->
    builder (first drafts, second debugs); 3 -> ladder. Keep in sync with the arm names in
    build_ultra_frontier_coding_bank.py / the replay manifest."""
    if not models:
        return None
    if len(models) == 1:
        return f"solo__{models[0]}"
    if len(models) == 2:
        return f"coord__builder__{models[0]}__debug__{models[1]}"
    if len(models) == 3:
        return f"coord__ladder__{models[0]}__{models[1]}__{models[2]}"
    return None


def _load_replay_outcomes(path: Path, lane: str | None) -> dict[str, set[str]]:
    """task_id -> set of arms (solo OR coord) that solved it (recorded in the seed manifest).

    Enables free offline GRPO: the conductor's emitted workflow is canonicalized to an arm
    and scored against the already-paid tournament outcome instead of a live worker call.
    """
    out: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if lane is not None and row.get("lane") != lane:
            continue
        task_id = row.get("source_task_id") or row.get("task_id")
        if not task_id:
            continue
        out[str(task_id)] = {
            a for a in (row.get("successful_arms") or []) if str(a).startswith(("solo__", "coord__"))
        }
    return out


def load_environment(
    pilot_config_path: str = str(DEFAULT_PILOT_CONFIG),
    task_manifest_path: str = str(DEFAULT_TASK_MANIFEST),
    task_name: str = "fugu_ultra_pilot",
    lane: str | None = None,
    max_examples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    provider_mode: str = "live",
    allow_yunwu_live: bool = False,
    artifact_dir: str | None = str(DEFAULT_ARTIFACT_DIR),
    cache_dir: str | None = "./.ultra_cache/grpo_pilot_worker_completions",
    workflow_record_cache_dir: str | None = "./.ultra_cache/grpo_pilot_workflow_records",
    live_safety_path: str | None = None,
    max_concurrency: int = 8,
    requests_per_minute: float | None = None,
    timeout_s: float = 300.0,
    max_retries: int = 2,
    worker_temperature: float = 0.2,
    worker_top_p: float = 1.0,
    worker_max_tokens: int = 8192,
    worker_reasoning_effort: str | None = "high",
    force_step_budget: StepBudget | None = None,
    max_task_chars: int = 12000,
    score_rollouts: bool = True,
    max_seq_len: int | None = None,
    replay_seed_manifest: str | None = None,
    max_turns: int = 1,
) -> vf.Environment:
    """Load the frozen Ultra pilot as a Verifiers SingleTurnEnv.

    ``max_turns > 1`` loads the Stage-2 multi-turn variant instead (plan -> execute+feedback ->
    repair-plan; terminal reward = last executed turn). Single-turn lanes are untouched."""

    pilot_path = Path(pilot_config_path).expanduser().resolve()
    task_path = Path(task_manifest_path).expanduser().resolve()
    pilot_config = _read_json(pilot_path)
    if not pilot_config.get("ready_for_pilot"):
        raise ValueError(f"pilot config is not ready_for_pilot: {pilot_path}")
    if force_step_budget is not None and force_step_budget not in {"short", "medium", "long", "max"}:
        raise ValueError(f"invalid force_step_budget: {force_step_budget!r}")
    if allow_yunwu_live:
        os.environ[YUNWU_LIVE_ALLOW_ENV] = "1"
    tasks = [TaskSpec.model_validate(row) for row in _read_jsonl(task_path)]
    replay_outcomes = (
        _load_replay_outcomes(Path(replay_seed_manifest).expanduser().resolve(), lane)
        if replay_seed_manifest
        else None
    )
    max_n = _normalize_max_examples(max_examples)
    safety_report = validate_live_worker_safety(
        pilot_config=pilot_config,
        lane=lane,
        provider_mode=provider_mode,
        allow_yunwu_live=allow_yunwu_live,
        live_safety_path=live_safety_path,
        max_examples=max_n,
        force_step_budget=force_step_budget,
    )

    dataset = _build_dataset(
        tasks=tasks,
        pilot_config=pilot_config,
        task_name=task_name,
        lane=lane,
        max_examples=max_n,
        seed=seed,
        shuffle=shuffle,
        max_task_chars=max_task_chars,
    )
    pool = _build_pool(
        pilot_config=pilot_config,
        provider_mode=provider_mode,
        cache_dir=cache_dir,
        max_concurrency=max_concurrency,
        requests_per_minute=requests_per_minute,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
    runtime = UltraPilotRuntime(
        pilot_config=pilot_config,
        tasks=tasks,
        pool=pool,
        sampling=Sampling(
            temperature=worker_temperature,
            top_p=worker_top_p,
            max_tokens=worker_max_tokens,
            seed=seed,
            reasoning_effort=worker_reasoning_effort,
        ),
        artifact_dir=Path(artifact_dir).expanduser().resolve() if artifact_dir else None,
        workflow_record_cache_dir=Path(workflow_record_cache_dir).expanduser().resolve()
        if workflow_record_cache_dir
        else None,
        force_step_budget=force_step_budget,
        replay_outcomes=replay_outcomes,
    )
    env_args = {
        "pilot_config_path": str(pilot_path),
        "task_manifest_path": str(task_path),
        "task_name": task_name,
        "lane": lane,
        "max_examples": max_examples,
        "provider_mode": provider_mode,
        "allow_yunwu_live": allow_yunwu_live,
        "live_safety_path": live_safety_path,
        "live_safety": safety_report,
    }
    if max_turns > 1:
        mt_rubric = vf.Rubric(
            funcs=[
                ultra_mt_reward,
                ultra_valid_for_training,
                ultra_workflow_parse_valid,
                ultra_grade_success,
                ultra_mt_turns,
                ultra_mt_converted,
            ],
            weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        mt_rubric.add_class_object("ultra_runtime", runtime)
        return FuguUltraMultiTurnEnv(
            ultra_runtime=runtime,
            max_turns=max_turns,
            dataset=dataset,
            rubric=mt_rubric,
            score_rollouts=score_rollouts,
            max_seq_len=max_seq_len,
            env_args={**env_args, "max_turns": max_turns},
        )

    rubric = vf.Rubric(
        funcs=[
            ultra_workflow_reward,
            ultra_valid_for_training,
            ultra_workflow_parse_valid,
            ultra_grade_success,
            ultra_workflow_cache_hit,
            ultra_redundancy_penalty,
        ],
        weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    rubric.add_class_object("ultra_runtime", runtime)

    env = FuguUltraPilotEnv(
        dataset=dataset,
        rubric=rubric,
        score_rollouts=score_rollouts,
        max_seq_len=max_seq_len,
        env_args=env_args,
    )
    env.ultra_runtime = runtime
    return env
