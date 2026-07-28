"""CRMArena offline harness: SQL query loop over the published org data.

CRMArena's tasks are answered by querying a Salesforce org; the benchmark
repo publishes that org's records as local SQLite (local_data/*.db), so the
lane runs fully offline: the worker explores the database through a
``run_sql`` tool (read-only, row-capped) and finishes by stating the answer
as plain text. Reward is CRMArena's own exact-match against the published
gold, normalized (case/whitespace), with ``None`` golds meaning "the correct
answer is that there is none" — the worker must say ``None``.

The fuzzy-match task type (knowledge_qa) is excluded upstream: its official
metric needs an LLM judge, and judge rewards are out of scope for training
(noisy, hackable) per the office-mixture triage.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from ..schemas import Grade, TaskSpec
from ..workers import Sampling, WorkerPool
from .base import StepInput, StepResult, register_harness
from .repo_artifacts import write_json

_MAX_TURNS_BY_BUDGET = {"short": 6, "medium": 10, "long": 16, "max": None}
_MAX_RESULT_ROWS = 50
_MAX_RESULT_CHARS = 4000

RUN_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": "Run one read-only SQLite query against the CRM database "
                       f"and return up to {_MAX_RESULT_ROWS} rows.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "A single SELECT statement."}},
            "required": ["query"],
        },
    },
}

SYSTEM_TEMPLATE = """\
You answer questions about a CRM system by querying its SQLite database.

Schema:
{schema}

{context}

Use the run_sql tool to inspect the data (SELECT only). When you know the
answer, reply WITHOUT calling a tool: state ONLY the answer itself (for
example an Id, a name, or a number). If the correct answer is that there is
no such record, reply exactly: None"""


def _payload(task: TaskSpec) -> dict[str, Any]:
    payload = task.grader.expected_answer
    return payload if isinstance(payload, dict) else {}


def _max_turns_for_step(task: TaskSpec, budget: str) -> int:
    payload = _payload(task)
    configured = int(payload.get("max_turns") or task.metadata.estimated_worker_calls or 16)
    cap = _MAX_TURNS_BY_BUDGET.get(budget, _MAX_TURNS_BY_BUDGET["medium"])
    return min(configured, cap) if cap is not None else configured


def normalize_answer(text: str) -> str:
    lines = [l.strip() for l in str(text).strip().splitlines() if l.strip()]
    final = lines[-1] if lines else ""
    return final.strip().strip(".").strip("'\"").casefold()


def _schema_ddl(db_path: str) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    return "\n".join(r[0].strip() for r in rows)


def run_sql_readonly(db_path: str, query: str) -> str:
    """Execute one query read-only; render a compact, capped result."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
        try:
            cursor = con.execute(query)
            columns = [d[0] for d in cursor.description or []]
            rows = cursor.fetchmany(_MAX_RESULT_ROWS + 1)
        finally:
            con.close()
    except sqlite3.Error as exc:
        return f"SQL error: {exc}"
    truncated = len(rows) > _MAX_RESULT_ROWS
    rows = rows[:_MAX_RESULT_ROWS]
    out = [" | ".join(columns)] if columns else []
    out += [" | ".join("" if v is None else str(v) for v in row) for row in rows]
    text = "\n".join(out) if out else "(no rows)"
    if len(text) > _MAX_RESULT_CHARS:
        text = text[:_MAX_RESULT_CHARS] + "\n[truncated]"
    if truncated:
        text += f"\n[only the first {_MAX_RESULT_ROWS} rows shown]"
    return text


@register_harness
class CrmQueryHarness:
    name = "crm_query"

    def __init__(self) -> None:
        self.db_path: str = ""
        self.transcript: list[dict[str, Any]] = []
        self.final_answer: str = ""
        self.answered: bool = False

    def _init(self, task: TaskSpec) -> None:
        if self.db_path:
            return
        payload = _payload(task)
        self.db_path = str(payload.get("db_path") or "")
        if not Path(self.db_path).is_file():
            raise FileNotFoundError(f"CRM database missing: {self.db_path}")
        system = SYSTEM_TEMPLATE.format(
            schema=_schema_ddl(self.db_path),
            context=str(payload.get("context") or "").strip(),
        )
        self.transcript = [{"role": "system", "content": system}]
        for message in task.input.messages:
            self.transcript.append(dict(message))

    def _messages(self, step: StepInput) -> list[dict[str, Any]]:
        messages = list(self.transcript)
        if step.prior_artifacts:
            blocks = [f"[Worker {a.get('worker_id')} result]\n{a.get('response', '')}"
                      for a in step.prior_artifacts]
            messages.append({"role": "user",
                             "content": "Authorized prior-step results:\n\n" + "\n\n".join(blocks)})
        if step.subtask.strip():
            messages.append({"role": "user", "content": f"Your subtask: {step.subtask}"})
        return messages

    async def run_step(self, step: StepInput, pool: WorkerPool, sampling: Sampling) -> StepResult:
        self._init(step.task)
        max_turns = _max_turns_for_step(step.task, step.budget)
        prompt_tokens = completion_tokens = 0
        cost = 0.0
        termination = "max_turns"

        for _ in range(max_turns if not self.answered else 0):
            comp = await pool.call_tools(step.worker_id, self._messages(step),
                                         [RUN_SQL_TOOL], sampling)
            prompt_tokens += comp.prompt_tokens
            completion_tokens += comp.completion_tokens
            cost += comp.cost_usd
            if comp.tool_calls:
                self.transcript.append({
                    "role": "assistant", "content": comp.content or "",
                    "tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.name,
                                                 "arguments": json.dumps(c.arguments)}}
                                   for c in comp.tool_calls],
                })
                for call in comp.tool_calls:
                    if call.name != "run_sql":
                        result = f"unknown tool {call.name!r}; only run_sql exists"
                    else:
                        result = await asyncio.to_thread(
                            run_sql_readonly, self.db_path,
                            str(call.arguments.get("query") or ""))
                    self.transcript.append({"role": "tool", "tool_call_id": call.id,
                                            "content": result})
                continue
            self.final_answer = comp.content or ""
            self.transcript.append({"role": "assistant", "content": self.final_answer})
            self.answered = True
            termination = "completed"
            break

        if self.answered and termination == "max_turns":
            termination = "completed"  # answered in an earlier workflow step

        messages_ref = None
        if step.artifact_dir:
            messages_ref = write_json(
                Path(step.artifact_dir) / "crm_query_transcript.json",
                {"answered": self.answered, "final_answer": self.final_answer,
                 "transcript": self.transcript})
        return StepResult(
            text=json.dumps({"answered": self.answered,
                             "answer": self.final_answer}, sort_keys=True),
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cost_usd=cost,
            termination=termination,
            messages_ref=messages_ref,
        )

    def grade(self, task: TaskSpec, final: StepResult) -> Grade:
        payload = _payload(task)
        gold = payload.get("answer")
        gold_norm = "none" if gold is None else normalize_answer(str(gold))
        got = normalize_answer(self.final_answer) if self.answered else ""
        score = 1.0 if got and got == gold_norm else 0.0
        return Grade(score=score, success=score >= task.grader.success_threshold,
                     details={"answered": self.answered})

    def close(self) -> None:
        self.db_path = ""
        self.transcript = []
        self.final_answer = ""
        self.answered = False
