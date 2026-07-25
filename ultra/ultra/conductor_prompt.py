"""Pure prompt and response helpers shared by conductor training and runtimes."""

from __future__ import annotations

import ast
import json
import re
from typing import Any


# These OOD examples are part of the accepted conductor interface. Keep them
# independent of the worker pool; runtime bindings supply the anonymous roles.
FEWSHOT_EXAMPLES = '''EXAMPLE 1:
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


def system_prompt(max_workflow_steps: int) -> str:
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
{FEWSHOT_EXAMPLES}"""


def _balanced_list(value: str, start: int) -> str | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for end in range(start, len(value)):
        char = value[end]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return value[start : end + 1]
    return None


def extract_workflow_payload(raw: str) -> str:
    """Translate paper-format conductor output into internal Workflow JSON."""
    text = raw.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]

    def grab(name: str) -> str | None:
        last = None
        for match in re.finditer(name + r"\s*=\s*(?=\[)", text):
            candidate = _balanced_list(text, match.end())
            if candidate is not None:
                last = candidate
        return last

    raw_ids = grab(r"model[_ ]?id")
    selector_field = "worker_id"
    if raw_ids is None:
        raw_ids = grab(r"profile[_ ]?ref")
        selector_field = "profile_ref"
    raw_subtasks = grab(r"subtasks")
    raw_access = grab(r"access[_ ]?list")
    if not (raw_ids and raw_subtasks and raw_access):
        return text
    try:
        ids = ast.literal_eval(raw_ids)
        subtasks = ast.literal_eval(raw_subtasks)
        access_entries = ast.literal_eval(raw_access)
    except (ValueError, SyntaxError):
        return text
    if not all(isinstance(item, list) for item in (ids, subtasks, access_entries)):
        return text

    steps = []
    for index in range(min(len(ids), len(subtasks), len(access_entries))):
        entry = access_entries[index]
        if isinstance(entry, str):
            entry = [entry]
        entry = entry or []
        if any(isinstance(item, str) and item.strip().lower() == "all" for item in entry):
            access = list(range(index))
        else:
            access = [int(item) for item in entry if not isinstance(item, str)]
        selector = (
            {"worker_id": int(ids[index])}
            if selector_field == "worker_id"
            else {"profile_ref": str(ids[index])}
        )
        steps.append({**selector, "subtask": str(subtasks[index]), "access": access})
    return json.dumps({"steps": steps})


def _worker_rows_for_lane(config: dict[str, Any], lane: str) -> list[dict[str, Any]]:
    rows = []
    for index, name in enumerate(config["lane_worker_masks"][lane]):
        identity = config["worker_pool"][name]
        rows.append(
            {
                "worker_id": index,
                "profile_ref": identity.get("profile_ref"),
                "role_prior": ", ".join(str(role) for role in identity.get("role_prior", [])),
            }
        )
    return rows


def _messages_text(task: Any, *, max_chars: int) -> str:
    parts = []
    for message in task.input.messages:
        parts.append(f"[{message.get('role', 'message')}]\n{str(message.get('content') or '')}")
    if task.input.repo is not None:
        repo = task.input.repo
        parts.append(
            f"[repo]\nurl={repo.url}\nbase_commit={repo.base_commit}\nsubdirectory={repo.subdirectory}"
        )
    if task.input.context_documents:
        documents = []
        for index, document in enumerate(task.input.context_documents, start=1):
            if isinstance(document, dict):
                title = document.get("title") or document.get("id") or f"document-{index}"
                content = document.get("text") or document.get("content") or json.dumps(document, sort_keys=True)
            else:
                title = f"document-{index}"
                content = str(document)
            documents.append(f"- {title}: {str(content)[:500]}")
        parts.append("[context documents]\n" + "\n".join(documents))
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[truncated for conductor prompt]"
    return text


def _capability_ref_system_prompt(max_workflow_steps: int) -> str:
    return f"""Your role is to plan a tool-using workflow over an anonymous set of capability profiles.
Choose up to {max_workflow_steps} positions. Each step has exactly three fields: profile_ref selects one supplied capability profile, subtask assigns concrete work, and access_positions contains only integer indexes of earlier workflow steps whose messages it may observe.
Choose profiles from their supplied capabilities and the task. No profile is a default or fallback. The first access_positions list must be empty. Profile references are never valid access positions, even when the same profile appears in an earlier step.
Return exactly one JSON object with a steps array and no prose: {{"steps":[{{"profile_ref":"one supplied reference","subtask":"task-specific tool-using work","access_positions":[]}}]}}"""


def prompt_for_task(
    task: Any,
    config: dict[str, Any],
    lane: str,
    *,
    max_task_chars: int,
) -> list[dict[str, str]]:
    workers = _worker_rows_for_lane(config, lane)
    max_workflow_steps = int(config["workflow_policy"]["max_workflow_steps"])
    capability_refs = config.get("selector_field") == "profile_ref"
    if capability_refs:
        worker_lines = [
            f"Profile {row['profile_ref']}: roles={row['role_prior']}" for row in workers
        ]
        system = _capability_ref_system_prompt(max_workflow_steps)
    else:
        worker_lines = [
            f"Model {row['worker_id']}: roles={row['role_prior']}" for row in workers
        ]
        system = system_prompt(max_workflow_steps)
    user = "\n\n".join(
        [
            "USER QUESTION:\n" + _messages_text(task, max_chars=max_task_chars),
            "AVAILABLE LANGUAGE MODELS:\n" + "\n".join(worker_lines),
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
