"""Counterfactual long-context tasks designed for worker disagreement.

The earlier generated long-context sources mostly test extraction plus light
revision precedence. These tasks add stale exact-format packets, cross-document
joins, and computed check fields so a worker must reject plausible distractors
instead of copying the nearest answer-shaped line.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .schemas import (
    EnvironmentSpec,
    GraderSpec,
    SourceRef,
    SplittingSpec,
    TaskInput,
    TaskMetadata,
    TaskSpec,
)

SOURCE_NAME = "longctx_counterfactual"
SOURCE_VERSION = "v1"

OWNERS = (
    "Mira Chen",
    "Omar Velez",
    "Nadia Flores",
    "Sofia Nguyen",
    "Rafael Costa",
    "Priya Shah",
    "Leah Ortiz",
    "Theo Martin",
)

REGIONS = (
    "eu-west-3",
    "us-east-2",
    "ap-south-1",
    "ca-central-1",
    "eu-central-1",
    "us-west-2",
)

SERVICES = (
    "Orion",
    "Kestrel",
    "Beacon",
    "Helio",
    "Iris",
    "Cobalt",
)


@dataclass(frozen=True)
class CounterfactualLongContextTask:
    task_id: str
    corpus: str
    question: str
    must_contain: tuple[str, ...]
    must_not_contain: tuple[str, ...]
    documents: tuple[dict[str, str], ...]


def _ledger_task(i: int) -> CounterfactualLongContextTask:
    service = SERVICES[i % len(SERVICES)]
    owner = OWNERS[(i + 2) % len(OWNERS)]
    stale_owner = OWNERS[(i + 5) % len(OWNERS)]
    region = REGIONS[(i + 1) % len(REGIONS)]
    stale_region = REGIONS[(i + 4) % len(REGIONS)]
    a0 = 12 + i
    b0 = 31 + 2 * i
    c0 = 7 + (i % 5)
    # Apply only active events in sequence order.
    a = ((a0 + 5 + i) * 2) - (3 + i % 4)
    b = (b0 - 4) + (8 + i % 6)
    c = (c0 * 3) - (2 + i % 3)
    seal = f"seal-{(17 * a + 19 * b + 23 * c + i) % 997:03d}"
    stale_seal = f"seal-{(17 * (a + 11) + 19 * (b + 7) + 23 * (c + 5) + i) % 997:03d}"
    answer = f"{service} / {owner} / {region} / A={a} B={b} C={c} / {seal}"
    stale = f"{service} / {stale_owner} / {stale_region} / A={a + 11} B={b + 7} C={c + 5} / {stale_seal}"
    task_id = f"ledger-crosscheck-{i + 1:02d}"
    return CounterfactualLongContextTask(
        task_id=task_id,
        corpus=f"counterfactual_ledger_{i // 4:02d}",
        question=(
            f"Resolve the active packet for service `{service}`. Return exactly "
            "`service / owner / region / A=<value> B=<value> C=<value> / seal`. Return only that line."
        ),
        must_contain=(answer,),
        must_not_contain=(stale_owner, stale_region, stale_seal, f"A={a + 11}", f"B={b + 7}", f"C={c + 5}"),
        documents=(
            {
                "title": "state rules",
                "text": (
                    f"Initial state for {service}: A={a0}, B={b0}, C={c0}. "
                    "Process active events in ascending sequence order. Ignore void, rehearsal, and projected events. "
                    "Seal is seal-XYZ where XYZ=(17*A + 19*B + 23*C + task_index) mod 997, padded to 3 digits. "
                    f"For this packet, task_index={i}."
                ),
            },
            {
                "title": "events early",
                "text": (
                    f"seq=01 active {service} A add {5 + i}. "
                    f"seq=02 active {service} B subtract 4. "
                    f"seq=03 rehearsal {service} C add 100. "
                    f"seq=04 active {service} C triple."
                ),
            },
            {
                "title": "events late",
                "text": (
                    f"seq=05 active {service} A double. "
                    f"seq=06 void {service} B add 70. "
                    f"seq=07 active {service} B add {8 + i % 6}. "
                    f"seq=08 active {service} C subtract {2 + i % 3}. "
                    f"seq=09 active {service} A subtract {3 + i % 4}."
                ),
            },
            {
                "title": "owner and region ledger",
                "text": (
                    f"active revision 2 service {service} owner {stale_owner} region {stale_region}. "
                    f"active revision 5 service {service} owner {owner} region {region}. "
                    "Use the highest active revision. Draft revisions do not count."
                ),
            },
            {
                "title": "stale exact packet",
                "text": f"A dashboard that included void and rehearsal events printed `{stale}`. The dashboard is stale.",
            },
            {
                "title": "nearby service packet",
                "text": (
                    f"{SERVICES[(i + 1) % len(SERVICES)]} / {OWNERS[(i + 1) % len(OWNERS)]} / "
                    f"{REGIONS[(i + 2) % len(REGIONS)]} / A={a} B={b + 1} C={c} / seal-111"
                ),
            },
        ),
    )


def _selection_task(i: int) -> CounterfactualLongContextTask:
    project = f"project-{chr(65 + i)}"
    candidates = []
    for j in range(5):
        stability = 4 + ((i + 2 * j) % 7)
        coverage = 5 + ((2 * i + j) % 6)
        risk = 1 + ((i + j) % 4)
        signed = (i + j) % 3 != 0
        retired = j == 4
        score = 3 * stability + 2 * coverage - 4 * risk + (5 if signed else 0)
        candidates.append(
            {
                "name": f"candidate {chr(65 + j)}",
                "stability": stability,
                "coverage": coverage,
                "risk": risk,
                "signed": signed,
                "retired": retired,
                "score": score,
                "lead": OWNERS[(i + j) % len(OWNERS)],
            }
        )
    active = [c for c in candidates if not c["retired"]]
    winner = max(active, key=lambda c: (c["score"], c["coverage"], c["name"]))
    plausible = max([c for c in active if c is not winner], key=lambda c: c["stability"])
    ticket = f"PLAN-{420 + i * 7}"
    answer = f"{project} / {winner['name']} / score={winner['score']} / {winner['lead']} / {ticket}"
    task_id = f"weighted-selection-{i + 1:02d}"
    return CounterfactualLongContextTask(
        task_id=task_id,
        corpus=f"counterfactual_selection_{i // 4:02d}",
        question=(
            f"Choose the valid mitigation for `{project}`. Return exactly "
            "`project / candidate / score=<value> / lead / ticket`. Return only that packet."
        ),
        must_contain=(answer,),
        must_not_contain=(plausible["name"], f"score={plausible['score']}", plausible["lead"], "candidate E"),
        documents=(
            {
                "title": "scoring rule",
                "text": (
                    "For non-retired candidates, score = 3*stability + 2*coverage - 4*risk + signed_bonus. "
                    "signed_bonus is 5 for signed candidates and 0 otherwise. Break ties by higher coverage, then candidate name."
                ),
            },
            *tuple(
                {
                    "title": candidate["name"],
                    "text": (
                        f"{'retired' if candidate['retired'] else 'active'} "
                        f"{'signed' if candidate['signed'] else 'unsigned'} {candidate['name']} for {project}: "
                        f"stability={candidate['stability']} coverage={candidate['coverage']} risk={candidate['risk']} "
                        f"lead={candidate['lead']}."
                    ),
                }
                for candidate in candidates
            ),
            {"title": "ticket ledger", "text": f"The selected mitigation for {project} uses approval ticket {ticket}."},
            {
                "title": "stale analyst packet",
                "text": (
                    f"A stale note chose {plausible['name']} because it had high stability and printed "
                    f"`{project} / {plausible['name']} / score={plausible['score']} / {plausible['lead']} / PLAN-000`."
                ),
            },
        ),
    )


def _timeline_task(i: int) -> CounterfactualLongContextTask:
    incident = f"IC-{70 + i}"
    owner = OWNERS[(i + 3) % len(OWNERS)]
    stale_owner = OWNERS[(i + 4) % len(OWNERS)]
    city = ("Lyon", "Oslo", "Porto", "Zurich", "Dublin", "Prague")[i % 6]
    stale_city = ("Milan", "Bergen", "Madrid", "Vienna", "Cork", "Warsaw")[i % 6]
    active_count = 3 + (i % 4)
    severity = "major" if active_count >= 5 else "standard"
    code = f"RL-{33 + i * 4}"
    answer = f"{incident} / {city} / {severity} / {owner} / {code}"
    stale = f"{incident} / {stale_city} / projected / {stale_owner} / RL-18"
    task_id = f"cutoff-timeline-{i + 1:02d}"
    return CounterfactualLongContextTask(
        task_id=task_id,
        corpus=f"counterfactual_timeline_{i // 4:02d}",
        question=(
            f"Apply the cutoff rules for `{incident}`. Return exactly "
            "`incident / location / severity / owner / release-code`. Return only that packet."
        ),
        must_contain=(answer,),
        must_not_contain=(stale_city, "projected", stale_owner, "RL-18"),
        documents=(
            {
                "title": "cutoff rules",
                "text": (
                    "Use active events at or before 2026-06-18T12:00Z. Ignore projected, void, and advisory events. "
                    "Severity is major if at least five active events remain after filtering; otherwise standard."
                ),
            },
            {
                "title": "events before cutoff",
                "text": (
                    f"2026-06-17T09:00Z active {incident} opened in {stale_city}. "
                    f"2026-06-17T18:30Z active {incident} routed toward {city}. "
                    f"2026-06-18T06:00Z active {incident} arrived in {city}. "
                    f"2026-06-18T08:15Z active {incident} assigned owner {owner}."
                ),
            },
            {
                "title": "more events",
                "text": (
                    f"2026-06-18T09:00Z {'active' if active_count >= 5 else 'advisory'} {incident} validation ping. "
                    f"2026-06-18T09:40Z {'active' if active_count >= 6 else 'void'} {incident} secondary scan. "
                    f"2026-06-18T13:40Z projected {incident} moved to {stale_city} with owner {stale_owner}."
                ),
            },
            {"title": "release ledger", "text": f"{incident} active release code at cutoff is {code}."},
            {"title": "stale packet", "text": f"After-cutoff dashboard printed `{stale}`. It includes projected events."},
            {"title": "nearby incident", "text": f"IC-{90 + i} / {city} / major / {stale_owner} / RL-18 is unrelated."},
        ),
    )


def _policy_task(i: int) -> CounterfactualLongContextTask:
    token = f"token-{chr(75 + i)}"
    ttl = 30 + 5 * (i % 5)
    rotate_hour = 7 + (i % 4)
    rotate_min = 15 if i % 2 else 45
    owner = OWNERS[(i + 6) % len(OWNERS)]
    stale_owner = OWNERS[(i + 1) % len(OWNERS)]
    waiver_ttl = ttl + 60
    stamp = f"POL-{700 + 9 * i}"
    answer = f"{token} / {ttl} minutes / {rotate_hour:02d}:{rotate_min:02d} UTC / {owner} / {stamp}"
    task_id = f"precedence-policy-{i + 1:02d}"
    return CounterfactualLongContextTask(
        task_id=task_id,
        corpus=f"counterfactual_policy_{i // 4:02d}",
        question=(
            f"Resolve the current signed policy for `{token}`. Return exactly "
            "`token / ttl / rotation-time / owner / stamp`. Return only that packet."
        ),
        must_contain=(answer,),
        must_not_contain=(f"{waiver_ttl} minutes", "06:00 UTC", stale_owner, "FAQ"),
        documents=(
            {
                "title": "precedence",
                "text": (
                    "Signed amendment beats policy memo. Policy memo beats FAQ. Emergency waiver beats both only while active. "
                    "Expired waivers and unsigned notes have no effect."
                ),
            },
            {"title": "faq", "text": f"FAQ says {token} TTL is {waiver_ttl} minutes and rotation time is 06:00 UTC."},
            {"title": "policy memo", "text": f"Signed policy memo says {token} owner is {stale_owner} and stamp is POL-101."},
            {"title": "ttl amendment", "text": f"Signed amendment says {token} TTL is {ttl} minutes."},
            {"title": "rotation amendment", "text": f"Signed amendment says {token} rotation time is {rotate_hour:02d}:{rotate_min:02d} UTC."},
            {"title": "owner amendment", "text": f"Signed amendment says {token} accountable owner is {owner}."},
            {"title": "stamp ledger", "text": f"The current policy stamp for {token} is {stamp}."},
            {"title": "expired waiver", "text": f"Emergency waiver set {token} TTL to {waiver_ttl} minutes but expired on 2026-05-31."},
            {"title": "unsigned note", "text": f"Unsigned note proposed owner {OWNERS[(i + 2) % len(OWNERS)]}; unsigned notes do not count."},
        ),
    )


def build_counterfactual_tasks(*, start: int = 0, groups: int = 6) -> tuple[CounterfactualLongContextTask, ...]:
    tasks: list[CounterfactualLongContextTask] = []
    for i in range(start, start + groups):
        tasks.append(_ledger_task(i))
        tasks.append(_selection_task(i))
        tasks.append(_timeline_task(i))
        tasks.append(_policy_task(i))
    return tuple(tasks)


TASKS = build_counterfactual_tasks()


def task_spec(task: CounterfactualLongContextTask) -> TaskSpec:
    group = f"{SOURCE_NAME}/{task.corpus}/{task.task_id}"
    return TaskSpec(
        task_id=f"{SOURCE_NAME}__{task.task_id}",
        capability="long_context",
        source=SourceRef(
            name=SOURCE_NAME,
            version=SOURCE_VERSION,
            policy="train_allowed",
            url_or_ref="generated://longctx_counterfactual",
        ),
        input=TaskInput(
            messages=[{"role": "user", "content": task.question}],
            context_documents=list(task.documents),
        ),
        environment=EnvironmentSpec(harness="long_context", wall_time_seconds=300),
        grader=GraderSpec(
            type="contains_all_absent",
            expected_answer={
                "must_contain": list(task.must_contain),
                "must_not_contain": list(task.must_not_contain),
            },
        ),
        splitting=SplittingSpec(
            group_id=task.corpus,
            split="grpo_train",
            contamination_group=group,
        ),
        metadata=TaskMetadata(
            domain="long_context",
            subdomain="counterfactual_state_verification",
            difficulty_estimate=0.95,
            tags=[
                "long-context",
                "generated",
                "counterfactual",
                "state-reconstruction",
                "cross-document-join",
                "stale-exact-answer-trap",
                "hard",
            ],
            requires_long_context=True,
            estimated_worker_calls=1,
        ),
    )


def materialize_long_context_counterfactual_tasks(
    *,
    out_jsonl: Path,
    report_out: Path | None = None,
    limit: int | None = None,
    offset_groups: int = 0,
    groups: int = 6,
) -> dict[str, Any]:
    task_pool = build_counterfactual_tasks(start=offset_groups, groups=groups)
    selected = list(task_pool[:limit] if limit is not None else task_pool)
    specs = [task_spec(task) for task in selected]
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for spec in specs:
            f.write(json.dumps(spec.model_dump(mode="json"), sort_keys=True) + "\n")

    report = {
        "version": "longctx_counterfactual_tasks_v1",
        "source": SOURCE_NAME,
        "task_count": len(specs),
        "offset_groups": offset_groups,
        "groups": groups,
        "out_jsonl": str(out_jsonl),
        "splits": sorted({spec.splitting.split for spec in specs}),
        "corpora": sorted({spec.splitting.group_id for spec in specs}),
        "live_calls": False,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
