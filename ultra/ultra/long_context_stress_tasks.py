"""Stress-style train-allowed long-context tasks.

These tasks require state reconstruction, arithmetic, or constraint scoring over
conflicting document fragments. They are meant as a harder source for discovery
when ordinary document-pack extraction is saturated.
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

SOURCE_NAME = "longctx_stress"
SOURCE_VERSION = "v1"


@dataclass(frozen=True)
class StressLongContextTask:
    task_id: str
    corpus: str
    question: str
    must_contain: tuple[str, ...]
    must_not_contain: tuple[str, ...]
    documents: tuple[dict[str, str], ...]


TASKS: tuple[StressLongContextTask, ...] = (
    StressLongContextTask(
        task_id="event-ledger-checksum",
        corpus="stress_event_ledger_alpha",
        question=(
            "Reconstruct the active ledger state. Return exactly "
            "`A=<value> / B=<value> / C=<value> / checksum=<value>`. Return only that line."
        ),
        must_contain=("A=49 / B=24 / C=25 / checksum=932",),
        must_not_contain=("A=68", "B=74", "C=125", "checksum=193"),
        documents=(
            {"title": "ledger rules", "text": "Initial state: A=18, B=27, C=9. Process events in ascending sequence. Ignore status=void and status=rehearsal. checksum=(7*A + 11*B + 13*C) mod 1000."},
            {"title": "events 001-004", "text": "seq=001 status=active target=A op=add value=5\nseq=002 status=active target=B op=subtract value=4\nseq=003 status=active target=C op=double\nseq=004 status=active target=A op=add value=11"},
            {"title": "events 005-008", "text": "seq=005 status=void target=B op=add value=50\nseq=006 status=active target=C op=subtract value=7\nseq=007 status=active target=B op=add value=9\nseq=008 status=active target=A op=double"},
            {"title": "events 009-012", "text": "seq=009 status=rehearsal target=C op=add value=100\nseq=010 status=active target=B op=subtract value=8\nseq=011 status=active target=C op=add value=14\nseq=012 status=active target=A op=subtract value=19"},
            {"title": "distractor ledger", "text": "A stale analyst note included void and rehearsal events and got A=68, B=74, C=125."},
            {"title": "format note", "text": "The final answer should include current A, B, C, and checksum only."},
        ),
    ),
    StressLongContextTask(
        task_id="candidate-score-selection",
        corpus="stress_candidate_scoring_beta",
        question=(
            "Apply the scoring formula and return the winning mitigation exactly as "
            "`candidate / score / lead`. Return only that packet."
        ),
        must_contain=("candidate C", "34", "Nadia Flores"),
        must_not_contain=("candidate A", "score 32", "Omar Velez", "candidate D"),
        documents=(
            {"title": "scoring rule", "text": "For active candidates, score = 3*stability + 2*coverage - 5*risk + signed_bonus. signed_bonus is 4 for signed candidates and 0 otherwise. Ignore retired candidates."},
            {"title": "candidate A", "text": "active signed candidate A: stability=8 coverage=7 risk=2 lead=Omar Velez."},
            {"title": "candidate B", "text": "active signed candidate B: stability=9 coverage=6 risk=3 lead=Mira Chen."},
            {"title": "candidate C", "text": "active unsigned candidate C: stability=7 coverage=9 risk=1 lead=Nadia Flores."},
            {"title": "candidate D", "text": "active signed candidate D: stability=6 coverage=10 risk=2 lead=Sofia Nguyen."},
            {"title": "candidate E", "text": "active signed candidate E: stability=10 coverage=5 risk=4 lead=Theo Martin."},
            {"title": "retired candidate", "text": "retired signed candidate R: stability=10 coverage=10 risk=0 lead=Leah Ortiz; retired candidates do not count."},
            {"title": "analyst note", "text": "A prior analyst preferred candidate A because it was signed; this note did not apply the full formula."},
        ),
    ),
    StressLongContextTask(
        task_id="invoice-net-reconciliation",
        corpus="stress_invoice_reconciliation_gamma",
        question=(
            "Reconcile Kestrel invoices through the cutoff. Return exactly "
            "`vendor / net amount / approval ticket`. Return only that packet."
        ),
        must_contain=("Kestrel / 742 / AP-77",),
        must_not_contain=("812", "Northwind", "AP-31", "refund pending"),
        documents=(
            {"title": "reconciliation rules", "text": "Use only Kestrel entries dated on or before 2026-06-30. Add approved charges, subtract approved credits, ignore void entries, pending refunds, and other vendors."},
            {"title": "Kestrel charges 1", "text": "2026-06-02 Kestrel approved charge 410. 2026-06-11 Kestrel approved charge 205."},
            {"title": "Kestrel credits", "text": "2026-06-14 Kestrel approved credit 88. 2026-06-18 Kestrel pending refund 70."},
            {"title": "Kestrel charges 2", "text": "2026-06-20 Kestrel approved charge 215. 2026-06-21 Kestrel void charge 120."},
            {"title": "after cutoff", "text": "2026-07-02 Kestrel approved charge 90. Entries after cutoff do not count."},
            {"title": "other vendor", "text": "2026-06-21 Northwind approved charge 812 with ticket AP-31."},
            {"title": "approval ledger", "text": "Kestrel June reconciliation approval ticket is AP-77."},
        ),
    ),
    StressLongContextTask(
        task_id="token-bucket-simulation",
        corpus="stress_token_bucket_delta",
        question=(
            "Simulate the bucket events and return exactly "
            "`final tokens / dropped requests / throttle mode`. Return only that packet."
        ),
        must_contain=("9 / 4 / soft",),
        must_not_contain=("22", "15 / hard", "maintenance", "training"),
        documents=(
            {"title": "bucket rules", "text": "Start with 20 tokens. Active consume events subtract requested tokens if available; if not, increment dropped requests by the token shortfall and leave tokens unchanged. Active refill events add tokens but cap at 25. Ignore training and maintenance events."},
            {"title": "mode rule", "text": "Throttle mode is hard if dropped requests are 6 or more, otherwise soft."},
            {"title": "events 1", "text": "seq=1 active consume 8\nseq=2 active consume 7\nseq=3 active refill 6"},
            {"title": "events 2", "text": "seq=4 training consume 9\nseq=5 active consume 15\nseq=6 active refill 4"},
            {"title": "events 3", "text": "seq=7 active consume 5\nseq=8 maintenance refill 10\nseq=9 active refill 12"},
            {"title": "events 4", "text": "seq=10 active consume 8\nseq=11 active consume 5"},
            {"title": "operator note", "text": "A stale monitor counted training and maintenance events, producing a hard throttle. Ignore that monitor."},
        ),
    ),
    StressLongContextTask(
        task_id="quorum-weighted-approval",
        corpus="stress_quorum_weighted_epsilon",
        question=(
            "Compute the active weighted approval for change CH-19. Return exactly "
            "`decision / weight / ticket`. Return only that packet."
        ),
        must_contain=("approved / 11 / CHG-884",),
        must_not_contain=("blocked", "9", "CHG-441", "withdrawn"),
        documents=(
            {"title": "approval rule", "text": "A change is approved if active weighted approval is at least 10. Security signatures weigh 4, operations weigh 3, finance weighs 2. Withdrawn, duplicate, and advisory signatures do not count."},
            {"title": "CH-19 signatures A", "text": "active security signature by Mira Chen ticket CHG-884. active operations signature by Leah Ortiz ticket CHG-884."},
            {"title": "CH-19 signatures B", "text": "withdrawn security signature by Tomas Ivers ticket CHG-441. active finance signature by Owen Blake ticket CHG-884."},
            {"title": "CH-19 signatures C", "text": "active finance signature by Priya Shah ticket CHG-884. duplicate finance signature by Priya Shah should not count twice."},
            {"title": "advisory note", "text": "Advisory security note said blocked at weight 9, before finance signatures arrived."},
            {"title": "CH-18 appendix", "text": "CH-18 was blocked under ticket CHG-441."},
        ),
    ),
    StressLongContextTask(
        task_id="alias-chain-current-record",
        corpus="stress_alias_resolution_zeta",
        question=(
            "Resolve the active alias chain for `crane` and return exactly "
            "`canonical service / owner / region / seal`. Return only that packet."
        ),
        must_contain=("Orion / Rafael Costa / eu-west-3 / seal-409",),
        must_not_contain=("Lynx", "Mason Reed", "us-east-1", "seal-118"),
        documents=(
            {"title": "alias rules", "text": "Follow active alias records until a canonical service is reached. If records conflict, use the highest active revision. Ignore retired and draft aliases."},
            {"title": "alias crane r1", "text": "active revision 1 alias crane -> Lynx."},
            {"title": "alias crane r2", "text": "draft revision 2 alias crane -> Heron; draft aliases do not count."},
            {"title": "alias crane r3", "text": "active revision 3 alias crane -> Orion."},
            {"title": "canonical Orion r4", "text": "active revision 4 canonical service Orion owner Rafael Costa region eu-west-3 seal seal-409."},
            {"title": "canonical Lynx r2", "text": "active revision 2 canonical service Lynx owner Mason Reed region us-east-1 seal seal-118."},
            {"title": "retired Orion r1", "text": "retired canonical service Orion owner Leah Ortiz region eu-central-1 seal seal-204."},
            {"title": "operator note", "text": "A stale dashboard still maps crane to Lynx. The dashboard is not authoritative."},
        ),
    ),
)


def task_spec(task: StressLongContextTask) -> TaskSpec:
    group = f"{SOURCE_NAME}/{task.corpus}/{task.task_id}"
    return TaskSpec(
        task_id=f"{SOURCE_NAME}__{task.task_id}",
        capability="long_context",
        source=SourceRef(
            name=SOURCE_NAME,
            version=SOURCE_VERSION,
            policy="train_allowed",
            url_or_ref="generated://longctx_stress",
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
            subdomain="stress_state_reconstruction",
            difficulty_estimate=0.92,
            tags=[
                "long-context",
                "generated",
                "stress",
                "state-reconstruction",
                "arithmetic",
                "conflicting-evidence",
                "hard",
            ],
            requires_long_context=True,
            estimated_worker_calls=1,
        ),
    )


def materialize_long_context_stress_tasks(
    *,
    out_jsonl: Path,
    report_out: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    selected = list(TASKS[:limit] if limit is not None else TASKS)
    specs = [task_spec(task) for task in selected]
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for spec in specs:
            f.write(json.dumps(spec.model_dump(mode="json"), sort_keys=True) + "\n")

    report = {
        "version": "longctx_stress_tasks_v1",
        "source": SOURCE_NAME,
        "task_count": len(specs),
        "out_jsonl": str(out_jsonl),
        "splits": sorted({spec.splitting.split for spec in specs}),
        "corpora": sorted({spec.splitting.group_id for spec in specs}),
        "live_calls": False,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
