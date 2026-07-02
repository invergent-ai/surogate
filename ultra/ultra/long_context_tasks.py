"""Generated train-allowed long-context document-pack tasks."""

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

SOURCE_NAME = "longctx_generated"
SOURCE_VERSION = "v1"


@dataclass(frozen=True)
class LongContextTask:
    task_id: str
    corpus: str
    question: str
    answer: str
    documents: list[dict[str, str]]


_SEED_TASKS: tuple[LongContextTask, ...] = (
    LongContextTask(
        task_id="incident-codename",
        corpus="ops_memos_alpha",
        question="Which incident codename was assigned after the second database failover?",
        answer="amber-lattice",
        documents=[
            {"title": "ops memo 01", "text": "The first failover used codename blue-harbor."},
            {"title": "ops memo 02", "text": "After the second database failover, the incident codename was amber-lattice."},
            {"title": "ops memo 03", "text": "A later cache flush was tracked as silver-cove."},
        ],
    ),
    LongContextTask(
        task_id="project-owner",
        corpus="roadmap_notes_beta",
        question="Who owns Project Kestrel according to the planning notes?",
        answer="Mira Chen",
        documents=[
            {"title": "roadmap intro", "text": "Project Heron is owned by Idris Patel."},
            {"title": "planning notes", "text": "For Project Kestrel, the accountable owner is Mira Chen."},
            {"title": "risk appendix", "text": "Kestrel risk review is scheduled after the billing migration."},
        ],
    ),
    LongContextTask(
        task_id="release-window",
        corpus="release_packets_gamma",
        question="What release window is approved for the API gateway migration?",
        answer="2026-08-14 02:00 UTC",
        documents=[
            {"title": "api gateway summary", "text": "The migration is approved for 2026-08-14 02:00 UTC."},
            {"title": "web client summary", "text": "The web client migration uses 2026-08-21 03:00 UTC."},
            {"title": "rollback plan", "text": "Rollback owners must be online for the API gateway window."},
        ],
    ),
    LongContextTask(
        task_id="vendor-escalation-owner",
        corpus="support_threads_delta",
        question="Who is listed as the escalation owner for the payment vendor timeout?",
        answer="Nadia Flores",
        documents=[
            {"title": "thread part 1", "text": "The search outage escalation owner is Tomas Ivers."},
            {"title": "thread part 2", "text": "Payment vendor timeout incidents should escalate to Nadia Flores."},
            {"title": "thread part 3", "text": "A separate refund queue delay is owned by Eli Moreno."},
        ],
    ),
    LongContextTask(
        task_id="archive-restore-code",
        corpus="backup_runbooks_epsilon",
        question="What restore code should be used for the archive bucket drill?",
        answer="restore-7184",
        documents=[
            {"title": "snapshot drill", "text": "Snapshot drills use restore code restore-1044."},
            {"title": "archive bucket drill", "text": "For the archive bucket drill, operators must use restore-7184."},
            {"title": "cold storage appendix", "text": "Cold storage restore tests are scheduled quarterly."},
        ],
    ),
    LongContextTask(
        task_id="contract-renewal-date",
        corpus="procurement_notes_zeta",
        question="What renewal date is recorded for the observability contract?",
        answer="2026-11-03",
        documents=[
            {"title": "compute contract", "text": "The compute contract renewal date is 2026-10-12."},
            {"title": "observability contract", "text": "Procurement recorded 2026-11-03 as the observability contract renewal date."},
            {"title": "storage contract", "text": "Storage contract renewal has not been approved."},
        ],
    ),
    LongContextTask(
        task_id="experiment-holdout-region",
        corpus="experiment_cards_eta",
        question="Which region is held out for the recommendation-ranking experiment?",
        answer="eu-central-1",
        documents=[
            {"title": "search ranking", "text": "Search ranking holds out us-west-2."},
            {"title": "recommendation ranking", "text": "The recommendation-ranking experiment holds out eu-central-1."},
            {"title": "checkout ranking", "text": "Checkout ranking runs in all regions."},
        ],
    ),
    LongContextTask(
        task_id="schema-migration-owner",
        corpus="database_notes_theta",
        question="Who owns the customer ledger schema migration?",
        answer="Ari Singh",
        documents=[
            {"title": "ledger migration overview", "text": "Ari Singh owns the customer ledger schema migration."},
            {"title": "inventory migration", "text": "Inventory schema migration is owned by Priya Raman."},
            {"title": "risk note", "text": "Ledger migration rollback requires a dual-write pause."},
        ],
    ),
    LongContextTask(
        task_id="incident-review-room",
        corpus="incident_reviews_iota",
        question="Which room is assigned for the cache saturation incident review?",
        answer="Room 4B",
        documents=[
            {"title": "database review", "text": "The database lock incident review is assigned to Room 2A."},
            {"title": "cache review", "text": "The cache saturation incident review is assigned to Room 4B."},
            {"title": "network review", "text": "Network packet loss review is remote-only."},
        ],
    ),
    LongContextTask(
        task_id="policy-exception-ticket",
        corpus="security_exceptions_kappa",
        question="What ticket ID tracks the temporary SSO bypass exception?",
        answer="SEC-4821",
        documents=[
            {"title": "vpn exception", "text": "Temporary VPN logging exception is tracked by SEC-4810."},
            {"title": "sso exception", "text": "The temporary SSO bypass exception is tracked by SEC-4821."},
            {"title": "mfa exception", "text": "The MFA migration exception was closed last quarter."},
        ],
    ),
    LongContextTask(
        task_id="customer-pilot-code",
        corpus="pilot_notes_lambda",
        question="What pilot code was assigned to the Northwind analytics customer?",
        answer="pilot-nw-73",
        documents=[
            {"title": "contoso pilot", "text": "Contoso analytics pilot uses code pilot-co-18."},
            {"title": "northwind pilot", "text": "Northwind analytics customer was assigned pilot code pilot-nw-73."},
            {"title": "fabrikam pilot", "text": "Fabrikam opted out of the analytics pilot."},
        ],
    ),
    LongContextTask(
        task_id="runbook-freeze-time",
        corpus="deployment_runbooks_mu",
        question="When does the checkout deployment freeze begin?",
        answer="2026-07-09 18:00 UTC",
        documents=[
            {"title": "search deployment", "text": "Search deployment freeze begins 2026-07-08 18:00 UTC."},
            {"title": "checkout deployment", "text": "Checkout deployment freeze begins 2026-07-09 18:00 UTC."},
            {"title": "email deployment", "text": "Email deployment has no freeze window."},
        ],
    ),
)


def _generated_tasks(count: int) -> list[LongContextTask]:
    projects = [
        "Atlas billing migration",
        "Beacon search rollout",
        "Cobalt warehouse sync",
        "Delta incident drill",
        "Ember policy review",
        "Fjord analytics pilot",
        "Granite schema update",
        "Helio security audit",
        "Iris capacity plan",
        "Juno vendor renewal",
    ]
    owners = [
        "Leah Ortiz",
        "Mason Reed",
        "Nora Patel",
        "Owen Blake",
        "Priya Shah",
        "Rafael Costa",
        "Sofia Nguyen",
        "Theo Martin",
    ]
    tasks: list[LongContextTask] = []
    for i in range(count):
        project = projects[i % len(projects)]
        owner = owners[i % len(owners)]
        code = f"ack-{4100 + i}"
        distractor_code = f"ack-{7100 + i}"
        corpus = f"generated_ops_pack_{i // 5:03d}"
        if i % 3 == 0:
            question = f"What acknowledgement code is recorded for {project}?"
            answer = code
            target_text = f"The {project} control note records acknowledgement code {code} after final review."
            distractor_text = f"A neighboring project used acknowledgement code {distractor_code}, but it is unrelated."
        elif i % 3 == 1:
            question = f"Who is the named owner for {project}?"
            answer = owner
            target_text = f"The owner field for {project} lists {owner} as accountable."
            distractor_text = f"The backup contact is {owners[(i + 3) % len(owners)]}, not the owner."
        else:
            answer = f"2026-{(i % 12) + 1:02d}-{(i % 25) + 1:02d} 09:00 UTC"
            question = f"When is the approved checkpoint for {project}?"
            target_text = f"The approved checkpoint for {project} is {answer}."
            distractor_text = f"The dry-run window for another effort is 2026-{((i + 4) % 12) + 1:02d}-15 09:00 UTC."
        tasks.append(
            LongContextTask(
                task_id=f"generated-docpack-{i + 1:03d}",
                corpus=corpus,
                question=question,
                answer=answer,
                documents=[
                    {"title": f"{project} overview", "text": f"{project} is tracked in the operations review packet."},
                    {"title": f"{project} control note", "text": target_text},
                    {"title": "nearby distractor", "text": distractor_text},
                    {
                        "title": "appendix",
                        "text": "Appendix entries contain unrelated review dates, backup contacts, and stale codes.",
                    },
                ],
            )
        )
    return tasks


_HARD_TASKS: tuple[LongContextTask, ...] = (
    LongContextTask(
        task_id="hard-security-exception-summary",
        corpus="hard_security_review_alpha",
        question=(
            "For the temporary SSO bypass, respond exactly as "
            "'ticket / approver / expiry date'."
        ),
        answer="SEC-4821 / Ren Ito / 2026-09-30",
        documents=[
            {"title": "exception register", "text": "The temporary SSO bypass exception is tracked by SEC-4821."},
            {"title": "approval memo", "text": "Ren Ito approved the temporary SSO bypass after risk review."},
            {"title": "expiry appendix", "text": "The SSO bypass exception expires on 2026-09-30 unless renewed."},
            {"title": "nearby vpn exception", "text": "Temporary VPN logging exception SEC-4810 was approved by Mara Bell."},
            {"title": "mfa migration note", "text": "MFA migration exception SEC-4799 expires on 2026-08-15."},
        ],
    ),
    LongContextTask(
        task_id="hard-release-rollback-summary",
        corpus="hard_release_packet_beta",
        question=(
            "For the API gateway migration, respond exactly as "
            "'window / rollback owner / rollback code'."
        ),
        answer="2026-08-14 02:00 UTC / Omar Velez / rbk-8842",
        documents=[
            {"title": "gateway schedule", "text": "The API gateway migration is approved for 2026-08-14 02:00 UTC."},
            {"title": "rollback staffing", "text": "Omar Velez is the rollback owner for the API gateway migration."},
            {"title": "rollback codes", "text": "Rollback code rbk-8842 belongs to the API gateway migration."},
            {"title": "web client schedule", "text": "The web client migration uses 2026-08-21 03:00 UTC and code rbk-2210."},
            {"title": "search migration", "text": "Search rollback owner is Priya Raman."},
        ],
    ),
    LongContextTask(
        task_id="hard-experiment-routing-summary",
        corpus="hard_experiment_cards_gamma",
        question=(
            "For the recommendation-ranking experiment, respond exactly as "
            "'holdout region / metric owner / launch gate'."
        ),
        answer="eu-central-1 / Sofia Nguyen / gate-rr-17",
        documents=[
            {"title": "experiment scope", "text": "The recommendation-ranking experiment holds out eu-central-1."},
            {"title": "metric ownership", "text": "Sofia Nguyen owns metrics for recommendation-ranking."},
            {"title": "launch gates", "text": "Recommendation-ranking cannot launch until gate-rr-17 is approved."},
            {"title": "search ranking", "text": "Search ranking holds out us-west-2 and uses gate-sr-09."},
            {"title": "checkout ranking", "text": "Checkout ranking metric owner is Theo Martin."},
        ],
    ),
    LongContextTask(
        task_id="hard-incident-action-summary",
        corpus="hard_incident_review_delta",
        question=(
            "For the cache saturation incident, respond exactly as "
            "'review room / action owner / due date'."
        ),
        answer="Room 4B / Nadia Flores / 2026-07-22",
        documents=[
            {"title": "review logistics", "text": "The cache saturation incident review is assigned to Room 4B."},
            {"title": "action register", "text": "Nadia Flores owns the cache saturation follow-up action."},
            {"title": "due dates", "text": "The cache saturation action is due on 2026-07-22."},
            {"title": "database incident", "text": "The database lock incident review is in Room 2A and due 2026-07-18."},
            {"title": "network incident", "text": "Network packet loss follow-up is owned by Tomas Ivers."},
        ],
    ),
    LongContextTask(
        task_id="hard-procurement-renewal-summary",
        corpus="hard_procurement_notes_epsilon",
        question=(
            "For the observability contract, respond exactly as "
            "'renewal date / procurement owner / approval code'."
        ),
        answer="2026-11-03 / Leah Ortiz / appr-6390",
        documents=[
            {"title": "contract dates", "text": "Procurement recorded 2026-11-03 as the observability contract renewal date."},
            {"title": "owner register", "text": "Leah Ortiz owns procurement follow-up for the observability contract."},
            {"title": "approval ledger", "text": "Approval code appr-6390 is attached to the observability renewal."},
            {"title": "compute contract", "text": "The compute contract renewal date is 2026-10-12 with approval appr-4220."},
            {"title": "storage contract", "text": "Storage procurement owner is Mason Reed."},
        ],
    ),
    LongContextTask(
        task_id="hard-customer-pilot-summary",
        corpus="hard_pilot_notes_zeta",
        question=(
            "For the Northwind analytics customer, respond exactly as "
            "'pilot code / success metric / escalation owner'."
        ),
        answer="pilot-nw-73 / weekly active dashboards / Eli Moreno",
        documents=[
            {"title": "northwind pilot", "text": "Northwind analytics customer was assigned pilot code pilot-nw-73."},
            {"title": "pilot metrics", "text": "The Northwind success metric is weekly active dashboards."},
            {"title": "support routing", "text": "Eli Moreno is the escalation owner for Northwind analytics."},
            {"title": "contoso pilot", "text": "Contoso analytics pilot uses code pilot-co-18 and metric report exports."},
            {"title": "fabrikam pilot", "text": "Fabrikam escalation owner is Nadia Flores."},
        ],
    ),
)


TASKS: tuple[LongContextTask, ...] = (*_SEED_TASKS, *_generated_tasks(125 - len(_SEED_TASKS)), *_HARD_TASKS)


def task_spec(task: LongContextTask) -> TaskSpec:
    group = f"{SOURCE_NAME}/{task.corpus}/{task.task_id}"
    return TaskSpec(
        task_id=f"{SOURCE_NAME}__{task.task_id}",
        capability="long_context",
        source=SourceRef(
            name=SOURCE_NAME,
            version=SOURCE_VERSION,
            policy="train_allowed",
            url_or_ref="generated://longctx",
        ),
        input=TaskInput(
            messages=[{"role": "user", "content": task.question}],
            context_documents=task.documents,
        ),
        environment=EnvironmentSpec(harness="long_context", wall_time_seconds=180),
        grader=GraderSpec(type="contains", expected_answer=task.answer),
        splitting=SplittingSpec(
            group_id=task.corpus,
            split="grpo_train",
            contamination_group=group,
        ),
        metadata=TaskMetadata(
            domain="long_context",
            subdomain="document_pack",
            tags=[
                "long-context",
                "generated",
                "document-pack",
                *(["hard", "multi_hop"] if task.task_id.startswith("hard-") else []),
            ],
            requires_long_context=True,
            estimated_worker_calls=1,
        ),
    )


def materialize_long_context_tasks(
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
        "version": "longctx_generated_tasks_v1",
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
