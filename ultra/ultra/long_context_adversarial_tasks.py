"""Adversarial train-allowed long-context tasks.

These are synthetic but intentionally not simple needle lookup. Each task asks the
worker to apply a small rule set over conflicting revisions, voided records, and
nearby distractors, then return only the current final fields.
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

SOURCE_NAME = "longctx_adversarial"
SOURCE_VERSION = "v1"


@dataclass(frozen=True)
class AdversarialLongContextTask:
    task_id: str
    corpus: str
    question: str
    must_contain: tuple[str, ...]
    must_not_contain: tuple[str, ...]
    documents: tuple[dict[str, str], ...]


TASKS: tuple[AdversarialLongContextTask, ...] = (
    AdversarialLongContextTask(
        task_id="state-reconstruction-access-ledger",
        corpus="adversarial_access_ledger_alpha",
        question=(
            "Apply the ledger rules and return the final vault-7 access packet exactly as "
            "`principals / owner / rotation code`. Return only that packet."
        ),
        must_contain=("Mira Chen, Theo Martin / Nadia Flores / rot-7319",),
        must_not_contain=("Owen Blake", "rot-1190", "Leah Ortiz", "vault-4"),
        documents=(
            {"title": "rules", "text": "For a vault, start from seed grants, apply active events in sequence order, ignore events marked void, drill, or superseded, and use the highest active owner record."},
            {"title": "seed grants", "text": "vault-7 initial principals: Mira Chen, Owen Blake. vault-4 initial principals: Leah Ortiz."},
            {"title": "event 014", "text": "active vault-7 revoke principal Owen Blake."},
            {"title": "event 015", "text": "void vault-7 grant principal Owen Blake; the approval checksum failed."},
            {"title": "event 016", "text": "active vault-7 grant principal Theo Martin."},
            {"title": "event 017", "text": "drill vault-7 grant principal Priya Shah; ignore drill events."},
            {"title": "owner record 1", "text": "active owner record for vault-7 at revision 3: Leah Ortiz."},
            {"title": "owner record 2", "text": "superseded owner record for vault-7 at revision 4: Rafael Costa."},
            {"title": "owner record 3", "text": "active owner record for vault-7 at revision 5: Nadia Flores."},
            {"title": "rotation record old", "text": "vault-7 rotation code rot-1190 was replaced during the June incident."},
            {"title": "rotation record current", "text": "active vault-7 rotation code is rot-7319."},
            {"title": "nearby vault", "text": "vault-4 owner is Leah Ortiz and rotation code is rot-4402."},
        ),
    ),
    AdversarialLongContextTask(
        task_id="revision-precedence-release-window",
        corpus="adversarial_release_window_beta",
        question=(
            "Using only active signed records, return the final Beacon rollout packet exactly as "
            "`window / rollback owner / gate`. Return only that packet."
        ),
        must_contain=("2026-10-04 03:30 UTC / Omar Velez / gate-bc-42",),
        must_not_contain=("2026-09-27 02:00 UTC", "Mason Reed", "gate-bc-17", "draft"),
        documents=(
            {"title": "precedence", "text": "Signed records beat unsigned notes. If two signed records conflict, use the later revision number. Ignore draft, rehearsal, and canceled records."},
            {"title": "beacon schedule r1", "text": "signed revision 1: Beacon rollout window 2026-09-27 02:00 UTC."},
            {"title": "beacon schedule r2", "text": "draft revision 2: Beacon rollout window 2026-10-01 01:00 UTC."},
            {"title": "beacon schedule r3", "text": "signed revision 3: Beacon rollout window 2026-10-04 03:30 UTC."},
            {"title": "canceled note", "text": "canceled signed note suggested 2026-10-05 04:00 UTC; cancellation was countersigned."},
            {"title": "rollback owner old", "text": "signed revision 1: Beacon rollback owner Mason Reed."},
            {"title": "rollback owner current", "text": "signed revision 4: Beacon rollback owner Omar Velez."},
            {"title": "gate r1", "text": "signed revision 1: Beacon launch gate gate-bc-17."},
            {"title": "gate r5", "text": "signed revision 5: Beacon launch gate gate-bc-42."},
            {"title": "rehearsal", "text": "rehearsal packet uses owner Priya Shah and gate rehearsal-9; rehearsal packets do not apply."},
            {"title": "atlas rollout", "text": "Atlas rollout is 2026-10-04 03:30 UTC but uses gate-at-08."},
        ),
    ),
    AdversarialLongContextTask(
        task_id="cutoff-state-cargo-route",
        corpus="adversarial_cargo_route_gamma",
        question=(
            "Apply the cutoff and status rules. Return the final container B7 packet exactly as "
            "`location / hold reason / release code`. Return only that packet."
        ),
        must_contain=("Lyon / customs mismatch / RL-42",),
        must_not_contain=("Milan", "RL-18", "temperature audit", "B9"),
        documents=(
            {"title": "rules", "text": "Use only active events with timestamp at or before 2026-06-18T12:00Z. Later events are projections. A hold event overrides transit until a release event at or before cutoff."},
            {"title": "B7 event 01", "text": "2026-06-17T09:00Z active container B7 arrived in Milan."},
            {"title": "B7 event 02", "text": "2026-06-17T18:30Z active container B7 departed Milan for Lyon."},
            {"title": "B7 event 03", "text": "2026-06-18T06:00Z active container B7 arrived in Lyon."},
            {"title": "B7 event 04", "text": "2026-06-18T08:15Z active container B7 placed on hold for customs mismatch."},
            {"title": "B7 event 05", "text": "2026-06-18T13:40Z projected container B7 release code RL-18; ignore projections after cutoff."},
            {"title": "release ledger", "text": "For a customs mismatch hold at Lyon, the active release code is RL-42."},
            {"title": "B9 event", "text": "Container B9 is held in Lyon for temperature audit and uses release code RL-18."},
            {"title": "voided scan", "text": "void container B7 scan says location Marseille; device clock failed."},
            {"title": "operator note", "text": "The B7 hold remains open at cutoff because no active release event exists before 12:00Z."},
        ),
    ),
    AdversarialLongContextTask(
        task_id="policy-precedence-token-controls",
        corpus="adversarial_policy_precedence_delta",
        question=(
            "Resolve the policy precedence chain for the mobile admin token. Return exactly "
            "`ttl / rotation time / accountable owner`. Return only that packet."
        ),
        must_contain=("45 minutes / 09:30 UTC / Leah Ortiz",),
        must_not_contain=("90 minutes", "08:00 UTC", "Rafael Costa", "FAQ"),
        documents=(
            {"title": "precedence rules", "text": "Signed amendment beats policy memo. Policy memo beats FAQ. Emergency waiver beats both only while active. Expired waivers have no effect."},
            {"title": "faq", "text": "FAQ says mobile admin token TTL is 90 minutes and rotation time is 08:00 UTC."},
            {"title": "policy memo", "text": "Signed policy memo says mobile admin token TTL is 60 minutes and accountable owner is Rafael Costa."},
            {"title": "amendment A", "text": "Signed amendment A says mobile admin token TTL is 45 minutes."},
            {"title": "amendment B", "text": "Signed amendment B says mobile admin token rotation time is 09:30 UTC."},
            {"title": "owner amendment", "text": "Signed owner amendment says Leah Ortiz is accountable owner for mobile admin token controls."},
            {"title": "emergency waiver", "text": "Emergency waiver changed TTL to 120 minutes but expired on 2026-05-31."},
            {"title": "desktop token", "text": "Desktop admin token owner is Rafael Costa and rotates at 08:00 UTC."},
            {"title": "unsigned note", "text": "Unsigned note asks whether rotation should move to 11:00 UTC; unsigned notes do not change policy."},
        ),
    ),
    AdversarialLongContextTask(
        task_id="quorum-approval-incident-action",
        corpus="adversarial_incident_quorum_epsilon",
        question=(
            "Apply quorum rules for incident IC-44 and return exactly "
            "`decision / action owner / approval ticket`. Return only that packet."
        ),
        must_contain=("approved / Priya Shah / SEC-9120",),
        must_not_contain=("blocked", "Tomas Ivers", "SEC-9044", "IC-43"),
        documents=(
            {"title": "quorum rules", "text": "An incident action is approved if it has at least two active security approvals and one active operations approval. Withdrawn approvals do not count."},
            {"title": "IC-44 proposal", "text": "Incident IC-44 proposed action owner Priya Shah."},
            {"title": "IC-44 security approval 1", "text": "active security approval for IC-44 by Mira Chen, ticket SEC-9120."},
            {"title": "IC-44 security approval 2", "text": "active security approval for IC-44 by Omar Velez, ticket SEC-9120."},
            {"title": "IC-44 security approval 3", "text": "withdrawn security approval for IC-44 by Tomas Ivers, ticket SEC-9044."},
            {"title": "IC-44 operations approval", "text": "active operations approval for IC-44 by Leah Ortiz."},
            {"title": "IC-43 proposal", "text": "Incident IC-43 owner Tomas Ivers is blocked pending more approvals."},
            {"title": "IC-44 meeting note", "text": "The meeting note says blocked, but meeting notes are advisory and do not override the quorum ledger."},
            {"title": "ticket appendix", "text": "Use the ticket shared by the active security approvals as the approval ticket."},
        ),
    ),
    AdversarialLongContextTask(
        task_id="constraint-satisfaction-maintenance-slot",
        corpus="adversarial_maintenance_slot_zeta",
        question=(
            "Choose the only maintenance slot satisfying all constraints. Return exactly "
            "`slot / shard / incident lead`. Return only that packet."
        ),
        must_contain=("2026-09-18 03:30 UTC / shard C / Sofia Nguyen",),
        must_not_contain=("2026-09-17 03:30 UTC", "shard B", "Owen Blake", "freeze"),
        documents=(
            {"title": "constraints", "text": "The slot must be outside the freeze, after dependency D-19, before audit A-7, and staffed by an active incident lead for the selected shard."},
            {"title": "freeze window", "text": "Freeze runs from 2026-09-16 00:00 UTC through 2026-09-17 23:59 UTC."},
            {"title": "dependency D-19", "text": "Dependency D-19 completes on 2026-09-18 02:00 UTC."},
            {"title": "audit A-7", "text": "Audit A-7 begins on 2026-09-18 06:00 UTC."},
            {"title": "candidate 1", "text": "Candidate slot 2026-09-17 03:30 UTC for shard B, lead Owen Blake."},
            {"title": "candidate 2", "text": "Candidate slot 2026-09-18 03:30 UTC for shard C, lead Sofia Nguyen."},
            {"title": "candidate 3", "text": "Candidate slot 2026-09-18 07:00 UTC for shard C, lead Sofia Nguyen."},
            {"title": "lead roster", "text": "Sofia Nguyen is active incident lead for shard C. Owen Blake is standby for shard B but not active this week."},
            {"title": "shard appendix", "text": "Shard C must be maintained before audit A-7; shard B was deferred."},
        ),
    ),
)


def task_spec(task: AdversarialLongContextTask) -> TaskSpec:
    group = f"{SOURCE_NAME}/{task.corpus}/{task.task_id}"
    return TaskSpec(
        task_id=f"{SOURCE_NAME}__{task.task_id}",
        capability="long_context",
        source=SourceRef(
            name=SOURCE_NAME,
            version=SOURCE_VERSION,
            policy="train_allowed",
            url_or_ref="generated://longctx_adversarial",
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
            subdomain="adversarial_state_reconstruction",
            difficulty_estimate=0.85,
            tags=[
                "long-context",
                "generated",
                "adversarial",
                "state-reconstruction",
                "conflicting-evidence",
                "hard",
            ],
            requires_long_context=True,
            estimated_worker_calls=1,
        ),
    )


def materialize_long_context_adversarial_tasks(
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
        "version": "longctx_adversarial_tasks_v1",
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
