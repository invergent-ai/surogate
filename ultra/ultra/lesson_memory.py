#!/usr/bin/env python3
"""Lesson memory + retrieval for the conductor (CORE-style).

Product component: regime- and situation-conditioned guidance injected into
the conductor decision prompt via ``guidelines_provider``.

Implements the two pieces CORE relies on at solve time:

  utility()  Beta-smoothed success estimate with asymmetric priors, so a
             lesson distilled from a winning decision starts mildly optimistic
             and one from a loss starts pessimistic.
  retrieve() ranks by cosine(state, lesson_origin) * utility(), CORE's rule
             combining contextual relevance with measured effectiveness.

Deliberate deviation: CORE embeds with a neural encoder. The conductor decides
inside a 7,680-token budget on a local service, and adding an embedding model
to the serving path buys little here — the retrieval key is a small, closed set
of workflow-state features (ownership phase, verification status, artifact
readiness, budget pressure), not free text. So similarity is computed over that
structured feature vector, which is cheaper, deterministic, and auditable.
Swapping in a text embedder later only changes `state_features`/`similarity`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Motifs are the curriculum's names for decision situations; they double as
# coarse state features so a lesson is retrieved for the situation it came from.
_MOTIF_FEATURES: dict[str, tuple[str, ...]] = {
    "unverified_completion_pending_verifier": (
        "owner_finished",
        "unverified_artifact",
        "verifier_pending",
    ),
    "failed_verification_repair_and_reverify": (
        "verification_failed",
        "defective_artifact",
        "repair_pending",
    ),
    "candidate_finisher_before_verifier": (
        "owner_finished",
        "candidate_artifact",
        "finisher_pending",
        "verifier_pending",
    ),
    "low_budget_deliverable_first": ("budget_low", "artifact_absent"),
    "stalled_owner_near_exhaustion": ("owner_stalled", "budget_low"),
    "active_private_loop_continue_before_handoff": (
        "owner_active",
        "recoverable_private_work",
    ),
}


@dataclass
class Lesson:
    """One natural-language decision rule with CORE's utility statistics."""

    lesson_id: str
    text: str
    label: str  # "specific" | "meta"
    motif: str
    features: frozenset[str]
    from_correct: bool = True
    reflection_type: str = "contrast"
    uses: int = 0
    wins: int = 0
    support: int = 1
    origins: list[str] = field(default_factory=list)

    def utility(self) -> float:
        """Beta-smoothed success rate with CORE's asymmetric priors."""
        if self.from_correct:
            alpha, beta = 2.0, 1.0
        else:
            alpha, beta = 1.0, 2.0
        return (self.wins + alpha) / (self.uses + alpha + beta)

    def to_json(self) -> dict[str, Any]:
        row = {
            "lesson_id": self.lesson_id,
            "text": self.text,
            "label": self.label,
            "motif": self.motif,
            "features": sorted(self.features),
            "from_correct": self.from_correct,
            "reflection_type": self.reflection_type,
            "uses": self.uses,
            "wins": self.wins,
            "support": self.support,
            "origins": self.origins,
        }
        return row

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "Lesson":
        return cls(
            lesson_id=row["lesson_id"],
            text=row["text"],
            label=row["label"],
            motif=row["motif"],
            features=frozenset(row.get("features", ())),
            from_correct=bool(row.get("from_correct", True)),
            reflection_type=row.get("reflection_type", "contrast"),
            uses=int(row.get("uses", 0)),
            wins=int(row.get("wins", 0)),
            support=int(row.get("support", 1)),
            origins=list(row.get("origins", ())),
        )


def state_features(state: Any) -> frozenset[str]:
    """Derive the retrieval key from a live control state.

    Mirrors the motif features above so lessons learned on synthetic
    branchpoints retrieve at the matching real workflow situation.
    """
    features: set[str] = set()
    budget = getattr(state, "budget", None)
    if budget is not None and getattr(budget, "paid_calls_remaining", 99) <= 3:
        features.add("budget_low")
    # Time regime: the single feature that separates Terminal-Bench-style
    # tasks (900-1800s, pass/fail, throughput-bound) from ALE-style tasks
    # (14,400s, partial credit, quality-bound). Strategy lessons tagged with
    # one of these retrieve only in their own regime.
    if budget is not None:
        limit = getattr(budget, "wall_time_limit_s", None)
        if limit is not None and limit > 0:
            features.add("short_horizon" if limit <= 2400 else "long_horizon")

    active = getattr(state, "active_position", None)
    if active is not None:
        progress = getattr(active, "progress", None) or {}
        if progress.get("completion_requested"):
            features.add("owner_finished")
        else:
            features.add("owner_active")
        if getattr(active, "artifacts", None):
            features.add("candidate_artifact")
            features.add("unverified_artifact")
        else:
            features.add("artifact_absent")

    for position in getattr(state, "positions", ()) or ():
        if getattr(position, "status", None) != "pending":
            continue
        tags = set(getattr(position, "capability_tags", ()) or ())
        subtask = str(getattr(position, "subtask", "") or "").casefold()
        if tags & {"verifier", "reviewer", "auditor"} or "verif" in subtask:
            features.add("verifier_pending")
        if tags & {"implementer", "coder", "drafter"} or "finish" in subtask:
            features.add("finisher_pending")
        if tags & {"debugger", "failure_analyst"} or "repair" in subtask:
            features.add("repair_pending")
    return frozenset(features)


# Strategy lessons come from whole-task contrasts rather than one branchpoint,
# so they carry no motif features. They stay retrievable at a modest fixed
# relevance: high enough to surface, low enough that a precisely matched
# situation-specific lesson outranks them.
WILDCARD_MOTIF = "whole_task_strategy"
WILDCARD_RELEVANCE = 0.30


def similarity(state: frozenset[str], lesson: Lesson) -> float:
    """Cosine similarity over the structured feature sets."""
    if lesson.motif == WILDCARD_MOTIF and lesson.features:
        # Regime-gated strategy lesson: fires at wildcard strength in its own
        # regime, never outside it.
        return WILDCARD_RELEVANCE if lesson.features & state else 0.0
    if lesson.motif == WILDCARD_MOTIF and not lesson.features:
        return WILDCARD_RELEVANCE
    if not state or not lesson.features:
        return 0.0
    overlap = len(state & lesson.features)
    if overlap == 0:
        return 0.0
    return overlap / math.sqrt(len(state) * len(lesson.features))


class LessonMemory:
    def __init__(self, lessons: list[Lesson]) -> None:
        self.lessons = lessons

    @classmethod
    def load(cls, path: Path) -> "LessonMemory":
        raw = json.loads(path.read_text())
        return cls([Lesson.from_json(row) for row in raw["lessons"]])

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {"lesson_count": len(self.lessons),
                 "lessons": [lesson.to_json() for lesson in self.lessons]},
                indent=2,
            )
        )

    def retrieve(
        self,
        state: Any,
        *,
        top_k: int = 4,
        strategy_k: int = 2,
        floor: float = 0.15,
    ) -> list[Lesson]:
        """Rank by relevance * utility (CORE's rule), over two reserved pools.

        Situation-specific lessons and whole-task strategy lessons answer
        different questions ("what should I decide here?" versus "how should
        effort be shaped across roles?"), so they are selected separately.
        Otherwise precisely matched decision lessons always crowd out strategy
        guidance, which is exactly the guidance that long-horizon failures
        (a planning role monopolising execution) need.
        """
        features = state if isinstance(state, frozenset) else state_features(state)
        specific: list[tuple[float, Lesson]] = []
        strategy: list[tuple[float, Lesson]] = []
        for lesson in self.lessons:
            relevance = similarity(features, lesson)
            if relevance < floor:
                continue
            bucket = strategy if lesson.motif == WILDCARD_MOTIF else specific
            bucket.append((relevance * lesson.utility(), lesson))
        specific.sort(key=lambda item: item[0], reverse=True)
        strategy.sort(key=lambda item: item[0], reverse=True)
        chosen = [lesson for _, lesson in specific[: max(0, top_k - strategy_k)]]
        chosen += [lesson for _, lesson in strategy[:strategy_k]]
        return chosen

    def render(self, lessons: list[Lesson]) -> str:
        if not lessons:
            return ""
        bullets = "\n".join(f"- {lesson.text}" for lesson in lessons)
        return (
            "LEARNED DECISION GUIDELINES (from prior verified outcomes at "
            "similar workflow states):\n" + bullets
        )


def build_from_reflection(path: Path) -> LessonMemory:
    """Build a deduplicated memory from generated contrast lessons."""
    raw = json.loads(path.read_text())
    by_text: dict[str, Lesson] = {}
    for index, row in enumerate(raw["lessons"]):
        text = row["lesson"].strip()
        key = text.casefold()
        motif = row.get("motif", "")
        features = frozenset(_MOTIF_FEATURES.get(motif, ()))
        existing = by_text.get(key)
        if existing is not None:
            existing.support += 1
            existing.features |= features
            existing.origins.append(row.get("scenario_id", ""))
            continue
        by_text[key] = Lesson(
            lesson_id=f"lesson-{index:04d}",
            text=text,
            label=row.get("label", "specific"),
            motif=motif,
            features=features,
            from_correct=True,
            reflection_type=row.get("reflection_type", "contrast"),
            origins=[row.get("scenario_id", "")],
        )
    return LessonMemory(list(by_text.values()))
