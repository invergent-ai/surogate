"""Outcome-based question curation for GRPO rollout collection (r5).

The r5 decision run measured ~70% of the candidate pool as always-solved and
~8% as never-solved for BOTH conductor policies (pool-intrinsic: kappa 0.506,
85% saturation overlap). Groups on those questions are uniform and carry zero
GRPO gradient, so rollout spend on them buys nothing. The fix, priced from
that same data (scratchpad/r5_gradient_yield.py): a cheap probe pass before
training and online attrition during it, which roughly DOUBLES gradient per
rollout at equal budget (1.84-2.08x vs the uncurated paper recipe).

Both mechanisms are strictly OUTCOME-based — a question is kept or dropped
only by the reward variance it produces when actually executed. No text,
keyword, language, or topic classification of any kind (hard rule: prompts
can be in any language).

* PROBE PASS (before training): m rollouts per candidate at the training
  temperature; keep a candidate iff its m rewards are not unanimous. m=3 is
  the measured cost optimum (~16-19 probe rollouts per retained question).
  Probe groups on retained questions are themselves valid (small) training
  groups and are returned so their cost is never wasted.
* ONLINE ATTRITION (during training): a retained question is dropped
  permanently once it returns `attrition_after` CONSECUTIVE uniform groups —
  as the policy improves, questions it has mastered stop paying and must not
  keep consuming rollouts. An informative group resets the streak.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ultra.grpo_rollout_collector import RolloutCollector

DEFAULT_PROBE_SIZE = 3       # measured optimum: cost per retained question
DEFAULT_ATTRITION_AFTER = 2  # consecutive uniform groups before a drop


def _is_informative(group: dict[str, Any]) -> bool:
    """Reward variance among scored rollouts; infra failures don't count."""
    rewards = [r for r in group.get("rewards", []) if r is not None]
    return len(set(rewards)) > 1


@dataclass
class ProbeFilter:
    """Pre-training difficulty filter: keep candidates with reward variance.

    Wraps a RolloutCollector configured for the TRAINING settings (same
    sampler temperature, same constrained workers) so the probe measures the
    distribution training will actually see. Results are journaled per
    candidate, so an interrupted probe run resumes without re-buying rollouts.

    Candidates are (question, gold, grader_type) triples — grading dispatches
    per question through the collector (mixed-grader pool); plain
    (question, gold) pairs are accepted for single-grader callers.

    `retention_target` stops the probe as soon as that many candidates are
    retained: with prior-ordered candidates the informative ones front-load,
    so the target is what makes ordering save money at all.
    """

    collector: RolloutCollector
    probe_size: int = DEFAULT_PROBE_SIZE
    retention_target: int | None = None
    # Candidates probed concurrently (each candidate is itself probe_size
    # parallel rollouts). Serial probing at ~5-8 min/candidate would take
    # days over a large pool; a small window keeps prior-ordering roughly
    # intact while the retention-target stop drains in-flight work.
    concurrency: int = 1
    journal_path: Path | None = None
    log: Callable[[str], None] = print

    def __post_init__(self) -> None:
        if self.probe_size < 2:
            raise ValueError("probe_size < 2 cannot observe reward variance")

    def _journal(self) -> dict[str, dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        if self.journal_path and self.journal_path.exists():
            for line in self.journal_path.read_text().split("\n"):
                if line.strip():
                    row = json.loads(line)
                    seen[row["question"]] = row
        return seen

    def probe(
        self, candidates: list[tuple]
    ) -> tuple[list[tuple], list[dict[str, Any]], dict[str, Any]]:
        """Probe candidates; return (retained, probe_groups, stats).

        `retained` preserves candidate order (same tuple shape as passed in).
        `probe_groups` holds the full group record of every INFORMATIVE probe
        (valid small training groups — returning them means probe spend is
        recycled, not written off). Stops early at `retention_target`.
        """
        import threading
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        original_group_size = self.collector.group_size
        self.collector.group_size = self.probe_size
        seen = self._journal()
        retained: list[tuple] = []
        probe_groups: list[dict[str, Any]] = []
        probed_now = infra_skipped = examined = 0
        journal_lock = threading.Lock()

        def evaluate(candidate, group):
            """Classify one completed probe; mutates counters under lock."""
            nonlocal infra_skipped
            scored = [r for r in group["rewards"] if r is not None]
            if len(scored) < self.probe_size:
                # Not a full observation — never classify a question from an
                # outage. Skip without verdict; the journal row is treated as
                # absent on the next run, so the question is re-probed rather
                # than silently lost.
                infra_skipped += 1
                return
            if _is_informative(group):
                retained.append(candidate)
                probe_groups.append(group)

        try:
            handle = (self.journal_path.open("a")
                      if self.journal_path else None)
            try:
                queue = list(candidates)
                queue.reverse()  # pop() from the front (prior order)
                workers = max(1, self.concurrency)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    in_flight: dict = {}

                    def target_hit() -> bool:
                        return (self.retention_target is not None
                                and len(retained) >= self.retention_target)

                    def submit_next() -> bool:
                        nonlocal examined, probed_now
                        while queue:
                            candidate = queue.pop()
                            question, gold = candidate[0], candidate[1]
                            grader_type = (candidate[2]
                                           if len(candidate) > 2 else None)
                            examined += 1
                            group = seen.get(question)
                            if group is not None and len(
                                [r for r in group["rewards"]
                                 if r is not None]
                            ) < self.probe_size:
                                group = None  # incomplete row: re-probe
                            if group is not None:
                                evaluate(candidate, group)  # journal reuse
                                if target_hit():
                                    return False
                                continue
                            probed_now += 1
                            future = pool.submit(
                                self.collector.collect_question,
                                question, gold, grader_type)
                            in_flight[future] = candidate
                            return True
                        return False

                    while not target_hit() and (in_flight or queue):
                        while (len(in_flight) < workers and queue
                               and not target_hit()):
                            if not submit_next():
                                break
                        if not in_flight:
                            break
                        done, _pending = wait(in_flight,
                                              return_when=FIRST_COMPLETED)
                        for future in done:
                            candidate = in_flight.pop(future)
                            group = future.result()
                            if handle:
                                with journal_lock:
                                    handle.write(json.dumps(group) + "\n")
                                    handle.flush()
                            evaluate(candidate, group)
                    # Target hit: drain in-flight (journal them; they still
                    # count — paid work is never discarded).
                    for future, candidate in list(in_flight.items()):
                        group = future.result()
                        if handle:
                            with journal_lock:
                                handle.write(json.dumps(group) + "\n")
                                handle.flush()
                        evaluate(candidate, group)
            finally:
                if handle:
                    handle.close()
        finally:
            self.collector.group_size = original_group_size
        stats = {
            "candidates": len(candidates),
            "examined": examined,
            "probed_now": probed_now,
            "probed_from_journal": examined - probed_now - infra_skipped,
            "retained": len(retained),
            "keep_rate": len(retained) / examined if examined else 0.0,
            "infra_skipped": infra_skipped,
            "probe_rollouts_spent": probed_now * self.probe_size,
            "stopped_at_target": (self.retention_target is not None
                                  and len(retained) >= self.retention_target),
        }
        self.log(
            f"[probe] kept {stats['retained']}/{stats['examined']} examined "
            f"of {stats['candidates']} candidates "
            f"(keep rate {stats['keep_rate']:.3f}, "
            f"{stats['probe_rollouts_spent']} rollouts spent, "
            f"{stats['infra_skipped']} skipped on infra"
            f"{', stopped at target' if stats['stopped_at_target'] else ''})"
        )
        return retained, probe_groups, stats


@dataclass
class AttritionTracker:
    """Drops a question after `attrition_after` consecutive uniform groups.

    Purely reactive bookkeeping over observed groups: call `observe` with
    each collected group; `active` answers whether a question is still worth
    rollouts. Dropped questions stay dropped (mastered questions do not come
    back as the policy only gets stronger on them).
    """

    attrition_after: int = DEFAULT_ATTRITION_AFTER
    _streaks: dict[str, int] = field(default_factory=dict)
    _dropped: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.attrition_after < 1:
            raise ValueError("attrition_after must be >= 1")

    def observe(self, group: dict[str, Any]) -> bool:
        """Record a group; returns True when its question remains active."""
        question = group["question"]
        if question in self._dropped:
            return False
        scored = [r for r in group.get("rewards", []) if r is not None]
        infra = int(group.get("infra_errors", 0))
        # A group where outages outnumber scores observes (almost) nothing
        # about difficulty: 2-of-16 scored agreeing is not "uniform". Require
        # a scored MAJORITY before a uniform group may advance the streak.
        if len(scored) < 2 or len(scored) <= infra:
            return True
        if _is_informative(group):
            self._streaks[question] = 0
            return True
        streak = self._streaks.get(question, 0) + 1
        self._streaks[question] = streak
        if streak >= self.attrition_after:
            self._dropped.add(question)
            return False
        return True

    def active(self, question: str) -> bool:
        return question not in self._dropped

    def filter(self, questions: list[tuple]) -> list[tuple]:
        """Keep active candidates; tuples of any arity, question first."""
        return [t for t in questions if self.active(t[0])]

    @property
    def dropped(self) -> frozenset[str]:
        return frozenset(self._dropped)

    def state(self) -> dict[str, Any]:
        """Serializable snapshot (for run journals / resume)."""
        return {"streaks": dict(self._streaks),
                "dropped": sorted(self._dropped)}

    @classmethod
    def from_state(cls, state: dict[str, Any],
                   attrition_after: int = DEFAULT_ATTRITION_AFTER
                   ) -> "AttritionTracker":
        tracker = cls(attrition_after=attrition_after)
        tracker._streaks = dict(state.get("streaks", {}))
        tracker._dropped = set(state.get("dropped", []))
        return tracker
