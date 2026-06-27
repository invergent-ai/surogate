"""Sequential-state + autonomous-research source adapters (EXECUTION-PENDING).

Target the sequential_sim and research_loop harnesses. Simulator seeds / scenario templates
are held out by split (ultra-data2 §9).
"""

from __future__ import annotations

from ..policy import SOURCE_POLICY
from ..schemas import TaskSpec
from .hf import make_taskspec
from .raw import RawRecordAdapter


class SequentialSimAdapter(RawRecordAdapter):
    """Sequential decision simulators: blindfold chess, logistics, scheduling, portfolio/game."""

    source_name = "sequential_sim"
    capability = "planning"
    policy = SOURCE_POLICY["sequential_sim"]
    harness = "sequential_sim"
    source_type = "simulator"

    def _to_spec(self, raw: dict, i: int) -> TaskSpec | None:
        if not raw.get("instruction"):
            return None
        scenario = raw.get("scenario", "sim")
        return make_taskspec(
            task_id=f"sequential_sim__{scenario}__{raw.get('seed', i)}",
            capability="planning",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="sequential_sim",
            grader_type="sim_reward",
            messages=[{"role": "user", "content": raw["instruction"]}],
            group_id=scenario,
            contamination_group=f"{scenario}::seed-family",
            domain="planning",
            estimated_worker_calls=raw.get("max_turns", 5),
            tags=["simulator", "planning"],
        )


class AutoResearchAdapter(RawRecordAdapter):
    """AutoResearch-style ML-optimization / program-synthesis loops (expensive, late-stage)."""

    source_name = "autoresearch"
    capability = "research"
    policy = SOURCE_POLICY["autoresearch"]
    harness = "research_loop"
    source_type = "simulator"

    def _to_spec(self, raw: dict, i: int) -> TaskSpec | None:
        if not raw.get("instruction"):
            return None
        return make_taskspec(
            task_id=f"autoresearch__{raw.get('id', i)}",
            capability="research",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="research_loop",
            grader_type="metric_improvement",
            messages=[{"role": "user", "content": raw["instruction"]}],
            group_id=raw.get("task", "research"),
            domain="research",
            estimated_worker_calls=raw.get("max_turns", 8),
            tags=["research", "optimization"],
        )
