"""Tool/dialogue source adapter (EXECUTION-PENDING: tool_dialog harness)."""

from __future__ import annotations

from ..policy import SOURCE_POLICY
from ..schemas import TaskSpec
from .hf import make_taskspec
from .raw import RawRecordAdapter


class TauBenchAdapter(RawRecordAdapter):
    """tau-bench-style tool/dialogue tasks (retail / airline / banking; custom domains).

    Each record carries a domain, a user-goal instruction, the domain API tool schemas, and
    a programmatic DB-state success check. Reward is ``db_state``, graded by the tool_dialog
    harness once built.
    """

    source_name = "tau_custom"
    capability = "tool_dialogue"
    policy = SOURCE_POLICY["tau_custom"]
    harness = "tool_dialog"

    def _to_spec(self, raw: dict, i: int) -> TaskSpec | None:
        if not raw.get("instruction"):
            return None
        domain = raw.get("domain", "tau")
        return make_taskspec(
            task_id=f"tau_custom__{domain}__{raw.get('id', i)}",
            capability="tool_dialogue",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="tool_dialog",
            grader_type="db_state",
            messages=[{"role": "user", "content": raw["instruction"]}],
            tools=raw.get("tools", []),
            group_id=domain,
            contamination_group=domain,
            domain=domain,
            requires_tools=True,
            estimated_worker_calls=raw.get("max_turns", 5),
            tags=["tool", "dialogue"],
        )
