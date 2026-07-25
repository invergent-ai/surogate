"""Harbor agent for a true solo-versus-typed-conductor comparison."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, override

from harbor.models.agent.context import AgentContext

from director.agentic.fugu_ultra_terminal import FuguUltraTerminalAgent


ARM_ENV = "FUGU_ROUTE2_COMPARISON_ARM"
SOLO_WORKER_ID = 0


class FuguRoute2ComparisonAgent(FuguUltraTerminalAgent):
    """Select one frozen comparison arm without changing product defaults."""

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        arm = os.environ.get(ARM_ENV)
        if arm not in {"solo", "conductor"}:
            raise RuntimeError(f"{ARM_ENV} must be 'solo' or 'conductor'")
        if arm == "conductor" and not os.environ.get("FUGU_TYPED_CONDUCTOR_MODEL"):
            raise RuntimeError("the conductor arm requires FUGU_TYPED_CONDUCTOR_MODEL")
        if arm == "solo":
            kwargs["solo_worker_id"] = SOLO_WORKER_ID
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name or f"fugu-route2-{arm}",
            **kwargs,
        )
        self._comparison_arm = arm

    @staticmethod
    @override
    def name() -> str:
        return "fugu-route2-comparison"

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "route2_comparison_arm": self._comparison_arm,
                "route2_true_solo_baseline": self._comparison_arm == "solo",
                "route2_typed_conductor": self._comparison_arm == "conductor",
            }
        )
        context.metadata = metadata
