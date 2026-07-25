from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class FuguAleConfig:
    """ALE-owned knobs for the host-side Fugu-Ultra deployer."""

    name: ClassVar[str] = "fugu-ultra"

    model: str = "fugu-27b-conductor"
    conductor_base_url: str = "http://localhost:8010/v1"
    pool_binding_path: str | None = None
    max_turns: int = 120
    task_budget_s: float = 14_400.0
    record_terminal_session: bool = False
    provider_base_url: str = "https://yunwu.ai/v1"
    fail_closed_provider_errors: bool = False
    conductor_temperature: float = 0.0
    conductor_seed: int = 0
    record_conductor_token_data: bool = False
    conductor_policy_revision: str | None = (
        "fugu-ale-r2-continue-balanced-20260722"
    )
    optimizer_sequence_len: int = 2_816
    conductor_max_input_tokens: int = 7_680
    conductor_max_output_tokens: int = 512
    # When set, run a single bound worker solo (no conductor) as an in-pool
    # baseline arm. Mutually exclusive with the typed conductor.
    solo_worker_id: int | None = None


@dataclass
class AmberSmokeConfig:
    """Zero-model, optimizer-ineligible ALE transport/grader smoke config."""

    name: ClassVar[str] = "fugu-ale-amber-zero-paid-smoke"

    model: str = "none"
    connect_timeout_s: float = 120.0
