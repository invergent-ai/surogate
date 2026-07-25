"""Adaptive rollout-depth budgeting for multi-turn RL.

Implements the coverage arm of TurnOPD (arXiv:2607.05804 §5.1): cap rollout depth
at the turn where trajectories stop producing new information, instead of always
running to the environment's ``max_turns``.

Why this is worth doing, measured on `environments/multihop-tools` with a
Qwen3.5-2B student (see the turn diagnostics in surogate/grpo/turn_stats.py):
survivor coverage goes flat after turn ~6 (25.0% at turn 7 -> 20.7% at turn 11).
Those tail turns are the *same* stuck trajectories being re-sampled — ~17.6% of
supervised tokens and ~12.6% of the loss budget for no new supervision, plus the
teacher prefill over every one of those tokens.

Only the coverage bound H_cov is implemented, deliberately. The paper also has an
efficiency arm H_eff (a survivor-weighted reverse-KL centroid) and takes
max(H_eff, H_cov). H_eff needs trainer-side per-turn KL routed back into the
orchestrator, and on our measured profile — which peaks at turn ~3 rather than
being monotonically front-loaded — it lands inside the range H_cov already
covers. The extra machinery buys nothing here.

THE BIAS TRAP (their "Dynamic Update"): a cap estimated from capped trajectories
ratchets downward forever, because truncated rollouts can never demonstrate that
a deeper horizon was needed. Statistics are therefore collected ONLY from
uncapped probe steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _quantile(sorted_values: list[int], q: float) -> float:
    """Linear-interpolated quantile of a pre-sorted list."""
    if not sorted_values:
        raise ValueError("_quantile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


@dataclass
class DepthObservation:
    """One rollout's outcome, as the controller sees it."""

    depth: int  # number of model turns taken
    success: bool


@dataclass
class RolloutDepthController:
    """Chooses a per-step rollout-depth cap from observed completion depths.

    Usage per orchestrator step:
        cap = controller.cap_for_step(step)   # None => run uncapped (a probe)
        ...  run rollouts, injecting `cap` ...
        controller.observe(step, observations)
    """

    enabled: bool = False
    max_depth: int = 12
    min_depth: int = 2
    # Quantile of the success-conditioned completion depth to cover (their p=0.80).
    quantile: float = 0.80
    # EMA weight on the newly measured horizon (their alpha_ema).
    ema: float = 0.3
    # Run one uncapped probe step every N steps. Probes are the only source of
    # depth statistics, so this also sets how fast the cap can grow back.
    probe_interval: int = 8
    # Steps at the start that are always uncapped, to seed the estimate.
    warmup_steps: int = 2
    # Minimum uncapped rollouts before a cap is emitted at all.
    min_observations: int = 8

    _smoothed: float | None = field(default=None, init=False)
    _cap: int | None = field(default=None, init=False)
    _n_observed: int = field(default=0, init=False)
    _last_measured: int | None = field(default=None, init=False)
    _used_success_cdf: bool = field(default=False, init=False)

    def is_probe_step(self, step: int) -> bool:
        """Probe steps run uncapped and are the only ones that update statistics."""
        if not self.enabled:
            return True
        if step < self.warmup_steps:
            return True
        if self.probe_interval <= 0:
            return False
        return step % self.probe_interval == 0

    def cap_for_step(self, step: int) -> int | None:
        """Depth cap for this step, or None to run to the environment's own max."""
        if not self.enabled or self.is_probe_step(step):
            return None
        return self._cap

    def observe(self, step: int, observations: list[DepthObservation]) -> None:
        """Fold a probe step's rollouts into the horizon estimate.

        Non-probe steps are ignored: their depths are censored by the cap we
        imposed, so including them would drive the cap monotonically down.
        """
        if not self.enabled or not self.is_probe_step(step) or not observations:
            return

        self._n_observed += len(observations)

        successes = sorted(o.depth for o in observations if o.success)
        if successes:
            population, self._used_success_cdf = successes, True
        else:
            # No successes yet: fall back to the full population (their §7.3).
            # It is more conservative — failed rollouts stall at the horizon and
            # so overestimate the depth actually needed.
            population, self._used_success_cdf = sorted(o.depth for o in observations), False

        measured = _quantile(population, self.quantile)
        self._last_measured = int(round(measured))

        if self._smoothed is None:
            self._smoothed = measured
        else:
            self._smoothed = (1.0 - self.ema) * self._smoothed + self.ema * measured

        if self._n_observed >= self.min_observations:
            self._cap = max(self.min_depth, min(self.max_depth, int(round(self._smoothed))))

    @property
    def cap(self) -> int | None:
        return self._cap

    def metrics(self) -> dict[str, float]:
        return {
            "depth/cap": float(self._cap) if self._cap is not None else float(self.max_depth),
            "depth/smoothed": float(self._smoothed) if self._smoothed is not None else 0.0,
            "depth/last_measured": float(self._last_measured) if self._last_measured is not None else 0.0,
            "depth/observations": float(self._n_observed),
            "depth/success_conditioned": 1.0 if self._used_success_cdf else 0.0,
        }
