"""Per-step routed rollout over an AgentEnv.

At each turn the router picks the worker (over the raw transcript), that worker emits one
shell action, the env executes it, and the observation is appended. The loop ends on the
submit sentinel, an env-signalled done, or the turn budget. Returns the terminal reward
plus the per-turn worker sequence (for the "which worker, when" analysis).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..fugu.inference import select_worker
from ..shared.transcript import Transcript
from ..shared.types import Sampling
from .actions import parse_action
from .env import AgentEnv
from .prompts import AGENT_SYSTEM, wrap_observation


@dataclass
class RolloutResult:
    reward: float
    turns: int
    submitted: bool
    worker_sequence: list[str] = field(default_factory=list)
    transcript: Transcript | None = None
    cost_usd: float = 0.0


async def agentic_rollout(
    router,
    pool,
    env: AgentEnv,
    *,
    max_turns: int = 30,
    sampling: Sampling | None = None,
    action_sampling: Sampling | None = None,
    allowed: "set[str] | list[str] | None" = None,
) -> RolloutResult:
    sampling = action_sampling or sampling or Sampling(temperature=0.2, max_tokens=2048)
    task = env.reset()
    tx = Transcript()
    tx.add("system", AGENT_SYSTEM)
    tx.add("user", task)

    worker_sequence: list[str] = []
    submitted = False
    turns = 0
    cost = 0.0
    for turns in range(1, max_turns + 1):
        # per-step routing over the raw transcript; allowed restricts the pool subset
        worker_id = select_worker(router, tx.render(), allowed=allowed)
        worker_sequence.append(worker_id)
        comp = await pool.call(worker_id, tx.as_messages(), sampling)
        cost += comp.cost_usd
        tx.add("assistant", comp.text)

        action = parse_action(comp.text)
        if action.submit:
            submitted = True
            break
        if action.command is None:
            tx.add("user", wrap_observation("No bash block found. Emit one ```bash command."))
            continue
        result = env.step(action.command)
        tx.add("user", wrap_observation(result.observation))
        if result.done:
            break

    reward = env.evaluate()
    return RolloutResult(
        reward=reward, turns=turns, submitted=submitted,
        worker_sequence=worker_sequence, transcript=tx, cost_usd=cost,
    )
