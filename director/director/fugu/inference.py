"""Decision-only inference for the Fugu router.

Single-step: featurize the query, argmax the head, dispatch the query to the selected
worker. Multi-turn: recompute the feature over the running transcript each turn and
route per step. The backbone never generates text — it only produces the routing
decision.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch

from ..shared.tasks import Task
from ..shared.transcript import Transcript, raw_query
from ..shared.types import Completion, Sampling
from ..shared.verifiers import get_grader
from .model import SelectionRouter

# The backbone forward runs on a single GPU but routing is called from many places at once —
# threaded harnesses (mini-swe-agent, terminal) AND event-loop rollouts (swe_pro/tau), often on
# the SAME router object (e.g. a CMA-ES candidate, or gen_router during generation). A single torch
# module isn't safe under concurrent forwards, so serialize every routing decision globally.
_GPU_LOCK = threading.Lock()


@torch.no_grad()
def select_worker(
    router: SelectionRouter, text: str, allowed: "set[str] | list[str] | None" = None
) -> str:
    """Pick the worker with the highest logit. ``allowed`` restricts the choice to a
    subset of worker ids (the product's per-request opt-out): disallowed workers are
    masked to -inf before the argmax, so selection falls back to the best *available*
    worker. Raises if ``allowed`` excludes the entire pool."""
    router.eval()
    with _GPU_LOCK:  # serialize the backbone forward across threads + the event loop
        logits = router.logits([text])[0]  # (L,)
        ids = router_worker_ids(router)
        if allowed is not None:
            allow = set(allowed)
            keep = torch.tensor([wid in allow for wid in ids], device=logits.device)
            if not bool(keep.any()):
                raise ValueError(f"allowed pool {allow} excludes all of {ids}")
            logits = logits.masked_fill(~keep, float("-inf"))
        j = int(logits.argmax(dim=-1).item())
    return ids[j]


def router_worker_ids(router: SelectionRouter) -> list[str]:
    ids = getattr(router, "worker_ids", None)
    if ids is None:
        raise AttributeError(
            "router has no worker_ids; call attach_worker_ids(router, pool.worker_ids)"
        )
    return ids


def attach_worker_ids(router: SelectionRouter, worker_ids: list[str]) -> None:
    """Bind the ordered worker ids so selection can name the chosen worker."""
    if len(worker_ids) != router.num_workers:
        raise ValueError(
            f"worker_ids has {len(worker_ids)} entries, router expects {router.num_workers}"
        )
    router.worker_ids = list(worker_ids)


@dataclass
class SingleResult:
    worker_id: str
    completion: Completion
    reward: float


async def answer_single(
    router: SelectionRouter, pool, task: Task, sampling: Sampling | None = None,
    allowed: "set[str] | list[str] | None" = None,
) -> SingleResult:
    sampling = sampling or Sampling()
    worker_id = select_worker(router, raw_query(task.prompt), allowed=allowed)
    comp = await pool.call(worker_id, task.messages(), sampling)
    reward = get_grader(task.grader)(comp.text, task.solution)
    return SingleResult(worker_id=worker_id, completion=comp, reward=reward)


@dataclass
class TurnStep:
    worker_id: str
    content: str


async def answer_multi_turn(
    router: SelectionRouter,
    pool,
    seed_messages: list[dict],
    max_turns: int,
    sampling: Sampling | None = None,
    user_responder=None,
    allowed: "set[str] | list[str] | None" = None,
) -> list[TurnStep]:
    """Per-turn routing over a transcript.

    ``user_responder(transcript) -> str | None`` supplies the next user message (or
    None to stop). With no responder this runs a single assistant turn. ``allowed``
    restricts routing to a worker subset (per-request opt-out).
    """
    sampling = sampling or Sampling()
    tx = Transcript(messages=list(seed_messages))
    steps: list[TurnStep] = []
    for _ in range(max_turns):
        worker_id = select_worker(router, tx.render(), allowed=allowed)
        comp = await pool.call(worker_id, tx.as_messages(), sampling)
        tx.add("assistant", comp.text)
        steps.append(TurnStep(worker_id=worker_id, content=comp.text))
        if user_responder is None:
            break
        nxt = user_responder(tx)
        if not nxt:
            break
        tx.add("user", nxt)
    return steps
