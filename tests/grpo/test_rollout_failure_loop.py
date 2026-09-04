"""A task that never completes a rollout must fail the run, not spin forever.

Observed 2026-09-03: vLLM rejected every rollout with a 400 it would never
accept, the scheduler re-scheduled each one, and the run sat in ``running``
with ``error`` null for twenty minutes holding two GPUs until a human
cancelled it.

Re-scheduling is right for a flaky rollout and wrong for a rejected one, and
the two are told apart at two levels: a group whose own rollouts keep failing
is a bad row and gets dropped, and groups that keep dying with nothing
completing in between mean the run itself cannot proceed.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

import pytest

from surogate.grpo.orchestrator.scheduler import (
    MAX_CONSECUTIVE_DROPPED_GROUPS,
    MAX_ROLLOUT_ATTEMPTS_PER_GROUP,
    GroupState,
    InflightRolloutInfo,
    RolloutFailureLoop,
    Scheduler,
)
from surogate.grpo.utils.logger import get_logger

REJECTION = "ModelError() -> BadRequestError('Error code: 400 - tool choice requires ...')"


class _StubScheduler(Scheduler):
    """A Scheduler with everything but the rollout loop stubbed out.

    Real ``generate_batch``, real re-scheduling, real bookkeeping — only the
    parts that would need an environment, a buffer and a live inference pool
    are replaced. The defect was in this wiring, not in a counter, so the
    guard is exercised through the loop it has to stop.
    """

    TASK = "toolenv"

    def __init__(self) -> None:
        self.logger = get_logger()
        self.step = 0
        self.batch_size = 8
        self.token_batch_size = None
        self.rollouts_per_example = 4
        self.json_logging = False
        self.update_policy_task = None
        self.inflight_requests = {}
        self.scoring_tasks = {}
        self.groups = {}
        self.next_group_id = 0
        self.deferred_group_scoring_tasks = set()
        self.checkpoint_ready = asyncio.Event()
        self.checkpoint_ready.set()
        self.failed_attempts_by_group = defaultdict(int)
        self.dropped_groups_by_task = defaultdict(int)
        self.empty_rollouts_by_task = defaultdict(int)
        self.errored_rollouts_by_task = defaultdict(int)
        self.total_rollouts_by_task = defaultdict(int)
        self.last_batch_generation_time = 0.0
        self.cancelled_rollouts_count = 0
        self.inflight_policy_update_task = None
        self.dropped_group_ids: list[int] = []
        self.buffer = _PassThroughBuffer()

    async def maybe_update_policy(self) -> None:
        return None

    async def update_policy_loop(self) -> None:
        await asyncio.Event().wait()

    async def drop_group(self, group_id: int) -> int:
        self.dropped_group_ids.append(group_id)
        return await super().drop_group(group_id)

    def _rollout(self, group_id: int) -> dict:
        raise NotImplementedError

    async def _fill_inflight_requests(self) -> None:
        while len(self.inflight_requests) < self.rollouts_per_example:
            group_id = next(
                (gid for gid, g in self.groups.items() if g.rollouts_to_schedule > 0), None
            )
            if group_id is None:
                group_id = self.next_group_id
                self.next_group_id += 1
                self.groups[group_id] = GroupState(
                    example={}, rollouts_to_schedule=self.rollouts_per_example
                )
            self.groups[group_id].rollouts_to_schedule -= 1
            outcome = self._rollout(group_id)
            task = asyncio.create_task(_resolved(outcome))
            self.inflight_requests[task] = InflightRolloutInfo(
                off_policy_steps=0, client_config=None, task=self.TASK, group_id=group_id
            )


async def _resolved(value: dict) -> dict:
    return value


class _PassThroughBuffer:
    """Accepts every group, which is what Buffer does with filtering off.

    Mirrors the two methods generate_batch calls: ``update`` takes the
    completed group, ``sample_rollouts`` hands back up to n and keeps the rest.
    """

    def __init__(self) -> None:
        self.rollout_buffer: list[dict] = []

    def update(self, rollouts: list[dict]) -> None:
        self.rollout_buffer.extend(rollouts)

    def sample_rollouts(self, n: int) -> list[dict]:
        n = min(n, len(self.rollout_buffer))
        sampled, self.rollout_buffer = self.rollout_buffer[-n:], self.rollout_buffer[:-n]
        return sampled


def _run_one_batch(scheduler: Scheduler):
    """Drive the real ``generate_batch`` once.

    ``asyncio.run``, not pytest-asyncio: the repo has no async plugin, and
    tests/grpo/test_depth_controller.py already drives coroutines this way.
    """

    async def go():
        try:
            return await asyncio.wait_for(scheduler.generate_batch(step=0), timeout=30)
        finally:
            await scheduler.stop()

    return asyncio.run(go())


# ── Everything is rejected: the run must fail ─────────────────────────


class _RejectedScheduler(_StubScheduler):
    def _rollout(self, group_id: int) -> dict:
        return {"trajectory": [], "error": {"error_chain_repr": REJECTION}}


def test_a_rejected_task_fails_the_batch_instead_of_re_scheduling_forever():
    scheduler = _RejectedScheduler()

    with pytest.raises(RolloutFailureLoop) as excinfo:
        _run_one_batch(scheduler)

    # The server's own message is the diagnosis — it named the exact missing
    # flags. A bare "run failed" leaves the same mystery in a terminal state.
    assert REJECTION in str(excinfo.value)
    assert scheduler.TASK in str(excinfo.value)
    assert len(scheduler.dropped_group_ids) == MAX_CONSECUTIVE_DROPPED_GROUPS


# ── One bad row: drop it, and let the run finish ──────────────────────


class _OnePoisonRowScheduler(_StubScheduler):
    """Group 0 can never succeed; every other example is fine.

    This is the case a per-task failure counter gets wrong. A rejected request
    fails in milliseconds and is re-scheduled at once, while a healthy rollout
    takes seconds, so at a cold start the poison row can out-count every
    success and kill a run whose dataset is otherwise fine.
    """

    POISON_GROUP = 0

    def _rollout(self, group_id: int) -> dict:
        if group_id == self.POISON_GROUP:
            return {"trajectory": [], "error": {"error_chain_repr": "bad row"}}
        return {
            "trajectory": [{"tokens": None, "response": {}}],
            "error": None,
            "example_id": group_id,
            "reward": 1.0,
        }


def test_one_unrunnable_example_is_dropped_and_the_batch_still_completes():
    scheduler = _OnePoisonRowScheduler()

    rollouts = _run_one_batch(scheduler)

    assert scheduler.dropped_group_ids == [_OnePoisonRowScheduler.POISON_GROUP]
    assert len(rollouts) == scheduler.batch_size
    # Dropping it cost a bounded number of attempts, not an unbounded retry.
    assert scheduler.errored_rollouts_by_task[scheduler.TASK] == MAX_ROLLOUT_ATTEMPTS_PER_GROUP


# ── The streak resets on progress ─────────────────────────────────────


def _bare_scheduler() -> Scheduler:
    """Carries only what ``_note_dropped_group`` reads."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.failed_attempts_by_group = defaultdict(int)
    scheduler.dropped_groups_by_task = defaultdict(int)
    return scheduler


def test_a_completed_rollout_clears_the_dropped_group_streak():
    scheduler = _bare_scheduler()

    for _ in range(MAX_CONSECUTIVE_DROPPED_GROUPS - 1):
        scheduler._note_dropped_group("flaky", "bad row")
    asyncio.run(scheduler._note_rollout_outcome("flaky", 99, None))

    # A run working through some bad rows still trains; only one where nothing
    # gets through in between is unrecoverable.
    for _ in range(MAX_CONSECUTIVE_DROPPED_GROUPS - 1):
        scheduler._note_dropped_group("flaky", "bad row")


def test_one_broken_task_does_not_ride_on_another_s_dropped_groups():
    scheduler = _bare_scheduler()

    for _ in range(MAX_CONSECUTIVE_DROPPED_GROUPS - 1):
        scheduler._note_dropped_group("env_a", "bad row")
        scheduler._note_dropped_group("env_b", "bad row")

    with pytest.raises(RolloutFailureLoop) as excinfo:
        scheduler._note_dropped_group("env_b", "bad row")
    assert "env_b" in str(excinfo.value)
