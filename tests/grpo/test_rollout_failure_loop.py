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
from types import SimpleNamespace

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

    Real ``generate_batch``, real re-scheduling, real bookkeeping. Only the
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
        self.max_inflight_rollouts = 4
        self.json_logging = False
        self.update_policy_task = None
        self.inflight_requests = {}
        self.scoring_tasks = {}
        self.groups = {}
        self.next_group_id = 0
        self.deferred_group_scoring_tasks = set()
        self.checkpoint_ready = asyncio.Event()
        self.checkpoint_ready.set()
        self.dropped_groups_by_task = defaultdict(int)
        self.total_dropped_groups = 0
        self.empty_rollouts_by_task = defaultdict(int)
        self.errored_rollouts_by_task = defaultdict(int)
        self.total_rollouts_by_task = defaultdict(int)
        self.last_batch_generation_time = 0.0
        self.cancelled_rollouts_count = 0
        self.inflight_policy_update_task = None
        self.dropped_group_ids: list[int] = []
        self.buffer = _PassThroughBuffer()
        self.prefetch_batches = True
        # A real Scheduler always has one; this stub predates the first field
        # it needed to read. The production default, so these tests see what a
        # real run sees - 600s is far beyond the 30s cap in _run_one_batch.
        self.config = SimpleNamespace(batch_stall_timeout=600)

    async def maybe_update_policy(self) -> None:
        return None

    async def update_policy_loop(self) -> None:
        await asyncio.Event().wait()

    async def drop_group(self, group_id: int) -> int:
        self.dropped_group_ids.append(group_id)
        return await super().drop_group(group_id)

    async def _produce(self, group_id: int) -> dict:
        return self._rollout(group_id)

    def _rollout(self, group_id: int) -> dict:
        raise NotImplementedError

    async def schedule_rollout(self, group_id: int) -> None:
        """The one method that would need a client, an env and a rate limiter.

        Overriding only this leaves the real ``_fill_inflight_requests`` and
        ``_schedule_next_request`` in the loop, so the guard is exercised
        against the scheduling code it ships beside -- including its capacity
        rule, which an earlier copy of this stub got wrong.
        """
        group = self.groups.get(group_id)
        if group is None or group.rollouts_to_schedule <= 0:
            return
        group.rollouts_to_schedule -= 1
        # Built inside the task, not at creation: a rollout that raises must
        # raise where a real one does, out of `finished_task.result()`.
        task = asyncio.create_task(self._produce(group_id))
        self.inflight_requests[task] = InflightRolloutInfo(
            off_policy_steps=0, client_config=None, task=self.TASK, group_id=group_id
        )


class _PassThroughBuffer:
    """Accepts every group, which is what Buffer does with filtering off.

    Mirrors the two methods generate_batch calls: ``update`` takes the
    completed group, ``sample_rollouts`` hands back up to n and keeps the rest.
    """

    def __init__(self) -> None:
        self.rollout_buffer: list[dict] = []

    def update(self, rollouts: list[dict]) -> None:
        self.rollout_buffer.extend(rollouts)

    def sample_examples(self, n: int) -> list[dict]:
        return [{} for _ in range(n)]

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

    # The server's own message is the diagnosis: it named the exact missing
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


# ── A flaky group is not a poison one ─────────────────────────────────


class _FlakyScheduler(_StubScheduler):
    """Every group fails every other attempt, and succeeds in between.

    The counter is "failures since this group last completed one", not
    "failures ever". Without the reset on success a merely flaky group
    accumulates toward the cap over its whole life and is eventually dropped
    as poison -- and groups do live across steps, since ``prefetch_batches``
    defaults to True and does not clear them at a batch boundary.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attempts_by_group: dict[int, int] = defaultdict(int)

    # Four failures for every success. A group needs four successes to
    # complete, so it accumulates exactly MAX_ROLLOUT_ATTEMPTS_PER_GROUP
    # failures over its life while never reaching five in a row. Without the
    # reset it is dropped as poison; with it, nothing is dropped.
    _FAILURES_PER_SUCCESS = 4

    def _rollout(self, group_id: int) -> dict:
        self.attempts_by_group[group_id] += 1
        if self.attempts_by_group[group_id] % (self._FAILURES_PER_SUCCESS + 1):
            return {"trajectory": [], "error": {"error_chain_repr": "flaky"}}
        return {
            "trajectory": [{"tokens": None, "response": {}}],
            "error": None, "example_id": group_id, "reward": 1.0,
        }


def test_a_group_that_fails_between_successes_is_never_dropped():
    scheduler = _FlakyScheduler()

    rollouts = _run_one_batch(scheduler)

    assert scheduler.dropped_group_ids == []
    assert len(rollouts) == scheduler.batch_size
    # It failed enough times to be dropped had the failures been counted
    # cumulatively; it just never failed MAX times in a row.
    assert (
        scheduler.errored_rollouts_by_task[scheduler.TASK]
        >= MAX_ROLLOUT_ATTEMPTS_PER_GROUP
    )


# ── A raised rollout is dropped, but never kills the run ──────────────


class _RaisingScheduler(_StubScheduler):
    """Every rollout raises rather than returning an error.

    That path drops its group on the first raise, with none of the per-group
    headroom the errored path gets, so counting those drops toward the streak
    let a brief transport outage end a run in under a second. It is therefore
    deliberately uncounted, and the batch spins instead -- which is the
    behaviour that path has always had.
    """

    def _rollout(self, group_id: int) -> dict:
        raise RuntimeError("transport went away")


def test_a_storm_of_raised_rollouts_does_not_fail_the_run():
    scheduler = _RaisingScheduler()

    async def go():
        try:
            # It never completes a batch; the point is which way it does not.
            await asyncio.wait_for(scheduler.generate_batch(step=0), timeout=2)
        finally:
            await scheduler.stop()

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())

    # Far past MAX_CONSECUTIVE_DROPPED_GROUPS worth of drops, and still alive.
    assert len(scheduler.dropped_group_ids) > MAX_CONSECUTIVE_DROPPED_GROUPS
    assert scheduler.dropped_groups_by_task[scheduler.TASK] == 0


# ── The streak resets on progress ─────────────────────────────────────


def _metrics(scheduler: Scheduler) -> dict:
    """``get_metrics`` on a scheduler built only for the counter paths."""
    scheduler.wait_for_ckpt_time = 0.0
    scheduler.update_weights_time = 0.0
    scheduler.step = 0
    scheduler.ckpt_step = 0
    scheduler.cancelled_rollouts_count = 0
    scheduler.inflight_requests = {}
    scheduler.groups = {}
    scheduler.empty_rollouts_by_task = defaultdict(int)
    scheduler.errored_rollouts_by_task = defaultdict(int)
    scheduler.total_rollouts_by_task = defaultdict(int)
    scheduler.inference_pool = SimpleNamespace(get_metrics=lambda: {})
    return scheduler.get_metrics()


def _bare_scheduler() -> Scheduler:
    """Carries only what ``_note_dropped_group`` reads."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.dropped_groups_by_task = defaultdict(int)
    scheduler.total_dropped_groups = 0
    return scheduler


def test_the_dropped_group_gauge_survives_a_success():
    """The streak answers "is this task failing now"; the gauge answers "how
    much training data has this run thrown away". A run dropping a group per
    step with successes in between reports 0 forever if they are the same
    number."""
    scheduler = _bare_scheduler()
    group = GroupState(example={}, rollouts_to_schedule=0)

    scheduler._note_dropped_group("flaky", "bad row")
    asyncio.run(scheduler._note_rollout_outcome("flaky", 99, group, None))

    assert scheduler.dropped_groups_by_task["flaky"] == 0

    # Through get_metrics, because that is where the gauge was wrong: it read
    # the streak, which this success has just zeroed.
    assert _metrics(scheduler)["scheduler/dropped_groups"] == 1


def test_a_completed_rollout_clears_the_dropped_group_streak():
    """A run working through some bad rows still trains; only one where
    nothing gets through in between is unrecoverable."""
    scheduler = _bare_scheduler()
    group = GroupState(example={}, rollouts_to_schedule=0)

    for _ in range(MAX_CONSECUTIVE_DROPPED_GROUPS - 1):
        scheduler._note_dropped_group("flaky", "bad row")
    asyncio.run(scheduler._note_rollout_outcome("flaky", 99, group, None))

    assert scheduler.dropped_groups_by_task["flaky"] == 0


def test_one_broken_task_does_not_ride_on_another_s_dropped_groups():
    scheduler = _bare_scheduler()

    for _ in range(MAX_CONSECUTIVE_DROPPED_GROUPS - 1):
        scheduler._note_dropped_group("env_a", "bad row")
        scheduler._note_dropped_group("env_b", "bad row")

    with pytest.raises(RolloutFailureLoop) as excinfo:
        scheduler._note_dropped_group("env_b", "bad row")
    assert "env_b" in str(excinfo.value)


# ── Rollouts all succeed, nothing progresses: the run must still fail ──


class _DiscardingBuffer(_PassThroughBuffer):
    """Accepts every group and hands none back.

    What difficulty filtering does to a group scored uniformly: the rollouts
    are fine, so nothing failure-shaped is recorded, and ``sample_rollouts``
    returns [] because a flat group carries no learning signal. A rubric that
    raises on every group drops them the same way.
    """

    def sample_rollouts(self, n: int) -> list[dict]:
        return []


class _NoProgressScheduler(_StubScheduler):
    """Every rollout succeeds; the buffer discards them all."""

    def __init__(self) -> None:
        super().__init__()
        self.buffer = _DiscardingBuffer()
        # Short, so the test does not sit for the 600s production default.
        self.config.batch_stall_timeout = 1

    def _rollout(self, group_id: int) -> dict:
        return {
            "trajectory": [{"tokens": None, "response": {}}],
            "error": None,
            "example_id": group_id,
            "reward": 0.0,
        }


def test_a_batch_that_never_progresses_fails_even_though_every_rollout_succeeds():
    """The case the rollout-failure guard above cannot see.

    Observed live 2026-09-23 on the repo's own tool environment: step 1 scored
    0.0000 across all 8 rollouts, the flat group was dropped whole, and the
    loop span with `Active tasks: 0` and both GPUs held until it was killed.
    Every rollout had completed normally, so the consecutive-failure streak
    was reset on each pass and never tripped.
    """
    scheduler = _NoProgressScheduler()

    with pytest.raises(RuntimeError) as excinfo:
        _run_one_batch(scheduler)

    message = str(excinfo.value)
    # Not RolloutFailureLoop: nothing failed. A reader who sees that exception
    # goes looking for a broken rollout and finds eight healthy ones.
    assert not isinstance(excinfo.value, RolloutFailureLoop)
    # The message has to point at the reward spread, which is the actual cause
    # and is not visible anywhere else in a terminal state.
    assert "batch progress stuck" in message
    assert "reward spread" in message


def test_the_watchdog_can_be_disabled():
    """None means no watchdog, for a run that legitimately stalls longer.

    Without this the test above would pass against a hard-coded timeout, and
    an operator with a slow environment would have no way out.

    Its own short bound rather than ``_run_one_batch``: this scheduler spins
    at full speed (rollouts complete instantly and are discarded), so the
    shared 30s cap would burn 30 real seconds of CPU on every suite run to
    learn one bit.
    """
    scheduler = _NoProgressScheduler()
    scheduler.config.batch_stall_timeout = None

    async def go():
        try:
            await asyncio.wait_for(scheduler.generate_batch(step=0), timeout=1)
        finally:
            await scheduler.stop()

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())
