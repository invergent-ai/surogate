"""A task that never completes a rollout must fail the run, not spin forever.

Observed 2026-09-03: vLLM rejected every rollout with a 400 it would never
accept, the scheduler re-scheduled each one, and the run sat in ``running``
with ``error`` null for twenty minutes holding two GPUs until a human
cancelled it. Re-scheduling is right for a flaky rollout and wrong for a
rejected one; the difference is whether anything ever gets through.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from surogate.grpo.orchestrator.scheduler import (
    MAX_CONSECUTIVE_ROLLOUT_FAILURES,
    RolloutFailureLoop,
    Scheduler,
)


def _scheduler() -> Scheduler:
    """A Scheduler carrying only what ``_note_rollout_outcome`` reads.

    Building a real one needs an environment, a buffer, a config and a live
    inference pool, none of which this decision touches.
    """
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.consecutive_failures_by_task = defaultdict(int)
    return scheduler


def test_a_rejected_task_fails_the_run_with_the_server_s_message():
    scheduler = _scheduler()
    reason = "ModelError() -> BadRequestError('Error code: 400 - tool choice requires ...')"

    for _ in range(MAX_CONSECUTIVE_ROLLOUT_FAILURES - 1):
        scheduler._note_rollout_outcome("toolenv", reason)

    with pytest.raises(RolloutFailureLoop) as excinfo:
        scheduler._note_rollout_outcome("toolenv", reason)

    # The server's own message is the diagnosis; a bare "run failed" would
    # leave the same 20-minute mystery, just in a terminal state.
    assert reason in str(excinfo.value)
    assert "toolenv" in str(excinfo.value)


def test_a_completed_rollout_clears_the_streak():
    scheduler = _scheduler()

    for _ in range(MAX_CONSECUTIVE_ROLLOUT_FAILURES - 1):
        scheduler._note_rollout_outcome("flaky", "empty trajectory")
    scheduler._note_rollout_outcome("flaky", None)

    # A task that fails most of the time still trains; only one that never
    # completes is unrecoverable.
    for _ in range(MAX_CONSECUTIVE_ROLLOUT_FAILURES - 1):
        scheduler._note_rollout_outcome("flaky", "empty trajectory")


def test_one_broken_task_does_not_ride_on_another_s_failures():
    scheduler = _scheduler()

    for _ in range(MAX_CONSECUTIVE_ROLLOUT_FAILURES - 1):
        scheduler._note_rollout_outcome("env_a", "empty trajectory")
        scheduler._note_rollout_outcome("env_b", "empty trajectory")

    with pytest.raises(RolloutFailureLoop) as excinfo:
        scheduler._note_rollout_outcome("env_b", "empty trajectory")
    assert "env_b" in str(excinfo.value)
