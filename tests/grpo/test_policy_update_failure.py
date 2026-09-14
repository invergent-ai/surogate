"""A policy push that dies detached must not be swallowed.

The push runs as its own task and the loop awaits it through `asyncio.shield`,
so cancelling the loop at a batch boundary leaves the push running with nobody
awaiting it. Every route that does await it discards what it raises:
`safe_cancel` catches `BaseException`, and the orchestrator's cleanup gathers
`stop()` with `return_exceptions=True`. So the failure is recorded on the task's
done-callback, which is the one point every push reaches, and the orchestrator
loop re-raises it.
"""

import asyncio

import pytest

from surogate.grpo.orchestrator.scheduler import Scheduler


def _scheduler(failure: BaseException | None = None):
    s = Scheduler.__new__(Scheduler)
    s.policy_update_error = None
    s.inflight_policy_update_task = None
    s.policy_update_lock = asyncio.Lock()
    s.checkpoint_ready = asyncio.Event()
    s.checkpoint_ready.set()

    async def _apply_policy_update(step):
        if failure is not None:
            raise failure

    s._apply_policy_update = _apply_policy_update
    return s


def _start_and_settle(s):
    """Start a push and let it finish, without awaiting it the way the loop does."""

    async def main():
        task = await s._get_or_start_policy_update_task(1)
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())


def test_a_failed_push_is_recorded_even_though_nobody_awaits_it():
    """The loop awaits the push through `asyncio.shield`, so cancelling the loop
    at a batch boundary leaves it running with nobody to receive its exception."""
    boom = RuntimeError("engine cannot reload full weights")
    s = _scheduler()

    async def _slow_then_fail(step):
        await asyncio.sleep(0.02)
        raise boom

    s._apply_policy_update = _slow_then_fail

    async def main():
        await s._get_or_start_policy_update_task(1)
        # Deliberately not awaited: the task outlives whoever started it.
        await asyncio.sleep(0.2)

    asyncio.run(main())
    assert s.policy_update_error is boom

    with pytest.raises(RuntimeError, match="reload full weights"):
        s.raise_if_policy_update_failed()


def test_a_failed_push_unblocks_the_batch_barrier():
    """`_apply_policy_update` clears `checkpoint_ready` and only sets it on the
    success path, so without this the run hangs instead of reporting."""
    s = _scheduler(RuntimeError("boom"))
    s.checkpoint_ready.clear()
    _start_and_settle(s)
    assert s.checkpoint_ready.is_set()


def test_draining_records_a_push_that_teardown_would_have_cancelled():
    boom = RuntimeError("boom")
    s = _scheduler()

    async def _slow_then_fail(step):
        await asyncio.sleep(0.02)
        raise boom

    s._apply_policy_update = _slow_then_fail

    async def main():
        await s._get_or_start_policy_update_task(1)
        await s.drain_policy_update()

    asyncio.run(main())
    assert s.policy_update_error is boom


def test_a_successful_push_records_nothing():
    s = _scheduler()
    _start_and_settle(s)
    assert s.policy_update_error is None
    s.raise_if_policy_update_failed()


def test_a_cancelled_push_is_not_a_failure():
    """Cancelling is how the loop stops at every batch boundary, not an error."""
    s = _scheduler()

    async def main():
        started = asyncio.Event()

        async def _slow(step):
            started.set()
            await asyncio.sleep(3600)

        s._apply_policy_update = _slow
        task = await s._get_or_start_policy_update_task(1)
        await started.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())
    assert s.policy_update_error is None
    s.raise_if_policy_update_failed()
