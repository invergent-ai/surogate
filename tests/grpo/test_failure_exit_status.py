"""A GRPO component failure must become a non-zero exit status.

Two halves of one gap, in both GRPO runners.

The trainer's crash handler called `logger.exception`, which `LoggerWrapper`
did not define, so the handler raised inside itself and the `error_event.set()`
beside it never ran: the watchdog was never told and the run hung in `running`
forever, holding its GPUs. The durable fix is the ordering -- set the channel
before doing anything fallible -- and the root fix is giving `LoggerWrapper`
the one stdlib method it was missing, since 100-odd other modules could reach
for it inside an `except` block just as easily.

And when a watchdog *does* abort, it does so by signalling its own process,
which the main thread could not tell apart from a user pressing Ctrl-C. It
swallowed the interrupt, returned normally, and the process exited 0 -- so ops
finalized a crashed run as `completed` with a null error.
"""

import sys
import threading
from unittest import mock

import pytest

# `split` and `colocate` reach the compiled extension through their config
# import, which has nothing to do with the failure plumbing under test. Use the
# real one where it is built (CI) and a stand-in where it is not, so these stay
# runnable without a GPU toolchain.
try:  # pragma: no cover - depends on the build environment
    import surogate._surogate  # noqa: F401
except ImportError:  # pragma: no cover
    import surogate

    _stub = mock.MagicMock()
    sys.modules["surogate._surogate"] = _stub
    surogate._surogate = _stub

from surogate.grpo import colocate, split  # noqa: E402
from surogate.grpo.abort import AbortReason  # noqa: E402
from surogate.utils.logger import get_logger  # noqa: E402


def test_the_logger_has_an_exception_method_that_keeps_the_traceback():
    """The root cause. `LoggerWrapper` mirrors the stdlib logger API but omitted
    this one member, so `logger.exception(...)` raised `AttributeError` from
    inside the very handler that was reporting a crash."""
    logger = get_logger()
    assert hasattr(logger, "exception")
    with mock.patch.object(logger, "error") as errored:
        logger.exception("boom")
    assert errored.call_args.kwargs["exc_info"] is True, "the traceback is the point"


def _crashing_trainer_module():
    """Both runners import the trainer inside `_run_trainer`, so swapping the
    module out is enough -- and avoids importing the real one, which drags in
    the whole engine."""
    fake = mock.MagicMock()
    fake.GRPOTrainer.side_effect = RuntimeError("boom")
    return fake


# ── the failure signal survives a broken logger ──────────────────────


def test_a_trainer_crash_sets_the_failure_event():
    ev = threading.Event()
    with mock.patch.dict(sys.modules, {"surogate.grpo.trainer": _crashing_trainer_module()}):
        split._run_trainer(mock.MagicMock(), ev)
    assert ev.is_set(), "the watchdog's only propagation channel must fire"


def test_the_event_is_set_even_when_logging_fails():
    """The regression test. Ordering is the durable fix: set the channel first,
    then log, so nothing wrong with the logging call can cost us the signal."""
    ev = threading.Event()
    broken = mock.MagicMock()
    broken.exception.side_effect = AttributeError("no such method")
    with (
        mock.patch.dict(sys.modules, {"surogate.grpo.trainer": _crashing_trainer_module()}),
        mock.patch.object(split, "logger", broken),
    ):
        with pytest.raises(AttributeError):
            split._run_trainer(mock.MagicMock(), ev)
    assert ev.is_set(), "a broken logger must not cost us the failure signal"


def test_the_colocate_event_is_set_even_when_logging_fails():
    """Same handler, same rule, the other runner."""
    ev = threading.Event()
    broken = mock.MagicMock()
    broken.exception.side_effect = AttributeError("no such method")
    with (
        mock.patch.dict(sys.modules, {"surogate.grpo.trainer": _crashing_trainer_module()}),
        mock.patch.object(colocate, "logger", broken),
    ):
        with pytest.raises(AttributeError):
            colocate._run_trainer(mock.MagicMock(), None, ev)
    assert ev.is_set()


# ── an abort leaves a reason behind, a clean shutdown does not ───────


def test_the_watchdog_records_why_it_aborted():
    trainer_failed = threading.Event()
    trainer_failed.set()
    abort = AbortReason()

    with (
        # The sentinel poll would otherwise sleep its full timeout on an empty
        # fd set; this test is about the trainer branch, not the wait.
        mock.patch.object(split.multiprocessing.connection, "wait", return_value=[]),
        mock.patch.object(split.os, "kill") as killed,
    ):
        split._watch_components([], trainer_failed, threading.Event(), abort)

    assert abort.reason, "the abort must leave a reason behind"
    assert "rainer" in abort.reason
    assert killed.called, "and it still signals the main thread"


def test_the_colocate_watchdog_records_why_it_aborted():
    error_event = threading.Event()
    error_event.set()
    abort = AbortReason()

    with mock.patch.object(colocate.os, "kill") as killed:
        colocate._watch_components(error_event, threading.Event(), abort)

    assert abort.reason, "both runners exit through the same ambiguity"
    assert killed.called


@pytest.mark.parametrize(
    "runner,call",
    [
        ("split", lambda abort, shutdown: split._watch_components([], threading.Event(), shutdown, abort)),
        ("colocate", lambda abort, shutdown: colocate._watch_components(threading.Event(), shutdown, abort)),
    ],
)
def test_a_clean_shutdown_records_nothing(runner, call):
    """A planned teardown must stay exit 0. Only a recorded reason is a failure,
    which is what keeps a user's Ctrl-C from being reported as a crash."""
    shutdown = threading.Event()
    shutdown.set()
    abort = AbortReason()

    with (
        mock.patch.object(split.os, "kill") as split_killed,
        mock.patch.object(colocate.os, "kill") as colocate_killed,
    ):
        call(abort, shutdown)

    assert abort.reason is None
    assert not split_killed.called and not colocate_killed.called
