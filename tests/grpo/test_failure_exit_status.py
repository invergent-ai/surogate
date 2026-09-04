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
    # Delegating to `error` adds a frame, and the location is read at a fixed
    # depth, so without the bump every exception log is tagged `logger.py`.
    assert errored.call_args.kwargs["_depth"] == 3


def test_the_formatter_actually_emits_the_traceback():
    """`exc_info=True` was cosmetic: `ColoredFormatter.format` overrides the base
    method wholesale and never appended the traceback, so every crash in the
    codebase logged its one-line message with no stack behind it."""
    import logging

    from surogate.utils.logger import ColoredFormatter

    try:
        raise ZeroDivisionError("division by zero")
    except ZeroDivisionError:
        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "Trainer thread crashed", None, sys.exc_info())

    rendered = ColoredFormatter().format(record)
    assert "Trainer thread crashed" in rendered
    assert "ZeroDivisionError" in rendered, "the stack is the reason exc_info was asked for"
    assert "Traceback (most recent call last)" in rendered


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


def test_a_trainer_import_failure_sets_the_failure_event():
    """The import used to sit outside the try, so the likeliest crash on a
    dependency bump -- and the parent commit is a verifiers 0.1.11 -> 0.3.0
    upgrade -- set no event, sent no signal, and hung the run holding GPUs."""
    ev = threading.Event()
    with mock.patch.dict(sys.modules, {"surogate.grpo.trainer": None}):
        # A None entry in sys.modules makes `from ... import X` raise ImportError.
        split._run_trainer(mock.MagicMock(), ev)
    assert ev.is_set(), "an import-time crash is still a crash"


def test_a_planned_teardown_is_not_reported_as_a_crash():
    """Teardown sets `shutdown_event` and then kills the vLLM subprocesses, so
    their sentinels fire exactly like a crash. The loop's own post-wait check
    catches the ordinary case; this pins the window underneath it, where the
    scan has already run and teardown flips the flag before the abort. It used
    to leave a swallowed SIGINT and a warning. Now that an abort raises, it
    would mark a successful run as failed.
    """
    proc = mock.MagicMock()
    proc.sentinel = "fd"
    proc.exitcode = -15
    abort = AbortReason()

    # is_set() is read three times per pass: the `while`, the post-wait guard,
    # and the final one. Teardown lands between the second and the third.
    shutdown = mock.MagicMock()
    shutdown.is_set.side_effect = [False, False, True]

    with (
        mock.patch.object(split.multiprocessing.connection, "wait", return_value=["fd"]),
        mock.patch.object(split.os, "kill") as killed,
    ):
        split._watch_components([(proc, "rollout vLLM")], threading.Event(), shutdown, abort)

    assert abort.reason is None, "a planned teardown must not be an abort"
    assert not killed.called


def test_a_late_trainer_crash_is_not_discarded_by_teardown():
    """The re-check added for planned teardown must not swallow a real crash.

    Teardown kills the vLLM subprocesses, so their sentinels fire exactly like a
    crash and have to be re-checked against `shutdown_event`. The trainer is
    different: teardown only *joins* that thread, so `trainer_failed` is never a
    consequence of shutting down. Applying the re-check to it as well downgraded
    a trainer that died on the final step to a clean exit 0.
    """
    trainer_failed = threading.Event()
    trainer_failed.set()
    abort = AbortReason()

    # is_set() is read by the `while`, then the post-wait guard. Teardown lands
    # right after, which used to discard the crash the watchdog had just seen.
    shutdown = mock.MagicMock()
    shutdown.is_set.side_effect = [False, False, True, True, True]

    with (
        mock.patch.object(split.multiprocessing.connection, "wait", return_value=[]),
        mock.patch.object(split.os, "kill") as killed,
    ):
        split._watch_components([], trainer_failed, shutdown, abort)

    assert abort.reason, "a trainer crash must survive a concurrent teardown"
    assert killed.called


def test_a_colocate_interrupt_marks_the_shutdown_planned():
    """A terminal Ctrl-C also reaches vLLM's EngineCore children, which colocate
    does not `setsid`. They die, the vLLM thread sets `error_event`, and the
    watchdog would report a user stop as a component crash. The SIGINT handler
    sets `shutdown_event` first so the watchdog returns quietly."""
    error_event = threading.Event()
    shutdown = threading.Event()
    abort = AbortReason()

    # What the handler does, then what the watchdog sees afterwards.
    shutdown.set()
    error_event.set()

    with mock.patch.object(colocate.os, "kill") as killed:
        colocate._watch_components(error_event, shutdown, abort)

    assert abort.reason is None, "an interrupt is not a crash"
    assert not killed.called


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
