"""A GRPO component failure must become a non-zero exit status.

Two halves of one gap. The trainer's crash handler called `logger.exception`,
which `LoggerWrapper` does not define, so the handler raised inside itself and
the `failure_event.set()` on the next line never ran: the watchdog was never
told and the run hung in `running` forever, holding its GPUs.

And when the watchdog *does* abort, it does so by signalling its own process,
which the main thread could not tell apart from a user pressing Ctrl-C. It
swallowed the interrupt, returned normally, and the process exited 0 — so ops
finalized a crashed run as `completed` with a null error.
"""

import sys
import threading
from unittest import mock

import pytest

#  reaches the compiled extension through its config import, which
# has nothing to do with the failure plumbing under test here. Use the real one
# where it is built (CI) and a stand-in where it is not, so these stay runnable
# without a GPU toolchain.
try:  # pragma: no cover - depends on the build environment
    import surogate._surogate  # noqa: F401
except ImportError:  # pragma: no cover
    import surogate

    _stub = mock.MagicMock()
    sys.modules["surogate._surogate"] = _stub
    surogate._surogate = _stub

from surogate.grpo import split  # noqa: E402
from surogate.utils.logger import get_logger  # noqa: E402


def test_the_logger_still_has_no_exception_method():
    """Pins the premise. If `LoggerWrapper` ever grows `.exception`, this fails
    and the handler below can be simplified — until then, using it is a bug."""
    assert not hasattr(get_logger(), "exception")


def _crashing_trainer_module():
    """`_run_trainer` imports the trainer inside the function, so swapping the
    module out is enough — and avoids importing the real one, which drags in
    the whole engine."""
    fake = mock.MagicMock()
    fake.GRPOTrainer.side_effect = RuntimeError("boom")
    return fake


def test_a_trainer_crash_sets_the_failure_event():
    ev = threading.Event()
    with mock.patch.dict(sys.modules, {"surogate.grpo.trainer": _crashing_trainer_module()}):
        split._run_trainer(mock.MagicMock(), ev)
    assert ev.is_set(), "the watchdog's only propagation channel must fire"


def test_the_event_is_set_even_when_logging_fails():
    """The regression test for the original bug.

    The handler logged first and set the event second, so anything wrong with
    the logging call cost us the signal entirely. Ordering is the durable fix:
    set the channel first, then log.
    """
    ev = threading.Event()
    broken = mock.MagicMock()
    broken.error.side_effect = AttributeError("no such method")
    with (
        mock.patch.dict(sys.modules, {"surogate.grpo.trainer": _crashing_trainer_module()}),
        mock.patch.object(split, "logger", broken),
    ):
        with pytest.raises(AttributeError):
            split._run_trainer(mock.MagicMock(), ev)
    assert ev.is_set(), "a broken logger must not cost us the failure signal"


def test_the_watchdog_records_why_it_aborted():
    """The main thread needs to tell a watchdog abort from a real Ctrl-C, so
    the reason has to be recorded before the signal is delivered."""
    trainer_failed = threading.Event()
    trainer_failed.set()
    shutdown = threading.Event()
    abort = split.AbortReason()

    with mock.patch.object(split.os, "kill") as killed:
        split._watch_components([], trainer_failed, shutdown, abort)

    assert abort.reason, "the abort must leave a reason behind"
    assert "rainer" in abort.reason
    assert killed.called, "and it still signals the main thread"


def test_a_clean_shutdown_records_nothing():
    shutdown = threading.Event()
    shutdown.set()
    abort = split.AbortReason()

    with mock.patch.object(split.os, "kill") as killed:
        split._watch_components([], threading.Event(), shutdown, abort)

    assert abort.reason is None
    assert not killed.called


def test_an_abort_reason_is_a_failure_not_an_interrupt():
    """A recorded reason means a component died: the process must exit
    non-zero. An empty one means the user pressed Ctrl-C, which stays 0."""
    assert split._exit_code_for(split.AbortReason("rollout vLLM died")) != 0
    assert split._exit_code_for(split.AbortReason()) == 0
