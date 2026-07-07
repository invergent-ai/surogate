"""Tests for GRPO-shutdown subprocess reaping in `surogate.grpo.split`.

The vLLM subprocess `setsid()`s so its immediate tree shares one process group,
but some workers (the vLLM `EngineCore`, the `multiprocessing.resource_tracker`)
end up outside that group, and when their parent dies mid-run they reparent to
PID 1 — so a `killpg` on the vLLM group misses them and a plain child-tree walk
can no longer find them. Shutdown must still leave nothing behind, else those
workers linger orphaned and strand the run.

The catch is `PR_SET_CHILD_SUBREAPER`: the pipeline marks itself a subreaper at
startup, so an orphaned descendant reparents to *it* instead of PID 1 and still
shows up in a recursive `children()` walk at shutdown, where it is reaped by PID.
"""

import multiprocessing as mp
import os
import signal
import time
from unittest import mock

import psutil

from surogate.grpo.split import (
    _reap_survivors,
    _set_child_subreaper,
    _terminate_vllm_tree,
)


def _signal_ready(ready) -> None:
    if ready is not None:
        ready.send(b"1")
        ready.close()


def _escaped_stubborn(ready=None):
    """A session leader that ignores SIGTERM, signalling once its handler is set.

    Models both a real escaped orphan (own group, so `killpg` can't reach it)
    and a hung vLLM leader that only SIGKILL can end. The readiness signal fires
    *after* the SIG_IGN handler is installed, so a test never races SIGTERM
    against an un-armed handler (which would let the process die on the default
    disposition and pass without exercising the SIGKILL escalation).
    """
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _signal_ready(ready)
    time.sleep(30)


def _responsive_leader(ready=None):
    """A vLLM-leader stand-in that exits on the default SIGTERM (the clean case)."""
    os.setsid()
    _signal_ready(ready)
    time.sleep(30)


def _start_ready(target):
    """Fork-start ``target`` and block until it signals its setup is complete."""
    ours, theirs = mp.Pipe()
    proc = mp.get_context("fork").Process(target=target, args=(theirs,))
    proc.start()
    theirs.close()
    assert ours.poll(timeout=5), "child never signalled ready"
    ours.recv()
    ours.close()
    return proc


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_group_gone(pgid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _group_alive(pgid):
            return True
        time.sleep(0.1)
    return False


def _kill_tree(pid: int) -> None:
    """Best-effort cleanup: SIGKILL a process and every descendant."""
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def test_reap_survivors_kills_processes_that_escaped_the_group():
    procs = [_start_ready(_escaped_stubborn) for _ in range(2)]
    try:
        snapshot = [psutil.Process(p.pid) for p in procs]
        assert all(s.is_running() for s in snapshot), "precondition: workers should be alive"

        _reap_survivors(snapshot, grace=0.5)

        for p in procs:
            p.join(timeout=5)
            assert not p.is_alive(), "an escaped SIGTERM-ignoring worker must be reaped by PID"
    finally:
        for p in procs:
            _kill_tree(p.pid)


def test_reap_survivors_is_a_noop_when_nothing_survives():
    _reap_survivors([], grace=0.5)  # must not raise


def test_terminate_vllm_tree_sigterms_a_responsive_leader():
    proc = _start_ready(_responsive_leader)
    try:
        pgid = proc.pid
        assert _group_alive(pgid), "precondition: the group should be alive"

        _terminate_vllm_tree(proc, grace=0.5)  # SIGTERM alone should end a responsive leader

        assert not proc.is_alive()
        assert _wait_group_gone(pgid), "the group must empty on a clean SIGTERM exit"
    finally:
        _kill_tree(proc.pid)


def test_terminate_vllm_tree_sigkills_a_stubborn_leader():
    proc = _start_ready(_escaped_stubborn)  # ignores SIGTERM
    try:
        pgid = proc.pid
        assert _group_alive(pgid), "precondition: the group should be alive"

        _terminate_vllm_tree(proc, grace=0.5)  # SIGTERM ignored -> escalate to SIGKILL

        assert not proc.is_alive(), "a leader that ignores SIGTERM must be SIGKILLed after the grace"
        assert _wait_group_gone(pgid)
    finally:
        _kill_tree(proc.pid)


def test_terminate_vllm_tree_is_a_noop_for_an_unstarted_process():
    proc = mp.get_context("fork").Process(target=time.sleep, args=(0,))
    _terminate_vllm_tree(proc, grace=0.5)  # never started: pid is None -> early return


def test_set_child_subreaper_is_best_effort_when_prctl_unavailable():
    # A platform without a usable libc (CDLL raises) must degrade, not crash.
    with mock.patch("surogate.grpo.split.ctypes.CDLL", side_effect=OSError("no libc")):
        _set_child_subreaper()  # must not raise


def _subreaper_parent(pipe):
    """Become a subreaper, then orphan a grandchild by exiting its parent.

    Mirrors ``grpo_split``: after ``PR_SET_CHILD_SUBREAPER`` a grandchild whose
    intermediate parent exits reparents to *this* process (the subreaper), not
    PID 1 — so a recursive ``children()`` walk from here still finds it. Runs in
    a forked subprocess so the prctl never touches the test runner itself.
    """
    _set_child_subreaper()
    if os.fork() == 0:  # intermediate
        gpid = os.fork()
        if gpid == 0:  # grandchild
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(30)
            os._exit(0)
        pipe.send(gpid)
        pipe.close()
        os._exit(0)  # exit -> grandchild reparents to the subreaper (us)
    pipe.close()
    time.sleep(30)  # stay alive as the reaper


def _ppid_of(pid: int):
    try:
        return psutil.Process(pid).ppid()
    except psutil.NoSuchProcess:
        return None


def test_child_subreaper_recaptures_a_reparented_grandchild():
    parent_conn, child_conn = mp.Pipe()
    parent = mp.get_context("fork").Process(target=_subreaper_parent, args=(child_conn,))
    parent.start()
    child_conn.close()  # only the subreaper subtree keeps the write end now
    try:
        assert parent_conn.poll(timeout=5), "subreaper child never reported the grandchild pid"
        gpid = parent_conn.recv()  # grandchild pid, still under its intermediate
        # once the intermediate exits, the orphan reparents to the subreaper
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _ppid_of(gpid) not in (parent.pid, None):
            time.sleep(0.05)

        assert _ppid_of(gpid) == parent.pid, "orphan must reparent to the subreaper, not PID 1"
        found = {p.pid for p in psutil.Process(parent.pid).children(recursive=True)}
        assert gpid in found, "recursive children() walk must find the reparented orphan"
    finally:
        _kill_tree(parent.pid)
        parent.join(timeout=5)
