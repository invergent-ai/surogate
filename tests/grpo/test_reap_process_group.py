"""Tests for GRPO-shutdown subprocess reaping in `surogate.grpo.split`.

The vLLM subprocess `setsid()`s so its immediate tree shares one process group,
but some workers (the vLLM `EngineCore`, the `multiprocessing.resource_tracker`)
end up in the *main* process's group / reparented to PID 1, so a `killpg` on the
vLLM group misses them. Shutdown must still leave nothing behind, else those
workers linger orphaned and strand the run.

The robust catch is a monitor thread that unions the descendant tree over the
whole run: a worker whose parent dies mid-run reparents to PID 1 and vanishes
from a one-shot teardown snapshot, but the monitor already recorded it while it
was still a descendant, so it can be reaped by PID at shutdown.
"""

import multiprocessing as mp
import os
import signal
import threading
import time
from threading import Thread

import psutil

from surogate.grpo.split import (
    _monitor_descendants,
    _reap_survivors,
    _snapshot_descendants,
    _terminate_vllm_tree,
)


def _escaped_stubborn():
    """A worker that `setsid`s into its OWN group and ignores SIGTERM.

    Models the real orphans, which escape the vLLM session group so `killpg`
    cannot reach them.
    """
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(30)


def _leader_with_ingroup_worker():
    """A session leader whose forked worker stays in the group and ignores SIGTERM."""
    os.setsid()
    if os.fork() == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(30)
        os._exit(0)
    time.sleep(30)


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


def test_reap_survivors_kills_processes_that_escaped_the_group():
    procs = [mp.get_context("fork").Process(target=_escaped_stubborn) for _ in range(2)]
    for p in procs:
        p.start()
    time.sleep(0.4)  # let them setsid + install the SIGTERM handler
    snapshot = [psutil.Process(p.pid) for p in procs]
    assert all(s.is_running() for s in snapshot), "precondition: workers should be alive"

    _reap_survivors(snapshot, grace=0.5)

    for p in procs:
        p.join(timeout=5)
        assert not p.is_alive(), "an escaped SIGTERM-ignoring worker must be reaped by PID"


def test_reap_survivors_is_a_noop_when_nothing_survives():
    _reap_survivors([], grace=0.5)  # must not raise


def test_terminate_vllm_tree_reaps_a_worker_in_the_group():
    proc = mp.get_context("fork").Process(target=_leader_with_ingroup_worker)
    proc.start()
    time.sleep(0.5)
    pgid = proc.pid
    assert _group_alive(pgid), "precondition: the group should be alive"

    _terminate_vllm_tree(proc, grace=0.5)

    assert not proc.is_alive()
    assert _wait_group_gone(pgid), "the in-group worker must be reaped; group must empty"


def test_terminate_vllm_tree_is_a_noop_for_an_unstarted_process():
    proc = mp.get_context("fork").Process(target=time.sleep, args=(0,))
    _terminate_vllm_tree(proc, grace=0.5)  # never started: pid is None -> early return


def _forks_a_grandchild_then_exits(pipe):
    """Child: fork a stubborn grandchild, report its pid, then exit.

    After this child exits the grandchild reparents to PID 1 and leaves the test
    process's live descendant tree — the exact shape of the real vLLM spawn
    worker (a multiprocessing spawn child of an EngineCore) that a one-shot
    teardown snapshot misses because its parent has already died.
    """
    gpid = os.fork()
    if gpid == 0:  # grandchild
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(30)
        os._exit(0)
    pipe.send(gpid)
    pipe.close()
    time.sleep(0.6)  # stay alive long enough for the monitor to snapshot us
    os._exit(0)  # exit -> grandchild reparents to PID 1


def _alive_and_not_zombie(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_snapshot_records_a_descendant_still_in_the_tree():
    proc = mp.get_context("fork").Process(target=_escaped_stubborn)
    proc.start()
    time.sleep(0.3)
    registry: dict = {}

    _snapshot_descendants(registry)

    assert any(k[0] == proc.pid for k in registry), "a live descendant must be recorded"
    _reap_survivors([registry[k] for k in registry if k[0] == proc.pid], grace=0.5)
    proc.join(timeout=5)
    assert not proc.is_alive()


def test_monitor_reaps_a_descendant_that_reparents_before_teardown():
    parent_conn, child_conn = mp.Pipe()
    registry: dict = {}
    stop = threading.Event()
    monitor = Thread(target=_monitor_descendants, args=(registry, stop), kwargs={"interval": 0.1})
    monitor.start()

    child = mp.get_context("fork").Process(target=_forks_a_grandchild_then_exits, args=(child_conn,))
    child.start()
    gpid = parent_conn.recv()  # grandchild pid, while both are still our descendants
    child.join(timeout=5)  # child exits -> grandchild reparents to PID 1
    time.sleep(0.2)
    stop.set()
    monitor.join(timeout=2)

    # The grandchild left the live tree when its parent died, but the monitor
    # recorded it earlier, so the union still holds it.
    key = next((k for k in registry if k[0] == gpid), None)
    assert key is not None, "monitor must record a descendant before it reparents to PID 1"
    assert _alive_and_not_zombie(gpid), "precondition: grandchild should still be alive"

    _reap_survivors([registry[key]], grace=0.5)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _alive_and_not_zombie(gpid):
        time.sleep(0.1)
    assert not _alive_and_not_zombie(gpid), "a reparented descendant must still be reaped"
