"""Split GRPO runner: the inference server and the trainer occupy disjoint sets of GPUs.

Unlike the co-locate runner, the server is launched as a separate process so it can be
given its own ``CUDA_VISIBLE_DEVICES``. The trainer runs in the parent process, which has
already had its own ``CUDA_VISIBLE_DEVICES`` set by the CLI before any CUDA-touching
import happens. Weight broadcast goes through the filesystem -- there is no shared GPU
memory between the two GPU sets.

Startup sequence (all components launch concurrently):
    1. Spawn the rollout server subprocess with ``CUDA_VISIBLE_DEVICES=<infer gpu ids>``.
    2. (Optional) Spawn a judge server subprocess with ``CUDA_VISIBLE_DEVICES=<judge gpu ids>``
       when ``orch_config.ruler.enabled`` and judge args are provided. Lets a single
       ``surogate grpo`` invocation own the entire RULER training topology -- no
       separate ``surogate grpo-infer`` for the judge.
    3. Start the surogate trainer in a background thread (parent process,
       ``CUDA_VISIBLE_DEVICES=<trainer gpu ids>`` already applied by the CLI).
    4. Run the orchestrator in the main async event loop. The orchestrator
       polls the server's /health internally and blocks until inference is ready.

Module-level imports here MUST stay free of CUDA-touching modules: this module is
re-imported in the spawned child, where nothing may touch a device until
``CUDA_VISIBLE_DEVICES`` has been re-set.
"""

import asyncio
import ctypes
import multiprocessing as mp
import multiprocessing.connection
import logging
import os
import signal
import sys
import threading
from threading import Thread

import psutil

from surogate.core.config.grpo_inference_config import GRPOInferenceConfig
from surogate.core.config.grpo_orch_config import GRPOOrchestratorConfig
from surogate.grpo.abort import AbortReason
from surogate.grpo.config import GRPOTrainConfig
from surogate.utils.logger import get_logger

logger = get_logger()


def _run_inference_subprocess(infer_config: GRPOInferenceConfig, infer_gpu_ids: str):
    """Entry point for the spawned inference subprocess.

    Sets ``CUDA_VISIBLE_DEVICES`` BEFORE anything touches a device. ``infer_gpu_ids``
    is a comma-separated string of real (driver-level) GPU indices -- the parent's
    mask does not propagate, because we override the env var here.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = infer_gpu_ids

    # Become a new session leader so anything the engine spawns shares our process
    # group. Lets the parent kill the whole tree atomically via os.killpg() on
    # shutdown, and detaches us from the terminal so terminal SIGINT no longer
    # reaches us -- the parent owns lifecycle.
    os.setsid()

    from surogate.grpo.inference.grpo_infer import grpo_infer

    grpo_infer(infer_config)


def _run_trainer(train_config: GRPOTrainConfig, failure_event: threading.Event):
    """Run the GRPO trainer in a background thread (parent process).

    Sets `failure_event` on any exception so the watchdog can interrupt the
    orchestrator. Re-raising here would be silently dropped by daemon-thread
    teardown -- the event is the propagation channel to the main thread.
    """
    try:
        # Inside the try: an ImportError here is a crash like any other, and
        # leaving it outside meant the most likely failure on a dependency bump
        # set no event, raised no signal, and hung the run in `running`.
        from surogate.grpo.trainer import GRPOTrainer

        GRPOTrainer(train_config).train()
    except Exception:
        # Set the channel FIRST: this used to log first, and the logging call
        # itself raised, so the line below never ran, the watchdog was never
        # told, and the run hung in `running` holding its GPUs. Whatever goes
        # wrong in logging, the signal must already be out.
        failure_event.set()
        logger.exception("Trainer thread crashed")


def _watch_components(
    server_procs: list[tuple["mp.Process", str]],
    trainer_failed: threading.Event,
    shutdown_event: threading.Event,
    abort: AbortReason,
) -> None:
    """Watchdog: aborts the pipeline if any server or the trainer dies unexpectedly.

    Sleeps on the union of all server sentinels with a 1s timeout so we can also
    poll the trainer's failure flag. On unexpected death we deliver SIGINT to
    the main thread, which has the existing SIGINT/KeyboardInterrupt teardown path.

    ``server_procs`` is a list of (process, label) tuples so the crash message can
    distinguish the rollout server's death from the judge's.
    """
    sentinels = [proc.sentinel for proc, _ in server_procs]
    crashed: str | None = None
    while not shutdown_event.is_set():
        ready = multiprocessing.connection.wait(sentinels, timeout=1.0)
        if shutdown_event.is_set():
            return
        for proc, label in server_procs:
            if proc.sentinel in ready:
                crashed = f"{label} subprocess died unexpectedly (exitcode={proc.exitcode})"
                break
        if crashed:
            # Only a sentinel can fire as a side effect of planned teardown, which
            # kills the server subprocesses; re-check before calling that a crash, or
            # a successful run gets marked failed. The trainer branch below needs
            # no such re-check: teardown only *joins* the trainer thread, so
            # `trainer_failed` is never a consequence of shutting down, and
            # discarding it here would downgrade a real late crash to exit 0.
            if shutdown_event.is_set():
                return
            break
        if trainer_failed.is_set():
            crashed = "Trainer thread crashed"
            break
    if crashed is None:
        return
    logger.error(f"{crashed} -- aborting GRPO pipeline")
    # Record before signalling: the main thread reads this to tell our own
    # abort apart from a user's Ctrl-C, and it must be set by the time the
    # signal lands.
    abort.reason = crashed
    # Wake the main thread out of asyncio.run; existing finally block tears down.
    os.kill(os.getpid(), signal.SIGINT)


def grpo_split(
    train_config: GRPOTrainConfig,
    infer_config: GRPOInferenceConfig,
    orch_config: GRPOOrchestratorConfig,
    infer_gpu_ids: list[int],
    trainer_gpu_ids: list[int],
    judge_infer_config: GRPOInferenceConfig | None = None,
    judge_gpu_ids: list[int] | None = None,
):
    """Run the full GRPO pipeline with the server and the trainer on disjoint GPU sets.

    When ``judge_infer_config`` and ``judge_gpu_ids`` are both provided AND
    ``orch_config.ruler.enabled`` is True, also spawns a second server serving
    the RULER judge model on the given GPU ids. This lets a single
    ``surogate grpo`` invocation own the entire RULER topology (rollout server +
    judge server + trainer + orchestrator). When the judge args are omitted, the
    orchestrator is expected to point at an externally-running judge, exactly
    as in the pre-judge-spawn workflow.

    The CLI is expected to have set ``os.environ["CUDA_VISIBLE_DEVICES"]`` to the
    trainer GPU ids BEFORE any CUDA-touching import. We do not set it again here
    because the parent has already imported the surogate stack by this point.
    """
    # Validate judge args FIRST so judge-config errors surface before unrelated
    # rollout/trainer overlap errors confuse the user.
    _validate_judge_args(
        judge_infer_config=judge_infer_config,
        judge_gpu_ids=judge_gpu_ids,
        orch_config=orch_config,
        infer_gpu_ids=infer_gpu_ids,
        trainer_gpu_ids=trainer_gpu_ids,
        rollout_infer_config=infer_config,
    )

    _check_disjoint_gpus(("--infer-gpus", infer_gpu_ids), ("--trainer-gpus", trainer_gpu_ids))
    _check_gpu_count_matches_topology("--infer-gpus", infer_gpu_ids, infer_config)
    if len(trainer_gpu_ids) != train_config.gpus:
        raise ValueError(f"--trainer-gpus has {len(trainer_gpu_ids)} GPU(s) but train.gpus = {train_config.gpus}")

    # QeRL noise lives in the base model's RMSNorm weights. The engine serves its
    # own resident base and hot-loads adapters only, so there is no way to hand it
    # a perturbed norm between steps; refusing here beats computing a sigma that
    # nothing ever applies.
    if train_config.noise_scheduler and train_config.noise_scheduler.enabled:
        raise ValueError(
            "noise_scheduler.enabled needs an inference server that applies QeRL noise to its "
            "base weights between steps, and the surogate engine does not"
        )

    spawn_judge = judge_infer_config is not None
    topology = f"inference GPUs: {infer_gpu_ids}, trainer GPUs: {trainer_gpu_ids}" + (
        f", judge GPUs: {judge_gpu_ids}" if spawn_judge else ""
    )
    logger.info(f"Starting GRPO pipeline (split mode) -- {topology}")

    # Become the reaper for our descendants BEFORE spawning any: workers that
    # reparent off a dead parent mid-run then land on us instead of PID 1, so the
    # shutdown child-tree walk can still find and kill them.
    _set_child_subreaper()

    # Spawn (not fork) so the child gets a fresh interpreter and doesn't inherit
    # any CUDA state from the parent's trainer-side GPU mask.
    ctx = mp.get_context("spawn")
    server_proc = _spawn_inference(ctx, infer_config, infer_gpu_ids, name="inference-subprocess")

    # SIGTERM -> SIGINT so a plain `kill <pid>` (and orchestrator code that calls
    # signal.raise_signal(SIGTERM) on stop) takes the same KeyboardInterrupt path
    # that Ctrl-C does, ensuring the finally block tears down the server tree.
    signal.signal(signal.SIGTERM, lambda *_: signal.raise_signal(signal.SIGINT))

    judge_proc: mp.Process | None = None
    if spawn_judge:
        assert judge_infer_config is not None and judge_gpu_ids is not None  # for type narrowing
        judge_proc = _spawn_inference(ctx, judge_infer_config, judge_gpu_ids, name="judge-subprocess")

    components = "rollout server, " + ("judge server, " if judge_proc is not None else "") + "trainer, and orchestrator"
    logger.info(f"Spawning {components} in parallel")
    server_proc.start()
    if judge_proc is not None:
        judge_proc.start()

    trainer_failed = threading.Event()
    trainer_thread = Thread(
        target=_run_trainer,
        args=(train_config, trainer_failed),
        daemon=True,
        name="grpo-trainer",
    )
    trainer_thread.start()

    # Watchdog watches every server we spawned plus the trainer thread. shutdown_event
    # suppresses spurious aborts during the planned teardown in the finally block.
    watched_servers: list[tuple[mp.Process, str]] = [(server_proc, "rollout server")]
    if judge_proc is not None:
        watched_servers.append((judge_proc, "judge server"))
    shutdown_event = threading.Event()
    abort = AbortReason()
    watchdog_thread = Thread(
        target=_watch_components,
        args=(watched_servers, trainer_failed, shutdown_event, abort),
        daemon=True,
        name="grpo-watchdog",
    )
    watchdog_thread.start()

    try:
        from surogate.grpo.orchestrator.grpo_orch import orchestrate

        asyncio.run(orchestrate(orch_config))
    except KeyboardInterrupt:
        # A watchdog abort arrives here as an ordinary Ctrl-C (see
        # `AbortReason`); the recorded reason is what tells them apart. The
        # watchdog has already logged the cause and the raise below carries it,
        # so only a genuine interrupt needs a line.
        if not abort.reason:
            logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"Split GRPO pipeline error: {e}")
        raise
    finally:
        # Mask FIRST, before anything that can be interrupted. These two lines
        # used to sit below the log and the event set, leaving a window where a
        # watchdog SIGINT would raise inside the `finally` itself -- skipping the
        # server reap entirely and stranding the process trees on their GPUs,
        # which is the exact outcome the trailing raise is placed after teardown
        # to avoid. Also covers a second Ctrl-C during cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        logger.info("Split GRPO pipeline shutting down")

        # Tell the watchdog this teardown is intentional; otherwise it would see
        # the server's planned death and re-fire SIGINT.
        shutdown_event.set()

        # Reap servers in parallel -- each can take up to 15s on SIGTERM->SIGKILL
        # escalation. Sequential teardown would double that on a typical RULER
        # topology. This is the graceful path: killpg lets each session release
        # its GPU cleanly.
        _terminate_inference_trees_in_parallel(watched_servers)

        # Trainer runs as a daemon thread; a brief join lets it flush logs, after
        # which it dies with the process. No reason to block for 30s here.
        trainer_thread.join(timeout=2.0)
        if trainer_thread.is_alive():
            logger.warning("Trainer thread still running; will exit with the process")

        watchdog_thread.join(timeout=2.0)

        # Reap anything the graceful teardown left behind so nothing keeps the
        # process tree -- and the pod -- alive. We set PR_SET_CHILD_SUBREAPER at
        # startup, so subprocesses that escaped their group or reparented off a
        # dead parent (server workers, and orchestrator env-server subprocess
        # trees) landed on us and show up in this recursive walk, where killpg
        # could not reach them. The trainer is an in-process C++ engine
        # (multi-GPU handled internally, no OS subprocess children of its own), so
        # reaping every descendant is safe even if its thread is still winding down.
        # Guard the whole sweep: it is the last statement in finally, so an error
        # here (e.g. psutil.AccessDenied in a hardened sandbox) must not replace the
        # original pipeline exception or skip cleanup.
        try:
            _reap_survivors(psutil.Process().children(recursive=True))
        except Exception as e:
            logger.warning(f"Survivor reap failed during shutdown: {e}")

        # Leave without running interpreter finalization -- on the clean path.
        #
        # The trainer is a daemon thread and may still be winding down -- the join
        # above waits two seconds and then says so. If it holds the stderr buffer's
        # lock when CPython finalizes, the interpreter aborts with
        # `_enter_buffered_busy` and the process exits 134, so a run that finished
        # every step and wrote every checkpoint reports itself as crashed. Every
        # subprocess has been reaped by this point and the only work left is
        # finalization itself, so flush what we own and go.
        #
        # An abort does not take this door: the raise below is what turns a dead
        # component into a non-zero exit, and exiting 0 here would hide it.
        if not abort.reason:
            logging.shutdown()
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:
                    pass
            os._exit(0)

    # After teardown, not instead of it: the finally block above reaps the server
    # process trees, and exiting early would strand them holding GPUs. Raising
    # here is what turns a dead component into a non-zero exit, so ops records
    # the run as failed rather than completed.
    if abort.reason:
        raise RuntimeError(f"GRPO pipeline aborted: {abort.reason}")


def _spawn_inference(
    ctx: mp.context.SpawnContext, infer_config: GRPOInferenceConfig, gpu_ids: list[int], *, name: str
) -> mp.Process:
    """Build (don't yet start) an inference subprocess pinned to the given GPU ids."""
    return ctx.Process(
        target=_run_inference_subprocess,
        args=(infer_config, ",".join(str(g) for g in gpu_ids)),
        name=name,
        daemon=False,
    )


def _check_disjoint_gpus(*labeled_sets: tuple[str, list[int]]) -> None:
    """Raise ValueError if any pair in ``labeled_sets`` shares a GPU id."""
    for i, (a_label, a_ids) in enumerate(labeled_sets):
        for b_label, b_ids in labeled_sets[i + 1 :]:
            overlap = sorted(set(a_ids) & set(b_ids))
            if overlap:
                raise ValueError(f"{a_label} and {b_label} overlap on {overlap}")


def _check_gpu_count_matches_topology(label: str, gpu_ids: list[int], infer_config: GRPOInferenceConfig) -> None:
    """Raise ValueError if ``len(gpu_ids) != dp * tp`` of the inference config."""
    expected = (infer_config.dp or 1) * (infer_config.tp or 1)
    if len(gpu_ids) != expected:
        raise ValueError(
            f"{label} has {len(gpu_ids)} GPU(s) but inference config implies "
            f"dp({infer_config.dp}) * tp({infer_config.tp}) = {expected}"
        )


def _validate_judge_args(
    *,
    judge_infer_config: GRPOInferenceConfig | None,
    judge_gpu_ids: list[int] | None,
    orch_config: GRPOOrchestratorConfig,
    infer_gpu_ids: list[int],
    trainer_gpu_ids: list[int],
    rollout_infer_config: GRPOInferenceConfig,
) -> None:
    """Validate the judge spawn args.

    Both ``judge_infer_config`` and ``judge_gpu_ids`` must be set together (or
    both omitted; this defensive check catches non-CLI callers -- the CLI layer
    rejects partial args earlier with a friendlier error). When set, RULER must
    be enabled, the judge GPU set must be disjoint from the rollout-server and
    trainer GPUs, the GPU count must match ``dp * tp``, and the judge port
    must differ from the rollout port.
    """
    if (judge_infer_config is None) != (judge_gpu_ids is None):
        raise ValueError(
            "--judge-infer and --judge-gpus must be provided together (or both omitted); "
            f"got judge_infer_config={'set' if judge_infer_config else 'unset'}, "
            f"judge_gpu_ids={'set' if judge_gpu_ids else 'unset'}"
        )
    if judge_infer_config is None:
        return

    assert judge_gpu_ids is not None
    if not orch_config.ruler or not orch_config.ruler.enabled:
        raise ValueError(
            "Judge spawn args were provided but orch_config.ruler.enabled is False. "
            "Either enable RULER (orch.yaml: ruler.enabled: true) or omit --judge-infer/--judge-gpus."
        )
    if not judge_gpu_ids:
        raise ValueError("--judge-gpus must list at least one GPU id")
    _check_disjoint_gpus(
        ("--judge-gpus", judge_gpu_ids),
        ("--infer-gpus", infer_gpu_ids),
        ("--trainer-gpus", trainer_gpu_ids),
    )
    _check_gpu_count_matches_topology("--judge-gpus", judge_gpu_ids, judge_infer_config)
    if judge_infer_config.port == rollout_infer_config.port:
        raise ValueError(
            f"Judge and rollout servers cannot share port {judge_infer_config.port}. "
            "Set a different `port` in the judge inference config (e.g. 8001)."
        )


# From <linux/prctl.h>. Not exposed by the stdlib, so we pass the raw constant
# to prctl(2) via ctypes.
_PR_SET_CHILD_SUBREAPER = 36


def _set_child_subreaper() -> None:
    """Mark this process as the reaper for its orphaned descendants (Linux).

    Workers a server spawns reparent to PID 1 when their parent exits mid-run,
    escaping both a teardown-time process-tree walk and a ``killpg`` on the
    server's session group. ``PR_SET_CHILD_SUBREAPER`` makes such orphans
    reparent to *us* instead, so a single ``children(recursive=True)`` walk at
    shutdown still finds them and they can be reaped by PID.
    ``prctl``/``PR_SET_CHILD_SUBREAPER`` is Linux-only, so on any other platform
    this is a no-op (the graceful ``killpg`` teardown still runs); on Linux it is
    best-effort, logging and continuing if the call fails in a restricted sandbox.

    Adopted descendants that die mid-run become zombies under us until the
    shutdown reap (or ``init`` when we exit) clears them. We deliberately do NOT
    install a ``SIGCHLD`` reaper for them -- a blanket ``waitpid(-1)`` would race
    ``multiprocessing`` for its own children and corrupt its bookkeeping -- and
    the accumulation is bounded by mid-run subprocess churn.
    """
    if sys.platform != "linux":
        return
    try:
        # CDLL(None) resolves against the already-loaded libc, so we don't hardcode
        # a SONAME (``libc.so.6`` is glibc-only; musl uses a different name).
        libc = ctypes.CDLL(None, use_errno=True)
        # prctl(2) takes int + four unsigned longs; declare them so ctypes doesn't
        # pass 32-bit c_int (undefined on LP64 arches like AArch64).
        libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        libc.prctl.restype = ctypes.c_int
        if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            errno = ctypes.get_errno()
            logger.warning(f"PR_SET_CHILD_SUBREAPER failed (errno={errno}); reparented workers may survive shutdown")
    except (OSError, AttributeError) as e:
        logger.warning(f"Could not set child subreaper ({e}); reparented workers may survive shutdown")


def _reap_survivors(procs: list["psutil.Process"], *, grace: float = 5.0) -> None:
    """SIGTERM then SIGKILL any of ``procs`` still alive, addressing them by PID.

    ``procs`` is the recursive child snapshot taken at shutdown. Because we set
    ``PR_SET_CHILD_SUBREAPER`` at startup, workers that escaped a server's session
    group and reparented off their dead parent have reparented to *us*, so they
    appear in that walk; killing by PID reaches them where a ``killpg`` on the
    server's group cannot.
    """
    alive = [p for p in procs if p.is_running()]
    if not alive:
        return
    # Guard on psutil.Error (not just NoSuchProcess): a process can die between
    # the snapshot and here (NoSuchProcess/ZombieProcess), and a signal can be
    # refused (AccessDenied). Neither should abort the sweep of the rest.
    for p in alive:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, still_alive = psutil.wait_procs(alive, timeout=grace)
    for p in still_alive:
        # ``name()`` reads /proc and raises ZombieProcess/NoSuchProcess if the
        # process died (as an adopted child would, becoming a zombie under us)
        # between the wait and here -- so both it and kill() must be guarded, or
        # the exception escapes the finally block and aborts shutdown.
        try:
            logger.warning(f"Force-killing leftover subprocess pid={p.pid} ({p.name()})")
            p.kill()
        except psutil.Error:
            pass


def _terminate_inference_trees_in_parallel(watched: list[tuple["mp.Process", str]]) -> None:
    """SIGTERM all server subprocesses concurrently and wait for them in parallel.

    Each call to ``_terminate_inference_tree`` can block up to ~15s waiting for
    the SIGTERM->SIGKILL escalation. With two servers (rollout + judge), serial
    teardown doubles shutdown latency for no reason. Threads are sufficient
    here -- the work is mostly waiting on ``Process.join``.
    """
    if len(watched) <= 1:
        for proc, label in watched:
            _terminate_inference_tree(proc, label=label)
        return
    threads = [
        Thread(target=_terminate_inference_tree, args=(proc,), kwargs={"label": label}, name=f"reap-{label}")
        for proc, label in watched
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _signal_inference(pid: int, sig: int) -> None:
    """Signal the server's process group, falling back to the bare PID.

    After the child ``setsid()``s, ``pid == pgid`` so ``killpg`` reaches the whole
    session. But if teardown races startup and fires before the child ran
    ``setsid()``, no group with that id exists yet and ``killpg`` raises
    ``ProcessLookupError`` -- the child is then still a lone process in our group,
    so signal it directly by PID (it has no workers yet). No-op if already gone.
    """
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _terminate_inference_tree(server_proc: "mp.Process", *, label: str = "inference server", grace: float = 10.0) -> None:
    """Graceful teardown of a server subprocess and its session group.

    The subprocess ``setsid()``s so anything it spawns shares its process group.
    We SIGTERM the group, wait ``grace`` for the leader to exit, and escalate to
    SIGKILL on the whole group only if the leader is still alive after the grace
    (a hung server). This is the fast, polite path that lets a responsive server
    release its GPU cleanly -- not the orphan catch. Workers that outlive a
    cleanly-exited leader, escaped the session group, or reparented off a dead
    parent are swept up by the subreaper-backed descendant reap in
    ``grpo_split``'s finally block. Escalating only when the leader survives the
    grace keeps the SIGKILL off a ``pgid`` that ``join`` already reaped (which the
    OS could have recycled) and preserves the "did not exit" diagnostic.
    """
    if not server_proc.is_alive() or server_proc.pid is None:
        return
    pid = server_proc.pid  # pid == pgid once the subprocess has setsid'd
    logger.info(f"Terminating {label} subprocess group")
    _signal_inference(pid, signal.SIGTERM)
    server_proc.join(timeout=grace)
    if server_proc.is_alive():
        logger.warning(f"{label} did not exit in {grace:.0f}s; sending SIGKILL to process group")
        _signal_inference(pid, signal.SIGKILL)
        server_proc.join(timeout=5.0)
