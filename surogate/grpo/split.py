"""Split GRPO runner: vLLM and the trainer occupy disjoint sets of GPUs.

Unlike the co-locate runner, vLLM is launched as a separate Python subprocess so it
can be given its own ``CUDA_VISIBLE_DEVICES``. The trainer runs in the parent process,
which has already had its own ``CUDA_VISIBLE_DEVICES`` set by the CLI before any torch
import happens. Weight broadcast goes through the filesystem backend — there is no
shared GPU memory between the two GPU sets.

Startup sequence (all components launch concurrently):
    1. Spawn rollout vLLM subprocess with ``CUDA_VISIBLE_DEVICES=<vllm gpu ids>``.
    2. (Optional) Spawn judge vLLM subprocess with ``CUDA_VISIBLE_DEVICES=<judge gpu ids>``
       when ``orch_config.ruler.enabled`` and judge args are provided. Lets a single
       ``surogate grpo`` invocation own the entire RULER training topology — no
       separate ``surogate grpo-infer`` for the judge.
    3. Start the surogate trainer in a background thread (parent process,
       ``CUDA_VISIBLE_DEVICES=<trainer gpu ids>`` already applied by the CLI).
    4. Run the orchestrator in the main async event loop. The orchestrator
       polls vLLM /health internally and blocks until inference is ready.

Module-level imports here MUST stay torch-free: this module is re-imported in the
spawned vLLM child, where torch must not be loaded until ``CUDA_VISIBLE_DEVICES``
has been re-set.
"""

import asyncio
import multiprocessing as mp
import multiprocessing.connection
import os
import signal
import threading
from threading import Thread

from surogate.core.config.grpo_inference_config import GRPOInferenceConfig
from surogate.core.config.grpo_orch_config import GRPOOrchestratorConfig
from surogate.grpo.config import GRPOTrainConfig
from surogate.utils.logger import get_logger

logger = get_logger()


def _run_vllm_subprocess(infer_config: GRPOInferenceConfig, vllm_gpu_ids: str):
    """Entry point for the spawned vLLM subprocess.

    Sets ``CUDA_VISIBLE_DEVICES`` BEFORE importing torch/vLLM. ``vllm_gpu_ids`` is a
    comma-separated string of real (driver-level) GPU indices — the parent's mask
    does not propagate, because we override the env var here.
    """
    import os
    import sys

    os.environ["CUDA_VISIBLE_DEVICES"] = vllm_gpu_ids

    # vLLM's server() calls parser.parse_args() without an explicit args= argument,
    # so it falls back to sys.argv. Spawned children inherit the parent's argv,
    # which contains our split-mode flags (--vllm-gpus etc.) that vLLM rejects.
    sys.argv = sys.argv[:1]

    # Become a new session leader so vLLM's engine workers (spawned by vLLM via
    # multiprocessing internally) share our process group. Lets the parent kill
    # the whole vLLM process tree atomically via os.killpg() on shutdown, and
    # detaches us from the terminal so terminal SIGINT no longer reaches us —
    # the parent owns lifecycle.
    os.setsid()

    # Filesystem broadcast — vLLM workers and the trainer don't share GPU memory.
    infer_config.weight_broadcast_type = "filesystem"

    from surogate.grpo.inference.grpo_infer import grpo_infer

    grpo_infer(infer_config)


def _run_trainer(train_config: GRPOTrainConfig, failure_event: threading.Event):
    """Run the GRPO trainer in a background thread (parent process).

    Sets `failure_event` on any exception so the watchdog can interrupt the
    orchestrator. Re-raising here would be silently dropped by daemon-thread
    teardown — the event is the propagation channel to the main thread.
    """
    from surogate.grpo.trainer import GRPOTrainer

    try:
        GRPOTrainer(train_config, external_weights=None).train()
    except Exception:
        logger.exception("Trainer thread crashed")
        failure_event.set()


def _watch_components(
    vllm_procs: list[tuple["mp.Process", str]],
    trainer_failed: threading.Event,
    shutdown_event: threading.Event,
) -> None:
    """Watchdog: aborts the pipeline if any vLLM or the trainer dies unexpectedly.

    Sleeps on the union of all vLLM sentinels with a 1s timeout so we can also
    poll the trainer's failure flag. On unexpected death we deliver SIGINT to
    the main thread, which has the existing SIGINT/KeyboardInterrupt teardown path.

    ``vllm_procs`` is a list of (process, label) tuples so the crash message can
    distinguish rollout-vLLM death from judge-vLLM death.
    """
    sentinels = [proc.sentinel for proc, _ in vllm_procs]
    crashed: str | None = None
    while not shutdown_event.is_set():
        ready = multiprocessing.connection.wait(sentinels, timeout=1.0)
        if shutdown_event.is_set():
            return
        for proc, label in vllm_procs:
            if proc.sentinel in ready:
                crashed = f"{label} subprocess died unexpectedly (exitcode={proc.exitcode})"
                break
        if crashed:
            break
        if trainer_failed.is_set():
            crashed = "Trainer thread crashed"
            break
    if crashed is None:
        return
    logger.error(f"{crashed} — aborting GRPO pipeline")
    # Wake the main thread out of asyncio.run; existing finally block tears down.
    os.kill(os.getpid(), signal.SIGINT)


def grpo_split(
    train_config: GRPOTrainConfig,
    infer_config: GRPOInferenceConfig,
    orch_config: GRPOOrchestratorConfig,
    vllm_gpu_ids: list[int],
    trainer_gpu_ids: list[int],
    judge_infer_config: GRPOInferenceConfig | None = None,
    judge_gpu_ids: list[int] | None = None,
):
    """Run the full GRPO pipeline with vLLM and trainer on disjoint GPU sets.

    When ``judge_infer_config`` and ``judge_gpu_ids`` are both provided AND
    ``orch_config.ruler.enabled`` is True, also spawns a second vLLM subprocess
    serving the RULER judge model on the given GPU ids. This lets a single
    ``surogate grpo`` invocation own the entire RULER topology (rollout vLLM +
    judge vLLM + trainer + orchestrator). When the judge args are omitted, the
    orchestrator is expected to point at an externally-running judge, exactly
    as in the pre-judge-spawn workflow.

    The CLI is expected to have set ``os.environ["CUDA_VISIBLE_DEVICES"]`` to the
    trainer GPU ids BEFORE importing torch. We do not set it again here because
    the parent has already imported the surogate stack by this point.
    """
    # Validate judge args FIRST so judge-config errors surface before unrelated
    # rollout/trainer overlap errors confuse the user.
    _validate_judge_args(
        judge_infer_config=judge_infer_config,
        judge_gpu_ids=judge_gpu_ids,
        orch_config=orch_config,
        vllm_gpu_ids=vllm_gpu_ids,
        trainer_gpu_ids=trainer_gpu_ids,
        rollout_infer_config=infer_config,
    )

    _check_disjoint_gpus(("--vllm-gpus", vllm_gpu_ids), ("--trainer-gpus", trainer_gpu_ids))
    _check_gpu_count_matches_topology("--vllm-gpus", vllm_gpu_ids, infer_config)
    if len(trainer_gpu_ids) != train_config.gpus:
        raise ValueError(f"--trainer-gpus has {len(trainer_gpu_ids)} GPU(s) but train.gpus = {train_config.gpus}")

    spawn_judge = judge_infer_config is not None
    topology = f"vLLM GPUs: {vllm_gpu_ids}, trainer GPUs: {trainer_gpu_ids}" + (
        f", judge GPUs: {judge_gpu_ids}" if spawn_judge else ""
    )
    logger.info(f"Starting GRPO pipeline (split mode) — {topology}")

    # No CUDA-IPC weight sharing in split mode — both sides go through disk.
    train_config.weight_broadcast_type = "filesystem"
    infer_config.weight_broadcast_type = "filesystem"

    # Spawn (not fork) so the child gets a fresh interpreter and doesn't inherit
    # any torch/CUDA state from the parent's trainer-side GPU mask.
    ctx = mp.get_context("spawn")
    vllm_proc = _spawn_vllm(ctx, infer_config, vllm_gpu_ids, name="vllm-subprocess")

    # SIGTERM → SIGINT so a plain `kill <pid>` (and orchestrator code that calls
    # signal.raise_signal(SIGTERM) on stop) takes the same KeyboardInterrupt path
    # that Ctrl-C does, ensuring the finally block tears down the vLLM tree.
    signal.signal(signal.SIGTERM, lambda *_: signal.raise_signal(signal.SIGINT))

    judge_proc: mp.Process | None = None
    if spawn_judge:
        assert judge_infer_config is not None and judge_gpu_ids is not None  # for type narrowing
        judge_proc = _spawn_vllm(ctx, judge_infer_config, judge_gpu_ids, name="judge-vllm-subprocess")

    components = "rollout vLLM, " + ("judge vLLM, " if judge_proc is not None else "") + "trainer, and orchestrator"
    logger.info(f"Spawning {components} in parallel")
    vllm_proc.start()
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

    # Watchdog watches every vLLM we spawned plus the trainer thread. shutdown_event
    # suppresses spurious aborts during the planned teardown in the finally block.
    watched_vllms: list[tuple[mp.Process, str]] = [(vllm_proc, "rollout vLLM")]
    if judge_proc is not None:
        watched_vllms.append((judge_proc, "judge vLLM"))
    shutdown_event = threading.Event()
    watchdog_thread = Thread(
        target=_watch_components,
        args=(watched_vllms, trainer_failed, shutdown_event),
        daemon=True,
        name="grpo-watchdog",
    )
    watchdog_thread.start()

    try:
        from surogate.grpo.orchestrator.grpo_orch import orchestrate

        asyncio.run(orchestrate(orch_config))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"Split GRPO pipeline error: {e}")
        raise
    finally:
        logger.info("Split GRPO pipeline shutting down")

        # Tell the watchdog this teardown is intentional; otherwise it would see
        # vLLM's planned death and re-fire SIGINT.
        shutdown_event.set()

        # Ignore further Ctrl-Cs/SIGTERMs so cleanup runs to completion. Without this,
        # a second Ctrl-C interrupts the vLLM teardown and leaks the subprocess tree.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        # Reap vLLMs in parallel — each can take up to 15s on SIGTERM→SIGKILL escalation.
        # Sequential teardown would double that on a typical RULER topology.
        _terminate_vllm_trees_in_parallel(watched_vllms)

        # Trainer runs as a daemon thread; a brief join lets it flush logs, after
        # which it dies with the process. No reason to block for 30s here.
        trainer_thread.join(timeout=2.0)
        if trainer_thread.is_alive():
            logger.warning("Trainer thread still running; will exit with the process")

        watchdog_thread.join(timeout=2.0)


def _spawn_vllm(
    ctx: mp.context.SpawnContext, infer_config: GRPOInferenceConfig, gpu_ids: list[int], *, name: str
) -> mp.Process:
    """Build (don't yet start) a vLLM subprocess pinned to the given GPU ids."""
    return ctx.Process(
        target=_run_vllm_subprocess,
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
    vllm_gpu_ids: list[int],
    trainer_gpu_ids: list[int],
    rollout_infer_config: GRPOInferenceConfig,
) -> None:
    """Validate the judge spawn args.

    Both ``judge_infer_config`` and ``judge_gpu_ids`` must be set together (or
    both omitted; this defensive check catches non-CLI callers — the CLI layer
    rejects partial args earlier with a friendlier error). When set, RULER must
    be enabled, the judge GPU set must be disjoint from rollout-vLLM and
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
        ("--vllm-gpus", vllm_gpu_ids),
        ("--trainer-gpus", trainer_gpu_ids),
    )
    _check_gpu_count_matches_topology("--judge-gpus", judge_gpu_ids, judge_infer_config)
    if judge_infer_config.port == rollout_infer_config.port:
        raise ValueError(
            f"Judge and rollout vLLM cannot share port {judge_infer_config.port}. "
            "Set a different `port` in the judge inference config (e.g. 8001)."
        )


def _terminate_vllm_trees_in_parallel(watched: list[tuple["mp.Process", str]]) -> None:
    """SIGTERM all vLLM subprocesses concurrently and wait for them in parallel.

    Each call to ``_terminate_vllm_tree`` can block up to ~15s waiting for the
    SIGTERM→SIGKILL escalation. With two vLLMs (rollout + judge), serial
    teardown doubles shutdown latency for no reason. Threads are sufficient
    here — the work is mostly waiting on ``Process.join``.
    """
    if len(watched) <= 1:
        for proc, label in watched:
            _terminate_vllm_tree(proc, label=label)
        return
    threads = [
        Thread(target=_terminate_vllm_tree, args=(proc,), kwargs={"label": label}, name=f"reap-{label}")
        for proc, label in watched
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _terminate_vllm_tree(vllm_proc: "mp.Process", *, label: str = "vLLM") -> None:
    """SIGTERM the vLLM process group, escalate to SIGKILL if it doesn't exit."""
    if not vllm_proc.is_alive() or vllm_proc.pid is None:
        return
    pgid = vllm_proc.pid  # vllm_proc called setsid() so its pid == its pgid
    logger.info(f"Terminating {label} subprocess group")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    vllm_proc.join(timeout=10.0)
    if vllm_proc.is_alive():
        logger.warning(f"{label} did not exit in 10s; sending SIGKILL to process group")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        vllm_proc.join(timeout=5.0)
