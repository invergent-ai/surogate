"""Split GRPO runner: vLLM and the trainer occupy disjoint sets of GPUs.

Unlike the co-locate runner, vLLM is launched as a separate Python subprocess so it
can be given its own ``CUDA_VISIBLE_DEVICES``. The trainer runs in the parent process,
which has already had its own ``CUDA_VISIBLE_DEVICES`` set by the CLI before any torch
import happens. Weight broadcast goes through the filesystem backend — there is no
shared GPU memory between the two GPU sets.

Startup sequence (all three components launch concurrently):
    1. Spawn vLLM subprocess with ``CUDA_VISIBLE_DEVICES=<vllm gpu ids>``.
    2. Start the surogate trainer in a background thread (parent process,
       ``CUDA_VISIBLE_DEVICES=<trainer gpu ids>`` already applied by the CLI).
    3. Run the orchestrator in the main async event loop. The orchestrator
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
    vllm_proc: "mp.Process",
    trainer_thread: Thread,
    trainer_failed: threading.Event,
    shutdown_event: threading.Event,
) -> None:
    """Watchdog: aborts the pipeline if vLLM or the trainer dies unexpectedly.

    Sleeps on `vllm_proc.sentinel` with a 1s timeout so we can also poll the
    trainer's failure flag. On unexpected death we deliver SIGINT to the main
    thread, which has the existing SIGINT/KeyboardInterrupt teardown path.
    """
    crashed: str | None = None
    while not shutdown_event.is_set():
        ready = multiprocessing.connection.wait([vllm_proc.sentinel], timeout=1.0)
        if shutdown_event.is_set():
            return
        if ready:
            crashed = f"vLLM subprocess died unexpectedly (exitcode={vllm_proc.exitcode})"
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
):
    """Run the full GRPO pipeline with vLLM and trainer on disjoint GPU sets.

    The CLI is expected to have set ``os.environ["CUDA_VISIBLE_DEVICES"]`` to the
    trainer GPU ids BEFORE importing torch. We do not set it again here because
    the parent has already imported the surogate stack by this point.
    """
    if set(vllm_gpu_ids) & set(trainer_gpu_ids):
        overlap = sorted(set(vllm_gpu_ids) & set(trainer_gpu_ids))
        raise ValueError(f"--vllm-gpus and --trainer-gpus overlap on {overlap}")

    expected_vllm = (infer_config.dp or 1) * (infer_config.tp or 1)
    if len(vllm_gpu_ids) != expected_vllm:
        raise ValueError(
            f"--vllm-gpus has {len(vllm_gpu_ids)} GPU(s) but inference config implies "
            f"dp({infer_config.dp}) * tp({infer_config.tp}) = {expected_vllm}"
        )
    if len(trainer_gpu_ids) != train_config.gpus:
        raise ValueError(f"--trainer-gpus has {len(trainer_gpu_ids)} GPU(s) but train.gpus = {train_config.gpus}")

    logger.info(f"Starting GRPO pipeline (split mode) — vLLM GPUs: {vllm_gpu_ids}, trainer GPUs: {trainer_gpu_ids}")

    # No CUDA-IPC weight sharing in split mode — both sides go through disk.
    train_config.weight_broadcast_type = "filesystem"
    infer_config.weight_broadcast_type = "filesystem"

    # Spawn (not fork) so the child gets a fresh interpreter and doesn't inherit
    # any torch/CUDA state from the parent's trainer-side GPU mask.
    ctx = mp.get_context("spawn")
    vllm_proc = ctx.Process(
        target=_run_vllm_subprocess,
        args=(infer_config, ",".join(str(g) for g in vllm_gpu_ids)),
        name="vllm-subprocess",
        daemon=False,
    )
    # SIGTERM → SIGINT so a plain `kill <pid>` (and orchestrator code that calls
    # signal.raise_signal(SIGTERM) on stop) takes the same KeyboardInterrupt path
    # that Ctrl-C does, ensuring the finally block tears down the vLLM tree.
    signal.signal(signal.SIGTERM, lambda *_: signal.raise_signal(signal.SIGINT))

    logger.info("Spawning vLLM, trainer, and orchestrator in parallel")
    vllm_proc.start()

    trainer_failed = threading.Event()
    trainer_thread = Thread(
        target=_run_trainer,
        args=(train_config, trainer_failed),
        daemon=True,
        name="grpo-trainer",
    )
    trainer_thread.start()

    # Watchdog must come up after both targets but before the orchestrator runs
    # so we never miss an early crash. shutdown_event suppresses spurious aborts
    # during the planned teardown in the finally block below.
    shutdown_event = threading.Event()
    watchdog_thread = Thread(
        target=_watch_components,
        args=(vllm_proc, trainer_thread, trainer_failed, shutdown_event),
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

        # Reap vLLM first — the heavy resource the user wants released.
        _terminate_vllm_tree(vllm_proc)

        # Trainer runs as a daemon thread; a brief join lets it flush logs, after
        # which it dies with the process. No reason to block for 30s here.
        trainer_thread.join(timeout=2.0)
        if trainer_thread.is_alive():
            logger.warning("Trainer thread still running; will exit with the process")

        watchdog_thread.join(timeout=2.0)


def _terminate_vllm_tree(vllm_proc: "mp.Process") -> None:
    """SIGTERM the vLLM process group, escalate to SIGKILL if it doesn't exit."""
    if not vllm_proc.is_alive() or vllm_proc.pid is None:
        return
    pgid = vllm_proc.pid  # vllm_proc called setsid() so its pid == its pgid
    logger.info("Terminating vLLM subprocess group")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    vllm_proc.join(timeout=10.0)
    if vllm_proc.is_alive():
        logger.warning("vLLM did not exit in 10s; sending SIGKILL to process group")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        vllm_proc.join(timeout=5.0)
