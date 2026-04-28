"""Split GRPO runner: vLLM and the trainer occupy disjoint sets of GPUs.

Unlike the co-locate runner, vLLM is launched as a separate Python subprocess so it
can be given its own ``CUDA_VISIBLE_DEVICES``. The trainer runs in the parent process,
which has already had its own ``CUDA_VISIBLE_DEVICES`` set by the CLI before any torch
import happens. Weight broadcast goes through the filesystem backend — there is no
shared GPU memory between the two GPU sets.

Startup sequence:
    1. Spawn vLLM subprocess with ``CUDA_VISIBLE_DEVICES=<vllm gpu ids>``.
    2. Poll vLLM /health until it serves.
    3. Start the surogate trainer in a background thread (parent process,
       ``CUDA_VISIBLE_DEVICES=<trainer gpu ids>`` already applied by the CLI).
    4. Run the orchestrator in the main async event loop.

Module-level imports here MUST stay torch-free: this module is re-imported in the
spawned vLLM child, where torch must not be loaded until ``CUDA_VISIBLE_DEVICES``
has been re-set.
"""

import asyncio
import multiprocessing as mp
import time
import urllib.error
import urllib.request
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

    # Filesystem broadcast — vLLM workers and the trainer don't share GPU memory.
    infer_config.weight_broadcast_type = "filesystem"

    from surogate.grpo.inference.grpo_infer import grpo_infer

    grpo_infer(infer_config)


def _wait_for_vllm_ready(proc: "mp.Process", host: str | None, port: int, timeout: float = 600.0) -> bool:
    """Poll vLLM /health until it returns 200, the subprocess dies, or we hit the timeout."""
    deadline = time.time() + timeout
    target_host = host or "localhost"
    url = f"http://{target_host}:{port}/health"
    last_err: Exception | None = None
    while time.time() < deadline:
        if not proc.is_alive():
            logger.error(f"vLLM subprocess exited during startup (exitcode={proc.exitcode})")
            return False
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = e
        time.sleep(2.0)
    logger.error(f"vLLM /health did not become ready within {timeout:.0f}s (last error: {last_err})")
    return False


def _run_trainer(train_config: GRPOTrainConfig):
    """Run the GRPO trainer in a background thread (parent process)."""
    from surogate.grpo.trainer import GRPOTrainer

    try:
        GRPOTrainer(train_config, external_weights=None).train()
    except Exception:
        logger.exception("Trainer thread crashed")
        raise


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
    vllm_proc.start()

    trainer_thread: Thread | None = None
    try:
        ready = _wait_for_vllm_ready(vllm_proc, infer_config.host, infer_config.port, timeout=600.0)
        if not ready:
            raise RuntimeError("vLLM did not become ready in time")
        logger.info("vLLM server ready")

        logger.info("Starting GRPO trainer")
        trainer_thread = Thread(
            target=_run_trainer,
            args=(train_config,),
            daemon=True,
            name="grpo-trainer",
        )
        trainer_thread.start()

        logger.info("Starting orchestrator")
        from surogate.grpo.orchestrator.grpo_orch import orchestrate

        asyncio.run(orchestrate(orch_config))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"Split GRPO pipeline error: {e}")
        raise
    finally:
        logger.info("Split GRPO pipeline shutting down")

        if trainer_thread is not None:
            trainer_thread.join(timeout=30.0)
            if trainer_thread.is_alive():
                logger.warning("Trainer thread did not finish within 30s")

        if vllm_proc.is_alive():
            logger.info("Terminating vLLM subprocess")
            vllm_proc.terminate()
            vllm_proc.join(timeout=10.0)
            if vllm_proc.is_alive():
                logger.warning("vLLM subprocess did not exit in 10s; killing")
                vllm_proc.kill()
                vllm_proc.join(timeout=5.0)
