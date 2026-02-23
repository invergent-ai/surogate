"""Unified GRPO runner: starts vLLM, trainer, and orchestrator in a single process.

In co-locate mode, vLLM's engine runs in a child process and the surogate trainer
runs in the parent. CUDA IPC is used to share quantized weight GPU memory between
the two processes (zero-copy).

The orchestrator runs in the main async event loop, communicating with vLLM via HTTP.

Startup sequence:
    1. Start vLLM HTTP server in a background thread (engine in child process)
    2. Wait for vLLM to be ready (init_app_state callback)
    3. Extract quantized weight GPU pointers from vLLM via CUDA IPC
    4. Start the trainer in a background thread (borrows vLLM's weights)
    5. Run the orchestrator in the main thread (async event loop)
"""

import asyncio
import logging
import sys
import time
from threading import Event, Thread

from surogate.core.config.grpo_inference_config import GRPOInferenceConfig
from surogate.core.config.grpo_orch_config import GRPOOrchestratorConfig
from surogate.grpo.config import GRPOTrainConfig
from surogate.grpo.inference.grpo_infer import setup_vllm_env
from surogate.utils.logger import get_logger

logger = get_logger()


def _run_vllm_server(
    infer_config: GRPOInferenceConfig,
    ready_event: Event,
    error_event: Event,
    engine_holder: list,
    loop_holder: list,
    shutdown_event: Event,
):
    """Run vLLM server in a background thread. Signals ready_event when serving.

    engine_holder: shared list; we append the engine_client (AsyncLLM) once
        init_app_state stores it so the main thread can use it for collective_rpc.
    loop_holder: shared list; we append the asyncio event loop so the main thread
        can schedule coroutines on it via asyncio.run_coroutine_threadsafe().
    shutdown_event: set by the main thread before stopping the loop, so we can
        distinguish intentional shutdown from unexpected errors.
    """
    try:
        import signal
        import uvloop
        from vllm.entrypoints.openai.api_server import run_server
        from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        import vllm.entrypoints.openai.api_server as _api_server_mod

        # Import our custom server module FIRST to apply its monkey patches
        # (custom_init_app_state, custom_build_app, etc.)
        from surogate.grpo.inference.vllm.server import WORKER_EXTENSION_CLS  # noqa: F401

        # Capture init_app_state AFTER server.py's monkey patches are applied,
        # so _prev_init_app_state is custom_init_app_state (not the vLLM original)
        _prev_init_app_state = _api_server_mod.init_app_state

        parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
        parser = make_arg_parser(parser)
        args = parser.parse_args(args=[], namespace=infer_config.to_vllm())
        assert args is not None
        validate_parsed_serve_args(args)

        args.worker_extension_cls = WORKER_EXTENSION_CLS[infer_config.weight_broadcast_type]

        # Wrap init_app_state to also capture the engine_client for weight extraction.
        async def _patched_init_app_state(engine_client, state, args, supported_tasks=None):
            await _prev_init_app_state(engine_client, state, args, supported_tasks)
            engine_holder.append(engine_client)
            ready_event.set()

        _api_server_mod.init_app_state = _patched_init_app_state

        # Both signal.signal() and loop.add_signal_handler() only work in the
        # main thread, but vLLM registers signal handlers in both places.
        # Suppress both when running in a background thread.
        _original_signal = signal.signal
        signal.signal = lambda *a, **kw: signal.SIG_DFL
        loop = None
        try:
            loop = uvloop.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.add_signal_handler = lambda *a, **kw: None
            # Share the loop so the main thread can schedule collective_rpc calls
            loop_holder.append(loop)
            loop.run_until_complete(run_server(args))
        except RuntimeError as e:
            # When the main thread calls loop.stop() for graceful shutdown,
            # run_until_complete() raises "Event loop stopped before Future completed".
            # This is expected — not an error.
            if shutdown_event.is_set():
                logger.info(f"vLLM server stopped (graceful shutdown)")
            else:
                raise
        finally:
            signal.signal = _original_signal
            _api_server_mod.init_app_state = _prev_init_app_state
            # Cancel all pending async tasks to suppress "Task was destroyed but
            # it is pending!" warnings (e.g., listen_for_disconnect coroutines).
            # Mute uvicorn's error logger during cancellation — it logs
            # CancelledError stack traces for in-flight ASGI requests.
            if loop is not None:
                uvicorn_logger = logging.getLogger("uvicorn.error")
                prev_level = uvicorn_logger.level
                uvicorn_logger.setLevel(logging.CRITICAL)
                try:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    # Give cancelled tasks a chance to run their cleanup.
                    # After loop.stop(), we need to restart before run_until_complete.
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
                finally:
                    uvicorn_logger.setLevel(prev_level)
                try:
                    loop.close()
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"vLLM server error: {e}")
        error_event.set()
        raise


def _extract_vllm_weights(
    engine_client,
    event_loop: asyncio.AbstractEventLoop,
    num_gpus: int,
) -> list[list[dict]] | None:
    """Extract quantized weight GPU pointers from vLLM via CUDA IPC.

    Returns per-GPU list of ExternalWeight dicts, or None if extraction fails.
    For DP mode (tp=1, dp=N), each GPU has a full model replica.
    """
    try:
        from surogate.grpo.inference.vllm.weight_extractor import extract_vllm_weights_via_ipc

        weights = extract_vllm_weights_via_ipc(engine_client, event_loop)

        if not weights:
            logger.warning("No quantized weights extracted from vLLM")
            return None

        # Replicate for all GPUs (in DP mode, all GPUs have identical weights)
        per_gpu_weights = [weights] * num_gpus
        logger.info(f"Extracted {len(weights)} quantized weights from vLLM "
                     f"for {num_gpus} GPU(s)")
        return per_gpu_weights

    except Exception as e:
        logger.warning(f"Weight extraction from vLLM failed: {e}. "
                       f"Falling back to disk loading.")
        import traceback
        traceback.print_exc()
        return None


def _run_trainer(
    train_config: GRPOTrainConfig,
    external_weights: list[list[dict]] | None,
    error_event: Event,
):
    """Run the GRPO trainer in a background thread."""
    try:
        from surogate.grpo.trainer import GRPOTrainer
        trainer = GRPOTrainer(train_config, external_weights=external_weights)
        trainer.train()
    except Exception as e:
        logger.error(f"Trainer error: {e}")
        error_event.set()
        raise


def grpo_unified(
    train_config: GRPOTrainConfig,
    infer_config: GRPOInferenceConfig,
    orch_config: GRPOOrchestratorConfig,
):
    """Run the full GRPO pipeline in a single process (co-locate mode).

    1. vLLM HTTP server (background thread, engine in child process)
    2. Extract quantized weight GPU pointers from vLLM via CUDA IPC
    3. Surogate C++ trainer (background thread, borrows vLLM's weights)
    4. Orchestrator (main async event loop)
    """
    logger.info("Starting unified GRPO pipeline (co-locate mode)")

    # Trainer uses filesystem broadcast for LoRA adapter updates — the vLLM
    # engine runs in a child process, so in-process shared state doesn't work.
    # Base weights are still zero-copy (shared via CUDA IPC at startup).
    # LoRA adapters are small (~10MB) so disk I/O is fine.
    train_config.weight_broadcast_type = "filesystem"
    # vLLM needs ColocateWeightUpdateWorker for extract_weight_ipc_handles()
    infer_config.weight_broadcast_type = "colocate"

    # Setup vLLM environment (must happen before importing vLLM)
    setup_vllm_env(infer_config)

    error_event = Event()
    vllm_ready = Event()
    shutdown_event = Event()  # Set before stopping vLLM loop for graceful shutdown
    engine_holder: list = []  # Shared: vLLM thread appends engine_client here
    loop_holder: list = []    # Shared: vLLM thread appends event loop here

    # Phase 1: Start vLLM server in background thread
    logger.info(f"Starting vLLM server on port {infer_config.port}" )
    vllm_thread = Thread(
        target=_run_vllm_server,
        args=(infer_config, vllm_ready, error_event, engine_holder, loop_holder,
              shutdown_event),
        daemon=True,
        name="vllm-server",
    )
    vllm_thread.start()

    # Wait for vLLM to be fully initialized (ready_event is set after init_app_state)
    vllm_ready.wait(timeout=300.0)
    if not vllm_ready.is_set() or error_event.is_set():
        logger.error("vLLM server failed to start")
        sys.exit(1)
    logger.info("vLLM server ready")

    # Phase 2: Extract quantized weight GPU pointers from vLLM via CUDA IPC
    engine_client = engine_holder[0] if engine_holder else None
    event_loop = loop_holder[0] if loop_holder else None

    external_weights = None
    if engine_client is not None and event_loop is not None:
        logger.info("Extracting quantized weights from vLLM via CUDA IPC")
        external_weights = _extract_vllm_weights(engine_client, event_loop, train_config.gpus)
    else:
        if engine_client is None:
            logger.warning("No engine client captured from vLLM — falling back to disk loading")
        if event_loop is None:
            logger.warning("No event loop captured from vLLM — falling back to disk loading")

    # Phase 3: Start trainer in background thread
    logger.info("Starting GRPO trainer")
    trainer_thread = Thread(
        target=_run_trainer,
        args=(train_config, external_weights, error_event),
        daemon=True,
        name="grpo-trainer",
    )
    trainer_thread.start()

    # Phase 4: Run orchestrator in main thread (async event loop)
    logger.info("Starting orchestrator")
    try:
        from surogate.grpo.orchestrator.grpo_orch import orchestrate
        asyncio.run(orchestrate(orch_config))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        raise
    finally:
        logger.info("Unified GRPO pipeline shutting down")

        # Wait for trainer to finish (it stops when orchestrator writes the stop signal)
        trainer_thread.join(timeout=30.0)
        if trainer_thread.is_alive():
            logger.warning("Trainer thread did not finish within 30s")

        # Stop vLLM event loop gracefully. Set shutdown_event first so the
        # vLLM thread knows the "Event loop stopped" RuntimeError is expected.
        shutdown_event.set()
        if event_loop is not None and not event_loop.is_closed():
            event_loop.call_soon_threadsafe(event_loop.stop)
            vllm_thread.join(timeout=10.0)
