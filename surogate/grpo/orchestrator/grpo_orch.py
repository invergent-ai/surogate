import asyncio
import dataclasses
import multiprocessing as mp
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from surogate.grpo.orchestrator.advantage import compute_advantages
from surogate.grpo.orchestrator.eval_utils import compute_eval_ckpt_step, get_eval_sampling_args
from surogate.grpo.orchestrator.event_loop_lag import EventLoopLagMonitor
from surogate.grpo.orchestrator.patches import monkey_patch_chat_completion_logprobs, monkey_patch_oai_iterable_types
from surogate.grpo.orchestrator.trajectories import interleave_rollout
from surogate.grpo.transport import TrainingBatch, TrainingSample, setup_training_batch_sender
from surogate.grpo.utils.asynyc_utils import safe_cancel
from surogate.grpo.utils.pathing import get_log_dir

# This monkey patch is necessary to avoid Pydantic validating fields using typing.Iterable (e.g. in multimodal or tool call messages) lazily which leads to tokenization errors, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1249
monkey_patch_oai_iterable_types()

# This monkey patch is necessary to avoid heavy CPU overhead from constructing the OAI ChatCompletion Pydantic model with logprobs, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1189
monkey_patch_chat_completion_logprobs()

# Import environment before any other imports

import pandas as pd
import verifiers as vf
from transformers import AutoProcessor, AutoTokenizer

from surogate.core.config.grpo_orch_config import GRPOBufferConfig, GRPOOrchestratorConfig
from surogate.utils.dict import DictDefault
from surogate.grpo.orchestrator.buffer import Buffer
from surogate.grpo.orchestrator.ckpt import Progress, setup_ckpt_manager
from surogate.grpo.orchestrator.eval_utils import evaluate_env
from surogate.grpo.orchestrator.filters import apply_filters, setup_filters
from surogate.grpo.orchestrator.scheduler import Scheduler
from surogate.grpo.orchestrator.utils import (
    compute_teacher_logprobs,
    get_sampling_args,
    get_weight_dir,
    print_benchmark,
    set_semaphore,
)
from surogate.grpo.orchestrator.vf_utils import (
    generate,
    get_completion_len,
    get_seq_len,
    intercept_vf_logging,
    setup_env_client,
    spawn_env_server,
    start_env_log_listener,
    task_uses_group_scoring,
    wait_for_env_servers,
)
from surogate.grpo.utils.client import (
    init_nccl_broadcast,
    setup_inference_pool,
)
from surogate.rewards.ruler import JudgeClientPool, build_ruler_rubric
from surogate.grpo.utils.logger import setup_logger
from surogate.grpo.utils.monitor import setup_monitor
from surogate.grpo.utils.temp_scheduling import compute_temperature
from surogate.grpo.utils.utils import (
    clean_exit,
    get_env_ids_to_install,
    install_env,
    resolve_latest_ckpt_step,
    strip_env_version,
    to_col_format,
)
from surogate.grpo.utils.vlm import is_vlm_model


@clean_exit
async def orchestrate(config: GRPOOrchestratorConfig):
    # Initialize the logger
    logger = setup_logger(
        config.log.level,
        log_file=Path(config.output_dir) / "logs" / "orchestrator.log" if config.log.file else None,
        json_logging=config.log.json_logging,
    )
    intercept_vf_logging(logger="verifiers.workers", level=config.log.vf_level)  # show logs from env clients
    logger.info("Starting orchestrator")

    event_loop_lag_monitor = EventLoopLagMonitor()
    event_loop_lag_monitor_task = asyncio.create_task(event_loop_lag_monitor.run())

    # Print warning if running in benchmark mode
    if config.bench:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

    # Save configs to output directory
    config_dir = Path(config.output_dir) / "control"
    config_dir.mkdir(parents=True, exist_ok=True)

    class _QuotedDumper(yaml.SafeDumper):
        """YAML dumper that quotes all string values."""

        pass

    _QuotedDumper.add_representer(
        str, lambda dumper, data: dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    )

    import json

    config_data = json.loads(json.dumps(dataclasses.asdict(config), default=str))
    with open(config_dir / "orch.yaml", "w") as f:
        yaml.dump(config_data, f, Dumper=_QuotedDumper)

    # Install environments
    env_ids_to_install = set()
    env_ids_to_install.update(get_env_ids_to_install(config.env))
    if config.eval is not None:
        env_ids_to_install.update(get_env_ids_to_install(config.eval.env))

    for env_id in env_ids_to_install:
        install_env(env_id)

    # Setup inference pool
    client_type = "openai_chat_completions_token" if config.use_token_client else "openai_chat_completions"
    if config.use_token_client:
        logger.warning(
            "Token-in-token-out (TITO) client is enabled. Only use this if your environment has a linear "
            "history and the chat template has the extension property."
        )
    inference_pool = await setup_inference_pool(config.client, model_name=config.model.name, client_type=client_type)

    # Setup teacher inference pool if configured
    if config.teacher_model:
        logger.info(
            f"Initializing teacher inference pool (base_url={', '.join(config.teacher_model.client.base_url)}, "
            f"model={config.teacher_model.model.name})"
        )
        teacher_inference_pool = await setup_inference_pool(
            config.teacher_model.client, model_name=config.teacher_model.model.name
        )
    else:
        teacher_inference_pool = None

    # Check if this is a vision-language model (used throughout for VLM-specific paths)
    is_vlm = is_vlm_model(config.model.name)

    # Load tokenizer and processor (processor only for VLM models)
    logger.info(f"Initializing tokenizer for {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)

    processor = None
    if is_vlm:
        logger.info(f"Loading VLM processor for {config.model.name}")
        processor = AutoProcessor.from_pretrained(config.model.name, trust_remote_code=True, use_fast=True)

    # Build rollout filters
    rollout_filters = setup_filters(config.filters, vocab_size=tokenizer.vocab_size)
    if rollout_filters:
        logger.info(f"Initialized {len(rollout_filters)} rollout filter(s): {[f.name for f in rollout_filters]}")

    # Setup monitor
    logger.info(f"Initializing monitor (report_to={config.report_to})")
    monitor = setup_monitor(
        wandb_config=config.report_to,
        output_dir=Path(config.output_dir),
        tokenizer=tokenizer,
        run_config=config,
    )

    # Load environment and extract dataset
    logger.info(
        f"Loading {len(config.env)} training environment(s) ({', '.join(env.name or env.id for env in config.env)})"
    )
    env_ids = [strip_env_version(env.id) for env in config.env]
    train_env_names = [env.name or env_id for env_id, env in zip(env_ids, config.env)]
    for env in config.env:
        if env.path and env.path not in sys.path:
            sys.path.insert(0, env.path)
    train_env_group = vf.EnvGroup(
        envs=[vf.load_environment(env_id, **env.args) for env_id, env in zip(env_ids, config.env)],
        env_names=train_env_names,
        map_kwargs=dict(writer_batch_size=1),  # set defensively to not error on map operations on large datasets
    )

    # Set up RULER (LLM-as-judge) before group-scoring auto-detection so the
    # detector picks RULER's group func up automatically.
    ruler_judge_pool: JudgeClientPool | None = None
    if config.ruler.enabled:
        logger.info(
            f"Initializing RULER (judge_model={config.ruler.judge_model}, mode={config.ruler.mode}, "
            f"weight={config.ruler.weight}, max_concurrent={config.ruler.max_concurrent_judges}, "
            f"endpoints={config.ruler.judge.base_url})"
        )
        ruler_judge_pool = JudgeClientPool(config.ruler.judge, config.ruler.max_concurrent_judges)
        for env_name, env in zip(train_env_names, train_env_group.envs):
            ruler_rubric = build_ruler_rubric(config.ruler, ruler_judge_pool)
            if config.ruler.mode == "replace":
                env.rubric = ruler_rubric
            else:
                # 'add' and 'metric' both compose with the existing rubric.
                env.rubric = vf.RubricGroup([env.rubric, ruler_rubric])
            logger.info(f"  attached RULER to env '{env_name}' (mode={config.ruler.mode})")

    verification_enabled = config.verification.enabled
    train_env_deferred_group_scoring_tasks = (
        {env_name for env_name in train_env_names if task_uses_group_scoring(train_env_group, env_name)}
        if verification_enabled
        else set()
    )
    for train_env_name, env_cfg in zip(train_env_names, config.env):
        env_cfg.extra_env_kwargs["score_rollouts"] = (
            verification_enabled and train_env_name not in train_env_deferred_group_scoring_tasks
        )
    if not verification_enabled:
        logger.info("Verification disabled; all training envs will skip scoring.")
    elif train_env_deferred_group_scoring_tasks:
        deferred_tasks = ", ".join(sorted(train_env_deferred_group_scoring_tasks))
        logger.info(
            f"Deferred group scoring enabled for training tasks: {deferred_tasks}. "
            "Rollouts run individually and are scored once each group completes."
        )

    # Forward log records emitted in spawned env subprocesses back to the
    # orchestrator's logger. One queue is shared across all envs; each env
    # tags its records with its name so the listener can prefix them. Queue
    # must come from the same `spawn` context that spawn_env_server uses.
    env_log_queue: mp.Queue = mp.get_context("spawn").Queue()
    env_log_thread, env_log_stop = start_env_log_listener(env_log_queue)

    def _stop_env_log_listener():
        env_log_stop.set()
        try:
            env_log_queue.put_nowait(None)
        except Exception:
            pass
        env_log_thread.join(timeout=2.0)

    try:
        train_env_addresses = []
        env_processes: list[mp.Process] = []
        for env_id, env, env_name in zip(env_ids, config.env, train_env_names):
            if env.address is None:
                address, process = spawn_env_server(
                    env_id=env_id,
                    env_args=env.args,
                    extra_env_kwargs=env.extra_env_kwargs,
                    env_path=env.path,
                    log_queue=env_log_queue,
                    log_prefix=env_name,
                    log_level="CRITICAL",
                    log_file=(get_log_dir(Path(config.output_dir)) / "train" / f"{env_name}.log").as_posix(),
                    log_file_level=config.log.vf_level,
                    json_logging=config.log.json_logging,
                )
                env_processes.append(process)
            else:
                if env_name in train_env_deferred_group_scoring_tasks:
                    logger.warning(
                        f"Training env {env_name} uses external server at {env.address}. "
                        "Ensure that server was started with score_rollouts=False."
                    )
                address = env.address
            logger.info(f"Connecting train environment {env_name} to server at {address}")
            train_env_addresses.append(address)
        train_env_clients = [
            setup_env_client(address=address, name=name) for name, address in zip(train_env_names, train_env_addresses)
        ]

        logger.info("Waiting for train environment servers to be ready")
        await wait_for_env_servers(train_env_clients)
        logger.success("Train environment servers ready")

        # this puts all train envs into server model
        # all calls to run_rollout and run_group will be routed to the server via the env client
        for env, env_client in zip(train_env_group.envs, train_env_clients):
            env.env_client = env_client

        if config.eval:
            env_ids = [strip_env_version(env.id) for env in config.eval.env]
            for env in config.eval.env:
                if env.path and env.path not in sys.path:
                    sys.path.insert(0, env.path)
            eval_envs = [vf.load_environment(env_id, **env.args) for env_id, env in zip(env_ids, config.eval.env)]
            eval_env_names = [env.name or env_id for env_id, env in zip(env_ids, config.eval.env)]
            eval_sampling_args = get_eval_sampling_args(config.eval.sampling)
            eval_env_addresses = []

            for env_id, env, eval_env_name in zip(env_ids, config.eval.env, eval_env_names):
                if env.address is None:
                    address, process = spawn_env_server(
                        env_id=env_id,
                        env_args=env.args,
                        extra_env_kwargs=env.extra_env_kwargs,
                        env_path=env.path,
                        log_queue=env_log_queue,
                        log_prefix=eval_env_name,
                        log_level="CRITICAL",
                        log_file=(get_log_dir(Path(config.output_dir)) / "eval" / f"{eval_env_name}.log").as_posix(),
                        log_file_level=config.log.vf_level,
                        json_logging=config.log.json_logging,
                    )
                    env_processes.append(process)
                else:
                    address = env.address
                logger.info(f"Connecting eval environment {eval_env_name} to server at {address}")
                eval_env_addresses.append(address)

            eval_env_clients = [
                setup_env_client(address=address, name=name)
                for name, address in zip(eval_env_names, eval_env_addresses)
            ]

            logger.info("Waiting for eval environment servers to be ready")
            await wait_for_env_servers(eval_env_clients)
            logger.success("Eval environment servers ready")

            # this puts all eval envs into server mode
            # all calls to run_rollout and run_group will be routed to the server via the env client
            for eval_env, eval_env_client in zip(eval_envs, eval_env_clients):
                eval_env.env_client = eval_env_client
        else:
            eval_envs: list[vf.Environment] = []
            eval_env_names: list[str] = []
            eval_sampling_args = {}

        # Setup buffer
        logger.info(f"Setting up buffer ({config.buffer})")
        train_dataset = train_env_group.get_dataset(seed=config.buffer.seed)
        buffer = Buffer(train_dataset, train_env_group.env_names, config.buffer)
        if config.val is not None:
            val_buffer_config = GRPOBufferConfig(DictDefault({"env_ratios": config.buffer.env_ratios}))
            val_dataset = train_env_group.get_eval_dataset(seed=val_buffer_config.seed)
            val_buffer = Buffer(val_dataset, train_env_group.env_names, val_buffer_config)
        else:
            val_buffer = None

        # Get checkpoint manager
        logger.info(f"Initializing checkpoint manager ({config.ckpt})")
        ckpt_manager = setup_ckpt_manager(Path(config.output_dir), config.ckpt)

        checkpoint_step = None
        if config.ckpt and config.ckpt.resume_step is not None and ckpt_manager is not None:
            if config.ckpt.resume_step == -1:
                checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
            else:
                checkpoint_step = config.ckpt.resume_step

        scheduler = Scheduler(
            env=train_env_group,
            buffer=buffer,
            inference_pool=inference_pool,
            max_inflight_rollouts=config.max_inflight_rollouts,
            max_async_level=config.max_async_level,
            max_off_policy_steps=config.max_off_policy_steps,
            strict_async_level=config.strict_async_level,
            tasks_per_minute=config.tasks_per_minute,
            lora_name=config.model.lora_adapter,
            deferred_group_scoring_tasks=train_env_deferred_group_scoring_tasks,
            config=config,
        )

        if checkpoint_step is not None and config.model.lora_adapter is not None:
            assert config.model.lora_adapter is not None
            scheduler.model_name = config.model.lora_adapter

        # Check health of the inference pool
        logger.info("Waiting for inference pool to be ready")
        await inference_pool.wait_for_ready(config.model.name)
        logger.success("Inference pool ready")

        # Check health of teacher inference server if configured
        if config.teacher_model and teacher_inference_pool:
            logger.info("Waiting for teacher inference pool to be ready")
            await teacher_inference_pool.wait_for_ready(config.teacher_model.model.name)
            logger.success("Teacher inference pool ready")

        # Block until the RULER judge pool is up. Without this, the first batch's
        # group-scoring HTTP calls can land while the judge model is still loading
        # — they'd be retried internally but inflate the first step's cost/latency.
        if ruler_judge_pool is not None:
            await ruler_judge_pool.wait_for_ready()

        # Set up weight broadcast backend
        logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
        if config.weight_broadcast.type == "nccl":
            await init_nccl_broadcast(
                inference_pool.admin_clients,
                config.weight_broadcast.host,
                config.weight_broadcast.port,
                config.weight_broadcast.timeout,
            )

        # Setup training batch sender for sending training examples to trainer
        logger.info(f"Initializing training batch sender ({config.rollout_transport})")
        training_batch_sender = setup_training_batch_sender(Path(config.output_dir), config.rollout_transport)

        # Track last online eval checkpoint step for this process
        last_eval_step = -1
        # Track previous ckpt_step to detect when ckpt_step jumps over eval interval boundaries
        prev_ckpt_step = -1

        # Reset weights to base model if starting from scratch
        progress = Progress()

        if checkpoint_step is not None and ckpt_manager is not None:
            ckpt_manager.load(progress, buffer, step=checkpoint_step)
            logger.info(f"Resuming training from checkpoint step {checkpoint_step}")
            scheduler.ckpt_step = progress.step  # Always resume from the latest checkpoint
            if config.eval and config.eval.skip_eval_on_resume:
                prev_ckpt_step = scheduler.ckpt_step
                last_eval_step = scheduler.ckpt_step
                logger.info(f"Skipping online eval on resume (ckpt_step={scheduler.ckpt_step})")
            else:
                # Allow eval at resumed step by setting prev_ckpt_step one behind
                prev_ckpt_step = scheduler.ckpt_step - 1

            # In NCCL mode, skip existence check - weights are broadcasted, not stored on disk
            check_exists = config.weight_broadcast.type != "nccl"
            wait_timeout = config.ckpt.wait_for_weights_timeout if config.ckpt else None
            weights_path = get_weight_dir(
                Path(config.output_dir), scheduler.ckpt_step, check_exists=check_exists, wait_timeout=wait_timeout
            )
            lora_name = config.model.lora_adapter
            await inference_pool.update_weights(weights_path, lora_name=lora_name, step=scheduler.ckpt_step)
        else:
            logger.info("Training from scratch")

        # Iterate over dataset in batches
        max_steps = config.max_steps or int(1e9)
        logger.info(f"Starting orchestrator loop (max_steps={max_steps or 'infinite'})")
        is_first_step = True
        await set_semaphore(config.max_concurrent or -1)

        # Persistent ThreadPoolExecutor for parallel rollout processing
        rollout_executor = ThreadPoolExecutor(max_workers=64)

        # Optional per-step metrics dump (one JSON object per line). Both training
        # and eval rows are written here when enabled — eval rows are emitted by
        # `_dump_eval_metrics` at every eval pass and carry an "eval/<env>/*" namespace.
        metrics_dump_file = None
        if config.dump_metrics:
            metrics_dump_path = "/tmp/grpo_metrics.jsonl"
            metrics_dump_file = open(metrics_dump_path, "a")
            logger.info(f"Dumping per-step metrics to {metrics_dump_path}")

        def _dump_eval_metrics(metrics_per_env: list[dict]) -> None:
            """Append one JSONL row per env eval to the dump file (if open)."""
            if metrics_dump_file is None:
                return
            for metrics in metrics_per_env:
                if not metrics:
                    continue
                entry = {"ts": time.time(), **metrics}
                metrics_dump_file.write(json.dumps(entry, default=str) + "\n")
            metrics_dump_file.flush()

        while True:
            # Check if this run has been evicted by the trainer
            evicted_path = Path(config.output_dir) / "control" / "evicted.txt"
            if evicted_path.exists():
                reason = evicted_path.read_text().strip()
                raise RuntimeError(f"Run evicted by trainer: {reason}")

            # Capture ckpt_step once for consistency (it's updated inside the scheduler)
            ckpt_step = scheduler.ckpt_step

            # Save checkpoint (if we are at an interval step and not at the first or last step)
            is_last_step = config.max_steps is not None and progress.step == config.max_steps - 1
            save_ckpt_time = 0
            if (
                ckpt_manager is not None
                and (config.ckpt and config.ckpt.interval)
                and not (is_first_step or is_last_step)
                and progress.step % config.ckpt.interval == 0
            ):
                logger.info(f"Saving checkpoint at step {progress.step}")
                save_ckpt_start_time = time.perf_counter()
                ckpt_manager.save(progress, buffer, step=progress.step)
                save_ckpt_time = time.perf_counter() - save_ckpt_start_time

            # Break if we have reached the maximum number of steps
            if config.max_steps and progress.step >= config.max_steps:
                break

            logger.info(f"Starting orchestrator step {progress.step}")
            step_start_time = time.perf_counter()

            # Run evals BEFORE training (blocking). Weight updates are paused via
            # scheduler.checkpoint_ready during eval to ensure consistent weights.
            # Use range check to handle ckpt_step jumping over interval boundaries.
            eval_ckpt_step = None
            if config.eval:
                eval_ckpt_step = compute_eval_ckpt_step(
                    ckpt_step=ckpt_step,
                    prev_ckpt_step=prev_ckpt_step,
                    last_eval_step=last_eval_step,
                    interval=config.eval.interval,
                    eval_base_model=config.eval.eval_base_model,
                )

            if eval_ckpt_step is not None:
                last_eval_step = ckpt_step
                if eval_ckpt_step != ckpt_step:
                    logger.info(f"Running evals for interval step {eval_ckpt_step} (current ckpt_step={ckpt_step})")
                else:
                    logger.info(f"Running evals for checkpoint step {ckpt_step}")

                # Pause weight updates and re-scheduling of training rollouts during eval
                # to avoid evaluating across different checkpoints and avoid congestion
                scheduler.checkpoint_ready.clear()

                # For heavy eval workloads, it might be necessary additionally cancel in-flight training rollouts
                if config.eval.cancel_inflight_rollouts_on_eval:
                    logger.info("Cancelling in-flight training rollouts before starting evals to avoid congestion.")
                    await scheduler.cancel_inflight_rollouts()

                results = await asyncio.gather(
                    *[
                        evaluate_env(
                            env=eval_env,
                            env_name=eval_env_name,
                            get_client=inference_pool.get_next_client,
                            model_name=scheduler.model_name,
                            sampling_args=eval_sampling_args,
                            num_examples=eval_env_config.num_examples or config.eval.num_examples,
                            rollouts_per_example=eval_env_config.rollouts_per_example
                            or config.eval.rollouts_per_example,
                            max_retries=eval_env_config.max_retries,
                            ckpt_step=ckpt_step,
                            step=progress.step,
                        )
                        for eval_env, eval_env_name, eval_env_config in zip(eval_envs, eval_env_names, config.eval.env)
                    ]
                )
                _dump_eval_metrics(results)

                # Resume weight updates
                scheduler.checkpoint_ready.set()

            # Update prev_ckpt_step for next iteration
            prev_ckpt_step = ckpt_step

            # Schedule generating the training batch
            temperature = compute_temperature(progress.step, config.sampling, config.max_steps)
            sampling_args = get_sampling_args(config.sampling, temperature=temperature)
            scheduler.set_sampling_args(sampling_args)
            train_task = asyncio.create_task(scheduler.generate_batch(step=progress.step))

            # Schedule running validation at the specified interval
            if val_buffer and config.val and progress.step % config.val.interval == 0:
                logger.info(f"Running validation for step {progress.step}")
                val_examples = val_buffer.sample_examples(config.val.num_examples)
                val_task = asyncio.create_task(
                    generate(
                        env=train_env_group,
                        model_name=scheduler.model_name,
                        examples=val_examples,
                        rollouts_per_example=config.val.rollouts_per_example,
                        sampling_args=sampling_args,
                        clients=inference_pool.clients,
                        pbar_description="Generating rollouts (val)",
                    )
                )
            else:
                val_task = asyncio.create_task(asyncio.sleep(0))  # Dummy task

            # Await train rollouts, process results and write batch to disk to consume by trainer
            await train_task
            generate_completions_time = scheduler.last_batch_generation_time
            train_rollouts = train_task.result()

            # Apply rollout filters (zeros reward/mask for degenerate generations)
            filter_metrics = apply_filters(rollout_filters, train_rollouts)

            # Compute advantages
            example_ids = [r["example_id"] for r in train_rollouts]
            num_rollouts = len(train_rollouts)
            num_unique_examples = len(set(example_ids))
            rewards = [r["reward"] for r in train_rollouts]
            completion_lens = [get_completion_len(r) for r in train_rollouts]
            advantages = compute_advantages(
                rewards,
                completion_lens,
                config.rollouts_per_example,
                config.advantage,
            )

            # Convert rollouts to training samples
            parallel_preprocess_start = time.perf_counter()

            # Process rollouts in parallel
            def process_rollout(rollout: vf.RolloutOutput) -> list[TrainingSample] | None:
                return interleave_rollout(rollout)

            loop = asyncio.get_event_loop()
            futures = [loop.run_in_executor(rollout_executor, process_rollout, r) for _, r in enumerate(train_rollouts)]
            results = await asyncio.gather(*futures)

            # Collect results and assign advantages
            train_examples: list[TrainingSample] = []
            rollout_prefill_lens: list[int] = []
            rollout_decode_lens: list[int] = []
            rollout_samples_per_rollout: list[int] = []
            num_prefill_tokens = 0
            num_decode_tokens = 0
            for rollout, advantage, samples in zip(train_rollouts, advantages, results):
                rollout_prefill_tokens = 0
                rollout_decode_tokens = 0
                if samples is not None:
                    rollout_samples_per_rollout.append(len(samples))
                    for sample in samples:
                        sample.advantage = advantage
                        sample.reward = rollout["reward"]
                        sample_decode_tokens = sum(sample.completion_mask)
                        sample_prefill_tokens = (
                            len(sample.prompt_ids) + len(sample.completion_mask) - sample_decode_tokens
                        )
                        rollout_decode_tokens += sample_decode_tokens
                        rollout_prefill_tokens += sample_prefill_tokens
                        train_examples.append(sample)
                else:
                    rollout_samples_per_rollout.append(0)
                rollout_prefill_lens.append(rollout_prefill_tokens)
                rollout_decode_lens.append(rollout_decode_tokens)
                num_prefill_tokens += rollout_prefill_tokens
                num_decode_tokens += rollout_decode_tokens

            parallel_preprocess_time = time.perf_counter() - parallel_preprocess_start
            logger.debug(
                f"Converted {len(train_rollouts)} rollouts ({num_unique_examples} unique examples) "
                f"to {len(train_examples)} training examples"
            )

            # Compute teacher logprobs if teacher model is configured
            teacher_logprobs_time = 0
            if config.teacher_model and teacher_inference_pool:
                logger.info(f"Computing teacher logprobs for {len(train_examples)} training examples")
                teacher_logprobs_start_time = time.perf_counter()
                teacher_logprobs_list = await compute_teacher_logprobs(
                    clients=teacher_inference_pool.clients,
                    model_name=config.teacher_model.model.name,
                    samples=train_examples,
                )
                for train_example, teacher_logprobs in zip(train_examples, teacher_logprobs_list):
                    train_example.teacher_logprobs = teacher_logprobs
                teacher_logprobs_time = time.perf_counter() - teacher_logprobs_start_time
                logger.debug(f"Computed teacher logprobs in {teacher_logprobs_time:.2f}s")

            training_batch = TrainingBatch(
                examples=train_examples,
                step=progress.step,
            )

            training_batch_sender.send(training_batch)

            # Await and process val results
            await val_task
            val_outputs = val_task.result()

            # Gather metrics in dataframes
            results_df = pd.DataFrame(
                {
                    "example_id": [rollout["example_id"] for rollout in train_rollouts],
                    "task": [rollout["task"] for rollout in train_rollouts],
                    "reward": [rollout["reward"] for rollout in train_rollouts],
                    "is_truncated": [rollout["is_truncated"] for rollout in train_rollouts],
                    "error": [rollout["error"] for rollout in train_rollouts],
                    "stop_condition": [rollout.get("stop_condition") for rollout in train_rollouts],
                    "seq_len": [get_seq_len(rollout) for rollout in train_rollouts],
                    "prefill_len": rollout_prefill_lens,
                    "decode_len": rollout_decode_lens,
                    "samples_per_rollout": rollout_samples_per_rollout,
                    "num_turns": [len(rollout["trajectory"]) for rollout in train_rollouts],
                    "generation_ms": [rollout["timing"]["generation_ms"] for rollout in train_rollouts],
                    "scoring_ms": [rollout["timing"]["scoring_ms"] for rollout in train_rollouts],
                }
            )

            # Gather individual reward function metrics
            metrics_df = pd.DataFrame([rollout["metrics"] for rollout in train_rollouts])

            # Aggregate RULER step totals (per-rollout values are written as
            # group-totals divided by group size, so summing across the batch
            # recovers the true step total).
            ruler_step_totals: dict[str, float] = {}
            if config.ruler.enabled and not metrics_df.empty:
                ruler_sum_cols = [
                    "ruler_judge_calls",
                    "ruler_input_tokens",
                    "ruler_output_tokens",
                    "ruler_judge_cost_usd",
                    "ruler_judge_latency_ms",
                ]
                for col in ruler_sum_cols:
                    if col in metrics_df.columns:
                        ruler_step_totals[f"ruler/total_{col[len('ruler_') :]}"] = float(metrics_df[col].sum())
                if "ruler_judge_failed" in metrics_df.columns:
                    ruler_step_totals["ruler/judge_failure_rate"] = float(metrics_df["ruler_judge_failed"].mean())
                if "ruler_score" in metrics_df.columns:
                    ruler_step_totals["ruler/score_mean"] = float(metrics_df["ruler_score"].mean())
                    ruler_step_totals["ruler/score_std"] = float(metrics_df["ruler_score"].std())

            val_results_df = (
                pd.DataFrame(
                    {
                        "example_id": [rollout["example_id"] for rollout in val_outputs],
                        "task": [rollout["task"] for rollout in val_outputs],
                        "reward": [rollout["reward"] for rollout in val_outputs],
                    }
                )
                if val_outputs is not None
                else None
            )

            # Update progress metrics and throughput
            num_tokens = int(results_df.seq_len.sum())
            progress.total_tokens += num_tokens
            progress.total_samples += num_rollouts
            progress.total_problems += num_unique_examples

            # Compute solve all and none tensors
            problem_indices = results_df.index // config.rollouts_per_example
            reward_sum_per_problem = results_df.groupby(problem_indices).reward.sum()
            solve_all = (reward_sum_per_problem == config.rollouts_per_example).mean()
            solve_none = (reward_sum_per_problem == 0).mean()
            effective_batch_size = 1 - solve_none - solve_all

            step_time = time.perf_counter() - step_start_time
            throughput = num_tokens / step_time
            to_log = {
                # Progress metrics
                "progress/tokens": num_tokens,
                "progress/prefill_tokens": num_prefill_tokens,
                "progress/decode_tokens": num_decode_tokens,
                "progress/samples": num_rollouts,
                "progress/problems": num_unique_examples,
                "progress/total_tokens": progress.total_tokens,
                "progress/total_samples": progress.total_samples,
                "progress/total_problems": progress.total_problems,
                "progress/ckpt_step": ckpt_step,  # Shared W&B axis
                # Sequence length metrics
                "seq_len/mean": results_df.groupby("example_id").seq_len.mean().mean(),
                "seq_len/max": results_df.groupby("example_id").seq_len.mean().max(),
                "seq_len/min": results_df.groupby("example_id").seq_len.mean().min(),
                "prefill_len/mean": results_df.groupby("example_id").prefill_len.mean().mean(),
                "prefill_len/max": results_df.groupby("example_id").prefill_len.mean().max(),
                "prefill_len/min": results_df.groupby("example_id").prefill_len.mean().min(),
                "decode_len/mean": results_df.groupby("example_id").decode_len.mean().mean(),
                "decode_len/max": results_df.groupby("example_id").decode_len.mean().max(),
                "decode_len/min": results_df.groupby("example_id").decode_len.mean().min(),
                "is_truncated/mean": results_df.groupby("example_id").is_truncated.mean().mean(),
                "is_truncated/max": results_df.groupby("example_id").is_truncated.mean().max(),
                "is_truncated/min": results_df.groupby("example_id").is_truncated.mean().min(),
                "stop_condition/generation_truncated": (
                    results_df.is_truncated & (results_df.stop_condition != "prompt_too_long")
                ).mean(),
                # Log rate of each stop condition (e.g. max_turns, prompt_too_long, has_error)
                **{
                    f"stop_condition/{sc}": rate
                    for sc, rate in results_df.stop_condition.dropna().value_counts(normalize=True).items()
                },
                # Seqs per rollout metrics
                "samples_per_rollout/mean": results_df.groupby("example_id").samples_per_rollout.mean().mean(),
                "samples_per_rollout/max": results_df.groupby("example_id").samples_per_rollout.mean().max(),
                "samples_per_rollout/min": results_df.groupby("example_id").samples_per_rollout.mean().min(),
                # Turn metrics
                "num_turns/mean": results_df.groupby("example_id").num_turns.mean().mean(),
                "num_turns/max": results_df.groupby("example_id").num_turns.mean().max(),
                "num_turns/min": results_df.groupby("example_id").num_turns.mean().min(),
                # Verifier timing metrics
                "generation_ms/mean": results_df.groupby("example_id").generation_ms.mean().mean(),
                "generation_ms/max": results_df.groupby("example_id").generation_ms.mean().max(),
                "generation_ms/min": results_df.groupby("example_id").generation_ms.mean().min(),
                "scoring_ms/mean": results_df.groupby("example_id").scoring_ms.mean().mean(),
                "scoring_ms/max": results_df.groupby("example_id").scoring_ms.mean().max(),
                "scoring_ms/min": results_df.groupby("example_id").scoring_ms.mean().min(),
                # Performance metrics
                "perf/throughput": throughput,
                # Train reward
                "reward/mean": results_df.reward.mean(),
                "reward/std": results_df.reward.std(),
                "reward/min": results_df.reward.min(),
                "reward/max": results_df.reward.max(),
                "reward/median": results_df.reward.median(),
                "sampling/temperature": temperature,
                # Batch metrics
                "batch/solve_none": solve_none,
                "batch/solve_all": solve_all,
                "batch/effective_batch_size": effective_batch_size,
                # Error metrics
                "error/mean": (~results_df.error.isna()).mean(),
                **{
                    f"error/{error}": error_rate
                    for error, error_rate in results_df.error.dropna()
                    .apply(lambda e: e.get("error") if isinstance(e, dict) else e)
                    .value_counts(normalize=True)
                    .items()
                },
                # Env metrics
                **{f"metrics/{metric}": metrics_df[metric].mean() for metric in metrics_df.columns},
                # RULER step totals (gated on config.ruler.enabled; empty when disabled)
                **ruler_step_totals,
                # Time metrics
                "time/step": step_time,
                "time/generate_completions": generate_completions_time,
                "time/teacher_logprobs": teacher_logprobs_time,
                "time/save_ckpt": save_ckpt_time,
                "time/parallel_preprocess": parallel_preprocess_time,
                # Scheduler metrics
                **scheduler.get_metrics(),
                # Buffer metrics
                **buffer.get_metrics(),
                # Event loop lag metrics
                **event_loop_lag_monitor.get_metrics(),
                # Rollout filter metrics
                **filter_metrics,
                # W&B axis
                "step": progress.step,
            }

            # If more than one env, add per-env metrics
            if results_df.task.nunique() > 1:
                per_env_reward = results_df.groupby("task").reward.mean().to_dict()
                to_log.update({f"reward/{env}": reward for env, reward in per_env_reward.items()})

                per_env_ratio = results_df.task.value_counts(normalize=True).to_dict()
                to_log.update({f"batch/{env}": ratio for env, ratio in per_env_ratio.items()})

            # Optionally, add val metrics
            if val_results_df is not None:
                to_log.update(
                    {
                        "val_reward/mean": val_results_df.reward.mean(),
                        "val_reward/std": val_results_df.reward.std(),
                        "val_reward/min": val_results_df.reward.min(),
                        "val_reward/max": val_results_df.reward.max(),
                        "val_reward/median": val_results_df.reward.median(),
                    }
                )

                if val_results_df.task.nunique() > 1:
                    per_env_reward = val_results_df.groupby("task").reward.mean().to_dict()
                    to_log.update({f"val_reward/{env}": reward for env, reward in per_env_reward.items()})

                    per_env_ratio = val_results_df.task.value_counts(normalize=True).to_dict()
                    to_log.update({f"val_batch/{env}": ratio for env, ratio in per_env_ratio.items()})

            # Log metrics to monitor(s)
            monitor.log(to_log, step=progress.step)

            # Dump raw per-step metrics to JSONL if enabled
            if metrics_dump_file is not None:
                entry = {"step": progress.step, "ts": time.time(), **to_log}
                metrics_dump_file.write(json.dumps(entry, default=str) + "\n")
                metrics_dump_file.flush()

            # Log samples to monitor(s) if enabled
            subset_train_rollouts = random.sample(train_rollouts, min(8, len(train_rollouts)))
            monitor.log_samples(subset_train_rollouts, step=progress.step)

            # Log distributions (rewards, advantages) if enabled
            monitor.log_distributions(
                distributions={
                    "rewards": rewards,
                    "advantages": advantages,
                },
                step=progress.step,
            )

            # Flush all accumulated metrics for this step
            monitor.flush(step=progress.step)

            step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Reward: {results_df.reward.mean():.4f} |{f' Val. Reward: {val_results_df.reward.mean():.4f} |' if val_results_df is not None else ''} Throughput: {throughput:.1f} tokens/s | Seq. Length: {results_df.groupby('example_id').seq_len.mean().mean():.1f} tokens/sample | Async Level: {scheduler.async_level} | Max. Off-Policy Level: {scheduler.max_off_policy_level}"
            logger.success(step_message)

            # Increment step
            progress.step += 1
            is_first_step = False

            event_loop_lag_monitor.reset()

        if config.eval:
            logger.info("Running final evals")
            results = await asyncio.gather(
                *[
                    evaluate_env(
                        env=eval_env,
                        env_name=eval_env_name,
                        get_client=inference_pool.get_next_client,
                        model_name=scheduler.model_name,
                        sampling_args=eval_sampling_args,
                        num_examples=eval_env_config.num_examples or config.eval.num_examples,
                        rollouts_per_example=eval_env_config.rollouts_per_example or config.eval.rollouts_per_example,
                        max_retries=eval_env_config.max_retries,
                        ckpt_step=ckpt_step,
                        step=progress.step,
                    )
                    for eval_env, eval_env_name, eval_env_config in zip(eval_envs, eval_env_names, config.eval.env)
                ]
            )
            _dump_eval_metrics(results)

        # Log final (immutable) samples and distributions to monitor(s)
        monitor.log_final_samples()
        monitor.save_final_summary()

        # Write final checkpoint
        if ckpt_manager is not None:
            logger.info("Writing final checkpoint")
            ckpt_manager.save(progress, buffer, step=progress.step)

        # Close metrics dump file
        if metrics_dump_file is not None:
            metrics_dump_file.close()

        # Close training batch sender
        training_batch_sender.close()

        # Shutdown rollout executor
        rollout_executor.shutdown(wait=False)

        # Stop scheduler
        await scheduler.stop()

        # Stop inference pool
        await inference_pool.stop()

        if teacher_inference_pool is not None:
            await teacher_inference_pool.stop()

        if ruler_judge_pool is not None:
            await ruler_judge_pool.aclose()

        # Cancel event loop lag monitor task
        await safe_cancel(event_loop_lag_monitor_task)

        # Shutdown env processes. Use SIGKILL to avoid noisy tracebacks from the
        # verifiers library's signal handler which raises exceptions inside running
        # coroutines, causing "Task exception was never retrieved" warnings.
        # Graceful cleanup isn't needed here since the inference pool and scheduler
        # are already stopped — no more requests are in flight.
        for process in env_processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=5)
    finally:
        _stop_env_log_listener()

    logger.success("Orchestrator finished.")

    # Optionally, print benchmark table
    if config.bench:
        print_benchmark(to_col_format(monitor.history))


def grpo_orchestrator(config: GRPOOrchestratorConfig):
    """Main entry-point for orchestrator."""
    asyncio.run(orchestrate(config))
