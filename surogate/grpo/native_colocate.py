"""Synchronous GRPO with one resident base shared by training and native serving."""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from pathlib import Path

from surogate.core.config.grpo_inference_config import SERVING_OFFLOAD_FIELDS
from surogate.grpo.shared_model import SharedModelServer, shared_execution
from surogate.grpo.shared_weights import adapter_modules, borrow_weights, write_shared_artifact
from surogate.utils.logger import get_logger

logger = get_logger()


def validate_configs(train, infer, orch):
    """Refuse modes whose weight storage cannot be borrowed safely yet."""
    if train.gpus != 1 or infer.tp != 1 or infer.dp != 1:
        raise ValueError("native GRPO colocate currently requires gpus: 1, tp: 1 and dp: 1")
    for name in SERVING_OFFLOAD_FIELDS:
        if getattr(infer, name, None) is not None:
            raise ValueError(
                f"native GRPO colocate does not support infer.{name}; "
                "use split-GPU GRPO or grpo-infer for serving offload settings"
            )
    if not train.lora or train.recipe != "bf16" or train.qlora_config is not None:
        raise ValueError("native GRPO colocate currently requires lora: true, recipe: bf16, without QLoRA")
    if train.master_dtype not in (None, "bf16"):
        raise ValueError("native GRPO colocate requires BF16 master weights")
    for name in ("offload_master", "offload_quants", "cpu_training", "dispatch_pp", "full_shard"):
        if getattr(train, name, False):
            raise ValueError(f"native GRPO colocate does not support {name}")
    if getattr(train, "ep_size", 1) != 1:
        raise ValueError("native GRPO colocate requires ep_size: 1")
    if train.noise_scheduler and train.noise_scheduler.enabled:
        raise ValueError("native GRPO colocate does not yet support QeRL weight noise")
    if train.transport_type != "filesystem" or orch.rollout_transport.type != "filesystem":
        raise ValueError("native GRPO colocate currently requires filesystem batch transport")
    if train.max_steps != orch.max_steps or train.max_steps <= 0:
        raise ValueError("native GRPO colocate requires matching positive max_steps in train and orch")
    if orch.sequence_len != train.sequence_len:
        raise ValueError("native GRPO colocate requires matching sequence_len in train and orch")
    if infer.max_model_len and infer.max_model_len > train.sequence_len:
        raise ValueError("max_model_len must not exceed the training sequence_len")
    if Path(orch.output_dir).resolve().parent != Path(train.output_dir).resolve():
        raise ValueError("orch.output_dir must be a run directory directly inside train.output_dir")
    if not Path(orch.output_dir).name.startswith("run_"):
        raise ValueError("orch.output_dir must use a run_ name, for example run_default")
    if infer.model != train.model or orch.model.name != train.model:
        raise ValueError("train, infer and orch must name the same model")
    if not infer.enable_lora or not orch.model.lora_adapter:
        raise ValueError("set infer.enable_lora: true and orch.model.lora_adapter")
    run = Path(orch.output_dir)
    output = Path(train.output_dir)
    if any(p.resolve() != run.resolve() for p in output.glob("run_*")):
        raise ValueError("native GRPO colocate requires one orchestrator run directory")
    if getattr(train, "lora_dropout", 0):
        raise ValueError("native GRPO colocate requires lora_dropout: 0 for policy scoring")
    config = json.loads((Path(train.model_dir) / "config.json").read_text())
    shared_execution(config, train.lora_target_modules)
    text = config.get("text_config", config)
    moe = text.get("num_experts", text.get("num_local_experts", text.get("n_routed_experts", 0)))
    if moe and getattr(train, "sequence_chunks", 1) > 1:
        raise ValueError("native MoE policy scoring requires sequence_chunks: 1; use long_context to reduce memory")
    if moe and not getattr(train, "doc_masking", True):
        raise ValueError("native MoE policy scoring requires doc_masking: true")
    if (moe and
            getattr(train, "lora_dtype", "fp32") != "bf16"):
        raise ValueError("MoE shared-model GRPO requires lora_dtype: bf16 for expert adapter training")


class SharedPolicy:
    """One phase owner, with an explicit version for every completed update."""

    def __init__(self, server, train, orch, *, start_step=0, checkpoints=None):
        self.server = server
        self.name = orch.model.lora_adapter
        self.scale = train.lora_alpha / train.lora_rank
        self.directory = Path(orch.output_dir) / "broadcasts"
        self.stop_event = threading.Event()
        self.start_step = start_step
        self.version = start_step - 1
        self.checkpoints = checkpoints
        self.training = True
        self.lock = threading.Lock()
        self.report_path = Path(train.output_dir) / "shared_weights.jsonl"

    def check_cancelled(self):
        if self.stop_event.is_set():
            raise InterruptedError("shared GRPO run stopped")

    def _report(self, phase):
        summary = dict(self.server.summary()) | {"phase": phase}
        with self.report_path.open("a") as stream:
            stream.write(json.dumps(summary) + "\n")
        logger.info(f"Shared GRPO {phase}: policy={summary['policy_version']}, "
                    f"base={summary['shared_base_bytes'] / 2**30:.3f} GiB, "
                    f"serving base allocation={summary['serving_base_allocated_bytes']} bytes, "
                    f"base upload={summary['base_upload_bytes']} bytes")

    def begin_training(self, step):
        with self.lock:
            self.check_cancelled()
            if self.training or step != self.version:
                raise RuntimeError(f"training batch {step} does not match serving policy {self.version}")
            self.server.begin_training()
            self.training = True
            self._report("training")

    def broadcast(self, trainer, step):
        with self.lock:
            self.check_cancelled()
            if step == self.version:
                return  # the existing trainer also publishes once at final teardown
            if not self.training or step != self.version + 1:
                raise RuntimeError(f"unexpected policy publication {step} after {self.version}")
            modules = [] if getattr(self.server, "uses_live_adapter", False) else adapter_modules(trainer, self.scale)
            self.server.publish(self.name, modules, step)
            self.version = step
            self.training = False
            self._report("rollouts")
            directory = self.directory / f"step_{step}"
            directory.mkdir(parents=True, exist_ok=True)
            # Coordination only. No model or adapter files are written per step.
            (directory / "STABLE").touch()
            if step > self.start_step:
                from surogate.grpo.runs import get_multi_run_manager

                manager = get_multi_run_manager()
                for idx in manager.used_idxs:
                    manager.ready_to_update[idx] = False

    def cleanup(self, step):
        pass

    def save_checkpoint(self, trainer, next_step):
        if self.checkpoints is not None:
            self.checkpoints.save_training(trainer, next_step)

    async def acknowledge(self, weight_dir, lora_name=None, step=0):
        with self.lock:
            if self.training or step != self.version or lora_name != self.name:
                raise RuntimeError(f"requested policy {step} is not the published shared policy")


class SharedInferencePool:
    """Use the normal rollout client and acknowledge already-published adapters."""

    def __init__(self, pool, policy):
        self.pool, self.policy = pool, policy

    def __getattr__(self, name):
        return getattr(self.pool, name)

    async def update_weights(self, weight_dir, lora_name=None, step=0):
        await self.policy.acknowledge(weight_dir, lora_name, step)


def grpo_native_colocate(train_config, infer_config, orch_config):
    validate_configs(train_config, infer_config, orch_config)
    from surogate.grpo.native_checkpoint import NativeCheckpointCoordinator
    from surogate.grpo.trainer import GRPOTrainer

    checkpoints = NativeCheckpointCoordinator(train_config, orch_config)
    checkpoints.prepare_resume()

    # Interleaving requires a complete batch from one policy before its update.
    train_config.max_async_level = 0
    orch_config.max_async_level = 0
    orch_config.strict_async_level = True
    # The native runner owns publication; the trainer's default broadcaster is
    # replaced before train() and never exports step adapters.
    Path(orch_config.output_dir).mkdir(parents=True, exist_ok=True)
    context = infer_config.max_model_len or train_config.sequence_len
    concurrency = infer_config.max_num_seqs or min(16, orch_config.batch_size or 16)
    settings = dict(host=infer_config.host or "127.0.0.1", port=infer_config.port or 8000,
                    device=0, model=orch_config.model.name, max_context=context,
                    prefill_chunk=min(getattr(infer_config, "decode_prefill_chunk", 256), context), max_concurrency=concurrency,
                    kv_capacity=context * concurrency, use_cuda_graph=True,
                    decode_cache_bytes=getattr(infer_config, "decode_cache_bytes", 0),
                    decode_prefix_entries=getattr(infer_config, "decode_prefix_entries", 32),
                    decode_memory_bytes=getattr(infer_config, "decode_memory_bytes", 0),
                    initial_policy_version=checkpoints.resume_step - 1,
                    rank=train_config.lora_rank)
    # Enforce the local server address; both components live in this process.
    orch_config.client.base_url = [f"http://127.0.0.1:{settings['port']}/v1"]
    trainer, server = None, None
    config = json.loads((Path(train_config.model_dir) / "config.json").read_text())
    text_config = config.get("text_config", config)
    if text_config.get("num_experts", text_config.get("num_local_experts", text_config.get("n_routed_experts", 0))):
        train_config.runtime_config.moe_rollout_parity = True
    if config.get("model_type") == "glm5_next" or text_config.get("model_type") in ("glm5_next", "glm5_next_text"):
        train_config.runtime_config.glm_rollout_parity = True
        logger.info("GLM policy scoring uses recurrent FLA KDA and fixed-reduction GEMMs.")
    generation_config = Path(train_config.model_dir) / "generation_config.json"
    if generation_config.is_file():
        eos = json.loads(generation_config.read_text()).get("eos_token_id")
        if eos is not None:
            settings["eos_token_id"] = eos
    execution = shared_execution(config, train_config.lora_target_modules, getattr(train_config, "tokenizer", None))
    with tempfile.TemporaryDirectory(prefix="surogate-shared-grpo-") as temporary:
        artifact = Path(temporary) / "model.sinfer"
        bindings = write_shared_artifact(train_config.model_dir, artifact) if execution == "serve" else None
        try:
            trainer = GRPOTrainer(train_config, resume_checkpoint=checkpoints.trainer_resume)
            if execution == "serve":
                from surogate import _surogate_serve
                weights = borrow_weights(trainer.trainer, bindings)
                server = _surogate_serve.SharedServer(str(artifact), weights, settings)
            else:
                logger.info("Shared-model rollouts use the resident training model.")
                server = SharedModelServer(trainer.trainer, train_config.tokenizer, config, settings)
            policy = SharedPolicy(server, train_config, orch_config, start_step=checkpoints.resume_step, checkpoints=checkpoints)
            trainer.phase_controller = policy
            trainer.broadcast = policy
            policy.broadcast(trainer.trainer, checkpoints.resume_step)
            asyncio.run(_run(trainer, policy, orch_config))
        finally:
            if server is not None:
                server.close()
            if trainer is not None:
                trainer.close()


async def _run(trainer, policy, config):
    from surogate.grpo.orchestrator.grpo_orch import orchestrate
    from surogate.grpo.utils.client import setup_inference_pool

    client_type = "openai_chat_completions_token" if config.use_token_client else "openai_chat_completions"
    pool = await setup_inference_pool(config.client, model_name=config.model.name, client_type=client_type)
    training = asyncio.create_task(asyncio.to_thread(trainer.train))
    orchestrator = asyncio.create_task(orchestrate(config, inference_pool=SharedInferencePool(pool, policy),
                                                  initial_policy_name=policy.name, prefetch_batches=False,
                                                  checkpoint_coordinator=getattr(policy, "checkpoints", None)))
    try:
        await asyncio.gather(asyncio.shield(training), orchestrator)
    finally:
        policy.stop_event.set()
        if not orchestrator.done():
            orchestrator.cancel()
        await asyncio.gather(orchestrator, return_exceptions=True)
        # Cancelling a to_thread task doesn't stop its thread. Let the packer's
        # cancellation check stop it before destroying any borrowed storage.
        if not training.done():
            await asyncio.shield(asyncio.gather(training, return_exceptions=True))
        await pool.stop()
