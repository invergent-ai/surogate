"""Main GRPO trainer

Training loop:
    1. Wait for batch from orchestrator (packer on master, transport to all ranks)
    2. For each micro-batch (packed sequence with multiple samples):
       a. step_grpo_native() on packed batch -> forward + native GRPO dloss + backward
    3. update_with_config() -> optimizer step
    4. Broadcast updated weights to inference engine

Document-level attention masking (Flash Attention varlen) can be enabled to
prevent cross-sample attention in packed sequences.
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np

from surogate import _surogate
from surogate.grpo.config import GRPOTrainConfig
from surogate.grpo.data import GRPODataLoader
from surogate.grpo.loss import compute_grpo_per_token_grads
from surogate.grpo.runs import get_multi_run_manager
from surogate.grpo.turn_stats import TurnAccumulator
from surogate.grpo.weight_broadcast import SurogateWeightBroadcast
from surogate.train.lr_schedule import LRSchedule
from surogate.train.metrics_writer import MetricsWriter
from surogate.utils.hf import get_model_weights_path
from surogate.utils.logger import get_logger
from surogate.utils.lora_compat import ensure_vllm_lora_compat
from surogate.utils.tensor import to_surogate_dtype

logger = get_logger()


def _filtered_config_for_logging(config: GRPOTrainConfig) -> dict:
    """Flatten config into JSON-serializable scalars for metrics_writer.log_config()."""
    raw = dict(vars(config))
    raw.pop("model_info", None)
    raw.pop("model", None)
    raw.pop("tokenizer", None)
    out: dict[str, bool | int | float | str] = {}
    for k, v in raw.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (bool, int, float, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _find_sample_boundaries(position_ids_flat: np.ndarray) -> list[tuple[int, int]]:
    """Find sample boundaries in packed position_ids.

    Packed sequences reset position_ids at each sample boundary (e.g.
    [0,1,2,0,1,0,1,2,3]).  Returns (start, end) tuples for each sample.
    """
    boundaries = [0]
    for i in range(1, len(position_ids_flat)):
        if position_ids_flat[i] == 0 and position_ids_flat[i - 1] != 0:
            # Only treat as a new sample if the next position is 1.
            if i + 1 < len(position_ids_flat) and position_ids_flat[i + 1] == 1:
                boundaries.append(i)
    ranges: list[tuple[int, int]] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(position_ids_flat)
        ranges.append((start, end))
    return ranges


class GRPOTrainer:
    """GRPO RL trainer using Surogate's C++ engine."""

    def __init__(self, config: GRPOTrainConfig, external_weights: list[list[dict]] | None = None):
        self.config = config

        # Build DSL IR for the model (same pattern as SurogateTrainerWrapper)
        from surogate.dsl.ir_builder import build_dsl_ir_for_model

        dsl_extra = {}
        if getattr(config, "ep_size", 1) > 1:
            dsl_extra["ep_size"] = config.ep_size
        ir_json = build_dsl_ir_for_model(config.model_dir, extra_config=dsl_extra or None)
        config.runtime_config.dsl_ir_json = ir_json

        # Compile JIT kernels (e.g. gated delta rule Triton kernels)
        from surogate.kernels.jit_compile import compile_jit_kernels

        jit_manifests = compile_jit_kernels(ir_json)
        if jit_manifests:
            config.runtime_config.jit_kernel_manifests = jit_manifests

        # Create C++ trainer using inherited config objects
        logger.info(f"Creating GRPO trainer for {config.model} ({config.gpus} GPUs)")
        self.trainer = _surogate.SurogateTrainer(
            ngpu=config.gpus,
            config=_surogate.PretrainedConfig.from_pretrained(config.model_dir, to_surogate_dtype(config.torch_dtype)),
            options=config.runtime_config,
            batch_size=config.per_device_train_batch_size,
            # Graph sequence length is the CHUNK size under chunked-sequence
            # training (trainer_seq_len = sequence_len // sequence_chunks),
            # exactly as the SFT wrapper passes it. Passing the full
            # sequence_len here built an unchunked graph whose saved-tensor
            # cache cannot fit long sequences (found 2026-07-27 bringing up
            # the first chunked GRPO run, on the 27B hybrid).
            seq_len=config.trainer_seq_len,
            grad_accum=1,  # Set dynamically per step via set_grad_accumulation()
            memcpy_all_gather=config.memcpy_all_gather,
            memcpy_send_recv=config.memcpy_send_recv,
            lora_config=config.lora_config,
            qlora_config=config.qlora_config,
        )

        # uses shift_tensor_right with pad_value=log(1/vocab_size).
        self._pad_logprob = None
        try:
            from surogate.core.model.hf_config import HfConfigFactory

            vocab_size = HfConfigFactory.get_config_attr(config.model_info.config, "vocab_size")
        except Exception:
            vocab_size = None
        if vocab_size:
            self._pad_logprob = float(np.log(1.0 / float(vocab_size)))

        # Import pretrained weights.
        #
        # Adapter-init protocol (2026-07-27): the SFT wrapper calls
        # configure_initial_adapter BEFORE import (adapter_init_mode=merge
        # registers the adapter via set_adapter_path so the native import
        # merges it in-stream) and import_initial_trainable_adapter after.
        # GRPOTrainer skipped both, silently IGNORING adapter_path — a run
        # configured to start from base+adapter would have trained from the
        # raw base.
        from surogate.train.adapter_init import (
            configure_initial_adapter,
            import_initial_trainable_adapter,
        )

        initial_adapter = configure_initial_adapter(
            config, self.trainer, fresh_run=True)
        model_weights_path = get_model_weights_path(config.model_dir)
        if external_weights is not None:
            # Zero-copy import from external GPU pointers (colocate mode with vLLM)
            logger.info(f"Importing weights from external GPU pointers (non-quantized from {model_weights_path})")
            self.trainer.import_weights_from_external(model_weights_path, external_weights)
        else:
            logger.info(f"Importing weights from {model_weights_path}")
            self.trainer.import_weights(model_weights_path)
        import_initial_trainable_adapter(self.trainer, initial_adapter)

        # loss_scale is computed dynamically per pack — see train() loop

        # LR schedule — max_steps here is "orchestrator steps" (one per pack() cycle).
        # The internal micro-step counter is separate.
        lr_max_steps = config.max_steps if config.max_steps > 0 else 1_000_000
        warmup_steps = config.warmup_steps
        if warmup_steps == 0 and config.warmup_ratio > 0:
            warmup_steps = int(lr_max_steps * config.warmup_ratio)

        self.lr_schedule = LRSchedule(
            base_lr=config.learning_rate,
            max_steps=lr_max_steps,
            warmup_steps=warmup_steps,
            cooldown_steps=config.cooldown_steps,
            final_lr=config.learning_rate * config.final_lr_fraction,
            schedule_type=config.lr_scheduler_type,
            wsd_decay_steps_fraction=config.wsd_decay_steps_fraction,
        )

        # Weight broadcast (with optional QeRL noise injection)
        if config.weight_broadcast_type == "colocate":
            from surogate.grpo.weight_broadcast import ColocateWeightBroadcast

            self.broadcast = ColocateWeightBroadcast(
                output_dir=config.output_dir,
                max_async_level=config.max_async_level,
                noise_config=config.noise_scheduler,
                base_model_dir=config.model_dir,
                max_steps=config.max_steps,
            )
        else:
            self.broadcast = SurogateWeightBroadcast(
                output_dir=config.output_dir,
                adapter_only=config.lora,
                max_async_level=config.max_async_level,
                noise_config=config.noise_scheduler,
                base_model_dir=config.model_dir,
                max_steps=config.max_steps,
            )

        # Data loader setup is deferred to train() since packer must run first
        self.data_loader: GRPODataLoader | None = None
        self.packer = None

        # Metrics from the Python loss path (turn_diagnostics mode only)
        self._diag_metrics: dict[str, float] | None = None
        self._turn_rows_path = Path(config.output_dir) / "turn_stats.jsonl"

        # dispatch-PP: stage the forward/backward across the GPU pool so a model
        # whose fp32 master does not fit resident can still train. GRPO drives the
        # two halves itself (staged forward -> Python per-token grads -> staged
        # backward seeded with them), because its gradients do not come from
        # cross-entropy.
        self._dispatch_pp = getattr(config, "parallelism", None) == "dispatch_pp"
        self._dpp_los: list[int] | None = None
        self._dpp_his: list[int] | None = None
        if self._dispatch_pp:
            from surogate.train.dispatch_pp import plan_stages

            self._dpp_los, self._dpp_his, n_layers, n_stages, max_stage, sb = plan_stages(
                config.model_info.config, config.gpus
            )
            logger.info(
                f"[dispatch_pp] GRPO: {n_stages} stages over {n_layers} layers "
                f"(max {max_stage} blocks/stage, stage_blocks={sb})"
            )

        # Optional metrics writers (driven by config.report_to)
        self.metrics_writer: MetricsWriter | None = None
        backends = config.report_to or []
        if isinstance(backends, str):
            backends = [backends]
        if "surogate" in backends:
            self.metrics_writer = MetricsWriter(output_path=config.surogate_metrics_path)
            self.metrics_writer.log_config(**_filtered_config_for_logging(config))

    def _copy_tokenizer_files(self, src_dir: str, dst_dir: str):
        """Copy tokenizer, vocab, and config files from source model to output directory."""
        tokenizer_files = [
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "chat_template.jinja",
            "generation_config.json",
        ]
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        for filename in tokenizer_files:
            src = src_path / filename
            if src.exists():
                shutil.copy(src, dst_path / filename)

    def _log_step(self, *, step, orch_step, step_metrics, result, lr, n_mb, vtc,
                  expected_loss_scale, step_time, turn_acc) -> None:
        """Turn diagnostics + step logging + metrics; shared by the native,
        diagnostic and dispatch-PP micro-step paths so all three report identically."""
        config = self.config
        # 5b. Turn-resolved diagnostics
        turn_summary: dict[str, float] = {}
        if turn_acc is not None:
            turn_summary = turn_acc.summary()
            self._write_turn_rows(step, orch_step, turn_acc)
            if turn_summary:
                logger.info(
                    f"  turns={turn_summary['turn/num_turns_observed']:.0f} "
                    f"deep/shallow_kl={turn_summary['turn/deep_shallow_kl_ratio']:.0%} "
                    f"deep_support={turn_summary['turn/deep_support']:.1%} "
                    f"deep_kl_budget={turn_summary['turn/deep_kl_budget']:.1%} "
                    f"deep_loss_budget={turn_summary['turn/deep_loss_budget']:.1%} "
                    f"turn0_budget={turn_summary['turn/turn0_loss_budget']:.1%}"
                )
            else:
                logger.warning(
                    "turn_diagnostics enabled but no turn ids reached the trainer "
                    "(single-turn env, or orchestrator predates the turn_ids plumbing)"
                )

        # 6. Logging
        if step % config.logging_steps == 0:
            logger.info(
                f"step={step} loss={step_metrics.get('policy_loss', 0):.4f} "
                f"grad_norm={result['norm']:.4f} lr={lr:.2e} "
                f"kl={step_metrics.get('mismatch_kl', 0):.4f} "
                f"masked={step_metrics.get('is_masked', 0):.2%} "
                f"tokens={step_metrics.get('total_tokens', 0)} "
                f"micro_batches={n_mb} "
                f"vtc={vtc} expected={expected_loss_scale} "
                f"time={step_time:.2f}s"
            )

            if self.metrics_writer is not None:
                duration_ms = step_time * 1000.0
                total_tokens = int(step_metrics.get("total_tokens", 0))
                tps = total_tokens / step_time if step_time > 0 else 0.0
                self.metrics_writer.track(
                    step,
                    **{
                        "train/policy_loss": float(step_metrics.get("policy_loss", 0.0)),
                        "train/mismatch_kl": float(step_metrics.get("mismatch_kl", 0.0)),
                        "train/masked_mismatch_kl": float(step_metrics.get("masked_mismatch_kl", 0.0)),
                        "train/unmasked_mismatch_kl": float(step_metrics.get("unmasked_mismatch_kl", 0.0)),
                        "train/is_masked": float(step_metrics.get("is_masked", 0.0)),
                        "train/is_masked_low": float(step_metrics.get("is_masked_low", 0.0)),
                        "train/is_masked_high": float(step_metrics.get("is_masked_high", 0.0)),
                        "train/teacher_kl": float(step_metrics.get("teacher_kl", 0.0)),
                        "train/keep_tokens": float(step_metrics.get("keep_tokens", 0.0)),
                        "train/total_tokens": float(total_tokens),
                        "train/grad_norm": float(result["norm"]),
                        "train/lr": float(lr),
                        "train/micro_batches": float(n_mb),
                        "train/vtc": float(vtc),
                        "train/expected_loss_scale": float(expected_loss_scale),
                        "train/duration_ms": duration_ms,
                        "train/tokens_per_second": tps,
                        "train/orch_step": float(orch_step),
                        **turn_summary,
                    },
                )

    def _prepare_micro_batch(self, mb, seq_len: int) -> dict:
        """Pad one micro-batch's per-token arrays to seq_len.

        Shared by the native, diagnostic and dispatch-PP paths so all three see
        identical inputs; the only difference between them is how the gradients
        are produced and applied.
        """
        orig_input_ids = mb["input_ids"]
        orig_position_ids = mb["position_ids"]
        orig_targets = mb["targets"]
        T_actual = orig_input_ids.shape[1]
        loss_mask_flat = mb["loss_mask"].flatten()

        sample_ranges = _find_sample_boundaries(orig_position_ids.flatten())

        input_padded = np.zeros((1, seq_len), dtype=np.int32)
        input_padded[0, :T_actual] = orig_input_ids[0, :T_actual]

        pos_padded = np.zeros((1, seq_len), dtype=np.int32)
        pos_padded[0, :T_actual] = orig_position_ids[0, :T_actual]
        if T_actual < seq_len:
            last_pos = int(pos_padded[0, T_actual - 1])
            pos_padded[0, T_actual:] = np.arange(last_pos + 1, last_pos + 1 + seq_len - T_actual, dtype=np.int32)

        temp_padded = np.ones((1, seq_len), dtype=np.float32)
        temp_padded[0, :T_actual] = mb["temperatures"].flatten()[:T_actual]

        targets_padded = np.full((1, seq_len), -100, dtype=np.int32)
        targets_padded[0, :T_actual] = orig_targets[0, :T_actual]
        tmask = loss_mask_flat[:T_actual].astype(bool)
        tmask_shifted = np.zeros_like(tmask)
        if T_actual > 1:
            tmask_shifted[:-1] = tmask[1:]
        targets_padded[0, :T_actual][~tmask_shifted] = -100

        inference_padded = np.zeros(seq_len, dtype=np.float32)
        inference_padded[:T_actual] = mb["inference_logprobs"].flatten()[:T_actual]
        advantages_padded = np.zeros(seq_len, dtype=np.float32)
        advantages_padded[:T_actual] = mb["advantages"].flatten()[:T_actual]
        loss_mask_padded = np.zeros(seq_len, dtype=np.uint8)
        loss_mask_padded[:T_actual] = loss_mask_flat[:T_actual].astype(np.uint8)

        teacher_padded = None
        if mb["teacher_logprobs"] is not None:
            teacher_padded = np.zeros(seq_len, dtype=np.float32)
            teacher_padded[:T_actual] = mb["teacher_logprobs"].flatten()[:T_actual]

        turn_ids_padded = None
        if mb.get("turn_ids") is not None:
            turn_ids_padded = np.full(seq_len, -1, dtype=np.int32)
            turn_ids_padded[:T_actual] = mb["turn_ids"].flatten()[:T_actual]

        return {
            "input": input_padded,
            "position_ids": pos_padded,
            "temperatures": temp_padded,
            "targets": targets_padded,
            "inference": inference_padded,
            "advantages": advantages_padded,
            "loss_mask": loss_mask_padded,
            "teacher": teacher_padded,
            "turn_ids": turn_ids_padded,
            "sample_ranges": sample_ranges,
            "T_actual": T_actual,
        }

    def _dispatch_pp_step(self, micro_batches, loss_scale: float, opt_config, step: int, turn_acc) -> dict:
        """One optimizer step under dispatch-PP.

        GRPO's gradients do not come from cross-entropy, so the fused dispatch step
        cannot be used as-is. Instead the two halves are driven explicitly:

          1. staged forward through the loss stage -> per-token logprobs
          2. Python forms the GRPO per-token gradients from them
          3. staged backward seeded with those gradients (custom_dloss), then the
             existing collect + optimizer + broadcast

        GRPO's micro-batches map onto dispatch-PP's M microbatches directly: each is
        already [1, seq_len], and GRPO takes exactly one optimizer step per
        orchestrator step, which is what the fused call does.
        """
        seq_len = self.config.sequence_len
        M = len(micro_batches)

        prepared = [self._prepare_micro_batch(mb, seq_len) for mb in micro_batches]
        inputs = np.ascontiguousarray(np.concatenate([p["input"] for p in prepared], axis=0))
        targets = np.ascontiguousarray(np.concatenate([p["targets"] for p in prepared], axis=0))

        logprob_rows = self.trainer.dispatch_pp_forward_logprobs_multigpu(
            inputs, targets, self._dpp_los, self._dpp_his, M
        )

        custom_dloss = np.zeros((M, seq_len), dtype=np.float32)
        metrics: dict = {}
        for i, p in enumerate(prepared):
            # Un-shift from target-slot layout to logical layout (see
            # _diagnostic_micro_step for why this is required).
            buf = np.asarray(logprob_rows[i, :seq_len], dtype=np.float32)
            trainer_lp = np.zeros(seq_len, dtype=np.float32)
            for start, end in p["sample_ranges"]:
                if end - start > 1:
                    trainer_lp[start + 1 : end] = buf[start : end - 1]

            grads, metrics = compute_grpo_per_token_grads(
                trainer_logprobs=trainer_lp,
                inference_logprobs=p["inference"],
                advantages=p["advantages"],
                loss_mask=p["loss_mask"].astype(bool),
                loss_config=self.config.loss,
                sample_ranges=p["sample_ranges"],
                teacher_logprobs=p["teacher"],
            )

            if turn_acc is not None and p["turn_ids"] is not None:
                reference_lp = p["teacher"] if p["teacher"] is not None else p["inference"]
                turn_acc.update(
                    turn_ids=p["turn_ids"],
                    loss_mask=p["loss_mask"].astype(bool),
                    per_token_kl=trainer_lp - reference_lp,
                    per_token_loss=grads,
                    sample_ranges=p["sample_ranges"],
                )

            for start, end in p["sample_ranges"]:
                if end - start > 1:
                    custom_dloss[i, start : end - 1] = grads[start + 1 : end]
        custom_dloss /= loss_scale

        loss = float(
            self.trainer.dispatch_pp_train_step_multigpu(
                inputs, targets, self._dpp_los, self._dpp_his, opt_config, step, False, M, custom_dloss
            )
        )
        self._diag_metrics = metrics
        return {"loss": loss, "norm": float(self.trainer.dispatch_pp_last_grad_norm())}

    def _write_turn_rows(self, step: int, orch_step: int, turn_acc) -> None:
        """Append per-turn detail for this step to turn_stats.jsonl (for plotting)."""
        rows = turn_acc.as_rows()
        if not rows:
            return
        self._turn_rows_path.parent.mkdir(parents=True, exist_ok=True)
        with self._turn_rows_path.open("a") as fh:
            for row in rows:
                fh.write(json.dumps({"step": step, "orch_step": orch_step, **row}) + "\n")

    def _diagnostic_micro_step(
        self,
        *,
        turn_acc,
        turn_ids,
        input_step,
        targets_step,
        pos_step,
        temp_step,
        inference_padded,
        advantages_padded,
        loss_mask_padded,
        teacher_padded,
        sample_ranges,
        loss_scale: float,
        seq_len: int,
    ) -> None:
        """Micro-step that also records turn-resolved statistics.

        Decomposes the fused native step into forward -> Python loss -> backward
        so per-token trainer logprobs become visible. The gradients are identical
        to ``step_grpo_native`` (same reference implementation, asserted by
        tests/grpo/test_native_formula.py); the only cost is the Python round
        trip on the [1, T] logprob buffer.
        """
        logprobs = self.trainer.forward_for_grpo(input_step, targets_step, pos_step, temp_step)

        # forward_for_grpo returns the CE loss buffer negated, so it is in TARGET
        # slot layout: buffer[t] = log p(input_ids[t+1]). The Python loss works in
        # logical token layout (aligned with loss_mask / inference_logprobs), so
        # un-shift within each packed sample. This is the exact inverse of the
        # shift applied to the gradients below, and mirrors how the native kernel
        # reads losses[out_idx] as the logprob of logical token out_idx+1.
        buf = np.asarray(logprobs[0, :seq_len], dtype=np.float32)
        trainer_lp = np.zeros(seq_len, dtype=np.float32)
        for start, end in sample_ranges:
            if end - start > 1:
                trainer_lp[start + 1 : end] = buf[start : end - 1]

        loss_mask_bool = loss_mask_padded.astype(bool)
        per_token_grads, metrics = compute_grpo_per_token_grads(
            trainer_logprobs=trainer_lp,
            inference_logprobs=inference_padded,
            advantages=advantages_padded,
            loss_mask=loss_mask_bool,
            loss_config=self.config.loss,
            sample_ranges=sample_ranges,
            teacher_logprobs=teacher_padded,
        )

        # Per-token reverse-KL contribution, k1 estimator: log pi_S - log pi_T.
        #
        # Rollouts are sampled on-policy from pi_S, so E[k1] = KL(pi_S || pi_T)
        # exactly — this is the quantity TurnOPD's per-turn diagnostics are
        # defined on, and the same log-ratio this stack already uses as the OPD
        # reward (teacher_tau * (teacher_lp - trainer_lp)).
        #
        # Do NOT use the k3 estimator exp(r)-r-1 here. It is also unbiased, but
        # its variance explodes: an undertrained student against a much larger
        # teacher produces occasional 10-15 nat gaps, and exp() lets a handful of
        # tokens dominate every per-turn mean (observed: "mean KL" of 2.4e6).
        reference_lp = teacher_padded if teacher_padded is not None else inference_padded
        per_token_kl = trainer_lp - reference_lp

        if turn_ids is not None:
            assert turn_ids.shape == loss_mask_bool.shape, (
                f"turn_ids {turn_ids.shape} must be padded to seq_len like loss_mask {loss_mask_bool.shape}"
            )
            turn_acc.update(
                turn_ids=turn_ids,
                loss_mask=loss_mask_bool,
                per_token_kl=per_token_kl,
                per_token_loss=per_token_grads,
                sample_ranges=sample_ranges,
            )

        # Shift into the target layout the LM-head backward expects: the gradient
        # for logical token t is written at slot t-1 within its own sample.
        shifted = np.zeros((1, seq_len), dtype=np.float32)
        for start, end in sample_ranges:
            if end - start > 1:
                shifted[0, start : end - 1] = per_token_grads[start + 1 : end]
        shifted /= loss_scale

        ngpu = self.config.gpus
        grads_step = np.ascontiguousarray(np.tile(shifted, (ngpu, 1)) if ngpu > 1 else shifted)
        self.trainer.backward_grpo(grads_step)

        self._diag_metrics = metrics

    def _setup_data(self, start_step: int = 0):
        """Set up data loader and optionally packer (master only)."""
        config = self.config

        # Build transport config
        if config.transport_type == "zmq":
            from surogate.core.config.grpo_orch_config import ZMQTransportConfig

            transport_config = ZMQTransportConfig({})
        else:
            from surogate.core.config.grpo_orch_config import FileSystemTransportConfig

            transport_config = FileSystemTransportConfig({})

        # dp_rank = 0 for single-node (all GPUs on same node share data)
        dp_rank = 0
        dp_world_size = 1  # Surogate handles multi-GPU internally

        # Initialize MultiRunManager singleton (required before packer)
        from surogate.grpo.packer import init_multi_run_manager, setup_grpo_packer

        init_multi_run_manager(output_dir=config.output_dir)

        # Setup packer on master (packs TrainingBatch -> MicroBatch)
        tokenizer = config.tokenizer
        self.packer = setup_grpo_packer(
            dp_world_size=dp_world_size,
            seq_len=config.sequence_len,
            pad_to_multiple_of=config.pad_to_multiple_of,
            tokenizer=tokenizer,
            transport_config=transport_config,
            start_step=start_step,
        )
        # chunked GRPO: no packed-doc isolation in chunked training attention
        self.packer.single_sample_bins = bool(getattr(config, 'single_sample_bins', False))

        # Setup data loader (receives packed MicroBatches)
        self.data_loader = GRPODataLoader(
            output_dir=config.output_dir,
            dp_rank=dp_rank,
            start_step=start_step,
            transport_config=transport_config,
        )

    def train(self):
        """Main GRPO training loop."""
        config = self.config
        max_steps = config.max_steps

        self._setup_data(start_step=0)

        # Get MultiRunManager — packer auto-increments progress[0].step after
        # each pack() call.
        mrm = get_multi_run_manager()

        logger.info("Starting GRPO training loop")
        logger.info(f"  Model: {config.model}")
        logger.info(f"  GPUs: {config.gpus}")
        logger.info(f"  Sequence length: {config.sequence_len}")
        logger.info("  Gradient accumulation: dynamic (from packer, no fixed cap)")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  LoRA: enabled={config.lora}, rank={config.lora_rank}, alpha={config.lora_alpha}")
        logger.info(f"  Recipe: {config.recipe}")
        logger.info(f"  Optimizer: {config.optimizer}")
        logger.info(
            f"  Loss: kl_tau={config.loss.kl_tau}, adv_tau={config.loss.adv_tau}, "
            f"ipo_mask_low={config.loss.ipo_mask_low}, ipo_mask_high={config.loss.ipo_mask_high}"
        )
        logger.info(f"  Doc masking: {config.doc_masking}")
        if config.noise_scheduler and config.noise_scheduler.enabled:
            ns = config.noise_scheduler
            logger.info(
                f"  QeRL noise: sigma_start={ns.sigma_start}, sigma_end={ns.sigma_end}, num_stages={ns.num_stages}"
            )
        if max_steps > 0:
            logger.info(f"  Max steps (orchestrator): {max_steps}")
        else:
            logger.info("  Running indefinitely (waiting for orchestrator)")

        step = 0  # Internal trainer step (one per grad_accum chunk, for LR schedule + logging)
        while True:
            orch_step = mrm.progress[0].step if 0 in mrm.progress else 0

            # 1. Broadcast weights (after first orchestrator step)
            if orch_step > 0:
                self.broadcast.broadcast(self.trainer, orch_step)
                self.broadcast.cleanup(orch_step)

            # Check if we've reached max steps
            if 0 < max_steps <= orch_step:
                logger.info(f"Reached max steps ({max_steps}), stopping")
                break

            # 2. Pack and wait for batch — packer increments progress[0].step
            self.packer.pack()
            self.data_loader.wait_for_batch()

            # 3. Get micro-batches
            micro_batches = self.data_loader.get_batch()
            if not micro_batches:
                logger.warning("No micro-batches received, retrying...")
                continue

            # Accumulate ALL micro-batches from one pack into a single optimizer step.
            # No fixed gradient_accumulation_steps — the count is determined dynamically by the packer each step.
            seq_len = config.sequence_len
            n_mb = len(micro_batches)

            # Tell the C++ engine how many micro-steps this optimizer step has.
            self.trainer.set_grad_accumulation(n_mb)

            # Divide total loss by the sum of loss_mask
            # across all micro-batches in this optimizer step.
            loss_scale = int(sum(int(mb["loss_mask"].sum()) for mb in micro_batches))
            loss_scale = max(loss_scale, 1)

            step_start = time.time()

            # Note: loss_scale normalization is applied explicitly to per-token grads
            # (loss = total_loss / loss_scale).

            turn_acc = TurnAccumulator() if config.turn_diagnostics else None

            lr = self.lr_schedule.get_lr(orch_step)
            opt_config = _surogate.OptimizerConfig(
                optimizer=config.optimizer,
                learning_rate=lr,
                weight_decay=config.weight_decay,
                grad_clip=config.max_grad_norm,
                adamw_beta1=config.adamw_beta1,
                adamw_beta2=config.adamw_beta2,
                adamw_epsilon=config.adamw_epsilon,
            )

            if self._dispatch_pp:
                # dispatch-PP folds forward+backward+optimizer into one staged pass
                # over all micro-batches, so it replaces both the loop below and the
                # update_with_config call further down.
                vtc = 0
                expected_loss_scale = loss_scale
                result = self._dispatch_pp_step(micro_batches, float(loss_scale), opt_config, step, turn_acc)
                step_metrics = dict(self._diag_metrics or {})
                step_time = time.time() - step_start
                self._log_step(
                    step=step,
                    orch_step=orch_step,
                    step_metrics=step_metrics,
                    result=result,
                    lr=lr,
                    n_mb=n_mb,
                    vtc=vtc,
                    expected_loss_scale=expected_loss_scale,
                    step_time=step_time,
                    turn_acc=turn_acc,
                )
                if config.save_steps > 0 and step > 0 and step % config.save_steps == 0 and config.checkpoint_dir:
                    logger.info(f"Saving checkpoint at step {step}...")
                    self.trainer.save_checkpoint(config.checkpoint_dir, step)
                step += 1
                continue

            # 4. Process each micro-batch (gradients accumulate in C++)
            for mb in micro_batches:
                # Original micro-batch data
                orig_input_ids = mb["input_ids"]  # [1, T_mb]
                orig_position_ids = mb["position_ids"]  # [1, T_mb]
                orig_targets = mb["targets"]  # [1, T_mb]
                advantages_flat = mb["advantages"].flatten()
                inference_lp = mb["inference_logprobs"].flatten()
                loss_mask_flat = mb["loss_mask"].flatten()
                temps_flat = mb["temperatures"].flatten()

                teacher_lp = None
                if mb["teacher_logprobs"] is not None:
                    teacher_lp = mb["teacher_logprobs"].flatten()

                T_actual = orig_input_ids.shape[1]
                ngpu = config.gpus

                # Find sample boundaries for per-sample logprob/gradient shifting
                pos_flat = orig_position_ids.flatten()
                sample_ranges = _find_sample_boundaries(pos_flat)

                # --- Build logprob targets (global next-token shift across packed sequence) ---
                # shift_tensor_left on the packed input.
                logprob_targets = np.full((1, seq_len), -100, dtype=np.int32)
                logprob_targets[0, :T_actual] = orig_targets[0, :T_actual]

                # Pad inputs to seq_len. Always pad position_ids sequentially
                # from the last real position so RoPE resets are preserved.
                input_padded = np.zeros((1, seq_len), dtype=np.int32)
                input_padded[0, :T_actual] = orig_input_ids[0, :T_actual]
                pos_padded = np.zeros((1, seq_len), dtype=np.int32)
                pos_padded[0, :T_actual] = orig_position_ids[0, :T_actual]
                if T_actual < seq_len:
                    last_pos = int(pos_padded[0, T_actual - 1])
                    pos_padded[0, T_actual:] = np.arange(
                        last_pos + 1, last_pos + 1 + seq_len - T_actual, dtype=np.int32
                    )
                temp_padded = np.ones((1, seq_len), dtype=np.float32)
                temp_padded[0, :T_actual] = temps_flat[:T_actual]

                # --- Build backward targets with loss mask applied ---
                # Mask out prompt tokens so CE forward produces 0 (= logprob 0) for them.
                # This target array is used for BOTH forward (logprob extraction) and backward.
                targets_padded = logprob_targets.copy()
                tmask = loss_mask_flat[:T_actual].astype(bool)
                tmask_shifted = np.zeros_like(tmask)
                if T_actual > 1:
                    tmask_shifted[:-1] = tmask[1:]
                targets_padded[0, :T_actual][~tmask_shifted] = -100

                # Tile for multi-GPU before forward (forward_for_grpo uses same layout as step_with_custom_loss)
                input_step = input_padded
                pos_step = pos_padded
                temp_step = temp_padded
                targets_step = targets_padded
                if ngpu > 1:
                    input_step = np.tile(input_padded, (ngpu, 1))
                    pos_step = np.tile(pos_padded, (ngpu, 1))
                    targets_step = np.tile(targets_padded, (ngpu, 1))
                    temp_step = np.tile(temp_padded, (ngpu, 1))

                inference_padded = np.zeros(seq_len, dtype=np.float32)
                inference_padded[:T_actual] = inference_lp[:T_actual]
                advantages_padded = np.zeros(seq_len, dtype=np.float32)
                advantages_padded[:T_actual] = advantages_flat[:T_actual]
                loss_mask_padded = np.zeros(seq_len, dtype=np.uint8)
                loss_mask_padded[:T_actual] = loss_mask_flat[:T_actual].astype(np.uint8)
                # Turn ids must be padded to seq_len like every other per-token
                # array; padding belongs to no turn.
                turn_ids_padded = None
                if mb.get("turn_ids") is not None:
                    turn_ids_padded = np.full(seq_len, -1, dtype=np.int32)
                    turn_ids_flat = mb["turn_ids"].flatten()
                    turn_ids_padded[:T_actual] = turn_ids_flat[:T_actual]
                teacher_padded = None
                if teacher_lp is not None:
                    teacher_padded = np.zeros(seq_len, dtype=np.float32)
                    teacher_padded[:T_actual] = teacher_lp[:T_actual]

                sample_starts = np.asarray([start for start, _ in sample_ranges], dtype=np.int32)
                sample_ends = np.asarray([end for _, end in sample_ranges], dtype=np.int32)

                if turn_acc is not None:
                    self._diagnostic_micro_step(
                        turn_acc=turn_acc,
                        turn_ids=turn_ids_padded,
                        input_step=input_step,
                        targets_step=targets_step,
                        pos_step=pos_step,
                        temp_step=temp_step,
                        inference_padded=inference_padded,
                        advantages_padded=advantages_padded,
                        loss_mask_padded=loss_mask_padded,
                        teacher_padded=teacher_padded,
                        sample_ranges=sample_ranges,
                        loss_scale=float(loss_scale),
                        seq_len=seq_len,
                    )
                    continue

                self.trainer.step_grpo_native(
                    input_step,
                    targets_step,
                    inference_padded,
                    advantages_padded,
                    loss_mask_padded,
                    sample_starts,
                    sample_ends,
                    position_ids=pos_step,
                    temperatures=temp_step,
                    teacher_logprobs=teacher_padded,
                    loss_scale=float(loss_scale),
                    ipo_mask_low=float(config.loss.ipo_mask_low),
                    ipo_mask_high=float(config.loss.ipo_mask_high),
                    adv_tau=float(config.loss.adv_tau),
                    teacher_tau=float(config.loss.teacher_tau),
                    kl_tau=float(config.loss.kl_tau),
                    ratio_clip=float(config.loss.ratio_clip),
                )

            # 5. Optimizer step — one per orchestrator step (lr/opt_config built above)
            # Read VTC before optimizer step for diagnostics
            vtc = self.trainer.get_valid_token_count(0)
            expected_loss_scale = loss_scale

            result = self.trainer.update_with_config(opt_config, step + 1)
            # The diagnostic path computes the loss in Python, so the native
            # accumulator is empty — use the Python metrics from the last
            # micro-batch instead.
            if turn_acc is not None:
                step_metrics = dict(self._diag_metrics or {})
            else:
                step_metrics = dict(self.trainer.get_grpo_native_metrics())

            step_time = time.time() - step_start

            self._log_step(
                step=step, orch_step=orch_step, step_metrics=step_metrics, result=result, lr=lr,
                n_mb=n_mb, vtc=vtc, expected_loss_scale=expected_loss_scale,
                step_time=step_time, turn_acc=turn_acc,
            )

            # 7. Checkpointing
            if config.save_steps > 0 and step > 0 and step % config.save_steps == 0 and config.checkpoint_dir:
                logger.info(f"Saving checkpoint at step {step}...")
                self.trainer.save_checkpoint(config.checkpoint_dir, step)

            step += 1

        # Final weight broadcast
        self.broadcast.broadcast(self.trainer, mrm.progress[0].step)

        # Save final adapter/model
        output_path = Path(config.output_dir)
        if config.lora:
            adapter_dir = output_path / "final_adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.export_adapter(str(adapter_dir))
            ensure_vllm_lora_compat(adapter_dir, config.model_dir)
            logger.info(f"Final LoRA adapter saved to {adapter_dir}")

            if config.merge_adapter:
                from surogate.utils.adapter_merge import merge_adapter

                merged_dir = output_path / "final_merged"
                merged_dir.mkdir(parents=True, exist_ok=True)
                try:
                    merge_adapter(
                        base_model_path=config.model_dir,
                        adapter_path=str(adapter_dir),
                        output_path=str(merged_dir),
                        max_shard_size="5GB",
                        cpu_offload=True,
                    )
                    self._copy_tokenizer_files(config.model_dir, str(merged_dir))
                    logger.info(f"Merged model saved to {merged_dir}")
                except Exception as e:
                    logger.error(f"Failed to merge adapter: {e}")
                    import traceback

                    logger.error(f"Traceback:\n{traceback.format_exc()}")
        else:
            model_dir = output_path / "final_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.export_model(str(model_dir))
            self._copy_tokenizer_files(config.model_dir, str(model_dir))
            logger.info(f"Final model saved to {model_dir}")

        logger.info(f"GRPO training complete after {step} steps")

    def close(self):
        if self.metrics_writer is not None:
            self.metrics_writer.close()
            self.metrics_writer = None


def grpo_train(config: GRPOTrainConfig):
    """Entry point for GRPO training."""
    trainer = GRPOTrainer(config)
    try:
        trainer.train()
    finally:
        trainer.close()
