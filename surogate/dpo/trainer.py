"""Offline DPO trainer.

Static (prompt, chosen, rejected) pairs are tokenized one sequence per row
(surogate/dpo/data.py). Each optimizer step runs `grad_accum` micro-steps; each
micro-step feeds `ngpu` host rows (distinct pairs per GPU = data-parallel
sharding), every host row holding B sequences = B/2 atomic pairs. step_dpo_native
computes the sigmoid-DPO gradient natively and backpropagates; the optimizer then
updates the LoRA adapter.

Reference log-probs (frozen start checkpoint, LoRA disabled) are computed INLINE
per micro-step on the EXACT same rows as the policy forward, not precomputed
offline. This is required for fp8: fp8-hybrid derives the activation scale from
the current batch's amax, so a sequence's log-probs depend on its batch-mates. An
offline pass (fixed sequential chunks) and the training loop (reshuffled batches)
would scale the same pair differently → a spurious nonzero margin at init (where
π_θ == π_ref). Computing the reference in the same micro-batch makes both forwards
share the identical scale, so at init the margin is exactly 0 (loss = log 2) under
fp8 and bf16 alike. Cost: one extra (no-backward) forward per micro-step.

No rollouts, no inference engine — purely offline.

For local-contrast training, ``reference_free`` skips the frozen-policy
forward and optimizes the chosen-vs-rejected likelihood gap directly.
``target_margin`` applies a SimPO-style minimum margin.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from surogate import _surogate
from surogate.core.config.dataset_config import PreferenceDatasetConfig
from surogate.core.datasets.datasets import disable_datasets_caching
from surogate.core.datasets.loader import load_dataset_with_config, pre_process
from surogate.dpo.config import DPOTrainConfig
from surogate.dpo.data import tokenize_preference_pairs
from surogate.train.lr_schedule import LRSchedule
from surogate.utils.hf import get_model_weights_path
from surogate.utils.logger import get_logger
from surogate.utils.lora_compat import ensure_surogate_lora_compat, ensure_vllm_lora_compat
from surogate.utils.tensor import to_surogate_dtype

logger = get_logger()


def _normalize_pref_row(row: dict) -> dict:
    """Strip the Arrow round-trip artifacts (all-columns rows with None fills)
    back to the minimal {prompt, chosen, rejected[, enable_thinking]} dict the
    tokenizer and the ref-sidecar digest expect."""
    out = {"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]}
    if row.get("enable_thinking") is not None:
        out["enable_thinking"] = bool(row["enable_thinking"])
    return out


def _load_pref_datasets(dataset_configs: list[PreferenceDatasetConfig], num_workers: int = 1) -> list[dict]:
    """Load configured `type: preference` datasets in declaration order through the
    shared dataset framework (local json/jsonl/parquet/csv files, dataset
    directories, or HF hub repos; honors subset/split/samples and the
    prompt/chosen/rejected field mappings)."""
    if not dataset_configs:
        raise ValueError("DPO requires at least one preference dataset")
    rows: list[dict] = []
    with disable_datasets_caching():
        for ds_cfg in dataset_configs:
            dataset = load_dataset_with_config(ds_cfg, num_workers=num_workers)
            ds_cfg.validate_columns(dataset.column_names)
            dataset = pre_process(dataset, ds_cfg, num_proc=num_workers, load_from_cache_file=False)
            if len(dataset) == 0:
                raise ValueError(f"no {{prompt, chosen, rejected}} rows in {ds_cfg.path}")
            rows.extend(_normalize_pref_row(row) for row in dataset.to_list())
    return rows


def _gpu_batch(pair_idx: list[int], batch, T: int):
    """Build one GPU's micro-batch from a list of pair indices (each pair = 2 rows).

    Returns flat per-token arrays of length len(pair_idx)*2*T and the sample/pair
    index arrays (token offsets into this GPU's [0, B*T) buffer).
    """
    seq_rows = []
    for p in pair_idx:
        seq_rows.append(2 * p)  # chosen
        seq_rows.append(2 * p + 1)  # rejected
    B = len(seq_rows)
    input_ids = batch.input_ids[seq_rows]  # [B, T]
    targets = batch.targets[seq_rows]
    position_ids = batch.position_ids[seq_rows]
    loss_mask = batch.loss_mask[seq_rows].reshape(-1)  # [B*T]
    sample_starts = np.array([i * T for i in range(B)], dtype=np.int32)
    sample_ends = np.array([i * T + int(batch.seq_len[seq_rows[i]]) for i in range(B)], dtype=np.int32)
    pair_chosen = np.array([2 * k for k in range(len(pair_idx))], dtype=np.int32)
    pair_rejected = np.array([2 * k + 1 for k in range(len(pair_idx))], dtype=np.int32)
    return {
        "input_ids": input_ids.astype(np.int32),
        "targets": targets.astype(np.int32),
        "position_ids": position_ids.astype(np.int32),
        "loss_mask": loss_mask.astype(np.uint8),
        "sample_starts": sample_starts,
        "sample_ends": sample_ends,
        "pair_chosen": pair_chosen,
        "pair_rejected": pair_rejected,
    }


def _reference_free_logprobs(
    gpu_batches: list[dict],
    B: int,
    T: int,
    beta: float,
    target_margin: float,
    length_norm: bool,
) -> np.ndarray:
    """Build synthetic references for direct likelihood-gap optimization.

    The native kernel subtracts the chosen reference sum (or mean) from the
    beta-scaled policy gap. Distribute the requested offset across chosen
    tokens in sum mode so the effective target does not grow with edit length.
    """
    refs = np.zeros((len(gpu_batches) * B, T), dtype=np.float32)
    if target_margin == 0:
        return refs
    for gpu_index, gpu_batch in enumerate(gpu_batches):
        mask = gpu_batch["loss_mask"].reshape(B, T)
        for sample_index in gpu_batch["pair_chosen"]:
            sample_index = int(sample_index)
            token_positions = np.flatnonzero(mask[sample_index])
            # The native loss reads ref_logprobs[t - 1] for loss_mask[t].
            token_positions = token_positions[token_positions > 0] - 1
            if not len(token_positions):
                continue
            offset = float(target_margin) / float(beta)
            if not length_norm:
                offset /= len(token_positions)
            row = gpu_index * B + sample_index
            refs[row, token_positions] = offset
    return refs


def dpo_main(config: DPOTrainConfig, args=None) -> None:
    ngpu = max(1, int(config.gpus or 1))
    B = int(config.per_device_train_batch_size)
    T = int(config.sequence_len)
    if B % 2 != 0:
        raise ValueError(f"per_device_train_batch_size must be even (pairs are 2 rows); got {B}")
    pairs_per_gpu = B // 2
    checkpoint_dir = config.checkpoint_dir or str(Path(config.output_dir))

    # --- compile the DSL IR + JIT kernels (mirrors SurogateTrainerWrapper) ---
    from surogate.dsl.ir_builder import build_dsl_ir_for_model

    dsl_extra = {}
    if getattr(config, "ep_size", 1) > 1:
        dsl_extra["ep_size"] = config.ep_size
    ir_json = build_dsl_ir_for_model(config.model_dir, extra_config=dsl_extra or None)
    config.runtime_config.dsl_ir_json = ir_json

    from surogate.kernels.jit_compile import compile_jit_kernels

    jit_manifests = compile_jit_kernels(ir_json)
    if jit_manifests:
        config.runtime_config.jit_kernel_manifests = jit_manifests

    # --- build trainer + import start weights -------------------------------
    logger.info(f"Creating DPO trainer for {config.model} ({ngpu} GPU(s), B={B}, T={T})")
    trainer = _surogate.SurogateTrainer(
        ngpu=ngpu,
        config=_surogate.PretrainedConfig.from_pretrained(config.model_dir, to_surogate_dtype(config.torch_dtype)),
        options=config.runtime_config,
        batch_size=B,
        seq_len=T,
        grad_accum=1,  # set dynamically per optimizer step
        memcpy_all_gather=config.memcpy_all_gather,
        memcpy_send_recv=config.memcpy_send_recv,
        lora_config=config.lora_config,
        qlora_config=config.qlora_config,
    )
    model_weights_path = get_model_weights_path(config.model_dir)
    logger.info(f"Importing start weights from {model_weights_path}")
    if config.adapter_path:
        # Native import expects Surogate key paths, while an adapter previously
        # prepared for vLLM may use the wrapper's language_model prefix. Work on
        # a temporary copy so the user's source adapter is never mutated.
        logger.info(f"Merging inherited adapter from {config.adapter_path} into start weights")
        with tempfile.TemporaryDirectory(prefix="surogate-dpo-adapter-") as temporary_dir:
            adapter_copy = Path(temporary_dir) / "adapter"
            shutil.copytree(config.adapter_path, adapter_copy)
            ensure_surogate_lora_compat(adapter_copy, config.model_dir)
            trainer.set_adapter_path(str(adapter_copy))
            trainer.import_weights(model_weights_path)
    else:
        trainer.import_weights(model_weights_path)

    start_step = 0
    if config.resume_from_checkpoint:
        latest_step = _surogate.find_latest_checkpoint(checkpoint_dir)
        if latest_step >= 0:
            checkpoint_path = Path(checkpoint_dir) / f"step_{latest_step:08d}"
            if config.lora:
                ensure_surogate_lora_compat(checkpoint_path, config.model_dir)
            logger.info(f"Resuming DPO from checkpoint step {latest_step}")
            trainer.load_checkpoint(checkpoint_dir, latest_step)
            start_step = latest_step
        else:
            logger.info("No DPO checkpoint found; starting from step 0")

    # --- tokenize preference pairs -----------------------------------------
    rows = _load_pref_datasets(config.datasets or [], num_workers=config.dataloader_num_workers or 1)
    batch = tokenize_preference_pairs(rows, config.tokenizer, max_len=T, span_mask=config.loss.span_mask)
    logger.info(f"Tokenized {batch.n_pairs} preference pairs (from {len(rows)} rows)")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    # Reference log-probs are computed INLINE per micro-step (see module docstring):
    # the frozen-ref forward must share the policy step's exact batch so fp8's
    # current-batch activation scaling produces a matching scale (margin == 0 at init).

    # --- schedule ----------------------------------------------------------
    pairs_per_step = pairs_per_gpu * ngpu * int(config.gradient_accumulation_steps)
    if config.max_steps and config.max_steps > 0:
        max_steps = int(config.max_steps)
    else:
        epochs = max(1, int(config.num_epochs or 1))
        max_steps = max(1, (batch.n_pairs * epochs) // pairs_per_step)
    warmup = config.warmup_steps or (int(max_steps * config.warmup_ratio) if config.warmup_ratio else 0)
    lr_schedule = LRSchedule(
        base_lr=config.learning_rate,
        max_steps=max_steps,
        warmup_steps=warmup,
        cooldown_steps=config.cooldown_steps,
        final_lr=config.learning_rate * config.final_lr_fraction,
        schedule_type=config.lr_scheduler_type,
        wsd_decay_steps_fraction=config.wsd_decay_steps_fraction,
    )
    # --- infinite shuffled pair stream -------------------------------------
    rng = np.random.RandomState(config.train_seed if config.train_seed is not None else 0)

    def pair_stream():
        while True:
            order = rng.permutation(batch.n_pairs)
            for p in order:
                yield int(p)

    stream = pair_stream()

    # Checkpoints identify the next optimizer step. Replaying the deterministic
    # pair stream restores the same data order as an uninterrupted run.
    for _ in range(start_step * pairs_per_step):
        next(stream)

    def next_pairs(n: int) -> list[int]:
        return [next(stream) for _ in range(n)]

    metrics_path = config.surogate_metrics_path
    logger.info(
        f"DPO training: {max_steps} steps x {pairs_per_step} pairs/step "
        f"(grad_accum={config.gradient_accumulation_steps}, beta={config.loss.dpo_beta}, "
        f"length_norm={config.loss.length_norm}, reference_free={config.loss.reference_free}, "
        f"target_margin={config.loss.target_margin})"
    )

    for step in range(start_step, max_steps):
        t0 = time.time()
        trainer.set_grad_accumulation(int(config.gradient_accumulation_steps))
        # one optimizer step normalizes by the mean over its pairs; gradients are
        # AVG-allreduced across GPUs, so compensate the per-GPU average back to a
        # global sum (/ ngpu), matching the native GRPO convention.
        effective_loss_scale = float(pairs_per_step) / float(ngpu)

        for _ in range(int(config.gradient_accumulation_steps)):
            gpu_batches = [_gpu_batch(next_pairs(pairs_per_gpu), batch, T) for _ in range(ngpu)]
            input_step = np.concatenate([gb["input_ids"] for gb in gpu_batches], axis=0)  # [ngpu*B, T]
            pos_step = np.concatenate([gb["position_ids"] for gb in gpu_batches], axis=0)
            targets_step = np.concatenate([gb["targets"] for gb in gpu_batches], axis=0)

            # Frozen reference (LoRA disabled) on the SAME rows/scale as the policy
            # forward below — fp8-correct (see module docstring). [ngpu*B, T] row order
            # matches input_step, so reshaping to [ngpu, B*T] aligns with the per-GPU layout.
            if config.loss.reference_free:
                ref_2d = _reference_free_logprobs(
                    gpu_batches,
                    B,
                    T,
                    beta=float(config.loss.dpo_beta),
                    target_margin=float(config.loss.target_margin),
                    length_norm=bool(config.loss.length_norm),
                )
            else:
                ref_2d = np.asarray(
                    trainer.compute_ref_logprobs_dpo(input_step, targets_step, position_ids=pos_step),
                    dtype=np.float32,
                )  # [ngpu*B, T]

            if ngpu > 1:
                ref_step = ref_2d.reshape(ngpu, B * T)  # [ngpu, B*T]
                loss_mask_step = np.stack([gb["loss_mask"] for gb in gpu_batches])
                sample_starts = np.stack([gb["sample_starts"] for gb in gpu_batches])  # [ngpu, B]
                sample_ends = np.stack([gb["sample_ends"] for gb in gpu_batches])
                pair_chosen = np.stack([gb["pair_chosen"] for gb in gpu_batches])  # [ngpu, B/2]
                pair_rejected = np.stack([gb["pair_rejected"] for gb in gpu_batches])
            else:
                gb = gpu_batches[0]
                ref_step = ref_2d.reshape(B * T)
                loss_mask_step = gb["loss_mask"]
                sample_starts = gb["sample_starts"]
                sample_ends = gb["sample_ends"]
                pair_chosen = gb["pair_chosen"]
                pair_rejected = gb["pair_rejected"]

            trainer.step_dpo_native(
                input_step,
                targets_step,
                ref_step,
                loss_mask_step,
                sample_starts,
                sample_ends,
                pair_chosen,
                pair_rejected,
                position_ids=pos_step,
                loss_scale=effective_loss_scale,
                beta=float(config.loss.dpo_beta),
                length_norm=1 if config.loss.length_norm else 0,
            )

        lr = lr_schedule.get_lr(step)
        opt_config = _surogate.OptimizerConfig(
            optimizer=config.optimizer,
            learning_rate=lr,
            weight_decay=config.weight_decay,
            grad_clip=config.max_grad_norm,
            adamw_beta1=config.adamw_beta1,
            adamw_beta2=config.adamw_beta2,
            adamw_epsilon=config.adamw_epsilon,
        )
        result = trainer.update_with_config(opt_config, step + 1)
        m = dict(trainer.get_dpo_native_metrics())
        dt = time.time() - t0

        if step % max(1, config.logging_steps) == 0:
            logger.info(
                f"step={step} dpo_loss={m.get('dpo_loss', 0):.4f} acc={m.get('dpo_accuracy', 0):.3f} "
                f"margin={m.get('dpo_margin', 0):.4f} grad_norm={result['norm']:.4f} lr={lr:.2e} "
                f"pairs={int(m.get('dpo_pairs', 0))} {dt * 1000:.0f}ms"
            )
            if metrics_path:
                try:  # metrics logging must never kill training
                    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(metrics_path, "a", encoding="utf-8") as mf:
                        mf.write(json.dumps({"step": step, "lr": lr, "grad_norm": result["norm"], **m}) + "\n")
                except OSError as exc:
                    logger.warning("could not write DPO metrics to %s: %s", metrics_path, exc)

        completed_step = step + 1
        if config.save_steps > 0 and completed_step % config.save_steps == 0:
            logger.info(f"Saving checkpoint at step {completed_step}...")
            trainer.save_checkpoint(checkpoint_dir, completed_step)
            if config.save_total_limit and config.save_total_limit > 0:
                removed = _surogate.clean_old_checkpoints(checkpoint_dir, config.save_total_limit, -1)
                if removed:
                    logger.info(f"Removed {removed} old checkpoints; keeping the newest {config.save_total_limit}")

    # --- final adapter ------------------------------------------------------
    out = Path(config.output_dir)
    if config.lora:
        adapter_dir = out / "final_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.export_adapter(str(adapter_dir))
        ensure_vllm_lora_compat(adapter_dir, config.model_dir)
        logger.info(f"Final LoRA adapter saved to {adapter_dir}")
    logger.info("DPO training complete.")
