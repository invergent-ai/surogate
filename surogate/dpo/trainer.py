"""Offline DPO trainer.

Static (prompt, chosen, rejected) pairs are tokenized one sequence per row
(surogate/dpo/data.py); a reference forward of the frozen start checkpoint
(LoRA disabled) is precomputed once and cached. Each optimizer step runs
`grad_accum` micro-steps; each micro-step feeds `ngpu` host rows (distinct pairs
per GPU = data-parallel sharding), every host row holding B sequences = B/2
atomic pairs. step_dpo_native computes the sigmoid-DPO gradient natively and
backpropagates; the optimizer then updates the LoRA adapter.

No rollouts, no inference engine — purely offline.
"""

import json
import time
from pathlib import Path

import numpy as np

from surogate import _surogate
from surogate.dpo.config import DPOTrainConfig
from surogate.dpo.data import (
    load_ref_sidecar,
    precompute_ref_logprobs,
    rows_digest,
    save_ref_sidecar,
    sidecar_hash,
    tokenize_preference_pairs,
)
from surogate.train.lr_schedule import LRSchedule
from surogate.utils.hf import get_model_weights_path
from surogate.utils.logger import get_logger
from surogate.utils.tensor import to_surogate_dtype

logger = get_logger()


def _load_pref_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "prompt" in r and "chosen" in r and "rejected" in r:
                rows.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]})
    if not rows:
        raise ValueError(f"no {{prompt, chosen, rejected}} rows in {path}")
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
    ref = batch.ref[seq_rows].reshape(-1)  # [B*T]
    sample_starts = np.array([i * T for i in range(B)], dtype=np.int32)
    sample_ends = np.array([i * T + int(batch.seq_len[seq_rows[i]]) for i in range(B)], dtype=np.int32)
    pair_chosen = np.array([2 * k for k in range(len(pair_idx))], dtype=np.int32)
    pair_rejected = np.array([2 * k + 1 for k in range(len(pair_idx))], dtype=np.int32)
    return {
        "input_ids": input_ids.astype(np.int32),
        "targets": targets.astype(np.int32),
        "position_ids": position_ids.astype(np.int32),
        "loss_mask": loss_mask.astype(np.uint8),
        "ref": ref.astype(np.float32),
        "sample_starts": sample_starts,
        "sample_ends": sample_ends,
        "pair_chosen": pair_chosen,
        "pair_rejected": pair_rejected,
    }


def dpo_main(config: DPOTrainConfig, args=None) -> None:
    ngpu = max(1, int(config.gpus or 1))
    B = int(config.per_device_train_batch_size)
    T = int(config.sequence_len)
    if B % 2 != 0:
        raise ValueError(f"per_device_train_batch_size must be even (pairs are 2 rows); got {B}")
    pairs_per_gpu = B // 2

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
    trainer.import_weights(model_weights_path)

    # --- tokenize preference pairs -----------------------------------------
    rows = _load_pref_rows(config.datasets[0].path)
    batch = tokenize_preference_pairs(rows, config.tokenizer, max_len=T)
    logger.info(f"Tokenized {batch.n_pairs} preference pairs (from {len(rows)} rows)")

    # --- reference logprobs (frozen start model), cached in a sidecar -------
    sidecar_path = str(Path(config.output_dir) / "ref_logprobs.npz")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    key = sidecar_hash(model=config.model_dir, max_len=T, n_rows=len(rows), rows_digest=rows_digest(rows))
    ref = load_ref_sidecar(sidecar_path, key)
    if ref is None or ref.shape[0] != batch.n_seq:
        logger.info("Precomputing reference logprobs (LoRA disabled)...")
        ref = precompute_ref_logprobs(trainer, batch, engine_b=B)
        save_ref_sidecar(sidecar_path, key, ref)
        logger.info(f"Reference logprobs cached -> {sidecar_path}")
    else:
        logger.info(f"Loaded cached reference logprobs <- {sidecar_path}")
    batch.ref = ref  # attach for _gpu_batch

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
    checkpoint_dir = config.checkpoint_dir or str(Path(config.output_dir))

    # --- infinite shuffled pair stream -------------------------------------
    rng = np.random.RandomState(config.train_seed if config.train_seed is not None else 0)

    def pair_stream():
        while True:
            order = rng.permutation(batch.n_pairs)
            for p in order:
                yield int(p)

    stream = pair_stream()

    def next_pairs(n: int) -> list[int]:
        return [next(stream) for _ in range(n)]

    metrics_path = config.surogate_metrics_path
    logger.info(
        f"DPO training: {max_steps} steps x {pairs_per_step} pairs/step "
        f"(grad_accum={config.gradient_accumulation_steps}, beta={config.loss.dpo_beta}, "
        f"length_norm={config.loss.length_norm})"
    )

    for step in range(max_steps):
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

            if ngpu > 1:
                ref_step = np.stack([gb["ref"] for gb in gpu_batches])  # [ngpu, B*T]
                loss_mask_step = np.stack([gb["loss_mask"] for gb in gpu_batches])
                sample_starts = np.stack([gb["sample_starts"] for gb in gpu_batches])  # [ngpu, B]
                sample_ends = np.stack([gb["sample_ends"] for gb in gpu_batches])
                pair_chosen = np.stack([gb["pair_chosen"] for gb in gpu_batches])  # [ngpu, B/2]
                pair_rejected = np.stack([gb["pair_rejected"] for gb in gpu_batches])
            else:
                gb = gpu_batches[0]
                ref_step = gb["ref"]
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

        if config.save_steps > 0 and step > 0 and step % config.save_steps == 0:
            logger.info(f"Saving checkpoint at step {step}...")
            trainer.save_checkpoint(checkpoint_dir, step)

    # --- final adapter ------------------------------------------------------
    out = Path(config.output_dir)
    if config.lora:
        adapter_dir = out / "final_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.export_adapter(str(adapter_dir))
        logger.info(f"Final LoRA adapter saved to {adapter_dir}")
    logger.info("DPO training complete.")
