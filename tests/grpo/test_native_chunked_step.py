"""Chunked native GRPO must start every optimizer step the way the unchunked step does (#270).

Micro-step 0 carries the step's start-of-step work: it zeroes the LoRA gradients, initialises the
FP8 delayed-scaling state and resets the MoE stats. `step_grpo_native_chunked` numbers its windows
from the end and skips the all-padding tail chunks, so a batch whose samples all end in chunk 0
never runs a micro-step 0. Its gradients were then accumulated onto the previous step's: the
second identical step reported about twice the first one's norm.
"""

import json
import os

import numpy as np
import pytest

CHUNK = 16
SEQ_CHUNKS = 2


@pytest.mark.gpu
@pytest.mark.slow
def test_chunked_native_grpo_steps_match_the_unchunked_step():
    _surogate = pytest.importorskip("surogate._surogate", reason="needs the built extension")

    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.utils.hf import get_model_weights_path
    from tests.test_onboarding_qwen3 import prepare_mini_model, resolve_model_path

    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3 weights not found; set QWEN3_MODEL_PATH or cache Qwen/Qwen3-0.6B")
    model_dir = prepare_mini_model(snapshot)

    full_len = CHUNK * SEQ_CHUNKS
    # The chunk RoPE tables are sized from the chunk; positions reach the full length.
    os.environ["SUROGATE_ROPE_MAX_SEQ"] = str(full_len)

    vocab_size = int(json.loads((model_dir / "config.json").read_text())["vocab_size"])
    rng = np.random.default_rng(0)
    inputs = rng.integers(0, vocab_size, size=(1, full_len), dtype=np.int32)
    targets = np.roll(inputs, -1, axis=1).astype(np.int32)
    # One sample, ending inside chunk 0: chunk 1 is all padding and is skipped.
    sample_end = 12
    position_ids = np.concatenate(
        [np.arange(sample_end, dtype=np.int32), np.arange(full_len - sample_end, dtype=np.int32)]
    ).reshape(1, full_len)
    loss_mask = np.zeros(full_len, dtype=np.uint8)
    loss_mask[4:sample_end] = 1
    advantages = (rng.normal(0.0, 1.0, size=full_len) * loss_mask).astype(np.float32)
    inference_logprobs = (rng.uniform(-12.0, -10.0, size=full_len) * loss_mask).astype(np.float32)

    config = _surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
    lora = _surogate.LoRAAdapterConfig(
        rank=8, alpha=16, dropout=0.0, dtype="bf16",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_rslora=False, train_router=False,
    )
    # lr = 0: the update flushes the step without moving the weights, so every step sees the same model.
    flush = _surogate.OptimizerConfig(optimizer="adamw_8bit", learning_rate=0.0, weight_decay=0.0, grad_clip=0.0)

    def grad_norms(seq_chunks, steps):
        options = _surogate.RuntimeOptions(
            offload_residual=False,
            use_cuda_graphs=False,
            offload_master=False,
            offload_grads=False,
            offload_optimizer=False,
            shard_gradients=True,
            use_zero_copy=False,
        )
        options.sequence_chunks = seq_chunks
        options.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))
        trainer = _surogate.SurogateTrainer(
            ngpu=1,
            config=config,
            options=options,
            batch_size=1,
            seq_len=full_len // seq_chunks,
            grad_accum=1,
            memcpy_all_gather=True,
            memcpy_send_recv=True,
            lora_config=lora,
            qlora_config=None,
        )
        trainer.import_weights(get_model_weights_path(str(model_dir)))
        norms = []
        for step in range(steps):
            trainer.step_grpo_native(
                inputs,
                targets,
                inference_logprobs,
                advantages,
                loss_mask,
                np.array([0], dtype=np.int32),
                np.array([sample_end], dtype=np.int32),
                position_ids=position_ids,
                loss_scale=float(loss_mask.sum()),
                kl_tau=0.1,
            )
            norms.append(float(trainer.update_with_config(flush, step + 1)["norm"]))
        del trainer
        return norms

    (reference,) = grad_norms(1, 1)
    chunked = grad_norms(SEQ_CHUNKS, 3)

    assert reference > 0.0
    for step, norm in enumerate(chunked):
        assert norm == pytest.approx(reference, rel=2e-2), (step, chunked, reference)
