"""Parity coverage for the native GRPO loss (E4 in the RL end-to-end findings).

The existing `test_native_formula.py` cannot establish parity with anything. Its
"expected" helper calls `compute_grpo_per_token_grads`, and so does the function
it checks (`loss.py:350`), so the assertion is `f(x) == f(x)`: it holds whatever
the formula is, and would keep holding if the formula were wrong.

Two different things were being conflated, and they are separated here.

1. **The Python wrapper.** What `compute_native_shifted_grpo_dloss_reference`
   adds over the base function is a per-sample next-token shift and a division
   by `loss_scale`. Both are checked below against values derived from the loss
   definition rather than from the code under test.

2. **The CUDA kernel.** `grpo_custom_dloss_kernel` has only ever been compared
   to the Python reference *by hand*. Nothing in the suite imports `_surogate`
   or calls `step_grpo_native`, so any future edit to either side diverges with
   a green suite. That test needs a GPU and lives at the bottom of this file.
"""

import json

import numpy as np
import pytest

from surogate.grpo.config import GRPOLossConfig
from surogate.grpo.loss import compute_native_shifted_grpo_dloss_reference

# A config where only the KL term is live, so the per-token gradient has a closed
# form: -2 * kl_tau * (trainer_logprob - inference_logprob) on unmasked tokens.
# Established independently by `test_sparse_outcome_advantage_preserves_full_completion_kl`
# in test_native_formula.py, which derives the same expression by hand.
KL_ONLY = GRPOLossConfig(ipo_mask_low=1.0, ipo_mask_high=1.0, adv_tau=1.0, teacher_tau=0.0, kl_tau=0.1)

TRAINER_LP = np.array([-7.0, -1.2, -0.8, -3.1, -2.0, -0.4, -5.0], dtype=np.float32)
INFERENCE_LP = np.array([-7.0, -1.4, -0.7, -2.8, -1.1, -0.5, -4.6], dtype=np.float32)
# Sample 2 starts at index 4 and its first token is unmasked on purpose: a global
# (rather than per-sample) shift would leak that gradient back into index 3.
LOSS_MASK = np.array([False, True, True, True, True, True, True])
SAMPLE_RANGES = [(0, 4), (4, 7)]


def _expected_unshifted_grads() -> np.ndarray:
    """The KL gradient, written from the loss definition, not from the code."""
    grads = -2.0 * KL_ONLY.kl_tau * (TRAINER_LP - INFERENCE_LP)
    return np.where(LOSS_MASK, grads, 0.0).astype(np.float32)


def _call(loss_scale: float) -> np.ndarray:
    return compute_native_shifted_grpo_dloss_reference(
        trainer_logprobs=TRAINER_LP,
        inference_logprobs=INFERENCE_LP,
        advantages=np.zeros(7, dtype=np.float32),
        loss_mask=LOSS_MASK,
        loss_config=KL_ONLY,
        sample_ranges=SAMPLE_RANGES,
        teacher_logprobs=None,
        loss_scale=loss_scale,
    )


def test_the_shift_moves_each_gradient_one_token_left_within_its_sample():
    """The kernel reads `losses[out_idx]` as the logprob of logical token
    `out_idx + 1`, so the reference has to hand back gradients in that layout."""
    grads = _expected_unshifted_grads()
    actual = _call(loss_scale=1.0)

    expected = np.zeros(7, dtype=np.float32)
    expected[0:3] = grads[1:4]  # sample (0, 4)
    expected[4:6] = grads[5:7]  # sample (4, 7)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_the_shift_does_not_cross_a_sample_boundary():
    """The last slot of each sample receives nothing. If the shift were applied
    across the whole packed row instead of per sample, index 3 would pick up
    sample 2's first gradient, which is non-zero precisely so this can fail."""
    grads = _expected_unshifted_grads()
    assert grads[4] != 0.0, "the fixture must make the leak detectable"

    actual = _call(loss_scale=1.0)
    assert actual[3] == 0.0, "sample 1's last slot must not receive sample 2's gradient"
    assert actual[6] == 0.0, "nor may the final slot of the row"


def test_the_result_is_divided_by_loss_scale():
    """`loss = total / loss_scale`, and the reference owes the caller the divided
    form: the engine does not divide again."""
    np.testing.assert_allclose(_call(loss_scale=4.0), _call(loss_scale=1.0) / 4.0, rtol=1e-6)


def test_a_one_token_sample_contributes_nothing():
    """`end - start > 1` guards the shift, so a length-1 range (a lone trailing
    pad token) leaves zeros rather than reading out of its own range."""
    actual = compute_native_shifted_grpo_dloss_reference(
        trainer_logprobs=TRAINER_LP[:4],
        inference_logprobs=INFERENCE_LP[:4],
        advantages=np.zeros(4, dtype=np.float32),
        loss_mask=np.array([False, True, True, True]),
        loss_config=KL_ONLY,
        sample_ranges=[(0, 3), (3, 4)],
        teacher_logprobs=None,
        loss_scale=1.0,
    )
    assert actual[3] == 0.0


# ── the actual CUDA parity, which needs hardware ─────────────────────


@pytest.mark.gpu
@pytest.mark.slow
def test_the_cuda_kernel_matches_the_python_reference_metrics():
    """The gap E4 names: nothing has ever run the kernel against the reference.

    `trainer.py` claims the decomposed path produces gradients identical to the
    fused `step_grpo_native`, and cites `test_native_formula.py` as the
    assertion. That test cannot assert it. This one can: both paths see the same
    weights and the same batch, so the kernel's own metrics -- read back through
    `get_grpo_native_metrics` -- must match what the Python reference computes
    from the same logprobs.

    Metrics rather than gradients on purpose: they are the kernel's arithmetic
    made observable without reading device memory, and they cover all the loss
    terms (policy loss, mismatch KL, keep/clip counts and their denominators).
    """
    _surogate = pytest.importorskip("surogate._surogate", reason="needs the built extension")

    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.grpo.loss import compute_native_grpo_metrics_reference
    from surogate.utils.hf import get_model_weights_path
    from tests.test_onboarding_qwen3 import prepare_mini_model, resolve_model_path

    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3 weights not found; set QWEN3_MODEL_PATH or cache Qwen/Qwen3-0.6B")
    model_dir = prepare_mini_model(snapshot)

    seq_len = 16
    vocab_size = int(json.loads((model_dir / "config.json").read_text())["vocab_size"])
    rng = np.random.default_rng(0)

    inputs = rng.integers(0, vocab_size, size=(1, seq_len), dtype=np.int32)
    targets = np.roll(inputs, -1, axis=1).astype(np.int32)
    position_ids = np.arange(seq_len, dtype=np.int32).reshape(1, seq_len)
    sample_ranges = [(0, seq_len)]

    # Prompt tokens masked out, completion tokens live, matching a real pack.
    loss_mask = np.zeros(seq_len, dtype=bool)
    loss_mask[4:] = True
    advantages = rng.normal(0.0, 1.0, size=seq_len).astype(np.float32) * loss_mask
    inference_logprobs = rng.uniform(-4.0, -0.1, size=seq_len).astype(np.float32)
    loss_scale = float(loss_mask.sum())

    config = _surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
    options = _surogate.RuntimeOptions(
        offload_residual=False,
        use_cuda_graphs=False,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        shard_gradients=True,
        use_zero_copy=False,
    )
    options.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))

    def _new_trainer():
        t = _surogate.SurogateTrainer(
            ngpu=1,
            config=config,
            options=options,
            batch_size=1,
            seq_len=seq_len,
            grad_accum=1,
            memcpy_all_gather=True,
            memcpy_send_recv=True,
            lora_config=None,
            qlora_config=None,
        )
        t.import_weights(get_model_weights_path(str(model_dir)))
        return t

    # A separate trainer per path: `forward_for_grpo` saves activations that
    # `step_grpo_native` would otherwise inherit. Same weights either way, so the
    # logprobs are the same.
    reference_trainer = _new_trainer()
    buf = np.asarray(
        reference_trainer.forward_for_grpo(inputs, targets, position_ids, None)[0, :seq_len],
        dtype=np.float32,
    )
    # forward_for_grpo returns the negated CE buffer in TARGET slot layout:
    # buf[t] = log p(input_ids[t+1]). Un-shift into logical layout, the exact
    # inverse of the shift the kernel applies. Mirrors trainer.py's diagnostic path.
    trainer_logprobs = np.zeros(seq_len, dtype=np.float32)
    for start, end in sample_ranges:
        if end - start > 1:
            trainer_logprobs[start + 1 : end] = buf[start : end - 1]

    expected = compute_native_grpo_metrics_reference(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=KL_ONLY,
        sample_ranges=sample_ranges,
        teacher_logprobs=None,
    )

    native_trainer = _new_trainer()
    native_trainer.step_grpo_native(
        inputs,
        targets,
        inference_logprobs,
        advantages,
        loss_mask.astype(np.uint8),
        np.array([s for s, _ in sample_ranges], dtype=np.int32),
        np.array([e for _, e in sample_ranges], dtype=np.int32),
        position_ids=position_ids,
        temperatures=None,
        teacher_logprobs=None,
        loss_scale=loss_scale,
        ipo_mask_low=float(KL_ONLY.ipo_mask_low),
        ipo_mask_high=float(KL_ONLY.ipo_mask_high),
        adv_tau=float(KL_ONLY.adv_tau),
        teacher_tau=float(KL_ONLY.teacher_tau),
        kl_tau=float(KL_ONLY.kl_tau),
        ratio_clip=float(KL_ONLY.ratio_clip),
    )
    actual = native_trainer.get_grpo_native_metrics()

    assert expected, "the reference must produce metrics to compare against"
    for key, expected_value in expected.items():
        assert key in actual, f"the kernel reports no {key}"
        # bf16 forward, so the logprobs feeding both sides carry real error;
        # the tolerance is on the arithmetic agreeing, not on bit equality.
        assert actual[key] == pytest.approx(expected_value, rel=1e-3, abs=1e-4), key
