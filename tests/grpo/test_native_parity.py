"""Parity coverage for the native GRPO loss (E4 in the RL end-to-end findings).

`test_native_formula.py` checks the Python loss against itself (see its module
docstring); nothing has ever checked it against the kernel. Two different things
were being conflated, and they are separated here.

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

# `ipo_mask_low/high = 1.0` masks nothing (probs_diff is always within +/-1), so
# the advantage term stays live; it contributes zero only because `_call` passes
# zero advantages. That leaves the KL term alone, with the closed form
# -2 * kl_tau * (trainer_logprob - inference_logprob) on unmasked tokens.
# Derived independently by `test_sparse_outcome_advantage_preserves_full_completion_kl`
# in test_native_formula.py, which writes out the same expression by hand.
KL_ONLY = GRPOLossConfig(ipo_mask_low=1.0, ipo_mask_high=1.0, adv_tau=1.0, teacher_tau=0.0, kl_tau=0.1)

# The GPU test uses production thresholds instead. With the 1.0 bounds above,
# |probs_diff| < 1 always holds, so nothing is ever masked and `is_masked`,
# `is_masked_low`, `is_masked_high` and `masked_mismatch_kl` compare 0 to 0 --
# four of the nine shared metrics proving nothing. 0.2 is the shipped default
# (`grpo/config.py`). That alone only fixes three of the four: the high side
# additionally needs probs_diff to go positive, which is why the GPU fixture
# derives its inference logprobs from the trainer's rather than from a fixed
# range that is always larger.
PRODUCTION_LIKE = GRPOLossConfig(ipo_mask_low=0.2, ipo_mask_high=0.2, adv_tau=1.0, teacher_tau=0.0, kl_tau=0.1)

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


def test_a_trailing_one_token_range_leaves_the_sample_before_it_alone():
    """A lone trailing pad token gets its own range now (see
    `_find_sample_boundaries`). The sample before it must shift exactly as it
    would have without that range -- the pad must neither absorb a gradient nor
    displace one."""
    grads = _expected_unshifted_grads()
    split = compute_native_shifted_grpo_dloss_reference(
        trainer_logprobs=TRAINER_LP,
        inference_logprobs=INFERENCE_LP,
        advantages=np.zeros(7, dtype=np.float32),
        loss_mask=LOSS_MASK,
        loss_config=KL_ONLY,
        sample_ranges=[(0, 4), (4, 6), (6, 7)],
        teacher_logprobs=None,
        loss_scale=1.0,
    )
    assert split[4] == pytest.approx(grads[5]), "the shortened sample still shifts"
    assert split[5] == 0.0, "its last slot receives nothing"
    assert split[6] == 0.0, "and the 1-token range contributes nothing"
    np.testing.assert_allclose(split[0:3], grads[1:4], rtol=1e-6, atol=1e-7)


# ── the actual CUDA parity, which needs hardware ─────────────────────


@pytest.mark.gpu
@pytest.mark.slow
def test_the_cuda_kernel_matches_the_python_reference_metrics():
    """The gap E4 names: nothing has ever run the kernel against the reference.

    Both paths see the same weights and the same batch, so the kernel's own
    metrics -- read back through `get_grpo_native_metrics` -- must match what the
    Python reference computes from the same logprobs.

    Metrics rather than gradients on purpose: they are the kernel's arithmetic
    made observable without reading device memory, and they cover all the loss
    terms (policy loss, mismatch KL, keep/clip counts and their denominators).

    First run 2026-09-04 against the published cu128 image: all 9 shared metrics
    agreed, policy_loss 17.431433 vs 17.431432. So the hand-verified claim that
    the two implementations match is now actually checked.
    """
    _surogate = pytest.importorskip("surogate._surogate", reason="needs the built extension")

    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.grpo.loss import compute_native_grpo_metrics_reference, unshift_to_logical
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
    # Two packed samples, not one: a single unpacked range never exercises the
    # per-sample shift, and this branch rewrote how those ranges are derived.
    # The positions must RESET per sample, as the packer emits them -- a
    # continuous arange alongside two ranges is contradictory input, and the
    # kernel faults on it with a misaligned address rather than rejecting it.
    sample_ranges = [(0, 8), (8, seq_len)]
    position_ids = np.concatenate([np.arange(e - s, dtype=np.int32) for s, e in sample_ranges]).reshape(1, seq_len)

    # Prompt masked, completion live -- PER SAMPLE. The first token of every
    # packed sample must be a masked prompt token: the kernel emits gradients
    # only for logical [start+1, end-1], so a sample whose first token is live
    # makes the kernel and the reference disagree by exactly that token. That is
    # an unvalidated invariant of the packed layout, not a divergence, and a
    # flat `loss_mask[4:] = True` across two samples silently breaks it.
    loss_mask = np.zeros(seq_len, dtype=bool)
    for start, end in sample_ranges:
        loss_mask[start + 4 : end] = True
    advantages = rng.normal(0.0, 1.0, size=seq_len).astype(np.float32) * loss_mask
    loss_scale = float(loss_mask.sum())
    # inference_logprobs is derived from the trainer's below, deliberately.

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
    # The production un-shift, not a copy of it: this test exists to check the
    # decomposed path against the kernel, and the kernel does its own shift in
    # C++, so sharing this cannot manufacture agreement between the two sides.
    trainer_logprobs = unshift_to_logical(buf, sample_ranges)
    # Nothing reads the reference trainer after this, and holding two full
    # trainers doubles peak VRAM on a box that is usually busy.
    del reference_trainer

    # Draw the inference logprobs NEAR the trainer's, not from a fixed range.
    # A truncated model over random token ids produces logprobs around
    # -log(vocab) ~ -12, so an independent draw from [-4, -0.1] leaves
    # importance_ratio = exp(trainer - inference) ~ 5e-5. The kernel's advantage
    # branch then contributes ~5e-6 of policy_loss, far under the 1e-3 tolerance:
    # the entire policy-gradient half could be deleted from the kernel and every
    # assertion here would still pass. Keeping the ratio near 1 puts the policy
    # term on the same order as the KL term, and makes probs_diff straddle zero
    # so both sides of the IPO mask are exercised rather than only the low side.
    inference_logprobs = (trainer_logprobs + rng.normal(0.0, 0.3, size=seq_len)).astype(np.float32)

    expected = compute_native_grpo_metrics_reference(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_config=PRODUCTION_LIKE,
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
        ipo_mask_low=float(PRODUCTION_LIKE.ipo_mask_low),
        ipo_mask_high=float(PRODUCTION_LIKE.ipo_mask_high),
        adv_tau=float(PRODUCTION_LIKE.adv_tau),
        teacher_tau=float(PRODUCTION_LIKE.teacher_tau),
        kl_tau=float(PRODUCTION_LIKE.kl_tau),
        ratio_clip=float(PRODUCTION_LIKE.ratio_clip),
    )
    actual = native_trainer.get_grpo_native_metrics()

    # Measured on a real run (Qwen3-0.6B, one batch, 2026-09-04): the kernel
    # reports 10 metrics, the reference 18, and 9 are shared. Each side has
    # terms the other lacks -- the reference computes OPD, replay,
    # ratio_clipped and policy_sample_count; the kernel reports teacher_kl --
    # so compare the intersection rather than demanding either be complete.
    assert actual, "the kernel must report metrics to compare against"
    core = {"policy_loss", "mismatch_kl", "keep_tokens", "total_tokens"}
    assert core <= set(actual), f"kernel metrics missing the core terms: {core - set(actual)}"

    compared = 0
    for key, actual_value in actual.items():
        if key not in expected:
            continue
        # bf16 forward, so the logprobs feeding both sides carry real error;
        # the tolerance is on the arithmetic agreeing, not on bit equality.
        assert actual_value == pytest.approx(expected[key], rel=1e-3, abs=1e-4), key
        compared += 1
    assert compared >= 9, f"only {compared} metrics were compared; 9 are shared"
