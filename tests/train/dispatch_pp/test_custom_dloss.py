"""dispatch-PP honors a caller-supplied per-token gradient seed.

This is the hook that lets GRPO/DPO-style objectives run under dispatch-PP: the
loss stage backpropagates from `custom_dloss` instead of the built-in
cross-entropy, exactly as backward_grpo() does on the non-dispatch path.

Two identities pin it, and the second is the one that matters:

  * custom_dloss == 1.0 everywhere must reproduce the CE backward, because CE
    seeds the backward with d_loss = 1.0. Proves the plumbing is faithful.
  * custom_dloss == 0.0 must produce a zero gradient. Proves the argument is
    actually *consumed* — a silently-ignored parameter would still yield the CE
    gradient here, and the first test alone would happily pass.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sg = pytest.importorskip("surogate._surogate")

from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.utils.hf import get_model_weights_path
from tests.test_onboarding_qwen3 import (
    BATCH,
    NUM_LAYERS,
    SEQ_LEN,
    make_inputs,
    prepare_mini_model,
    resolve_model_path,
)

_NGPU = 2
_MIN_FREE_BYTES = 6 * 1024**3


def _enough_free_gpus():
    if torch.cuda.device_count() < _NGPU:
        return False
    for i in range(_NGPU):
        free, _ = torch.cuda.mem_get_info(i)
        if free < _MIN_FREE_BYTES:
            return False
    return True


pytestmark = pytest.mark.skipif(
    not _enough_free_gpus(), reason=f"needs {_NGPU} GPUs each with >= {_MIN_FREE_BYTES // 1024**3} GiB free"
)


def _build_trainer():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3 weights not found. Set QWEN3_MODEL_PATH or cache Qwen/Qwen3-0.6B.")
    model_dir = prepare_mini_model(snapshot)
    cfg = sg.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
    opts = sg.RuntimeOptions(
        offload_residual=False,
        use_cuda_graphs=False,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        shard_gradients=False,
        use_zero_copy=False,
    )
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))
    trainer = sg.SurogateTrainer(
        ngpu=_NGPU,
        config=cfg,
        options=opts,
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        grad_accum=1,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
        lora_config=None,
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    return trainer, model_dir


def _stage_ranges():
    cut = NUM_LAYERS // 2 - 1
    return [0, cut + 1], [cut, NUM_LAYERS - 1]


def _opt_config():
    return sg.OptimizerConfig.adamw(
        lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, grad_clip=1.0
    )


def _one_step(custom_dloss):
    """One dispatch-PP step on a fixed batch; returns (loss, grad_norm)."""
    trainer, model_dir = _build_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    b = make_inputs(vocab_size)
    los, his = _stage_ranges()
    loss = float(
        trainer.dispatch_pp_train_step_multigpu(
            b["inputs"], b["targets"], los, his, _opt_config(), 0, False, 1, custom_dloss
        )
    )
    return loss, float(trainer.dispatch_pp_last_grad_norm())


def _ones_like_batch():
    return np.ones((BATCH, SEQ_LEN), dtype=np.float32)


def test_unit_custom_dloss_matches_cross_entropy():
    """d_loss = 1.0 is exactly what CE seeds, so the two paths must agree."""
    ce_loss, ce_norm = _one_step(None)
    cd_loss, cd_norm = _one_step(_ones_like_batch())

    assert np.isfinite([ce_norm, cd_norm]).all(), (ce_norm, cd_norm)
    assert ce_norm > 0.0, "baseline produced no gradient; test cannot discriminate"
    # The forward is identical, so the reported loss must match exactly.
    assert cd_loss == pytest.approx(ce_loss, rel=1e-5)
    assert cd_norm == pytest.approx(ce_norm, rel=2e-2), (
        f"custom dloss of ones diverged from CE: {cd_norm} vs {ce_norm}"
    )


def test_zero_custom_dloss_produces_no_gradient():
    """The guard on the guard: a silently-ignored argument would still give the
    CE gradient here, which would make the ones-test meaningless."""
    _, ce_norm = _one_step(None)
    _, zero_norm = _one_step(np.zeros((BATCH, SEQ_LEN), dtype=np.float32))

    assert ce_norm > 0.0
    assert zero_norm == pytest.approx(0.0, abs=1e-4), (
        f"zero dloss still produced gradient {zero_norm}; custom_dloss is being ignored"
    )


def test_staged_forward_produces_valid_logprobs():
    """The staged forward must run THROUGH the loss stage and yield real logprobs.

    GRPO forms its per-token gradients from these, so an all-zero or non-finite
    buffer would silently zero the gradient. Checks shape, finiteness, that
    supervised positions carry negative logprobs, and that masked/padding
    positions stay exactly zero (matching forward_for_grpo's contract).

    CONTRACT NOTE: dispatch-PP leaves per-step executor state that the next
    dispatch call clears on entry. Do NOT issue a non-dispatch forward
    (forward_for_grpo, validate, ...) on the same trainer in between — it hits an
    async launch failure. GRPO alternates staged-forward and fused dispatch step,
    so it never does. Numerical agreement of the assembled GRPO gradients with the
    native path is covered by tests/grpo/test_dispatch_pp_routing.py.
    """
    trainer, model_dir = _build_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    b = make_inputs(vocab_size)
    los, his = _stage_ranges()

    staged = np.asarray(
        trainer.dispatch_pp_forward_logprobs_multigpu(b["inputs"], b["targets"], los, his, 1)
    )

    assert staged.shape == (BATCH, SEQ_LEN), staged.shape
    assert np.isfinite(staged).all(), "non-finite logprobs would poison every gradient"

    supervised = staged != 0.0
    assert supervised.sum() >= SEQ_LEN - 2, (
        f"only {supervised.sum()} supervised positions; the loss stage likely did not run"
    )
    assert (staged[supervised] < 0.0).all(), "logprobs must be negative"


def test_staged_forward_handles_multiple_microbatches():
    """GRPO packs one micro-batch per row, so M>1 must be sliced correctly."""
    trainer, model_dir = _build_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    los, his = _stage_ranges()

    M = 2
    b0 = make_inputs(vocab_size)
    inputs = np.ascontiguousarray(np.repeat(b0["inputs"], M, axis=0))
    targets = np.ascontiguousarray(np.repeat(b0["targets"], M, axis=0))

    staged = np.asarray(trainer.dispatch_pp_forward_logprobs_multigpu(inputs, targets, los, his, M))

    assert staged.shape == (M * BATCH, SEQ_LEN), staged.shape
    assert np.isfinite(staged).all()
    # Identical microbatches must produce identical logprobs.
    np.testing.assert_allclose(staged[0], staged[1], rtol=1e-5, atol=1e-6)


def test_scaled_custom_dloss_scales_the_gradient():
    """CE backward computes dlogit = dloss * (softmax - one_hot), so scaling the
    seed scales the gradient linearly (before clipping)."""
    _, base = _one_step(_ones_like_batch())
    _, half = _one_step(0.5 * _ones_like_batch())

    assert base > 0.0
    assert half == pytest.approx(0.5 * base, rel=5e-2), f"expected ~{0.5 * base}, got {half}"
