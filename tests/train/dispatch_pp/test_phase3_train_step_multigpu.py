"""Phase-3 / integration gate: the multi-GPU fused dispatch-PP step converges.

This closes the loop across the GPU pool: ``dispatch_pp_train_step_multigpu``
runs the backward dispatch round-robin (stages on different GPUs, boundary gradients
handed GPU->host->GPU), collects every stage's gradients onto the master GPU, runs
the optimizer there, then **broadcasts** the updated weights back to every replica so
the pool stays consistent for the next step. Repeated on a fixed batch the loss must
fall — proving multi-GPU dispatch training with cross-GPU weight consistency.

(Every parameter trains: the lowest stage runs the trailing embedding-backward op and
collects its gradient, the loss-owning stage collects lm_head / final_norm, and each
block's gradients come from its own stage.)
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


def _stage_ranges(splits):
    los, his, lo = [], [], 0
    for cut in splits:
        los.append(lo)
        his.append(cut)
        lo = cut + 1
    los.append(lo)
    his.append(NUM_LAYERS - 1)
    return los, his


def test_multigpu_dispatch_train_step_converges():
    trainer, model_dir = _build_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    b = make_inputs(vocab_size)
    opt_config = sg.OptimizerConfig.adamw(
        lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, grad_clip=1.0
    )
    los, his = _stage_ranges([NUM_LAYERS // 2 - 1])  # 2 stages -> GPU 0 and GPU 1

    n_steps = 15
    losses = [
        float(trainer.dispatch_pp_train_step_multigpu(b["inputs"], b["targets"], los, his, opt_config, i))
        for i in range(n_steps)
    ]
    print("multigpu dispatch train losses:", [round(x, 3) for x in losses], flush=True)

    assert all(np.isfinite(losses)), losses
    assert losses[0] > 0.0
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    assert losses[-1] < 0.9 * losses[0], f"loss barely moved: first={losses[0]:.3f} last={losses[-1]:.3f}"
    drops = sum(1 for a, c in zip(losses, losses[1:]) if c <= a + 1e-3 * abs(a))
    assert drops >= (n_steps - 1) * 0.7, f"trajectory not monotone enough: {losses}"


def test_multigpu_dispatch_train_step_stale_converges():
    """One-step-stale mode (stale=True) defers each step's optimizer update by one
    step — every step trains on weights one update behind, the RoundPipe v1 default.
    It must still converge on a fixed batch (controlled staleness does not break
    training). dispatch_pp_flush_pending applies the final deferred grads."""
    trainer, model_dir = _build_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    b = make_inputs(vocab_size)
    opt_config = sg.OptimizerConfig.adamw(
        lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, grad_clip=1.0
    )
    los, his = _stage_ranges([NUM_LAYERS // 2 - 1])

    n_steps = 16
    losses = [
        float(trainer.dispatch_pp_train_step_multigpu(b["inputs"], b["targets"], los, his, opt_config, i, True))
        for i in range(n_steps)
    ]
    trainer.dispatch_pp_flush_pending(opt_config)
    print("multigpu stale dispatch train losses:", [round(x, 3) for x in losses], flush=True)

    assert all(np.isfinite(losses)), losses
    assert losses[0] > 0.0
    # Controlled one-step staleness still drives the loss down (allowing the one-step
    # lag: the first two steps see the same pre-update weights).
    assert losses[-1] < 0.9 * losses[0], f"stale loss barely moved: first={losses[0]:.3f} last={losses[-1]:.3f}"
