"""Phase-3 / integration gate: a fused dispatch-PP training step converges.

This is the end-to-end keystone: a single ``dispatch_pp_train_step`` chains the
sub-range executor's forward (computing the loss) -> backward (filling the grad
store) -> optimizer update, all on one GPU. Running it repeatedly on a *fixed* batch
must drive the loss down — proving the dispatch executor + optimizer actually train,
not just match a reference in isolation.

Single GPU: the cross-GPU stage handoff (forward + backward) is validated separately
in test_phase2_multigpu; this test closes the loop with a real optimizer update.
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
    SEQ_LEN,
    make_inputs,
    prepare_mini_model,
    resolve_model_path,
)

_MIN_FREE_BYTES = 6 * 1024**3


def _enough_free_gpu():
    if torch.cuda.device_count() < 1:
        return False
    free, _ = torch.cuda.mem_get_info(0)
    return free >= _MIN_FREE_BYTES


pytestmark = pytest.mark.skipif(
    not _enough_free_gpu(), reason=f"needs 1 GPU with >= {_MIN_FREE_BYTES // 1024**3} GiB free"
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
        ngpu=1,
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


def test_dispatch_train_step_converges_on_fixed_batch():
    trainer, model_dir = _build_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    b = make_inputs(vocab_size)
    opt_config = sg.OptimizerConfig.adamw(
        lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, grad_clip=1.0
    )

    n_steps = 15
    losses = [
        float(trainer.dispatch_pp_train_step(b["inputs"], b["targets"], opt_config, i))
        for i in range(n_steps)
    ]
    print("dispatch train losses:", [round(x, 4) for x in losses], flush=True)

    # Loss must be a sane positive scalar throughout.
    assert all(np.isfinite(losses)), losses
    assert losses[0] > 0.0
    # Overfitting one fixed batch should drive the loss well down.
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    assert losses[-1] < 0.9 * losses[0], f"loss barely moved: first={losses[0]:.3f} last={losses[-1]:.3f}"
    # The trajectory should be predominantly downward (allow minor bf16 wobble).
    drops = sum(1 for a, b_ in zip(losses, losses[1:]) if b_ <= a + 1e-3)
    assert drops >= (n_steps - 1) * 0.7, f"trajectory not monotone enough: {losses}"
