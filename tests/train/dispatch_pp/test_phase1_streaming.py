"""Phase-1 gate: single-GPU weight streaming parity.

A model whose block work-weights are streamed per block from pinned CPU memory
(``offload_master=True`` -> ``DslWeightManager::needs_block_gather()`` -> per-block
gather/release in the executor) must produce the SAME forward hidden state and the
SAME per-block weight-grad norms as a resident-weights run. This proves dispatch-PP
can run with weights streamed (not resident), reusing the existing streaming stack.
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

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Phase-1 weight-streaming parity needs a GPU"
)


def _build_trainer(model_dir, *, offload_master):
    cfg = sg.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
    opts = sg.RuntimeOptions(
        offload_residual=False,
        use_cuda_graphs=False,
        offload_master=offload_master,  # streams block work-weights from pinned CPU
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
    return trainer


def _model_dir():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3 weights not found. Set QWEN3_MODEL_PATH or cache Qwen/Qwen3-0.6B.")
    return prepare_mini_model(snapshot)


def test_streamed_forward_matches_resident():
    """Per-block weight streaming must not change the forward hidden state."""
    model_dir = _model_dir()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    inputs = make_inputs(vocab_size)["inputs"]

    resident = _build_trainer(model_dir, offload_master=False)
    res_hidden = np.asarray(resident.dispatch_pp_debug_forward_hidden(inputs))
    del resident

    streamed = _build_trainer(model_dir, offload_master=True)
    str_hidden = np.asarray(streamed.dispatch_pp_debug_forward_hidden(inputs))

    np.testing.assert_allclose(res_hidden, str_hidden, rtol=1e-3, atol=1e-3)


def test_streamed_grads_match_resident():
    """Per-block weight streaming must not change the per-block weight-grad norms."""
    model_dir = _model_dir()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    batch = make_inputs(vocab_size)

    resident = _build_trainer(model_dir, offload_master=False)
    res_grads = np.asarray(
        resident.dispatch_pp_debug_grad_norms_whole(batch["inputs"], batch["targets"])
    )
    del resident

    streamed = _build_trainer(model_dir, offload_master=True)
    str_grads = np.asarray(
        streamed.dispatch_pp_debug_grad_norms_whole(batch["inputs"], batch["targets"])
    )

    assert res_grads.shape == (NUM_LAYERS,)
    np.testing.assert_allclose(res_grads, str_grads, rtol=2e-2, atol=2e-2)
