"""Phase-0 gate: numerical parity of contiguous block sub-range execution vs whole-graph.

Single GPU, weights resident. Proves the compiled executor can stop after block k
and resume at block k+1 with a CPU-boundary activation handoff, matching the
whole-graph forward hidden state and per-block backward grad norms.
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
    not torch.cuda.is_available(), reason="Phase-0 sub-range parity needs a GPU"
)


def _build_tiny_trainer():
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


def test_subrange_forward_matches_whole_graph():
    """Running blocks [0..L) as two sub-ranges [0..k] then [k+1..L) must produce
    the same hidden state as a single whole-graph forward, within tolerance."""
    trainer, model_dir = _build_tiny_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    inputs = make_inputs(vocab_size)["inputs"]
    split_after = (NUM_LAYERS // 2) - 1

    whole = np.asarray(trainer.dispatch_pp_debug_forward_hidden(inputs))
    part = np.asarray(trainer.dispatch_pp_debug_forward_subranges(inputs, split_after))
    np.testing.assert_allclose(whole, part, rtol=1e-2, atol=1e-2)


def test_subrange_backward_grad_matches_whole_graph():
    """The bounded (block-range, forced-eager) backward executor must produce the
    same per-block weight grads as the default whole-graph backward, within
    tolerance. This proves the executor can run the backward over an explicit op
    sub-range correctly; multi-stage gradient-accumulation handoff across separate
    invocations is a scheduler concern (deferred)."""
    trainer, model_dir = _build_tiny_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    batch = make_inputs(vocab_size)
    split_after = (NUM_LAYERS // 2) - 1

    g_whole = np.asarray(
        trainer.dispatch_pp_debug_grad_norms_whole(batch["inputs"], batch["targets"])
    )
    g_part = np.asarray(
        trainer.dispatch_pp_debug_grad_norms_subranges(batch["inputs"], batch["targets"], split_after)
    )
    np.testing.assert_allclose(g_whole, g_part, rtol=2e-2, atol=2e-2)
