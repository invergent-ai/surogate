"""Phase-2 gate: multi-GPU round-robin forward AND backward dispatch.

Partition the model into contiguous block stages and dispatch them round-robin
across the GPU pool (stage i -> GPU i % ngpu), handing the boundary state from one
GPU to the next *through host memory* (no NCCL). This exercises the dispatch-PP
stateless pool: any stage runs on any GPU, rendezvous through CPU.

Forward: the two-tensor fused-residual boundary (blocks[hi].res_att and
blocks[hi].mlp_down = x) is read by name on the sending GPU and bound into the same
graph tids on the receiving GPU; the final hidden matches whole-graph forward.

Backward: stages run in reverse order (loss-owning stage first). Each stage's ops
are selected by their owning block layer (robust to boundary view ops), runs one
GPU on the full batch with the DP grad/loss all-reduce skipped, and hands the
boundary gradients (d_blocks[hi].res_att / .mlp_down) to the next lower stage's GPU
through host memory. Per-block weight-grad norms match whole-graph backward within
bf16 tolerance.
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
_MIN_FREE_BYTES = 6 * 1024**3  # the DSL stack alone is ~3 GB per GPU


def _enough_free_gpus():
    if torch.cuda.device_count() < _NGPU:
        return False
    # The trainer uses physical GPUs 0..ngpu-1; skip if any is busy (e.g. shared box).
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


def test_multigpu_dispatch_runs_end_to_end():
    """Round-robin dispatch across GPUs runs and returns the right-shaped output —
    proves the stateless-pool orchestration + residual handoff plumbing."""
    trainer, model_dir = _build_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    inputs = make_inputs(vocab_size)["inputs"]
    single = np.asarray(trainer.dispatch_pp_debug_forward_hidden(inputs))
    los, his = _stage_ranges([NUM_LAYERS // 2 - 1])
    multi = np.asarray(trainer.dispatch_pp_debug_forward_hidden_multigpu(inputs, los, his))
    assert multi.shape == single.shape and multi.size > 0


def test_multigpu_dispatch_matches_single_gpu():
    """Final hidden from round-robin multi-GPU dispatch matches whole-graph forward.

    The two-tensor fused-residual boundary — residual-after-attention
    (blocks[hi].res_att) and x (blocks[hi].mlp_down) — is read by name on the
    sending GPU (kept live by the preserve-last-block hook) and bound into the
    same graph tids on the receiving GPU through host memory. Tolerance is
    bf16-level (the cross-GPU path reorders nothing mathematically; residual is
    bf16 rounding)."""
    trainer, model_dir = _build_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    inputs = make_inputs(vocab_size)["inputs"]
    single = np.asarray(trainer.dispatch_pp_debug_forward_hidden(inputs))
    los, his = _stage_ranges([NUM_LAYERS // 2 - 1])
    multi = np.asarray(trainer.dispatch_pp_debug_forward_hidden_multigpu(inputs, los, his))
    np.testing.assert_allclose(single, multi, rtol=2e-2, atol=2e-2)


def test_multigpu_dispatch_round_robin_wrap():
    """More stages than GPUs (one block per stage) exercises round-robin GPU reuse
    and multiple host handoffs; still matches whole-graph forward within bf16 tol."""
    trainer, model_dir = _build_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    inputs = make_inputs(vocab_size)["inputs"]
    single = np.asarray(trainer.dispatch_pp_debug_forward_hidden(inputs))
    los, his = _stage_ranges(list(range(NUM_LAYERS - 1)))  # [0]|[1]|[2]|[3]
    multi = np.asarray(trainer.dispatch_pp_debug_forward_hidden_multigpu(inputs, los, his))
    np.testing.assert_allclose(single, multi, rtol=5e-2, atol=5e-2)


def test_multigpu_backward_runs_end_to_end():
    """Round-robin backward dispatch across GPUs runs and returns per-block grad
    norms — proves the reverse-order stage scheduling + cross-GPU gradient handoff
    plumbing (no missing boundary tensors)."""
    trainer, model_dir = _build_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    b = make_inputs(vocab_size)
    los, his = _stage_ranges([NUM_LAYERS // 2 - 1])
    norms = np.asarray(trainer.dispatch_pp_debug_grad_norms_multigpu(b["inputs"], b["targets"], los, his))
    assert norms.shape == (NUM_LAYERS,)
    assert np.all(norms > 0.0)


def test_multigpu_backward_matches_single_gpu():
    """Per-block weight-grad norms from two-stage multi-GPU backward match whole-graph
    backward. The boundary gradients (d_blocks[hi].res_att / .mlp_down) cross GPUs
    through host memory; tolerance is bf16-level (the separate per-stage forward
    recompute + bf16 boundary rounding)."""
    trainer, model_dir = _build_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    b = make_inputs(vocab_size)
    whole = np.asarray(trainer.dispatch_pp_debug_grad_norms_whole(b["inputs"], b["targets"]))
    los, his = _stage_ranges([NUM_LAYERS // 2 - 1])
    multi = np.asarray(trainer.dispatch_pp_debug_grad_norms_multigpu(b["inputs"], b["targets"], los, his))
    np.testing.assert_allclose(multi, whole, rtol=1e-1, atol=1e-1)


def test_multigpu_backward_round_robin_wrap():
    """One block per backward stage (more stages than GPUs) exercises round-robin GPU
    reuse and multiple gradient handoffs; per-block grad norms still match whole-graph
    backward within bf16 tolerance."""
    trainer, model_dir = _build_trainer()
    vocab_size = json.loads((Path(model_dir) / "config.json").read_text())["vocab_size"]
    b = make_inputs(vocab_size)
    whole = np.asarray(trainer.dispatch_pp_debug_grad_norms_whole(b["inputs"], b["targets"]))
    los, his = _stage_ranges(list(range(NUM_LAYERS - 1)))  # [0]|[1]|[2]|[3]
    multi = np.asarray(trainer.dispatch_pp_debug_grad_norms_multigpu(b["inputs"], b["targets"], los, his))
    np.testing.assert_allclose(multi, whole, rtol=1e-1, atol=1e-1)
