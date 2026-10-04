"""Streamed block weights under CUDA graphs: the capture risks of #237.

Each test runs the same steps eagerly and under CUDA graphs and compares the losses and gradient norms.

1. An internal (per-pass) graph skips every block gather while it captures (handle_layer_start), so it
   replays whatever the prefetch slots held at capture time. #227 kept the internal forward graph away
   from sharded weights only, and offload_master streams block weights the same way. A capture before
   any gather reads the masters in place and is correct, so an eager pass runs first, as
   train_step_graphed's warmup does before an evaluation.
2. The memcpy all-gather waits on events recorded on the other ranks' streams, which a stream capture
   forbids. The NCCL gather is covered by test_zero3_nccl_parity.py; this path was not.
3. When a packed batch has more documents than the captured cap, train_step_graphed captures again
   without an eager warmup.

Qwen3 (attention only) is used because it has no capture-unsafe ops: a model with them makes
train_step_graphed fall back to an eager step, and nothing would be captured.

The ZeRO-3 tests need two GPUs. On a host whose GPU peer-to-peer path is broken, run with
NCCL_P2P_DISABLE=1.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:
    import surogate._surogate as _surogate
except ImportError:
    pytest.skip("surogate._surogate C++ extension not built", allow_module_level=True)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

SEQ = 256


@pytest.fixture(scope="module")
def model_dir():
    if torch.cuda.device_count() < 1:
        pytest.skip("needs a GPU")
    from tests import test_onboarding_qwen3 as onboarding

    snapshot = onboarding.resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3-0.6B not available (set QWEN3_MODEL_PATH)")
    return onboarding.prepare_mini_model(snapshot)


def _needs_two_gpus():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")


def _build(model_dir, ngpu: int, graphs: bool, *, zero3=False, offload_master=False, memcpy_all_gather=False):
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    opts = _surogate.RuntimeOptions(
        recipe="bf16",
        use_cuda_graphs=graphs,
        offload_master=offload_master,
        offload_grads=False,
        offload_optimizer=False,
        offload_residual=False,
        shard_weights=zero3,
        shard_gradients=zero3,
    )
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))
    manifests = compile_jit_kernels(opts.dsl_ir_json)
    if manifests:
        opts.jit_kernel_manifests = manifests
    trainer = _surogate.SurogateTrainer(
        ngpu=ngpu,
        config=_surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16"),
        options=opts,
        batch_size=1,
        seq_len=SEQ,
        grad_accum=1,
        memcpy_all_gather=memcpy_all_gather,
        memcpy_send_recv=False,
        lora_config=None,
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    return trainer


def _rows(ngpu: int, doc_lens=(SEQ,), seed=0):
    """One row, the same on every rank, so the averaged gradient is the single-GPU one.

    `doc_lens` packs that many documents into the row: positions restart at each one, and a document's
    last token has no target."""
    assert sum(doc_lens) == SEQ
    rng = np.random.default_rng(seed)
    x = rng.integers(10, 8000, size=(1, SEQ), dtype=np.int32)
    y = np.concatenate([x[:, 1:], np.full((1, 1), -100, np.int32)], axis=1)
    pos = np.concatenate([np.arange(n, dtype=np.int32) for n in doc_lens])[None, :]
    for end in np.cumsum(doc_lens):
        y[0, end - 1] = -100
    rep = lambda a: np.ascontiguousarray(np.repeat(a, ngpu, axis=0).astype(np.int32))  # noqa: E731
    return rep(x), rep(y), rep(pos)


def _optimizer(lr: float):
    return _surogate.OptimizerConfig(optimizer="adamw_8bit", learning_rate=lr, grad_clip=0.0)


def _release(trainer):
    del trainer
    torch.cuda.empty_cache()


def _eval_and_step(trainer, ngpu: int) -> list[float]:
    """An eager pass (log-probs are forward-only, which never runs a graph), two validations, then two
    forward+backward steps at a zero learning rate: the first validation captures the internal forward
    graph, every later forward replays it."""
    x, y, pos = _rows(ngpu)
    trainer.compute_logprobs(np.ascontiguousarray(x[:1]), np.ascontiguousarray(y[:1]), False, None, None)
    out = [float(trainer.validate(x, y, pos)) for _ in range(2)]
    for step in range(2):
        trainer.step(x, y, pos)
        out.append(float(dict(trainer.update_with_config(_optimizer(0.0), step))["norm"]))
    return out


def _assert_close(got, want, what, rtol=1e-2):
    assert np.all(np.isfinite(got)), (what, got)
    for i, (g, w) in enumerate(zip(got, want)):
        assert abs(g - w) <= rtol * abs(w), (what, i, got, want)


@pytest.fixture(scope="module")
def eager(model_dir):
    trainer = _build(model_dir, ngpu=1, graphs=False)
    out = _eval_and_step(trainer, ngpu=1)
    _release(trainer)
    return out


def test_offload_master_internal_forward_graph_matches_eager(model_dir, eager):
    """Risk 1: offload_master without sharding streams block weights into the prefetch slots too."""
    trainer = _build(model_dir, ngpu=1, graphs=True, offload_master=True)
    got = _eval_and_step(trainer, ngpu=1)
    _release(trainer)
    _assert_close(got[:2], eager[:2], "validation loss")
    _assert_close(got[2:], eager[2:], "gradient norm", rtol=5e-2)


@pytest.mark.parametrize(
    "ngpu,streamed", [(1, {"offload_master": True}), (2, {"zero3": True})], ids=["offload_master", "zero3"]
)
def test_evaluation_after_graphed_step_matches_eager(model_dir, ngpu, streamed):
    """Risk 1 as the issue puts it: train_step_graphed re-enables the internal graphs when it returns,
    and the next evaluation runs on whatever the full-step graph left in the prefetch slots. The step
    moves the weights, so an evaluation that read the slots the graph gathered before its update would
    show. The evaluation also waits on the slot events, which the graph last recorded inside its
    capture."""
    if ngpu > 1:
        _needs_two_gpus()
    x, y, pos = _rows(ngpu)
    out = []
    for graphs in (False, True):
        trainer = _build(model_dir, ngpu, graphs, **streamed)
        trainer.train_step_graphed(x, y, pos, _optimizer(1e-3), 1)
        out.append([float(trainer.validate(x, y, pos)) for _ in range(2)])
        _release(trainer)
    _assert_close(out[1], out[0], "validation loss")


def _graphed_steps(trainer, ngpu: int, batches, lr: float) -> list[tuple[float, float]]:
    out = []
    for step, doc_lens in enumerate(batches):
        result = dict(trainer.train_step_graphed(*_rows(ngpu, doc_lens, seed=step), _optimizer(lr), step + 1))
        out.append((float(result["loss"]), float(result["norm"])))
    return out


def test_zero3_memcpy_all_gather_full_step_graph_matches_eager(model_dir):
    """Risk 2: the full-step graph over the memcpy all-gather (capture, then a replay)."""
    _needs_two_gpus()
    batches = [(SEQ,), (SEQ,)]
    trainer = _build(model_dir, ngpu=1, graphs=False)
    want = _graphed_steps(trainer, 1, batches, lr=0.0)
    _release(trainer)
    trainer = _build(model_dir, ngpu=2, graphs=True, zero3=True, memcpy_all_gather=True)
    got = _graphed_steps(trainer, 2, batches, lr=0.0)
    _release(trainer)
    _assert_close([l for l, _ in got], [l for l, _ in want], "loss")
    _assert_close([n for _, n in got], [n for _, n in want], "gradient norm", rtol=5e-2)


def test_zero3_recapture_after_doc_cap_growth_matches_eager(model_dir):
    """Risk 3: two documents capture with a cap of two; five force a capture with no warmup, then a
    replay. The optimizer moves the weights, so a replay that read a stale prefetch slot would show."""
    _needs_two_gpus()
    batches = [(128, 128), (64, 48, 48, 48, 48), (48, 64, 48, 48, 48)]
    trainer = _build(model_dir, ngpu=2, graphs=False, zero3=True)
    want = _graphed_steps(trainer, 2, batches, lr=1e-3)
    _release(trainer)
    trainer = _build(model_dir, ngpu=2, graphs=True, zero3=True)
    got = _graphed_steps(trainer, 2, batches, lr=1e-3)
    _release(trainer)
    _assert_close([l for l, _ in got], [l for l, _ in want], "loss")
    _assert_close([n for _, n in got], [n for _, n in want], "gradient norm", rtol=5e-2)
