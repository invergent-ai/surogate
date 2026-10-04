"""A CUDA-graphed multi-GPU step reduces every gradient, as the eager step does, and both clip and
report the norm of the averaged gradient.

The eager backward reduces each layer's gradients at the layer's end (DslGradStore::notify_block), and
the reduction at the end of the step then skips them. A CUDA graph capture skips the layer-end work,
so the end of the step has to reduce the layer gradients too. Since layer gradients are found by
their DSL names ("blocks[N].<name>"), a graphed ZeRO-1 or ZeRO-2 step failed to capture
(cudaErrorStreamCaptureIsolation), and a graphed ZeRO-3 step updated each rank's shard from that
rank's own gradient.

Under ZeRO-2/3 the eager step leaves each layer gradient reduce-scattered: a rank holds its own slice
reduced and the rest unreduced. The norm summed every rank's whole buffers, so it differed from rank to
rank and from the graphed step's, which reduces every gradient whole. 32-bit AdamW read the unreduced
rest of the buffer. With two micro-steps the reduce-scatter ran on each, so the second one accumulated
onto the first one's reduced slice.

Each rank gets a different row here: with the same row everywhere, an unreduced gradient equals the
reduced one. Needs two GPUs. On a host whose GPU peer-to-peer path is broken, run with
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
LR = 1e-3

# (shard_weights, shard_gradients) per ZeRO level
LEVELS = {"zero1": (False, False), "zero2": (False, True), "zero3": (True, True)}


@pytest.fixture(scope="module")
def model_dir():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    from tests import test_onboarding_qwen3_5 as onboarding

    snapshot = onboarding.resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3.5-0.8B not available (set QWEN3_5_MODEL_PATH)")
    return onboarding.prepare_mini_model(snapshot)


def _build(model_dir, level: str, graphs: bool, grad_accum: int):
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    shard_weights, shard_gradients = LEVELS[level]
    opts = _surogate.RuntimeOptions(
        recipe="bf16",
        use_cuda_graphs=graphs,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        offload_residual=False,
        shard_weights=shard_weights,
        shard_gradients=shard_gradients,
    )
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))
    manifests = compile_jit_kernels(opts.dsl_ir_json)
    if manifests:
        opts.jit_kernel_manifests = manifests
    trainer = _surogate.SurogateTrainer(
        ngpu=2,
        config=_surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16"),
        options=opts,
        batch_size=1,
        seq_len=SEQ,
        grad_accum=grad_accum,
        memcpy_all_gather=False,
        memcpy_send_recv=False,
        lora_config=None,
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    return trainer


def _rows(grad_accum: int):
    """A different row on each rank in each micro-step."""
    rows = 2 * grad_accum
    rng = np.random.default_rng(0)
    x = rng.integers(10, 8000, size=(rows, SEQ), dtype=np.int32)
    y = np.concatenate([x[:, 1:], np.full((rows, 1), -100, np.int32)], axis=1).astype(np.int32)
    pos = np.tile(np.arange(SEQ, dtype=np.int32), (rows, 1))
    return x, y, pos


def _step(model_dir, level: str, optimizer: str, grad_accum: int, graphs: bool, out):
    trainer = _build(model_dir, level, graphs, grad_accum)
    config = _surogate.OptimizerConfig(optimizer=optimizer, learning_rate=LR, grad_clip=0.0)
    result = dict(trainer.train_step_graphed(*_rows(grad_accum), config, 1))
    trainer.export_model(str(out))
    del trainer
    torch.cuda.empty_cache()
    return float(result["loss"]), float(result["norm"])


def _assert_same_model(path_a, path_b):
    """Adam's first step is sign-like, so an element whose gradient is ~0 may step either way: a few
    elements are a learning rate or two apart, the rest at most a bf16 rounding. A rank that stepped on
    its own gradient instead of the average differs wherever the two signs differ."""
    from safetensors.torch import load_file

    a_all = load_file(str(path_a / "model.safetensors"))
    b_all = load_file(str(path_b / "model.safetensors"))
    assert a_all.keys() == b_all.keys()
    for name, tensor in a_all.items():
        a, b = tensor.float(), b_all[name].float()
        far = ~torch.isclose(a, b, rtol=2**-7, atol=1e-6)
        assert int(far.sum()) <= max(2, 0.05 * far.numel()), (name, int(far.sum()), far.numel())
        assert float((a - b).abs().max()) <= 2.5 * LR, name


@pytest.mark.parametrize(
    "level,optimizer",
    [(level, "adamw_8bit") for level in LEVELS] + [("zero2", "adamw")],
)
def test_graphed_step_reduces_every_gradient(model_dir, level, optimizer, tmp_path):
    eager_loss, eager_norm = _step(model_dir, level, optimizer, 1, graphs=False, out=tmp_path / "eager")
    graphed_loss, graphed_norm = _step(model_dir, level, optimizer, 1, graphs=True, out=tmp_path / "graphed")
    assert abs(graphed_loss - eager_loss) <= 1e-2 * abs(eager_loss), (graphed_loss, eager_loss)
    assert abs(graphed_norm - eager_norm) <= 3e-2 * eager_norm, (graphed_norm, eager_norm)
    _assert_same_model(tmp_path / "eager", tmp_path / "graphed")


def test_zero2_accumulates_micro_steps_as_zero1(model_dir, tmp_path):
    """ZeRO-2 reduce-scatters a layer's gradients once, on the last micro-step, as ZeRO-1 all-reduces them.

    The reference is the eager ZeRO-1 step rather than a graphed one: a captured embedding backward keeps
    the token layout it computed on the host at capture time, so a graphed step with two micro-steps
    sends the first micro-step's embedding gradient to the second one's token rows."""
    zero1_loss, zero1_norm = _step(model_dir, "zero1", "adamw_8bit", 2, graphs=False, out=tmp_path / "zero1")
    zero2_loss, zero2_norm = _step(model_dir, "zero2", "adamw_8bit", 2, graphs=False, out=tmp_path / "zero2")
    assert abs(zero2_loss - zero1_loss) <= 1e-2 * abs(zero1_loss), (zero2_loss, zero1_loss)
    assert abs(zero2_norm - zero1_norm) <= 3e-2 * zero1_norm, (zero2_norm, zero1_norm)
    _assert_same_model(tmp_path / "zero1", tmp_path / "zero2")
