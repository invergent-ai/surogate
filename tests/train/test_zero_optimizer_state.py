"""ZeRO shards the 8-bit AdamW state across the ranks, and the step still computes the single-GPU update (#231).

Below ZeRO-3 every rank holds every master whole, and at ZeRO-3 the embedding and lm_head stay whole
too. Each rank used to keep AdamW state for all of them: at ZeRO-1 every rank held the whole model's
state. Now each rank keeps state for, and updates, its own 1/world slice of such a master, and the
slices are all-gathered after the step. Every rank gets the same rows, so the averaged gradients equal
the single-GPU step's.

Needs two GPUs. On a host whose GPU peer-to-peer path is broken, run with NCCL_P2P_DISABLE=1.
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


def _build(model_dir, ngpu: int, level: str = "zero1", graphs: bool = False):
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
        ngpu=ngpu,
        config=_surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16"),
        options=opts,
        batch_size=1,
        seq_len=SEQ,
        grad_accum=1,
        memcpy_all_gather=False,
        memcpy_send_recv=False,
        lora_config=None,
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    return trainer


def _rows(ngpu: int, seed: int = 0):
    """The same row on every rank, so the averaged gradient is the single-GPU one."""
    rng = np.random.default_rng(seed)
    x = np.repeat(rng.integers(10, 8000, size=(1, SEQ), dtype=np.int32), ngpu, axis=0)
    y = np.concatenate([x[:, 1:], np.full((ngpu, 1), -100, np.int32)], axis=1).astype(np.int32)
    pos = np.tile(np.arange(SEQ, dtype=np.int32), (ngpu, 1))
    return x, y, pos


def _config():
    return _surogate.OptimizerConfig(optimizer="adamw_8bit", learning_rate=LR, grad_clip=0.0)


def _state_bytes(trainer, gpu: int) -> int:
    """Device bytes of the 8-bit AdamW state (moments and scales) on one rank."""
    return int(trainer.get_allocator_info(gpu)["AdamW8bit_OptState"]["device"])


def _release(trainer):
    del trainer
    torch.cuda.empty_cache()


def _assert_same_model(path_a, path_b):
    """Both runs reduce the same gradient, in a different order. Adam's first step is sign-like, so an
    element whose gradient is ~0 may step either way: a few elements are a learning rate or two apart,
    the rest at most a bf16 rounding. A slice that missed the all-gather would hold the imported value,
    an Adam step away in nearly every element."""
    from safetensors.torch import load_file

    a_all = load_file(str(path_a / "model.safetensors"))
    b_all = load_file(str(path_b / "model.safetensors"))
    assert a_all.keys() == b_all.keys()
    for name, tensor in a_all.items():
        assert b_all[name].dtype == tensor.dtype and b_all[name].shape == tensor.shape, name
        a, b = b_all[name].float(), tensor.float()
        far = ~torch.isclose(a, b, rtol=2**-7, atol=1e-6)
        assert int(far.sum()) <= max(2, 0.05 * far.numel()), (name, int(far.sum()), far.numel())
        assert float((a - b).abs().max()) <= 2.5 * LR, name


@pytest.fixture(scope="module")
def single_gpu(model_dir, tmp_path_factory):
    """State size and exported model of one single-GPU step."""
    out = tmp_path_factory.mktemp("single")
    trainer = _build(model_dir, ngpu=1)
    trainer.train_step_graphed(*_rows(1), _config(), 1)
    state = _state_bytes(trainer, 0)
    trainer.export_model(str(out / "step1"))
    _release(trainer)
    return state, out


@pytest.mark.parametrize("level", list(LEVELS))
def test_each_rank_keeps_half_the_state_and_computes_the_single_gpu_step(model_dir, single_gpu, level, tmp_path):
    single_state, single_out = single_gpu
    trainer = _build(model_dir, ngpu=2, level=level)
    trainer.train_step_graphed(*_rows(2), _config(), 1)
    for gpu in (0, 1):
        # Every master is split evenly here; only the per-tensor group padding differs.
        assert abs(_state_bytes(trainer, gpu) - single_state / 2) <= 0.01 * single_state, (gpu, single_state)
    trainer.export_model(str(tmp_path / "step1"))
    _release(trainer)
    _assert_same_model(single_out / "step1", tmp_path / "step1")


def test_zero1_cuda_graph_step_gathers_the_slices(model_dir, single_gpu, tmp_path):
    """The all-gather of the updated slices runs inside the captured optimizer step."""
    _, single_out = single_gpu
    trainer = _build(model_dir, ngpu=2, level="zero1", graphs=True)
    trainer.train_step_graphed(*_rows(2), _config(), 1)
    trainer.export_model(str(tmp_path / "step1"))
    _release(trainer)
    _assert_same_model(single_out / "step1", tmp_path / "step1")


def _optimizer_state_files(ckpt) -> dict[str, bytes]:
    return {str(f.relative_to(ckpt)): f.read_bytes() for f in sorted(ckpt.rglob("adam.*")) if f.is_file()}


def test_zero1_resume_restores_each_ranks_slice_of_the_state(model_dir, tmp_path):
    """Each rank writes its own slice of the state and reads it back: a resumed run saves the same state
    files again, and its next step matches the step of the run that wrote the checkpoint.

    The reference is that run itself rather than a single-GPU run: a second step compounds the
    run-to-run spread of the first (an element that stepped the other way changes every gradient)."""
    trainer = _build(model_dir, ngpu=2, level="zero1")
    trainer.train_step_graphed(*_rows(2), _config(), 1)
    trainer.save_checkpoint(str(tmp_path / "ckpt"), 1)
    trainer.train_step_graphed(*_rows(2, seed=1), _config(), 2)
    trainer.export_model(str(tmp_path / "continued"))
    _release(trainer)

    trainer = _build(model_dir, ngpu=2, level="zero1")
    trainer.load_checkpoint(str(tmp_path / "ckpt"), 1)
    trainer.save_checkpoint(str(tmp_path / "resaved"), 1)
    trainer.train_step_graphed(*_rows(2, seed=1), _config(), 2)
    trainer.export_model(str(tmp_path / "resumed"))
    _release(trainer)

    saved = _optimizer_state_files(tmp_path / "ckpt")
    assert len(saved) == 4, sorted(saved)  # moments and variances, one file per rank
    assert _optimizer_state_files(tmp_path / "resaved") == saved
    _assert_same_model(tmp_path / "continued", tmp_path / "resumed")
