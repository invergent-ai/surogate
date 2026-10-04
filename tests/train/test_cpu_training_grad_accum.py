"""cpu_training with gradient accumulation computes the on-device step.

Full fine-tune under cpu_training streams each layer's gradients to pinned host memory. On
micro-steps after the first they land in one staging buffer per weight name, shared by every
layer, and used to be added into the host gradients only once, after the last micro-step's
backward: every layer received the gradient of the last layer staged (layer 0), and the micro-steps
between the first and the last were dropped. Each layer's staged copy is now added as it lands.

Both runs below train the same two rows as two micro-steps, so their gradient norm and the weights
after one update must agree; the bug moved the norm and flipped the sign of the update in most
elements of every layer but the first.
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
GRAD_ACCUM = 2
LR = 1e-3


@pytest.fixture(scope="module")
def model_dir():
    if torch.cuda.device_count() < 1:
        pytest.skip("needs a GPU")
    from tests import test_onboarding_qwen3_5 as onboarding

    snapshot = onboarding.resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3.5-0.8B not available (set QWEN3_5_MODEL_PATH)")
    return onboarding.prepare_mini_model(snapshot)


def _build(model_dir, cpu_training: bool):
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    opts = _surogate.RuntimeOptions(
        recipe="bf16",
        use_cuda_graphs=False,
        cpu_training=cpu_training,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        offload_residual=False,
        shard_weights=False,
        shard_gradients=False,
    )
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))
    manifests = compile_jit_kernels(opts.dsl_ir_json)
    if manifests:
        opts.jit_kernel_manifests = manifests
    trainer = _surogate.SurogateTrainer(
        ngpu=1,
        config=_surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16"),
        options=opts,
        batch_size=1,
        seq_len=SEQ,
        grad_accum=GRAD_ACCUM,
        memcpy_all_gather=False,
        memcpy_send_recv=False,
        lora_config=None,
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    return trainer


def _rows():
    """One distinct row per micro-step."""
    rng = np.random.default_rng(0)
    x = rng.integers(10, 8000, size=(GRAD_ACCUM, SEQ), dtype=np.int32)
    y = np.concatenate([x[:, 1:], np.full((GRAD_ACCUM, 1), -100, np.int32)], axis=1).astype(np.int32)
    pos = np.tile(np.arange(SEQ, dtype=np.int32), (GRAD_ACCUM, 1))
    return x, y, pos


def _step(model_dir, cpu_training: bool, export_dir) -> tuple[float, float]:
    trainer = _build(model_dir, cpu_training)
    config = _surogate.OptimizerConfig(optimizer="adamw_8bit", learning_rate=LR, grad_clip=0.0)
    result = dict(trainer.train_step_graphed(*_rows(), config, 1))
    trainer.export_model(str(export_dir))
    del trainer
    torch.cuda.empty_cache()
    return float(result["loss"]), float(result["norm"])


def test_cpu_training_grad_accum_matches_on_device(model_dir, tmp_path):
    from safetensors.torch import load_file

    ref_loss, ref_norm = _step(model_dir, cpu_training=False, export_dir=tmp_path / "device")
    loss, norm = _step(model_dir, cpu_training=True, export_dir=tmp_path / "cpu")
    assert np.isfinite(norm)
    assert abs(loss - ref_loss) <= 1e-2 * abs(ref_loss), (loss, ref_loss)
    assert abs(norm - ref_norm) <= 5e-2 * ref_norm, (norm, ref_norm)

    device = load_file(str(tmp_path / "device" / "model.safetensors"))
    cpu = load_file(str(tmp_path / "cpu" / "model.safetensors"))
    assert cpu.keys() == device.keys()
    for name, tensor in device.items():
        # The two runs reduce the same gradient in a different order and update it on different
        # devices. Adam's first step is sign-like, so an element whose gradient is ~0 may step either
        # way: a few elements are up to two learning rates apart, the rest at most a bf16 rounding.
        a, b = cpu[name].float(), tensor.float()
        far = ~torch.isclose(a, b, rtol=2**-7, atol=1e-6)
        assert int(far.sum()) <= max(2, 0.05 * far.numel()), (name, int(far.sum()), far.numel())
        assert float((a - b).abs().max()) <= 2.5 * LR, name
