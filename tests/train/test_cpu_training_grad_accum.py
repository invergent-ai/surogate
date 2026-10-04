"""cpu_training full fine-tuning computes the on-device step, with and without gradient accumulation.

Full fine-tune under cpu_training streams each layer's gradients through device slots to pinned
host memory, where the CPU optimizer steps the host masters. Four things kept it from training:

- The grad store filed a gradient under its layer only for "blocks.N." / "layers.N." names. DSL
  names are "blocks[N].<name>", so every layer's gradients were placeholders never bound to a slot
  or copied to the host: the CPU optimizer stepped on zeros for every block parameter, and the
  reported norm was that of the embeddings, head and final norm alone.
- With the layers filed, they share one host staging buffer per weight name on micro-steps after
  the first, which used to be added into the host gradients once, after the last micro-step: every
  layer received the last layer staged, and the micro-steps in between were dropped. Each layer's
  staged copy is now added as it lands.
- The backward moved every op that only writes a parameter gradient (here the view of each
  linear-attention conv weight's gradient back to the weight's shape) after the embedding
  backward, past the end of its layer, where the layer's gradients had already gone to the host:
  every conv weight but the first layer's stayed put.
- The export read the work copies, which under offloaded masters are prefetch slots (whichever layer
  was gathered last) and stale non-block copies; it now reads the masters, as under ZeRO-3.

Both runs below train the same rows as GRAD_ACCUM micro-steps with the options SFTConfig sets for
cpu_training (offloaded masters and gradients), so their loss, gradient norm and the weights after
one update must agree.
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


@pytest.fixture(scope="module")
def model_dir():
    if torch.cuda.device_count() < 1:
        pytest.skip("needs a GPU")
    from tests import test_onboarding_qwen3_5 as onboarding

    snapshot = onboarding.resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3.5-0.8B not available (set QWEN3_5_MODEL_PATH)")
    return onboarding.prepare_mini_model(snapshot)


def _build(model_dir, cpu_training: bool, grad_accum: int):
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    opts = _surogate.RuntimeOptions(
        recipe="bf16",
        use_cuda_graphs=False,
        cpu_training=cpu_training,
        # What SFTConfig sets for a cpu_training full fine-tune: masters and gradients on the host.
        offload_master=cpu_training,
        offload_grads=cpu_training,
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
        grad_accum=grad_accum,
        memcpy_all_gather=False,
        memcpy_send_recv=False,
        lora_config=None,
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    return trainer


def _rows(grad_accum: int):
    """One distinct row per micro-step."""
    rng = np.random.default_rng(0)
    x = rng.integers(10, 8000, size=(grad_accum, SEQ), dtype=np.int32)
    y = np.concatenate([x[:, 1:], np.full((grad_accum, 1), -100, np.int32)], axis=1).astype(np.int32)
    pos = np.tile(np.arange(SEQ, dtype=np.int32), (grad_accum, 1))
    return x, y, pos


def _step(model_dir, cpu_training: bool, grad_accum: int, export_dir) -> tuple[float, float]:
    trainer = _build(model_dir, cpu_training, grad_accum)
    config = _surogate.OptimizerConfig(optimizer="adamw_8bit", learning_rate=LR, grad_clip=0.0)
    result = dict(trainer.train_step_graphed(*_rows(grad_accum), config, 1))
    trainer.export_model(str(export_dir))
    del trainer
    torch.cuda.empty_cache()
    return float(result["loss"]), float(result["norm"])


@pytest.mark.parametrize("grad_accum", [1, 3])
def test_cpu_training_matches_on_device(model_dir, tmp_path, grad_accum):
    from safetensors.torch import load_file

    ref_loss, ref_norm = _step(model_dir, False, grad_accum, tmp_path / "device")
    loss, norm = _step(model_dir, True, grad_accum, tmp_path / "cpu")
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
