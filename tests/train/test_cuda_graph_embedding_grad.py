"""A CUDA-graphed training step computes the embedding gradient of the tokens it runs on (#288).

A full fine-tune captures its whole step once and replays it. The embedding backward grouped the
step's token positions by token on the host and copied the groups to the GPU, so the capture kept the
capture step's groups: every later step added each position's gradient to the row of the token that
sat there in the capture step. With two micro-steps both read the groups of the last one captured.

Each case compares a graphed run with an eager one: they run the same kernels in the same order, so
every exported tensor matches. A misplaced embedding gradient shows up in the embedding alone.
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
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from tests import test_onboarding_qwen3_5 as onboarding

    snapshot = onboarding.resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3.5-0.8B not available (set QWEN3_5_MODEL_PATH)")
    return onboarding.prepare_mini_model(snapshot)


def _build(model_dir, graphs: bool, grad_accum: int):
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    opts = _surogate.RuntimeOptions(
        recipe="bf16",
        use_cuda_graphs=graphs,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        offload_residual=False,
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


def _rows(seed: int, rows: int):
    rng = np.random.default_rng(seed)
    x = rng.integers(10, 8000, size=(rows, SEQ), dtype=np.int32)
    y = np.concatenate([x[:, 1:], np.full((rows, 1), -100, np.int32)], axis=1).astype(np.int32)
    pos = np.tile(np.arange(SEQ, dtype=np.int32), (rows, 1))
    return x, y, pos


def _train(model_dir, graphs: bool, grad_accum: int, steps: int, out):
    trainer = _build(model_dir, graphs, grad_accum)
    config = _surogate.OptimizerConfig(optimizer="adamw_8bit", learning_rate=LR, grad_clip=0.0)
    for step in range(steps):
        trainer.train_step_graphed(*_rows(step, grad_accum), config, step + 1)
    trainer.export_model(str(out))
    del trainer
    torch.cuda.empty_cache()


def _assert_same_model(path_a, path_b):
    from safetensors.torch import load_file

    a_all = load_file(str(path_a / "model.safetensors"))
    b_all = load_file(str(path_b / "model.safetensors"))
    assert a_all.keys() == b_all.keys()
    for name, tensor in a_all.items():
        a, b = tensor.float(), b_all[name].float()
        far = ~torch.isclose(a, b, rtol=2**-7, atol=1e-6)
        assert int(far.sum()) <= max(2, 1e-3 * far.numel()), (name, int(far.sum()), far.numel())


@pytest.mark.parametrize(
    "grad_accum,steps",
    [(1, 2), (2, 1)],
    ids=["step-after-capture", "two-micro-steps"],
)
def test_graphed_step_computes_the_eager_embedding_gradient(model_dir, grad_accum, steps, tmp_path):
    _train(model_dir, graphs=False, grad_accum=grad_accum, steps=steps, out=tmp_path / "eager")
    _train(model_dir, graphs=True, grad_accum=grad_accum, steps=steps, out=tmp_path / "graphed")
    _assert_same_model(tmp_path / "eager", tmp_path / "graphed")
