"""ZeRO-3 on the NCCL all-gather path computes the single-GPU step.

The onboarding tests all run `memcpy_all_gather=True`, so the default NCCL gather of sharded
weights had no coverage: the in-place slot bug of #226 shipped through it. And under the
full-step CUDA graph a backward that never released its prefetch slots overwrote the layer it
was reading (the ZeRO-3 "loss right, gradient norm 1e7..1e32" divergence). Every rank gets the
same rows, so the averaged gradients (and their norm) equal the single-GPU step's.

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


@pytest.fixture(scope="module")
def model_dir():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    from tests import test_onboarding_qwen3_5 as onboarding

    snapshot = onboarding.resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3.5-0.8B not available (set QWEN3_5_MODEL_PATH)")
    return onboarding.prepare_mini_model(snapshot)


def _step(model_dir, ngpu: int, graphs: bool, zero3: bool) -> tuple[float, float]:
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
        memcpy_all_gather=False,  # the NCCL path
        memcpy_send_recv=False,
        lora_config=None,
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    rng = np.random.default_rng(0)
    x = np.repeat(rng.integers(10, 8000, size=(1, SEQ), dtype=np.int32), ngpu, axis=0)
    y = np.concatenate([x[:, 1:], np.full((ngpu, 1), -100, np.int32)], axis=1).astype(np.int32)
    pos = np.tile(np.arange(SEQ, dtype=np.int32), (ngpu, 1))
    config = _surogate.OptimizerConfig(optimizer="adamw_8bit", learning_rate=0.0, grad_clip=0.0)
    result = dict(trainer.train_step_graphed(x, y, pos, config, 1))
    del trainer
    torch.cuda.empty_cache()
    return float(result["loss"]), float(result["norm"])


@pytest.fixture(scope="module")
def reference(model_dir):
    return _step(model_dir, ngpu=1, graphs=False, zero3=False)


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "cuda_graphs"])
def test_zero3_nccl_matches_single_gpu(model_dir, reference, graphs):
    ref_loss, ref_norm = reference
    loss, norm = _step(model_dir, ngpu=2, graphs=graphs, zero3=True)
    assert np.isfinite(norm)
    assert abs(loss - ref_loss) <= 1e-2 * abs(ref_loss), (loss, ref_loss)
    assert abs(norm - ref_norm) <= 5e-2 * ref_norm, (norm, ref_norm)


_RESHAPE = """
import sys
import numpy as np
import surogate._surogate as _surogate
from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.kernels.jit_compile import compile_jit_kernels
from surogate.utils.hf import get_model_weights_path

model_dir = sys.argv[1]
opts = _surogate.RuntimeOptions(recipe="bf16", use_cuda_graphs=False, offload_master=False, offload_grads=False,
                                offload_optimizer=False, offload_residual=False, shard_weights=True,
                                shard_gradients=True)
opts.recompute = "false"
opts.dsl_ir_json = build_dsl_ir_for_model(model_dir)
manifests = compile_jit_kernels(opts.dsl_ir_json)
if manifests:
    opts.jit_kernel_manifests = manifests
trainer = _surogate.SurogateTrainer(ngpu=2, config=_surogate.PretrainedConfig.from_pretrained(model_dir, "bf16"),
                                    options=opts, batch_size=1, seq_len=%d, grad_accum=1, memcpy_all_gather=False,
                                    memcpy_send_recv=False, lora_config=None, qlora_config=None)
trainer.import_weights(get_model_weights_path(model_dir))
x = np.random.default_rng(0).integers(10, 8000, size=(1, %d), dtype=np.int32)
y = np.roll(x, -1, axis=1).astype(np.int32)
assert np.isfinite(np.asarray(trainer.compute_logprobs(x, y, False, None, None))).all()
try:
    trainer.compute_logprobs(np.ascontiguousarray(x[:, :%d]), np.ascontiguousarray(y[:, :%d]), False, None, None)
except (RuntimeError, ValueError) as error:
    print("REFUSED:", error)
""" % (SEQ, SEQ, SEQ // 2, SEQ // 2)


def test_zero3_pass_at_another_shape_is_refused_not_a_crash(model_dir, tmp_path):
    """A pass at a shape other than the trainer's used to recompile, release the arenas the ZeRO-3 weight
    slabs live in, and rebind from the freed memory: SIGSEGV in cudaMemcpyAsync inside
    DslWeightManager::rebind_to_persistent_arena (#230). It must be refused with an error instead."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = tmp_path / "reshape.py"
    script.write_text(_RESHAPE)
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(script), str(Path(model_dir).resolve())], capture_output=True,
                            text=True, timeout=900, env=dict(os.environ), cwd=str(repo))
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    assert "REFUSED:" in result.stdout and "is not supported" in result.stdout, output[-4000:]
