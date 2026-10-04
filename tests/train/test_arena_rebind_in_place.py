"""The weight-manager, gradient and LoRA slabs are bound in place, not copied (#232).

#227 freed the parameters' own storage before the persistent arena was allocated. The weight manager
(ZeRO-3, offloaded or FP32 masters), the full fine-tune gradients and the LoRA adapters still copied
into their arena and freed afterwards, so building a trainer briefly held each of those slabs twice.
Each now frees before the arena exists and binds every tensor to its slot. A tensor bound to the
wrong slot would change the step, so each configuration's first step is compared with the plain
bf16 one on the same row: the work weights, gradients and (zero-initialised) adapters are the same.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

try:
    import surogate._surogate  # noqa: F401
except ImportError:
    pytest.skip("surogate._surogate C++ extension not built", allow_module_level=True)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

SEQ = 256

_STEP = """
import json
import sys

import numpy as np
import surogate._surogate as _surogate
from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.utils.hf import get_model_weights_path

model_dir, cfg = sys.argv[1], json.loads(sys.argv[2])
opts = _surogate.RuntimeOptions(recipe="bf16", use_cuda_graphs=False, offload_master=cfg.get("offload_master", False),
                                offload_grads=False, offload_optimizer=False, offload_residual=False,
                                master_dtype=cfg.get("master_dtype", ""))
opts.dsl_ir_json = build_dsl_ir_for_model(model_dir)
lora = _surogate.LoRAAdapterConfig(rank=8) if cfg.get("lora") else None
trainer = _surogate.SurogateTrainer(ngpu=1, config=_surogate.PretrainedConfig.from_pretrained(model_dir, "bf16"),
                                    options=opts, batch_size=1, seq_len=SEQ, grad_accum=1, memcpy_all_gather=True,
                                    memcpy_send_recv=True, lora_config=lora, qlora_config=None)
trainer.import_weights(get_model_weights_path(model_dir))
x = np.random.default_rng(0).integers(10, 8000, size=(1, SEQ), dtype=np.int32)
y = np.concatenate([x[:, 1:], np.full((1, 1), -100, np.int32)], axis=1).astype(np.int32)
pos = np.arange(SEQ, dtype=np.int32)[None, :].copy()
config = _surogate.OptimizerConfig(optimizer="adamw_8bit", learning_rate=0.0, grad_clip=0.0)
result = trainer.train_step_graphed(x, y, pos, config, 0)
print("STEP", json.dumps({"loss": float(result["loss"]), "norm": float(result["norm"])}))
""".replace("SEQ", str(SEQ))


@pytest.fixture(scope="module")
def model_dir():
    from tests import test_onboarding_qwen3 as onboarding

    snapshot = onboarding.resolve_model_path()
    if snapshot is None:
        pytest.skip("Qwen3-0.6B not available (set QWEN3_MODEL_PATH)")
    return Path(onboarding.prepare_mini_model(snapshot)).resolve()


def _step(model_dir: Path, **cfg) -> tuple[dict, dict]:
    """One step in a fresh process; its result and each owner's SUROGATE_DEBUG_ARENA_CONSUME counts."""
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", _STEP, str(model_dir), json.dumps(cfg)],
        capture_output=True,
        text=True,
        timeout=900,
        env=dict(os.environ, SUROGATE_DEBUG_ARENA_CONSUME="1"),
        cwd=str(repo),
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]
    step = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("STEP "))[5:])
    counts = {}
    for line in result.stderr.splitlines():
        match = re.match(r"\[arena-consume (\S+)\] (.*)", line)
        if match:
            counts[match.group(1)] = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", match.group(2))}
    return step, counts


def _bound_in_place(counts: dict, owner: str) -> None:
    moved = counts[owner]
    assert moved["bound_in_place"] > 0, (owner, moved)
    assert all(value == 0 for key, value in moved.items() if key.startswith("rebound")), (owner, moved)


@pytest.fixture(scope="module")
def reference(model_dir):
    step, counts = _step(model_dir)
    _bound_in_place(counts, "accumulator")
    return step


@pytest.mark.parametrize(
    "cfg, owner",
    [
        ({"master_dtype": "fp32"}, "dsl_wm_persistent"),
        ({"offload_master": True}, "dsl_wm_persistent"),
        ({"lora": True}, "lora_persistent"),
    ],
    ids=["fp32_masters", "offload_master", "lora"],
)
def test_slabs_bind_in_place_and_step_unchanged(model_dir, reference, cfg, owner):
    step, counts = _step(model_dir, **cfg)
    _bound_in_place(counts, owner)
    assert abs(step["loss"] - reference["loss"]) <= 1e-3 * abs(reference["loss"]), (step, reference)
    if not cfg.get("lora"):  # a LoRA step's norm is the adapters'
        _bound_in_place(counts, "accumulator")
        assert abs(step["norm"] - reference["norm"]) <= 2e-2 * reference["norm"], (step, reference)
