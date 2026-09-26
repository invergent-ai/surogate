"""The trainer's LoRA getters name every adapter tensor as the exported adapter does (#220).

get_lora_weights feeds GRPO's adapter publication and get_lora_gradients the gradient diagnostics. Both
spelled the module path themselves, and drifted from the export: `self_attn.q_proj/k_proj/v_proj/o_proj`
where a native fused-QKV model (Spark 2.5) exports `self_attn.q_k_v_proj` and `self_attn.out_proj`, the plain
layer path where GLM-5-Next exports under `language_model`, and no fused `gate_up_proj`, shared-expert or
router adapters at all. Grouped MoE experts are the one intended difference: the getters hand out the
[experts, ...] tensor, the export writes one tensor per expert.
"""

import re

import numpy as np
import pytest
import torch
from safetensors import safe_open

from tests.grpo.test_batched_decode_gpu import make_trainer

pytestmark = [pytest.mark.gpu, pytest.mark.slow,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

EXPERT = re.compile(r"\.experts\.(\d+\.)?")


def split(names):
    """(everything but the experts, the expert projections with their index dropped)."""
    plain = {n for n in names if ".experts." not in n}
    experts = {EXPERT.sub(".experts.", n) for n in names if ".experts." in n}
    return plain, experts


@pytest.mark.parametrize("case", ["llama", "lfm2", "spark", "qwen3_5", "qwen3_5_moe", "gpt_oss", "glm"])
def test_getters_name_tensors_as_the_export_does(tmp_path, case):
    model = tmp_path / "model"
    model.mkdir()
    trainer = make_trainer(model, case, sequence=128)
    trainer.export_adapter(str(tmp_path / "adapter"))
    with safe_open(str(tmp_path / "adapter" / "adapter_model.safetensors"), "pt") as f:
        exported = split(f.keys())

    assert split(trainer.get_lora_weights(0).keys()) == exported

    rng = np.random.default_rng(3)
    inputs = rng.integers(3, 97, size=(2, 128), dtype=np.int32)
    trainer.step_with_custom_loss(inputs, np.roll(inputs, -1, axis=1),
                                  rng.normal(0, 0.01, size=inputs.shape).astype(np.float32))
    assert split(trainer.get_lora_gradients(0).keys()) == exported
    if case == "spark":
        assert {".self_attn.q_k_v_proj.", ".self_attn.out_proj."} <= {
            m[0] for n in exported[0] if (m := re.search(r"\.self_attn\.\w+\.", n))}
