"""GRPO's fp32-gradient default applies to adapter runs only.

The default exists to keep a small adapter's gradients precise while the master
weights stay BF16. With ``lora: false`` the same line allocates fp32 gradients
for every weight against that BF16 master, which the backward rejects with
``matmul_backward: weight-grad output tensor shape/dtype mismatch``.

These read the mutated config dict rather than a finished object: the gate runs
on the incoming mapping before ``super().__init__``, so it is observable without
a model checkout on disk. Construction is expected to fail afterwards for
unrelated reasons (no model), which is why it is swallowed.
"""

import pytest

from surogate.grpo.config import GRPOTrainConfig
from surogate.utils.dict import DictDefault


def _gradient_dtype_for(**cfg) -> str | None:
    d = DictDefault({"model": "local-test-model", "output_dir": "/tmp/o", **cfg})
    try:
        GRPOTrainConfig(d)
    except Exception:
        pass  # model loading, not the gate under test
    return d.get("gradient_dtype")


@pytest.mark.parametrize("lora", [True, None])
def test_an_adapter_run_still_gets_fp32_gradients(lora):
    """``None`` means the key was absent; the engine defaults ``lora`` to true."""
    cfg = {} if lora is None else {"lora": lora}
    assert _gradient_dtype_for(**cfg) == "fp32"


def test_a_full_finetune_inherits_the_model_dtype():
    assert _gradient_dtype_for(lora=False) is None


def test_an_explicit_choice_is_never_overridden():
    assert _gradient_dtype_for(lora=True, gradient_dtype="bf16") == "bf16"
    assert _gradient_dtype_for(lora=False, gradient_dtype="fp32") == "fp32"
