"""Shared-expert weights must not be FP8/FP4-quantized.

The C++ matmul dispatcher (csrc/src/runtime/ops/matmul.cpp:253, guarded by
``is_shared_expert_weight_name``) deliberately forces shared-expert matmuls down
the BF16 path — they never go through the FP8/FP4 recipe. So the shared-expert
*weights* must stay full precision; if the import-time quantizer turns them into
FP8 (because ``quantizable=True``), the BF16 fallback kernel gets a quantized
weight and crashes:

    dispatch_matmul ... weight='blocks[0].shared_expert_up' used_recipe=0
    DType mismatch (class): expected BF16, got F8_E4M3

This mirrors how router weights are already kept full precision
(dsl_model.cpp:81). Fast, GPU-free: traces the module and inspects param specs.
"""

from __future__ import annotations

import pytest

from surogate.dsl.nn import Proxy, Tracer, _current_tracer
from surogate.dsl.modules.moe import MoESharedExpert, NemotronSharedExpert


def _shared_expert_params(mod):
    tracer = Tracer()
    token = _current_tracer.set(tracer)
    try:
        x = Proxy("x", tracer.graph.input("x"))
        mod(x)
    finally:
        _current_tracer.reset(token)
    return tracer.params


@pytest.mark.parametrize(
    "mod",
    [
        MoESharedExpert(d_model=2048, shared_expert_intermediate=512),
        NemotronSharedExpert(d_model=2048, shared_expert_intermediate=512),
    ],
    ids=["MoESharedExpert", "NemotronSharedExpert"],
)
def test_shared_expert_weights_not_quantizable(mod):
    params = _shared_expert_params(mod)
    assert params, "expected the shared expert to register weight params"

    quantizable = [name for name, spec in params.items() if spec.quantizable]
    assert not quantizable, (
        "shared-expert weights are forced to BF16 by the matmul dispatcher, so "
        "they must be registered quantizable=False (else fp8-hybrid import "
        f"quantizes them and the BF16 matmul crashes): {quantizable}"
    )
