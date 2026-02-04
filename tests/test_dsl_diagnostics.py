import json

import pytest

from surogate.dsl import (
    Array,
    Param,
    Tensor,
    block,
    compile_model,
    forward,
    graph,
    model,
    save,
)
from surogate.dsl.errors import DSLError, ErrorCode


def _get_first_error_code(payload: str) -> str:
    doc = json.loads(payload)
    assert doc["success"] is False
    assert "errors" in doc and doc["errors"]
    return doc["errors"][0]["code"]


def _get_warning_codes(payload: str) -> set[str]:
    doc = json.loads(payload)
    warnings = doc.get("warnings", [])
    return {w["code"] for w in warnings}


def test_compile_model_unknown_name_returns_e002():
    payload = compile_model("DefinitelyNotARealModel", {}, raise_on_error=False)
    assert _get_first_error_code(payload) == "E002"


def test_compile_model_unknown_name_raises_when_raise_on_error():
    with pytest.raises(DSLError) as excinfo:
        compile_model("DefinitelyNotARealModel", {}, raise_on_error=True)
    assert excinfo.value.code == ErrorCode.E002


def test_compile_model_missing_forward_emits_e012():
    @model
    class NoForwardModel:
        def __init__(self):
            pass

    payload = compile_model(NoForwardModel, {}, raise_on_error=False)
    assert _get_first_error_code(payload) == "E012"


def test_stackedblocks_missing_n_layers_emits_e012():
    @block
    class DummyBlock:
        @forward
        def forward(
            self,
            x: Tensor["B", "T", "C"],
            residual: Tensor["B", "T", "C"],
            position_ids: Tensor["T", "int32"],
        ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
            with graph() as g:
                x2 = g.copy(x)
                r2 = g.copy(residual)
                return x2, r2

    @model
    class MissingNLayersModel:
        blocks = Param(Array["n_layers", "DummyBlock"])

        @forward
        def forward(
            self,
            x: Tensor["B", "T", "C"],
            residual: Tensor["B", "T", "C"],
            position_ids: Tensor["T", "int32"],
        ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
            with graph() as g:
                # Intentionally omit n_layers
                return g.call(
                    "StackedBlocks",
                    x,
                    residual,
                    position_ids,
                    num_outputs=2,
                    blocks="blocks",
                )

    payload = compile_model(MissingNLayersModel, {}, raise_on_error=False)
    assert _get_first_error_code(payload) == "E012"


def test_warning_w001_shadows_primitive():
    @model
    class matmul:  # intentionally shadows the primitive name "matmul"
        @forward
        def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
            with graph() as g:
                return g.copy(x)

    payload = compile_model(matmul, {}, raise_on_error=False)
    assert "W001" in _get_warning_codes(payload)


def test_warning_w004_unused_saved_tensor():
    @model
    class SavedTensorNotInGraphModel:
        @save("does_not_exist")
        @forward
        def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
            with graph() as g:
                return g.copy(x)

    payload = compile_model(SavedTensorNotInGraphModel, {}, raise_on_error=False)
    assert "W004" in _get_warning_codes(payload)
