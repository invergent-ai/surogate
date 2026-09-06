"""The GLM-5.3-Flash converter's one value transform.

llama.cpp stores the KDA decay as `ssm_a = -exp(A_log)` -- its own graph multiplies by the
tensor as it lies -- while the engine's gate is written in terms of `A_log`, as the checkpoint
is. Bound as stored, the gate ran at `exp(-exp(A_log))`: every linear-attention state forgot
its past at one rate whatever the token said, and a NumPy reference written from the same
reading agreed with the engine at every probe. What caught it was llama.cpp's graph for the
architecture, which is the reading these tests pin.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from surogate.serve.convert.common.gguf_source import FP32
from surogate.serve.convert.glm5_next import convert as glm_convert
from surogate.serve.convert.glm5_next import recipe as rcp


class _Store:
    """A GGUF source that answers `float32` from a fixed array and records what was asked."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.asked: list[str] = []

    def float32(self, name: str) -> np.ndarray:
        self.asked.append(name)
        return self.values.copy()


def _spec(count: int) -> SimpleNamespace:
    return SimpleNamespace(name="text/layers/4/kda/a_log", format=FP32, shape=(count,))


def test_a_log_is_the_log_of_the_negated_store():
    store = _Store([-4.7707, -7.1018, -3.3352, -1.4618])
    payload = glm_convert.materialize_unspoken(store, "text/layers/4/kda/a_log", _spec(4))
    assert store.asked == ["blk.4.ssm_a"]
    got = np.frombuffer(payload, dtype=np.float32)
    np.testing.assert_allclose(got, np.log(-store.values), rtol=1e-6)
    # The gate multiplies by exp(A_log): the round trip must give back the stored magnitude.
    np.testing.assert_allclose(np.exp(got), -store.values, rtol=1e-6)


def test_a_store_that_is_not_a_negated_exponential_is_refused():
    store = _Store([-4.77, 0.5, -1.46])
    with pytest.raises(ValueError, match="ssm_a"):
        glm_convert.materialize_unspoken(store, "text/layers/4/kda/a_log", _spec(3))


def test_nothing_else_is_materialised_here():
    with pytest.raises(KeyError):
        glm_convert.materialize_unspoken(_Store([-1.0]), "text/layers/4/kda/decay_bias", _spec(1))


def test_only_the_recurrent_layers_carry_a_decay():
    geometry = SimpleNamespace(layers=8, is_attention=lambda layer: layer % 4 == 3)
    assert rcp.materialized_objects(geometry) == frozenset(
        f"text/layers/{layer}/kda/a_log" for layer in (0, 1, 2, 4, 5, 6)
    )
