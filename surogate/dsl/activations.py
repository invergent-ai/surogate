"""Activation registry for the Python DSL.

Immutable, enum-like namespace of frozen ``ActivationSpec`` constants.
HF-config strings are resolved once at the boundary via
``activation_from_name(...)``; downstream code circulates typed specs.

See ``MIGRATION.md`` Phase 8.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Mapping


@dataclass(frozen=True, kw_only=True)
class ActivationSpec:
    """Describes an activation for the DSL -> C++ pipeline.

    ``cpp_op`` / ``cpp_backward_op`` reference the op names registered in
    ``csrc/src/runtime/executor/op_registrations.cpp``. When ``gated`` is
    true, the activation folds a ``[gate | up]`` concatenated tensor into a
    single fused kernel named by ``gated_cpp_op``.
    """

    name: str
    cpp_op: str
    cpp_backward_op: str
    gated: bool = False
    gated_cpp_op: str = ""
    attrs: Mapping[str, Any] = field(default_factory=dict)


class Activation:
    """Known activations. Immutable, IDE-discoverable."""

    SILU: ClassVar[ActivationSpec] = ActivationSpec(
        name="silu",
        cpp_op="silu",
        cpp_backward_op="silu_backward",
        gated=True,
        gated_cpp_op="swiglu",
    )
    GELU: ClassVar[ActivationSpec] = ActivationSpec(
        name="gelu",
        cpp_op="gelu",
        cpp_backward_op="gelu_backward",
    )
    RELU2: ClassVar[ActivationSpec] = ActivationSpec(
        name="relu2",
        cpp_op="relu2",
        cpp_backward_op="relu2_backward",
    )
    GPT_OSS: ClassVar[ActivationSpec] = ActivationSpec(
        name="gpt_oss_act",
        cpp_op="gpt_oss_moe_act",
        cpp_backward_op="gpt_oss_moe_act_backward",
        gated=True,
        gated_cpp_op="gpt_oss_moe_act",
        attrs={"alpha": 1.702, "limit": 7.0},
    )


_BY_NAME: Mapping[str, ActivationSpec] = MappingProxyType(
    {a.name: a for a in vars(Activation).values() if isinstance(a, ActivationSpec)}
)


def activation_from_name(name: str) -> ActivationSpec:
    """Resolve an HF-config activation string to a spec.

    HF configs use ``hidden_act`` values like ``"silu"`` / ``"gelu"``; this
    helper is the single boundary call that converts those strings into a
    typed ``ActivationSpec``.
    """
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"Unknown activation '{name}'. Known: {sorted(_BY_NAME)}")
    return spec


__all__ = [
    "Activation",
    "ActivationSpec",
    "activation_from_name",
]
