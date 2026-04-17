"""Quantization strategy registry for the Python DSL.

Mirrors HF's ``HfQuantizer`` lifecycle-hook pattern on the DSL side. Each
strategy knows which params to quantize, what HF weight suffixes to
look for (for pre-quantized safetensors), and what hardware it requires.
This is the DSL counterpart of the C++ ``IQuantizer`` in
``csrc/src/runtime/qlora/``.

See ``MIGRATION.md`` Phase 8.7.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Mapping

if TYPE_CHECKING:
    from .specs import ParamSpec


class QuantizationStrategy(ABC):
    """Base class for DSL quantization strategies.

    Subclasses answer three questions: which params need quantization,
    which HF weight-file suffixes carry the pre-quantized state, and
    whether the current hardware supports the format.
    """

    @abstractmethod
    def param_needs_quantization(self, param_name: str, param_spec: "ParamSpec") -> bool:
        """Return True if ``param`` should be quantized.

        Called during graph compilation to decide which weights get
        routed through the QLoRA pipeline.
        """

    @abstractmethod
    def weight_suffixes(self) -> Mapping[str, str]:
        """HF weight-file suffixes for pre-quantized models.

        Keys are the logical roles (``"data"``, ``"scales"``,
        ``"absmax"``, ``"quant_state"``); values are the suffix
        appended to the base parameter name in the safetensors index.
        """

    def validate_hardware(self, sm_version: int) -> None:
        """Raise if the current GPU can't run this format. Default: no-op."""
        return None


class BnBNF4Strategy(QuantizationStrategy):
    """BitsAndBytes NF4. Works on any CUDA-capable GPU."""

    def param_needs_quantization(self, param_name, param_spec):
        return param_spec.quantizable and "norm" not in param_name

    def weight_suffixes(self):
        return {
            "data": ".weight",
            "absmax": ".absmax",
            "quant_state": ".quant_state",
        }


class FP8Strategy(QuantizationStrategy):
    """FP8 E4M3 per-block scales. Requires Ada/Hopper (SM89+)."""

    def param_needs_quantization(self, param_name, param_spec):
        return param_spec.quantizable

    def weight_suffixes(self):
        return {"data": ".weight", "scales": ".weight_scale_inv"}

    def validate_hardware(self, sm_version: int) -> None:
        if sm_version < 89:
            raise RuntimeError(f"FP8 quantization requires SM89+ (Ada/Hopper); got SM{sm_version}")


class FP4Strategy(QuantizationStrategy):
    """NVFP4 E2M1 with 2D block scales. Requires Blackwell (SM100+)."""

    def param_needs_quantization(self, param_name, param_spec):
        return param_spec.quantizable

    def weight_suffixes(self):
        return {"data": ".weight", "scales": ".weight_scale"}

    def validate_hardware(self, sm_version: int) -> None:
        if sm_version < 100:
            raise RuntimeError(f"NVFP4 quantization requires SM100+ (Blackwell); got SM{sm_version}")


class MXFP4Strategy(QuantizationStrategy):
    """MXFP4 with E8M0 shared exponents — used for HF pre-quantized
    GPT-OSS checkpoints."""

    def param_needs_quantization(self, param_name, param_spec):
        return param_spec.quantizable

    def weight_suffixes(self):
        return {"data": ".blocks", "scales": ".scales"}


@dataclass(frozen=True, kw_only=True)
class QuantSpec:
    """Quantization kind = identity + strategy class."""

    name: str
    strategy: type[QuantizationStrategy]


class Quant:
    """Known quantization kinds. Immutable, IDE-discoverable."""

    BNB_NF4: ClassVar[QuantSpec] = QuantSpec(name="bnb_nf4", strategy=BnBNF4Strategy)
    FP8: ClassVar[QuantSpec] = QuantSpec(name="fp8", strategy=FP8Strategy)
    FP4: ClassVar[QuantSpec] = QuantSpec(name="fp4", strategy=FP4Strategy)
    MXFP4: ClassVar[QuantSpec] = QuantSpec(name="mxfp4", strategy=MXFP4Strategy)


_BY_NAME: Mapping[str, QuantSpec] = MappingProxyType(
    {q.name: q for q in vars(Quant).values() if isinstance(q, QuantSpec)}
)


def quant_from_name(name: str) -> QuantSpec:
    """Resolve a quantization kind string to a spec."""
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"Unknown quantization '{name}'. Known: {sorted(_BY_NAME)}")
    return spec


__all__ = [
    "BnBNF4Strategy",
    "FP4Strategy",
    "FP8Strategy",
    "MXFP4Strategy",
    "Quant",
    "QuantSpec",
    "QuantizationStrategy",
    "quant_from_name",
]
