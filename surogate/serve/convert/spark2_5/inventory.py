"""Spark serving inventory, with every dimension supplied by config.json."""

from dataclasses import dataclass, field

from surogate.serve.convert.common import declaration
from surogate.serve.convert.common.inventory import (
    BF16,
    W8,
    ResourceSpec,
    TensorSpec,
    tensor_spec,
)

MODEL_ID = TARGET_KEY = "spark2_5"
WEIGHTS_ID = "groupwise-int"
RESOURCE_SPECS = tuple(
    ResourceSpec("frontend/" + name)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    )
)


@dataclass(frozen=True, slots=True)
class Geometry:
    hidden: int
    layers: int
    intermediate: int
    vocab: int
    query_heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    sliding_rotary_dim: int
    rope_theta: float
    sliding_rope_theta: float
    sliding_window: int
    max_context: int
    rms_epsilon: float
    layer_types: tuple[str, ...]
    declared: declaration.Declaration = field(repr=False, compare=False)
    tied_output_head: bool

    @property
    def query_size(self):
        return self.query_heads * self.head_dim

    @property
    def kv_size(self):
        return self.kv_heads * self.head_dim


def build_tensor_specs(g: Geometry) -> tuple[TensorSpec, ...]:
    return tuple(tensor_spec(obj.name, obj.shape, {"bf16": BF16, "w8": W8}[obj.format]) for obj in g.declared.objects())


def build_object_specs(g: Geometry):
    return RESOURCE_SPECS + build_tensor_specs(g)
