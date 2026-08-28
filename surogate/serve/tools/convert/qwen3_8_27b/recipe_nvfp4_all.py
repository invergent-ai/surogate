"""Single-source recipe for the all-NVFP4 Qwen3.8-27B artifact.

`sakamakismile/Qwen3.8-27B-MTP-NVFP4` is a compressed-tensors export whose
ignore list holds only the vision tower, so every language linear ships as
packed E2M1 codes with per-16 E4M3 block scales, a weight global scale and a
calibrated activation global scale - the same field names and the same
divisor convention `recipe_nvfp4` already reads. It also carries the bf16
embedding, lm_head, MTP block and vision tower, so one checkpoint supplies the
whole artifact and there is no separate official source to align against.

Two matrices need work on the way in: `in_proj_a` / `in_proj_b` are quantised
here but the artifact stores the fused control projection dense, so they are
decoded back to BF16.
"""

from __future__ import annotations

import torch

from surogate.serve.tools.convert.common.safetensors import ShardReader
from surogate.serve.tools.convert.qwen3_6.common import recipe as family_recipe
from surogate.serve.tools.convert.qwen3_6_27b import recipe as official_recipe

from . import inventory_nvfp4_all as inventory
from .recipe_nvfp4 import (
    InputDivisorRecipe,
    MatrixPart,
    MatrixSource,
    Nvfp4WeightRecipe,
    RowRange,
    _all,
    _q_part,
    _source,
    materialize_input_divisor,
    materialize_nvfp4_weight,
)

QUANTIZED_REPOSITORY = "sakamakismile/Qwen3.8-27B-MTP-NVFP4"
QUANTIZED_REVISION = "6d98dc1f1d5259c9582794014b73852baf20f805"
BASE_REPOSITORY = QUANTIZED_REPOSITORY
BASE_REVISION = QUANTIZED_REVISION

WEIGHT_FIELD = "weight_packed"
SCALE_FIELD = "weight_scale"
GLOBAL_SCALE_FIELD = "weight_global_scale"
INPUT_SCALE_FIELD = "input_global_scale"

# E2M1 code -> value, for the two control projections the artifact wants dense.
_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _build_matrix_recipes() -> tuple[
    tuple[Nvfp4WeightRecipe, ...], tuple[InputDivisorRecipe, ...]
]:
    weights: list[Nvfp4WeightRecipe] = []
    divisors: list[InputDivisorRecipe] = []

    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"

        if layer in inventory.FULL_ATTENTION_LAYERS:
            query = _source(source_prefix + "self_attn.q_proj", 12288, 5120)
            key = _source(source_prefix + "self_attn.k_proj", 1024, 5120)
            value = _source(source_prefix + "self_attn.v_proj", 1024, 5120)
            output = _source(source_prefix + "self_attn.o_proj", 5120, 6144)
            fused = (query, key, value)
            weights.extend(
                (
                    Nvfp4WeightRecipe(
                        object_prefix + "attention/query_key_gate_value",
                        (14336, 5120),
                        (
                            _q_part(query, False),
                            _all(key),
                            _q_part(query, True),
                            _all(value),
                        ),
                        fused,
                    ),
                    Nvfp4WeightRecipe(
                        object_prefix + "attention/output",
                        output.shape,
                        (_all(output),),
                        (output,),
                    ),
                )
            )
            divisors.extend(
                (
                    InputDivisorRecipe(
                        object_prefix
                        + "attention/input_projection/input_scale_divisor",
                        fused,
                        (object_prefix + "attention/query_key_gate_value",),
                    ),
                    InputDivisorRecipe(
                        object_prefix
                        + "attention/output_projection/input_scale_divisor",
                        (output,),
                        (object_prefix + "attention/output",),
                    ),
                )
            )
        else:
            query_key_value = _source(
                source_prefix + "linear_attn.in_proj_qkv", 10240, 5120
            )
            z = _source(source_prefix + "linear_attn.in_proj_z", 6144, 5120)
            output = _source(source_prefix + "linear_attn.out_proj", 5120, 6144)
            weights.extend(
                (
                    Nvfp4WeightRecipe(
                        object_prefix + "gdn/query_key_value",
                        query_key_value.shape,
                        (_all(query_key_value),),
                        (query_key_value,),
                    ),
                    Nvfp4WeightRecipe(
                        object_prefix + "gdn/z", z.shape, (_all(z),), (z,)
                    ),
                    Nvfp4WeightRecipe(
                        object_prefix + "gdn/output",
                        output.shape,
                        (_all(output),),
                        (output,),
                    ),
                )
            )
            divisors.extend(
                (
                    InputDivisorRecipe(
                        object_prefix + "gdn/input_projection/input_scale_divisor",
                        (query_key_value,),
                        (object_prefix + "gdn/query_key_value",),
                    ),
                    InputDivisorRecipe(
                        object_prefix + "gdn/z_projection/input_scale_divisor",
                        (z,),
                        (object_prefix + "gdn/z",),
                    ),
                    InputDivisorRecipe(
                        object_prefix + "gdn/output_projection/input_scale_divisor",
                        (output,),
                        (object_prefix + "gdn/output",),
                    ),
                )
            )

        gate = _source(source_prefix + "mlp.gate_proj", 17408, 5120)
        up = _source(source_prefix + "mlp.up_proj", 17408, 5120)
        down = _source(source_prefix + "mlp.down_proj", 5120, 17408)
        weights.extend(
            (
                Nvfp4WeightRecipe(
                    object_prefix + "mlp/gate_up",
                    (34816, 5120),
                    (_all(gate), _all(up)),
                    (gate, up),
                ),
                Nvfp4WeightRecipe(
                    object_prefix + "mlp/down", down.shape, (_all(down),), (down,)
                ),
            )
        )
        divisors.extend(
            (
                InputDivisorRecipe(
                    object_prefix + "mlp/gate_up_projection/input_scale_divisor",
                    (gate, up),
                    (object_prefix + "mlp/gate_up",),
                ),
                InputDivisorRecipe(
                    object_prefix + "mlp/down_projection/input_scale_divisor",
                    (down,),
                    (object_prefix + "mlp/down",),
                ),
            )
        )

    return tuple(weights), tuple(divisors)


NVFP4_WEIGHT_RECIPES, INPUT_DIVISOR_RECIPES = _build_matrix_recipes()
NVFP4_WEIGHTS_BY_NAME = {item.object_name: item for item in NVFP4_WEIGHT_RECIPES}
INPUT_DIVISORS_BY_NAME = {item.object_name: item for item in INPUT_DIVISOR_RECIPES}

# Norms, a_log, dt_bias and the convolution are the mixed contract's objects, unchanged:
# this export ships them the same way. Only the fused control projection differs, because
# in_proj_a/in_proj_b arrive quantised here.
DEQUANTIZED_CONTROL_LAYERS = inventory.GDN_LAYERS

OFFICIAL_TENSOR_SPECS = tuple(
    spec
    for spec in inventory.TENSOR_SPECS
    if spec.name == "text/token_embedding"
    or spec.name == "text/output_head"
    or spec.name.startswith("text/draft_head")
    or spec.name.startswith("mtp/")
    or spec.name.startswith("vision/")
)


def decode_nvfp4_dense(source_name: str, shape: tuple[int, ...], reader: ShardReader) -> torch.Tensor:
    """in_proj_a / in_proj_b ship quantised; the artifact wants them dense."""

    packed = reader.get(f"{source_name}.{WEIGHT_FIELD}")
    scales = reader.get(f"{source_name}.{SCALE_FIELD}")
    divisor = float(
        reader.get(f"{source_name}.{GLOBAL_SCALE_FIELD}").reshape(()).to(torch.float32)
    )
    if not divisor > 0.0:
        raise ValueError(f"{source_name}: weight global scale must be positive")
    rows, half = packed.shape
    low = (packed & 0x0F).to(torch.long)
    high = (packed >> 4).to(torch.long)
    values = torch.empty(rows, half * 2, dtype=torch.float32)
    values[:, 0::2] = _E2M1_VALUES[low]
    values[:, 1::2] = _E2M1_VALUES[high]
    block_scales = scales.to(torch.float32).repeat_interleave(16, dim=1)
    dense = values * block_scales / divisor
    if tuple(dense.shape) != tuple(shape):
        raise ValueError(
            f"{source_name}: dequantised shape {tuple(dense.shape)} != {tuple(shape)}"
        )
    return dense.to(torch.bfloat16)


def materialize_control_projection(layer: int, reader: ShardReader) -> torch.Tensor:
    prefix = f"model.language_model.layers.{layer}.linear_attn."
    a = decode_nvfp4_dense(prefix + "in_proj_a", (48, 5120), reader)
    b = decode_nvfp4_dense(prefix + "in_proj_b", (48, 5120), reader)
    return torch.cat((a, b), dim=0)


# The GDN input projection fuses in_proj_qkv and in_proj_z into one 16384-row object, and an
# NVFP4 object carries a single weight divisor. This export gives the two sources *different*
# weight global scales (layer 0: 6176 against 11264), so no one divisor states both exactly.
# Restating one half onto the other's divisor was measured at ~2 % mean relative error with
# 93 % of its values moving - a second quantisation of every GDN gate weight, which is not a
# trade to make silently. Splitting the object into two NVFP4 weights is the way through, and
# it needs an NVFP4 qkv|z path in gdn_input_proj and its conv snapshot/record twins (today's
# SplitGdnInputProjectionPayload is the Q4/Q5 qk|value_z split, a different arrangement).
FUSED_GDN_DIVISOR_BLOCKER = (
    "in_proj_qkv and in_proj_z carry different weight_global_scale words in this export, so "
    "the fused gdn/query_key_value_z object cannot state both halves exactly. See the note in "
    "recipe_nvfp4_all.py: the artifact needs the projection split into two NVFP4 objects."
)


def validate_recipe() -> None:
    """Every NVFP4 object in the inventory has exactly one recipe, and a divisor."""

    expected = {
        spec.name for spec in inventory.TENSOR_SPECS if spec.format == inventory.NVFP4
    }
    produced = set(NVFP4_WEIGHTS_BY_NAME)
    if expected != produced:
        missing = sorted(expected - produced)[:4]
        extra = sorted(produced - expected)[:4]
        raise ValueError(
            f"all-NVFP4 recipe coverage mismatch; missing={missing} extra={extra}"
        )
    divisor_names = {spec.name for spec in inventory.INPUT_SCALE_DIVISOR_SPECS}
    if divisor_names != set(INPUT_DIVISORS_BY_NAME):
        raise ValueError("activation divisor coverage does not match the inventory")
    for item in NVFP4_WEIGHT_RECIPES:
        rows = sum(part.output_rows for part in item.parts)
        if rows != item.shape[0]:
            raise ValueError(
                f"{item.object_name}: parts cover {rows} rows, shape wants {item.shape[0]}"
            )


__all__ = [
    "BASE_REPOSITORY",
    "BASE_REVISION",
    "GLOBAL_SCALE_FIELD",
    "INPUT_DIVISORS_BY_NAME",
    "INPUT_DIVISOR_RECIPES",
    "INPUT_SCALE_FIELD",
    "NVFP4_WEIGHTS_BY_NAME",
    "NVFP4_WEIGHT_RECIPES",
    "OFFICIAL_TENSOR_SPECS",
    "QUANTIZED_REPOSITORY",
    "QUANTIZED_REVISION",
    "SCALE_FIELD",
    "WEIGHT_FIELD",
    "decode_nvfp4_dense",
    "materialize_control_projection",
    "materialize_input_divisor",
    "materialize_nvfp4_weight",
    "validate_recipe",
]
