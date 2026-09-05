"""Hugging Face source recipe for Qwen3.6-35B-A3B."""

from __future__ import annotations

from pathlib import Path

from surogate.serve.convert.common.safetensors import ShardReader
from surogate.serve.convert.common.recipe import (
    Cast,
    Concat,
    DraftHeadTokenIds,
    Expression,
    GatherRows,
    Reshape,
    SOURCE_DTYPE,
    Slice,
    SourcePreflight,
    SourceTensor,
    TensorRecipe,
    Transpose,
    build_vision_recipes,
    expression_shape,
    expression_sources,
    materialize_expression,
    materialize_recipe,
    preflight_source_reader,
    source,
    source_requirements as _common_source_requirements,
    validate_recipe_coverage as _common_validate_recipe_coverage,
)
from surogate.serve.convert.common.declaration import declare, derive_recipes
from surogate.dsl.ir_builder import resolve_architecture

from . import inventory


DRAFT_ROWS = 131072
DRAFT_RANKING_PATH = (
    "freq_corpus/fixtures/ranking/ranking.train.counts.i64"
)


# Objects whose op reads the row-split W8 planes. Their weights are still read from the GGUF --
# Q8_0 and W8G32_F16S hold the same numbers, so the loader rearranges them on the device -- and
# this list is what tells the planner to take that route rather than the native one. Both the
# converter and the ingest bridge read it, because the bridge must keep dequantising exactly what
# the converter still builds itself.
NATIVE_EXCLUDE_SUFFIXES = (
    "attention/query_key_gate_value",
    "gdn/query_key_value_z",
    "gdn/output",
    "moe/shared_gate_up",
    "moe/shared_down",
)


def _build_declared_recipes() -> tuple[TensorRecipe, ...]:
    """The text stack and the MTP head, from the training declaration.

    Not written here: the declaration maps every parameter to its checkpoint tensor
    (`hf_mapping`) and says which parameters each artifact object is built from, in row
    order (`ServeObject.components`). The expert objects carry `transform="flatten_experts"`,
    which is what turns the training graph's expert-major `[E, rows, cols]` parameter into
    the rows the artifact stores — and, where the checkpoint fuses gate and up, also
    accepts the stacked spelling a GGUF of the same model keeps them in.

    The DFlash scorer comes along: it ships as its own checkpoint with its tensors at
    the root and the training graph has no scorer, so its section names those tensors
    directly (`ServeObject.source`) instead of naming parameters that do not exist.

    `flat_sources=False` keeps the source names in the nested `model.language_model.`
    dialect this converter's recipes, its preflight and its GGUF repack plan are all
    written against; the reader canonicalises both spellings, so this is a convention,
    not a constraint. It does not reach the scorer, whose names are already at the root.
    """
    config = inventory.hf_config_for()
    declaration = declare(resolve_architecture(config), config, flat_sources=False)
    return derive_recipes(declaration, capabilities={"text", "dflash"})


def _build_draft_head_recipes() -> tuple[TensorRecipe, ...]:
    return (
        TensorRecipe(
            "text/draft_head",
            GatherRows(
                source("lm_head.weight", (248320, 2048)),
                token_ids_object="text/draft_head_token_ids",
                rows=DRAFT_ROWS,
            ),
        ),
        TensorRecipe(
            "text/draft_head_token_ids",
            DraftHeadTokenIds(
                ranking_path=DRAFT_RANKING_PATH,
                tokenizer_resource="frontend/tokenizer_config.json",
                vocab_rows=248320,
                tokenizer_id_count=248077,
                rows=DRAFT_ROWS,
            ),
        ),
    )


def _build_mtp_recipes() -> tuple[TensorRecipe, ...]:
    source_prefix = "mtp.layers.0."
    object_prefix = "mtp/layer/"
    q_proj = source_prefix + "self_attn.q_proj.weight"
    recipes = [
        TensorRecipe("mtp/input_projection", source("mtp.fc.weight", (2048, 4096))),
        TensorRecipe(
            "mtp/embedding_norm",
            source("mtp.pre_fc_norm_embedding.weight", (2048,)),
        ),
        TensorRecipe(
            "mtp/hidden_norm",
            source("mtp.pre_fc_norm_hidden.weight", (2048,)),
        ),
        TensorRecipe(
            object_prefix + "input_norm",
            source(source_prefix + "input_layernorm.weight", (2048,)),
        ),
        TensorRecipe(
            object_prefix + "attention/query_key_gate_value",
            Concat(
                (
                    _attention_part(q_proj, gate=False),
                    source(source_prefix + "self_attn.k_proj.weight", (512, 2048)),
                    _attention_part(q_proj, gate=True),
                    source(source_prefix + "self_attn.v_proj.weight", (512, 2048)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            object_prefix + "attention/query_norm",
            source(source_prefix + "self_attn.q_norm.weight", (256,)),
        ),
        TensorRecipe(
            object_prefix + "attention/key_norm",
            source(source_prefix + "self_attn.k_norm.weight", (256,)),
        ),
        TensorRecipe(
            object_prefix + "attention/output",
            source(source_prefix + "self_attn.o_proj.weight", (2048, 4096)),
        ),
        TensorRecipe(
            object_prefix + "post_attention_norm",
            source(source_prefix + "post_attention_layernorm.weight", (2048,)),
        ),
    ]
    recipes.extend(_moe_recipes(source_prefix + "mlp.", object_prefix + "moe/"))
    recipes.append(
        TensorRecipe("mtp/final_norm", source("mtp.norm.weight", (2048,)))
    )
    return tuple(recipes)


def _in_inventory_order(recipes, specs) -> tuple[TensorRecipe, ...]:
    """`recipes` as the inventory lists them. The declaration emits its objects in the
    order it declares them, which is not the order the container writes them, and
    `validate_recipe_coverage` compares the two lists position by position."""
    by_name = {item.object_name: item for item in recipes}
    ordered = tuple(by_name.pop(spec.name) for spec in specs)
    if by_name:
        raise ValueError(f"recipes for objects the inventory does not list: {sorted(by_name)[:6]}")
    return ordered


_DECLARED_RECIPE_SPECS = _build_declared_recipes() + _build_draft_head_recipes() + build_vision_recipes(2048)

#: The two source checkpoints this target reads, kept apart because they are opened by
#: two readers and preflighted separately: the model, and the DFlash scorer beside it.
BASE_RECIPE_SPECS = _in_inventory_order(
    (r for r in _DECLARED_RECIPE_SPECS if not r.object_name.startswith("dflash/")),
    inventory.TENSOR_SPECS[: -len(inventory.DFLASH_TENSOR_SPECS)],
)
DFLASH_RECIPE_SPECS = _in_inventory_order(
    (r for r in _DECLARED_RECIPE_SPECS if r.object_name.startswith("dflash/")),
    inventory.DFLASH_TENSOR_SPECS,
)


def build_recipes(geometry=None) -> tuple[TensorRecipe, ...]:
    """Where every artifact object comes from, for a checkpoint of this size.

    The dense family reads its dimensions off the checkpoint because one target serves
    every published size of it. This target serves 35B-A3B and nothing else -- every width
    in this module is that checkpoint's, written out -- so the geometry is accepted for a
    uniform call across the families and has nothing here to vary.
    """
    return BASE_RECIPE_SPECS + DFLASH_RECIPE_SPECS


RECIPE_SPECS = build_recipes()
BASE_RECIPES_BY_NAME = {item.object_name: item for item in BASE_RECIPE_SPECS}
DFLASH_RECIPES_BY_NAME = {
    item.object_name: item for item in DFLASH_RECIPE_SPECS
}
RECIPES_BY_NAME = {item.object_name: item for item in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    """Validate exact output pairing and both source-checkpoint inventories."""

    _common_validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)
    if len(RECIPE_SPECS) != 934 or len(RECIPES_BY_NAME) != 934:
        raise ValueError("35B recipe does not contain exactly 934 tensor transforms")
    base_requirements = base_source_requirements()
    # 1,045 distinct sources when every routed gate_up names its fused HF spelling, plus the two
    # stacked GGUF spellings its AnyOf also accepts, across 40 layers: 1,045 + 80 + the two the
    # first layer's pair introduces. The count is a typo guard, so it counts every spelling a
    # recipe may read rather than the subset one checkpoint happens to satisfy.
    if len(base_requirements) != 1127:
        raise ValueError(
            f"35B base recipe covers {len(base_requirements)} unique sources, "
            "expected 1127"
        )
    dflash_requirements = dflash_source_requirements()
    if len(dflash_requirements) != 69:
        raise ValueError(
            f"35B DFlash recipe covers {len(dflash_requirements)} unique sources, "
            "expected 69"
        )
    if {
        item.dtype
        for item in (*base_requirements.values(), *dflash_requirements.values())
    } != {SOURCE_DTYPE}:
        raise ValueError("35B source recipes must contain only BF16 tensors")


def base_source_requirements() -> dict[str, SourceTensor]:
    return _common_source_requirements(BASE_RECIPE_SPECS)


def dflash_source_requirements() -> dict[str, SourceTensor]:
    return _common_source_requirements(DFLASH_RECIPE_SPECS)


def _preflight_exact_source(
    reader: ShardReader,
    recipes: tuple[TensorRecipe, ...],
    requirements: dict[str, SourceTensor],
    label: str,
) -> SourcePreflight:
    # The reader folds VL-style nesting (model.language_model.* -> model.*)
    # while this recipe addresses sources in the nested dialect; compare the
    # inventory in the folded dialect so both spellings agree (the 27B NVFP4
    # recipe does the same).
    def _folded(name: str) -> str:
        prefix = "model.language_model."
        return "model." + name[len(prefix):] if name.startswith(prefix) else name

    with reader:
        actual_names = {_folded(name) for name in reader.names}
    required_names = {_folded(name) for name in requirements}
    if actual_names != required_names:
        missing = sorted(required_names - actual_names)
        extra = sorted(actual_names - required_names)
        details = []
        if missing:
            details.append(f"missing={missing[:8]!r}")
        if extra:
            details.append(f"extra={extra[:8]!r}")
        raise ValueError(
            f"{label} source inventory differs from its exact tensor contract"
            + (": " + ", ".join(details) if details else "")
        )
    with reader:
        return preflight_source_reader(reader, recipes)


def preflight_base_sources(model_dir: str | Path) -> SourcePreflight:
    model = Path(model_dir)
    return _preflight_exact_source(
        ShardReader.from_index(model / "model.safetensors.index.json"),
        BASE_RECIPE_SPECS,
        base_source_requirements(),
        "35B base checkpoint",
    )


def preflight_dflash_sources(model_dir: str | Path) -> SourcePreflight:
    model = Path(model_dir)
    return _preflight_exact_source(
        ShardReader.from_file(model / "model.safetensors"),
        DFLASH_RECIPE_SPECS,
        dflash_source_requirements(),
        "35B DFlash checkpoint",
    )


__all__ = [
    "BASE_RECIPES_BY_NAME",
    "BASE_RECIPE_SPECS",
    "Cast",
    "Concat",
    "DFLASH_RECIPES_BY_NAME",
    "DFLASH_RECIPE_SPECS",
    "DRAFT_RANKING_PATH",
    "DRAFT_ROWS",
    "DraftHeadTokenIds",
    "Expression",
    "GatherRows",
    "RECIPE_SPECS",
    "RECIPES_BY_NAME",
    "Reshape",
    "SOURCE_DTYPE",
    "ShardReader",
    "Slice",
    "SourcePreflight",
    "SourceTensor",
    "TensorRecipe",
    "Transpose",
    "expression_shape",
    "expression_sources",
    "materialize_expression",
    "materialize_recipe",
    "base_source_requirements",
    "dflash_source_requirements",
    "preflight_base_sources",
    "preflight_dflash_sources",
    "validate_recipe_coverage",
]
