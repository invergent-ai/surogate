"""Hugging Face source recipe for Qwen3.6-35B-A3B."""

from __future__ import annotations

from pathlib import Path

from surogate.serve.tools.convert.common.safetensors import ShardReader
from surogate.serve.tools.convert.common.recipe import (
    AnyOf,
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
    attention_qproj_part,
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


def _attention_part(source_name: str, *, gate: bool) -> Expression:
    return attention_qproj_part(
        source_name,
        gate,
        num_heads=16,
        hidden_size=2048,
    )


def _moe_recipes(source_prefix: str, object_prefix: str) -> tuple[TensorRecipe, ...]:
    """Build one expert-major sparse-MoE source transform.

    The routed gate/up source is already expert-major and half-split inside
    each expert.  A contiguous reshape therefore implements exactly
    ``stored_row(e,p,r) = e*1024 + p*512 + r`` without a permutation.
    """

    return (
        TensorRecipe(
            object_prefix + "router_shared_gate",
            Concat(
                (
                    source(source_prefix + "gate.weight", (256, 2048)),
                    source(source_prefix + "shared_expert_gate.weight", (1, 2048)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            object_prefix + "routed_gate_up",
            # Two spellings of the same rows. An HF checkpoint fuses gate and up into one
            # expert-major tensor; a GGUF of the same model keeps them as two stacked
            # tensors, and concatenating those on the expert's output axis reproduces
            # stored_row(e, p, r) = e*1024 + p*512 + r exactly -- so a GGUF's routed experts
            # are a row gather over blocks, not something that has to be dequantised.
            AnyOf((
                Reshape(
                    source(source_prefix + "experts.gate_up_proj", (256, 1024, 2048)),
                    (262144, 2048),
                ),
                Reshape(
                    Concat(
                        (
                            source(source_prefix + "experts.gate_proj", (256, 512, 2048)),
                            source(source_prefix + "experts.up_proj", (256, 512, 2048)),
                        ),
                        1,
                    ),
                    (262144, 2048),
                ),
            )),
        ),
        TensorRecipe(
            object_prefix + "routed_down",
            Reshape(
                source(source_prefix + "experts.down_proj", (256, 2048, 512)),
                (524288, 512),
            ),
        ),
        TensorRecipe(
            object_prefix + "shared_gate_up",
            Concat(
                (
                    source(
                        source_prefix + "shared_expert.gate_proj.weight",
                        (512, 2048),
                    ),
                    source(
                        source_prefix + "shared_expert.up_proj.weight",
                        (512, 2048),
                    ),
                ),
                0,
            ),
        ),
        TensorRecipe(
            object_prefix + "shared_down",
            source(
                source_prefix + "shared_expert.down_proj.weight",
                (2048, 512),
            ),
        ),
    )


def _build_text_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "text/token_embedding",
            source("model.language_model.embed_tokens.weight", (248320, 2048)),
        )
    ]

    for layer in inventory.TEXT_LAYERS:
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            TensorRecipe(
                object_prefix + "input_norm",
                source(source_prefix + "input_layernorm.weight", (2048,)),
            )
        )

        if layer in inventory.FULL_ATTENTION_LAYERS:
            q_proj = source_prefix + "self_attn.q_proj.weight"
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "attention/query_key_gate_value",
                        Concat(
                            (
                                _attention_part(q_proj, gate=False),
                                source(
                                    source_prefix + "self_attn.k_proj.weight",
                                    (512, 2048),
                                ),
                                _attention_part(q_proj, gate=True),
                                source(
                                    source_prefix + "self_attn.v_proj.weight",
                                    (512, 2048),
                                ),
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
                        source(
                            source_prefix + "self_attn.o_proj.weight",
                            (2048, 4096),
                        ),
                    ),
                )
            )
        else:
            convolution = source(
                source_prefix + "linear_attn.conv1d.weight",
                (8192, 1, 4),
            )
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "gdn/a_log",
                        Cast(
                            source(source_prefix + "linear_attn.A_log", (32,)),
                            inventory.FP32,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        Cast(
                            source(source_prefix + "linear_attn.dt_bias", (32,)),
                            inventory.FP32,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/convolution",
                        Transpose(
                            Reshape(Slice(convolution, 1, 0, 1), (8192, 4)),
                            (1, 0),
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/a_b_projection",
                        Concat(
                            (
                                source(
                                    source_prefix + "linear_attn.in_proj_a.weight",
                                    (32, 2048),
                                ),
                                source(
                                    source_prefix + "linear_attn.in_proj_b.weight",
                                    (32, 2048),
                                ),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/query_key_value_z",
                        Concat(
                            (
                                source(
                                    source_prefix + "linear_attn.in_proj_qkv.weight",
                                    (8192, 2048),
                                ),
                                source(
                                    source_prefix + "linear_attn.in_proj_z.weight",
                                    (4096, 2048),
                                ),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/norm",
                        source(source_prefix + "linear_attn.norm.weight", (128,)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/output",
                        source(
                            source_prefix + "linear_attn.out_proj.weight",
                            (2048, 4096),
                        ),
                    ),
                )
            )

        recipes.append(
            TensorRecipe(
                object_prefix + "post_attention_norm",
                source(source_prefix + "post_attention_layernorm.weight", (2048,)),
            )
        )
        recipes.extend(
            _moe_recipes(
                source_prefix + "mlp.",
                object_prefix + "moe/",
            )
        )

    recipes.extend(
        (
            TensorRecipe(
                "text/final_norm",
                source("model.language_model.norm.weight", (2048,)),
            ),
            TensorRecipe(
                "text/output_head",
                source("lm_head.weight", (248320, 2048)),
            ),
        )
    )
    return tuple(recipes)


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


def _build_dflash_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "dflash/feature_projection",
            source("fc.weight", (2048, 16384)),
        ),
        TensorRecipe(
            "dflash/context_norm",
            source("hidden_norm.weight", (2048,)),
        ),
    ]
    for layer in inventory.DFLASH_LAYERS:
        source_prefix = f"layers.{layer}."
        object_prefix = f"dflash/layers/{layer}/"
        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + "input_norm",
                    source(source_prefix + "input_layernorm.weight", (2048,)),
                ),
                TensorRecipe(
                    object_prefix + "attention/query_key_value",
                    Concat(
                        (
                            source(
                                source_prefix + "self_attn.q_proj.weight",
                                (4096, 2048),
                            ),
                            source(
                                source_prefix + "self_attn.k_proj.weight",
                                (1024, 2048),
                            ),
                            source(
                                source_prefix + "self_attn.v_proj.weight",
                                (1024, 2048),
                            ),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "attention/query_norm",
                    source(source_prefix + "self_attn.q_norm.weight", (128,)),
                ),
                TensorRecipe(
                    object_prefix + "attention/key_norm",
                    source(source_prefix + "self_attn.k_norm.weight", (128,)),
                ),
                TensorRecipe(
                    object_prefix + "attention/output",
                    source(
                        source_prefix + "self_attn.o_proj.weight",
                        (2048, 4096),
                    ),
                ),
                TensorRecipe(
                    object_prefix + "post_attention_norm",
                    source(
                        source_prefix + "post_attention_layernorm.weight",
                        (2048,),
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    Concat(
                        (
                            source(
                                source_prefix + "mlp.gate_proj.weight",
                                (6144, 2048),
                            ),
                            source(
                                source_prefix + "mlp.up_proj.weight",
                                (6144, 2048),
                            ),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/down",
                    source(
                        source_prefix + "mlp.down_proj.weight",
                        (2048, 6144),
                    ),
                ),
            )
        )
    recipes.append(
        TensorRecipe("dflash/final_norm", source("norm.weight", (2048,)))
    )
    return tuple(recipes)


BASE_RECIPE_SPECS = (
    _build_text_recipes()
    + _build_draft_head_recipes()
    + _build_mtp_recipes()
    + build_vision_recipes(2048)
)
DFLASH_RECIPE_SPECS = _build_dflash_recipes()
RECIPE_SPECS = BASE_RECIPE_SPECS + DFLASH_RECIPE_SPECS
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
