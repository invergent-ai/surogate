"""Emits a serving artifact's object inventory from the DSL declaration.

The converter used to state the geometry a third time — `LAYERS = 48`,
`HIDDEN = 2560`, `HC_COUNT = 4` — and spell out every object name and shape
beside it. None of that is new information: the declaration already knows the
layer schedule and the widths, and since `ServeObject` entries live on the block
schemas it now knows the artifact names, numeric formats and fused compositions
too. So the inventory is derived rather than written.

What is *not* derived, and should not be: the repacking each object needs on the
way in — untiling GDN value heads, unfolding a norm's `+1`, splitting an
interleaved query/gate projection. Those are algorithms, and they stay in the
converter. The declaration names them (`ServeObject.transform`) so that the set
of transforms in play is visible from the model, but it does not try to express
them.

Run: python emit_inventory.py <model_dir> [--diff <converter.module>]
"""

from __future__ import annotations

import sys
from typing import Any

from from_dsl import _compile, _config, _module  # noqa: PLC2701 - same package, one contract


def geometry(config: dict[str, Any]) -> dict[str, int]:
    """The symbols `ServeObject.shape` entries are written against.

    Everything here is derived from the declaration; nothing is a constant that
    could disagree with it.
    """

    hidden = config["d_model"]
    heads_v = config["linear_num_value_heads"]
    dim_v = config["linear_value_head_dim"]
    key_dim = config["linear_num_key_heads"] * config["linear_key_head_dim"]
    value_dim = heads_v * dim_v
    conv_dim = 2 * key_dim + value_dim
    query_size = config["num_query_heads"] * config["head_size"]
    kv_size = config["num_kv_heads"] * config["head_size"]
    # MoE-only quantities: a dense declaration simply has none of them.
    experts = config.get("num_experts", 0)
    expert_ffn = config["d_ff"]
    shared = config.get("shared_expert_intermediate", 0)
    ngram, per_gram = config.get("ngram_size", 0), config.get("heads_per_ngram", 0)
    ple_heads = (ngram - 1) * per_gram if ngram else 0

    return {
        "C": hidden,
        "TwoC": 2 * hidden,
        "M": expert_ffn,
        "TwoM": 2 * expert_ffn,
        "DraftVocab": config.get("draft_head_vocab", 0),
        "Vocab": config["vocab_size"],
        "HeadDim": config["head_size"],
        "QuerySize": query_size,
        "AttnFusedRows": 2 * query_size + 2 * kv_size,
        "HcCount": config.get("hc_count", 0),
        "HcWidth": config.get("hc_count", 0) * hidden,
        "HcLowRank": config.get("hc_lowrank", 0),
        "Hv": heads_v,
        "TwoHv": 2 * heads_v,
        "Vd": dim_v,
        "ValueDim": value_dim,
        "ConvK": config["linear_conv_kernel_dim"],
        "ConvDim": conv_dim,
        "GdnFusedRows": conv_dim + value_dim,
        "RouterRows": experts + 1,
        "RoutedGateUpRows": experts * 2 * expert_ffn,
        "RoutedDownRows": experts * hidden,
        "SharedM": shared,
        "SharedGateUpRows": 2 * shared,
        "IndexerDim": config.get("indexer_head_dim", 0),
        "IndexerQueryRows": config.get("indexer_n_heads", 0) * config.get("indexer_head_dim", 0),
        "PleEmbed": config.get("ple_embed_dim", 0),
        "PleConvKernel": config.get("ple_conv_kernel_size", 0),
        "PleHeads": ple_heads,
        "PleMultipliers": 2 * ngram,
        # Vision tower, when the declaration carries one.
        "VisionHidden": config.get("vision_hidden", 0),
        "VisionIntermediate": config.get("vision_intermediate", 0),
        "VisionQkvRows": config.get("vision_qkv_rows", 0),
        "VisionPatchRows": config.get("vision_patch_rows", 0),
        "VisionPositionEmbeddings": config.get("vision_position_embeddings", 0),
        "VisionMergerHidden": config.get("vision_merger_hidden", 0),
        # DFlash scorer.
        "DflashHeadDim": config.get("dflash_head_dim", 0),
        "DflashQkvRows": config.get("dflash_qkv_rows", 0),
        "DflashAttnCols": config.get("dflash_attn_cols", 0),
        "DflashGateUpRows": config.get("dflash_gate_up_rows", 0),
        "DflashFfn": config.get("dflash_ffn", 0),
        "DflashFeatureRows": config.get("dflash_feature_rows", 0),
    }


def resolve(shape: tuple[str | int, ...], symbols: dict[str, int]) -> tuple[int, ...]:
    resolved = []
    for dim in shape:
        if isinstance(dim, int):
            resolved.append(dim)
        elif dim in symbols:
            resolved.append(symbols[dim])
        else:
            raise KeyError(f"serve object shape references unknown symbol {dim!r}")
    return tuple(resolved)


def inventory_for(architecture: str, hf_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Every tensor a serving artifact stores, from the declaration alone."""

    from surogate.dsl.decorators import _block_registry, _model_registry  # noqa: PLC2701

    ir = _compile(architecture, hf_config)
    config = _config(ir)
    symbols = geometry(config)

    spec = next(
        (s for s in _model_registry.values()
         if s.hf_config and architecture in (s.hf_config.architecture, s.hf_config.model_type)),
        None,
    )
    if spec is None or not getattr(spec, "_nn_model_class", None):
        raise ValueError(f"no DSL model registered for {architecture}")
    model_class = spec._nn_model_class  # noqa: SLF001

    model_objects = getattr(model_class, "_serve_objects_", ())
    layer_objects = getattr(model_class, "_serve_layer_objects_", {})
    block_classes = getattr(model_class, "_serve_blocks_", {})
    if not block_classes:
        raise ValueError(
            f"{model_class.__name__} declares no _serve_blocks_; a model without serve "
            f"objects cannot have its artifact inventory derived"
        )

    block_types = _block_types(config)

    out: list[dict[str, Any]] = []

    def emit(name: str, obj) -> None:
        out.append({
            "name": name,
            "shape": resolve(obj.shape, symbols),
            "format": obj.format,
            "components": obj.components,
            "transform": obj.transform,
        })

    leading = [o for o in model_objects if o.name.endswith("token_embedding")]
    trailing = [o for o in model_objects if o not in leading]
    for obj in leading:
        emit(obj.name, obj)
    for layer, block_type in enumerate(block_types):
        prefix = f"text/layers/{layer}/"
        for marker, objects in layer_objects.items():
            if _layer_matches(marker, layer, config):
                for obj in objects:
                    emit(prefix + obj.name, obj)
        for obj in block_classes[block_type].schema.serve_objects:
            emit(prefix + obj.name, obj)
    for obj in trailing:
        emit(obj.name, obj)

    for section in getattr(model_class, "_serve_sections_", ()):
        count = section.repeat if isinstance(section.repeat, int) else config[section.repeat]
        for index in range(count):
            prefix = section.prefix if count == 1 else f"{section.prefix}{index}/"
            for obj in section.objects:
                emit(prefix + obj.name, obj)
    return out


def _layer_matches(marker: str, layer: int, config: dict[str, Any]) -> bool:
    """Which layers a conditional object group lands on. `ple` follows the
    declaration's `ple_layer_ids`, which is 1-based where the engine is 0-based."""

    if marker == "ple":
        ids = config.get("ple_layer_ids") or []
        return bool(ids) and layer == ids[0] - 1
    raise ValueError(f"unknown layer-object marker {marker!r}")


def _block_types(config: dict[str, Any]) -> list[str]:
    types = config.get("layer_types")
    if types:
        return ["attention" if t == "full_attention" else "mamba" for t in types]
    interval = config["full_attention_interval"]
    return ["attention" if (i + 1) % interval == 0 else "mamba"
            for i in range(config["n_layers"])]


def main() -> int:
    from surogate.dsl.ir_builder import load_hf_config, resolve_architecture

    model_dir = sys.argv[1]
    hf_config = load_hf_config(model_dir)
    architecture = resolve_architecture(hf_config)
    emitted = inventory_for(architecture, hf_config)
    print(f"{architecture}: {len(emitted)} artifact tensors emitted from the declaration")

    if "--diff" not in sys.argv:
        return 0

    import importlib

    module = importlib.import_module(sys.argv[sys.argv.index("--diff") + 1])
    committed = {s.name: (tuple(s.shape), s.format) for s in module.TENSOR_SPECS}
    derived = {o["name"]: (o["shape"], o["format"]) for o in emitted}

    missing = sorted(set(committed) - set(derived))
    extra = sorted(set(derived) - set(committed))
    differing = sorted(
        n for n in set(committed) & set(derived)
        if committed[n][0] != derived[n][0]
        or not formats_agree(derived[n][1], committed[n][1], module)
    )

    print(f"  committed {len(committed)}, derived {len(derived)}")
    for label, names in (("only in converter", missing), ("only in declaration", extra)):
        if names:
            print(f"  {len(names)} {label}: {', '.join(names[:6])}"
                  f"{' ...' if len(names) > 6 else ''}")
    for name in differing[:10]:
        print(f"  DIFFERS {name}: converter={committed[name]} declaration={derived[name]}")
    if not (missing or extra or differing):
        print("  the converter's inventory is exactly what the declaration implies")
    return len(missing) + len(extra) + len(differing)


#: Formats the declaration pins exactly. Anything else is the profile's choice.
_FIXED_FORMATS = ("bf16", "fp32", "i32")


def _format_name(fmt: str, module) -> str:
    return {"w8": module.W8, "bf16": module.BF16, "fp32": module.FP32,
            "i32": module.I32}.get(fmt, fmt)


def formats_agree(declared: str, committed: str, module) -> bool:
    """A declared `quantised` accepts whatever width the export profile picked —
    the 35B stores routed experts Q4 and their down projections Q5 where the 0.8B
    stores both W8. What the declaration pins is that a norm is *never* quantised."""

    if declared == "quantised":
        return committed not in (module.BF16, module.FP32, module.I32)
    return _format_name(declared, module) == committed


if __name__ == "__main__":
    sys.exit(main())
