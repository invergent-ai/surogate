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
    experts, expert_ffn = config["num_experts"], config["d_ff"]
    shared = config["shared_expert_intermediate"]
    ngram, per_gram = config.get("ngram_size", 0), config.get("heads_per_ngram", 0)
    ple_heads = (ngram - 1) * per_gram if ngram else 0

    return {
        "C": hidden,
        "M": expert_ffn,
        "Vocab": config["vocab_size"],
        "HeadDim": config["head_size"],
        "QuerySize": query_size,
        "AttnFusedRows": 2 * query_size + 2 * kv_size,
        "HcCount": config["hc_count"],
        "HcWidth": config["hc_count"] * hidden,
        "HcLowRank": config["hc_lowrank"],
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

    model_objects = getattr(model_class, "_serve_objects_", None)
    if model_objects is None:
        module = sys.modules[model_class.__module__]
        model_objects = getattr(module, "QWEN4_EXP_MODEL_SERVE_OBJECTS", ())
        ple_objects = getattr(module, "QWEN4_EXP_PLE_SERVE_OBJECTS", ())
    else:
        ple_objects = ()

    # Block schemas, by the type tag the stack scheduled them under.
    from surogate.dsl.blocks.qwen4_exp import Qwen4ExpAttentionBlock, Qwen4ExpLinearBlock

    by_type = {"attention": Qwen4ExpAttentionBlock, "mamba": Qwen4ExpLinearBlock}
    block_types = _block_types(config)
    ple_layer = (config["ple_layer_ids"][0] - 1) if config.get("ple_layer_ids") else None

    out: list[dict[str, Any]] = []

    def emit(name: str, obj) -> None:
        out.append({
            "name": name,
            "shape": resolve(obj.shape, symbols),
            "format": obj.format,
            "components": obj.components,
            "transform": obj.transform,
        })

    for obj in model_objects:
        if obj.name == "text/token_embedding":
            emit(obj.name, obj)
    for layer, block_type in enumerate(block_types):
        prefix = f"text/layers/{layer}/"
        if layer == ple_layer:
            for obj in ple_objects:
                emit(prefix + obj.name, obj)
        for obj in by_type[block_type].schema.serve_objects:
            emit(prefix + obj.name, obj)
    for obj in model_objects:
        if obj.name != "text/token_embedding":
            emit(obj.name, obj)
    return out


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
    derived = {o["name"]: (o["shape"], _format_name(o["format"], module)) for o in emitted}

    missing = sorted(set(committed) - set(derived))
    extra = sorted(set(derived) - set(committed))
    differing = sorted(n for n in set(committed) & set(derived) if committed[n] != derived[n])

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


def _format_name(fmt: str, module) -> str:
    return {"w8": module.W8, "bf16": module.BF16, "fp32": module.FP32,
            "i32": module.I32}.get(fmt, fmt)


if __name__ == "__main__":
    sys.exit(main())
