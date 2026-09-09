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

from surogate.serve.convert.common.declaration import (
    block_types as _block_types,
    inventory_for,
    layer_matches as _layer_matches,
    resolve,
    symbols_for as geometry,
)

__all__ = ["geometry", "resolve", "inventory_for", "formats_agree", "_block_types", "_layer_matches"]


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
    stores both W8. What the declaration pins is that a norm is *never* quantised.

    `profiled` widens that to include BF16, and pins only that the object is a weight:
    the vision tower's default profile stores it whole, and `--vision-storage quantized`
    narrows it, so neither width is the declaration's to fix."""

    if declared == "quantised":
        return committed not in (module.BF16, module.FP32, module.I32)
    if declared == "profiled":
        return committed not in (module.FP32, module.I32)
    return _format_name(declared, module) == committed


if __name__ == "__main__":
    sys.exit(main())
