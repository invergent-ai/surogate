"""Serving objects derived from a resolved hybrid MoE checkpoint."""

from surogate.serve.convert.common.inventory import (
    BF16, FP32, I32, Q4, Q5, Q6, W8, RESOURCE_SPECS, TensorSpec,
    StoredObjectSpec, tensor_spec, build_vision_specs as _vision_specs,
)
from surogate.serve.convert.common.qwen3_5 import (
    Geometry, geometry_block, vision_geometry_block, vision_tower,
    geometry_from_config as _from_config, geometry_from_checkpoint as _from_checkpoint,
)

MODEL_ID = TARGET_KEY = "qwen3_5_moe"
WEIGHTS_ID = "groupwise-int"


def geometry_from_config(config, **kwargs):
    return _from_config(config, mixture=True, **kwargs)


def geometry_from_checkpoint(root, config=None, **kwargs):
    return _from_checkpoint(root, config, mixture=True, **kwargs)


def hf_config_for(g: Geometry):
    return g.declared.hf_config


def _format(obj):
    if obj.format != "quantised":
        return obj.format.upper()
    if obj.name.startswith("text/layers/"):
        if obj.name.endswith("moe/routed_gate_up"):
            return Q4
        if obj.name.endswith("moe/routed_down"):
            return Q5
    return Q6 if obj.name == "text/output_head" else W8


def build_text_specs(g: Geometry) -> tuple[TensorSpec, ...]:
    specs = []
    for obj in g.declared.objects(capabilities={"text"}):
        if obj.name.startswith("text/draft_head") or (obj.name.startswith("mtp/") and not g.mtp_layers):
            continue
        shape = (g.gdn_value_head_dim,) if obj.name.endswith("/gdn/norm") else obj.shape
        specs.append(tensor_spec(obj.name, tuple(shape), _format(obj)))
    specs.extend((tensor_spec("text/draft_head", (g.draft_vocab, g.hidden), Q4),
                  tensor_spec("text/draft_head_token_ids", (g.draft_vocab,), I32)))
    return tuple(specs)


def build_dflash_specs(g: Geometry, dflash) -> tuple[TensorSpec, ...]:
    if dflash is None:
        return ()
    return tuple(tensor_spec(obj.name, obj.shape, _format(obj))
                 for obj in dflash.declaration(g).objects(capabilities={"text", "dflash"})
                 if obj.name.startswith("dflash/"))


def build_tensor_specs(g: Geometry, *, dflash=None) -> tuple[TensorSpec, ...]:
    tower = vision_tower(g)
    vision = _vision_specs(g.hidden, **tower) if tower else ()
    return build_text_specs(g) + vision + build_dflash_specs(g, dflash)


def build_object_specs(g: Geometry, *, dflash=None) -> tuple[StoredObjectSpec, ...]:
    return RESOURCE_SPECS + build_tensor_specs(g, dflash=dflash)
