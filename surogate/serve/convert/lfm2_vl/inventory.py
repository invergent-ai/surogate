"""Checkpoint-derived LFM2-VL text, vision, and projector objects."""

from dataclasses import dataclass, fields
import json
import math

from surogate.serve.convert.common import declaration, conversion
from surogate.serve.convert.common.checkpoint import positive_int
from surogate.serve.convert.common.inventory import RESOURCE_SPECS as ALL_RESOURCES, ResourceSpec
from surogate.serve.convert.lfm2 import inventory as text_inventory

ARCHITECTURE = "Lfm2VlForConditionalGeneration"
TARGET_KEY = MODEL_ID = "lfm2_vl"
WEIGHTS_ID = "groupwise-int"
CAPABILITIES = ("text", "vision")
TEXT_RESOURCES = tuple(s for s in ALL_RESOURCES if not s.name.endswith("preprocessor_config.json"))
RESOURCE_SPECS = (*TEXT_RESOURCES, ResourceSpec("frontend/preprocessor_config.json"))
tensor_specs = text_inventory.tensor_specs


@dataclass(frozen=True, slots=True)
class Geometry(text_inventory.Geometry):
    vision: dict


def geometry_from_config(config):
    text = dict(config["text_config"])
    base = text_inventory.geometry_from_config(text)
    source = {**base.declared.hf_config, **config}
    declared = declaration.declare(ARCHITECTURE, source)
    vc = config["vision_config"]
    for key in ("hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads",
                "num_patches", "patch_size", "num_channels"):
        positive_int(vc, key)
    if vc.get("model_type") != "siglip2_vision_model" or vc.get("hidden_act") != "gelu_pytorch_tanh":
        raise ValueError("LFM2-VL requires a SigLIP2 vision tower with gelu_pytorch_tanh")
    if vc["hidden_size"] % vc["num_attention_heads"] or vc["hidden_size"] // vc["num_attention_heads"] not in (64, 72):
        raise ValueError("LFM2-VL vision attention supports head dimensions 64 and 72")
    if math.isqrt(vc["num_patches"]) ** 2 != vc["num_patches"]:
        raise ValueError("SigLIP2 position embeddings must form a square table")
    if config.get("downsample_factor") != 2 or vc["patch_size"] != 16 or vc["num_channels"] != 3:
        raise ValueError("LFM2-VL currently requires RGB patches of size 16 and downsample_factor=2")
    if config.get("projector_hidden_act") != "gelu" or not config.get("projector_bias", True):
        raise ValueError("LFM2-VL requires a GELU projector with bias")
    vision = dict(
        siglip2=1, projector_hidden=positive_int(config, "projector_hidden_size"),
        projector_norm=int(config.get("projector_use_layernorm", True)),
        hidden=vc["hidden_size"], layers=vc["num_hidden_layers"],
        intermediate=vc["intermediate_size"], heads=vc["num_attention_heads"],
        patch_dim=vc["num_channels"] * vc["patch_size"] ** 2,
        merge=config["downsample_factor"], position_embeddings=vc["num_patches"],
        rotary_dim=0, rope_theta=0.0, norm_epsilon=float(vc["layer_norm_eps"]),
        output_hidden=base.hidden,
    )
    return Geometry(**{f.name: getattr(base, f.name) for f in fields(base) if f.name != "declared"},
                    declared=declared, vision=vision)


def declared_objects(geometry):
    from .recipe import vision_recipes
    objects = list(geometry.declared.objects(capabilities={"text"}))
    for recipe in vision_recipes(geometry):
        from surogate.serve.convert.common.recipe import expression_shape
        objects.append(declaration.DeclaredObject(
            name=recipe.object_name, shape=expression_shape(recipe.expression), format="bf16",
            components=(), transform="",
        ))
    return objects


def load_resources(model, specs):
    resources = list(conversion.load_resources(model, TEXT_RESOURCES))
    root = json.loads((model / "config.json").read_text())
    processor_path = model / "processor_config.json"
    processor = json.loads(processor_path.read_text()) if processor_path.exists() else {}
    image_path = model / "preprocessor_config.json"
    image = (json.loads(image_path.read_text()) if image_path.exists()
             else processor.get("image_processor"))
    if not isinstance(image, dict):
        raise ValueError("LFM2-VL checkpoint does not contain image processor settings")
    image = dict(image)
    image["image_processor_type"] = "Lfm2VlImageProcessor"
    image["image_token_id"] = root["image_token_id"]
    image["use_image_special_tokens"] = processor.get("use_image_special_tokens", root.get("use_image_special_tokens", True))
    resources.append(conversion.ResourcePayload("frontend/preprocessor_config.json", json.dumps(image).encode()))
    return tuple(resources)
