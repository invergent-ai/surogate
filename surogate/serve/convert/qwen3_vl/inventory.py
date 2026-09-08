"""Qwen3-VL serving objects, derived from the checkpoint and training declaration."""

from dataclasses import dataclass
import math

from surogate.serve.artifact.geometry import validate_resolved_geometry
from surogate.serve.convert.common import declaration
from surogate.serve.convert.common.checkpoint import positive_int, dense_geometry
from surogate.serve.convert.common.inventory import (
    RESOURCE_SPECS, BF16, W8, tensor_spec, build_vision_specs,
)
from surogate.serve.convert.common.qwen3_5 import vision_geometry_block, vision_tower
from surogate.serve.convert.qwen3.inventory import Geometry as TextGeometry

ARCHITECTURE = "Qwen3VLForConditionalGeneration"
TARGET_KEY = MODEL_ID = "qwen3_vl"
WEIGHTS_ID = "groupwise-int"


@dataclass(frozen=True, slots=True)
class Geometry(TextGeometry):
    vision: dict
    deepstack_indexes: tuple[int, ...]
    mrope_sections: tuple[int, int, int]


def geometry_from_config(config):
    if config.get("architectures") != [ARCHITECTURE] or config.get("model_type") != TARGET_KEY:
        raise ValueError("expected a dense Qwen3-VL checkpoint")
    text = config["text_config"]
    for name in ("hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size",
                 "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings"):
        positive_int(text, name)
    if text.get("hidden_act") != "silu" or text.get("attention_bias", False):
        raise ValueError("Qwen3-VL requires SiLU and attention_bias=false")
    if text.get("use_sliding_window", False) or text.get("sliding_window"):
        raise ValueError("Qwen3-VL sliding-window attention is unsupported")
    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default" or not rope.get("mrope_interleaved"):
        raise ValueError("Qwen3-VL requires default interleaved MRoPE")
    sections = rope.get("mrope_section")
    if (not isinstance(sections, (list, tuple)) or len(sections) != 3 or
        any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in sections) or
        sum(sections) * 2 != text["head_dim"] or
        sections[1] > (text["head_dim"] // 2 + 1) // 3 or
        sections[2] > text["head_dim"] // 6):
        raise ValueError("mrope_section must partition the rotary pairs into temporal/height/width")
    source = {**text, **config}
    source["rope_theta"] = rope.get("rope_theta", text.get("rope_theta"))
    declared = declaration.declare(ARCHITECTURE, source)
    vc = config["vision_config"]
    vision = vision_geometry_block(config, text_hidden=text["hidden_size"])
    if (vc.get("hidden_act") != "gelu_pytorch_tanh" or
        vc["hidden_size"] // vc["num_heads"] not in (64, 72) or
        (vc["in_channels"], vc["temporal_patch_size"], vc["patch_size"], vc["spatial_merge_size"]) != (3, 2, 16, 2)):
        raise ValueError("unsupported Qwen3-VL vision activation, attention head, or patch geometry")
    if math.isqrt(vc["num_position_embeddings"]) ** 2 != vc["num_position_embeddings"]:
        raise ValueError("vision position embeddings must form a square table")
    indexes = vc.get("deepstack_visual_indexes")
    if (not isinstance(indexes, list) or
        any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < vc["depth"] for i in indexes) or
        indexes != sorted(set(indexes)) or len(indexes) > text["num_hidden_layers"]):
        raise ValueError("deepstack_visual_indexes must be distinct increasing vision layer indexes")
    vision["deepstack_layers"] = len(indexes)
    g = Geometry(
        layers=text["num_hidden_layers"], hidden=text["hidden_size"],
        intermediate=text["intermediate_size"], vocab=text["vocab_size"],
        query_heads=text["num_attention_heads"], kv_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"], declared=declared, vision=vision,
        deepstack_indexes=tuple(indexes), mrope_sections=tuple(sections),
    )
    geometry_block(g, token_domain=g.vocab)
    return g


def geometry_block(g, *, token_domain):
    values = dense_geometry(g, token_domain=token_domain)
    values.update(zip(("mrope_temporal", "mrope_height", "mrope_width"), g.mrope_sections))
    return validate_resolved_geometry(values)


def merger_tensors(g):
    """Intermediate vision mergers normalize after concatenating a patch group."""
    merged = g.vision["hidden"] * g.vision["merge"] ** 2
    for index, layer in enumerate(g.deepstack_indexes):
        for name, source, shape, fmt in (
            ("fc1", "linear_fc1.weight", (merged, merged), W8),
            ("fc1_bias", "linear_fc1.bias", (merged,), BF16),
            ("fc2", "linear_fc2.weight", (g.hidden, merged), W8),
            ("fc2_bias", "linear_fc2.bias", (g.hidden,), BF16),
            ("norm/weight", "norm.weight", (merged,), BF16),
            ("norm/bias", "norm.bias", (merged,), BF16),
        ):
            yield (f"vision/layers/{layer}/deepstack/{name}",
                   f"model.visual.deepstack_merger_list.{index}.{source}", shape, fmt)


def build_tensor_specs(g):
    text = tuple(tensor_spec(o.name, o.shape, {"quantised": W8, "bf16": BF16}[o.format])
                 for o in g.declared.objects(capabilities={"text"}))
    # Keep intermediate visual features at the same weight precision as the text stack.
    vision = tuple(tensor_spec(s.name, s.shape, BF16 if s.format == BF16 else W8)
                   for s in build_vision_specs(g.hidden, **vision_tower(g)))
    return (*text, *vision,
            *(tensor_spec(name, shape, fmt) for name, _, shape, fmt in merger_tensors(g)))


def build_object_specs(g):
    return (*build_tensor_specs(g), *RESOURCE_SPECS)
