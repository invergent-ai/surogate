"""Flash-Next's Qwen vision projector, paired with native text GGUF weights."""

import json
import math

from surogate.serve.convert.common import vision_gguf
from surogate.serve.convert.common.checkpoint import positive_int
from surogate.serve.convert.common.recipe import Concat, Reshape, SourceTensor, TensorRecipe


def config_from_gguf(text, vision):
    if text.kv("general.architecture") != "qwen4exp":
        raise ValueError("expected qwen4exp text weights")
    if vision.kv("clip.projector_type") != "qwen3vl_merger" or vision.kv("clip.use_gelu") is not True:
        raise ValueError("Flash-Next requires a Qwen3-VL GELU vision projector")
    values = {name: vision.kv("clip.vision." + key) for name, key in {
        "depth": "block_count", "hidden_size": "embedding_length",
        "intermediate_size": "feed_forward_length", "num_heads": "attention.head_count",
        "patch_size": "patch_size", "spatial_merge_size": "spatial_merge_size",
        "out_hidden_size": "projection_dim", "image_size": "image_size",
    }.items()}
    for name in values:
        positive_int(values, name)
    if values["out_hidden_size"] != text.kv("qwen4exp.embedding_length"):
        raise ValueError("text and projector hidden sizes disagree")
    if values["patch_size"] != 16 or values["spatial_merge_size"] != 2:
        raise ValueError("Flash-Next vision requires patch size 16 and merge size 2")
    image_size = values.pop("image_size")
    if image_size % values["patch_size"]:
        raise ValueError("projector image size must be divisible by patch size")
    if vision.kv("clip.vision.is_deepstack_layers") != [False] * values["depth"]:
        raise ValueError("Flash-Next projector must declare no deepstack layers")
    eps = vision.kv("clip.vision.attention.layer_norm_epsilon")
    if not isinstance(eps, (int, float)) or not math.isclose(eps, 1e-6, rel_tol=1e-5):
        raise ValueError("Flash-Next vision requires layer norm epsilon 1e-6")
    return dict(values, in_channels=3, temporal_patch_size=2,
                num_position_embeddings=(image_size // values["patch_size"]) ** 2,
                hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[])


def find_projector(text_path, explicit=None):
    return vision_gguf.find_projector(text_path, explicit, config_from_gguf)


def build_recipes(g):
    from .inventory import build_vision_specs

    specs = build_vision_specs(g)
    if not specs:
        return ()
    hidden, patch_dim = specs[0].shape
    config = g.declared.hf_config["vision_config"]
    channels, size = config["in_channels"], config["patch_size"]
    patch = tuple(Reshape(SourceTensor("v.patch_embd.weight" + (f".{i}" if i else ""),
                                      (hidden, channels, size, size)),
                          (hidden, channels, 1, size, size)) for i in range(config["temporal_patch_size"]))
    recipes = [TensorRecipe("vision/patch_embedding", Reshape(Concat(patch, 2), (hidden, patch_dim)))]
    names = {"attention/qkv": "attn_qkv", "attention/output": "attn_out",
             "mlp/fc1": "ffn_up", "mlp/fc2": "ffn_down", "norm1": "ln1", "norm2": "ln2"}
    for spec in specs[1:]:
        name = spec.name
        if name == "vision/patch_embedding_bias":
            source = "v.patch_embd.bias"
        elif name == "vision/position_embedding":
            source = "v.position_embd.weight"
        elif name.startswith("vision/merger/"):
            tail = name.removeprefix("vision/merger/")
            key = {"fc1": "mm.0", "fc2": "mm.2", "norm/weight": "v.post_ln",
                   "norm/bias": "v.post_ln"}[tail.removesuffix("_bias")]
            source = key + (".bias" if tail.endswith(("_bias", "/bias")) else ".weight")
        else:
            parts = name.split("/")
            tail = "/".join(parts[3:])
            key = tail.removesuffix("_bias").removesuffix("/weight").removesuffix("/bias")
            source = f"v.blk.{parts[2]}.{names[key]}" + (".bias" if tail.endswith(("_bias", "/bias")) else ".weight")
        recipes.append(TensorRecipe(name, SourceTensor(source, spec.shape)))
    return tuple(recipes)


def preprocessor_config(projector):
    channels = {}
    for name in ("image_mean", "image_std"):
        value = projector.kv("clip.vision." + name)
        if (not isinstance(value, list) or len(value) != 3 or
                any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in value) or
                (name == "image_std" and min(value) <= 0)):
            raise ValueError(f"projector must declare valid {name}")
        channels[name] = value
    return json.dumps(dict(channels, image_processor_type="Qwen3VLImageProcessor",
                           processor_class="Qwen3VLProcessor", patch_size=16,
                           temporal_patch_size=2, merge_size=2,
                           size={"shortest_edge": projector.kv("clip.vision.image_min_pixels", 65536),
                                 "longest_edge": projector.kv("clip.vision.image_max_pixels", 16777216)})).encode()
