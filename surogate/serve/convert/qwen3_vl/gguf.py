"""Qwen3-VL text GGUF plus vision projector, with native text weights kept in place."""

import json
import math
from pathlib import Path

from surogate.serve.artifact.container import ArtifactIdentity
from surogate.serve.convert.common import vision_gguf
from surogate.serve.convert.common.gguf_source import GgufSource
from surogate.serve.convert.common.recipe import (
    Concat,
    Reshape,
    SourceTensor,
    TensorRecipe,
)
from surogate.serve.gguf.bridge import synthesised_config

from . import inventory


def config_from_gguf(text, vision):
    arch = text.kv("general.architecture")
    if arch not in ("qwen3vl", "qwen3vlmoe"):
        raise ValueError("expected a Qwen3-VL dense or MoE text GGUF")
    if (
        text.kv(f"{arch}.rope.scaling.type", "none") not in ("none", "default")
        or text.kv(f"{arch}.rope.scaling.factor", 1.0) != 1.0
        or text.kv(f"{arch}.attention.sliding_window", 0) != 0
    ):
        raise ValueError("Qwen3-VL GGUF requires default RoPE and full attention")
    if arch == "qwen3vlmoe" and (
        text.kv(f"{arch}.expert_gating_func", 1) != 1
        or text.kv(f"{arch}.expert_weights_norm", True) is not True
        or text.kv(f"{arch}.expert_weights_scale", 1.0) != 1.0
    ):
        raise ValueError("Qwen3-VL-MoE GGUF requires normalized softmax routing with unit scale")
    if vision.kv("clip.projector_type", vision.kv("clip.vision.projector_type")) != "qwen3vl_merger":
        raise ValueError("--mmproj must contain a Qwen3-VL vision projector")
    if vision.kv("clip.use_gelu", True) is not True:
        raise ValueError("Qwen3-VL vision projector requires GELU")
    tc = synthesised_config(text, arch)
    if tc is None:
        raise ValueError("Qwen3-VL GGUF is missing its text geometry or tokenizer metadata")
    sections = list(text.kv(f"{arch}.rope.dimension_sections", []))
    if len(sections) == 4 and sections[-1] == 0:
        sections.pop()
    tc["rope_scaling"] = {"rope_type": "default", "mrope_interleaved": True, "mrope_section": sections}
    moe = arch == "qwen3vlmoe"
    if moe and not tc["intermediate_size"]:
        tc["intermediate_size"] = tc["moe_intermediate_size"]
    layers = vision.kv("clip.vision.block_count")
    deepstack = vision.kv("clip.vision.is_deepstack_layers")
    if (
        not isinstance(deepstack, (list, tuple))
        or len(deepstack) != layers
        or any(type(value) is not bool for value in deepstack)
    ):
        raise ValueError("vision GGUF must declare its complete deepstack layer schedule")
    indexes = [i for i, enabled in enumerate(deepstack) if enabled]
    if text.kv(f"{arch}.n_deepstack_layers") != len(indexes):
        raise ValueError("text GGUF and --mmproj disagree on deepstack feature count")
    projection = vision.kv("clip.vision.projection_dim")
    if projection != tc["hidden_size"]:
        raise ValueError("text GGUF and --mmproj disagree on decoder hidden size")
    image_size, patch_size = vision.kv("clip.vision.image_size"), vision.kv("clip.vision.patch_size")
    if not image_size or not patch_size or image_size % patch_size:
        raise ValueError("vision GGUF image size must be a multiple of its patch size")
    eps = vision.kv("clip.vision.attention.layer_norm_epsilon")
    if eps is None or not math.isclose(eps, 1e-6, rel_tol=1e-5):
        raise ValueError("Qwen3-VL vision layer norm epsilon must be 1e-6")
    vc = {
        "depth": layers,
        "hidden_size": vision.kv("clip.vision.embedding_length"),
        "intermediate_size": vision.kv("clip.vision.feed_forward_length"),
        "num_heads": vision.kv("clip.vision.attention.head_count"),
        "in_channels": 3,
        "temporal_patch_size": 2,
        "patch_size": patch_size,
        "spatial_merge_size": vision.kv("clip.vision.spatial_merge_size", 2),
        "num_position_embeddings": (image_size // patch_size) ** 2,
        "out_hidden_size": projection,
        "hidden_act": "gelu_pytorch_tanh",
        "deepstack_visual_indexes": indexes,
    }
    return {
        "model_type": "qwen3_vl_moe" if moe else "qwen3_vl",
        "architectures": ["Qwen3VLMoeForConditionalGeneration" if moe else inventory.ARCHITECTURE],
        "tie_word_embeddings": tc["tie_word_embeddings"],
        "text_config": tc,
        "vision_config": vc,
    }


def find_projector(text_path, explicit=None):
    return vision_gguf.find_projector(text_path, explicit,
        lambda text, vision: inventory.geometry_from_config(config_from_gguf(text, vision)))


def build_recipes(g, tied):
    recipes = []

    def add(name, source, shape):
        recipes.append(TensorRecipe(name, SourceTensor(source, shape)))

    add("text/token_embedding", "token_embd.weight", (g.vocab, g.hidden))
    add("text/final_norm", "output_norm.weight", (g.hidden,))
    add("text/output_head", "token_embd.weight" if tied else "output.weight", (g.vocab, g.hidden))
    for layer in range(g.layers):
        p, b = f"text/layers/{layer}/", f"blk.{layer}."
        add(p + "input_norm", b + "attn_norm.weight", (g.hidden,))
        recipes.append(
            TensorRecipe(
                p + "attention/query_key_value",
                Concat(
                    tuple(
                        SourceTensor(b + f"attn_{axis}.weight", (rows, g.hidden))
                        for axis, rows in (("q", g.query_size), ("k", g.kv_size), ("v", g.kv_size))
                    ),
                    0,
                ),
            )
        )
        add(p + "attention/query_norm", b + "attn_q_norm.weight", (g.head_dim,))
        add(p + "attention/key_norm", b + "attn_k_norm.weight", (g.head_dim,))
        add(p + "attention/output", b + "attn_output.weight", (g.hidden, g.query_size))
        add(p + "post_attention_norm", b + "ffn_norm.weight", (g.hidden,))
        if g.experts:
            add(p + "moe/router", b + "ffn_gate_inp.weight", (g.experts, g.hidden))
            gate_up = Concat(
                tuple(
                    SourceTensor(b + f"ffn_{half}_exps.weight", (g.experts, g.intermediate, g.hidden))
                    for half in ("gate", "up")
                ),
                1,
            )
            recipes.append(
                TensorRecipe(p + "moe/routed_gate_up", Reshape(gate_up, (g.experts * 2 * g.intermediate, g.hidden)))
            )
            recipes.append(
                TensorRecipe(
                    p + "moe/routed_down",
                    Reshape(
                        SourceTensor(b + "ffn_down_exps.weight", (g.experts, g.hidden, g.intermediate)),
                        (g.experts * g.hidden, g.intermediate),
                    ),
                )
            )
        else:
            recipes.append(
                TensorRecipe(
                    p + "mlp/gate_up",
                    Concat(
                        tuple(
                            SourceTensor(b + f"ffn_{half}.weight", (g.intermediate, g.hidden))
                            for half in ("gate", "up")
                        ),
                        0,
                    ),
                )
            )
            add(p + "mlp/down", b + "ffn_down.weight", (g.hidden, g.intermediate))
    v = g.vision
    patch = tuple(
        Reshape(SourceTensor("v.patch_embd.weight" + suffix, (v["hidden"], 3, 16, 16)), (v["hidden"], 3, 1, 16, 16))
        for suffix in ("", ".1")
    )
    recipes.append(TensorRecipe("vision/patch_embedding", Reshape(Concat(patch, 2), (v["hidden"], 1536))))
    add("vision/patch_embedding_bias", "v.patch_embd.bias", (v["hidden"],))
    add("vision/position_embedding", "v.position_embd.weight", (v["position_embeddings"], v["hidden"]))
    names = {
        "attention/qkv": "attn_qkv",
        "attention/output": "attn_out",
        "mlp/fc1": "ffn_up",
        "mlp/fc2": "ffn_down",
        "norm1": "ln1",
        "norm2": "ln2",
    }
    # The inventory owns all dimensions and order; only source naming is GGUF-specific.
    by_name = {r.object_name: r for r in recipes}
    for spec in inventory.build_tensor_specs(g):
        if spec.name in by_name:
            continue
        name = spec.name
        if "/deepstack/" in name:
            parts = name.split("/")
            tail = ".".join(parts[4:]).replace("_bias", ".bias")
            if not tail.endswith((".weight", ".bias")):
                tail += ".weight"
            source = f"v.deepstack.{parts[2]}.{tail}"
        elif name.startswith("vision/merger/"):
            tail = name.removeprefix("vision/merger/")
            key = tail.replace("_bias", "")
            key = {"fc1": "mm.0", "fc2": "mm.2", "norm/weight": "v.post_ln", "norm/bias": "v.post_ln"}[key]
            source = key + (".bias" if tail.endswith(("_bias", "/bias")) else ".weight")
        else:
            parts = name.split("/")
            tail = "/".join(parts[3:])
            key = tail.removesuffix("_bias").removesuffix("/weight").removesuffix("/bias")
            source = f"v.blk.{parts[2]}.{names[key]}" + (".bias" if tail.endswith(("_bias", "/bias")) else ".weight")
        by_name[name] = TensorRecipe(name, SourceTensor(source, spec.shape))
    return tuple(by_name[s.name] for s in inventory.build_tensor_specs(g))


def convert(text_path, projector_path, output, *, device="cpu"):
    source = GgufSource(Path(text_path), extra=[Path(projector_path)])
    try:
        config = config_from_gguf(source.readers[0], source.readers[-1])
        g = inventory.geometry_from_config(config)
        specs = tuple(
            inventory.tensor_spec(s.name, s.shape, inventory.BF16) if s.name.startswith("vision/") else s
            for s in inventory.build_tensor_specs(g)
        )
        recipes = build_recipes(g, config["tie_word_embeddings"])
        resources, domain = vision_gguf.frontend_resources(source.readers[0], inventory.RESOURCE_SPECS)
        image = {
            "image_processor_type": "Qwen3VLImageProcessor", "processor_class": "Qwen3VLProcessor",
            "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
            "image_mean": source.kv("clip.vision.image_mean"), "image_std": source.kv("clip.vision.image_std"),
            "size": {"shortest_edge": source.kv("clip.vision.image_min_pixels", 65536),
                     "longest_edge": source.kv("clip.vision.image_max_pixels", 16777216)},
        }
        if not image["image_mean"] or not image["image_std"]:
            raise ValueError("vision GGUF is missing image normalization metadata")
        resources["frontend/preprocessor_config.json"] = json.dumps(image).encode()
        resources["frontend/video_preprocessor_config.json"] = json.dumps({
            **image, "video_processor_type": "Qwen3VLVideoProcessor", "fps": 2.0,
            "min_frames": 4, "max_frames": 768}).encode()
        return vision_gguf.write_artifact(source, output,
            identity=ArtifactIdentity(g.architecture, inventory.WEIGHTS_ID, architecture=g.architecture),
            geometry=inventory.geometry_block(g, token_domain=domain), vision_geometry=g.vision,
            layer_types=["full_attention"] * g.layers, specs=specs, recipes=recipes, resources=resources,
            device=device)

    finally:
        source.close()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--mmproj", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    convert(args.gguf, args.mmproj, args.out, device=args.device)


if __name__ == "__main__":
    main()
