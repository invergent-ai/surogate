"""Prepare an LFM2-VL text GGUF and its matching SigLIP2 projector."""

import json
from pathlib import Path

from surogate.serve.artifact.container import ArtifactIdentity
from surogate.serve.convert.common import vision_gguf
from surogate.serve.convert.common.gguf_source import GgufSource
from surogate.serve.convert.common.recipe import Concat, Reshape, SourceTensor, TensorRecipe, Transpose
from surogate.serve.convert.lfm2.convert import _geometry_block
from surogate.serve.gguf.lfm2 import config_from_gguf as text_config_from_gguf

from . import inventory


def config_from_gguf(text, vision):
    if text.kv("general.architecture") != "lfm2" or vision.kv("clip.projector_type") != "lfm2":
        raise ValueError("--mmproj must pair LFM2 text with an LFM2-VL projector")
    tc = text_config_from_gguf(text)
    if vision.kv("clip.vision.projection_dim") != tc["hidden_size"]:
        raise ValueError("text GGUF and --mmproj disagree on decoder hidden size")
    patch = vision.kv("clip.vision.patch_size")
    size = vision.kv("clip.vision.image_size")
    if not patch or not size or size % patch or vision.kv("clip.use_gelu", True) is not True:
        raise ValueError("unsupported LFM2-VL vision patch geometry or activation")
    projection = vision.tensor("mm.1.weight")
    if projection is None or len(projection.shape) != 2:
        raise ValueError("LFM2-VL projector is missing its first projection")
    tokens = text.kv("tokenizer.ggml.tokens")
    if "<image>" not in tokens:
        raise ValueError("LFM2-VL text tokenizer has no <image> token")
    return {
        "architectures": [inventory.ARCHITECTURE],
        "model_type": "lfm2_vl",
        "text_config": tc,
        "image_token_id": tokens.index("<image>"),
        "downsample_factor": vision.kv("clip.vision.projector.scale_factor", 2),
        "projector_hidden_size": projection.shape[1],
        "projector_hidden_act": "gelu",
        "projector_bias": True,
        "projector_use_layernorm": vision.tensor("mm.input_norm.weight") is not None,
        "vision_config": {
            "model_type": "siglip2_vision_model",
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": vision.kv("clip.vision.embedding_length"),
            "intermediate_size": vision.kv("clip.vision.feed_forward_length"),
            "num_hidden_layers": vision.kv("clip.vision.block_count"),
            "num_attention_heads": vision.kv("clip.vision.attention.head_count"),
            "num_channels": 3,
            "patch_size": patch,
            "num_patches": (size // patch) ** 2,
            "layer_norm_eps": vision.kv("clip.vision.attention.layer_norm_epsilon"),
        },
    }


def find_projector(text_path, explicit=None):
    return vision_gguf.find_projector(
        text_path, explicit, lambda text, vision: inventory.geometry_from_config(config_from_gguf(text, vision))
    )


def build_recipes(g):
    recipes = []

    def add(name, source, shape):
        recipes.append(TensorRecipe(name, SourceTensor(source, shape)))

    add("text/token_embedding", "token_embd.weight", (g.vocab, g.hidden))
    add(
        "text/output_head",
        "token_embd.weight"
        if g.declared.hf_config["text_config"].get("tie_word_embeddings", True)
        else "output.weight",
        (g.vocab, g.hidden),
    )
    add("text/final_norm", "token_embd_norm.weight", (g.hidden,))
    for layer in range(g.layers):
        p, b = f"text/layers/{layer}/", f"blk.{layer}."
        add(p + "input_norm", b + "attn_norm.weight", (g.hidden,))
        add(p + "post_attention_norm", b + "ffn_norm.weight", (g.hidden,))
        if g.is_attention(layer):
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
        else:
            add(p + "conv/in_proj", b + "shortconv.in_proj.weight", (3 * g.hidden, g.hidden))
            add(p + "conv/out_proj", b + "shortconv.out_proj.weight", (g.hidden, g.hidden))
            recipes.append(
                TensorRecipe(
                    p + "conv/convolution",
                    Transpose(SourceTensor(b + "shortconv.conv.weight", (g.hidden, g.conv_kernel)), (1, 0)),
                )
            )
        recipes.append(
            TensorRecipe(
                p + "mlp/gate_up",
                Concat(
                    tuple(
                        SourceTensor(b + f"ffn_{half}.weight", (g.intermediate, g.hidden)) for half in ("gate", "up")
                    ),
                    0,
                ),
            )
        )
        add(p + "mlp/down", b + "ffn_down.weight", (g.hidden, g.intermediate))
    v = g.vision
    h, m = v["hidden"], v["intermediate"]
    recipes.append(
        TensorRecipe(
            "vision/patch_embedding",
            Reshape(Transpose(SourceTensor("v.patch_embd.weight", (h, 3, 16, 16)), (0, 2, 3, 1)), (h, 768)),
        )
    )
    add("vision/patch_embedding_bias", "v.patch_embd.bias", (h,))
    add("vision/position_embedding", "v.position_embd.weight", (v["position_embeddings"], h))
    for layer in range(v["layers"]):
        p, b = f"vision/layers/{layer}/", f"v.blk.{layer}."
        for suffix, shape, name in (("weight", (h, h), "attention/qkv"), ("bias", (h,), "attention/qkv_bias")):
            recipes.append(
                TensorRecipe(
                    p + name,
                    Concat(tuple(SourceTensor(b + f"attn_{axis}." + suffix, shape) for axis in ("q", "k", "v")), 0),
                )
            )
        for name, source, shape in (
            ("attention/output", "attn_out.weight", (h, h)),
            ("attention/output_bias", "attn_out.bias", (h,)),
            ("mlp/fc1", "ffn_up.weight", (m, h)),
            ("mlp/fc1_bias", "ffn_up.bias", (m,)),
            ("mlp/fc2", "ffn_down.weight", (h, m)),
            ("mlp/fc2_bias", "ffn_down.bias", (h,)),
            ("norm1/weight", "ln1.weight", (h,)),
            ("norm1/bias", "ln1.bias", (h,)),
            ("norm2/weight", "ln2.weight", (h,)),
            ("norm2/bias", "ln2.bias", (h,)),
        ):
            add(p + name, b + source, shape)
    for suffix in ("weight", "bias"):
        add("vision/post_norm/" + suffix, "v.post_ln." + suffix, (h,))
        if v["projector_norm"]:
            add("vision/merger/norm/" + suffix, "mm.input_norm." + suffix, (4 * h,))
    ph = v["projector_hidden"]
    for name, src, shape in (
        ("fc1", "1.weight", (ph, 4 * h)),
        ("fc1_bias", "1.bias", (ph,)),
        ("fc2", "2.weight", (g.hidden, ph)),
        ("fc2_bias", "2.bias", (g.hidden,)),
    ):
        add("vision/merger/" + name, "mm." + src, shape)
    by_name = {r.object_name: r for r in recipes}
    return tuple(by_name[s.name] for s in inventory.tensor_specs(inventory.declared_objects(g)))


def convert(text_path, projector_path, output, *, device="cpu"):
    source = GgufSource(Path(text_path), extra=[Path(projector_path)])
    try:
        config = config_from_gguf(source.readers[0], source.readers[-1])
        g = inventory.geometry_from_config(config)
        specs = inventory.tensor_specs(inventory.declared_objects(g))
        resources, domain = vision_gguf.frontend_resources(source.readers[0], inventory.TEXT_RESOURCES)
        if source.kv("clip.vision.image_mean") != [0.5] * 3 or source.kv("clip.vision.image_std") != [0.5] * 3:
            raise ValueError("LFM2-VL GGUF requires image mean/std of 0.5")
        resources["frontend/preprocessor_config.json"] = json.dumps(
            {
                "image_processor_type": "Lfm2VlImageProcessor",
                "image_token_id": config["image_token_id"],
                "min_image_tokens": 64,
                "max_image_tokens": 256,
                "resample": 2,
                "use_image_special_tokens": True,
            }
        ).encode()
        return vision_gguf.write_artifact(
            source,
            output,
            identity=ArtifactIdentity("lfm2_vl", inventory.WEIGHTS_ID, architecture="lfm2_vl"),
            geometry=_geometry_block(g, token_domain=domain),
            vision_geometry=g.vision,
            layer_types=g.layer_types,
            specs=specs,
            recipes=build_recipes(g),
            resources=resources,
            device=device,
            native_text=False,
        )
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
