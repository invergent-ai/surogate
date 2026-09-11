"""Prepare Gemma text GGUFs with their matching image projectors."""

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from surogate.serve.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceObject,
    ResourceSpec,
    TensorSpec,
)
from surogate.serve.convert.common import vision_gguf
from surogate.serve.convert.common.gguf_source import GgufSource
from surogate.serve.convert.common.recipe import Concat, Reshape, SourceTensor, TensorRecipe, Transpose
from surogate.serve.gguf.bridge import synthesised_config

from . import inventory
from .convert import resources_for


def config_from_gguf(text, vision):
    arch = text.kv("general.architecture")
    kind = vision.kv("clip.vision.projector_type", vision.kv("clip.projector_type"))
    if (arch, kind) not in (("gemma3", "gemma3"), ("gemma4", "gemma4v"), ("gemma4", "gemma4uv")):
        raise ValueError("Gemma text and vision projector families do not match")
    tc = synthesised_config(text, arch)
    if vision.kv("clip.vision.projection_dim") != tc["hidden_size"]:
        raise ValueError("Gemma text and projector disagree on decoder hidden size")
    version, free = (3 if arch == "gemma3" else 4), kind == "gemma4uv"
    h = vision.kv("clip.vision.embedding_length")
    patch = vision.kv("clip.vision.patch_size")
    merge = vision.kv("clip.vision.projector.scale_factor", 4 if version == 3 else 3)
    position = vision.tensor("v.position_embd.weight")
    if position is None or not h or not patch or not merge:
        raise ValueError("Gemma projector is missing its patch or position geometry")
    vc = {
        "model_type": "siglip_vision_model" if version == 3 else "gemma4_unified_vision" if free else "gemma4_vision",
        "hidden_size": h,
        "intermediate_size": vision.kv("clip.vision.feed_forward_length"),
        "num_hidden_layers": vision.kv("clip.vision.block_count"),
        "num_attention_heads": vision.kv("clip.vision.attention.head_count"),
        "patch_size": patch,
        "image_size": vision.kv("clip.vision.image_size"),
        "hidden_act": "gelu_pytorch_tanh",
        "layer_norm_eps": vision.kv("clip.vision.attention.layer_norm_epsilon", 1e-6),
        "rms_norm_eps": vision.kv("clip.vision.attention.layer_norm_epsilon", 1e-6),
        "pooling_kernel_size": merge,
        "position_embedding_size": position.shape[1],
        "rope_parameters": {"rope_type": "default", "rope_theta": 100.0},
        "standardize": vision.tensor("v.std_bias") is not None,
        "use_clipped_linears": vision.tensor("v.blk.0.attn_q.input_min") is not None,
    }
    if free:
        vc.update(mm_embed_dim=h, output_proj_dims=h, model_patch_size=patch * merge, mm_posemb_size=position.shape[1])
    # The E-series uses causal image attention; other Gemma 4 releases use
    # bidirectional image blocks in the sliding layers.
    if version == 4:
        tc["use_bidirectional_attention"] = None if tc.get("hidden_size_per_layer_input") else "vision"
    tokens = text.kv("tokenizer.ggml.tokens")
    image_token = "<image_soft_token>" if version == 3 else "<|image|>"
    if image_token not in tokens:
        raise ValueError("Gemma GGUF tokenizer has no image token")
    config = {
        "model_type": "gemma3" if version == 3 else "gemma4_unified" if free else "gemma4",
        "architectures": ["Gemma3ForConditionalGeneration" if version == 3 else "Gemma4UnifiedForConditionalGeneration" if free else "Gemma4ForConditionalGeneration"],
        "text_config": tc,
        "vision_config": vc,
        "image_token_id": tokens.index(image_token),
        "boi_token_id": tokens.index("<start_of_image>" if version == 3 else "<|image>"),
        "eoi_token_id": tokens.index("<end_of_image>" if version == 3 else "<image|>"),
    }
    if version == 3:
        size = vc["image_size"]
        if not size or size % (patch * merge):
            raise ValueError("Gemma 3 image size does not match its patch pooling")
        config["mm_tokens_per_image"] = (size // (patch * merge)) ** 2
    elif "<|video|>" in tokens:
        config["video_token_id"] = tokens.index("<|video|>")
    return config


def find_projector(text_path, explicit=None):
    return vision_gguf.find_projector(
        text_path, explicit, lambda text, vision: inventory.geometry_from_config(config_from_gguf(text, vision))
    )


def build_recipes(g):
    specs, original = inventory.vision_recipes(g)
    v = g.vision
    h, patch = v["hidden"], int((v["patch_dim"] // 3) ** 0.5)

    def source(expr):
        name = expr.name
        if v["encoder_free"]:
            role = name.removeprefix("model.embed_vision.")
            if role.startswith("patch_ln1."):
                suffix = role.split(".")[-1]
                # GGUF exports flatten the CHW permutation into a vector.
                raw = Reshape(SourceTensor("v.patch_norm.1." + suffix, (v["patch_dim"],)), (3, patch, patch))
                return Reshape(Transpose(raw, (1, 2, 0)), expr.shape)
            if role == "patch_dense.weight":
                raw = Reshape(SourceTensor("v.patch_embd.weight", (h, v["patch_dim"])), (h, 3, patch, patch))
                return Reshape(Transpose(raw, (0, 2, 3, 1)), expr.shape)
            mapping = {
                "patch_dense.bias": "v.patch_embd.bias",
                "patch_ln2.weight": "v.patch_norm.2.weight",
                "patch_ln2.bias": "v.patch_norm.2.bias",
                "pos_norm.weight": "v.patch_norm.3.weight",
                "pos_norm.bias": "v.patch_norm.3.bias",
                "multimodal_embedder.embedding_projection.weight": "mm.input_projection.weight",
            }
            return SourceTensor(mapping[role], expr.shape)
        root = "model.vision_tower.vision_model." if v["gemma_version"] == 3 else "model.vision_tower."
        if name.startswith("model.multi_modal_projector."):
            return SourceTensor(
                "mm.soft_emb_norm.weight" if "soft_emb_norm" in name else "mm.input_projection.weight", expr.shape
            )
        if name == "model.embed_vision.embedding_projection.weight":
            return SourceTensor("mm.input_projection.weight", expr.shape)
        role = name.removeprefix(root)
        if role == "patch_embedder.input_proj.weight":
            return Reshape(
                Transpose(SourceTensor("v.patch_embd.weight", (h, 3, patch, patch)), (0, 2, 3, 1)), expr.shape
            )
        simple = {
            "embeddings.patch_embedding.weight": "v.patch_embd.weight",
            "embeddings.patch_embedding.bias": "v.patch_embd.bias",
            "embeddings.position_embedding.weight": "v.position_embd.weight",
            "patch_embedder.position_embedding_table": "v.position_embd.weight",
            "post_layernorm.weight": "v.post_ln.weight",
            "post_layernorm.bias": "v.post_ln.bias",
            "std_bias": "v.std_bias",
            "std_scale": "v.std_scale",
        }
        if role in simple:
            return SourceTensor(simple[role], expr.shape)
        pieces = role.split(".")
        if pieces[:2] != ["encoder", "layers"]:
            raise ValueError(f"unknown Gemma vision tensor: {name}")
        layer, tail = pieces[2], ".".join(pieces[3:]).replace(".linear.", ".")
        modules = {
            "self_attn.q_proj": "attn_q",
            "self_attn.k_proj": "attn_k",
            "self_attn.v_proj": "attn_v",
            "self_attn.out_proj": "attn_out",
            "self_attn.o_proj": "attn_out",
            "self_attn.q_norm": "attn_q_norm",
            "self_attn.k_norm": "attn_k_norm",
            "layer_norm1": "ln1",
            "layer_norm2": "ln2",
            "input_layernorm": "ln1",
            "post_attention_layernorm": "attn_post_norm",
            "pre_feedforward_layernorm": "ln2",
            "post_feedforward_layernorm": "ffn_post_norm",
            "mlp.fc1": "ffn_up",
            "mlp.fc2": "ffn_down",
            "mlp.gate_proj": "ffn_gate",
            "mlp.up_proj": "ffn_up",
            "mlp.down_proj": "ffn_down",
        }
        module, suffix = tail.rsplit(".", 1)
        mapped = "v.blk." + layer + "." + modules[module] + "." + suffix
        if not expr.shape:
            return Reshape(SourceTensor(mapped, (1,)), ())
        return SourceTensor(mapped, expr.shape)

    def rewrite(expr):
        if isinstance(expr, SourceTensor):
            return source(expr)
        if isinstance(expr, Concat):
            return replace(expr, sources=tuple(rewrite(e) for e in expr.sources))
        return replace(expr, source=rewrite(expr.source))

    recipes = []
    for recipe in original:
        if v["encoder_free"] and recipe.object_name == "vision/position_embedding":
            expr = Reshape(
                SourceTensor("v.position_embd.weight", (2, v["position_embeddings"], h)),
                (2 * v["position_embeddings"], h),
            )
        else:
            expr = rewrite(recipe.expression)
        recipes.append(TensorRecipe(recipe.object_name, expr))
    return specs, tuple(recipes)


def combine(text_path, vision_path, output):
    with Artifact(text_path) as text, Artifact(vision_path) as vision:
        replacement = {obj.name for obj in vision.objects}
        pairs = [(text, obj, 0) for obj in text.objects if obj.name not in replacement]
        pairs += [(vision, obj, len(text.external)) for obj in vision.objects]
        specs = []
        for _owner, obj, offset in pairs:
            if isinstance(obj, ResourceObject):
                specs.append(ResourceSpec(name=obj.name, bytes=obj.bytes, encoding=obj.encoding))
            else:
                specs.append(
                    TensorSpec(
                        obj.name,
                        obj.shape,
                        obj.format,
                        obj.layout,
                        runs=tuple((s + offset, o, b) for s, o, b in obj.runs),
                        transform=obj.transform,
                        group_map=obj.group_map,
                        segments=obj.segments,
                    )
                )
        with ArtifactWriter(
            output,
            text.identity,
            specs,
            external=(*text.external, *vision.external),
            geometry=text.geometry,
            vision_geometry=vision.vision_geometry,
            layer_types=text.layer_types,
        ) as writer:
            for owner, obj, _ in pairs:
                if not getattr(obj, "runs", ()):
                    writer.write(obj.name, owner.payload(obj))


def convert(text_path, projector_path, output, *, device="cpu"):
    from surogate.serve.ingest import _ensure_from_gguf

    source = GgufSource(Path(text_path), extra=[Path(projector_path)])
    try:
        config = config_from_gguf(source.readers[0], source.readers[-1])
        g = inventory.geometry_from_config(config)
        specs, recipes = build_recipes(g)
        text_artifact = _ensure_from_gguf(Path(text_path), include_vision=False)
        with tempfile.TemporaryDirectory(prefix="surogate-gemma-projector-") as directory:
            root = Path(directory)
            with Artifact(text_artifact) as text:
                for obj in text.objects:
                    if isinstance(obj, ResourceObject):
                        (root / obj.name.removeprefix("frontend/")).write_bytes(text.payload(obj))
            image = {
                "size": {
                    "height": config["vision_config"].get("image_size", 896),
                    "width": config["vision_config"].get("image_size", 896),
                },
                "resample": 2 if g.vision["gemma_version"] == 3 else 3,
                "max_soft_tokens": config.get("mm_tokens_per_image", 280),
            }
            (root / "preprocessor_config.json").write_text(json.dumps(image))
            if g.vision["gemma_version"] == 4:
                (root / "video_preprocessor_config.json").write_text(
                    json.dumps({"max_soft_tokens": 70, "num_frames": 32})
                )
            resources = resources_for(root, g)
            projection = root / "projector.sinfer"

            def transform(spec, tensor):
                if spec.name == "vision/projection_norm":
                    tensor = tensor.float() - 1
                return tensor.float() if spec.format == "FP32" else tensor

            vision_gguf.write_artifact(
                source,
                projection,
                identity=ArtifactIdentity(g.target, "groupwise-int", architecture=g.target),
                geometry={},
                vision_geometry=g.vision,
                layer_types=None,
                specs=specs,
                recipes=recipes,
                resources=resources,
                device=device,
                tensor_transform=transform,
            )
            combine(text_artifact, projection, output)
        Path(str(output) + ".conversion.json").write_text(
            json.dumps(
                {"architecture": g.target, "vision": g.vision, "sources": [str(p) for p in source.shards]}, indent=2
            )
            + "\n"
        )
        return Path(output)
    finally:
        source.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    convert(args.gguf, args.mmproj, args.out, device=args.device)


if __name__ == "__main__":
    main()
