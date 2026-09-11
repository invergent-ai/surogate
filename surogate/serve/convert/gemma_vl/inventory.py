"""Gemma vision contracts alongside the existing checkpoint-derived text inventories."""

import importlib
import math
from dataclasses import dataclass

from surogate.serve.convert.common.checkpoint import dense_geometry, positive_int
from surogate.serve.convert.common.inventory import BF16, FP32, tensor_spec
from surogate.serve.convert.common.recipe import Concat, Reshape, SourceTensor, TensorRecipe, Transpose


@dataclass(frozen=True)
class Geometry:
    text: object
    vision: dict
    target: str
    config: dict


def geometry_from_config(config):
    vc, tc = config["vision_config"], dict(config["text_config"])
    version = 3 if config["model_type"] == "gemma3" else 4
    if version == 3:
        target = "gemma3"
        tc.update(architectures=["Gemma3ForCausalLM"], model_type="gemma3_text")
        module = importlib.import_module("surogate.serve.convert.gemma3.recipe")
        text = module.geometry_from_config(tc)
    else:
        target = (
            "gemma4_moe"
            if tc.get("enable_moe_block")
            else "gemma4_e"
            if tc.get("num_kv_shared_layers", 0) or tc.get("hidden_size_per_layer_input", 0)
            else "gemma4"
        )
        module = importlib.import_module("surogate.serve.convert." + target + ".inventory")
        text = module.geometry_from_config(config)
    if tc.get("use_bidirectional_attention") not in (None, False, "vision"):
        raise ValueError("Gemma vision serving requires causal text attention")
    free = vc.get("model_type") == "gemma4_unified_vision"
    if vc.get("model_type") not in ("siglip_vision_model", "gemma4_vision", "gemma4_unified_vision"):
        raise ValueError("unsupported Gemma vision encoder")
    if not free and vc.get("hidden_act", vc.get("hidden_activation")) != "gelu_pytorch_tanh":
        raise ValueError("Gemma vision requires tanh GELU")
    hidden = positive_int(vc, "mm_embed_dim" if free else "hidden_size")
    if free:
        if vc.get("model_patch_size") != 48 or vc.get("output_proj_dims") != hidden:
            raise ValueError("unsupported Gemma unified patch or projector geometry")
        merge, patch, positions = 1, 48, positive_int(vc, "mm_posemb_size")
    elif version == 3:
        patch, image = positive_int(vc, "patch_size"), positive_int(vc, "image_size")
        tokens = positive_int(config, "mm_tokens_per_image")
        if image % patch or math.isqrt(tokens) ** 2 != tokens or (image // patch) % math.isqrt(tokens):
            raise ValueError("Gemma 3 pooling must partition its square image grid")
        merge = image // patch // math.isqrt(tokens)
        positions = (image // patch) ** 2
    else:
        patch = positive_int(vc, "patch_size")
        merge = positive_int(vc, "pooling_kernel_size")
        positions = positive_int(vc, "position_embedding_size")
    heads = 1 if free else positive_int(vc, "num_attention_heads")
    if not free and (
        hidden % heads
        or hidden // heads not in (64, 72)
        or vc.get("num_key_value_heads", heads) != heads
        or vc.get("head_dim", hidden // heads) != hidden // heads
    ):
        raise ValueError("Gemma vision requires equal Q/K/V head counts and head width 64 or 72")
    rope = vc.get("rope_parameters") or {}
    if version == 4 and not free and rope.get("rope_type", "default") != "default":
        raise ValueError("Gemma vision requires default spatial rotary embeddings")
    vision = dict(
        gemma_version=version,
        gemma_pad_token=int(tc.get("pad_token_id", 0) or 0),
        encoder_free=int(free),
        clipped_linears=int(vc.get("use_clipped_linears", False)),
        standardize=int(vc.get("standardize", False)),
        attention_mode=1 if version == 3 else 2 if tc.get("use_bidirectional_attention") == "vision" else 0,
        max_image_tokens=positive_int(config, "mm_tokens_per_image") if version == 3 else 1120,
        layers=0 if free else positive_int(vc, "num_hidden_layers"),
        hidden=hidden,
        intermediate=hidden if free else positive_int(vc, "intermediate_size"),
        heads=heads,
        patch_dim=3 * patch * patch,
        merge=merge,
        position_embeddings=positions,
        rotary_dim=hidden // heads if version == 4 and not free else 0,
        output_hidden=text.hidden,
        rope_theta=float(rope.get("rope_theta", 100.0)) if version == 4 and not free else 0.0,
        norm_epsilon=float(vc.get("rms_norm_eps", vc.get("layer_norm_eps", 1e-6))),
    )
    return Geometry(text, vision, target, config)


def text_specs_and_recipes(g):
    inv = importlib.import_module("surogate.serve.convert." + g.target + ".inventory")
    recipe = importlib.import_module("surogate.serve.convert." + g.target + ".recipe")
    tied = g.config.get("tie_word_embeddings", g.config["text_config"].get("tie_word_embeddings", True))
    if g.target == "gemma3":
        return inv.build_stored_tensor_specs(g.text, tied_output_head=tied), recipe.build_recipes(
            g.text, tied_output_head=tied
        )
    return inv.tensor_specs(inv.stored_objects(g.text, tied_output_head=tied)), recipe.build_recipes(g.text)


def geometry_block(g, *, token_domain):
    if g.target == "gemma3":
        metadata = dense_geometry(g.text, token_domain=token_domain)
        config = g.config["text_config"]
        metadata.update(
            attention_scale=positive_int(config, "query_pre_attn_scalar") ** -0.5,
            sliding_window=positive_int(config, "sliding_window"),
            sliding_rope_theta=float(config["rope_local_base_freq"]),
            embedding_scale=g.text.hidden**0.5,
        )
        return metadata
    module = importlib.import_module("surogate.serve.convert." + g.target + ".convert")
    return module._geometry_block(g.text, token_domain=token_domain)


def vision_recipes(g):
    v = g.vision
    h, m = v["hidden"], v["intermediate"]
    recipes, specs = [], []

    def add(name, expression, shape, numeric=BF16):
        specs.append(tensor_spec("vision/" + name, shape, numeric))
        recipes.append(TensorRecipe("vision/" + name, expression))

    def weight(name, source, shape, numeric=BF16):
        add(name, SourceTensor(source, shape), shape, numeric)

    if v["encoder_free"]:
        root = "model.embed_vision."
        for name, src, width in (
            ("patch_norm1", "patch_ln1", v["patch_dim"]),
            ("patch_norm2", "patch_ln2", h),
            ("position_norm", "pos_norm", h),
        ):
            for suffix in ("weight", "bias"):
                weight(name + "/" + suffix, root + src + "." + suffix, (width,))
        weight("patch_embedding", root + "patch_dense.weight", (h, v["patch_dim"]))
        weight("patch_embedding_bias", root + "patch_dense.bias", (h,))
        add(
            "position_embedding",
            Reshape(
                Transpose(SourceTensor(root + "pos_embedding", (v["position_embeddings"], 2, h)), (1, 0, 2)),
                (2 * v["position_embeddings"], h),
            ),
            (2 * v["position_embeddings"], h),
        )
        weight("projection", root + "multimodal_embedder.embedding_projection.weight", (g.text.hidden, h))
    elif v["gemma_version"] == 3:
        root = "model.vision_tower.vision_model."
        patch = g.config["vision_config"]["patch_size"]
        add(
            "patch_embedding",
            Reshape(
                SourceTensor(root + "embeddings.patch_embedding.weight", (h, 3, patch, patch)), (h, v["patch_dim"])
            ),
            (h, v["patch_dim"]),
        )
        weight("patch_embedding_bias", root + "embeddings.patch_embedding.bias", (h,))
        weight("position_embedding", root + "embeddings.position_embedding.weight", (v["position_embeddings"], h))
        for layer in range(v["layers"]):
            p, src = f"layers/{layer}/", root + f"encoder.layers.{layer}."
            for suffix, shape, name in (("weight", (h, h), "attention/qkv"), ("bias", (h,), "attention/qkv_bias")):
                out_shape = (3 * h, h) if suffix == "weight" else (3 * h,)
                add(
                    p + name,
                    Concat(
                        tuple(
                            SourceTensor(src + f"self_attn.{axis}_proj." + suffix, shape) for axis in ("q", "k", "v")
                        ),
                        0,
                    ),
                    out_shape,
                )
            for name, source, shape in (
                ("attention/output", "self_attn.out_proj.weight", (h, h)),
                ("attention/output_bias", "self_attn.out_proj.bias", (h,)),
                ("mlp/fc1", "mlp.fc1.weight", (m, h)),
                ("mlp/fc1_bias", "mlp.fc1.bias", (m,)),
                ("mlp/fc2", "mlp.fc2.weight", (h, m)),
                ("mlp/fc2_bias", "mlp.fc2.bias", (h,)),
                ("norm1/weight", "layer_norm1.weight", (h,)),
                ("norm1/bias", "layer_norm1.bias", (h,)),
                ("norm2/weight", "layer_norm2.weight", (h,)),
                ("norm2/bias", "layer_norm2.bias", (h,)),
            ):
                weight(p + name, src + source, shape)
        for suffix in ("weight", "bias"):
            weight("post_norm/" + suffix, root + "post_layernorm." + suffix, (h,))
        weight("projection_norm", "model.multi_modal_projector.mm_soft_emb_norm.weight", (h,))
        add(
            "projection",
            Transpose(
                SourceTensor("model.multi_modal_projector.mm_input_projection_weight", (h, g.text.hidden)), (1, 0)
            ),
            (g.text.hidden, h),
        )
    else:
        root = "model.vision_tower."
        weight("patch_embedding", root + "patch_embedder.input_proj.weight", (h, v["patch_dim"]))
        add(
            "position_embedding",
            Reshape(
                SourceTensor(root + "patch_embedder.position_embedding_table", (2, v["position_embeddings"], h)),
                (2 * v["position_embeddings"], h),
            ),
            (2 * v["position_embeddings"], h),
        )
        for layer in range(v["layers"]):
            p, src = f"layers/{layer}/", root + f"encoder.layers.{layer}."
            for name, module, shape in (
                ("attention/query", "self_attn.q_proj", (h, h)),
                ("attention/key", "self_attn.k_proj", (h, h)),
                ("attention/value", "self_attn.v_proj", (h, h)),
                ("attention/output", "self_attn.o_proj", (h, h)),
                ("mlp/gate", "mlp.gate_proj", (m, h)),
                ("mlp/up", "mlp.up_proj", (m, h)),
                ("mlp/down", "mlp.down_proj", (h, m)),
            ):
                weight(p + name, src + module + ".linear.weight", shape)
                if v["clipped_linears"]:
                    add(
                        p + name + "/clip",
                        Concat(
                            tuple(
                                Reshape(SourceTensor(src + module + "." + bound, ()), (1,))
                                for bound in ("input_min", "input_max", "output_min", "output_max")
                            ),
                            0,
                        ),
                        (4,),
                        FP32,
                    )
            for name, module, width in (
                ("input_norm", "input_layernorm", h),
                ("post_attention_norm", "post_attention_layernorm", h),
                ("pre_feedforward_norm", "pre_feedforward_layernorm", h),
                ("post_feedforward_norm", "post_feedforward_layernorm", h),
                ("attention/query_norm", "self_attn.q_norm", h // v["heads"]),
                ("attention/key_norm", "self_attn.k_norm", h // v["heads"]),
            ):
                weight(p + name, src + module + ".weight", (width,))
        if v["standardize"]:
            for name in ("std_bias", "std_scale"):
                weight(name, root + name, (h,), FP32)
        weight("projection", "model.embed_vision.embedding_projection.weight", (g.text.hidden, h))
    return tuple(specs), tuple(recipes)
