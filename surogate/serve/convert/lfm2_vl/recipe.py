"""The text mapping comes from the DSL; the external vision tower follows SigLIP2."""

from surogate.serve.convert.common.recipe import SourceTensor, TensorRecipe, Concat
from surogate.serve.convert.lfm2.recipe import open_reader, preflight_sources
from surogate.serve.convert.common.declaration import derive_recipes


def vision_recipes(g):
    v = g.vision
    h, m = v["hidden"], v["intermediate"]
    root = "model.vision_tower.vision_model."
    out = []

    def add(name, source, shape):
        out.append(TensorRecipe("vision/" + name, SourceTensor(source, shape)))

    add("patch_embedding", root + "embeddings.patch_embedding.weight", (h, v["patch_dim"]))
    add("patch_embedding_bias", root + "embeddings.patch_embedding.bias", (h,))
    add("position_embedding", root + "embeddings.position_embedding.weight", (v["position_embeddings"], h))
    for layer in range(v["layers"]):
        prefix = f"layers/{layer}/"
        src = root + f"encoder.layers.{layer}."
        for suffix, shape in (("weight", (h, h)), ("bias", (h,))):
            out.append(TensorRecipe("vision/" + prefix + ("attention/qkv" if suffix == "weight" else "attention/qkv_bias"),
                Concat(tuple(SourceTensor(src + f"self_attn.{part}_proj.{suffix}", shape) for part in ("q", "k", "v")), 0)))
        for name, path, shape in (
            ("attention/output", "self_attn.out_proj.weight", (h, h)),
            ("attention/output_bias", "self_attn.out_proj.bias", (h,)),
            ("mlp/fc1", "mlp.fc1.weight", (m, h)), ("mlp/fc1_bias", "mlp.fc1.bias", (m,)),
            ("mlp/fc2", "mlp.fc2.weight", (h, m)), ("mlp/fc2_bias", "mlp.fc2.bias", (h,)),
            ("norm1/weight", "layer_norm1.weight", (h,)), ("norm1/bias", "layer_norm1.bias", (h,)),
            ("norm2/weight", "layer_norm2.weight", (h,)), ("norm2/bias", "layer_norm2.bias", (h,)),
        ):
            add(prefix + name, src + path, shape)
    add("post_norm/weight", root + "post_layernorm.weight", (h,))
    add("post_norm/bias", root + "post_layernorm.bias", (h,))
    src = "model.multi_modal_projector."
    merged, ph = h * v["merge"] ** 2, v["projector_hidden"]
    if v["projector_norm"]:
        add("merger/norm/weight", src + "layer_norm.weight", (merged,))
        add("merger/norm/bias", src + "layer_norm.bias", (merged,))
    for name, path, shape in (
        ("merger/fc1", "linear_1.weight", (ph, merged)), ("merger/fc1_bias", "linear_1.bias", (ph,)),
        ("merger/fc2", "linear_2.weight", (g.hidden, ph)), ("merger/fc2_bias", "linear_2.bias", (g.hidden,)),
    ):
        add(name, src + path, shape)
    return tuple(out)


def build_recipes(geometry):
    return (*derive_recipes(geometry.declared, capabilities={"text"}, tied_output_head=True),
            *vision_recipes(geometry))
