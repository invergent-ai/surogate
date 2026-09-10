"""Serving declarations for an auxiliary DFlash checkpoint."""

from ..block_schema import ServeObject, ServeSection

#: DFlash: a small dense stack that scores draft continuations. It comes from its
#: own checkpoint rather than this model's config, so its geometry is supplied by the resolved auxiliary checkpoint. That checkpoint keeps its tensors at the root and
#: the training graph has no scorer, so the section names them directly rather than
#: through parameters that do not exist.
DFLASH_SERVE_SECTION = ServeSection(
    prefix="dflash/layers/",
    hf_prefix="layers.{index}.",
    objects=(
        ServeObject("input_norm", "bf16", ("C",), source="input_layernorm.weight"),
        ServeObject(
            "attention/query_key_value",
            "quantised",
            ("DflashQkvRows", "C"),
            source=(
                ("self_attn.q_proj.weight", ("DflashAttnCols", "C")),
                ("self_attn.k_proj.weight", ("DflashKvRows", "C")),
                ("self_attn.v_proj.weight", ("DflashKvRows", "C")),
            ),
        ),
        ServeObject("attention/query_norm", "bf16", ("DflashHeadDim",), source="self_attn.q_norm.weight"),
        ServeObject("attention/key_norm", "bf16", ("DflashHeadDim",), source="self_attn.k_norm.weight"),
        ServeObject("attention/output", "quantised", ("C", "DflashAttnCols"), source="self_attn.o_proj.weight"),
        ServeObject("post_attention_norm", "bf16", ("C",), source="post_attention_layernorm.weight"),
        ServeObject(
            "mlp/gate_up",
            "quantised",
            ("DflashGateUpRows", "C"),
            source=(("mlp.gate_proj.weight", ("DflashFfn", "C")), ("mlp.up_proj.weight", ("DflashFfn", "C"))),
        ),
        ServeObject("mlp/down", "quantised", ("C", "DflashFfn"), source="mlp.down_proj.weight"),
    ),
    repeat="dflash_layers",
    capability="dflash",
)

DFLASH_HEAD_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject(
        "dflash/feature_projection",
        "quantised",
        ("C", "DflashFeatureRows"),
        scope="model",
        capability="dflash",
        source="fc.weight",
    ),
    ServeObject("dflash/context_norm", "bf16", ("C",), scope="model", capability="dflash", source="hidden_norm.weight"),
)

DFLASH_TAIL_OBJECTS: tuple[ServeObject, ...] = (
    ServeObject("dflash/final_norm", "bf16", ("C",), scope="model", capability="dflash", source="norm.weight"),
)
