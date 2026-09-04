"""Persistent-object contract for every export of the interleaved gated-delta family.

Two things vary and nothing else does: the checkpoint's dimensions, and what its export did
to the weights. Both are parameters here — `Geometry` and `Export` — so one object list serves
Qwen3.5, Qwen3.6 and Qwen3.8 at every published size and quantisation.

This module contains only target storage roles. Source-checkpoint mapping and materialization
live in the sibling conversion recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from surogate.serve.convert.common.inventory import (
    tied_duplicate_objects,
    BF16,
    BLOCK_SCALE_LAYOUT,
    CONTIGUOUS_LAYOUT,
    DIRECT_FORMATS,
    FORMAT_NAMES,
    FP8,
    FP32,
    I32,
    LAYOUT_NAMES,
    LogicalAliasSpec,
    LogicalRowViewSpec,
    NVFP4,
    Q4,
    Q5,
    Q6,
    RESOURCE_ENCODING,
    RESOURCE_SPECS,
    ROW_SCALE_LAYOUT,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    VISION_LAYERS,
    W8,
    build_vision_specs as _family_vision_specs,
    tensor_spec as _family_tensor_spec,
)
from surogate.serve.convert.common.inventory import FP8_BLOCK as FP8_BLOCK_FORMAT, BLOCK128_LAYOUT


#: The identity of the checkpoint being converted. One converter serves the family, so this
#: is resolved per checkpoint by `model_id_for`; the constant is the registered size.
MODEL_ID = "qwen3.5-2b"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "qwen3_5"

#: The width at which this family becomes the 27B. It is a size key, not a dimension any
#: shape is built from: two things about a registered artifact follow the generation rather
#: than the numbers — the vision tower it carries and whether its checkpoint ties the output
#: head — and `is_27b` is the single place that decides them.
_HIDDEN_27B = 5120

#: Source `model_type` spellings that mean Qwen3.8. It shares every dimension with the 3.6,
#: so only what the checkpoint calls itself tells the two apart.
_QWEN3_8_MODEL_TYPES = ("qwen3_8", "qwen38")
QWEN3_8_MODEL_ID = "qwen3.8-27b"


def model_id_for(geometry: "Geometry", model_type: str | None = None) -> str:
    """The model id an artifact of this size claims, which the engine resolves its profile
    from. The family's target answers for every size.

    Qwen3.8-27B is dimensionally identical to Qwen3.6-27B and differs only in how its
    exports quantise, so the source's own `model_type` is what separates them; without one,
    a 27B checkpoint is taken for the 3.6.
    """
    declared = model_type or geometry.model_type
    if geometry.hidden == _HIDDEN_27B and str(declared) in _QWEN3_8_MODEL_TYPES:
        return QWEN3_8_MODEL_ID
    return {
        1024: "qwen3.5-0.8b",
        2048: "qwen3.5-2b",
        2560: "qwen3.5-4b",
        _HIDDEN_27B: "qwen3.6-27b",
    }.get(geometry.hidden, MODEL_ID)


@dataclass(frozen=True)
class Geometry:
    """The dimensions the object list depends on, read from `config.json` at convert time.

    One converter serves every size of this family: the numbers below are the 2B's, and they
    are the defaults only so a caller that has no checkpoint in hand still gets a valid list.

    `model_type` is not a dimension and takes no part in equality — it only carries what the
    source called itself, for the one decision the dimensions cannot make.
    """

    layers: int = 24
    hidden: int = 2048
    intermediate: int = 6144
    vocab: int = 248320
    query_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 256
    gdn_key_heads: int = 16
    gdn_key_head_dim: int = 128
    gdn_value_heads: int = 16
    gdn_value_head_dim: int = 128
    gdn_conv_kernel: int = 4
    full_attention_interval: int = 4
    model_type: str | None = field(default=None, compare=False)

    @property
    def query_size(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def key_dim(self) -> int:
        return self.gdn_key_heads * self.gdn_key_head_dim

    @property
    def value_dim(self) -> int:
        return self.gdn_value_heads * self.gdn_value_head_dim

    @property
    def convolution_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(range(3, self.layers, self.full_attention_interval))

    @property
    def gdn_layers(self) -> tuple[int, ...]:
        full = set(self.full_attention_layers)
        return tuple(layer for layer in range(self.layers) if layer not in full)


def is_27b(geometry: Geometry) -> bool:
    """Whether this is the family's 27B, whose checkpoint differs by more than its numbers:
    it unties the output head, and it carries the 27-layer vision tower where every smaller
    size carries a 24-layer one."""
    return geometry.hidden == _HIDDEN_27B


def geometry_from_config(config) -> Geometry:
    """The dimensions this checkpoint declares. `config` is the whole `config.json`."""
    text = config.get("text_config", config)
    return Geometry(
        layers=int(text["num_hidden_layers"]),
        hidden=int(text["hidden_size"]),
        intermediate=int(text["intermediate_size"]),
        vocab=int(text["vocab_size"]),
        query_heads=int(text["num_attention_heads"]),
        kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        gdn_key_heads=int(text["linear_num_key_heads"]),
        gdn_key_head_dim=int(text["linear_key_head_dim"]),
        gdn_value_heads=int(text["linear_num_value_heads"]),
        gdn_value_head_dim=int(text["linear_value_head_dim"]),
        gdn_conv_kernel=int(text["linear_conv_kernel_dim"]),
        full_attention_interval=int(text.get("full_attention_interval", 4)),
        # The root type is the family identity; the nested one is its "_text" variant.
        model_type=config.get("model_type", text.get("model_type")),
    )


#: GGUF architecture strings this family is exported under, and the `model_type` each means.
#: 3.5 and 3.6 share one string; 3.8 has its own, which is the only thing that tells it from
#: the 3.6 at the same dimensions.
_GGUF_ARCHITECTURES = (("qwen35", "qwen3_5"), ("qwen38", "qwen3_8"))


def geometry_from_gguf(kv) -> Geometry:
    """The dimensions this GGUF declares, for the repack plan.

    The plan is made before the bridge writes a `config.json`, so the numbers come from the
    file's own key-values. `ssm.group_count` is the linear-attention head count and
    `ssm.state_size` its head width; the value heads follow from the inner size.
    """
    for arch, model_type in _GGUF_ARCHITECTURES:
        blocks = int(kv(f"{arch}.block_count", 0) or 0)
        # llama.cpp counts the MTP (nextn) block among the blocks; the text core is the rest.
        blocks -= int(kv(f"{arch}.nextn_predict_layers", 0) or 0)
        if blocks:
            break
    else:
        raise ValueError("GGUF declares no architecture of this family")
    heads = int(kv(f"{arch}.attention.head_count"))
    key_heads = int(kv(f"{arch}.ssm.group_count"))
    key_head_dim = int(kv(f"{arch}.ssm.state_size"))
    inner = int(kv(f"{arch}.ssm.inner_size"))
    return Geometry(
        layers=blocks,
        hidden=int(kv(f"{arch}.embedding_length")),
        intermediate=int(kv(f"{arch}.feed_forward_length")),
        query_heads=heads,
        kv_heads=int(kv(f"{arch}.attention.head_count_kv")),
        head_dim=int(kv(f"{arch}.attention.key_length")),
        gdn_key_heads=key_heads,
        gdn_key_head_dim=key_head_dim,
        gdn_value_heads=inner // key_head_dim,
        gdn_value_head_dim=key_head_dim,
        gdn_conv_kernel=int(kv(f"{arch}.ssm.conv_kernel")),
        model_type=model_type,
    )


GEOMETRY = Geometry()

#: The 27B. Named because the exports that only exist at that size state their per-layer
#: exceptions in its layer numbering.
GEOMETRY_27B = Geometry(
    layers=64,
    hidden=_HIDDEN_27B,
    intermediate=17408,
    vocab=248320,
    query_heads=24,
    kv_heads=4,
    head_dim=256,
    gdn_key_heads=16,
    gdn_key_head_dim=128,
    gdn_value_heads=48,
    gdn_value_head_dim=128,
    gdn_conv_kernel=4,
)


# ---------------------------------------------------------------------------
# Export profiles
# ---------------------------------------------------------------------------

#: The published exports of this architecture, spelled as the engine's `WeightsProfile`
#: enumerators name them (csrc/src/serve/api/targets/qwen3_5/package.h). Quantisation is a
#: parameter of the conversion, never a separate converter.
GROUPWISE_INT = "groupwise-int"
NVFP4_MIXED_BF16 = "nvfp4-mixed-bf16"
NVFP4_UNIFORM = "nvfp4-uniform"
NVFP4_MLP_ONLY = "nvfp4-mlp-only"
NVFP4_ALL = "nvfp4-all"
#: Hugging Face fine-grained FP8: every projection E4M3 with an FP32 scale per 128x128
#: block (`weight_scale_inv`), fused parents, byte-wide endpoints. Any size of the family.
FP8_BLOCK = "fp8-block"
PROFILES = (GROUPWISE_INT, NVFP4_MIXED_BF16, NVFP4_UNIFORM, NVFP4_MLP_ONLY, NVFP4_ALL, FP8_BLOCK)

#: The `weights_id` half of the artifact identity each profile writes. The engine resolves
#: the profile back from (model_id, weights_id), so these strings are the contract: `nvfp4`
#: means two different profiles and is told apart by the model id, exactly as
#: `Package::resolve_weights` does it.
WEIGHTS_IDS = {
    GROUPWISE_INT: "groupwise-int",
    NVFP4_UNIFORM: "nvfp4-mixed",
    NVFP4_MIXED_BF16: "nvfp4",
    NVFP4_MLP_ONLY: "nvfp4",
    NVFP4_ALL: "nvfp4-all",
    FP8_BLOCK: "fp8-block",
}


def weights_id_for(profile: str) -> str:
    return WEIGHTS_IDS[profile]


def profile_for(model_id: str, weights_id: str) -> str:
    """The profile an artifact of this identity was written with; the engine's own rule."""
    if weights_id == "nvfp4":
        return NVFP4_MLP_ONLY if model_id == QWEN3_8_MODEL_ID else NVFP4_MIXED_BF16
    for profile, published in WEIGHTS_IDS.items():
        if published == weights_id and profile != NVFP4_MLP_ONLY:
            return profile
    raise ValueError(f"no profile writes {model_id!r}/{weights_id!r}")


#: How the fused projections are cut into objects. The engine binds whichever it finds, so
#: an export that types the halves apart stores them apart.
FUSED = "fused"
QUERY_KEY_AND_GATE_VALUE = "query_key+gate_value"
QUERY_KEY_AND_VALUE_Z = "query_key+value_z"
QUERY_KEY_VALUE_AND_Z = "query_key_value+z"
SPLIT_A_B = "a_projection+b_projection"
FUSED_A_B = "a_b_projection"


@dataclass(frozen=True)
class Export:
    """Everything one published export does to the object list, and nothing else.

    `attention_input`, `gdn_input` and `mlp` carry one format per object stored for that
    role, so a storage choice and the widths that go with it cannot drift apart.
    `exceptions` names the layers an export left at another width — measured from the
    published file, not derived, because nothing in the checkpoint states them.
    """

    attention_storage: str
    gdn_storage: str
    control_storage: str
    vocabulary: str
    draft_head: str
    attention_input: tuple[str, ...]
    attention_output: str
    gdn_input: tuple[str, ...]
    gdn_output: str
    mlp: tuple[str, str]
    exceptions: Mapping[str, tuple[str, tuple[int, ...]]] = field(default_factory=dict)
    #: What the export ships. The 4B NVFP4 release carries neither draft block nor tower.
    mtp: bool = True
    vision: bool = True

    def width(self, role: str, layer: int, default: str) -> str:
        exception = self.exceptions.get(role)
        if exception is not None and layer in exception[1]:
            return exception[0]
        return default


#: The 3.6-27B additive NVFP4 export left these layers in BF16, and the 3.8 MLP-only export
#: left its last eight MLPs in FP8. Both are properties of the published files.
_BF16_ATTENTION_INPUT_LAYERS = (3, 7, 11, 15, 19, 23)
_BF16_ATTENTION_OUTPUT_LAYERS = (3, 7)
_BF16_GDN_OUTPUT_LAYERS = (4,)
_FP8_MLP_LAYERS = tuple(range(56, 64))


def export_for(profile: str, geometry: Geometry = GEOMETRY) -> Export:
    """The export table for one profile at one size.

    Only the group-wise profile varies with the checkpoint, and only in what the endpoints
    and the fused projections are stored as: a K-quant repack of the 27B types the halves of
    each projection apart and writes a Q6 vocabulary, the 3.8 writes the same graph with a
    byte-wide one, and every smaller size is byte-wide throughout.
    """
    if profile == GROUPWISE_INT:
        if not is_27b(geometry):
            return Export(
                attention_storage=FUSED, gdn_storage=FUSED, control_storage=SPLIT_A_B,
                vocabulary=W8, draft_head=W8,
                attention_input=(W8,), attention_output=W8,
                gdn_input=(W8,), gdn_output=W8, mlp=(W8, W8),
            )
        vocabulary = W8 if model_id_for(geometry) == QWEN3_8_MODEL_ID else Q6
        return Export(
            attention_storage=QUERY_KEY_AND_GATE_VALUE,
            gdn_storage=QUERY_KEY_AND_VALUE_Z,
            control_storage=SPLIT_A_B,
            vocabulary=vocabulary, draft_head=Q4,
            attention_input=(Q4, Q5), attention_output=Q5,
            gdn_input=(Q4, Q5), gdn_output=Q5, mlp=(Q4, Q5),
        )
    if profile == NVFP4_UNIFORM:
        return Export(
            attention_storage=FUSED, gdn_storage=FUSED, control_storage=SPLIT_A_B,
            vocabulary=W8, draft_head=W8,
            attention_input=(NVFP4,), attention_output=NVFP4,
            gdn_input=(NVFP4,), gdn_output=NVFP4, mlp=(NVFP4, NVFP4),
            mtp=False, vision=False,
        )
    if profile == NVFP4_MIXED_BF16:
        return Export(
            attention_storage=FUSED, gdn_storage=FUSED, control_storage=SPLIT_A_B,
            vocabulary=W8, draft_head=Q4,
            attention_input=(NVFP4,), attention_output=NVFP4,
            gdn_input=(NVFP4,), gdn_output=NVFP4, mlp=(NVFP4, NVFP4),
            exceptions={
                "attention_input": (BF16, _BF16_ATTENTION_INPUT_LAYERS),
                "attention_output": (BF16, _BF16_ATTENTION_OUTPUT_LAYERS),
                "gdn_output": (BF16, _BF16_GDN_OUTPUT_LAYERS),
            },
        )
    if profile == NVFP4_MLP_ONLY:
        return Export(
            attention_storage=FUSED, gdn_storage=FUSED, control_storage=FUSED_A_B,
            vocabulary=FP8, draft_head=Q4,
            attention_input=(FP8,), attention_output=FP8,
            gdn_input=(FP8,), gdn_output=FP8, mlp=(NVFP4, NVFP4),
            exceptions={"mlp": (FP8, _FP8_MLP_LAYERS)},
        )
    if profile == FP8_BLOCK:
        return Export(
            attention_storage=FUSED, gdn_storage=FUSED, control_storage=SPLIT_A_B,
            vocabulary=W8, draft_head=W8,
            attention_input=(FP8_BLOCK_FORMAT,), attention_output=FP8_BLOCK_FORMAT,
            gdn_input=(FP8_BLOCK_FORMAT,), gdn_output=FP8_BLOCK_FORMAT,
            mlp=(FP8_BLOCK_FORMAT, FP8_BLOCK_FORMAT),
            mtp=False, vision=False,
        )
    if profile == NVFP4_ALL:
        return Export(
            attention_storage=FUSED, gdn_storage=QUERY_KEY_VALUE_AND_Z,
            control_storage=FUSED_A_B,
            vocabulary=FP8, draft_head=Q4,
            attention_input=(NVFP4,), attention_output=NVFP4,
            gdn_input=(NVFP4, NVFP4), gdn_output=NVFP4, mlp=(NVFP4, NVFP4),
        )
    raise ValueError(f"unknown export profile: {profile!r}")


# ---------------------------------------------------------------------------
# Object list
# ---------------------------------------------------------------------------


def tensor_spec(name: str, shape: tuple[int, ...], numeric_format: str) -> TensorSpec:
    """A spec with the layout its format is stored in; NVFP4 carries block scales and FP8
    a scale per row, which the shared table does not cover."""
    if numeric_format == NVFP4:
        return TensorSpec(name, shape, numeric_format, BLOCK_SCALE_LAYOUT)
    if numeric_format == FP8:
        return TensorSpec(name, shape, numeric_format, ROW_SCALE_LAYOUT)
    if numeric_format == FP8_BLOCK_FORMAT:
        return TensorSpec(name, shape, numeric_format, BLOCK128_LAYOUT)
    return _family_tensor_spec(name, shape, numeric_format)


_tensor = tensor_spec

#: The scale a projection's activations are divided by, published beside the weight that
#: needs it. Only an NVFP4 weight has one.
_DIVISOR_SITES = {
    "attention/query_key_gate_value": "attention/input_projection",
    "attention/query_key": "attention/input_projection",
    "attention/output": "attention/output_projection",
    "gdn/query_key_value_z": "gdn/input_projection",
    "gdn/query_key_value": "gdn/input_projection",
    "gdn/z": "gdn/z_projection",
    "gdn/output": "gdn/output_projection",
    "mlp/gate_up": "mlp/gate_up_projection",
    "mlp/down": "mlp/down_projection",
}


def _weight(specs: list[TensorSpec], prefix: str, role: str,
            shape: tuple[int, ...], numeric_format: str) -> None:
    """Append one weight and, when it is NVFP4, the divisor that decodes it."""
    specs.append(_tensor(prefix + role, shape, numeric_format))
    if numeric_format == NVFP4:
        specs.append(_tensor(prefix + _DIVISOR_SITES[role] + "/input_scale_divisor", (), FP32))


def attention_input_roles(g: Geometry, export: Export) -> tuple[tuple[str, tuple[int, int]], ...]:
    """The objects q|k|gate|v is stored as, with the rows each holds."""
    if export.attention_storage == QUERY_KEY_AND_GATE_VALUE:
        half = (g.query_size + g.kv_size, g.hidden)
        return (("attention/query_key", half), ("attention/gate_value", half))
    return (("attention/query_key_gate_value",
             (2 * g.query_size + 2 * g.kv_size, g.hidden)),)


def gdn_input_roles(g: Geometry, export: Export) -> tuple[tuple[str, tuple[int, int]], ...]:
    """The objects q|k|v|z is stored as, cut where the export types them apart."""
    if export.gdn_storage == QUERY_KEY_AND_VALUE_Z:
        return (("gdn/query_key", (2 * g.key_dim, g.hidden)),
                ("gdn/value_z", (2 * g.value_dim, g.hidden)))
    if export.gdn_storage == QUERY_KEY_VALUE_AND_Z:
        return (("gdn/query_key_value", (g.convolution_dim, g.hidden)),
                ("gdn/z", (g.value_dim, g.hidden)))
    return (("gdn/query_key_value_z", (g.convolution_dim + g.value_dim, g.hidden)),)


def gdn_control_roles(g: Geometry, export: Export) -> tuple[tuple[str, tuple[int, int]], ...]:
    if export.control_storage == FUSED_A_B:
        return (("gdn/a_b_projection", (2 * g.gdn_value_heads, g.hidden)),)
    return (("gdn/a_projection", (g.gdn_value_heads, g.hidden)),
            ("gdn/b_projection", (g.gdn_value_heads, g.hidden)))


def build_text_core_specs(g: Geometry = GEOMETRY,
                          profile: str = GROUPWISE_INT) -> tuple[TensorSpec, ...]:
    export = export_for(profile, g)
    full = set(g.full_attention_layers)
    specs: list[TensorSpec] = [
        _tensor("text/token_embedding", (g.vocab, g.hidden), export.vocabulary),
    ]

    for layer in range(g.layers):
        prefix = f"text/layers/{layer}/"
        specs.append(_tensor(prefix + "input_norm", (g.hidden,), BF16))

        if layer in full:
            for (role, shape), width in zip(attention_input_roles(g, export),
                                            export.attention_input):
                _weight(specs, prefix, role, shape,
                        export.width("attention_input", layer, width))
            specs.extend(
                (
                    _tensor(prefix + "attention/query_norm", (g.head_dim,), BF16),
                    _tensor(prefix + "attention/key_norm", (g.head_dim,), BF16),
                )
            )
            _weight(specs, prefix, "attention/output", (g.hidden, g.query_size),
                    export.width("attention_output", layer, export.attention_output))
        else:
            specs.extend(
                (
                    _tensor(prefix + "gdn/a_log", (g.gdn_value_heads,), FP32),
                    _tensor(prefix + "gdn/dt_bias", (g.gdn_value_heads,), FP32),
                    _tensor(prefix + "gdn/convolution",
                            (g.gdn_conv_kernel, g.convolution_dim), BF16),
                )
            )
            for role, shape in gdn_control_roles(g, export):
                specs.append(_tensor(prefix + role, shape, BF16))
            for (role, shape), width in zip(gdn_input_roles(g, export), export.gdn_input):
                _weight(specs, prefix, role, shape,
                        export.width("gdn_input", layer, width))
            specs.append(_tensor(prefix + "gdn/norm", (g.gdn_key_head_dim,), BF16))
            _weight(specs, prefix, "gdn/output", (g.hidden, g.value_dim),
                    export.width("gdn_output", layer, export.gdn_output))

        specs.append(_tensor(prefix + "post_attention_norm", (g.hidden,), BF16))
        _weight(specs, prefix, "mlp/gate_up", (2 * g.intermediate, g.hidden),
                export.width("mlp", layer, export.mlp[0]))
        _weight(specs, prefix, "mlp/down", (g.hidden, g.intermediate),
                export.width("mlp", layer, export.mlp[1]))

    specs.extend(
        (
            _tensor("text/final_norm", (g.hidden,), BF16),
            _tensor("text/output_head", (g.vocab, g.hidden), export.vocabulary),
        )
    )
    return tuple(specs)


#: The draft head's shortlist: a fixed count of frequent tokens, not a model dimension.
DRAFT_VOCAB = 131072


def build_draft_head_specs(g: Geometry = GEOMETRY,
                           profile: str = GROUPWISE_INT) -> tuple[TensorSpec, ...]:
    return (
        _tensor("text/draft_head", (DRAFT_VOCAB, g.hidden), export_for(profile, g).draft_head),
        _tensor("text/draft_head_token_ids", (DRAFT_VOCAB,), I32),
    )


def build_mtp_specs(g: Geometry = GEOMETRY) -> tuple[TensorSpec, ...]:
    # The draft block is byte-wide under every export: it is one layer, and the widths the
    # quantised exports chose for the stack buy nothing here.
    return (
        _tensor("mtp/input_projection", (g.hidden, 2 * g.hidden), W8),
        _tensor("mtp/embedding_norm", (g.hidden,), BF16),
        _tensor("mtp/hidden_norm", (g.hidden,), BF16),
        _tensor("mtp/layer/input_norm", (g.hidden,), BF16),
        _tensor("mtp/layer/attention/query_key_gate_value",
                (2 * g.query_size + 2 * g.kv_size, g.hidden), W8),
        _tensor("mtp/layer/attention/query_norm", (g.head_dim,), BF16),
        _tensor("mtp/layer/attention/key_norm", (g.head_dim,), BF16),
        _tensor("mtp/layer/attention/output", (g.hidden, g.query_size), W8),
        _tensor("mtp/layer/post_attention_norm", (g.hidden,), BF16),
        _tensor("mtp/layer/mlp/gate_up", (2 * g.intermediate, g.hidden), W8),
        _tensor("mtp/layer/mlp/down", (g.hidden, g.intermediate), W8),
        _tensor("mtp/final_norm", (g.hidden,), BF16),
    )


def vision_tower(g: Geometry) -> dict[str, int]:
    """The tower this size carries. Its dimensions do not follow the text size — only the
    width it projects into does — so each size registers one rather than deriving it."""
    if is_27b(g):
        return dict(layers=27, hidden=1152, intermediate=4304, qkv_rows=3456,
                    merger_hidden=4608)
    return dict(layers=24, hidden=1024, intermediate=4096, qkv_rows=3072,
                merger_hidden=4096)


def build_vision_specs(g: Geometry = GEOMETRY) -> tuple[TensorSpec, ...]:
    return _family_vision_specs(g.hidden, **vision_tower(g))


def build_tensor_specs(geometry: Geometry = GEOMETRY, *, profile: str = GROUPWISE_INT,
                       mtp: bool | None = None,
                       vision: bool | None = None) -> tuple[TensorSpec, ...]:
    """The tensor list for one checkpoint and export, which a repack plan is made against."""
    export = export_for(profile, geometry)
    tensors = (build_text_core_specs(geometry, profile)
               + build_draft_head_specs(geometry, profile))
    if export.mtp if mtp is None else mtp:
        tensors += build_mtp_specs(geometry)
    if export.vision if vision is None else vision:
        tensors += build_vision_specs(geometry)
    return tensors


# A community GGUF export of this family is text-only: it drops the vision tower the way it
# often drops the MTP block. The loader already probes for one (`binder.has("vision/...")`)
# and only refuses `--vision` against an artifact without it, so the artifact may omit the
# vision/* objects entirely.
def active_specs(*, mtp: bool | None = None, vision: bool | None = None,
                 geometry: Geometry = GEOMETRY,
                 profile: str = GROUPWISE_INT) -> tuple[tuple, tuple]:
    """(tensor_specs, object_specs) for the requested artifact variant."""
    tensors = build_tensor_specs(geometry, profile=profile, mtp=mtp, vision=vision)
    return tensors, RESOURCE_SPECS + tensors


# ---------------------------------------------------------------------------
# Logical views
# ---------------------------------------------------------------------------


def build_logical_row_views(g: Geometry = GEOMETRY,
                            profile: str = GROUPWISE_INT) -> tuple[LogicalRowViewSpec, ...]:
    """The fixed row windows a kernel reads out of a stored parent.

    Every window names the object that actually holds it, so an export that stores a
    projection whole and one that stores it as a typed pair describe the same four
    attention rows against different parents. A row that an export stores as an object of
    its own gets no window: it is already addressable.
    """
    export = export_for(profile, g)
    full = g.full_attention_layers
    gdn = g.gdn_layers
    every_layer = tuple(range(g.layers))
    text = "text/layers/{l}/"

    def view(name, parent, begin, rows, layers):
        return LogicalRowViewSpec(name, parent, begin, begin + rows, (rows, g.hidden), layers)

    views: list[LogicalRowViewSpec] = []
    if export.attention_storage == QUERY_KEY_AND_GATE_VALUE:
        query_key, gate_value = text + "attention/query_key", text + "attention/gate_value"
        gate_begin = 0
    else:
        query_key = gate_value = text + "attention/query_key_gate_value"
        gate_begin = g.query_size + g.kv_size
    views.extend(
        (
            view(text + "attention/query", query_key, 0, g.query_size, full),
            view(text + "attention/key", query_key, g.query_size, g.kv_size, full),
            view(text + "attention/output_gate", gate_value, gate_begin, g.query_size, full),
            view(text + "attention/value", gate_value, gate_begin + g.query_size, g.kv_size,
                 full),
        )
    )

    if export.gdn_storage == QUERY_KEY_AND_VALUE_Z:
        query_key, value_z = text + "gdn/query_key", text + "gdn/value_z"
        views.extend(
            (
                view(text + "gdn/query", query_key, 0, g.key_dim, gdn),
                view(text + "gdn/key", query_key, g.key_dim, g.key_dim, gdn),
                view(text + "gdn/value", value_z, 0, g.value_dim, gdn),
                view(text + "gdn/z", value_z, g.value_dim, g.value_dim, gdn),
            )
        )
    else:
        qkv = text + ("gdn/query_key_value" if export.gdn_storage == QUERY_KEY_VALUE_AND_Z
                      else "gdn/query_key_value_z")
        views.extend(
            (
                view(text + "gdn/query", qkv, 0, g.key_dim, gdn),
                view(text + "gdn/key", qkv, g.key_dim, g.key_dim, gdn),
                view(text + "gdn/value", qkv, 2 * g.key_dim, g.value_dim, gdn),
            )
        )
        if export.gdn_storage != QUERY_KEY_VALUE_AND_Z:
            views.append(view(text + "gdn/z", qkv, g.convolution_dim, g.value_dim, gdn))

    if export.control_storage == FUSED_A_B:
        a_b = text + "gdn/a_b_projection"
        views.extend(
            (
                view(text + "gdn/a_projection", a_b, 0, g.gdn_value_heads, gdn),
                view(text + "gdn/b_projection", a_b, g.gdn_value_heads, g.gdn_value_heads,
                     gdn),
            )
        )

    views.extend(
        (
            view(text + "mlp/gate", text + "mlp/gate_up", 0, g.intermediate, every_layer),
            view(text + "mlp/up", text + "mlp/gate_up", g.intermediate, g.intermediate,
                 every_layer),
        )
    )

    # The draft block always stores the one fused parent, whatever the stack does.
    mtp = "mtp/layer/"
    mtp_parent = mtp + "attention/query_key_gate_value"
    views.extend(
        (
            view(mtp + "attention/query", mtp_parent, 0, g.query_size, None),
            view(mtp + "attention/key", mtp_parent, g.query_size, g.kv_size, None),
            view(mtp + "attention/output_gate", mtp_parent, g.query_size + g.kv_size,
                 g.query_size, None),
            view(mtp + "attention/value", mtp_parent, 2 * g.query_size + g.kv_size,
                 g.kv_size, None),
            view(mtp + "mlp/gate", mtp + "mlp/gate_up", 0, g.intermediate, None),
            view(mtp + "mlp/up", mtp + "mlp/gate_up", g.intermediate, g.intermediate, None),
        )
    )
    return tuple(views)


def build_alias_specs(g: Geometry = GEOMETRY) -> tuple[LogicalAliasSpec, ...]:
    return (
        LogicalAliasSpec("mtp/token_embedding", ("text/token_embedding",)),
        LogicalAliasSpec("mtp/full_output_head", ("text/output_head",)),
        LogicalAliasSpec(
            "mtp/optimized_proposal_head",
            ("text/draft_head", "text/draft_head_token_ids"),
        ),
        LogicalAliasSpec(
            "text/layers/{l}/gdn/channel_major_convolution",
            ("text/layers/{l}/gdn/convolution",),
            layers=g.gdn_layers,
            axis_order=(1, 0),
        ),
    )


# ---------------------------------------------------------------------------
# The registered size, for callers with no checkpoint in hand
# ---------------------------------------------------------------------------

FULL_ATTENTION_LAYERS = GEOMETRY.full_attention_layers
GDN_LAYERS = GEOMETRY.gdn_layers

TEXT_CORE_TENSOR_SPECS = build_text_core_specs()
DRAFT_HEAD_TENSOR_SPECS = build_draft_head_specs()
MTP_TENSOR_SPECS = build_mtp_specs()
VISION_TENSOR_SPECS = build_vision_specs()

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

# surogate vendor patch (PATCHES.md #15): community GGUF exports frequently
# strip the MTP (nextn) block; such checkpoints convert to an artifact that
# omits the mtp/* objects entirely (the loader binds MTP only when present
# and MTP speculation is refused with a clear error). The draft head stays:
# it derives from the embedding, which every export carries.
TENSOR_SPECS_NO_MTP = TEXT_CORE_TENSOR_SPECS + DRAFT_HEAD_TENSOR_SPECS + VISION_TENSOR_SPECS
OBJECT_SPECS_NO_MTP: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS_NO_MTP

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

LOGICAL_ROW_VIEW_SPECS = build_logical_row_views()
ALIAS_SPECS = build_alias_specs()


# ---------------------------------------------------------------------------
# One export's complete inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportInventory:
    """Every list a conversion of one (checkpoint, export) pair needs.

    A conversion asks for this instead of importing an inventory module per export, so the
    recipe that fills an artifact and the contract that describes it are built from the one
    profile table and cannot describe different artifacts. The names are the ones an
    inventory module published, because that is what this stands in for.
    """

    geometry: Geometry
    profile: str
    export: Export
    MODEL_ID: str
    WEIGHTS_ID: str
    TARGET_KEY: str
    TENSOR_SPECS: tuple[TensorSpec, ...]
    OBJECT_SPECS: tuple[StoredObjectSpec, ...]
    TEXT_CORE_TENSOR_SPECS: tuple[TensorSpec, ...]
    LOGICAL_ROW_VIEW_SPECS: tuple[LogicalRowViewSpec, ...]
    ALIAS_SPECS: tuple[LogicalAliasSpec, ...]

    RESOURCE_SPECS = RESOURCE_SPECS
    TensorSpec = TensorSpec
    ResourceSpec = ResourceSpec
    BF16 = BF16
    FP32 = FP32
    I32 = I32
    NVFP4 = NVFP4
    FP8 = FP8
    W8 = W8
    Q4 = Q4
    Q5 = Q5
    Q6 = Q6

    @property
    def FULL_ATTENTION_LAYERS(self) -> tuple[int, ...]:
        return self.geometry.full_attention_layers

    @property
    def GDN_LAYERS(self) -> tuple[int, ...]:
        return self.geometry.gdn_layers

    def specs_of(self, numeric_format: str) -> tuple[TensorSpec, ...]:
        return tuple(spec for spec in self.TENSOR_SPECS if spec.format == numeric_format)

    @property
    def NVFP4_TENSOR_SPECS(self) -> tuple[TensorSpec, ...]:
        return self.specs_of(NVFP4)

    @property
    def FP8_TENSOR_SPECS(self) -> tuple[TensorSpec, ...]:
        return self.specs_of(FP8)

    @property
    def INPUT_SCALE_DIVISOR_SPECS(self) -> tuple[TensorSpec, ...]:
        return tuple(spec for spec in self.TENSOR_SPECS
                     if spec.format == FP32 and spec.name.endswith("/input_scale_divisor"))

    @property
    def FORMAT_COUNTS(self) -> dict[str, int]:
        # Only what this artifact actually holds: the family's vocabulary spans every export,
        # and a tally of formats an artifact does not use says nothing about it.
        counts = {name: sum(spec.format == name for spec in self.TENSOR_SPECS)
                  for name in FORMAT_NAMES}
        return {name: count for name, count in counts.items() if count}

    @property
    def LAYOUT_COUNTS(self) -> dict[str, int]:
        counts = {name: sum(spec.layout == name for spec in self.TENSOR_SPECS)
                  for name in LAYOUT_NAMES}
        return {name: count for name, count in counts.items() if count}

    def layers_at(self, role: str, numeric_format: str) -> tuple[int, ...]:
        """The layers whose `role` this export wrote at `numeric_format`.

        A recipe reads a layer from a different source field where the export left it at
        another width, so it asks which layers those are rather than restating the list.
        """
        default = {
            "attention_input": self.export.attention_input[0],
            "attention_output": self.export.attention_output,
            "gdn_input": self.export.gdn_input[0],
            "gdn_output": self.export.gdn_output,
            "mlp": self.export.mlp[0],
        }[role]
        layers = (self.FULL_ATTENTION_LAYERS if role.startswith("attention")
                  else self.GDN_LAYERS if role.startswith("gdn")
                  else tuple(range(self.geometry.layers)))
        exception = self.export.exceptions.get(role)
        if exception is None:
            return layers if numeric_format == default else ()
        other, listed = exception
        if numeric_format == other:
            return tuple(layer for layer in layers if layer in set(listed))
        if numeric_format == default:
            return tuple(layer for layer in layers if layer not in set(listed))
        return ()

    def validate_inventory(self) -> None:
        """No object may be named twice, and every quantised weight must have its divisor."""
        names = tuple(spec.name for spec in self.OBJECT_SPECS)
        if len(names) != len(set(names)):
            raise ValueError(f"{self.profile} inventory contains duplicate object names")
        divisors = {spec.name for spec in self.INPUT_SCALE_DIVISOR_SPECS}
        if len(divisors) != len(self.NVFP4_TENSOR_SPECS):
            raise ValueError(
                f"{self.profile} inventory has {len(divisors)} activation divisors for "
                f"{len(self.NVFP4_TENSOR_SPECS)} NVFP4 weights"
            )


def export_inventory(profile: str = GROUPWISE_INT, geometry: Geometry = GEOMETRY, *,
                     mtp: bool | None = None,
                     vision: bool | None = None) -> ExportInventory:
    """The complete inventory for one checkpoint converted under one export profile."""
    tensors, objects = active_specs(mtp=mtp, vision=vision, geometry=geometry,
                                    profile=profile)
    return ExportInventory(
        geometry=geometry,
        profile=profile,
        export=export_for(profile, geometry),
        MODEL_ID=model_id_for(geometry),
        WEIGHTS_ID=weights_id_for(profile),
        TARGET_KEY=TARGET_KEY,
        TENSOR_SPECS=tensors,
        OBJECT_SPECS=objects,
        TEXT_CORE_TENSOR_SPECS=build_text_core_specs(geometry, profile),
        LOGICAL_ROW_VIEW_SPECS=build_logical_row_views(geometry, profile),
        ALIAS_SPECS=build_alias_specs(geometry),
    )


#: The 27B geometry a Qwen3.8 checkpoint has, which its exports are described against.
GEOMETRY_QWEN3_8 = Geometry(**{**vars(GEOMETRY_27B), "model_type": "qwen3_8"})
